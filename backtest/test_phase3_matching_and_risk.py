"""
backtest/test_phase3_matching_and_risk.py
Phase 3 撮合引擎微观流动性、动态冲击滑点与高频风控状态机专有测试。

测试覆盖：
1. 单 Bar 成交量占比上限 (Volume Participation Cap) 约束：
   - 大单超额买入时受限截断至 Bar 容量上限。
   - 低流动性不足一手时，验证一手保底机制与流动性放弃机制。
2. 高频流动性与波动率冲击滑点模型 (Dynamic Slippage Model):
   - 基于平方根冲击定律 (Square-root Law) 验证冲击与参与率、振幅的正相关性。
   - 验证滑点被 max_slippage 严格截断封顶。
   - 100% 向下兼容固定滑点 (fixed)。
3. 隔夜跳空防守规则 (Overnight Gap Protection):
   - 隔夜向下跳空开盘直接跌破止损线时，以真实跳空开盘价成交，杜绝理想化止损价幻觉。
4. 日内平仓模式 (Intraday Exit at EOD / 不隔夜):
   - 尾盘最后一根 Bar 强制平仓防守，持仓在收盘时平仓归零。
5. BacktestEngine 全链路动态流动性撮合集成测试。
"""

import unittest
from datetime import datetime, date, time
import pandas as pd
import numpy as np

from backtest.core.types import Position, TimeFrame
from backtest.core.market_rules import MarketProfile
from backtest.execution.simulated_broker import SimulatedBroker
from backtest.execution.risk_manager import ExitStateMachine
from backtest.engine.backtest_engine import KronosBacktesterPro


class TestVolumeParticipationCap(unittest.TestCase):
    """测试单 Bar 成交量占比上限 (Volume Participation Cap)"""

    def setUp(self):
        self.profile_cn = MarketProfile.for_stock("600036")  # A 股，每手 100 股
        self.profile_us = MarketProfile.for_stock("AAPL")    # 美股，每手 1 股

    def test_volume_cap_limits_buy_shares(self):
        """测试大额申报时被单 Bar 成交量上限截断"""
        broker = SimulatedBroker(
            profile=self.profile_cn,
            initial_capital=1000000.0,
            volume_participation_cap=0.05,  # 上限 5%
        )
        # 单 Bar 成交量为 10,000 股，5% 上限为 500 股
        bar_vol = 10000.0
        pos = broker.execute_buy(
            trade_date=pd.Timestamp("2026-03-02 10:00:00"),
            symbol="600036",
            open_price=30.0,
            shares_to_buy=2000,  # 算法原拟买入 2000 股
            target_horizon=5,
            active_stop_pct=0.04,
            stop_price=28.8,
            bar_idx=0,
            close_price=30.2,
            low_price=29.9,
            bar_volume=bar_vol,
            bar_high=30.5,
        )
        self.assertIsNotNone(pos)
        # 严格截断到 500 股
        self.assertEqual(pos.shares, 500)
        self.assertEqual(broker.trades[-1]['shares'], 500)

    def test_volume_cap_min_lot_guarantee(self):
        """测试流动性极度贫乏时 (上限计算不足 1 手): 保底 vs 放弃"""
        # 1. 开启一手保底 (min_lot_guarantee=True)
        broker_guarantee = SimulatedBroker(
            profile=self.profile_cn,
            initial_capital=100000.0,
            volume_participation_cap=0.05,  # 5%
            min_lot_guarantee=True,
        )
        # Bar 成交量 800 股，5% 上限仅 40 股，不足一手 100 股
        pos_guarantee = broker_guarantee.execute_buy(
            trade_date=pd.Timestamp("2026-03-02 10:00:00"),
            symbol="600036",
            open_price=30.0,
            shares_to_buy=500,
            target_horizon=5,
            active_stop_pct=0.04,
            stop_price=28.8,
            bar_idx=0,
            close_price=30.2,
            low_price=29.9,
            bar_volume=800.0,
        )
        self.assertIsNotNone(pos_guarantee)
        self.assertEqual(pos_guarantee.shares, 100)  # 保底买入 1 手 (100股)

        # 2. 关闭一手保底 (min_lot_guarantee=False)
        broker_no_guarantee = SimulatedBroker(
            profile=self.profile_cn,
            initial_capital=100000.0,
            volume_participation_cap=0.05,
            min_lot_guarantee=False,
        )
        pos_no_guarantee = broker_no_guarantee.execute_buy(
            trade_date=pd.Timestamp("2026-03-02 10:00:00"),
            symbol="600036",
            open_price=30.0,
            shares_to_buy=500,
            target_horizon=5,
            active_stop_pct=0.04,
            stop_price=28.8,
            bar_idx=0,
            close_price=30.2,
            low_price=29.9,
            bar_volume=800.0,
        )
        self.assertIsNone(pos_no_guarantee)  # 流动性不足一手，直接安全弃单


class TestDynamicSlippageModel(unittest.TestCase):
    """测试基于平方根冲击定律与波动率的动态滑点模型"""

    def setUp(self):
        self.profile = MarketProfile.for_stock("AAPL")

    def test_fixed_slippage_remains_constant(self):
        """固定滑点模式下，无论成交量多大，滑点始终为恒定基础值"""
        broker = SimulatedBroker(
            profile=self.profile,
            slippage_rate=0.001,
            slippage_model="fixed",
        )
        slip = broker.compute_slippage(
            base_price=150.0,
            shares=10000,
            bar_volume=20000.0,
            bar_high=155.0,
            bar_low=148.0,
        )
        self.assertEqual(slip, 0.001)

    def test_dynamic_slippage_scales_with_participation_and_spread(self):
        """动态滑点模式下：参与率与波动幅度越大，冲击滑点越高"""
        broker = SimulatedBroker(
            profile=self.profile,
            slippage_rate=0.001,
            slippage_model="dynamic",
            slippage_impact_coef=0.1,
            max_slippage=0.02,
        )

        # 1. 微小参与率 (100 股 / 1,000,000 股，振幅 0.5%)
        slip_low = broker.compute_slippage(
            base_price=100.0,
            shares=100,
            bar_volume=1000000.0,
            bar_high=100.5,
            bar_low=100.0,
        )
        # 接近基础滑点 0.001
        self.assertAlmostEqual(slip_low, 0.001, delta=0.0001)

        # 2. 较大参与率 (10,000 股 / 20,000 股 = 50%，振幅 4%)
        slip_high = broker.compute_slippage(
            base_price=100.0,
            shares=10000,
            bar_volume=20000.0,
            bar_high=104.0,
            bar_low=100.0,
        )
        # 显著大于基础滑点
        self.assertGreater(slip_high, slip_low)
        self.assertGreater(slip_high, 0.003)

        # 3. 极端冲击受 max_slippage 封顶
        slip_extreme = broker.compute_slippage(
            base_price=100.0,
            shares=100000,
            bar_volume=10000.0,
            bar_high=120.0,
            bar_low=90.0,
        )
        self.assertEqual(slip_extreme, 0.02)


class TestOvernightGapProtection(unittest.TestCase):
    """测试隔夜向下跳空防守规则"""

    def setUp(self):
        self.profile = MarketProfile.for_stock("600036")
        self.exit_sm = ExitStateMachine(
            hard_stop_loss=0.04,
            intraday_stop=True,
            overnight_gap_protection=True,
        )

    def test_gap_down_fills_at_open_price(self):
        """次日大幅向下跳空开盘跌破止损线，成交价以跳空开盘价而非理想止损价成交"""
        pos = Position(
            symbol="600036",
            shares=1000,
            avg_entry_price=100.0,
            entry_date=pd.Timestamp("2026-03-02 10:00:00"),
            stop_price=96.0,  # 止损线 96.0
        )
        # 模拟次日跨日数据: 开盘跳空低开到 93.0 (低于止损线 96.0)
        row_mock = pd.Series({
            "open": 93.0,
            "high": 94.0,
            "low": 92.5,
            "close": 93.5,
            "volume": 50000,
        }, name=pd.Timestamp("2026-03-03 09:30:00"))

        should_exit, reason, fill_px = self.exit_sm.check_intraday_exit(
            row=row_mock,
            position=pos,
            profile=self.profile,
            idx=1,
            limit_down=0,
        )

        self.assertTrue(should_exit)
        # 真实穿透成交价为跳空开盘价 93.0，杜绝 96.0 的虚假回测幻觉
        self.assertEqual(fill_px, 93.0)
        self.assertIn("隔夜跳空止损", reason)


class TestIntradayExitAtEod(unittest.TestCase):
    """测试日内平仓模式 (尾盘强制离场不隔夜)"""

    def setUp(self):
        self.profile_us = MarketProfile.for_stock("US_AAPL")
        self.exit_sm = ExitStateMachine(
            hard_stop_loss=0.04,
            intraday_exit_at_eod=True,
        )

    def test_eod_forces_exit_on_last_bar(self):
        """在最后一根 Bar 时且允许平仓时，触发日内清仓"""
        pos = Position(
            symbol="AAPL",
            shares=100,
            avg_entry_price=150.0,
            entry_date=pd.Timestamp("2026-03-02 09:30:00"),
            stop_price=144.0,
        )
        # 美股为 T+0，入场当日即可平仓
        row_last_bar = pd.Series({
            "open": 152.0,
            "high": 153.0,
            "low": 151.8,
            "close": 152.5,
            "volume": 20000,
        }, name=pd.Timestamp("2026-03-02 15:30:00"))

        reason, px = self.exit_sm.evaluate_eod_exit(
            row=row_last_bar,
            position=pos,
            profile=self.profile_us,
            idx=12,
            is_last_bar_of_day=True,
        )
        self.assertEqual(reason, "日内平仓模式(尾盘不隔夜)")
        self.assertEqual(px, 152.5)


class TestBacktestEnginePhase3Integration(unittest.TestCase):
    """测试 BacktestEngine 动态冲击与成交量占比上限全流程集成"""

    def test_engine_runs_with_dynamic_slippage_and_cap(self):
        """验证在开启 dynamic slippage 与 volume cap 时引擎平稳运行且交易台账生成"""
        engine = KronosBacktesterPro(data_dir=".", model_dir=".", initial_capital=100000.0)
        engine.set_market("AAPL", market="US")

        # 构造 10 天模拟信号矩阵
        dates = pd.bdate_range(start="2026-03-02", periods=10)
        df_records = []
        px = 100.0
        for i, d in enumerate(dates):
            exec_sig = 1 if i == 1 else 0
            df_records.append({
                "date": d,
                "open": px,
                "high": px + 1.0,
                "low": px - 1.0,
                "close": px + 0.5,
                "volume": 50000.0,
                "exec_signal": exec_sig,
                "atr_pct": 0.02,
                "exec_target_horizon": 3,
            })
            px += 0.5

        test_df = pd.DataFrame(df_records).set_index("date")

        backtest_df, trades = engine.run_backtest_engine(
            test_df,
            volume_participation_cap=0.10,
            slippage_model="dynamic",
            slippage_impact_coef=0.1,
            max_slippage=0.02,
        )

        self.assertFalse(backtest_df.empty)
        # 验证产生了交易台账
        self.assertGreater(len(trades), 0)
        # 验证每笔交易都记录了有效滑点率
        for t in trades:
            self.assertIn("slippage_rate", t)
            self.assertGreater(t["slippage_rate"], 0)


if __name__ == "__main__":
    unittest.main()
