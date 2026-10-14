"""
backtest/test_backtest_engine.py
量化执行引擎核心组件单测（无网络、纯内存数据）。
测试覆盖:
1. RiskBudgetSizer: 风险预算定头寸、单票上限、一手保底机制、连亏降仓熔断
2. ExitStateMachine: 盘中硬止损触发、跌停顺延拦截、收盘移动止盈、模型证伪时间止损、动量跃迁护航
3. SimulatedBroker: 买卖撮合、扣费税费、持仓状态精准复位
4. KronosBacktesterPro: 协同撮合委托与完整状态机撮合闭环
"""
import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = Path(__file__).resolve().parent
for p in (str(ROOT), str(BACKTEST_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backtest.core.market_rules import MarketProfile
from backtest.core.types import Position
from backtest.execution.sizer import RiskBudgetSizer
from backtest.execution.risk_manager import ExitStateMachine
from backtest.execution.simulated_broker import SimulatedBroker
from backtest.engine.backtest_engine import KronosBacktesterPro


class TestRiskBudgetSizer(unittest.TestCase):
    """头寸规模与风险预算测算测试"""

    def setUp(self):
        self.profile_cn = MarketProfile.for_stock("300418")
        self.profile_us = MarketProfile.for_stock("AAPL")

    def test_standard_risk_budget_sizing(self):
        """测试按风险预算 (2%) 及止损幅度推导股数"""
        sizer = RiskBudgetSizer(risk_per_trade=0.02, max_position_pct=0.30)
        # 总本金 100,000，可用现金 100,000，买入价 20 元，硬止损 4%
        # 单笔风险额 = 100,000 * 0.02 = 2,000 元
        # 每股风险 = 20 * 0.04 = 0.8 元 => 理论股数 = 2,000 / 0.8 = 2,500 股
        # 最大单票市值上限 = 100,000 * 0.30 = 30,000 元 => 最多 1,500 股
        shares, active_stop_pct, active_trail_pct, stop_price = sizer.compute_size(
            portfolio_value=100000.0,
            cash=100000.0,
            buy_price=20.0,
            profile=self.profile_cn,
            hard_stop_loss=0.04,
            trailing_stop=0.025,
        )
        self.assertEqual(shares, 1500)
        self.assertAlmostEqual(active_stop_pct, 0.04)
        self.assertAlmostEqual(stop_price, 20.0 * 0.96)

    def test_consecutive_losses_penalty(self):
        """测试连续亏损降仓机制"""
        sizer = RiskBudgetSizer(risk_per_trade=0.02, max_position_pct=0.50)
        s0, _, _, _ = sizer.compute_size(
            portfolio_value=100000.0,
            cash=100000.0,
            buy_price=20.0,
            profile=self.profile_cn,
            consecutive_losses=0,
            hard_stop_loss=0.04,
        )
        s2, _, _, _ = sizer.compute_size(
            portfolio_value=100000.0,
            cash=100000.0,
            buy_price=20.0,
            profile=self.profile_cn,
            consecutive_losses=2,
            hard_stop_loss=0.04,
        )
        # 连亏 2 次以上应下调 50%
        self.assertEqual(s2, s0 // 2)

    def test_min_lot_guarantee(self):
        """测试一手保底机制（针对高单价标的）"""
        # 美股高单价 500 美元一股，本金 50,000 美元
        # 极小风险预算：0.0001 (5美元 < 单股止损 20 美元)，常规计算得 0 股
        sizer_micro = RiskBudgetSizer(
            risk_per_trade=0.0001,
            max_position_pct=0.30,
            min_lot_guarantee=True,
        )
        shares, _, _, _ = sizer_micro.compute_size(
            portfolio_value=50000.0,
            cash=50000.0,
            buy_price=500.0,
            profile=self.profile_us,
            hard_stop_loss=0.04,
        )
        self.assertEqual(shares, 1)  # 一手保底生效


class TestExitStateMachine(unittest.TestCase):
    """出场状态机测试"""

    def setUp(self):
        self.profile_cn = MarketProfile.for_stock("300418")
        self.profile_us = MarketProfile.for_stock("AAPL")
        self.sm = ExitStateMachine(
            hard_stop_loss=0.04,
            trailing_stop=0.025,
            trailing_stop_min_profit=0.02,
            max_holding_days=10,
            intraday_stop=True,
        )

    def test_intraday_stop_triggered(self):
        """测试盘中触及止损价触发平仓"""
        pos = Position(
            symbol="300418",
            shares=1000,
            avg_entry_price=20.0,
            stop_price=19.2,  # 4% 止损
            entry_bar_idx=0,
            holding_bars=1,
        )
        # 次日 bar (idx=1, T+1 满足)，当日最低价 19.0 触及 19.2
        row = SimpleNamespace(
            Index=pd.Timestamp("2026-01-06"),
            open=19.5,
            high=19.8,
            low=19.0,
            close=19.1,
            close_prev=20.0,
        )
        should_exit, reason, fill_px = self.sm.check_intraday_exit(
            row=row,
            position=pos,
            profile=self.profile_cn,
            idx=1,
            limit_down=0,
        )
        self.assertTrue(should_exit)
        self.assertIn("盘中止损", reason)
        self.assertEqual(fill_px, 19.2)

    def test_trailing_stop_eod(self):
        """测试收盘移动止盈触发（大幅冲高后回撤超过阈值）"""
        pos = Position(
            symbol="300418",
            shares=1000,
            avg_entry_price=20.0,
            peak_price=24.0,   # 冲高到 24 (+20%)
            holding_bars=3,
            target_horizon=7,
            entry_bar_idx=0,
            active_stop_pct=0.04,
            active_trail_pct=0.025,
        )
        # 收盘从 24 回撤到 23.2 (-3.33% > 2.5%)，浮盈 (23.2-20)/20 = 16% > 2%
        row = SimpleNamespace(
            Index=pd.Timestamp("2026-01-08"),
            close=23.2,
            close_prev=23.8,
            ma_fast=22.0,
            exec_bearish=0,
        )
        pending_reason, _ = self.sm.evaluate_eod_exit(
            row=row,
            position=pos,
            profile=self.profile_cn,
            idx=3,
        )
        self.assertIsNotNone(pending_reason)
        self.assertIn("移动止盈", pending_reason)

    def test_time_stop_model_falsified(self):
        """测试模型证伪时间止损（达到目标时界且处于浮亏）"""
        pos = Position(
            symbol="300418",
            shares=1000,
            avg_entry_price=20.0,
            peak_price=20.5,
            holding_bars=5,
            target_horizon=5,  # 达标持仓期
            entry_bar_idx=0,
            active_stop_pct=0.04,
            active_trail_pct=0.025,
        )
        # 收盘 19.8 处于浮亏 (-1%)
        row = SimpleNamespace(
            Index=pd.Timestamp("2026-01-10"),
            close=19.8,
            close_prev=19.9,
            ma_fast=20.1,
            exec_bearish=0,
        )
        pending_reason, _ = self.sm.evaluate_eod_exit(
            row=row,
            position=pos,
            profile=self.profile_cn,
            idx=5,
        )
        self.assertIsNotNone(pending_reason)
        self.assertIn("模型信号证伪", pending_reason)

    def test_momentum_transition(self):
        """测试【动量跃迁】：触达目标时界但趋势强劲，取消强平并移交移动止盈"""
        pos = Position(
            symbol="300418",
            shares=1000,
            avg_entry_price=20.0,
            peak_price=22.0,
            holding_bars=5,
            target_horizon=5,
            entry_bar_idx=0,
            active_stop_pct=0.04,
            active_trail_pct=0.025,
        )
        # 收盘 21.9（贴近最高价 22.0，回撤仅 -0.45%），站在快均线 21.0 之上，且浮盈 9.5%
        row = SimpleNamespace(
            Index=pd.Timestamp("2026-01-10"),
            close=21.9,
            close_prev=21.5,
            ma_fast=21.0,
            exec_bearish=0,
        )
        pending_reason, _ = self.sm.evaluate_eod_exit(
            row=row,
            position=pos,
            profile=self.profile_cn,
            idx=5,
        )
        # 动量跃迁会取消平仓 (pending_reason is None)
        self.assertIsNone(pending_reason)
        # 并且将 target_horizon 延长至最大持仓时界 10 日
        self.assertEqual(pos.target_horizon, 10)


class TestSimulatedBroker(unittest.TestCase):
    """仿真撮合交易经纪商测试"""

    def setUp(self):
        self.profile_cn = MarketProfile.for_stock("300418")
        self.broker = SimulatedBroker(
            profile=self.profile_cn,
            initial_capital=100000.0,
            slippage_rate=0.001,
        )

    def test_broker_execution_and_reset(self):
        """测试开仓买入、平仓卖出及持仓重置"""
        # 1. 开仓买入 1000 股 @ 20.0
        pos = self.broker.execute_buy(
            trade_date=pd.Timestamp("2026-01-05"),
            symbol="300418",
            open_price=20.0,
            shares_to_buy=1000,
            target_horizon=5,
            active_stop_pct=0.04,
            stop_price=19.2,
            bar_idx=0,
            close_price=20.5,
            low_price=19.8,
            active_trail_pct=0.025,
        )
        self.assertIsNotNone(pos)
        self.assertTrue(pos.is_open)
        self.assertEqual(pos.shares, 1000)
        self.assertIn("300418", self.broker.positions)

        # 2. 平仓卖出 @ 22.0
        cash, fee, trade_pnl, pnl_amount = self.broker.execute_sell(
            trade_date=pd.Timestamp("2026-01-08"),
            symbol="300418",
            position=pos,
            fill_price=22.0,
            reason="波峰到达止盈",
            holding_days=3,
        )
        # 检查持仓是否复位归零
        self.assertFalse(pos.is_open)
        self.assertEqual(pos.shares, 0)
        self.assertEqual(pos.holding_bars, 0)
        self.assertGreater(cash, 100000.0)  # 盈利平仓
        self.assertGreater(pnl_amount, 0)
        self.assertEqual(len(self.broker.trades), 2)  # 1 buy + 1 sell


class TestBacktestEngineIntegration(unittest.TestCase):
    """回测引擎状态机逐日撮合闭环测试"""

    def test_engine_run_with_mock_data(self):
        """使用生成的数据流验证 run_backtest_engine 状态机流转"""
        dates = pd.date_range("2026-01-01", periods=20, freq="B")
        prices = [10.0 + i * 0.2 for i in range(20)]
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1000000] * 20,
            # 第 2 天触发买入
            "exec_signal": [1 if i == 2 else 0 for i in range(20)],
            "target_horizon": [5] * 20,
            "exec_position_scale": [1.0] * 20,
            "atr_pct": [0.02] * 20,
            "open_limit_up": [0] * 20,
            "open_limit_down": [0] * 20,
            "close_prev": [prices[max(0, i-1)] for i in range(20)],
            "ma_fast": [p * 0.99 for p in prices],
            "exec_bearish": [0] * 20,
        }, index=dates)

        backtester = KronosBacktesterPro(
            data_dir=str(BACKTEST_DIR / "data"),
            model_dir=str(BACKTEST_DIR / "yuce"),
            initial_capital=50000.0,
            market_profile=MarketProfile.for_stock("AAPL"),
            enable_log=False,
        )
        backtester.current_stock_code = "AAPL"

        backtest_df, trades = backtester.run_backtest_engine(
            df=df,
            hard_stop_loss=0.04,
            trailing_stop=0.025,
            max_holding_days=5,
            min_lot_guarantee=True,
        )
        self.assertGreater(len(backtest_df), 0)
        self.assertGreater(len(trades), 0)
        # 买入成交
        self.assertEqual(trades[0]["action"], "BUY")


if __name__ == "__main__":
    unittest.main()
