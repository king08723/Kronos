"""
backtest/test_phase1_multitimeframe.py
Phase 1 核心领域与交易日历重构专有单元测试。
测试覆盖：
1. TimeFrame 多时间粒度 (1m/5m/15m/30m/60m/1d/1w) 属性推导、频率字符串与年化因子校准。
2. Position 自然日与 Bar 步长双时钟推进逻辑 (holding_bars & holding_days)。
3. A 股 T+1 交易制度在高频分钟线下的严格合规锁定：
   - 同一交易自然日内无论经过多少根分钟 Bar，坚决拒绝平仓。
   - 跨越到次日交易时段首 Bar (如 09:30)，立刻解除锁定允许撮合。
4. 港股/美股 T+0 制度在高频分钟线下的及时出场：同日同分钟触及止损立即成交。
5. ExitStateMachine 离散出场状态机在高频分钟线下的 T+1 锁定与次日顺延撮合验证。
"""

import unittest
from datetime import datetime
import pandas as pd
import numpy as np

from backtest.core.types import TimeFrame, Position, OrderSide, OrderType, TradeRecord
from backtest.core.market_rules import MarketProfile
from backtest.core.timeframes import is_same_trading_day, get_trading_days_passed, get_annualization_factor
from backtest.execution.risk_manager import ExitStateMachine
from backtest.execution.simulated_broker import SimulatedBroker


class TestTimeFrameCore(unittest.TestCase):
    """测试多时间粒度属性与转换"""

    def test_timeframe_properties(self):
        tf_1m = TimeFrame.from_str("1m")
        self.assertEqual(tf_1m, TimeFrame.M1)
        self.assertEqual(tf_1m.minutes, 1)
        self.assertTrue(tf_1m.is_intraday)
        self.assertEqual(tf_1m.pandas_freq, "1min")
        self.assertEqual(tf_1m.label, "1分钟")
        self.assertEqual(tf_1m.bars_per_day("CN"), 240)
        self.assertEqual(tf_1m.bars_per_day("HK"), 330)
        self.assertEqual(tf_1m.bars_per_day("US"), 390)

        tf_30m = TimeFrame.from_str("30min")
        self.assertEqual(tf_30m, TimeFrame.M30)
        self.assertEqual(tf_30m.minutes, 30)
        self.assertTrue(tf_30m.is_intraday)
        self.assertEqual(tf_30m.pandas_freq, "30min")
        self.assertEqual(tf_30m.bars_per_day("CN"), 8)

        tf_60m = TimeFrame.from_str("1h")
        self.assertEqual(tf_60m, TimeFrame.H1)
        self.assertEqual(tf_60m.minutes, 60)
        self.assertTrue(tf_60m.is_intraday)
        self.assertEqual(tf_60m.pandas_freq, "60min")
        self.assertEqual(tf_60m.bars_per_day("CN"), 4)

        tf_1d = TimeFrame.from_str("daily")
        self.assertEqual(tf_1d, TimeFrame.D1)
        self.assertFalse(tf_1d.is_intraday)
        self.assertEqual(tf_1d.pandas_freq, "1D")
        self.assertEqual(tf_1d.bars_per_day("CN"), 1)

    def test_annual_factor_scaling(self):
        # A 股 244 交易日，1 分钟线年化因子应为 sqrt(244 * 240) = sqrt(58560) ≈ 242.0
        f_1m = get_annualization_factor("1m", market="CN")
        self.assertAlmostEqual(f_1m, np.sqrt(244 * 240), places=3)

        # 日线年化因子 sqrt(244) ≈ 15.62
        f_1d = get_annualization_factor("1d", market="CN")
        self.assertAlmostEqual(f_1d, np.sqrt(244), places=3)


class TestPositionDualClock(unittest.TestCase):
    """测试 Position 的自然日与 Bar 步长双时钟系统"""

    def test_holding_bars_and_days_progression(self):
        pos = Position(
            symbol="600036",
            shares=1000,
            avg_entry_price=30.0,
            entry_date=pd.Timestamp("2026-03-02 09:35:00"),
            holding_bars=0,
            holding_days=0,
        )

        # 同一天经过 10 根分钟线
        for i in range(1, 11):
            pos.update_market_price(
                high=30.5, low=29.8, close=30.2,
                current_time=pd.Timestamp(f"2026-03-02 09:{35+i:02d}:00")
            )
        self.assertEqual(pos.holding_bars, 10)
        self.assertEqual(pos.holding_days, 0)  # 同日内天数跨度仍为 0

        # 次日开盘第 1 根 Bar
        pos.update_market_price(
            high=31.0, low=30.1, close=30.8,
            current_time=pd.Timestamp("2026-03-03 09:30:00")
        )
        self.assertEqual(pos.holding_bars, 11)
        self.assertEqual(pos.holding_days, 1)  # 跨过 1 个自然日

    def test_t_plus_one_strict_enforcement_in_minute_bars(self):
        """A 股 1 分钟线下 T+1 严格防偷跑测试"""
        pos_cn = Position(
            symbol="300418",
            shares=2000,
            avg_entry_price=20.0,
            entry_date=pd.Timestamp("2026-03-02 09:35:00"),
            entry_bar_idx=5,
        )

        # 1. 在当天 09:36 (idx=6)，虽然 idx - entry_bar_idx = 1，但因同属于 2026-03-02，严禁平仓！
        can_exit_min1 = pos_cn.can_exit(
            current_time=pd.Timestamp("2026-03-02 09:36:00"),
            settlement_days=1,
            current_idx=6,
        )
        self.assertFalse(can_exit_min1, "A 股同日第 1 分钟禁止平仓")

        # 2. 在当天收盘 15:00 (idx=240)，即使过了 235 根 Bar，严禁平仓！
        can_exit_eod = pos_cn.can_exit(
            current_time=pd.Timestamp("2026-03-02 15:00:00"),
            settlement_days=1,
            current_idx=240,
        )
        self.assertFalse(can_exit_eod, "A 股同日收盘 Bar 同样禁止平仓")

        # 3. 次日 2026-03-03 09:30 (idx=241)，自然日跨越，解除锁定允许平仓！
        can_exit_next_day = pos_cn.can_exit(
            current_time=pd.Timestamp("2026-03-03 09:30:00"),
            settlement_days=1,
            current_idx=241,
        )
        self.assertTrue(can_exit_next_day, "A 股次日开盘首 Bar 应准许平仓")

    def test_t_plus_zero_in_us_hk(self):
        """美股/港股 T+0 制度在高频下允许同日平仓"""
        pos_us = Position(
            symbol="AAPL",
            shares=100,
            avg_entry_price=180.0,
            entry_date=pd.Timestamp("2026-03-02 09:35:00"),
            entry_bar_idx=5,
        )
        # 美股 settlement_days=0，在下一分钟即可平仓
        can_exit = pos_us.can_exit(
            current_time=pd.Timestamp("2026-03-02 09:36:00"),
            settlement_days=0,
            current_idx=6,
        )
        self.assertTrue(can_exit, "美股 T+0 允许同日同分钟平仓")


class TestExitStateMachineIntraday(unittest.TestCase):
    """测试状态机在高频分钟线下的止损与顺延"""

    def setUp(self):
        self.profile_cn = MarketProfile.for_stock("300418")
        self.profile_us = MarketProfile.for_stock("AAPL")
        self.sm = ExitStateMachine(
            hard_stop_loss=0.04,
            trailing_stop=0.025,
            intraday_stop=True,
        )

    def test_a_share_intraday_stop_blocked_on_same_day(self):
        """A 股 1 分钟线当天买入后触及止损价，应被 T+1 规则拦截，禁止平仓"""
        pos = Position(
            symbol="300418",
            shares=1000,
            avg_entry_price=20.0,
            stop_price=19.2,  # -4% 止损线
            entry_date=pd.Timestamp("2026-03-02 09:35:00"),
            entry_bar_idx=5,
        )

        class MockBar:
            Index = pd.Timestamp("2026-03-02 10:15:00")
            close = 19.1
            low = 19.0        # 盘中大幅击穿止损价 19.2
            open = 19.3
            limit_down = 0

        row = MockBar()
        should_exit, reason, fill = self.sm.check_intraday_exit(
            row=row,
            position=pos,
            profile=self.profile_cn,
            idx=45,
            limit_down=0,
        )
        self.assertFalse(should_exit, "A 股同日触及止损价不得触发盘中平仓")

    def test_a_share_intraday_stop_executes_on_next_day(self):
        """次日触及止损价或顺延止损应准许平仓"""
        pos = Position(
            symbol="300418",
            shares=1000,
            avg_entry_price=20.0,
            stop_price=19.2,
            entry_date=pd.Timestamp("2026-03-02 09:35:00"),
            entry_bar_idx=5,
        )

        class MockBarNextDay:
            Index = pd.Timestamp("2026-03-03 09:31:00")
            close = 19.1
            low = 19.0
            open = 19.1
            limit_down = 0

        row = MockBarNextDay()
        should_exit, reason, fill = self.sm.check_intraday_exit(
            row=row,
            position=pos,
            profile=self.profile_cn,
            idx=241,
            limit_down=0,
        )
        self.assertTrue(should_exit, "次日触及止损线应准许触发平仓")
        self.assertIn("盘中止损", reason)


if __name__ == "__main__":
    unittest.main()
