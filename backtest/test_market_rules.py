"""
三市场交易规则与组合合成单测（无网络、不跑模型）。
可用: python -m unittest backtest.test_market_rules -v
或:   pytest backtest/test_market_rules.py -v
"""
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
for p in (str(ROOT), str(EXAMPLES), str(BACKTEST_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backtest.core.market_rules import (  # noqa: E402
    SymbolResolver,
    MarketProfile,
    HK_STAMP_DUTY_CUTOVER,
    compute_open_limit_flags,
    validate_stock_code,
)


def _approx(a, b, rel=1e-6):
    return abs(a - b) <= max(abs(a), abs(b), 1.0) * rel


class TestSymbolResolve(unittest.TestCase):
    def test_us_class_shares(self):
        clean, mkt, _ = SymbolResolver.resolve('BRK.B')
        self.assertEqual(mkt, 'US')
        self.assertEqual(clean, 'BRK.B')
        clean2, mkt2, _ = SymbolResolver.resolve('BRK-B')
        self.assertEqual(mkt2, 'US')
        self.assertEqual(clean2, 'BRK.B')

    def test_prefixes(self):
        self.assertEqual(SymbolResolver.resolve('HK_00700')[0:2], ('00700', 'HK'))
        self.assertEqual(SymbolResolver.resolve('US_AAPL')[0:2], ('AAPL', 'US'))
        self.assertEqual(SymbolResolver.resolve('300418')[1], 'CN')

    def test_validate_aligned(self):
        ok, _, code, mkt = validate_stock_code('HK_00700')
        self.assertTrue(ok)
        self.assertEqual((code, mkt), ('00700', 'HK'))
        ok, _, code, mkt = validate_stock_code('BRK.B')
        self.assertTrue(ok)
        self.assertEqual(mkt, 'US')


class TestShareRounding(unittest.TestCase):
    def test_cn_boards(self):
        star = MarketProfile.for_stock('688001')
        self.assertEqual(star.lot_size, 200)
        self.assertEqual(star.round_shares(250), 250)
        self.assertEqual(star.round_shares(150), 0)

        chi = MarketProfile.for_stock('300418')
        self.assertEqual(chi.round_shares(150), 150)

        main = MarketProfile.for_stock('600036')
        self.assertEqual(main.round_shares(150), 100)
        self.assertEqual(main.round_shares(99), 0)

    def test_bse_and_st_limits(self):
        bse = MarketProfile.for_stock('830799')
        self.assertAlmostEqual(bse.price_limit_pct, 0.30)
        st = MarketProfile.for_stock('600000', stock_name='*ST示例')
        self.assertAlmostEqual(st.price_limit_pct, 0.05)
        normal = MarketProfile.for_stock('600000', stock_name='浦发银行')
        self.assertAlmostEqual(normal.price_limit_pct, 0.10)

    def test_hk_lots(self):
        self.assertEqual(MarketProfile.for_stock('00941').lot_size, 500)
        self.assertEqual(MarketProfile.for_stock('00700').lot_size, 100)
        self.assertEqual(MarketProfile.for_stock('00700').round_shares(250), 200)


class TestFees(unittest.TestCase):
    def test_hk_stamp_cutover(self):
        hk = MarketProfile.for_stock('00700')
        gross = 100_000.0
        before = date(2023, 11, 16)
        self.assertLess(before, HK_STAMP_DUTY_CUTOVER)
        buy_before = hk.buy_cost(gross, trade_date=before)
        buy_after = hk.buy_cost(gross, trade_date=date(2023, 11, 17))
        self.assertTrue(_approx(buy_before, 80 + 130 + 10.5))
        self.assertTrue(_approx(buy_after, 80 + 100 + 10.5))
        self.assertGreater(buy_before, buy_after)

    def test_us_sell_exceeds_buy(self):
        us = MarketProfile.for_stock('AAPL')
        gross = 10_000.0
        buy = us.buy_cost(gross, trade_date=date(2024, 6, 1))
        sell = us.sell_cost(gross, trade_date=date(2024, 6, 1))
        self.assertAlmostEqual(buy, 0.0)
        self.assertGreater(sell, buy)
        self.assertTrue(_approx(sell, gross * 0.000028))

    def test_cn_transfer_fee(self):
        sh = MarketProfile.for_stock('600036')
        sz = MarketProfile.for_stock('000001')
        self.assertGreater(sh.transfer_fee_rate, 0)
        self.assertEqual(sz.transfer_fee_rate, 0.0)
        self.assertTrue(_approx(sh.buy_cost(100_000.0), 31.0))
        self.assertTrue(_approx(sz.buy_cost(100_000.0), 30.0))


class TestOpenLimit(unittest.TestCase):
    def test_open_not_close(self):
        """开盘未封、收盘封板可买；开盘封涨停不可买。"""
        prof = MarketProfile.for_stock('600036')
        df = pd.DataFrame({
            'open': [10.5, 11.00],
            'high': [11.00, 11.00],
            'low': [10.4, 10.9],
            'close': [11.00, 11.00],
            'close_prev': [10.0, 10.0],
        }, index=pd.to_datetime(['2024-01-02', '2024-01-03']))
        out = compute_open_limit_flags(df, prof)
        self.assertEqual(int(out['open_limit_up'].iloc[0]), 0)
        self.assertEqual(int(out['open_limit_up'].iloc[1]), 1)
        self.assertTrue(prof.is_open_limit_up(11.00, 10.0))
        self.assertFalse(prof.is_open_limit_up(10.5, 10.0))

    def test_annual_days(self):
        self.assertEqual(MarketProfile.for_stock('600036').annual_trading_days, 244)
        self.assertEqual(MarketProfile.for_stock('00700').annual_trading_days, 246)
        self.assertEqual(MarketProfile.for_stock('AAPL').annual_trading_days, 252)


class TestPortfolioCurve(unittest.TestCase):
    def test_skips_closed_market_days(self):
        from app_backtest_dashboard import calculate_portfolio_equity_curve

        with tempfile.TemporaryDirectory() as td:
            cn = os.path.join(td, '600036')
            hk = os.path.join(td, '00700')
            os.makedirs(cn)
            os.makedirs(hk)
            pd.DataFrame({
                'date': ['2024-01-02', '2024-01-04'],
                'portfolio_value': [100.0, 110.0],
            }).to_csv(os.path.join(cn, '600036_daily_backtest.csv'), index=False)
            pd.DataFrame({
                'date': ['2024-01-02', '2024-01-03', '2024-01-04'],
                'portfolio_value': [100.0, 105.0, 107.1],
            }).to_csv(os.path.join(hk, '00700_daily_backtest.csv'), index=False)

            curve = calculate_portfolio_equity_curve(td)
            self.assertIsNotNone(curve)
            # 01-03 仅港股 +5%；01-04 等权 (10%+2%)/2 = 6%
            self.assertTrue(_approx(curve['Portfolio_Basket'].iloc[0], 1.05))
            self.assertTrue(_approx(curve['Portfolio_Basket'].iloc[1], 1.05 * 1.06))


class TestSignalCausality(unittest.TestCase):
    def test_no_lookahead_on_same_day_close(self):
        """
        严谨因果律检验：
        早盘开盘时生成撮合信号（如入场判断、均线确认、Z-Score、去偏收益），
        在 t-1 日及之前数据、t 日开盘价与预测保持完全不变的前提下，
        人为极端篡改 t 日的收盘价（尚未发生的未来价格），
        验证 t 日早盘的入场决策与确认状态绝不受任何影响（杜绝 Lookahead bias）。
        """
        from run_backtest_kronos_pro import KronosBacktesterPro
        bt = KronosBacktesterPro(data_dir=str(BACKTEST_DIR), model_dir=str(BACKTEST_DIR))
        bt.set_market('600036', market='CN')

        dates = pd.date_range('2024-01-01', periods=60, freq='B')
        np.random.seed(42)
        base_p = 10.0 + np.cumsum(np.random.randn(60) * 0.1)

        hist_df = pd.DataFrame({
            'open': base_p,
            'high': base_p * 1.02,
            'low': base_p * 0.98,
            'close': base_p,
            'volume': 1000000,
        }, index=dates)

        pred_df = pd.DataFrame({
            'pred_close': base_p * 1.03,
            'pred_close_1d': base_p * 1.01,
            'pred_min_down': -0.02,
            'forecast_horizon': 5,
        }, index=dates)

        sig_orig = bt.align_and_generate_signals(hist_df.copy(), pred_df.copy())

        # 篡改第 40 天收盘价（从 10 翻倍至 20，模拟盘中大暴涨）
        hist_df_tampered = hist_df.copy()
        hist_df_tampered.iloc[40, hist_df_tampered.columns.get_loc('close')] = 20.0
        sig_tampered = bt.align_and_generate_signals(hist_df_tampered.copy(), pred_df.copy())

        t_date = dates[40]
        self.assertEqual(
            sig_orig.loc[t_date, 'confirm'],
            sig_tampered.loc[t_date, 'confirm'],
            "均线短期确认 (confirm) 泄露了当天收盘价！"
        )
        self.assertEqual(
            sig_orig.loc[t_date, 'exec_signal'],
            sig_tampered.loc[t_date, 'exec_signal'],
            "早盘执行信号 (exec_signal) 泄露了当天收盘价！"
        )
        self.assertAlmostEqual(
            sig_orig.loc[t_date, 'signal_score'],
            sig_tampered.loc[t_date, 'signal_score'],
            places=6,
            msg="信号强度 (signal_score) 泄露了当天收盘价！"
        )
        self.assertAlmostEqual(
            sig_orig.loc[t_date, 'expected_return_adj'],
            sig_tampered.loc[t_date, 'expected_return_adj'],
            places=6,
            msg="去偏后预期收益 (expected_return_adj) 泄露了当天收盘价！"
        )

    def test_warmup_period_no_future_leakage(self):
        """
        验证滚动自校准预热期杜绝未来函数：
        在样本前 1~10 天样本不足时，严禁使用全样本 quantile 填充，
        篡改未来（第 20 天起）的数据绝对不能改变前 10 天的预热期门槛与信号状态。
        """
        from run_backtest_kronos_pro import KronosBacktesterPro
        bt = KronosBacktesterPro(data_dir=str(BACKTEST_DIR), model_dir=str(BACKTEST_DIR))
        bt.set_market('600036', market='CN')

        dates = pd.date_range('2024-01-01', periods=50, freq='B')
        base_p = 10.0 + np.sin(np.linspace(0, 3.14 * 4, 50)) * 0.5

        hist_df = pd.DataFrame({
            'open': base_p,
            'high': base_p * 1.01,
            'low': base_p * 0.99,
            'close': base_p,
            'volume': 1000000,
        }, index=dates)

        pred_df = pd.DataFrame({
            'pred_close': base_p * 1.05,
            'pred_close_1d': base_p * 1.01,
            'pred_min_down': -0.01,
            'forecast_horizon': 5,
        }, index=dates)

        sig_orig = bt.align_and_generate_signals(hist_df.copy(), pred_df.copy())

        # 极端篡改第 25 天后的价格（未来数据翻三倍）
        hist_df_tampered = hist_df.copy()
        hist_df_tampered.iloc[25:, hist_df_tampered.columns.get_loc('close')] *= 3.0
        hist_df_tampered.iloc[25:, hist_df_tampered.columns.get_loc('open')] *= 3.0
        hist_df_tampered.iloc[25:, hist_df_tampered.columns.get_loc('high')] *= 3.0
        hist_df_tampered.iloc[25:, hist_df_tampered.columns.get_loc('low')] *= 3.0

        pred_df_tampered = pred_df.copy()
        pred_df_tampered.iloc[25:, pred_df_tampered.columns.get_loc('pred_close')] *= 3.0

        sig_tampered = bt.align_and_generate_signals(hist_df_tampered.copy(), pred_df_tampered.copy())

        # 检验对齐后的前 15 天数据（严格处于第 25 天篡改点之前，杜绝未来暴涨渗透到历史信号）
        valid_dates = list(sig_orig.index)[:15]
        for d in valid_dates:
            self.assertEqual(sig_orig.loc[d, 'exec_signal'], sig_tampered.loc[d, 'exec_signal'])
            self.assertEqual(sig_orig.loc[d, 'veto_reason'], sig_tampered.loc[d, 'veto_reason'])
            self.assertAlmostEqual(sig_orig.loc[d, 'signal_score'], sig_tampered.loc[d, 'signal_score'], places=5)
            self.assertAlmostEqual(sig_orig.loc[d, 'entry_threshold'], sig_tampered.loc[d, 'entry_threshold'], places=5)

        # 全时段因果性截断探针验证
        res = bt.check_causality(hist_df, pred_df, n_probe=8)
        self.assertTrue(res.get('passed', False), f"因果律检验未通过: {res.get('不一致项')}")



class TestLimitDownStopLossProtection(unittest.TestCase):
    def test_stop_loss_blocked_on_limit_down_lock(self):
        """
        跌停板流动性防线检验：
        持仓期间若盘中触及止损价，但当天收盘被一字跌停或封死跌停板，
        撮合引擎严禁在当日虚构止损成交，必须阻断并顺延至次日流动性恢复后撮合。
        """
        from run_backtest_kronos_pro import KronosBacktesterPro
        bt = KronosBacktesterPro(data_dir=str(BACKTEST_DIR), model_dir=str(BACKTEST_DIR))
        bt.set_market('600036', market='CN')

        dates = pd.date_range('2024-01-01', periods=4, freq='B')
        # Day 0: 买入
        # Day 1: 盘中击穿止损(low=8.5)，但收盘封死跌停(close=9.0, is_close_limit_down=True)
        # Day 2: 跌停打开(open=8.8, low=8.7, high=9.1, close=9.0)，允许卖出撮合
        signal_df = pd.DataFrame({
            'open': [10.0, 9.8, 8.8, 8.8],
            'high': [10.2, 9.8, 9.1, 8.9],
            'low': [9.9, 8.5, 8.7, 8.7],
            'close': [10.0, 9.0, 9.0, 8.8],
            'close_prev': [10.0, 10.0, 9.0, 9.0],
            'open_limit_up': [0, 0, 0, 0],
            'close_limit_up': [0, 0, 0, 0],
            'open_limit_down': [0, 0, 0, 0],
            'close_limit_down': [0, 1, 0, 0],
            'is_close_limit_down': [False, True, False, False],
            'long_entry': [True, False, False, False],
            'exec_signal': [1, 0, 0, 0],
            'exec_bearish': [0, 0, 0, 0],
            'exec_target_horizon': [5, 5, 5, 5],
            'exec_position_scale': [1.0, 0.0, 0.0, 0.0],
            'signal_score': [2.0, 0.0, 0.0, 0.0],
            'expected_return_adj': [0.05, 0.0, 0.0, 0.0],
            'pred_min_down': [-0.01, -0.05, -0.02, -0.02],
            'ma_fast': [9.5, 9.5, 9.5, 9.5],
            'ma_slow': [9.0, 9.0, 9.0, 9.0],
            'forecast_horizon': [5, 5, 5, 5],
            'target_horizon': [5, 5, 5, 5],
        }, index=dates)

        backtest_df, trades = bt.run_backtest_engine(
            signal_df, hard_stop_loss=0.04, intraday_stop=True,
            risk_per_trade=1.0, max_position_pct=1.0
        )

        # 验证 Day 1 (dates[1]) 没有成交止损卖出，平仓成交只能顺延在 Day 2 (dates[2])
        self.assertEqual(len(trades), 2)  # [BUY, SELL]
        sell_trade = trades[1]
        self.assertEqual(sell_trade['action'], 'SELL')
        self.assertEqual(pd.to_datetime(sell_trade['date']), dates[2], "未在跌停打开的次日成交，或在跌停当日违规成交！")


class TestDebiasHorizonConsistency(unittest.TestCase):
    def test_skill_evaluation_using_1d_pred(self):
        """
        预测时界与技能体检量纲一致性检验：
        当存在 pred_close_1d 时，体检优先对齐 1 日真实收益，IC 评估口径为 '1日预期'。
        """
        from run_backtest_kronos_pro import KronosBacktesterPro
        bt = KronosBacktesterPro(data_dir=str(BACKTEST_DIR), model_dir=str(BACKTEST_DIR))

        dates = pd.date_range('2024-01-01', periods=30, freq='B')
        ret_real = np.linspace(-0.02, 0.05, 30)
        ret_pred = ret_real + 0.005  # 正相关单调序列
        base_prev = 10.0
        df = pd.DataFrame({
            'close': base_prev * (1.0 + ret_real),
            'close_prev': [base_prev] * 30,
            'pred_close': base_prev * (1.0 + ret_pred * 2.0),     # 5日多日预测
            'pred_close_1d': base_prev * (1.0 + ret_pred),         # 1日单日预测
        }, index=dates)

        skill = bt.evaluate_prediction_skill(df)
        self.assertEqual(skill.get('评估口径'), '1日预期')
        self.assertGreater(skill.get('IC', 0), 0.9)  # 严格单调正相关，IC 应当接近 1.0 且显著


class TestReversalAwareDebias(unittest.TestCase):
    def test_reversal_aware_debias_behavior(self):
        """
        验证反转感知去偏 (方案 1)：
        1. 在下跌背景 (ma_fast < ma_slow) 且模型预测向上反转时：
           历史积累的正向偏差 (pred_bias > 0) 不应惩罚反转信号，
           expected_return_adj 保持等于 expected_return (reversal_debias_weight=0.0)。
        2. 当关闭反转感知 (reversal_aware_debias=False) 时，
           expected_return_adj 会被扣除 pred_bias。
        3. 在顺势上涨背景 (ma_fast >= ma_slow) 且模型预测向上时：
           依然严格执行去偏，扣除同向泡沫。
        """
        from run_backtest_kronos_pro import KronosBacktesterPro
        bt = KronosBacktesterPro(data_dir=str(BACKTEST_DIR), model_dir=str(BACKTEST_DIR))

        # 构建一段持续下跌的市场数据以形成均线空头排列和正向预测误差 (pred > real)
        dates = pd.date_range('2024-01-01', periods=80, freq='B')
        prices = 100.0 * np.exp(-0.005 * np.arange(80))  # 持续下行
        hist_df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': 10000.0,
            'amount': 1000000.0,
        }, index=dates)

        # 预测数据：前 70 天持续稍微高估，在第 75 天产生强反转预期
        pred_close = prices * 1.02  # 持续存在 +2% 的高估
        pred_close_1d = prices * 1.01
        # 第 75 天 (dates[75]): 给出强烈的底部反转信号
        pred_close[75] = prices[75] * 1.05
        pred_close_1d[75] = prices[75] * 1.03

        pred_df = pd.DataFrame({
            'pred_close': pred_close,
            'pred_close_1d': pred_close_1d,
            'pred_slope': np.full(80, 0.005),  # 预测斜率向上
            'target_horizon': np.full(80, 5),
        }, index=dates)

        # 1. 开启反转感知去偏 (默认行为)
        sig_reversal = bt.align_and_generate_signals(
            hist_df.copy(), pred_df.copy(),
            reversal_aware_debias=True,
            reversal_debias_weight=0.0
        )
        t_rev = dates[75]
        self.assertTrue(sig_reversal.loc[t_rev, 'is_bull_reversal'])
        self.assertAlmostEqual(
            sig_reversal.loc[t_rev, 'expected_return_adj'],
            sig_reversal.loc[t_rev, 'expected_return'],
            places=5,
            msg="反转感知模式下，底部多头反转预期不应被历史高估偏差扣减！"
        )

        # 2. 关闭反转感知去偏 (机械去偏对比)
        sig_mechanical = bt.align_and_generate_signals(
            hist_df.copy(), pred_df.copy(),
            reversal_aware_debias=False
        )
        self.assertLess(
            sig_mechanical.loc[t_rev, 'expected_return_adj'],
            sig_mechanical.loc[t_rev, 'expected_return'],
            msg="机械去偏模式下，历史高估偏差会被机械扣除！"
        )


class TestZScoreEntryMode(unittest.TestCase):
    def test_z_score_entry_mode_behavior(self):
        """
        验证方案 A (entry_mode='z_score')：
        1. 默认无需 60 天预热期，样本初期即可正常评估信噪比；
        2. signal_score > z_score_threshold 时通过门槛审查；
        3. signal_score <= z_score_threshold 时否决原因明确为信噪比未达标。
        """
        from run_backtest_kronos_pro import KronosBacktesterPro
        bt = KronosBacktesterPro(data_dir=str(BACKTEST_DIR), model_dir=str(BACKTEST_DIR))

        dates = pd.date_range('2024-01-01', periods=30, freq='B')
        base_p = 10.0 + np.linspace(0, 1.0, 30)
        hist_df = pd.DataFrame({
            'open': base_p,
            'high': base_p * 1.01,
            'low': base_p * 0.99,
            'close': base_p,
            'volume': 100000.0,
            'amount': 1000000.0,
        }, index=dates)

        # 预测：前 15 天弱信号 (预期涨幅仅 0.5%)，后 15 天强信号 (预期涨幅 5%)
        pred_close = base_p.copy()
        pred_close[:15] *= 1.005
        pred_close[15:] *= 1.05

        pred_df = pd.DataFrame({
            'pred_close': pred_close,
            'pred_slope': np.full(30, 0.005),
            'target_horizon': np.full(30, 5),
        }, index=dates)

        # 默认模式即为方案 A (entry_mode='z_score', z_score_threshold=0.70)
        sig = bt.align_and_generate_signals(hist_df, pred_df, z_score_threshold=0.70)

        # 1. 验证不存在预热期不足的否决
        self.assertFalse((sig['veto_reason'] == '滚动校准窗口预热期不足').any())

        # 2. 验证门槛值稳定在 0.70 左右
        valid_idx = sig.dropna(subset=['signal_score']).index
        self.assertTrue((sig.loc[valid_idx, 'entry_threshold'] >= 0.70).all())

        # 3. 验证后半段高信噪比日期的入场放行
        high_score_days = sig[sig['signal_score'] > sig['entry_threshold']]
        self.assertGreater(len(high_score_days), 0)


if __name__ == '__main__':
    unittest.main()



