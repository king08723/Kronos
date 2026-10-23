"""
test_phase4_mtf_strategy.py
Phase 4 专项测试套件：策略层跨周期 (MTF) 对齐与高频特征工程

覆盖内容:
1. MultiTimeframeAligner.align_higher_to_lower 严格因果律检验 (日线对齐日内，绝对零未来函数)
2. MultiTimeframeAligner.align_higher_to_lower 通用高频对齐 (60m 对齐 15m，严格点对点因果守恒)
3. MultiTimeframeAligner.auto_align_dual_timeframe 自动合成日线并对齐双周期共振特征
4. MultiTimeframeAligner.broadcast_daily_to_intraday 日度预测广播至日内所有 Bar
5. KronosStrategy 对高频时钟与日度预测的自适应检测与广播映射
6. KronosStrategy 高频年化波动率因子与自适应窗口缩放检验
7. KronosStrategy 宏观趋势滤网 (MTF Trend Filter) 准入与否决矩阵检验
8. KronosBacktesterPro 跨周期端到端完整闭环回测验证
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

from backtest.core.types import TimeFrame
from backtest.core.market_rules import MarketProfile
from backtest.core.mtf_aligner import MultiTimeframeAligner
from backtest.strategies.kronos_strategy import KronosStrategy
from backtest.engine.backtest_engine import KronosBacktesterPro


class TestMultiTimeframeAligner(unittest.TestCase):
    """测试跨周期特征对齐器 (MultiTimeframeAligner) 的严格因果律"""

    def setUp(self):
        # 构造 3 个交易日的日线数据 (2026-03-02, 2026-03-03, 2026-03-04)
        daily_dates = pd.to_datetime(['2026-03-02', '2026-03-03', '2026-03-04'])
        self.daily_df = pd.DataFrame({
            'close': [10.0, 11.0, 12.0],
            'ma20': [9.5, 10.2, 10.8],
        }, index=daily_dates)

        # 构造 2026-03-03 与 2026-03-04 的 30m 高频 Bar (每个交易日 8 根)
        times = []
        for d in ['2026-03-03', '2026-03-04']:
            times.extend(pd.to_datetime([
                f'{d} 10:00:00', f'{d} 10:30:00', f'{d} 11:00:00', f'{d} 11:30:00',
                f'{d} 13:30:00', f'{d} 14:00:00', f'{d} 14:30:00', f'{d} 15:00:00'
            ]))
        self.intraday_df = pd.DataFrame({
            'open': np.linspace(10.5, 12.5, len(times)),
            'close': np.linspace(10.6, 12.6, len(times)),
            'volume': 10000,
        }, index=pd.DatetimeIndex(times))

    def test_daily_to_intraday_causality(self):
        """测试日线对齐到日内 30m 时，100% 杜绝未来函数 (t 日任何时刻只能看到 t-1 日收盘指标)"""
        aligned = MultiTimeframeAligner.align_higher_to_lower(
            lower_df=self.intraday_df,
            higher_df=self.daily_df,
            prefix="htf_",
            shift_higher=True,
        )

        # 检验对齐后行数保持不变
        self.assertEqual(len(aligned), len(self.intraday_df))
        self.assertIn('htf_ma20', aligned.columns)
        self.assertIn('htf_close', aligned.columns)

        # 检验 2026-03-03 的所有 8 根 Bar，其 htf_ma20 必须全部等于 2026-03-02 的 ma20 (9.5)
        day1_bars = aligned.loc['2026-03-03']
        self.assertEqual(len(day1_bars), 8)
        for _, row in day1_bars.iterrows():
            self.assertEqual(row['htf_ma20'], 9.5)
            self.assertEqual(row['htf_close'], 10.0)

        # 检验 2026-03-04 的所有 8 根 Bar，其 htf_ma20 必须全部等于 2026-03-03 的 ma20 (10.2)
        day2_bars = aligned.loc['2026-03-04']
        self.assertEqual(len(day2_bars), 8)
        for _, row in day2_bars.iterrows():
            self.assertEqual(row['htf_ma20'], 10.2)
            self.assertEqual(row['htf_close'], 11.0)

    def test_intraday_to_intraday_causality(self):
        """测试 60m 对齐到 15m 时的因果律守恒"""
        times_60m = pd.to_datetime(['2026-03-02 10:30', '2026-03-02 11:30', '2026-03-02 14:00', '2026-03-02 15:00'])
        df_60m = pd.DataFrame({'val_60': [100.0, 110.0, 120.0, 130.0]}, index=times_60m)

        times_15m = pd.to_datetime([
            '2026-03-02 10:15', '2026-03-02 10:30', '2026-03-02 10:45',
            '2026-03-02 11:00', '2026-03-02 11:15', '2026-03-02 11:30'
        ])
        df_15m = pd.DataFrame({'val_15': range(len(times_15m))}, index=times_15m)

        # shift_higher=True 严格排除当前正在闭合的 Bar (h_time < l_time)
        aligned_strict = MultiTimeframeAligner.align_higher_to_lower(
            lower_df=df_15m,
            higher_df=df_60m,
            prefix="htf_",
            shift_higher=True,
        )
        # 10:15 与 10:30 时，均未有在当前时刻之前完成的 60m Bar，必须为 NaN
        self.assertTrue(pd.isna(aligned_strict.loc['2026-03-02 10:15:00', 'htf_val_60']))
        self.assertTrue(pd.isna(aligned_strict.loc['2026-03-02 10:30:00', 'htf_val_60']))
        # 10:45 时，10:30 闭合的 60m Bar 已成为历史，值应为 100.0
        self.assertEqual(aligned_strict.loc['2026-03-02 10:45:00', 'htf_val_60'], 100.0)
        self.assertEqual(aligned_strict.loc['2026-03-02 11:00:00', 'htf_val_60'], 100.0)
        self.assertEqual(aligned_strict.loc['2026-03-02 11:15:00', 'htf_val_60'], 100.0)
        # 11:30 时，由于排除当前闭合的 11:30 Bar，依然取 10:30 闭合的 100.0
        self.assertEqual(aligned_strict.loc['2026-03-02 11:30:00', 'htf_val_60'], 100.0)

        # shift_higher=False 允许右闭合对齐 (h_time <= l_time)
        aligned_exact = MultiTimeframeAligner.align_higher_to_lower(
            lower_df=df_15m,
            higher_df=df_60m,
            prefix="htf_",
            shift_higher=False,
        )
        self.assertEqual(aligned_exact.loc['2026-03-02 10:30:00', 'htf_val_60'], 100.0)
        self.assertEqual(aligned_exact.loc['2026-03-02 11:30:00', 'htf_val_60'], 110.0)

    def test_auto_align_dual_timeframe(self):
        """测试 auto_align_dual_timeframe 单份 30m 数据内存无损聚合日线并计算双均线对齐"""
        days = ['2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05', '2026-03-06']
        times = []
        for d in days:
            times.extend(pd.to_datetime([
                f'{d} 10:00:00', f'{d} 10:30:00', f'{d} 11:00:00', f'{d} 11:30:00',
                f'{d} 13:30:00', f'{d} 14:00:00', f'{d} 14:30:00', f'{d} 15:00:00'
            ]))
        np.random.seed(42)
        close_prices = 10.0 + np.cumsum(np.random.randn(len(times)) * 0.1)
        df_30m = pd.DataFrame({
            'open': close_prices,
            'high': close_prices + 0.1,
            'low': close_prices - 0.1,
            'close': close_prices,
            'volume': 10000,
        }, index=pd.DatetimeIndex(times))

        aligned = MultiTimeframeAligner.auto_align_dual_timeframe(
            lower_df=df_30m,
            market="CN",
            ma_fast_window=2,
            ma_slow_window=3,
            prefix="htf_",
        )

        self.assertIn("htf_ma_fast", aligned.columns)
        self.assertIn("htf_ma_slow", aligned.columns)
        self.assertIn("htf_trend_up", aligned.columns)

        # 检验同一日内的所有 Bar 共享相同的宏观日线特征
        for d in days[3:]:
            sub = aligned.loc[d]
            self.assertEqual(sub["htf_ma_fast"].nunique(), 1)
            self.assertEqual(sub["htf_ma_slow"].nunique(), 1)
            self.assertEqual(sub["htf_trend_up"].nunique(), 1)

    def test_broadcast_daily_to_intraday(self):
        """测试 broadcast_daily_to_intraday 将日度预测精确广播至日内所有 Bar"""
        daily_preds = pd.DataFrame({
            'pred_close': [10.5, 11.2, 11.8],
            'target_horizon': [3, 4, 5],
        }, index=pd.to_datetime(['2026-03-02', '2026-03-03', '2026-03-04']))

        intraday_times = pd.date_range('2026-03-02 09:30', '2026-03-04 15:00', freq='30min')
        intraday_df = pd.DataFrame({'close': range(len(intraday_times))}, index=intraday_times)

        broadcasted = MultiTimeframeAligner.broadcast_daily_to_intraday(
            intraday_df=intraday_df,
            daily_df=daily_preds,
        )

        self.assertEqual(len(broadcasted), len(intraday_df))
        self.assertEqual(broadcasted.loc['2026-03-02 10:00:00', 'pred_close'], 10.5)
        self.assertEqual(broadcasted.loc['2026-03-02 14:30:00', 'pred_close'], 10.5)
        self.assertEqual(broadcasted.loc['2026-03-03 10:30:00', 'pred_close'], 11.2)
        self.assertEqual(broadcasted.loc['2026-03-04 15:00:00', 'pred_close'], 11.8)


class TestKronosStrategyMTFIntegration(unittest.TestCase):
    """测试 KronosStrategy 在多时间周期下的自适应特征工程与滤网"""

    def setUp(self):
        self.days = pd.date_range('2026-03-01', periods=15, freq='B')
        times = []
        for d in self.days:
            d_str = d.strftime('%Y-%m-%d')
            for t_str in ['10:00:00', '10:30:00', '11:00:00', '11:30:00', '13:30:00', '14:00:00', '14:30:00', '15:00:00']:
                times.append(pd.to_datetime(f'{d_str} {t_str}'))

        np.random.seed(42)
        prices = 20.0 + np.cumsum(np.random.randn(len(times)) * 0.1)
        self.hist_30m = pd.DataFrame({
            'open': prices,
            'high': prices + 0.1,
            'low': prices - 0.1,
            'close': prices,
            'volume': 50000,
        }, index=pd.DatetimeIndex(times))

        # 日度预测序列
        daily_closes = self.hist_30m.groupby(self.hist_30m.index.date)['close'].last()
        self.pred_daily = pd.DataFrame({
            'pred_close': [p * 1.03 for p in daily_closes],
            'target_horizon': [3] * len(self.days),
            'optimal_horizon': [3] * len(self.days),
            'pred_path_max': [p * 1.05 for p in daily_closes],
        }, index=pd.to_datetime([d.strftime('%Y-%m-%d') for d in self.days]))

    def test_kronos_strategy_auto_broadcast_daily_to_intraday(self):
        """测试 KronosStrategy 在面对日度预测与 30m 历史数据时，自动检测并广播对齐"""
        strat = KronosStrategy(profile=MarketProfile.for_market('CN'))
        sig_df = strat.align_and_generate_signals(
            hist_df=self.hist_30m,
            pred_df=self.pred_daily,
            min_overlap_days=5,
            use_mtf_trend_filter=True,
            dynamic_calib=False,
            calib_window=20,
        )

        self.assertGreater(len(sig_df), 50)
        self.assertIn('raw_signal', sig_df.columns)
        self.assertIn('signal_score', sig_df.columns)
        self.assertIn('htf_ma_fast', sig_df.columns)
        self.assertIn('htf_ma_slow', sig_df.columns)

    def test_annual_factor_scaling(self):
        """验证年化波动率精确使用 30m 年化折算因子，而非日线的 sqrt(252)"""
        strat = KronosStrategy(profile=MarketProfile.for_market('CN'))
        sig_df = strat.align_and_generate_signals(
            hist_df=self.hist_30m,
            pred_df=self.pred_daily,
            min_overlap_days=5,
            use_mtf_trend_filter=False,
            dynamic_calib=False,
            calib_window=20,
        )

        # 30m A 股年化因子为 sqrt(244 * 8) ≈ 44.181
        expected_factor = TimeFrame.M30.annual_factor(market='CN')
        self.assertAlmostEqual(expected_factor, np.sqrt(244 * 8), places=3)

        # 检查 sig_df['ann_vol'] 是否等于 volatility * expected_factor
        valid_row = sig_df.dropna(subset=['volatility', 'ann_vol']).iloc[-1]
        calc_ann_vol = valid_row['volatility'] * expected_factor
        self.assertAlmostEqual(valid_row['ann_vol'], calc_ann_vol, places=4)

    def test_mtf_trend_filter_rejection(self):
        """测试宏观大趋势滤网 (MTF Filter) 对下跌趋势的入场否决矩阵"""
        strat = KronosStrategy(profile=MarketProfile.for_market('CN'))
        sig_df = strat.align_and_generate_signals(
            hist_df=self.hist_30m,
            pred_df=self.pred_daily,
            min_overlap_days=5,
            use_mtf_trend_filter=True,
            dynamic_calib=False,
            calib_window=20,
        )

        # 筛选下跌趋势且无短期确认的行
        down_trend_rows = sig_df[(sig_df['htf_ma_fast'] < sig_df['htf_ma_slow']) & (sig_df['close_prev'] <= sig_df['htf_ma_fast'])]
        if len(down_trend_rows) > 0:
            # 必须全部被一票否决
            for _, r in down_trend_rows.iterrows():
                self.assertEqual(r['raw_signal'], 0)
                self.assertIn('下跌趋势且无短期确认', r['veto_reason'])


class TestBacktesterProMTFEndToEnd(unittest.TestCase):
    """测试 KronosBacktesterPro 引擎层支持 timeframe 参数并闭环执行"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.temp_dir.name, "historical_data")
        self.pred_dir = os.path.join(self.temp_dir.name, "yuce")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

        # 构造 15 天 30m 数据并保存为 300418_30m.csv
        days = pd.date_range('2026-03-01', periods=15, freq='B')
        times = []
        for d in days:
            d_str = d.strftime('%Y-%m-%d')
            for t_str in ['10:00:00', '10:30:00', '11:00:00', '11:30:00', '13:30:00', '14:00:00', '14:30:00', '15:00:00']:
                times.append(pd.to_datetime(f'{d_str} {t_str}'))

        np.random.seed(42)
        close_prices = 20.0 + np.cumsum(np.random.randn(len(times)) * 0.1)
        hist_df = pd.DataFrame({
            'date': times,
            'open': close_prices,
            'high': close_prices + 0.15,
            'low': close_prices - 0.15,
            'close': close_prices,
            'volume': 100000,
            'amount': close_prices * 100000,
        })
        hist_df.to_csv(os.path.join(self.data_dir, "300418_30m.csv"), index=False)

        # 构造预测文件
        daily_closes = hist_df.groupby(hist_df['date'].dt.date)['close'].last()
        pred_df = pd.DataFrame({
            'date': [d.strftime('%Y-%m-%d') for d in days],
            'pred_close': [p * 1.02 for p in daily_closes],
            'pred_path_max': [p * 1.04 for p in daily_closes],
            'optimal_horizon': [3] * len(days),
            'target_horizon': [3] * len(days),
        })
        pred_df.to_csv(os.path.join(self.pred_dir, "300418_predictions.csv"), index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_complete_backtest_with_30m_timeframe(self):
        """端到端验证 KronosBacktesterPro 传入 timeframe='30m' 能够成功加载、对齐并完成回测"""
        backtester = KronosBacktesterPro(
            data_dir=self.data_dir,
            model_dir=self.pred_dir,
            initial_capital=100000.0,
            enable_log=False,
        )

        metrics, signal_df, backtest_df, trades, fig = backtester.run_complete_backtest(
            stock_code="300418",
            output_dir=self.temp_dir.name,
            timeframe=TimeFrame.M30,
            show_plot=False,
            min_lot_guarantee=True,
            hard_stop_loss=0.04,
            use_mtf_trend_filter=True,
            dynamic_calib=False,
        )

        self.assertIsNotNone(metrics)
        self.assertGreater(len(signal_df), 50)
        self.assertGreater(len(backtest_df), 50)
        self.assertIn('portfolio_value', backtest_df.columns)
        self.assertIn('cum_return', backtest_df.columns)


if __name__ == '__main__':
    unittest.main()
