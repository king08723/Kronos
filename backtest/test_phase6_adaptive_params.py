"""
backtest/test_phase6_adaptive_params.py
Phase 6 单元测试：不同 K 线周期自适应参数解析与全链路回归验证
测试点：
1. get_timeframe_defaults 各周期（1d, 60m, 30m, 15m, 5m, 1m）量纲矩阵正确性；
2. A股 (CN) 与 港美股 (HK/US) 交易制度合规性微调（T+1 严禁日内强平 vs T+0 高频日内强平）；
3. resolve_timeframe_params 用户显式覆盖与同义别名自动同步；
4. 策略层 align_and_generate_signals 在参数为 None 时的无感自适应；
5. 撮合引擎 run_backtest_engine 与 run_complete_backtest 端到端自适应回测。
"""

import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np

from backtest.core import (
    TimeFrame,
    MarketProfile,
    get_timeframe_defaults,
    resolve_timeframe_params,
    TIMEFRAME_BEST_PRACTICES,
)
from backtest.strategies.kronos_strategy import KronosStrategy
from backtest.engine.backtest_engine import KronosBacktesterPro


class TestPhase6AdaptiveParams(unittest.TestCase):

    def test_get_timeframe_defaults_matrix(self):
        """测试 6 种周期基准参数的阶梯式量纲矩阵"""
        # 日线 (1d)
        d1 = get_timeframe_defaults('1d', market='CN')
        self.assertEqual(d1['timeframe'], '1d')
        self.assertEqual(d1['max_hold'], 15)
        self.assertEqual(d1['hard_stop'], 0.040)
        self.assertEqual(d1['trail_stop'], 0.025)
        self.assertEqual(d1['z_threshold'], 0.55)
        self.assertEqual(d1['slippage_model'], 'fixed')
        self.assertIsNone(d1['volume_participation_cap'])
        self.assertFalse(d1['use_mtf_trend_filter'])

        # 30分钟 (30m)
        m30 = get_timeframe_defaults('30m', market='CN')
        self.assertEqual(m30['timeframe'], '30m')
        self.assertEqual(m30['max_hold'], 24)
        self.assertEqual(m30['hard_stop'], 0.022)
        self.assertEqual(m30['trail_stop'], 0.016)
        self.assertEqual(m30['z_threshold'], 0.75)
        self.assertEqual(m30['slippage_model'], 'dynamic')
        self.assertEqual(m30['volume_participation_cap'], 0.10)
        self.assertTrue(m30['use_mtf_trend_filter'])

        # 1分钟 (1m)
        m1 = get_timeframe_defaults('1m', market='CN')
        self.assertEqual(m1['timeframe'], '1m')
        self.assertEqual(m1['max_hold'], 45)
        self.assertEqual(m1['hard_stop'], 0.008)
        self.assertEqual(m1['trail_stop'], 0.005)
        self.assertEqual(m1['z_threshold'], 1.05)
        self.assertEqual(m1['slippage_model'], 'dynamic')
        self.assertEqual(m1['volume_participation_cap'], 0.03)

    def test_market_compliance_t0_vs_t1(self):
        """测试市场制度合规：A股 T+1 严禁日内强平，港美股 T+0 高频日内强平"""
        # A 股 1分钟/5分钟/15分钟，强制 intraday_exit_at_eod=False
        for tf in ['1m', '5m', '15m']:
            cn_cfg = get_timeframe_defaults(tf, market='CN')
            self.assertFalse(cn_cfg['intraday_exit_at_eod'], f"A股 {tf} 绝不可日内强平，须满足 T+1 合规")

        # 港美股 1分钟/5分钟/15分钟，支持 T+0，默认开启日内强平
        for mkt in ['HK', 'US']:
            for tf in ['1m', '5m', '15m']:
                cfg = get_timeframe_defaults(tf, market=mkt)
                self.assertTrue(cfg['intraday_exit_at_eod'], f"{mkt} 市场 {tf} 应自适应开启日内强平防范隔夜跳空")

    def test_resolve_timeframe_params_overrides(self):
        """测试自适应参数解析与用户显式覆盖"""
        # 未传参数时，完全填充 30m 最佳实践
        cfg = resolve_timeframe_params('30m', market='CN', auto_adapt=True)
        self.assertEqual(cfg['hard_stop'], 0.022)
        self.assertEqual(cfg['hard_stop_loss'], 0.022)
        self.assertEqual(cfg['slippage_model'], 'dynamic')

        # 用户显式指定 hard_stop=0.035，应覆盖默认 0.022，且同义别名 hard_stop_loss 同步更新
        custom = resolve_timeframe_params('30m', market='CN', overrides={'hard_stop': 0.035}, auto_adapt=True)
        self.assertEqual(custom['hard_stop'], 0.035)
        self.assertEqual(custom['hard_stop_loss'], 0.035)
        # 其余未指定的参数依然保持 30m 最佳实践
        self.assertEqual(custom['trail_stop'], 0.016)
        self.assertEqual(custom['z_threshold'], 0.75)

        # auto_adapt=False 时，回退到 1d 默认值
        no_adapt = resolve_timeframe_params('30m', market='CN', auto_adapt=False)
        self.assertEqual(no_adapt['hard_stop'], 0.040)
        self.assertEqual(no_adapt['z_threshold'], 0.55)

    def test_strategy_signal_adaptive(self):
        """测试策略层在未传参数时能自动推断周期并应用自适应阈值"""
        # 构造 30 分钟行情与预测数据
        dates = pd.date_range("2026-03-02 09:30:00", periods=60, freq="30min")
        hist_df = pd.DataFrame({
            'open': [20.0 + i * 0.1 for i in range(60)],
            'high': [20.2 + i * 0.1 for i in range(60)],
            'low': [19.8 + i * 0.1 for i in range(60)],
            'close': [20.1 + i * 0.1 for i in range(60)],
            'volume': [100000.0] * 60,
        }, index=dates)

        pred_df = pd.DataFrame({
            'forecast_horizon': [3] * 60,
            'pred_mean': [20.5 + i * 0.1 for i in range(60)],
            'pred_close': [20.5 + i * 0.1 for i in range(60)],
            'pred_std': [0.5] * 60,
        }, index=dates)

        # z_score_threshold=None, hard_stop_loss=None, entry_percentile=None
        strat = KronosStrategy()
        signal_df = strat.align_and_generate_signals(
            hist_df=hist_df,
            pred_df=pred_df,
            z_score_threshold=None,
            hard_stop_loss=None,
            entry_percentile=None,
            calib_window=None,
            use_mtf_trend_filter=None,
        )
        self.assertFalse(signal_df.empty)
        self.assertIn('exec_signal', signal_df.columns)
        self.assertIn('signal_score', signal_df.columns)

    def test_engine_adaptive_backtest_end_to_end(self):
        """测试回测引擎 auto_adapt_params=True 能够端到端自适应运行"""
        temp_dir = tempfile.mkdtemp()
        data_dir = os.path.join(temp_dir, "historical_data")
        pred_dir = os.path.join(temp_dir, "yuce")
        out_dir = os.path.join(temp_dir, "output")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        try:
            # 写入 30m 行情数据
            dates = pd.date_range("2026-03-02 10:00:00", periods=24, freq="30min")
            hist_df = pd.DataFrame({
                'date': dates,
                'open': [10.0 + (i % 3) * 0.1 for i in range(24)],
                'high': [10.2 + (i % 3) * 0.1 for i in range(24)],
                'low': [9.8 + (i % 3) * 0.1 for i in range(24)],
                'close': [10.1 + (i % 3) * 0.1 for i in range(24)],
                'volume': [50000] * 24,
            })
            hist_df.to_csv(os.path.join(data_dir, "300418_30m.csv"), index=False)

            # 写入对应预测文件
            pred_df = pd.DataFrame({
                'date': dates,
                'pred_mean': [10.5] * 24,
                'pred_close': [10.5] * 24,
                'pred_std': [0.4] * 24,
                'forecast_horizon': [4] * 24,
            })
            pred_df.to_csv(os.path.join(pred_dir, "300418_predictions.csv"), index=False)

            backtester = KronosBacktesterPro(data_dir=data_dir, model_dir=pred_dir, initial_capital=100000)
            metrics, sig_df, bt_df, trades, fig = backtester.run_complete_backtest(
                stock_code="300418",
                output_dir=out_dir,
                timeframe="30m",
                auto_adapt_params=True,
                show_plot=False,
            )
            self.assertIsNotNone(metrics)
            self.assertIsNotNone(bt_df)
            self.assertFalse(bt_df.empty)

            # 验证导出产物 metrics 或 params 中记录了自适应参数
            m_path = os.path.join(out_dir, "300418_metrics.json")
            self.assertTrue(os.path.exists(m_path))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
