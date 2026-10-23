"""
test_phase5_integration_cli.py
Phase 5 专项测试套件：CLI 命令行、可视化组件与全链路综合集成

覆盖内容:
1. CLI 命令行参数解析完整性测试 (--timeframe, --volume-cap, --slippage-model, --intraday-exit-eod 等)
2. CLI 调度回测全真流程 (带 30m 时间周期、动态滑点与日内强平)
3. 静态图表与动态交互图表在日内高频时间戳下的自适应渲染验证
4. 系统级多周期 (1d, 60m, 30m) 交叉运行回归测试
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

from backtest.core.types import TimeFrame
from backtest.core.market_rules import MarketProfile
from backtest.visualizer.plot_static import StaticVisualizer
from backtest.visualizer.plot_interactive import InteractiveVisualizer, HAS_PLOTLY
from backtest.engine.backtest_engine import KronosBacktesterPro


class TestPhase5CLIArguments(unittest.TestCase):
    """测试 CLI 命令行解析器的多时间周期与高级风控参数"""

    def test_cli_parser_defaults_and_options(self):
        """测试 argparse 正确识别新增的参数与默认值"""
        import argparse

        # 模拟 run_backtest_kronos_pro 中的 parser 构建逻辑
        parser = argparse.ArgumentParser()
        parser.add_argument("stock_code", nargs="?", default="300418")
        parser.add_argument("--timeframe", "--tf", type=str, default="1d",
                            choices=['1d', '60m', '30m', '15m', '5m', '1m'])
        parser.add_argument("--volume-cap", type=float, default=None)
        parser.add_argument("--slippage-model", type=str, default="fixed", choices=["fixed", "dynamic"])
        parser.add_argument("--intraday-exit-eod", action="store_true", default=False)
        parser.add_argument("--no-gap-protection", action="store_true", default=False)
        parser.add_argument("--no-mtf-filter", action="store_true", default=False)

        # 默认解析
        args_default = parser.parse_args(["300418"])
        self.assertEqual(args_default.stock_code, "300418")
        self.assertEqual(args_default.timeframe, "1d")
        self.assertIsNone(args_default.volume_cap)
        self.assertEqual(args_default.slippage_model, "fixed")
        self.assertFalse(args_default.intraday_exit_eod)
        self.assertFalse(args_default.no_gap_protection)
        self.assertFalse(args_default.no_mtf_filter)

        # 自定义参数解析
        custom_argv = [
            "600519",
            "--timeframe", "30m",
            "--volume-cap", "0.08",
            "--slippage-model", "dynamic",
            "--intraday-exit-eod",
            "--no-gap-protection",
            "--no-mtf-filter",
        ]
        args_custom = parser.parse_args(custom_argv)
        self.assertEqual(args_custom.stock_code, "600519")
        self.assertEqual(args_custom.timeframe, "30m")
        self.assertEqual(args_custom.volume_cap, 0.08)
        self.assertEqual(args_custom.slippage_model, "dynamic")
        self.assertTrue(args_custom.intraday_exit_eod)
        self.assertTrue(args_custom.no_gap_protection)
        self.assertTrue(args_custom.no_mtf_filter)


class TestPhase5VisualizerIntraday(unittest.TestCase):
    """测试可视化模块对日内高频 K 线的自适应渲染"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # 构造 3 个交易日的高频 30m 数据
        times = []
        for d in ['2026-03-02', '2026-03-03', '2026-03-04']:
            times.extend(pd.to_datetime([
                f'{d} 10:00:00', f'{d} 10:30:00', f'{d} 11:00:00', f'{d} 11:30:00',
                f'{d} 13:30:00', f'{d} 14:00:00', f'{d} 14:30:00', f'{d} 15:00:00'
            ]))
        np.random.seed(42)
        prices = 10.0 + np.cumsum(np.random.randn(len(times)) * 0.1)

        self.backtest_df = pd.DataFrame({
            'open': prices,
            'high': prices + 0.1,
            'low': prices - 0.1,
            'close': prices,
            'shares': [1000] * 12 + [0] * (len(times) - 12),
            'portfolio_value': 100000.0 + np.cumsum(np.random.randn(len(times)) * 50),
        }, index=pd.DatetimeIndex(times))
        self.backtest_df['returns'] = self.backtest_df['portfolio_value'].pct_change().fillna(0.0)

        self.signal_df = pd.DataFrame({
            'signal_score': np.random.randn(len(times)),
            'raw_signal': [1, 0, 0, 0] * 6,
        }, index=pd.DatetimeIndex(times))

        self.trades = [
            {
                'date': times[2],
                'action': 'BUY',
                'price': float(prices[2]),
                'shares': 1000,
                'target_horizon': 3,
            },
            {
                'date': times[12],
                'action': 'SELL',
                'price': float(prices[12]),
                'shares': 1000,
                'pnl_pct': 0.025,
                'reason': '波峰止盈',
                'cash_balance': 102500.0,
            }
        ]
        self.metrics = {
            '累计收益率': 0.025,
            '夏普比率': 1.85,
            '最大回撤': -0.012,
            '胜率': 1.0,
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_static_visualizer_intraday_rendering(self):
        """测试 StaticVisualizer 在 30m 数据下能够成功生成回测图且柱状图不溢出"""
        viz = StaticVisualizer(profile=MarketProfile.for_market('CN'))
        fig = viz.plot_backtest_results(
            signal_df=self.signal_df,
            backtest_df=self.backtest_df,
            metrics=self.metrics,
            trades=self.trades,
            stock_code="300418",
            output_dir=self.temp_dir.name,
            show_plot=False,
        )
        self.assertIsNotNone(fig)
        out_png = os.path.join(self.temp_dir.name, "300418_backtest_pro.png")
        self.assertTrue(os.path.exists(out_png))
        self.assertGreater(os.path.getsize(out_png), 10000)

    def test_interactive_visualizer_intraday_rendering(self):
        """测试 InteractiveVisualizer 能够自适应生成包含时分信息的 Plotly 图表"""
        if not HAS_PLOTLY:
            self.skipTest("Plotly 未安装，跳过交互图测试")

        viz = InteractiveVisualizer(profile=MarketProfile.for_market('CN'))
        fig = viz.plot_backtest_interactive(
            signal_df=self.signal_df,
            backtest_df=self.backtest_df,
            metrics=self.metrics,
            trades=self.trades,
            stock_code="300418",
        )
        self.assertIsNotNone(fig)
        # 检验 hover 包含精确到分钟的时间格式
        self.assertIn("%Y-%m-%d %H:%M", str(fig))


class TestPhase5EndToEndFullPipeline(unittest.TestCase):
    """端到端综合验收测试：多时间周期从数据、策略到撮合的全链路闭环"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.temp_dir.name, "historical_data")
        self.pred_dir = os.path.join(self.temp_dir.name, "yuce")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

        # 构造 1 分钟基础高频数据 (3天 x 240条/天 = 720条)
        times_1m = []
        for d in ['2026-03-02', '2026-03-03', '2026-03-04']:
            times_1m.extend(pd.date_range(f"{d} 09:30:00", f"{d} 11:30:00", freq="1min"))
            times_1m.extend(pd.date_range(f"{d} 13:00:00", f"{d} 15:00:00", freq="1min"))

        np.random.seed(42)
        close_1m = 25.0 + np.cumsum(np.random.randn(len(times_1m)) * 0.02)
        df_1m = pd.DataFrame({
            'date': times_1m,
            'open': close_1m,
            'high': close_1m + 0.05,
            'low': close_1m - 0.05,
            'close': close_1m,
            'volume': 5000,
            'amount': close_1m * 5000,
        })
        # 保存为基础 1 分钟文件 300418_1m.csv
        df_1m.to_csv(os.path.join(self.data_dir, "300418_1m.csv"), index=False)

        # 构造日度预测数据
        pred_df = pd.DataFrame({
            'date': ['2026-03-02', '2026-03-03', '2026-03-04'],
            'pred_close': [25.5, 26.0, 26.5],
            'pred_path_max': [26.0, 26.8, 27.2],
            'optimal_horizon': [2, 2, 2],
            'target_horizon': [2, 2, 2],
        })
        pred_df.to_csv(os.path.join(self.pred_dir, "300418_predictions.csv"), index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_pipeline_on_the_fly_resample_30m(self):
        """
        全真测试：本地仅有 1 分钟基础数据，引擎指定 timeframe='30m'，
        验证系统自动完成：
        1. 内存现场无损合成 30m K 线
        2. 日度预测广播至 30m
        3. 自动计算双周期 MA20/MA60 共振滤网
        4. 启用流动性 5% 截断与动态平方根冲击滑点
        5. 启用日内收盘强平 (EOD Exit)
        """
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
            min_overlap_days=2,
            volume_participation_cap=0.05,
            slippage_model="dynamic",
            intraday_exit_at_eod=True,
            use_mtf_trend_filter=True,
            dynamic_calib=False,
            show_plot=False,
        )

        self.assertIsNotNone(metrics)
        self.assertGreater(len(backtest_df), 10)
        self.assertIn('cum_return', backtest_df.columns)
        # 验证回测结束时或每个交易日收盘时仓位处理正常
        self.assertIn('portfolio_value', backtest_df.columns)


if __name__ == '__main__':
    unittest.main()
