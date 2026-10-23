"""
backtest/test_parameter_plateau.py
单元测试：高性能参数平原优化器 (ParameterPlateauOptimizer)
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backtest.core.market_rules import MarketProfile
from backtest.engine.backtest_engine import KronosBacktesterPro
from backtest.engine.parameter_plateau import ParameterPlateauOptimizer


class TestParameterPlateauOptimizer(unittest.TestCase):
    def setUp(self):
        # 构造仿真 30m 高频数据 (400 根 Bar)
        start = datetime(2026, 3, 2, 9, 30)
        dates = [start + timedelta(minutes=30 * i) for i in range(400)]
        np.random.seed(42)
        close_prices = 100.0 + np.cumsum(np.random.randn(400) * 0.5)

        self.signal_df = pd.DataFrame(
            {
                'close': close_prices,
                'open': close_prices + np.random.randn(400) * 0.1,
                'high': close_prices + np.abs(np.random.randn(400) * 0.3),
                'low': close_prices - np.abs(np.random.randn(400) * 0.3),
                'exec_signal': np.random.choice([0, 1], size=400, p=[0.85, 0.15]),
                'atr_pct': 0.015,
                'exec_position_scale': 1.0,
                'exec_target_horizon': 5,
                'open_limit_up': 0,
                'open_limit_down': 0,
            },
            index=pd.DatetimeIndex(dates, name='date'),
        )

        self.engine = KronosBacktesterPro(
            data_dir="",
            model_dir="",
            initial_capital=100000.0,
            market_profile=MarketProfile.for_stock("600036"),
            enable_log=False,
        )

    def test_grid_search_and_plateau_analysis(self):
        opt = ParameterPlateauOptimizer(
            engine=self.engine,
            stock_code="600036",
            stock_name="招商银行",
            timeframe="30m",
        )

        # 扫描 3 x 3 x 2 = 18 组参数网格
        param_grid = {
            'hard_stop_loss': [0.03, 0.04, 0.05],
            'trailing_stop': [0.015, 0.02, 0.025],
            'max_holding_days': [5, 10],
        }

        results_df = opt.run_grid_search(
            signal_df=self.signal_df,
            param_grid=param_grid,
            max_workers=2,
        )

        self.assertEqual(len(results_df), 18)
        self.assertIn('夏普比率', results_df.columns)
        self.assertIn('累计收益率', results_df.columns)
        self.assertTrue((results_df['status'] == 'SUCCESS').all())

        # 分析参数平原
        analysis = opt.analyze_plateau(
            results_df=results_df,
            param_x='hard_stop_loss',
            param_y='trailing_stop',
            metric='夏普比率',
        )

        self.assertIn('plateau_optimal', analysis)
        self.assertIn('raw_optimal', analysis)
        self.assertIn('diagnostic', analysis)
        print(f"平原分析诊断: {analysis['diagnostic']}")


if __name__ == '__main__':
    unittest.main()
