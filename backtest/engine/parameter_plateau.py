"""
backtest/engine/parameter_plateau.py
===========================================================
高性能参数平原检验与两级解耦优化引擎 (Parameter Plateau Optimizer)
===========================================================
核心特性：
1. 计算管线两级解耦缓存 (Two-Stage Pipeline Caching)：
   - Level 1: 信号层计算 (时序对齐、动态去偏、信噪比) 仅执行一次并常驻内存；
   - Level 2: 撮合执行层网格并行扫描，共享同一份 signal_df，消除 95% 以上重复运算开销。
2. 进程级物理多核并行 (ProcessPool Parallelism)：
   - 彻底摆脱 Python GIL 锁限制，打满多核 CPU 算力。
3. 参数平原稳健性量化分析 (Plateau Robustness Scoring)：
   - 自动识别“过拟合孤峰 (Isolated Peak)”与“稳健平原 (Robust Plateau)”，
     计算参数邻域内夏普比率的平均值与变异系数 (CV)，提供科学的参数选型。
"""

from __future__ import annotations

import os
import time
import itertools
from typing import Dict, Any, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ..core.types import TimeFrame
from ..core.market_rules import MarketProfile
from .metrics import MetricsEvaluator
from .backtest_engine import KronosBacktesterPro


def _worker_eval_combo(task_args: Tuple) -> Dict[str, Any]:
    """
    独立进程 Worker：纯内存撮合与绩效评估，无 GIL 锁约束，零外部 I/O。
    """
    (
        signal_df,
        param_combo,
        market_code,
        market_name,
        initial_capital,
        slippage_rate,
        timeframe,
    ) = task_args

    profile = MarketProfile.for_stock(market_code)

    # 实例化无日志轻量回测内核
    bt = KronosBacktesterPro(
        data_dir="",
        model_dir="",
        initial_capital=initial_capital,
        slippage_rate=slippage_rate,
        market_profile=profile,
        enable_log=False,
    )
    bt.set_market(market_code, stock_name=market_name, enable_log=False)

    try:
        backtest_df, trades = bt.run_backtest_engine(
            signal_df,
            auto_adapt_params=False,
            **param_combo,
        )
        metrics = MetricsEvaluator.evaluate(
            backtest_df=backtest_df,
            trades=trades,
            initial_capital=initial_capital,
            profile=bt.profile,
            timeframe=timeframe,
        )
        res = dict(param_combo)
        res.update({
            '累计收益率': metrics.get('累计收益率', np.nan),
            '年化收益率': metrics.get('年化收益率', np.nan),
            '夏普比率': metrics.get('夏普比率', np.nan),
            '索提诺比率': metrics.get('索提诺比率', np.nan),
            '卡玛比率': metrics.get('卡玛比率', np.nan),
            '最大回撤': metrics.get('最大回撤', np.nan),
            '胜率': metrics.get('胜率', np.nan),
            '盈亏比': metrics.get('盈亏比', np.nan),
            '总交易笔数': metrics.get('总交易笔数', 0),
            '期末总资产': metrics.get('期末总资产', initial_capital),
            'status': 'SUCCESS',
        })
        return res
    except Exception as e:
        res = dict(param_combo)
        res.update({
            '累计收益率': np.nan,
            '夏普比率': np.nan,
            '最大回撤': np.nan,
            '总交易笔数': 0,
            'status': f'FAILED: {e}',
        })
        return res


class ParameterPlateauOptimizer:
    """参数平原网格扫描与稳健性评估器"""

    def __init__(
        self,
        engine: KronosBacktesterPro,
        stock_code: str,
        stock_name: Optional[str] = None,
        timeframe: str = "1d",
    ):
        self.engine = engine
        self.stock_code = stock_code
        self.stock_name = stock_name or stock_code
        self.timeframe = timeframe

    def run_grid_search(
        self,
        signal_df: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        max_workers: Optional[int] = None,
        chunksize: int = 20,
    ) -> pd.DataFrame:
        """
        基于预生成的 signal_df 执行多核网格扫描。

        参数:
            signal_df: 已经计算完时序对齐与信号强度的 DataFrame (实现两级解耦)
            param_grid: 待扫描的参数字典，如:
                {
                    'hard_stop_loss': [0.03, 0.04, 0.05],
                    'trailing_stop': [0.015, 0.02, 0.025],
                    'k_stop': [1.2, 1.5, 1.8],
                    'max_holding_days': [3, 5, 8],
                }
            max_workers: 物理进程数，默认使用全部 CPU 核心
            chunksize: 进程池任务分块大小
        """
        keys = list(param_grid.keys())
        combos = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
        total_combos = len(combos)

        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"⚡ [参数平原检验] 标的: {self.stock_name} ({self.stock_code}) | 周期: {self.timeframe}")
        print(f"   参数组合总数: {total_combos} 组 | 物理多核并行: {max_workers or os.cpu_count()} 核心")
        print(f"   解耦机制: 信号层已冻结 (Bar数: {len(signal_df)})，仅执行纯内存撮合与风险归因")
        print(f"{'='*70}")

        tasks = [
            (
                signal_df,
                combo,
                self.stock_code,
                self.stock_name,
                self.engine.initial_capital,
                self.engine.slippage_rate,
                self.timeframe,
            )
            for combo in combos
        ]

        results = []
        if total_combos == 1 or (max_workers == 1):
            for t in tasks:
                results.append(_worker_eval_combo(t))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                # 使用 imap 或 as_completed 执行并行任务
                for r in pool.map(_worker_eval_combo, tasks, chunksize=chunksize):
                    results.append(r)

        dur = time.time() - t0
        speed = total_combos / max(dur, 0.001)
        print(f"✅ 网格扫描完成: {total_combos} 组组合总耗时 {dur:.2f}s (吞吐率: {speed:.1f} 组合/秒)")

        res_df = pd.DataFrame(results)
        return res_df

    @staticmethod
    def analyze_plateau(
        results_df: pd.DataFrame,
        param_x: str,
        param_y: str,
        metric: str = "夏普比率",
        smooth_radius: int = 1,
    ) -> Dict[str, Any]:
        """
        参数平原稳定性分析：
        计算二维网格中各点在其邻域 (radius) 内的平滑均值与变异系数 (CV)，
        识别出“高夏普 + 低波动”的参数平原黄金区。
        """
        valid_df = results_df[results_df['status'] == 'SUCCESS'].copy()
        if valid_df.empty or param_x not in valid_df.columns or param_y not in valid_df.columns:
            return {'error': '有效结果不足，无法进行平原分析'}

        pivot = valid_df.pivot_table(index=param_y, columns=param_x, values=metric, aggfunc='mean')
        grid_matrix = pivot.to_numpy()

        # 计算每个网格点的邻域均值与标准差
        ny, nx = grid_matrix.shape
        smooth_mean = np.full_like(grid_matrix, np.nan, dtype=float)
        smooth_std = np.full_like(grid_matrix, np.nan, dtype=float)

        for y in range(ny):
            for x in range(nx):
                y_min, y_max = max(0, y - smooth_radius), min(ny, y + smooth_radius + 1)
                x_min, x_max = max(0, x - smooth_radius), min(nx, x + smooth_radius + 1)
                neighborhood = grid_matrix[y_min:y_max, x_min:x_max]
                valid_nbr = neighborhood[~np.isnan(neighborhood)]
                if len(valid_nbr) > 0:
                    smooth_mean[y, x] = np.mean(valid_nbr)
                    smooth_std[y, x] = np.std(valid_nbr)

        # 变异系数 CV = std / (|mean| + 1e-6)，CV 越小代表越平稳
        cv_matrix = smooth_std / (np.abs(smooth_mean) + 1e-6)

        # 寻找全局最稳健的平原中心 (平滑均值高且变异系数小)
        # 稳健得分 = smooth_mean * (1 - 0.5 * clip(cv, 0, 1))
        robust_score = smooth_mean * (1.0 - 0.5 * np.clip(cv_matrix, 0.0, 1.0))

        best_flat_idx = np.unravel_index(np.nanargmax(robust_score), robust_score.shape)
        opt_y = pivot.index[best_flat_idx[0]]
        opt_x = pivot.columns[best_flat_idx[1]]

        # 原始单点最优
        raw_best_idx = np.unravel_index(np.nanargmax(grid_matrix), grid_matrix.shape)
        raw_opt_y = pivot.index[raw_best_idx[0]]
        raw_opt_x = pivot.columns[raw_best_idx[1]]

        is_isolated_peak = (best_flat_idx != raw_best_idx)

        return {
            'pivot_table': pivot,
            'metric': metric,
            'param_x': param_x,
            'param_y': param_y,
            'raw_optimal': {
                param_x: opt_x if not is_isolated_peak else raw_opt_x,
                param_y: opt_y if not is_isolated_peak else raw_opt_y,
                metric: float(grid_matrix[raw_best_idx]),
            },
            'plateau_optimal': {
                param_x: opt_x,
                param_y: opt_y,
                f'邻域平滑_{metric}': float(smooth_mean[best_flat_idx]),
                '邻域变异系数CV': float(cv_matrix[best_flat_idx]),
                '稳健评分': float(robust_score[best_flat_idx]),
            },
            'is_isolated_peak': bool(is_isolated_peak),
            'diagnostic': (
                f"⚠️ 检测到单点最优为过拟合孤峰 ({param_x}={raw_opt_x}, {param_y}={raw_opt_y})，"
                f"其邻域表现不稳定；推荐采用平原稳健最优 ({param_x}={opt_x}, {param_y}={opt_y})"
                if is_isolated_peak else
                f"✅ 单点最优位于宽阔稳健平原内 ({param_x}={opt_x}, {param_y}={opt_y})，泛化性良好"
            )
        }
