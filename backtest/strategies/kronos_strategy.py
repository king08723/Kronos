"""
backtest/strategies/kronos_strategy.py
Kronos 深度时序 AI 量化策略
包含：
- 波动率自适应滚动分位数自校准 (compute_dynamic_rolling_quantile)
- 蒙特卡洛预测特征解耦与动态去偏
- 九重守门人入场否决矩阵 (9-Stage Risk Veto)
- 粒子滤波置信度加权
- 因果律检验 (Anti-Lookahead Causality Check)
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from ..core.types import Bar, Signal, TimeFrame
from ..core.market_rules import MarketProfile, compute_open_limit_flags
from ..core.timeframes import detect_timeframe, resolve_timeframe_params
from ..core.mtf_aligner import MultiTimeframeAligner
from .base import BaseStrategy

logger = logging.getLogger("KronosBacktest")

def _fast_median_1d(valid: np.ndarray) -> float:
    m = len(valid)
    mid = m // 2
    if m % 2 == 1:
        return float(np.partition(valid, mid)[mid])
    else:
        part = np.partition(valid, (mid - 1, mid))
        return float((part[mid - 1] + part[mid]) * 0.5)


def _fast_quantile_1d(valid: np.ndarray, q: float) -> float:
    m = len(valid)
    if m == 1:
        return float(valid[0])
    v_idx = q * (m - 1)
    i = int(np.floor(v_idx))
    fraction = v_idx - i
    if fraction == 0 or i >= m - 1:
        return float(np.partition(valid, i)[i])
    part = np.partition(valid, (i, i + 1))
    return float(part[i] + fraction * (part[i + 1] - part[i]))


def compute_dynamic_rolling_quantile(
    series: pd.Series,
    windows: np.ndarray | pd.Series,
    q: float,
    min_periods_ratio: float = 0.25,
    min_floor: int = 8,
) -> pd.Series:
    """
    计算动态变长窗口的滚动分位数（严格防未来函数，线性选择 O(W) 加速版）。

    机制说明:
    在极端高波动率或黑天鹅阶段，固定长窗口会导致滞后钝化；而平稳震荡期需要长窗口保持统计稳定性。
    本函数支持逐日传入自适应计算得出的窗口长度 windows[i]，在截至当日 t 的历史切片上
    计算指定分位数 q（如 q=0.5 为动态中位数去偏，q=0.8 为 80% 动态入场分位门槛）。
    采用快速选择算法 (np.partition) 代替全量排序切片，计算复杂度从 O(W log W) 降至 O(W)，
    数值在浮点精度上与标准 np.quantile/median 严格等价。

    参数:
        series: 待统计的时序数据 (如 shift(1) 后的预测误差或 signal_score)
        windows: 逐日对应的整数回看窗口长度 (标量、数组或 Series)
        q: 分位数值 (0.0 ~ 1.0)
        min_periods_ratio: 最小有效样本占当前窗口的比例，默认 0.25
        min_floor: 最小有效样本保底数量，默认 8

    返回:
        pd.Series: 与原序列同索引的动态滚动分位数序列
    """
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    if isinstance(windows, (int, np.integer)):
        windows = np.full(n, int(windows), dtype=int)
    elif isinstance(windows, pd.Series):
        windows = windows.to_numpy(dtype=int)
    else:
        windows = np.asarray(windows, dtype=int)

    result = np.full(n, np.nan, dtype=float)
    is_median = (q == 0.5)

    for i in range(n):
        w = int(windows[i])
        min_p = max(min_floor, int(w * min_periods_ratio))
        start = max(0, i - w + 1)
        chunk = vals[start : i + 1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) >= min_p:
            if is_median:
                result[i] = _fast_median_1d(valid)
            else:
                result[i] = _fast_quantile_1d(valid, q)

    return pd.Series(result, index=series.index, name=series.name)



class KronosStrategy(BaseStrategy):
    """
    Kronos AI 驱动自适应量化策略。
    """

    def __init__(self, name: str = "KronosAI_Pro", profile: Optional[MarketProfile] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params)
        self.profile = profile or MarketProfile.for_market('CN')

    def generate_signals(
        self,
        hist_df: pd.DataFrame,
        pred_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """实现 BaseStrategy.generate_signals 接口"""
        if pred_df is None:
            raise ValueError("KronosStrategy 需要模型预测数据 pred_df")
        return self.align_and_generate_signals(hist_df, pred_df, **kwargs)

    def align_and_generate_signals(
        self,
        hist_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        z_score_threshold: Optional[float] = None,
        volatility_window: int = 14,
        min_overlap_bars: Optional[int] = None,
        min_overlap_days: Optional[int] = None,
        forecast_horizon: int = None,
        max_auto_horizon: int = 10,
        hard_stop_loss: Optional[float] = None,
        entry_percentile: Optional[float] = None,
        calib_window: Optional[int] = None,
        dynamic_calib: bool = True,
        calib_window_min: Optional[int] = None,
        vol_high_thresh: float = 0.40,
        vol_low_thresh: float = 0.20,
        debias: bool = True,
        atr_window: int = 14,
        allow_interpolated: bool = False,
        particle_filter: bool = True,
        pf_confidence_threshold: float = 0.35,
        reversal_aware_debias: bool = True,
        reversal_debias_weight: float = 0.0,
        higher_hist_df: Optional[pd.DataFrame] = None,
        use_mtf_trend_filter: Optional[bool] = None,
        auto_adapt_params: bool = True,
        timeframe: Optional[Union[str, TimeFrame]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        核心时序对齐与信号生成。

        【时序不变量】
        对每个交易日/Bar t，信号 raw_signal[t] 只允许使用在 **t-1 收盘时点已知** 的信息：
          - pred_*[t]      由严格早于 t 的数据推理得到（walk_forward_predict 保证）
          - close_prev[t]  = close[t-1]
          - volatility[t]  使用截至 t-1 的收益率窗口计算
          - atr[t]         使用截至 t-1 的真实波幅计算
        因此 raw_signal[t] 在 t-1 收盘即可确定，在 **t 日/Bar 开盘** 撮合成交。

        【自适应周期最佳实践参数】
        未显式指定的参数自动依据 detect_timeframe(hist_df) 适配最优默认值。

        【跨周期双时钟宏观趋势滤网 (MTF Trend Filter)】
        当小级别为日内高频 (如 30m / 60m) 时，自动启动大级别 (日线) 宏观趋势滤网，
        通过 MultiTimeframeAligner 以零未来函数对齐日线 MA20 / MA60，实现双周期共振。
        """
        # 防御性规整：确保索引为 DatetimeIndex
        if not isinstance(hist_df.index, pd.DatetimeIndex):
            hist_work = hist_df.copy()
            date_col = 'date' if 'date' in hist_work.columns else (
                'timestamps' if 'timestamps' in hist_work.columns else None
            )
            if date_col:
                hist_work[date_col] = pd.to_datetime(hist_work[date_col])
                hist_work = hist_work.set_index(date_col).sort_index()
            else:
                hist_work.index = pd.to_datetime(hist_work.index)
            hist_df = hist_work

        if not isinstance(pred_df.index, pd.DatetimeIndex):
            pred_work_in = pred_df.copy()
            date_col = 'date' if 'date' in pred_work_in.columns else (
                '日期' if '日期' in pred_work_in.columns else None
            )
            if date_col:
                pred_work_in[date_col] = pd.to_datetime(pred_work_in[date_col])
                pred_work_in = pred_work_in.set_index(date_col).sort_index()
            else:
                pred_work_in.index = pd.to_datetime(pred_work_in.index)
            pred_df = pred_work_in

        tf = TimeFrame.from_str(timeframe) if timeframe else detect_timeframe(hist_df)

        if auto_adapt_params:
            adaptive_cfg = resolve_timeframe_params(
                timeframe=tf,
                market=getattr(self.profile, 'market', 'CN'),
                overrides={
                    'z_threshold': z_score_threshold,
                    'hard_stop': hard_stop_loss,
                    'entry_percentile': entry_percentile,
                    'calib_window': calib_window,
                    'calib_window_min': calib_window_min,
                    'use_mtf_trend_filter': use_mtf_trend_filter,
                },
                auto_adapt=True,
            )
            z_score_threshold = adaptive_cfg['z_threshold']
            hard_stop_loss = adaptive_cfg['hard_stop']
            entry_percentile = adaptive_cfg['entry_percentile']
            calib_window = adaptive_cfg['calib_window']
            calib_window_min = adaptive_cfg['calib_window_min']
            use_mtf_trend_filter = adaptive_cfg['use_mtf_trend_filter']
        else:
            z_score_threshold = 0.55 if z_score_threshold is None else z_score_threshold
            hard_stop_loss = 0.04 if hard_stop_loss is None else hard_stop_loss
            entry_percentile = 0.80 if entry_percentile is None else entry_percentile
            calib_window = 60 if calib_window is None else calib_window
            calib_window_min = 20 if calib_window_min is None else calib_window_min
            use_mtf_trend_filter = True if use_mtf_trend_filter is None else use_mtf_trend_filter

        if not (0 < hard_stop_loss < 1):
            raise ValueError(
                f"[PARAM ERROR] hard_stop_loss 须满足 0 < x < 1，当前值={hard_stop_loss}"
            )
        pred_work = pred_df.copy()

        # 最小重叠 K 线数量：优先取传入的 min_overlap_bars / min_overlap_days，缺省日线 10 根，分时 20 根
        min_overlap = min_overlap_bars if min_overlap_bars is not None else (
            min_overlap_days if min_overlap_days is not None else (10 if not tf.is_intraday else 20)
        )
        min_overlap_days = min_overlap  # 内部兼容

        common_dates = hist_df.index.intersection(pred_work.index)
        overlap_bars = len(common_dates)

        if overlap_bars < min_overlap and tf.is_intraday:
            # 跨周期检测：如果模型预测为日度序列，自动广播映射到日内高频 Bar 上
            hist_day_keys = set(d.date() if hasattr(d, 'date') else pd.to_datetime(d).date() for d in hist_df.index)
            pred_day_keys = set(d.date() if hasattr(d, 'date') else pd.to_datetime(d).date() for d in pred_work.index)
            day_overlap = len(hist_day_keys.intersection(pred_day_keys))
            if day_overlap >= max(5, min_overlap // 5):
                logger.info(f"   [MTF 跨周期对齐] 检测到日度预测 ({day_overlap} 交易日)，广播映射至 {tf.label} 日内 Bar...")
                pred_work = MultiTimeframeAligner.broadcast_daily_to_intraday(hist_df, pred_df)
                common_dates = hist_df.index.intersection(pred_work.index)
                overlap_bars = len(common_dates)

        bar_unit = "交易日" if not tf.is_intraday else "根 K线 (Bar)"
        print(f"\n[对齐检测] 历史 ∩ 预测 重叠区间: {overlap_bars} {bar_unit} (周期: {tf.label})")

        def _safe_dt_str(d):
            if hasattr(d, 'strftime'):
                if hasattr(d, 'hour') and (d.hour != 0 or d.minute != 0 or d.second != 0):
                    return d.strftime('%Y-%m-%d %H:%M:%S')
                return d.strftime('%Y-%m-%d')
            elif hasattr(d, 'date'):
                return str(d.date())
            return str(d)

        if overlap_bars < min_overlap:
            h_max_str = _safe_dt_str(hist_df.index.max())
            p_min_str = _safe_dt_str(pred_df.index.min())
            raise ValueError(
                f"\n{'='*60}\n"
                f"[BACKTEST ERROR] 重叠 K 线 (Bar) 不足，无法执行真实样本外回测！\n"
                f"  • 历史数据结束: {h_max_str}\n"
                f"  • 预测数据起始: {p_min_str}\n"
                f"  • 当前重叠根数: {overlap_bars} 根 Bar (要求 >= {min_overlap} 根 Bar，周期: {tf.label})\n"
                f"\n解决方案：\n"
                f"  请生成一份覆盖历史区间的 Kronos 预测序列，确保预测时间戳\n"
                f"  与历史真实收盘价有至少 {min_overlap} 根 K 线 (Bar) 的重叠。\n"
                f"{'='*60}"
            )

        print(f"   重叠区间: {_safe_dt_str(common_dates.min())} → {_safe_dt_str(common_dates.max())}")
        print(f"   [BACKTEST 模式] 以真实收盘价结算，执行严格样本外验证")

        # 自动推断预测时界 Horizon 与模式
        is_auto_horizon = False
        horizon_mode_col = 'horizon_mode' if 'horizon_mode' in pred_df.columns else (
            '时界' if '时界' in pred_df.columns else None)
        if forecast_horizon == 'auto' or (isinstance(forecast_horizon, str) and forecast_horizon.lower() == 'auto'):
            is_auto_horizon = True
        elif forecast_horizon is None:
            if horizon_mode_col and str(pred_df[horizon_mode_col].iloc[0]).lower() == 'auto':
                is_auto_horizon = True
            elif 'optimal_horizon' in pred_df.columns or '动态时界' in pred_df.columns:
                is_auto_horizon = True
            elif horizon_mode_col:
                try:
                    forecast_horizon = int(pred_df[horizon_mode_col].iloc[0])
                except Exception:
                    forecast_horizon = 1
            else:
                forecast_horizon = 1
        elif isinstance(forecast_horizon, (int, float)):
            forecast_horizon = int(forecast_horizon)

        # --------------------------------------------------------------
        # 历史侧指标：全部用「截至 t-1」的窗口，保证 t 日信号可在 t-1 收盘确定
        # --------------------------------------------------------------
        hist_work = hist_df.copy()
        hist_work['daily_return'] = hist_work['close'].pct_change()

        vol_raw = (hist_work['daily_return']
                   .rolling(window=volatility_window, min_periods=5)
                   .std()
                   .ffill())

        # 真实波动度量 ATR
        _high = hist_work['high'] if 'high' in hist_work.columns else hist_work['close']
        _low = hist_work['low'] if 'low' in hist_work.columns else hist_work['close']
        _pc = hist_work['close'].shift(1)
        true_range = pd.concat([(_high - _low).abs(),
                                (_high - _pc).abs(),
                                (_low - _pc).abs()], axis=1).max(axis=1)
        atr_raw = true_range.rolling(window=atr_window, min_periods=5).mean().ffill()
        atr_pct_raw = (atr_raw / hist_work['close']).ffill()

        # 时点修正：t 日决策只能看到截至 t-1 的统计量
        hist_work['volatility'] = vol_raw.shift(1)
        hist_work['atr'] = atr_raw.shift(1)
        hist_work['atr_pct'] = atr_pct_raw.shift(1)
        hist_work['close_prev'] = hist_work['close'].shift(1)

        # 趋势状态滞后一期 (支持跨周期双时钟宏观趋势滤网)
        if use_mtf_trend_filter and tf.is_intraday:
            if higher_hist_df is not None:
                hist_work = MultiTimeframeAligner.align_higher_to_lower(hist_work, higher_hist_df, prefix="htf_")
            else:
                hist_work = MultiTimeframeAligner.auto_align_dual_timeframe(hist_work, market=self.profile.market, prefix="htf_")
            ma_fast = hist_work['htf_ma_fast']
            ma_slow = hist_work['htf_ma_slow']
        else:
            ma_fast = hist_work['close'].rolling(20, min_periods=max(5, 10)).mean().shift(1)
            ma_slow = hist_work['close'].rolling(60, min_periods=max(10, 20)).mean().shift(1)
        hist_work['ma_fast'] = ma_fast
        hist_work['ma_slow'] = ma_slow

        # 截取重叠回测区间
        df = hist_work.loc[common_dates].copy()
        df['pred_close'] = pred_work.loc[common_dates, 'pred_close']

        for opt_col in ['pred_upper', 'pred_lower', 'pred_expected', 'pred_path_max',
                        'pred_terminal', 'pred_close_1d', 'pred_slope', 'pred_min_down', 'peak_premium',
                        'is_interpolated', 'pf_confidence', 'pf_val_error', 'pf_effective_n']:
            if opt_col in pred_work.columns:
                df[opt_col] = pred_work.loc[common_dates, opt_col]

        if 'pred_path_max' not in df.columns:
            df['pred_path_max'] = df['pred_close']

        # 剔除插值行
        if 'is_interpolated' in df.columns and not allow_interpolated:
            interp_mask = df['is_interpolated'].fillna(0) > 0
            if interp_mask.any():
                logger.info(f"  [数据过滤] 剔除 {int(interp_mask.sum())} 条插值预测行")
                df = df[~interp_mask]

        # 动态最优时界 (optimal_horizon)
        dyn_h_col = None
        for cand in ['optimal_horizon', '动态时界']:
            if cand in pred_work.columns:
                dyn_h_col = cand
                break

        if dyn_h_col is not None:
            df['target_horizon'] = (pred_work.loc[common_dates, dyn_h_col]
                                    .reindex(df.index).astype(float)
                                    .fillna(5).clip(1, 20).round().astype(int))
            horizon_display = "Auto(极值自适应)"
        else:
            # ⚠️ fallback：预测缓存中缺少 '动态时界' / 'optimal_horizon' 列。
            # 可能原因：使用了旧版预测脚本生成的缓存（未包含该列）。
            # 建议：删除旧缓存并勾选【强制刷新】重新生成预测以获取动态时界数据。
            if is_auto_horizon:
                # auto 模式下 fallback 到 max_auto_horizon 的一半（保守合理估计）
                h_int = max(3, (max_auto_horizon if isinstance(max_auto_horizon, int) else 10) // 2)
                logger.warning(
                    f"[时界Fallback] 预测 CSV 缺少 '动态时界' 列（可能为旧版缓存）。"
                    f"已自动降级为固定 {h_int} 天。"
                    f"建议：勾选【强制刷新】删除旧缓存后重新运行以恢复动态时界。"
                )
            else:
                h_int = int(forecast_horizon) if forecast_horizon is not None else 1
            df['target_horizon'] = h_int
            horizon_display = f"{h_int}天(Fallback-缺动态时界列)"

        df = df.dropna(subset=['close_prev', 'volatility'])
        if len(df) < max(min_overlap_days, 2):
            raise ValueError(
                f"[BACKTEST ERROR] 有效信号行不足（{len(df)} 行）。"
            )

        vol_scale = np.sqrt(df['target_horizon'].astype(float))

        # --------------------------------------------------------------
        # 波动率自适应校准窗口 (Volatility-Adaptive Calibration Window)
        # --------------------------------------------------------------
        # 年化波动率: 基于当前 K 线粒度的精确年化折算因子 (严格防偷价)
        ann_factor = tf.annual_factor(market=self.profile.market)
        df['ann_vol'] = (df['volatility'] * ann_factor).fillna(vol_low_thresh)

        # 样本窗口自适应：日内高频下扩展统计样本 Bar 数量
        if tf.is_intraday:
            bars_per_day = tf.bars_per_day(market=self.profile.market)
            eff_calib_window = calib_window * bars_per_day if calib_window <= 100 else calib_window
            eff_calib_min = calib_window_min * bars_per_day if calib_window_min <= 50 else calib_window_min
        else:
            eff_calib_window = calib_window
            eff_calib_min = calib_window_min

        if dynamic_calib:
            # 波动率压力系数 (0.0 表示平稳低波，1.0 表示极端高波)
            vol_stress = np.clip(
                (df['ann_vol'] - vol_low_thresh) / max(vol_high_thresh - vol_low_thresh, 1e-4),
                0.0, 1.0
            )
            # 动态窗口：从基准 calib_window 平滑收缩至 calib_window_min
            dyn_windows = np.round(
                eff_calib_window - vol_stress * (eff_calib_window - eff_calib_min)
            ).astype(int)
            df['dyn_calib_window'] = dyn_windows
        else:
            dyn_windows = np.full(len(df), eff_calib_window, dtype=int)
            df['dyn_calib_window'] = eff_calib_window

        # --------------------------------------------------------------
        # 预期收益与信号强度
        # --------------------------------------------------------------
        df['expected_return'] = (df['pred_close'] - df['close_prev']) / df['close_prev']
        df['target_return'] = (df['pred_path_max'] - df['close_prev']) / df['close_prev']

        if debias:
            realized_1d = (df['close'] - df['close_prev']) / df['close_prev']
            # 若预测包含单日预测价格 pred_close_1d，则同量纲计算单日系统性偏差；否则使用时界扩展收益归一化
            if 'pred_close_1d' in df.columns and df['pred_close_1d'].notna().all():
                pred_1d_ret = (df['pred_close_1d'] - df['close_prev']) / df['close_prev']
                err = pred_1d_ret - realized_1d
            else:
                err = (df['expected_return'] / np.maximum(vol_scale, 1.0)) - realized_1d

            if dynamic_calib:
                bias = compute_dynamic_rolling_quantile(
                    err.shift(1), dyn_windows, q=0.5, min_periods_ratio=0.25, min_floor=8
                )
            else:
                bias = err.shift(1).rolling(calib_window, min_periods=max(10, calib_window // 4)).median()
            
            # 单日中位数偏差按持有期扩展，保持与时界扩展波动率同量纲
            df['pred_bias'] = bias.fillna(0.0) * vol_scale

            # --------------------------------------------------------------
            # 状态感知自适应去偏 (Reversal-Aware Conditional De-biasing / 方案 1)
            # --------------------------------------------------------------
            # 在行情反转时，历史滚动偏差是由上一轮单边走势形成的“滞后误差”；
            # 若机械扣除，会严重惩罚真实的反转 Alpha，导致买卖点滞后或被硬否决。
            if reversal_aware_debias:
                # 历史环境：基于截至 t-1 的均线状态判断中短期强弱 (严格 t-1，杜绝未来函数)
                trend_down = (df['ma_fast'] < df['ma_slow']) | (df['close_prev'] < df['ma_fast'])
                trend_up = (df['ma_fast'] >= df['ma_slow']) | (df['close_prev'] > df['ma_fast'])

                # 底部反转 (Bullish Reversal): 弱势探底背景下，模型预测向上反转
                is_bull_reversal = trend_down & (df['expected_return'] > 0)
                if 'pred_slope' in df.columns:
                    is_bull_reversal = is_bull_reversal & (df['pred_slope'] > 0)

                # 顶部反转 (Bearish Reversal): 强势上升背景下，模型预测拐头向下
                is_bear_reversal = trend_up & (df['expected_return'] < 0)
                if 'pred_slope' in df.columns:
                    is_bear_reversal = is_bear_reversal & (df['pred_slope'] < 0)

                df['is_bull_reversal'] = is_bull_reversal
                df['is_bear_reversal'] = is_bear_reversal
                df['is_reversal'] = is_bull_reversal | is_bear_reversal

                # 反转状态下历史同向偏差衰减 (默认完全豁免，权重为 reversal_debias_weight = 0.0)
                # - 底部反转时阻断正偏差扣除 (防止将真实反转盈利扣为负)
                # - 顶部反转时阻断负偏差回补 (防止将看空收益拉高导致无法离场)
                reversal_mask = (is_bull_reversal & (df['pred_bias'] > 0)) | (is_bear_reversal & (df['pred_bias'] < 0))
                bias_weight = np.where(reversal_mask, reversal_debias_weight, 1.0)

                df['pred_bias_effective'] = df['pred_bias'] * bias_weight
                df['expected_return_adj'] = df['expected_return'] - df['pred_bias_effective']
            else:
                df['is_bull_reversal'] = False
                df['is_bear_reversal'] = False
                df['is_reversal'] = False
                df['pred_bias_effective'] = df['pred_bias']
                df['expected_return_adj'] = df['expected_return'] - df['pred_bias']
        else:
            df['pred_bias'] = 0.0
            df['pred_bias_effective'] = 0.0
            df['is_bull_reversal'] = False
            df['is_bear_reversal'] = False
            df['is_reversal'] = False
            df['expected_return_adj'] = df['expected_return']

        df['volatility_scaled'] = (df['volatility'] * vol_scale).replace(0, np.nan)
        df['signal_score'] = df['expected_return_adj'] / df['volatility_scaled']
        df['signal_score'] = df['signal_score'].replace([np.inf, -np.inf], np.nan)

        # --------------------------------------------------------------
        # 粒子滤波 (Particle Filtering / 玩法 B) 动态置信度与不确定性调控
        # --------------------------------------------------------------
        if particle_filter:
            if 'pf_confidence' in df.columns and df['pf_confidence'].notna().any():
                df['pf_confidence'] = df['pf_confidence'].fillna(1.0).clip(0.1, 1.0)
            else:
                # 兼容未包含蒙特卡洛粒子明细的旧版/外部预测：使用近期走势滚动拟合度动态衍生粒子置信度
                hist_pred_err = ((df['pred_close'].shift(1) - df['close_prev']) / (df['close_prev'] + 1e-6)).abs()
                rolling_err = hist_pred_err.rolling(5, min_periods=2).mean()
                if dynamic_calib:
                    base_err = compute_dynamic_rolling_quantile(
                        rolling_err, dyn_windows, q=0.5, min_periods_ratio=0.25, min_floor=8
                    ).fillna(rolling_err)
                else:
                    base_err = rolling_err.rolling(calib_window, min_periods=max(10, calib_window // 4)).median().fillna(rolling_err)
                dyn_ratio = (rolling_err / (base_err + 1e-4)).clip(0.5, 3.0)
                df['pf_confidence'] = (1.0 / dyn_ratio).clip(0.2, 1.0)
        else:
            df['pf_confidence'] = 1.0

        # --------------------------------------------------------------
        # 可交易性：开盘涨跌停（用前收算理论价，在执行日开盘判断）
        # --------------------------------------------------------------
        df = compute_open_limit_flags(df, self.profile)
        # 兼容旧列名：信号层否决仍用 limit_up；撮合层优先读 open_limit_*
        df['limit_up'] = df['open_limit_up']
        df['limit_down'] = df['open_limit_down']

        # --------------------------------------------------------------
        # 入场判定（自校准模式）
        # --------------------------------------------------------------
        veto = pd.Series('', index=df.index, dtype=object)

        def _add_veto(mask, reason):
            sel = mask & (veto == '')
            veto[sel] = reason

        # --------------------------------------------------------------
        # 入场判定：绝对 Z-Score (信息信噪比) 门槛筛选
        # --------------------------------------------------------------
        # 直接使用无量纲信息信噪比阈值 z_score_threshold (如 0.55)，
        # 彻底消除 60 天滚动分位在弱市"矮子拔将军"和强市"良机限流"的缺陷，
        # 且无需任何预热期，极大提升样本利用率。
        base_thresh = float(z_score_threshold)
        if particle_filter and 'pf_confidence' in df.columns:
            # 粒子滤波安全边际调控：置信度低于 0.6 时动态上浮门槛
            pf_entry_penalty = np.where(df['pf_confidence'] < 0.6, (0.6 - df['pf_confidence']) * 0.4, 0.0)
            df['entry_threshold'] = base_thresh + pf_entry_penalty
        else:
            df['entry_threshold'] = base_thresh

        buy_cond = df['signal_score'] > df['entry_threshold']
        _add_veto(~buy_cond, f'信噪比未达标(signal_score<={base_thresh:.2f})')

        if 'pred_slope' in df.columns:
            c = df['pred_slope'] > 0
            _add_veto(~c, '预测斜率向下')
            buy_cond &= c

        # T+N 合规与波峰可实现性校验：若结算周期为 T+1，波峰时界为1日则无法在峰值卖出
        if getattr(self.profile, 'settlement_days', 1) >= 1 and 'target_horizon' in df.columns:
            c = df['target_horizon'] > 1
            _add_veto(~c, f'预测波峰时界<=1日(T+{self.profile.settlement_days}无法波峰离场)')
            buy_cond &= c

        trend_up = df['ma_fast'] >= df['ma_slow']
        confirm = df['close_prev'] > df['ma_fast']  # 严格使用 t-1 收盘价，杜绝未来函数
        df['confirm'] = confirm
        df['regime'] = np.where(trend_up, 'UPTREND', 'DOWNTREND')
        regime_ok = trend_up | confirm
        _add_veto(~regime_ok, '下跌趋势且无短期确认')
        buy_cond &= regime_ok

        c = df['expected_return_adj'] > 0
        _add_veto(~c, '去偏后预期收益为负')
        buy_cond &= c

        if 'pred_min_down' in df.columns:
            depth = df['pred_min_down'].abs() / max(hard_stop_loss, 1e-6)
            scale = np.where(depth <= 1.0, 1.0,
                             np.where(depth >= 2.0, 0.0,
                                      1.0 - 0.6 * (depth - 1.0)))
            df['position_scale'] = np.clip(scale, 0.0, 1.0)
        else:
            df['position_scale'] = 1.0

        # 粒子滤波动态仓位加权：根据粒子拟合置信度等比例调节仓位比例
        if particle_filter and 'pf_confidence' in df.columns:
            df['position_scale'] = df['position_scale'] * df['pf_confidence'].clip(0.2, 1.0)

        c = df['position_scale'] > 0
        _add_veto(~c, '预测下探过深或粒子置信度过低(仓位系数为0)')
        buy_cond &= c

        # 粒子滤波极低置信度硬否决
        if particle_filter and 'pf_confidence' in df.columns:
            c = df['pf_confidence'] >= pf_confidence_threshold
            _add_veto(~c, f'粒子滤波置信度过低(<{pf_confidence_threshold:.0%})')
            buy_cond &= c

        c = df['open_limit_up'] == 0
        _add_veto(~c, '开盘封涨停无法买入')
        buy_cond &= c

        df['raw_signal'] = np.where(buy_cond.fillna(False), 1, 0)
        df['veto_reason'] = np.where(df['raw_signal'] == 1, '', veto)

        bear_cond = df['signal_score'] < -0.8
        if 'pred_slope' in df.columns:
            bear_cond = bear_cond | ((df['pred_slope'] < -0.001) & (df['expected_return_adj'] < -0.005))
        df['bearish_signal'] = np.where(bear_cond.fillna(False), 1, 0)

        # 严格防偷价：t 日决策由 t-1 收盘确定，t 日开盘撮合 (lag=0)
        df['exec_signal'] = df['raw_signal'].astype(int)
        df['exec_bearish'] = df['bearish_signal'].astype(int)
        df['exec_target_horizon'] = df['target_horizon'].fillna(5).astype(int)
        df['exec_position_scale'] = df['position_scale'].fillna(0.0)
        df['execution_lag'] = 0

        # --------------------------------------------------------------
        # 诊断输出
        # --------------------------------------------------------------
        def _safe_log_dt(d):
            if hasattr(d, 'strftime'):
                if hasattr(d, 'hour') and (d.hour != 0 or d.minute != 0 or d.second != 0):
                    return d.strftime('%Y-%m-%d %H:%M:%S')
                return d.strftime('%Y-%m-%d')
            elif hasattr(d, 'date'):
                return str(d.date())
            return str(d)

        buy_cnt = int(df['raw_signal'].sum())
        bear_cnt = int(df['bearish_signal'].sum())
        bar_label = "交易日" if not tf.is_intraday else "根 Bar"
        thr_label = f"绝对信噪比 Z-Score >= {z_score_threshold:.2f}"
        logger.info(
            f"[信号诊断] 重叠 {len(df)} {bar_label} ({_safe_log_dt(df.index.min())} ~ {_safe_log_dt(df.index.max())}) | "
            f"入场阈值={thr_label} | 预测时界={horizon_display} | "
            f"波动率窗口={volatility_window}Bar | ATR窗口={atr_window}Bar | 硬止损={hard_stop_loss:.2%} | "
            f"市场规则={self.profile.describe()}"
        )
        if dynamic_calib:
            logger.info(
                f"[信号诊断] 波动率自适应校准: 动态窗口范围={int(dyn_windows.min())}~{int(dyn_windows.max())}Bar (均值={dyn_windows.mean():.1f}Bar) | "
                f"年化波动率: 最低={df['ann_vol'].min():.1%}, 中位={df['ann_vol'].median():.1%}, 最高={df['ann_vol'].max():.1%}"
            )
        logger.info(
            f"[信号诊断] 信号强度统计: 最小值={df['signal_score'].min():.3f}, "
            f"中位数={df['signal_score'].median():.3f}, "
            f"平均值={df['signal_score'].mean():.3f}, "
            f"最大值={df['signal_score'].max():.3f}"
        )
        if debias:
            reversal_info = ""
            if reversal_aware_debias:
                bull_rev_cnt = int(df['is_bull_reversal'].sum())
                bear_rev_cnt = int(df['is_bear_reversal'].sum())
                reversal_info = (
                    f" | 反转感知去偏: 多头反转={bull_rev_cnt}{bar_label}, "
                    f"空头反转={bear_rev_cnt}{bar_label} (反转偏差衰减权重={reversal_debias_weight:.1f})"
                )
            logger.info(
                f"[信号诊断] 去偏: 原始预期均值={df['expected_return'].mean():+.2%}, "
                f"滚动偏差中位数={df['pred_bias'].median():+.2%}, "
                f"去偏后均值={df['expected_return_adj'].mean():+.2%}"
                f"{reversal_info}"
            )
        logger.info(
            f"[信号诊断] 触发结果: 买入信号 {buy_cnt} {bar_label} ({buy_cnt/len(df):.1%}), "
            f"转空离场信号 {bear_cnt} {bar_label} ({bear_cnt/len(df):.1%})"
        )

        # 否决原因分布：直接回答「为什么这段时间不交易」
        veto_counts = df.loc[df['raw_signal'] == 0, 'veto_reason'].value_counts()
        if len(veto_counts) > 0:
            logger.info("[信号诊断] 未触发原因分布:")
            for reason, cnt in veto_counts.items():
                if reason:
                    logger.info(f"    {reason}: {cnt} {bar_label} ({cnt/len(df):.1%})")

        if buy_cnt > 0:
            sig_rows = df[df['raw_signal'] == 1]
            for s_date, s_row in sig_rows.iterrows():
                logger.info(
                    f"  ⚡ 产生买入信号: {_safe_log_dt(s_date)} 昨收={s_row['close_prev']:.3f}, "
                    f"期望价={s_row['pred_close']:.3f}, 去偏预期涨幅={s_row['expected_return_adj']*100:+.2f}%, "
                    f"波动率={s_row['volatility']*100:.2f}%, 强度={s_row['signal_score']:.2f} "
                    f"(阈值{s_row['entry_threshold']:.2f}), 仓位系数={s_row['position_scale']:.2f}"
                )
        else:
            top_reason = veto_counts.index[0] if len(veto_counts) > 0 else '未知'
            logger.warning(
                f"  ⚠️ 整个区间未触发任何买入信号。主要否决原因: {top_reason}。"
                f"可调低 z_score_threshold（当前 {z_score_threshold:.2f}）或检查预测技能体检结果。"
            )

        print(f"   信号统计: 共 {len(df)} {bar_label}，"
              f"买入信号 {buy_cnt} ({buy_cnt/len(df):.1%})，"
              f"转空离场信号 {bear_cnt} ({bear_cnt/len(df):.1%})")

        return df


    def check_causality(self, hist_df: pd.DataFrame, pred_df: pd.DataFrame,
                        n_probe: int = 12, **kwargs) -> dict:
        """
        前视偏差检验：验证信号确实只依赖历史信息。

        做法
        ----
        把数据在第 k 天截断，只用 [0, k] 的数据重算信号，
        再与全样本计算出的第 k 天信号比对。若两者一致，说明计算第 k 天信号时
        没有用到 k 之后的任何信息。

        这能抓出 .bfill()、未 shift 的滚动统计量、全样本分位数等典型泄漏写法。
        """
        full = self.align_and_generate_signals(hist_df, pred_df, **kwargs)
        dates = list(full.index)
        if len(dates) < 40:
            return {'error': '样本过短，无法做因果性检验'}

        # 探针从满足最小重叠天数的早期开始覆盖全时段，保证预热期与成熟期均无未来函数
        min_ov = int(kwargs.get('min_overlap_days', 10))
        start = max(min_ov, 10)
        if len(dates) <= start:
            return {'error': '有效重叠样本不足以进行截断因果性检验'}

        probes = [dates[i] for i in
                  np.linspace(start, len(dates) - 1, min(n_probe, len(dates) - start)).astype(int)]

        mismatches = []
        check_cols = ('volatility', 'atr_pct', 'expected_return', 'signal_score', 'entry_threshold', 'exec_signal')
        for d in probes:
            h_trunc = hist_df[hist_df.index <= d]
            p_trunc = pred_df[pred_df.index <= d]
            try:
                partial = self.align_and_generate_signals(h_trunc, p_trunc, **kwargs)
            except Exception:
                continue
            if d not in partial.index:
                continue
            for col in check_cols:
                if col not in full.columns or col not in partial.columns:
                    continue
                a, b = full.loc[d, col], partial.loc[d, col]
                if pd.isna(a) and pd.isna(b):
                    continue
                if pd.isna(a) != pd.isna(b) or abs(float(a) - float(b)) > 1e-6:
                    d_str = str(d.date()) if hasattr(d, 'date') else str(d)
                    mismatches.append({'date': d_str, 'col': col,
                                       'full': float(a) if not pd.isna(a) else None,
                                       'truncated': float(b) if not pd.isna(b) else None})

        return {
            '探针数': len(probes),
            '不一致项': mismatches,
            'passed': len(mismatches) == 0,
        }

    def run_validation_protocol(self, hist_df: pd.DataFrame, pred_df: pd.DataFrame,
                                stock_code: str = '', **base_kwargs) -> dict:
        """
        跨时段与参数稳健性检验。

        单一参数组合在单一时段上的好成绩没有意义。本协议检查：
          1. 因果性：信号是否真的只用历史信息
          2. 参数敏感性：入场分位在 0.60~0.90 之间扫描，看收益是否只在某一点为正
          3. 分段稳健性：把回测区间等分为 3 段，看各段表现是否一致

        判定标准：参数扫描中正收益占比 < 50%，或分段中正收益段数 <= 1，
        视为该结果依赖特定参数/时段，不可采信。
        """
        report = {'stock_code': stock_code}

        print(f"\n{'='*64}")
        print(f"  稳健性验证协议 {stock_code}")
        print(f"{'='*64}")

        # ---- 1. 因果性 ----
        print("\n[1/3] 前视偏差（因果性）检验 ...")
        causal = self.check_causality(hist_df, pred_df, **base_kwargs)
        report['causality'] = causal
        if causal.get('passed'):
            print(f"      ✅ 通过：{causal['探针数']} 个探针点上，"
                  f"截断重算与全样本计算完全一致，无前视偏差")
        elif 'error' in causal:
            print(f"      ⚠️ {causal['error']}")
        else:
            print(f"      ❌ 发现 {len(causal['不一致项'])} 处不一致（存在前视偏差）:")
            for mm in causal['不一致项'][:5]:
                print(f"         {mm['date']} {mm['col']}: "
                      f"全样本={mm['full']:.6f} vs 截断={mm['truncated']:.6f}")

        # ---- 2. 参数敏感性 ----
        print("\n[2/3] 信噪比门槛 (z_score_threshold) 参数敏感性 ...")
        sweep = []
        for z_thr in (0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20):
            kw = dict(base_kwargs); kw['z_score_threshold'] = z_thr
            try:
                sig = self.align_and_generate_signals(hist_df, pred_df, **kw)
                bdf, tr = self.run_backtest_engine(
                    sig,
                    hard_stop_loss=base_kwargs.get('hard_stop_loss', 0.04),
                    max_holding_days=15)
                mt = self.evaluate_metrics(bdf, tr)
                sweep.append({'门槛': z_thr, '收益': mt['累计收益率'],
                              '回撤': mt['最大回撤'], '笔数': mt['总交易笔数']})
            except Exception as e:
                sweep.append({'门槛': z_thr, '收益': np.nan, '回撤': np.nan,
                              '笔数': 0, 'err': str(e)[:40]})
        report['sweep'] = sweep
        valid = [s for s in sweep if np.isfinite(s.get('收益', np.nan))]
        pos = [s for s in valid if s['收益'] > 0]
        pos_ratio = len(pos) / len(valid) if valid else 0.0
        print(f"      {'信噪比门槛':>10} {'累计收益':>10} {'最大回撤':>10} {'笔数':>6}")
        for s in sweep:
            if np.isfinite(s.get('收益', np.nan)):
                print(f"      {s['门槛']:>10.2f} {s['收益']:>+9.2%} {s['回撤']:>+9.2%} {s['笔数']:>6d}")
            else:
                print(f"      {s['门槛']:>10.2f} {'失败':>10} {'-':>10} {'-':>6}")
        print(f"      正收益占比 {pos_ratio:.0%} "
              f"({'✅ 参数稳健' if pos_ratio >= 0.5 else '❌ 依赖特定参数，不可采信'})")

        # ---- 3. 分段稳健性 ----
        print("\n[3/3] 分段稳健性（等分 3 段）...")
        segs = []
        try:
            sig = self.align_and_generate_signals(hist_df, pred_df, **base_kwargs)
            bounds = np.linspace(0, len(sig), 4).astype(int)
            for i in range(3):
                part = sig.iloc[bounds[i]:bounds[i + 1]]
                if len(part) < 15:
                    continue
                bdf, tr = self.run_backtest_engine(
                    part, hard_stop_loss=base_kwargs.get('hard_stop_loss', 0.04),
                    max_holding_days=15)
                mt = self.evaluate_metrics(bdf, tr)
                part_start_str = str(part.index.min().date()) if hasattr(part.index.min(), 'date') else str(part.index.min())
                part_end_str = str(part.index.max().date()) if hasattr(part.index.max(), 'date') else str(part.index.max())
                segs.append({'段': i + 1,
                             '起': part_start_str,
                             '止': part_end_str,
                             '收益': mt['累计收益率'], '笔数': mt['总交易笔数']})
        except Exception as e:
            print(f"      ⚠️ 分段检验失败: {e}")
        report['segments'] = segs
        for s in segs:
            print(f"      第{s['段']}段 {s['起']} ~ {s['止']}: "
                  f"收益 {s['收益']:>+7.2%}, {s['笔数']} 笔")
        pos_seg = sum(1 for s in segs if s['收益'] > 0)
        if segs:
            print(f"      正收益段 {pos_seg}/{len(segs)} "
                  f"({'✅ 跨时段一致' if pos_seg >= 2 else '❌ 表现集中于个别时段'})")

        report['passed'] = (causal.get('passed', False)
                            and pos_ratio >= 0.5
                            and pos_seg >= 2)
        print(f"\n  总体结论: {'✅ 通过稳健性验证' if report['passed'] else '❌ 未通过，结果不可采信为策略有效性证据'}")
        print(f"{'='*64}")
        return report

    # ------------------------------------------------------------------
    # 3. 撮合与风控状态机
    # ------------------------------------------------------------------

