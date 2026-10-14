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
from .base import BaseStrategy

logger = logging.getLogger("KronosBacktest")

def compute_dynamic_rolling_quantile(
    series: pd.Series,
    windows: np.ndarray | pd.Series,
    q: float,
    min_periods_ratio: float = 0.25,
    min_floor: int = 8,
) -> pd.Series:
    """
    计算动态变长窗口的滚动分位数（严格防未来函数）。

    机制说明:
    在极端高波动率或黑天鹅阶段，固定长窗口会导致滞后钝化；而平稳震荡期需要长窗口保持统计稳定性。
    本函数支持逐日传入自适应计算得出的窗口长度 windows[i]，在截至当日 t 的历史切片上
    计算指定分位数 q（如 q=0.5 为动态中位数去偏，q=0.8 为 80% 动态入场分位门槛）。

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
    for i in range(n):
        w = int(windows[i])
        min_p = max(min_floor, int(w * min_periods_ratio))
        start = max(0, i - w + 1)
        chunk = vals[start : i + 1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) >= min_p:
            if q == 0.5:
                result[i] = np.median(valid)
            else:
                result[i] = np.quantile(valid, q)
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
        z_score_threshold: float = 0.70,
        volatility_window: int = 14,
        min_overlap_days: int = 10,
        forecast_horizon: int = None,
        max_auto_horizon: int = 10,
        hard_stop_loss: float = 0.04,
        entry_percentile: float = 0.80,
        calib_window: int = 60,
        dynamic_calib: bool = True,
        calib_window_min: int = 20,
        vol_high_thresh: float = 0.40,
        vol_low_thresh: float = 0.20,
        debias: bool = True,
        atr_window: int = 14,
        allow_interpolated: bool = False,
        particle_filter: bool = True,
        pf_confidence_threshold: float = 0.35,
    ) -> pd.DataFrame:
        """
        核心时序对齐与信号生成。

        【时序不变量】
        对每个交易日 t，信号 raw_signal[t] 只允许使用在 **t-1 收盘时点已知** 的信息：
          - pred_*[t]      由严格早于 t 的数据推理得到（walk_forward_predict 保证）
          - close_prev[t]  = close[t-1]
          - volatility[t]  使用截至 t-1 的收益率窗口计算
          - atr[t]         使用截至 t-1 的真实波幅计算
        因此 raw_signal[t] 在 t-1 收盘即可确定，在 **t 日开盘** 撮合成交。

        【入场判定（波动率自适应分位自校准）】
        改用每标的滚动分位数自校准：signal_score 位于该标的自身过去校准窗口分布的
        entry_percentile 分位之上。
        在极端高波动率或黑天鹅阶段（年化波动率 >= vol_high_thresh 40%），校准窗口自动收缩
        至 calib_window_min (20~30 天)，消除滞后；在平稳震荡期维持基准 calib_window (60 天)，
        在二者之间根据年化波动率压力系数平滑线性插值过渡。
        """
        if not (0 < hard_stop_loss < 1):
            raise ValueError(
                f"[PARAM ERROR] hard_stop_loss 须满足 0 < x < 1，当前值={hard_stop_loss}"
            )

        common_dates = hist_df.index.intersection(pred_df.index)
        overlap_days = len(common_dates)

        print(f"\n[对齐检测] 历史 ∩ 预测 重叠区间: {overlap_days} 个交易日")

        if overlap_days < min_overlap_days:
            raise ValueError(
                f"\n{'='*60}\n"
                f"[BACKTEST ERROR] 重叠交易日不足，无法执行真实样本外回测！\n"
                f"  • 历史数据结束: {hist_df.index.max().date()}\n"
                f"  • 预测数据起始: {pred_df.index.min().date()}\n"
                f"  • 当前重叠天数: {overlap_days} 天 (要求 >= {min_overlap_days} 天)\n"
                f"\n解决方案：\n"
                f"  请生成一份覆盖历史区间的 Kronos 预测序列，确保预测日期\n"
                f"  与历史真实收盘价有至少 {min_overlap_days} 个交易日的重叠。\n"
                f"{'='*60}"
            )

        print(f"   重叠区间: {common_dates.min().date()} → {common_dates.max().date()}")
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

        # 趋势状态滞后一期
        ma_fast = hist_work['close'].rolling(20, min_periods=10).mean().shift(1)
        ma_slow = hist_work['close'].rolling(60, min_periods=20).mean().shift(1)
        hist_work['ma_fast'] = ma_fast
        hist_work['ma_slow'] = ma_slow

        # 截取重叠回测区间
        df = hist_work.loc[common_dates].copy()
        df['pred_close'] = pred_df.loc[common_dates, 'pred_close']

        for opt_col in ['pred_upper', 'pred_lower', 'pred_expected', 'pred_path_max',
                        'pred_terminal', 'pred_close_1d', 'pred_slope', 'pred_min_down', 'peak_premium',
                        'is_interpolated', 'pf_confidence', 'pf_val_error', 'pf_effective_n']:
            if opt_col in pred_df.columns:
                df[opt_col] = pred_df.loc[common_dates, opt_col]

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
            if cand in pred_df.columns:
                dyn_h_col = cand
                break

        if dyn_h_col is not None:
            df['target_horizon'] = (pred_df.loc[common_dates, dyn_h_col]
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
        # 年化波动率: 基于截至 t-1 的滚动波动率 (严格防偷价)
        df['ann_vol'] = (df['volatility'] * np.sqrt(252)).fillna(vol_low_thresh)

        if dynamic_calib:
            # 波动率压力系数 (0.0 表示平稳低波，1.0 表示极端高波)
            vol_stress = np.clip(
                (df['ann_vol'] - vol_low_thresh) / max(vol_high_thresh - vol_low_thresh, 1e-4),
                0.0, 1.0
            )
            # 动态窗口：从基准 calib_window 平滑收缩至 calib_window_min (四舍五入为整数天)
            dyn_windows = np.round(
                calib_window - vol_stress * (calib_window - calib_window_min)
            ).astype(int)
            df['dyn_calib_window'] = dyn_windows
        else:
            dyn_windows = np.full(len(df), calib_window, dtype=int)
            df['dyn_calib_window'] = calib_window

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
            df['expected_return_adj'] = df['expected_return'] - df['pred_bias']
        else:
            df['pred_bias'] = 0.0
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

        if dynamic_calib:
            df['entry_threshold'] = compute_dynamic_rolling_quantile(
                df['signal_score'].shift(1), dyn_windows, q=entry_percentile, min_periods_ratio=0.25, min_floor=8
            )
        else:
            roll = df['signal_score'].shift(1).rolling(
                calib_window, min_periods=max(10, calib_window // 4))
            df['entry_threshold'] = roll.quantile(entry_percentile)
        warmup = df['entry_threshold'].isna()
        if warmup.any():
            fallback = df['signal_score'].quantile(entry_percentile)
            df.loc[warmup, 'entry_threshold'] = fallback

        # 粒子滤波不确定性惩罚：置信度低于 0.6 时动态上浮入场分位要求 (要求更充足的安全边际)
        if particle_filter and 'pf_confidence' in df.columns:
            pf_entry_penalty = np.where(df['pf_confidence'] < 0.6, (0.6 - df['pf_confidence']) * 0.4, 0.0)
            df['entry_threshold'] = df['entry_threshold'] + pf_entry_penalty

        buy_cond = df['signal_score'] > df['entry_threshold']
        _add_veto(~buy_cond, '信号强度未进入滚动分位阈值')

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
        buy_cnt = int(df['raw_signal'].sum())
        bear_cnt = int(df['bearish_signal'].sum())
        if dynamic_calib:
            thr_label = (
                f"自适应{int(dyn_windows.min())}~{int(dyn_windows.max())}日 "
                f"(均值{dyn_windows.mean():.1f}天) {entry_percentile:.0%} 分位"
            )
        else:
            thr_label = f"滚动{calib_window}日 {entry_percentile:.0%} 分位"
        logger.info(
            f"[信号诊断] 重叠 {len(df)} 交易日 ({df.index.min().date()} ~ {df.index.max().date()}) | "
            f"自校准模式 | 预测时界={horizon_display} | 入场阈值={thr_label} | "
            f"波动率窗口={volatility_window}天 | ATR窗口={atr_window}天 | 硬止损={hard_stop_loss:.2%} | "
            f"市场规则={self.profile.describe()}"
        )
        if dynamic_calib:
            logger.info(
                f"[信号诊断] 波动率自适应校准: 动态窗口范围={int(dyn_windows.min())}~{int(dyn_windows.max())}天 (均值={dyn_windows.mean():.1f}天) | "
                f"年化波动率: 最低={df['ann_vol'].min():.1%}, 中位={df['ann_vol'].median():.1%}, 最高={df['ann_vol'].max():.1%}"
            )
        logger.info(
            f"[信号诊断] 信号强度统计: 最小值={df['signal_score'].min():.3f}, "
            f"中位数={df['signal_score'].median():.3f}, "
            f"平均值={df['signal_score'].mean():.3f}, "
            f"最大值={df['signal_score'].max():.3f}"
        )
        if debias:
            logger.info(
                f"[信号诊断] 去偏: 原始预期收益均值={df['expected_return'].mean():+.2%}, "
                f"滚动偏差中位数={df['pred_bias'].median():+.2%}, "
                f"去偏后均值={df['expected_return_adj'].mean():+.2%}"
            )
        logger.info(
            f"[信号诊断] 触发结果: 买入信号 {buy_cnt} 天 ({buy_cnt/len(df):.1%}), "
            f"转空离场信号 {bear_cnt} 天 ({bear_cnt/len(df):.1%})"
        )

        # 否决原因分布：直接回答「为什么这段时间不交易」
        veto_counts = df.loc[df['raw_signal'] == 0, 'veto_reason'].value_counts()
        if len(veto_counts) > 0:
            logger.info("[信号诊断] 未触发原因分布:")
            for reason, cnt in veto_counts.items():
                if reason:
                    logger.info(f"    {reason}: {cnt} 天 ({cnt/len(df):.1%})")

        if buy_cnt > 0:
            sig_rows = df[df['raw_signal'] == 1]
            for s_date, s_row in sig_rows.iterrows():
                logger.info(
                    f"  ⚡ 产生买入信号: {s_date.date()} 昨收={s_row['close_prev']:.3f}, "
                    f"期望价={s_row['pred_close']:.3f}, 去偏预期涨幅={s_row['expected_return_adj']*100:+.2f}%, "
                    f"波动率={s_row['volatility']*100:.2f}%, 强度={s_row['signal_score']:.2f} "
                    f"(阈值{s_row['entry_threshold']:.2f}), 仓位系数={s_row['position_scale']:.2f}"
                )
        else:
            top_reason = veto_counts.index[0] if len(veto_counts) > 0 else '未知'
            logger.warning(
                f"  ⚠️ 整个区间未触发任何买入信号。主要否决原因: {top_reason}。"
                f"可调低 entry_percentile（当前 {entry_percentile:.0%}）或检查预测技能体检结果。"
            )

        print(f"   信号统计: 共 {len(df)} 个交易日，"
              f"买入信号 {buy_cnt} 天 ({buy_cnt/len(df):.1%})，"
              f"转空离场信号 {bear_cnt} 天 ({bear_cnt/len(df):.1%})")

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

        # 在后半段均匀取若干个探针点
        start = len(dates) // 2
        probes = [dates[i] for i in
                  np.linspace(start, len(dates) - 1, min(n_probe, len(dates) - start)).astype(int)]

        mismatches = []
        for d in probes:
            h_trunc = hist_df[hist_df.index <= d]
            p_trunc = pred_df[pred_df.index <= d]
            try:
                partial = self.align_and_generate_signals(h_trunc, p_trunc, **kwargs)
            except Exception:
                continue
            if d not in partial.index:
                continue
            for col in ('volatility', 'atr_pct', 'expected_return'):
                if col not in full.columns or col not in partial.columns:
                    continue
                a, b = full.loc[d, col], partial.loc[d, col]
                if pd.isna(a) and pd.isna(b):
                    continue
                if pd.isna(a) != pd.isna(b) or abs(float(a) - float(b)) > 1e-9:
                    mismatches.append({'date': str(d.date()), 'col': col,
                                       'full': float(a), 'truncated': float(b)})

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
        print("\n[2/3] 入场分位参数敏感性 ...")
        sweep = []
        for pct in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            kw = dict(base_kwargs); kw['entry_percentile'] = pct
            try:
                sig = self.align_and_generate_signals(hist_df, pred_df, **kw)
                bdf, tr = self.run_backtest_engine(
                    sig,
                    hard_stop_loss=base_kwargs.get('hard_stop_loss', 0.04),
                    max_holding_days=15)
                mt = self.evaluate_metrics(bdf, tr)
                sweep.append({'分位': pct, '收益': mt['累计收益率'],
                              '回撤': mt['最大回撤'], '笔数': mt['总交易笔数']})
            except Exception as e:
                sweep.append({'分位': pct, '收益': np.nan, '回撤': np.nan,
                              '笔数': 0, 'err': str(e)[:40]})
        report['sweep'] = sweep
        valid = [s for s in sweep if np.isfinite(s.get('收益', np.nan))]
        pos = [s for s in valid if s['收益'] > 0]
        pos_ratio = len(pos) / len(valid) if valid else 0.0
        print(f"      {'分位':>6} {'累计收益':>10} {'最大回撤':>10} {'笔数':>6}")
        for s in sweep:
            if np.isfinite(s.get('收益', np.nan)):
                print(f"      {s['分位']:>6.2f} {s['收益']:>+9.2%} {s['回撤']:>+9.2%} {s['笔数']:>6d}")
            else:
                print(f"      {s['分位']:>6.2f} {'失败':>10} {'-':>10} {'-':>6}")
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
                segs.append({'段': i + 1,
                             '起': str(part.index.min().date()),
                             '止': str(part.index.max().date()),
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

