"""
backtest/core/mtf_aligner.py
跨时间周期特征对齐器 (Multi-Timeframe Feature Aligner)
实现因果律严格守恒的跨周期特征提取与映射，彻底消除未来函数 (Zero Lookahead Bias)。

核心能力:
1. align_higher_to_lower:
   将大级别数据（如日线 Higher Timeframe）严格基于已完成 Bar 映射到小级别数据（如 30m / 60m Lower Timeframe）。
   关键防偷价设计:
   - 日线对齐日内 Bar 时：小周期在 t 日（哪怕是 14:30）只能看到 t-1 日收盘的大周期指标！
   - 小周期 Bar 时间戳早于大周期完成时刻，强制通过 shift(1) 将大周期闭合切片后置匹配。
2. auto_align_dual_timeframe:
   若只有单份高频数据 (如 30m)，自动利用 SessionResampler 在内存中无损合成大周期并计算特征，然后以零未来函数精确对齐回小周期。
3. broadcast_daily_to_intraday:
   支持将日度数据（如日线预测、宏观特征）按交易日精确广播到日内所有高频 Bar 上。
"""

from __future__ import annotations

import logging
from typing import Optional, List, Callable, Union, Any
from datetime import datetime, date
import pandas as pd
import numpy as np

from .types import TimeFrame
from .timeframes import detect_timeframe
from .resampler import SessionResampler

logger = logging.getLogger("KronosBacktest")


class MultiTimeframeAligner:
    """因果律严格守恒的跨周期特征对齐器"""

    @classmethod
    def align_higher_to_lower(
        cls,
        lower_df: pd.DataFrame,
        higher_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        prefix: str = "htf_",
        shift_higher: bool = True,
    ) -> pd.DataFrame:
        """
        将大级别 DataFrame 的特征对齐到小级别 DataFrame 上。

        参数:
            lower_df: 小级别高频数据 (如 30m, 必须具备 DatetimeIndex)
            higher_df: 大级别数据 (如 日线或 60m, 必须具备 DatetimeIndex)
            feature_cols: 需要对齐的大级别列名列表 (若为 None 则对齐全部非 OHLCV 衍生列或全部列)
            prefix: 对齐后大级别列名前缀 (默认 "htf_")
            shift_higher: 是否强制向前位移 1 期以彻底杜绝未来函数 (默认 True)

        返回:
            pd.DataFrame: 与 lower_df 索引相同，且拼接了已对齐的大级别特征列。
        """
        if lower_df.empty:
            return lower_df.copy()
        if higher_df.empty:
            return lower_df.copy()

        l_df = lower_df.copy()
        h_df = higher_df.copy()

        if not isinstance(l_df.index, pd.DatetimeIndex):
            l_df.index = pd.to_datetime(l_df.index)
        if not isinstance(h_df.index, pd.DatetimeIndex):
            h_df.index = pd.to_datetime(h_df.index)

        l_df = l_df.sort_index()
        h_df = h_df.sort_index()

        lower_tf = detect_timeframe(l_df)
        higher_tf = detect_timeframe(h_df)

        target_cols = feature_cols if feature_cols is not None else list(h_df.columns)
        h_subset = h_df[target_cols].copy()

        # 模式 A: 大周期为日线 (D1)，小周期为日内 (M1, M5, M15, M30, H1)
        # 规则：日内任意时间戳 (如 2026-03-02 10:30)，对齐 shifted 后的 2026-03-02 日线行（即 2026-03-01 收盘的特征）
        if higher_tf == TimeFrame.D1 and lower_tf.is_intraday:
            if shift_higher:
                h_subset = h_subset.shift(1)
            h_daily = h_subset.copy()
            h_daily['__date_key'] = h_daily.index.date
            h_daily = h_daily.drop_duplicates(subset=['__date_key'], keep='last').set_index('__date_key')

            l_dates = l_df.index.date
            aligned_features = h_daily.reindex(l_dates)
            aligned_features.index = l_df.index
            aligned_features = aligned_features.add_prefix(prefix)
            return pd.concat([l_df, aligned_features], axis=1)

        # 模式 B: 通用时间序列基于 merge_asof backward 对齐 (如 15m 对齐 60m，或日内同频对齐)
        # 若 shift_higher 为 True，采用 allow_exact_matches=False 严格匹配 h_time < l_time，
        # 彻底杜绝使用当前时刻正在闭合的大 Bar，同时避免数据序列整体下移一整期的多余滞后。
        l_df_with_key = l_df.copy()
        l_df_with_key['__merge_time'] = l_df_with_key.index
        h_subset_with_key = h_subset.copy()
        h_subset_with_key['__merge_time'] = h_subset_with_key.index

        merged = pd.merge_asof(
            l_df_with_key,
            h_subset_with_key.add_prefix(prefix),
            left_on='__merge_time',
            right_on=f'{prefix}__merge_time',
            direction='backward',
            allow_exact_matches=not shift_higher,
        )
        merged.index = l_df.index
        cols_to_drop = [c for c in ['__merge_time', f'{prefix}__merge_time'] if c in merged.columns]
        return merged.drop(columns=cols_to_drop)

    @classmethod
    def auto_align_dual_timeframe(
        cls,
        lower_df: pd.DataFrame,
        market: str = "CN",
        ma_fast_window: int = 20,
        ma_slow_window: int = 60,
        prefix: str = "htf_",
    ) -> pd.DataFrame:
        """
        全自动双周期对齐引擎：
        从单份日内高频 K 线中，利用 SessionResampler 无损合成日线并计算大级别均线与趋势，
        再严格无未来函数地映射回小周期，实现双周期共振 (Dual-Timeframe Synergy)。
        """
        lower_tf = detect_timeframe(lower_df)
        min_p_fast = max(1, min(5, ma_fast_window))
        min_p_slow = max(1, min(10, ma_slow_window))

        if not lower_tf.is_intraday:
            # 若本身就是日线或更低频，直接计算自身均线并加上前缀
            res = lower_df.copy()
            res[f"{prefix}ma_fast"] = res["close"].rolling(ma_fast_window, min_periods=min_p_fast).mean().shift(1)
            res[f"{prefix}ma_slow"] = res["close"].rolling(ma_slow_window, min_periods=min_p_slow).mean().shift(1)
            res[f"{prefix}trend_up"] = res[f"{prefix}ma_fast"] >= res[f"{prefix}ma_slow"]
            return res

        # 1. 在内存中利用 SessionResampler 聚合成纯净日线
        daily_df = SessionResampler.resample_to_timeframe(
            lower_df,
            target_timeframe=TimeFrame.D1,
            market=market,
        )

        # 2. 在日线级别计算宏观趋势特征
        daily_features = pd.DataFrame(index=daily_df.index)
        daily_features["ma_fast"] = daily_df["close"].rolling(ma_fast_window, min_periods=min_p_fast).mean()
        daily_features["ma_slow"] = daily_df["close"].rolling(ma_slow_window, min_periods=min_p_slow).mean()
        daily_features["trend_up"] = daily_features["ma_fast"] >= daily_features["ma_slow"]
        daily_features["close_prev"] = daily_df["close"]

        # 3. 对齐映射回小周期 (严格 shift(1) 杜绝未来函数)
        aligned = cls.align_higher_to_lower(
            lower_df=lower_df,
            higher_df=daily_features,
            prefix=prefix,
            shift_higher=True,
        )
        return aligned

    @classmethod
    def broadcast_daily_to_intraday(
        cls,
        intraday_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        将按交易日生成的日度数据 (如每日预测 pred_df) 精确广播映射到日内所有 K 线 Bar 上。
        """
        if intraday_df.empty or daily_df.empty:
            return intraday_df.copy()

        i_df = intraday_df.copy()
        d_df = daily_df.copy()

        if not isinstance(i_df.index, pd.DatetimeIndex):
            i_df.index = pd.to_datetime(i_df.index)
        if not isinstance(d_df.index, pd.DatetimeIndex):
            d_df.index = pd.to_datetime(d_df.index)

        target_cols = feature_cols if feature_cols is not None else list(d_df.columns)
        d_sub = d_df[target_cols].copy()
        d_sub['__date_key'] = d_sub.index.date
        d_sub = d_sub.drop_duplicates(subset=['__date_key'], keep='last').set_index('__date_key')

        i_dates = i_df.index.date
        aligned = d_sub.reindex(i_dates)
        aligned.index = i_df.index

        # 合并 (若列名已存在则使用广播后的值覆盖或补充)
        for c in target_cols:
            i_df[c] = aligned[c]
        return i_df
