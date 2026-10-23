"""
backtest/core/resampler.py
交易所交易时段感知的高精度 K 线合成引擎 (Session-Aware Resampler)

核心解决：
1. 交易所时段不规则与午休断点问题：
   - A 股 (CN): 上午 09:30-11:30 (2小时)，下午 13:00-15:00 (2小时)。
     - 60分钟线：严格合成 4 根 Bar (10:30, 11:30, 14:00, 15:00)。杜绝标准 resample('1h') 产生的午休断裂与空 Bar。
     - 30分钟线：严格合成 8 根 Bar (10:00, 10:30, 11:00, 11:30, 13:30, 14:00, 14:30, 15:00)。
     - 15分钟线：严格合成 16 根 Bar。
     - 5分钟线：严格合成 48 根 Bar。
   - 港股 (HK): 上午 09:30-12:00 (2.5小时)，下午 13:00-16:00 (3.0小时)。
     - 30分钟线：严格合成 11 根 Bar (上午 5 根 + 下午 6 根)。
   - 美股 (US): 09:30-16:00 (6.5小时连续交易)。
     - 30分钟线：严格合成 13 根 Bar。
     - 60分钟线：严格合成 7 根 Bar。
2. OHLCV 物理守恒：
   - open: 第一根 Bar 的 open
   - high: 窗口内最高价 max(high)
   - low: 窗口内最低价 min(low)
   - close: 最后一根 Bar 的 close
   - volume: 成交量严格相加 sum(volume)
   - amount: 成交额严格相加 sum(amount)
3. 严格自然日隔离：绝不跨日拼接，隔夜不污染。
"""

from __future__ import annotations

import logging
from typing import Union, Optional, List, Tuple
from datetime import time, datetime
import pandas as pd
import numpy as np

from .types import TimeFrame
from .timeframes import detect_timeframe

logger = logging.getLogger("KronosBacktest")


class SessionResampler:
    """交易时段感知的多周期 K 线合成器"""

    # 预定义各市场右闭合聚合时间节点 (HH:MM:SS)
    CN_60M_CUTOFFS = [
        time(10, 30, 0),
        time(11, 30, 0),
        time(14, 0, 0),
        time(15, 0, 0),
    ]

    CN_30M_CUTOFFS = [
        time(10, 0, 0),
        time(10, 30, 0),
        time(11, 0, 0),
        time(11, 30, 0),
        time(13, 30, 0),
        time(14, 0, 0),
        time(14, 30, 0),
        time(15, 0, 0),
    ]

    CN_15M_CUTOFFS = [
        time(9, 45, 0),
        time(10, 0, 0),
        time(10, 15, 0),
        time(10, 30, 0),
        time(10, 45, 0),
        time(11, 0, 0),
        time(11, 15, 0),
        time(11, 30, 0),
        time(13, 15, 0),
        time(13, 30, 0),
        time(13, 45, 0),
        time(14, 0, 0),
        time(14, 15, 0),
        time(14, 30, 0),
        time(14, 45, 0),
        time(15, 0, 0),
    ]

    HK_30M_CUTOFFS = [
        time(10, 0, 0),
        time(10, 30, 0),
        time(11, 0, 0),
        time(11, 30, 0),
        time(12, 0, 0),
        time(13, 30, 0),
        time(14, 0, 0),
        time(14, 30, 0),
        time(15, 0, 0),
        time(15, 30, 0),
        time(16, 0, 0),
    ]

    HK_60M_CUTOFFS = [
        time(10, 30, 0),
        time(11, 30, 0),
        time(12, 0, 0),
        time(14, 0, 0),
        time(15, 0, 0),
        time(16, 0, 0),
    ]

    US_30M_CUTOFFS = [
        time(10, 0, 0),
        time(10, 30, 0),
        time(11, 0, 0),
        time(11, 30, 0),
        time(12, 0, 0),
        time(12, 30, 0),
        time(13, 0, 0),
        time(13, 30, 0),
        time(14, 0, 0),
        time(14, 30, 0),
        time(15, 0, 0),
        time(15, 30, 0),
        time(16, 0, 0),
    ]

    US_60M_CUTOFFS = [
        time(10, 30, 0),
        time(11, 30, 0),
        time(12, 30, 0),
        time(13, 30, 0),
        time(14, 30, 0),
        time(15, 30, 0),
        time(16, 0, 0),
    ]

    @classmethod
    def get_session_cutoffs(
        cls,
        target_tf: TimeFrame,
        market: str = "CN",
    ) -> Optional[List[time]]:
        """获取目标市场与时间粒度对应的标准切片节点"""
        m = (market or "CN").upper()
        if m == "CN":
            if target_tf == TimeFrame.H1:
                return cls.CN_60M_CUTOFFS
            elif target_tf == TimeFrame.M30:
                return cls.CN_30M_CUTOFFS
            elif target_tf == TimeFrame.M15:
                return cls.CN_15M_CUTOFFS
        elif m == "HK":
            if target_tf == TimeFrame.H1:
                return cls.HK_60M_CUTOFFS
            elif target_tf == TimeFrame.M30:
                return cls.HK_30M_CUTOFFS
        elif m == "US":
            if target_tf == TimeFrame.H1:
                return cls.US_60M_CUTOFFS
            elif target_tf == TimeFrame.M30:
                return cls.US_30M_CUTOFFS
        return None

    @classmethod
    def resample_to_timeframe(
        cls,
        df: pd.DataFrame,
        target_timeframe: Union[str, TimeFrame],
        market: str = "CN",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        amount_col: Optional[str] = "amount",
    ) -> pd.DataFrame:
        """
        全功能交易时段感知 K 线合成主入口。
        自动兼容任意低级别输入 (1m/5m/15m) 并精确合成为 (5m/15m/30m/60m/1d)。
        """
        if df.empty:
            return df.copy()

        target_tf = TimeFrame.from_str(target_timeframe)
        current_tf = detect_timeframe(df)

        # 若目标粒度与当前粒度相同，直接返回副本
        if current_tf == target_tf:
            return df.copy()

        # 校验：严禁低频下采样为高频 (如日线合成1分钟线)
        if not current_tf.is_intraday and target_tf.is_intraday:
            raise ValueError(
                f"[RESAMPLE ERROR] 无法将低频数据 ({current_tf.label}) 下采样为高频数据 ({target_tf.label})！"
            )
        if current_tf.is_intraday and target_tf.is_intraday and current_tf.minutes > target_tf.minutes:
            raise ValueError(
                f"[RESAMPLE ERROR] 无法将当前周期 ({current_tf.label}, {current_tf.minutes}m) "
                f"下采样为更细周期 ({target_tf.label}, {target_tf.minutes}m)！"
            )

        # 确保 DataFrame 索引为单调递增的 DatetimeIndex
        work_df = df.copy()
        if not isinstance(work_df.index, pd.DatetimeIndex):
            time_cols = [c for c in ("date", "datetime", "timestamp", "time", "时间", "日期") if c in work_df.columns]
            if time_cols:
                work_df["__datetime"] = pd.to_datetime(work_df[time_cols[0]])
                work_df = work_df.set_index("__datetime")
            else:
                work_df.index = pd.to_datetime(work_df.index)
        work_df = work_df.sort_index()

        # 1. 如果目标是日线 (1d/D1)
        if target_tf == TimeFrame.D1:
            return cls._resample_to_daily(
                work_df, open_col, high_col, low_col, close_col, volume_col, amount_col
            )

        # 2. 如果存在特定的 Session Cutoffs 规则（如 A股 30m / 60m / 15m）
        cutoffs = cls.get_session_cutoffs(target_tf, market=market)
        if cutoffs:
            return cls._resample_by_cutoffs(
                work_df, cutoffs, target_tf, open_col, high_col, low_col, close_col, volume_col, amount_col
            )

        # 3. 常规分周期聚合（按交易自然日切片并使用 Pandas Resample 兜底）
        return cls._resample_general_intraday(
            work_df, target_tf, open_col, high_col, low_col, close_col, volume_col, amount_col
        )

    @classmethod
    def _resample_to_daily(
        cls,
        df: pd.DataFrame,
        open_col: str,
        high_col: str,
        low_col: str,
        close_col: str,
        volume_col: str,
        amount_col: Optional[str],
    ) -> pd.DataFrame:
        """日内高频 K 线严格按交易自然日聚合成日 K 线"""
        agg_dict = {
            open_col: "first",
            high_col: "max",
            low_col: "min",
            close_col: "last",
            volume_col: "sum",
        }
        if amount_col and amount_col in df.columns:
            agg_dict[amount_col] = "sum"

        for col in df.columns:
            if col not in agg_dict:
                agg_dict[col] = "last"

        # 严格按自然日分组，杜绝跨天混淆
        daily_df = df.groupby(df.index.date).agg(agg_dict)
        daily_df.index = pd.to_datetime(daily_df.index)
        daily_df.index.name = "date"
        return daily_df.dropna(subset=[open_col, close_col])

    @classmethod
    def _resample_by_cutoffs(
        cls,
        df: pd.DataFrame,
        cutoffs: List[time],
        target_tf: TimeFrame,
        open_col: str,
        high_col: str,
        low_col: str,
        close_col: str,
        volume_col: str,
        amount_col: Optional[str],
    ) -> pd.DataFrame:
        """
        按交易所指定时间切片节点进行精准右闭合聚合。
        彻底消除 11:30~13:00 午休及跨日产生的空 Bar。
        """
        agg_dict = {
            open_col: "first",
            high_col: "max",
            low_col: "min",
            close_col: "last",
            volume_col: "sum",
        }
        if amount_col and amount_col in df.columns:
            agg_dict[amount_col] = "sum"
        for col in df.columns:
            if col not in agg_dict:
                agg_dict[col] = "last"

        # 为每一行映射归属的 Bar 时间戳
        # 规则：寻找大于等于当前时间点 t.time() 的第一个 cutoff；
        # 若超过最后一个 cutoff，则归入最后一个 cutoff (例如尾盘集合竞价 15:00:xx 归入 15:00:00)
        daily_dfs = []
        for d, day_group in df.groupby(df.index.date):
            target_timestamps = []
            for ts in day_group.index:
                t_val = ts.time()
                assigned_cutoff = None
                for c in cutoffs:
                    if t_val <= c:
                        assigned_cutoff = c
                        break
                if assigned_cutoff is None:
                    assigned_cutoff = cutoffs[-1]
                # 拼接生成该 Bar 的完整绝对时间戳
                target_timestamps.append(datetime.combine(d, assigned_cutoff))

            day_group = day_group.copy()
            day_group["__bar_timestamp"] = target_timestamps
            resampled_day = day_group.groupby("__bar_timestamp").agg(agg_dict)
            daily_dfs.append(resampled_day)

        if not daily_dfs:
            return pd.DataFrame()

        result = pd.concat(daily_dfs)
        result.index = pd.to_datetime(result.index)
        result.index.name = "date"
        return result.dropna(subset=[open_col, close_col]).sort_index()

    @classmethod
    def _resample_general_intraday(
        cls,
        df: pd.DataFrame,
        target_tf: TimeFrame,
        open_col: str,
        high_col: str,
        low_col: str,
        close_col: str,
        volume_col: str,
        amount_col: Optional[str],
    ) -> pd.DataFrame:
        """通用日内 Resample（按自然日分组后在日内进行右闭合聚合）"""
        rule = f"{target_tf.minutes}min"
        agg_dict = {
            open_col: "first",
            high_col: "max",
            low_col: "min",
            close_col: "last",
            volume_col: "sum",
        }
        if amount_col and amount_col in df.columns:
            agg_dict[amount_col] = "sum"
        for col in df.columns:
            if col not in agg_dict:
                agg_dict[col] = "last"

        daily_dfs = []
        for d, day_group in df.groupby(df.index.date):
            resampled = day_group.resample(rule, label="right", closed="right").agg(agg_dict)
            daily_dfs.append(resampled.dropna(subset=[open_col, close_col]))

        if not daily_dfs:
            return pd.DataFrame()

        result = pd.concat(daily_dfs)
        result.index.name = "date"
        return result.sort_index()
