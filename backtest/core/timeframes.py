"""
backtest/core/timeframes.py
多时间粒度 (Multi-Timeframe) 与交易所交易日历时段工具
支持日线 (1d) 以及 15分钟 (15m)、30分钟 (30m)、60分钟等日内高频 K 线的无缝处理与年化校准。
"""

from __future__ import annotations

from typing import Union, Optional, Tuple
import pandas as pd
import numpy as np

from .types import TimeFrame


# 交易所常规交易时段配置 (本地时间)
MARKET_TRADING_SESSIONS = {
    'CN': [('09:30', '11:30'), ('13:00', '15:00')],                     # 4.0小时 (240分钟)
    'HK': [('09:30', '12:00'), ('13:00', '16:00')],                     # 5.5小时 (330分钟)
    'US': [('09:30', '16:00')],                                         # 6.5小时 (390分钟, 美东时间)
}


def get_annualization_factor(
    timeframe: Union[str, TimeFrame] = TimeFrame.D1,
    market: str = 'CN',
    annual_trading_days: Optional[int] = None,
) -> float:
    """
    计算特定市场与时间粒度下的年化折算因子。
    公式: sqrt(annual_trading_days * bars_per_day)

    示例:
    - CN 日线: sqrt(244 * 1) ≈ 15.62
    - HK 日线: sqrt(246 * 1) ≈ 15.68
    - US 日线: sqrt(252 * 1) ≈ 15.87
    - HK 15分钟: 1日22根Bar, sqrt(246 * 22) = sqrt(5412) ≈ 73.57
    - US 15分钟: 1日26根Bar, sqrt(252 * 26) = sqrt(6552) ≈ 80.94
    - US 30分钟: 1日13根Bar, sqrt(252 * 13) = sqrt(3276) ≈ 57.24
    """
    tf = TimeFrame.from_str(timeframe)
    return tf.annual_factor(market=market, annual_trading_days=annual_trading_days)


def detect_timeframe(df: pd.DataFrame) -> TimeFrame:
    """
    通过 DataFrame 索引或时间列自动推断 K 线时间粒度。
    若时间差中位数为 1 天或以上返回 D1，若在 10~20 分钟返回 M15，20~40 分钟返回 M30 等。
    """
    if df.empty:
        return TimeFrame.D1

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        time_col = None
        for col in ('date', 'datetime', 'timestamp', 'time'):
            if col in df.columns:
                time_col = col
                break
        if time_col:
            idx = pd.to_datetime(df[time_col])
        else:
            return TimeFrame.D1

    if len(idx) < 2:
        return TimeFrame.D1

    # 过滤重复索引计算时间差中位数
    unique_idx = pd.Series(idx).drop_duplicates().sort_values()
    if len(unique_idx) < 2:
        return TimeFrame.D1

    diffs = unique_idx.diff().dropna()
    median_sec = diffs.median().total_seconds()

    if median_sec >= 86400 * 0.8:
        return TimeFrame.D1
    elif 3000 <= median_sec <= 4200:
        return TimeFrame.H1
    elif 1500 <= median_sec < 3000:
        return TimeFrame.M30
    elif 600 <= median_sec < 1500:
        return TimeFrame.M15
    elif 180 <= median_sec < 600:
        return TimeFrame.M5
    elif median_sec < 180:
        return TimeFrame.M1
    return TimeFrame.D1


def resample_bars(
    df: pd.DataFrame,
    target_timeframe: Union[str, TimeFrame],
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: str = 'volume',
    amount_col: Optional[str] = 'amount',
) -> pd.DataFrame:
    """
    将低级别 K 线 (如 1m/5m/15m) 合成为高级别 K 线 (如 30m/60m/1d)。
    严格保持 OHLCV 物理正确性：
    - open: 第一根的 open
    - high: 最大值
    - low: 最小值
    - close: 最后一根的 close
    - volume / amount: 求和
    """
    tf = TimeFrame.from_str(target_timeframe)
    if not isinstance(df.index, pd.DatetimeIndex):
        time_cols = [c for c in ('date', 'datetime', 'timestamp') if c in df.columns]
        if time_cols:
            df = df.copy()
            df.index = pd.to_datetime(df[time_cols[0]])

    rule_map = {
        TimeFrame.M5: '5min',
        TimeFrame.M15: '15min',
        TimeFrame.M30: '30min',
        TimeFrame.H1: '1h',
        TimeFrame.D1: '1D',
    }
    rule = rule_map.get(tf, '1D')

    agg_dict = {
        open_col: 'first',
        high_col: 'max',
        low_col: 'min',
        close_col: 'last',
        volume_col: 'sum',
    }
    if amount_col and amount_col in df.columns:
        agg_dict[amount_col] = 'sum'

    for col in df.columns:
        if col not in agg_dict:
            agg_dict[col] = 'last'

    resampled = df.resample(rule, label='right', closed='right').agg(agg_dict)
    resampled = resampled.dropna(subset=[open_col, close_col])
    return resampled
