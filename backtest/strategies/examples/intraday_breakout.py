"""
backtest/strategies/examples/intraday_breakout.py
日内多周期突破策略示例 (支持 15m / 30m / 日线，港美股极佳适配)
展示如何继承 BaseStrategy 极简实现一个生产级量化策略：
1. 均线与通道动量过滤
2. 动态 ATR 波动率自适应入场
3. 自动生成与统一撮合引擎、出场状态机及实盘网关兼容的信号矩阵
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from ..base import BaseStrategy
from ...core.types import TimeFrame
from ...core.market_rules import MarketProfile


class IntradayBreakoutStrategy(BaseStrategy):
    """
    高流动性港美股日内波段突破策略 (Intraday Breakout & Trend Following)
    - 周期适配: 15m, 30m, 60m 或 日线
    - 规则:
      1. 顺势判定: 价格站上快速 EMA (如 EMA20 > EMA50)
      2. 突破入场: 突破前 N 根 Bar 的最高价 (Donchian 通道高点)
      3. 动态止损: 基于 ATR 的波动率止损与目标时界设置
    """

    def __init__(
        self,
        name: str = "IntradayBreakout_15m",
        fast_window: int = 20,
        slow_window: int = 50,
        channel_window: int = 20,
        atr_window: int = 14,
        target_horizon_bars: int = 12,
        profile: Optional[MarketProfile] = None,
    ):
        super().__init__(name=name)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.channel_window = channel_window
        self.atr_window = atr_window
        self.target_horizon_bars = target_horizon_bars
        self.profile = profile or MarketProfile.for_market('US')

    def generate_signals(
        self,
        hist_df: pd.DataFrame,
        pred_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        在行情矩阵上计算技术指标并生成买入、离场与否决信号。
        严格防偷价 (Anti-Lookahead): 使用 shift(1) 保证 t 决策只用 t-1 之前的信息。
        """
        df = hist_df.copy()

        # 1. 基础时序指标 (严守 t-1 因果律)
        close = df['close']
        high = df['high']
        low = df['low']

        # 前收盘价
        df['close_prev'] = close.shift(1)

        # 均线系统
        df['ema_fast'] = close.ewm(span=self.fast_window, adjust=False).mean().shift(1)
        df['ema_slow'] = close.ewm(span=self.slow_window, adjust=False).mean().shift(1)

        # 唐奇安通道 (前 N 根最高最低)
        df['channel_high'] = high.rolling(window=self.channel_window).max().shift(1)
        df['channel_low'] = low.rolling(window=self.channel_window).min().shift(1)

        # ATR 波动率
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_window).mean().shift(1)
        df['atr'] = atr
        df['atr_pct'] = atr / df['close_prev']
        df['volatility'] = df['close_prev'].pct_change().rolling(self.atr_window).std()

        # 2. 信号生成与守门人逻辑
        # 条件1: 顺大势 (EMA fast > slow 且 昨收 > EMA fast)
        trend_bullish = (df['ema_fast'] > df['ema_slow']) & (df['close_prev'] > df['ema_fast'])
        
        # 条件2: 突破通道上轨
        breakout = df['close_prev'] >= df['channel_high'] * 0.998

        # 买入信号初步触发
        raw_buy = trend_bullish & breakout

        # 初始化输出字段
        df['exec_signal'] = 0
        df['exec_bearish'] = 0
        df['exec_position_scale'] = 1.0
        df['exec_target_horizon'] = self.target_horizon_bars
        df['veto_reason'] = ""

        # 填充信号与否决原因
        for i in range(len(df)):
            if raw_buy.iloc[i]:
                df.iloc[i, df.columns.get_loc('exec_signal')] = 1
                df.iloc[i, df.columns.get_loc('exec_position_scale')] = 1.0
            else:
                reasons = []
                if not trend_bullish.iloc[i]:
                    reasons.append("逆大势(EMA未多头排列)")
                if not breakout.iloc[i]:
                    reasons.append("未突破通道上轨")
                df.iloc[i, df.columns.get_loc('veto_reason')] = "; ".join(reasons)

            # 转空离场条件: 跌破通道下轨或 EMA 死叉
            if df['close_prev'].iloc[i] < df['channel_low'].iloc[i] or df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i]:
                df.iloc[i, df.columns.get_loc('exec_bearish')] = 1

        # 剥离前序因指标计算产生的空值
        clean_df = df.dropna(subset=['close_prev', 'ema_slow', 'atr']).copy()
        return clean_df
