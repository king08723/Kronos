"""
backtest/strategies
策略层：包含策略抽象基类、Kronos AI 预测策略及多周期日内突破策略范例
"""

from .base import BaseStrategy
from .kronos_strategy import KronosStrategy, compute_dynamic_rolling_quantile
from .examples.intraday_breakout import IntradayBreakoutStrategy
