"""
backtest/strategies/base.py
量化策略抽象基类 (BaseStrategy)
支持：
1. 矩阵式/向量化批量信号生成 (generate_signals) -> 满足高吞吐历史回测
2. 事件驱动逐 Bar 回调 (on_bar) -> 无缝对接 15m/30m 日内实盘行情流与仿真交易
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd

from ..core.types import Bar, Signal, Order, Position, TimeFrame
from ..core.market_rules import MarketProfile


class BaseStrategy(ABC):
    """量化策略抽象基类"""

    def __init__(self, name: str = "BaseStrategy", params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params: Dict[str, Any] = params or {}
        self.profile: Optional[MarketProfile] = None
        self.timeframe: TimeFrame = TimeFrame.D1

    def set_market(self, profile: MarketProfile) -> None:
        """注入当前标的交易所规则画像"""
        self.profile = profile

    def set_timeframe(self, timeframe: TimeFrame) -> None:
        """设置当前策略运行的时间粒度 (1d / 30m / 15m 等)"""
        self.timeframe = timeframe

    def initialize(self, context: Optional[Dict[str, Any]] = None) -> None:
        """策略初始化钩子，可用于预热指标、连接数据源等"""
        pass

    @abstractmethod
    def generate_signals(
        self,
        hist_df: pd.DataFrame,
        pred_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        批量/矩阵式信号生成（回测模式）
        
        入参:
            hist_df: 历史行情数据 (包含 open, high, low, close, volume 等)
            pred_df: 可选的模型预测数据 (若为 AI 驱动策略)
        
        返回:
            pd.DataFrame: 包含执行信号列的矩阵，至少包含：
                - exec_signal: 1 (买入), 0 (观望), -1 (转空离场)
                - exec_position_scale: 仓位系数 (0.0 ~ 1.0)
                - exec_target_horizon: 预期持仓周期 (整数，单位为当前 Bar 数量)
                - veto_reason: 若被风控否决，记录拦截原因
        """
        raise NotImplementedError

    def on_bar(
        self,
        bar: Bar,
        current_position: Optional[Position] = None,
        account_cash: float = 0.0,
    ) -> Optional[Signal]:
        """
        事件驱动逐 Bar 处理（日内实盘/仿真交易模式）
        当新的 15m/30m/1d K 线生成时触发。
        """
        return None
