"""
backtest/execution/broker_base.py
统一交易经纪商/执行接口 (Broker Interface)
统一回测仿真撮合 (SimulatedBroker) 与实盘接口 (Futu / Tiger / IBKR / Alpaca)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

from ..core.types import Order, Position, OrderSide, OrderType, OrderStatus
from ..core.market_rules import MarketProfile


class BaseBroker(ABC):
    """统一经纪商抽象接口"""

    def __init__(self, profile: MarketProfile, initial_capital: float = 100000.0):
        self.profile = profile
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []

    @abstractmethod
    def submit_order(self, order: Order, current_bar: Any = None) -> Order:
        """提交订单"""
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        raise NotImplementedError

    @abstractmethod
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """获取账户当前总净值 (现金 + 持仓总市值)"""
        raise NotImplementedError

    def get_position(self, symbol: str) -> Position:
        """获取指定标的持仓，不存在则返回空持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
