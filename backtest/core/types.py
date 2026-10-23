"""
backtest/core/types.py
量化回测与实盘统一底层领域模型 (Domain Types)
支持跨市场 (CN/HK/US)、多时间周期 (1d/30m/15m/5m/1m)、事件驱动与状态机。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd


class TimeFrame(str, Enum):
    """K线时间粒度定义与推导"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "60m"
    D1 = "1d"
    W1 = "1w"

    @property
    def minutes(self) -> int:
        mapping = {
            TimeFrame.M1: 1,
            TimeFrame.M5: 5,
            TimeFrame.M15: 15,
            TimeFrame.M30: 30,
            TimeFrame.H1: 60,
            TimeFrame.D1: 1440,
            TimeFrame.W1: 10080,
        }
        return mapping.get(self, 1440)

    @property
    def is_intraday(self) -> bool:
        return self in (TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.M30, TimeFrame.H1)

    @property
    def pandas_freq(self) -> str:
        """返回适配 Pandas Resample 的标准频率字符串"""
        mapping = {
            TimeFrame.M1: "1min",
            TimeFrame.M5: "5min",
            TimeFrame.M15: "15min",
            TimeFrame.M30: "30min",
            TimeFrame.H1: "60min",
            TimeFrame.D1: "1D",
            TimeFrame.W1: "1W",
        }
        return mapping.get(self, "1D")

    @property
    def label(self) -> str:
        """中文友好名称"""
        mapping = {
            TimeFrame.M1: "1分钟",
            TimeFrame.M5: "5分钟",
            TimeFrame.M15: "15分钟",
            TimeFrame.M30: "30分钟",
            TimeFrame.H1: "60分钟",
            TimeFrame.D1: "日线",
            TimeFrame.W1: "周线",
        }
        return mapping.get(self, "日线")

    def bars_per_day(self, market: str = 'CN') -> int:
        """
        根据交易所交易时段计算每日包含的 Bar 数量：
        - CN (A股): 4.0 小时 (240分钟)
        - HK (港股): 5.5 小时 (330分钟: 09:30-12:00, 13:00-16:00)
        - US (美股): 6.5 小时 (390分钟: 09:30-16:00 EST)
        """
        market = (market or 'CN').upper()
        if market == 'US':
            trading_mins = 390
        elif market == 'HK':
            trading_mins = 330
        else:
            trading_mins = 240

        if not self.is_intraday:
            return 1
        return max(1, trading_mins // self.minutes)

    def annual_factor(self, market: str = 'CN', annual_trading_days: int = None) -> float:
        """
        根据市场与时间粒度精确计算年化乘数 (sqrt(bars_per_year))
        用于夏普比率、年化波动率等指标的科学折算。
        """
        market = (market or 'CN').upper()
        if annual_trading_days is None:
            default_days = {'CN': 244, 'HK': 246, 'US': 252}
            annual_trading_days = default_days.get(market, 252)

        bars_per_year = annual_trading_days * self.bars_per_day(market)
        return math.sqrt(bars_per_year)

    @classmethod
    def from_str(cls, val: Union[str, 'TimeFrame']) -> 'TimeFrame':
        if isinstance(val, TimeFrame):
            return val
        if hasattr(val, 'value'):
            s = str(val.value).strip().lower()
        else:
            s = str(val).strip().lower()

        if s.startswith('timeframe.'):
            s = s.split('.', 1)[1]

        mapping = {
            '1m': cls.M1, '1min': cls.M1, 'm1': cls.M1, '1分钟': cls.M1,
            '5m': cls.M5, '5min': cls.M5, 'm5': cls.M5, '5分钟': cls.M5,
            '15m': cls.M15, '15min': cls.M15, 'm15': cls.M15, '15分钟': cls.M15,
            '30m': cls.M30, '30min': cls.M30, 'm30': cls.M30, '30分钟': cls.M30,
            '60m': cls.H1, '60min': cls.H1, '1h': cls.H1, '1hour': cls.H1, 'h1': cls.H1, '60分钟': cls.H1,
            'd': cls.D1, '1d': cls.D1, 'day': cls.D1, 'daily': cls.D1, '日线': cls.D1,
            'w': cls.W1, '1w': cls.W1, 'week': cls.W1, 'weekly': cls.W1, '周线': cls.W1,
        }
        return mapping.get(s, cls.D1)


class MarketType(str, Enum):
    CN = "CN"
    HK = "HK"
    US = "US"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Bar:
    """标准化单一行情 Bar (日线或分钟线)"""
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    amount: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """标准化交易决策信号"""
    symbol: str
    timestamp: pd.Timestamp
    action: int                     # 1: 买入, -1: 卖出, 0: 观望
    score: float = 0.0              # 信号强度 (如 Z-Score)
    position_scale: float = 1.0     # 仓位比例调节系数 (0.0 ~ 1.0)
    target_horizon: int = 5         # 预期持仓时界 (Bar 周期数)
    stop_loss_pct: float = 0.04     # 建议止损幅度
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    veto_reason: str = ""           # 否决或拦截原因 (若被风控拦截)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """订单对象（回测仿真撮合与实盘报单通用）"""
    order_id: str
    symbol: str
    side: OrderSide
    qty: int
    price: float
    order_type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.PENDING
    create_time: Optional[pd.Timestamp] = None
    filled_time: Optional[pd.Timestamp] = None
    filled_price: float = 0.0
    filled_qty: int = 0
    commission: float = 0.0
    reason: str = ""
    client_tag: str = ""


@dataclass
class Position:
    """持仓对象（跟踪成本、浮盈、峰谷值与双轨持仓周期：自然日与Bar步长）"""
    symbol: str
    shares: int = 0
    avg_entry_price: float = 0.0
    entry_date: Optional[pd.Timestamp] = None
    entry_commission: float = 0.0
    peak_price: float = 0.0
    trough_price: float = 0.0
    holding_bars: int = 0
    holding_days: int = 0
    entry_bar_idx: Optional[int] = None
    stop_price: Optional[float] = None
    target_horizon: Optional[int] = None
    active_stop_pct: float = 0.04
    active_trail_pct: float = 0.025
    trailing_active: bool = False

    @property
    def is_open(self) -> bool:
        return self.shares > 0

    def update_market_price(
        self,
        high: float,
        low: float,
        close: float,
        current_time: Optional[Any] = None,
    ) -> None:
        if not self.is_open:
            return
        self.holding_bars += 1
        self.peak_price = max(self.peak_price, high)
        self.trough_price = min(self.trough_price, low) if self.trough_price > 0 else low

        if current_time is not None and self.entry_date is not None:
            try:
                c_date = current_time.date() if hasattr(current_time, 'date') else pd.to_datetime(current_time).date()
                e_date = self.entry_date.date() if hasattr(self.entry_date, 'date') else pd.to_datetime(self.entry_date).date()
                self.holding_days = max(0, (c_date - e_date).days)
            except Exception:
                self.holding_days = self.holding_bars
        else:
            self.holding_days = self.holding_bars

    def can_exit(
        self,
        current_time: Any = None,
        settlement_days: int = 1,
        current_idx: Optional[int] = None,
    ) -> bool:
        """
        合规出场判断（T+N合规）：
        - 若 settlement_days == 0 (港股/美股 T+0)，开仓后即可平仓。
        - 若 settlement_days >= 1 (A股 T+1):
          1. 优先使用真实时间戳判断自然日换日跨度 (current_time.date() > entry_date.date())。
             在 1m/15m/30m/60m 等日内 K 线回测中，彻底防止同一自然日内提前卖出。
          2. 若无时间戳信息（纯数字索引 Mock 测试），回退为 (current_idx - entry_bar_idx) >= settlement_days。
        """
        if not self.is_open:
            return False
        if settlement_days <= 0:
            return True

        if current_time is not None and self.entry_date is not None:
            try:
                c_date = current_time.date() if hasattr(current_time, 'date') else pd.to_datetime(current_time).date()
                e_date = self.entry_date.date() if hasattr(self.entry_date, 'date') else pd.to_datetime(self.entry_date).date()
                return c_date > e_date
            except Exception:
                pass

        if current_idx is not None and self.entry_bar_idx is not None:
            return (current_idx - self.entry_bar_idx) >= settlement_days

        return True

    def unrealized_pnl(self, current_price: float) -> float:
        if not self.is_open or self.avg_entry_price <= 0:
            return 0.0
        return (current_price - self.avg_entry_price) * self.shares

    def unrealized_return(self, current_price: float) -> float:
        if not self.is_open or self.avg_entry_price <= 0:
            return 0.0
        return (current_price - self.avg_entry_price) / self.avg_entry_price

    def drawdown_from_peak(self, current_price: float) -> float:
        if not self.is_open or self.peak_price <= 0:
            return 0.0
        return (current_price - self.peak_price) / self.peak_price


@dataclass
class TradeRecord:
    """成交流水记录字典标准化"""
    date: pd.Timestamp
    action: str
    price: float
    shares: int
    commission: float
    pnl_pct: float
    pnl_amt: float
    holding_days: int
    reason: str
    cash_balance: float
    entry_date: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    symbol: str = ""
    target_horizon: Optional[int] = None
    stop_pct: Optional[float] = None
    stop_price: Optional[float] = None
    holding_bars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date,
            'action': self.action,
            'price': self.price,
            'shares': self.shares,
            'commission': self.commission,
            'pnl_pct': self.pnl_pct,
            'pnl_amt': self.pnl_amt,
            'holding_days': self.holding_days,
            'holding_bars': self.holding_bars if self.holding_bars > 0 else self.holding_days,
            'reason': self.reason,
            'cash_balance': self.cash_balance,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'mfe_pct': self.mfe_pct,
            'mae_pct': self.mae_pct,
            'symbol': self.symbol,
            'target_horizon': self.target_horizon,
            'stop_pct': self.stop_pct,
            'stop_price': self.stop_price,
        }
