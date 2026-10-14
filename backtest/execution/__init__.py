"""
backtest/execution
撮合与执行层：统一经纪商接口、仿真撮合、实盘网关适配、仓位管理与出场状态机
"""

from .broker_base import BaseBroker
from .simulated_broker import SimulatedBroker
from .sizer import RiskBudgetSizer
from .risk_manager import ExitStateMachine
from .live_broker_adapter import LiveBrokerAdapter, FutuBrokerAdapter, InteractiveBrokersAdapter
