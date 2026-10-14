"""
backtest
Kronos 量化回测与实盘仿真系统
包含：
- core: 多市场微观制度与统一领域模型
- data: 行情获取、模型预测与本地缓存
- engine: 回测执行引擎与全维度评估归因
- execution: 仿真撮合、风控状态机与经纪商适配
- strategies: Kronos AI 策略与经典技术指标
- visualizer: 静态与交互式高保真可视化
- ui: Streamlit 交互式量化看板
"""

from .core.market_rules import MarketProfile, SymbolResolver
from .core.types import Order, Position, OrderType, OrderSide, TimeFrame
from .engine.backtest_engine import KronosBacktesterPro
from .execution.sizer import RiskBudgetSizer
from .execution.risk_manager import ExitStateMachine
from .execution.simulated_broker import SimulatedBroker

__all__ = [
    "MarketProfile",
    "SymbolResolver",
    "Order",
    "Position",
    "OrderType",
    "OrderSide",
    "TimeFrame",
    "KronosBacktesterPro",
    "RiskBudgetSizer",
    "ExitStateMachine",
    "SimulatedBroker",
]
