"""
backtest/core
底层核心包：领域模型、市场规则画像、多时间周期日历与统一日志系统
"""

from .types import (
    TimeFrame,
    MarketType,
    OrderSide,
    OrderType,
    OrderStatus,
    Bar,
    Signal,
    Order,
    Position,
    TradeRecord,
)

from .market_rules import (
    SymbolResolver,
    MarketProfile,
    HK_COMMON_LOT_SIZES,
    HK_STAMP_DUTY_CUTOVER,
    compute_open_limit_flags,
    detect_market_type,
    validate_stock_code,
)

from .timeframes import (
    get_annualization_factor,
    detect_timeframe,
    resample_bars,
    MARKET_TRADING_SESSIONS,
    is_same_trading_day,
    get_trading_days_passed,
    TIMEFRAME_BEST_PRACTICES,
    get_timeframe_defaults,
    resolve_timeframe_params,
)

from .resampler import (
    SessionResampler,
)

from .mtf_aligner import (
    MultiTimeframeAligner,
)

from .logger import (
    setup_logger,
    attach_stock_logger,
    detach_stock_logger,
    detach_all_stock_loggers,
    set_logging_state,
    clear_logs,
)
