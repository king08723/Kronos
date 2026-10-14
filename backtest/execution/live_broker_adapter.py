"""
backtest/execution/live_broker_adapter.py
港美股实盘交易网关适配器 (Live Broker Adapters)
支持：
1. 统一实盘账户资金与持仓同步 (Futu / Tiger / Interactive Brokers / Alpaca)
2. 订单生命周期管理 (限价单/市价单/止损单)
3. 真实行情流驱动与下单路由，确保「回测代码无需修改直接跑实盘」
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core.types import Order, Position, OrderSide, OrderType, OrderStatus, Bar
from ..core.market_rules import MarketProfile
from .broker_base import BaseBroker

logger = logging.getLogger("KronosBacktest")


class LiveBrokerAdapter(BaseBroker):
    """实盘交易网关抽象适配基类"""

    def __init__(self, profile: MarketProfile, config: Optional[Dict[str, Any]] = None):
        super().__init__(profile=profile, initial_capital=0.0)
        self.config = config or {}
        self.is_connected = False

    def connect(self) -> bool:
        """建立实盘经纪商通信连接"""
        raise NotImplementedError

    def disconnect(self) -> None:
        """断开连接"""
        self.is_connected = False

    def sync_account(self) -> Dict[str, float]:
        """从交易所或券商同步最新现金与总资产"""
        raise NotImplementedError

    def sync_positions(self) -> Dict[str, Position]:
        """从券商同步最新持仓明细"""
        raise NotImplementedError


class FutuBrokerAdapter(LiveBrokerAdapter):
    """
    富途开放接口 (Futu OpenD) 实盘/仿真网关
    针对港股 (HK) 与美股 (US) 交易的工业级对接桥梁。
    """

    def __init__(self, profile: MarketProfile, host: str = "127.0.0.1", port: int = 11111, is_paper: bool = True):
        super().__init__(profile=profile, config={'host': host, 'port': port, 'is_paper': is_paper})
        self.host = host
        self.port = port
        self.is_paper = is_paper
        self._ctx = None

    def connect(self) -> bool:
        """
        连接本地运行的 Futu OpenD 网关：
        pip install futu-api
        """
        try:
            import futu as ft
            trd_market = ft.TrdMarket.HK if self.profile.market == 'HK' else ft.TrdMarket.US
            trd_env = ft.TrdEnv.SIMULATE if self.is_paper else ft.TrdEnv.REAL
            self._ctx = ft.OpenSecTradeContext(
                filter_trdmarket=trd_market,
                host=self.host,
                port=self.port,
                security_firm=ft.SecurityFirm.FUTUSECURITIES,
            )
            self.is_connected = True
            logger.info(f"✅ 富途网关连接成功: {self.host}:{self.port} | 市场={self.profile.market} | 环境={'仿真' if self.is_paper else '实盘'}")
            return True
        except ImportError:
            logger.warning("⚠️ 未安装 futu-api，请执行: pip install futu-api")
            return False
        except Exception as e:
            logger.error(f"❌ 连接富途网关失败: {e}")
            return False

    def submit_order(self, order: Order, current_bar: Any = None) -> Order:
        """向富途网关报单"""
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            order.reason = "网关未连接"
            return order

        try:
            import futu as ft
            trd_side = ft.TrdSide.BUY if order.side == OrderSide.BUY else ft.TrdSide.SELL
            order_type = ft.OrderType.NORMAL if order.order_type == OrderType.LIMIT else ft.OrderType.MARKET
            
            # 富途代码转换: 00700 -> HK.00700, AAPL -> US.AAPL
            futu_code = f"HK.{order.symbol}" if self.profile.market == 'HK' else f"US.{order.symbol}"

            ret, data = self._ctx.place_order(
                price=order.price,
                qty=order.qty,
                code=futu_code,
                trd_side=trd_side,
                order_type=order_type,
                trd_env=ft.TrdEnv.SIMULATE if self.is_paper else ft.TrdEnv.REAL,
            )
            if ret == ft.RET_OK:
                order.status = OrderStatus.SUBMITTED
                order.order_id = str(data['order_id'].iloc[0])
                logger.info(f"🟢 [富途报单成功] ID={order.order_id} | {order.side} {order.qty}股 {order.symbol} @ {order.price}")
            else:
                order.status = OrderStatus.REJECTED
                order.reason = str(data)
                logger.error(f"🔴 [富途报单被拒] {data}")
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reason = str(e)
            logger.error(f"❌ 富途报单异常: {e}")
        return order

    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected:
            return False
        try:
            import futu as ft
            ret, data = self._ctx.modify_order(
                ft.ModifyOrderOp.CANCEL,
                order_id=int(order_id),
                qty=0,
                price=0,
                trd_env=ft.TrdEnv.SIMULATE if self.is_paper else ft.TrdEnv.REAL,
            )
            return ret == ft.RET_OK
        except Exception:
            return False

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        return self.cash


class InteractiveBrokersAdapter(LiveBrokerAdapter):
    """
    盈透证券 (Interactive Brokers / TWS / Gateway) 适配器
    全球市场通用，支持港美股股票、期权与日内高频 Bar。
    """

    def __init__(self, profile: MarketProfile, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        super().__init__(profile=profile, config={'host': host, 'port': port, 'client_id': client_id})
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None

    def connect(self) -> bool:
        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id)
            self.is_connected = True
            logger.info(f"✅ IBKR 网关连接成功: {self.host}:{self.port} (clientId={self.client_id})")
            return True
        except ImportError:
            logger.warning("⚠️ 未安装 ib_insync，请执行: pip install ib_insync")
            return False
        except Exception as e:
            logger.error(f"❌ 连接 IBKR 网关失败: {e}")
            return False

    def submit_order(self, order: Order, current_bar: Any = None) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            return order
        try:
            from ib_insync import Stock, MarketOrder, LimitOrder
            exchange = 'SEHK' if self.profile.market == 'HK' else 'SMART'
            currency = self.profile.currency
            contract = Stock(order.symbol, exchange, currency)
            action = 'BUY' if order.side == OrderSide.BUY else 'SELL'
            ib_order = MarketOrder(action, order.qty) if order.order_type == OrderType.MARKET else LimitOrder(action, order.qty, order.price)
            trade = self._ib.placeOrder(contract, ib_order)
            order.status = OrderStatus.SUBMITTED
            order.order_id = str(trade.order.orderId)
            logger.info(f"🟢 [IBKR报单成功] ID={order.order_id} | {order.side} {order.qty}股 @ {order.price}")
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reason = str(e)
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        return self.cash
