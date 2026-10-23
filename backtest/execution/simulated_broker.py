"""
backtest/execution/simulated_broker.py
全真仿真撮合经纪商 (SimulatedBroker)
完整实现微观交易制度：
- 报价滑点 (slippage_rate)
- 交易所规费、印花税、双边佣金与最低消费 (MarketProfile)
- 涨跌停封死判定与平仓指令顺延
- T+N 合规合规流转 (A股 T+1，港美股 T+0 当日平仓)
- 逐笔流水账 (trades) 与逐日盯市净值 (MTM)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..core.types import Order, Position, OrderSide, OrderStatus
from ..core.market_rules import MarketProfile
from .broker_base import BaseBroker

logger = logging.getLogger("KronosBacktest")


class SimulatedBroker(BaseBroker):
    """仿真撮合交易经纪商 (支持成交量占比约束与冲击滑点模型)"""

    def __init__(
        self,
        profile: MarketProfile,
        initial_capital: float = 100000.0,
        slippage_rate: float = 0.001,
        volume_participation_cap: Optional[float] = None,
        slippage_model: str = "fixed",
        slippage_impact_coef: float = 0.1,
        max_slippage: float = 0.02,
        min_lot_guarantee: bool = True,
    ):
        super().__init__(profile=profile, initial_capital=initial_capital)
        self.slippage_rate = slippage_rate
        self.volume_participation_cap = volume_participation_cap
        self.slippage_model = slippage_model.lower() if slippage_model else "fixed"
        self.slippage_impact_coef = slippage_impact_coef
        self.max_slippage = max_slippage
        self.min_lot_guarantee = min_lot_guarantee
        self.trades: List[Dict[str, Any]] = []

    def compute_slippage(
        self,
        base_price: float,
        shares: int,
        bar_volume: Optional[float] = None,
        bar_high: Optional[float] = None,
        bar_low: Optional[float] = None,
    ) -> float:
        """
        根据滑点模型计算实际执行滑点费率。
        - "fixed": 固定滑点 (self.slippage_rate)
        - "dynamic": 基于单 Bar 成交量占比与波动率的非线性流动性冲击模型 (Square-root Law)
        """
        if self.slippage_model != "dynamic":
            return float(self.slippage_rate)

        if bar_volume is None or pd.isna(bar_volume) or bar_volume <= 0 or shares <= 0:
            return float(self.slippage_rate)

        # 1. 参与率 (Participation Ratio)
        participation = min(1.0, float(shares) / float(bar_volume))

        # 2. 价格振幅相对值 (High-Low Spread)
        if (
            bar_high is not None
            and bar_low is not None
            and not pd.isna(bar_high)
            and not pd.isna(bar_low)
            and bar_high >= bar_low
            and base_price > 0
        ):
            spread = (bar_high - bar_low) / base_price
        else:
            spread = 0.005  # 默认 0.5% 相对振幅兜底

        # 3. 平方根冲击模型: impact = impact_coef * sqrt(participation) * spread
        impact = self.slippage_impact_coef * (participation ** 0.5) * spread
        eff_slippage = float(np.clip(self.slippage_rate + impact, self.slippage_rate, self.max_slippage))
        return eff_slippage

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """按现价逐日盯市计算总净值"""
        equity = self.cash
        for sym, pos in self.positions.items():
            if pos.is_open:
                px = current_prices.get(sym, pos.avg_entry_price)
                equity += pos.shares * px
        return equity

    def submit_order(self, order: Order, current_bar: Any = None) -> Order:
        """撮合订单，计算税费与滑点"""
        # 回测引擎使用定制化撮合，此处保留标准接口以实现 BaseBroker 规范
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

    def execute_sell(
        self,
        trade_date: pd.Timestamp,
        symbol: str,
        position: Position,
        fill_price: float,
        reason: str,
        holding_days: int = 0,
        holding_bars: Optional[int] = None,
        stock_code_tag: str = "",
        bar_volume: Optional[float] = None,
        bar_high: Optional[float] = None,
        bar_low: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """
        执行卖出平仓撮合，扣除税费与滑点并回笼资金。
        返回: (cash, fee, trade_pnl_pct, pnl_amount)
        """
        cs = self.profile.currency_symbol
        eff_slippage = self.compute_slippage(
            base_price=fill_price,
            shares=position.shares,
            bar_volume=bar_volume,
            bar_high=bar_high,
            bar_low=bar_low,
        )
        sell_price = fill_price * (1.0 - eff_slippage)
        gross = position.shares * sell_price
        fee = self.profile.sell_cost(gross, trade_date=trade_date)
        avg_entry = position.avg_entry_price

        trade_pnl = (sell_price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
        mfe = (position.peak_price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
        mae = (position.trough_price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
        pnl_amount = (sell_price - avg_entry) * position.shares - fee - position.entry_commission

        self.cash = self.cash + gross - fee

        date_str = trade_date.date() if hasattr(trade_date, 'date') else str(trade_date)
        bars_held = holding_bars if holding_bars is not None else position.holding_bars
        duration_desc = f"{holding_days}天" if bars_held == holding_days else f"{holding_days}天({bars_held}Bar)"
        slip_desc = f" (滑点={eff_slippage:.2%})" if eff_slippage != self.slippage_rate else ""
        logger.info(
            f"[{date_str}] {stock_code_tag}🔴 执行平仓: {position.shares}股 @ {sell_price:.3f}{slip_desc} | "
            f"持仓: {duration_desc} | 原因: 【{reason}】 | "
            f"单笔盈亏: {trade_pnl*100:+.2f}% ({cs}{pnl_amount:+,.2f}) | "
            f"MFE={mfe*100:+.2f}%, MAE={mae*100:+.2f}% | "
            f"税费: {fee:.2f} | 账户资金: {self.cash:,.2f}"
        )

        trade_record = {
            'date': trade_date,
            'action': 'SELL',
            'price': round(sell_price, 4),
            'shares': position.shares,
            'commission': round(fee, 2),
            'slippage_rate': round(eff_slippage, 6),
            'pnl_pct': round(trade_pnl, 6),
            'pnl_amt': round(pnl_amount, 2),
            'holding_days': holding_days,
            'holding_bars': bars_held if bars_held > 0 else holding_days,
            'reason': reason,
            'cash_balance': round(self.cash, 2),
            'entry_date': position.entry_date,
            'entry_price': round(position.avg_entry_price, 4),
            'mfe_pct': round(mfe, 6),
            'mae_pct': round(mae, 6),
            'symbol': symbol or self.profile.market,
        }
        self.trades.append(trade_record)

        # 重置持仓状态
        position.shares = 0
        position.avg_entry_price = 0.0
        position.peak_price = 0.0
        position.trough_price = 0.0
        position.holding_bars = 0
        position.holding_days = 0
        position.entry_date = None
        position.entry_commission = 0.0
        position.target_horizon = None
        position.entry_bar_idx = None
        position.stop_price = None

        return self.cash, fee, trade_pnl, pnl_amount

    def execute_buy(
        self,
        trade_date: pd.Timestamp,
        symbol: str,
        open_price: float,
        shares_to_buy: int,
        target_horizon: int,
        active_stop_pct: float,
        stop_price: float,
        bar_idx: int,
        close_price: float,
        low_price: float,
        active_trail_pct: float = 0.025,
        stock_code_tag: str = "",
        bar_volume: Optional[float] = None,
        bar_high: Optional[float] = None,
    ) -> Optional[Position]:
        """
        执行买入开仓撮合，支持单 Bar 成交量占比上限约束与动态冲击滑点。
        """
        # 1. 单 Bar 成交量占比上限约束 (Volume Participation Cap)
        if self.volume_participation_cap is not None and self.volume_participation_cap > 0:
            if bar_volume is not None and not pd.isna(bar_volume) and bar_volume > 0:
                max_executable_raw = bar_volume * self.volume_participation_cap
                max_executable = self.profile.round_shares(max_executable_raw)
                if shares_to_buy > max_executable:
                    date_str = trade_date.date() if hasattr(trade_date, 'date') else str(trade_date)
                    if max_executable >= self.profile.lot_size:
                        logger.info(
                            f"[{date_str}] {stock_code_tag}⚠️ 触及单Bar流动性上限: 拟买入 {shares_to_buy} 股 > "
                            f"Bar容量上限({bar_volume:,.0f} * {self.volume_participation_cap:.1%})，受限截断为 {max_executable} 股"
                        )
                        shares_to_buy = max_executable
                    else:
                        if self.min_lot_guarantee:
                            shares_to_buy = self.profile.lot_size
                            logger.info(
                                f"[{date_str}] {stock_code_tag}🛡️ 流动性上限不足1手但触发一手保底: 买入 1 手 ({shares_to_buy}股)"
                            )
                        else:
                            logger.warning(
                                f"[{date_str}] {stock_code_tag}❌ 流动性枯竭: Bar成交量({bar_volume:,.0f})容量不足1手({self.profile.lot_size}股)，开仓放弃"
                            )
                            return None

        # 2. 计算实际执行滑点
        eff_slippage = self.compute_slippage(
            base_price=open_price,
            shares=shares_to_buy,
            bar_volume=bar_volume,
            bar_high=bar_high,
            bar_low=low_price,
        )
        buy_price = open_price * (1.0 + eff_slippage)
        gross_cost = shares_to_buy * buy_price
        fee = self.profile.buy_cost(gross_cost, trade_date=trade_date)
        total_outflow = gross_cost + fee

        if self.cash < total_outflow:
            return None

        self.cash -= total_outflow
        date_str = trade_date.date() if hasattr(trade_date, 'date') else str(trade_date)
        slip_desc = f" (滑点={eff_slippage:.2%})" if eff_slippage != self.slippage_rate else ""

        logger.info(
            f"[{date_str}] {stock_code_tag}🟢 开仓买入: {shares_to_buy}股 @ {buy_price:.3f}{slip_desc} | "
            f"目标时界: {target_horizon}日 | 止损={active_stop_pct:.2%}"
            f"(价 {stop_price:.3f}) | 耗资: {total_outflow:,.2f} (费用: {fee:.2f}) | "
            f"剩余现金: {self.cash:,.2f}"
        )

        pos = Position(
            symbol=symbol,
            shares=shares_to_buy,
            avg_entry_price=buy_price,
            entry_date=trade_date,
            entry_commission=fee,
            peak_price=close_price,
            trough_price=low_price if low_price > 0 else close_price,
            holding_bars=0,
            entry_bar_idx=bar_idx,
            stop_price=stop_price,
            target_horizon=target_horizon,
            active_stop_pct=active_stop_pct,
            active_trail_pct=active_trail_pct,
        )
        self.positions[symbol] = pos

        trade_record = {
            'date': trade_date,
            'action': 'BUY',
            'price': round(buy_price, 4),
            'shares': shares_to_buy,
            'commission': round(fee, 2),
            'slippage_rate': round(eff_slippage, 6),
            'pnl_pct': 0.0,
            'pnl_amt': 0.0,
            'holding_days': 0,
            'holding_bars': 0,
            'reason': f"信号开仓(目标H={target_horizon}日)",
            'cash_balance': round(self.cash, 2),
            'target_horizon': target_horizon,
            'stop_pct': round(active_stop_pct, 4),
            'stop_price': round(stop_price, 4),
            'entry_date': trade_date,
            'entry_price': round(buy_price, 4),
            'mfe_pct': 0.0,
            'mae_pct': 0.0,
            'symbol': symbol or self.profile.market,
        }
        self.trades.append(trade_record)
        return pos
