"""
backtest/execution/sizer.py
机构级风险预算定仓 (Risk Budgeting Sizing)
包含：
1. 动态 ATR 止损反推下单股数
2. 现金与单标的仓位上限硬约束
3. 连续亏损自适应降仓防守
4. 港美股高单价标的一手保底机制 (Min-Lot Guarantee)
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional
import numpy as np

from ..core.market_rules import MarketProfile

logger = logging.getLogger("KronosBacktest")


class RiskBudgetSizer:
    """机构级风险预算定仓器"""

    def __init__(
        self,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.80,
        k_stop: float = 1.5,
        k_trail: float = 1.2,
        min_stop: float = 0.02,
        max_stop: float = 0.12,
        consecutive_loss_cut: int = 2,
        min_lot_guarantee: bool = True,
    ):
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.k_stop = k_stop
        self.k_trail = k_trail
        self.min_stop = min_stop
        self.max_stop = max_stop
        self.consecutive_loss_cut = consecutive_loss_cut
        self.min_lot_guarantee = min_lot_guarantee

    def compute_size(
        self,
        portfolio_value: float,
        cash: float,
        buy_price: float,
        profile: MarketProfile,
        atr_pct: Optional[float] = None,
        position_scale: float = 1.0,
        consecutive_losses: int = 0,
        hard_stop_loss: float = 0.04,
        trailing_stop: float = 0.025,
        trade_date=None,
        stock_code: str = "",
    ) -> Tuple[int, float, float, float]:
        """
        根据风险预算与微观规则计算可买入股数与自适应止损线。
        
        返回:
            (shares_to_buy: int, active_stop_pct: float, active_trail_pct: float, stop_price: float)
        """
        code_tag = f"[{stock_code}] " if stock_code else ""

        # 1. 动态自适应止损止盈幅度
        if atr_pct is None or np.isnan(atr_pct) or atr_pct <= 0:
            active_stop_pct = hard_stop_loss
            active_trail_pct = trailing_stop
        else:
            active_stop_pct = float(np.clip(self.k_stop * atr_pct, self.min_stop, self.max_stop))
            active_trail_pct = float(np.clip(self.k_trail * atr_pct, self.min_stop, self.max_stop))

        # 2. 连续亏损降仓防守
        scale = float(position_scale or 1.0)
        if self.consecutive_loss_cut > 0 and consecutive_losses >= self.consecutive_loss_cut:
            scale *= 0.5

        # 3. 风险预算定仓推导
        risk_budget = portfolio_value * self.risk_per_trade * scale
        stop_distance = active_stop_pct * buy_price
        raw_shares = (risk_budget / stop_distance) if stop_distance > 0 else 0

        # 4. 仓位上限与现金可用硬约束
        cap_by_pct = (portfolio_value * self.max_position_pct) / buy_price if buy_price > 0 else 0
        cap_by_cash = (cash * 0.98) / buy_price if buy_price > 0 else 0
        raw_shares = min(raw_shares, cap_by_pct, cap_by_cash)

        # 5. 交易所微观规则整手截断
        shares_to_buy = profile.round_shares(raw_shares)

        # 6. 一手保底机制 (针对高单价港美股自适应激活)
        if shares_to_buy < profile.lot_size and self.min_lot_guarantee:
            one_lot_cost = profile.lot_size * buy_price
            one_lot_total = one_lot_cost + profile.buy_cost(one_lot_cost, trade_date=trade_date)
            if cash >= one_lot_total and (one_lot_cost <= portfolio_value * self.max_position_pct):
                shares_to_buy = profile.lot_size
                logger.info(
                    f"[{trade_date.date() if hasattr(trade_date, 'date') else trade_date}] {code_tag}"
                    f"🛡️ 触发一手保底: 计算股数 {raw_shares:.1f} 股不足一手({profile.lot_size}股)，"
                    f"账户现金充足，保底买入 1 手 ({shares_to_buy}股)"
                )

        stop_price = buy_price * (1.0 - active_stop_pct)
        return shares_to_buy, active_stop_pct, active_trail_pct, stop_price
