"""
backtest/execution/risk_manager.py
离散状态机出场与多维风控防守中枢 (Exit State Machine)
包含：
1. 盘中止损与跳空保护 (Intraday Stop with Limit-Down Check)
2. 动态 ATR 硬止损 (Hard Stop Loss)
3. 激活门槛峰值回撤移动止盈 (Trailing Stop)
4. 预测波峰止盈与「动量跃迁」护航 (Peak Horizon Take-Profit & Momentum Transition)
5. 信号证伪时间止损与最大持仓时界强制离场
6. 模型波段走弱主动规避
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from ..core.market_rules import MarketProfile
from ..core.types import Position

logger = logging.getLogger("KronosBacktest")


class ExitStateMachine:
    """多维动态出场状态机"""

    def __init__(
        self,
        hard_stop_loss: float = 0.04,
        trailing_stop: float = 0.025,
        trailing_stop_min_profit: float = 0.02,
        max_holding_days: int = 5,
        intraday_stop: bool = True,
    ):
        self.hard_stop_loss = hard_stop_loss
        self.trailing_stop = trailing_stop
        self.trailing_stop_min_profit = trailing_stop_min_profit
        self.max_holding_days = max_holding_days
        self.intraday_stop = intraday_stop

    def check_intraday_exit(
        self,
        row: Any,
        position: Position,
        profile: MarketProfile,
        idx: int,
        limit_down: int,
        stock_code_tag: str = "",
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        盘中触及止损价检查（仅在满足 T+N 合规且市场未跌停时触发）。
        返回: (should_exit_intraday, reason, fill_price)
        """
        can_exit_today = (
            position.entry_bar_idx is None
            or (idx - position.entry_bar_idx) >= profile.settlement_days
        )
        if not (self.intraday_stop and can_exit_today):
            return False, None, None

        stop_price = position.stop_price
        if stop_price is None or position.shares <= 0:
            return False, None, None

        close_price = row.close
        low_price = getattr(row, 'low', close_price)
        if pd.isna(low_price):
            low_price = close_price
        open_price = getattr(row, 'open', close_price)
        if pd.isna(open_price):
            open_price = close_price
        close_prev_val = getattr(row, 'close_prev', np.nan)

        if low_price <= stop_price and not limit_down:
            # 检查收盘是否跌停封死
            is_close_limit_down = False
            if profile.price_limit_pct is not None and not pd.isna(close_prev_val):
                down_limit_px = round(close_prev_val * (1.0 - profile.price_limit_pct), 2)
                is_close_limit_down = (close_price <= down_limit_px + 0.01)

            if is_close_limit_down:
                date_str = row.Index.date() if hasattr(row.Index, 'date') else str(row.Index)
                logger.info(
                    f"[{date_str}] {stock_code_tag}⛔ 盘中触及止损价({stop_price:.3f})但当日收盘封死跌停，"
                    f"无法成交，平仓指令顺延次日"
                )
                return False, f"跌停顺延-盘中止损受阻(触及{stop_price:.3f})", stop_price
            else:
                fill = stop_price if open_price > stop_price else open_price
                return True, f'盘中止损(触及{stop_price:.3f})', fill

        return False, None, None

    def evaluate_eod_exit(
        self,
        row: Any,
        position: Position,
        profile: MarketProfile,
        idx: int,
        stock_code_tag: str = "",
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        收盘盯市评估（决定次日开盘是否执行平仓指令）。
        包含硬止损、移动止盈、波峰止盈、动量跃迁、时间止损与波段走弱。
        返回: (pending_close_reason, pending_close_price)
        """
        can_exit_today = (
            position.entry_bar_idx is None
            or (idx - position.entry_bar_idx) >= profile.settlement_days
        )
        close_price = row.close
        avg_entry_price = position.avg_entry_price
        peak_price = position.peak_price
        holding_days = position.holding_bars
        active_target_horizon = position.target_horizon
        active_stop_pct = position.active_stop_pct
        active_trail_pct = position.active_trail_pct

        drawdown_from_cost = (close_price - avg_entry_price) / avg_entry_price
        drawdown_from_peak = (close_price - peak_price) / peak_price
        ma_fast_val = getattr(row, 'ma_fast', None)
        exec_bearish = int(getattr(row, 'exec_bearish', 0))

        pending_close_reason = None
        pending_close_price = None

        if drawdown_from_cost <= -active_stop_pct:
            pending_close_reason = f"硬止损({drawdown_from_cost:.2%})"
        elif (drawdown_from_peak <= -active_trail_pct
              and drawdown_from_cost >= self.trailing_stop_min_profit):
            pending_close_reason = f"移动止盈(峰值回撤{drawdown_from_peak:.2%})"
        elif (active_target_horizon is not None and holding_days >= active_target_horizon):
            if drawdown_from_cost <= 0:
                # 浮亏：模型预测波峰已过，假说彻底证伪，时间止损
                pending_close_reason = (
                    f"时间止损-模型信号证伪({holding_days}日/{active_target_horizon}日目标,"
                    f"浮亏{drawdown_from_cost:.2%})"
                )
            else:
                # 浮盈检验：判断主升浪延续还是波峰滞涨
                is_near_high = (drawdown_from_peak >= -0.015)
                is_trend_intact = (ma_fast_val is not None and not pd.isna(ma_fast_val) and close_price > ma_fast_val)
                has_profit_cushion = (drawdown_from_cost >= max(0.02, active_stop_pct * 1.2))

                if is_near_high and is_trend_intact and has_profit_cushion:
                    # 🚀【动量跃迁】：动量强劲，取消强平，移交收紧后的移动止盈护航
                    position.active_trail_pct = max(0.015, active_trail_pct * 0.7)
                    position.target_horizon = self.max_holding_days
                    date_str = row.Index.date() if hasattr(row.Index, 'date') else str(row.Index)
                    logger.info(
                        f"[{date_str}] {stock_code_tag}🚀 触达目标时界({holding_days}日)但动量强劲: "
                        f"浮盈={drawdown_from_cost*100:+.2f}%, 峰值回撤={drawdown_from_peak*100:+.2f}% | "
                        f"取消强平，状态跃迁至移动止盈护航(收紧回撤至{position.active_trail_pct:.2%})"
                    )
                else:
                    pending_close_reason = (
                        f"预测波峰到达止盈({holding_days}日/{active_target_horizon}日目标,"
                        f"浮盈{drawdown_from_cost:.2%})"
                    )
        elif holding_days >= self.max_holding_days:
            pending_close_reason = f"持仓超时({holding_days}日)"
        elif exec_bearish == 1:
            pending_close_reason = "模型波段走弱主动离场"

        # T+N 未满足时撤回平仓决定，下一根 bar 再判
        if pending_close_reason and not can_exit_today:
            date_str = row.Index.date() if hasattr(row.Index, 'date') else str(row.Index)
            logger.debug(f"[{date_str}] {stock_code_tag}⏳ T+{profile.settlement_days} 限制，"
                         f"平仓指令【{pending_close_reason}】顺延")
            pending_close_reason = None
            pending_close_price = None

        if pending_close_reason:
            date_str = row.Index.date() if hasattr(row.Index, 'date') else str(row.Index)
            logger.info(
                f"[{date_str}] {stock_code_tag}⚠️ 触发平仓预警: 【{pending_close_reason}】 | "
                f"收盘价={close_price:.3f}, 成本={avg_entry_price:.3f}, "
                f"浮盈={drawdown_from_cost*100:+.2f}%, 峰值回撤={drawdown_from_peak*100:+.2f}%"
            )
        else:
            date_str = row.Index.date() if hasattr(row.Index, 'date') else str(row.Index)
            logger.debug(
                f"[{date_str}] {stock_code_tag}🔵 持仓第{holding_days}天: 现价={close_price:.3f}, "
                f"浮盈={drawdown_from_cost*100:+.2f}%, 峰值回撤={drawdown_from_peak*100:+.2f}%"
            )

        return pending_close_reason, pending_close_price
