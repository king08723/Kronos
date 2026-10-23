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
from datetime import datetime, date
from typing import Optional, Tuple, Any
import pandas as pd
import numpy as np

from ..core.market_rules import MarketProfile
from ..core.types import Position

logger = logging.getLogger("KronosBacktest")


def _extract_time(row: Any) -> Any:
    """从命名元组、Pandas Series 或字典/对象中鲁棒提取时间戳"""
    if hasattr(row, 'Index') and row.Index is not None:
        return row.Index
    if hasattr(row, 'name') and row.name is not None and isinstance(row.name, (pd.Timestamp, datetime)):
        return row.name
    if hasattr(row, 'date') and row.date is not None:
        return row.date
    if hasattr(row, 'name') and row.name is not None:
        return row.name
    return None


class ExitStateMachine:
    """多维动态出场状态机"""

    def __init__(
        self,
        hard_stop_loss: float = 0.04,
        trailing_stop: float = 0.025,
        trailing_stop_min_profit: float = 0.02,
        max_holding_days: int = 5,
        intraday_stop: bool = True,
        k_near_high_atr: float = 0.8,
        k_trail_atr: float = 1.0,
        min_near_high_pct: float = 0.015,
        max_near_high_pct: float = 0.035,
        min_trail_pct: float = 0.018,
        max_trail_pct: float = 0.042,
        intraday_exit_at_eod: bool = False,
        overnight_gap_protection: bool = True,
    ):
        self.hard_stop_loss = hard_stop_loss
        self.trailing_stop = trailing_stop
        self.trailing_stop_min_profit = trailing_stop_min_profit
        self.max_holding_days = max_holding_days
        self.intraday_stop = intraday_stop
        self.k_near_high_atr = k_near_high_atr
        self.k_trail_atr = k_trail_atr
        self.min_near_high_pct = min_near_high_pct
        self.max_near_high_pct = max_near_high_pct
        self.min_trail_pct = min_trail_pct
        self.max_trail_pct = max_trail_pct
        self.intraday_exit_at_eod = intraday_exit_at_eod
        self.overnight_gap_protection = overnight_gap_protection

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
        current_time = _extract_time(row)
        can_exit_today = position.can_exit(
            current_time=current_time,
            settlement_days=profile.settlement_days,
            current_idx=idx,
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

            date_str = current_time.date() if hasattr(current_time, 'date') else str(current_time)
            if is_close_limit_down:
                logger.info(
                    f"[{date_str}] {stock_code_tag}⛔ 盘中触及止损价({stop_price:.3f})但当日收盘封死跌停，"
                    f"无法成交，平仓指令顺延次日"
                )
                return False, f"跌停顺延-盘中止损受阻(触及{stop_price:.3f})", stop_price
            else:
                if self.overnight_gap_protection and open_price < stop_price:
                    fill = open_price
                    reason = f"盘中止损-隔夜跳空止损(开盘{open_price:.3f}破止损{stop_price:.3f})"
                elif open_price <= stop_price:
                    fill = open_price
                    reason = f"盘中止损(触及{stop_price:.3f})"
                else:
                    fill = stop_price
                    reason = f"盘中止损(触及{stop_price:.3f})"
                return True, reason, fill

        return False, None, None

    def evaluate_eod_exit(
        self,
        row: Any,
        position: Position,
        profile: MarketProfile,
        idx: int,
        stock_code_tag: str = "",
        is_last_bar_of_day: bool = False,
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        收盘盯市评估（决定次日开盘是否执行平仓指令）。
        包含硬止损、移动止盈、波峰止盈、动量跃迁与持仓时间限制。
        返回: (pending_close_reason, pending_close_price)
        """
        current_time = _extract_time(row)
        can_exit_today = position.can_exit(
            current_time=current_time,
            settlement_days=profile.settlement_days,
            current_idx=idx,
        )
        close_price = row.close
        avg_entry_price = position.avg_entry_price
        peak_price = position.peak_price
        holding_bars = position.holding_bars
        holding_days = position.holding_days if position.holding_days > 0 else position.holding_bars
        active_target_horizon = position.target_horizon
        active_stop_pct = position.active_stop_pct
        active_trail_pct = position.active_trail_pct

        drawdown_from_cost = (close_price - avg_entry_price) / avg_entry_price if avg_entry_price > 0 else 0.0
        drawdown_from_peak = (close_price - peak_price) / peak_price if peak_price > 0 else 0.0
        ma_fast_val = getattr(row, 'ma_fast', None)

        pending_close_reason = None
        pending_close_price = None

        date_str = current_time.date() if hasattr(current_time, 'date') else str(current_time)

        # 常驻记忆 (Latch)：当持仓期间最高价浮盈曾跨过激活门槛时，锁定激活状态开启移动止盈
        peak_gain_from_cost = (peak_price - avg_entry_price) / avg_entry_price if avg_entry_price > 0 else 0.0
        if peak_gain_from_cost >= self.trailing_stop_min_profit and not getattr(position, 'trailing_active', False):
            position.trailing_active = True
            logger.info(
                f"[{date_str}] {stock_code_tag}🎯 峰值浮盈达到激活门槛({peak_gain_from_cost*100:+.2f}% >= {self.trailing_stop_min_profit*100:.1f}%)，"
                f"锁定移动止盈激活状态(Latch: 回撤容忍={active_trail_pct:.2%})"
            )

        if drawdown_from_cost <= -active_stop_pct:
            pending_close_reason = f"硬止损({drawdown_from_cost:.2%})"
        elif getattr(position, 'trailing_active', False) and drawdown_from_peak <= -active_trail_pct:
            pending_close_reason = f"移动止盈(峰值回撤{drawdown_from_peak:.2%})"
        elif (active_target_horizon is not None and holding_bars >= active_target_horizon):
            unit_str = "日" if holding_days == holding_bars else "Bar"
            if drawdown_from_cost <= 0:
                # 浮亏：模型预测波峰已过，假说彻底证伪，时间止损
                pending_close_reason = (
                    f"时间止损-模型信号证伪({holding_bars}{unit_str}/{active_target_horizon}{unit_str}目标,"
                    f"浮亏{drawdown_from_cost:.2%})"
                )
            else:
                # 浮盈检验：判断主升浪延续还是波峰滞涨
                atr_pct_val = getattr(row, 'atr_pct', np.nan)
                current_atr_pct = (
                    float(atr_pct_val)
                    if (atr_pct_val is not None and not pd.isna(atr_pct_val) and atr_pct_val > 0)
                    else None
                )

                if current_atr_pct is not None:
                    near_high_tol = float(np.clip(
                        self.k_near_high_atr * current_atr_pct,
                        self.min_near_high_pct,
                        self.max_near_high_pct,
                    ))
                else:
                    near_high_tol = self.min_near_high_pct

                is_near_high = (drawdown_from_peak >= -near_high_tol)
                is_trend_intact = (ma_fast_val is not None and not pd.isna(ma_fast_val) and close_price > ma_fast_val)
                has_profit_cushion = (drawdown_from_cost >= max(0.02, active_stop_pct * 1.2))

                if is_near_high and is_trend_intact and has_profit_cushion:
                    # 🚀【动量跃迁】：动量强劲，取消强平，移交收紧后的移动止盈护航
                    if current_atr_pct is not None:
                        dynamic_trail = float(np.clip(
                            self.k_trail_atr * current_atr_pct,
                            self.min_trail_pct,
                            self.max_trail_pct,
                        ))
                        tightened_trail = min(position.active_trail_pct, dynamic_trail)
                        position.active_trail_pct = max(self.min_trail_pct, tightened_trail)
                    else:
                        position.active_trail_pct = max(self.min_trail_pct, active_trail_pct * 0.75)

                    position.target_horizon = self.max_holding_days
                    logger.info(
                        f"[{date_str}] {stock_code_tag}🚀 触达目标时界({holding_bars}{unit_str})但动量强劲: "
                        f"浮盈={drawdown_from_cost*100:+.2f}%, 峰值回撤={drawdown_from_peak*100:+.2f}%(容许门限={near_high_tol*100:.2f}%) | "
                        f"取消强平，状态跃迁至动态ATR移动止盈护航(收紧回撤至{position.active_trail_pct:.2%})"
                    )
                else:
                    pending_close_reason = (
                        f"预测波峰到达止盈({holding_bars}{unit_str}/{active_target_horizon}{unit_str}目标,"
                        f"浮盈{drawdown_from_cost:.2%})"
                    )
        elif (holding_days >= self.max_holding_days if holding_days > 0 else holding_bars >= self.max_holding_days):
            pending_close_reason = f"持仓超时({holding_days}日)"
        elif self.intraday_exit_at_eod and is_last_bar_of_day and can_exit_today:
            pending_close_reason = "日内平仓模式(尾盘不隔夜)"
            pending_close_price = close_price
        # 注：根据方案 1 已移除 exec_bearish 走弱提前平仓逻辑，防止在主升浪中因微小预测扰动截断盈利。
        # 出场决策完全交由价格行为（硬止损、动态 ATR 移动止盈）与周期管理（波峰止盈、动量跃迁、持仓超时）接管。

        # T+N 未满足时撤回平仓决定，下一根 bar 再判
        if pending_close_reason and not can_exit_today:
            logger.debug(f"[{date_str}] {stock_code_tag}⏳ T+{profile.settlement_days} 限制，"
                         f"平仓指令【{pending_close_reason}】顺延")
            pending_close_reason = None
            pending_close_price = None

        if pending_close_reason:
            logger.info(
                f"[{date_str}] {stock_code_tag}⚠️ 触发平仓预警: 【{pending_close_reason}】 | "
                f"收盘价={close_price:.3f}, 成本={avg_entry_price:.3f}, "
                f"浮盈={drawdown_from_cost*100:+.2f}%, 峰值回撤={drawdown_from_peak*100:+.2f}%"
            )
        else:
            logger.debug(
                f"[{date_str}] {stock_code_tag}🔵 持仓第{holding_days}天: 现价={close_price:.3f}, "
                f"浮盈={drawdown_from_cost*100:+.2f}%, 峰值回撤={drawdown_from_peak*100:+.2f}%"
            )

        return pending_close_reason, pending_close_price
