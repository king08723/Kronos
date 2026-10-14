"""
backtest/engine/metrics.py
量化策略绩效评估与深度风险归因体系 (Performance & Risk Attribution)
支持：
- 严格年化收益与夏普比率计算 (量纲对齐)
- 索提诺比率 (只惩罚下行波动) 与卡玛比率 (Calmar)
- 最大回撤 (基于净值真实序列)
- 真实成交金额口径盈亏比 (Profit Factor)
- 最大潜在盈利 (MFE) 与最大潜在逆行 (MAE) 逐笔交易质量诊断
- 自动适配多时间粒度 (1d / 30m / 15m) 年化折算乘数
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from ..core.types import TimeFrame
from ..core.market_rules import MarketProfile
from ..core.timeframes import detect_timeframe


class MetricsEvaluator:
    """绩效与风险指标评估器"""

    @staticmethod
    def evaluate(
        backtest_df: pd.DataFrame,
        trades: List[Dict[str, Any]],
        initial_capital: float,
        profile: MarketProfile,
        timeframe: Optional[Union[str, TimeFrame]] = None,
    ) -> Dict[str, Any]:
        """
        计算全套量化绩效指标，自动根据标的市场和时间粒度校准年化参数。
        """
        returns = backtest_df['returns'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        total_bars = len(backtest_df)

        if total_bars < 2:
            return {'error': '数据点不足，无法计算指标'}

        final_value = backtest_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital

        # 推断时间粒度与年化因子
        tf = TimeFrame.from_str(timeframe) if timeframe else detect_timeframe(backtest_df)
        ann_days = float(getattr(profile, 'annual_trading_days', 252) or 252)
        bars_per_day = tf.bars_per_day(profile.market)
        bars_per_year = ann_days * bars_per_day
        ann_factor = math.sqrt(bars_per_year)

        # 样本充分性 (至少相当于半年交易日)
        annualizable = total_bars >= (120 * bars_per_day)
        annual_return = ((1.0 + total_return) ** (bars_per_year / total_bars) - 1.0) if annualizable else np.nan

        volatility = returns.std(ddof=1) * ann_factor

        # 夏普比率 (无风险利率日频/Bar频折算)
        risk_free = profile.risk_free_rate
        bar_rf = (1.0 + risk_free) ** (1.0 / bars_per_year) - 1.0
        excess = returns - bar_rf
        sharpe = (excess.mean() / excess.std(ddof=1) * ann_factor
                  if excess.std(ddof=1) > 1e-12 else 0.0)

        # 索提诺比率
        downside = excess[excess < 0]
        sortino = (excess.mean() / downside.std(ddof=1) * ann_factor
                   if len(downside) > 1 and downside.std(ddof=1) > 1e-12 else np.nan)

        # 最大回撤
        pv = backtest_df['portfolio_value']
        peak = pv.expanding().max()
        drawdowns = (pv - peak) / peak
        max_drawdown = float(drawdowns.min())

        calmar = (annual_return / abs(max_drawdown)
                  if annualizable and abs(max_drawdown) > 1e-8 else np.nan)

        # 交易统计
        sell_trades = [t for t in trades if t['action'] == 'SELL' and t['reason'] != '回测结束强平']
        forced = [t for t in trades if t['action'] == 'SELL' and t['reason'] == '回测结束强平']
        win_trades = [t for t in sell_trades if t['pnl_pct'] > 0]
        lose_trades = [t for t in sell_trades if t['pnl_pct'] <= 0]

        win_rate = len(win_trades) / len(sell_trades) if sell_trades else 0.0
        avg_pnl = float(np.mean([t['pnl_pct'] for t in sell_trades])) if sell_trades else 0.0

        def _amount(t):
            return t['pnl_pct'] * t['price'] * t['shares'] / (1 + t['pnl_pct']) if t['pnl_pct'] > -1 else 0.0

        total_gain = sum(_amount(t) for t in win_trades)
        total_loss = abs(sum(_amount(t) for t in lose_trades))
        if total_loss > 1e-8:
            profit_factor = total_gain / total_loss
        elif total_gain > 0:
            profit_factor = float('inf')
        else:
            profit_factor = float('nan')

        stat_reliable = len(sell_trades) >= 20
        exposure = float((backtest_df['shares'] > 0).mean())

        all_sells = sell_trades + forced
        tot_commission = sum(t.get('commission', 0.0) for t in trades)
        tot_pnl_amt = sum(t.get('pnl_amt', 0.0) for t in all_sells)
        avg_hold_days = float(np.mean([t['holding_days'] for t in sell_trades])) if sell_trades else 0.0
        max_pnl_win = float(max([t['pnl_pct'] for t in win_trades])) if win_trades else 0.0
        max_pnl_lose = float(min([t['pnl_pct'] for t in lose_trades])) if lose_trades else 0.0
        avg_mfe = float(np.mean([t['mfe_pct'] for t in sell_trades])) if sell_trades and 'mfe_pct' in sell_trades[0] else 0.0
        avg_mae = float(np.mean([t['mae_pct'] for t in sell_trades])) if sell_trades and 'mae_pct' in sell_trades[0] else 0.0

        return {
            '累计收益率':   total_return,
            '年化收益率':   annual_return,
            '年化波动率':   volatility,
            '夏普比率':     sharpe,
            '索提诺比率':   sortino,
            '卡玛比率':     calmar,
            '最大回撤':     max_drawdown,
            '总交易笔数':   len(sell_trades),
            '期末强平笔数': len(forced),
            '胜率':         win_rate,
            '盈亏比':       profit_factor,
            '单笔平均盈亏': avg_pnl,
            '持仓占比':     exposure,
            '期末总资产':   final_value,
            '总手续费':     tot_commission,
            '总盈亏金额':   tot_pnl_amt,
            '平均持仓天数': avg_hold_days,
            '最大单笔盈利': max_pnl_win,
            '最大单笔亏损': max_pnl_lose,
            '平均MFE':      avg_mfe,
            '平均MAE':      avg_mae,
            '回测天数':     total_bars if not tf.is_intraday else int(total_bars / bars_per_day),
            '统计可靠':     stat_reliable,
            '可年化':       annualizable,
        }
