"""
backtest/visualizer/plot_static.py
Matplotlib 专业 4 联回测图绘制模块
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from ..core.market_rules import MarketProfile

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class StaticVisualizer:
    def __init__(self, profile: MarketProfile = None):
        self.profile = profile or MarketProfile.for_market('CN')
    def plot_backtest_results(
        self,
        signal_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
        metrics: dict,
        trades: list,
        stock_code: str,
        output_dir: str,
        show_plot: bool = False,
    ):
        """
        绘制四联回测图：
          1. 价格 K 线 + 买卖信号标注
          2. 策略净值 vs. 买入持有基准
          3. 水下回撤曲线
          4. Z-Score 信号强度
        """
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(4, 1, figure=fig, height_ratios=[3, 2, 1.5, 1.5], hspace=0.35)

        ax1 = fig.add_subplot(gs[0])   # 价格 + 信号
        ax2 = fig.add_subplot(gs[1])   # 净值曲线
        ax3 = fig.add_subplot(gs[2])   # 回撤
        ax4 = fig.add_subplot(gs[3])   # Z-Score

        dates = backtest_df.index
        close = backtest_df['close']
        cs = self.profile.currency_symbol
        cur = self.profile.currency

        # ---- 子图1：价格走势 + 买卖标注 ----
        ax1.plot(dates, close, color='#4A90D9', linewidth=1.5, label='真实收盘价', zorder=2)
        ax1.fill_between(dates, close, close.min() * 0.95, alpha=0.05, color='#4A90D9')

        buy_trades  = [t for t in trades if t['action'] == 'BUY']
        sell_trades_list = [t for t in trades if t['action'] == 'SELL']

        for t in buy_trades:
            if t['date'] in backtest_df.index:
                ax1.scatter(t['date'], t['price'], marker='^', color='#E74C3C',
                            s=120, zorder=5, label='_buy')
                ax1.annotate(f"买\n{cs}{t['price']:.2f}", xy=(t['date'], t['price']),
                             xytext=(0, -22), textcoords='offset points',
                             fontsize=7, color='#E74C3C', ha='center')

        for t in sell_trades_list:
            if t['date'] in backtest_df.index:
                color = '#27AE60' if t['pnl_pct'] > 0 else '#8E44AD'
                ax1.scatter(t['date'], t['price'], marker='v', color=color,
                            s=120, zorder=5, label='_sell')
                reason_tag = t.get('reason', '')
                if '波峰' in reason_tag:
                    short_reason = '波峰止盈'
                elif '超时' in reason_tag:
                    short_reason = '持仓超时'
                elif '硬止损' in reason_tag:
                    short_reason = '硬止损'
                elif '移动止盈' in reason_tag:
                    short_reason = '移动止盈'
                elif '走弱' in reason_tag or '转空' in reason_tag:
                    short_reason = '趋势拐头'
                else:
                    short_reason = reason_tag[:6]
                ax1.annotate(f"卖({t['pnl_pct']:+.1%})\n[{short_reason}]\n{cs}{t['price']:.2f}",
                             xy=(t['date'], t['price']),
                             xytext=(0, 12), textcoords='offset points',
                             fontsize=7.5, fontweight='bold', color=color, ha='center')

        # 信号区间背景
        in_position = backtest_df['shares'] > 0
        ax1.fill_between(dates, close.min() * 0.95, close.max() * 1.02,
                         where=in_position, alpha=0.08, color='#F39C12', label='持仓区间')

        buy_patch   = mpatches.Patch(color='#E74C3C', label='买入 ▲')
        sell_patch  = mpatches.Patch(color='#27AE60', label='卖出（盈） ▼')
        sell_patch2 = mpatches.Patch(color='#8E44AD', label='卖出（亏） ▼')
        hold_patch  = mpatches.Patch(color='#F39C12', alpha=0.3, label='持仓区间')
        ax1.legend(handles=[buy_patch, sell_patch, sell_patch2, hold_patch],
                   loc='upper left', fontsize=8)
        ax1.set_title(f'{stock_code}  Kronos Pro 回测 — 样本外验证',
                      fontsize=13, fontweight='bold', pad=8)
        ax1.set_ylabel(f'价格 ({cur})', fontsize=10)
        ax1.grid(True, alpha=0.25)
        ax1.tick_params(axis='x', labelbottom=False)

        # ---- 子图2：策略净值 vs. 买入持有 ----
        strategy_cum = (1 + backtest_df['returns'].fillna(0)).cumprod()
        bh_returns   = close.pct_change().fillna(0)
        bh_cum       = (1 + bh_returns).cumprod()

        ax2.plot(dates, strategy_cum, color='#E74C3C', linewidth=2,
                 label=f'策略净值  ({metrics["累计收益率"]:+.2%})')
        ax2.plot(dates, bh_cum, color='#7F8C8D', linewidth=1.5,
                 linestyle='--', label=f'买入持有 ({(close.iloc[-1]/close.iloc[0]-1):+.2%})')
        ax2.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
        ax2.set_ylabel('累计净值', fontsize=10)
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(True, alpha=0.25)
        ax2.tick_params(axis='x', labelbottom=False)

        # 指标文本框
        mdd = metrics.get('最大回撤', 0)
        sharpe = metrics.get('夏普比率', 0)
        calmar = metrics.get('卡玛比率', 0)
        wr = metrics.get('胜率', 0)
        pf = metrics.get('盈亏比', float('nan'))
        pf_str = f"{pf:.2f}" if not np.isnan(pf) else "N/A"
        info_text = (
            f"夏普: {sharpe:.2f}  |  卡玛: {calmar:.2f}  |  MaxDD: {mdd:.2%}\n"
            f"胜率: {wr:.1%}  |  盈亏比: {pf_str}  |  "
            f"交易笔数: {metrics.get('总交易笔数', 0)}"
        )
        ax2.text(0.01, 0.03, info_text, transform=ax2.transAxes, fontsize=8.5,
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDFEFE', alpha=0.85))

        # ---- 子图3：回撤曲线 ----
        dd = (strategy_cum - strategy_cum.expanding().max()) / strategy_cum.expanding().max()
        ax3.fill_between(dates, dd, 0, alpha=0.55, color='#E74C3C', label='回撤')
        ax3.axhline(0, color='gray', linewidth=0.8)
        ax3.set_ylabel('回撤', fontsize=10)
        ax3.set_ylim(min(dd.min() * 1.2, -0.01), 0.005)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.25)
        ax3.tick_params(axis='x', labelbottom=False)

        from ..core.timeframes import detect_timeframe
        tf = detect_timeframe(backtest_df)
        bar_width = (tf.minutes / 1440.0) * 0.8 if tf.is_intraday else 0.8

        # ---- 子图4：Z-Score 信号强度 ----
        if 'signal_score' in signal_df.columns:
            z_scores = signal_df['signal_score'].reindex(dates)
            ax4.bar(dates, z_scores.clip(-4, 4), color=np.where(z_scores > 0, '#2ECC71', '#E74C3C'),
                    alpha=0.7, width=bar_width, label='Z-Score')
            ax4.axhline(0, color='gray', linewidth=0.8)
            # 默认阈值线（标注）
            ax4.axhline(1.2, color='orange', linewidth=1, linestyle='--', alpha=0.7, label='入场阈值(1.2σ)')
            ax4.set_ylabel('Z-Score', fontsize=10)
            ax4.legend(fontsize=8, loc='upper right')
            ax4.grid(True, alpha=0.2)
            ax4.set_xlabel('时间' if tf.is_intraday else '日期', fontsize=10)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f'{stock_code}_backtest_pro.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"📊 图表已保存: {out_path}")

        if show_plot:
            plt.show()

        return fig


def plot_backtest_results(
    signal_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    metrics: dict,
    trades: list,
    stock_code: str,
    output_dir: str,
    profile: MarketProfile = None,
    show_plot: bool = False,
):
    viz = StaticVisualizer(profile=profile)
    return viz.plot_backtest_results(
        signal_df=signal_df,
        backtest_df=backtest_df,
        metrics=metrics,
        trades=trades,
        stock_code=stock_code,
        output_dir=output_dir,
        show_plot=show_plot,
    )
