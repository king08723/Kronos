"""
backtest/visualizer/plot_interactive.py
Plotly 交互式 4 联回测图绘制模块
"""

import pandas as pd
import numpy as np
from ..core.market_rules import MarketProfile

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

class InteractiveVisualizer:
    def __init__(self, profile: MarketProfile = None):
        self.profile = profile or MarketProfile.for_market('CN')
    def plot_backtest_interactive(
        self,
        signal_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
        metrics: dict,
        trades: list,
        stock_code: str,
        z_score_threshold: float = 1.2,
        crosshair_mode: str = "x",
    ):
        """
        绘制专业交互式四联回测图（Plotly 引擎，支持 4 轴联动、滚轮缩放、平移、悬停详情）：
          1. 价格曲线 + 买入/卖出信号点 + 持仓区间遮罩
          2. 策略净值 vs. 买入持有基准
          3. 水下动态回撤曲线 (Underwater Drawdown)
          4. Z-Score 信号强度柱状图 (红绿柱 + 入场阈值线)
        """
        if not HAS_PLOTLY or backtest_df is None or backtest_df.empty:
            return None

        dates = backtest_df.index
        close = backtest_df['close']
        cs = self.profile.currency_symbol
        cur = self.profile.currency

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            row_heights=[0.38, 0.24, 0.18, 0.20],
            subplot_titles=[
                f"<b>{stock_code} 价格走势与买卖信号执行 ({cur})</b>",
                "<b>策略累计净值 vs 买入持有基准</b>",
                "<b>水下动态回撤 (Underwater Drawdown)</b>",
                "<b>Z-Score 预测信号强度 (动量驱动)</b>"
            ]
        )

        # ---- 子图1：价格走势 + 买卖标注 + 持仓区间 ----
        # 1. 价格折线
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=close,
                mode='lines',
                name='真实收盘价',
                line=dict(color='#2980B9', width=2),
                hovertemplate="<b>日期</b>: %{x|%Y-%m-%d}<br><b>收盘价</b>: " + cs + "%{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )

        # 2. 买入信号
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        if buy_trades:
            b_dates = [t['date'] for t in buy_trades]
            b_prices = [t['price'] for t in buy_trades]
            b_texts = [f"买<br>{cs}{t['price']:.2f}" for t in buy_trades]
            b_hovers = [
                f"<b>【买入开仓】</b><br>日期: {t['date'].strftime('%Y-%m-%d')}<br>成交价: {cs}{t['price']:.2f}<br>买入股数: {t.get('shares', 0)} 股<br>目标时界: {t.get('target_horizon', 'N/A')} 天"
                for t in buy_trades
            ]
            fig.add_trace(
                go.Scatter(
                    x=b_dates,
                    y=b_prices,
                    mode='markers+text',
                    name='买入信号 ▲',
                    marker=dict(symbol='triangle-up', size=13, color='#E74C3C', line=dict(width=1, color='white')),
                    text=b_texts,
                    textposition="bottom center",
                    textfont=dict(size=9, color='#C0392B'),
                    hovertext=b_hovers,
                    hoverinfo="text"
                ),
                row=1, col=1
            )

        # 3. 卖出信号
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        if sell_trades:
            s_dates = [t['date'] for t in sell_trades]
            s_prices = [t['price'] for t in sell_trades]
            s_colors = ['#27AE60' if t['pnl_pct'] > 0 else '#8E44AD' for t in sell_trades]
            s_texts = []
            s_hovers = []
            for t in sell_trades:
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
                s_texts.append(f"卖({t['pnl_pct']:+.1%})<br>[{short_reason}]<br>{cs}{t['price']:.2f}")
                s_hovers.append(
                    f"<b>【卖出平仓】</b><br>"
                    f"日期: {t['date'].strftime('%Y-%m-%d')}<br>"
                    f"平仓价: {cs}{t['price']:.2f}<br>"
                    f"单笔盈亏: {t['pnl_pct']:+.2%}<br>"
                    f"触发原因: {t.get('reason', 'N/A')}<br>"
                    f"持仓股数: {t.get('shares', 0)} 股<br>"
                    f"账户资金: {cs}{t.get('cash_balance', 0):,.2f}"
                )
            fig.add_trace(
                go.Scatter(
                    x=s_dates,
                    y=s_prices,
                    mode='markers+text',
                    name='卖出平仓 ▼',
                    marker=dict(symbol='triangle-down', size=13, color=s_colors, line=dict(width=1, color='white')),
                    text=s_texts,
                    textposition="top center",
                    textfont=dict(size=9, color='#2C3E50'),
                    hovertext=s_hovers,
                    hoverinfo="text"
                ),
                row=1, col=1
            )

        # 4. 持仓区间遮罩
        if 'shares' in backtest_df.columns:
            in_pos = (backtest_df['shares'] > 0).astype(int)
            diff = in_pos.diff().fillna(in_pos.iloc[0])
            starts = backtest_df.index[diff == 1].tolist()
            ends = backtest_df.index[diff == -1].tolist()
            if len(in_pos) > 0 and in_pos.iloc[0] == 1 and (not starts or starts[0] > backtest_df.index[0]):
                starts.insert(0, backtest_df.index[0])
            if len(in_pos) > 0 and in_pos.iloc[-1] == 1:
                ends.append(backtest_df.index[-1])
            for s_date, e_date in zip(starts, ends):
                fig.add_vrect(
                    x0=s_date,
                    x1=e_date,
                    fillcolor="#F39C12",
                    opacity=0.12,
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )

        # ---- 子图2：策略净值 vs. 买入持有 ----
        strategy_cum = (1 + backtest_df['returns'].fillna(0)).cumprod()
        bh_returns = close.pct_change().fillna(0)
        bh_cum = (1 + bh_returns).cumprod()

        strat_ret = metrics.get('累计收益率', 0)
        bh_ret = (close.iloc[-1] / close.iloc[0] - 1) if len(close) > 0 else 0

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=strategy_cum,
                mode='lines',
                name=f"策略净值 ({strat_ret:+.2%})",
                line=dict(color='#E74C3C', width=2.4),
                hovertemplate="<b>日期</b>: %{x|%Y-%m-%d}<br><b>策略累计净值</b>: %{y:.3f}<extra></extra>"
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bh_cum,
                mode='lines',
                name=f"买入持有 ({bh_ret:+.2%})",
                line=dict(color='#7F8C8D', width=1.6, dash='dash'),
                hovertemplate="<b>日期</b>: %{x|%Y-%m-%d}<br><b>基准累计净值</b>: %{y:.3f}<extra></extra>"
            ),
            row=2, col=1
        )
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", line_width=1, row=2, col=1)

        mdd = metrics.get('最大回撤', 0)
        sharpe = metrics.get('夏普比率', 0)
        calmar = metrics.get('卡玛比率', 0)
        wr = metrics.get('胜率', 0)
        info_text = f"夏普: {sharpe:.2f} | 卡玛: {calmar:.2f} | MaxDD: {mdd:.2%} | 胜率: {wr:.1%}"
        fig.add_annotation(
            xref="x2 domain", yref="y2 domain",
            x=0.01, y=0.05,
            text=info_text,
            showarrow=False,
            font=dict(size=11, color="#2C3E50"),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(180, 180, 180, 0.6)",
            borderwidth=1,
            row=2, col=1
        )

        # ---- 子图3：水下回撤曲线 ----
        dd = (strategy_cum - strategy_cum.expanding().max()) / strategy_cum.expanding().max()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=dd,
                mode='lines',
                fill='tozeroy',
                name='水下回撤',
                line=dict(color='#E74C3C', width=1.2),
                fillcolor='rgba(231, 76, 60, 0.35)',
                hovertemplate="<b>日期</b>: %{x|%Y-%m-%d}<br><b>回撤幅度</b>: %{y:.2%}<extra></extra>"
            ),
            row=3, col=1
        )
        fig.add_hline(y=0.0, line_color="gray", line_width=0.8, row=3, col=1)

        # ---- 子图4：Z-Score 信号强度 ----
        if 'signal_score' in signal_df.columns:
            z_scores = signal_df['signal_score'].reindex(dates).clip(-4, 4)
            bar_colors = [
                'rgba(46, 204, 113, 0.75)' if val >= 0 else 'rgba(231, 76, 60, 0.75)'
                for val in z_scores
            ]
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=z_scores,
                    name='Z-Score 信号强度',
                    marker_color=bar_colors,
                    hovertemplate="<b>日期</b>: %{x|%Y-%m-%d}<br><b>Z-Score</b>: %{y:.2f}σ<extra></extra>"
                ),
                row=4, col=1
            )
            fig.add_hline(y=0.0, line_color="gray", line_width=0.8, row=4, col=1)
            fig.add_hline(
                y=z_score_threshold,
                line_dash="dash",
                line_color="#E67E22",
                line_width=1.2,
                annotation_text=f"入场阈值 (+{z_score_threshold}σ)",
                annotation_position="top right",
                row=4, col=1
            )

        # 全局布局与样式
        fig.update_layout(
            height=980,
            hovermode=crosshair_mode,
            spikedistance=-1,    # 全图任意垂直高度吸附触发时间线
            hoverdistance=100,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="rgba(200, 200, 200, 0.5)",
                borderwidth=1
            ),
            margin=dict(l=60, r=40, t=50, b=40),
            plot_bgcolor="rgba(250, 250, 250, 0.8)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        # 开启 4 轴完美拉通对齐的十字光标时间指示线 (Spikeline)
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(220, 224, 230, 0.6)",
            showspikes=True,
            spikemode="across",        # 垂直穿透整个子图绘图区
            spikesnap="hovered data",  # 统一吸附到当前日期，让 4 个子图垂直时间线绝对共振对齐
            spikethickness=1.2,
            spikedash="dash",          # 虚线标尺
            spikecolor="#34495E"       # 显眼的深灰蓝时间标尺线
        )
        fig.update_yaxes(showgrid=True, gridcolor="rgba(220, 224, 230, 0.6)")
        fig.update_yaxes(title_text=f"价格 ({cur})", row=1, col=1)
        fig.update_yaxes(title_text="累计净值", row=2, col=1)
        fig.update_yaxes(title_text="回撤", tickformat=".1%", row=3, col=1)
        fig.update_yaxes(title_text="Z-Score", row=4, col=1)

        return fig


def plot_backtest_interactive(
    signal_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    metrics: dict,
    trades: list,
    stock_code: str,
    z_score_threshold: float = 1.2,
    crosshair_mode: str = 'x',
    profile: MarketProfile = None,
):
    viz = InteractiveVisualizer(profile=profile)
    return viz.plot_backtest_interactive(
        signal_df=signal_df,
        backtest_df=backtest_df,
        metrics=metrics,
        trades=trades,
        stock_code=stock_code,
        z_score_threshold=z_score_threshold,
        crosshair_mode=crosshair_mode,
    )
