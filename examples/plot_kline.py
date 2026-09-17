# -*- coding: utf-8 -*-
"""
基于 matplotlib.pyplot 封装的专业交互式 K 线蜡烛图模块
完全独立文件封装，供其它模块直接导入调用。

特性:
1. 原生 matplotlib.pyplot 弹窗渲染，无需浏览器或 HTML 文件
2. 真实行情与预测行情以独立的蜡烛图（Candlestick）形态呈现 (红涨绿跌)
3. 鼠标滚轮自由缩放 (Zoom)、鼠标右键/中键拖拽平移 (Pan)
4. 鼠标点击任意 K 线，即时弹出显示详细 OHLCV 数据卡片与十字定位线
5. 完美适配 Mac/Linux 中文字体
"""

from typing import Optional, Tuple
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# 适配中文与负号
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class InteractiveKLineChart:
    """
    Matplotlib 交互式 K 线图封装类
    负责绘制 K 线与成交量，并绑定鼠标交互事件（滚轮缩放、拖动平移、点击显示 OHLCV）。
    """
    def __init__(
        self,
        kline_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        title: str = "Kronos 行情走势与预测对比 (交互式K线蜡烛图)",
        color_up: str = "#EB4D4B",       # 真实阳线颜色 (中国红)
        color_down: str = "#2ECC71",     # 真实阴线颜色 (中国绿)
        pred_color_up: str = "#F39C12",  # 预测阳线颜色 (醒目橙金)
        pred_color_down: str = "#3498DB",# 预测阴线颜色 (醒目天蓝)
        dark_mode: bool = False          # 是否启用券商深色主题
    ):
        self.kline_df = kline_df.copy()
        self.pred_df = pred_df.copy()
        self.title = title
        self.color_up = color_up
        self.color_down = color_down
        self.pred_color_up = pred_color_up
        self.pred_color_down = pred_color_down
        self.dark_mode = dark_mode

        # 统一列名
        self.kline_df.columns = [str(c).lower() for c in self.kline_df.columns]
        self.pred_df.columns = [str(c).lower() for c in self.pred_df.columns]

        self.total_len = len(self.kline_df)
        self.pred_len = len(self.pred_df)
        self.history_len = max(0, self.total_len - self.pred_len)

        # 对齐预测数据索引
        self.pred_df.index = self.kline_df.index[-self.pred_len:]

        # 提取时间字符串
        time_col = None
        for c in ['timestamps', 'timestamp', 'date', 'datetime']:
            if c in self.kline_df.columns:
                time_col = c
                break

        if time_col:
            self.dates = self.kline_df[time_col].astype(str).tolist()
        elif isinstance(self.kline_df.index, pd.DatetimeIndex):
            self.dates = self.kline_df.index.astype(str).tolist()
        else:
            self.dates = [f"#{i}" for i in range(self.total_len)]

        # 交互拖动状态
        self._is_panning = False
        self._pan_start_x = None
        self._pan_start_xlim = None
        self._cursor_line = None
        self._annot = None

    def draw(self) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        # 设置主题背景
        bg_color = "#131722" if self.dark_mode else "#FFFFFF"
        text_color = "#E0E3EB" if self.dark_mode else "#222222"
        grid_color = "#242938" if self.dark_mode else "#F0F0F0"

        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}
        )
        self.fig.patch.set_facecolor(bg_color)
        self.ax1.set_facecolor(bg_color)
        self.ax2.set_facecolor(bg_color)

        x = np.arange(self.total_len)

        # 1. 绘制真实行情蜡烛图 (Ground Truth)
        real_opens = self.kline_df['open'].values
        real_highs = self.kline_df['high'].values
        real_lows = self.kline_df['low'].values
        real_closes = self.kline_df['close'].values
        real_vols = self.kline_df['volume'].values

        is_up = real_closes >= real_opens
        real_colors = np.where(is_up, self.color_up, self.color_down)

        # 真实影线
        self.ax1.vlines(x, real_lows, real_highs, color=real_colors, linewidth=1.0, alpha=0.9)
        # 真实实体
        candle_bottoms = np.minimum(real_opens, real_closes)
        candle_heights = np.abs(real_closes - real_opens)
        candle_heights = np.maximum(candle_heights, 0.001)  # 避免平盘高度为0不显示
        self.ax1.bar(
            x, candle_heights, bottom=candle_bottoms,
            width=0.6, color=real_colors, edgecolor=real_colors, alpha=0.9,
            label='真实走势 (Ground Truth)'
        )

        # 2. 绘制预测区间蜡烛图 (Prediction)
        if self.pred_len > 0:
            pred_x = np.arange(self.history_len, self.total_len)
            pred_opens = self.pred_df['open'].values
            pred_highs = self.pred_df['high'].values
            pred_lows = self.pred_df['low'].values
            pred_closes = self.pred_df['close'].values
            pred_vols = self.pred_df['volume'].values

            pred_is_up = pred_closes >= pred_opens
            pred_colors = np.where(pred_is_up, self.pred_color_up, self.pred_color_down)

            # 预测影线 (虚线或稍微粗一点区分)
            self.ax1.vlines(pred_x, pred_lows, pred_highs, color=pred_colors, linewidth=1.2, alpha=0.9)
            # 预测蜡烛实体 (用中空/半透明条形区别真实K线)
            p_bottoms = np.minimum(pred_opens, pred_closes)
            p_heights = np.abs(pred_closes - pred_opens)
            p_heights = np.maximum(p_heights, 0.001)
            self.ax1.bar(
                pred_x, p_heights, bottom=p_bottoms,
                width=0.45, color=pred_colors, edgecolor=pred_colors,
                alpha=0.65, label='Kronos预测走势 (Prediction)'
            )

            # 绘制预测起点垂直分割线
            self.ax1.axvline(
                x=self.history_len - 0.5, color='#E74C3C', linestyle='--', linewidth=1.5,
                label='预测起点'
            )
            self.ax2.axvline(x=self.history_len - 0.5, color='#E74C3C', linestyle='--', linewidth=1.5)

            # 在预测区间添加浅色高亮背景
            self.ax1.axvspan(
                self.history_len - 0.5, self.total_len - 0.5,
                color='#FFA726', alpha=0.08, label='预测区间'
            )

        # 3. 绘制成交量副图 (Volume)
        self.ax2.bar(
            x, real_vols, width=0.6, color=real_colors, alpha=0.7,
            label='真实成交量'
        )
        if self.pred_len > 0:
            self.ax2.bar(
                pred_x, pred_vols, width=0.35, color=pred_colors, alpha=0.8,
                label='预测成交量'
            )

        # 4. 坐标轴与网格美化
        self.ax1.set_title(self.title, fontsize=13, fontweight='bold', color=text_color, pad=12)
        self.ax1.set_ylabel("价格 (Price)", fontsize=11, color=text_color)
        self.ax2.set_ylabel("成交量 (Volume)", fontsize=11, color=text_color)
        self.ax2.set_xlabel("时间序号 (可在图上用鼠标滚轮缩放，右键拖拽，左键点击查看OHLCV)", fontsize=10, color=text_color)

        self.ax1.grid(True, linestyle=':', alpha=0.5, color=grid_color)
        self.ax2.grid(True, linestyle=':', alpha=0.5, color=grid_color)
        self.ax1.legend(loc='upper left', fontsize=9)
        self.ax2.legend(loc='upper left', fontsize=9)

        # X轴刻度稀疏化显示真实时间
        step = max(1, self.total_len // 8)
        tick_indices = list(range(0, self.total_len, step))
        if tick_indices[-1] != self.total_len - 1:
            tick_indices.append(self.total_len - 1)
        self.ax2.set_xticks(tick_indices)
        self.ax2.set_xticklabels(
            [self.dates[i] for i in tick_indices],
            rotation=20, ha='right', fontsize=9, color=text_color
        )

        # 默认视口：聚焦在最近 80 根 K 线，保证蜡烛清晰饱满
        init_left = max(0, self.total_len - 80)
        init_right = self.total_len + 1
        self.ax1.set_xlim(init_left, init_right)

        # 5. 初始化交互悬停/点击标注框 (Annotation)
        self._annot = self.ax1.annotate(
            "", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6", fc="#2C3E50", ec="none", alpha=0.9),
            color="white", fontsize=9,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#E74C3C")
        )
        self._annot.set_visible(False)

        # 顶部提示栏
        self._info_text = self.ax1.text(
            0.02, 0.96, "[提示] 滚轮缩放时间轴 | 右键按住拖动画布 | 左键点击任一K线显示详细OHLCV",
            transform=self.ax1.transAxes, fontsize=9, color="#888888",
            bbox=dict(boxstyle="square,pad=0.2", fc=bg_color, ec="none", alpha=0.8)
        )

        # 6. 绑定鼠标事件 (滚轮缩放、右键拖动、左键点击)
        self._connect_events()

        plt.tight_layout()
        return self.fig, (self.ax1, self.ax2)

    def _connect_events(self):
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

    def _on_scroll(self, event):
        """鼠标滚轮缩放时间轴"""
        if event.inaxes not in (self.ax1, self.ax2):
            return
        base_scale = 1.25
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale

        cur_xlim = self.ax1.get_xlim()
        cur_width = cur_xlim[1] - cur_xlim[0]
        xdata = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2

        new_width = cur_width * scale_factor
        # 限制最小显示 10 根，最大不超过全长 1.5 倍
        if 10 < new_width < self.total_len * 1.5:
            rel_pos = (xdata - cur_xlim[0]) / cur_width
            new_left = xdata - new_width * rel_pos
            new_right = xdata + new_width * (1 - rel_pos)
            self.ax1.set_xlim(new_left, new_right)
            self.fig.canvas.draw_idle()

    def _on_button_press(self, event):
        """鼠标按下：左键点击查数据，右键/中键按下准备拖拽"""
        if event.inaxes not in (self.ax1, self.ax2):
            return

        if event.button == 1:  # 左键单击查看 OHLCV
            self._show_kline_info(event)
        elif event.button in (2, 3):  # 中键或右键拖拽
            self._is_panning = True
            self._pan_start_x = event.xdata
            self._pan_start_xlim = self.ax1.get_xlim()

    def _on_button_release(self, event):
        """鼠标松开：结束拖拽"""
        if event.button in (2, 3):
            self._is_panning = False
            self._pan_start_x = None

    def _on_motion(self, event):
        """鼠标移动：拖拽平移视口"""
        if self._is_panning and event.inaxes in (self.ax1, self.ax2) and event.xdata is not None:
            dx = event.xdata - self._pan_start_x
            new_xlim = [self._pan_start_xlim[0] - dx, self._pan_start_xlim[1] - dx]
            self.ax1.set_xlim(new_xlim)
            self.fig.canvas.draw_idle()

    def _show_kline_info(self, event):
        """在点击位置弹出详细 OHLCV 悬停气泡"""
        if event.xdata is None:
            return
        idx = int(round(event.xdata))
        if idx < 0 or idx >= self.total_len:
            return

        # 获取数据
        t = self.dates[idx]
        o = self.kline_df['open'].iloc[idx]
        h = self.kline_df['high'].iloc[idx]
        l = self.kline_df['low'].iloc[idx]
        c = self.kline_df['close'].iloc[idx]
        v = self.kline_df['volume'].iloc[idx]
        chg_pct = (c - o) / o * 100 if o != 0 else 0
        tag = "[涨]" if c >= o else "[跌]"

        lines = [
            f"时间: {t}",
            f"真实行情: {tag} ({chg_pct:+.2f}%)",
            f"  开: {o:.2f} | 高: {h:.2f}",
            f"  低: {l:.2f} | 收: {c:.2f}",
            f"  量: {v:,.0f}"
        ]

        # 如果落在预测区间内，同时展示预测值
        if idx >= self.history_len:
            p_idx = idx - self.history_len
            po = self.pred_df['open'].iloc[p_idx]
            ph = self.pred_df['high'].iloc[p_idx]
            pl = self.pred_df['low'].iloc[p_idx]
            pc = self.pred_df['close'].iloc[p_idx]
            pv = self.pred_df['volume'].iloc[p_idx]
            err = pc - c
            lines.extend([
                f"--------------------",
                f"Kronos 预测值 (误差: {err:+.2f}):",
                f"  开: {po:.2f} | 高: {ph:.2f}",
                f"  低: {pl:.2f} | 收: {pc:.2f}",
                f"  量: {pv:,.0f}"
            ])

        info_str = "\n".join(lines)

        # 更新指示框位置与文字
        self._annot.xy = (idx, c)
        self._annot.set_text(info_str)
        self._annot.set_visible(True)

        # 绘制/更新十字参考竖线
        if self._cursor_line:
            self._cursor_line.remove()
        self._cursor_line = self.ax1.axvline(idx, color="#E74C3C", linestyle=":", linewidth=1.0)

        # 同时在顶部提示栏简报
        self._info_text.set_text(f"选中: {t} | 收盘: {c:.2f} | 涨跌: {chg_pct:+.2f}% | 成交量: {v:,.0f}")
        self._info_text.set_color("#FF5722")

        self.fig.canvas.draw_idle()


def plot_prediction(
    kline_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    title: str = "Kronos 行情走势与预测对比 (交互式K线蜡烛图)",
    dark_mode: bool = False
) -> None:
    """
    独立封装的 K 线显示函数，供其它模块直接调用。
    基于 matplotlib.pyplot 弹出独立窗口，支持蜡烛图展示、滚轮缩放、拖拽平移、点击显示 OHLCV。

    参数说明:
    ----------
    kline_df : pd.DataFrame
        历史行情与真实未来的拼接数据，包含:
        - 时间列: 'timestamps', 'timestamp', 'date' 之一，或时间索引
        - 'open', 'high', 'low', 'close', 'volume'
    pred_df : pd.DataFrame
        模型预测输出的未来数据，包含:
        - 'open', 'high', 'low', 'close', 'volume'
    title : str, 可选
        窗口主标题
    dark_mode : bool, 可选
        是否开启券商深色主题，默认为 False (经典白底)
    """
    chart = InteractiveKLineChart(kline_df, pred_df, title=title, dark_mode=dark_mode)
    chart.draw()
    plt.show()


# 兼容原函数名与自定义扩展命名
plot_prediction_candlestick = plot_prediction


if __name__ == "__main__":
    # 本地快速测试
    print(">>> 正在启动 Matplotlib 交互式 K 线图本地自测...")
    dates = pd.date_range("2024-06-01 09:30", periods=80, freq="5min")
    np.random.seed(42)

    c = 100.0
    opens, highs, lows, closes, vols = [], [], [], [], []
    for _ in range(80):
        step = np.random.randn() * 0.4
        o = c
        c = o + step
        h = max(o, c) + abs(np.random.randn() * 0.2)
        l = min(o, c) - abs(np.random.randn() * 0.2)
        v = np.random.randint(200, 1500)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c); vols.append(v)

    mock_kline = pd.DataFrame({
        'timestamps': dates,
        'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': vols
    })
    pred_len = 25
    mock_pred = mock_kline.tail(pred_len).copy()
    mock_pred['open'] += np.random.randn(pred_len) * 0.15
    mock_pred['close'] += np.random.randn(pred_len) * 0.15

    # 运行展示
    chart = InteractiveKLineChart(mock_kline, mock_pred)
    fig, axes = chart.draw()
    plt.show()
    plt.savefig("./outputs/test_mpl_kline.png", dpi=150)
    print(">>> 自测图已生成至 ./outputs/test_mpl_kline.png")
