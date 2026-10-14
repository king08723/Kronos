"""
backtest/visualizer
可视化层：静态图绘制 (Matplotlib)、交互式图绘制 (Plotly) 与全图贯通十字光标
"""

from .crosshair import inject_continuous_crosshair_script
from .plot_static import StaticVisualizer, plot_backtest_results
from .plot_interactive import InteractiveVisualizer, plot_backtest_interactive, HAS_PLOTLY
