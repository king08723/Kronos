import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 添加项目路径以便导入自定义模块
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

# 设置中文字体（跨平台兼容 macOS、Windows、Linux）
plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'STHeiti',  # macOS
    'SimHei', 'Microsoft YaHei',  # Windows
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC',  # Linux
    'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def ensure_output_directory(output_dir):
    """确保输出目录存在，如果不存在则创建"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    return output_dir


def prepare_stock_data(csv_file_path, stock_code):
    """
    准备股票数据，转换为Kronos模型需要的格式

    参数:
    csv_file_path: CSV文件路径
    stock_code: 股票代码，用于显示信息

    返回:
    df: 处理后的DataFrame
    """
    print(f"正在加载和预处理股票 {stock_code} 数据...")

    # 读取CSV文件
    # 【示例数据: 原始 df 结构】
    #       日期    开盘价  最高价  最低价  收盘价      成交量         成交额
    # 0 2024-01-02  35.20  36.50  34.80  36.10  1205000.0  4.312000e+07
    # 1 2024-01-03  36.00  37.10  35.90  36.80  1350000.0  4.950000e+07
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

    # 检查数据列名并重命名为标准格式
    # 【标准映射字典】
    column_mapping = {
        '日期': 'timestamps',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '成交额': 'amount'
    }

    # 只重命名存在的列
    actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=actual_mapping)

    # 确保时间戳列存在并转换为datetime格式
    if 'timestamps' not in df.columns:
        # 如果数据有日期索引，重置索引
        if df.index.name == '日期':
            df = df.reset_index()
            df = df.rename(columns={'日期': 'timestamps'})

    df['timestamps'] = pd.to_datetime(df['timestamps'])

    # 按时间排序（由旧到新升序排列）
    df = df.sort_values('timestamps').reset_index(drop=True)

    # 【示例数据: 处理后的 df 结构，形状如 (242, 7)】
    #   timestamps   open   high    low  close     volume        amount
    # 0 2024-01-02  35.20  36.50  34.80  36.10  1205000.0  4.312000e+07
    # 1 2024-01-03  36.00  37.10  35.90  36.80  1350000.0  4.950000e+07
    print(f"✅ 数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
    print(f"数据列: {df.columns.tolist()}")

    return df


def calculate_prediction_parameters(df, target_days=100):
    """
    根据目标预测天数计算合适的参数

    参数:
    df: 股票数据DataFrame
    target_days: 目标预测天数（自然日）

    返回:
    lookback: 回看期数
    pred_len: 预测期数
    """
    # 计算平均交易日数量（考虑节假日）
    # 【示例】total_days = (2024-12-31 - 2024-01-02).days ≈ 364 天（自然日跨度）
    total_days = (df['timestamps'].max() - df['timestamps'].min()).days
    # 【示例】trading_days = 242（A股实际开市交易的天数）
    trading_days = len(df)
    # 【示例】trading_ratio = 242 / 364 ≈ 0.665（交易日/自然日比例，通常在 0.68 左右）
    trading_ratio = trading_days / total_days if total_days > 0 else 0.7  # 交易日比例

    # 计算目标预测的交易日数量
    # 【示例】target_days=100 时，pred_trading_days = int(100 * 0.665) = 66 天
    pred_trading_days = int(target_days * trading_ratio)

    # 设置回看期数为预测期数的2-3倍，但不超过数据总量的70%
    # 【示例】max_lookback = int(242 * 0.7) = 169
    # 【示例】lookback = min(66 * 2, 169, 242 - 66) = min(132, 169, 176) = 132（回看历史 132 个交易日）
    # 【示例】pred_len = min(66, 242 - 132) = 66（预测未来 66 个交易日）
    max_lookback = int(len(df) * 0.7)
    lookback = min(pred_trading_days * 2, max_lookback, len(df) - pred_trading_days)
    pred_len = min(pred_trading_days, len(df) - lookback)

    print(f"📊 参数计算:")
    print(f"  目标预测天数: {target_days} 天（自然日）")
    print(f"  预计交易日数量: {pred_trading_days} 天")
    print(f"  回看期数 (lookback): {lookback}")
    print(f"  预测期数 (pred_len): {pred_len}")

    return lookback, pred_len


def generate_future_dates_with_holidays(last_date, pred_len):
    """
    生成未来的交易日日期，考虑中国节假日

    参数:
    last_date: 最后一个历史数据的日期
    pred_len: 预测期数

    返回:
    future_dates: 未来的交易日日期列表
    """
    # 中国主要节假日（需要根据实际情况调整）
    # 【示例数据: 过滤名单】包含需要排除的法定休市日期
    holidays_2025 = [
        # 2025年国庆节假期（通常为10月1日-10月8日）
        datetime(2025, 10, 1), datetime(2025, 10, 2), datetime(2025, 10, 3),
        datetime(2025, 10, 4), datetime(2025, 10, 5), datetime(2025, 10, 6),
        datetime(2025, 10, 7), datetime(2025, 10, 8),  # 添加10月8日
        # 周末调休等可以根据需要添加
    ]

    future_dates = []
    current_date = last_date + timedelta(days=1)

    while len(future_dates) < pred_len:
        # 如果是工作日（周一到周五 weekday()<5）且不是节假日
        if current_date.weekday() < 5 and current_date not in holidays_2025:
            future_dates.append(current_date)
        current_date += timedelta(days=1)

    # 【示例数据: future_dates 列表，长度为 pred_len (如 66)】
    # [Timestamp('2025-01-02'), Timestamp('2025-01-03'), Timestamp('2025-01-06'), ...]
    print(f"📅 生成的未来交易日: 共 {len(future_dates)} 天")
    print(f"   起始日期: {future_dates[0].strftime('%Y-%m-%d')}")
    print(f"   结束日期: {future_dates[-1].strftime('%Y-%m-%d')}")

    # 显示节假日信息
    holiday_count = sum(1 for date in holidays_2025 if date > last_date)
    print(f"   包含节假日: {holiday_count} 天")

    return future_dates[:pred_len]


def plot_prediction_with_details(kline_df, pred_df, future_dates, stock_code="002354", stock_name="股票", pred_len=100,
                                 output_dir=".", sample_paths=None):
    """
    绘制详细的预测结果图表 - 支持蒙特卡洛多路径与扇形置信区间 (Confidence Ribbon)
    """
    # 确保输出目录存在
    ensure_output_directory(output_dir)

    # 确保数据长度一致
    min_len = min(len(pred_df), len(future_dates))
    pred_df = pred_df.iloc[:min_len]
    future_dates = future_dates[:min_len]

    # 设置预测数据的索引为未来日期
    pred_df.index = future_dates

    # 准备价格数据
    # 【示例】sr_close: 历史收盘价序列 (Series，索引为 timestamps, 值为 float)
    # 【示例】sr_pred_close: 预测收盘价序列 (Series，索引为 future_dates, 值为 float)
    sr_close = kline_df.set_index('timestamps')['close']
    sr_pred_close = pred_df['close']
    sr_close.name = '历史数据'
    sr_pred_close.name = "预测数据"

    # 准备成交量数据
    sr_volume = kline_df.set_index('timestamps')['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = '历史数据'
    sr_pred_volume.name = "预测数据"

    # 合并数据
    # 【示例】close_df 包含两列 ['历史数据', '预测数据']，形状为 (lookback + pred_len, 2)
    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    hist_close = close_df['历史数据'].dropna()
    pred_close = close_df['预测数据'].dropna()
    hist_volume = volume_df['历史数据'].dropna()
    pred_volumes = volume_df['预测数据'].dropna()

    # 准备完整的OHLCV数据集（按时间排序），用于鼠标点击时精确定位与数据提取
    # 【示例】hist_records 结构: 历史 K 线的完整 OHLCV 记录
    hist_records = kline_df.copy()
    hist_records['timestamps'] = pd.to_datetime(hist_records['timestamps'])
    hist_records = hist_records.set_index('timestamps')
    hist_records['data_type'] = '历史'

    # 【示例】pred_records 结构: 预测未来的 OHLCV 及置信区间记录
    pred_records = pred_df.copy()
    pred_records.index = pd.to_datetime(pred_records.index)
    pred_records['data_type'] = '预测'

    # 汇总完整OHLCV记录并去重
    # 【示例】combined_ohlcv: 包含历史+预测所有交易日的完整行情总表，形状如 (132 + 66, 10)
    combined_ohlcv = pd.concat([hist_records, pred_records]).sort_index()
    combined_ohlcv = combined_ohlcv[~combined_ohlcv.index.duplicated(keep='last')]
    all_dates = combined_ohlcv.index
    all_x_nums = mdates.date2num(all_dates.to_pydatetime())

    # 创建更大的图表
    fig = plt.figure(figsize=(18, 14))

    # 使用GridSpec创建更灵活的布局
    gs = plt.GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1])

    ax1 = fig.add_subplot(gs[0])  # 价格图表
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # 成交量图表（与价格图共享X轴时间）
    ax3 = fig.add_subplot(gs[2])  # 价格变动图表

    # 1. 价格图表 - 更大更清晰
    # 只显示最近200个交易日的历史数据，避免图表过于拥挤
    recent_history = hist_close.iloc[-min(200, len(hist_close)):]
    line_hist = ax1.plot(recent_history.index, recent_history.values, label='历史价格', color='#1f77b4', linewidth=2.5, alpha=0.9, marker='.', markersize=4)

    # 绘制蒙特卡洛多路径独立采样轨迹（浅橙细虚线）
    if sample_paths is not None and len(sample_paths) > 1:
        for p_idx, sp in enumerate(sample_paths):
            sp_c = sp['close'].iloc[:min_len]
            lbl = f'各采样路径 ({len(sample_paths)}条)' if p_idx == 0 else ""
            ax1.plot(pred_close.index, sp_c.values, color='#ff9f43',
                     linewidth=0.9, linestyle='--', alpha=0.4, label=lbl)

    # 绘制扇形置信区间 (Confidence Ribbon)
    if 'close_upper' in pred_df.columns and 'close_lower' in pred_df.columns:
        valid_l = min(len(pred_close), len(pred_df['close_upper']))
        upper_v = pred_df['close_upper'].iloc[:valid_l].values
        lower_v = pred_df['close_lower'].iloc[:valid_l].values
        ax1.fill_between(pred_close.index[:valid_l], lower_v, upper_v,
                         color='#ff7f0e', alpha=0.25, label='置信区间 (Confidence Ribbon)')

    # 绘制预测均值折线
    line_pred = ax1.plot(pred_close.index, pred_close.values, label='预测走势 (均值)',
                         color='#ff7f0e', linewidth=2.5, linestyle='-', marker='o', markersize=4)

    # 添加预测起始点的标记
    prediction_start_date = pred_close.index[0] if len(pred_close) > 0 else hist_close.index[-1]
    prediction_start_price = hist_close.iloc[-1]
    ax1.axvline(x=prediction_start_date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.annotate('预测起点', xy=(prediction_start_date, prediction_start_price),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.set_ylabel('收盘价 (元)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{stock_name}({stock_code}) 股票价格预测 - 未来{pred_len}个交易日\n[💡 滚轮: 缩放时间轴 | 按住左键拖动: 左右平移 | 单击: 查看OHLCV/成交量/置信区间 | 双击: 重置视野 | 右键: 清除提示]',
                  fontsize=14, fontweight='bold', pad=15)

    # 设置x轴日期格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 设置y轴格式
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    # 2. 成交量图表 - 显示历史与预测成交量
    recent_hist_vol = hist_volume.iloc[-min(200, len(hist_volume)):]
    if len(recent_hist_vol) > 0:
        ax2.bar(recent_hist_vol.index, recent_hist_vol.values,
                alpha=0.6, color='#1f77b4', label='历史成交量', width=0.8)
    if len(pred_volumes) > 0:
        ax2.bar(pred_volumes.index, pred_volumes.values,
                alpha=0.7, color='#ff7f0e', label='预测成交量', width=0.8)

    ax2.set_ylabel('成交量 (股)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-4:.0f}万' if x >= 1e4 else f'{x:.0f}'))

    # 3. 价格变动图表 - 优化显示
    if len(pred_close) > 0 and len(hist_close) > 0:
        price_change = pred_close - hist_close.iloc[-1]
        colors = ['green' if x >= 0 else 'red' for x in price_change]

        # 每5个交易日显示一个标签，避免过于拥挤
        bars = ax3.bar(range(len(price_change)), price_change.values, alpha=0.8, color=colors)

        # 在关键点添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if i % 10 == 0 or i == len(bars) - 1 or abs(height) > price_change.std():  # 每10天或最后一天或显著波动
                ax3.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:+.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=8, fontweight='bold')

        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    ax3.set_ylabel('价格变动 (元)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('交易日', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 设置x轴标签
    if len(pred_close) > 0:
        # 每10个交易日显示一个标签
        xticks_positions = list(range(0, len(pred_close), max(1, len(pred_close) // 10)))
        if len(pred_close) - 1 not in xticks_positions:
            xticks_positions.append(len(pred_close) - 1)
        ax3.set_xticks(xticks_positions)
        ax3.set_xticklabels([f'D{i + 1}' for i in xticks_positions])

    # 添加详细的统计信息框
    if len(pred_close) > 0 and len(hist_close) > 0:
        last_hist_val = hist_close.iloc[-1]
        last_pred_val = pred_close.iloc[-1]
        change_ratio = (last_pred_val / last_hist_val - 1) * 100
        pred_stats = {
            '股票代码': stock_code,
            '股票名称': stock_name,
            '当前价格': f"{last_hist_val:.2f} 元",
            '预测结束价格': f"{last_pred_val:.2f} 元",
            '预测涨跌幅': f"{change_ratio:+.2f}%",
            '预测期间最高价': f"{pred_close.max():.2f} 元",
            '预测期间最低价': f"{pred_close.min():.2f} 元",
            '预测波动率': f"{pred_close.std():.2f} 元",
            '预测起始日期': f"{pred_close.index[0].strftime('%Y-%m-%d')}",
            '预测结束日期': f"{pred_close.index[-1].strftime('%Y-%m-%d')}",
            '预测交易日数': f"{len(pred_close)} 天"
        }
        if 'close_upper' in pred_df.columns and 'close_lower' in pred_df.columns:
            last_idx = len(pred_close) - 1
            pred_stats['期末置信区间'] = f"{pred_df['close_lower'].iloc[last_idx]:.2f} ~ {pred_df['close_upper'].iloc[last_idx]:.2f} 元"

        stats_text = "\n".join([f"{k}: {v}" for k, v in pred_stats.items()])
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                 verticalalignment='bottom')

    # 记录默认初始视野，便于双击一键复位
    default_xlim = ax1.get_xlim()
    default_ylim = ax1.get_ylim()

    # ==================== 交互功能：滚轮缩放、鼠标拖动与点击查看 ====================
    # 1. 价格图浮窗 (OHLCV)
    annot_ax1 = ax1.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round4,pad=0.6", fc="#FFFFF2", ec="#2B4C7E", lw=1.5, alpha=0.96),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="#2B4C7E", lw=1.5),
        fontsize=10,
        zorder=30
    )
    annot_ax1.set_visible(False)
    highlight_ax1 = ax1.scatter([], [], s=140, facecolors='none', edgecolors='#FF0000', linewidths=2.5, zorder=35)

    # 2. 成交量图浮窗 (Volume)
    annot_ax2 = ax2.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round4,pad=0.6", fc="#F0F8FF", ec="#E65100", lw=1.5, alpha=0.96),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="#E65100", lw=1.5),
        fontsize=10,
        zorder=30
    )
    annot_ax2.set_visible(False)
    highlight_ax2 = ax2.scatter([], [], s=120, facecolors='none', edgecolors='#E65100', linewidths=2.5, zorder=35)

    # 3. 联动垂直指示线
    vline_ax1 = ax1.axvline(x=np.nan, color='#888888', linestyle=':', alpha=0.7, linewidth=1.2, visible=False)
    vline_ax2 = ax2.axvline(x=np.nan, color='#888888', linestyle=':', alpha=0.7, linewidth=1.2, visible=False)

    # 拖拽状态记录
    drag_state = {
        'is_pressed': False,
        'is_dragging': False,
        'press_x': None,
        'press_xlim': None,
    }

    def auto_adjust_axes():
        """根据当前可见时间窗口，自动自适应价格轴和成交量轴的Y范围"""
        cur_xlim = ax1.get_xlim()
        mask = (all_x_nums >= cur_xlim[0]) & (all_x_nums <= cur_xlim[1])
        if np.any(mask):
            sub_df = combined_ohlcv.iloc[mask]
            # 价格Y轴自适应
            vis_high = sub_df['high'].max() if 'high' in sub_df.columns else sub_df['close'].max()
            vis_low = sub_df['low'].min() if 'low' in sub_df.columns else sub_df['close'].min()
            if pd.notnull(vis_high) and pd.notnull(vis_low) and vis_high > vis_low:
                p_margin = (vis_high - vis_low) * 0.1
                ax1.set_ylim([vis_low - p_margin, vis_high + p_margin])

            # 成交量Y轴自适应
            vis_vol = sub_df['volume'].max() if 'volume' in sub_df.columns else None
            if pd.notnull(vis_vol) and vis_vol > 0:
                ax2.set_ylim([0, vis_vol * 1.15])

    def on_scroll(event):
        """鼠标滚轮缩放事件：以鼠标所在时间为中心等比放大/缩小"""
        if event.inaxes not in [ax1, ax2]:
            return
        if event.xdata is None:
            return

        cur_xlim = ax1.get_xlim()
        cur_xrange = cur_xlim[1] - cur_xlim[0]

        # 向上滚放大(缩小跨度)，向下滚缩小(放大跨度)
        is_zoom_in = False
        if hasattr(event, 'step') and event.step != 0:
            is_zoom_in = event.step > 0
        elif hasattr(event, 'button'):
            is_zoom_in = (event.button == 'up')

        scale_factor = 0.8 if is_zoom_in else 1.25
        new_range = cur_xrange * scale_factor

        # 限制最大缩放倍数（最少 3 天，最多不超过全部数据的 2 倍）
        total_data_range = max(all_x_nums) - min(all_x_nums)
        if new_range < 3 or new_range > total_data_range * 2.5:
            return

        # 以鼠标所在的 X 坐标为锚点进行等比缩放
        xdata = event.xdata
        ratio = (xdata - cur_xlim[0]) / cur_xrange
        new_xmin = xdata - new_range * ratio
        new_xmax = xdata + new_range * (1.0 - ratio)

        ax1.set_xlim([new_xmin, new_xmax])
        auto_adjust_axes()
        fig.canvas.draw_idle()

    def on_press(event):
        """鼠标按下：记录起始点，准备拖拽或点击"""
        if event.inaxes not in [ax1, ax2]:
            return

        if event.button == 1:  # 左键按下
            drag_state['is_pressed'] = True
            drag_state['is_dragging'] = False
            drag_state['press_x'] = event.x
            drag_state['press_xlim'] = ax1.get_xlim()

        elif event.button == 3:  # 右键点击清除所有浮窗提示
            annot_ax1.set_visible(False)
            annot_ax2.set_visible(False)
            highlight_ax1.set_offsets(np.empty((0, 2)))
            highlight_ax2.set_offsets(np.empty((0, 2)))
            vline_ax1.set_visible(False)
            vline_ax2.set_visible(False)
            fig.canvas.draw_idle()

    def on_motion(event):
        """鼠标移动：按住左键移动时左右拖拽平移图表"""
        if not drag_state['is_pressed'] or drag_state['press_x'] is None:
            return
        if event.x is None:
            return

        dx_pixels = event.x - drag_state['press_x']
        # 移动超过 3 个像素触发拖动模式
        if abs(dx_pixels) > 3:
            drag_state['is_dragging'] = True

        if drag_state['is_dragging']:
            # 将像素差转换为数据坐标差
            inv = ax1.transData.inverted()
            x0_data = inv.transform((drag_state['press_x'], 0))[0]
            x1_data = inv.transform((event.x, 0))[0]
            dx_data = x1_data - x0_data

            orig_xmin, orig_xmax = drag_state['press_xlim']
            new_xmin = orig_xmin - dx_data
            new_xmax = orig_xmax - dx_data

            ax1.set_xlim([new_xmin, new_xmax])
            auto_adjust_axes()
            fig.canvas.draw_idle()

    def handle_click_info(event):
        """处理单击查看 OHLCV 或 Volume 数据"""
        if event.xdata is None or event.inaxes not in [ax1, ax2]:
            return

        dist = np.abs(all_x_nums - event.xdata)
        idx = int(np.argmin(dist))
        nearest_x = all_x_nums[idx]
        row = combined_ohlcv.iloc[idx]
        dt = all_dates[idx]

        # 更新纵向参考虚线
        vline_ax1.set_xdata([nearest_x, nearest_x])
        vline_ax1.set_visible(True)
        vline_ax2.set_xdata([nearest_x, nearest_x])
        vline_ax2.set_visible(True)

        # 点击价格图 ax1 -> 展示 OHLCV 详情
        if event.inaxes == ax1:
            curr_open = row.get('open', np.nan)
            curr_high = row.get('high', np.nan)
            curr_low = row.get('low', np.nan)
            curr_close = row.get('close', np.nan)
            curr_vol = row.get('volume', np.nan)
            curr_amt = row.get('amount', np.nan)
            dtype = row.get('data_type', '数据')

            prev_close = combined_ohlcv.iloc[idx - 1]['close'] if idx > 0 else curr_close
            chg = curr_close - prev_close
            chg_pct = (chg / prev_close) * 100 if prev_close else 0.0
            color_mark = "🔴" if chg >= 0 else "🟢"
            chg_str = f"{color_mark} {chg:+.2f} ({chg_pct:+.2f}%)"

            vol_hands = curr_vol / 100 if pd.notnull(curr_vol) else 0
            if pd.notnull(curr_amt) and curr_amt > 0:
                amt_str = f"{curr_amt / 1e8:.2f} 亿元" if curr_amt >= 1e8 else f"{curr_amt / 1e4:.2f} 万元"
            else:
                amt_str = "无"

            # 蒙特卡洛置信区间展示
            conf_info = ""
            if dtype == '预测' and 'close_upper' in row and 'close_lower' in row:
                c_up = row['close_upper']
                c_low = row['close_lower']
                if pd.notnull(c_up) and pd.notnull(c_low):
                    span = c_up - c_low
                    span_pct = (span / curr_close) * 100 if curr_close else 0.0
                    conf_info = (
                        f"\n──────────────────────\n"
                        f"🔮 蒙特卡洛置信区间:\n"
                        f"乐观上限 (90%):  {c_up:.2f} 元\n"
                        f"悲观下限 (10%):  {c_low:.2f} 元\n"
                        f"区间跨度:        {span:.2f} 元 (±{span_pct/2:.1f}%)"
                    )

            ohlcv_text = (
                f"📅 {dt.strftime('%Y-%m-%d')} 【{dtype}行情】\n"
                f"──────────────────────\n"
                f"开盘价 (Open):   {curr_open:.2f} 元\n"
                f"最高价 (High):   {curr_high:.2f} 元\n"
                f"最低价 (Low):    {curr_low:.2f} 元\n"
                f"收盘价 (Close):  {curr_close:.2f} 元\n"
                f"涨跌幅 (Change): {chg_str}\n"
                f"成交量 (Volume): {curr_vol:,.0f} 股 ({vol_hands:,.0f} 手)\n"
                f"成交额 (Amount): {amt_str}"
                f"{conf_info}"
            )

            annot_ax1.xy = (nearest_x, curr_close)
            annot_ax1.set_text(ohlcv_text)
            annot_ax1.set_visible(True)
            annot_ax2.set_visible(False)
            highlight_ax1.set_offsets([[nearest_x, curr_close]])
            highlight_ax2.set_offsets(np.empty((0, 2)))
            fig.canvas.draw_idle()

        # 点击成交量图 ax2 -> 展示 Volume 详情
        elif event.inaxes == ax2:
            curr_vol = row.get('volume', 0)
            curr_amt = row.get('amount', 0)
            curr_close = row.get('close', np.nan)
            dtype = row.get('data_type', '数据')

            vol_hands = curr_vol / 100 if pd.notnull(curr_vol) else 0
            if curr_vol >= 1e8:
                vol_scale = f"{curr_vol / 1e8:.2f} 亿股"
            elif curr_vol >= 1e4:
                vol_scale = f"{curr_vol / 1e4:.2f} 万股"
            else:
                vol_scale = f"{curr_vol:,.0f} 股"

            if pd.notnull(curr_amt) and curr_amt > 0:
                amt_str = f"{curr_amt / 1e8:.2f} 亿元" if curr_amt >= 1e8 else f"{curr_amt / 1e4:.2f} 万元"
            else:
                amt_str = "无"

            vol_text = (
                f"📊 {dt.strftime('%Y-%m-%d')} 【{dtype}成交量】\n"
                f"──────────────────────\n"
                f"成交量 (股):  {curr_vol:,.0f} 股 ({vol_scale})\n"
                f"成交量 (手):  {vol_hands:,.0f} 手\n"
                f"成交额:       {amt_str}\n"
                f"当日收盘价:   {curr_close:.2f} 元"
            )

            annot_ax2.xy = (nearest_x, curr_vol)
            annot_ax2.set_text(vol_text)
            annot_ax2.set_visible(True)
            annot_ax1.set_visible(False)
            highlight_ax2.set_offsets([[nearest_x, curr_vol]])
            highlight_ax1.set_offsets(np.empty((0, 2)))
            fig.canvas.draw_idle()

    def on_release(event):
        """鼠标松开：若未拖拽则执行点击查看或双击复位"""
        if not drag_state['is_pressed']:
            return

        was_dragging = drag_state['is_dragging']
        drag_state['is_pressed'] = False
        drag_state['is_dragging'] = False
        drag_state['press_x'] = None
        drag_state['press_xlim'] = None

        # 如果刚才处于拖拽平移，松开时不触发点击弹窗
        if was_dragging:
            return

        # 双击左键：一键重置恢复初始视野
        if event.dblclick:
            ax1.set_xlim(default_xlim)
            ax1.set_ylim(default_ylim)
            annot_ax1.set_visible(False)
            annot_ax2.set_visible(False)
            highlight_ax1.set_offsets(np.empty((0, 2)))
            highlight_ax2.set_offsets(np.empty((0, 2)))
            vline_ax1.set_visible(False)
            vline_ax2.set_visible(False)
            auto_adjust_axes()
            fig.canvas.draw_idle()
            return

        # 普通单击：展示浮窗
        handle_click_info(event)

    # 注册事件监听：滚轮缩放、拖拽平移、单击展示、双击复位
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.tight_layout()

    # 保存高分辨率图片到指定目录（保存时浮窗处于隐藏状态，保证图片整洁）
    chart_filename = os.path.join(output_dir, f'{stock_code}_prediction_chart.png')
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 预测图表已保存: {chart_filename}")

    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)

    return close_df, volume_df


def generate_prediction_report(close_df, volume_df, pred_df, future_dates, stock_code="002354", stock_name="股票",
                               output_dir="."):
    """
    生成预测报告
    """
    # 确保输出目录存在
    ensure_output_directory(output_dir)

    print(f"\n{'=' * 70}")
    print(f"📊 {stock_name}({stock_code}) 股票预测报告")
    print(f"{'=' * 70}")

    hist_close = close_df['历史数据'].dropna()
    pred_close = close_df['预测数据'].dropna()
    vol_pred = volume_df['预测数据'].dropna()

    if len(pred_close) == 0 or len(hist_close) == 0:
        print("❌ 没有有效的预测数据可生成报告")
        return

    # 确保所有数组长度一致
    min_len = min(len(pred_close), len(vol_pred), len(future_dates))
    valid_pred_close = pred_close.iloc[:min_len]
    valid_vol_pred = vol_pred.iloc[:min_len]
    valid_future_dates = future_dates[:min_len]

    # 基本统计
    historical_close = hist_close.iloc[-1]
    predicted_close = valid_pred_close.iloc[-1]
    price_change_pct = (predicted_close / historical_close - 1) * 100

    print(f"🔮 预测概览:")
    print(f"   当前价格: {historical_close:.2f} 元")
    if 'close_upper' in pred_df.columns and 'close_lower' in pred_df.columns:
        last_up = pred_df['close_upper'].iloc[min_len - 1]
        last_low = pred_df['close_lower'].iloc[min_len - 1]
        print(f"   预测结束价格: {predicted_close:.2f} 元 (置信区间: {last_low:.2f} ~ {last_up:.2f} 元)")
    else:
        print(f"   预测结束价格: {predicted_close:.2f} 元")
    print(f"   预测涨跌幅: {price_change_pct:+.2f}%")
    print(f"   预测期间: {min_len} 个交易日")
    print(
        f"   预测时间范围: {valid_future_dates[0].strftime('%Y-%m-%d')} 到 {valid_future_dates[-1].strftime('%Y-%m-%d')}")

    print(f"\n📈 价格预测统计:")
    print(f"   预测最高价: {valid_pred_close.max():.2f} 元")
    print(f"   预测最低价: {valid_pred_close.min():.2f} 元")
    print(f"   预测平均价: {valid_pred_close.mean():.2f} 元")
    print(f"   价格波动率: {valid_pred_close.std():.2f} 元")

    print(f"\n📊 成交量预测统计:")
    print(f"   预测平均成交量: {valid_vol_pred.mean():,.0f} 手")
    print(f"   预测最大成交量: {valid_vol_pred.max():,.0f} 手")
    print(f"   预测最小成交量: {valid_vol_pred.min():,.0f} 手")

    # 保存详细预测数据到指定目录 - 确保所有数组长度一致
    detail_dict = {
        '日期': valid_future_dates,
        '预测收盘价_均值': valid_pred_close.values,
    }
    if 'close_upper' in pred_df.columns and 'close_lower' in pred_df.columns:
        detail_dict['预测收盘价_上限(90%)'] = pred_df['close_upper'].iloc[:min_len].values
        detail_dict['预测收盘价_下限(10%)'] = pred_df['close_lower'].iloc[:min_len].values
    detail_dict.update({
        '预测成交量': valid_vol_pred.values,
        '价格变动(元)': (valid_pred_close.values - historical_close),
        '价格变动(%)': ((valid_pred_close.values / historical_close - 1) * 100)
    })

    prediction_details = pd.DataFrame(detail_dict)
    prediction_file = os.path.join(output_dir, f'{stock_code}_detailed_predictions.csv')
    prediction_details.to_csv(prediction_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 详细预测数据已保存: {prediction_file}")


def main(stock_code="002354", stock_name="天娱数科", data_dir="./data", pred_days=100, output_dir="./output", sample_count=5):
    """
    主函数：执行股票价格预测

    参数:
    stock_code: 股票代码
    stock_name: 股票名称
    data_dir: 数据文件目录
    pred_days: 预测天数（自然日）
    output_dir: 输出文件目录
    sample_count: 蒙特卡洛采样路径数（>=2 时自动启用扇形置信区间）
    """
    # 构建数据文件路径
    csv_file_path = os.path.join(data_dir, f"{stock_code}_stock_data.csv")

    print(f"🎯 开始 {stock_name}({stock_code}) 股票价格预测")
    print("=" * 70)
    print(f"数据文件: {csv_file_path}")
    print(f"预测天数: {pred_days} 天（自然日）")
    print(f"采样路径: {sample_count} 条蒙特卡洛集成")
    print(f"输出目录: {output_dir}")

    # 检查数据文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"❌ 数据文件不存在: {csv_file_path}")
        print("请先运行数据获取脚本生成股票数据文件")
        return

    # 确保输出目录存在
    ensure_output_directory(output_dir)

    try:
        # 1. 加载模型和分词器
        print("\n步骤1: 加载Kronos模型和分词器...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        print("✅ 模型加载完成")

        # 2. 实例化预测器
        print("步骤2: 初始化预测器...")
        predictor = KronosPredictor(model, tokenizer, device=None, max_context=512)
        print("✅ 预测器初始化完成")

        # 3. 准备数据
        print("步骤3: 准备股票数据...")
        df = prepare_stock_data(csv_file_path, stock_code)

        # 4. 计算预测参数
        print("步骤4: 计算预测参数...")
        lookback, pred_len = calculate_prediction_parameters(df, target_days=pred_days)

        if pred_len <= 0:
            print("❌ 数据量不足，无法进行预测")
            return

        print(f"✅ 最终参数 - 回看期: {lookback}, 预测期: {pred_len}")

        # 5. 准备输入数据
        print("步骤5: 准备输入数据...")
        # 使用最新的数据作为输入
        # 【示例数据: x_df 为纯 OHLCV 数值矩阵，形状如 (lookback=132, 6)】
        #       open   high    low  close     volume        amount
        # 0    34.20  34.80  33.90  34.50  1500000.0  5.175000e+07
        # ...
        # 131  38.10  39.50  37.80  39.20  2100000.0  8.190000e+07
        x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        # 【示例数据: x_timestamp 为历史对应时间戳序列，形状如 (132,)】
        # 0      2024-06-20
        # ...
        # 131    2024-12-31
        x_timestamp = df.iloc[-lookback:]['timestamps'].reset_index(drop=True)

        # 生成未来日期（考虑节假日）
        # 【示例数据: future_dates 为未来预测交易日列表，长度如 pred_len=66】
        # [Timestamp('2025-01-02'), Timestamp('2025-01-03'), ..., Timestamp('2025-04-10')]
        last_historical_date = df['timestamps'].iloc[-1]
        future_dates = generate_future_dates_with_holidays(last_historical_date, pred_len)

        print(f"输入数据形状: {x_df.shape}")
        print(f"历史数据时间范围: {x_timestamp.iloc[0]} 到 {x_timestamp.iloc[-1]}")
        print(f"预测时间范围: {future_dates[0]} 到 {future_dates[-1]}")

        # 6. 执行预测（支持蒙特卡洛多路径集成与扇形置信区间）
        if sample_count > 1:
            print(f"步骤6: 执行蒙特卡洛多路径集成预测 (采样路径数: {sample_count})...")
            # 【示例数据: sample_paths 是长度为 sample_count(如 5) 的列表】
            # 每个元素是一个独立预测的 DataFrame，形状为 (pred_len=66, 6)，代表一种可能的未来平行走势
            sample_paths = predictor.predict_batch(
                df_list=[x_df] * sample_count,
                x_timestamp_list=[x_timestamp] * sample_count,
                y_timestamp_list=[pd.Series(future_dates)] * sample_count,
                pred_len=pred_len,
                T=1.0,         # 采样温度：控制波动性大小 (T=1.0 为自然波动，T<1 保守，T>1 激进)
                top_p=0.9,     # 核采样截断阈值：过滤掉累计概率超过 90% 的低概率尾部极端异常 Token
                sample_count=1,
                verbose=True
            )

            # 聚合各采样轨迹矩阵
            # 【示例数据: close_matrix 堆叠矩阵，形状为 (sample_count, pred_len)，如 (5, 66)】
            # 每行是一条独立采样路径在未来 66 天的收盘价数值
            # [
            #   [39.5, 40.1, 39.8, ..., 45.2],  <- 路径 1 (中度看涨)
            #   [39.1, 38.6, 38.2, ..., 41.0],  <- 路径 2 (震荡回调)
            #   ...
            #   [39.8, 41.2, 42.5, ..., 48.0],  <- 路径 5 (强力突破)
            # ]
            close_matrix = np.stack([p['close'].values for p in sample_paths], axis=0)
            vol_matrix = np.stack([p['volume'].values for p in sample_paths], axis=0)
            open_matrix = np.stack([p['open'].values for p in sample_paths], axis=0)
            high_matrix = np.stack([p['high'].values for p in sample_paths], axis=0)
            low_matrix = np.stack([p['low'].values for p in sample_paths], axis=0)
            amt_matrix = np.stack([p['amount'].values for p in sample_paths], axis=0)

            # 【示例数据: 置信区间分位数，形状均为 (pred_len=66,)】
            # c_upper: 90% 分位数（未来每一步的乐观走势上限），如 [39.7, 41.0, ..., 47.5]
            # c_lower: 10% 分位数（未来每一步的悲观走势下限），如 [38.9, 38.2, ..., 40.5]
            c_upper = np.percentile(close_matrix, 90, axis=0) if sample_count >= 5 else np.max(close_matrix, axis=0)
            c_lower = np.percentile(close_matrix, 10, axis=0) if sample_count >= 5 else np.min(close_matrix, axis=0)

            # 【示例数据: 最终预测汇总表 pred_df，形状为 (66, 9)】
            # 包含开高低收均值、成交量/额均值、置信上下限及标准差，索引为 future_dates
            pred_df = pd.DataFrame({
                'open': np.mean(open_matrix, axis=0),
                'high': np.mean(high_matrix, axis=0),
                'low': np.mean(low_matrix, axis=0),
                'close': np.mean(close_matrix, axis=0),
                'volume': np.mean(vol_matrix, axis=0),
                'amount': np.mean(amt_matrix, axis=0),
                'close_upper': c_upper,
                'close_lower': c_lower,
                'close_std': np.std(close_matrix, axis=0)
            }, index=pd.Series(future_dates))
            print(f"✅ 完成 {sample_count} 条蒙特卡洛轨迹集成与扇形置信区间推导")
        else:
            print("步骤6: 执行单路径价格预测...")
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=pd.Series(future_dates),
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=True
            )
            sample_paths = None
            print("✅ 预测完成")

        # 7. 显示预测结果
        print("\n步骤7: 显示预测结果...")
        print("预测数据前5行:")
        # 确保预测数据长度与未来日期一致
        min_len = min(len(pred_df), len(future_dates))
        pred_df = pred_df.iloc[:min_len]
        pred_df.index = future_dates[:min_len]
        print(pred_df[['open', 'high', 'low', 'close', 'volume']].head())

        # 8. 可视化结果（绘制扇形置信区间与独立采样轨迹）
        print("步骤8: 生成可视化图表...")
        # 使用最后一部分历史数据和预测数据
        kline_df = df.iloc[-lookback:].reset_index(drop=True)
        close_df, volume_df = plot_prediction_with_details(kline_df, pred_df, future_dates, stock_code, stock_name,
                                                           pred_len, output_dir, sample_paths=sample_paths)

        # 9. 生成预测报告
        print("步骤9: 生成预测报告...")
        generate_prediction_report(close_df, volume_df, pred_df, future_dates, stock_code, stock_name, output_dir)

        print(f"\n🎉 {stock_name}({stock_code}) 股票预测完成!")
        print("生成的文件:")
        print(f"  📊 {os.path.join(output_dir, stock_code + '_prediction_chart.png')} - 预测图表")
        print(f"  📋 {os.path.join(output_dir, stock_code + '_detailed_predictions.csv')} - 详细预测数据")

        # 显示预测总结
        hist_close_valid = close_df['历史数据'].dropna()
        pred_close_valid = close_df['预测数据'].dropna()
        if len(pred_close_valid) > 0 and len(hist_close_valid) > 0:
            print(f"\n📈 预测总结:")
            historical_price = hist_close_valid.iloc[-1]
            predicted_price = pred_close_valid.iloc[-1]
            change_pct = (predicted_price / historical_price - 1) * 100

            print(f"  当前价格: {historical_price:.2f} 元")
            print(f"  预测价格(均值): {predicted_price:.2f} 元")
            if 'close_upper' in pred_df.columns and 'close_lower' in pred_df.columns:
                print(f"  置信区间: {pred_df['close_lower'].iloc[min_len-1]:.2f} ~ {pred_df['close_upper'].iloc[min_len-1]:.2f} 元")
            print(f"  预期涨跌: {change_pct:+.2f}%")
            print(
                f"  预测时间: {future_dates[0].strftime('%Y-%m-%d')} 到 {future_dates[min_len - 1].strftime('%Y-%m-%d')}")

            if change_pct > 10:
                print(f"  🚀 模型预测未来{pred_len}个交易日大幅看涨 (+{change_pct:.1f}%)")
            elif change_pct > 5:
                print(f"  📈 模型预测未来{pred_len}个交易日看涨 (+{change_pct:.1f}%)")
            elif change_pct > 0:
                print(f"  ↗️ 模型预测未来{pred_len}个交易日微涨 (+{change_pct:.1f}%)")
            elif change_pct > -5:
                print(f"  ↘️ 模型预测未来{pred_len}个交易日微跌 ({change_pct:.1f}%)")
            elif change_pct > -10:
                print(f"  📉 模型预测未来{pred_len}个交易日看跌 ({change_pct:.1f}%)")
            else:
                print(f"  🔻 模型预测未来{pred_len}个交易日大幅看跌 ({change_pct:.1f}%)")

    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# 使用方法说明
if __name__ == "__main__":
    """
    股票预测工具 - 支持多股票预测

    使用方法：
    修改下面的 STOCK_CONFIG 来预测不同的股票
    """

    # ==================== 在这里修改股票配置 ====================
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    STOCK_CONFIG = {
        "stock_code": "300418",  # 股票代码
        "stock_name": "昆仑万维",  # 股票名称
        "data_dir": os.path.join(CURRENT_DIR, "data"),  # 数据文件目录
        "pred_days": 100,  # 预测100个自然日
        "sample_count": 5,  # 蒙特卡洛采样路径数 (>=2 自动开启扇形置信区间)
        "output_dir": os.path.join(CURRENT_DIR, "yuce")  # 输出文件目录
    }

    # 其他股票配置示例：
    # STOCK_CONFIG = {"stock_code": "000001", "stock_name": "平安银行", "data_dir": os.path.join(CURRENT_DIR, "data"), "pred_days": 100, "sample_count": 5, "output_dir": os.path.join(CURRENT_DIR, "yuce")}
    # STOCK_CONFIG = {"stock_code": "600036", "stock_name": "招商银行", "data_dir": os.path.join(CURRENT_DIR, "data"), "pred_days": 100, "sample_count": 5, "output_dir": os.path.join(CURRENT_DIR, "yuce")}
    # STOCK_CONFIG = {"stock_code": "300750", "stock_name": "宁德时代", "data_dir": os.path.join(CURRENT_DIR, "data"), "pred_days": 100, "sample_count": 5, "output_dir": os.path.join(CURRENT_DIR, "yuce")}
    # =========================================================

    print("🤖 智能股票预测工具")
    print("=" * 70)
    print(f"当前预测股票: {STOCK_CONFIG['stock_name']}({STOCK_CONFIG['stock_code']})")
    print(f"数据目录: {STOCK_CONFIG['data_dir']}")
    print(f"预测天数: {STOCK_CONFIG['pred_days']} 天（自然日）")
    print(f"蒙特卡洛采样: {STOCK_CONFIG['sample_count']} 条路径 (扇形置信区间)")
    print(f"输出目录: {STOCK_CONFIG['output_dir']}")
    print()

    # 运行主程序
    main(**STOCK_CONFIG)

    print(f"\n💡 提示：要预测其他股票，请修改代码中的 STOCK_CONFIG 变量")