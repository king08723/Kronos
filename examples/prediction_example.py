import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# 将项目根目录添加到 Python 模块搜索路径，以便导入 model 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosTokenizer, KronosPredictor

# 适配 Mac/Linux 环境下的中文字体显示
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def plot_prediction(kline_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    """
    绘制真实行情与预测行情的对比折线图（收盘价 + 成交量）

    参数说明:
    ----------
    kline_df : pd.DataFrame
        历史行情与真实未来的拼接数据，结构包含:
        - index: 行索引（通常为整数序号或 DatetimeIndex）
        - 'close': float, 真实收盘价序列 (Ground Truth)
        - 'volume': float, 真实成交量序列 (Ground Truth)
    pred_df : pd.DataFrame
        模型预测输出的未来数据，结构包含:
        - index: 与未来真实数据对应的索引 (Shape: [pred_len, 6])
        - 'close': float, 预测收盘价序列 (Prediction)
        - 'volume': float, 预测成交量序列 (Prediction)
    """
    # 对齐预测数据的索引，使其与 kline_df 尾部的真实预测区间一致
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    
    # 提取收盘价对比序列
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    # 提取成交量对比序列
    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    # 合并为用于绘图的 DataFrame
    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    # 创建双子图画布 (上方收盘价，下方成交量)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 1. 绘制收盘价对比
    ax1.plot(close_df['Ground Truth'], label='Ground Truth (真实值)', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction (预测值)', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price (收盘价)', fontsize=12)
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. 绘制成交量对比
    ax2.plot(volume_df['Ground Truth'], label='Ground Truth (真实成交量)', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction (预测成交量)', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume (成交量)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 步骤 1: 加载分词器（Tokenizer）与核心预测模型（Model）
# ==============================================================================
# 关键数据结构与类型说明:
# - KronosTokenizer (PyTorch Module):
#     负责把连续的 OHLCV 浮点数通过矢量量化（BSQuantizer）转换为离散 Token ID，
#     以及将模型生成的 Token ID 还原解码为价格浮点数。
# - Kronos (PyTorch Module):
#     生成式自回归 Transformer 主模型，学习时序演变并一步步预测未来的 Token。
print(">>> [Step 1] 加载 Tokenizer 与模型权重...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")  # 可选: Kronos-mini, Kronos-small, Kronos-base


# ==============================================================================
# 步骤 2: 实例化高级预测器（KronosPredictor）
# ==============================================================================
# 关键参数与类型说明:
# - model: Kronos 神经网络实例
# - tokenizer: KronosTokenizer 编解码器实例
# - device: 运算设备 (None 会自动按顺序探测: CUDA -> MPS(Mac芯片) -> CPU)
# - max_context (int): 模型自注意力机制能够接收的最大历史窗口长度 (默认 512)
# - clip (float): 归一化极值截断阈值 (默认 5.0，防止极大异常值干扰)
print(">>> [Step 2] 实例化 KronosPredictor 预测器...")
predictor = KronosPredictor(model, tokenizer, max_context=512)


# ==============================================================================
# 步骤 3: 准备数据与时间段划分
# ==============================================================================
print(">>> [Step 3] 读取并解析示例行情数据...")

# 兼容当前工作目录无论在项目根目录还是 examples/ 目录都能正确找到数据文件
data_path = "./data/XSHG_5min_600977.csv"
if not os.path.exists(data_path):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "XSHG_5min_600977.csv")

df = pd.read_csv(data_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])  # 必须转为 datetime64 类型以供提取时间周期特征

"""
【数据示例 1: df (原始数据集的实际内容展示)】
数据类型: pandas.DataFrame, 形状 (Shape): [2500, 7]
列说明: timestamps(时间戳), open(开), high(高), low(低), close(收), volume(量), amount(额)

           timestamps   open   high    low  close  volume     amount
0 2024-06-18 11:15:00  11.27  11.28  11.26  11.27   379.0   427161.0
1 2024-06-18 11:20:00  11.27  11.28  11.27  11.27   277.0   312192.0
2 2024-06-18 11:25:00  11.27  11.27  11.26  11.27   380.0   427954.0
3 2024-06-18 11:30:00  11.26  11.27  11.26  11.27   761.0   856971.0
4 2024-06-18 13:05:00  11.27  11.27  11.25  11.26  1439.0  1620733.0
"""

# 定义滑动窗口参数:
# - lookback (int): 作为条件输入的历史观测步长 (400 根 5分钟 K线，约等于 33.3 个交易小时)
# - pred_len (int): 预测未来的步长 (120 根 5分钟 K线，约等于 10 个交易小时)
lookback: int = 400
pred_len: int = 120

# 1. 历史观测数据输入 (Shape: [lookback, 6] -> [400, 6])
x_df: pd.DataFrame = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]

"""
【数据示例 2: x_df (传入模型作为历史条件的行情特征)】
数据类型: pandas.DataFrame, 形状 (Shape): [400, 6]
不含 timestamps 列，仅保留纯数值行情特征:

    open   high    low  close  volume     amount
0  11.27  11.28  11.26  11.27   379.0   427161.0
1  11.27  11.28  11.27  11.27   277.0   312192.0
2  11.27  11.27  11.26  11.27   380.0   427954.0
3  11.26  11.27  11.26  11.27   761.0   856971.0
4  11.27  11.27  11.25  11.26  1439.0  1620733.0
"""

# 2. 历史时间戳 (Series of Timestamp, 长度: 400)
x_timestamp: pd.Series = df.loc[:lookback-1, 'timestamps']

# 3. 未来时间戳 (Series of Timestamp, 长度: 120)
y_timestamp: pd.Series = df.loc[lookback:lookback+pred_len-1, 'timestamps']

"""
【数据示例 3: x_timestamp 与 y_timestamp (时间戳序列)】
数据类型: pandas.Series, dtype: datetime64[ns]
说明: 模型会提取 [minute, hour, weekday, day, month] 作为时间位置嵌入辅助预测。

x_timestamp.head():
0   2024-06-18 11:15:00
1   2024-06-18 11:20:00
2   2024-06-18 11:25:00
3   2024-06-18 11:30:00
4   2024-06-18 13:05:00
Name: timestamps, dtype: datetime64[ns]

y_timestamp.head():
400   2024-06-28 14:05:00
401   2024-06-28 14:10:00
402   2024-06-28 14:15:00
403   2024-06-28 14:20:00
404   2024-06-28 14:25:00
Name: timestamps, dtype: datetime64[ns]
"""

print(f"  * 历史输入窗口形态: {x_df.shape} (包含 {len(x_df)} 根 K 线)")
print(f"  * 目标预测步长: {pred_len} 个时间步")


# ==============================================================================
# 步骤 4: 执行生成式预测（Inference）
# ==============================================================================
# 核心方法说明: predictor.predict(...)
# 调节超参数:
# - T (float): Temperature 温度系数。
#       T < 1.0 时采样更确定、保守、平滑；
#       T > 1.0 时波动与多样性更大。
# - top_p (float): 核采样截断阈值 (0.0 ~ 1.0)，只在累积概率达 top_p 的 Token 中采样。
# - sample_count (int): 蒙特卡洛并行生成采样数，>1 时会对多组预测轨迹取均值，提升稳定性。
# - verbose (bool): 是否在控制台显示 step-by-step 进度条。
print(">>> [Step 4] 开始自回归时序预测...")
pred_df: pd.DataFrame = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

"""
【数据示例 4: pred_df (模型自回归推理输出的未来行情预测结果)】
数据类型: pandas.DataFrame, 形状 (Shape): [120, 6]
索引 (Index): 对应的未来时间戳 y_timestamp

                          open       high        low      close       volume
timestamps                                                                  
2024-06-28 14:05:00  10.822159  10.834056  10.811458  10.820602   523.151550
2024-06-28 14:10:00  10.820820  10.832644  10.809989  10.818702   679.045288
2024-06-28 14:15:00  10.817475  10.828601  10.807862  10.816195   553.784180
2024-06-28 14:20:00  10.815714  10.826743  10.806481  10.814709   538.001343
2024-06-28 14:25:00  10.809531  10.818578  10.787449  10.804219  1246.797607
"""


# ==============================================================================
# 步骤 5: 结果输出与对比可视化
# ==============================================================================
print("\n>>> [Step 5] 预测完成！未来部分 K 线预测值预览 (前 5 行):")
print(pred_df[['open', 'high', 'low', 'close', 'volume']].head())

# 拼接历史与未来真实数据用于对照画图
kline_df = df.loc[:lookback+pred_len-1]

# 弹出 Matplotlib 曲线对比图 (关闭窗口后脚本结束)
print("\n>>> 正在生成行情对比折线图 (蓝线: 真实走势, 红线: Kronos预测走势)...")
plot_prediction(kline_df, pred_df)
