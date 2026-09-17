# Kronos Examples 学习与实战指南

本项目在 `examples/` 目录下提供了由浅入深的完整示例，涵盖从基础单序列推理、批量并行推理、真实金融数据获取，到量化策略回测与 GUI 可视化全流程。

---

## 目录
- [一、 由浅入深进阶路线图](#一-由浅入深进阶路线图)
- [二、 基础篇：单序列预测与变体对比](#二-基础篇单序列预测与变体对比)
- [三、 核心调优技巧：提升预测精确度与稳定性](#三-核心调优技巧提升预测精确度与稳定性)
- [四、 下一步进阶：批量推理与并发处理](#四-下一步进阶批量推理与并发处理)

---

## 一、 由浅入深进阶路线图

| 阶段 | 示例文件 | 核心学习目标与特点 | 复杂度 |
| :--- | :--- | :--- | :--- |
| **Stage 1: 基础单序列** | [prediction_example.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_example.py)<br>[prediction_wo_vol_example.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_wo_vol_example.py) | **基础入门 / 极简变体**：模型加载、单序列数据格式组织（全特征 OHLCVA vs 纯价格 OHLC）。 | ★☆☆☆☆ |
| **Stage 2: 批处理与并发** | [prediction_batch_example.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_batch_example.py) | **API扩展与性能加速**：从单样本 `predict` 跨越到 `predict_batch`，理解多个时间窗口并行组织与 GPU 批量自回归采样。 | ★★☆☆☆ |
| **Stage 3: 真实市场日线适配** | [prediction_cn_markets_day.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_cn_markets_day.py) | **市场规则适配**：针对 A 股日 K 线、交易日历机制、真实市场时间戳对齐与长跨度预测。 | ★★★☆☆ |
| **Stage 4: 实时数据获取与工程管道** | [get_akshare_date_2024-2025_x.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/get_akshare_date_2024-2025_x.py)<br>[prediction_akshare_2024-2025.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_akshare_2024-2025.py)<br>[get_date_new.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/get_date_new.py)<br>[prediction_new.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_new.py) | **端到端数据闭环**：通过 AKShare 动态抓取真实股票行情、自动化清洗、多股票批量特征提取与实时推理。 | ★★★★☆ |
| **Stage 5: 交互界面与量化策略闭环** | [prediction_new_GUI.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_new_GUI.py)<br>[run_backtest_kronos.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/run_backtest_kronos.py) | **工业级量化应用**：<br>1. GUI 图形交互界面操作；<br>2. 结合 Kronos 预测信号的量化回测系统（包含仓位管理、夏普比率、最大回撤等回测评价体系）。 | ★★★★★ |

---

## 二、 基础篇：单序列预测与变体对比

### 1. [prediction_example.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_example.py) vs [prediction_wo_vol_example.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_wo_vol_example.py)

`wo_vol` 代表 **without volume（不包含成交量）**。

| 比较维度 | [prediction_example.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_example.py) | [prediction_wo_vol_example.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_wo_vol_example.py) |
| :--- | :--- | :--- |
| **输入特征** | 6 维完整特征：`open`, `high`, `low`, `close`, `volume`, `amount` | 4 维纯价格特征：`open`, `high`, `low`, `close` |
| **预测目标** | 同时预测未来的**价格走势**与**成交量** | 仅预测未来的**价格走势** |
| **底层实现** | 原生 6 维直接送入 Tokenizer 编码 | `KronosPredictor` 自动为缺失的 `volume/amount` 补 0 |
| **可视化图表** | **双子图**（上方收盘价对比，下方成交量对比） | **单子图**（仅绘制收盘价对比） |
| **适用场景** | 交易所股票/期货/ETF 等有真实量能的数据 | 外汇即期 (Forex Spot)、宏观经济指标、债券收益率，或过滤极端量能噪音 |

### 2. 底层容错机制
在 `KronosPredictor` 源码中（[kronos.py:L528-530](file:///Users/jyy/Documents/github_projects/Kronos/model/kronos.py#L528-L530)），框架会自动侦测输入列，并在缺失成交量时自动填充 0.0，因此即便只提供 OHLC 四价也不会产生形状不匹配错误：
```python
if self.vol_col not in df.columns:
    df[self.vol_col] = 0.0  # 自动填充为 0.0
    df[self.amt_vol] = 0.0  # 自动填充为 0.0
```

---

## 三、 核心调优技巧：提升预测精确度与稳定性

### 1. 为什么每次预测的曲线图形都不一样？
Kronos 是**生成式自回归概率模型（类似于 LLM 大语言模型）**：
1. **多项式概率采样**：在每一步预测时，模型通过 `sample_from_logits` 根据概率分布进行 `torch.multinomial(probs, num_samples=1)` 随机抽样；
2. **自回归蝴蝶效应**：未来 120 步中，第 $i$ 步微小的抽样差异会被当作真实历史反馈给第 $i+1$ 步，导致预测路径像分叉树枝一样演化出不同可能；
3. **蒙特卡洛金融意义**：这代表未来市场可能演化的不同平行路径（Scenario Simulation）。

### 2. 想要更加精确稳定，应该如何调优？

#### ① 缩短预测步长 `pred_len`（最关键）：
- **默认 120 步的代价**：对于 5 分钟 K 线，$120 \times 5 = 600$ 分钟，相当于 **2.5 个完整 A 股交易日**。跨 2.5 天逐根预测 K 线难度极大，且后面大部分输入是模型自己的“幻觉递归”。
- **优化建议**：将 `pred_len` 缩短到 **12 ~ 24**（未来 1 ~ 2 小时），95% 以上的历史上下文都是真实值，趋势判断的准确率会成倍提升。

#### ② 降低采样温度 `T = 0.2 ~ 0.3`：
- **原理**：调低温度系数 $T$ 会强化高置信度 Token 的比重，过滤长尾奇异噪音，走势从“毛刺乱飞”转为平滑确定的主趋势。
- **注意**：$T$ 不宜低于 0.1，过低会导致波动率坍缩（走势变平）。推荐设置在 `0.2 ~ 0.4` 之间。

#### ③ 固定随机种子（调试与完全复现）：
```python
import torch, numpy as np, random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(42)  # 固定种子后每次运行图形完全一致
```

---

## 四、 下一步进阶：批量推理与并发处理

👉 **[prediction_batch_example.py](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_batch_example.py)**

### 核心亮点：
1. **多样本并行**：单次调用 `predictor.predict()` 无法充分榨干 GPU/MPS 算力；使用 `predictor.predict_batch()` 可以将多个窗口/股票的输入打包成一个 Tensor 批次并发预测。
2. **数据切片准备**：
   - `df_list`: 多个历史切片 DataFrame 列表
   - `x_timestamp_list`: 对应的历史时间戳列表
   - `y_timestamp_list`: 对应的未来时间戳列表
3. **批量自回归生成**：内部实现 Batch Tokenization 与并行解码，大幅提升全市场选股与批量回测时的推理吞吐率。
