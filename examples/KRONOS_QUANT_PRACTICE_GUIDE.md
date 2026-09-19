# Kronos 金融时序大模型量化实战指南：参数调优、多周期建模与工程落地

> **编写视角**：量化投资专家 & AI 架构师  
> **适用范围**：Kronos 自回归时序模型在 A 股日线、分钟线预测，蒙特卡洛多路径集成，短中长线参数调优与实操交易。

---

## 目录
1. [模型特性与自回归机理](#一模型特性与自回归机理)
2. [多路径采样与置信区间原理（蒙特卡洛）](#二多路径采样与置信区间原理)
3. [周期场景与参数搭配矩阵（短线 / 中线 / 长线）](#三周期场景与参数搭配矩阵)
4. [黄金比例法则：Lookback 与 Pred_len 的协同](#四黄金比例法则lookback-与-pred_len-的协同)
5. [1～5 天超短线预测专题（分钟线落地实践）](#五15-天超短线预测专题分钟线落地实践)
6. [量化工程落地四大实操避坑指南](#六量化工程落地四大实操避坑指南)
7. [代码实装参考：自适应双层采样模板](#七代码实装参考自适应双层采样模板)

---

## 一、模型特性与自回归机理

Kronos 是基于 **离散分词器（Tokenizer） + 自回归 Transformer** 架构的时序基础大模型。

### 1. 核心数学机制
模型根据历史输入 $X_{1:t}$，自回归依次生成下一个时刻的 Token：
$$P(Y_{1:L} \mid X) = \prod_{\tau=1}^L P(y_\tau \mid X, y_1, \dots, y_{\tau-1})$$

* **自注意力机制优势**：擅长捕捉 **30 ~ 120 步** 的中长程依赖、技术形态与量价相关性。
* **自回归误差滚雪球（Exposure Bias）**：预测步数 $L$ 越长，喂给模型的“自身生成的数据”越多，自回归误差呈非线性放大。
* **方差 vs 偏差权衡**：
  * **低温度 ($T \le 0.3$)**：方差极小，抗噪声强，体现大趋势引力；但在长程推演下容易均值收敛、模式衰减。
  * **高温度 ($T \ge 0.8$)**：保留真实市场波动弹性；但长程推演下极易发散，产生极端离群值。

---

## 二、多路径采样与置信区间原理

在 [`prediction_akshare_2024-2025.py`](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_akshare_2024-2025.py) 中，系统支持蒙特卡洛多路径采样与扇形置信区间推导。

### 1. 为什么不建议把不同超参数混合在一起算简单均值？
* **统计学一致性**：蒙特卡洛置信区间的数学前提是从**同一个后验分布 $P(Y \mid X)$** 中进行独立同分布（i.i.d.）抽样。若将 $T=0.3$ 和 $T=1.0$ 的样本粗暴求算术平均，在统计学上失去了明确的置信带意义。
* **信号互相削弱**：低温度的清晰趋势容易被高温采样的随机游走噪声带偏。

### 2. 最佳设计：双层职责分离（主基准线 + 压力测试带）
* **核心主趋势线（Base Line）**：
  * 使用**低/中等温度**生成 1~3 条路径，取均值作为 `pred_df['close']`。
  * **作用**：代表模型确信度最高的主干趋势，用于判断多空大方向、买卖信号触发点。
* **波动置信区间（Risk Ribbon / 上下轨）**：
  * 使用**标准/偏高温度**运行 6~10 条蒙特卡洛采样，提取每天的 **P10（下界）** 和 **P90（上界）** 作为 `close_lower` 与 `close_upper`。
  * **作用**：进行压力测试，评估极端行情下的潜在振幅，用于设置防守止损位与挂单止盈位。

---

## 三、周期场景与参数搭配矩阵

不同交易跨度下，参数绝对不能“一刀切”。以下为量化实战推荐配置：

| 交易战术场景 | 目标预测天数 ($L$) | 建议回看天数 (Lookback) | 核心主趋势线 ($T_{base}, p_{base}$) | 波动上下轨 ($T_{risk}, p_{risk}$) | 采样路径数及分位数 | 核心量化逻辑 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **超短/波段**<br>(日K线) | **3 ~ 7 天** | 30 ~ 50 天<br>(约 1.5~2.5 个月) | **$T = 0.45 \sim 0.50$**<br>$top\_p = 0.85$ | **$T = 0.75 \sim 0.80$**<br>$top\_p = 0.90$ | 5 条<br>取 P15 / P85 | **保留多空博弈弹性。**<br>$T$ 不能过低，否则压制成单调直线，丢掉局部波浪与拐点；短线发散小，上下轨只需适度扩散。 |
| **中线波段趋势**<br>(日K线) | **15 ~ 30 天**<br>(约 1 个月) | 60 ~ 100 天<br>(约 3~5 个月，覆盖单季度财报) | **$T = 0.35 \sim 0.40$**<br>$top\_p = 0.80$ | **$T = 0.85 \sim 0.90$**<br>$top\_p = 0.90$ | 8 条<br>取 P10 / P90 | **Kronos 的黄金舒适区。**<br>步数适中，滤除日内杂音，推演箱体与通道走势极佳；上下轨模拟中期黑天鹅扰动。 |
| **中长线战略布局**<br>(日K线) | **45 ~ 60 天**<br>(约 1 个季度) | 120 ~ 180 天<br>(约半年至大半年，覆盖完整牛熊波段) | **$T = 0.25 \sim 0.30$**<br>$top\_p = 0.75$ | **$T = 0.80 \sim 0.85$**<br>$top\_p = 0.85$ | 10 条<br>取 P20 / P80 | **大势定性，严禁超高温。**<br>步数达 60 步，长线上下轨严禁使用 $T \ge 1.0$（否则 80 步后极端离群值会导致置信带撑大数倍而失效）。 |

> **⚠️ 量化红线提示**：在日线级别下，**强烈不建议单次硬性预测超过 60 个交易日（如直接预测 100 天）**。任何纯时序模型在 60 步后的绝对价格预测能力都会衰减为纯粹的统计先验惯性。

---

## 四、黄金比例法则：Lookback 与 Pred_len 的协同

### 1. 核心比例：$2:1 \le \text{Lookback} : \text{Pred\_len} \le 4:1$
* **为什么不能低于 2:1？**
  根据采样定理与波浪分析，要预测未来一个波段的走向，模型必须在历史中看到 **2~3 个完整的同级别波浪振荡**，才能准确定位当前的支撑阻力区间与动量衰减度。
* **为什么不能超过 4:1 或盲目追求 2~3 年长历史？**
  * **市场状态漂移（Regime Shift）**：A股政策、流动性、板块逻辑在 6~12 个月内就会彻底迭代，过远的历史数据会形成“数据污染”。
  * **注意力稀释（Attention Dilution）**：输入超过 250 根日线后，自注意力矩阵权重被过度平摊，导致模型对最近 5~10 天的异动突破钝化。

### 2. 个股属性自适应
* **大盘蓝筹/周期稳健股（低波动）**：惯性大、周期长，**适当放大 Lookback 至 120~150 天**。
* **科技/题材成长股（高波动）**：情绪驱动快、记忆周期短，**适当缩短 Lookback 至 30~50 天**，聚焦当期最新成交密集区。

---

## 五、1～5 天超短线预测专题（分钟线落地实践）

### 1. 为什么“日K线”做 1~5 天预测不理想？
1. **步长仅 1~5 步**：Transformer 注意力机制尚未充分展开；且离散 Tokenizer 存在量化步长误差，在微小涨跌中误差被放大。
2. **外生变量主导**：隔夜美股外盘、盘后利好/利空公告、集合竞价情绪驱动了次日 70% 的涨跌，纯时序日线无法捕捉突发消息。
3. **白噪声主导**：日频超短线极度接近随机游走（Random Walk）。

### 2. 破局之道：降维切入“15分钟 / 30分钟 K 线”
当切换为 **15 分钟 K 线** 时：
* 1 个交易日 = 16 根 K 线；
* **预测 1~3 天 = 预测未来 16 ~ 48 根 K 线；预测 5 天 = 80 根 K 线**。
* **优势**：完整落入 Kronos 的 40~80 步最佳推演区间，能够清晰刻画**早盘拉升、午后回踩、尾盘抢筹、量价背离**等日内微观技术形态。

### 3. 分钟级实战落地配置表

```python
# 专为 1~5 天（15分钟级别）定制的推荐实操配置
SHORT_TERM_MIN15_CONFIG = {
    "pred_days": 3,              # 推荐单次预测 2~3 个交易日（实战准确率最高）
    "pred_len": 48,              # 3天 * 16根 = 48 根 15分钟 K线
    "lookback": 192,             # 回看 12 个交易日（192 根 15分钟 K线，完整覆盖两周交易节律）
    
    # 核心主线（确定性走势中枢）
    "base_params": {
        "T": 0.38,
        "top_p": 0.85,
        "sample_count": 2
    },
    
    # 波动上下轨（日内高抛低吸与止盈止损边界）
    "risk_params": {
        "T": 0.80,
        "top_p": 0.90,
        "sample_count": 6,
        "percentile": (15, 85)
    }
}
```

---

## 六、量化工程落地四大实操避坑指南

### 1. 隔夜跳空（Overnight Gap）的物理断点
* **问题**：15:00 收盘到次日 09:30 开盘存在 18.5 小时停牌，跳空高开/低开是常态。
* **方案**：时间戳序列必须剔除休市时段，保持连续交易索引；在开盘第一根 Bar 的上下轨容忍度主动扩大 20%~30%。

### 2. 充分学习日内“U型波动率（U-Shape Volatility）”
* **问题**：早盘 09:30~10:00 剧烈波动，午盘 11:00~14:00 极度缩量。
* **方案**：分钟级回看期**必须 $\ge 10$ 个交易日**（即 160 根以上 15分钟线），确保模型注意力能够学习到每天固定时段的缩量与放量特征。

### 3. T+1 制度与可执行性状态机
* **问题**：模型可能预测上午买入、下午暴跌卖出，但 A 股当天买入无法卖出。
* **方案**：实盘交易执行器中必须内嵌 T+1 状态机：当天买入的头寸锁死平仓权限，直到次日开盘方可解锁止盈止损。

### 4. 双尺度协同架构（日线定方向 + 分钟线抓买卖点）
* **大周期（日线模型）**：判断未来 1~2 周的大趋势是处于上升通道还是破位寻底；
* **小周期（15分钟模型）**：在日线多头共振前提下，推演未来 1~3 天的分时回踩低吸点。

---

## 七、代码实装参考：自适应双层采样模板

可直接嵌入 [`prediction_akshare_2024-2025.py`](file:///Users/jyy/Documents/github_projects/Kronos/examples/prediction_akshare_2024-2025.py) 的参数调度器：

```python
import numpy as np
import pandas as pd

def get_adaptive_params(pred_len: int, timeframe: str = "day"):
    """
    根据预测步长与周期自动适配最优化采样超参数
    """
    if timeframe == "min15":
        return {
            "lookback": min(240, max(160, pred_len * 3)),
            "base_T": 0.38, "base_top_p": 0.85, "base_samples": 2,
            "risk_T": 0.80, "risk_top_p": 0.90, "risk_samples": 6,
            "percentiles": (15, 85)
        }
    
    # 日线级别
    if pred_len <= 15:       # 超短线波段
        return {
            "lookback": 45,
            "base_T": 0.48, "base_top_p": 0.85, "base_samples": 2,
            "risk_T": 0.78, "risk_top_p": 0.90, "risk_samples": 6,
            "percentiles": (15, 85)
        }
    elif pred_len <= 45:     # 中线趋势
        return {
            "lookback": 90,
            "base_T": 0.38, "base_top_p": 0.80, "base_samples": 3,
            "risk_T": 0.88, "risk_top_p": 0.90, "risk_samples": 8,
            "percentiles": (10, 90)
        }
    else:                    # 季度长线
        return {
            "lookback": 150,
            "base_T": 0.28, "base_top_p": 0.75, "base_samples": 3,
            "risk_T": 0.82, "risk_top_p": 0.85, "risk_samples": 10,
            "percentiles": (20, 80)
        }


def execute_dual_layer_prediction(predictor, x_df, x_timestamp, future_dates, pred_len, cfg):
    """
    执行双层自适应预测：核心趋势线 + 波动压力带
    """
    # 1. 核心主趋势推演 (Base Line)
    base_paths = predictor.predict_batch(
        df_list=[x_df] * cfg["base_samples"],
        x_timestamp_list=[x_timestamp] * cfg["base_samples"],
        y_timestamp_list=[pd.Series(future_dates)] * cfg["base_samples"],
        pred_len=pred_len,
        T=cfg["base_T"],
        top_p=cfg["base_top_p"],
        sample_count=1,
        verbose=False
    )
    
    # 2. 波动上下轨推演 (Risk Ribbon)
    risk_paths = predictor.predict_batch(
        df_list=[x_df] * cfg["risk_samples"],
        x_timestamp_list=[x_timestamp] * cfg["risk_samples"],
        y_timestamp_list=[pd.Series(future_dates)] * cfg["risk_samples"],
        pred_len=pred_len,
        T=cfg["risk_T"],
        top_p=cfg["risk_top_p"],
        sample_count=1,
        verbose=False
    )
    
    # 聚合特征矩阵
    base_close = np.stack([p['close'].values for p in base_paths], axis=0)
    risk_close = np.stack([p['close'].values for p in risk_paths], axis=0)
    
    p_low, p_high = cfg["percentiles"]
    c_upper = np.percentile(risk_close, p_high, axis=0)
    c_lower = np.percentile(risk_close, p_low, axis=0)
    
    pred_df = pd.DataFrame({
        'close': np.mean(base_close, axis=0),  # 核心趋势线
        'close_upper': c_upper,                # 压力上轨
        'close_lower': c_lower,                # 防守下轨
        'close_std': np.std(risk_close, axis=0)
    }, index=pd.Series(future_dates))
    
    return pred_df, risk_paths
```

---

## 八、总结与实操心法

1. **短线（1~5天）求精度**：放弃纯日线，拥抱 **15分钟/30分钟线**，让模型推演 30~80 步的微观波段。
2. **中线（15~30天）求通道**：日线黄金区，重点看突破与支撑，运用 **$T=0.38$ 主线 + 扇形置信带** 进行波段持股。
3. **长线（60天+）只定性**：绝对价格误差随自回归累积放大，只看大趋势引力方向，**切忌将远期价格当作精确止盈目标**。
4. **动态滚动（Rolling Walk-Forward）胜过死守一次性预测**：每日/每周期滚动更新最新 K 线，方为量化交易常胜之道。
