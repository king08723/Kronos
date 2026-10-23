# Kronos 信号与风控诊断矩阵数据结构 (signal_df) 说明书

> **物理存储路径**：`backtest/batch/{SYMBOL}/{SYMBOL}_signals.csv`（如 [AAPL_signals.csv](file:///Users/jyy/Documents/github_projects/Kronos/backtest/batch/AAPL/AAPL_signals.csv)）与 `backtest/output/{SYMBOL}_signals.csv`  
> **内存对象名称**：`signal_df` (`pandas.DataFrame`)  
> **产生函数**：[KronosStrategy.align_and_generate_signals](file:///Users/jyy/Documents/github_projects/Kronos/backtest/strategies/kronos_strategy.py#L97)  
> **核心消费模块**：
> - 预测能力前置体检：[SkillEvaluator.evaluate](file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/skill_evaluator.py#L24)
> - 回测撮合状态机：[KronosBacktesterPro.run_backtest_engine](file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/backtest_engine.py#L227)
> - 风险预算定仓：[RiskBudgetSizer.compute_size](file:///Users/jyy/Documents/github_projects/Kronos/backtest/execution/sizer.py#L45)

---

## 一、数据结构背景与核心定位

`signal_df` 是策略研究与模拟交易执行之间的核心枢纽。它将历史真实行情（`hist_df`）与模型滚动前瞻预测（`pred_df`）在时间维度上精准合并，衍生出波动率自适应特征、反转感知去偏特征、信噪比得分，并运行 **九重入场风控否决过滤器**，最终输出可执行的买卖指令矩阵。

其最核心的技术保障是：**严格时序对齐，彻底杜绝未来函数（Look-ahead Bias）**。  
所有技术指标与去偏偏差均采用 `shift(1)` 处理，确保在第 `t` 日决策时，模型只能使用截至 `t-1` 日收盘已确定的全部已知数据。

---

## 二、字段分类与架构体系总览

`signal_df` 包含 60 余个关键字段，逻辑上划分为 8 大模块：

| 模块类别 | 代表性字段 | 核心业务功能 |
| :--- | :--- | :--- |
| **1. 基础行情与衍生** | `open`, `close`, `high`, `low`, `volume`, `amount`, `close_prev`, `daily_return` | 记录真实 K 线价格与成交数据 |
| **2. 波动率与技术形态** | `volatility`, `atr`, `atr_pct`, `ma_fast`, `ma_slow` | 刻画历史价格离散度与长短期均线趋势 |
| **3. 预测序列继承** | `pred_close`, `pred_expected`, `pred_path_max`, `pred_min_down`, `pred_slope`, `optimal_horizon` | 继承自 `pred_df` 的 AI 推演特征 |
| **4. 动态校准与时界** | `target_horizon`, `ann_vol`, `dyn_calib_window` | 动态伸缩窗口，高波收缩、低波放宽 |
| **5. 收益测算与自适应去偏**| `expected_return`, `target_return`, `pred_bias`, `expected_return_adj`, `is_reversal` | 剥离系统性预测偏置，反转态智能豁免 |
| **6. 信号评分与风控门槛** | `volatility_scaled`, `signal_score`, `entry_threshold` | 无量纲信息信噪比打分（Z-Score） |
| **7. 交易状态与趋势确认** | `open_limit_up`, `open_limit_down`, `confirm`, `regime` | 开盘涨跌停可交易性与均线短期确认 |
| **8. 决策输出与执行映射** | `position_scale`, `raw_signal`, `veto_reason`, `exec_signal`, `exec_position_scale` | 九重否决过滤后的最终开平仓决策 |

---

## 三、各模块关键字段深度解析

### 1. 基础行情与技术形态模块
- `close_prev`：`t-1` 日真实收盘价（`close.shift(1)`），作为计算当日所有涨跌幅与预测收益率的统一基准。
- `volatility`：截至 `t-1` 日的滚动收盘对数收益率标准差（通常 14 天窗口）。
- `atr` / `atr_pct`：真实波幅（Average True Range）及其占前收盘价的百分比（`atr / close_prev`），用于动态防守止损线设定。
- `ma_fast` / `ma_slow`：快速均线（20日）与慢速均线（60日），全部滞后一期（`shift(1)`），用于判定中期市场环境（`regime`）。

### 2. 波动率自适应校准模块
- `ann_vol`：年化波动率（`volatility * sqrt(252)`）。
- `dyn_calib_window`：**波动率自适应校准窗口**。在高波动市场（如年化波动率超过 45%）时，历史偏差的参考价值衰减加快，窗口从基准 45 天平滑收缩至 15 天；在低波平稳市场，窗口平滑扩展以积累更多样本。

### 3. 预期收益与状态感知去偏（Reversal-Aware De-biasing）模块
- `expected_return`：原始预测期望收益率：
  ```text
  expected_return = (pred_expected - close_prev) / close_prev
  ```
- `target_return`：潜在波峰回报率：
  ```text
  target_return = (pred_path_max - close_prev) / close_prev
  ```
- `pred_bias`：模型历史预测误差的滚动中位数（经时界扩展）：
  ```text
  err_1d = ((pred_close_1d - close_prev) / close_prev) - realized_1d
  pred_bias = rolling_median(err_1d, dyn_calib_window) * sqrt(target_horizon)
  ```
- `is_bull_reversal` / `is_bear_reversal`：**反转感知状态标记**：
  - 底部反转（`is_bull_reversal`）：在均线弱势（`close_prev < ma_fast`）背景下，模型预测向上反转且 `pred_slope > 0`；
  - 顶部反转（`is_bear_reversal`）：在均线强势背景下，模型预测拐头向下且 `pred_slope < 0`。
- `pred_bias_effective`：有效扣除偏差。在发生反转时，历史同向偏差是由上一轮单边走势形成的“滞后误差”；系统将偏差衰减权重置为 0，**阻断历史正偏差的机械扣除**，防止将真实的反转 Alpha 误杀。
- `expected_return_adj`：去偏后的真实无偏预期收益：
  ```text
  expected_return_adj = expected_return - pred_bias_effective
  ```

### 4. 信号评分与信噪比（Z-Score）模块
- `volatility_scaled`：时界扩展波动率（不确定性分母）：
  ```text
  volatility_scaled = volatility * sqrt(target_horizon)
  ```
- `signal_score`：**无量纲信息信噪比评分（Z-Score）**：
  ```text
  signal_score = expected_return_adj / volatility_scaled
  ```
  衡量单位波动风险下的超额预期收益，彻底消除传统滚动分位数在弱市“矮子拔将军”和强市“良机限流”的缺陷。
- `entry_threshold`：动态入场门槛。基准通常为 `0.70`；当粒子滤波置信度 `pf_confidence < 0.6` 时，门槛动态上浮增加安全边际：
  ```text
  entry_threshold = base_thresh + max(0.0, (0.6 - pf_confidence) * 0.4)
  ```

### 5. 动态仓位系数（Position Scale）
- `position_scale`：介于 `0.0 ~ 1.0` 之间，由预测最大下探与粒子滤波置信度双重决定：
  1. **下探深度惩罚**：若 `pred_min_down` 超过硬止损（如 4%），仓位平滑缩水；若达到止损线的 2 倍以上，系数归零。
  2. **粒子置信度调节**：直接乘以 `pf_confidence` 截断值。

---

## 四、九重入场风控否决机制（Veto Reasons）深度剖析

在生成买入信号时，系统对每一行候选信号进行严密的九重串联风控拦截。任何一项未达标，买入指令即被一票否决，原因记录在 `veto_reason` 列中：

| 否决代码/文本 | 触发条件 | 量化设计原理与防范风险 |
| :--- | :--- | :--- |
| **信噪比未达标** | `signal_score <= entry_threshold` | 预期超额收益无法覆盖波动风险，防止在噪音中频繁交易 |
| **预测斜率向下** | `pred_slope <= 0` | 即使点位略有微涨，但整体拟合趋势向下，杜绝逆势摸顶买入 |
| **时界<=1日无法离场** | `target_horizon <= 1`（T+1 市场） | 在 A 股等非 T+0 市场，买入当日无法卖出，波峰无法兑现 |
| **下跌趋势且无短期确认** | `ma_fast < ma_slow` 且 `close_prev <= ma_fast` | 避免在右侧单边下跌主跌浪中盲目接飞刀 |
| **去偏后预期收益为负** | `expected_return_adj <= 0` | 原始预测虽然为正，但扣除系统性乐观偏差后无利可图 |
| **预测下探过深或置信度过低** | `position_scale <= 0` | 预测最大亏损空间过大（超 2 倍止损），风险预算不允许暴露 |
| **粒子滤波置信度过低** | `pf_confidence < pf_confidence_threshold` | 模型对近期真实行情跟踪严重失真，模型失去可信度 |
| **开盘封涨停无法买入** | `open_limit_up == 1` | 撮合物理不可行性风控，杜绝不可成交的虚假收益 |
| **仓位预算或账户资金不足** | `shares_to_buy < lot_size` | 风险预算分配后不足 1 股/1 手，无法发起交易 |

---

## 五、执行映射（Execution Mapping）与防未来函数闭环

```text
[截至 t-1 日收盘数据 + t-1 日生成的 pred_df]
                    │
                    ▼
  KronosStrategy 运算衍生特征与九重风控检查
                    │
                    ├─ raw_signal (原始开仓意图: 0 或 1)
                    ├─ veto_reason (若未通过，写入明确的否决理由)
                    └─ position_scale (仓位控制系数)
                    │
                    ▼ 执行映射 (写入 signal_df)
  df['exec_signal'] = df['raw_signal'].astype(int)
  df['exec_target_horizon'] = df['target_horizon'].astype(int)
  df['exec_position_scale'] = df['position_scale'].astype(float)
  df['execution_lag'] = 0
                    │
                    ▼ 模拟撮合层消费 (BacktestEngine)
  在第 t 日以 open_price (开盘价) 撮合买入，使用 exec_position_scale 定量
```

通过这一机制：
1. 信号诊断表中的每一行，真实记录了历史每一天策略的“思考过程”与“风控诊断”；
2. 即使未开仓，也能从 `veto_reason` 明确获知是由于信噪比不足、趋势不佳、还是下探过深，为策略调优提供了全量可解释性支持。
