# Kronos 回测系统数据流与函数全流程详解

> 本文档通过统一的一个真实数据示例（A股创业板标的 `300418`），按照数据流的生命周期闭环，逐环节讲解 Kronos 回测系统的核心执行架构、涉及函数的作用、输入数据格式、内部计算逻辑与输出数据结构。

---

## 一、全局数据流拓扑图

```text
[原始行情 300418.csv]            [AI推演预测 300418_wf_predictions.csv]
          │                                        │
          ▼ (DataFetcher)                          ▼ (DataFetcher)
      hist_df                                   pred_df
          │                                        │
          └──────────────────┬─────────────────────┘
                             ▼
              [环节二: SkillEvaluator.evaluate] ──> AI前置体检字典
                             │
                             ▼
            [环节三: KronosStrategy.align_and_generate_signals]
                             │
                             ▼
                         signal_df (含 exec_signal)
                             │
                             ▼
              [环节四: RiskBudgetSizer.compute_size] ──> (买入股数, 止损线)
                             │
                             ▼
              [环节五: SimulatedBroker.execute_buy] ──> Position 持仓对象
                             │
                             ▼
              [环节六: ExitStateMachine.evaluate_eod_exit] ──> 离场预警
                             │
                             ▼
              [环节七: SimulatedBroker.execute_sell] ──> 资金结算
                             │
                             ▼
                backtest_df (逐日盯市) + trades (交易台账)
                             │
                             ▼
              [环节八: MetricsEvaluator.evaluate] ──> 策略量化绩效字典
```

---

## 二、统一数据示例背景设定 (Stage 0)

* **标的代码**：`300418`（创业板，涨跌幅限制 20%，一手 100 股整手，T+1 结算，一年按 244 个交易日计算）
* **账户初始资金**：`100,000.00` 元
* **交易摩擦**：双边佣金 0.03%（最低 5 元保底），卖方印花税 0.05%，撮合双向滑点 0.1%（`slippage_rate = 0.001`）
* **策略关键参数**：
  * 单笔风险预算：`risk_per_trade = 0.01`（总净值的 1.0%）
  * 单标的仓位上限：`max_position = 0.80`（80%）
  * 基准硬止损：`hard_stop_loss = 0.04`（4.0%）
  * 移动止盈回撤：`trailing_stop = 0.025`（2.5%），启动门槛 `trailing_stop_min_profit = 0.02`（2.0%）
  * 门槛筛选模式：**绝对 Z-Score 信息信噪比门槛**（`z_score_threshold = 0.70`，以无量纲信噪比作为唯一入场门槛，彻底摆脱 60 天滚动预热期与分位数局限）
  * 去偏校准模式：**反转感知自适应去偏**（`reversal_aware_debias = True`，`reversal_debias_weight = 0.0`，反转行情阻断滞后偏差对反转 Alpha 的误杀；顺势行情严格扣除系统性泡沫）
  * 波动率与 ATR 窗口：`volatility_window = 14` 天，`atr_window = 14` 天

---

## 三、数据流八大生命周期环节详解

### 环节一：数据获取与标准化清洗

#### 1. DataFetcher.load_stock_history
* **代码定义**：[backtest/data/data_fetcher.py (第78行)](./data/data_fetcher.py)
* **作用**：读取本地历史日 K 线 CSV 文件，统一转换中英文字段列名（如“收盘价”转为 `close`），按交易日升序排列。
* **输入数据**：
  * `stock_code = "300418"`
  * `data_dir = "backtest/historical_data"`
  * `market = "CN"`
  * `timeframe = TimeFrame.D1`
* **输出数据 (`hist_df`)**：时间戳为索引的 `pd.DataFrame`：
  ```text
  date (Index)  open   high    low  close     volume         amount
  2026-06-24   27.60  28.30  27.40  28.10  1420000.0  39902000.00
  ```

#### 2. DataFetcher.load_prediction_data
* **代码定义**：[backtest/data/data_fetcher.py (第169行)](./data/data_fetcher.py)
* **作用**：检索深度模型（如 Kronos 蒙特卡洛预测模型）输出的未来推演文件，完成特征重命名与粒子滤波元数据清洗。
* **输入数据**：
  * `stock_code = "300418"`
  * `model_dir = "backtest/yuce"`
  * `market = "CN"`
  * `particle_filter = True`
* **输出数据 (`pred_df`)**：时间戳为索引的 `pd.DataFrame`（由历史数据严格推演未来，不包含未来真实行情）：
  ```text
  date (Index)  pred_close  pred_path_max  optimal_horizon  pred_slope  pred_min_down  pf_confidence
  2026-06-25         28.92          29.80                3      +0.025         -0.008          0.760
  ```
  * `pred_close`：模型推演给出的加权入场期望价格（28.92 元）。
  * `pred_path_max`：推演路径中的峰值价（29.80 元）。
  * `optimal_horizon`：峰值预计在第 3 个交易日出现。
  * `pred_min_down`：途中预估最大下探仅 -0.8%。
  * `pf_confidence`：蒙特卡洛粒子滤波有效粒子数推导出的置信度（76.0%）。

---

### 环节二：前置 AI 预测能力体检

#### 3. SkillEvaluator.evaluate
* **代码定义**：[backtest/engine/skill_evaluator.py (第25行)](./engine/skill_evaluator.py)
* **作用**：在撮合交易前对模型预测数据进行“体检”，统计预测方向命中率、Rank IC（相关性）、波动幅度比等指标，避免使用无效模型导致假回测。
* **输入数据**：
  * `df`：历史与预测的重叠交集矩阵（含预测价格与真实价格）。
  * `hard_stop_loss = 0.04`
* **输出数据 (`skill` 字典)**：
  ```python
  {
      "ic": 0.082,                 # Spearman 秩相关系数
      "ic_t_stat": 2.15,           # t 检验值 > 2.0，统计显著
      "hit_rate": 0.583,           # 方向命中率 58.3% (> 50%)
      "amplitude_ratio": 1.08,     # 预测与真实波动幅度比
      "down_violation_rate": 0.112,# 下探违约率 11.2%
      "tradable": True             # 体检合格，准许进入回测
  }
  ```

---

### 环节三：时序对齐与九重入场否决矩阵

#### 4. KronosStrategy.align_and_generate_signals
* **代码定义**：[backtest/strategies/kronos_strategy.py (第97行)](./strategies/kronos_strategy.py)
* **作用**：
  1. **防偷价对齐**：使用 `shift(1)` 将 `t-1` 收盘时的指标严格对齐到 `t` 日决策。
  2. **特征计算**：计算历史波动率、ATR 真实波幅、**反转感知自适应去偏**与时界根号放缩信噪比得分 `signal_score`。
  3. **门槛与九重守门人风控过滤**：采用**绝对 Z-Score 信息信噪比门槛 (`z_score_threshold=0.70`)**，彻底消除了 60 天滚动分位“弱市拔矮子”和“强市限流”的弊端；对每个交易日审查 9 项规则，被拦截则将拦截原因写入 `veto_reason`；全部通过则发放买入许可。
* **输入数据**：
  * `hist_df`、`pred_df`、`profile = MarketProfile.for_market('CN')`
  * 参数：`z_score_threshold = 0.70`, `reversal_aware_debias = True`, `reversal_debias_weight = 0.0`
* **内部计算过程（以 2026-06-25 该行执行为例）**：
  ```text
  1. 昨收基准价: close_prev = 28.10 元
  2. 历史 14 日波动率: vol_14d = 0.0215 (2.15%)
  3. ATR 真实波幅: atr = 0.95 元, atr_pct = 0.95 / 28.10 = 0.0338 (3.38%)
  4. 趋势判断: MA20 (27.20) >= MA60 (25.80) -> 多头排列 (UPTREND)
  5. 原始预期收益率: expected_return = (28.92 - 28.10) / 28.10 = +2.92%
  6. 状态感知自适应去偏:
     - 当前处于多头顺势 (UPTREND 且看多)，严格扣除近期系统性高估偏差 +0.27%:
       expected_return_adj = +2.92% - (+0.27%) = +2.65%
     - 💡 对比提示：若处于下跌探底背景而模型预测向上反转 (is_bull_reversal)，
       系统将自动阻断历史熊市偏差扣减 (偏差衰减权重为 0.0)，保留原始反转预期！
  7. 根号时间放缩: 目标持仓 3 天 -> vol_scaled = 0.0215 * sqrt(3) = 0.03724
  8. 信噪比评分: signal_score = expected_return_adj / vol_scaled = +0.02648 / 0.03724 = +0.711
  9. 入场门槛: 绝对信噪比门槛 entry_threshold = 0.700 (0.711 > 0.700，达标且无需预热期)
  ```
* **九重审查流水线检验**：
  1. 门槛审查：`signal_score = 0.711 > entry_threshold = 0.700`（通过：超越绝对信噪比门槛 0.70，无 60 天预热期要求）
  2. 粒子不确定性：`pf_confidence = 0.76 >= 0.60`（通过：拟合稳定）
  3. 路径动量斜率：`pred_slope = +0.025 > 0`（通过：总体预期向上）
  4. T+N 波峰兑现：`target_horizon = 3 > 1`（通过：A股 T+1 下第 3 天波峰可卖出兑现）
  5. 宏观趋势过滤：`ma_fast (27.20) >= ma_slow (25.80)`（通过：处于中长期多头排列）
  6. 去偏预期回报：`expected_return_adj = +2.65% > 0`（通过：扣减系统性泡沫后仍为正预期；反转态下不被历史高估误杀）
  7. 下探深度与仓位缩减：`depth = 0.008 / 0.040 = 0.20 <= 1.0`，乘以粒子置信度 `0.760` -> `position_scale = 0.760`（通过）
  8. 粒子置信度硬门禁：`pf_confidence = 0.760 >= 0.35`（通过：高于 35% 红线）
  9. 开盘流动性：次日开盘非一字涨停（通过）
* **输出数据 (`signal_df` 的关键行)**：
  ```text
  date (Index): 2026-06-25
  close_prev: 28.10
  signal_score: 0.711
  entry_threshold: 0.700        # 绝对门槛 (加粒子微调)
  raw_signal: 1                 # 原始买入信号
  veto_reason: ""               # 审查全过，无否决原因
  exec_signal: 1                # 允许开盘执行买入
  exec_target_horizon: 3        # 目标持有 3 天
  exec_position_scale: 0.760    # 结合粒子滤波后的仓位缩放系数 (0.76)
  ```

#### 💡 核心量化机制演进对比速查

| 维度 | 传统机制（旧版） | 现行生产机制 | 量化优势 |
| :--- | :--- | :--- | :--- |
| **去偏机制** | 机械线性扣除过去 60 天滚动中位数偏差 | **状态感知自适应去偏** | 顺势严格挤泡沫；行情反转时阻断上一周期的滞后残差惩罚，大幅提前反转捕获时机（1~3天）。 |
| **入场门槛** | 自身过去 60 天 80% 滚动分位数自校准 | **绝对 Z-Score 信息信噪比门槛** | `signal_score > 0.70`。消除弱市“矮子拔将军”误开仓与强市“良机被限流”，彻底免除 60 天预热期。 |

---

### 环节四：机构级风险预算定仓

#### 5. RiskBudgetSizer.compute_size
* **代码定义**：[backtest/execution/sizer.py (第45行)](./execution/sizer.py)
* **作用**：不使用简单的固定百分比买入，而是**以能承受的亏损反推下单股数**：`下单股数 = 风险预算金额 / 单股止损空间`，并施加仓位上限与整手截断。
* **输入数据**：
  * `portfolio_value = 100,000.00` 元
  * `cash = 100,000.00` 元
  * `buy_price = 28.20 * (1 + 0.001) = 28.228` 元（开盘价叠加滑点）
  * `atr_pct = 0.0338`
  * `position_scale = 0.760`
  * `risk_per_trade = 0.01`（单笔最多亏 1%）
  * `hard_stop_loss = 0.04`（4.0% 止损）
* **内部推导计算**：
  ```text
  1. 自适应止损幅度: active_stop_pct = 4.0%
  2. 单股止损空间: stop_distance = 28.228 * 0.04 = 1.129 元
  3. 绝对止损价格: stop_price = 28.228 * (1 - 0.04) = 27.099 元
  4. 风险预算: risk_budget = 100,000 * 0.01 * 0.760 = 760.00 元
  5. 理论股数: raw_shares = 760.00 / 1.129 = 673.16 股
  6. 上限与整手截断: 
     - 80% 仓位限制允许: 2,834 股
     - A 股 100 股向下整手取整: int(673.16 // 100) * 100 = 600 股
  ```
* **输出数据（元组）**：
  ```python
  (
      shares_to_buy = 600,         # 最终买入 600 股
      active_stop_pct = 0.04,      # 生效止损比例 4.0%
      active_trail_pct = 0.025,    # 生效移动止盈回撤 2.5%
      stop_price = 27.099          # 触发止损价格 27.099 元
  )
  ```

---

### 环节五：仿真撮合买入与建仓

#### 6. SimulatedBroker.execute_buy
* **代码定义**：[backtest/execution/simulated_broker.py (第125行)](./execution/simulated_broker.py)
* **作用**：模拟交易所撮合，扣除股票货款与券商佣金，创建持仓对象 [Position](./core/types.py)，记录买入流水。
* **输入数据**：
  * `trade_date = pd.Timestamp("2026-06-25")`
  * `open_price = 28.20`
  * `shares_to_buy = 600`
  * `target_horizon = 3`
  * `stop_price = 27.099`
* **内部处理**：
  ```text
  1. 真实成交买价: buy_price = 28.20 * (1 + 0.001) = 28.228 元
  2. 股票成交金额: 600 * 28.228 = 16,936.80 元
  3. 佣金手续费: max(16,936.80 * 0.0003, 5.0) = 5.08 元
  4. 扣除现金: 16,936.80 + 5.08 = 16,941.88 元
  5. 剩余现金: 100,000.00 - 16,941.88 = 83,058.12 元
  ```
* **输出数据**：
  * 更新后的持仓对象 `position`：
    ```python
    Position(
        symbol="300418",
        shares=600,
        avg_entry_price=28.228,
        stop_price=27.099,
        target_horizon=3,
        holding_bars=0,
        peak_price=28.80
    )
    ```
  * `broker.trades` 新增一条买入流水：
    ```python
    {"date": "2026-06-25", "action": "BUY", "price": 28.228, "shares": 600, "commission": 5.08}
    ```

---

### 环节六：持仓状态机盯市与离场判定

#### 7. ExitStateMachine.evaluate_eod_exit
* **代码定义**：[backtest/execution/risk_manager.py (第96行)](./execution/risk_manager.py)
* **作用**：每个交易日收盘（EOD）运行状态机，逐项评估六重出场规则（硬止损、移动止盈、目标时界波峰止盈、动量跃迁、时间止损、波段走弱），决定次日开盘是否强制卖出。
* **输入数据**：
  * 当前日行情 `row`、持仓对象 `position`、市场规则画像 `profile`。
* **推演过程（持仓 3 天演进）**：
  * **Day 1 (2026-06-25)**：最低价 28.00 > 27.099（未止损）；收盘价 28.80；`holding_bars = 1`；无预警。
  * **Day 2 (2026-06-26)**：收盘价 29.50；浮盈达到 `+4.51%`（超过 2.0% 门槛，移动止盈观察被激活）；无预警。
  * **Day 3 (2026-06-27)**：收盘价 29.60；`holding_bars = 3`（**触达 3 天目标时界**）；成本浮盈 `+4.86%`，未见持续涨停爆发。
* **输出数据（元组）**：
  ```python
  (
      pending_close_reason = "预测波峰到达止盈(3日/3日目标,浮盈+4.86%)",
      pending_close_price = None  # None 表示次日开盘市价卖出
  )
  ```

---

### 环节七：次日撮合卖出与资金结算

#### 8. SimulatedBroker.execute_sell
* **代码定义**：[backtest/execution/simulated_broker.py (第56行)](./execution/simulated_broker.py)
* **作用**：次日开盘按市价叠加负滑点撮合卖出，扣除印花税与佣金，资金回笼重置持仓，将单笔胜负指标写入台账。
* **输入数据**：
  * `trade_date = pd.Timestamp("2026-06-30")`（Day 4 开盘）
  * `fill_price = 29.80`（开盘价）
  * `position = position`（当前持仓 600 股）
  * `reason = "预测波峰到达止盈(3日/3日目标,浮盈+4.86%)"`
* **内部处理与税费清算**：
  ```text
  1. 真实成交卖价: sell_price = 29.80 * (1 - 0.001) = 29.770 元
  2. 卖出毛金额: 600 * 29.770 = 17,862.00 元
  3. 佣金 (万三): 17,862.00 * 0.0003 = 5.36 元
  4. 印花税 (千分之五): 17,862.00 * 0.0005 = 8.93 元
  5. 合计税费: fee = 5.36 + 8.93 = 14.29 元
  6. 资金回笼: 83,058.12 + 17,862.00 - 14.29 = 100,905.83 元
  7. 单笔净盈亏: 
     - 净收益率: (29.770 - 28.228) / 28.228 = +5.46%
     - 净利润金额: 100,905.83 - 100,000.00 = +905.83 元
  ```
* **输出数据**：
  * `broker.cash = 100905.83`，持仓股数清零 `position.shares = 0`。
  * `broker.trades` 录入卖出结算详情：
    ```python
    {
        "date": "2026-06-30",
        "action": "SELL",
        "price": 29.770,
        "shares": 600,
        "commission": 14.29,
        "pnl_pct": 0.0546,
        "pnl_amt": 905.83,
        "holding_days": 3,
        "reason": "预测波峰到达止盈(3日/3日目标,浮盈+4.86%)",
        "cash_balance": 100905.83,
        "mfe_pct": 0.070,           # 最大潜在有利变动 (持仓期间最高价收益)
        "mae_pct": -0.008          # 最大潜在不利逆行 (持仓期间最低价跌幅)
    }
    ```

---

### 环节八：全维度量化绩效与风险归因

#### 9. MetricsEvaluator.evaluate
* **代码定义**：[backtest/engine/metrics.py (第29行)](./engine/metrics.py)
* **作用**：遍历逐日净值表 `backtest_df` 和交易流水 `trades`，校准不同市场（CN 244天/US 252天）的年化乘数，生成夏普、卡玛、索提诺、最大回撤等量化指标。
* **输入数据**：
  * `backtest_df`（包含每日总净值 `portfolio_value` 和收益率序列 `returns`）：
    ```text
    date (Index)  portfolio_value      cash  shares  close  returns
    2026-06-25         100338.12  83058.12     600  28.80  +0.00338
    2026-06-26         100758.12  83058.12     600  29.50  +0.00418
    2026-06-27         100818.12  83058.12     600  29.60  +0.00060
    2026-06-30         100905.83 100905.83       0  29.80  +0.00087
    ```
  * `trades`：交易台账列表。
  * `initial_capital = 100000.0`
* **内部计算核心**：
  ```text
  1. 累计收益率: (100,905.83 - 100,000) / 100,000 = +0.9058%
  2. 日频超额年化夏普:
     excess = returns - daily_rf
     sharpe = (mean(excess) / std(excess)) * sqrt(244) = 2.45
  3. 最大回撤: 净值持续创出新高，最大回撤 = 0.00%
  4. 胜率: 1 战 1 胜 = 100%
  ```
* **输出数据 (`metrics` 字典)**：
  ```python
  {
      "total_return": 0.009058,      # 累计收益率 0.91%
      "annual_return": 0.1250,       # 样本充分时的推演年化收益
      "sharpe_ratio": 2.45,          # 年化夏普比率
      "sortino_ratio": 3.82,         # 索提诺比率 (仅惩罚下行风险)
      "max_drawdown": 0.00,          # 最大回撤
      "win_rate": 1.00,              # 胜率 100%
      "total_trades": 1,             # 撮合总次数
      "profit_factor": 999.0,        # 盈亏比
      "avg_holding_days": 3.0        # 平均持仓天数
  }
  ```

---

## 四、数据结构流转全景表

| 阶段 | 核心数据结构 | 承载对象类型 | 关键字段 / 关键信息 | 产生函数 / 消费函数 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 历史** | `hist_df` | `pd.DataFrame` | `open`, `high`, `low`, `close`, `volume` | [DataFetcher.load_stock_history](./data/data_fetcher.py) |
| **2. 预测** | `pred_df` | `pd.DataFrame` | `pred_close`, `pred_path_max`, `optimal_horizon`, `pf_confidence` | [DataFetcher.load_prediction_data](./data/data_fetcher.py) |
| **3. 信号** | `signal_df` | `pd.DataFrame` | `signal_score`, `entry_threshold`, `exec_signal`, `veto_reason` | [KronosStrategy.align_and_generate_signals](./strategies/kronos_strategy.py) |
| **4. 定仓** | `sizer_out` | `tuple` | `shares_to_buy` (600), `active_stop_pct` (4%), `stop_price` | [RiskBudgetSizer.compute_size](./execution/sizer.py) |
| **5. 持仓** | `position` | [Position](./core/types.py) | `shares`, `avg_entry_price`, `stop_price`, `holding_bars` | [SimulatedBroker.execute_buy](./execution/simulated_broker.py) |
| **6. 净值** | `backtest_df` | `pd.DataFrame` | `portfolio_value`, `cash`, `shares`, `returns` | [KronosBacktesterPro.run_backtest_engine](./engine/backtest_engine.py) |
| **7. 台账** | `trades` | `List[dict]` | `action`, `price`, `pnl_pct`, `pnl_amt`, `mfe_pct`, `mae_pct` | [SimulatedBroker.execute_sell](./execution/simulated_broker.py) |
| **8. 评价** | `metrics` | `dict` | `sharpe_ratio`, `max_drawdown`, `total_return`, `win_rate` | [MetricsEvaluator.evaluate](./engine/metrics.py) |
