# Kronos 批量量化回测系统数据流与程序代码全流程详解

> 本文档详细解析运行 [backtest/run_backtest_batch.py](./run_backtest_batch.py) 时，批量任务调度与单只股票回测引擎的完整执行流程。
> 按照数据流转顺序，从**股票池解析输入**、**各股票独立模拟撮合回测**，一直到**多市场绩效排行榜生成与结果文件保存**的全过程，结合项目真实回测日志（以美股标的 **AAPL / 苹果公司** 为实战案例），通俗易懂地讲解每个环节的函数功能、参数含义、输入输出数据结构以及生成文件的对应关系。

---

## 一、双层数据流整体流程图

运行 `python backtest/run_backtest_batch.py` 实际上包含两层核心流程：**外层批量调度流程（Batch Orchestration）** 与 **内层单只股票回测流程（Single Stock Pipeline）**。

### 1. 批量调度总体流程图 (`run_backtest_batch.py`)

```text
[用户输入/股票池配置] (CLI 参数 / --stock-file / --stocks-inline / 默认股票池)
            │
            ▼ 阶段 0: 股票池规范化解析 (parse_stock_item + SymbolResolver)
     stock_pool 规范化列表 [{'stock_code', 'stock_name', 'market'}, ...]
            │
            ▼ 阶段 1: [1/4] 并发准备历史行情 (BatchBacktester.prepare_data_pool)
     ThreadPoolExecutor 多线程并发拉取历史 K 线 -> 保存到 backtest/historical_data/{code}.csv
     数据天数充足性检查: len(hist_df) >= lookback + test_days
            │
            ▼ 产出: valid_data_pool (检查合格的股票池字典)
            │
            ▼ 阶段 2: [2/4] 批量回测调度器 (BatchBacktester.run_batch_backtest)
     串行 / 多线程并行调度 (--parallel-stocks N) ──> 执行 BatchBacktester.run_single_stock
     (深度学习模型预测阶段使用互斥锁 _INFERENCE_LOCK 排队，避免 GPU 冲突，已有预测文件则直接复用)
            │
   ┌────────┴────────────────────────┬────────────────────────┐
   ▼ 标的 1 (AAPL 美股)              ▼ 标的 2 (MSFT 美股)     ▼ 标的 N (TSLA/300418)
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│ 单只股票回测流程 (详见下图) │ │ 单只股票回测流程 (详见下图) │ │ 单只股票回测流程 (详见下图) │
│ 保存到 batch/{code}/ 目录 │ │ 保存到 batch/{code}/ 目录 │ │ 保存到 batch/{code}/ 目录 │
└───────────────────────┘ └───────────────────────┘ └───────────────────────┘
            │                                 │                        │
            └─────────────────────────────────┼────────────────────────┘
                                              ▼
            ▼ 阶段 3: [3/4] 绩效排行榜聚合汇总 (BatchBacktester.generate_ranking_summary)
     收集所有标的 metrics 与 skill ──> 多指标量化排序 (年化/夏普/回撤/胜率)
     保存结果: backtest/batch/batch_backtest_ranking.csv
     控制台打印: 全量大盘绩效排行榜 + A股/港股/美股分盘统计
            │
            ▼ 阶段 4: [4/4] 回测图表与明细汇总 (BatchBacktester.export_visualizations_summary)
     终端汇总打印各标的图表与报表输出目录
```

---

### 2. 单只股票量化回测执行流程 (`run_single_stock`)

> **重要执行顺序提示**：在真实回测内核代码中，必须**先**执行时间对齐与特征计算（`align_and_generate_signals`，使用 `shift(1)` 防止提前偷看未来价格），将历史真实价格与模型预测价格合并生成 `signal_df`，**随后**才能使用 `signal_df` 进行前置 AI 预测能力体检（`evaluate_prediction_skill`）。

```text
[历史日K线 backtest/historical_data/AAPL.csv]
          │
          ▼ 环节一: KronosDataFetcher.fetch_history
      hist_df (真实历史日 K 线)
          │
          ▼ 环节二: KronosDataFetcher.walk_forward_predict (仅 --auto 模式且无本地缓存时推演)
      pred_df (滚动前瞻预测数据) ──> 保存到 backtest/yuce/AAPL_wf_predictions.csv
          │
          ▼ 环节三: KronosBacktesterPro.set_market
      根据标的市场 (US) 初始化交易规则配置 MarketProfile ($50,000/1股起/T+0/252天)，配置专属独立日志
          │
          ▼ 环节四: KronosStrategy.align_and_generate_signals (时间对齐防未来数据 + 九项入场风控检查)
      signal_df (包含 shifted 特征、signal_score、门槛值、exec_signal、veto_reason)
          │
          ▼ 环节五: SkillEvaluator.evaluate (AI 预测能力体检，--require-skill 准入门槛)
      skill 字典 (Rank IC、方向命中率、波动幅度比、下探违约率、tradable 标志)
          │
          ▼ 环节六: KronosBacktesterPro.run_backtest_engine (逐日模拟撮合状态机)
          │   每日开盘循环:
          │   ├─ A. 执行上一交易日平仓指令 (SimulatedBroker.execute_sell)
          │   ├─ B. 信号触发开仓买入:
          │   │      ├─ RiskBudgetSizer.compute_size (基于风险预算计算买入股数 + 仓位上限 + 整手取整)
          │   │      └─ SimulatedBroker.execute_buy (滑点撮合、扣款、建立持仓 Position)
          │   ├─ C. 持仓每日跟踪与离场判定:
          │   │      ├─ ExitStateMachine.check_intraday_exit (盘中跌破止损线离场)
          │   │      └─ ExitStateMachine.evaluate_eod_exit (收盘波峰止盈/移动止盈/持仓超时离场)
          │   └─ D. 每日结算总资产净值 portfolio_value
          │
          ▼ 产出: backtest_df (每日资产净值表) + trades (逐笔成交台账)
          │
          ▼ 环节七: MetricsEvaluator.evaluate (量化绩效指标计算)
      metrics 字典 (累计收益率、年化收益率、夏普比率、卡玛比率、最大回撤、胜率等)
          │
          ▼ 环节八: KronosBacktesterPro.export_backtest_artifacts (保存回测明细与指标文件)
      在 backtest/batch/AAPL/ 目录下保存 5 个结果文件:
      ├─ AAPL_trades.csv (逐笔交易明细表)
      ├─ AAPL_daily_equity.csv / AAPL_daily_backtest.csv (每日净值与仓位表)
      ├─ AAPL_signals.csv (逐日预测与风控诊断明细表)
      └─ AAPL_metrics.json (绩效指标与参数配置快照)
          │
          ▼ 环节九: plot_backtest_results (生成可视化回测图表)
      保存图表: backtest/batch/AAPL/AAPL_backtest_pro.png
```

---

## 二、统一数据示例背景设定 (美股 AAPL 实战背景)

根据项目实际批量回测运行日志 [backtest/batch/AAPL/AAPL_backtest.log](./batch/AAPL/AAPL_backtest.log) 与结果快照，基准配置如下：

* **标的代码**：`AAPL`（苹果公司，美股 NASDAQ，无涨跌幅限制，以 1 股为单位，T+0 结算，一年按 252 个交易日计算）
* **账户初始资金**：`$50,000.00` 美元（按美股市场规则自动配置）
* **交易成本与摩擦**：双边佣金 0.00%（最低 $0），卖方印花税 0.00%，买卖滑点 0.05% (`slippage_rate = 0.0005`)，卖出仅按实际扣除 SEC/FINRA 极低规费（如单笔约 $0.36 ~ $0.50）
* **策略关键参数**：
  * 回测区间：`120` 个交易日（`2026-04-24 ~ 2026-10-14`）
  * 单笔风险预算：`risk_per_trade = 0.01`（总净值的 1.0%）
  * 单只股票仓位上限：`max_position = 0.80`（80%）
  * 基准硬止损：`hard_stop_loss = 0.04`（4.0%）
  * 移动止盈回撤：`trailing_stop = 0.025`（2.5%），启动门槛 `trailing_stop_min_profit = 0.02`（2.0%）
  * 信号入场门槛：**Z-Score 预测信噪比门槛**（`z_score_threshold = 0.70`，要求预测收益与波动率之比达到该标准才允许买入）
  * 波动率与 ATR 计算周期：`volatility_window = 14` 天，`atr_window = 14` 天

---

## 三、批量调度 4 个主要阶段详解 (`run_backtest_batch.py`)

### 阶段 0：股票池解析与标准化配置

* **核心函数**：[parse_stock_item (第89行)](./run_backtest_batch.py#L89) 与 [load_stock_pool_from_file (第105行)](./run_backtest_batch.py#L105)
* **执行逻辑**：
  1. 接收命令行传入的个股代码（如 `AAPL`、`MSFT`、`TSLA`）、行内字符串（`--stocks-inline`）或文件（`--stock-file`）。
  2. 调用 [SymbolResolver.resolve](./core/market_rules.py#L45) 进行代码正则与后缀识别：
     * 纯英文字母代码（如 `AAPL`, `MSFT`）自动识别为 `US`（美股），设置结算币种为 `USD ($)`，最小交易单位为 1 股，T+0 交易机制，年化交易日 252 天；
     * 6 位纯数字识别为 `CN`（A股），设置货币为 `CNY (¥)`，每手 100 股，T+1 交易机制；
     * 5 位数字或 `.HK` 识别为 `HK`（港股），设置货币为 `HKD (HK$)`。
* **输出数据**：
  ```python
  [
      {"stock_code": "AAPL", "stock_name": "苹果公司", "market": "US"},
      {"stock_code": "MSFT", "stock_name": "微软",     "market": "US"},
      {"stock_code": "TSLA", "stock_name": "特斯拉",   "market": "US"},
      {"stock_code": "NVDA", "stock_name": "英伟达",   "market": "US"},
  ]
  ```

---

### 阶段 1：[1/4] 并发准备历史行情

* **核心函数**：[BatchBacktester.prepare_data_pool (第163行)](./run_backtest_batch.py#L163)
* **执行逻辑**：
  1. 创建最多 6 个并发线程的线程池（`max_workers=6`），并发调用 [fetcher.fetch_history](./data/kronos_predictor.py#L90)。
  2. 优先检查本地 `backtest/historical_data/{stock_code}.csv` 是否存在且数据充足；若不存在或已过期（`force_refresh=True`），则调用数据接口重新下载。
  3. **数据天数充足性检查**：
     计算所需最少交易日数 `min_days = lookback + test_days`（例如 90 + 120 = 210 个交易日）。若某只股票的历史日 K 线少于 `min_days`，系统会在控制台输出警告并将其移入 `failed` 失败列表，避免后续因为历史数据过短导致计算报错。
* **输出数据**：
  * `valid`：通过检查的股票字典，包含每只股票的 `hist_df` 数据与总交易日数。
  * 控制台输出：`✅ 数据就绪: 合格 4 只, 过滤 0 只`。

---

### 阶段 2：[2/4] 批量回测调度与异常容错处理

* **核心函数**：[BatchBacktester.run_batch_backtest (第367行)](./run_backtest_batch.py#L367)
* **调度机制**：
  1. 支持单线程顺序执行（`--parallel-stocks 1`）或多线程并发执行（`--parallel-stocks N`）。
  2. **显卡互斥排队机制**：
     针对 Mac Apple Silicon (MPS/Metal) 在 PyTorch 多线程并发时容易出现驱动底层冲突（Metal Assertion Failure）的问题，程序在 [KronosDataFetcher.walk_forward_predict](./data/kronos_predictor.py#L23) 中加入了互斥锁 `_INFERENCE_LOCK = threading.Lock()`。
     并发回测时，深度学习模型推理阶段会自动排队执行；当预测完成后，后续的信号计算、模拟撮合与绩效统计则会在各自线程中全速并发运行。
  3. **单股票异常容错设计**：
     如果某只股票因为数据缺失、体检未通过或参数问题抛出异常，外层 `try...except` 会捕获该异常，调用 `_fail_result()` 将其标记为 `FAILED`，记录错误原因后继续回测下一只股票，绝不会因为单只股票出错导致整个批量任务中断退出。

---

### 阶段 3：[3/4] 绩效排行榜聚合汇总

* **核心函数**：[BatchBacktester.generate_ranking_summary (第466行)](./run_backtest_batch.py#L466)
* **执行逻辑**：
  1. 遍历所有股票回测返回的 `metrics` 与 `skill` 结果字典。
  2. 汇总提取期末资产、累计收益率、年化收益率、夏普比率、卡玛比率、最大回撤、胜率、交易笔数等关键指标。
  3. 按照**年化收益率(%)从高到低排序**，重新生成排名序号（1, 2, 3...）。
  4. 自动保存为汇总 CSV 文件：`backtest/batch/batch_backtest_ranking.csv`。
* **实际运行生成的排行榜示例**：
  ```text
  排名  股票代码  股票名称  市场  结算币种   期末资产    回测状态  累计收益(%)  年化收益(%)  夏普比率  卡玛比率  最大回撤(%)  胜率(%)  交易笔数  体检建议
  1     MSFT      微软      US    USD        $51,772.11  成功      +3.54%       +7.59%        0.66      5.07      -1.50%        66.7%   3         需谨慎 🟡
  2     AAPL      苹果公司  US    USD        $50,994.14  成功      +1.99%       +4.22%       -0.17     14.18      -0.30%       100.0%   2         需谨慎 🟡
  3     TSLA      特斯拉    US    USD        $50,723.63  成功      +1.45%       +3.06%       -0.59      2.66      -1.15%        80.0%   5         需谨慎 🟡
  4     NVDA      英伟达    US    USD        $50,000.00  成功       0.00%        0.00%        0.00       N/A       0.00%         0.0%   0         需谨慎 🟡
  ```

---

### 阶段 4：[4/4] 回测图表与明细汇总

* **核心函数**：[BatchBacktester.export_visualizations_summary (第552行)](./run_backtest_batch.py#L552)
* **执行逻辑**：
  在控制台汇总打印所有成功回测股票的结果输出目录，方便直接找到每只股票的报表和图表：
  ```text
  🎨 [4/4] 回测图表与明细汇总
     📊 [US] 苹果公司(AAPL) -> /Users/jyy/Documents/github_projects/Kronos/backtest/batch/AAPL
     📊 [US] 微软(MSFT)     -> /Users/jyy/Documents/github_projects/Kronos/backtest/batch/MSFT
  ```

---

## 四、单只股票回测的 9 个核心执行环节详解（以美股 AAPL 为例）

> 当 `BatchBacktester` 执行到单只股票时，调用的核心方法为 [run_single_stock (第224行)](./run_backtest_batch.py#L224)。以下严格依据实际回测日志 [AAPL_backtest.log](./batch/AAPL/AAPL_backtest.log) 与真实交易数据，通俗剖析其 9 个执行步骤的具体过程。

### 环节一：历史行情读取与标准化清洗

* **代码位置**：[backtest/data/data_fetcher.py (第78行)](./data/data_fetcher.py#L78)
* **调用函数**：`DataFetcher.load_stock_history(stock_code, data_dir, market)`
* **输入参数**：
  * `stock_code = "AAPL"`
  * `data_dir = "backtest/historical_data"`
  * `market = "US"`
* **内部处理**：读取 CSV 文件，将列名统一转为标准小写英文（`open`, `high`, `low`, `close`, `volume`），过滤停牌或缺失的无效行，并转换为按交易日期升序排列的 `pd.DataFrame`。
* **输出数据 (`hist_df`) 真实片段**：
  ```text
  date (Index)    open    high     low   close      volume         amount
  2026-04-30    270.01  275.50  267.66  270.86  91848230.0  24878011577.80
  2026-05-01    278.36  286.70  277.87  279.64  79915442.0  22347554200.88
  ```

---

### 环节二：Walk-Forward 滚动前瞻预测与结果保存

* **代码位置**：[backtest/data/kronos_predictor.py (第135行)](./data/kronos_predictor.py#L135)
* **调用函数**：`KronosDataFetcher.walk_forward_predict(...)`
* **触发条件**：设置了 `auto_predict=True` 且本地文件 `backtest/yuce/AAPL_wf_predictions.csv` 不存在时。
* **输入参数**：
  * `hist_df`：完整历史日 K 线数据
  * `test_days = 120`：回测验证的交易天数
  * `lookback = 90`：每次预测参考的历史窗口天数（对齐 Kronos 官方微调基准）
  * `sample_count = 20`：蒙特卡洛预测路径采样数
* **预测机制与双重检查锁 (DCL)**：
  1. **检查本地缓存**：首先检查 `backtest/yuce/` 目录下是否已有现成的预测文件，如果有就直接读取，省去耗时的重复模型计算；
  2. **加锁排队推演**：如果本地还没有预测文件，则获取互斥锁 `_INFERENCE_LOCK`，二次确认文件仍不存在后，排队调用深度学习模型，避免多线程并发抢占 GPU 导致崩溃；
  3. **逐日滚动预测**：在 `test_days` 窗口内的每一个交易日，截取过去 90 天的历史数据作为输入，预测未来几天的价格走势，并通过粒子滤波计算置信度与预期涨幅高点；
  4. **保存预测结果**：将预测数据保存到 `backtest/yuce/AAPL_wf_predictions.csv`，后续回测可以直接复用该文件。
* **输出数据 (`pred_df`) 真实片段 (对齐 2026-05-01 入场点)**：
  ```text
  date (Index)  pred_close  pred_path_max  optimal_horizon  pred_slope  pred_min_down  pf_confidence
  2026-05-01        278.08         281.25                5     +0.0014         -0.052          1.000
  ```

---

### 环节三：市场规则初始化与独立日志配置

* **代码位置**：[backtest/engine/backtest_engine.py (第108行)](./engine/backtest_engine.py#L108)
* **调用函数**：`KronosBacktesterPro.set_market(stock_code, market, log_dir, enable_log)`
* **内部处理**：
  1. 调用 [MarketProfile.for_stock](./core/market_rules.py#L140) 设置美股交易规则（每手 1 股起，佣金 0.00%，卖方印花税 0.00%，T+0 结算，无涨跌停限制，初始资金 `$50,000.00` 美元，年化交易日 252 天）。
  2. 释放上一只股票的日志处理器，调用 [attach_stock_logger](./core/logger.py#L65) 把当前股票的日志输出到独立文件 `backtest/batch/AAPL/AAPL_backtest.log`，保证每只股票的日志单独记录、互不干扰。
* **日志输出**：
  ```text
  [AAPL] [市场规则] AAPL -> US/us (USD) | 1股起 | 佣金0.0000%(最低$0) | 卖方印花税0.000% | T+0 | 无涨跌停 | 年化交易日252
  ```

---

### 环节四：时间对齐（防止使用未来数据）与九项入场风控检查

* **代码位置**：[backtest/strategies/kronos_strategy.py (第97行)](./strategies/kronos_strategy.py#L97)
* **调用函数**：`KronosStrategy.align_and_generate_signals(hist_df, pred_df, ...)`
* **输入参数**：
  * `hist_df`、`pred_df`
  * `z_score_threshold = 0.70`（预测收益信噪比门槛）
  * `reversal_aware_debias = True`（反转感知自适应去偏模式）：
    * **通俗大白话**：“**顺势时挤干水分防忽悠，拐点时网开一面抓反转**”。
    * **为什么需要它？** 
      1. *平时的去偏逻辑*：AI 模型有时容易“盲目乐观”，系统平时会统计模型过去吹了多少牛皮（历史高估偏差），并在计算预期收益时把这部分水分挤掉（预期收益减去历史偏差），防止盲目追高；
      2. *传统死板去偏的痛点*：如果一只股票连跌了两个月，系统会累积出很大的正向偏差（认为模型过去一直过度乐观）。当**真正的大底出现、行情即将暴涨反转**时，模型敏锐抓住了底部拐点预测上涨 +2.5%，但死板的去偏逻辑会强行扣掉过去两个月累积的 3% 偏差，算出来变成 -0.5%，硬生生把一个绝佳的黄金抄底机会给扼杀否决了；
      3. *开启本参数的作用*：当系统检测到底部均线弱势但模型预测强势抬头时，会立刻识别出“当前正在发生绝地反击”，**自动豁免或暂停扣除历史偏差**，完整保留珍贵的反转爆发 Alpha，彻底解决抄底买点滞后或被误杀的问题。
* **内部计算过程（以 2026-05-01 实际买入决策为例）**：
  ```text
  1. 避免未来数据泄露: 将 2026-04-30 收盘后产出的预测指标向后平移一天（使用 shift(1)），对齐到 2026-05-01 开盘，确保决策只使用开盘前已有的信息
  2. 基准昨收价: close_prev = 270.86 美元
  3. 历史波动率: vol_14d = 0.01615 (1.62%)
  4. 趋势判断: MA20 (264.67) >= MA60 (261.87) -> 多头排列 (UPTREND)
  5. 原始预期收益率: expected_return = (278.08 - 270.86) / 270.86 = +2.67%
  6. 去偏校准: 顺势运行且无过度泡沫，预期收益保持 expected_return_adj = +2.67%
  7. 根号时间放缩: 目标持仓 5 天 -> vol_scaled = 0.01615 * sqrt(5) = 0.03611
  8. 信噪比评分: signal_score = 0.02667 / 0.03611 = +0.7386
  9. 入场门槛检验: 设定门槛 entry_threshold = 0.700 (0.7386 > 0.700，信噪比达标)
  ```
* **九项入场风控逐一检查**：
  1. 门槛审查：`signal_score = 0.739 > 0.700`（通过：预测信噪比高于门槛 0.700）
  2. 粒子滤波不确定性：`pf_confidence = 1.00 >= 0.60`（通过：预测路径高度收敛）
  3. 路径动量斜率：`pred_slope = +0.0014 > 0`（通过：预测走势总体向上）
  4. 目标持仓天数：`target_horizon = 5 > 1`（通过：美股 T+0 机制下持有 5 天预计达到收益高点）
  5. 趋势过滤：`ma_fast (264.67) >= ma_slow (261.87)`（通过：中长期均线多头排列）
  6. 去偏预期收益：`expected_return_adj = +2.67% > 0`（通过：预期收益为正）
  7. 回撤幅度与仓位调整：`position_scale = 0.826`（通过：根据潜在回撤微调仓位）
  8. 粒子滤波置信度门槛：`pf_confidence = 1.00 >= 0.35`（通过：达到最低可信度标准）
  9. 开盘流动性：美股开盘正常撮合（通过）
* **输出数据 (`signal_df` 的关键行)**：
  ```text
  date (Index): 2026-05-01
  open: 278.36
  close_prev: 270.86
  signal_score: 0.739
  entry_threshold: 0.700
  exec_signal: 1                # 允许开盘执行买入
  exec_target_horizon: 5        # 目标持有 5 天
  exec_position_scale: 0.826    # 仓位调整系数 0.826
  veto_reason: ""               # 检查全部通过，无否决原因
  ```

---

### 环节五：前置 AI 预测能力体检与准入门槛

* **代码位置**：[backtest/engine/skill_evaluator.py (第25行)](./engine/skill_evaluator.py#L25)
* **调用函数**：`KronosBacktesterPro.evaluate_prediction_skill(signal_df, hard_stop_loss)`
* **输入数据**：已合并对齐历史价格与预测指标的 `signal_df`。
* **执行逻辑**：
  在实际模拟资金交易前，对模型的预测表现进行“体检”，统计预测方向命中率、Rank IC、波动幅度比等指标。
  若命令行指定了 `--require-skill` 且评估结果为 `skill['tradable'] == False`，程序在此处会直接抛出 `ValueError` 中止该股票后续回测，并在排行榜记录体检未通过的原因，避免在预测完全不准的股票上浪费回测时间。
* **输出数据 (`skill` 字典) 真实快照**：
  ```python
  {
      "IC": 0.039,                 # Spearman 秩相关系数
      "IC_t值": 0.426,             # t 统计量
      "方向命中率": 0.483,         # 方向命中率 48.3%
      "幅度比": 0.554,             # 预测幅度与真实波动幅度之比
      "下探跌破率": 0.067,         # 跌破硬止损线的概率仅 6.7%
      "粒子滤波平均置信度": 1.000, # 粒子滤波置信度极高
      "tradable": True             # 允许进入后续模拟交易
  }
  ```

---

### 环节六：基于风险预算确定买入仓位（动态定仓）

* **代码位置**：[backtest/execution/sizer.py (第45行)](./execution/sizer.py#L45)
* **调用函数**：`RiskBudgetSizer.compute_size(...)`
* **执行逻辑**：
  不采用简单的全仓买入或固定比例买入，而是**根据账户单笔允许承受的最大亏损金额，反推计算应该买入的股数**：
  `下单股数 = 风险预算金额 / 单股止损空间`。
* **内部推导计算（美股 AAPL 实测数据）**：
  ```text
  1. 真实开盘买入价: buy_price = 278.36 * (1 + 0.0005) = 278.4992 美元 (开盘价叠加 0.05% 美股滑点)
  2. 动态止损幅度: active_stop_pct = 3.27% (0.0327)
  3. 单股止损空间: stop_distance = 278.4992 * 0.0327 = 9.1069 美元
  4. 绝对止损价格: stop_price = 278.4992 * (1 - 0.0327) = 269.4007 美元
  5. 风险预算金额: risk_budget = 50,000 * 0.01 * 0.8260 = 413.00 美元
  6. 理论买入股数: raw_shares = 413.00 / 9.1069 = 45.35 股
  7. 仓位上限与整手截断:
     - 80% 仓位限制允许的最大股数: 143 股
     - 美股以 1 股为单位向下取整: int(45.35 // 1) = 45 股
  ```
* **输出数据**：
  ```python
  (
      shares_to_buy = 45,          # 最终买入 45 股
      active_stop_pct = 0.0327,    # 生效止损比例 3.27%
      active_trail_pct = 0.025,    # 生效移动止盈回撤 2.5%
      stop_price = 269.4007        # 触发止损价格 269.4007 美元
  )
  ```

---

### 环节七：模拟撮合买入与建立持仓

* **代码位置**：[backtest/execution/simulated_broker.py (第125行)](./execution/simulated_broker.py#L125)
* **调用函数**：`SimulatedBroker.execute_buy(...)`
* **内部处理**：
  扣除购买股票所需资金（`45 * 278.4992 = 12,532.46` 美元），美股佣金为 $0，扣减现金余额后，创建并返回 [Position](./core/types.py#L75) 持仓对象，并在交易流水 `broker.trades` 中记录买入明细。
* **输出数据**：
  * 持仓对象：`Position(symbol="AAPL", shares=45, avg_entry_price=278.4992, stop_price=269.4007, target_horizon=5)`
  * 账户可用现金：`50,000.00 - 12,532.46 = 37,467.54` 美元。
* **日志输出**：
  ```text
  [2026-05-01] [AAPL] 🟢 开仓买入: 45股 @ 278.499 | 目标时界: 5日 | 止损=3.27%(价 269.401) | 耗资: 12,532.46 (费用: 0.00) | 剩余现金: 37,467.54
  ```

---

### 环节八：持仓每日跟踪、止盈止损离场判定与模拟卖出

* **代码位置**：[backtest/engine/backtest_engine.py (第388-440行)](./engine/backtest_engine.py#L388) 与 [backtest/execution/risk_manager.py (第96行)](./execution/risk_manager.py#L96)
* **离场判定两大执行窗口（Execution Gateways）与多维离散状态机**：
  1. **盘中硬止损窗口 (`check_intraday_exit`)**：每个交易日盘中监测最低价 `low` 是否跌破止损线 `stop_price`。若破位且符合 T+0 规则，立即触发盘中以市价止损卖出防范闪崩；
  2. **收盘规则评估窗口 (`evaluate_eod_exit`)**：若盘中未跌破止损，收盘时由状态机多维综合评估：
     * **硬止损 / 移动止盈**：收盘亏损破线或大幅浮盈后自峰值回撤超标时止盈；
     * **时间止损**：到达预测波峰天数但浮亏（模型假说被市场证伪），果断出局释放资金；
     * **动态 ATR 动量跃迁护航（让利润奔跑）**：达到预测天数且动量强劲（紧贴高点且在均线上方），自动取消强平并移交动态 ATR 移动止盈护航，顺延至最大持仓日（双向截断防范赶顶虚胖与震荡磨损）；
     * **预测波峰止盈**：达到目标持仓天数且高位滞涨（未达动量跃迁门槛），落袋为安。
* **AAPL 第一笔交易持仓演进与波峰止盈过程**：
  * **Day 1 ~ Day 4 (2026-05-01 ~ 2026-05-06)**：收盘价温和上涨，始终未跌破止损价 269.401；
  * **Day 5 (2026-05-07)**：收盘价 286.92 美元，成本 278.4992，浮盈 `+3.02%`，`holding_bars = 5`（**达到预期的 5 天持仓目标**），触发平仓提示：
    ```text
    [2026-05-07] [AAPL] ⚠️ 触发平仓预警: 【预测波峰到达止盈(5日/5日目标,浮盈3.02%)】 | 收盘价=286.920, 成本=278.499, 浮盈=+3.02%, 峰值回撤=-1.60%
    ```
  * **Day 6 (2026-05-08 开盘卖出)**：开盘价 289.49 美元，`SimulatedBroker.execute_sell` 执行卖出撮合：
    * 实际成交卖价（扣除 0.05% 滑点）：`289.49 * (1 - 0.0005) = 289.3453` 美元；
    * 卖出总金额：`45 * 289.3453 = 13,020.54` 美元；
    * 规费扣除：美股 SEC/FINRA 规费 `$0.36`；
    * 资金到账后现金：`37,467.54 + 13,020.54 - 0.36 = 50,487.71` 美元；
    * 单笔净收益：`+487.71` 美元（净收益率 `+3.89%`）；
    * 持仓期间最大浮盈（MFE）：`+4.70%`，持仓期间最大浮亏（MAE）：`-1.48%`。
* **日志输出**：
  ```text
  [2026-05-08] [AAPL] 🔴 执行平仓: 45股 @ 289.345 | 持仓: 5天 | 原因: 【预测波峰到达止盈(5日/5日目标,浮盈3.02%)】 | 单笔盈亏: +3.89% ($+487.71) | MFE=+4.70%, MAE=-1.48% | 税费: 0.36 | 账户资金: 50,487.71
  ```

---

### 环节九：每日资产净值结算与绩效指标统计

* **代码位置**：[backtest/engine/metrics.py (第29行)](./engine/metrics.py#L29)
* **调用函数**：`MetricsEvaluator.evaluate(backtest_df, trades, initial_capital, profile)`
* **内部处理**：
  遍历每日净值表 `backtest_df`，根据美股市场年化天数（252 天）计算年化收益率、年化波动率、夏普比率、卡玛比率（年化收益/最大回撤）、索提诺比率、历史最大回撤以及胜率。
* **输出数据 (`metrics` 字典) 真实快照**：
  ```python
  {
      "累计收益率": 0.01988,        # 累计收益率 +1.99%
      "年化收益率": 0.04221,        # 年化收益率 4.22%
      "年化波动率": 0.01489,        # 年化波动率 1.49% (走势平稳)
      "最大回撤": -0.00298,         # 历史最大回撤仅 -0.30%
      "卡玛比率": 14.18,            # 卡玛比率 14.18 (收益对回撤比良好)
      "胜率": 1.00,                 # 胜率 100% (2 战 2 胜)
      "总交易笔数": 2,              # 交易次数 (另有 1 笔期末自动清仓)
      "期末总资产": 50994.14,       # 期末资金 $50,994.14 美元
      "平均持仓天数": 5.5,          # 平均每笔持仓 5.5 天
      "平均MFE": 0.0364,            # 平均单笔最大有利变动 +3.64%
      "平均MAE": -0.0152            # 平均单笔最大不利逆行仅 -1.52%
  }
  ```

---

## 五、输出文件目录结构与说明

运行 `python backtest/run_backtest_batch.py` 时，系统会按照以下目录结构生成并保存所有数据与结果文件，方便查阅：

```text
Kronos/
├── backtest/
│   ├── historical_data/                        # [输入/本地缓存] 阶段 1 拉取的历史行情数据
│   │   ├── AAPL.csv                            # 苹果公司原始日 K 线 (美股)
│   │   ├── MSFT.csv                            # 微软原始日 K 线 (美股)
│   │   └── TSLA.csv                            # 特斯拉原始日 K 线 (美股)
│   │
│   ├── yuce/                                   # [输入/本地缓存] 阶段 2 模型预测结果缓存
│   │   ├── AAPL_wf_predictions.csv             # AAPL 滚动前瞻预测序列
│   │   ├── MSFT_wf_predictions.csv             # MSFT 滚动前瞻预测序列
│   │   └── TSLA_wf_predictions.csv             # TSLA 滚动前瞻预测序列
│   │
│   └── batch/                                  # [回测输出] 批量回测结果总目录
│       │
│       ├── batch_backtest_ranking.csv          # 🏆 阶段 3 生成: 所有股票绩效汇总排行榜
│       │
│       ├── AAPL/                               # AAPL 专属回测结果目录
│       │   ├── AAPL_trades.csv                 # 逐笔交易明细表 (含开平仓、规费、MFE/MAE)
│       │   ├── AAPL_daily_equity.csv           # 每日资产净值与仓位明细表
│       │   ├── AAPL_daily_backtest.csv         # 每日资产净值表 (历史兼容格式)
│       │   ├── AAPL_signals.csv                # 逐日预测价、Z-Score、动态门槛与风控原因表
│       │   ├── AAPL_metrics.json               # 量化指标、AI 体检结果与参数快照
│       │   ├── AAPL_backtest_pro.png           # 四联量化回测复盘全景图
│       │   └── AAPL_backtest.log               # 该股票完整的运行日志
│       │
│       ├── MSFT/                               # MSFT 专属回测结果目录
│       │   └── ...
│       └── TSLA/                               # TSLA 专属回测结果目录
│           └── ...
```

### 输出文件主要字段说明

1. **`batch_backtest_ranking.csv`（大盘排行榜）**：
   * 包含字段：`排名`, `股票代码`, `股票名称`, `市场`, `结算币种`, `期末资产`, `回测状态`, `累计收益率(%)`, `年化收益率(%)`, `夏普比率`, `卡玛比率`, `最大回撤(%)`, `胜率(%)`, `盈亏比`, `交易笔数`, `单笔均收益(%)`, `技能得分`, `体检建议`。
2. **`AAPL_trades.csv`（逐笔交易台账）**：
   * 包含字段：`date`, `action` (BUY/SELL), `price`, `shares`, `commission` (交易规费), `pnl_pct` (单笔净收益率), `pnl_amt` (净盈亏金额), `holding_days`, `reason` (平仓触发原因), `cash_balance`, `mfe_pct` (持仓期最大有利变动), `mae_pct` (持仓期最大不利逆行)。
3. **`AAPL_signals.csv`（信号与风控诊断表）**：
   * 包含字段：`open`, `close`, `pred_close`, `pred_path_max`, `signal_score`, `entry_threshold`, `raw_signal`, `veto_reason` (风控否决原因), `exec_signal`, `exec_position_scale`, `exec_target_horizon`。
4. **`AAPL_metrics.json`（量化指标快照）**：
   * 包含完整的 JSON 键值：`stock_code`, `market`, `initial_capital`, `strategy_params`, `metrics`, `skill_evaluation`。

---

## 六、核心数据结构流转全景表

| 阶段 | 核心数据结构 | 对应 Python 类型 | 关键字段 / 关键信息 | 产生函数 / 消费函数 | 对应保存的文件 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0. 股票池** | `stock_pool` | `List[dict]` | `stock_code`, `stock_name`, `market` | [load_stock_pool_from_file](./run_backtest_batch.py#L105) | 输入 CSV 或命令行参数 |
| **1. 历史行情** | `hist_df` | `pd.DataFrame` | `open`, `high`, `low`, `close`, `volume` | [KronosDataFetcher.fetch_history](./data/kronos_predictor.py#L90) | `backtest/historical_data/AAPL.csv` |
| **2. 预测序列** | `pred_df` | `pd.DataFrame` | `pred_close`, `pred_path_max`, `optimal_horizon`, `pf_confidence` | [KronosDataFetcher.walk_forward_predict](./data/kronos_predictor.py#L135) | `backtest/yuce/AAPL_wf_predictions.csv` |
| **3. 信号矩阵** | `signal_df` | `pd.DataFrame` | `signal_score`, `entry_threshold`, `exec_signal`, `veto_reason` | [KronosStrategy.align_and_generate_signals](./strategies/kronos_strategy.py#L97) | `backtest/batch/AAPL/AAPL_signals.csv` |
| **4. 预测体检** | `skill` | `dict` | `IC` (0.039), `方向命中率` (48.3%), `tradable` | [SkillEvaluator.evaluate](./engine/skill_evaluator.py#L25) | 保存在 `AAPL_metrics.json` |
| **5. 风险定仓** | `sizer_out` | `tuple` | `shares_to_buy` (45), `active_stop_pct` (3.27%), `stop_price` ($269.40) | [RiskBudgetSizer.compute_size](./execution/sizer.py#L45) | 内存计算，用于指导买入下单 |
| **6. 持仓状态** | `position` | [Position](./core/types.py#L75) | `shares` (45), `avg_entry_price` ($278.50), `holding_bars` | [SimulatedBroker.execute_buy](./execution/simulated_broker.py#L125) | 内存对象，用于跟踪持仓状态 |
| **7. 每日净值** | `backtest_df` | `pd.DataFrame` | `portfolio_value`, `cash`, `shares`, `returns`, `drawdown` | [KronosBacktesterPro.run_backtest_engine](./engine/backtest_engine.py#L227) | `backtest/batch/AAPL/AAPL_daily_equity.csv` |
| **8. 交易台账** | `trades` | `List[dict]` | `action`, `price`, `pnl_pct`, `pnl_amt`, `mfe_pct`, `mae_pct` | [SimulatedBroker.execute_sell](./execution/simulated_broker.py#L56) | `backtest/batch/AAPL/AAPL_trades.csv` |
| **9. 绩效统计** | `metrics` | `dict` | `sharpe_ratio`, `max_drawdown` (-0.30%), `total_return` (+1.99%) | [MetricsEvaluator.evaluate](./engine/metrics.py#L29) | `backtest/batch/AAPL/AAPL_metrics.json` |
| **10. 大盘排名** | `ranking_df` | `pd.DataFrame` | `排名` (第2名), `AAPL`, `年化收益率` (4.22%), `卡玛比率` (14.18) | [BatchBacktester.generate_ranking_summary](./run_backtest_batch.py#L466) | `backtest/batch/batch_backtest_ranking.csv` |

---

## 七、常见疑问排查与实战对照

### Q1: 美股回测中的“一手保底（Min-Lot Guarantee）”与整手截断规则是怎样的？
* **解答**：美股交易以 1 股为单位，与 A 股（100 股整手）和港股（根据证券定义的自定义手数，如 100/500 股）不同。在美股市场模式下，`RiskBudgetSizer` 在扣除风险预算后，直接取整为整数股数（例如 45.35 股截断为 45 股）。一手保底机制在美股高单价标的（如股价高昂时计算股数不足 1 股）但仓位预算允许时，保底买入 1 股。

### Q2: 为什么美股回测中没有佣金和印花税？
* **解答**：根据 [MarketProfile.for_stock](./core/market_rules.py#L140) 中的美股交易规则：美股主流零售券商大多实行 0 佣金（`commission_rate = 0.0`），且美股无卖方印花税（`stamp_duty_sell = 0.0`）。系统仅按真实的交易所规费（SEC Fee / FINRA TAF）扣除极微额费用（如 45 股卖出仅扣除 $0.36 规费），完全还原美股全真交易成本。

### Q3: 为什么旧文档说 `SkillEvaluator.evaluate` 是环节二，但代码里是在对齐之后才执行？
* **解答**：这是旧文档在撰写时的顺序疏漏。在真实业务逻辑中，Rank IC 和方向命中率必须对比**真实历史价格变动**与**模型预测变动**，只有在 `align_and_generate_signals` 完成时间对齐并把历史行情与预测数据合并后，才具备计算体检指标的数据基础。因此代码中的顺序必然是：**加载数据 -> 时间对齐与信号生成 -> 预测质量体检 -> 模拟撮合回测**。

### Q4: 运行批量回测时，显卡显存或 MPS 会不会因为多线程并发而崩溃？
* **解答**：完全不会。`KronosDataFetcher` 中实现了全局互斥锁 `_INFERENCE_LOCK` 与双重检查锁（DCL）。多个线程并发运行时，深度学习模型推理阶段会自动排队安全执行；当第一个股票预测完成并保存文件后，后续如果检测到已有缓存文件，就直接读取复用，既彻底避免了底层显卡驱动冲突，又节省了重复推演的时间。
