# Kronos 逐笔交易成交台账数据结构 (trades) 说明书

> **物理存储路径**：`backtest/batch/{SYMBOL}/{SYMBOL}_trades.csv`（如 [AAPL_trades.csv](file:///Users/jyy/Documents/github_projects/Kronos/backtest/batch/AAPL/AAPL_trades.csv)）  
> **内存对象名称**：`trades` (`List[dict]`)  
> **产生函数**：[SimulatedBroker.execute_buy](file:///Users/jyy/Documents/github_projects/Kronos/backtest/execution/simulated_broker.py#L125) 与 [SimulatedBroker.execute_sell](file:///Users/jyy/Documents/github_projects/Kronos/backtest/execution/simulated_broker.py#L56)  
> **核心消费模块**：
> - 量化绩效指标评估：[MetricsEvaluator.evaluate](file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/metrics.py#L29)
> - 回测收益与盈亏可视化：[plot_backtest_results](file:///Users/jyy/Documents/github_projects/Kronos/backtest/run_backtest_kronos_pro.py#L210)

---

## 一、数据结构背景与核心定位

`trades` 是记录回测过程中每一笔买入与卖出交易的 **流水台账表**。与只记录每日资金总额的净值表不同，交易台账以 **逐笔交易订单（Order/Trade）** 为粒度，完整记录了从开仓点位、持仓成本、止损预期、手续费滑点扣除，到离场触发原因、最终盈亏、以及持仓过程中的极大盈利/亏损变动路径。

它是评估策略 **执行胜率、盈亏比、离场效率（MFE/MAE）以及摩擦成本** 的最基础凭证。

---

## 二、数据表头与 18 个字段全景总览

| 字段名 | 数据类型 | 典型示例值 | 业务含义与核心功能 |
| :--- | :--- | :--- | :--- |
| `date` | `str (YYYY-MM-DD)` | `2026-05-08` | 交易指令撮合成交的日期 |
| `action` | `str` | `BUY` / `SELL` | 交易行为动作（买入开仓或卖出平仓） |
| `price` | `float64` | `289.3453` | 实际撮合成交价（已计入买卖双向滑点） |
| `shares` | `int64` | `45` | 实际成交股数（经一手整手规则截断取整） |
| `commission` | `float64` | `0.36` | 本次交易所扣除的手续费、印花税与规费总和 |
| `pnl_pct` | `float64` | `0.038945` | 单笔交易净收益率（扣除双边所有摩擦费用） |
| `pnl_amt` | `float64` | `487.71` | 单笔交易净盈亏绝对金额（账户币种） |
| `holding_days` | `int64` | `5` | 该笔持仓实际持有的交易日天数 |
| `reason` | `str` | `预测波峰到达止盈(5日/5日目标,浮盈3.02%)` | 本次平仓指令触发的具体状态机原因 |
| `cash_balance` | `float64` | `50487.71` | 本次交易撮合完成后的账户可用现金余额 |
| `target_horizon` | `float64` | `5.0` | 开仓时根据 AI 模型推演绑定的最优持有天数 |
| `stop_pct` | `float64` | `0.0327` | 开仓时确立的主动动态止损比例（3.27%） |
| `stop_price` | `float64` | `269.4007` | 开仓时确立的绝对止损触发价位 |
| `entry_date` | `str (YYYY-MM-DD)` | `2026-05-01` | 该笔持仓最初买入成交的日期 |
| `entry_price` | `float64` | `278.4992` | 该笔持仓买入成交的原始均价成本 |
| `mfe_pct` | `float64` | `0.047041` | **持仓期最大有利变动 (MFE)**：持仓最大潜在浮盈 |
| `mae_pct` | `float64` | `-0.014827` | **持仓期最大不利逆行 (MAE)**：持仓最大潜在浮亏 |
| `symbol` | `str` | `AAPL` | 交易标的代码 |

---

## 三、核心量化机制与计算原理

### 1. 全真成本与滑点撮合机制
在执行买入和卖出时，系统严格还原真实交易所摩擦：
- **买入撮合成交价**：`buy_price = open_price * (1.0 + slippage_rate)`（美股默认万分之 5 滑点）。
- **卖出撮合成交价**：`sell_price = fill_price * (1.0 - slippage_rate)`。
- **交易费用扣除**：
  - 美股：0 佣金、0 印花税，仅按真实 SEC/FINRA 规则扣除极微额规费（如 45 股卖出仅扣 $0.36）；
  - A 股：双边万分之 2.5 佣金（最低 5 元起），卖方千分之 0.5 印花税；
  - 港股：包含双边佣金、从价印花税与交易所交易费。

### 2. 单笔盈亏计算
- **净盈亏金额 (`pnl_amt`)**：
  ```text
  pnl_amt = (sell_price - entry_price) * shares - total_commission
  ```
- **净收益率 (`pnl_pct`)**：
  ```text
  pnl_pct = pnl_amt / (entry_price * shares)
  ```

### 3. 六大离场机理（`reason` 字段解析）
平仓动作全部由退出状态机 [ExitStateMachine](file:///Users/jyy/Documents/github_projects/Kronos/backtest/execution/risk_manager.py#L96) 严格判定触发：

| 触发原因标识 | 触发条件 | 量化设计原理与逻辑 |
| :--- | :--- | :--- |
| **盘中止损** | 盘中最低价触及 `stop_price` | 保护本金第一底线，盘中一旦击穿止损线立即斩仓 |
| **收盘硬止损** | 收盘跌幅超过 `active_stop_pct` | 应对跳空低开或盘中未触及但在收盘破位的情形 |
| **移动止盈 (Trailing Stop)** | 利润冲高后从波峰回撤超过阈值 | 在浮盈达到一定比例后锁定已有利润，防止盈利变亏损 |
| **预测波峰到达止盈** | `holding_days >= target_horizon` 且盈利 | 达到 AI 模型预测的最高波峰天数，及时高位兑现利润 |
| **时间止损-模型信号证伪** | `holding_days >= target_horizon` 但浮亏 | 模型预测的波峰周期已过但价格未涨反跌，逻辑证伪主动认赔出局 |
| **回测结束强平** | 运行至回测区间最后 1 个交易日 | 将所有未平仓头寸按最后收盘价折现，便于统计完整业绩闭环 |

---

## 四、持仓路径质量分析：MFE 与 MAE 的实战价值

`mfe_pct` 与 `mae_pct` 是机构量化交易中评价交易执行质量的核心维度：

```text
买入点 (entry_price)
  ├── 向上冲刺至最高价 high_price ──> MFE (最大有利变动，如 +4.70%)
  └── 向下回调至最低价 low_price  ──> MAE (最大不利逆行，如 -1.48%)
                                        │
                                        ▼ 最终卖出 (pnl_pct, 如 +3.89%)
```

1. **MFE（Maximum Favorable Excursion，最大有利变动）**：
   - 记录持仓期间如果卖在最高点能拿到的最大浮盈；
   - **诊断意义**：如果某一笔交易的 `mfe_pct` 达到 +8%，但最终平仓 `pnl_pct` 只有 +1%，说明止盈策略过于迟钝，回吐了过多利润；如果 `mfe_pct` 与 `pnl_pct` 非常接近，说明高位兑现离场极其精准。
2. **MAE（Maximum Adverse Excursion，最大不利逆行）**：
   - 记录持仓期间经历过的最大浮亏；
   - **诊断意义**：如果大部分最终盈利的交易，其 `mae_pct` 都在 -1.5% 以内，说明买点选时极为精准，承受的心理回撤很小；如果常年在 -3.5%（濒临 4% 止损线）边缘徘徊后才拉起，说明开仓过早，需要优化买点过滤。
