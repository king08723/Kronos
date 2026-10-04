# Kronos Pro 回测引擎说明

对应脚本：`backtest/run_backtest_kronos_pro.py`  
多市场规则：`backtest/market_rules.py`  
批量回测：`backtest/run_backtest_batch.py`  
看板：`streamlit run backtest/app_backtest_dashboard.py`（单标的 UI 也可 `streamlit run backtest/run_backtest_kronos_pro.py`）

本文说明**当前代码真实在做什么**，并补一层策略直觉。CLI 里仍有 `--z-threshold`，但入场已改为滚动分位，该参数主要用于图表标注，不再决定买不买。

---

## 怎么跑

```bash
# 推荐：拉行情 + Walk-Forward 预测 + 回测
python backtest/run_backtest_kronos_pro.py 300418 --auto
python backtest/run_backtest_kronos_pro.py 00700 --auto
python backtest/run_backtest_kronos_pro.py AAPL --auto

# 只用已有预测（须与历史重叠 ≥ 10 个交易日）
python backtest/run_backtest_kronos_pro.py 300418

# 批量
python backtest/run_backtest_batch.py --auto --stock-file backtest/my_stocks.csv
```

默认数据流：

| 用途 | 路径 |
|------|------|
| 行情缓存 | `examples/data/` |
| 预测缓存 | `examples/yuce/{code}_wf_predictions_auto.csv` |
| 回测图 / 日志 | `examples/backtest/` |

常用参数（与代码默认值一致）：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--horizon` | `auto` | `auto` = 极值自适应时界；也可写整数天数 |
| `--max-auto-horizon` | `10` | auto 模式向前推演的最大交易日 |
| `--lookback` | `120` | 每步预测用的历史窗口 |
| `--test-days` | `120` | 回测验证窗口 |
| `--samples` | `20` | 蒙特卡洛路径数（&lt;20 时分位数退化为极值） |
| `--entry-percentile` | `0.80` | 滚动分位入场：取自身信号分布前 20% |
| `--calib-window` | `60` | 分位阈值与去偏的滚动窗口 |
| `--hard-stop` | `0.04` | 硬止损基准 4%（实际可被 ATR 改写） |
| `--trail-stop` | `0.025` | 移动止盈回撤 2.5% |
| `--max-hold` | `15` | 兜底最大持仓天数（不是固定拿 5 天就走） |
| `--risk-per-trade` | `0.01` | 单笔风险预算 = 净值 × 1% |
| `--max-position` | `0.80` | 单标的仓位上限 80% |
| `--market` | 自动推断 | 强制 `CN` / `HK` / `US` |

---

## 地基：先记住这 5 个概念

1. **OHLC**：每天一根 K 线的开高低收。买卖撮合用开盘价加滑点，盯市用收盘价。
2. **开仓 / 平仓**：空仓时才开新仓，不追高加仓。仓位按风险预算算，**不是永远满仓或空仓**。
3. **回测**：用历史数据按规则重演。本脚本就是这件事。
4. **交易成本**：佣金 + 印花税 + 滑点。A 股默认佣金万三（最低 5 元）、卖方印花税 0.05%、滑点 0.1%；港股/美股规则不同，见 `market_rules.py`。
5. **波动率**：这只股票平时抖得多厉害。信号强度要把预期收益除以「对应时界的正常抖动」。

---

## 这套策略在干什么（一句话）

让 Kronos 对每个交易日预测未来一段价格路径（默认最多 10 天、20 条蒙特卡洛样本）。  
若这条路径在**去偏之后**仍然够强、方向向上、趋势环境允许、路上下探不太深，就在**次日开盘**买入；持仓后用硬止损、移动止盈、波峰时界、转空离场、盘中止损把亏损封住、让利润跑。

---

## 为什么不能「预测涨就买」

| 坑 | 说明 | 代码里的对应物 |
|----|------|----------------|
| 噪声坑 | 日波动 3% 的股票涨 1.5% 可能只是抖动 | `signal_score = 去偏预期收益 / (日波动 × √时界)`，再和滚动分位比 |
| 摩擦成本坑 | 小行情扣完费用不划算 | 分位门槛 + 风险预算定仓，小信号买得少甚至不买 |
| 假路径坑 | 终点涨、中途先暴跌 | 斜率必须 &gt; 0；下探过深则 `position_scale=0` 直接否决 |
| 环境坑 | 大跌趋势里接飞刀 | 快慢均线：下跌趋势且无短期确认则否决 |
| 作弊坑 | 用当天收盘价假装当天开盘就成交 | 预测只用 t 之前的数据；`raw_signal[t]` 在 t-1 收盘已知，**t 日开盘**撮合 |

---

## 开仓：现在不是「固定 Z&gt;0.7 三道门」

旧说明把入场写成固定 Z-Score &gt; 0.7。当前实现是 **自适应分位 + 多层否决**。

### 信号强度（还是 Z 的形状，阈值变了）

```
expected_return     = (pred_close - 昨收) / 昨收
expected_return_adj = expected_return - 滚动预测偏差中位数   # 默认开启去偏
signal_score        = expected_return_adj / (vol_14d × √target_horizon)
```

- `pred_close` 绑定 **无偏期望** `pred_expected`，不拿波峰价做入场。
- `target_horizon` 在 auto 模式下是路径波峰时界（`optimal_horizon`），不是写死的 5。
- 入场条件：`signal_score` 高于过去 `calib_window` 日的 `entry_percentile` 分位（默认 60 日滚动、80% 分位）。预热不足时退化为全样本同口径分位。

### 还必须同时通过

- 预测斜率 `pred_slope > 0`
- 去偏后预期收益 `> 0`
- 趋势：快均线在慢均线之上，或收盘站上快均线
- 预测下探不能太深：相对硬止损深度 ≥ 2 倍时仓位系数归零，直接不买
- 开盘未封涨停
- T+N 市场：波峰时界必须 &gt; 1 日（否则结算周期内卖不到峰值）

### 买多少

不再是「现金 98% 一把梭」。

1. 按 ATR 得到本笔止损幅度（夹在最小/最大止损之间，缺 ATR 则用 `--hard-stop`）。
2. `股数 ≈ (净值 × 风险预算 × 下探仓位系数) / (止损幅度 × 买入价)`。
3. 再截断到：仓位上限 80%、现金 98%、市场整手（A 股 100 股 / 港股每手 / 美股 1 股）。
4. 算出不足一手但现金和仓位上限够一手时，默认 **一手保底**（可用 `--no-min-lot-guarantee` 关掉）。
5. 连续亏损达到阈值会把仓位系数再打五折。

---

## 防偷价：决策和成交必须错开

错误做法：用「当天开盘到当天收盘」的涨幅当当天就能赚到的钱。开盘那一刻你不可能知道收盘价。

当前对齐（`execution_lag = 0`，但信息截止在 t-1）：

```
t-1 日收盘 → 用截至 t-1 的行情跑模型 → 得到对准交易日 t 的 pred_* 和 raw_signal
t   日开盘 → 用真实开盘价 ± 滑点成交
```

Walk-Forward 保证 `pred_*[t]` 不看 t 日及之后的真实 K 线。这和旧文「今天收盘决策、明天开盘买」是同一原则，只是行索引对齐方式改成了「预测行日期 = 可成交日」。

---

## 平仓：不止四把刀

收盘后扫描持仓，命中则记下 `pending_close_reason`，**下一根可交易 K 线开盘**执行（跌停卖不出则顺延；T+N 未满则撤回、下一根再判）。

| 规则 | 当前行为 |
|------|----------|
| 硬止损 | 相对成本亏损达到本笔 `active_stop_pct`（默认基准 -4%，可 ATR 自适应） |
| 移动止盈 | 浮盈已过 +2%，且从持仓最高价回撤达到本笔 `active_trail_pct`（默认 2.5%） |
| 波峰时界 | 持仓天数 ≥ 本笔 `optimal_horizon`：浮亏则时间止损；浮盈且动量仍强则**取消到期强平**，收紧移动止盈并顺延到 `--max-hold`；否则按「预测波峰到达」止盈 |
| 超时兜底 | 持仓达到 `--max-hold`（默认 15 日）必须走 |
| 模型转空 | `signal_score < -0.5`，**或** `斜率&lt;0 且去偏预期收益&lt;0`（不是旧文里的「两者同时成立」） |
| 盘中止损 | 默认开启：当日最低价触及止损价则按止损价（若开盘已跳空则按开盘）卖出；`--no-intraday-stop` 可关 |
| 期末强平 | 回测结束仍持仓，按最后收盘价减滑点平掉 |

非对称结构没变：亏损有封顶，盈利靠移动止盈和动量顺延去跑。只是「拿满 5 天必须走」已经不是默认行为。

---

## `_summarize_paths()`：一条路径如何变成 pred_df 的一列

对应代码：`KronosDataFetcher._summarize_paths`（约 L326–L447）。  
对每个目标日 t，模型输出形状 `(n_samples, pred_len)` 的 close/high/low。默认 `n_samples=20`，`pred_len` 在 auto 下是 `--max-auto-horizon`（10），不是固定 5。

### 入场用：`pred_expected` → 下游的 `pred_close`

- **auto**：全时界均值路径再取均值（与 argmax **解耦**，避免选峰正偏）。
- **固定时界**：用均值路径的**终点** `mean_path[-1]`。

旧说明写成 `mean(mean_path)` 且举例 5 天，只覆盖 auto 模式。

### 止盈目标用：`pred_path_max`

对**每一条**采样路径先取最高价，再取这些峰值的**中位数**。  
不是旧说明里的 `argmax(mean_path)`——跨路径先平均会把尖峰抹平。

### 时界：`optimal_horizon`

- **auto**：各路径峰值出现时间的中位数 + 1。
- **固定时界**：等于 `eval_horizon`。

这个值进入止损状态机，作为「预测波峰到达」的持仓时钟，不是画图装饰。

### 斜率：`pred_slope`

对**整条** `mean_path` 做线性回归，斜率再除以决策时的真实昨收，得到每日百分比。  
不是旧说明里的只拟合到波峰 `mean_path[:opt_horizon]`（那种写法会把「因为选了峰所以斜率为正」做成同义反复）。

### 最大下探：`pred_min_down`

每条路径在 `[:opt_horizon]` 内先取最低价，再取保守分位（默认 10% 分位）。  
不是先把路径平均再取最低——平均会低估极端下探。

### 同一条均值路径的分工（示意）

```
价格
 ^
 |                 ★ 逐路径峰值的中位数 → pred_path_max（只用于止盈目标）
 |            ↗        ↘
 |       ●                 ●
 |  ●                           ●
 +------------------------------------→ 交易日
   Day1  Day2  ...              DayN

pred_close / pred_expected = auto 下全时界均值（入场）
pred_slope                 = 全时界回归斜率（方向过滤）
optimal_horizon            = 各路径峰值时间的中位数（持仓时钟）
```

每个 Walk-Forward 日期都会重新压缩一次，所以 pred_df 每一行都有这些列。斜率为负的行会被入场否决。

---

## 市场规则（不是只做 A 股 100 股）

`MarketProfile.for_stock` 按代码推断：

- **A 股**：100 股整手、T+1、涨跌停、佣金万三最低 5 元、卖方印花税 0.05%、沪市过户费、滑点 0.1%，默认本金 10 万。
- **港股**：每手股数查表（未收录默认 100 并警告）、T+0、双边印花税（2023-11-17 起 0.10%）、滑点 0.15%，默认本金 30 万。
- **美股**：1 股、T+0、无印花税/佣金建模、滑点 0.05%，默认本金 5 万。

不做空、不建模盘前盘后与碎股。

---

## 相关入口

- 单标的 CLI / 内置 Streamlit：`run_backtest_kronos_pro.py`
- 股票池批量 + 排行榜：`run_backtest_batch.py`（产物在 `backtest/batch/`）
- 统一看板（批量横截面 + 单标的下钻）：`app_backtest_dashboard.py`
- 规则单测：`python -m pytest backtest/test_market_rules.py`
