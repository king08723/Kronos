# Kronos 历史行情日 K 线数据结构 (hist_df) 说明书

> **物理存储路径**：`backtest/historical_data/{SYMBOL}.csv`（如 [AAPL.csv](file:///Users/jyy/Documents/github_projects/Kronos/backtest/historical_data/AAPL.csv)）  
> **内存对象名称**：`hist_df` (`pandas.DataFrame`)  
> **产生函数**：[KronosDataFetcher.fetch_history](file:///Users/jyy/Documents/github_projects/Kronos/backtest/data/kronos_predictor.py#L487)  
> **核心消费模块**：
> - 滚动预测推演输入：[KronosDataFetcher.walk_forward_predict](file:///Users/jyy/Documents/github_projects/Kronos/backtest/data/kronos_predictor.py#L573)
> - 策略特征计算与对齐：[KronosStrategy.align_and_generate_signals](file:///Users/jyy/Documents/github_projects/Kronos/backtest/strategies/kronos_strategy.py#L97)
> - 模拟撮合真实行情回放：[KronosBacktesterPro.run_backtest_engine](file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/backtest_engine.py#L227)

---

## 一、数据结构背景与工程防伪原则

`hist_df` 是 Kronos 回测系统的基石。所有 AI 大模型的历史提示词构建（Prompt）、真实盈亏撮合（Fill）、以及前置技能评估的对标真值，全部依赖于该数据结构。

在 Kronos 的工程设计中，遵循以下最高安全原则：
1. **禁用模拟兜底（Zero Synthetic Tolerance）**：
   在历史拉取中显式配置 `allow_fallback=False`。如果网络异常或证券代码不存在，系统直接报错中断，**坚决杜绝用几何布朗运动或随机数生成的虚假 K 线流入量化回测**。
2. **多市场代码全自动规范化（Symbol Resolution）**：
   通过 [SymbolResolver](file:///Users/jyy/Documents/github_projects/Kronos/backtest/core/market_rules.py#L80) 自动识别 A 股（沪市 `sh.` / 深市 `sz.`）、港股（5 位纯数字）与美股（纳斯达克/纽交所英文字母），抹平跨市场数据源格式差异。
3. **时效本地缓存防击穿**：
   优先读取带有时间戳校验的本地 Parquet/CSV 缓存；仅在强制刷新或缓存过期时才调用底层 API（AKShare / YFinance）。

---

## 二、标准数据结构与字段规范

回测引擎内部统一约束 `hist_df` 必须是以日期为索引（单调递增）的标准 DataFrame：

| 字段名称 | 存储形态 | 数据类型 | 物理单位 | 业务含义与用途 |
| :--- | :--- | :--- | :--- | :--- |
| `date` (Index) | 行索引 | `pd.DatetimeIndex` | 交易日 `YYYY-MM-DD` | 确保严格的时间升序，杜绝乱序数据 |
| `open` | 数据列 | `float64` | 账户法币元/美元 | 当日开盘价：**开仓撮合的基准价格** |
| `high` | 数据列 | `float64` | 账户法币元/美元 | 当日最高价：更新持仓峰值 `peak_price` 与计算 MFE |
| `low` | 数据列 | `float64` | 账户法币元/美元 | 当日最低价：**盘中止损触发检查** 与计算 MAE |
| `close` | 数据列 | `float64` | 账户法币元/美元 | 当日收盘价：每日盯市估值与收盘离场评估 |
| `volume` | 数据列 | `float64` | 股 (美股) / 手 (A股) | 当日成交量：流动性校验与成交量放大确认 |
| `amount` | 数据列 | `float64` | 账户法币元/美元 | 当日成交额（若缺失自动用 `close * volume` 补齐） |

---

## 三、标准数据切片示例

```text
>>> print(hist_df.head(3))
                          open      high       low     close       volume         amount
date                                                                                    
2026-04-22 00:00:00  268.5000  271.2000  267.1000  270.8000   45,120,000  12,218,500,000
2026-04-23 00:00:00  271.0000  274.5000  269.8000  272.9400   42,800,000  11,681,832,000
2026-04-24 00:00:00  272.2700  272.5700  269.1600  270.5700   38,157,110  10,324,169,252
```

---

## 四、下游三大关键数据流应用

1. **输入大模型（Prompt Window）**：
   在 `walk_forward_predict` 中，截取当前决策日前 `lookback` 天（如 120 天）的 OHLCV 序列，作为大模型预测未来行情的先验特征。
2. **衍生策略风控指标**：
   在 `align_and_generate_signals` 中，基于真实收盘价序列滚动计算 14 天收益率波动率、20/60 日均线、以及 14 天真实波幅（ATR）。
3. **全真状态机模拟撮合**：
   在 `run_backtest_engine` 每日循环中：
   - 开盘时：使用 `row.open` 模拟买入开仓；
   - 盘中时：比对 `row.low` 是否击穿盘中止损线；
   - 收盘时：使用 `row.close` 重新评估持仓浮盈浮亏与总资产净值。
