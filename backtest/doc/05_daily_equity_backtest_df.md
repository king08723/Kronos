# Kronos 每日资产净值与持仓数据结构 (backtest_df) 说明书

> **物理存储路径**：`backtest/batch/{SYMBOL}/{SYMBOL}_daily_equity.csv` 与 `{SYMBOL}_daily_backtest.csv`（如 [AAPL_daily_equity.csv](file:///Users/jyy/Documents/github_projects/Kronos/backtest/batch/AAPL/AAPL_daily_equity.csv)）  
> **内存对象名称**：`backtest_df` (`pandas.DataFrame`)  
> **产生函数**：[KronosBacktesterPro.run_backtest_engine](file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/backtest_engine.py#L448)  
> **核心消费模块**：
> - 投资组合与风险指标评估：[MetricsEvaluator.evaluate](file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/metrics.py#L29)
> - 净值曲线与回撤走势绘图：[plot_backtest_results](file:///Users/jyy/Documents/github_projects/Kronos/backtest/run_backtest_kronos_pro.py#L210)

---

## 一、数据结构背景与核心定位

`backtest_df` 是回测引擎每日对账户资产进行 **盯市结算（Mark-to-Market, MTM）** 生成的日频时序数据表。无论当天是否有买卖成交，系统在每日收盘后都会对账户可用现金、当前持仓数量、持仓公允市值、总净值、单日收益率、动态回撤以及相对于标的自身买入持有（Buy & Hold）的基准超额进行全量结算。

它是计算年化收益率、夏普比率、索提诺比率、最大回撤、卡玛比率等 **经典量化投资组合绩效指标** 的最核心时序底表。

---

## 二、数据表头与 14 个字段全景总览

| 字段名 | 数据类型 | 典型示例值 | 业务含义与核心计算逻辑 |
| :--- | :--- | :--- | :--- |
| `date` | `str (YYYY-MM-DD)` | `2026-05-01` | 结算交易日 |
| `portfolio_value` | `float64` | `50051.34` | **当日总资产净值**：`cash + shares * close` |
| `cash` | `float64` | `37467.54` | 当日收盘账户可用现金余额 |
| `shares` | `int64` | `45` | 当日收盘持仓总股数 |
| `close` | `float64` | `279.64` | 当日标的公允收盘价 |
| `exec_signal` | `int64 (0/1)` | `1` | 当日开盘执行的开仓动作标记 |
| `returns` | `float64` | `0.001027` | 当日账户总资产净值收益率 |
| `position_market_value` | `float64` | `12583.80` | 当日收盘持仓公允市值：`shares * close` |
| `position_pct` | `float64` | `0.2514` | **当前仓位暴露比率**：持仓市值占总净值比例 |
| `cum_return` | `float64` | `0.001027` | 策略自回测起点以来的累计净值收益率 |
| `drawdown` | `float64` | `0.000000` | 当日相对历史净值高水位线的动态回撤率 |
| `benchmark_return` | `float64` | `0.032415` | 标的股票自身的单日真实收益率（基准日收益） |
| `benchmark_cum_return` | `float64` | `0.033522` | 标的股票买入持有策略（Buy & Hold）累计收益率 |
| `excess_return` | `float64` | `-0.032495` | **策略超额收益 (Alpha)**：策略累计收益减去基准累计收益 |

---

## 三、核心量化指标计算原理

### 1. 每日盯市结算（Mark-to-Market）
每日收盘后，资产净值严格按照以下恒等式结算：
```text
portfolio_value[t] = cash[t] + shares[t] * close[t]
position_market_value[t] = shares[t] * close[t]
position_pct[t] = position_market_value[t] / portfolio_value[t]
```

### 2. 每日收益率与累计收益率
- **单日收益率 (`returns`)**：
  ```text
  returns[t] = (portfolio_value[t] - portfolio_value[t-1]) / portfolio_value[t-1]
  ```
  *(注：空仓期间该值为 0.0，完美规避市场下跌回撤)*
- **累计收益率 (`cum_return`)**：
  ```text
  cum_return[t] = (portfolio_value[t] - initial_capital) / initial_capital
  ```

### 3. 高水位线（High-Water Mark）与动态回撤 (`drawdown`)
回撤衡量投资者在任何时点可能遭受的最大心理与资金承压：
- **历史净值峰值（高水位线）**：
  ```text
  peak_value[t] = max(portfolio_value[0], portfolio_value[1], ..., portfolio_value[t])
  ```
- **当日动态回撤 (`drawdown`)**：
  ```text
  drawdown[t] = (portfolio_value[t] - peak_value[t]) / peak_value[t]
  ```
  该值恒为小于等于 0 的负数（或 0）。整个序列的最小值即为策略的 **最大回撤（Max Drawdown, MDD）**。

### 4. 基准对照与超额收益（Alpha）
系统自动将当前策略与 **“第一天满仓买入并一直持有该股票”** 的纯多头被动基准进行实时同台竞技：
- `benchmark_return[t] = (close[t] - close[t-1]) / close[t-1]`
- `benchmark_cum_return[t] = (close[t] - close[0]) / close[0]`
- `excess_return[t] = cum_return[t] - benchmark_cum_return[t]`

---

## 四、数据实战应用与下游链路

```text
[AAPL_daily_equity.csv (backtest_df)]
                   │
                   ▼
  MetricsEvaluator 绩效指标评估 (metrics.py)
  ├─ 年化收益率: (1 + cum_return[-1]) ^ (252 / 回测天数) - 1
  ├─ 年化波动率: std(returns) * sqrt(252)
  ├─ 夏普比率 (Sharpe): (年化收益率 - 无风险利率) / 年化波动率
  ├─ 索提诺比率 (Sortino): 超额收益 / 下行半方差波动率 (Downside Deviation)
  ├─ 最大回撤 (Max Drawdown): min(drawdown)
  └─ 卡玛比率 (Calmar): 年化收益率 / abs(最大回撤)
                   │
                   ▼
  可视化引擎绘图 (AAPL_backtest_pro.png)
  ├─ 子图一: 策略净值走势 vs 标的买入持有基准走势
  ├─ 子图二: 仓位暴露比例与可用现金变动
  └─ 子图三: 水下动态回撤区域图 (Underwater Plot)
```
