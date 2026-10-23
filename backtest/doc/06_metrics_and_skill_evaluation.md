# Kronos 量化绩效指标与 AI 预测体检数据结构 (metrics & skill) 说明书

> **物理存储路径**：`backtest/batch/{SYMBOL}/{SYMBOL}_metrics.json`（如 [AAPL_metrics.json](file:///Users/jyy/Documents/github_projects/Kronos/backtest/batch/AAPL/AAPL_metrics.json)）  
> **内存对象名称**：`metrics` (`dict`) 与 `skill_evaluation` (`dict`)  
> **产生函数**：
> - 策略绩效评估：[MetricsEvaluator.evaluate](file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/metrics.py#L29)
> - AI 预测体检：[SkillEvaluator.evaluate](file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/skill_evaluator.py#L24)
> **核心消费模块**：
> - 批量回测排名生成：[BatchBacktester.generate_ranking_summary](file:///Users/jyy/Documents/github_projects/Kronos/backtest/run_backtest_batch.py#L466)
> - 终端回测报告打印：[MetricsEvaluator.print_report](file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/metrics.py#L110)
> - WebUI 绩效仪表盘：[app_backtest_dashboard.py](file:///Users/jyy/Documents/github_projects/Kronos/backtest/app_backtest_dashboard.py)

---

## 一、数据结构背景与双层评估体系

`metrics.json` 是 Kronos 回测系统的最终量化裁决报告。Kronos 采用业界首创的 **“AI 前置体检 + 策略实战撮合”** 双层评估架构：

```text
                  ┌──────────────────────────────────────────────┐
                  │ 1. AI 预测能力前置体检 (skill_evaluation)      │
                  │    - 检验模型本身是否有 Alpha 预测能力         │
                  │    - 评估 IC、t检验、方向命中率、下探违约率     │
                  │    - 若无能力，直接一票否决准入 (tradable=False)│
                  └──────────────────────┬───────────────────────┘
                                         │ 准入通过 (Pass)
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 2. 策略实战撮合绩效评估 (metrics)             │
                  │    - 结合风控、定仓、滑点摩擦的真实交易成绩    │
                  │    - 评估年化收益、夏普比率、卡玛比率、最大回撤│
                  └──────────────────────────────────────────────┘
```

该 JSON 快照将两个维度的全部参数与结果永久固化，杜绝了“策略再精巧却在交易一个毫无预测能力的噪音模型”的重大隐患。

---

## 二、JSON 顶层结构全览

文件包含 6 个核心根节点：
```json
{
  "stock_code": "AAPL",
  "market": "US",
  "initial_capital": 50000.0,
  "slippage_rate": 0.0005,
  "exported_at": "2026-10-15 23:14:32",
  "strategy_params": { ... },
  "metrics": { ... },
  "skill_evaluation": { ... }
}
```

---

## 三、AI 预测能力前置体检模块 (`skill_evaluation`)

源码实现见 [skill_evaluator.py](file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/skill_evaluator.py)。

| 键名 | 典型示例值 | 量化数学定义与业务解读 |
| :--- | :--- | :--- |
| `评估口径` | `1日预期` | 优先使用 `pred_close_1d` 对标次日真实收益，测算单日选时纯度 |
| `样本数` | `120.0` | 参与有效评估的非空交易日总数（必须 >= 20 条） |
| `IC` | `0.0392` | **Spearman 秩相关系数 (Information Coefficient)**：衡量预测排序与真实收益排序的相关度 |
| `IC_t值` | `0.4257` | IC 的统计显著性 t 统计量：`t = IC * sqrt((N-2)/(1-IC^2))` |
| `IC显著` | `0.0` (False) | 判断 `abs(t) >= 1.96`（在 95% 置信水平下是否显著异于 0） |
| `方向命中率` | `0.4833` | 涨跌方向预测正确的比例：`mean((predicted > 0) == (realized > 0))` |
| `预测收益均值` | `+0.113%` | 模型单日预测收益率的时间序列算术均值 |
| `真实收益均值` | `+0.205%` | 标的次日真实收益率的时间序列算术均值 |
| `幅度比` | `0.5537` | **预测收益均值 / 真实收益均值**（衡量模型涨幅预期是偏激进还是偏保守） |
| `幅度偏差` | `-0.0009` | 预测均值减去真实均值，反映整体系统性高估或低估 |
| `下探跌破率` | `0.0667` | 真实盘中最低价击穿模型预测最大下探 `pred_min_down` 的频率（要求 < 10%） |
| `粒子滤波平均置信度` | `1.0000` | 粒子滤波近期走势吻合度的均值 |
| `粒子滤波验证误差` | `0.0000` | 粒子在已知走势上的加权跟踪相对误差 |
| `不通过原因` | `["IC 不显著(t=0.43)"]` | 若未能通过体检门禁，详细列出被拦截的具体原因列表 |
| `tradable` | `false` | **标的准入终审结论**：是否具备可实盘/可回测的统计学有效性 |

---

## 四、策略实战撮合绩效模块 (`metrics`)

源码实现见 [metrics.py](file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/metrics.py)。

### 1. 收益与风险维度
- **累计收益率 (`累计收益率`)**：`portfolio_value[-1] / initial_capital - 1.0`（如 `+1.99%`）。
- **年化收益率 (`年化收益率`)**：
  ```text
  annual_return = (1 + 累计收益率) ^ (252 / 回测交易日数) - 1.0 (如 +4.22%)
  ```
- **年化波动率 (`年化波动率`)**：日收益率标准差乘根号 252：`std(daily_returns) * sqrt(252)`。
- **最大回撤 (`最大回撤`)**：历史最高净值回撤的最深低谷：`min(drawdown)`（如 `-0.30%`）。

### 2. 风险调整收益比率
- **夏普比率 (`夏普比率`, Sharpe Ratio)**：
  ```text
  sharpe = (年化收益率 - 无风险利率) / 年化波动率
  ```
  *(注：默认无风险利率取 2.5% 或 4.5% 美债利率)*
- **索提诺比率 (`索提诺比率`, Sortino Ratio)**：
  仅以 **下行波动率（Downside Deviation）** 作为分母，不对上行盈利波动进行惩罚。
- **卡玛比率 (`卡玛比率`, Calmar Ratio)**：
  ```text
  calmar = 年化收益率 / abs(最大回撤)
  ```
  示例中卡玛比率高达 `14.18`，反映出策略在极小的回撤代价（0.30%）下实现了稳健的正收益。

### 3. 交易行为统计
- **总交易笔数 (`总交易笔数`)**：完整闭环的平仓交易总笔数（如 2 笔）。
- **胜率 (`胜率`)**：盈利交易笔数 / 总交易笔数（示例中 2 笔全胜，胜率 100%）。
- **盈亏比 (`盈亏比`)**：平均盈利金额 / 平均亏损金额（若无亏损笔数则为 `inf` 或 `null`）。
- **持仓时间占比 (`持仓占比`, Exposure)**：持仓交易日天数 / 回测总交易日天数（如 `10.83%`，说明策略绝大部分时间空仓等待，绝非盲目死扛）。
- **平均持仓天数 (`平均持仓天数`)**：平均每笔交易持有天数（如 `5.5 天`，与 AI 推演最优波峰时界 5~6 天完全吻合）。
- **路径质量指标**：
  - `平均MFE`：单笔平均最大有利冲高幅度（如 `+3.64%`）；
  - `平均MAE`：单笔平均最大不利回撤幅度（如 `-1.52%`）。

---

## 五、批量大盘排名对照 (`batch_backtest_ranking.csv`)

在执行多标的批量回测时，`BatchBacktester` 会自动抽取所有标的的 `metrics.json`，汇总生成大盘横向天梯榜 [batch_backtest_ranking.csv](file:///Users/jyy/Documents/github_projects/Kronos/backtest/batch/batch_backtest_ranking.csv)：

```text
排名,股票代码,股票名称,市场,期末资产,累计收益率(%),年化收益率(%),夏普比率,卡玛比率,最大回撤(%),胜率(%),技能得分,体检建议
1,MSFT,微软,US,$51,772.11,3.54,7.59,0.66,5.07,-1.50,66.67,0,需谨慎 🟡
2,AAPL,苹果公司,US,$50,994.14,1.99,4.22,-0.17,14.18,-0.30,100.0,0,需谨慎 🟡
3,TSLA,特斯拉,US,$50,723.63,1.45,3.06,-0.59,2.66,-1.15,80.0,0,需谨慎 🟡
4,NVDA,英伟达,US,$50,000.00,0.00,0.00,0.00,,0.00,0.0,0,需谨慎 🟡
```

这实现了从单标的深度诊断到跨标的多资产横向遴选的完整闭环。
