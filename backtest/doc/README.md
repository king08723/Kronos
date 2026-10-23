# Kronos 量化回测核心数据结构流转规范与文档总索引

本目录收录了 Kronos 机构级量化回测系统中，各个生命周期阶段核心数据结构的 **单体技术说明书**。  
所有文档严格遵循 **代码级字段映射**、**量化数学公式纯文本化（杜绝 LaTeX）** 与 **全真数据流向闭环** 标准编写，与架构指南 [backtest_dataflow_guide.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/backtest_dataflow_guide.md) 保持完全同步。

---

## 一、核心数据结构文档目录索引

| 序号 | 数据结构名称 | 对应 Python 类型 | 物理存储文件 | 核心业务职责 | 技术说明书直达 |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **01** | **历史行情日K线** (`hist_df`) | `pd.DataFrame` | `backtest/historical_data/{SYMBOL}.csv` | 提供真实未篡改的 OHLCV 基石行情，杜绝合成模拟假数据 | [01_hist_df_historical_data.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/doc/01_hist_df_historical_data.md) |
| **02** | **滚动前瞻预测序列** (`pred_df`) | `pd.DataFrame` | `backtest/yuce/{SYMBOL}_wf_predictions_auto.csv` | 蒙特卡洛多路径推演与粒子滤波置信度，解耦无偏期望与波峰止盈 | [02_pred_df_predictions.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/doc/02_pred_df_predictions.md) |
| **03** | **信号风控诊断矩阵** (`signal_df`) | `pd.DataFrame` | `backtest/batch/{SYMBOL}/{SYMBOL}_signals.csv` | 时间对齐严格杜绝未来函数，自适应去偏，九重入场风控否决过滤 | [03_signal_df_signals_and_veto.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/doc/03_signal_df_signals_and_veto.md) |
| **04** | **逐笔交易成交台账** (`trades`) | `List[dict]` | `backtest/batch/{SYMBOL}/{SYMBOL}_trades.csv` | 记录真实撮合价格、滑点费用、6 大离场机理以及 MFE/MAE 路径质量 | [04_trades_transaction_ledger.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/doc/04_trades_transaction_ledger.md) |
| **05** | **每日资产净值表** (`backtest_df`) | `pd.DataFrame` | `backtest/batch/{SYMBOL}/{SYMBOL}_daily_equity.csv` | 每日盯市结算、现金与仓位暴露跟踪、动态回撤与买入持有超额 | [05_daily_equity_backtest_df.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/doc/05_daily_equity_backtest_df.md) |
| **06** | **量化绩效与AI体检** (`metrics` & `skill`) | `dict` | `backtest/batch/{SYMBOL}/{SYMBOL}_metrics.json` | 固化双层评估体系：前置 Rank IC 技能门禁与实战夏普/卡玛/胜率报告 | [06_metrics_and_skill_evaluation.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/doc/06_metrics_and_skill_evaluation.md) |

---

## 二、端到端数据流转全景架构

```text
[01. 真实历史行情 hist_df] ───────────────┐
                                          │
                                          ▼
[02. 滚动前瞻预测序列 pred_df] ───────────┼──> [03. 信号风控诊断矩阵 signal_df]
 (AAPL_wf_predictions_auto.csv)           │     (时间严格 shift(1) 对齐 + 九重否决)
                                          │                      │
                                          │                      ├─> [06. AI技能体检 skill]
                                          │                      │    (Rank IC, t检验, 门禁)
                                          │                      │
                                          ▼                      ▼
                           [回测内核撮合状态机 BacktestEngine]
                                          │
                                          ├─ 逐笔开平仓 ──> [04. 逐笔交易台账 trades]
                                          │                (滑点费用、MFE/MAE、6大离场)
                                          │
                                          ├─ 每日盯市 ────> [05. 每日净值表 backtest_df]
                                          │                (动态净值、仓位占比、动态回撤)
                                          │
                                          ▼
                           [06. 策略量化绩效快照 metrics.json]
                           (夏普比率、卡玛比率、最大回撤、大盘排名汇总)
```

---

## 三、各文档定位与使用指南

1. **做 AI 预测模型调优与特征工程**：重点查阅 [02_pred_df_predictions.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/doc/02_pred_df_predictions.md) 与 [06_metrics_and_skill_evaluation.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/doc/06_metrics_and_skill_evaluation.md)，关注单日期望、波峰溢价、粒子滤波置信度与 Rank IC 显著性。
2. **做交易策略信号与开仓风控**：重点查阅 [03_signal_df_signals_and_veto.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/doc/03_signal_df_signals_and_veto.md)，关注信噪比 Z-Score 门槛、反转感知去偏与九重否决原因诊断。
3. **做交易执行与退出状态机分析**：重点查阅 [04_trades_transaction_ledger.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/doc/04_trades_transaction_ledger.md)，关注单笔收益、6 种退出机理以及 MFE（盈利回吐）与 MAE（逆行风险）。
4. **做投资组合风险与资金曲线分析**：重点查阅 [05_daily_equity_backtest_df.md](file:///Users/jyy/Documents/github_projects/Kronos/backtest/doc/05_daily_equity_backtest_df.md)，关注每日盯市、仓位暴露度（Exposure）以及最大回撤深度。
