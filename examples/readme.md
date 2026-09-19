# Kronos 示例与量化实战指南

本目录包含 Kronos 金融时序大模型在 A 股等市场的预测脚本、回测框架与深度实战指南。

## 核心文档推荐
* 📖 **[Kronos 金融时序大模型量化实战指南](KRONOS_QUANT_PRACTICE_GUIDE.md)**
  > 涵盖自回归原理、短中长线参数搭配矩阵（T, top_p）、Lookback 黄金比例法则、1~5天分钟级（15分钟/30分钟）K线实操专题与代码模版。

## 常用脚本说明
1. **`prediction_akshare_2024-2025.py`**：基于 akshare 数据的日线预测，支持蒙特卡洛多路径采样与扇形置信区间（Confidence Ribbon）。
2. **`run_backtest_kronos.py`**：基于预测数据与交易信号的量化回测评估脚本。
3. **`get_akshare_date_2024-2025_x.py`**：A 股历史与最新行情数据下载及清洗工具。
4. **`prediction_cn_markets_day.py`**：结合 baostock / yfinance 的 A 股日线预测基准。
