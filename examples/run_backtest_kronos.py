# run_backtest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体（兼容 macOS 和 Windows）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class KronosBacktester:
    """
    Kronos模型回测类
    
    【核心数据流概览】
    1. 历史行情 (hist_df: DataFrame) ──┐
                                     ├──> 对齐与信号生成 (combined_df: DataFrame) ──> 回测模拟 (run_backtest) ──> 指标归因 (metrics: dict)
    2. 模型预测 (pred_df: DataFrame) ──┘
    """

    def __init__(self, data_dir: str, model_dir: str, initial_capital: float = 100000.0):
        """
        初始化回测器

        参数与示例:
        --------------------------------------------------
        data_dir: str
            历史行情数据所在目录路径
            示例值: "/Users/jyy/Documents/github_projects/Kronos/examples/data"
        model_dir: str
            模型预测结果所在目录路径
            示例值: "/Users/jyy/Documents/github_projects/Kronos/examples/yuce"
        initial_capital: float, 默认 100000.0 (十万元)
            初始账户可用资金
            示例值: 100000.0
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.initial_capital = initial_capital
        # 缓存回测结果字典，结构示例: {"300418": {"metrics": {...}, "trades": [...]}}
        self.results = {}

    def load_historical_data(self, stock_code: str) -> pd.DataFrame:
        """
        加载并标准化历史日K线行情数据

        参数:
            stock_code (str): 股票代码，示例: "300418" 或 "03750"

        返回:
            pd.DataFrame: 以日期为索引的历史行情表格。
            -----------------------------------------------------------------------------------------
            【返回数据结构与示例数据】:
            索引类型: pd.DatetimeIndex (名称: 'date')
            列结构:
                open   (float64): 开盘价 (元)
                high   (float64): 最高价 (元)
                low    (float64): 最低价 (元)
                close  (float64): 收盘价 (元)
                volume (float64): 成交量 (股)
                amount (float64): 成交金额 (元)

            示例 DataFrame 打印样式:
                                open   high    low  close       volume        amount
            date
            2025-09-19 00:00:00 41.50  45.50  41.32  43.45  139928000.0  6.079872e+09
            2025-09-22 00:00:00 43.00  43.38  42.00  42.65   84946700.0  3.622977e+09
            2026-09-18 00:00:00 44.80  45.90  44.50  45.33   62100000.0  2.812000e+09
            -----------------------------------------------------------------------------------------
        """
        # 兼容多种历史数据文件命名
        possible_files = [
            os.path.join(self.data_dir, f"{stock_code}_stock_data.csv"),
            os.path.join(self.data_dir, f"{stock_code}_all_history.csv"),
            os.path.join(self.data_dir, f"{stock_code}.csv")
        ]

        csv_file = None
        for f in possible_files:
            if os.path.exists(f):
                csv_file = f
                break

        if csv_file is None:
            raise FileNotFoundError(f"数据文件不存在，已查找路径: {possible_files}")

        print(f"✅ 找到历史数据文件: {csv_file}")
        df = pd.read_csv(csv_file, encoding='utf-8-sig')

        # 检查列名并标准化
        column_mapping = {
            '日期': 'date',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume',
            '成交额': 'amount'
        }

        # 重命名列
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index()

        print(f"✅ 加载历史数据: {len(df)} 条记录")
        print(f"时间范围: {df.index.min()} 到 {df.index.max()}")

        return df

    def load_predictions(self, stock_code: str) -> pd.DataFrame:
        """
        加载模型前瞻预测数据

        参数:
            stock_code (str): 股票代码，示例: "300418"

        返回:
            pd.DataFrame: 以日期为索引的未来时段预测序列。
            -----------------------------------------------------------------------------------------
            【返回数据结构与示例数据】:
            索引类型: pd.DatetimeIndex (名称: 'date')
            列结构:
                predicted_close  (float64): 模型预测的收盘价 (元)
                predicted_volume (float64): 模型预测的成交量 (股，可选)

            示例 DataFrame 打印样式:
                                 predicted_close  predicted_volume
            date
            2026-09-21 00:00:00        44.214916      5.795489e+07
            2026-09-22 00:00:00        44.221474      4.851408e+07
            2026-09-23 00:00:00        45.326720      5.811489e+07
            2026-09-24 00:00:00        44.732090      6.184539e+07
            -----------------------------------------------------------------------------------------
        """
        # 尝试不同的预测文件命名
        pred_files = [
            os.path.join(self.model_dir, f"{stock_code}_kronos_predictions.csv"),
            os.path.join(self.model_dir, f"{stock_code}_detailed_predictions.csv"),
            os.path.join(self.model_dir, f"{stock_code}_comprehensive_predictions.csv"),
            os.path.join(self.model_dir, f"{stock_code}_predictions.csv")
        ]

        pred_df = None
        for pred_file in pred_files:
            if os.path.exists(pred_file):
                pred_df = pd.read_csv(pred_file, encoding='utf-8-sig')
                print(f"✅ 找到预测文件: {pred_file}")
                break

        if pred_df is None:
            raise FileNotFoundError(f"未找到预测文件，请检查目录: {self.model_dir}")

        # 标准化列名（兼容不同预测输出格式）
        column_mapping = {
            '日期': 'date',
            '预测收盘价_均值': 'predicted_close',
            '预测收盘价': 'predicted_close',
            '基础预测收盘价': 'predicted_close',
            '增强预测收盘价': 'predicted_close',
            '收盘价': 'predicted_close',
            '预测成交量': 'predicted_volume',
            '成交量': 'predicted_volume'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in pred_df.columns and new_col not in pred_df.columns:
                pred_df = pred_df.rename(columns={old_col: new_col})

        pred_df['date'] = pd.to_datetime(pred_df['date'])
        pred_df.set_index('date', inplace=True)
        pred_df = pred_df.sort_index()

        print(f"✅ 加载预测数据: {len(pred_df)} 条记录")
        print(f"预测时间范围: {pred_df.index.min()} 到 {pred_df.index.max()}")

        return pred_df

    def align_data(self, hist_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        对齐历史数据和预测数据的时间范围，确保回测仅在模型预测的“未来时区”生效

        参数:
            hist_df: 历史数据表 (DataFrame)
            pred_df: 预测数据表 (DataFrame)

        返回:
            pred_df_aligned (pd.DataFrame): 过滤后的预测数据。
            -----------------------------------------------------------------------------------------
            【示例对比】:
            输入历史范围: 2025-09-19 至 2026-09-18 (最后历史日期: 2026-09-18)
            原始预测范围: 2026-09-21 至 2026-12-29
            输出对齐范围: 2026-09-21 至 2026-12-29 (pred_df.index > 2026-09-18)
            -----------------------------------------------------------------------------------------
        """
        # 找到历史数据的最后日期 (示例: Timestamp('2026-09-18 00:00:00'))
        last_hist_date = hist_df.index.max()

        # 筛选预测数据，从历史数据结束后开始
        pred_df_aligned = pred_df[pred_df.index > last_hist_date]

        if len(pred_df_aligned) == 0:
            # 如果没有未来的预测数据，使用所有预测数据
            pred_df_aligned = pred_df.copy()
            print("⚠️ 警告：预测数据没有未来的日期，使用所有预测数据")

        print(f"✅ 数据对齐: 历史数据结束于 {last_hist_date}, 预测数据从 {pred_df_aligned.index.min()} 开始")

        return pred_df_aligned

    def calculate_trading_signals(self, hist_df: pd.DataFrame, pred_df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
        """
        基于预测收益率计算多空/持仓交易信号

        参数:
            hist_df (pd.DataFrame): 历史数据
            pred_df (pd.DataFrame): 预测数据
            threshold (float): 触发交易的预期涨跌幅阈值，默认 0.02 (即 2%)
                               例如: 预测收益率 > +2% 产生买入信号(+1)，< -2% 产生卖出信号(-1)

        返回:
            pd.DataFrame: 合并并带有交易信号的数据表 (combined_df)
            -----------------------------------------------------------------------------------------
            【返回数据结构与示例数据】:
            索引: pd.DatetimeIndex ('date')
            字段结构:
                actual      (float64): 历史真实收盘价 (未来时段为 NaN)
                predicted   (float64): 模型预测收盘价 (历史时段为 NaN)
                pred_return (float64): 预测收益率 = predicted[t] / predicted[t-1] - 1
                signal      (int64):   瞬时触发脉冲 (1: 买入, -1: 卖出, 0: 无动作)
                position    (float64): 目标持仓状态 (1.0: 多头持仓, -1.0: 空头持仓, 0.0: 空仓观望)

            示例表格展示:
                                actual  predicted  pred_return  signal  position
            date
            2026-09-18 00:00:00  45.33        NaN          NaN       0       0.0  <- 历史最后一天
            2026-09-21 00:00:00    NaN      44.21          NaN       0       0.0  <- 预测第1天(无前值)
            2026-09-22 00:00:00    NaN      44.22     0.000227       0       0.0  <- 波动 < 2%，维持空仓
            2026-09-23 00:00:00    NaN      45.33     0.025100       1       1.0  <- 涨幅 +2.51% > 2%，买入！
            2026-09-24 00:00:00    NaN      44.73    -0.013238       0       1.0  <- 跌幅 1.32% < 2%，维持持仓
            -----------------------------------------------------------------------------------------
        """
        # 对齐数据
        pred_df = self.align_data(hist_df, pred_df)

        # 合并历史数据和预测数据
        # 示例: combined 包含了历史 actual (如 45.33) 与未来 predicted (如 44.21, 45.33...)
        combined = pd.concat([
            hist_df[['close']].rename(columns={'close': 'actual'}),
            pred_df[['predicted_close']].rename(columns={'predicted_close': 'predicted'})
        ], axis=1)

        # 计算预测收益率 (例: 45.33 / 44.22 - 1 = +0.0251)
        combined['pred_return'] = combined['predicted'].pct_change()

        # 生成交易信号 (脉冲触发)
        # 阈值判断示例: pred_return = 0.0251 > 0.02 -> 产生 signal = 1
        combined['signal'] = 0
        combined['signal'] = np.where(combined['pred_return'] > threshold, 1,  # 买入信号 (+1)
                                      np.where(combined['pred_return'] < -threshold, -1, 0))  # 卖出信号 (-1)

        # 过滤信号：避免频繁交易（使用前向填充 ffill 维持持仓仓位状态）
        # 示例: [0, 0, 1, 0, 0] -> replace(0, nan) -> [nan, nan, 1, nan, nan] -> ffill() -> [nan, nan, 1, 1, 1] -> fillna(0) -> [0, 0, 1, 1, 1]
        combined['position'] = combined['signal'].replace(0, np.nan).ffill().fillna(0)

        return combined

    def run_backtest(self, combined_df: pd.DataFrame) -> tuple:
        """
        按时间序列逐步遍历并模拟交易撮合

        参数:
            combined_df (pd.DataFrame): 包含价格与持仓信号的合并数据表 (由 calculate_trading_signals 生成)

        返回:
            tuple: (backtest_results, trades)
            -----------------------------------------------------------------------------------------
            【返回值 1: backtest_results (pd.DataFrame) 账户日度时序记录】
            索引: pd.DatetimeIndex ('date')
            列结构:
                capital   (float64): 当日收盘账户资产总净值 (现金 + 持仓市值，单位: 元)
                position  (float64): 当日收盘持股数量 (正数代表多头，0 代表空仓，单位: 股)
                returns   (float64): 账户资产日收益率 (当天资产净值 / 前一天资产净值 - 1)
                price     (float64): 当日计算估值采用的标的基准价格 (元)

            示例 DataFrame 打印样式:
                                     capital  position   returns      price
            date
            2026-09-18 00:00:00  100000.0000       0.0  0.000000  45.330000  <- 初始状态
            2026-09-21 00:00:00  100000.0000       0.0  0.000000  44.214916  <- 观望维持
            2026-09-22 00:00:00  100000.0000       0.0  0.000000  44.221474  <- 观望维持
            2026-09-23 00:00:00  100000.0700    2206.0  0.000001  45.326720  <- 触发买入2206股
            2026-09-24 00:00:00   98679.0600    2206.0 -0.013210  44.732090  <- 持仓浮动净值随价格变动

            -----------------------------------------------------------------------------------------
            【返回值 2: trades (List[Dict[str, Any]]) 交易明细记录清单】
            结构示例:
            [
                {
                    'date': Timestamp('2026-09-23 00:00:00'),  # 成交日期
                    'action': 'BUY',                            # 动作: 'BUY' (买入建仓) 或 'SELL' (卖出平仓)
                    'price': 45.32672,                          # 成交价格 (元)
                    'shares': 2206.0,                           # 成交股数 (正数)
                    'capital': 0.0700                           # 交易执行后剩余的现金余额 (元)
                }
            ]
            -----------------------------------------------------------------------------------------
        """
        # 初始化资金和持仓
        # capital (float): 可用现金余额，初始值示例 100000.0
        capital = self.initial_capital
        # position (float): 当前持股数量，初始值 0.0
        position = 0
        # trades (list): 交易历史流水账本列表
        trades = []

        # 回测记录表格构建
        backtest_results = pd.DataFrame(index=combined_df.index)
        backtest_results['capital'] = capital
        backtest_results['position'] = 0
        backtest_results['returns'] = 0.0
        backtest_results['price'] = combined_df['actual'].combine_first(combined_df['predicted'])

        for i, (date, row) in enumerate(combined_df.iterrows()):
            # current_price (float): 当前交易日参考价格，示例 45.33
            current_price = row['actual'] if not pd.isna(row['actual']) else row['predicted']
            # signal (float): 当日持仓目标，1.0(持有多仓), -1.0(持有空仓), 0.0(空仓)
            signal = row['position']

            # 跳过无效价格
            if pd.isna(current_price):
                continue

            # 执行交易模拟（T+0/T+1 简化模型，按当日信号在当前价格全仓调仓）
            if i > 0:  # 从第二天开始
                prev_position = backtest_results['position'].iloc[i - 1] if i > 0 else 0

                # 平仓信号: 上一日有持仓，而今日目标持仓为 0
                if prev_position != 0 and signal == 0:
                    capital = position * current_price
                    position = 0
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': prev_position,
                        'capital': capital
                    })

                # 开仓信号: 上一日无持仓，而今日目标持仓非 0 (如 signal == 1)
                elif prev_position == 0 and signal != 0:
                    # 计算可买股数（假设全仓满仓买入）
                    shares = int(capital / current_price)
                    if shares > 0:
                        position = shares * signal
                        capital -= shares * current_price
                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares * signal,
                            'capital': capital
                        })

            # 更新当日持仓总市值 (portfolio_value: float = 现金 + 持仓股数 * 收盘价)
            portfolio_value = capital + position * current_price

            # 记录当日结果
            backtest_results.loc[date, 'capital'] = portfolio_value
            backtest_results.loc[date, 'position'] = position
            backtest_results.loc[date, 'price'] = current_price

            # 计算日收益率 returns = (V_t - V_{t-1}) / V_{t-1}
            if i > 0:
                prev_value = backtest_results['capital'].iloc[i - 1]
                if prev_value > 0:
                    backtest_results.loc[date, 'returns'] = (portfolio_value - prev_value) / prev_value

        return backtest_results, trades

    def calculate_metrics(self, backtest_results: pd.DataFrame, trades: list) -> dict:
        """
        计算量化回测核心绩效与风险归因指标

        参数:
            backtest_results (pd.DataFrame): run_backtest 返回的每日净值与持仓表
            trades (list): 交易历史流水账列表

        返回:
            dict: 包含各项量化指标的字典。
            -----------------------------------------------------------------------------------------
            【返回数据结构与示例数据】:
            {
                '总收益率': -0.0581,         # float: (期末资产 - 初始资金) / 初始资金, 如 -5.81%
                '年化收益率': -0.0478,       # float: 复利年化收益率 (基于 252 交易日换算), 如 -4.78%
                '波动率': 0.1207,           # float: 收益率标准差 * sqrt(252), 年化波动率 12.07%
                '夏普比率': -0.6446,         # float: (年化收益率 - 无风险利率3%) / 波动率
                '最大回撤': -0.1824,         # float: 历史资产高点到低点的最大跌幅比例, 如 -18.24%
                '胜率': 0.60,               # float: 盈利交易次数 / 总完成对冲交易次数, 如 60.0%
                '平均交易收益': 0.015,       # float: 每笔已平仓交易的平均收益率, 如 1.50%
                '交易次数': 1,               # int: 累计产生的交易委托动作数 (买入+卖出)
                '最终资金': 94187.91        # float: 回测截止时账户资产最终净值 (元)
            }
            -----------------------------------------------------------------------------------------
        """
        returns = backtest_results['returns'].replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) == 0:
            return {
                '总收益率': 0.0,
                '年化收益率': 0.0,
                '波动率': 0.0,
                '夏普比率': 0.0,
                '最大回撤': 0.0,
                '胜率': 0.0,
                '平均交易收益': 0.0,
                '交易次数': 0,
                '最终资金': self.initial_capital
            }

        # 1. 累计总收益率 (Total Return)
        total_return = (backtest_results['capital'].iloc[-1] - self.initial_capital) / self.initial_capital
        # 2. 年化收益率 (Annualized Return, 按一年252个交易日复利推算)
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        # 3. 年化波动率 (Volatility)
        volatility = returns.std() * np.sqrt(252)

        # 4. 夏普比率 (Sharpe Ratio, 假设中国国债无风险利率为 3%)
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # 5. 最大回撤 (Maximum Drawdown)
        # cumulative_returns (Series): 资产累计净值曲线 (以1.0为起点)
        cumulative_returns = (1 + returns).cumprod()
        # peak (Series): 历史最高水位线
        peak = cumulative_returns.expanding().max()
        # drawdown (Series): 当前相对最高点的回撤比率序列
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 6. 交易统计 (按买入-卖出配对计算单笔盈亏)
        trade_returns = []
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy = buy_trades[i]
            sell = sell_trades[i]
            trade_return = (sell['price'] - buy['price']) / buy['price']
            trade_returns.append(trade_return)

        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0.0
        avg_trade_return = float(np.mean(trade_returns)) if trade_returns else 0.0

        metrics = {
            '总收益率': total_return,
            '年化收益率': annual_return,
            '波动率': volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '胜率': win_rate,
            '平均交易收益': avg_trade_return,
            '交易次数': len(trades),
            '最终资金': backtest_results['capital'].iloc[-1]
        }

        return metrics

    def plot_backtest_results(self, backtest_results: pd.DataFrame, metrics: dict, stock_code: str, output_dir: str, show_plot: bool = True):
        """
        绘制回测三联图（资金净值曲线、累计收益 vs 基准、动态回撤区间）

        参数:
            backtest_results (pd.DataFrame): 包含 capital, returns, price 等字段
            metrics (dict): calculate_metrics 返回的指标字典
            stock_code (str): 标的代码，示例 "300418"
            output_dir (str): 图表输出保存目录，示例 "./examples/backtest"
            show_plot (bool): 是否调用 plt.show() 弹窗显示，GUI/Streamlit下通常设为 False

        返回:
            matplotlib.figure.Figure: 生成的画布对象
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # 1. 资金曲线
        ax1.plot(backtest_results.index, backtest_results['capital'],
                 linewidth=2, label='策略资金曲线', color='#1f77b4')
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--',
                    label=f'初始资金 ({self.initial_capital:,.0f}元)')
        ax1.set_ylabel('资金 (元)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'{stock_code} Kronos模型回测结果', fontsize=14, fontweight='bold')

        # 2. 收益率曲线
        cumulative_returns = (1 + backtest_results['returns'].fillna(0)).cumprod()
        ax2.plot(backtest_results.index, cumulative_returns,
                 linewidth=2, label='策略累计收益', color='#2ca02c')

        # 基准收益（买入持有）
        price_returns = backtest_results['price'].pct_change().fillna(0)
        benchmark_returns = (1 + price_returns).cumprod()
        ax2.plot(backtest_results.index, benchmark_returns,
                 linewidth=2, label='基准收益（买入持有）', color='#ff7f0e', alpha=0.7)

        ax2.set_ylabel('累计收益', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 回撤曲线
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        ax3.fill_between(backtest_results.index, drawdown, 0,
                         alpha=0.3, color='red', label='回撤')
        ax3.set_ylabel('回撤', fontsize=12)
        ax3.set_xlabel('日期', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 添加指标文本卡片
        metrics_text = (
            f"总收益率: {metrics['总收益率']:.2%}\n"
            f"年化收益率: {metrics['年化收益率']:.2%}\n"
            f"夏普比率: {metrics['夏普比率']:.2f}\n"
            f"最大回撤: {metrics['最大回撤']:.2%}\n"
            f"胜率: {metrics['胜率']:.2%}\n"
            f"交易次数: {metrics['交易次数']}\n"
            f"最终资金: {metrics['最终资金']:,.0f}元"
        )

        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                    facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()

        # 保存图表至指定目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            chart_file = os.path.join(output_dir, f'{stock_code}_backtest_results.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"📊 回测图表已保存: {chart_file}")

        if show_plot:
            plt.show()

        return fig

    def run_complete_backtest(self, stock_code: str, output_dir: str, threshold: float = 0.02, show_plot: bool = False):
        """
        全流程一键回测管线：加载行情与预测 -> 信号生成 -> 模拟撮合 -> 指标计算 -> 图表渲染

        参数与示例:
            stock_code (str): 标的代码，示例: "300418"
            output_dir (str): 图表输出路径，示例: "./examples/backtest"
            threshold (float): 交易信号灵敏度阈值，示例: 0.02 (即 2%)
            show_plot (bool): 是否弹窗展示图表，默认 False

        返回:
            tuple: (metrics, backtest_results, trades, fig)
                metrics: dict, 量化指标字典
                backtest_results: pd.DataFrame, 每日资金与持仓变化表
                trades: list, 交易行为列表
                fig: matplotlib.figure.Figure, 渲染的图表对象
        """
        print(f"🎯 开始 {stock_code} 回测分析")
        print("=" * 50)

        try:
            # 1. 加载数据
            print("步骤1: 加载历史数据和预测数据...")
            hist_df = self.load_historical_data(stock_code)
            pred_df = self.load_predictions(stock_code)

            # 2. 计算交易信号
            print("步骤2: 计算交易信号...")
            combined_df = self.calculate_trading_signals(hist_df, pred_df, threshold)

            # 3. 运行回测
            print("步骤3: 运行回测...")
            backtest_results, trades = self.run_backtest(combined_df)

            # 4. 计算指标
            print("步骤4: 计算回测指标...")
            metrics = self.calculate_metrics(backtest_results, trades)

            # 5. 绘制结果
            print("步骤5: 生成回测图表...")
            fig = self.plot_backtest_results(backtest_results, metrics, stock_code, output_dir, show_plot=show_plot)

            # 6. 打印详细报告
            print("\n" + "=" * 70)
            print(f"📊 {stock_code} 回测报告")
            print("=" * 70)
            for key, value in metrics.items():
                if isinstance(value, float):
                    if '率' in key or '收益' in key or '回撤' in key:
                        print(f"  {key}: {value:.2%}")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

            print(f"\n交易记录 (共{len(trades)}次交易):")
            for i, trade in enumerate(trades[-10:], 1):  # 显示最后10次交易
                print(f"  交易{i}: {trade['date'].strftime('%Y-%m-%d')} "
                      f"{trade['action']} {abs(trade['shares'])}股 @ {trade['price']:.2f}元")

            return metrics, backtest_results, trades, fig

        except Exception as e:
            print(f"❌ 回测过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None


def get_available_stocks(data_dir: str, model_dir: str) -> list:
    """
    扫描历史数据目录与模型预测目录，提取同时或任一存在的可用股票代码集合

    参数:
        data_dir (str): 历史数据目录，示例: "./examples/data"
        model_dir (str): 预测数据目录，示例: "./examples/yuce"

    返回:
        list: 排好序的标的代码列表。
        -----------------------------------------------------------------------------------------
        【返回数据示例】:
        ['300418', '600519', '000001']
        -----------------------------------------------------------------------------------------
    """
    stocks = set()
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('_stock_data.csv') or f.endswith('_all_history.csv'):
                code = f.split('_')[0]
                if code.isalnum():
                    stocks.add(code)
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith('_predictions.csv') or f.endswith('_detailed_predictions.csv') or f.endswith('_comprehensive_predictions.csv'):
                code = f.split('_')[0]
                if code.isalnum():
                    stocks.add(code)
    stock_list = sorted(list(stocks))
    return stock_list if stock_list else ["300418"]


def streamlit_main():
    """
    Streamlit Web 交互看板入口
    
    【核心会话状态 st.session_state["backtest_data"] 数据结构示例】:
    {
        "stock_code": "300418",                 # str: 当前回测股票代码
        "initial_capital": 100000.0,            # float: 初始本金 (元)
        "metrics": {                            # dict: 量化评价指标字典
            "总收益率": -0.0581,
            "年化收益率": -0.0478,
            "夏普比率": -0.64,
            "最大回撤": -0.1824,
            "胜率": 0.0,
            "平均交易收益": 0.0,
            "交易次数": 1,
            "最终资金": 94187.91
        },
        "backtest_results": pd.DataFrame,       # DataFrame: 包含 capital, position, returns, price
        "trades": [                             # list: 交易记录列表
            {
                'date': Timestamp('2026-09-23 00:00:00'),
                'action': 'BUY',
                'price': 45.33,
                'shares': 2206.0,
                'capital': 0.07
            }
        ],
        "fig": matplotlib.figure.Figure         # Figure: 资金/收益/回撤三合一图表
    }
    """
    import streamlit as st

    st.set_page_config(
        page_title="Kronos 量化回测看板",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Kronos 量化模型回测系统")
    st.caption("基于时间序列预测模型的量化交易信号生成、组合回测与风险归因看板")

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(CURRENT_DIR, "data")
    model_dir = os.path.join(CURRENT_DIR, "yuce")
    output_dir = os.path.join(CURRENT_DIR, "backtest")

    available_stocks = get_available_stocks(data_dir, model_dir)

    with st.sidebar:
        st.header("⚙️ 回测策略配置")

        default_idx = available_stocks.index("300418") if "300418" in available_stocks else 0
        stock_option = st.selectbox(
            "选择股票标的",
            options=available_stocks + ["自定义输入"],
            index=default_idx
        )

        if stock_option == "自定义输入":
            selected_stock = st.text_input("输入股票代码 (如 300418)", value="300418")
        else:
            selected_stock = stock_option

        initial_capital = st.number_input(
            "初始账户资金 (元)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )

        threshold_pct = st.slider(
            "交易触发阈值 (%)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="当模型预测的预期涨跌幅超过该阈值时触发买入或卖出信号"
        )
        # threshold (float): 交易信号阈值，示例 0.02
        threshold = threshold_pct / 100.0

        run_btn = st.button("🚀 运行回测分析", type="primary")

    # 首次加载或点击按钮时执行回测
    if "backtest_data" not in st.session_state or run_btn:
        with st.spinner(f"正在加载 {selected_stock} 历史数据并运行回测..."):
            backtester = KronosBacktester(
                data_dir=data_dir,
                model_dir=model_dir,
                initial_capital=initial_capital
            )
            metrics, backtest_results, trades, fig = backtester.run_complete_backtest(
                stock_code=selected_stock,
                output_dir=output_dir,
                threshold=threshold,
                show_plot=False
            )
            if metrics is not None:
                st.session_state["backtest_data"] = {
                    "stock_code": selected_stock,
                    "initial_capital": initial_capital,
                    "metrics": metrics,
                    "backtest_results": backtest_results,
                    "trades": trades,
                    "fig": fig
                }
            else:
                st.error(f"❌ 未能完成 {selected_stock} 的回测，请检查数据文件和预测文件是否存在！")

    if "backtest_data" in st.session_state:
        data = st.session_state["backtest_data"]
        metrics = data["metrics"]
        backtest_results = data["backtest_results"]
        trades = data["trades"]
        fig = data["fig"]
        stock_code = data["stock_code"]
        init_cap = data.get("initial_capital", 100000)

        # 1. 核心量化指标卡片
        with st.container(border=True):
            st.subheader(f"📊 {stock_code} 核心量化绩效概览")
            c1, c2, c3, c4, c5 = st.columns(5)

            tot_ret = metrics['总收益率']
            c1.metric(
                "总收益率",
                f"{tot_ret:.2%}",
                delta=f"{tot_ret:.2%}",
                delta_color="normal"
            )
            c2.metric(
                "年化收益率",
                f"{metrics['年化收益率']:.2%}"
            )
            c3.metric(
                "夏普比率",
                f"{metrics['夏普比率']:.2f}"
            )
            c4.metric(
                "最大回撤",
                f"{metrics['最大回撤']:.2%}",
                delta_color="inverse"
            )
            final_cap = metrics['最终资金']
            c5.metric(
                "最终净值 (元)",
                f"¥{final_cap:,.2f}",
                delta=f"{(final_cap - init_cap):+,.2f}元"
            )

        # 统计小面板
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.info(f"📈 **年化波动率**: {metrics['波动率']:.2%}")
        sc2.info(f"🎯 **交易胜率**: {metrics['胜率']:.2%}")
        sc3.info(f"🔄 **总交易次数**: {metrics['交易次数']} 次")
        sc4.info(f"💰 **单笔平均收益**: {metrics['平均交易收益']:.2%}")

        # 2. 标签页展示
        tab_chart, tab_trades, tab_data = st.tabs([
            "📈 综合回测曲线 (资金/基准/回撤)",
            "📋 交易执行明细",
            "📑 每日账户净值记录"
        ])

        with tab_chart:
            if fig is not None:
                st.pyplot(fig)
            else:
                st.warning("暂无图表数据")

        with tab_trades:
            if trades:
                trade_df = pd.DataFrame(trades)
                if 'shares' in trade_df.columns:
                    trade_df['shares'] = trade_df['shares'].apply(lambda x: abs(int(x)))
                if 'price' in trade_df.columns:
                    trade_df['price'] = trade_df['price'].apply(lambda x: f"¥{x:.2f}")
                if 'capital' in trade_df.columns:
                    trade_df['capital'] = trade_df['capital'].apply(lambda x: f"¥{x:,.2f}")
                trade_df.rename(columns={
                    'date': '成交日期',
                    'action': '买卖方向',
                    'price': '成交单价',
                    'shares': '成交股数',
                    'capital': '变动后账户资金'
                }, inplace=True)
                st.dataframe(trade_df, width="stretch")
            else:
                st.info("在当前阈值与时序区间内，策略未触发任何成交动作。")

        with tab_data:
            display_df = backtest_results.copy()
            st.dataframe(display_df, width="stretch")


def cli_main():
    """
    终端命令行入口
    
    运行示例:
        python examples/run_backtest_kronos.py 300418
    """
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # target_stock (str): 目标标的代码，默认 "300418"
    target_stock = sys.argv[1] if len(sys.argv) > 1 else "300418"

    # BACKTEST_CONFIG (dict): 回测全局执行配置项
    # 示例结构:
    # {
    #     "stock_code": "300418",
    #     "data_dir": "/Users/.../examples/data",
    #     "model_dir": "/Users/.../examples/yuce",
    #     "output_dir": "/Users/.../examples/backtest",
    #     "initial_capital": 100000.0,
    #     "threshold": 0.02
    # }
    BACKTEST_CONFIG = {
        "stock_code": target_stock,
        "data_dir": os.path.join(CURRENT_DIR, "data"),
        "model_dir": os.path.join(CURRENT_DIR, "yuce"),
        "output_dir": os.path.join(CURRENT_DIR, "backtest"),
        "initial_capital": 100000,
        "threshold": 0.02
    }

    print("🤖 Kronos模型回测系统 (CLI 模式)")
    print("=" * 50)
    print(f"回测股票: {BACKTEST_CONFIG['stock_code']}")
    print(f"初始资金: {BACKTEST_CONFIG['initial_capital']:,.0f}元")
    print(f"交易阈值: {BACKTEST_CONFIG['threshold']:.1%}")
    print()

    backtester = KronosBacktester(
        data_dir=BACKTEST_CONFIG["data_dir"],
        model_dir=BACKTEST_CONFIG["model_dir"],
        initial_capital=BACKTEST_CONFIG["initial_capital"]
    )

    metrics, results, trades, fig = backtester.run_complete_backtest(
        stock_code=BACKTEST_CONFIG["stock_code"],
        output_dir=BACKTEST_CONFIG["output_dir"],
        threshold=BACKTEST_CONFIG["threshold"],
        show_plot=False
    )

    if metrics:
        print(f"\n✅ {BACKTEST_CONFIG['stock_code']} 回测完成!")
        print(f"📁 图表与结果保存在: {BACKTEST_CONFIG['output_dir']}")


def main():
    """
    自适应主函数：自动识别 Streamlit 运行环境与 CLI 命令行环境
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx() is not None
    except Exception:
        in_streamlit = False

    if in_streamlit:
        streamlit_main()
    else:
        cli_main()


if __name__ == "__main__":
    main()