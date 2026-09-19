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
    """

    def __init__(self, data_dir, model_dir, initial_capital=100000):
        """
        初始化回测器

        参数:
        data_dir: 数据目录
        model_dir: 模型预测结果目录
        initial_capital: 初始资金
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.initial_capital = initial_capital
        self.results = {}

    def load_historical_data(self, stock_code):
        """
        加载历史数据
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

    def load_predictions(self, stock_code):
        """
        加载模型预测结果
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

    def align_data(self, hist_df, pred_df):
        """
        对齐历史数据和预测数据的时间范围
        """
        # 找到历史数据的最后日期
        last_hist_date = hist_df.index.max()

        # 筛选预测数据，从历史数据结束后开始
        pred_df_aligned = pred_df[pred_df.index > last_hist_date]

        if len(pred_df_aligned) == 0:
            # 如果没有未来的预测数据，使用所有预测数据
            pred_df_aligned = pred_df.copy()
            print("⚠️ 警告：预测数据没有未来的日期，使用所有预测数据")

        print(f"✅ 数据对齐: 历史数据结束于 {last_hist_date}, 预测数据从 {pred_df_aligned.index.min()} 开始")

        return pred_df_aligned

    def calculate_trading_signals(self, hist_df, pred_df, threshold=0.02):
        """
        计算交易信号
        """
        # 对齐数据
        pred_df = self.align_data(hist_df, pred_df)

        # 合并历史数据和预测数据
        combined = pd.concat([
            hist_df[['close']].rename(columns={'close': 'actual'}),
            pred_df[['predicted_close']].rename(columns={'predicted_close': 'predicted'})
        ], axis=1)

        # 计算预测收益率
        combined['pred_return'] = combined['predicted'].pct_change()

        # 生成交易信号
        combined['signal'] = 0
        combined['signal'] = np.where(combined['pred_return'] > threshold, 1,  # 买入信号
                                      np.where(combined['pred_return'] < -threshold, -1, 0))  # 卖出信号

        # 过滤信号：避免频繁交易（信号前向填充维持仓位）
        combined['position'] = combined['signal'].replace(0, np.nan).ffill().fillna(0)

        return combined

    def run_backtest(self, combined_df):
        """
        运行回测
        """
        # 初始化资金和持仓
        capital = self.initial_capital
        position = 0
        trades = []

        # 回测记录
        backtest_results = pd.DataFrame(index=combined_df.index)
        backtest_results['capital'] = capital
        backtest_results['position'] = 0
        backtest_results['returns'] = 0.0
        backtest_results['price'] = combined_df['actual'].combine_first(combined_df['predicted'])

        for i, (date, row) in enumerate(combined_df.iterrows()):
            current_price = row['actual'] if not pd.isna(row['actual']) else row['predicted']
            signal = row['position']

            # 跳过无效价格
            if pd.isna(current_price):
                continue

            # 执行交易
            if i > 0:  # 从第二天开始
                prev_position = backtest_results['position'].iloc[i - 1] if i > 0 else 0

                # 平仓信号
                if prev_position != 0 and signal == 0:
                    # 平仓
                    capital = position * current_price
                    position = 0
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': prev_position,
                        'capital': capital
                    })

                # 开仓信号
                elif prev_position == 0 and signal != 0:
                    # 计算可买股数（假设全仓交易）
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

            # 更新持仓市值
            portfolio_value = capital + position * current_price

            # 记录结果
            backtest_results.loc[date, 'capital'] = portfolio_value
            backtest_results.loc[date, 'position'] = position
            backtest_results.loc[date, 'price'] = current_price

            # 计算日收益率
            if i > 0:
                prev_value = backtest_results['capital'].iloc[i - 1]
                if prev_value > 0:
                    backtest_results.loc[date, 'returns'] = (portfolio_value - prev_value) / prev_value

        return backtest_results, trades

    def calculate_metrics(self, backtest_results, trades):
        """
        计算回测指标
        """
        returns = backtest_results['returns'].replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) == 0:
            return {
                '总收益率': 0,
                '年化收益率': 0,
                '波动率': 0,
                '夏普比率': 0,
                '最大回撤': 0,
                '胜率': 0,
                '平均交易收益': 0,
                '交易次数': 0,
                '最终资金': self.initial_capital
            }

        total_return = (backtest_results['capital'].iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        # 波动率
        volatility = returns.std() * np.sqrt(252)

        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 交易统计
        trade_returns = []
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy = buy_trades[i]
            sell = sell_trades[i]
            trade_return = (sell['price'] - buy['price']) / buy['price']
            trade_returns.append(trade_return)

        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0

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

    def plot_backtest_results(self, backtest_results, metrics, stock_code, output_dir, show_plot=True):
        """
        绘制回测结果图表
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

        # 添加指标文本
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

        # 保存图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            chart_file = os.path.join(output_dir, f'{stock_code}_backtest_results.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"📊 回测图表已保存: {chart_file}")

        if show_plot:
            plt.show()

        return fig

    def run_complete_backtest(self, stock_code, output_dir, threshold=0.02, show_plot=False):
        """
        运行完整的回测流程
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


def get_available_stocks(data_dir, model_dir):
    """
    扫描数据目录与预测目录，提取可用的股票代码列表
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
    终端命令行入口（兼容 python run_backtest_kronos.py [stock_code]）
    """
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    target_stock = sys.argv[1] if len(sys.argv) > 1 else "300418"

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
        show_plot=True
    )

    if metrics:
        print(f"\n✅ {BACKTEST_CONFIG['stock_code']} 回测完成!")
        print(f"📁 结果保存在: {BACKTEST_CONFIG['output_dir']}")


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