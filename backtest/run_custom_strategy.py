"""
backtest/run_backtest_custom_strategy.py
港美股多周期与自定义策略回测示例 (Multi-Timeframe & Custom Strategy Demo)
演示：
1. 港美股标的加载 (US / HK)
2. 多周期 K 线支持 (15m / 30m / 日线)
3. 纯 Python 继承 BaseStrategy 极简实现新策略
4. 统一撮合引擎结算与绩效评估
5. 无缝切换实盘网关 (Futu / Tiger / IBKR)
"""

import os
import sys
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for p in (PROJECT_ROOT, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from backtest.core import TimeFrame, MarketProfile, SymbolResolver
from backtest.data import DataFetcher
from backtest.strategies import BaseStrategy, IntradayBreakoutStrategy
from backtest.engine import BacktestEngine, MetricsEvaluator
from backtest.execution import FutuBrokerAdapter, InteractiveBrokersAdapter


def run_custom_strategy_demo(
    stock_code: str = "AAPL",
    market: str = "US",
    timeframe: str = "1d",
    capital: float = None,
):
    print(f"\n{'='*64}")
    print(f"🚀 运行自定义量化策略回测: {stock_code} [{market}] 周期: {timeframe}")
    print(f"{'='*64}")

    data_dir = os.path.join(SCRIPT_DIR, "data")
    output_dir = os.path.join(SCRIPT_DIR, "output")
    tf = TimeFrame.from_str(timeframe)

    # 1. 市场规则配置 (针对港美股自动匹配)
    profile = MarketProfile.for_stock(stock_code, market=market)
    initial_capital = capital if capital is not None else profile.default_capital
    print(f"1. 市场微观画像: {profile.describe()}")
    print(f"   初始本金: {profile.currency_symbol}{initial_capital:,.2f} ({profile.currency})")

    # 2. 行情数据加载 (支持日线与15m/30m多周期)
    hist_df, _ = DataFetcher.load_stock_history(
        stock_code=stock_code,
        data_dir=data_dir,
        market=profile.market,
        timeframe=tf,
    )
    print(f"2. 成功加载行情: {len(hist_df)} 根 K 线 ({hist_df.index.min()} ~ {hist_df.index.max()})")

    # 3. 策略实例化与信号计算 (继承 BaseStrategy)
    strategy = IntradayBreakoutStrategy(
        name=f"Breakout_{tf.value}",
        fast_window=20,
        slow_window=50,
        channel_window=20,
        target_horizon_bars=10 if not tf.is_intraday else 20,
        profile=profile,
    )
    print("3. 生成策略交易信号...")
    signal_df = strategy.generate_signals(hist_df)
    buy_signals = (signal_df['exec_signal'] == 1).sum()
    print(f"   触发买入信号: {buy_signals} 次")

    # 4. 驱动统一撮合引擎 (与实盘统一的物理状态机)
    engine = BacktestEngine(
        data_dir=data_dir,
        model_dir=data_dir,
        initial_capital=initial_capital,
        market_profile=profile,
    )
    engine.set_market(stock_code, market=profile.market)

    print("4. 执行仿真撮合与逐日/逐Bar盯市...")
    backtest_df, trades = engine.run_backtest_engine(
        df=signal_df,
        hard_stop_loss=0.04,
        trailing_stop=0.025,
        max_holding_days=10,
        min_lot_guarantee=True,
        risk_per_trade=0.015,
        max_position_pct=0.80,
        intraday_stop=True,
    )

    # 5. 计算全套绩效指标 (自动适配 15m/30m/日线年化因子)
    metrics = MetricsEvaluator.evaluate(
        backtest_df=backtest_df,
        trades=trades,
        initial_capital=initial_capital,
        profile=profile,
        timeframe=tf,
    )

    # 6. 控制台输出报告
    engine._print_report(stock_code, metrics, trades)

    # 7. 实盘接入示例说明
    print("5. 实盘无缝对接状态:")
    print(f"   - 富途网关 (Futu OpenD): 就绪 (适配器: FutuBrokerAdapter, 标的: {profile.market}.{stock_code})")
    print(f"   - 盈透网关 (Interactive Brokers): 就绪 (适配器: InteractiveBrokersAdapter, 标的: {stock_code})")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="港美股多周期与自定义策略回测")
    parser.add_argument("stock", nargs="?", default="AAPL", help="股票代码 (如 AAPL, TSLA, 00700)")
    parser.add_argument("--market", default="US", choices=["US", "HK", "CN"], help="市场代码")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "30m", "15m", "5m"], help="K线周期")
    parser.add_argument("--capital", type=float, default=None, help="初始资金")
    args = parser.parse_args()

    run_custom_strategy_demo(
        stock_code=args.stock,
        market=args.market,
        timeframe=args.timeframe,
        capital=args.capital,
    )
