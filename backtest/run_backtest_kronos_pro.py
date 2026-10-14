"""
run_backtest_kronos_pro.py
Kronos Pro 工业级量化回测中枢门面 (Facade)
支持：
- A股 / 港股 / 美股 全真制度回测 (微观摩擦/涨跌停/T+N/整手/多币种)
- 1d / 30m / 15m 多时间粒度 K 线回测与年化校准
- 策略插件化架构与实盘网关对接
- 命令行 CLI 与 Streamlit 交互式量化中枢
"""

from __future__ import annotations

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from typing import Union, Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")
for _p in (PROJECT_ROOT, SCRIPT_DIR, EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def resolve_data_dir() -> str:
    """回测行情数据存储目录：全部固定收敛至 backtest/data"""
    local_data = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(local_data, exist_ok=True)
    return local_data


def resolve_pred_dir() -> str:
    """回测预测数据存储目录：全部固定收敛至 backtest/yuce"""
    local_pred = os.path.join(SCRIPT_DIR, "yuce")
    os.makedirs(local_pred, exist_ok=True)
    return local_pred


# ---------------------------------------------------------------------------
# 领域模型、市场规则画像与日志基础设施 (重导出保持 100% 向下兼容)
# ---------------------------------------------------------------------------
from backtest.core import (
    SymbolResolver,
    MarketProfile,
    HK_COMMON_LOT_SIZES,
    HK_STAMP_DUTY_CUTOVER,
    compute_open_limit_flags,
    detect_market_type,
    validate_stock_code,
    TimeFrame,
    MarketType,
    OrderSide,
    OrderType,
    OrderStatus,
    Order,
    Position,
    Signal,
    Bar,
    TradeRecord,
    setup_logger,
    attach_stock_logger,
    detach_stock_logger,
    set_logging_state,
    clear_logs,
    get_annualization_factor,
    detect_timeframe,
    resample_bars,
)

# ---------------------------------------------------------------------------
# 数据层、策略层、执行层与引擎层 (模块化内核重导出)
# ---------------------------------------------------------------------------
from backtest.data import (
    CacheManager,
    DataFetcher,
    KronosDataFetcher,
    KronosPredictorPipeline,
)

from backtest.strategies import (
    BaseStrategy,
    KronosStrategy,
    compute_dynamic_rolling_quantile,
    IntradayBreakoutStrategy,
)

from backtest.execution import (
    BaseBroker,
    SimulatedBroker,
    RiskBudgetSizer,
    ExitStateMachine,
    LiveBrokerAdapter,
    FutuBrokerAdapter,
    InteractiveBrokersAdapter,
)

from backtest.engine import (
    MetricsEvaluator,
    SkillEvaluator,
    BacktestEngine,
    KronosBacktesterPro,
)

from backtest.visualizer import (
    inject_continuous_crosshair_script,
    StaticVisualizer,
    plot_backtest_results,
    InteractiveVisualizer,
    plot_backtest_interactive,
)

# 获取可用股票扫描函数 (向下兼容)
get_available_stocks = DataFetcher.get_available_stocks

logger = logging.getLogger("KronosBacktest")

def streamlit_main():
    """Streamlit 交互回测看板（工业级 Pro 版）"""
    from backtest.ui.dashboard_single_view import render_single_backtest_dashboard
    render_single_backtest_dashboard()


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def cli_main():
    """
    命令行入口。

    用法:
        # 模式1：自动获取数据 + Walk-Forward 预测 + 回测（推荐，需要 Kronos 模型）
        python backtest/run_backtest_kronos_pro.py 300418 --auto

        # 可选参数（--auto 模式）:
        #   --test-days N       回测验证窗口长度，默认 60 个交易日
        #   --lookback N        每步预测的历史窗口，默认 120 个交易日
        #   --step-size N       Walk-Forward 滑动步长，默认 1（逐日），>1 加速但精度略降
        #   --samples N         蒙特卡洛采样数，默认 3
        #   --history-days N    拉取历史数据总天数，默认 400

        # 模式2：使用现有预测文件回测（要求预测文件与历史有 >=10 天重叠）
        python backtest/run_backtest_kronos_pro.py 300418
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Kronos Pro 量化回测引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stock_code", nargs="?", default="300418",
                        help="股票代码，例如 300418（默认）")
    parser.add_argument("--auto", action="store_true",
                        help="自动模式：调用 Kronos 模型做 Walk-Forward 预测后再回测")
    parser.add_argument("--horizon", type=str, default="auto",
                        help="预测时界跨度：auto (极值自适应，默认) 或 整数天数 (1=单日超短, 3=短线波段, 5=标准波段)")
    parser.add_argument("--max-auto-horizon", type=int, default=10,
                        help="Auto 模式下向前推演最大天数，默认 10")
    parser.add_argument("--test-days", type=int, default=120,
                        help="回测验证窗口长度（交易日数），默认 120 (约半年)")
    parser.add_argument("--lookback", type=int, default=120,
                        help="每步预测的历史窗口长度（交易日数），默认 120")
    parser.add_argument("--step-size", type=int, default=1,
                        help="Walk-Forward 步长，默认 1（逐日精确）。>1 跳步插值仅供预览，插值行会被排除在评估外")
    parser.add_argument("--samples", type=int, default=20,
                        help="蒙特卡洛采样路径数，默认 20（分位数估计所需的最低样本量）")
    parser.add_argument("--history-days", type=int, default=700,
                        help="拉取历史行情的总天数，默认 700 (自然日)")
    parser.add_argument("--capital", type=float, default=None,
                        help="初始资金（缺省按市场自动匹配：CN=10万, HK=30万, US=5万）")
    parser.add_argument("--z-threshold", type=float, default=0.70,
                        help="Z-Score 入场阈值，默认 0.70 (推荐 0.65~0.75，黄金平衡点)")
    parser.add_argument("--hard-stop", type=float, default=0.04,
                        help="可配置硬止损线，默认 0.04 (4%%)；同时约束买入侧预测最大下探过滤")
    parser.add_argument("--trail-stop", type=float, default=0.025,
                        help="移动止盈回撤，默认 0.025 (2.5%%)")
    parser.add_argument("--max-hold", type=int, default=15,
                        help="最大持仓天数，默认 15 (Auto模式下以波峰止盈优先)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="强制从网络重新拉取行情数据（忽略本地缓存）")
    parser.add_argument("--entry-percentile", type=float, default=0.80,
                        help="入场分位阈值，默认 0.80（取该标的自身信号分布的滚动前20%%）")
    parser.add_argument("--calib-window", type=int, default=60,
                        help="分位阈值与去偏的基准滚动校准窗口，默认 60 交易日")
    parser.add_argument("--no-dynamic-calib", action="store_true",
                        help="关闭波动率自适应校准窗口（默认开启：高波动收缩至20~30天，平稳期维持60天）")
    parser.add_argument("--calib-window-min", type=int, default=20,
                        help="极端高波动率下校准窗口收缩下限，默认 20 交易日")
    parser.add_argument("--vol-high-thresh", type=float, default=0.40,
                        help="极端高波动率年化阈值（默认 0.40 即 40%%，达到或超过时窗口收缩至下限）")
    parser.add_argument("--vol-low-thresh", type=float, default=0.20,
                        help="平稳低波动率年化阈值（默认 0.20 即 20%%，低于或等于时保持基准窗口）")
    parser.add_argument("--risk-per-trade", type=float, default=0.01,
                        help="单笔风险预算占净值比例，默认 0.01 (1%%)")
    parser.add_argument("--max-position", type=float, default=0.80,
                        help="单标的仓位上限，默认 0.80")
    parser.add_argument("--no-min-lot-guarantee", action="store_true",
                        help="关闭一手保底机制（默认开启，港美股高单价标的自适应）")
    parser.add_argument("--no-intraday-stop", action="store_true",
                        help="关闭盘中止损（默认开启，遵守市场 T+N 合规）")
    parser.add_argument("--require-skill", action="store_true",
                        help="预测技能体检未通过时直接中止回测")
    parser.add_argument("--market", type=str, default=None, choices=[None, 'CN', 'HK', 'US'],
                        help="强制指定市场规则，默认按股票代码自动推断")
    parser.add_argument("--particle-filter", "--pf", action="store_true",
                        help="启用粒子滤波 (玩法B)：以最近真实走势检验蒙特卡洛预测路径并动态重分配权重")
    parser.add_argument("--pf-window", type=int, default=3,
                        help="粒子滤波验证窗口天数，默认 3")
    parser.add_argument("--pf-threshold", type=float, default=0.35,
                        help="粒子滤波置信度准入阈值，默认 0.35")
    parser.add_argument("--no-log", action="store_true",
                        help="关闭本地文件日志落盘（仅控制台输出，不生成 .log 文件）")
    parser.add_argument("--clear-log", action="store_true",
                        help="运行前回测前，清空当前标的的独立日志文件")
    parser.add_argument("--clear-all-logs", action="store_true",
                        help="清空 output 目录下所有回测日志文件（包括 kronos_backtest.log 与全部标的日志）")

    args = parser.parse_args()
    target_stock = args.stock_code

    backtest_dir = os.path.join(SCRIPT_DIR, "output")

    if args.clear_all_logs:
        cleared = clear_logs(log_dir=backtest_dir, clear_all=True)
        print(f"🧹 已清空全部回测日志 (共 {len(cleared)} 个文件):")
        for p in cleared:
            print(f"   - {p}")
        if len(sys.argv) <= 2 or sys.argv[1] == "--clear-all-logs":
            sys.exit(0)

    if args.clear_log:
        cleared = clear_logs(log_dir=backtest_dir, stock_code=target_stock)
        if cleared:
            print(f"🧹 已清空标的 [{target_stock}] 历史运行日志: {cleared[0]}")
        else:
            print(f"ℹ️ 标的 [{target_stock}] 暂无历史日志或已清空")

    if args.no_log:
        set_logging_state(enabled=True, file_logging=False, logger=logger, log_dir=backtest_dir)
        print("📝 日志文件落盘已关闭 (--no-log)，仅终端输出控制台信息。")
    else:
        set_logging_state(enabled=True, file_logging=True, logger=logger, log_dir=backtest_dir)

    horizon_val = args.horizon
    if isinstance(horizon_val, str) and horizon_val.isdigit():
        horizon_val = int(horizon_val)

    prof = MarketProfile.for_stock(target_stock, market=args.market)
    initial_capital = args.capital if args.capital is not None else prof.default_capital

    print("=" * 60)
    print("  Kronos Pro 量化回测引擎 (CLI)")
    print("=" * 60)
    print(f"  标的代码  : {target_stock} [{prof.market}市场]")
    print(f"  初始本金  : {prof.currency_symbol}{initial_capital:,.2f} ({prof.currency})")
    print(f"  运行模式  : {'[AUTO] 自动获取数据 + Walk-Forward 预测 + 回测' if args.auto else '[MANUAL] 使用现有预测文件回测'}")
    print(f"  预测时界  : {'Auto (极值自适应)' if horizon_val == 'auto' else f'{horizon_val} 天'}")
    if args.particle_filter:
        print(f"  粒子滤波  : 已启用 (玩法B / 验证窗口={args.pf_window}天 / 门槛={args.pf_threshold:.0%})")
    print()

    data_dir    = resolve_data_dir()
    pred_dir    = resolve_pred_dir()

    hist_df = None
    pred_df = None

    # ----------------------------------------------------------------
    # AUTO 模式：自动拉取行情 + Walk-Forward 预测
    # ----------------------------------------------------------------
    if args.auto:
        h_desc = f"Auto(最大推演{args.max_auto_horizon}天)" if horizon_val == 'auto' else f"{horizon_val}天"
        print(f"[AUTO] 参数：horizon={h_desc}, test_days={args.test_days}, lookback={args.lookback}, "
              f"step_size={args.step_size}, samples={args.samples}, "
              f"history_days={args.history_days}, particle_filter={args.particle_filter}")
        print()
        fetcher = KronosDataFetcher()
        try:
            hist_df, pred_df = fetcher.run_auto_pipeline(
                stock_code=target_stock,
                data_dir=data_dir,
                pred_output_dir=pred_dir,
                history_days=args.history_days,
                test_days=args.test_days,
                lookback=args.lookback,
                step_size=args.step_size,
                sample_count=args.samples,
                forecast_horizon=horizon_val,
                max_auto_horizon=args.max_auto_horizon,
                force_refresh=args.force_refresh,
                particle_filter=args.particle_filter,
                pf_val_window=args.pf_window,
            )
        except Exception as e:
            print(f"\n❌ AUTO 模式失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # ----------------------------------------------------------------
    # 执行回测
    # ----------------------------------------------------------------
    backtester = KronosBacktesterPro(
        data_dir=data_dir,
        model_dir=pred_dir,
        initial_capital=initial_capital,
        enable_log=not args.no_log,
    )

    if args.auto and hist_df is not None and pred_df is not None:
        # AUTO 模式：直接用已获取的数据，跳过文件加载
        backtester.set_market(target_stock, market=args.market, log_dir=backtest_dir, enable_log=not args.no_log)
        clean_code = getattr(backtester, 'current_stock_code', target_stock)
        print(f"   市场规则: {backtester.profile.describe()}")

        logger.info(f"[{clean_code}] ==================== KRONOS QUANT BACKTEST START (AUTO) ====================")
        logger.info(f"[{clean_code}] 标的代码: {target_stock} | 市场规则: {backtester.profile.describe()} | 初始资金: {backtester.profile.currency_symbol}{backtester.initial_capital:,.2f}")
        logger.info(f"[{clean_code}] 策略配置: 止损={args.hard_stop:.2%}, 移动止盈={args.trail_stop:.2%}, 最大持仓={args.max_hold}天, 动态校准={'开启' if not args.no_dynamic_calib else '关闭'}, 风险预算={args.risk_per_trade:.2%}")

        print("\n[Step 2] 时序对齐 & 信号生成")
        signal_df = backtester.align_and_generate_signals(
            hist_df, pred_df,
            z_score_threshold=args.z_threshold,
            forecast_horizon=horizon_val,
            max_auto_horizon=args.max_auto_horizon if hasattr(args, 'max_auto_horizon') else 10,
            hard_stop_loss=args.hard_stop,
            entry_percentile=args.entry_percentile,
            calib_window=args.calib_window,
            dynamic_calib=not args.no_dynamic_calib,
            calib_window_min=args.calib_window_min,
            vol_high_thresh=args.vol_high_thresh,
            vol_low_thresh=args.vol_low_thresh,
            particle_filter=args.particle_filter,
            pf_confidence_threshold=args.pf_threshold,
        )

        print("\n[Step 2b] 预测技能体检")
        skill = backtester.evaluate_prediction_skill(signal_df, hard_stop_loss=args.hard_stop)
        backtester.print_skill_report(skill, target_stock)
        if args.require_skill and not skill.get('tradable', False):
            reason = skill.get('error') or '; '.join(skill.get('不通过原因', []))
            print(f"\n❌ [准入拦截] {target_stock} 未通过预测技能体检，已中止回测。\n   原因: {reason}")
            sys.exit(2)

        print("\n[Step 3] 逐日撮合状态机")
        backtest_df, trades = backtester.run_backtest_engine(
            signal_df,
            hard_stop_loss=args.hard_stop,
            trailing_stop=args.trail_stop,
            max_holding_days=args.max_hold,
            min_lot_guarantee=not args.no_min_lot_guarantee,
            risk_per_trade=args.risk_per_trade,
            max_position_pct=args.max_position,
            intraday_stop=not args.no_intraday_stop,
        )
        print("\n[Step 4] 绩效指标计算")
        metrics = backtester.evaluate_metrics(backtest_df, trades)
        metrics['_skill'] = skill
        print("\n[Step 5] 生成图表")
        backtester.plot_backtest_results(
            signal_df, backtest_df, metrics, trades,
            target_stock, backtest_dir, show_plot=False,
        )
        print("\n[Step 6] 导出量化分析数据集")
        backtester.export_backtest_artifacts(
            stock_code=target_stock,
            output_dir=backtest_dir,
            signal_df=signal_df,
            backtest_df=backtest_df,
            trades=trades,
            metrics=metrics,
            params=vars(args),
        )
        backtester._print_report(target_stock, metrics, trades)
        logger.info(f"[{clean_code}] ==================== KRONOS QUANT BACKTEST COMPLETED ====================\n")
    else:
        # MANUAL 模式：从文件加载
        result = backtester.run_complete_backtest(
            stock_code=target_stock,
            output_dir=backtest_dir,
            z_score_threshold=args.z_threshold,
            hard_stop_loss=args.hard_stop,
            trailing_stop=args.trail_stop,
            max_holding_days=args.max_hold,
            show_plot=False,
            entry_percentile=args.entry_percentile,
            calib_window=args.calib_window,
            dynamic_calib=not args.no_dynamic_calib,
            calib_window_min=args.calib_window_min,
            vol_high_thresh=args.vol_high_thresh,
            vol_low_thresh=args.vol_low_thresh,
            risk_per_trade=args.risk_per_trade,
            max_position_pct=args.max_position,
            min_lot_guarantee=not args.no_min_lot_guarantee,
            intraday_stop=not args.no_intraday_stop,
            require_skill=args.require_skill,
            market=args.market,
            forecast_horizon=horizon_val,
            max_auto_horizon=getattr(args, 'max_auto_horizon', 10),
            particle_filter=args.particle_filter,
            pf_confidence_threshold=args.pf_threshold,
            enable_log=not args.no_log,
            clear_log=args.clear_log,
        )
        metrics = result[0]

    if metrics:
        print(f"✅ 回测完成。图表保存于 {backtest_dir}/{target_stock}_backtest_pro.png")


def main():
    """自适应主入口：自动区分 Streamlit / CLI 环境"""
    try:
        from streamlit.runtime import exists as streamlit_exists
        in_streamlit = streamlit_exists()
    except Exception:
        in_streamlit = False

    if in_streamlit:
        streamlit_main()
    else:
        cli_main()


if __name__ == "__main__":
    main()

