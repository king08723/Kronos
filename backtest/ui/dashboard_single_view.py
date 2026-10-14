"""
backtest/ui/dashboard_single_view.py
单标的深度穿透与交互回测看板 (Single Stock Quant Backtest Dashboard)
支持：
- Auto 模式 (自动拉取行情 + Walk-Forward 滚动预测 + 回测) 与 Manual 模式
- 动态最优时界 (Auto Horizon) 与固定时界
- 波动率自适应滚动分位数校准
- 四联回测图 (静态 Matplotlib / 动态 Plotly + 贯通十字标尺)
- 决策诊断日志与逐笔成交流水明细穿透
"""

import os
import sys
import inspect
import logging
import traceback
import pandas as pd
import numpy as np
import streamlit as st

from ..core import (
    MarketProfile,
    SymbolResolver,
    clear_logs,
    set_logging_state,
)
from ..data import (
    KronosDataFetcher,
    DataFetcher,
)
from ..engine import (
    KronosBacktesterPro,
)
from ..visualizer import (
    inject_continuous_crosshair_script,
    HAS_PLOTLY,
)

logger = logging.getLogger("KronosBacktest")

UI_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = os.path.dirname(UI_DIR)

def resolve_data_dir() -> str:
    """回测行情数据存储目录：优先收敛至独立解耦目录 backtest/historical_data，向下兼容 backtest/data"""
    hist_data = os.path.join(BACKTEST_DIR, "historical_data")
    if os.path.exists(hist_data):
        return hist_data
    local_data = os.path.join(BACKTEST_DIR, "data")
    os.makedirs(local_data, exist_ok=True)
    return local_data

def resolve_pred_dir() -> str:
    local_pred = os.path.join(BACKTEST_DIR, "yuce")
    os.makedirs(local_pred, exist_ok=True)
    return local_pred

get_available_stocks = DataFetcher.get_available_stocks


def render_single_backtest_dashboard():
    """Streamlit 交互回测看板（工业级 Pro 版）"""
    # 平滑兼容 Streamlit 容器宽度参数 (新版: width='stretch', 旧版: use_container_width=True)
    _stretch_kw = (
        {'width': 'stretch'}
        if 'width' in inspect.signature(st.button).parameters
        else {'use_container_width': True}
    )

    try:
        st.set_page_config(
            page_title="Kronos Pro 量化回测",
            page_icon="📈",
            layout="wide"
        )
    except Exception:
        pass

    data_dir   = resolve_data_dir()
    model_dir  = resolve_pred_dir()
    output_dir = os.path.join(BACKTEST_DIR, "output")

    st.title("📈 Kronos Pro — 工业级量化回测系统")
    st.caption(
        "**严格防偷价 (t→t+1)** · Z-Score自适应信号 · A股真实摩擦 · 三重止损状态机 · "
        "Walk-Forward 样本外验证"
    )

    available_stocks = get_available_stocks(data_dir, model_dir)

    with st.sidebar:
        st.header("⚙️ 回测参数配置")

        # ---- 数据来源模式 ----
        st.subheader("📡 数据来源")
        data_mode = st.radio(
            "选择数据获取方式",
            options=["🤖 Auto（自动拉取行情 + Walk-Forward 预测）",
                     "📂 Manual（使用现有预测文件）"],
            index=0,
            help=(
                "Auto 模式：自动从 AKShare 拉取历史行情，用 Kronos 模型"
                "做 Walk-Forward 逐步预测，生成真实样本外预测序列。\n\n"
                "Manual 模式：使用 yuce/ 目录中已有的预测 CSV 文件，"
                "要求预测文件与历史数据有 ≥10 个交易日重叠。"
            )
        )
        is_auto_mode = data_mode.startswith("🤖")

        if is_auto_mode:
            st.caption("ℹ️ Auto 模式需要 Kronos 模型已加载，首次运行会有约 1~2 分钟初始化时间。")
        else:
            st.info(
                "**Manual 模式要求**：`yuce/` 目录中的预测文件必须与历史行情有 ≥10 天重叠区间。",
                icon="📋"
            )

        st.divider()

        # ---- 标的选择 ----
        st.subheader("标的选择")
        stock_candidates = sorted(set(available_stocks + ["300418", "600036", "000001"]))
        default_idx = stock_candidates.index("300418") if "300418" in stock_candidates else 0
        stock_option = st.selectbox("选择标的", stock_candidates + ["自定义输入"],
                                    index=default_idx)
        if stock_option == "自定义输入":
            selected_stock = st.text_input("输入股票代码", value="300418")
        else:
            selected_stock = stock_option

        # ---- Walk-Forward 参数（仅 Auto 模式显示）----
        if is_auto_mode:
            st.subheader("🔄 Walk-Forward 参数")
            horizon_mode = st.radio(
                "预测时界模式 (Horizon Mode)",
                options=["🤖 Auto 极值自适应 (推荐 · 动态最优买卖点)", "⚙️ 固定预测时界 (Fixed Days)"],
                index=0,
                help="【Auto 极值自适应】：前瞻推演 10 天走势，自动识别未来波峰极大值与拐点，动态确定每笔交易的最优持仓时界。\n【固定预测时界】：传统固定天数 Horizon 回测。"
            )
            if horizon_mode.startswith("🤖"):
                forecast_horizon = "auto"
                max_auto_horizon = st.slider(
                    "最大推演搜索时界 (天)", min_value=5, max_value=20, value=10, step=1,
                    help=(
                        "Auto 模式下模型向前推演的最大天数（推荐 8~12 交易日）。"
                        "算法在该区间内搜寻期望收益最高的波峰点作为 optimal_horizon。\n\n"
                        "⚠️ 超过 15 天时，Kronos 自回归误差会快速累积，"
                        "argmax 趋向于漂移至末端，导致 optimal_horizon 虚高；"
                        "5~12 天为最佳精度区间。"
                    )
                )
                if max_auto_horizon > 12:
                    st.warning(
                        f"⚠️ 当前推演时界 **{max_auto_horizon} 天** 超出建议范围（≤12天）。"
                        f"Kronos 自回归在此跨度下误差累积显著，预测波峰位置可靠性下降。"
                        f"建议调低至 **8~12 天**。",
                        icon="⚠️"
                    )
            else:
                forecast_horizon = st.selectbox(
                    "选择固定时界 (Horizon)",
                    options=[1, 3, 5, 10],
                    index=2,
                    format_func=lambda x: f"{x} 天 ({'单日超短' if x==1 else '短线波段' if x==3 else '标准波段(推荐)' if x==5 else '双周中波段'})"
                )
                max_auto_horizon = 10

            history_days = st.slider(
                "历史数据获取天数", min_value=200, max_value=1200,
                value=700, step=50,
                help="从 AKShare 拉取的历史日K线总天数（自然日）。回测半年（120交易日）需至少 650 天以上。"
            )
            test_days = st.slider(
                "回测验证窗口 (交易日)", min_value=20, max_value=300,
                value=120, step=5,
                help="历史数据最后 N 个交易日作为回测验证区间，默认 120 个交易日（约半年）"
            )
            wf_lookback = st.slider(
                "预测输入窗口 (交易日)", min_value=60, max_value=200,
                value=120, step=10,
                help="每次滚动预测时使用的历史窗口长度"
            )
            min_needed_natural_days = int((test_days + wf_lookback) * 1.5)
            if history_days < min_needed_natural_days:
                st.warning(
                    f"⚠️ 当前回测窗口({test_days}日) + 输入窗口({wf_lookback}日) 需要至少 "
                    f"{test_days + wf_lookback} 个交易日（约 {min_needed_natural_days} 自然日）。\n"
                    f"建议将【历史数据获取天数】调至 **{min_needed_natural_days}** 天以上，避免数据不足报错！"
                )

            step_size = st.slider(
                "滑动步长", min_value=1, max_value=10,
                value=1, step=1,
                help=("=1 逐日精确预测（正式验收必须用 1）；>1 跳步预测并线性插值，"
                      "插值会人为平滑序列、虚高 IC，仅适合快速预览，"
                      "插值行会被自动排除在信号与技能体检之外")
            )
            wf_samples = st.slider(
                "蒙特卡洛采样数", min_value=1, max_value=60,
                value=20, step=1,
                help=("每个时间步的独立采样路径数。≥20 才能支撑分位数估计；"
                      "采样数过低还会放大 argmax 波峰正偏（3 条采样在纯噪声上就有 +3% 虚高）。"
                      "采用批量并行推理，提高采样数几乎不增加耗时")
            )
            force_refresh = st.checkbox("强制刷新行情缓存", value=False)
        else:
            forecast_horizon = "auto"
            max_auto_horizon = 10
            history_days = 700
            test_days = 120
            wf_lookback = 120
            step_size = 1
            wf_samples = 20
            force_refresh = False

        st.divider()

        # ---- 资金与信号参数 ----
        temp_profile = MarketProfile.for_stock(selected_stock)
        st.subheader("资金设置")
        initial_capital = st.number_input(
            f"初始账户资金 ({temp_profile.currency})", min_value=1000, max_value=100000000,
            value=int(temp_profile.default_capital), step=5000,
            help=f"{temp_profile.market} 市场推荐基准资金: {temp_profile.currency_symbol}{temp_profile.default_capital:,.0f}"
        )

        st.subheader("策略与体检")
        min_lot_guarantee = st.checkbox(
            "一手保底机制（港美股高单价适配）", value=True,
            help="若按风险预算计算出的股数不足一手，在现金及仓位上限允许时保底买入 1 手，避免港股、美股高单价标的因整手向下抹零而零成交。"
        )
        require_skill = st.checkbox(
            "预测技能未达标时拦截回测", value=False,
            help=("开启后，若该标的的预测 IC 不显著或为负，将直接中止回测。"
                  "预测没有排序能力时，任何参数组合跑出的正收益都只是路径运气。")
        )

        st.subheader("信号参数")
        entry_percentile = st.slider(
            "入场分位阈值", min_value=0.50, max_value=0.95,
            value=0.80, step=0.05,
            help=("取该标的自身信号分布的滚动分位数作为阈值，替代固定 Z 值。\n"
                  "0.80 表示只在信号强度进入自身历史前 20% 时入场。\n"
                  "调低 = 交易更频繁。不同标的的预测量纲差异极大，分位数阈值能自动适配。")
        )
        calib_window = st.slider(
            "基准校准窗口 (交易日)", min_value=20, max_value=120,
            value=60, step=10,
            help="分位数阈值与预测去偏所用的基准回看窗口（平稳期维持此长度）"
        )
        dynamic_calib = st.checkbox(
            "启用波动率自适应校准窗口", value=True,
            help="极端高波动率或黑天鹅阶段（如年化波动率>40%）窗口自适应收缩至20~30天，避免60天窗口迟钝；震荡平稳期维持基准窗口"
        )
        if dynamic_calib:
            calib_window_min = st.slider(
                "高波动自适应窗口下限 (天)", min_value=15, max_value=40,
                value=20, step=5,
                help="当标的年化波动率达到或超过40%时采用的最小统计窗口，推荐20~30天"
            )
        else:
            calib_window_min = 20
        vol_high_thresh = 0.40
        vol_low_thresh = 0.20
        z_threshold = 0.70
        vol_window = st.slider(
            "波动率计算窗口 (交易日)", min_value=5, max_value=30,
            value=14, step=1
        )

        st.subheader("风控参数")
        hard_stop = st.slider(
            "基准硬止损线 (%)", min_value=1.0, max_value=10.0,
            value=4.0, step=0.5,
            help="实际止损线由 ATR 自适应（1.5×ATR，限定在 2%~12%），该值用于保底与参照。"
        ) / 100.0
        trail_stop = st.slider(
            "移动止盈回撤 (%)", min_value=1.0, max_value=8.0,
            value=2.5, step=0.5,
            help="持仓高点回撤超过此幅度且盈利>2%时止盈。同样按 ATR 自适应"
        ) / 100.0
        risk_per_trade = st.slider(
            "单笔风险预算 (%)", min_value=0.25, max_value=5.0,
            value=1.0, step=0.25,
            help=("每笔交易允许亏损的净值占比，仓位 = 风险预算 / 止损距离。\n"
                  "波动越大的标的仓位自动越小。")
        ) / 100.0
        max_position_pct = st.slider(
            "单标的仓位上限 (%)", min_value=10, max_value=100,
            value=80, step=5,
            help="风险预算算出的仓位还会受此上限约束"
        ) / 100.0
        intraday_stop = st.checkbox(
            "盘中止损（遵守市场 T+N）", value=True,
            help="开启后，当日最低价击穿止损价即在当日成交（跳空低开按开盘价）。A股严格遵守T+1开仓当日不触发，港美股T+0开仓当日亦可立即止损。"
        )
        def_hold = 15 if forecast_horizon == "auto" else max(5, int(forecast_horizon))
        max_hold = st.slider(
            "最大持仓天数", min_value=2, max_value=30,
            value=def_hold, step=1,
            help="单笔交易最大持有天数。Auto 模式下以预测波峰止盈优先，超时平仓作为兜底风控"
        )

        st.subheader("交易成本")
        commission_bps = st.number_input(
            "券商佣金 (万分之)", min_value=0.1, max_value=10.0,
            value=3.0, step=0.5
        )
        slippage_bps = st.number_input(
            "滑点预估 (万分之)", min_value=0.0, max_value=20.0,
            value=10.0, step=1.0
        )

        st.divider()
        st.subheader("📝 诊断日志设置")
        enable_file_log = st.toggle("开启运行日志落盘", value=True, help="关闭后回测过程将不再向本地磁盘写入 .log 文件")
        set_logging_state(enabled=True, file_logging=enable_file_log, logger=logger, log_dir=output_dir)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🗑️ 清空标的日志", help=f"清空 {selected_stock} 的专属日志", **_stretch_kw):
                cleared = clear_logs(log_dir=output_dir, stock_code=selected_stock)
                if cleared:
                    st.toast(f"已清空 {selected_stock} 运行日志", icon="🧹")
                else:
                    st.toast(f"未找到 {selected_stock} 日志文件", icon="ℹ️")
                st.rerun()
        with col_c2:
            if st.button("🧹 清空全部日志", help="清空 output 目录下所有 .log 日志", **_stretch_kw):
                cleared = clear_logs(log_dir=output_dir, clear_all=True)
                st.toast(f"已清空全部回测日志 (共 {len(cleared)} 个文件)", icon="🧹")
                st.rerun()

        st.divider()
        run_btn = st.button("🚀 运行 Pro 回测", type="primary", **_stretch_kw)

    # ---------- 执行回测 ----------
    run_key = (selected_stock, is_auto_mode, initial_capital, z_threshold, vol_window,
               hard_stop, trail_stop, max_hold, commission_bps, slippage_bps,
               min_lot_guarantee, require_skill, entry_percentile, calib_window,
               dynamic_calib, calib_window_min, vol_high_thresh, vol_low_thresh,
               risk_per_trade, max_position_pct, intraday_stop,
               forecast_horizon if is_auto_mode else "auto",
               max_auto_horizon if is_auto_mode else 10,
               history_days if is_auto_mode else 0,
               test_days if is_auto_mode else 0,
               wf_lookback if is_auto_mode else 0,
               step_size if is_auto_mode else 0,
               wf_samples if is_auto_mode else 0,
               force_refresh if is_auto_mode else False)

    backtester = KronosBacktesterPro(
        data_dir=data_dir,
        model_dir=model_dir,
        initial_capital=float(initial_capital),
        commission_rate=commission_bps / 10000,
        slippage_rate=slippage_bps / 10000,
        enable_log=enable_file_log,
    )

    is_auto_h = (forecast_horizon == "auto" or (isinstance(forecast_horizon, str) and forecast_horizon.lower() == "auto"))
    cache_suffix = "_auto" if is_auto_h else (f"_h{forecast_horizon}" if int(forecast_horizon) > 1 else "")
    wf_cache_file = os.path.join(model_dir, f"{selected_stock}_wf_predictions{cache_suffix}.csv")
    has_wf_cache = (not force_refresh) and os.path.exists(wf_cache_file)

    current_loaded_stock = st.session_state.get("bt_data", {}).get("stock_code")
    stock_changed = (current_loaded_stock != selected_stock)
    need_initial_load = ("bt_data" not in st.session_state) or stock_changed

    should_execute = False
    is_auto_preloading = False

    if run_btn:
        should_execute = True
    elif need_initial_load:
        if (is_auto_mode and has_wf_cache) or (not is_auto_mode):
            should_execute = True
            is_auto_preloading = True

    if should_execute:
        if is_auto_mode:
            h_label = "Auto 极值自适应" if is_auto_h else f"Horizon={forecast_horizon}天"
            if has_wf_cache:
                spinner_msg = f"📁 命中 {selected_stock} 已有 ({h_label}) 预测缓存，正在秒级执行回测与结算..."
            else:
                spinner_msg = (
                    f"正在自动获取 {selected_stock} 行情并执行 Walk-Forward 多时界预测 ({h_label})...\n"
                    f"（step_size={step_size} 时约需 {test_days // step_size * 2}~"
                    f"{test_days // step_size * 5} 秒，请耐心等待）"
                )
            with st.spinner(spinner_msg):
                try:
                    fetcher = KronosDataFetcher()
                    hist_df, pred_df = fetcher.run_auto_pipeline(
                        stock_code=selected_stock,
                        data_dir=data_dir,
                        pred_output_dir=model_dir,
                        history_days=history_days,
                        test_days=test_days,
                        lookback=wf_lookback,
                        step_size=step_size,
                        sample_count=wf_samples,
                        forecast_horizon=forecast_horizon,
                        max_auto_horizon=max_auto_horizon,
                        force_refresh=force_refresh,
                    )
                    backtester.set_market(selected_stock)
                    signal_df = backtester.align_and_generate_signals(
                        hist_df, pred_df,
                        z_score_threshold=z_threshold,
                        volatility_window=vol_window,
                        forecast_horizon=forecast_horizon,
                        max_auto_horizon=max_auto_horizon,
                        hard_stop_loss=hard_stop,
                        entry_percentile=entry_percentile,
                        calib_window=calib_window,
                        dynamic_calib=dynamic_calib,
                        calib_window_min=calib_window_min,
                        vol_high_thresh=vol_high_thresh,
                        vol_low_thresh=vol_low_thresh,
                    )
                    skill = backtester.evaluate_prediction_skill(signal_df, hard_stop_loss=hard_stop)
                    backtester.print_skill_report(skill, selected_stock)
                    if require_skill and not skill.get('tradable', False):
                        reason = skill.get('error') or '; '.join(skill.get('不通过原因', []))
                        st.error(
                            f"❌ {selected_stock} 未通过预测技能体检，已中止回测。\n\n"
                            f"原因：{reason}\n\n"
                            f"预测没有排序能力时，任何参数组合跑出的正收益都只是路径运气。",
                            icon="🚫")
                        st.stop()
                    backtest_df, trades = backtester.run_backtest_engine(
                        signal_df,
                        hard_stop_loss=hard_stop,
                        trailing_stop=trail_stop,
                        max_holding_days=max_hold,
                        min_lot_guarantee=min_lot_guarantee,
                        risk_per_trade=risk_per_trade,
                        max_position_pct=max_position_pct,
                        intraday_stop=intraday_stop,
                    )
                    metrics = backtester.evaluate_metrics(backtest_df, trades)
                    metrics['_skill'] = skill
                    fig = backtester.plot_backtest_results(
                        signal_df, backtest_df, metrics, trades,
                        selected_stock, output_dir, show_plot=False,
                    )
                except Exception as e:
                    if is_auto_preloading:
                        logger.warning(f"自动预加载 {selected_stock} 失败: {e}")
                        metrics = None
                    else:
                        st.error(f"❌ Auto 模式失败：{e}", icon="🚫")
                        st.code(traceback.format_exc())
                        st.stop()
        else:
            with st.spinner(f"正在对 {selected_stock} 执行样本外回测..."):
                try:
                    result = backtester.run_complete_backtest(
                        stock_code=selected_stock,
                        output_dir=output_dir,
                        z_score_threshold=z_threshold,
                        volatility_window=vol_window,
                        hard_stop_loss=hard_stop,
                        trailing_stop=trail_stop,
                        max_holding_days=max_hold,
                        show_plot=False,
                        entry_percentile=entry_percentile,
                        calib_window=calib_window,
                        dynamic_calib=dynamic_calib,
                        calib_window_min=calib_window_min,
                        vol_high_thresh=vol_high_thresh,
                        vol_low_thresh=vol_low_thresh,
                        risk_per_trade=risk_per_trade,
                        max_position_pct=max_position_pct,
                        min_lot_guarantee=min_lot_guarantee,
                        intraday_stop=intraday_stop,
                        require_skill=require_skill,
                    )
                    metrics, signal_df, backtest_df, trades, fig = result
                except Exception as me:
                    if is_auto_preloading:
                        logger.warning(f"自动预加载 {selected_stock} 失败: {me}")
                        metrics = None
                    else:
                        st.error(f"❌ 回测失败: {me}")
                        metrics = None

        if metrics is not None:
            plotly_fig = None
            if HAS_PLOTLY:
                try:
                    plotly_fig = backtester.plot_backtest_interactive(
                        signal_df, backtest_df, metrics, trades,
                        selected_stock, z_score_threshold=z_threshold
                    )
                except Exception as pe:
                    logger.warning(f"生成 Plotly 交互图失败: {pe}")

            st.session_state["bt_data"] = {
                "stock_code": selected_stock,
                "mode": "Auto" if is_auto_mode else "Manual",
                "metrics": metrics,
                "signal_df": signal_df,
                "backtest_df": backtest_df,
                "trades": trades,
                "fig": fig,
                "plotly_fig": plotly_fig,
                "z_threshold": z_threshold,
                "initial_capital": float(initial_capital),
                "auto_loaded": is_auto_preloading,
            }
            st.session_state["bt_key"] = run_key
        elif not is_auto_preloading:
            st.error(
                "❌ 回测失败：预测数据与历史数据无足够重叠，请检查控制台输出的错误信息，"
                "或切换到 Auto 模式自动生成预测数据。",
                icon="🚫"
            )
            st.stop()

    if "bt_data" not in st.session_state:
        st.info("👈 请在左侧侧边栏配置参数后，点击「🚀 运行 Pro 回测」启动量化回测系统。", icon="💡")
        st.stop()

    data = st.session_state["bt_data"]
    if data.get("auto_loaded", False):
        st.caption("⚡ 系统已自动命中缓存载入当前标的回测结果。如需修改参数重新计算，请点击左侧「🚀 运行 Pro 回测」。")
    elif "bt_key" in st.session_state and st.session_state["bt_key"] != run_key:
        st.warning("⚠️ 检测到左侧配置参数已被修改（当前仍展示上一次回测结果）。请点击左侧「🚀 运行 Pro 回测」按钮以应用新参数运行。", icon="🔔")

    data       = st.session_state["bt_data"]
    metrics    = data["metrics"]
    backtest_df= data["backtest_df"]
    signal_df  = data["signal_df"]
    trades     = data["trades"]
    fig        = data["fig"]
    plotly_fig = data.get("plotly_fig")
    sc         = data["stock_code"]
    init_cap   = data["initial_capital"]

    cur_profile = MarketProfile.for_stock(sc)
    cs = cur_profile.currency_symbol
    cur = cur_profile.currency

    if plotly_fig is None and HAS_PLOTLY and backtest_df is not None:
        try:
            plotly_fig = backtester.plot_backtest_interactive(
                signal_df, backtest_df, metrics, trades, sc,
                z_score_threshold=data.get("z_threshold", 1.2)
            )
            data["plotly_fig"] = plotly_fig
        except Exception as pe:
            logger.warning(f"动态生成交互式图表失败: {pe}")

    # ---------- 核心指标卡片 ----------
    with st.container(border=True):
        st.subheader(f"📊 {sc} [{cur_profile.market}] · 核心绩效概览  [样本外 Backtest]")
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        tot_ret = metrics['累计收益率']
        ann_ret = metrics['年化收益率']
        sharpe  = metrics['夏普比率']
        calmar  = metrics['卡玛比率']
        mdd     = metrics['最大回撤']
        final   = metrics['期末总资产']

        c1.metric("累计收益率", f"{tot_ret:+.2%}", delta=f"{tot_ret:+.2%}", delta_color="normal")
        c2.metric("年化收益率", f"{ann_ret:+.2%}")
        c3.metric("夏普比率",   f"{sharpe:.3f}")
        c4.metric("卡玛比率",   f"{calmar:.3f}")
        c5.metric("最大回撤",   f"{mdd:.2%}", delta_color="inverse")
        c6.metric("期末总资产", f"{cs}{final:,.0f}", delta=f"{final-init_cap:+,.0f} {cur}")

    s1, s2, s3, s4, s5 = st.columns(5)
    pf = metrics.get('盈亏比', float('nan'))
    pf_str = f"{pf:.2f}" if not (isinstance(pf, float) and pf != pf) else "N/A"
    s1.info(f"📈 **年化波动率**: {metrics['年化波动率']:.2%}")
    s2.info(f"🎯 **胜率**: {metrics['胜率']:.1%}")
    s3.info(f"⚖️ **盈亏比**: {pf_str}")
    s4.info(f"🔄 **总交易笔数**: {int(metrics['总交易笔数'])} 笔")
    s5.info(f"💰 **单笔平均盈亏**: {metrics['单笔平均盈亏']:+.2%}")

    # ---------- 标签页 ----------
    tab_chart, tab_signal, tab_trades, tab_data = st.tabs([
        "📈 四联回测图", "🔬 信号详情", "📋 交易流水", "📑 每日净值"
    ])

    with tab_chart:
        if plotly_fig is not None:
            plotly_fig.update_layout(hovermode="x", spikedistance=-1)
            st.plotly_chart(plotly_fig, width="stretch", key=f"pro_plotly_{sc}")
            inject_continuous_crosshair_script()
        elif fig is not None:
            st.pyplot(fig, **_stretch_kw)
        else:
            st.warning("图表生成失败")

        if trades:
            st.subheader("📋 交易执行记录明细 (含买入与卖出)")
            summary_trades = []
            for idx, t in enumerate(trades, 1):
                act = t['action']
                act_label = "🟢 买入建仓" if act == 'BUY' else "🔴 卖出平仓"
                pnl_disp = f"{t['pnl_pct']:+.2%}" if act == 'SELL' else "-"
                summary_trades.append({
                    '序号': idx,
                    '成交日期': t['date'].strftime('%Y-%m-%d'),
                    '操作动作': act_label,
                    '成交价': f"{cs}{t['price']:.3f}",
                    '持仓股数': f"{t['shares']:,}",
                    '单笔盈亏': pnl_disp,
                    '触发原因': t.get('reason', 'N/A'),
                    '账户总资金': f"{cs}{t['cash_balance']:,.2f}",
                })
            st.dataframe(pd.DataFrame(summary_trades), width="stretch")

    with tab_signal:
        st.markdown("**Z-Score 信号序列（防偷价验证）**")
        if signal_df is not None:
            cols_to_show = ['close', 'pred_close', 'expected_return']
            col_names = ['真实收盘价', '预测价', '预期收益率']
            if 'dyn_calib_window' in signal_df.columns:
                cols_to_show.append('dyn_calib_window')
                col_names.append('自适应窗口(天)')
            if 'ann_vol' in signal_df.columns:
                cols_to_show.append('ann_vol')
                col_names.append('年化波动率')
            else:
                cols_to_show.append('volatility')
                col_names.append('历史波动率')
            cols_to_show.extend(['signal_score', 'raw_signal', 'exec_signal'])
            col_names.extend(['Z-Score', '原始信号(t日)', '执行信号(t+1日)'])

            display_sig = signal_df[cols_to_show].copy()
            display_sig.columns = col_names
            fmt_dict = {
                '真实收盘价': '{:.3f}',
                '预测价': '{:.3f}',
                '预期收益率': '{:+.3%}',
                'Z-Score': '{:+.3f}',
            }
            if '自适应窗口(天)' in display_sig.columns:
                fmt_dict['自适应窗口(天)'] = '{:d}'
            if '年化波动率' in display_sig.columns:
                fmt_dict['年化波动率'] = '{:.1%}'
            if '历史波动率' in display_sig.columns:
                fmt_dict['历史波动率'] = '{:.4f}'

            st.dataframe(
                display_sig.style.format(fmt_dict),
                width="stretch"
            )

    with tab_trades:
        st.markdown("**逐笔成交流水账（含税费与持仓天数）**")
        if trades:
            tdf = pd.DataFrame(trades)
            st.dataframe(tdf, **_stretch_kw)
        else:
            st.info("回测区间无成交记录")

    with tab_data:
        if backtest_df is not None:
            display_bt = backtest_df.copy()
            display_bt['portfolio_value'] = display_bt['portfolio_value'].apply(lambda x: f"{cs}{x:,.2f}")
            display_bt['cash'] = display_bt['cash'].apply(lambda x: f"{cs}{x:,.2f}")
            display_bt['returns'] = display_bt['returns'].apply(lambda x: f"{x:+.4%}")
            st.dataframe(display_bt, **_stretch_kw)

    # ---------- 运行日志与决策诊断流水 ----------
    st.divider()
    with st.expander("📜 后台运行日志与决策诊断流水 (Diagnostics)", expanded=False):
        clean_code, _, _ = SymbolResolver.resolve(selected_stock)
        clean_code = clean_code or selected_stock
        stock_log_file = os.path.join(output_dir, f"{clean_code}_backtest.log")
        global_log_file = os.path.join(output_dir, "kronos_backtest.log")
        active_log = stock_log_file if os.path.exists(stock_log_file) else global_log_file

        status_text = "🟢 文件日志落盘已开启" if enable_file_log else "⚪ 文件日志落盘已关闭 (仅内存/控制台)"
        st.markdown(f"**当前状态**: {status_text} | **当前标的**: `{clean_code}`")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🗑️ 清空当前标的日志", key="clear_stock_log_btn_exp", **_stretch_kw):
                cleared = clear_logs(log_dir=output_dir, stock_code=clean_code)
                if cleared:
                    st.toast(f"已清空 {clean_code} 运行日志", icon="🧹")
                else:
                    st.toast(f"未找到 {clean_code} 日志文件", icon="ℹ️")
                st.rerun()
        with col_b2:
            if st.button("🧹 清空所有回测日志", key="clear_all_logs_btn_exp", **_stretch_kw):
                cleared = clear_logs(log_dir=output_dir, clear_all=True)
                st.toast(f"已清空全部回测日志 (共 {len(cleared)} 个文件)", icon="🧹")
                st.rerun()

        if os.path.exists(active_log) and os.path.getsize(active_log) > 0:
            try:
                with open(active_log, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                st.caption(f"📁 日志文件落盘位置: `{active_log}` (实时显示最新 {min(len(lines), 150)} 行)")
                st.code("".join(lines[-150:]), language="log")
            except Exception as e:
                st.warning(f"读取日志失败: {e}")
        else:
            st.info(f"暂无日志记录，或文件已被清空。 (目标: `{active_log}`)")
