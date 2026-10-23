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
    TimeFrame,
    MarketProfile,
    SymbolResolver,
    clear_logs,
    set_logging_state,
    get_timeframe_defaults,
    resolve_timeframe_params,
)

# 跨市场标杆股票快速索引与多语言标签画像
POPULAR_STOCKS_INFO = {
    # A 股
    "300418": ("昆仑万维", "CN"),
    "600036": ("招商银行", "CN"),
    "000001": ("平安银行", "CN"),
    "603288": ("海天味业", "CN"),
    # 港股
    "00700": ("腾讯控股", "HK"),
    "09988": ("阿里巴巴-W", "HK"),
    "03690": ("美团-W", "HK"),
    "00941": ("中国移动", "HK"),
    # 美股
    "AAPL": ("苹果公司", "US"),
    "TSLA": ("特斯拉", "US"),
    "MSFT": ("微软", "US"),
    "NVDA": ("英伟达", "US"),
}
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

    c_hdr1, c_hdr2 = st.columns([8, 2])
    with c_hdr1:
        st.title("📈 Kronos Pro — 工业级量化回测系统")
        st.caption(
            "**严格防偷价 (t→t+1)** · Z-Score自适应信号 · 跨市场真实摩擦与撮合 · 三重止损状态机 · "
            "Walk-Forward 样本外验证"
        )
    with c_hdr2:
        st.write("")
        if st.button("⬅️ 返回批量大盘", key="btn_back_to_batch", **_stretch_kw):
            st.session_state["app_main_work_mode"] = "🏆 股票池批量回测大盘 (Batch Hub)"
            st.rerun()

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

        # 联动大盘批量池与全市场常用标的
        active_batch_pool = st.session_state.get("active_batch_pool", [])
        pool_codes = [it["stock_code"] for it in active_batch_pool if isinstance(it, dict) and "stock_code" in it]

        candidate_set = set(available_stocks + list(POPULAR_STOCKS_INFO.keys()) + pool_codes)
        stock_candidates = sorted(candidate_set)

        # 智能同步选中标的：
        # 若用户在批量大盘切换了不同的预设池，优先同步该池的首个标的；否则维持单标的看板用户的手动选择
        current_batch_preset = st.session_state.get("active_batch_preset")
        last_synced_preset = st.session_state.get("_last_synced_batch_preset")
        if current_batch_preset and current_batch_preset != last_synced_preset and pool_codes:
            st.session_state["_last_synced_batch_preset"] = current_batch_preset
            preferred_stock = pool_codes[0]
            st.session_state["single_active_stock"] = preferred_stock
        else:
            preferred_stock = st.session_state.get("single_active_stock")
            if not preferred_stock or preferred_stock not in stock_candidates:
                if pool_codes and pool_codes[0] in stock_candidates:
                    preferred_stock = pool_codes[0]
                elif "300418" in stock_candidates:
                    preferred_stock = "300418"
                else:
                    preferred_stock = stock_candidates[0] if stock_candidates else "300418"

        default_idx = stock_candidates.index(preferred_stock) if preferred_stock in stock_candidates else 0

        def _format_stock_label(sc: str) -> str:
            if sc == "自定义输入":
                return "✏️ 自定义输入 (Manual Input)"
            if sc in POPULAR_STOCKS_INFO:
                name, mkt = POPULAR_STOCKS_INFO[sc]
                flag = {"CN": "🇨🇳", "HK": "🇭🇰", "US": "🇺🇸"}.get(mkt, "🌐")
                return f"{flag} {sc} ({name})"
            prof = MarketProfile.for_stock(sc)
            flag = {"CN": "🇨🇳", "HK": "🇭🇰", "US": "🇺🇸"}.get(prof.market, "🌐")
            return f"{flag} {sc} ({prof.market})"

        stock_option = st.selectbox(
            "选择标的",
            stock_candidates + ["自定义输入"],
            index=default_idx,
            format_func=_format_stock_label,
            key="sv_stock_selectbox",
        )
        if stock_option == "自定义输入":
            selected_stock = st.text_input("输入股票代码", value=preferred_stock, key="sv_stock_custom_input")
        else:
            selected_stock = stock_option
        st.session_state["single_active_stock"] = selected_stock

        # ---- K线周期选择 ----
        timeframe_options = ["1d", "60m", "30m", "15m", "5m", "1m"]
        synced_tf = st.session_state.get("active_timeframe", "1d")
        default_tf_idx = timeframe_options.index(synced_tf) if synced_tf in timeframe_options else 0

        timeframe_opt = st.selectbox(
            "K线时间粒度 (TimeFrame)",
            options=timeframe_options,
            index=default_tf_idx,
            key="sv_timeframe_selector",
            format_func=lambda x: {
                "1d": "日线 (1d · 默认)",
                "60m": "60分钟 (60m · 4根/日)",
                "30m": "30分钟 (30m · 8根/日)",
                "15m": "15分钟 (15m · 16根/日)",
                "5m": "5分钟 (5m · 48根/日)",
                "1m": "1分钟 (1m · 240根/日)",
            }.get(x, x),
            help="选择回测的目标时间颗粒度。支持自动加载精确匹配文件或从底层高频分钟线现场无损合成。"
        )
        st.session_state["active_timeframe"] = timeframe_opt

        auto_adapt_params = st.toggle(
            "🤖 启用周期最优参数自适应 (Auto-Adapt)",
            value=True,
            help="开启后：切换 1d/60m/30m/15m/5m/1m 时，系统自动将止损止盈、信噪比、最大持仓时界、滑点冲击模型等适配至量化实盘最佳实践！仍可在下方手动微调。"
        )

        stock_mkt = MarketProfile.for_stock(selected_stock).market if selected_stock else 'CN'
        tf_obj = TimeFrame.from_str(timeframe_opt)
        tf_defaults = get_timeframe_defaults(timeframe=timeframe_opt, market=stock_mkt)
        is_intraday = tf_obj.is_intraday
        unit_label = "K线根数 (Bars)" if is_intraday else "交易日"

        with st.container(border=True):
            st.markdown(f"**🎯 周期量化画像 ({timeframe_opt})**：{tf_defaults.get('description', '')}")
            c_r1, c_r2 = st.columns(2)
            with c_r1:
                st.caption(f"🛡️ 推荐止损: `{tf_defaults['hard_stop']*100:.1f}%`")
                st.caption(f"⏳ 最大持仓: `{tf_defaults['max_hold']}` Bar")
                st.caption(f"📶 信噪比门槛: `Z≥{tf_defaults['z_threshold']}`")
            with c_r2:
                st.caption(f"📈 移动止盈: `{tf_defaults['trail_stop']*100:.1f}%`")
                st.caption(f"💧 滑点模型: `{tf_defaults['slippage_model']}`")
                st.caption(f"⚡ MTF双周期: `{'启用' if tf_defaults['use_mtf_trend_filter'] else '无'}`")

        # ---- Walk-Forward 参数（仅 Auto 模式显示）----
        if is_auto_mode:
            st.subheader("🔄 Walk-Forward 参数")
            horizon_mode = st.radio(
                "预测时界模式 (Horizon Mode)",
                options=["🤖 Auto 极值自适应 (推荐 · 动态最优买卖点)", "⚙️ 固定预测时界 (Fixed Days)"],
                index=0,
                help="【Auto 极值自适应】：前瞻推演 10 天走势，自动识别未来波峰极大值与拐点，动态确定每笔交易的最优持仓时界。\n【固定预测时界】：传统固定天数 Horizon 回测。"
            )
            def_test_bars = int(tf_defaults.get('test_bars', 120))
            def_lookback_bars = int(tf_defaults.get('lookback_bars', 90))
            def_step_size = int(tf_defaults.get('step_size', 1))
            def_max_horizon = int(tf_defaults.get('max_auto_horizon', 10))
            def_hist_days = int(tf_defaults.get('fetch_days', 700))

            if horizon_mode.startswith("🤖"):
                forecast_horizon = "auto"
                max_auto_horizon = st.slider(
                    f"最大推演搜索时界 ({unit_label})",
                    min_value=2 if is_intraday else 5,
                    max_value=30 if is_intraday else 20,
                    value=def_max_horizon if auto_adapt_params else 10,
                    step=1,
                    key=f"sv_max_auto_h_{timeframe_opt}",
                    help=(
                        f"Auto 模式下模型向前推演的最大步数（{unit_label}）。"
                        "算法在该区间内搜寻期望收益最高的波峰点作为 optimal_horizon。"
                    )
                )
                if (not is_intraday and max_auto_horizon > 12) or (is_intraday and max_auto_horizon > 24):
                    st.warning(
                        f"⚠️ 当前推演时界 **{max_auto_horizon} {unit_label}** 偏大，"
                        f"Kronos 自回归在此跨度下误差累积显著，建议调低。",
                        icon="⚠️"
                    )
            else:
                forecast_horizon = st.selectbox(
                    f"选择固定时界 ({unit_label})",
                    options=[1, 3, 5, 10] if not is_intraday else [2, 4, 8, 16],
                    index=2,
                    key=f"sv_fixed_horizon_{timeframe_opt}",
                    format_func=lambda x: f"{x} {unit_label}"
                )
                max_auto_horizon = def_max_horizon

            history_days = st.slider(
                "历史数据获取天数 (自然日)",
                min_value=15 if is_intraday else 200,
                max_value=1200,
                value=def_hist_days if auto_adapt_params else 700,
                step=10 if is_intraday else 50,
                key=f"sv_hist_days_{timeframe_opt}",
                help="从数据源拉取的历史行情总天数（自然日）。"
            )
            test_days = st.slider(
                f"回测验证窗口 ({unit_label})",
                min_value=20,
                max_value=600 if is_intraday else 300,
                value=def_test_bars if auto_adapt_params else 120,
                step=10 if is_intraday else 5,
                key=f"sv_test_days_{timeframe_opt}",
                help=f"历史数据最后 N 个 {unit_label} 作为回测验证区间"
            )
            wf_lookback = st.slider(
                f"预测输入窗口 ({unit_label})",
                min_value=30 if is_intraday else 60,
                max_value=500 if is_intraday else 200,
                value=def_lookback_bars if auto_adapt_params else 90,
                step=10,
                key=f"sv_lookback_{timeframe_opt}",
                help=f"每次滚动预测时使用的历史窗口长度 ({unit_label})"
            )
            step_size = st.slider(
                "滑动步长 (Step Size)",
                min_value=1, max_value=10,
                value=def_step_size if auto_adapt_params else 1,
                step=1,
                key=f"sv_step_size_{timeframe_opt}",
                help="=1 逐根精确预测；>1 跳步预测并线性插值（提速一倍以上）"
            )
            wf_samples = st.slider(
                "蒙特卡洛采样数", min_value=1, max_value=60,
                value=35, step=1,
                key=f"sv_samples_{timeframe_opt}",
                help="每个时间步的独立采样路径数。推荐 ≥35 支撑高精度分位数估计与粒子滤波。"
            )
            force_refresh = st.checkbox("强制刷新行情与预测缓存", value=False, key=f"sv_force_ref_{timeframe_opt}")
        else:
            forecast_horizon = "auto"
            max_auto_horizon = int(tf_defaults.get('max_auto_horizon', 10))
            history_days = int(tf_defaults.get('fetch_days', 700))
            test_days = int(tf_defaults.get('test_bars', 120))
            wf_lookback = int(tf_defaults.get('lookback_bars', 90))
            step_size = int(tf_defaults.get('step_size', 1))
            wf_samples = 35
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
        default_entry_pct = float(tf_defaults['entry_percentile']) if auto_adapt_params else 0.80
        entry_percentile = st.slider(
            "入场分位阈值", min_value=0.50, max_value=0.98,
            value=default_entry_pct, step=0.01,
            key=f"sv_entry_pct_{timeframe_opt}",
            help=("取该标的自身信号分布的滚动分位数作为阈值，替代固定 Z 值。\n"
                  "0.80 表示只在信号强度进入自身历史前 20% 时入场。\n"
                  "调低 = 交易更频繁。不同标的的预测量纲差异极大，分位数阈值能自动适配。")
        )
        default_calib_win = int(tf_defaults['calib_window']) if auto_adapt_params else 60
        calib_window = st.slider(
            "基准校准窗口 (Bar数/天数)", min_value=5, max_value=120,
            value=default_calib_win, step=5,
            key=f"sv_calib_win_{timeframe_opt}",
            help="分位数阈值与预测去偏所用的基准回看窗口（平稳期维持此长度）"
        )
        dynamic_calib = st.checkbox(
            "启用波动率自适应校准窗口", value=True,
            key=f"sv_dynamic_calib_{timeframe_opt}",
            help="极端高波动率或黑天鹅阶段（如年化波动率>40%）窗口自适应收缩至20~30天，避免60天窗口迟钝；震荡平稳期维持基准窗口"
        )
        if dynamic_calib:
            default_calib_min = int(tf_defaults['calib_window_min']) if auto_adapt_params else 20
            calib_window_min = st.slider(
                "高波动自适应窗口下限", min_value=2, max_value=40,
                value=default_calib_min, step=1,
                key=f"sv_calib_min_{timeframe_opt}",
                help="当标的年化波动率达到或超过40%时采用的最小统计窗口"
            )
        else:
            calib_window_min = 20
        vol_high_thresh = 0.40
        vol_low_thresh = 0.20

        default_z = float(tf_defaults['z_threshold']) if auto_adapt_params else 0.55
        z_threshold = st.slider(
            "绝对信噪比入场门槛 (Z-Score)", min_value=0.20, max_value=2.00,
            value=default_z, step=0.05,
            key=f"sv_z_thresh_{timeframe_opt}",
            help="预测超额收益相对预期波动率的标准差倍数门槛，自适应按周期的信噪比推荐"
        )
        vol_window = st.slider(
            "波动率计算窗口 (交易日)", min_value=5, max_value=30,
            value=14, step=1,
            key=f"sv_vol_win_{timeframe_opt}"
        )

        st.subheader("风控参数")
        default_hard_stop = round(tf_defaults['hard_stop'] * 100, 2) if auto_adapt_params else 4.0
        hard_stop = st.slider(
            "基准硬止损线 (%)", min_value=0.2, max_value=15.0,
            value=default_hard_stop, step=0.1,
            key=f"sv_hard_stop_{timeframe_opt}",
            help="实际止损线由 ATR 自适应（1.5×ATR，限定在 2%~12%），该值用于保底与参照。"
        ) / 100.0

        default_trail_stop = round(tf_defaults['trail_stop'] * 100, 2) if auto_adapt_params else 2.5
        trail_stop = st.slider(
            "移动止盈回撤 (%)", min_value=0.1, max_value=10.0,
            value=default_trail_stop, step=0.1,
            key=f"sv_trail_stop_{timeframe_opt}",
            help="持仓高点回撤超过此幅度且盈利达标时止盈。高频周期自动收紧"
        ) / 100.0

        risk_per_trade = st.slider(
            "单笔风险预算 (%)", min_value=0.25, max_value=5.0,
            value=1.0, step=0.25,
            key=f"sv_risk_per_trade_{timeframe_opt}",
            help=("每笔交易允许亏损的净值占比，仓位 = 风险预算 / 止损距离。\n"
                  "波动越大的标的仓位自动越小。")
        ) / 100.0
        max_position_pct = st.slider(
            "单标的仓位上限 (%)", min_value=10, max_value=100,
            value=80, step=5,
            key=f"sv_max_pos_{timeframe_opt}",
            help="风险预算算出的仓位还会受此上限约束"
        ) / 100.0
        intraday_stop = st.checkbox(
            "盘中止损（遵守市场 T+N）", value=True,
            key=f"sv_intraday_stop_{timeframe_opt}",
            help="开启后，当日最低价击穿止损价即在当日成交（跳空低开按开盘价）。A股严格遵守T+1开仓当日不触发，港美股T+0开仓当日亦可立即止损。"
        )

        default_max_hold = int(tf_defaults['max_hold']) if auto_adapt_params else (15 if forecast_horizon == "auto" else max(5, int(forecast_horizon)))
        max_hold = st.slider(
            "最大持仓周期数 (Bar数/天数)", min_value=1, max_value=120,
            value=default_max_hold, step=1,
            key=f"sv_max_hold_{timeframe_opt}",
            help="单笔交易最大持有周期数。Auto 模式下以预测波峰止盈优先，超时平仓作为兜底风控"
        )

        st.subheader("交易成本")
        commission_bps = st.number_input(
            "券商佣金 (万分之)", min_value=0.1, max_value=10.0,
            value=3.0, step=0.5,
            key=f"sv_comm_bps_{timeframe_opt}"
        )
        slippage_bps = st.number_input(
            "滑点预估 (万分之)", min_value=0.0, max_value=20.0,
            value=10.0, step=1.0,
            key=f"sv_slip_bps_{timeframe_opt}"
        )

        st.subheader("高频流动性与跨周期共振")
        default_cap_enable = (tf_defaults['volume_participation_cap'] is not None) if auto_adapt_params else False
        enable_volume_cap = st.checkbox(
            "启用单 Bar 流动性截断",
            value=default_cap_enable,
            key=f"sv_enable_vcap_{timeframe_opt}",
            help="单笔大单在单根 Bar 撮合时的成交量上限（默认 10%），超出部分整手截断，防止市场冲击"
        )
        if enable_volume_cap:
            default_cap_val = int((tf_defaults['volume_participation_cap'] or 0.10) * 100) if auto_adapt_params else 10
            volume_cap_val = st.slider(
                "单 Bar 成交量占比上限 (%)", min_value=1, max_value=20,
                value=default_cap_val, step=1,
                key=f"sv_vcap_val_{timeframe_opt}"
            ) / 100.0
        else:
            volume_cap_val = None

        default_slip_idx = (1 if tf_defaults['slippage_model'] == 'dynamic' else 0) if auto_adapt_params else 0
        slippage_mode = st.radio(
            "滑点模型",
            options=["固定滑点 (Fixed)", "平方根动态冲击滑点 (Dynamic)"],
            index=default_slip_idx,
            key=f"sv_slip_mode_{timeframe_opt}",
            help="【固定滑点】：按固定基准点差执行；【动态滑点】：基于市场微观平方根冲击定律与实时成交量占比动态计算真实摩擦。"
        )
        eff_slippage_model = "dynamic" if "Dynamic" in slippage_mode else "fixed"

        default_eod_exit = bool(tf_defaults['intraday_exit_at_eod']) if auto_adapt_params else False
        intraday_exit_eod = st.checkbox(
            "日内收盘强平 (不留隔夜仓)",
            value=default_eod_exit,
            key=f"sv_eod_exit_{timeframe_opt}",
            help="在交易日最后一根 Bar 强制清平所有多头仓位，仅做日内日冲交易，规避隔夜系统性风险（A股T+1自动保持合规）。"
        )
        default_gap_prot = bool(tf_defaults['overnight_gap_protection']) if auto_adapt_params else True
        overnight_gap_protection = st.checkbox(
            "开启隔夜跳空防守",
            value=default_gap_prot,
            key=f"sv_gap_prot_{timeframe_opt}",
            help="次日低开跌破止损线以实际开盘价真实穿透止损，避免盘中止损线失效产生虚假高收益。"
        )
        default_mtf = bool(tf_defaults['use_mtf_trend_filter']) if auto_adapt_params else True
        mtf_trend_filter = st.checkbox(
            "跨周期宏观趋势滤网 (MTF Filter)",
            value=default_mtf,
            key=f"sv_mtf_{timeframe_opt}",
            help="在小周期（如 30m / 60m）回测时，以零未来函数引入日线级别 MA20/MA60 宏观大趋势滤网进行双周期共振准入。"
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
    run_key = (selected_stock, timeframe_opt, is_auto_mode, initial_capital, z_threshold, vol_window,
               hard_stop, trail_stop, max_hold, commission_bps, slippage_bps,
               min_lot_guarantee, require_skill, entry_percentile, calib_window,
               dynamic_calib, calib_window_min, vol_high_thresh, vol_low_thresh,
               risk_per_trade, max_position_pct, intraday_stop,
               volume_cap_val, eff_slippage_model, intraday_exit_eod,
               overnight_gap_protection, mtf_trend_filter, auto_adapt_params,
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
    tf_tag = f"_{timeframe_opt.lower()}" if timeframe_opt and timeframe_opt != "1d" else ""
    wf_cache_candidates = [
        os.path.join(model_dir, f"{selected_stock}{tf_tag}_wf_predictions{cache_suffix}.csv"),
        os.path.join(model_dir, f"{selected_stock}_wf_predictions{cache_suffix}.csv"),
    ]
    wf_cache_file = next((p for p in wf_cache_candidates if os.path.exists(p)), wf_cache_candidates[0])
    has_wf_cache = (not force_refresh) and os.path.exists(wf_cache_file)

    current_loaded_stock = st.session_state.get("bt_data", {}).get("stock_code")
    current_loaded_tf = st.session_state.get("bt_data", {}).get("timeframe")
    stock_changed = (current_loaded_stock != selected_stock) or (current_loaded_tf != timeframe_opt)

    if run_btn:
        should_execute = True
    else:
        should_execute = False
        if stock_changed and "bt_data" in st.session_state:
            # 标的或周期切换后，若尚未点击运行，清空旧标的的历史渲染，提示用户配置参数并启动
            st.session_state.pop("bt_data", None)
            st.session_state.pop("bt_key", None)

    if should_execute:
        if is_auto_mode:
            h_label = "Auto 极值自适应" if is_auto_h else f"Horizon={forecast_horizon}步"
            if has_wf_cache:
                spinner_msg = f"📁 命中 {selected_stock} ({timeframe_opt} · {h_label}) 预测缓存，正在秒级执行回测与结算..."
            else:
                spinner_msg = (
                    f"正在自动获取 {selected_stock} 行情并执行 Walk-Forward 多时界预测 ({timeframe_opt} · {h_label})...\n"
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
                        timeframe=timeframe_opt,
                        test_bars=test_days,
                        lookback_bars=wf_lookback,
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
                        use_mtf_trend_filter=mtf_trend_filter,
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
                        volume_participation_cap=volume_cap_val,
                        slippage_model=eff_slippage_model,
                        intraday_exit_at_eod=intraday_exit_eod,
                        overnight_gap_protection=overnight_gap_protection,
                        auto_adapt_params=auto_adapt_params,
                    )
                    metrics = backtester.evaluate_metrics(backtest_df, trades)
                    metrics['_skill'] = skill
                    fig = backtester.plot_backtest_results(
                        signal_df, backtest_df, metrics, trades,
                        selected_stock, output_dir, show_plot=False,
                    )
                except Exception as e:
                    st.error(f"❌ Auto 模式失败：{e}", icon="🚫")
                    st.code(traceback.format_exc())
                    st.stop()
        else:
            with st.spinner(f"正在对 {selected_stock} 执行样本外回测..."):
                try:
                    result = backtester.run_complete_backtest(
                        stock_code=selected_stock,
                        output_dir=output_dir,
                        timeframe=timeframe_opt,
                        use_mtf_trend_filter=mtf_trend_filter,
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
                        volume_participation_cap=volume_cap_val,
                        slippage_model=eff_slippage_model,
                        intraday_exit_at_eod=intraday_exit_eod,
                        overnight_gap_protection=overnight_gap_protection,
                        auto_adapt_params=auto_adapt_params,
                    )
                    metrics, signal_df, backtest_df, trades, fig = result
                except Exception as me:
                    st.error(f"❌ 回测失败: {me}", icon="🚫")
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
                "timeframe": timeframe_opt,
                "mode": "Auto" if is_auto_mode else "Manual",
                "metrics": metrics,
                "signal_df": signal_df,
                "backtest_df": backtest_df,
                "trades": trades,
                "fig": fig,
                "plotly_fig": plotly_fig,
                "z_threshold": z_threshold,
                "initial_capital": float(initial_capital),
            }
            st.session_state["bt_key"] = run_key
        else:
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
    if "bt_key" in st.session_state and st.session_state["bt_key"] != run_key:
        st.warning("⚠️ 检测到左侧配置参数已被修改（当前仍展示上一次回测结果）。请点击左侧「🚀 运行 Pro 回测」按钮以应用新参数运行。", icon="🔔")
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
                t_date = t.get('date', t.get('datetime', ''))
                t_date_str = t_date.strftime('%Y-%m-%d %H:%M') if hasattr(t_date, 'strftime') else str(t_date)[:16]
                summary_trades.append({
                    '序号': idx,
                    '成交日期': t_date_str,
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
