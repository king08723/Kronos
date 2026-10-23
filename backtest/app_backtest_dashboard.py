"""
app_backtest_dashboard.py
===========================================================
Kronos 量化策略研发与回测中枢看板 (Unified Quant Backtest Hub)
===========================================================
整合「股票池批量回测大盘」与「单标的微观深度归因」，
支持 1分钟/30分钟/60分钟/日线等多周期回测、高频流动性整手截断、
平方根动态冲击滑点、日内强平、跨周期宏观趋势共振与穿透下钻。
"""

import os
import sys
import time
import json
import inspect
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")
for p in [PROJECT_ROOT, CURRENT_DIR, EXAMPLES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_backtest_kronos_pro import (
    KronosDataFetcher,
    KronosBacktesterPro,
    MarketProfile,
    SymbolResolver,
    get_available_stocks,
    setup_logger,
    inject_continuous_crosshair_script,
    clear_logs,
    set_logging_state,
)

from run_backtest_batch import (
    BatchBacktester,
    DEFAULT_BATCH_CONFIG,
    parse_stock_item,
    load_stock_pool_from_file,
)

from backtest.core import (
    get_timeframe_defaults,
    resolve_timeframe_params,
    TIMEFRAME_BEST_PRACTICES,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# 回测产物收敛目录
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")
BATCH_OUTPUT_DIR = os.path.join(CURRENT_DIR, "batch")

def resolve_data_dir() -> str:
    hist_data = os.path.join(CURRENT_DIR, "historical_data")
    if os.path.exists(hist_data):
        return hist_data
    local_data = os.path.join(CURRENT_DIR, "data")
    os.makedirs(local_data, exist_ok=True)
    return local_data

def resolve_pred_dir() -> str:
    local_pred = os.path.join(CURRENT_DIR, "yuce")
    os.makedirs(local_pred, exist_ok=True)
    return local_pred

DATA_DIR = resolve_data_dir()
PRED_DIR = resolve_pred_dir()

# 预设股票池定义
PRESET_POOLS = {
    "🌐 跨市场标杆基准池 (6只)": [
        {"stock_code": "300418", "stock_name": "昆仑万维", "market": "CN"},
        {"stock_code": "600036", "stock_name": "招商银行", "market": "CN"},
        {"stock_code": "00700",  "stock_name": "腾讯控股", "market": "HK"},
        {"stock_code": "09988",  "stock_name": "阿里巴巴-W", "market": "HK"},
        {"stock_code": "AAPL",   "stock_name": "苹果公司", "market": "US"},
        {"stock_code": "TSLA",   "stock_name": "特斯拉",   "market": "US"},
    ],
    "🇨🇳 A股高波与蓝筹精选池 (4只)": [
        {"stock_code": "300418", "stock_name": "昆仑万维", "market": "CN"},
        {"stock_code": "600036", "stock_name": "招商银行", "market": "CN"},
        {"stock_code": "000001", "stock_name": "平安银行", "market": "CN"},
        {"stock_code": "603288", "stock_name": "海天味业", "market": "CN"},
    ],
    "🇭🇰 港股科技与金融核心池 (4只)": [
        {"stock_code": "00700",  "stock_name": "腾讯控股", "market": "HK"},
        {"stock_code": "09988",  "stock_name": "阿里巴巴-W", "market": "HK"},
        {"stock_code": "03690",  "stock_name": "美团-W",   "market": "HK"},
        {"stock_code": "00941",  "stock_name": "中国移动", "market": "HK"},
    ],
    "🇺🇸 美股科技与新能源领头羊 (4只)": [
        {"stock_code": "AAPL",   "stock_name": "苹果公司", "market": "US"},
        {"stock_code": "TSLA",   "stock_name": "特斯拉",   "market": "US"},
        {"stock_code": "MSFT",   "stock_name": "微软",     "market": "US"},
        {"stock_code": "NVDA",   "stock_name": "英伟达",   "market": "US"},
    ],
}

STOCK_NAMES = {it['stock_code']: it['stock_name'] for pool in PRESET_POOLS.values() for it in pool}


# ===========================================================================
# 辅助数据加载与投资组合计算工具
# ===========================================================================

def load_existing_batch_ranking(output_dir: str = BATCH_OUTPUT_DIR) -> Optional[pd.DataFrame]:
    """读取已有批量回测排行榜 CSV"""
    ranking_csv = os.path.join(output_dir, "batch_backtest_ranking.csv")
    if os.path.exists(ranking_csv):
        try:
            df = pd.read_csv(ranking_csv, encoding='utf-8-sig')
            return df
        except Exception:
            return None
    return None


def get_existing_batch_stocks(output_dir: str = BATCH_OUTPUT_DIR) -> List[str]:
    """获取 batch 目录中已有结果的股票列表"""
    if not os.path.exists(output_dir):
        return []
    codes = []
    for item in os.listdir(output_dir):
        p = os.path.join(output_dir, item)
        if os.path.isdir(p) and not item.startswith('.'):
            codes.append(item)
    return sorted(codes)


def load_stock_batch_detail(stock_code: str, output_dir: str = BATCH_OUTPUT_DIR) -> Dict[str, Any]:
    """加载单只股票在批量回测目录中的明细数据"""
    sdir = os.path.join(output_dir, stock_code)
    detail: Dict[str, Any] = {
        'code': stock_code,
        'metrics': {},
        'trades': None,
        'daily': None,
        'signals': None,
        'plot_img': None,
    }
    if not os.path.exists(sdir):
        return detail

    m_path = os.path.join(sdir, f"{stock_code}_metrics.json")
    if os.path.exists(m_path):
        try:
            with open(m_path, 'r', encoding='utf-8') as f:
                detail['metrics'] = json.load(f)
        except Exception:
            pass

    t_path = os.path.join(sdir, f"{stock_code}_trades.csv")
    if os.path.exists(t_path):
        try:
            detail['trades'] = pd.read_csv(t_path, encoding='utf-8-sig')
        except Exception:
            pass

    for d_name in [f"{stock_code}_daily_backtest.csv", f"{stock_code}_backtest.csv", f"{stock_code}_daily_equity.csv"]:
        d_path = os.path.join(sdir, d_name)
        if os.path.exists(d_path):
            try:
                detail['daily'] = pd.read_csv(d_path, encoding='utf-8-sig')
                break
            except Exception:
                pass

    s_path = os.path.join(sdir, f"{stock_code}_signals.csv")
    if os.path.exists(s_path):
        try:
            detail['signals'] = pd.read_csv(s_path, encoding='utf-8-sig')
        except Exception:
            pass

    log_path = os.path.join(sdir, f"{stock_code}_backtest.log")
    if not os.path.exists(log_path):
        cand = os.path.join(os.path.dirname(os.path.dirname(sdir)), "output", f"{stock_code}_backtest.log")
        if os.path.exists(cand):
            log_path = cand
        else:
            cand_global = os.path.join(os.path.dirname(os.path.dirname(sdir)), "output", "kronos_backtest.log")
            if os.path.exists(cand_global):
                log_path = cand_global
    if os.path.exists(log_path):
        detail['log_file'] = log_path

    img_path = os.path.join(sdir, f"{stock_code}_backtest_pro.png")
    if os.path.exists(img_path):
        detail['plot_img'] = img_path

    return detail


def calculate_portfolio_equity_curve(output_dir: str = BATCH_OUTPUT_DIR) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, float]]:
    """
    合成投资组合等权收益率曲线与各标的横向对比：
    各标的先算每个时步的收益率，再对「当前有效」的标的做等权平均。
    休市或无行情标的不参与权重，禁止 ffill 把休市填成 0 收益。
    返回: (组合合成曲线 DataFrame, 各单标的累计收益 DataFrame, 组合统计指标字典)
    """
    codes = get_existing_batch_stocks(output_dir)
    ret_series = []
    cum_series = []

    for code in codes:
        sdir = os.path.join(output_dir, code)
        target_df = None
        for cand in [f"{code}_daily_backtest.csv", f"{code}_backtest.csv", f"{code}_daily_equity.csv"]:
            p = os.path.join(sdir, cand)
            if os.path.exists(p):
                try:
                    target_df = pd.read_csv(p, encoding='utf-8-sig')
                    break
                except Exception:
                    continue

        if target_df is not None and not target_df.empty:
            date_col = next((c for c in ['datetime', 'date', 'Date', '日期'] if c in target_df.columns), target_df.columns[0])
            val_col = next((c for c in ['portfolio_value', 'net_value', '净值'] if c in target_df.columns), None)
            if val_col:
                try:
                    df = target_df.copy()
                    df['ts'] = pd.to_datetime(df[date_col])
                    df = df.sort_values('ts')
                    nav = df[val_col].astype(float)
                    daily_ret = nav.pct_change()
                    cum_ret = (nav / nav.iloc[0] - 1.0) * 100

                    sub_ret = pd.DataFrame({'ts': df['ts'].values, f'{code}_ret': daily_ret.values}).set_index('ts')
                    sub_cum = pd.DataFrame({'ts': df['ts'].values, f'{code}': cum_ret.values}).set_index('ts')
                    ret_series.append(sub_ret)
                    cum_series.append(sub_cum)
                except Exception:
                    continue

    if not ret_series:
        return None, None, {}

    # 外连接对齐时间序列，保留 NaN
    merged_ret = pd.concat(ret_series, axis=1).sort_index()
    merged_cum = pd.concat(cum_series, axis=1).sort_index() if cum_series else None
    if merged_ret.empty:
        return None, None, {}

    # 等权合成：仅对有效标的做平均
    merged_ret['Portfolio_Step_Return'] = merged_ret.mean(axis=1, skipna=True)
    valid_returns = merged_ret['Portfolio_Step_Return'].dropna()
    if valid_returns.empty:
        return None, None, {}

    basket = (1.0 + valid_returns).cumprod()
    portfolio_df = pd.DataFrame({
        'ts': basket.index,
        'Portfolio_Basket': basket.values,
    })
    portfolio_df['Portfolio_Cumulative_Return(%)'] = (portfolio_df['Portfolio_Basket'] - 1.0) * 100
    portfolio_df['Portfolio_Drawdown(%)'] = (
        portfolio_df['Portfolio_Basket'] / portfolio_df['Portfolio_Basket'].cummax() - 1.0
    ) * 100

    # 计算组合风险指标
    n_steps = len(valid_returns)
    total_cum_ret = portfolio_df['Portfolio_Cumulative_Return(%)'].iloc[-1]
    max_dd = abs(portfolio_df['Portfolio_Drawdown(%)'].min())

    # 年化收益率与波动率估算 (假定基准年化步长为 252)
    ann_factor = 252.0
    mean_ret = valid_returns.mean()
    std_ret = valid_returns.std()

    ann_ret = ((1.0 + total_cum_ret / 100.0) ** (ann_factor / max(1, n_steps)) - 1.0) * 100 if n_steps > 0 else 0.0
    ann_vol = (std_ret * np.sqrt(ann_factor)) * 100 if std_ret > 0 else 0.0
    rf = 2.0  # 2% 无风险利率
    sharpe = (ann_ret - rf) / max(1e-6, ann_vol) if ann_vol > 0 else 0.0
    calmar = (ann_ret / max(0.01, max_dd)) if max_dd > 0 else 0.0

    stats = {
        'total_cum_ret': total_cum_ret,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'active_assets': len(codes),
    }

    return portfolio_df.reset_index(drop=True), merged_cum, stats


# ===========================================================================
# 视图 1：股票池批量回测大盘 (Batch Hub)
# ===========================================================================

def render_batch_mode():
    st.header("🏆 股票池批量量化回测大盘与横截面筛选")
    st.caption("多时间周期 · 动态滑点与流动性截断 · 宏观大趋势共振 · 组合绩效大盘与全真归因")

    # 1. 侧边栏：多维度参数配置
    with st.sidebar:
        st.subheader(":material/dataset: 1. 股票池设定")
        pool_mode = st.radio(
            "选择股票池来源",
            options=["🌟 推荐精品预设池", "📄 上传文件 (.csv/.json)", "✏️ 内联自定义录入"],
            index=0,
        )

        selected_pool = []
        if pool_mode.startswith("🌟"):
            preset_name = st.selectbox("选择预设池", list(PRESET_POOLS.keys()), index=0)
            selected_pool = PRESET_POOLS[preset_name]
            st.session_state["active_batch_preset"] = preset_name
        elif pool_mode.startswith("📄"):
            uploaded_file = st.file_uploader("上传股票池文件", type=["csv", "txt", "json"])
            if uploaded_file is not None:
                try:
                    temp_p = os.path.join(CURRENT_DIR, f"_temp_pool_{int(time.time())}.csv")
                    with open(temp_p, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    selected_pool = load_stock_pool_from_file(temp_p)
                    if os.path.exists(temp_p):
                        os.remove(temp_p)
                    st.success(f"成功解析 {len(selected_pool)} 只标的！")
                except Exception as ex:
                    st.error(f"解析文件失败: {ex}")
            else:
                st.caption("提示：CSV 格式需包含 `stock_code` 列，可选 `stock_name`, `market`。")
                selected_pool = DEFAULT_BATCH_CONFIG["stock_pool"]
            st.session_state["active_batch_preset"] = "file_upload"
        else:
            inline_text = st.text_area(
                "输入标的（格式: 代码:名称:市场，逗号隔开）",
                value="300418:昆仑万维:CN, 600036:招商银行:CN, 00700:腾讯控股:HK, AAPL:苹果公司:US",
                help="例: 300418:昆仑万维:CN, 00700:腾讯控股:HK, AAPL:苹果公司:US"
            )
            selected_pool = [parse_stock_item(it) for it in inline_text.split(',') if it.strip()]
            st.session_state["active_batch_preset"] = "custom_inline"

        st.session_state["active_batch_pool"] = selected_pool

        st.caption(f"当前池包含 **{len(selected_pool)}** 只标的：")
        pool_summary_items = [f"`{it.get('stock_code')}` {it.get('stock_name')} ({it.get('market')})" for it in selected_pool]
        st.write(", ".join(pool_summary_items[:8]) + ("..." if len(pool_summary_items) > 8 else ""))

        st.divider()

        # 2. K线周期与时界推演
        st.subheader(":material/schedule: 2. K线周期与时界推演")
        timeframe_options = ["1d", "60m", "30m", "15m", "5m", "1m"]
        synced_tf = st.session_state.get("active_timeframe", "1d")
        default_batch_tf_idx = timeframe_options.index(synced_tf) if synced_tf in timeframe_options else 0
        timeframe = st.selectbox(
            "K线时间粒度 (TimeFrame)",
            options=timeframe_options,
            index=default_batch_tf_idx,
            key="batch_timeframe_selector",
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
        st.session_state["active_timeframe"] = timeframe

        auto_adapt_params = st.toggle(
            "🤖 启用周期最优参数自适应 (推荐)",
            value=True,
            help="开启后：切换 1d/60m/30m/15m/5m/1m 时，系统自动将止损、移动止盈、最大持仓时界、信噪比阈值、滑点冲击模型与日内平仓对齐量化实盘最佳实践！仍可在下方手动调整覆盖。"
        )
        tf_defaults = get_timeframe_defaults(timeframe=timeframe, market="CN")

        with st.container(border=True):
            st.markdown(f"**🎯 周期量化画像 ({timeframe})**：{tf_defaults.get('description', '')}")
            m_c1, m_c2, m_c3 = st.columns(3)
            m_c1.caption(f"🛡️ 推荐止损: `{tf_defaults['hard_stop']*100:.1f}%`")
            m_c2.caption(f"📈 移动止盈: `{tf_defaults['trail_stop']*100:.1f}%`")
            m_c3.caption(f"⏳ 最大持仓: `{tf_defaults['max_hold']}` Bar")
            m_c4, m_c5, m_c6 = st.columns(3)
            m_c4.caption(f"📶 信噪比门槛: `Z≥{tf_defaults['z_threshold']}`")
            m_c5.caption(f"💧 滑点模型: `{tf_defaults['slippage_model']}`")
            m_c6.caption(f"⚡ MTF双周期: `{'启用' if tf_defaults['use_mtf_trend_filter'] else '无'}`")

        horizon_choice = st.radio(
            "时界模式 (Forecast Horizon)",
            options=["🤖 Auto 极值自适应 (动态波峰最优持仓)", "⚙️ 固定时界 (Fixed Horizon)"],
            index=0,
            help="【Auto 极值自适应】：前瞻推演走势，自动寻找未来波峰极大值，动态确定最优买卖点。"
        )
        default_max_horizon = int(tf_defaults.get('max_auto_horizon', 10)) if auto_adapt_params else 10
        if horizon_choice.startswith("🤖"):
            horizon_val = "auto"
            max_auto_horizon = st.slider(
                "最大推演搜索跨度 (K线根数 / Bars)", min_value=2, max_value=25, value=default_max_horizon, step=1,
                key=f"max_auto_horizon_{timeframe}",
                help="Auto 模式向前推演搜索的最大跨度（日线为交易日，分时为 Bar 根数）。"
            )
            if max_auto_horizon > 15:
                st.warning("⚠️ 推演时界超过 15 步，模型波峰定位精度可能下降，建议 6~12 步。", icon="⚠️")
        else:
            horizon_val = st.selectbox("固定时界 (K线根数 / Bars)", [1, 2, 3, 5, 8, 10, 16, 20], index=2, key=f"fixed_horizon_{timeframe}")
            max_auto_horizon = horizon_val if isinstance(horizon_val, int) else 10

        # 时间窗口自适应（严格以 K 线根数 Bar 数量为准）
        is_intra = (timeframe != '1d')
        default_test_bars = int(tf_defaults.get('test_bars', 200 if is_intra else 120)) if auto_adapt_params else (200 if is_intra else 120)
        default_lookback_bars = int(tf_defaults.get('lookback_bars', 160 if is_intra else 90)) if auto_adapt_params else (160 if is_intra else 90)
        default_hist_days = int(tf_defaults.get('history_days', 90 if is_intra else 700)) if auto_adapt_params else (90 if is_intra else 700)

        # 换算等效交易天数提示
        bars_per_day_est = {'1d': 1, '60m': 4, '30m': 8, '15m': 16, '5m': 48, '1m': 240}.get(timeframe, 8)
        approx_test_days = round(default_test_bars / bars_per_day_est, 1)
        approx_lookback_days = round(default_lookback_bars / bars_per_day_est, 1)

        if is_intra:
            test_bars = st.slider(
                "回测验证窗口 (K线根数 / Bars)", min_value=30, max_value=600, value=default_test_bars, step=10,
                key=f"test_bars_{timeframe}",
                help=f"{timeframe} 推荐约 {default_test_bars} 根 Bar 作为样本外测试区间（以每日约 {bars_per_day_est} 根 Bar 计，约合 {approx_test_days} 个交易日）"
            )
            lookback_bars = st.slider(
                "预测输入窗口 (K线根数 / Bars)", min_value=30, max_value=500, value=default_lookback_bars, step=10,
                key=f"lookback_bars_{timeframe}",
                help=f"{timeframe} 推荐约 {default_lookback_bars} 根 Bar 历史特征输入（约合 {approx_lookback_days} 个交易日，拟合 Kronos 512 上下文）"
            )
            history_days = st.slider(
                "历史数据获取天数 (自然日)", min_value=15, max_value=365, value=default_hist_days, step=5,
                key=f"hist_days_{timeframe}",
                help=f"数据源分时 K 线历史拉取跨度，推荐 {default_hist_days} 天"
            )
        else:
            test_bars = st.slider(
                "回测验证窗口 (K线根数 / 交易日)", min_value=20, max_value=250, value=default_test_bars, step=10,
                key=f"test_bars_{timeframe}",
                help="日线级别 1 根 Bar = 1 交易日。最后 N 个交易日作为样本外测试区间"
            )
            lookback_bars = st.slider(
                "预测输入窗口 (K线根数 / 交易日)", min_value=30, max_value=200, value=default_lookback_bars, step=5,
                key=f"lookback_bars_{timeframe}",
                help="日线级别 1 根 Bar = 1 交易日。每次预测使用的历史窗口长度，默认 90 天对齐官方训练先验"
            )
            history_days = st.slider(
                "历史数据获取天数 (自然日)", min_value=200, max_value=1200, value=default_hist_days, step=50,
                key=f"hist_days_{timeframe}"
            )

        lookback = lookback_bars  # 兼容别名
        def_step_size = int(tf_defaults.get('step_size', 1)) if auto_adapt_params else 1
        step_size = st.slider(
            "滚动滑动步长 (Step Size)",
            min_value=1, max_value=5, value=def_step_size, step=1,
            key=f"batch_step_size_{timeframe}",
            help=(
                "【Step Size = 1】：逐根 Bar 执行真实 Kronos 预测，无任何插值，信号时间戳完整连续（推荐高精度评估）。\n"
                "【Step Size > 1】：跳步推演并线性插值中间点（提速 100%+）。为保证量化严谨性，回测系统会自动剔除插值点，仅在真实预测时刻出信号。"
            )
        )
        samples = st.slider("蒙特卡洛采样数", min_value=1, max_value=60, value=35, step=1,
                            help="每个时步独立采样路径数，推荐 35 保证分位数估计稳定")
        force_refresh = st.checkbox("强制刷新网络行情与预测", value=False)

        st.divider()

        # 3. 高频流动性与冲击滑点
        st.subheader(":material/water_drop: 3. 高频流动性与滑点")
        default_cap_enable = (tf_defaults['volume_participation_cap'] is not None) if auto_adapt_params else False
        enable_volume_cap = st.checkbox(
            "启用单 Bar 流动性截断",
            value=default_cap_enable,
            key=f"enable_volume_cap_{timeframe}",
            help="单笔大单在单根 Bar 撮合时的成交量上限，超出部分整手截断，防止市场冲击"
        )
        if enable_volume_cap:
            default_cap_val = int((tf_defaults['volume_participation_cap'] or 0.10) * 100) if auto_adapt_params else 10
            volume_cap_val = st.slider(
                "单 Bar 成交量占比上限 (%)", 1, 20, default_cap_val, 1,
                key=f"volume_cap_val_{timeframe}"
            ) / 100.0
        else:
            volume_cap_val = None

        default_slip_idx = (1 if tf_defaults['slippage_model'] == 'dynamic' else 0) if auto_adapt_params else 0
        slippage_mode = st.radio(
            "滑点模型",
            options=["固定滑点 (Fixed)", "平方根动态冲击滑点 (Dynamic)"],
            index=default_slip_idx,
            key=f"slippage_mode_{timeframe}",
            help="【固定滑点】：按基准点差执行；【动态滑点】：基于微观市场平方根冲击模型与实时成交占比计算。"
        )
        eff_slippage_model = "dynamic" if "Dynamic" in slippage_mode else "fixed"

        col_cost1, col_cost2 = st.columns(2)
        with col_cost1:
            commission_bps = st.number_input("佣金 (万分之)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
        with col_cost2:
            slippage_bps = st.number_input("基准滑点 (万分之)", min_value=0.0, max_value=30.0, value=10.0, step=1.0)

        st.divider()

        # 4. 资金风控与宏观大趋势
        st.subheader(":material/shield: 4. 资金风控与趋势滤网")
        capital_input = st.number_input("全局统一初始资金 (0=按市场自适应)", min_value=0, value=0, step=50000,
                                        help="设为 0 时：A股默认10万，港股默认30万，美股默认5万")
        final_capital = None if capital_input == 0 else float(capital_input)

        default_z = float(tf_defaults['z_threshold']) if auto_adapt_params else 0.55
        z_threshold = st.slider(
            "入场信噪比门槛 (Z-Score)", min_value=0.20, max_value=2.00, value=default_z, step=0.05,
            key=f"z_thresh_{timeframe}",
            help="预测超额收益相对预期波动率的标准差倍数门槛，自适应按周期的信噪比推荐"
        )

        default_pct = float(tf_defaults['entry_percentile']) if auto_adapt_params else 0.80
        entry_percentile = st.slider(
            "入场自身分位阈值", 0.50, 0.98, default_pct, 0.01,
            key=f"entry_pct_{timeframe}",
            help="仅在信号强度进入自身历史前百分比时入场，自动消除量纲偏差"
        )

        default_hard_stop = round(tf_defaults['hard_stop'] * 100, 2) if auto_adapt_params else 4.0
        hard_stop = st.slider(
            "基准硬止损线 (%)", 0.2, 15.0, default_hard_stop, 0.1,
            key=f"hard_stop_{timeframe}",
            help="周期自适应推荐基准止损线，高频分钟线对应更紧密止损"
        ) / 100.0

        default_trail_stop = round(tf_defaults['trail_stop'] * 100, 2) if auto_adapt_params else 2.5
        trail_stop = st.slider(
            "移动止盈回撤 (%)", 0.1, 10.0, default_trail_stop, 0.1,
            key=f"trail_stop_{timeframe}",
            help="持仓最高点回撤止盈幅度"
        ) / 100.0

        default_max_hold = int(tf_defaults['max_hold']) if auto_adapt_params else 15
        max_hold = st.slider(
            "最大持仓周期数 (Bar数/天数)", 1, 120, default_max_hold, 1,
            key=f"max_hold_{timeframe}",
            help="单笔交易最大持有周期数（日线为天数，分钟线为 Bar 根数）"
        )

        min_lot_guarantee = st.checkbox("一手保底机制（防港美股高单价抹零）", value=True)
        intraday_stop = st.checkbox("盘中止损（遵守市场 T+N 合规）", value=True)

        default_eod_exit = bool(tf_defaults['intraday_exit_at_eod']) if auto_adapt_params else False
        intraday_exit_eod = st.checkbox(
            "日内收盘强平 (不留隔夜仓)", value=default_eod_exit,
            key=f"eod_exit_{timeframe}",
            help="在交易日最后一根 Bar 强制清平所有多头仓位，仅做日冲交易，规避隔夜系统性风险（A股T+1自动保持合规）。"
        )

        default_gap_prot = bool(tf_defaults['overnight_gap_protection']) if auto_adapt_params else True
        overnight_gap_protection = st.checkbox(
            "开启隔夜跳空防守", value=default_gap_prot,
            key=f"gap_prot_{timeframe}",
            help="次日低开跌破止损线以实际开盘价真实穿透止损，避免盘中止损线失效产生虚高收益。"
        )

        default_mtf = bool(tf_defaults['use_mtf_trend_filter']) if auto_adapt_params else True
        mtf_trend_filter = st.checkbox(
            "跨周期宏观趋势滤网 (MTF Filter)", value=default_mtf,
            key=f"mtf_flt_{timeframe}",
            help="在小周期（如 30m / 60m）回测时，以零未来函数引入日线级别 MA20/MA60 宏观大趋势滤网进行双周期共振准入。"
        )
        require_skill = st.checkbox("预测技能体检未通过时直接拦截", value=False)

        st.divider()

        # 5. 算力并发与日志
        st.subheader(":material/memory: 5. 算力并发与日志")
        max_workers = st.slider("行情数据并发准备线程", 1, 12, 6)
        parallel_stocks = st.slider(
            "模型预测回测并发标的数", 1, 4, 1,
            help="注意：在 Apple Silicon (Mac MPS) 环境下，深度学习推理已启用底层互斥锁排队保护；建议保持为 1 以获最佳稳定性。"
        )
        enable_file_log = st.toggle("开启运行日志落盘", value=True)
        set_logging_state(enabled=True, file_logging=enable_file_log, log_dir=BATCH_OUTPUT_DIR)

        st.divider()

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            run_batch_btn = st.button("🚀 启动批量回测", type="primary", width="stretch")
        with col_b2:
            dry_run_btn = st.button("🔍 预检行情", width="stretch")

    # 2. 状态与进度反馈区
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    # 执行预检
    if dry_run_btn:
        with st.spinner(f"正在并发拉取各标的 {timeframe} 行情并预检有效性..."):
            engine = BatchBacktester(log_dir=BATCH_OUTPUT_DIR)
            valid = engine.prepare_data_pool(
                stock_list=selected_pool, data_dir=DATA_DIR,
                history_days=history_days, lookback=lookback_bars,
                test_days=test_bars, force_refresh=force_refresh,
                max_workers=max_workers, timeframe=timeframe,
                lookback_bars=lookback_bars, test_bars=test_bars,
            )
            st.success(f"✅ 行情预检完成：**{len(valid)}** 只标的数据充足合格，可正常回测！")
            preview_rows = []
            col_bar_name = '日K线总条数' if timeframe == '1d' else f'{timeframe}K线总条数'
            for c, item in valid.items():
                preview_rows.append({
                    '代码': c, '名称': item['stock_name'], '市场': item['market'],
                    col_bar_name: item['total_bars'], '状态': '合格 🟢'
                })
            st.dataframe(pd.DataFrame(preview_rows), width="stretch")

    # 执行批量回测
    if run_batch_btn:
        engine = BatchBacktester(log_dir=BATCH_OUTPUT_DIR)
        bar = progress_placeholder.progress(0, text=f"[1/4] 正在并发准备各标的 {timeframe} 历史行情...")

        def _data_cb(done, total, code, name, status):
            pct = int((done / max(1, total)) * 30)
            bar.progress(pct, text=f"[1/4] 准备行情 ({done}/{total}): {name}({code}) -> {status}")

        valid = engine.prepare_data_pool(
            stock_list=selected_pool, data_dir=DATA_DIR,
            history_days=history_days, lookback=lookback_bars,
            test_days=test_bars, force_refresh=force_refresh,
            max_workers=max_workers, progress_callback=_data_cb,
            timeframe=timeframe,
            lookback_bars=lookback_bars, test_bars=test_bars,
        )

        if not valid:
            st.error("❌ 无标的通过行情有效性预检，无法启动批量回测。")
        else:
            bar.progress(35, text=f"[2/4] 行情就绪（{len(valid)}只），开始时序推演与状态机撮合...")
            status_container = status_placeholder.container(border=True)
            adapt_str = "自适应最佳实践" if auto_adapt_params else "手动配置"
            status_container.info(f"🚀 正在对 {len(valid)} 只标的执行批量量化回测流水线 (周期: {timeframe} | 参数模式: {adapt_str})...")

            def _bt_cb(done, total, code, name, st_val, dur, res):
                pct = 35 + int((done / max(1, total)) * 60)
                tag = "✅ 成功" if st_val == 'SUCCESS' else "❌ 失败"
                bar.progress(pct, text=f"[2/4] 回测中 ({done}/{total}): {name}({code}) {tag} (耗时{dur:.1f}s)")

            results = engine.run_batch_backtest(
                valid_data_pool=valid, data_dir=DATA_DIR,
                pred_dir=PRED_DIR, backtest_batch_dir=BATCH_OUTPUT_DIR,
                parallel_stocks=parallel_stocks,
                auto_predict=True, test_days=test_bars,
                lookback=lookback_bars, step_size=step_size,
                sample_count=samples, forecast_horizon=horizon_val,
                max_auto_horizon=max_auto_horizon,
                force_refresh=force_refresh,
                test_bars=test_bars, lookback_bars=lookback_bars,
                initial_capital=final_capital,
                z_threshold=z_threshold,
                entry_percentile=entry_percentile,
                hard_stop=hard_stop, trail_stop=trail_stop,
                max_hold=max_hold,
                min_lot_guarantee=min_lot_guarantee,
                intraday_stop=intraday_stop,
                require_skill=require_skill,
                generate_plots=True,
                enable_log=enable_file_log,
                timeframe=timeframe,
                volume_participation_cap=volume_cap_val,
                slippage_model=eff_slippage_model,
                intraday_exit_at_eod=intraday_exit_eod,
                overnight_gap_protection=overnight_gap_protection,
                use_mtf_trend_filter=mtf_trend_filter,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                progress_callback=_bt_cb,
                auto_adapt_params=auto_adapt_params,
            )

            bar.progress(95, text="[3/4] 正在统计横截面排行榜与投资组合指标...")
            ranking_df = engine.generate_ranking_summary(results, BATCH_OUTPUT_DIR)
            engine.export_visualizations_summary(results, BATCH_OUTPUT_DIR)
            bar.progress(100, text="🎉 批量量化回测全部完毕！")
            st.session_state["batch_ranking"] = ranking_df
            st.session_state["batch_results"] = results
            st.toast("🎉 批量回测已执行完毕，最新绩效大盘已刷新！", icon="🚀")

    # 3. 自动扫描与展示结果
    ranking_df = st.session_state.get("batch_ranking")
    if ranking_df is None:
        ranking_df = load_existing_batch_ranking(BATCH_OUTPUT_DIR)
        if ranking_df is not None:
            st.session_state["batch_ranking"] = ranking_df

    if ranking_df is None or ranking_df.empty:
        st.info("💡 尚未检测到批量回测历史记录，请在左侧侧边栏配置参数后点击「🚀 启动批量回测」。", icon="ℹ️")
        return

    # ---------- 顶层 KPI 概览卡片 ----------
    valid_sub = ranking_df[ranking_df['回测状态'] == '成功'] if '回测状态' in ranking_df.columns else ranking_df
    total_count = len(ranking_df)
    success_count = len(valid_sub)

    with st.container(border=True):
        st.subheader(":material/analytics: 股票池综合表现 KPI 概览")
        k1, k2, k3, k4, k5, k6 = st.columns(6)

        avg_ann = valid_sub['年化收益率(%)'].mean() if not valid_sub.empty and '年化收益率(%)' in valid_sub.columns else 0.0
        avg_cum = valid_sub['累计收益率(%)'].mean() if not valid_sub.empty and '累计收益率(%)' in valid_sub.columns else 0.0
        avg_dd  = valid_sub['最大回撤(%)'].mean() if not valid_sub.empty and '最大回撤(%)' in valid_sub.columns else 0.0
        avg_wr  = valid_sub['胜率(%)'].mean() if not valid_sub.empty and '胜率(%)' in valid_sub.columns else 0.0

        tradable_cnt = sum(1 for v in valid_sub['体检建议'] if '可实盘' in str(v)) if '体检建议' in valid_sub.columns else 0
        tradable_rate = (tradable_cnt / max(1, success_count)) * 100

        top_stock = valid_sub.iloc[0]['股票名称'] if not valid_sub.empty else "N/A"
        top_ann = valid_sub.iloc[0]['年化收益率(%)'] if not valid_sub.empty else 0.0

        k1.metric("标的规模", f"{total_count} 只", delta=f"{success_count} 只成功")
        k2.metric("池平均年化", f"{avg_ann:+.2f}%", delta=f"{avg_cum:+.2f}% 累计")
        k3.metric("池平均胜率", f"{avg_wr:.1f}%")
        k4.metric("池平均最大回撤", f"{avg_dd:.2f}%", delta_color="inverse")
        k5.metric("实盘达标率", f"{tradable_rate:.0f}%", delta=f"{tradable_cnt}/{success_count} 标的")
        k6.metric("最佳收益标的", f"{top_stock}", delta=f"{top_ann:+.2f}% 年化")

    # ---------- 核心功能 Tab 布局 ----------
    tab_rank, tab_market, tab_portfolio, tab_drill = st.tabs([
        "🏆 策略绩效排行榜", "🌐 多市场分盘洞察", "📈 组合等权合成走势", "🎯 标的穿透下钻 (Drill-Down)"
    ])

    # TAB 1: 策略绩效排行榜
    with tab_rank:
        st.markdown("**量化策略全标的绩效横截面排行榜**")
        c_filter1, c_filter2, c_filter3 = st.columns([3, 3, 3])
        with c_filter1:
            market_filter = st.selectbox("市场筛选", ["全部市场", "A股 (CN)", "港股 (HK)", "美股 (US)"], index=0)
        with c_filter2:
            status_filter = st.selectbox("体检建议筛选", ["全部状态", "仅看可实盘 🟢", "仅看需谨慎 🟡"], index=0)
        with c_filter3:
            ret_filter = st.selectbox("收益筛选", ["全部收益", "仅看正收益 (盈)", "仅看负收益 (亏)"], index=0)

        disp_df = ranking_df.copy()
        if market_filter != "全部市场":
            m_code = {'A股 (CN)': 'CN', '港股 (HK)': 'HK', '美股 (US)': 'US'}.get(market_filter)
            if '市场' in disp_df.columns:
                disp_df = disp_df[disp_df['市场'] == m_code]
        if status_filter == "仅看可实盘 🟢" and '体检建议' in disp_df.columns:
            disp_df = disp_df[disp_df['体检建议'].str.contains('可实盘', na=False)]
        elif status_filter == "仅看需谨慎 🟡" and '体检建议' in disp_df.columns:
            disp_df = disp_df[disp_df['体检建议'].str.contains('需谨慎', na=False)]
        if ret_filter == "仅看正收益 (盈)" and '年化收益率(%)' in disp_df.columns:
            disp_df = disp_df[disp_df['年化收益率(%)'] > 0]
        elif ret_filter == "仅看负收益 (亏)" and '年化收益率(%)' in disp_df.columns:
            disp_df = disp_df[disp_df['年化收益率(%)'] <= 0]

        col_config = {
            "累计收益率(%)": st.column_config.NumberColumn("累计收益率", format="%.2f%%"),
            "年化收益率(%)": st.column_config.NumberColumn("年化收益率", format="%.2f%%"),
            "最大回撤(%)": st.column_config.NumberColumn("最大回撤", format="%.2f%%"),
            "胜率(%)": st.column_config.NumberColumn("胜率", format="%.1f%%"),
            "夏普比率": st.column_config.NumberColumn("夏普比率", format="%.2f"),
            "卡玛比率": st.column_config.NumberColumn("卡玛比率", format="%.2f"),
            "盈亏比": st.column_config.NumberColumn("盈亏比", format="%.2f"),
            "交易笔数": st.column_config.NumberColumn("交易笔数", format="%d 笔"),
            "单笔均收益(%)": st.column_config.NumberColumn("单笔均收益", format="%.2f%%"),
        }

        st.dataframe(disp_df, column_config=col_config, width="stretch")

        csv_data = disp_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="💾 导出当前筛选排行榜 (CSV)",
            data=csv_data,
            file_name=f"kronos_batch_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        st.markdown("---")
        c_quick1, c_quick2 = st.columns([6, 4])
        with c_quick1:
            st.markdown("💡 **想要深入查看某只个股的走势、图表与成交流水？**")
            st.caption("1. **直接查看**：点击上方第 4 个标签页 **「🎯 标的穿透下钻 (Drill-Down)」** 即可在当前大盘内免重算浏览；\n2. **深度微调**：点击右侧按钮跳转至 **「🔬 单标的微观深度回测」** 专属工作台进行独立参数微调。")
        with c_quick2:
            batch_codes = [c for c in disp_df['代码'].tolist() if c] if '代码' in disp_df.columns else []
            if batch_codes:
                q_col1, q_col2 = st.columns([4, 6])
                with q_col1:
                    sel_goto = st.selectbox("选择目标标的", batch_codes, key="quick_goto_stock_sel", label_visibility="collapsed")
                with q_col2:
                    if st.button(f"🔬 深度穿透 {sel_goto}", key="btn_quick_goto_pro", width="stretch", type="primary"):
                        st.session_state["single_active_stock"] = sel_goto
                        st.session_state["app_main_work_mode"] = "🔬 单标的微观深度回测 (Single-Stock Pro)"
                        st.rerun()

    # TAB 2: 多市场分盘洞察
    with tab_market:
        st.markdown("**各市场微观结构与回测绩效横向对比**")
        st.caption("A股 (T+1、分板块手数与涨跌停、沪市过户费) vs 港股 (每手股数、双边印花税) vs 美股 (T+0、1股起买、卖方规费)")

        mkt_rows = []
        if '市场' in valid_sub.columns:
            for mkt, mkt_label in [('CN', 'A股市场'), ('HK', '港股市场'), ('US', '美股市场')]:
                sub = valid_sub[valid_sub['市场'] == mkt]
                if not sub.empty:
                    mkt_rows.append({
                        '市场分类': mkt_label,
                        '合格标的数': len(sub),
                        '平均年化(%)': round(sub['年化收益率(%)'].mean(), 2),
                        '平均累计(%)': round(sub['累计收益率(%)'].mean(), 2),
                        '平均最大回撤(%)': round(sub['最大回撤(%)'].mean(), 2),
                        '平均胜率(%)': round(sub['胜率(%)'].mean(), 1),
                        '总交易笔数': int(sub['交易笔数'].sum()),
                    })

        if mkt_rows:
            mkt_df = pd.DataFrame(mkt_rows)
            st.dataframe(mkt_df, width="stretch")

            # 柱状图对比
            chart_data = mkt_df[['市场分类', '平均年化(%)', '平均最大回撤(%)']].set_index('市场分类')
            st.bar_chart(chart_data)
        else:
            st.info("暂无分市场数据。")

        with st.expander("💡 市场微观结构与量化交易制度解析", expanded=True):
            st.markdown("""
- **美股 (US)**：T+0 交易，支持 1 股起买；默认零佣金，卖方计入 SEC 与 TAF 规费；年化基准 252 交易日。
- **港股 (HK)**：按标的自定义每手股数申报（如腾讯 100 股、移动 500 股）；T+0；双边印花税（0.10%）；系统启用一手保底机制防高价股抹零。
- **A股 (CN)**：严格执行 T+1 制度（开仓当日禁止平仓）；主板 100 股整数倍，科创板 200 股起；涨跌停限制（主板 ±10%，双创 ±20%，北交所 ±30%）；沪市另计过户费。
            """)

    # TAB 3: 投资组合等权合成走势
    with tab_portfolio:
        st.markdown("**股票池等权合成投资组合走势 (Portfolio Synthetic Basket)**")
        st.caption("按各标的在每个时步的收益率等权平均后复利；休市标的不参与当日权重分配，真实反映跨市场多品种资产组合表现。")

        portfolio_curve, multi_cum_df, port_stats = calculate_portfolio_equity_curve(BATCH_OUTPUT_DIR)
        if portfolio_curve is not None and not portfolio_curve.empty:
            # 组合关键指标卡片
            with st.container(border=True):
                p_c1, p_c2, p_c3, p_c4, p_c5, p_c6 = st.columns(6)
                p_c1.metric("组合累计收益率", f"{port_stats.get('total_cum_ret', 0):+.2f}%")
                p_c2.metric("组合年化收益率", f"{port_stats.get('ann_ret', 0):+.2f}%")
                p_c3.metric("年化波动率", f"{port_stats.get('ann_vol', 0):.2f}%")
                p_c4.metric("组合夏普比率", f"{port_stats.get('sharpe', 0):.2f}")
                p_c5.metric("组合历史最大回撤", f"{port_stats.get('max_dd', 0):.2f}%", delta_color="inverse")
                p_c6.metric("组合卡玛比率", f"{port_stats.get('calmar', 0):.2f}")

            # 组合走势图
            chart_df = portfolio_curve[['ts', 'Portfolio_Cumulative_Return(%)', 'Portfolio_Drawdown(%)']].set_index('ts')
            st.subheader("📈 投资组合等权复利净值曲线 (%)")
            st.line_chart(chart_df['Portfolio_Cumulative_Return(%)'])

            c_sub1, c_sub2 = st.columns(2)
            with c_sub1:
                st.subheader("📉 组合动态水下回撤曲线 (%)")
                st.line_chart(chart_df['Portfolio_Drawdown(%)'])
            with c_sub2:
                if multi_cum_df is not None and not multi_cum_df.empty:
                    st.subheader("📊 标的间收益贡献对比")
                    st.line_chart(multi_cum_df)
                else:
                    st.info("单标的数据无需进行横向对比。")
        else:
            st.info("暂无足够的逐日/逐Bar净值数据以合成组合曲线。")

    # TAB 4: 标的穿透下钻 (Drill-Down Detail)
    with tab_drill:
        st.markdown("**🎯 标的穿透下钻诊断 (Drill-Down)**")
        st.caption("从大盘排行无缝穿透至任意单只股票，查看微观四联回测图、成交流水对账表与逐日/逐Bar净值")

        all_batch_stocks = get_existing_batch_stocks(BATCH_OUTPUT_DIR)
        if not all_batch_stocks:
            st.info("暂无可下钻的标的数据")
            return

        preferred_drill = st.session_state.get("single_active_stock")
        default_drill_idx = all_batch_stocks.index(preferred_drill) if preferred_drill in all_batch_stocks else 0

        c_dsel1, c_dsel2 = st.columns([7, 3])
        with c_dsel1:
            target_c = st.selectbox(
                "选择下钻标的",
                all_batch_stocks,
                index=default_drill_idx,
                key="drill_stock_selector",
                format_func=lambda x: f"{x} ({STOCK_NAMES.get(x, '')})" if x in STOCK_NAMES else x
            )
            st.session_state["single_active_stock"] = target_c
        with c_dsel2:
            st.write("")
            st.write("")
            if st.button("🔬 前往单标的深度回测微调", key=f"btn_jump_pro_{target_c}", width="stretch"):
                st.session_state["single_active_stock"] = target_c
                st.session_state["app_main_work_mode"] = "🔬 单标的微观深度回测 (Single-Stock Pro)"
                st.rerun()

        detail = load_stock_batch_detail(target_c, BATCH_OUTPUT_DIR)

        prof = MarketProfile.for_stock(target_c)
        m = detail.get('metrics', {})

        # 单标的 KPI 卡片
        if m:
            with st.container(border=True):
                st.subheader(f"🔍 {target_c} [{prof.market}] · 核心指标")
                d1, d2, d3, d4, d5, d6 = st.columns(6)
                cs = prof.currency_symbol
                d1.metric("累计收益率", f"{m.get('累计收益率', 0):+.2%}")
                d2.metric("年化收益率", f"{m.get('年化收益率', 0):+.2%}")
                d3.metric("夏普比率", f"{m.get('夏普比率', 0):.2f}")
                d4.metric("最大回撤", f"{m.get('最大回撤', 0):.2%}", delta_color="inverse")
                d5.metric("胜率", f"{m.get('胜率', 0):.1%}")
                d6.metric("期末资产", f"{cs}{m.get('期末总资产', 0):,.0f}")

        # 图表与流水
        sub_d1, sub_d2, sub_d3, sub_d4, sub_d5 = st.tabs([
            "📈 回测全景四联图", "📋 逐笔成交流水", "📑 净值流水与明细", "🎯 预测与信号明细", "📄 运行诊断日志"
        ])
        with sub_d1:
            if HAS_PLOTLY and detail.get('daily') is not None:
                d_df = detail['daily'].copy()
                d_col = next((c for c in ['datetime', 'date', 'Date', '日期'] if c in d_df.columns), d_df.columns[0])
                d_df['date_dt'] = pd.to_datetime(d_df[d_col])
                d_df = d_df.set_index('date_dt')

                t_records = []
                if detail.get('trades') is not None and not detail['trades'].empty:
                    t_copy = detail['trades'].copy()
                    t_date_col = next((c for c in ['datetime', 'date', 'Date', '日期'] if c in t_copy.columns), 'date')
                    t_copy['date'] = pd.to_datetime(t_copy[t_date_col])
                    t_records = t_copy.to_dict('records')

                bt_pro = KronosBacktesterPro(data_dir=DATA_DIR, model_dir=PRED_DIR, enable_log=False)
                bt_pro.set_market(target_c, market=prof.market, enable_log=False)
                try:
                    p_fig = bt_pro.plot_backtest_interactive(
                        signal_df=pd.DataFrame(),
                        backtest_df=d_df,
                        metrics=m,
                        trades=t_records,
                        stock_code=target_c,
                        crosshair_mode="x"
                    )
                finally:
                    bt_pro.cleanup_logger()
                if p_fig is not None:
                    st.plotly_chart(p_fig, width="stretch", key=f"drill_plotly_{target_c}")
                    inject_continuous_crosshair_script()
                else:
                    st.warning("生成交互式图表失败")
            elif detail.get('plot_img') and os.path.exists(detail['plot_img']):
                st.image(detail['plot_img'], caption=f"{target_c} 量化回测四联诊断图", width="stretch")
            else:
                st.info("未找到该标的的预生成图表文件。")

        with sub_d2:
            trades_df = detail.get('trades')
            if trades_df is not None and not trades_df.empty:
                f_opt = st.selectbox("成交方向过滤", ["全部流水", "仅买入 (BUY)", "仅卖出 (SELL)"], index=0, key=f"tf_{target_c}")
                if f_opt == "仅买入 (BUY)":
                    trades_df = trades_df[trades_df['action'] == 'BUY']
                elif f_opt == "仅卖出 (SELL)":
                    trades_df = trades_df[trades_df['action'] == 'SELL']
                st.dataframe(trades_df, width="stretch")
            else:
                st.info("该标的在回测区间内未发生任何交易闭环。")

        with sub_d3:
            daily_df = detail.get('daily')
            if daily_df is not None and not daily_df.empty:
                st.dataframe(daily_df, width="stretch")
            else:
                st.info("暂无逐日/逐Bar净值数据。")

        with sub_d4:
            signals_df = detail.get('signals')
            if signals_df is not None and not signals_df.empty:
                st.dataframe(signals_df, width="stretch")
            else:
                st.info("暂无预测与信号明细数据（在回测运行完成后自动落盘）。")

        with sub_d5:
            col_l1, col_l2, col_l3 = st.columns([5, 2, 2])
            with col_l1:
                st.markdown(f"**标的 `{target_c}` 诊断日志流水**")
            with col_l2:
                if st.button("🗑️ 清空当前标的日志", key=f"btn_clr_{target_c}", width="stretch"):
                    cleared = clear_logs(log_dir=OUTPUT_DIR, stock_code=target_c)
                    if cleared:
                        st.toast(f"已清空 {target_c} 运行日志", icon="🧹")
                    else:
                        st.toast(f"未检索到 {target_c} 独立日志", icon="ℹ️")
                    st.rerun()
            with col_l3:
                if st.button("🧹 清空所有回测日志", key=f"btn_clr_all_{target_c}", width="stretch"):
                    cleared = clear_logs(log_dir=OUTPUT_DIR, clear_all=True)
                    st.toast(f"已清空全部回测日志 (共 {len(cleared)} 个文件)", icon="🧹")
                    st.rerun()

            log_file = detail.get('log_file')
            if log_file and os.path.exists(log_file) and os.path.getsize(log_file) > 0:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                stock_lines = [l for l in lines if target_c in l]
                disp_lines = stock_lines[-250:] if stock_lines else lines[-250:]
                st.caption(f"日志文件: `{log_file}` (展示最新 {len(disp_lines)} 行)")
                st.code("".join(disp_lines), language="log")
            else:
                st.info("暂未检索到对应标的的日志文件，或日志已被清空。")


# ===========================================================================
# 视图 2：单标的微观深度回测 (Single-Stock Pro)
# ===========================================================================

def render_single_stock_mode():
    from run_backtest_kronos_pro import streamlit_main as pro_single_main
    pro_single_main()


# ===========================================================================
# 主入口
# ===========================================================================

def streamlit_main():
    st.set_page_config(
        page_title="Kronos 量化策略研发与回测中枢",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📈 Kronos 量化策略研发与回测中枢 (Quant Backtest Hub)")
    st.caption("严格防偷价 (t→t+1) · Z-Score自适应波峰信号 · 多周期全真撮合与微观摩擦 · 投资组合横截面洞察")

    # 顶层工作模式切换
    mode_options = ["🏆 股票池批量回测大盘 (Batch Hub)", "🔬 单标的微观深度回测 (Single-Stock Pro)"]
    if "app_main_work_mode" not in st.session_state or st.session_state["app_main_work_mode"] not in mode_options:
        st.session_state["app_main_work_mode"] = mode_options[0]

    mode = st.segmented_control(
        "工作模式切换",
        options=mode_options,
        default=st.session_state["app_main_work_mode"],
        key="app_main_work_mode",
        label_visibility="collapsed",
    )

    if mode and mode.startswith("🔬"):
        render_single_stock_mode()
    else:
        render_batch_mode()


def streamlit_batch_main():
    """专供 run_backtest_batch.py 调用的入口"""
    streamlit_main()


if __name__ == "__main__":
    streamlit_main()
