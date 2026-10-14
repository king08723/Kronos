"""
app_backtest_dashboard.py
===========================================================
Kronos 量化策略研发与回测中枢看板 (Unified Quant Backtest Hub)
===========================================================
整合「股票池批量回测大盘」与「单标的微观深度归因」，
提供多标的横截面筛选、多市场分盘统计、等权投资组合走势与单标的穿透下钻 (Drill-Down)。
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
# 回测体系根目录与数据目录配置
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

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# 回测产物全部收敛于 backtest 目录下
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")
BATCH_OUTPUT_DIR = os.path.join(CURRENT_DIR, "batch")

# 行情与预测数据：全部统一收敛于 backtest/data 与 backtest/yuce
def resolve_data_dir() -> str:
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


# ===========================================================================
# 辅助数据加载工具
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
    """获取 batch_backtest 目录中已有结果的股票列表"""
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

    d_path = os.path.join(sdir, f"{stock_code}_daily_backtest.csv")
    if not os.path.exists(d_path):
        d_path = os.path.join(sdir, f"{stock_code}_daily_equity.csv")
    if os.path.exists(d_path):
        try:
            detail['daily'] = pd.read_csv(d_path, encoding='utf-8-sig')
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


def calculate_portfolio_equity_curve(output_dir: str = BATCH_OUTPUT_DIR) -> Optional[pd.DataFrame]:
    """
    合成投资组合等权收益率曲线：
    各标的先算日收益率，再对「当日有行情」的标的做等权平均。
    某市场休市日不参与权重，禁止 ffill 把休市填成 0 收益。
    这是跨市场收益率合成，不是换汇后的资产加总。
    """
    codes = get_existing_batch_stocks(output_dir)
    ret_series = []
    for code in codes:
        d_path = os.path.join(output_dir, code, f"{code}_daily_backtest.csv")
        if os.path.exists(d_path):
            try:
                df = pd.read_csv(d_path, encoding='utf-8-sig')
                date_col = next((c for c in ['date', 'Date', '日期'] if c in df.columns), df.columns[0])
                val_col = next((c for c in ['portfolio_value', 'net_value', '净值'] if c in df.columns), None)
                if val_col:
                    df['date_dt'] = pd.to_datetime(df[date_col])
                    df = df.sort_values('date_dt')
                    # 日收益率；首日与停牌/缺失日为 NaN，合成时跳过
                    nav = df[val_col].astype(float)
                    daily_ret = nav.pct_change()
                    sub = pd.DataFrame({
                        'date_dt': df['date_dt'].values,
                        f'{code}_ret': daily_ret.values,
                    }).set_index('date_dt')
                    ret_series.append(sub)
            except Exception:
                continue

    if not ret_series:
        return None

    # 外连接对齐，保留 NaN（休市/无行情），不 ffill
    merged = pd.concat(ret_series, axis=1).sort_index()
    if merged.empty:
        return None

    # 等权：只对当日非 NaN 的标的取均值；全日皆 NaN 则组合收益为 NaN
    merged['Portfolio_Daily_Return'] = merged.mean(axis=1, skipna=True)
    # 用有效日复利累积净值（跳过全休市日）
    valid = merged['Portfolio_Daily_Return'].dropna()
    if valid.empty:
        return None
    basket = (1.0 + valid).cumprod()
    out = pd.DataFrame({
        'date_dt': basket.index,
        'Portfolio_Basket': basket.values,
    })
    out['Portfolio_Cumulative_Return(%)'] = (out['Portfolio_Basket'] - 1.0) * 100
    out['Portfolio_Drawdown(%)'] = (
        out['Portfolio_Basket'] / out['Portfolio_Basket'].cummax() - 1.0
    ) * 100
    return out.reset_index(drop=True)


# ===========================================================================
# 视图 1：股票池批量回测大盘 (Batch Screening)
# ===========================================================================

def render_batch_mode():
    st.header("🏆 股票池批量回测大盘与横截面筛选")
    st.caption("多标的并发数据准备 · 时界动态推演 · 严格防偷价 · 组合级绩效排行榜与多市场横向洞察")

    # 1. 侧边栏：股票池管理与批量参数
    with st.sidebar:
        st.subheader("📁 1. 股票池设定")
        pool_mode = st.radio(
            "选择股票池来源",
            options=["🌟 推荐精品预设池", "📄 上传文件 (.csv/.json)", "✏️ 内联自定义录入"],
            index=0,
        )

        selected_pool = []
        if pool_mode.startswith("🌟"):
            preset_name = st.selectbox("选择预设池", list(PRESET_POOLS.keys()), index=0)
            selected_pool = PRESET_POOLS[preset_name]
        elif pool_mode.startswith("📄"):
            uploaded_file = st.file_uploader("上传股票池文件", type=["csv", "txt", "json"])
            if uploaded_file is not None:
                try:
                    # 写入临时文件解析
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
        else:
            inline_text = st.text_area(
                "输入标的（格式: 代码:名称:市场，逗号隔开）",
                value="300418:昆仑万维:CN, 600036:招商银行:CN, 00700:腾讯控股:HK, AAPL:苹果公司:US",
                help="例: 300418:昆仑万维:CN, 00700:腾讯控股:HK, AAPL:苹果公司:US"
            )
            selected_pool = [parse_stock_item(it) for it in inline_text.split(',') if it.strip()]

        st.caption(f"当前池包含 **{len(selected_pool)}** 只标的：")
        pool_summary_items = [f"`{it.get('stock_code')}` {it.get('stock_name')} ({it.get('market')})" for it in selected_pool]
        st.write(", ".join(pool_summary_items[:8]) + ("..." if len(pool_summary_items) > 8 else ""))

        st.divider()

        # 2. 预测时界与回测区间
        st.subheader("🔄 2. 推演与时界参数")
        horizon_choice = st.radio(
            "时界模式 (Forecast Horizon)",
            options=["🤖 Auto 极值自适应 (动态波峰最优持仓)", "⚙️ 固定时界 (Fixed Horizon)"],
            index=0,
            help="【Auto 极值自适应】：模型推演 10 天走势，自动寻找未来收益风险比最佳的波峰拐点。"
        )
        if horizon_choice.startswith("🤖"):
            horizon_val = "auto"
            max_auto_horizon = st.slider(
                "最大推演搜索天数", min_value=5, max_value=20, value=10, step=1,
                help=(
                    "Auto 模式下 Kronos 向前推演的最大天数（推荐 8~12 交易日）。"
                    "超过 15 天后自回归误差快速累积，波峰位置可靠性明显下降。"
                )
            )
            if max_auto_horizon > 12:
                st.warning(
                    f"⚠️ 推演时界 **{max_auto_horizon} 天** 超出建议范围（≤12天），"
                    f"Kronos 波峰定位精度会明显下降。建议调低至 **8~12 天**。",
                    icon="⚠️"
                )
        else:
            horizon_val = st.selectbox("固定时界天数", [1, 3, 5, 10], index=2)
            max_auto_horizon = 10

        test_days = st.slider("回测验证窗口 (交易日)", min_value=20, max_value=250, value=120, step=10,
                              help="历史行情最后 N 个交易日作为验证样本外区间")
        lookback = st.slider("预测输入窗口 (交易日)", min_value=60, max_value=200, value=120, step=10)
        history_days = st.slider("历史数据获取天数 (自然日)", min_value=300, max_value=1200, value=700, step=50)
        samples = st.slider("蒙特卡洛采样数", min_value=1, max_value=40, value=20, step=1)
        force_refresh = st.checkbox("强制刷新网络行情与预测", value=False)

        st.divider()

        # 3. 策略风控与资金模型
        st.subheader("🛡️ 3. 资金与风控机制")
        capital_input = st.number_input("全局统一初始资金 (0=按市场自适应)", min_value=0, value=0, step=50000,
                                        help="设为 0 时：A股默认10万，港股默认30万，美股默认5万")
        final_capital = None if capital_input == 0 else float(capital_input)

        hard_stop = st.slider("基准硬止损线 (%)", 1.0, 10.0, 4.0, 0.5) / 100.0
        trail_stop = st.slider("移动止盈回撤 (%)", 1.0, 8.0, 2.5, 0.5) / 100.0
        max_hold = st.slider("最大持仓交易日", 2, 30, 15, 1)
        min_lot_guarantee = st.checkbox("一手保底机制（防港美股高单价抹零）", value=True)
        intraday_stop = st.checkbox("盘中止损（遵守市场 T+N 合规）", value=True)
        require_skill = st.checkbox("预测技能体检未通过时直接拦截", value=False)

        st.divider()

        # 4. 并发控制
        st.subheader("⚡ 4. 算力与并发")
        max_workers = st.slider("行情数据并发准备线程", 1, 12, 6)
        parallel_stocks = st.slider("模型预测回测并发标的数", 1, 4, 1,
                                    help="注意：深度学习模型预测消耗 GPU/CPU 算力与显存，建议设为 1 串行执行，避免显存溢出。")

        st.divider()

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            run_batch_btn = st.button("🚀 启动批量回测", type="primary", width="stretch")
        with col_b2:
            dry_run_btn = st.button("🔍 预检行情", width="stretch")

    # 2. 状态与进度提示区
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    # 执行预检
    if dry_run_btn:
        with st.spinner("正在并发拉取各标的行情并预检有效性..."):
            engine = BatchBacktester(log_dir=BATCH_OUTPUT_DIR)
            valid = engine.prepare_data_pool(
                stock_list=selected_pool, data_dir=DATA_DIR,
                history_days=history_days, lookback=lookback,
                test_days=test_days, force_refresh=force_refresh,
                max_workers=max_workers,
            )
            st.success(f"✅ 行情预检完成：**{len(valid)}** 只标的数据充足合格，可正常回测！")
            preview_rows = []
            for c, item in valid.items():
                preview_rows.append({
                    '代码': c, '名称': item['stock_name'], '市场': item['market'],
                    '日K线总条数': item['total_bars'], '状态': '合格 🟢'
                })
            st.dataframe(pd.DataFrame(preview_rows), width="stretch")

    # 执行批量回测
    if run_batch_btn:
        engine = BatchBacktester(log_dir=BATCH_OUTPUT_DIR)
        bar = progress_placeholder.progress(0, text="[1/4] 正在拉取历史行情数据...")

        def _data_cb(done, total, code, name, status):
            pct = int((done / max(1, total)) * 30)
            bar.progress(pct, text=f"[1/4] 准备行情 ({done}/{total}): {name}({code}) -> {status}")

        valid = engine.prepare_data_pool(
            stock_list=selected_pool, data_dir=DATA_DIR,
            history_days=history_days, lookback=lookback,
            test_days=test_days, force_refresh=force_refresh,
            max_workers=max_workers, progress_callback=_data_cb,
        )

        if not valid:
            st.error("❌ 无标的通过行情有效性预检，无法启动批量回测。")
        else:
            bar.progress(35, text=f"[2/4] 行情就绪（{len(valid)}只），开始时序推演与回测撮合...")
            status_container = status_placeholder.container(border=True)
            status_container.info(f"🚀 正在对 {len(valid)} 只标的执行批量量化回测流水线...")

            def _bt_cb(done, total, code, name, st_val, dur, res):
                pct = 35 + int((done / max(1, total)) * 60)
                tag = "✅ 成功" if st_val == 'SUCCESS' else "❌ 失败"
                bar.progress(pct, text=f"[2/4] 回测计算中 ({done}/{total}): {name}({code}) {tag} (耗时{dur:.1f}s)")

            results = engine.run_batch_backtest(
                valid_data_pool=valid, data_dir=DATA_DIR,
                pred_dir=PRED_DIR, backtest_batch_dir=BATCH_OUTPUT_DIR,
                parallel_stocks=parallel_stocks,
                auto_predict=True, test_days=test_days,
                lookback=lookback, step_size=1,
                sample_count=samples, forecast_horizon=horizon_val,
                max_auto_horizon=max_auto_horizon,
                force_refresh=force_refresh,
                initial_capital=final_capital,
                hard_stop=hard_stop, trail_stop=trail_stop,
                max_hold=max_hold,
                min_lot_guarantee=min_lot_guarantee,
                intraday_stop=intraday_stop,
                require_skill=require_skill,
                generate_plots=True,
                progress_callback=_bt_cb,
            )

            bar.progress(95, text="[3/4] 正在统计多市场横截面排行榜...")
            ranking_df = engine.generate_ranking_summary(results, BATCH_OUTPUT_DIR)
            engine.export_visualizations_summary(results, BATCH_OUTPUT_DIR)
            bar.progress(100, text="🎉 批量量化回测全部完毕！")
            st.session_state["batch_ranking"] = ranking_df
            st.session_state["batch_results"] = results
            st.success("🎉 批量回测已执行完毕，最新绩效大盘已刷新落盘！")

    # 3. 自动扫描与展示结果（无论刚跑完还是读取已有结果）
    ranking_df = st.session_state.get("batch_ranking")
    if ranking_df is None:
        ranking_df = load_existing_batch_ranking(BATCH_OUTPUT_DIR)
        if ranking_df is not None:
            st.session_state["batch_ranking"] = ranking_df

    if ranking_df is None or ranking_df.empty:
        st.info("💡 尚未检测到批量回测历史记录，请在左侧侧边栏配置股票池后点击「🚀 启动批量回测」。", icon="ℹ️")
        return

    # ---------- 顶层 KPI 组合概览卡片 ----------
    valid_sub = ranking_df[ranking_df['回测状态'] == '成功'] if '回测状态' in ranking_df.columns else ranking_df
    total_count = len(ranking_df)
    success_count = len(valid_sub)

    with st.container(border=True):
        st.subheader("📊 股票池综合表现 KPI 概览")
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

    # TAB 1: 排行榜
    with tab_rank:
        st.markdown("**量化策略全标的绩效横截面排行榜**")
        c_filter1, c_filter2 = st.columns([3, 4])
        with c_filter1:
            market_filter = st.selectbox("市场筛选", ["全部市场", "A股 (CN)", "港股 (HK)", "美股 (US)"], index=0)
        with c_filter2:
            status_filter = st.selectbox("体检建议筛选", ["全部状态", "仅看可实盘 🟢", "仅看需谨慎 🟡"], index=0)

        disp_df = ranking_df.copy()
        if market_filter != "全部市场":
            m_code = {'A股 (CN)': 'CN', '港股 (HK)': 'HK', '美股 (US)': 'US'}.get(market_filter)
            if '市场' in disp_df.columns:
                disp_df = disp_df[disp_df['市场'] == m_code]
        if status_filter == "仅看可实盘 🟢" and '体检建议' in disp_df.columns:
            disp_df = disp_df[disp_df['体检建议'].str.contains('可实盘', na=False)]
        elif status_filter == "仅看需谨慎 🟡" and '体检建议' in disp_df.columns:
            disp_df = disp_df[disp_df['体检建议'].str.contains('需谨慎', na=False)]

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

    # TAB 2: 多市场分盘洞察
    with tab_market:
        st.markdown("**各市场微观结构与回测绩效横向对比**")
        st.caption(
            "A股 (T+1、分板块手数与涨跌停、沪市过户费) vs "
            "港股 (每手股数、双边印花税) vs "
            "美股 (T+0、1股起买、卖方 SEC/TAF)"
        )

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

        with st.expander("💡 市场微观结构对策略表现的影响分析", expanded=True):
            st.markdown("""
- **美股 (US)**：T+0、1 股起买；默认零佣金，卖方另计 SEC/TAF 近似规费；年化按 252 交易日。
- **港股 (HK)**：按标的每手股数申报（如腾讯 100、中国移动 500）；T+0；双边印花税（2023-11-17 起 0.10%）；一手保底防高价股抹零。
- **A股 (CN)**：T+1；主板整手 100 股；创业板最少 100、超出按 1 股；科创板最少 200、超出按 1 股；北交所涨跌停 ±30%；ST ±5%；开盘相对前收判断能否成交；沪市另计过户费。
            """)

    # TAB 3: 投资组合等权合成走势
    with tab_portfolio:
        st.markdown("**股票池等权合成投资组合走势 (Portfolio Synthetic Basket)**")
        st.caption(
            "按各标的日收益率等权平均后复利；某市场休市日不参与当日权重。"
            "这是收益率合成，不是换汇后的资产加总（CNY/HKD/USD 期末资产不可直接相加）。"
        )

        portfolio_curve = calculate_portfolio_equity_curve(BATCH_OUTPUT_DIR)
        if portfolio_curve is not None and not portfolio_curve.empty:
            chart_df = portfolio_curve[['date_dt', 'Portfolio_Cumulative_Return(%)', 'Portfolio_Drawdown(%)']].set_index('date_dt')
            st.line_chart(chart_df['Portfolio_Cumulative_Return(%)'])

            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.subheader("📉 组合最大回撤动态走势")
                st.line_chart(chart_df['Portfolio_Drawdown(%)'])
            with c_p2:
                final_port_ret = portfolio_curve['Portfolio_Cumulative_Return(%)'].iloc[-1]
                max_port_dd = portfolio_curve['Portfolio_Drawdown(%)'].min()
                st.metric("投资组合期末累计收益率", f"{final_port_ret:+.2f}%")
                st.metric("投资组合历史最大回撤", f"{max_port_dd:.2f}%", delta_color="inverse")
                st.caption("注：仅对当日有行情的标的等权；休市不填平为 0 收益。")
        else:
            st.info("暂无足够的逐日净值数据以合成组合曲线。")

    # TAB 4: 🎯 单标的穿透下钻 (Drill-Down Detail)
    with tab_drill:
        st.markdown("**🎯 标的穿透下钻诊断 (Drill-Down)**")
        st.caption("从大盘排行无缝穿透至任意单只股票，查看微观 4 联回测图、成交流水对账表与逐日净值")

        all_batch_stocks = get_existing_batch_stocks(BATCH_OUTPUT_DIR)
        if not all_batch_stocks:
            st.info("暂无可下钻的标的数据")
            return

        target_c = st.selectbox("选择下钻标的", all_batch_stocks, index=0)
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
            "📈 回测全景四联图", "📋 逐笔成交流水", "📑 逐日净值表", "🎯 预测与信号明细", "📄 运行诊断日志"
        ])
        with sub_d1:
            if HAS_PLOTLY and detail.get('daily') is not None:
                d_df = detail['daily'].copy()
                d_col = next((c for c in ['date', 'Date', '日期'] if c in d_df.columns), d_df.columns[0])
                d_df['date_dt'] = pd.to_datetime(d_df[d_col])
                d_df = d_df.set_index('date_dt')

                t_records = []
                if detail.get('trades') is not None and not detail['trades'].empty:
                    t_copy = detail['trades'].copy()
                    t_copy['date'] = pd.to_datetime(t_copy['date'])
                    t_records = t_copy.to_dict('records')

                bt_pro = KronosBacktesterPro(data_dir=DATA_DIR, model_dir=PRED_DIR)
                bt_pro.set_market(target_c, market=prof.market)
                p_fig = bt_pro.plot_backtest_interactive(
                    signal_df=pd.DataFrame(),
                    backtest_df=d_df,
                    metrics=m,
                    trades=t_records,
                    stock_code=target_c,
                    crosshair_mode="x"
                )
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
                st.info("暂无逐日净值数据。")

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
    st.caption("严格防偷价 (t→t+1) · Z-Score自适应波峰信号 · A/港/美全市场微观摩擦 · 多标的横截面筛选与组合配置")

    # 顶层工作模式切换
    mode = st.segmented_control(
        "工作模式切换",
        options=["🏆 股票池批量回测大盘 (Batch Screening)", "🔬 单标的微观深度回测 (Single-Stock Pro)"],
        default="🏆 股票池批量回测大盘 (Batch Screening)",
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
