"""
run_backtest_kronos_pro.py
============================
Kronos 量化回测引擎 —— 工业级重构版 (Pro)

核心修复与特性：
1. [数据层] 强制样本外对齐：仅在 hist_df 与 pred_df 有足够重叠区间（>=10天）时执行
           真实 Backtest；若无重叠则报错退出，要求用户补充含重叠区间的数据。
2. [策略层] Z-Score 自适应信号：expected_return 以当日真实收盘价为基准；
           彻底废弃 ffill() 死锁写法，改为逐日状态机管理仓位。
3. [策略层] 严格 t → t+1 防偷价：raw_signal 在 t 日收盘后产出，
           exec_signal = raw_signal.shift(1)，仅在 t+1 日撮合。
4. [执行层] A 股真实市场摩擦：100 股整手、双边佣金万三（最低5元）、
           卖方印花税千分之五、买卖双向滑点 0.1%、禁止裸做空。
5. [风控层] 三重止损状态机：可配置硬止损（默认 -4%）、移动止盈（高点回撤-2.5%）、
           持仓超时（5交易日）；买入侧动态下探过滤同步受硬止损约束。
6. [新增] 自动获取数据 + Walk-Forward 滚动预测：
           KronosDataFetcher 自动调用 fetch_or_load_stock_data 拉取历史行情，
           再调用 KronosPredictor 在历史区间内逐步滚动生成覆盖回测窗口的预测序列，
           彻底解决「预测数据与历史数据无重叠」的根本问题。

=============================================================================
核心数据流转与示例数据全景 (Data Flow & Sample Data Overview)
=============================================================================
1. 原始历史行情 (hist_df): pd.DataFrame (索引为 DatetimeIndex)
   -------------------------------------------------------------------------
   date (Index)        |  open  |  high  |   low  | close  |    volume  |       amount
   2026-06-23 00:00:00 | 27.200 | 27.850 | 26.900 | 27.500 | 1150000.0  | 31450000.000
   2026-06-24 00:00:00 | 27.600 | 28.300 | 27.400 | 28.100 | 1420000.0  | 39680000.000
   2026-06-25 00:00:00 | 28.200 | 29.000 | 28.000 | 28.800 | 1680000.0  | 48020000.000

2. 模型预测输出 (pred_df): pd.DataFrame (索引为 DatetimeIndex)
   -------------------------------------------------------------------------
   date (Index)        | pred_close | pred_upper(90%) | pred_lower(10%)
   2026-06-24 00:00:00 |     28.450 |          29.100 |          27.800
   2026-06-25 00:00:00 |     28.920 |          29.500 |          28.300
   2026-06-26 00:00:00 |     28.300 |          28.850 |          27.750

3. 信号对齐矩阵 (signal_df): pd.DataFrame (含严格防偷价滞后信号)
   -------------------------------------------------------------------------
   date        | close | close_prev | pred_close | exp_ret | vol_14d | Z-Score | raw_sig | exec_sig
   2026-06-24  | 28.10 |      27.50 |     28.450 | +3.45%  |   2.10% |  +1.643 |       1 |       0 (昨日无)
   2026-06-25  | 28.80 |      28.10 |     28.920 | +2.92%  |   2.15% |  +1.358 |       1 |       1 (今开盘买)
   2026-06-26  | 28.50 |      28.80 |     28.300 | -1.74%  |   2.20% |  -0.791 |       0 |       1 (昨买生效)
   2026-06-27  | 27.90 |      28.50 |     27.600 | -3.16%  |   2.25% |  -1.404 |       0 |       0 (空仓观望)

4. 回测净值曲线 (backtest_df): pd.DataFrame (逐日 Mark-to-Market)
   -------------------------------------------------------------------------
   date        | portfolio_value |       cash | shares | close | exec_sig |  returns
   2026-06-24  |      100,000.00 | 100,000.00 |      0 | 28.10 |        0 |  0.0000%
   2026-06-25  |      100,780.20 |   2,978.40 |  3,400 | 28.80 |        1 | +0.7802%
   2026-06-26  |       99,760.20 |   2,978.40 |  3,400 | 28.50 |        1 | -1.0121%

5. 交易流水记录 (trades): list[dict]
   -------------------------------------------------------------------------
   [
     # 开仓买入示例
     {
       'date': Timestamp('2026-06-25 00:00:00'), 'action': 'BUY',
       'price': 28.228,          # 开盘价 28.20 * (1 + 0.1% 滑点)
       'shares': 3400,           # A股 100 股整手
       'commission': 28.79,      # 万三佣金（>= 5元）
       'pnl_pct': 0.0, 'reason': '信号开仓', 'cash_balance': 2978.40
     },
     # 触发风控平仓卖出示例
     {
       'date': Timestamp('2026-07-02 00:00:00'), 'action': 'SELL',
       'price': 30.170,          # 开盘价 30.20 * (1 - 0.1% 滑点)
       'shares': 3400,
       'commission': 82.02,      # 佣金万三 + 卖方千分之五印花税
       'pnl_pct': +0.0688,       # 单笔收益率 +6.88%
       'reason': '移动止盈(峰值回撤-2.5%)', 'cash_balance': 105478.38
     }
   ]

6. 绩效评价结果 (metrics): dict
   -------------------------------------------------------------------------
   {
     '累计收益率': 0.1582,       # +15.82% 总投资回报
     '年化收益率': 0.7240,       # +72.40% 复合年化折算
     '年化波动率': 0.2135,       # 21.35% 年化收益波动率
     '夏普比率':   3.2700,       # 风险调整后超额回报 (无风险利率2.5%)
     '卡玛比率':  12.4400,       # 年化收益率 / |最大回撤|
     '最大回撤':  -0.0582,       # -5.82% 历史上净值自最高点最大跌幅
     '总交易笔数': 6,            # 完整买卖闭环的交易次数
     '胜率':       0.6667,       # 66.67% (4胜2负)
     '盈亏比':     2.4500,       # 盈利总额 / 亏损总额
     '单笔平均盈亏': 0.0264,     # +2.64% 单笔交易平均收益
     '期末总资产': 115820.50     # 回测结束时账户总净资产（元）
   }
=============================================================================

运行方式:
    # 自动获取数据 + Walk-Forward 预测 + 回测（推荐）
    python backtest/run_backtest_kronos_pro.py 300418 --auto

    # 仅回测（使用现有预测文件，要求有重叠区间）
    python backtest/run_backtest_kronos_pro.py 300418

    # Streamlit 看板
    streamlit run backtest/run_backtest_kronos_pro.py
"""

import os
import sys
import logging
import warnings
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

# 脚本位于仓库根目录 backtest/；产物收敛至 backtest/output/
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
# 后台日志记录系统
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str = None) -> logging.Logger:
    """配置支持控制台与文件落盘的全局诊断 Logger"""
    logger = logging.getLogger("KronosBacktest")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # 控制台 StreamHandler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # 日志写到 backtest/output/，与回测图放在同一目录
        if log_dir is None:
            log_dir = os.path.join(SCRIPT_DIR, "output")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "kronos_backtest.log")
        fh = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

logger = setup_logger()


import re
import argparse

# ---------------------------------------------------------------------------
# 标的与市场规则（实现见 market_rules，此处再导出保持兼容）
# ---------------------------------------------------------------------------
from market_rules import (  # noqa: E402
    SymbolResolver,
    MarketProfile,
    HK_COMMON_LOT_SIZES,
    HK_STAMP_DUTY_CUTOVER,
    compute_open_limit_flags,
)

# ---------------------------------------------------------------------------
# 自动数据获取 + Walk-Forward 预测模块
# ---------------------------------------------------------------------------

class KronosDataFetcher:
    """
    自动行情拉取 + Walk-Forward 滚动预测器

    职责
    ----
    1. 调用项目已有的 fetch_or_load_stock_data / prepare_stock_data
       从 AKShare/本地缓存 获取完整历史日K线。
    2. 将历史数据切分为「训练窗口」与「回测验证窗口」。
    3. 在回测验证窗口内执行 **Walk-Forward 逐步滚动预测**：
       对验证窗口的每一个交易日 t，以 t 之前的 lookback 个历史日
       作为输入，调用 KronosPredictor.predict 预测 t 日的价格；
       由此生成一条与真实历史严格对齐的「样本外预测序列」。
    4. 将预测序列保存为标准 CSV，供 KronosBacktesterPro 直接读取。

    Walk-Forward 时序示意图
    -----------------------
    | <--- lookback ---> | t | t+1 | ... |
    |  历史训练数据输入   | ↑ 预测目标（单步） |
                         |  下一步：滑动一格  |
    | <--- lookback ----> | t+1 | t+2 | ... |

    注意
    ----
    * 单步预测（pred_len=1）精度最高，但速度最慢；
      可通过 step_size 参数控制每次滑动步长，加快速度（默认=1）。
    * 若 step_size > 1，中间日期用线性插值填充，适合快速验证。
    """

    def __init__(self, model_name="NeoQuasar/Kronos-base",
                 tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                 device=None, max_context=512):
        self._predictor = None
        self._model_name = model_name
        self._tokenizer_name = tokenizer_name
        # 在 macOS 上，PyTorch MPS (Metal) 驱动在长序列循环自回归时存在已知系统级竞态崩溃缺陷 (IOGPUMetalCommandBuffer assertion)
        # 而 Apple Silicon 统一内存的 CPU 执行 Kronos 仅需 0.19s/次，因此 macOS 默认选用稳健高速的 cpu
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
            elif sys.platform == "darwin":
                device = "cpu"
            else:
                device = "cpu"
        self._device = device
        self._max_context = max_context

    def _ensure_predictor(self):
        """懒加载：首次调用时才初始化模型，避免仅做数据检查时浪费显存"""
        if self._predictor is not None:
            return
        print("\n🧠 正在加载 Kronos 模型（首次使用，请稍候）...")
        # 动态导入，避免在不使用 --auto 时强制要求模型依赖
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from model import Kronos, KronosTokenizer, KronosPredictor
            tokenizer = KronosTokenizer.from_pretrained(self._tokenizer_name)
            model = Kronos.from_pretrained(self._model_name)
            self._predictor = KronosPredictor(model, tokenizer,
                                               device=self._device,
                                               max_context=self._max_context)
            print(f"✅ 模型加载完成 (device: {self._predictor.device})")
        except ImportError as e:
            raise ImportError(
                f"无法加载 Kronos 模型模块: {e}\n"
                "请确认项目根目录的 model/ 包存在，且已安装所有依赖。"
            )

    def _sample_paths(self, x_df, x_timestamp, y_timestamp, pred_len,
                      n_samples, temperature, top_p):
        """
        采集 n_samples 条**相互独立**的蒙特卡洛预测路径，保留逐路径明细。

        为什么不能直接用 predict(sample_count=N)
        --------------------------------------
        model/kronos.py 的 auto_regressive_inference 在内部就把 N 条路径按
        `np.mean(preds, axis=1)` 平均掉了，调用方只能拿到均值路径。而 P2/P3 需要
        逐路径的极值与分位数，一旦提前平均就永久丢失了尾部信息。

        高效做法
        --------
        predict_batch 传入 N 份相同输入、sample_count=1：批内每一行各自独立做
        multinomial 采样，因此一次批量前向即可得到 N 条独立路径，
        比循环调用 predict N 次快得多。

        返回
        ----
        dict[str, np.ndarray]，每个值形状为 (n_samples, pred_len)；
        采样全部失败时返回 None。
        """
        n_samples = max(1, int(n_samples))

        # 快路径：一次批量前向拿到 N 条独立路径
        if n_samples > 1 and hasattr(self._predictor, 'predict_batch'):
            try:
                pred_dfs = self._predictor.predict_batch(
                    df_list=[x_df] * n_samples,
                    x_timestamp_list=[x_timestamp] * n_samples,
                    y_timestamp_list=[y_timestamp] * n_samples,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=1,
                    verbose=False,
                )
                out = {}
                for col in ('close', 'low', 'high'):
                    if col in pred_dfs[0].columns:
                        out[col] = np.array([d[col].values.astype(float) for d in pred_dfs])
                if 'close' in out:
                    return out
            except Exception as e:
                print(f"   ⚠️ 批量采样失败，回退为逐条采样: {e}")

        # 慢路径回退：逐条调用 predict
        collected = {'close': [], 'low': [], 'high': []}
        for _ in range(n_samples):
            try:
                r = self._predictor.predict(
                    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                    pred_len=pred_len, T=temperature, top_p=top_p,
                    sample_count=1, verbose=False,
                )
                for col in ('close', 'low', 'high'):
                    if col in r.columns:
                        collected[col].append(r[col].values.astype(float))
            except Exception as e:
                print(f"   ⚠️ 单条采样异常: {e}")
                break

        if not collected['close']:
            return None
        return {k: np.array(v) for k, v in collected.items() if v}

    @staticmethod
    def _summarize_paths(paths, min_down_quantile, prev_real_close,
                         is_auto_horizon, eval_horizon):
        """
        把逐路径采样结果汇总为单日预测指标（P1/P2/P3 修正的核心所在）。

        参数
        ----
        paths : dict[str, np.ndarray]
            _sample_paths 的输出，每项形状 (n_samples, pred_len)。
            示例（n_samples=3, pred_len=5，即3条路径、每条预测5个交易日）：
            {
                'close': np.array([
                    [28.10, 28.50, 29.20, 28.80, 28.60],  # 路径1
                    [28.20, 28.90, 29.50, 29.10, 28.40],  # 路径2
                    [28.30, 28.60, 28.70, 29.00, 29.30],  # 路径3
                ]),  # shape: (3, 5)
                'low': np.array([
                    [27.80, 28.20, 28.90, 28.50, 28.30],
                    [28.00, 28.60, 29.10, 28.80, 28.10],
                    [28.10, 28.30, 28.40, 28.70, 29.00],
                ]),  # shape: (3, 5)
                'high': np.array([
                    [28.40, 28.80, 29.50, 29.10, 28.90],
                    [28.50, 29.10, 29.80, 29.40, 28.70],
                    [28.60, 28.90, 29.00, 29.30, 29.60],
                ]),  # shape: (3, 5)
            }
            # 注：实际运行时 n_samples 默认为 20，pred_len = eval_horizon（如 10）。
            # n_samples < 20 时 pred_upper/pred_lower 退化为极大/极小值而非分位数（P3 修正）。
        min_down_quantile : float
            最大下探的保守分位，例如 0.10
        prev_real_close : float
            决策时点已知的最后一个真实收盘价（预测基准）
        is_auto_horizon : bool
            是否为极值自适应模式
        eval_horizon : int
            向前推演的最大步长

        返回
        ----
        dict，含 pred_expected / pred_path_max / optimal_horizon / pred_min_down 等
        """
        close_arr = paths['close']                       # (n_samples, pred_len)
        n_samples = close_arr.shape[0]
        mean_path = np.mean(close_arr, axis=0)           # 各时点的期望价格

        # ---- 止盈目标：逐路径计算极值（彻底解决跨路径平均抹平尖峰的问题） ----
        per_path_peaks = np.max(close_arr, axis=1)       # (n_samples,) 逐路径最高价
        per_path_peak_idx = np.argmax(close_arr, axis=1) # (n_samples,) 逐路径最高点发生时间

        # 稳健波峰目标：取中位数（消除时间异步导致的平滑效应，还原真实波峰高度）
        pred_path_max = float(np.median(per_path_peaks))

        # 最佳时界：取各路径波峰出现时间的中位数 + 1
        pred_len = close_arr.shape[1]
        if is_auto_horizon:
            peak_idx = int(round(float(np.median(per_path_peak_idx))))
            opt_horizon = peak_idx + 1
        else:
            peak_idx = int(np.argmax(mean_path))
            opt_horizon = eval_horizon

        # ---- 入场判定：无偏期望值（P1 修正） ----
        # 关键：该估计量必须与 argmax **完全解耦**。
        # 若用 mean_path[:opt_horizon] 取均值，窗口右端由 argmax 选出，
        # 选择偏差仍会残留（合成检验：3 采样下仍有 +1.77% 虚高）。
        # 改用与 argmax 无关的全时界均值后，同口径检验偏差降至 +0.04%。
        if is_auto_horizon:
            pred_expected = float(np.mean(mean_path))
        else:
            pred_expected = float(mean_path[-1])
        pred_terminal = float(mean_path[-1])

        # ---- 分位数：样本量不足时不谎称置信区间（P3 修正） ----
        safe_peak_idx = min(max(0, peak_idx), pred_len - 1)
        enough_for_quantile = n_samples >= 20
        if enough_for_quantile:
            upper_val = float(np.quantile(close_arr[:, safe_peak_idx], 0.90))
            lower_val = float(np.quantile(close_arr[:, safe_peak_idx], 0.10))
        else:
            upper_val = float(np.max(close_arr[:, safe_peak_idx]))
            lower_val = float(np.min(close_arr[:, safe_peak_idx]))

        # ---- 最大下探：每条路径先各自取最小，再取保守分位（P2 修正） ----
        # 修正前是 np.mean(low_arr, axis=0) 先跨路径平均再取极小，
        # 平均会把极端路径抹平，导致下探被系统性低估。
        low_arr = paths.get('low', close_arr)
        seg = low_arr[:, :opt_horizon] if opt_horizon >= 1 else low_arr
        per_path_min = np.min(seg, axis=1)               # (n_samples,) 逐路径最低价
        if n_samples >= 5:
            min_low_val = float(np.quantile(per_path_min, min_down_quantile))
        else:
            min_low_val = float(np.min(per_path_min))

        min_down_pct = ((min_low_val - prev_real_close) / prev_real_close
                        if prev_real_close > 0 else 0.0)

        # ---- 斜率：基于全推演时界拟合客观趋势（修复事后选峰导致的同义反复/正偏） ----
        full_len = len(mean_path)
        if full_len > 1:
            slope_val = float(np.polyfit(np.arange(full_len), mean_path, 1)[0])
            slope_pct = slope_val / prev_real_close if prev_real_close > 0 else 0.0
        else:
            slope_pct = ((pred_expected - prev_real_close) / prev_real_close
                         if prev_real_close > 0 else 0.0)

        # 波峰溢价：量化 argmax 正偏的大小，供诊断与体检使用
        peak_premium = (pred_path_max / pred_expected - 1.0) if pred_expected > 0 else 0.0

        return {
            'pred_expected': pred_expected,
            'pred_path_max': pred_path_max,
            'pred_terminal': pred_terminal,
            'pred_close_1d': float(mean_path[0]),
            'pred_upper': upper_val,
            'pred_lower': lower_val,
            'pred_slope': slope_pct,
            'pred_min_down': min_down_pct,
            'optimal_horizon': float(opt_horizon),
            'peak_premium': float(peak_premium),
            'n_samples': int(n_samples),
        }

    # ------------------------------------------------------------------
    # 预测列名标准化 与 缓存参数指纹（P1 语义迁移 / P4 复现性）
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        把预测 CSV 的中文列名标准化为英文内部名，并兼容旧版文件。

        语义迁移要点（P1）
        ------------------
        * pred_close —— 下游入场判定统一使用的价格，现绑定为**无偏期望值**。
        * pred_path_max —— 路径最大值，仅供止盈目标使用。

        旧版文件只有 `预测收盘价_均值` 一列，其含义随模式而变：
          - 固定时界模式：该值即时界终值，等价于期望值，可安全使用；
          - auto 模式：该值是 argmax 取出的路径最大值，带结构性正偏，
            此时 pred_expected 置为 NaN 并打上 legacy_biased 标记，
            由信号层决定是拒绝还是降级处理。
        """
        out = df.copy()
        rename = {}
        # 多个源列可能映射到同一目标名（例如文件里同时有「预测收盘价_均值」和
        # 「预测收盘价」）。只认第一个，否则会产生重名列，后续赋值会抛
        # "Cannot set a DataFrame with multiple columns"。
        taken = set()

        def _claim(col, target):
            if target in taken or target in out.columns:
                return
            rename[col] = target
            taken.add(target)

        for c in out.columns:
            cs = str(c)
            if cs == '预测期望收盘价':
                _claim(c, 'pred_expected')
            elif cs == '预测波峰收盘价':
                _claim(c, 'pred_path_max')
            elif cs == '预测时界终值':
                _claim(c, 'pred_terminal')
            elif cs == '预测收盘价_1日':
                _claim(c, 'pred_close_1d')
            elif cs.startswith('预测上界') or cs.startswith('预测收盘价_上限'):
                _claim(c, 'pred_upper')
            elif cs.startswith('预测下界') or cs.startswith('预测收盘价_下限'):
                _claim(c, 'pred_lower')
            elif cs == '预测斜率':
                _claim(c, 'pred_slope')
            elif cs == '预测最大下探':
                _claim(c, 'pred_min_down')
            elif cs == '波峰溢价':
                _claim(c, 'peak_premium')
            elif cs in ('动态时界', 'optimal_horizon'):
                _claim(c, 'optimal_horizon')
            elif cs == '采样路径数':
                _claim(c, 'n_samples')
            elif cs == '是否插值':
                _claim(c, 'is_interpolated')
            elif cs == '时界':
                _claim(c, 'horizon_mode')
            elif cs in ('预测收盘价_均值', '预测收盘价', '增强预测收盘价',
                        '基础预测收盘价', '收盘价'):
                _claim(c, '_legacy_close')
        out = out.rename(columns=rename)

        # 旧版文件兼容处理
        if '_legacy_close' in out.columns:
            legacy_auto = False
            if 'horizon_mode' in out.columns:
                legacy_auto = str(out['horizon_mode'].iloc[0]).lower() == 'auto'
            elif 'optimal_horizon' in out.columns:
                legacy_auto = True

            if 'pred_path_max' not in out.columns:
                out['pred_path_max'] = out['_legacy_close']
            if 'pred_expected' not in out.columns:
                # auto 旧文件的均值列其实是路径最大值，不能当期望用
                out['pred_expected'] = np.nan if legacy_auto else out['_legacy_close']
            out['legacy_biased'] = bool(legacy_auto)
            out = out.drop(columns=['_legacy_close'])
        else:
            out['legacy_biased'] = False

        # 下游统一入口：pred_close 绑定无偏期望值
        if 'pred_expected' in out.columns:
            out['pred_close'] = out['pred_expected']
        if 'pred_path_max' not in out.columns and 'pred_close' in out.columns:
            out['pred_path_max'] = out['pred_close']

        for col in ('pred_close', 'pred_expected', 'pred_path_max', 'pred_terminal',
                    'pred_close_1d', 'pred_upper', 'pred_lower', 'pred_slope',
                    'pred_min_down', 'peak_premium', 'optimal_horizon',
                    'n_samples', 'is_interpolated'):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors='coerce')
        return out

    @staticmethod
    def _build_fingerprint(**kwargs) -> dict:
        """生成预测缓存的参数指纹；任一生成参数变化都会导致指纹不匹配。"""
        fp = {k: (str(v) if not isinstance(v, (int, float, bool)) else v)
              for k, v in kwargs.items()}
        fp['schema_version'] = 2      # v2 = P1/P2/P3 修正后的列结构
        return fp

    @staticmethod
    def _fingerprint_path(csv_path: str) -> str:
        return csv_path + '.meta.json'

    @classmethod
    def _write_cache_fingerprint(cls, csv_path: str, fingerprint: dict):
        import json
        try:
            with open(cls._fingerprint_path(csv_path), 'w', encoding='utf-8') as f:
                json.dump(fingerprint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"   ⚠️ 写入缓存指纹失败（不影响预测结果）: {e}")

    @classmethod
    def _check_cache_fingerprint(cls, csv_path: str, expected: dict):
        """
        校验缓存指纹。

        返回 (是否匹配, 不匹配原因描述)。缺失指纹文件视为不匹配，
        因为旧缓存的生成参数不可考，继续复用会造成无法归因的实验污染。
        """
        import json
        fp_path = cls._fingerprint_path(csv_path)
        if not os.path.exists(fp_path):
            return False, "缺少参数指纹文件（旧版缓存，生成参数不可考）"
        try:
            with open(fp_path, 'r', encoding='utf-8') as f:
                actual = json.load(f)
        except Exception as e:
            return False, f"指纹文件读取失败: {e}"

        diffs = []
        for k, v in expected.items():
            if k not in actual:
                diffs.append(f"{k}: 缺失 -> {v}")
            elif str(actual[k]) != str(v):
                diffs.append(f"{k}: {actual[k]} -> {v}")
        if diffs:
            return False, "参数已变更 (" + ", ".join(diffs) + ")"
        return True, ""

    @staticmethod
    def _import_data_utils():
        """动态导入数据工具函数（来自 prediction_akshare_2024_2025_copy）"""
        # 预测工具仍在 examples/，与本脚本同进程复用
        if EXAMPLES_DIR not in sys.path:
            sys.path.insert(0, EXAMPLES_DIR)
        try:
            from prediction_akshare_2024_2025_copy import (
                fetch_or_load_stock_data,
                prepare_stock_data,
                generate_future_dates_with_holidays,
                resolve_date_range,
                ensure_output_directory,
                detect_market_type,
                validate_stock_code,
            )
        except ImportError:
            from examples.prediction_akshare_2024_2025_copy import (
                fetch_or_load_stock_data,
                prepare_stock_data,
                generate_future_dates_with_holidays,
                resolve_date_range,
                ensure_output_directory,
                detect_market_type,
                validate_stock_code,
            )
        return (fetch_or_load_stock_data, prepare_stock_data,
                generate_future_dates_with_holidays, detect_market_type,
                validate_stock_code)

    def fetch_history(
        self,
        stock_code: str,
        data_dir: str,
        history_days: int = 400,
        market: str = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        拉取历史日K线数据并标准化为 KronosBacktesterPro 所需格式。

        参数示例 (Parameters)
        --------------------
        stock_code : str
            股票代码，示例: '300418' (A股代码) 或 'AAPL' (美股代码)
        data_dir : str
            数据存储目录路径，示例: './examples/data'
        history_days : int, default=400
            拉取回溯自然天数 (转换为交易日通常约为 history_days * 5/7)，示例: 400
        market : str, optional
            市场类型，'sz' (深市) / 'sh' (沪市) / 'us' (美股)，None 则自动推断
        force_refresh : bool, default=False
            是否强制绕过本地缓存重新调用 API 下载行情

        返回值数据结构与示例 (Returns: pd.DataFrame)
        ------------------------------------------
        索引: pd.DatetimeIndex (名称为 'date', 单调递增)
        数据列:
            - open   (float64): 当日开盘价 (元)
            - high   (float64): 当日最高价 (元)
            - low    (float64): 当日最低价 (元)
            - close  (float64): 当日收盘价 (元)
            - volume (float64): 成交量 (股 / 手)
            - amount (float64): 成交金额 (元)

        DataFrame 结构示例:
        >>> print(df.head(3))
                                  open    high     low   close       volume          amount
        date                                                                               
        2025-01-02 00:00:00  25.1000  26.300  24.800  25.900  1,250,000.0   32,150,000.00
        2025-01-03 00:00:00  25.8000  26.500  25.500  26.200  1,400,000.0   36,400,000.00
        2025-01-06 00:00:00  26.3000  27.100  26.000  26.850  1,820,000.0   48,550,000.00
        """
        clean_code, det_market, prefix = SymbolResolver.resolve(stock_code, market=market)

        (fetch_or_load_stock_data, prepare_stock_data,
         _, _, validate_stock_code) = self._import_data_utils()

        # 校验与规范化代码
        is_valid, err_msg, norm_code, _ = validate_stock_code(
            clean_code, det_market
        )
        normalized_code = norm_code if is_valid else clean_code

        # 统一走 fetch_or_load_stock_data：内含缓存新鲜度判定，禁止模拟 K 线进入回测
        market_label = {'CN': 'A股', 'HK': '港股', 'US': '美股'}.get(det_market, det_market)
        print(f"\n📡 获取 [{market_label}] {normalized_code} 历史行情（{history_days} 天，"
              f"{'强制刷新' if force_refresh else '时效缓存'}）...")
        raw_df, _ = fetch_or_load_stock_data(
            stock_code=normalized_code,
            data_dir=data_dir,
            history_days=history_days,
            force_refresh=force_refresh,
            market=det_market,
            allow_fallback=False,
        )
        if raw_df is None or len(raw_df) == 0:
            raise ValueError(
                f"未能获取 {normalized_code} 的真实行情数据，请检查网络或股票代码。"
                f"（回测已禁用模拟兜底数据）"
            )

        df = prepare_stock_data(raw_df, normalized_code)

        # 转换为回测引擎标准格式（DatetimeIndex + 标准列名）
        df = df.rename(columns={'timestamps': 'date'})
        df = df.set_index('date').sort_index()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'amount' not in df.columns and 'volume' in df.columns:
            df['amount'] = df['close'] * df['volume']

        print(f"✅ 历史行情: {df.index.min().date()} → {df.index.max().date()} ({len(df)} 条)")
        return df

    def walk_forward_predict(
        self,
        hist_df: pd.DataFrame,
        stock_code: str,
        output_dir: str,
        test_days: int = 120,
        lookback: int = 120,
        step_size: int = 2,
        sample_count: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
        forecast_horizon: Union[int, str] = 'auto',
        max_auto_horizon: int = 10,
        force_refresh: bool = False,
        min_down_quantile: float = 0.10,
    ) -> pd.DataFrame:
        """
        Walk-Forward 逐步滚动预测，生成覆盖回测窗口的样本外多时界/自适应极值预测序列。

        参数示例 (Parameters)
        --------------------
        hist_df : pd.DataFrame
            完整历史行情（索引为 DatetimeIndex，含 open/high/low/close/volume/amount 列）
        stock_code : str
            股票代码，示例: '300418'
        output_dir : str
            预测 CSV 输出目录，示例: './examples/yuce'
        test_days : int, default=120
            回测验证窗口长度（最后 N 个交易日，默认半年 120 交易日）, 示例: 120
        lookback : int, default=120
            每次预测使用的历史窗口长度（交易日数）, 示例: 120
        step_size : int, default=2
            滑动步长（1=最精确逐日预测，2=加速跳步预测+线性插值推荐）, 示例: 2
        sample_count : int, default=20
            蒙特卡洛采样路径数。需要 >= 20 才能支撑分位数估计（P3 修正前默认为 3，
            用 3 条样本计算 90/10 分位不具统计意义）, 示例: 20
        temperature : float, default=1.0
            生成温度系数，示例: 1.0
        top_p : float, default=0.9
            核采样阈值，示例: 0.9
        forecast_horizon : Union[int, str], default='auto'
            预测时界跨度：'auto' (极值自适应，推荐)，或具体天数 (1=单日超短，3=短线波段，5=标准波段)
        max_auto_horizon : int, default=10
            Auto 模式下向前推演搜索波峰的最大步长，默认 10 交易日
        force_refresh : bool, default=False
            是否强制重新生成预测，默认 False（若已有对应时界预测文件则直接秒级复用）
        min_down_quantile : float, default=0.10
            最大下探的保守分位（P2 修正）。对「每条路径各自的最低价」取该分位，
            0.10 表示取 10% 最差情形。绝不可先跨路径平均再取极小值。

        【P1/P2/P3 修正说明】
        ---------------------
        本方法产出两个语义不同的价格预测，下游不得混用：

        * pred_expected —— 路径期望值（无偏）。用于**入场判定**。
        * pred_path_max —— 路径最大值（最大值统计量，结构性正偏）。仅用于**止盈目标**。

        修正前 auto 模式把 argmax 得到的路径最大值直接当作期望收益去和波动率比，
        对 300418 造成相对 1 日预测价均值 +6.66% 的系统性虚高。
        """
        is_auto_horizon = (
            forecast_horizon == 'auto'
            or (isinstance(forecast_horizon, str) and forecast_horizon.lower() == 'auto')
        )
        if is_auto_horizon:
            eval_horizon = int(max_auto_horizon)
            cache_suffix = "_auto"
        else:
            eval_horizon = int(forecast_horizon)
            cache_suffix = f"_h{eval_horizon}" if eval_horizon > 1 else ""

        clean_code, _, prefix = SymbolResolver.resolve(stock_code)

        # 优先检查本地是否已有对应时界的 Walk-Forward 预测缓存
        out_candidates = [
            os.path.join(output_dir, f"{clean_code}_wf_predictions{cache_suffix}.csv"),
            os.path.join(output_dir, f"{prefix}_wf_predictions{cache_suffix}.csv"),
            os.path.join(output_dir, f"{clean_code}_wf_predictions.csv"),
            os.path.join(output_dir, f"{prefix}_wf_predictions.csv"),
        ]
        target_path = next((p for p in out_candidates if os.path.exists(p)), out_candidates[0])
        out_path = os.path.join(output_dir, f"{clean_code}_wf_predictions{cache_suffix}.csv")

        # P4：缓存命中必须同时满足「行数足够」与「参数指纹一致」。
        # 修正前只检查行数，导致用户在侧边栏改了 lookback/step_size/samples 之后
        # 仍会静默复用参数完全不同的旧缓存，实验结果无法归因。
        expected_fp = self._build_fingerprint(
            stock_code=clean_code, test_days=test_days, lookback=lookback,
            step_size=step_size, sample_count=sample_count,
            forecast_horizon=forecast_horizon, max_auto_horizon=max_auto_horizon,
            temperature=temperature, top_p=top_p, min_down_quantile=min_down_quantile,
            hist_last_date=str(hist_df.index.max().date()), n_hist=len(hist_df),
        )

        if not force_refresh and os.path.exists(target_path):
            fp_ok, fp_reason = self._check_cache_fingerprint(target_path, expected_fp)
            if not fp_ok:
                print(f"\n🔄 已有预测缓存与当前参数不一致，将重新生成: {target_path}")
                print(f"   原因: {fp_reason}")
            else:
                try:
                    cached_pred = pd.read_csv(target_path, encoding='utf-8-sig')
                    date_col = '日期' if '日期' in cached_pred.columns else cached_pred.columns[0]
                    if len(cached_pred) >= min(test_days, 10):
                        print(f"\n💾 命中已有 Walk-Forward 预测数据: {target_path}")
                        print(f"   共 {len(cached_pred)} 条预测记录，参数指纹一致，直接复用。")
                        print(f"   💡 如需重新生成预测，请勾选【强制刷新】或传入 --force-refresh。")
                        result_df = cached_pred.set_index(date_col)
                        result_df.index = pd.to_datetime(result_df.index)
                        result_df = self._normalize_pred_columns(result_df)
                        if 'pred_close' in result_df.columns:
                            return result_df
                    else:
                        print(f"\n🔄 缓存记录数 {len(cached_pred)} 不足 {min(test_days, 10)}，将重新生成")
                except Exception as e:
                    print(f"⚠️ 读取已有预测文件失败，将重新生成: {e}")

        (_, _, generate_future_dates_with_holidays, *_) = self._import_data_utils()

        if len(hist_df) < lookback + test_days:
            raise ValueError(
                f"历史数据不足：需要至少 {lookback + test_days} 个交易日，"
                f"实际只有 {len(hist_df)} 个。请增大 history_days 或减小 test_days/lookback。"
            )

        self._ensure_predictor()

        # 切分训练区间和回测区间
        train_df  = hist_df.iloc[:-test_days]
        test_df   = hist_df.iloc[-test_days:]
        test_dates = test_df.index.tolist()

        horizon_label = f"Auto (最大探测={eval_horizon}天)" if is_auto_horizon else f"固定={eval_horizon}天"
        print(f"\n📈 Walk-Forward 预测配置 (时界模式: {horizon_label}):")
        print(f"   训练区间: {train_df.index.min().date()} → {train_df.index.max().date()} ({len(train_df)} 日)")
        print(f"   回测区间: {test_df.index.min().date()} → {test_df.index.max().date()} ({len(test_df)} 日)")
        print(f"   lookback={lookback}, step_size={step_size}, sample_count={sample_count}")

        # 准备 OHLCV 数据列（Kronos 标准输入）
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        available_cols = [c for c in feature_cols if c in hist_df.columns]

        pred_records = {}  # date -> dict of metrics

        # 确定需要实际推理的"锚点"日期
        anchor_indices = list(range(0, len(test_dates), step_size))
        if anchor_indices[-1] != len(test_dates) - 1:
            anchor_indices.append(len(test_dates) - 1)

        import time
        t0 = time.time()

        for seq_idx, anchor_idx in enumerate(anchor_indices):
            target_date = test_dates[anchor_idx]

            # 输入窗口：target_date 之前的 lookback 个交易日
            all_up_to_target = hist_df[hist_df.index < target_date]
            if len(all_up_to_target) < lookback:
                print(f"   ⚠️ {target_date.date()} 前历史不足 {lookback} 日，跳过")
                continue

            window_df = all_up_to_target.iloc[-lookback:]
            x_df = window_df[available_cols].copy()
            for col in available_cols:
                x_df[col] = pd.to_numeric(x_df[col], errors='coerce').fillna(0.0)

            x_timestamp = pd.Series(window_df.index)

            # 确定从 target_date 起未来 eval_horizon 个交易日的序列
            if anchor_idx + eval_horizon <= len(test_dates):
                target_future_dates = test_dates[anchor_idx : anchor_idx + eval_horizon]
            else:
                avail_dates = test_dates[anchor_idx:]
                needed = eval_horizon - len(avail_dates)
                extra_dates = generate_future_dates_with_holidays(avail_dates[-1], needed + 5)[:needed]
                target_future_dates = avail_dates + [pd.Timestamp(d) for d in extra_dates]

            y_timestamp = pd.Series(target_future_dates)

            # 蒙特卡洛多路径采样（保留逐路径明细，供 P2/P3 计算极值与分位）
            paths = self._sample_paths(
                x_df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=eval_horizon,
                n_samples=sample_count,
                temperature=temperature,
                top_p=top_p,
            )
            if paths is None:
                print(f"   ⚠️ 预测失败，跳过 [{target_date.date()}]")
                continue

            prev_real_close = float(all_up_to_target['close'].iloc[-1])
            rec = self._summarize_paths(
                paths=paths,
                min_down_quantile=min_down_quantile,
                prev_real_close=prev_real_close,
                is_auto_horizon=is_auto_horizon,
                eval_horizon=eval_horizon,
            )
            pred_records[target_date] = rec

            elapsed = time.time() - t0
            opt_horizon = int(rec['optimal_horizon'])
            opt_str = f" [动态波峰 H*={opt_horizon}日]" if is_auto_horizon else ""
            print(f"   [{seq_idx+1}/{len(anchor_indices)}] {target_date.date()} "
                  f"期望={rec['pred_expected']:.3f} 波峰={rec['pred_path_max']:.3f}"
                  f"(溢价{rec['peak_premium']*100:+.1f}%){opt_str} "
                  f"(斜率={rec['pred_slope']*100:+.2f}%) "
                  f"真实={hist_df.loc[target_date, 'close']:.3f} "
                  f"(累计耗时 {elapsed:.1f}s)")

            # 显存/内存释放保护
            import gc
            gc.collect()

        # 标记哪些日期是真实推理出来的锚点，哪些是插值填充的
        anchor_date_set = set(pred_records.keys())

        # step_size > 1 时：用线性插值填充锚点之间的日期
        # 注意：插值序列会被人为平滑，从而虚高 IC，因此带插值的数据禁止进入评估路径。
        if step_size > 1 and len(pred_records) >= 2:
            anchor_dates = sorted(pred_records.keys())
            anchor_nums = [test_dates.index(d) for d in anchor_dates]
            metric_keys = ['pred_expected', 'pred_path_max', 'pred_terminal', 'pred_close_1d',
                           'pred_upper', 'pred_lower', 'pred_slope', 'pred_min_down',
                           'optimal_horizon', 'peak_premium', 'n_samples']
            anchor_vals = {k: [pred_records[d][k] for d in anchor_dates] for k in metric_keys}

            for i, td in enumerate(test_dates):
                if td not in pred_records:
                    pred_records[td] = {
                        k: float(np.interp(i, anchor_nums, anchor_vals[k]))
                        for k in metric_keys
                    }

        # 构建输出 DataFrame
        sorted_dates = sorted(pred_records.keys())
        quantile_label = (f"样本{sample_count}条_90分位" if sample_count >= 20
                          else f"样本{sample_count}条_极大值")
        lower_label = (f"样本{sample_count}条_10分位" if sample_count >= 20
                       else f"样本{sample_count}条_极小值")
        pred_df = pd.DataFrame({
            '日期':             [d.strftime('%Y-%m-%d') for d in sorted_dates],
            # 入场判定用（无偏期望值）
            '预测期望收盘价':   [pred_records[d]['pred_expected'] for d in sorted_dates],
            # 止盈目标用（路径最大值，结构性正偏，禁止用于入场）
            '预测波峰收盘价':   [pred_records[d]['pred_path_max'] for d in sorted_dates],
            '预测时界终值':     [pred_records[d]['pred_terminal'] for d in sorted_dates],
            '预测收盘价_1日':   [pred_records[d]['pred_close_1d'] for d in sorted_dates],
            f'预测上界({quantile_label})': [pred_records[d]['pred_upper'] for d in sorted_dates],
            f'预测下界({lower_label})':   [pred_records[d]['pred_lower'] for d in sorted_dates],
            '预测斜率':         [pred_records[d]['pred_slope'] for d in sorted_dates],
            '预测最大下探':     [pred_records[d]['pred_min_down'] for d in sorted_dates],
            '波峰溢价':         [pred_records[d]['peak_premium'] for d in sorted_dates],
            '动态时界':         [int(round(pred_records[d]['optimal_horizon'])) for d in sorted_dates],
            '时界':             ['auto' if is_auto_horizon else eval_horizon] * len(sorted_dates),
            '采样路径数':       [int(round(pred_records[d]['n_samples'])) for d in sorted_dates],
            # 该行是真实推理还是插值产物：评估路径据此拒绝插值数据
            '是否插值':         [0 if d in anchor_date_set else 1 for d in sorted_dates],
        })

        # 保存 CSV + 参数指纹（P4：防止参数变更后静默复用旧缓存）
        os.makedirs(output_dir, exist_ok=True)
        pred_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        self._write_cache_fingerprint(out_path, self._build_fingerprint(
            stock_code=stock_code, test_days=test_days, lookback=lookback,
            step_size=step_size, sample_count=sample_count,
            forecast_horizon=forecast_horizon, max_auto_horizon=max_auto_horizon,
            temperature=temperature, top_p=top_p, min_down_quantile=min_down_quantile,
            hist_last_date=str(hist_df.index.max().date()), n_hist=len(hist_df),
        ))

        interp_cnt = sum(1 for d in sorted_dates if d not in anchor_date_set)
        print(f"\n✅ Walk-Forward 预测完成，结果已保存: {out_path}")
        print(f"   覆盖 {len(sorted_dates)} 个交易日 "
              f"({sorted_dates[0].date()} → {sorted_dates[-1].date()})")
        print(f"   真实推理 {len(sorted_dates) - interp_cnt} 条，插值填充 {interp_cnt} 条，"
              f"每条 {sample_count} 采样路径")
        if interp_cnt > 0:
            print(f"   ⚠️ 含插值数据，禁止用于技能评估与正式验收（插值会人为平滑序列、虚高 IC）")

        # 返回标准格式
        result_df = pred_df.set_index('日期')
        result_df.index = pd.to_datetime(result_df.index)
        return self._normalize_pred_columns(result_df)

    # 兼容别名
    generate_walk_forward_predictions = walk_forward_predict

    def run_auto_pipeline(
        self,
        stock_code: str,
        data_dir: str,
        pred_output_dir: str,
        history_days: int = 700,
        test_days: int = 120,
        lookback: int = 120,
        step_size: int = 1,
        sample_count: int = 20,
        forecast_horizon: Union[int, str] = 'auto',
        max_auto_horizon: int = 10,
        market: str = None,
        force_refresh: bool = False,
        min_down_quantile: float = 0.10,
    ) -> tuple:
        """
        一键自动管线：拉取行情 → Walk-Forward 预测 → 返回 (hist_df, pred_df)

        返回的两个 DataFrame 直接满足 KronosBacktesterPro.align_and_generate_signals 的输入要求。
        """
        clean_code, det_market, _ = SymbolResolver.resolve(stock_code, market=market)
        market_label = {'CN': 'A股', 'HK': '港股', 'US': '美股'}.get(det_market, det_market)
        print(f"\n{'='*60}")
        print(f"🚀 启动 Kronos 自动管线: [{market_label}] {clean_code}")
        print(f"{'='*60}")

        # 1. 获取历史数据
        hist_df = self.fetch_history(
            stock_code=clean_code,
            data_dir=data_dir,
            history_days=history_days,
            market=det_market,
            force_refresh=force_refresh,
        )

        # 2. Walk-Forward 预测
        pred_df = self.walk_forward_predict(
            hist_df=hist_df,
            stock_code=clean_code,
            output_dir=pred_output_dir,
            test_days=test_days,
            lookback=lookback,
            step_size=step_size,
            sample_count=sample_count,
            forecast_horizon=forecast_horizon,
            max_auto_horizon=max_auto_horizon,
            force_refresh=force_refresh,
            min_down_quantile=min_down_quantile,
        )

        return hist_df, pred_df


# ---------------------------------------------------------------------------
# 核心回测引擎
# ---------------------------------------------------------------------------

class KronosBacktesterPro:
    """
    Kronos 量化回测引擎 Pro 版

    现代自适应量化架构：
    - 多市场微观画像 (A股/港股/美股)
    - 样本外时序防偷价 (t-1收盘决策 -> t日开盘撮合)
    - 滚动分位数自校准信号强度
    - 动态 ATR 止损 + 预测波峰主动止盈
    - 一手保底机制 (Min-Lot Guarantee)
    """

    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        initial_capital: float = None,
        commission_rate: float = None,
        min_commission: float = None,
        stamp_duty_rate: float = None,
        slippage_rate: float = None,
        market_profile: 'MarketProfile' = None,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir

        # 市场规则画像：默认 A 股，可由 set_market 依股票代码自动切换
        self.profile = market_profile or MarketProfile.for_market('CN')
        self.initial_capital = initial_capital if initial_capital is not None else self.profile.default_capital
        # 滑点：未显式指定时跟随市场默认（A 0.10% / HK 0.15% / US 0.05%）
        self._slippage_override = slippage_rate is not None
        self.slippage_rate = (
            slippage_rate if self._slippage_override else self.profile.default_slippage
        )

        # 显式传入的费率覆盖 profile 默认值
        self._override = {}
        if commission_rate is not None:
            self.profile.commission_rate = commission_rate
            self._override['commission_rate'] = commission_rate
        if min_commission is not None:
            self.profile.min_commission = min_commission
            self._override['min_commission'] = min_commission
        if stamp_duty_rate is not None:
            self.profile.stamp_duty_sell = stamp_duty_rate
            self._override['stamp_duty_sell'] = stamp_duty_rate

    def set_market(self, stock_code: str, market: str = None, stock_name: str = None):
        """依股票代码/名称切换市场规则，并保留用户显式覆盖的费率项"""
        self.profile = MarketProfile.for_stock(
            stock_code, market=market, stock_name=stock_name
        )
        for k, v in self._override.items():
            setattr(self.profile, k, v)
        # 未手动指定滑点时，跟随新市场默认值
        if not self._slippage_override:
            self.slippage_rate = self.profile.default_slippage
        logger.info(f"[市场规则] {stock_code} -> {self.profile.describe()}")
        return self.profile

    @property
    def commission_rate(self):
        return self.profile.commission_rate

    @property
    def min_commission(self):
        return self.profile.min_commission

    @property
    def stamp_duty_rate(self):
        return self.profile.stamp_duty_sell

    # ------------------------------------------------------------------
    # 1. 数据加载
    # ------------------------------------------------------------------

    def load_data(self, stock_code: str) -> tuple:
        """
        加载并标准化历史行情与模型预测数据（支持跨市场智能寻址）。
        """
        clean_code, det_market, prefix = SymbolResolver.resolve(stock_code, market=self.profile.market)
        if self.profile.market != det_market:
            self.set_market(clean_code, market=det_market)

        # ---- 历史行情候选 ----
        hist_candidates = [
            os.path.join(self.data_dir, f"{prefix}_stock_data.csv"),
            os.path.join(self.data_dir, f"{clean_code}_stock_data.csv"),
            os.path.join(self.data_dir, f"{clean_code}_all_history.csv"),
            os.path.join(self.data_dir, f"{clean_code}.csv"),
        ]
        hist_candidates = list(dict.fromkeys(hist_candidates))
        hist_file = next((p for p in hist_candidates if os.path.exists(p)), None)
        if not hist_file:
            raise FileNotFoundError(
                f"[DATA ERROR] 未找到 {stock_code} ({det_market}) 的历史行情文件。\n"
                f"已查找路径:\n" + "\n".join(f"  {p}" for p in hist_candidates)
            )

        print(f"✅ 历史数据: {hist_file}")
        hist_df = pd.read_csv(hist_file, encoding='utf-8-sig')

        # 列名标准化
        hist_col_map = {
            '日期': 'date', 'timestamps': 'date',
            '开盘价': 'open',  'open': 'open',
            '收盘价': 'close', 'close': 'close',
            '最高价': 'high',  'high': 'high',
            '最低价': 'low',   'low': 'low',
            '成交量': 'volume','volume': 'volume',
            '成交额': 'amount','amount': 'amount',
        }
        hist_df = hist_df.rename(columns={k: v for k, v in hist_col_map.items() if k in hist_df.columns})

        for drop_col in ['index', 'Unnamed: 0', '股票代码', 'stock_code', 'pct_chg']:
            if drop_col in hist_df.columns:
                hist_df = hist_df.drop(columns=[drop_col])

        hist_df['date'] = pd.to_datetime(hist_df['date'])
        hist_df = hist_df.set_index('date').sort_index()
        if hist_df.index.has_duplicates:
            dup_cnt = int(hist_df.index.duplicated().sum())
            hist_df = hist_df[~hist_df.index.duplicated(keep='last')]
            logger.warning(f"[数据清洗] 历史行情存在 {dup_cnt} 个重复日期，已保留最后一条")

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in hist_df.columns:
                hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')

        print(f"   历史范围: {hist_df.index.min().date()} → {hist_df.index.max().date()}  ({len(hist_df)} 条)")

        # ---- 模型预测候选 ----
        pred_candidates = [
            os.path.join(self.model_dir, f"{clean_code}_wf_predictions_auto.csv"),
            os.path.join(self.model_dir, f"{prefix}_wf_predictions_auto.csv"),
            os.path.join(self.model_dir, f"{clean_code}_wf_predictions.csv"),
            os.path.join(self.model_dir, f"{prefix}_wf_predictions.csv"),
            os.path.join(self.model_dir, f"{clean_code}_wf_predictions_h3.csv"),
            os.path.join(self.model_dir, f"{clean_code}_wf_predictions_h5.csv"),
            os.path.join(self.model_dir, f"{clean_code}_detailed_predictions.csv"),
            os.path.join(self.model_dir, f"{clean_code}_kronos_predictions.csv"),
            os.path.join(self.model_dir, f"{clean_code}_predictions.csv"),
        ]
        pred_candidates = list(dict.fromkeys(pred_candidates))
        pred_file = next((p for p in pred_candidates if os.path.exists(p)), None)
        if not pred_file:
            raise FileNotFoundError(
                f"[DATA ERROR] 未找到 {stock_code} ({det_market}) 的预测文件。\n"
                f"已查找路径:\n" + "\n".join(f"  {p}" for p in pred_candidates)
            )

        print(f"✅ 预测数据: {pred_file}")
        pred_df = pd.read_csv(pred_file, encoding='utf-8-sig')

        date_src = next((c for c in ('日期', 'date', 'Date') if c in pred_df.columns),
                        pred_df.columns[0])
        pred_df = pred_df.rename(columns={date_src: 'date'})
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        pred_df = pred_df.set_index('date').sort_index()
        if pred_df.index.has_duplicates:
            dup_cnt = int(pred_df.index.duplicated().sum())
            pred_df = pred_df[~pred_df.index.duplicated(keep='last')]
            logger.warning(f"[数据清洗] 预测数据存在 {dup_cnt} 个重复日期，已保留最后一条")

        pred_df = KronosDataFetcher._normalize_pred_columns(pred_df)

        if 'pred_close' not in pred_df.columns:
            raise ValueError(
                f"[DATA ERROR] 预测文件 {pred_file} 中未找到可用的预测收盘价列。\n"
                f"当前列名: {list(pred_df.columns)}"
            )

        print(f"   预测范围: {pred_df.index.min().date()} → {pred_df.index.max().date()}  ({len(pred_df)} 条)")

        return hist_df, pred_df

    # ------------------------------------------------------------------
    # 2. 时序对齐与信号生成
    # ------------------------------------------------------------------

    def align_and_generate_signals(
        self,
        hist_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        z_score_threshold: float = 0.70,
        volatility_window: int = 14,
        min_overlap_days: int = 10,
        forecast_horizon: int = None,
        max_auto_horizon: int = 10,
        hard_stop_loss: float = 0.04,
        entry_percentile: float = 0.80,
        calib_window: int = 60,
        debias: bool = True,
        atr_window: int = 14,
        allow_interpolated: bool = False,
    ) -> pd.DataFrame:
        """
        核心时序对齐与信号生成。

        【时序不变量】
        对每个交易日 t，信号 raw_signal[t] 只允许使用在 **t-1 收盘时点已知** 的信息：
          - pred_*[t]      由严格早于 t 的数据推理得到（walk_forward_predict 保证）
          - close_prev[t]  = close[t-1]
          - volatility[t]  使用截至 t-1 的收益率窗口计算
          - atr[t]         使用截至 t-1 的真实波幅计算
        因此 raw_signal[t] 在 t-1 收盘即可确定，在 **t 日开盘** 撮合成交。

        【入场判定（自适应分位自校准）】
        改用每标的滚动分位数自校准：signal_score 位于该标的自身过去 calib_window 日分布的
        entry_percentile 分位之上，阈值随标的自身波动与市场环境自动校准。
        """
        if not (0 < hard_stop_loss < 1):
            raise ValueError(
                f"[PARAM ERROR] hard_stop_loss 须满足 0 < x < 1，当前值={hard_stop_loss}"
            )

        common_dates = hist_df.index.intersection(pred_df.index)
        overlap_days = len(common_dates)

        print(f"\n[对齐检测] 历史 ∩ 预测 重叠区间: {overlap_days} 个交易日")

        if overlap_days < min_overlap_days:
            raise ValueError(
                f"\n{'='*60}\n"
                f"[BACKTEST ERROR] 重叠交易日不足，无法执行真实样本外回测！\n"
                f"  • 历史数据结束: {hist_df.index.max().date()}\n"
                f"  • 预测数据起始: {pred_df.index.min().date()}\n"
                f"  • 当前重叠天数: {overlap_days} 天 (要求 >= {min_overlap_days} 天)\n"
                f"\n解决方案：\n"
                f"  请生成一份覆盖历史区间的 Kronos 预测序列，确保预测日期\n"
                f"  与历史真实收盘价有至少 {min_overlap_days} 个交易日的重叠。\n"
                f"{'='*60}"
            )

        print(f"   重叠区间: {common_dates.min().date()} → {common_dates.max().date()}")
        print(f"   [BACKTEST 模式] 以真实收盘价结算，执行严格样本外验证")

        # 自动推断预测时界 Horizon 与模式
        is_auto_horizon = False
        horizon_mode_col = 'horizon_mode' if 'horizon_mode' in pred_df.columns else (
            '时界' if '时界' in pred_df.columns else None)
        if forecast_horizon == 'auto' or (isinstance(forecast_horizon, str) and forecast_horizon.lower() == 'auto'):
            is_auto_horizon = True
        elif forecast_horizon is None:
            if horizon_mode_col and str(pred_df[horizon_mode_col].iloc[0]).lower() == 'auto':
                is_auto_horizon = True
            elif 'optimal_horizon' in pred_df.columns or '动态时界' in pred_df.columns:
                is_auto_horizon = True
            elif horizon_mode_col:
                try:
                    forecast_horizon = int(pred_df[horizon_mode_col].iloc[0])
                except Exception:
                    forecast_horizon = 1
            else:
                forecast_horizon = 1
        elif isinstance(forecast_horizon, (int, float)):
            forecast_horizon = int(forecast_horizon)

        # --------------------------------------------------------------
        # 历史侧指标：全部用「截至 t-1」的窗口，保证 t 日信号可在 t-1 收盘确定
        # --------------------------------------------------------------
        hist_work = hist_df.copy()
        hist_work['daily_return'] = hist_work['close'].pct_change()

        vol_raw = (hist_work['daily_return']
                   .rolling(window=volatility_window, min_periods=5)
                   .std()
                   .ffill())

        # 真实波动度量 ATR
        _high = hist_work['high'] if 'high' in hist_work.columns else hist_work['close']
        _low = hist_work['low'] if 'low' in hist_work.columns else hist_work['close']
        _pc = hist_work['close'].shift(1)
        true_range = pd.concat([(_high - _low).abs(),
                                (_high - _pc).abs(),
                                (_low - _pc).abs()], axis=1).max(axis=1)
        atr_raw = true_range.rolling(window=atr_window, min_periods=5).mean().ffill()
        atr_pct_raw = (atr_raw / hist_work['close']).ffill()

        # 时点修正：t 日决策只能看到截至 t-1 的统计量
        hist_work['volatility'] = vol_raw.shift(1)
        hist_work['atr'] = atr_raw.shift(1)
        hist_work['atr_pct'] = atr_pct_raw.shift(1)
        hist_work['close_prev'] = hist_work['close'].shift(1)

        # 趋势状态滞后一期
        ma_fast = hist_work['close'].rolling(20, min_periods=10).mean().shift(1)
        ma_slow = hist_work['close'].rolling(60, min_periods=20).mean().shift(1)
        hist_work['ma_fast'] = ma_fast
        hist_work['ma_slow'] = ma_slow

        # 截取重叠回测区间
        df = hist_work.loc[common_dates].copy()
        df['pred_close'] = pred_df.loc[common_dates, 'pred_close']

        for opt_col in ['pred_upper', 'pred_lower', 'pred_expected', 'pred_path_max',
                        'pred_terminal', 'pred_slope', 'pred_min_down', 'peak_premium',
                        'is_interpolated']:
            if opt_col in pred_df.columns:
                df[opt_col] = pred_df.loc[common_dates, opt_col]

        if 'pred_path_max' not in df.columns:
            df['pred_path_max'] = df['pred_close']

        # 剔除插值行
        if 'is_interpolated' in df.columns and not allow_interpolated:
            interp_mask = df['is_interpolated'].fillna(0) > 0
            if interp_mask.any():
                logger.info(f"  [数据过滤] 剔除 {int(interp_mask.sum())} 条插值预测行")
                df = df[~interp_mask]

        # 动态最优时界 (optimal_horizon)
        dyn_h_col = None
        for cand in ['optimal_horizon', '动态时界']:
            if cand in pred_df.columns:
                dyn_h_col = cand
                break

        if dyn_h_col is not None:
            df['target_horizon'] = (pred_df.loc[common_dates, dyn_h_col]
                                    .reindex(df.index).astype(float)
                                    .fillna(5).clip(1, 20).round().astype(int))
            horizon_display = "Auto(极值自适应)"
        else:
            # ⚠️ fallback：预测缓存中缺少 '动态时界' / 'optimal_horizon' 列。
            # 可能原因：使用了旧版预测脚本生成的缓存（未包含该列）。
            # 建议：删除旧缓存并勾选【强制刷新】重新生成预测以获取动态时界数据。
            if is_auto_horizon:
                # auto 模式下 fallback 到 max_auto_horizon 的一半（保守合理估计）
                h_int = max(3, (max_auto_horizon if isinstance(max_auto_horizon, int) else 10) // 2)
                logger.warning(
                    f"[时界Fallback] 预测 CSV 缺少 '动态时界' 列（可能为旧版缓存）。"
                    f"已自动降级为固定 {h_int} 天。"
                    f"建议：勾选【强制刷新】删除旧缓存后重新运行以恢复动态时界。"
                )
            else:
                h_int = int(forecast_horizon) if forecast_horizon is not None else 1
            df['target_horizon'] = h_int
            horizon_display = f"{h_int}天(Fallback-缺动态时界列)"

        df = df.dropna(subset=['close_prev', 'volatility'])
        if len(df) < max(min_overlap_days, 2):
            raise ValueError(
                f"[BACKTEST ERROR] 有效信号行不足（{len(df)} 行）。"
            )

        vol_scale = np.sqrt(df['target_horizon'].astype(float))

        # --------------------------------------------------------------
        # 预期收益与信号强度
        # --------------------------------------------------------------
        df['expected_return'] = (df['pred_close'] - df['close_prev']) / df['close_prev']
        df['target_return'] = (df['pred_path_max'] - df['close_prev']) / df['close_prev']

        if debias:
            realized = (df['close'] - df['close_prev']) / df['close_prev']
            err = (df['expected_return'] - realized)
            bias = err.shift(1).rolling(calib_window, min_periods=max(10, calib_window // 4)).median()
            df['pred_bias'] = bias.fillna(0.0)
            df['expected_return_adj'] = df['expected_return'] - df['pred_bias']
        else:
            df['pred_bias'] = 0.0
            df['expected_return_adj'] = df['expected_return']

        df['volatility_scaled'] = (df['volatility'] * vol_scale).replace(0, np.nan)
        df['signal_score'] = df['expected_return_adj'] / df['volatility_scaled']
        df['signal_score'] = df['signal_score'].replace([np.inf, -np.inf], np.nan)

        # --------------------------------------------------------------
        # 可交易性：开盘涨跌停（用前收算理论价，在执行日开盘判断）
        # --------------------------------------------------------------
        df = compute_open_limit_flags(df, self.profile)
        # 兼容旧列名：信号层否决仍用 limit_up；撮合层优先读 open_limit_*
        df['limit_up'] = df['open_limit_up']
        df['limit_down'] = df['open_limit_down']

        # --------------------------------------------------------------
        # 入场判定（自校准模式）
        # --------------------------------------------------------------
        veto = pd.Series('', index=df.index, dtype=object)

        def _add_veto(mask, reason):
            sel = mask & (veto == '')
            veto[sel] = reason

        roll = df['signal_score'].shift(1).rolling(
            calib_window, min_periods=max(10, calib_window // 4))
        df['entry_threshold'] = roll.quantile(entry_percentile)
        warmup = df['entry_threshold'].isna()
        if warmup.any():
            fallback = df['signal_score'].quantile(entry_percentile)
            df.loc[warmup, 'entry_threshold'] = fallback

        buy_cond = df['signal_score'] > df['entry_threshold']
        _add_veto(~buy_cond, '信号强度未进入滚动分位阈值')

        if 'pred_slope' in df.columns:
            c = df['pred_slope'] > 0
            _add_veto(~c, '预测斜率向下')
            buy_cond &= c

        # T+N 合规与波峰可实现性校验：若结算周期为 T+1，波峰时界为1日则无法在峰值卖出
        if getattr(self.profile, 'settlement_days', 1) >= 1 and 'target_horizon' in df.columns:
            c = df['target_horizon'] > 1
            _add_veto(~c, f'预测波峰时界<=1日(T+{self.profile.settlement_days}无法波峰离场)')
            buy_cond &= c

        trend_up = df['ma_fast'] >= df['ma_slow']
        confirm = df['close'] > df['ma_fast']
        df['regime'] = np.where(trend_up, 'UPTREND', 'DOWNTREND')
        regime_ok = trend_up | confirm
        _add_veto(~regime_ok, '下跌趋势且无短期确认')
        buy_cond &= regime_ok

        c = df['expected_return_adj'] > 0
        _add_veto(~c, '去偏后预期收益为负')
        buy_cond &= c

        if 'pred_min_down' in df.columns:
            depth = df['pred_min_down'].abs() / max(hard_stop_loss, 1e-6)
            scale = np.where(depth <= 1.0, 1.0,
                             np.where(depth >= 2.0, 0.0,
                                      1.0 - 0.6 * (depth - 1.0)))
            df['position_scale'] = np.clip(scale, 0.0, 1.0)
            c = df['position_scale'] > 0
            _add_veto(~c, '预测下探过深(仓位系数为0)')
            buy_cond &= c
        else:
            df['position_scale'] = 1.0

        c = df['open_limit_up'] == 0
        _add_veto(~c, '开盘封涨停无法买入')
        buy_cond &= c

        df['raw_signal'] = np.where(buy_cond.fillna(False), 1, 0)
        df['veto_reason'] = np.where(df['raw_signal'] == 1, '', veto)

        bear_cond = df['signal_score'] < -0.5
        if 'pred_slope' in df.columns:
            bear_cond = bear_cond | ((df['pred_slope'] < 0) & (df['expected_return_adj'] < 0))
        df['bearish_signal'] = np.where(bear_cond.fillna(False), 1, 0)

        # 严格防偷价：t 日决策由 t-1 收盘确定，t 日开盘撮合 (lag=0)
        df['exec_signal'] = df['raw_signal'].astype(int)
        df['exec_bearish'] = df['bearish_signal'].astype(int)
        df['exec_target_horizon'] = df['target_horizon'].fillna(5).astype(int)
        df['exec_position_scale'] = df['position_scale'].fillna(0.0)
        df['execution_lag'] = 0

        # --------------------------------------------------------------
        # 诊断输出
        # --------------------------------------------------------------
        buy_cnt = int(df['raw_signal'].sum())
        bear_cnt = int(df['bearish_signal'].sum())
        thr_label = f"滚动{calib_window}日 {entry_percentile:.0%} 分位"
        logger.info(
            f"[信号诊断] 重叠 {len(df)} 交易日 ({df.index.min().date()} ~ {df.index.max().date()}) | "
            f"自校准模式 | 预测时界={horizon_display} | 入场阈值={thr_label} | "
            f"波动率窗口={volatility_window}天 | ATR窗口={atr_window}天 | 硬止损={hard_stop_loss:.2%} | "
            f"市场规则={self.profile.describe()}"
        )
        logger.info(
            f"[信号诊断] 信号强度统计: 最小值={df['signal_score'].min():.3f}, "
            f"中位数={df['signal_score'].median():.3f}, "
            f"平均值={df['signal_score'].mean():.3f}, "
            f"最大值={df['signal_score'].max():.3f}"
        )
        if debias:
            logger.info(
                f"[信号诊断] 去偏: 原始预期收益均值={df['expected_return'].mean():+.2%}, "
                f"滚动偏差中位数={df['pred_bias'].median():+.2%}, "
                f"去偏后均值={df['expected_return_adj'].mean():+.2%}"
            )
        logger.info(
            f"[信号诊断] 触发结果: 买入信号 {buy_cnt} 天 ({buy_cnt/len(df):.1%}), "
            f"转空离场信号 {bear_cnt} 天 ({bear_cnt/len(df):.1%})"
        )

        # 否决原因分布：直接回答「为什么这段时间不交易」
        veto_counts = df.loc[df['raw_signal'] == 0, 'veto_reason'].value_counts()
        if len(veto_counts) > 0:
            logger.info("[信号诊断] 未触发原因分布:")
            for reason, cnt in veto_counts.items():
                if reason:
                    logger.info(f"    {reason}: {cnt} 天 ({cnt/len(df):.1%})")

        if buy_cnt > 0:
            sig_rows = df[df['raw_signal'] == 1]
            for s_date, s_row in sig_rows.iterrows():
                logger.info(
                    f"  ⚡ 产生买入信号: {s_date.date()} 昨收={s_row['close_prev']:.3f}, "
                    f"期望价={s_row['pred_close']:.3f}, 去偏预期涨幅={s_row['expected_return_adj']*100:+.2f}%, "
                    f"波动率={s_row['volatility']*100:.2f}%, 强度={s_row['signal_score']:.2f} "
                    f"(阈值{s_row['entry_threshold']:.2f}), 仓位系数={s_row['position_scale']:.2f}"
                )
        else:
            top_reason = veto_counts.index[0] if len(veto_counts) > 0 else '未知'
            logger.warning(
                f"  ⚠️ 整个区间未触发任何买入信号。主要否决原因: {top_reason}。"
                f"可调低 entry_percentile（当前 {entry_percentile:.0%}）或检查预测技能体检结果。"
            )

        print(f"   信号统计: 共 {len(df)} 个交易日，"
              f"买入信号 {buy_cnt} 天 ({buy_cnt/len(df):.1%})，"
              f"转空离场信号 {bear_cnt} 天 ({bear_cnt/len(df):.1%})")

        return df

    # ------------------------------------------------------------------
    # 2b. 预测技能体检（准入门槛）
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_prediction_skill(df: pd.DataFrame, hard_stop_loss: float = 0.04) -> dict:
        """
        在回测之前体检预测本身有没有可用的预测能力。

        策略再精巧也救不回一个没有技能的预测源，所以把「能不能交易这个标的」
        变成一道显式的前置检查，而不是等回测跑完看收益猜原因。

        三项核心指标
        ------------
        IC (信息系数)
            预测收益与真实收益的 Spearman 秩相关。衡量**排序**能力。
            |IC| < 0.03 基本等同噪声。
        magnitude_ratio (幅度比)
            预测收益均值 / 真实收益均值。远大于 1 说明幅度被系统性高估，
            此时 Z-Score 分子虚高，阈值形同虚设。
        breach_rate (下探跌破率)
            实际最低价跌破预测最大下探的比例。该值应接近预测所用的保守分位
            （默认 10%）。若达到 50%+，说明下探预测不具备风险边界能力，
            不能用作硬过滤条件。

        返回
        ----
        dict，含各项指标与 tradable 准入结论
        """
        out = {}
        need = {'pred_close', 'close', 'close_prev'}
        if not need.issubset(df.columns) or len(df) < 20:
            return {'error': f'样本不足或缺列，无法体检（行数={len(df)}）', 'tradable': False}

        realized = ((df['close'] - df['close_prev']) / df['close_prev']).astype(float)
        predicted = ((df['pred_close'] - df['close_prev']) / df['close_prev']).astype(float)
        mask = realized.notna() & predicted.notna()
        realized, predicted = realized[mask], predicted[mask]
        n = len(realized)
        out['样本数'] = n
        if n < 20:
            return {'error': f'有效样本仅 {n} 条', 'tradable': False}

        # Spearman 秩相关（不依赖 scipy）
        ic = float(predicted.rank().corr(realized.rank()))
        out['IC'] = ic
        # t 统计量与显著性（近似）
        if abs(ic) < 1 - 1e-12 and n > 2:
            t_stat = ic * np.sqrt((n - 2) / (1 - ic ** 2))
        else:
            t_stat = 0.0
        out['IC_t值'] = float(t_stat)
        out['IC显著'] = bool(abs(t_stat) > 2.0)

        # 方向命中率
        out['方向命中率'] = float(((predicted > 0) == (realized > 0)).mean())

        # 幅度校准
        mean_pred, mean_real = float(predicted.mean()), float(realized.mean())
        out['预测收益均值'] = mean_pred
        out['真实收益均值'] = mean_real
        out['幅度比'] = (mean_pred / mean_real) if abs(mean_real) > 1e-9 else float('inf')
        out['幅度偏差'] = mean_pred - mean_real

        # 下探预测校准
        if 'pred_min_down' in df.columns and 'low' in df.columns:
            sub = df.loc[mask]
            actual_down = ((sub['low'] - sub['close_prev']) / sub['close_prev']).astype(float)
            pmd = sub['pred_min_down'].astype(float)
            valid = actual_down.notna() & pmd.notna()
            if valid.sum() >= 20:
                out['下探跌破率'] = float((actual_down[valid] < pmd[valid]).mean())

        # 准入判定
        reasons = []
        if abs(ic) < 0.03:
            reasons.append(f'IC={ic:.3f} 接近噪声(<0.03)')
        if not out['IC显著']:
            reasons.append(f'IC 不显著(t={t_stat:.2f})')
        if ic < 0:
            reasons.append(f'IC 为负({ic:.3f})，预测方向与真实相反')
        out['不通过原因'] = reasons
        out['tradable'] = len(reasons) == 0
        return out

    @staticmethod
    def print_skill_report(skill: dict, stock_code: str = ''):
        """打印预测技能体检报告"""
        print(f"\n{'='*64}")
        print(f"  预测技能体检 {stock_code}")
        print(f"{'='*64}")
        if 'error' in skill:
            print(f"  ❌ {skill['error']}")
            return
        print(f"  样本数            : {skill['样本数']}")
        print(f"  IC (秩相关)       : {skill['IC']:+.4f}  (t={skill['IC_t值']:+.2f}, "
              f"{'显著' if skill['IC显著'] else '不显著'})")
        print(f"  方向命中率        : {skill['方向命中率']:.1%}")
        print(f"  预测收益均值      : {skill['预测收益均值']:+.3%}")
        print(f"  真实收益均值      : {skill['真实收益均值']:+.3%}")
        mr = skill['幅度比']
        print(f"  幅度比(预测/真实) : {mr:.1f}x" if np.isfinite(mr) else "  幅度比(预测/真实) : inf")
        if '下探跌破率' in skill:
            br = skill['下探跌破率']
            flag = '✅' if br <= 0.20 else ('⚠️' if br <= 0.35 else '❌')
            print(f"  下探跌破率        : {br:.1%} {flag}  (目标≈10%)")
        if skill['tradable']:
            print(f"  结论              : ✅ 通过准入，可进入回测")
        else:
            print(f"  结论              : ❌ 未通过准入")
            for r in skill['不通过原因']:
                print(f"       - {r}")
        print(f"{'='*64}")

    # ------------------------------------------------------------------
    # 2c. 验证协议：因果性 / 参数敏感性 / 分段稳健性
    # ------------------------------------------------------------------

    def check_causality(self, hist_df: pd.DataFrame, pred_df: pd.DataFrame,
                        n_probe: int = 12, **kwargs) -> dict:
        """
        前视偏差检验：验证信号确实只依赖历史信息。

        做法
        ----
        把数据在第 k 天截断，只用 [0, k] 的数据重算信号，
        再与全样本计算出的第 k 天信号比对。若两者一致，说明计算第 k 天信号时
        没有用到 k 之后的任何信息。

        这能抓出 .bfill()、未 shift 的滚动统计量、全样本分位数等典型泄漏写法。
        """
        full = self.align_and_generate_signals(hist_df, pred_df, **kwargs)
        dates = list(full.index)
        if len(dates) < 40:
            return {'error': '样本过短，无法做因果性检验'}

        # 在后半段均匀取若干个探针点
        start = len(dates) // 2
        probes = [dates[i] for i in
                  np.linspace(start, len(dates) - 1, min(n_probe, len(dates) - start)).astype(int)]

        mismatches = []
        for d in probes:
            h_trunc = hist_df[hist_df.index <= d]
            p_trunc = pred_df[pred_df.index <= d]
            try:
                partial = self.align_and_generate_signals(h_trunc, p_trunc, **kwargs)
            except Exception:
                continue
            if d not in partial.index:
                continue
            for col in ('volatility', 'atr_pct', 'expected_return'):
                if col not in full.columns or col not in partial.columns:
                    continue
                a, b = full.loc[d, col], partial.loc[d, col]
                if pd.isna(a) and pd.isna(b):
                    continue
                if pd.isna(a) != pd.isna(b) or abs(float(a) - float(b)) > 1e-9:
                    mismatches.append({'date': str(d.date()), 'col': col,
                                       'full': float(a), 'truncated': float(b)})

        return {
            '探针数': len(probes),
            '不一致项': mismatches,
            'passed': len(mismatches) == 0,
        }

    def run_validation_protocol(self, hist_df: pd.DataFrame, pred_df: pd.DataFrame,
                                stock_code: str = '', **base_kwargs) -> dict:
        """
        跨时段与参数稳健性检验。

        单一参数组合在单一时段上的好成绩没有意义。本协议检查：
          1. 因果性：信号是否真的只用历史信息
          2. 参数敏感性：入场分位在 0.60~0.90 之间扫描，看收益是否只在某一点为正
          3. 分段稳健性：把回测区间等分为 3 段，看各段表现是否一致

        判定标准：参数扫描中正收益占比 < 50%，或分段中正收益段数 <= 1，
        视为该结果依赖特定参数/时段，不可采信。
        """
        report = {'stock_code': stock_code}

        print(f"\n{'='*64}")
        print(f"  稳健性验证协议 {stock_code}")
        print(f"{'='*64}")

        # ---- 1. 因果性 ----
        print("\n[1/3] 前视偏差（因果性）检验 ...")
        causal = self.check_causality(hist_df, pred_df, **base_kwargs)
        report['causality'] = causal
        if causal.get('passed'):
            print(f"      ✅ 通过：{causal['探针数']} 个探针点上，"
                  f"截断重算与全样本计算完全一致，无前视偏差")
        elif 'error' in causal:
            print(f"      ⚠️ {causal['error']}")
        else:
            print(f"      ❌ 发现 {len(causal['不一致项'])} 处不一致（存在前视偏差）:")
            for mm in causal['不一致项'][:5]:
                print(f"         {mm['date']} {mm['col']}: "
                      f"全样本={mm['full']:.6f} vs 截断={mm['truncated']:.6f}")

        # ---- 2. 参数敏感性 ----
        print("\n[2/3] 入场分位参数敏感性 ...")
        sweep = []
        for pct in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            kw = dict(base_kwargs); kw['entry_percentile'] = pct
            try:
                sig = self.align_and_generate_signals(hist_df, pred_df, **kw)
                bdf, tr = self.run_backtest_engine(
                    sig,
                    hard_stop_loss=base_kwargs.get('hard_stop_loss', 0.04),
                    max_holding_days=15)
                mt = self.evaluate_metrics(bdf, tr)
                sweep.append({'分位': pct, '收益': mt['累计收益率'],
                              '回撤': mt['最大回撤'], '笔数': mt['总交易笔数']})
            except Exception as e:
                sweep.append({'分位': pct, '收益': np.nan, '回撤': np.nan,
                              '笔数': 0, 'err': str(e)[:40]})
        report['sweep'] = sweep
        valid = [s for s in sweep if np.isfinite(s.get('收益', np.nan))]
        pos = [s for s in valid if s['收益'] > 0]
        pos_ratio = len(pos) / len(valid) if valid else 0.0
        print(f"      {'分位':>6} {'累计收益':>10} {'最大回撤':>10} {'笔数':>6}")
        for s in sweep:
            if np.isfinite(s.get('收益', np.nan)):
                print(f"      {s['分位']:>6.2f} {s['收益']:>+9.2%} {s['回撤']:>+9.2%} {s['笔数']:>6d}")
            else:
                print(f"      {s['分位']:>6.2f} {'失败':>10} {'-':>10} {'-':>6}")
        print(f"      正收益占比 {pos_ratio:.0%} "
              f"({'✅ 参数稳健' if pos_ratio >= 0.5 else '❌ 依赖特定参数，不可采信'})")

        # ---- 3. 分段稳健性 ----
        print("\n[3/3] 分段稳健性（等分 3 段）...")
        segs = []
        try:
            sig = self.align_and_generate_signals(hist_df, pred_df, **base_kwargs)
            bounds = np.linspace(0, len(sig), 4).astype(int)
            for i in range(3):
                part = sig.iloc[bounds[i]:bounds[i + 1]]
                if len(part) < 15:
                    continue
                bdf, tr = self.run_backtest_engine(
                    part, hard_stop_loss=base_kwargs.get('hard_stop_loss', 0.04),
                    max_holding_days=15)
                mt = self.evaluate_metrics(bdf, tr)
                segs.append({'段': i + 1,
                             '起': str(part.index.min().date()),
                             '止': str(part.index.max().date()),
                             '收益': mt['累计收益率'], '笔数': mt['总交易笔数']})
        except Exception as e:
            print(f"      ⚠️ 分段检验失败: {e}")
        report['segments'] = segs
        for s in segs:
            print(f"      第{s['段']}段 {s['起']} ~ {s['止']}: "
                  f"收益 {s['收益']:>+7.2%}, {s['笔数']} 笔")
        pos_seg = sum(1 for s in segs if s['收益'] > 0)
        if segs:
            print(f"      正收益段 {pos_seg}/{len(segs)} "
                  f"({'✅ 跨时段一致' if pos_seg >= 2 else '❌ 表现集中于个别时段'})")

        report['passed'] = (causal.get('passed', False)
                            and pos_ratio >= 0.5
                            and pos_seg >= 2)
        print(f"\n  总体结论: {'✅ 通过稳健性验证' if report['passed'] else '❌ 未通过，结果不可采信为策略有效性证据'}")
        print(f"{'='*64}")
        return report

    # ------------------------------------------------------------------
    # 3. 撮合与风控状态机
    # ------------------------------------------------------------------

    def run_backtest_engine(
        self,
        df: pd.DataFrame,
        hard_stop_loss: float = 0.04,
        trailing_stop: float = 0.025,
        trailing_stop_min_profit: float = 0.02,
        max_holding_days: int = 5,
        min_lot_guarantee: bool = True,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.80,
        k_stop: float = 1.5,
        k_trail: float = 1.2,
        min_stop: float = 0.02,
        max_stop: float = 0.12,
        intraday_stop: bool = True,
        consecutive_loss_cut: int = 2,
    ) -> tuple:
        """
        逐日仿真撮合引擎（状态机）。

        【撮合规则设计】
        - 开仓/平仓撮合价格：使用当日开盘价（open）乘滑点 (1 ± slippage_rate)。
          因为 exec_signal / pending_close_reason 是前一日收盘后计算出的决策，在今日开盘立即撮合。
        - 每日盯市结算（Mark-to-Market）：以当日真实收盘价 close 结算总账户资产 portfolio_value。
        - 交易约束：多市场制度全真建模（A股100股整手/港股自定义每手/美股1股，T+1/T+0，印花税及交易所规费）。

        参数示例 (Parameters)
        --------------------
        df : pd.DataFrame
            信号对齐矩阵（由 align_and_generate_signals 产出，含 open/close/exec_signal 等列）
        hard_stop_loss : float, default=0.04
            可配置硬止损线（相比成本价）。示例: 0.04 表示亏损达 -4.0% 时止损
        trailing_stop : float, default=0.025
            移动止盈回撤阈值。示例: 0.025 表示自持仓期间最高价回撤 -2.5% 时离场
        trailing_stop_min_profit : float, default=0.02
            移动止盈启动门槛。示例: 0.02 表示必须在浮盈超过 +2.0% 后才激活峰值回撤止盈
        max_holding_days : int, default=5
            单笔交易最大持仓天数。示例: 5 表示持满 5 个交易日必须平仓
        min_lot_guarantee : bool, default=True
            一手保底机制。若风险预算计算股数经每手截断为 0，但在资金和仓位上限允许时保底买入 1 手。
        """
        profile = self.profile
        cash = self.initial_capital       # 账户可用现金
        position_shares = 0               # 当前持股数量（0 表示空仓）
        avg_entry_price = 0.0             # 持仓成本均价（含滑点）
        peak_price = 0.0                  # 持仓期间最高价
        holding_days = 0                  # 连续持仓天数
        active_target_horizon = None      # 当前持仓的目标时界
        entry_bar_idx = None              # 开仓所在的 bar 序号，用于 T+N 合规判定
        active_stop_pct = hard_stop_loss  # 本笔交易实际生效的止损幅度
        active_trail_pct = trailing_stop  # 本笔交易实际生效的移动止盈幅度
        stop_price = None                 # 本笔交易的绝对止损价
        consecutive_losses = 0            # 连续亏损计数，用于降仓

        pending_close_reason = None       # 次日开盘待执行的平仓指令
        pending_close_price = None        # 指定成交价（盘中止损时使用）

        trades = []
        records = []
        rows = list(df.iterrows())

        def _sell(date, shares, price, reason, cur_cash):
            """按市场规则结算一笔卖出，返回 (现金, 费用, 单笔收益率)"""
            gross = shares * price
            fee = profile.sell_cost(gross, trade_date=date)
            pnl = (price - avg_entry_price) / avg_entry_price if avg_entry_price > 0 else 0.0
            return cur_cash + gross - fee, fee, pnl

        for idx, (date, row) in enumerate(rows):
            open_price = row['open'] if not pd.isna(row.get('open', np.nan)) else row['close']
            close_price = row['close']
            low_price = row.get('low', close_price)
            if pd.isna(low_price):
                low_price = close_price
            exec_signal = int(row['exec_signal'])
            exec_bearish = int(row.get('exec_bearish', 0))
            # 优先使用开盘涨跌停标志（相对前收）；兼容旧 limit_* 列
            limit_down = int(row.get('open_limit_down', row.get('limit_down', 0)))
            limit_up = int(row.get('open_limit_up', row.get('limit_up', 0)))

            # ----------------------------------------------------------------
            # A. 执行上一交易日决定的平仓指令
            # ----------------------------------------------------------------
            if pending_close_reason and position_shares > 0:
                if limit_down:
                    # 开盘封跌停卖不出，指令顺延到下一可交易日
                    logger.info(f"[{date.date()}] ⛔ 开盘封跌停无法卖出，平仓指令顺延: 【{pending_close_reason}】")
                else:
                    if pending_close_price is not None:
                        # 盘中止损：若开盘已跳空低于止损价，则按开盘价成交
                        fill = min(pending_close_price, open_price)
                    else:
                        fill = open_price
                    sell_price = fill * (1 - self.slippage_rate)
                    cash, fee, trade_pnl = _sell(date, position_shares, sell_price,
                                                 pending_close_reason, cash)
                    logger.info(
                        f"[{date.date()}] 🔴 执行平仓: {position_shares}股 @ {sell_price:.3f} | "
                        f"原因: 【{pending_close_reason}】 | 单笔盈亏: {trade_pnl*100:+.2f}% | "
                        f"税费: {fee:.2f} | 账户资金: {cash:,.2f}"
                    )
                    trades.append({
                        'date': date, 'action': 'SELL',
                        'price': round(sell_price, 4), 'shares': position_shares,
                        'commission': round(fee, 2), 'pnl_pct': round(trade_pnl, 6),
                        'reason': pending_close_reason, 'cash_balance': round(cash, 2),
                    })
                    consecutive_losses = consecutive_losses + 1 if trade_pnl <= 0 else 0
                    position_shares = 0
                    avg_entry_price = 0.0
                    peak_price = 0.0
                    holding_days = 0
                    active_target_horizon = None
                    entry_bar_idx = None
                    stop_price = None
                    pending_close_reason = None
                    pending_close_price = None

            # ----------------------------------------------------------------
            # B. 开仓买入
            # ----------------------------------------------------------------
            if position_shares == 0 and exec_signal == 1 and not limit_up:
                buy_price = open_price * (1 + self.slippage_rate)
                portfolio_value = cash        # 空仓时总净值即现金

                # 本笔交易的止损幅度：ATR 自适应
                atr_pct = row.get('atr_pct', np.nan)
                if pd.isna(atr_pct) or atr_pct <= 0:
                    active_stop_pct = hard_stop_loss
                    active_trail_pct = trailing_stop
                else:
                    active_stop_pct = float(np.clip(k_stop * atr_pct, min_stop, max_stop))
                    active_trail_pct = float(np.clip(k_trail * atr_pct, min_stop, max_stop))

                # 风险预算定仓：单笔最大亏损 = 净值 * risk_per_trade
                scale = float(row.get('exec_position_scale', 1.0) or 0.0)
                if consecutive_loss_cut > 0 and consecutive_losses >= consecutive_loss_cut:
                    scale *= 0.5   # 连亏降仓
                risk_budget = portfolio_value * risk_per_trade * scale
                stop_distance = active_stop_pct * buy_price
                raw_shares = (risk_budget / stop_distance) if stop_distance > 0 else 0
                # 仓位上限与现金约束
                cap_by_pct = (portfolio_value * max_position_pct) / buy_price if buy_price > 0 else 0
                cap_by_cash = (cash * 0.98) / buy_price if buy_price > 0 else 0
                raw_shares = min(raw_shares, cap_by_pct, cap_by_cash)

                shares_to_buy = profile.round_shares(raw_shares)

                # 一手保底机制 (Min-Lot Guarantee):
                # 针对港美股高单价股票（如腾讯、小米、美股大盘蓝筹），若严格风控算出的股数经整手截断后为 0，
                # 且当前账户现金和最大仓位允许至少买入 1 手，则保底买入 1 手，防止交易完全停摆。
                if shares_to_buy < profile.lot_size and min_lot_guarantee:
                    one_lot_cost = profile.lot_size * buy_price
                    one_lot_total = one_lot_cost + profile.buy_cost(one_lot_cost, trade_date=date)
                    if cash >= one_lot_total and (one_lot_cost <= portfolio_value * max_position_pct):
                        shares_to_buy = profile.lot_size
                        logger.info(
                            f"[{date.date()}] 🛡️ 触发一手保底: 计算股数 {raw_shares:.1f} 股不足一手({profile.lot_size}股)，"
                            f"账户现金充足，保底买入 1 手 ({shares_to_buy}股)"
                        )

                if shares_to_buy >= profile.lot_size:
                    gross_cost = shares_to_buy * buy_price
                    fee = profile.buy_cost(gross_cost, trade_date=date)
                    total_outflow = gross_cost + fee

                    if cash >= total_outflow:
                        cash -= total_outflow
                        position_shares = shares_to_buy
                        avg_entry_price = buy_price
                        peak_price = close_price
                        holding_days = 0
                        entry_bar_idx = idx
                        stop_price = buy_price * (1 - active_stop_pct)
                        active_target_horizon = int(row.get('exec_target_horizon', max_holding_days) or max_holding_days)

                        logger.info(
                            f"[{date.date()}] 🟢 开仓买入: {shares_to_buy}股 @ {buy_price:.3f} | "
                            f"目标时界: {active_target_horizon}日 | 止损={active_stop_pct:.2%}"
                            f"(价 {stop_price:.3f}) | 耗资: {total_outflow:,.2f} (费用: {fee:.2f}) | "
                            f"剩余现金: {cash:,.2f}"
                        )
                        trades.append({
                            'date': date, 'action': 'BUY',
                            'price': round(buy_price, 4), 'shares': shares_to_buy,
                            'commission': round(fee, 2), 'pnl_pct': 0.0,
                            'reason': f"信号开仓(目标H={active_target_horizon}日)",
                            'cash_balance': round(cash, 2),
                            'target_horizon': active_target_horizon,
                            'stop_pct': round(active_stop_pct, 4),
                        })
            elif position_shares == 0 and exec_signal == 1 and limit_up:
                logger.info(f"[{date.date()}] ⛔ 开盘封涨停无法买入，放弃本次开仓信号")

            # ----------------------------------------------------------------
            # C. 持仓盯市与离场决策
            # ----------------------------------------------------------------
            if position_shares > 0:
                holding_days += 1
                peak_price = max(peak_price, close_price)

                # T+N 合规：A股当日买入不可当日卖出（settlement_days=1），港美股T+0可当日离场
                can_exit_today = (entry_bar_idx is None
                                  or (idx - entry_bar_idx) >= profile.settlement_days)

                # ---- 盘中止损：在允许交易的当日立即成交 ----
                if (intraday_stop and can_exit_today
                        and stop_price is not None and low_price <= stop_price
                        and not limit_down):
                    fill = stop_price if open_price > stop_price else open_price  # 跳空低开按开盘价
                    sell_price = fill * (1 - self.slippage_rate)
                    cash, fee, trade_pnl = _sell(date, position_shares, sell_price, '盘中止损', cash)
                    logger.info(
                        f"[{date.date()}] 🔴 盘中止损成交: {position_shares}股 @ {sell_price:.3f} | "
                        f"止损价={stop_price:.3f}, 当日最低={low_price:.3f} | "
                        f"单笔盈亏: {trade_pnl*100:+.2f}% | 账户资金: {cash:,.2f}"
                    )
                    trades.append({
                        'date': date, 'action': 'SELL',
                        'price': round(sell_price, 4), 'shares': position_shares,
                        'commission': round(fee, 2), 'pnl_pct': round(trade_pnl, 6),
                        'reason': f'盘中止损(触及{stop_price:.3f})',
                        'cash_balance': round(cash, 2),
                    })
                    consecutive_losses = consecutive_losses + 1 if trade_pnl <= 0 else 0
                    position_shares = 0
                    avg_entry_price = 0.0
                    peak_price = 0.0
                    holding_days = 0
                    active_target_horizon = None
                    entry_bar_idx = None
                    stop_price = None
                    pending_close_reason = None
                    pending_close_price = None
                    records.append({
                        'date': date,
                        'portfolio_value': cash,
                        'cash': cash,
                        'shares': 0,
                        'close': close_price,
                        'exec_signal': exec_signal,
                    })
                    continue

                drawdown_from_cost = (close_price - avg_entry_price) / avg_entry_price
                drawdown_from_peak = (close_price - peak_price) / peak_price
                ma_fast_val = row.get('ma_fast', None)

                if drawdown_from_cost <= -active_stop_pct:
                    pending_close_reason = f"硬止损({drawdown_from_cost:.2%})"
                    pending_close_price = None
                elif (drawdown_from_peak <= -active_trail_pct
                      and drawdown_from_cost >= trailing_stop_min_profit):
                    pending_close_reason = f"移动止盈(峰值回撤{drawdown_from_peak:.2%})"
                    pending_close_price = None
                elif (active_target_horizon is not None
                      and holding_days >= active_target_horizon):
                    if drawdown_from_cost <= 0:
                        # 浮亏：模型预测波峰窗口已过，假说彻底证伪，坚决时间止损！
                        pending_close_reason = (
                            f"时间止损-模型信号证伪({holding_days}日/{active_target_horizon}日目标,"
                            f"浮亏{drawdown_from_cost:.2%})"
                        )
                        pending_close_price = None
                    else:
                        # 浮盈检验：判断是「主升浪延续」还是「波峰滞涨」
                        is_near_high = (drawdown_from_peak >= -0.015)  # 离最高点回撤不足 1.5%
                        is_trend_intact = (ma_fast_val is not None and not pd.isna(ma_fast_val) and close_price > ma_fast_val)
                        has_profit_cushion = (drawdown_from_cost >= max(0.02, active_stop_pct * 1.2))

                        if is_near_high and is_trend_intact and has_profit_cushion:
                            # 🚀【动量跃迁】：动量强劲，取消到期一刀切强平，移交收紧后的移动止盈护航续抱！
                            active_trail_pct = max(0.015, active_trail_pct * 0.7)
                            active_target_horizon = max_holding_days  # 顺延至最大兜底持仓天数
                            logger.info(
                                f"[{date.date()}] 🚀 触达目标时界({holding_days}日)但动量强劲: "
                                f"浮盈={drawdown_from_cost*100:+.2f}%, 峰值回撤={drawdown_from_peak*100:+.2f}% | "
                                f"取消强平，状态跃迁至移动止盈护航(收紧回撤至{active_trail_pct:.2%})"
                            )
                        else:
                            # 滞涨或已现回撤：波峰确实到达，获利了结离场
                            pending_close_reason = (
                                f"预测波峰到达止盈({holding_days}日/{active_target_horizon}日目标,"
                                f"浮盈{drawdown_from_cost:.2%})"
                            )
                            pending_close_price = None
                elif holding_days >= max_holding_days:
                    pending_close_reason = f"持仓超时({holding_days}日)"
                    pending_close_price = None
                elif exec_bearish == 1:
                    pending_close_reason = "模型波段走弱主动离场"
                    pending_close_price = None

                # T+N 未满足时撤回本次平仓决定，下一根 bar 再判
                if pending_close_reason and not can_exit_today:
                    logger.debug(f"[{date.date()}] ⏳ T+{profile.settlement_days} 限制，"
                                 f"平仓指令【{pending_close_reason}】顺延")
                    pending_close_reason = None
                    pending_close_price = None

                if pending_close_reason:
                    logger.info(
                        f"[{date.date()}] ⚠️ 触发平仓预警: 【{pending_close_reason}】 | "
                        f"收盘价={close_price:.3f}, 成本={avg_entry_price:.3f}, "
                        f"浮盈={drawdown_from_cost*100:+.2f}%, 峰值回撤={drawdown_from_peak*100:+.2f}%"
                    )
                else:
                    logger.debug(
                        f"[{date.date()}] 🔵 持仓第{holding_days}天: 现价={close_price:.3f}, "
                        f"浮盈={drawdown_from_cost*100:+.2f}%, 峰值回撤={drawdown_from_peak*100:+.2f}%"
                    )
            else:
                logger.debug(f"[{date.date()}] ⚪ 空仓观望 (exec_signal={exec_signal}, "
                             f"否决原因={row.get('veto_reason', '')})")

            # ----------------------------------------------------------------
            # D. 每日盯市净值结算
            # ----------------------------------------------------------------
            records.append({
                'date': date,
                'portfolio_value': cash + position_shares * close_price,
                'cash': cash,
                'shares': position_shares,
                'close': close_price,
                'exec_signal': exec_signal,
            })

        # 回测结束仍持仓则强制平仓
        if position_shares > 0 and records:
            last_date = records[-1]['date']
            last_close = records[-1]['close']
            sell_price = last_close * (1 - self.slippage_rate)
            cash, fee, trade_pnl = _sell(last_date, position_shares, sell_price, '回测结束强平', cash)
            logger.info(
                f"[{last_date.date()}] 🏁 回测期末强平: {position_shares}股 @ {sell_price:.3f} | "
                f"盈亏: {trade_pnl*100:+.2f}% | 最终总资金: {cash:,.2f}"
            )
            trades.append({
                'date': last_date, 'action': 'SELL',
                'price': round(sell_price, 4), 'shares': position_shares,
                'commission': round(fee, 2), 'pnl_pct': round(trade_pnl, 6),
                'reason': '回测结束强平', 'cash_balance': round(cash, 2),
            })
            records[-1]['portfolio_value'] = cash
            records[-1]['shares'] = 0

        logger.info(
            f"[状态机结算] 回测完成: 总交易笔数 {len([t for t in trades if t['action']=='SELL'])} 笔 | "
            f"期末总净值: {records[-1]['portfolio_value']:,.2f}"
        )

        backtest_df = pd.DataFrame(records).set_index('date')
        backtest_df['returns'] = backtest_df['portfolio_value'].pct_change().fillna(0.0)

        return backtest_df, trades

    # ------------------------------------------------------------------
    # 4. 多维度绩效指标
    # ------------------------------------------------------------------

    def evaluate_metrics(self, backtest_df: pd.DataFrame, trades: list) -> dict:
        """
        计算全套量化绩效与风险归因指标。

        参数示例 (Parameters)
        --------------------
        backtest_df : pd.DataFrame
            每日账户净值记录（含 portfolio_value, returns 等字段）
        trades : list[dict]
            交易流水列表（含 action, pnl_pct, reason 等字段）

        返回数据结构与示例 (Returns: dict)
        ----------------------------------
        字典包含以下量化指标（含数据类型及典型数值示例）:
        {
            '累计收益率':    0.1582,       # float: 整个回测期总资产收益率 (+15.82%)
            '年化收益率':    0.7240,       # float: 按 252 个交易日复利折算年化收益 (+72.40%)
            '年化波动率':    0.2135,       # float: 日收益率标准差 * sqrt(252) (21.35%)
            '夏普比率':      3.2700,       # float: (年化收益率 - 无风险利率2.5%) / 年化波动率
            '卡玛比率':     12.4400,       # float: 年化收益率 / |最大回撤| (衡量单位回撤承担的回报)
            '最大回撤':     -0.0582,       # float: 历史最高净值至低谷的最大跌幅 (-5.82%)
            '总交易笔数':    6,            # int: 完整开平仓闭环交易次数 (不含回测期末未平仓强平)
            '胜率':          0.6667,       # float: 盈利笔数 / 总交易笔数 (66.67%, 即 4 胜 2 负)
            '盈亏比':        2.4500,       # float: 总盈利金额 / 总亏损金额 (平均每亏1元能赚2.45元)
            '单笔平均盈亏':  0.0264,       # float: 每笔交易平均盈利率 (+2.64%)
            '期末总资产':    115820.50,    # float: 回测结束时账户的总资产金额 (元)
        }
        """
        returns = backtest_df['returns'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        total_days = len(backtest_df)

        if total_days < 2:
            return {'error': '数据点不足，无法计算指标'}

        final_value = backtest_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # 年化：样本过短时外推会失真（20 天赚 5% 外推成年化 87%），
        # 低于半年则不外推，直接标记为不可年化。
        # 交易日基数按市场：A股244 / 港股246 / 美股252
        ann_days = float(getattr(self.profile, 'annual_trading_days', 252) or 252)
        annualizable = total_days >= 120
        annual_return = ((1 + total_return) ** (ann_days / total_days) - 1) if annualizable else np.nan

        volatility = returns.std(ddof=1) * np.sqrt(ann_days)

        # 夏普：用日频超额收益直接年化，不再拿「复利年化收益」除以「日频年化波动」。
        # 两者量纲不一致，原写法在高收益区间会系统性高估夏普。
        risk_free = self.profile.risk_free_rate
        daily_rf = (1 + risk_free) ** (1 / ann_days) - 1
        excess = returns - daily_rf
        sharpe = (excess.mean() / excess.std(ddof=1) * np.sqrt(ann_days)
                  if excess.std(ddof=1) > 1e-12 else 0.0)

        # 索提诺：只惩罚下行波动
        downside = excess[excess < 0]
        sortino = (excess.mean() / downside.std(ddof=1) * np.sqrt(ann_days)
                   if len(downside) > 1 and downside.std(ddof=1) > 1e-12 else np.nan)

        # 最大回撤：直接用真实净值序列，避免 (1+returns).cumprod() 的浮点累积误差
        pv = backtest_df['portfolio_value']
        peak = pv.expanding().max()
        drawdowns = (pv - peak) / peak
        max_drawdown = float(drawdowns.min())

        calmar = (annual_return / abs(max_drawdown)
                  if annualizable and abs(max_drawdown) > 1e-8 else np.nan)

        # 期末强平是回测截断造成的，不计入胜率；但其盈亏仍体现在净值里
        sell_trades = [t for t in trades if t['action'] == 'SELL' and t['reason'] != '回测结束强平']
        forced = [t for t in trades if t['action'] == 'SELL' and t['reason'] == '回测结束强平']
        win_trades = [t for t in sell_trades if t['pnl_pct'] > 0]
        lose_trades = [t for t in sell_trades if t['pnl_pct'] <= 0]

        win_rate = len(win_trades) / len(sell_trades) if sell_trades else 0.0
        avg_pnl = float(np.mean([t['pnl_pct'] for t in sell_trades])) if sell_trades else 0.0

        # 盈亏比改用金额口径。原实现把各笔收益率直接相加，
        # 不同仓位规模的交易被等权对待，与实际盈亏不符。
        def _amount(t):
            return t['pnl_pct'] * t['price'] * t['shares'] / (1 + t['pnl_pct']) if t['pnl_pct'] > -1 else 0.0
        total_gain = sum(_amount(t) for t in win_trades)
        total_loss = abs(sum(_amount(t) for t in lose_trades))
        if total_loss > 1e-8:
            profit_factor = total_gain / total_loss
        elif total_gain > 0:
            profit_factor = float('inf')   # 全胜，无亏损样本
        else:
            profit_factor = float('nan')

        # 交易样本不足时，胜率/盈亏比不具统计意义
        stat_reliable = len(sell_trades) >= 20
        exposure = float((backtest_df['shares'] > 0).mean())

        return {
            '累计收益率':   total_return,
            '年化收益率':   annual_return,
            '年化波动率':   volatility,
            '夏普比率':     sharpe,
            '索提诺比率':   sortino,
            '卡玛比率':     calmar,
            '最大回撤':     max_drawdown,
            '总交易笔数':   len(sell_trades),
            '期末强平笔数': len(forced),
            '胜率':         win_rate,
            '盈亏比':       profit_factor,
            '单笔平均盈亏': avg_pnl,
            '持仓占比':     exposure,
            '期末总资产':   final_value,
            '回测天数':     total_days,
            '统计可靠':     stat_reliable,
            '可年化':       annualizable,
        }

    # ------------------------------------------------------------------
    # 5. 可视化
    # ------------------------------------------------------------------

    def plot_backtest_results(
        self,
        signal_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
        metrics: dict,
        trades: list,
        stock_code: str,
        output_dir: str,
        show_plot: bool = False,
    ):
        """
        绘制四联回测图：
          1. 价格 K 线 + 买卖信号标注
          2. 策略净值 vs. 买入持有基准
          3. 水下回撤曲线
          4. Z-Score 信号强度
        """
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(4, 1, figure=fig, height_ratios=[3, 2, 1.5, 1.5], hspace=0.35)

        ax1 = fig.add_subplot(gs[0])   # 价格 + 信号
        ax2 = fig.add_subplot(gs[1])   # 净值曲线
        ax3 = fig.add_subplot(gs[2])   # 回撤
        ax4 = fig.add_subplot(gs[3])   # Z-Score

        dates = backtest_df.index
        close = backtest_df['close']
        cs = self.profile.currency_symbol
        cur = self.profile.currency

        # ---- 子图1：价格走势 + 买卖标注 ----
        ax1.plot(dates, close, color='#4A90D9', linewidth=1.5, label='真实收盘价', zorder=2)
        ax1.fill_between(dates, close, close.min() * 0.95, alpha=0.05, color='#4A90D9')

        buy_trades  = [t for t in trades if t['action'] == 'BUY']
        sell_trades_list = [t for t in trades if t['action'] == 'SELL']

        for t in buy_trades:
            if t['date'] in backtest_df.index:
                ax1.scatter(t['date'], t['price'], marker='^', color='#E74C3C',
                            s=120, zorder=5, label='_buy')
                ax1.annotate(f"买\n{cs}{t['price']:.2f}", xy=(t['date'], t['price']),
                             xytext=(0, -22), textcoords='offset points',
                             fontsize=7, color='#E74C3C', ha='center')

        for t in sell_trades_list:
            if t['date'] in backtest_df.index:
                color = '#27AE60' if t['pnl_pct'] > 0 else '#8E44AD'
                ax1.scatter(t['date'], t['price'], marker='v', color=color,
                            s=120, zorder=5, label='_sell')
                reason_tag = t.get('reason', '')
                if '波峰' in reason_tag:
                    short_reason = '波峰止盈'
                elif '超时' in reason_tag:
                    short_reason = '持仓超时'
                elif '硬止损' in reason_tag:
                    short_reason = '硬止损'
                elif '移动止盈' in reason_tag:
                    short_reason = '移动止盈'
                elif '走弱' in reason_tag or '转空' in reason_tag:
                    short_reason = '趋势拐头'
                else:
                    short_reason = reason_tag[:6]
                ax1.annotate(f"卖({t['pnl_pct']:+.1%})\n[{short_reason}]\n{cs}{t['price']:.2f}",
                             xy=(t['date'], t['price']),
                             xytext=(0, 12), textcoords='offset points',
                             fontsize=7.5, fontweight='bold', color=color, ha='center')

        # 信号区间背景
        in_position = backtest_df['shares'] > 0
        ax1.fill_between(dates, close.min() * 0.95, close.max() * 1.02,
                         where=in_position, alpha=0.08, color='#F39C12', label='持仓区间')

        buy_patch   = mpatches.Patch(color='#E74C3C', label='买入 ▲')
        sell_patch  = mpatches.Patch(color='#27AE60', label='卖出（盈） ▼')
        sell_patch2 = mpatches.Patch(color='#8E44AD', label='卖出（亏） ▼')
        hold_patch  = mpatches.Patch(color='#F39C12', alpha=0.3, label='持仓区间')
        ax1.legend(handles=[buy_patch, sell_patch, sell_patch2, hold_patch],
                   loc='upper left', fontsize=8)
        ax1.set_title(f'{stock_code}  Kronos Pro 回测 — 样本外验证',
                      fontsize=13, fontweight='bold', pad=8)
        ax1.set_ylabel(f'价格 ({cur})', fontsize=10)
        ax1.grid(True, alpha=0.25)
        ax1.tick_params(axis='x', labelbottom=False)

        # ---- 子图2：策略净值 vs. 买入持有 ----
        strategy_cum = (1 + backtest_df['returns'].fillna(0)).cumprod()
        bh_returns   = close.pct_change().fillna(0)
        bh_cum       = (1 + bh_returns).cumprod()

        ax2.plot(dates, strategy_cum, color='#E74C3C', linewidth=2,
                 label=f'策略净值  ({metrics["累计收益率"]:+.2%})')
        ax2.plot(dates, bh_cum, color='#7F8C8D', linewidth=1.5,
                 linestyle='--', label=f'买入持有 ({(close.iloc[-1]/close.iloc[0]-1):+.2%})')
        ax2.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
        ax2.set_ylabel('累计净值', fontsize=10)
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(True, alpha=0.25)
        ax2.tick_params(axis='x', labelbottom=False)

        # 指标文本框
        mdd = metrics.get('最大回撤', 0)
        sharpe = metrics.get('夏普比率', 0)
        calmar = metrics.get('卡玛比率', 0)
        wr = metrics.get('胜率', 0)
        pf = metrics.get('盈亏比', float('nan'))
        pf_str = f"{pf:.2f}" if not np.isnan(pf) else "N/A"
        info_text = (
            f"夏普: {sharpe:.2f}  |  卡玛: {calmar:.2f}  |  MaxDD: {mdd:.2%}\n"
            f"胜率: {wr:.1%}  |  盈亏比: {pf_str}  |  "
            f"交易笔数: {metrics.get('总交易笔数', 0)}"
        )
        ax2.text(0.01, 0.03, info_text, transform=ax2.transAxes, fontsize=8.5,
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDFEFE', alpha=0.85))

        # ---- 子图3：回撤曲线 ----
        dd = (strategy_cum - strategy_cum.expanding().max()) / strategy_cum.expanding().max()
        ax3.fill_between(dates, dd, 0, alpha=0.55, color='#E74C3C', label='回撤')
        ax3.axhline(0, color='gray', linewidth=0.8)
        ax3.set_ylabel('回撤', fontsize=10)
        ax3.set_ylim(min(dd.min() * 1.2, -0.01), 0.005)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.25)
        ax3.tick_params(axis='x', labelbottom=False)

        # ---- 子图4：Z-Score 信号强度 ----
        if 'signal_score' in signal_df.columns:
            z_scores = signal_df['signal_score'].reindex(dates)
            ax4.bar(dates, z_scores.clip(-4, 4), color=np.where(z_scores > 0, '#2ECC71', '#E74C3C'),
                    alpha=0.7, width=0.8, label='Z-Score')
            ax4.axhline(0, color='gray', linewidth=0.8)
            # 默认阈值线（标注）
            ax4.axhline(1.2, color='orange', linewidth=1, linestyle='--', alpha=0.7, label='入场阈值(1.2σ)')
            ax4.set_ylabel('Z-Score', fontsize=10)
            ax4.legend(fontsize=8, loc='upper right')
            ax4.grid(True, alpha=0.2)
            ax4.set_xlabel('日期', fontsize=10)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f'{stock_code}_backtest_pro.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"📊 图表已保存: {out_path}")

        if show_plot:
            plt.show()

        return fig

    def plot_backtest_interactive(
        self,
        signal_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
        metrics: dict,
        trades: list,
        stock_code: str,
        z_score_threshold: float = 1.2,
        crosshair_mode: str = "x",
    ):
        """
        绘制专业交互式四联回测图（Plotly 引擎，支持 4 轴联动、滚轮缩放、平移、悬停详情）：
          1. 价格曲线 + 买入/卖出信号点 + 持仓区间遮罩
          2. 策略净值 vs. 买入持有基准
          3. 水下动态回撤曲线 (Underwater Drawdown)
          4. Z-Score 信号强度柱状图 (红绿柱 + 入场阈值线)
        """
        if not HAS_PLOTLY or backtest_df is None or backtest_df.empty:
            return None

        dates = backtest_df.index
        close = backtest_df['close']
        cs = self.profile.currency_symbol
        cur = self.profile.currency

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            row_heights=[0.38, 0.24, 0.18, 0.20],
            subplot_titles=[
                f"<b>{stock_code} 价格走势与买卖信号执行 ({cur})</b>",
                "<b>策略累计净值 vs 买入持有基准</b>",
                "<b>水下动态回撤 (Underwater Drawdown)</b>",
                "<b>Z-Score 预测信号强度 (动量驱动)</b>"
            ]
        )

        # ---- 子图1：价格走势 + 买卖标注 + 持仓区间 ----
        # 1. 价格折线
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=close,
                mode='lines',
                name='真实收盘价',
                line=dict(color='#2980B9', width=2),
                hovertemplate="<b>日期</b>: %{x|%Y-%m-%d}<br><b>收盘价</b>: " + cs + "%{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )

        # 2. 买入信号
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        if buy_trades:
            b_dates = [t['date'] for t in buy_trades]
            b_prices = [t['price'] for t in buy_trades]
            b_texts = [f"买<br>{cs}{t['price']:.2f}" for t in buy_trades]
            b_hovers = [
                f"<b>【买入开仓】</b><br>日期: {t['date'].strftime('%Y-%m-%d')}<br>成交价: {cs}{t['price']:.2f}<br>买入股数: {t.get('shares', 0)} 股<br>目标时界: {t.get('target_horizon', 'N/A')} 天"
                for t in buy_trades
            ]
            fig.add_trace(
                go.Scatter(
                    x=b_dates,
                    y=b_prices,
                    mode='markers+text',
                    name='买入信号 ▲',
                    marker=dict(symbol='triangle-up', size=13, color='#E74C3C', line=dict(width=1, color='white')),
                    text=b_texts,
                    textposition="bottom center",
                    textfont=dict(size=9, color='#C0392B'),
                    hovertext=b_hovers,
                    hoverinfo="text"
                ),
                row=1, col=1
            )

        # 3. 卖出信号
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        if sell_trades:
            s_dates = [t['date'] for t in sell_trades]
            s_prices = [t['price'] for t in sell_trades]
            s_colors = ['#27AE60' if t['pnl_pct'] > 0 else '#8E44AD' for t in sell_trades]
            s_texts = []
            s_hovers = []
            for t in sell_trades:
                reason_tag = t.get('reason', '')
                if '波峰' in reason_tag:
                    short_reason = '波峰止盈'
                elif '超时' in reason_tag:
                    short_reason = '持仓超时'
                elif '硬止损' in reason_tag:
                    short_reason = '硬止损'
                elif '移动止盈' in reason_tag:
                    short_reason = '移动止盈'
                elif '走弱' in reason_tag or '转空' in reason_tag:
                    short_reason = '趋势拐头'
                else:
                    short_reason = reason_tag[:6]
                s_texts.append(f"卖({t['pnl_pct']:+.1%})<br>[{short_reason}]<br>{cs}{t['price']:.2f}")
                s_hovers.append(
                    f"<b>【卖出平仓】</b><br>"
                    f"日期: {t['date'].strftime('%Y-%m-%d')}<br>"
                    f"平仓价: {cs}{t['price']:.2f}<br>"
                    f"单笔盈亏: {t['pnl_pct']:+.2%}<br>"
                    f"触发原因: {t.get('reason', 'N/A')}<br>"
                    f"持仓股数: {t.get('shares', 0)} 股<br>"
                    f"账户资金: {cs}{t.get('cash_balance', 0):,.2f}"
                )
            fig.add_trace(
                go.Scatter(
                    x=s_dates,
                    y=s_prices,
                    mode='markers+text',
                    name='卖出平仓 ▼',
                    marker=dict(symbol='triangle-down', size=13, color=s_colors, line=dict(width=1, color='white')),
                    text=s_texts,
                    textposition="top center",
                    textfont=dict(size=9, color='#2C3E50'),
                    hovertext=s_hovers,
                    hoverinfo="text"
                ),
                row=1, col=1
            )

        # 4. 持仓区间遮罩
        if 'shares' in backtest_df.columns:
            in_pos = (backtest_df['shares'] > 0).astype(int)
            diff = in_pos.diff().fillna(in_pos.iloc[0])
            starts = backtest_df.index[diff == 1].tolist()
            ends = backtest_df.index[diff == -1].tolist()
            if len(in_pos) > 0 and in_pos.iloc[0] == 1 and (not starts or starts[0] > backtest_df.index[0]):
                starts.insert(0, backtest_df.index[0])
            if len(in_pos) > 0 and in_pos.iloc[-1] == 1:
                ends.append(backtest_df.index[-1])
            for s_date, e_date in zip(starts, ends):
                fig.add_vrect(
                    x0=s_date,
                    x1=e_date,
                    fillcolor="#F39C12",
                    opacity=0.12,
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )

        # ---- 子图2：策略净值 vs. 买入持有 ----
        strategy_cum = (1 + backtest_df['returns'].fillna(0)).cumprod()
        bh_returns = close.pct_change().fillna(0)
        bh_cum = (1 + bh_returns).cumprod()

        strat_ret = metrics.get('累计收益率', 0)
        bh_ret = (close.iloc[-1] / close.iloc[0] - 1) if len(close) > 0 else 0

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=strategy_cum,
                mode='lines',
                name=f"策略净值 ({strat_ret:+.2%})",
                line=dict(color='#E74C3C', width=2.4),
                hovertemplate="<b>日期</b>: %{x|%Y-%m-%d}<br><b>策略累计净值</b>: %{y:.3f}<extra></extra>"
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bh_cum,
                mode='lines',
                name=f"买入持有 ({bh_ret:+.2%})",
                line=dict(color='#7F8C8D', width=1.6, dash='dash'),
                hovertemplate="<b>日期</b>: %{x|%Y-%m-%d}<br><b>基准累计净值</b>: %{y:.3f}<extra></extra>"
            ),
            row=2, col=1
        )
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", line_width=1, row=2, col=1)

        mdd = metrics.get('最大回撤', 0)
        sharpe = metrics.get('夏普比率', 0)
        calmar = metrics.get('卡玛比率', 0)
        wr = metrics.get('胜率', 0)
        info_text = f"夏普: {sharpe:.2f} | 卡玛: {calmar:.2f} | MaxDD: {mdd:.2%} | 胜率: {wr:.1%}"
        fig.add_annotation(
            xref="x2 domain", yref="y2 domain",
            x=0.01, y=0.05,
            text=info_text,
            showarrow=False,
            font=dict(size=11, color="#2C3E50"),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(180, 180, 180, 0.6)",
            borderwidth=1,
            row=2, col=1
        )

        # ---- 子图3：水下回撤曲线 ----
        dd = (strategy_cum - strategy_cum.expanding().max()) / strategy_cum.expanding().max()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=dd,
                mode='lines',
                fill='tozeroy',
                name='水下回撤',
                line=dict(color='#E74C3C', width=1.2),
                fillcolor='rgba(231, 76, 60, 0.35)',
                hovertemplate="<b>日期</b>: %{x|%Y-%m-%d}<br><b>回撤幅度</b>: %{y:.2%}<extra></extra>"
            ),
            row=3, col=1
        )
        fig.add_hline(y=0.0, line_color="gray", line_width=0.8, row=3, col=1)

        # ---- 子图4：Z-Score 信号强度 ----
        if 'signal_score' in signal_df.columns:
            z_scores = signal_df['signal_score'].reindex(dates).clip(-4, 4)
            bar_colors = [
                'rgba(46, 204, 113, 0.75)' if val >= 0 else 'rgba(231, 76, 60, 0.75)'
                for val in z_scores
            ]
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=z_scores,
                    name='Z-Score 信号强度',
                    marker_color=bar_colors,
                    hovertemplate="<b>日期</b>: %{x|%Y-%m-%d}<br><b>Z-Score</b>: %{y:.2f}σ<extra></extra>"
                ),
                row=4, col=1
            )
            fig.add_hline(y=0.0, line_color="gray", line_width=0.8, row=4, col=1)
            fig.add_hline(
                y=z_score_threshold,
                line_dash="dash",
                line_color="#E67E22",
                line_width=1.2,
                annotation_text=f"入场阈值 (+{z_score_threshold}σ)",
                annotation_position="top right",
                row=4, col=1
            )

        # 全局布局与样式
        fig.update_layout(
            height=980,
            hovermode=crosshair_mode,
            spikedistance=-1,    # 全图任意垂直高度吸附触发时间线
            hoverdistance=100,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="rgba(200, 200, 200, 0.5)",
                borderwidth=1
            ),
            margin=dict(l=60, r=40, t=50, b=40),
            plot_bgcolor="rgba(250, 250, 250, 0.8)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        # 开启 4 轴完美拉通对齐的十字光标时间指示线 (Spikeline)
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(220, 224, 230, 0.6)",
            showspikes=True,
            spikemode="across",        # 垂直穿透整个子图绘图区
            spikesnap="hovered data",  # 统一吸附到当前日期，让 4 个子图垂直时间线绝对共振对齐
            spikethickness=1.2,
            spikedash="dash",          # 虚线标尺
            spikecolor="#34495E"       # 显眼的深灰蓝时间标尺线
        )
        fig.update_yaxes(showgrid=True, gridcolor="rgba(220, 224, 230, 0.6)")
        fig.update_yaxes(title_text=f"价格 ({cur})", row=1, col=1)
        fig.update_yaxes(title_text="累计净值", row=2, col=1)
        fig.update_yaxes(title_text="回撤", tickformat=".1%", row=3, col=1)
        fig.update_yaxes(title_text="Z-Score", row=4, col=1)

        return fig

    # ------------------------------------------------------------------
    # 6. 全流程管线
    # ------------------------------------------------------------------

    def run_complete_backtest(
        self,
        stock_code: str,
        output_dir: str,
        z_score_threshold: float = 1.2,
        volatility_window: int = 14,
        hard_stop_loss: float = 0.04,
        trailing_stop: float = 0.025,
        max_holding_days: int = 5,
        show_plot: bool = False,
        entry_percentile: float = 0.80,
        calib_window: int = 60,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.80,
        min_lot_guarantee: bool = True,
        intraday_stop: bool = True,
        require_skill: bool = False,
        market: str = None,
        forecast_horizon: str = None,
        max_auto_horizon: int = 10,
        **kwargs,
    ) -> tuple:
        """
        一键完整回测管线：市场规则 → 加载 → 技能体检 → 对齐/信号 → 撮合 → 指标 → 绘图

        require_skill=True 时，预测技能体检未通过将直接中止回测，
        避免在没有预测能力的标的上继续调参。

        返回: (metrics, signal_df, backtest_df, trades, fig)
        """
        print(f"\n{'='*60}")
        print(f"🎯  Kronos Pro 回测引擎  —  {stock_code}")
        print(f"{'='*60}")

        try:
            print("\n[Step 1] 设定市场规则 & 加载数据")
            # 依股票代码切换市场规则（整手、印花税、T+N、涨跌停、结算币种）
            self.set_market(stock_code, market=market)
            print(f"   市场规则: {self.profile.describe()}")

            hist_df, pred_df = self.load_data(stock_code)

            print("\n[Step 2] 时序对齐 & 信号生成")
            signal_df = self.align_and_generate_signals(
                hist_df, pred_df,
                z_score_threshold=z_score_threshold,
                volatility_window=volatility_window,
                forecast_horizon=forecast_horizon,
                hard_stop_loss=hard_stop_loss,
                entry_percentile=entry_percentile,
                calib_window=calib_window,
                max_auto_horizon=max_auto_horizon,
            )

            print("\n[Step 2b] 预测技能体检")
            skill = self.evaluate_prediction_skill(signal_df, hard_stop_loss=hard_stop_loss)
            self.print_skill_report(skill, stock_code)
            if require_skill and not skill.get('tradable', False):
                reason = skill.get('error') or '; '.join(skill.get('不通过原因', []))
                raise ValueError(
                    f"\n[准入拦截] {stock_code} 未通过预测技能体检，已中止回测。\n"
                    f"  原因: {reason}\n"
                    f"  说明: 预测无排序能力时，任何参数组合的正收益都只是路径运气。\n"
                    f"  如需强行回测，请设置 require_skill=False。"
                )

            print("\n[Step 3] 逐日撮合状态机")
            backtest_df, trades = self.run_backtest_engine(
                signal_df,
                hard_stop_loss=hard_stop_loss,
                trailing_stop=trailing_stop,
                max_holding_days=max_holding_days,
                min_lot_guarantee=min_lot_guarantee,
                risk_per_trade=risk_per_trade,
                max_position_pct=max_position_pct,
                intraday_stop=intraday_stop,
            )

            print("\n[Step 4] 绩效指标计算")
            metrics = self.evaluate_metrics(backtest_df, trades)
            metrics['_skill'] = skill

            print("\n[Step 5] 生成图表")
            fig = self.plot_backtest_results(
                signal_df, backtest_df, metrics, trades,
                stock_code, output_dir, show_plot=show_plot,
            )

            self._print_report(stock_code, metrics, trades)
            return metrics, signal_df, backtest_df, trades, fig

        except ValueError as e:
            print(str(e))
            return None, None, None, None, None
        except Exception as e:
            import traceback
            print(f"\n❌ 回测异常: {e}")
            traceback.print_exc()
            return None, None, None, None, None

    def _print_report(self, stock_code: str, metrics: dict, trades: list):
        """控制台输出格式化回测报告"""
        print(f"\n{'='*60}")
        print(f"📊  {stock_code} 回测报告  [样本外 Backtest 模式]")
        print(f"{'='*60}")
        fmt_pct = {'累计收益率', '年化收益率', '年化波动率', '最大回撤', '胜率',
                   '单笔平均盈亏', '持仓占比'}
        cur = self.profile.currency
        cs = self.profile.currency_symbol
        for k, v in metrics.items():
            if k.startswith('_'):
                continue
            if isinstance(v, bool):
                print(f"  {k:<12}: {'是' if v else '否'}")
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                # 样本不足时年化/卡玛等指标本就不该给数，显式标注而非填 0
                label = '不适用' if k in ('年化收益率', '卡玛比率') and not metrics.get('可年化', True) else 'N/A'
                if isinstance(v, float) and v == float('inf'):
                    label = '∞ (无亏损样本)'
                print(f"  {k:<12}: {label}")
            elif k in fmt_pct:
                print(f"  {k:<12}: {v:>+.2%}")
            elif k in ('总交易笔数', '期末强平笔数', '回测天数'):
                print(f"  {k:<12}: {int(v)}")
            elif k == '期末总资产':
                print(f"  {k:<12}: {cur} {v:>,.2f}")
            else:
                print(f"  {k:<12}: {v:.4f}")

        if not metrics.get('统计可靠', True):
            print(f"\n  ⚠️ 交易样本仅 {metrics.get('总交易笔数', 0)} 笔（<20），"
                  f"胜率与盈亏比不具统计意义，不可作为参数选择依据。")
        if not metrics.get('可年化', True):
            print(f"  ⚠️ 回测仅 {metrics.get('回测天数', 0)} 天（<120），"
                  f"年化与卡玛比率已标记为不适用，短样本外推会严重失真。")

        sell_t = [t for t in trades if t['action'] == 'SELL']
        if sell_t:
            print(f"\n  平仓原因分布:")
            reason_counts = {}
            for t in sell_t:
                reason_counts[t['reason']] = reason_counts.get(t['reason'], 0) + 1
            for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {cnt} 次")

        print(f"\n  近期交易流水 (最多10条):")
        for t in trades[-10:]:
            pnl_str = f"  盈亏: {t['pnl_pct']:+.2%}" if t['action'] == 'SELL' else ""
            print(f"    {t['date'].strftime('%Y-%m-%d')}  "
                  f"{t['action']:<4} {t['shares']:>6}股 @ {cs}{t['price']:.3f}"
                  f"{pnl_str}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def get_available_stocks(data_dir: str, model_dir: str) -> list:
    """
    扫描 data_dir 和 model_dir，返回同时拥有历史数据和预测文件的股票代码列表。
    优先返回两侧均有文件的标的（交集），若为空则返回并集。
    全面支持 A股 (如 300418)、港股 (如 HK_00700/00700)、美股 (如 US_AAPL/AAPL)。
    """
    def extract_codes(directory: str, suffixes: list) -> set:
        codes = set()
        if not os.path.isdir(directory):
            return codes
        for fname in os.listdir(directory):
            for suffix in suffixes:
                if fname.endswith(suffix):
                    raw_stem = fname[:-len(suffix)]
                    clean_code, _, _ = SymbolResolver.resolve(raw_stem)
                    if clean_code:
                        codes.add(clean_code)
        return codes

    hist_codes = extract_codes(
        data_dir,
        ['_stock_data.csv', '_all_history.csv']
    )
    pred_codes = extract_codes(
        model_dir,
        ['_wf_predictions_auto.csv', '_wf_predictions.csv',
         '_kronos_predictions.csv', '_detailed_predictions.csv',
         '_comprehensive_predictions.csv', '_streamlit_predictions.csv',
         '_predictions.csv']
    )

    both = sorted(hist_codes & pred_codes)
    return both if both else sorted(hist_codes | pred_codes) or ['300418']


def inject_continuous_crosshair_script():
    """
    注入全图贯通的垂直十字光标时间标尺 (Continuous Crosshair Timeline across 4 subplots)
    使 4 联子图在鼠标悬停移动时共用一条无断点的垂直虚线参考线，贯穿全部 4 个子图。
    """
    import streamlit as st
    html_code = """
    <script>
    (function() {
        const doc = window.parent.document || document;
        
        function setupCrosshair() {
            const chartDivs = doc.querySelectorAll('.stPlotlyChart');
            chartDivs.forEach(chartContainer => {
                if (chartContainer.dataset.crosshairInit === 'true') return;
                
                const plotlyEl = chartContainer.querySelector('.js-plotly-plot');
                if (!plotlyEl) return;
                
                chartContainer.dataset.crosshairInit = 'true';
                chartContainer.style.position = 'relative';

                // 创建全局贯通的垂直时间虚线标尺
                let vline = chartContainer.querySelector('.quant-crosshair-vline');
                if (!vline) {
                    vline = doc.createElement('div');
                    vline.className = 'quant-crosshair-vline';
                    vline.style.position = 'absolute';
                    vline.style.top = '30px';
                    vline.style.bottom = '35px';
                    vline.style.width = '0px';
                    vline.style.borderLeft = '1.5px dashed #E74C3C';
                    vline.style.pointerEvents = 'none';
                    vline.style.zIndex = '9999';
                    vline.style.display = 'none';
                    vline.style.boxShadow = '0 0 3px rgba(231, 76, 60, 0.4)';
                    chartContainer.appendChild(vline);
                }

                // 监听鼠标在图表区域内的移动
                chartContainer.addEventListener('mousemove', function(e) {
                    const rect = chartContainer.getBoundingClientRect();
                    const mouseX = e.clientX - rect.left;
                    
                    // 尝试从 Plotly 内部 SVG 捕捉精准吸附的 xspikeline 坐标
                    let targetX = mouseX;
                    const spike = chartContainer.querySelector('line.xspikeline, line.spikeline');
                    if (spike) {
                        const sx = parseFloat(spike.getAttribute('x1'));
                        if (!isNaN(sx) && sx > 0) {
                            targetX = sx;
                        }
                    }

                    // 绘图区域边界检测 (留出左 Y 轴刻度 58px 与 右边距 35px)
                    const minX = 58;
                    const maxX = rect.width - 35;
                    if (targetX >= minX && targetX <= maxX) {
                        vline.style.left = targetX + 'px';
                        vline.style.display = 'block';
                    } else {
                        vline.style.display = 'none';
                    }
                });

                chartContainer.addEventListener('mouseleave', function() {
                    vline.style.display = 'none';
                });
            });
        }

        setupCrosshair();
        if (!window._quantCrosshairTimer) {
            window._quantCrosshairTimer = setInterval(setupCrosshair, 400);
        }
    })();
    </script>
    """
    try:
        st.html(html_code, unsafe_allow_javascript=True)
    except Exception:
        try:
            import streamlit.components.v1 as components
            components.html(html_code, height=0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Streamlit 看板
# ---------------------------------------------------------------------------

def streamlit_main():
    """Streamlit 交互回测看板（工业级 Pro 版）"""
    import streamlit as st
    import inspect

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
    output_dir = os.path.join(SCRIPT_DIR, "output")

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
                # 动态精度提示
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
            # 自动数据量校验安全提示
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
                  "调低 = 交易更频繁。不同标的的预测量纲差异极大"
                  "（同一模型下 300418 预测均值 +6.4%，603288 仅 +0.01%），"
                  "分位数阈值能自动适配，固定 Z 值不能。")
        )
        calib_window = st.slider(
            "滚动校准窗口 (交易日)", min_value=20, max_value=120,
            value=60, step=10,
            help="分位数阈值与预测去偏所用的回看窗口"
        )
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

        run_btn = st.button("🚀 运行 Pro 回测", type="primary", **_stretch_kw)

    # ---------- 执行回测 ----------
    run_key = (selected_stock, is_auto_mode, initial_capital, z_threshold, vol_window,
               hard_stop, trail_stop, max_hold, commission_bps, slippage_bps,
               min_lot_guarantee, require_skill, entry_percentile, calib_window,
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
    )

    # 智能执行与预加载决策：
    # 1. 用户点击「🚀 运行 Pro 回测」：按当前配置强制执行
    # 2. 用户未点击，但当前 Session 尚无回测数据或切换了股票标的：自动尝试秒级加载，消除白屏停滞
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
        # 如果是 Auto 模式且命中预测缓存，或者 Manual 模式，则自动快速运行（<0.3秒）
        if (is_auto_mode and has_wf_cache) or (not is_auto_mode):
            should_execute = True
            is_auto_preloading = True

    if should_execute:
        if is_auto_mode:
            # ---- AUTO 模式：拉取行情 + Walk-Forward 预测 + 回测 ----
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
                        import traceback
                        st.code(traceback.format_exc())
                        st.stop()
        else:
            # ---- MANUAL 模式：使用现有预测文件 ----
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

    # 提示参数修改状态或自动载入状态
    data       = st.session_state["bt_data"]
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

    # 兼容历史 session：若尚未生成交互图但支持 Plotly，则即时生成
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

        # 直接在图表下方呈现全量交易执行记录与平仓原因
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
            display_sig = signal_df[['close', 'pred_close', 'expected_return',
                                      'volatility', 'signal_score',
                                      'raw_signal', 'exec_signal']].copy()
            display_sig.columns = ['真实收盘价', '预测价', '预期收益率', '历史波动率',
                                    'Z-Score', '原始信号(t日)', '执行信号(t+1日)']
            st.dataframe(
                display_sig.style.format({
                    '真实收盘价': '{:.3f}',
                    '预测价': '{:.3f}',
                    '预期收益率': '{:+.3%}',
                    '历史波动率': '{:.4f}',
                    'Z-Score': '{:+.3f}',
                }),
                width="stretch"
            )
            st.caption(
                "注：**原始信号** 在 t 日收盘后生成；"
                "**执行信号** = 原始信号向后移动1天，消除未来函数偷价。"
            )

    with tab_trades:
        if trades:
            sub_tab_flow, sub_tab_roundtrip = st.tabs([
                "📋 逐笔成交流水 (买入+卖出全量对账)", "🎯 交易轮次闭环盈亏 (Round-Trip)"
            ])

            with sub_tab_flow:
                col_f1, col_f2 = st.columns([3, 5])
                with col_f1:
                    filter_opt = st.selectbox(
                        "流水方向筛选",
                        options=["全部流水 (买入 + 卖出)", "仅买入建仓 (BUY)", "仅卖出平仓 (SELL)"],
                        index=0
                    )

                full_rows = []
                for idx, t in enumerate(trades, 1):
                    act = t['action']
                    if "仅买入" in filter_opt and act != 'BUY':
                        continue
                    if "仅卖出" in filter_opt and act != 'SELL':
                        continue

                    act_label = "🟢 买入建仓" if act == 'BUY' else "🔴 卖出平仓"
                    amt = t['price'] * t['shares']
                    pnl_str = f"{t['pnl_pct']:+.2%}" if act == 'SELL' else "-"

                    full_rows.append({
                        '序号': idx,
                        '成交日期': t['date'].strftime('%Y-%m-%d'),
                        '方向': act_label,
                        '成交均价': f"{cs}{t['price']:.3f}",
                        '成交股数': f"{t['shares']:,}",
                        '成交金额': f"{cs}{amt:,.2f}",
                        '手续税费': f"{cs}{t['commission']:.2f}",
                        '单笔盈亏率': pnl_str,
                        '触发原因': t.get('reason', 'N/A'),
                        '剩余可用资金': f"{cs}{t['cash_balance']:,.2f}",
                    })

                if full_rows:
                    st.dataframe(pd.DataFrame(full_rows), width="stretch")
                else:
                    st.info("当前筛选条件下无流水记录", icon="ℹ️")

            with sub_tab_roundtrip:
                # 智能配对 BUY 与对应平仓的 SELL
                round_trips = []
                current_buy = None
                for t in trades:
                    if t['action'] == 'BUY':
                        current_buy = t
                    elif t['action'] == 'SELL' and current_buy is not None:
                        buy_amt = current_buy['price'] * current_buy['shares']
                        sell_amt = t['price'] * t['shares']
                        total_fees = current_buy['commission'] + t['commission']
                        net_profit_yuan = sell_amt - buy_amt - total_fees
                        hold_days = (t['date'] - current_buy['date']).days

                        target_h = current_buy.get('target_horizon', 'N/A')
                        h_disp = f"{target_h} 天" if target_h != 'N/A' and target_h is not None else "-"
                        round_trips.append({
                            '交易轮次': f"第 {len(round_trips)+1} 笔",
                            '买入日期': current_buy['date'].strftime('%Y-%m-%d'),
                            '买入价': f"{cs}{current_buy['price']:.3f}",
                            '平仓日期': t['date'].strftime('%Y-%m-%d'),
                            '平仓价': f"{cs}{t['price']:.3f}",
                            '持股数': f"{t['shares']:,}",
                            '动态目标时界': h_disp,
                            '持仓自然日': f"{hold_days} 天",
                            '单笔盈亏率': f"{t['pnl_pct']:+.2%}",
                            f'净收益 ({cur})': f"{cs}{net_profit_yuan:+,.2f}",
                            '平仓触发原因': t.get('reason', 'N/A'),
                        })
                        current_buy = None

                if round_trips:
                    rt_df = pd.DataFrame(round_trips)
                    st.dataframe(rt_df, width="stretch")
                else:
                    st.info("暂无可配对的完整开平仓闭环记录", icon="ℹ️")
        else:
            st.info("策略在当前参数下未触发任何交易记录", icon="ℹ️")

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
        log_file = os.path.join(output_dir, "kronos_backtest.log")
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                st.caption(f"📁 日志文件落盘位置: `{log_file}` (实时显示最新 150 行)")
                st.code("".join(lines[-150:]), language="log")
                if st.button("🗑️ 清空后台运行日志", key="clear_log_btn"):
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write("")
                    st.rerun()
            except Exception as e:
                st.warning(f"读取日志失败: {e}")
        else:
            st.info("暂无运行日志记录")


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
                        help="分位阈值与去偏的滚动校准窗口，默认 60 交易日")
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

    args = parser.parse_args()
    target_stock = args.stock_code

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
    print()

    data_dir    = resolve_data_dir()
    pred_dir    = resolve_pred_dir()
    backtest_dir= os.path.join(SCRIPT_DIR, "output")

    hist_df = None
    pred_df = None

    # ----------------------------------------------------------------
    # AUTO 模式：自动拉取行情 + Walk-Forward 预测
    # ----------------------------------------------------------------
    if args.auto:
        h_desc = f"Auto(最大推演{args.max_auto_horizon}天)" if horizon_val == 'auto' else f"{horizon_val}天"
        print(f"[AUTO] 参数：horizon={h_desc}, test_days={args.test_days}, lookback={args.lookback}, "
              f"step_size={args.step_size}, samples={args.samples}, "
              f"history_days={args.history_days}")
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
    )

    if args.auto and hist_df is not None and pred_df is not None:
        # AUTO 模式：直接用已获取的数据，跳过文件加载
        backtester.set_market(target_stock, market=args.market)
        print(f"   市场规则: {backtester.profile.describe()}")

        print("\n[Step 2] 时序对齐 & 信号生成")
        signal_df = backtester.align_and_generate_signals(
            hist_df, pred_df,
            z_score_threshold=args.z_threshold,
            forecast_horizon=horizon_val,
            max_auto_horizon=args.max_auto_horizon if hasattr(args, 'max_auto_horizon') else 10,
            hard_stop_loss=args.hard_stop,
            entry_percentile=args.entry_percentile,
            calib_window=args.calib_window,
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
        backtester._print_report(target_stock, metrics, trades)
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
            risk_per_trade=args.risk_per_trade,
            max_position_pct=args.max_position,
            min_lot_guarantee=not args.no_min_lot_guarantee,
            intraday_stop=not args.no_intraday_stop,
            require_skill=args.require_skill,
            market=args.market,
            forecast_horizon=horizon_val,
            max_auto_horizon=getattr(args, 'max_auto_horizon', 10),
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

