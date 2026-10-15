"""
backtest/data/kronos_predictor.py
Kronos AI 预测器与 Walk-Forward 滚动前向推演管线
支持蒙特卡洛多路径采样、特征解耦、粒子滤波重分配与缓存参数指纹防污染。
"""

from __future__ import annotations

import os
import sys
import json
import logging
import threading
from datetime import datetime, date
from typing import Union, Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd

from ..core.market_rules import SymbolResolver
from .cache import CacheManager
from .data_fetcher import DataFetcher, _import_data_utils

# 全局模型推演线程锁：防止 Apple Silicon (MPS) 或多线程环境下并发调用 PyTorch 模型导致的 Metal 驱动断言冲突
_INFERENCE_LOCK = threading.Lock()

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, 'examples')

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
        # 硬件加速设备优先级：CUDA > Apple Silicon MPS (Metal) > CPU
        if device is None or device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device
        self._max_context = max_context

    def _ensure_predictor(self):
        """懒加载：首次调用时才初始化模型，避免仅做数据检查时浪费显存"""
        if self._predictor is not None:
            return
        print(f"\n🧠 正在加载 Kronos 模型 (目标设备: {self._device}，首次使用请稍候)...")
        # 动态导入，避免在不使用 --auto 时强制要求模型依赖
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from model import Kronos, KronosTokenizer, KronosPredictor
            tokenizer = KronosTokenizer.from_pretrained(self._tokenizer_name)
            model = Kronos.from_pretrained(self._model_name)
            try:
                self._predictor = KronosPredictor(model, tokenizer,
                                                   device=self._device,
                                                   max_context=self._max_context)
            except Exception as e_dev:
                if self._device == "mps":
                    print(f"⚠️ Apple Silicon MPS 设备初始化异常 ({e_dev})，自动降级回退至 CPU...")
                    self._device = "cpu"
                    self._predictor = KronosPredictor(model, tokenizer,
                                                       device="cpu",
                                                       max_context=self._max_context)
                else:
                    raise
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
                         is_auto_horizon, eval_horizon,
                         weights=None, val_error=0.0):
        """
        把逐路径采样结果汇总为单日预测指标（含粒子滤波动态加权与无偏期望值修正）。

        参数
        ----
        paths : dict[str, np.ndarray]
            _sample_paths 的输出，每项形状 (n_samples, pred_len)。
        min_down_quantile : float
            最大下探的保守分位，例如 0.10
        prev_real_close : float
            决策时点已知的最后一个真实收盘价（预测基准）
        is_auto_horizon : bool
            是否为极值自适应模式
        eval_horizon : int
            向前推演的最大步长
        weights : np.ndarray, optional
            粒子滤波动态权重向量 (n_samples,)，默认等权 1/N
        val_error : float, optional
            粒子在验证窗口上的平均加权跟踪误差 (NRMSE)

        返回
        ----
        dict，含 pred_expected / pred_path_max / optimal_horizon / pred_min_down / pf_confidence 等
        """
        close_arr = paths['close']                       # (n_samples, pred_len)
        n_samples = close_arr.shape[0]

        # ---- 粒子滤波权重归一化与有效样本数 (ESS) ----
        if weights is None:
            w_arr = np.ones(n_samples, dtype=float) / float(n_samples)
        else:
            w_arr = np.asarray(weights, dtype=float)
            w_sum = float(np.sum(w_arr))
            w_arr = w_arr / w_sum if w_sum > 0 else (np.ones(n_samples, dtype=float) / float(n_samples))

        sum_w_sq = float(np.sum(w_arr ** 2))
        ess = float(1.0 / sum_w_sq) if sum_w_sq > 0 else float(n_samples)
        pf_confidence = float(np.clip(ess / float(n_samples), 0.0, 1.0))

        # 辅助加权分位数（稳健平滑插值）
        def _w_quantile(vals, q):
            v = np.asarray(vals, dtype=float)
            idx = np.argsort(v)
            sorted_v = v[idx]
            sorted_w = w_arr[idx]
            cum_w = np.cumsum(sorted_w)
            cum_w = cum_w / cum_w[-1]
            return float(np.interp(q, cum_w, sorted_v))

        # 粒子加权期望价格路径 (Weighted Mean Path)
        mean_path = np.sum(close_arr * w_arr[:, None], axis=0)  # 各时点的期望价格

        # ---- 止盈目标：逐路径计算极值并进行加权中位数统计 ----
        per_path_peaks = np.max(close_arr, axis=1)       # (n_samples,) 逐路径最高价
        per_path_peak_idx = np.argmax(close_arr, axis=1) # (n_samples,) 逐路径最高点发生时间

        # 稳健加权波峰目标：取加权中位数
        pred_path_max = _w_quantile(per_path_peaks, 0.5)

        # 最佳时界：取加权中位数 + 1
        pred_len = close_arr.shape[1]
        if is_auto_horizon:
            peak_idx = int(round(_w_quantile(per_path_peak_idx, 0.5)))
            opt_horizon = peak_idx + 1
        else:
            peak_idx = int(np.argmax(mean_path))
            opt_horizon = eval_horizon

        # ---- 入场判定：动态最优时界当天的跨路径加权期望价格 ----
        # 严格对齐持仓时界：取持有至 opt_horizon 当天的跨路径期望收盘价，消除时界错配与全区间均值带来的形态污染
        target_idx = min(max(0, opt_horizon - 1), pred_len - 1)
        pred_expected = float(mean_path[target_idx])
        pred_terminal = float(mean_path[-1])

        # ---- 加权分位数 ----
        safe_peak_idx = min(max(0, peak_idx), pred_len - 1)
        enough_for_quantile = n_samples >= 5
        if enough_for_quantile:
            upper_val = _w_quantile(close_arr[:, safe_peak_idx], 0.90)
            lower_val = _w_quantile(close_arr[:, safe_peak_idx], 0.10)
        else:
            upper_val = float(np.max(close_arr[:, safe_peak_idx]))
            lower_val = float(np.min(close_arr[:, safe_peak_idx]))

        # ---- 最大下探：加权保守分位 ----
        low_arr = paths.get('low', close_arr)
        seg = low_arr[:, :opt_horizon] if opt_horizon >= 1 else low_arr
        per_path_min = np.min(seg, axis=1)               # (n_samples,) 逐路径最低价
        if n_samples >= 5:
            min_low_val = _w_quantile(per_path_min, min_down_quantile)
        else:
            min_low_val = float(np.min(per_path_min))

        min_down_pct = ((min_low_val - prev_real_close) / prev_real_close
                        if prev_real_close > 0 else 0.0)

        # ---- 斜率：基于加权期望拟合客观趋势 ----
        full_len = len(mean_path)
        if full_len > 1:
            slope_val = float(np.polyfit(np.arange(full_len), mean_path, 1)[0])
            slope_pct = slope_val / prev_real_close if prev_real_close > 0 else 0.0
        else:
            slope_pct = ((pred_expected - prev_real_close) / prev_real_close
                         if prev_real_close > 0 else 0.0)

        # 波峰溢价
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
            'pf_confidence': pf_confidence,
            'pf_val_error': float(val_error),
            'pf_effective_n': float(ess),
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
            elif cs in ('粒子滤波置信度', 'pf_confidence'):
                _claim(c, 'pf_confidence')
            elif cs in ('粒子滤波验证误差', 'pf_val_error'):
                _claim(c, 'pf_val_error')
            elif cs in ('有效粒子数', 'pf_effective_n'):
                _claim(c, 'pf_effective_n')
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
                    'n_samples', 'is_interpolated', 'pf_confidence', 'pf_val_error',
                    'pf_effective_n'):
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
        particle_filter: bool = False,
        pf_val_window: int = 3,
        pf_temperature: float = None,
        market: str = None,
    ) -> pd.DataFrame:
        """
        Walk-Forward 逐步滚动预测，生成覆盖回测窗口的样本外多时界/自适应极值预测序列。
        支持【玩法B：粒子滤波动态重分配】——基于最近真实数据回测拟合度重加权蒙特卡洛预测路径。

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
        particle_filter : bool, default=False
            是否启用粒子滤波（玩法B）：让模型先预测最近已知的验证窗口，
            基于真实数据拟合度 Softmax 重新分配 20 条采样路径的权重。
        pf_val_window : int, default=3
            粒子滤波验证窗口长度（交易日数），默认 3 日。
        pf_temperature : float, optional
            粒子滤波 Softmax 温度系数，None 则自适应取误差标准差。

        【P1/P2/P3 修正说明】
        ---------------------
        本方法产出两个语义不同的价格预测，下游不得混用：

        * pred_expected —— 动态最优时界（optimal_horizon）当天的跨路径加权期望收盘价（无偏）。用于**入场判定**。
        * pred_path_max —— 路径最大值（最大值统计量，结构性正偏）。仅用于**止盈目标**。
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

        if particle_filter:
            cache_suffix = f"{cache_suffix}_pf"

        clean_code, det_market, prefix = SymbolResolver.resolve(stock_code, market=market)

        # 优先检查本地是否已有对应时界的 Walk-Forward 预测缓存
        if particle_filter:
            out_candidates = [
                os.path.join(output_dir, f"{clean_code}_wf_predictions{cache_suffix}.csv"),
                os.path.join(output_dir, f"{prefix}_wf_predictions{cache_suffix}.csv"),
            ]
        else:
            out_candidates = [
                os.path.join(output_dir, f"{clean_code}_wf_predictions{cache_suffix}.csv"),
                os.path.join(output_dir, f"{prefix}_wf_predictions{cache_suffix}.csv"),
                os.path.join(output_dir, f"{clean_code}_wf_predictions.csv"),
                os.path.join(output_dir, f"{prefix}_wf_predictions.csv"),
            ]
        target_path = next((p for p in out_candidates if os.path.exists(p)), out_candidates[0])
        out_path = os.path.join(output_dir, f"{clean_code}_wf_predictions{cache_suffix}.csv")

        # P4：缓存命中必须同时满足「行数足够」与「参数指纹一致」。
        expected_fp = self._build_fingerprint(
            stock_code=clean_code, test_days=test_days, lookback=lookback,
            step_size=step_size, sample_count=sample_count,
            forecast_horizon=forecast_horizon, max_auto_horizon=max_auto_horizon,
            temperature=temperature, top_p=top_p, min_down_quantile=min_down_quantile,
            particle_filter=particle_filter, pf_val_window=pf_val_window,
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

        # 针对并发场景进行 GPU/MPS 推理互斥保护（避免 Apple Silicon Metal 驱动 MTLCommandBuffer 并发断言崩溃）
        waiting_lock = not _INFERENCE_LOCK.acquire(blocking=False)
        if waiting_lock:
            print(f"   ⏳ [GPU调度] 标的 {clean_code} 排队等待 GPU/模型资源（MPS/Metal 驱动互斥保护中）...")
            _INFERENCE_LOCK.acquire()
        try:
            # 双重检查锁（DCL）：在当前线程等待锁期间，前面的线程可能已计算完成并保存了缓存
            if not force_refresh and os.path.exists(target_path):
                fp_ok, _ = self._check_cache_fingerprint(target_path, expected_fp)
                if fp_ok:
                    try:
                        cached_pred = pd.read_csv(target_path, encoding='utf-8-sig')
                        date_col = '日期' if '日期' in cached_pred.columns else cached_pred.columns[0]
                        if len(cached_pred) >= min(test_days, 10):
                            print(f"\n💾 [GPU调度复用] 命中已有 Walk-Forward 预测数据: {target_path}")
                            result_df = cached_pred.set_index(date_col)
                            result_df.index = pd.to_datetime(result_df.index)
                            result_df = self._normalize_pred_columns(result_df)
                            if 'pred_close' in result_df.columns:
                                return result_df
                    except Exception:
                        pass

            self._ensure_predictor()

            # 切分训练区间和回测区间
            train_df  = hist_df.iloc[:-test_days]
            test_df   = hist_df.iloc[-test_days:]
            test_dates = test_df.index.tolist()

            mkt_tag = {'CN': 'A股', 'HK': '港股', 'US': '美股'}.get(det_market, det_market)
            horizon_label = f"Auto (最大探测={eval_horizon}天)" if is_auto_horizon else f"固定={eval_horizon}天"
            print(f"\n📈 Walk-Forward 预测配置 [{mkt_tag} {clean_code}] (时界模式: {horizon_label}):")
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

                # -------------------------------------------------------------
                # 粒子滤波 (Particle Filtering / 玩法B)：配置验证窗口
                # -------------------------------------------------------------
                use_pf_this_step = (
                    particle_filter and len(all_up_to_target) >= lookback + pf_val_window
                )

                if use_pf_this_step:
                    # 把提示词窗口切分为：[前 lookback 天作为Prompt] + [后 pf_val_window 天作为已知验证走势]
                    prompt_df = all_up_to_target.iloc[-(lookback + pf_val_window): -pf_val_window]
                    val_df = all_up_to_target.iloc[-pf_val_window:]
                    x_df = prompt_df[available_cols].copy()
                    for col in available_cols:
                        x_df[col] = pd.to_numeric(x_df[col], errors='coerce').fillna(0.0)
                    x_timestamp = pd.Series(prompt_df.index)
                    val_timestamps = list(val_df.index)
                    real_val_closes = val_df['close'].values.astype(float)
                else:
                    window_df = all_up_to_target.iloc[-lookback:]
                    x_df = window_df[available_cols].copy()
                    for col in available_cols:
                        x_df[col] = pd.to_numeric(x_df[col], errors='coerce').fillna(0.0)
                    x_timestamp = pd.Series(window_df.index)
                    val_timestamps = []
                    real_val_closes = None

                # 确定从 target_date 起未来 eval_horizon 个交易日的序列
                if anchor_idx + eval_horizon <= len(test_dates):
                    target_future_dates = test_dates[anchor_idx : anchor_idx + eval_horizon]
                else:
                    avail_dates = test_dates[anchor_idx:]
                    needed = eval_horizon - len(avail_dates)
                    extra_dates = generate_future_dates_with_holidays(avail_dates[-1], needed + 5, market=det_market, verbose=False)[:needed]
                    target_future_dates = avail_dates + [pd.Timestamp(d) for d in extra_dates]

                # 拼接验证窗口与未来预测窗口
                full_target_dates = val_timestamps + list(target_future_dates)
                y_timestamp = pd.Series(full_target_dates)
                full_pred_len = len(full_target_dates)

                # 蒙特卡洛多路径采样（保留逐路径明细，供 P2/P3 计算极值与分位）
                paths = self._sample_paths(
                    x_df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=full_pred_len,
                    n_samples=sample_count,
                    temperature=temperature,
                    top_p=top_p,
                )
                if paths is None:
                    print(f"   ⚠️ 预测失败，跳过 [{target_date.date()}]")
                    continue

                # -------------------------------------------------------------
                # 粒子滤波动态重分配权重：基于对最近真实走势的拟合度打分
                # -------------------------------------------------------------
                weights = None
                val_error = 0.0
                if use_pf_this_step and 'close' in paths:
                    val_pred_arr = paths['close'][:, :pf_val_window]  # (n_samples, pf_val_window)
                    # 计算各粒子相对真实收盘价的加权相对误差 (Normalized RMSE)
                    rel_err = (val_pred_arr - real_val_closes[None, :]) / (real_val_closes[None, :] + 1e-6)
                    path_nrmse = np.sqrt(np.mean(rel_err ** 2, axis=1))  # (n_samples,)

                    # 动态自适应 Softmax 温度系数
                    err_std = float(np.std(path_nrmse))
                    temp = pf_temperature if pf_temperature is not None else max(err_std, 0.015)
                    # 数值稳定性 Softmax 计算
                    exp_scores = np.exp(-(path_nrmse - np.min(path_nrmse)) / temp)
                    weights = exp_scores / np.sum(exp_scores)
                    val_error = float(np.average(path_nrmse, weights=weights))

                    # 截断提取 target_date 及以后的未来推演部分
                    future_paths = {k: v[:, pf_val_window:] for k, v in paths.items()}
                else:
                    future_paths = paths

                prev_real_close = float(all_up_to_target['close'].iloc[-1])
                rec = self._summarize_paths(
                    paths=future_paths,
                    min_down_quantile=min_down_quantile,
                    prev_real_close=prev_real_close,
                    is_auto_horizon=is_auto_horizon,
                    eval_horizon=eval_horizon,
                    weights=weights,
                    val_error=val_error,
                )
                pred_records[target_date] = rec

                elapsed = time.time() - t0
                opt_horizon = int(rec['optimal_horizon'])
                opt_str = f" [动态波峰 H*={opt_horizon}日]" if is_auto_horizon else ""
                pf_str = f" [粒子置信={rec['pf_confidence']:.1%}, 误差={rec['pf_val_error']:.2%}]" if use_pf_this_step else ""
                print(f"   [{seq_idx+1}/{len(anchor_indices)}] {target_date.date()} "
                      f"期望={rec['pred_expected']:.3f} 波峰={rec['pred_path_max']:.3f}"
                      f"(溢价{rec['peak_premium']*100:+.1f}%){opt_str}{pf_str} "
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
                               'optimal_horizon', 'peak_premium', 'n_samples',
                               'pf_confidence', 'pf_val_error', 'pf_effective_n']
                anchor_vals = {k: [pred_records[d].get(k, 0.0) for d in anchor_dates] for k in metric_keys}

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
                '粒子滤波置信度':   [pred_records[d].get('pf_confidence', 1.0) for d in sorted_dates],
                '粒子滤波验证误差': [pred_records[d].get('pf_val_error', 0.0) for d in sorted_dates],
                '有效粒子数':       [pred_records[d].get('pf_effective_n', sample_count) for d in sorted_dates],
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
                particle_filter=particle_filter, pf_val_window=pf_val_window,
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

        finally:
            _INFERENCE_LOCK.release()

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
        particle_filter: bool = False,
        pf_val_window: int = 3,
        pf_temperature: float = None,
    ) -> tuple:
        """
        一键自动管线：拉取行情 → Walk-Forward 预测（可配置粒子滤波） → 返回 (hist_df, pred_df)

        返回的两个 DataFrame 直接满足 KronosBacktesterPro.align_and_generate_signals 的输入要求。
        """
        clean_code, det_market, _ = SymbolResolver.resolve(stock_code, market=market)
        market_label = {'CN': 'A股', 'HK': '港股', 'US': '美股'}.get(det_market, det_market)
        print(f"\n{'='*60}")
        print(f"🚀 启动 Kronos 自动管线: [{market_label}] {clean_code}"
              f"{' [粒子滤波重分配开启]' if particle_filter else ''}")
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
            particle_filter=particle_filter,
            pf_val_window=pf_val_window,
            pf_temperature=pf_temperature,
            market=det_market,
        )

        return hist_df, pred_df




# 别名映射
KronosPredictorPipeline = KronosDataFetcher
