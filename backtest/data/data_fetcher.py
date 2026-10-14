"""
backtest/data/data_fetcher.py
多市场 (A股/港股/美股) 与多周期 (日线/15m/30m/60m) 行情获取与加载器。
具备本地缓存智能匹配、列名标准化、高频数据加载与可用标的扫描能力。
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional, Tuple, List, Set
import pandas as pd
import numpy as np

from ..core.types import TimeFrame
from ..core.market_rules import SymbolResolver, MarketProfile
from .cache import CacheManager

logger = logging.getLogger("KronosBacktest")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")


def _import_data_utils():
    """动态导入数据拉取底层工具函数"""
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
        try:
            from examples.prediction_akshare_2024_2025_copy import (
                fetch_or_load_stock_data,
                prepare_stock_data,
                generate_future_dates_with_holidays,
                resolve_date_range,
                ensure_output_directory,
                detect_market_type,
                validate_stock_code,
            )
        except ImportError:
            fetch_or_load_stock_data = None
            prepare_stock_data = None
            generate_future_dates_with_holidays = None
            resolve_date_range = None
            ensure_output_directory = None
            from ..core.market_rules import (
                detect_market_type as _core_detect,
                validate_stock_code as _core_validate,
            )
            detect_market_type = _core_detect
            validate_stock_code = _core_validate
    return (
        fetch_or_load_stock_data,
        prepare_stock_data,
        generate_future_dates_with_holidays,
        resolve_date_range,
        ensure_output_directory,
        detect_market_type,
        validate_stock_code,
    )


class DataFetcher:
    """行情数据拉取与清洗器"""

    @staticmethod
    def load_stock_history(
        stock_code: str,
        data_dir: str,
        market: str = None,
        timeframe: Union[str, TimeFrame] = TimeFrame.D1,
    ) -> Tuple[pd.DataFrame, MarketProfile]:
        """
        加载本地行情文件并标准化清洗（支持日线及15m/30m多周期）。
        """
        tf = TimeFrame.from_str(timeframe)
        profile = MarketProfile.for_stock(stock_code, market=market)
        clean_code, det_market, prefix = SymbolResolver.resolve(stock_code, market=profile.market)

        # 候选文件查找
        hist_candidates = []
        if tf.is_intraday:
            hist_candidates.extend([
                os.path.join(data_dir, f"{prefix}_{tf.value}.csv"),
                os.path.join(data_dir, f"{clean_code}_{tf.value}.csv"),
                os.path.join(data_dir, f"{prefix}_stock_data_{tf.value}.csv"),
                os.path.join(data_dir, f"{clean_code}_stock_data_{tf.value}.csv"),
            ])

        hist_candidates.extend([
            os.path.join(data_dir, f"{prefix}_stock_data.csv"),
            os.path.join(data_dir, f"{clean_code}_stock_data.csv"),
            os.path.join(data_dir, f"{clean_code}_all_history.csv"),
            os.path.join(data_dir, f"{clean_code}.csv"),
        ])
        hist_candidates = list(dict.fromkeys(hist_candidates))
        hist_file = next((p for p in hist_candidates if os.path.exists(p)), None)

        if not hist_file:
            raise FileNotFoundError(
                f"[DATA ERROR] 未找到 {stock_code} ({det_market}, 周期={tf.value}) 的行情文件。\n"
                f"已查找路径:\n" + "\n".join(f"  {p}" for p in hist_candidates)
            )

        print(f"✅ 历史数据: {hist_file}")
        hist_df = pd.read_csv(hist_file, encoding='utf-8-sig')

        # 列名标准化
        hist_col_map = {
            '日期': 'date', 'timestamps': 'date', '时间': 'date', 'datetime': 'date',
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
            logger.warning(f"[数据清洗] 行情数据存在 {dup_cnt} 个重复时间戳，已保留最后一条")

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in hist_df.columns:
                hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')

        if 'amount' not in hist_df.columns and 'volume' in hist_df.columns:
            hist_df['amount'] = hist_df['close'] * hist_df['volume']

        print(f"   历史范围: {hist_df.index.min()} → {hist_df.index.max()}  ({len(hist_df)} 条)")
        return hist_df, profile

    @staticmethod
    def load_prediction_data(
        stock_code: str,
        model_dir: str,
        market: str = None,
        particle_filter: bool = False,
    ) -> pd.DataFrame:
        """
        智能检索并加载模型预测文件，执行去重与列名映射。
        """
        clean_code, det_market, prefix = SymbolResolver.resolve(stock_code, market=market)

        pred_candidates = []
        if particle_filter:
            pred_candidates.extend([
                os.path.join(model_dir, f"{clean_code}_wf_predictions_auto_pf.csv"),
                os.path.join(model_dir, f"{prefix}_wf_predictions_auto_pf.csv"),
                os.path.join(model_dir, f"{clean_code}_wf_predictions_pf.csv"),
                os.path.join(model_dir, f"{prefix}_wf_predictions_pf.csv"),
            ])
        pred_candidates.extend([
            os.path.join(model_dir, f"{clean_code}_wf_predictions_auto.csv"),
            os.path.join(model_dir, f"{prefix}_wf_predictions_auto.csv"),
            os.path.join(model_dir, f"{clean_code}_wf_predictions.csv"),
            os.path.join(model_dir, f"{prefix}_wf_predictions.csv"),
            os.path.join(model_dir, f"{clean_code}_wf_predictions_h3.csv"),
            os.path.join(model_dir, f"{clean_code}_wf_predictions_h5.csv"),
            os.path.join(model_dir, f"{clean_code}_detailed_predictions.csv"),
            os.path.join(model_dir, f"{clean_code}_kronos_predictions.csv"),
            os.path.join(model_dir, f"{clean_code}_predictions.csv"),
        ])
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

        from .kronos_predictor import KronosDataFetcher
        pred_df = KronosDataFetcher._normalize_pred_columns(pred_df)

        if 'pred_close' not in pred_df.columns:
            # 尝试由 pred_close_mean 降级回退
            if 'pred_close_mean' in pred_df.columns:
                pred_df['pred_close'] = pred_df['pred_close_mean']
            else:
                raise ValueError(
                    f"[DATA ERROR] 预测文件 {pred_file} 中未找到可用的预测收盘价列。\n"
                    f"当前列名: {list(pred_df.columns)}"
                )

        print(f"   预测范围: {pred_df.index.min().date()} → {pred_df.index.max().date()}  ({len(pred_df)} 条)")
        return pred_df

    @staticmethod
    def fetch_history(
        stock_code: str,
        data_dir: str,
        history_days: int = 400,
        market: str = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        拉取历史日K线数据并标准化为回测引擎所需格式。
        """
        clean_code, det_market, prefix = SymbolResolver.resolve(stock_code, market=market)
        (fetch_or_load_stock_data, prepare_stock_data,
         _, _, validate_stock_code) = _import_data_utils()

        if fetch_or_load_stock_data is None:
            raise ImportError("未找到 prediction_akshare_2024_2025_copy 数据获取模块")

        is_valid, err_msg, norm_code, _ = validate_stock_code(clean_code, det_market)
        normalized_code = norm_code if is_valid else clean_code

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
            )

        df = prepare_stock_data(raw_df, normalized_code)
        df = df.rename(columns={'timestamps': 'date'})
        df = df.set_index('date').sort_index()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'amount' not in df.columns and 'volume' in df.columns:
            df['amount'] = df['close'] * df['volume']

        print(f"✅ 历史行情: {df.index.min().date()} → {df.index.max().date()} ({len(df)} 条)")
        return df

    @staticmethod
    def get_available_stocks(data_dir: str, model_dir: str) -> List[str]:
        """
        扫描 data_dir 和 model_dir，返回同时拥有历史数据和预测文件的股票代码列表。
        全面支持 A股 (如 300418)、港股 (如 HK_00700/00700)、美股 (如 US_AAPL/AAPL)。
        """
        def extract_codes(directory: str, suffixes: list) -> Set[str]:
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
            ['_stock_data.csv', '_all_history.csv', '_stock_data_15m.csv', '_stock_data_30m.csv']
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
