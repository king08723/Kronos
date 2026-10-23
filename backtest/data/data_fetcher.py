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
    def _read_and_clean_csv(file_path: str) -> pd.DataFrame:
        """读取并标准化清洗 K 线 CSV 文件"""
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        # 列名标准化映射
        hist_col_map = {
            '日期': 'date', 'timestamps': 'date', '时间': 'date', 'datetime': 'date',
            '开盘价': 'open',  'open': 'open',
            '收盘价': 'close', 'close': 'close',
            '最高价': 'high',  'high': 'high',
            '最低价': 'low',   'low': 'low',
            '成交量': 'volume','volume': 'volume',
            '成交额': 'amount','amount': 'amount',
        }
        df = df.rename(columns={k: v for k, v in hist_col_map.items() if k in df.columns})

        for drop_col in ['index', 'Unnamed: 0', '股票代码', 'stock_code', 'pct_chg']:
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])

        date_col = 'date' if 'date' in df.columns else df.columns[0]
        df['date'] = pd.to_datetime(df[date_col])
        df = df.set_index('date').sort_index()

        if df.index.has_duplicates:
            dup_cnt = int(df.index.duplicated().sum())
            df = df[~df.index.duplicated(keep='last')]
            logger.warning(f"[数据清洗] 行情数据存在 {dup_cnt} 个重复时间戳，已保留最后一条")

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'amount' not in df.columns and 'volume' in df.columns:
            df['amount'] = df['close'] * df['volume']

        return df

    @staticmethod
    def load_stock_history(
        stock_code: str,
        data_dir: str,
        market: str = None,
        timeframe: Union[str, TimeFrame] = TimeFrame.D1,
    ) -> Tuple[pd.DataFrame, MarketProfile]:
        """
        加载本地行情文件并标准化清洗（支持日线及 1m/5m/15m/30m/60m 多周期，并具备智能即时合成能力）。
        """
        from ..core.resampler import SessionResampler
        from ..core.timeframes import detect_timeframe

        tf = TimeFrame.from_str(timeframe)
        profile = MarketProfile.for_stock(stock_code, market=market)
        clean_code, det_market, prefix = SymbolResolver.resolve(stock_code, market=profile.market)

        # 候选查找目录：支持当前传入目录与解耦目录 (historical_data / data) 双向兼容
        search_dirs = [data_dir]
        base_name = os.path.basename(os.path.normpath(data_dir))
        parent_dir = os.path.dirname(os.path.normpath(data_dir))
        if base_name == 'data':
            alt_dir = os.path.join(parent_dir, 'historical_data')
            if os.path.exists(alt_dir) and alt_dir not in search_dirs:
                search_dirs.insert(0, alt_dir)
        elif base_name == 'historical_data':
            alt_dir = os.path.join(parent_dir, 'data')
            if os.path.exists(alt_dir) and alt_dir not in search_dirs:
                search_dirs.append(alt_dir)

        # 1. 第一优先级：精确匹配当前目标周期的文件
        exact_candidates = []
        for sdir in search_dirs:
            if tf.is_intraday:
                exact_candidates.extend([
                    os.path.join(sdir, f"{prefix}_{tf.value}.csv"),
                    os.path.join(sdir, f"{clean_code}_{tf.value}.csv"),
                    os.path.join(sdir, f"{prefix}_stock_data_{tf.value}.csv"),
                    os.path.join(sdir, f"{clean_code}_stock_data_{tf.value}.csv"),
                ])
            else:
                exact_candidates.extend([
                    os.path.join(sdir, f"{prefix}_stock_data.csv"),
                    os.path.join(sdir, f"{clean_code}_stock_data.csv"),
                    os.path.join(sdir, f"{clean_code}_all_history.csv"),
                    os.path.join(sdir, f"{clean_code}.csv"),
                    os.path.join(sdir, f"{prefix}.csv"),
                ])
        exact_candidates = list(dict.fromkeys(exact_candidates))
        exact_file = next((p for p in exact_candidates if os.path.exists(p)), None)

        if exact_file:
            print(f"✅ 历史数据 (精确匹配): {exact_file}")
            hist_df = DataFetcher._read_and_clean_csv(exact_file)
            file_tf = detect_timeframe(hist_df)
            # 若文件中实际粒度更细，使用 SessionResampler 精确对齐
            if file_tf != tf:
                if (file_tf.is_intraday and tf.is_intraday and file_tf.minutes < tf.minutes) or (not tf.is_intraday):
                    print(f"   [时段合成] 将基础数据 ({file_tf.label}) 精确合成为目标周期 ({tf.label})...")
                    hist_df = SessionResampler.resample_to_timeframe(hist_df, target_timeframe=tf, market=profile.market)
            print(f"   历史范围: {hist_df.index.min()} → {hist_df.index.max()}  ({len(hist_df)} 条)")
            return hist_df, profile

        # 2. 第二优先级：未找到精确匹配文件，尝试从更细粒度的高频基础数据即时合成
        all_intraday_tfs = [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.M30, TimeFrame.H1]
        eligible_base_tfs = [btf for btf in all_intraday_tfs if not tf.is_intraday or btf.minutes < tf.minutes]

        base_file = None
        base_tf = None
        for btf in eligible_base_tfs:
            base_candidates = []
            for sdir in search_dirs:
                base_candidates.extend([
                    os.path.join(sdir, f"{prefix}_{btf.value}.csv"),
                    os.path.join(sdir, f"{clean_code}_{btf.value}.csv"),
                    os.path.join(sdir, f"{prefix}_stock_data_{btf.value}.csv"),
                    os.path.join(sdir, f"{clean_code}_stock_data_{btf.value}.csv"),
                ])
            base_candidates = list(dict.fromkeys(base_candidates))
            found_p = next((p for p in base_candidates if os.path.exists(p)), None)
            if found_p:
                base_file = found_p
                base_tf = btf
                break

        if base_file:
            print(f"✅ 找到基础高频数据: {base_file} ({base_tf.label})")
            base_df = DataFetcher._read_and_clean_csv(base_file)
            print(f"   [时段合成] 正在使用 SessionResampler 从 {base_tf.label} 无损合成为 {tf.label}...")
            hist_df = SessionResampler.resample_to_timeframe(base_df, target_timeframe=tf, market=profile.market)
            print(f"   合成完成: {hist_df.index.min()} → {hist_df.index.max()}  ({len(hist_df)} 条)")
            return hist_df, profile

        # 3. 第三优先级：本地无匹配文件且无法合成，自动触发网络在线拉取并落盘
        if tf.is_intraday:
            target_save_dir = search_dirs[0]
            print(f"🌐 本地未找到 {stock_code} ({tf.label}) 行情，正在自动从网络拉取分时数据...")
            fetched_res = DataFetcher.fetch_intraday_history(
                stock_code=stock_code,
                timeframe=tf,
                market=profile.market,
                save_dir=target_save_dir,
            )
            if fetched_res is not None:
                fetched_df, saved_path = fetched_res
                print(f"✅ 自动获取并落盘成功: {saved_path} ({len(fetched_df)} 条)")
                clean_df = DataFetcher._read_and_clean_csv(saved_path)
                return clean_df, profile

        # 4. 兜底未找到报错
        all_searched = exact_candidates + [
            os.path.join(sdir, f"{clean_code}_{btf.value}.csv")
            for sdir in search_dirs for btf in eligible_base_tfs
        ]
        all_searched = list(dict.fromkeys(all_searched))
        raise FileNotFoundError(
            f"[DATA ERROR] 未找到 {stock_code} ({det_market}, 周期={tf.label}) 的行情文件或可合成的高频底座文件，且网络拉取未成功。\n"
            f"已查找路径:\n" + "\n".join(f"  {p}" for p in all_searched[:10])
        )

    @staticmethod
    def fetch_intraday_history(
        stock_code: str,
        timeframe: Union[str, TimeFrame] = TimeFrame.M30,
        market: str = None,
        save_dir: str = None,
        limit: int = 1000,
        force_refresh: bool = False,
    ) -> Optional[Tuple[pd.DataFrame, str]]:
        """
        从极速直连数据源（东方财富）联网获取分时K线数据（支持美股/港股/A股，1m/5m/15m/30m/60m），
        清洗并自动落盘为标准回测格式 CSV。
        """
        import requests
        tf = TimeFrame.from_str(timeframe)
        klt_map = {
            '1m': '1', '5m': '5', '15m': '15', '30m': '30', '60m': '60'
        }
        klt = klt_map.get(tf.value)
        if not klt:
            logger.warning(f"[fetch_intraday_history] 暂不支持非分时周期的在线拉取: {tf.value}")
            return None

        clean_code, det_market, prefix = SymbolResolver.resolve(stock_code, market=market)

        # 确定候选 secid
        secids = []
        if det_market == 'US' or clean_code.isalpha():
            secids = [f'105.{clean_code}', f'106.{clean_code}', f'107.{clean_code}']
        elif det_market == 'HK' or (clean_code.isdigit() and len(clean_code) == 5):
            secids = [f'116.{clean_code.zfill(5)}']
        else:
            if clean_code.startswith(('60', '68')):
                secids = [f'1.{clean_code}', f'0.{clean_code}']
            elif clean_code.startswith(('4', '8', '92')):
                secids = [f'0.{clean_code}', f'2.{clean_code}']
            else:
                secids = [f'0.{clean_code}', f'1.{clean_code}']

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://quote.eastmoney.com/',
        }

        # 确定保存路径
        if save_dir is None:
            save_dir = os.path.join(PROJECT_ROOT, "backtest", "historical_data")
            if not os.path.exists(save_dir):
                save_dir = os.path.join(PROJECT_ROOT, "backtest", "data")
        os.makedirs(save_dir, exist_ok=True)

        if det_market in ('HK', 'US'):
            out_filename = f"{prefix}_{tf.value}.csv"
        else:
            out_filename = f"{clean_code}_{tf.value}.csv"
        target_csv_path = os.path.join(save_dir, out_filename)

        if not force_refresh and os.path.exists(target_csv_path):
            try:
                cached_df = pd.read_csv(target_csv_path, encoding='utf-8-sig')
                if len(cached_df) >= 30:
                    return cached_df, target_csv_path
            except Exception:
                pass

        # 优先检索与拉取策略：美股走 Sina / Yahoo / Eastmoney 多源互备
        df_res = None

        # --- 数据源 A: 美股优先新浪分时极速接口 ---
        if det_market == 'US':
            sina_type_map = {'5m': '5', '15m': '15', '30m': '30', '60m': '60'}
            sina_type = sina_type_map.get(tf.value)
            if sina_type:
                try:
                    url_sina = f"https://stock.finance.sina.com.cn/usstock/api/json.php/US_MinKService.getMinK?symbol={clean_code.lower()}&type={sina_type}"
                    resp_sina = requests.get(url_sina, headers=headers, timeout=6, proxies={'http': None, 'https': None})
                    if resp_sina.status_code == 200:
                        data_sina = resp_sina.json()
                        if isinstance(data_sina, list) and len(data_sina) >= 10:
                            records = []
                            for row in data_sina:
                                o_p = float(row.get('o', 0))
                                c_p = float(row.get('c', 0))
                                h_p = float(row.get('h', 0))
                                l_p = float(row.get('l', 0))
                                vol = float(row.get('v', 0))
                                amt = float(row.get('a', c_p * vol))
                                records.append({
                                    '日期': row.get('d'),
                                    '开盘价': o_p,
                                    '收盘价': c_p,
                                    '最高价': h_p,
                                    '最低价': l_p,
                                    '成交量': vol,
                                    '成交额': amt,
                                })
                            df_res = pd.DataFrame(records)
                            logger.info(f"[fetch_intraday_history] 成功通过 [新浪财经] 获取美股 {clean_code} ({tf.label}) 共 {len(df_res)} 条分时数据")
                except Exception as ex_sina:
                    logger.debug(f"[fetch_intraday_history] 新浪美股接口异常: {ex_sina}")

            # --- 数据源 B: 美股备选 Yahoo Finance 直连 ---
            if df_res is None or df_res.empty:
                try:
                    url_yf = f"https://query1.finance.yahoo.com/v8/finance/chart/{clean_code}?range=1mo&interval={tf.value}"
                    resp_yf = requests.get(url_yf, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=6)
                    if resp_yf.status_code == 200:
                        yf_json = resp_yf.json()
                        chart_res = yf_json.get('chart', {}).get('result', [])
                        if chart_res:
                            res_obj = chart_res[0]
                            ts_list = res_obj.get('timestamp', [])
                            quotes = res_obj.get('indicators', {}).get('quote', [{}])[0]
                            opens = quotes.get('open', [])
                            highs = quotes.get('high', [])
                            lows = quotes.get('low', [])
                            closes = quotes.get('close', [])
                            vols = quotes.get('volume', [])
                            records = []
                            for i, ts in enumerate(ts_list):
                                if closes[i] is not None:
                                    dt_str = pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d %H:%M:%S')
                                    records.append({
                                        '日期': dt_str,
                                        '开盘价': float(opens[i] if opens[i] is not None else closes[i]),
                                        '收盘价': float(closes[i]),
                                        '最高价': float(highs[i] if highs[i] is not None else closes[i]),
                                        '最低价': float(lows[i] if lows[i] is not None else closes[i]),
                                        '成交量': float(vols[i] if vols[i] is not None else 0),
                                        '成交额': float((closes[i] or 0) * (vols[i] or 0)),
                                    })
                            if len(records) >= 10:
                                df_res = pd.DataFrame(records)
                                logger.info(f"[fetch_intraday_history] 成功通过 [Yahoo Finance] 获取美股 {clean_code} ({tf.label}) 共 {len(df_res)} 条分时数据")
                except Exception as ex_yf:
                    logger.debug(f"[fetch_intraday_history] Yahoo Finance 直连异常: {ex_yf}")

        # --- 数据源 C: 东方财富直连 (全市场通用底座) ---
        if df_res is None or df_res.empty:
            for secid in secids:
                url = (
                    f"http://push2his.eastmoney.com/api/qt/stock/kline/get"
                    f"?secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
                    f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
                    f"&klt={klt}&fqt=1&end=20500101&lmt={limit}"
                )
                try:
                    resp = requests.get(url, headers=headers, timeout=6, proxies={'http': None, 'https': None})
                    if resp.status_code == 200:
                        res_json = resp.json()
                        klines = res_json.get('data', {}).get('klines', []) if res_json.get('data') else []
                        if klines and len(klines) >= 10:
                            records = []
                            for item in klines:
                                p = item.split(',')
                                if len(p) >= 6:
                                    close_p = float(p[2])
                                    vol = float(p[5])
                                    amt = float(p[6]) if len(p) > 6 else close_p * vol
                                    records.append({
                                        '日期': p[0],
                                        '开盘价': float(p[1]),
                                        '收盘价': close_p,
                                        '最高价': float(p[3]),
                                        '最低价': float(p[4]),
                                        '成交量': vol,
                                        '成交额': amt,
                                    })
                            df_res = pd.DataFrame(records)
                            logger.info(f"[fetch_intraday_history] 成功通过 [东方财富] 获取 {secid} ({tf.label}) 共 {len(df_res)} 条分时数据")
                            break
                except Exception as e:
                    logger.debug(f"[fetch_intraday_history] 尝试请求 {secid} 失败: {e}")
                    continue

        if df_res is not None and not df_res.empty:
            df_res.to_csv(target_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"[fetch_intraday_history] 分时数据已成功保存至: {target_csv_path}")
            return df_res, target_csv_path

        logger.error(f"[fetch_intraday_history] 所有数据源均未获取到 {stock_code} ({det_market}, {tf.label}) 分时行情")
        return None

    @staticmethod
    def load_prediction_data(
        stock_code: str,
        model_dir: str,
        market: str = None,
        particle_filter: bool = False,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        智能检索并加载模型预测文件，执行去重与列名映射。
        优先匹配具体分时周期的预测文件（如 AAPL_30m_wf_predictions_auto.csv），
        若未找到则平滑降级至通用/日线预测文件。
        """
        clean_code, det_market, prefix = SymbolResolver.resolve(stock_code, market=market)

        tf_tag = ""
        if timeframe and str(timeframe).lower() not in ("1d", "none"):
            tf_tag = f"_{str(timeframe).lower()}"

        pred_candidates = []
        if tf_tag:
            if particle_filter:
                pred_candidates.extend([
                    os.path.join(model_dir, f"{clean_code}{tf_tag}_wf_predictions_auto_pf.csv"),
                    os.path.join(model_dir, f"{prefix}{tf_tag}_wf_predictions_auto_pf.csv"),
                    os.path.join(model_dir, f"{clean_code}{tf_tag}_wf_predictions_pf.csv"),
                    os.path.join(model_dir, f"{prefix}{tf_tag}_wf_predictions_pf.csv"),
                ])
            pred_candidates.extend([
                os.path.join(model_dir, f"{clean_code}{tf_tag}_wf_predictions_auto.csv"),
                os.path.join(model_dir, f"{prefix}{tf_tag}_wf_predictions_auto.csv"),
                os.path.join(model_dir, f"{clean_code}{tf_tag}_wf_predictions.csv"),
                os.path.join(model_dir, f"{prefix}{tf_tag}_wf_predictions.csv"),
            ])

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
        if not particle_filter:
            # 降级保底：若未开启 --pf 但存在 _pf 预测文件，自动匹配复用
            if tf_tag:
                pred_candidates.extend([
                    os.path.join(model_dir, f"{clean_code}{tf_tag}_wf_predictions_auto_pf.csv"),
                    os.path.join(model_dir, f"{prefix}{tf_tag}_wf_predictions_auto_pf.csv"),
                ])
            pred_candidates.extend([
                os.path.join(model_dir, f"{clean_code}_wf_predictions_auto_pf.csv"),
                os.path.join(model_dir, f"{prefix}_wf_predictions_auto_pf.csv"),
                os.path.join(model_dir, f"{clean_code}_wf_predictions_pf.csv"),
                os.path.join(model_dir, f"{prefix}_wf_predictions_pf.csv"),
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
                raise ValueError(
                    f"[DATA ERROR] 预测文件 {pred_file} 中未找到可用的预测收盘价列。\n"
                    f"当前列名: {list(pred_df.columns)}"
                )
        def _safe_dt_str(d):
            if hasattr(d, 'strftime'):
                if hasattr(d, 'hour') and (d.hour != 0 or d.minute != 0 or d.second != 0):
                    return d.strftime('%Y-%m-%d %H:%M:%S')
                return d.strftime('%Y-%m-%d')
            elif hasattr(d, 'date'):
                return str(d.date())
            return str(d)

        print(f"   预测范围: {_safe_dt_str(pred_df.index.min())} → {_safe_dt_str(pred_df.index.max())}  ({len(pred_df)} 根 Bar/条)")
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

        min_dt_str = _safe_dt_str(df.index.min())
        max_dt_str = _safe_dt_str(df.index.max())
        print(f"✅ 历史行情: {min_dt_str} → {max_dt_str} ({len(df)} 根 Bar/条)")
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

        hist_suffixes = [
            '_stock_data.csv', '_all_history.csv',
            '_1m.csv', '_5m.csv', '_15m.csv', '_30m.csv', '_60m.csv',
            '_stock_data_1m.csv', '_stock_data_5m.csv', '_stock_data_15m.csv',
            '_stock_data_30m.csv', '_stock_data_60m.csv',
        ]
        hist_codes = extract_codes(data_dir, hist_suffixes)
        base_name = os.path.basename(os.path.normpath(data_dir))
        parent_dir = os.path.dirname(os.path.normpath(data_dir))
        alt_name = 'historical_data' if base_name == 'data' else ('data' if base_name == 'historical_data' else None)
        if alt_name:
            alt_dir = os.path.join(parent_dir, alt_name)
            if os.path.isdir(alt_dir):
                hist_codes.update(extract_codes(alt_dir, hist_suffixes))
        pred_codes = extract_codes(
            model_dir,
            ['_wf_predictions_auto.csv', '_wf_predictions.csv',
             '_kronos_predictions.csv', '_detailed_predictions.csv',
             '_comprehensive_predictions.csv', '_streamlit_predictions.csv',
             '_predictions.csv']
        )

        both = sorted(hist_codes & pred_codes)
        return both if both else sorted(hist_codes | pred_codes) or ['300418']
