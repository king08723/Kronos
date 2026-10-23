"""
run_backtest_batch.py
===========================================================
Kronos 批量量化回测系统 (Batch Stock Backtest Engine)
===========================================================
参考 prediction_batch.py 的池化设计与 run_backtest_kronos_pro.py 的回测内核，
实现面向多标的股票池的批量、自动化回测评估系统。

核心特性：
1. 股票池灵活导入：支持 CSV 文件 (--stock-file)、命令行内联 (--stocks-inline)、
   位置参数或内置默认配置。
2. 多线程并发数据准备：多源行情并发拉取，时效感知缓存判定。
3. 预测与回测隔离编排：
   - 预测缓存共享 backtest/yuce/ 或 examples/yuce/，与单股 --auto 模式完全复用；
   - 回测输出隔离于 backtest/batch/{stock_code}/。
4. 容错与异常隔离：单只标的失败不影响整体流程。
5. 默认串行，支持 --parallel-stocks N 并行。
6. 绩效大盘排行榜：batch_backtest_ranking.csv。

运行示例：
    python backtest/run_backtest_batch.py --auto
    python backtest/run_backtest_batch.py --auto --stock-file backtest/my_stocks.csv
    python backtest/run_backtest_batch.py 300418 600036 AAPL --auto
    python backtest/run_backtest_batch.py 300418 600036 --dry-run
"""

import os
import sys
import time
import json
import socket
import argparse
import warnings
from typing import Optional, List, Dict, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

socket.setdefaulttimeout(10.0)
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")
for p in [PROJECT_ROOT, CURRENT_DIR, EXAMPLES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

def resolve_data_dir() -> str:
    """回测行情数据存储目录：优先收敛至独立解耦目录 backtest/historical_data，向下兼容 backtest/data"""
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

from run_backtest_kronos_pro import (
    KronosDataFetcher,
    KronosBacktesterPro,
    MarketProfile,
    SymbolResolver,
    setup_logger,
    clear_logs,
    set_logging_state,
)
from backtest.core import resolve_timeframe_params, get_timeframe_defaults, TimeFrame

try:
    from prediction_akshare_2024_2025_copy import (
        validate_stock_code,
        ensure_output_directory,
    )
except ImportError:
    from examples.prediction_akshare_2024_2025_copy import (
        validate_stock_code,
        ensure_output_directory,
    )


# ---------------------------------------------------------------------------
# 股票池解析工具
# ---------------------------------------------------------------------------

def parse_stock_item(raw_item) -> dict:
    """规范化股票条目为 {'stock_code', 'stock_name', 'market'}，通过 SymbolResolver 自动推断市场与标准化代码"""
    if isinstance(raw_item, dict):
        code = str(raw_item.get('stock_code', raw_item.get('code', ''))).strip()
        name = str(raw_item.get('stock_name', raw_item.get('name', code))).strip()
        market = raw_item.get('market', None)
    else:
        parts = str(raw_item).strip().split(':')
        code = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else code
        market = parts[2].strip() if len(parts) > 2 else None

    clean_code, det_mkt, _ = SymbolResolver.resolve(code, market=market)
    return {'stock_code': clean_code, 'stock_name': name, 'market': det_mkt}


def load_stock_pool_from_file(file_path: str) -> list:
    """
    从 CSV/TXT/JSON 文件加载股票池。
    CSV 格式: stock_code,stock_name,market（market 可空，按代码自动推断）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"股票池文件不存在: {file_path}")
    ext = os.path.splitext(file_path)[-1].lower()
    pool = []
    if ext in ('.csv', '.txt'):
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        col_code = next((c for c in ['stock_code', 'code', '代码', '股票代码'] if c in df.columns), df.columns[0])
        col_name = next((c for c in ['stock_name', 'name', '名称', '股票名称'] if c in df.columns), None)
        col_mkt = next((c for c in ['market', '市场'] if c in df.columns), None)
        for _, row in df.iterrows():
            code = str(row[col_code]).strip()
            name = str(row[col_name]).strip() if col_name else code
            mkt = str(row[col_mkt]).strip() if col_mkt and pd.notna(row[col_mkt]) else None
            pool.append({'stock_code': code, 'stock_name': name, 'market': mkt})
    elif ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get('stock_pool', [])
        pool = [parse_stock_item(it) for it in items]
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    return pool


# ---------------------------------------------------------------------------
# 批量回测引擎
# ---------------------------------------------------------------------------

class BatchBacktester:
    """批量股票量化回测编排引擎"""

    def __init__(self, model_name="NeoQuasar/Kronos-base",
                 tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                 device=None, max_context=512, log_dir=None):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.max_context = max_context
        self.logger = setup_logger(log_dir)
        self._fetcher = None

    def _get_fetcher(self) -> KronosDataFetcher:
        if self._fetcher is None:
            self._fetcher = KronosDataFetcher(
                model_name=self.model_name,
                tokenizer_name=self.tokenizer_name,
                device=self.device,
                max_context=self.max_context,
            )
        return self._fetcher

    # ---- 步骤 1: 并发数据准备 ----

    def prepare_data_pool(self, stock_list, data_dir, history_days=700,
                          lookback=90, test_days=120, force_refresh=False,
                          max_workers=6, progress_callback=None, timeframe="1d",
                          lookback_bars=None, test_bars=None):
        tf_obj = TimeFrame.from_str(timeframe)
        eff_lookback = lookback_bars if lookback_bars is not None else lookback
        eff_test = test_bars if test_bars is not None else test_days
        print("\n" + "=" * 75)
        print(f"📡 [1/4] 并发准备历史行情 ({tf_obj.label} · 目标: {len(stock_list)} 只, 线程: {max_workers})")
        print("=" * 75)
        ensure_output_directory(data_dir)
        fetcher = self._get_fetcher()
        min_bars = eff_lookback + eff_test

        def _fetch(item):
            item = parse_stock_item(item)
            raw_code, name = item['stock_code'], item['stock_name']
            raw_mkt = item.get('market')
            ok, err, norm_code, det_mkt = validate_stock_code(raw_code, raw_mkt)
            if not ok:
                return raw_code, name, None, det_mkt or 'UNKNOWN', f"代码校验失败: {err}"
            try:
                hist_df = fetcher.fetch_history(
                    stock_code=norm_code, data_dir=data_dir,
                    history_days=history_days, market=det_mkt,
                    force_refresh=force_refresh,
                    timeframe=timeframe,
                )
                if hist_df is None or len(hist_df) == 0:
                    return norm_code, name, None, det_mkt, "行情为空"
                if len(hist_df) < min_bars:
                    bar_unit = "条" if tf_obj.is_intraday else "日"
                    return norm_code, name, None, det_mkt, \
                        f"{tf_obj.label}K线({len(hist_df)}{bar_unit})不足{min_bars}{bar_unit}"
                return norm_code, name, hist_df, det_mkt, "OK"
            except Exception as e:
                return norm_code, name, None, det_mkt, str(e)

        valid, failed = {}, []
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futs = {exe.submit(_fetch, s): s for s in stock_list}
            done_cnt = 0
            total_cnt = len(stock_list)
            for f in as_completed(futs):
                done_cnt += 1
                code, name, df, mkt, st = f.result()
                if df is not None:
                    valid[code] = {
                        'stock_code': code, 'stock_name': name,
                        'market': mkt, 'hist_df': df,
                        'total_bars': len(df),
                    }
                else:
                    failed.append((code, name, mkt, st))
                    tag = {'CN': 'A股', 'HK': '港股', 'US': '美股'}.get(mkt, mkt or '')
                    print(f"   ⚠️ 过滤 [{tag}] {name}({code}): {st}")

                if progress_callback:
                    try:
                        progress_callback(done_cnt, total_cnt, code, name, "OK" if df is not None else f"过滤: {st}")
                    except Exception:
                        pass
        print(f"\n✅ 数据就绪: 合格 {len(valid)} 只, 过滤 {len(failed)} 只")
        return valid

    # ---- 步骤 2: 单股完整流水线 ----

    def run_single_stock(self, stock_info, data_dir, pred_dir,
                         backtest_batch_dir, auto_predict=True,
                         test_days=120, lookback=90, step_size=1,
                         sample_count=35, forecast_horizon="auto",
                         max_auto_horizon=10, force_refresh=False,
                         min_down_quantile=0.10,
                         initial_capital=None, z_threshold: Optional[float] = None,
                         hard_stop: Optional[float] = None, trail_stop: Optional[float] = None,
                         max_hold: Optional[int] = None,
                         min_lot_guarantee=True, entry_percentile: Optional[float] = None,
                         calib_window: Optional[int] = None, dynamic_calib=True,
                         calib_window_min: Optional[int] = None, vol_high_thresh=0.40,
                         vol_low_thresh=0.20, risk_per_trade=0.01,
                         max_position=0.80, intraday_stop=True,
                         require_skill=False, generate_plots=True,
                         enable_log: bool = True,
                         timeframe: str = "1d",
                         volume_participation_cap: Optional[float] = None,
                         slippage_model: Optional[str] = None,
                         slippage_impact_coef: float = 0.1,
                         max_slippage: float = 0.02,
                         intraday_exit_at_eod: Optional[bool] = None,
                         overnight_gap_protection: Optional[bool] = None,
                         use_mtf_trend_filter: Optional[bool] = None,
                         commission_bps: Optional[float] = None,
                         slippage_bps: Optional[float] = None,
                         auto_adapt_params: bool = True,
                         test_bars: Optional[int] = None,
                         lookback_bars: Optional[int] = None,
                         **kwargs):
        code = stock_info['stock_code']
        name = stock_info['stock_name']
        market = stock_info.get('market')
        hist_df = stock_info.get('hist_df')

        eff_test_bars = test_bars if test_bars is not None else test_days
        eff_lookback_bars = lookback_bars if lookback_bars is not None else lookback
        test_days = eff_test_bars
        lookback = eff_lookback_bars

        stock_out = os.path.join(backtest_batch_dir, code)
        ensure_output_directory(stock_out)

        prof = MarketProfile.for_stock(code, market=market, stock_name=name)
        cap = initial_capital if initial_capital is not None else prof.default_capital

        if auto_adapt_params:
            adaptive_cfg = resolve_timeframe_params(
                timeframe=timeframe,
                market=prof.market,
                overrides={
                    'z_threshold': z_threshold,
                    'hard_stop': hard_stop,
                    'trail_stop': trail_stop,
                    'max_hold': max_hold,
                    'entry_percentile': entry_percentile,
                    'calib_window': calib_window,
                    'calib_window_min': calib_window_min,
                    'volume_participation_cap': volume_participation_cap,
                    'slippage_model': slippage_model,
                    'intraday_exit_at_eod': intraday_exit_at_eod,
                    'overnight_gap_protection': overnight_gap_protection,
                    'use_mtf_trend_filter': use_mtf_trend_filter,
                },
                auto_adapt=True,
            )
            z_threshold = adaptive_cfg['z_threshold']
            hard_stop = adaptive_cfg['hard_stop']
            trail_stop = adaptive_cfg['trail_stop']
            max_hold = adaptive_cfg['max_hold']
            entry_percentile = adaptive_cfg['entry_percentile']
            calib_window = adaptive_cfg['calib_window']
            calib_window_min = adaptive_cfg['calib_window_min']
            volume_participation_cap = adaptive_cfg['volume_participation_cap']
            slippage_model = adaptive_cfg['slippage_model']
            intraday_exit_at_eod = adaptive_cfg['intraday_exit_at_eod']
            overnight_gap_protection = adaptive_cfg['overnight_gap_protection']
            use_mtf_trend_filter = adaptive_cfg['use_mtf_trend_filter']
        else:
            z_threshold = 0.55 if z_threshold is None else z_threshold
            hard_stop = 0.04 if hard_stop is None else hard_stop
            trail_stop = 0.025 if trail_stop is None else trail_stop
            max_hold = 15 if max_hold is None else max_hold
            entry_percentile = 0.80 if entry_percentile is None else entry_percentile
            calib_window = 60 if calib_window is None else calib_window
            calib_window_min = 20 if calib_window_min is None else calib_window_min
            slippage_model = "fixed" if slippage_model is None else slippage_model
            intraday_exit_at_eod = False if intraday_exit_at_eod is None else intraday_exit_at_eod
            overnight_gap_protection = True if overnight_gap_protection is None else overnight_gap_protection
            use_mtf_trend_filter = True if use_mtf_trend_filter is None else use_mtf_trend_filter

        print(f"\n{'='*70}")
        print(f"🚀 回测: [{prof.market}] {name} ({code}) | 周期: {timeframe} | 本金: {prof.currency_symbol}{cap:,.2f} ({prof.currency})")
        print(f"   规则: {prof.describe()}")
        print(f"   滑点: {slippage_model} | 流动性截断: {volume_participation_cap or '无限制'} | 日内强平: {intraday_exit_at_eod} | MTF共振: {use_mtf_trend_filter}")
        print(f"   防守: 硬止损 {hard_stop:.2%} | 移动止盈 {trail_stop:.2%} | 最多持有 {max_hold} Bar | 信噪比门槛 {z_threshold}")
        print(f"   输出: {stock_out}")
        print(f"{'='*70}")

        fetcher = self._get_fetcher()
        pred_df = None

        # 预测阶段 (若需要自动滚动预测)
        if auto_predict:
            if hist_df is None:
                hist_df = fetcher.fetch_history(
                    stock_code=code, data_dir=data_dir,
                    history_days=700, market=market,
                    force_refresh=force_refresh,
                )
            pred_df = fetcher.walk_forward_predict(
                hist_df=hist_df, stock_code=code,
                output_dir=pred_dir, test_days=test_days,
                lookback=lookback, step_size=step_size,
                sample_count=sample_count,
                forecast_horizon=forecast_horizon,
                max_auto_horizon=max_auto_horizon,
                force_refresh=force_refresh,
                min_down_quantile=min_down_quantile,
                market=market,
                test_bars=eff_test_bars,
                lookback_bars=eff_lookback_bars,
                timeframe=timeframe,
            )

        enable_log = kwargs.get('enable_log', enable_log)

        # 回测引擎
        bt = KronosBacktesterPro(
            data_dir=data_dir, model_dir=pred_dir,
            initial_capital=cap,
            enable_log=enable_log,
        )
        try:
            bt.set_market(code, market=market, stock_name=name, log_dir=stock_out, enable_log=enable_log)
            if commission_bps is not None:
                bt.profile.commission_rate = float(commission_bps) / 10000.0
            if slippage_bps is not None:
                bt.slippage_rate = float(slippage_bps) / 10000.0

            if hist_df is None or pred_df is None:
                h_df, p_df = bt.load_data(code, timeframe=timeframe)
                if hist_df is None:
                    hist_df = h_df
                if pred_df is None:
                    pred_df = p_df

            signal_df = bt.align_and_generate_signals(
                hist_df, pred_df, z_score_threshold=z_threshold,
                forecast_horizon=forecast_horizon,
                max_auto_horizon=max_auto_horizon,
                hard_stop_loss=hard_stop,
                entry_percentile=entry_percentile,
                calib_window=calib_window,
                dynamic_calib=dynamic_calib,
                calib_window_min=calib_window_min,
                vol_high_thresh=vol_high_thresh,
                vol_low_thresh=vol_low_thresh,
                timeframe=timeframe,
                use_mtf_trend_filter=use_mtf_trend_filter,
            )

            skill = bt.evaluate_prediction_skill(signal_df, hard_stop_loss=hard_stop)
            bt.print_skill_report(skill, code)
            if require_skill and not skill.get('tradable', False):
                reason = skill.get('error') or '; '.join(skill.get('不通过原因', []))
                raise ValueError(f"技能体检拦截: {reason}")

            backtest_df, trades = bt.run_backtest_engine(
                signal_df, hard_stop_loss=hard_stop,
                trailing_stop=trail_stop, max_holding_days=max_hold,
                min_lot_guarantee=min_lot_guarantee,
                risk_per_trade=risk_per_trade,
                max_position_pct=max_position, intraday_stop=intraday_stop,
                volume_participation_cap=volume_participation_cap,
                slippage_model=slippage_model,
                slippage_impact_coef=slippage_impact_coef,
                max_slippage=max_slippage,
                intraday_exit_at_eod=intraday_exit_at_eod,
                overnight_gap_protection=overnight_gap_protection,
            )

            metrics = bt.evaluate_metrics(backtest_df, trades)
            metrics['_skill'] = skill
            metrics['timeframe'] = timeframe
            metrics['slippage_model'] = slippage_model

            # 隔离落盘全套量化分析数据集 (trades, daily_equity, daily_backtest, signals, metrics)
            bt.export_backtest_artifacts(
                stock_code=code,
                output_dir=stock_out,
                signal_df=signal_df,
                backtest_df=backtest_df,
                trades=trades,
                metrics=metrics,
                params={
                    'hard_stop': hard_stop,
                    'trail_stop': trail_stop,
                    'max_hold': max_hold,
                    'min_lot_guarantee': min_lot_guarantee,
                    'risk_per_trade': risk_per_trade,
                    'max_position': max_position,
                    'intraday_stop': intraday_stop,
                    'timeframe': timeframe,
                    'volume_participation_cap': volume_participation_cap,
                    'slippage_model': slippage_model,
                    'intraday_exit_at_eod': intraday_exit_at_eod,
                    'overnight_gap_protection': overnight_gap_protection,
                    'use_mtf_trend_filter': use_mtf_trend_filter,
                }
            )

            if generate_plots:
                try:
                    bt.plot_backtest_results(
                        signal_df, backtest_df, metrics, trades,
                        code, stock_out, show_plot=False,
                    )
                except Exception as e:
                    print(f"   ⚠️ 图表异常: {e}")

            bt._print_report(code, metrics, trades)

            return {
                'stock_code': code, 'stock_name': name,
                'market': bt.profile.market,
                'currency': bt.profile.currency,
                'currency_symbol': bt.profile.currency_symbol,
                'status': 'SUCCESS', 'metrics': metrics,
                'skill': skill, 'trades_count': len(trades),
                'stock_out_dir': stock_out,
            }
        finally:
            bt.cleanup_logger()

    # ---- 步骤 3: 批量调度 ----

    def run_batch_backtest(self, valid_data_pool, data_dir, pred_dir,
                           backtest_batch_dir, parallel_stocks=1,
                           progress_callback=None, **kwargs):
        codes = list(valid_data_pool.keys())
        total = len(codes)
        print("\n" + "=" * 75)
        print(f"📈 [2/4] 批量回测 (标的: {total} 只, 并发: {parallel_stocks})")
        print("=" * 75)
        if parallel_stocks > 1:
            print(f"💡 [算力调度] 并发标的数: {parallel_stocks}。深度学习 GPU 推理阶段已启用线程安全互斥保护（避免 Apple Silicon MPS/Metal 驱动断言冲突），回测撮合与归因分析并发执行。")

        ensure_output_directory(backtest_batch_dir)
        ensure_output_directory(kwargs.get('pred_dir', pred_dir))
        results = {}
        t0 = time.time()

        def _fail_result(c, si, err):
            mkt = si.get('market')
            prof = MarketProfile.for_stock(c, market=mkt, stock_name=si.get('stock_name'))
            return {
                'stock_code': c,
                'stock_name': si['stock_name'],
                'market': prof.market,
                'currency': prof.currency,
                'currency_symbol': prof.currency_symbol,
                'status': 'FAILED',
                'error': str(err), 'metrics': {}, 'skill': {},
            }

        if parallel_stocks <= 1:
            for i, c in enumerate(codes):
                si = valid_data_pool[c]
                ts = time.time()
                try:
                    r = self.run_single_stock(
                        stock_info=si, data_dir=data_dir,
                        pred_dir=pred_dir,
                        backtest_batch_dir=backtest_batch_dir, **kwargs)
                    results[c] = r
                    dur = time.time() - ts
                    print(f"   ✅ [{i+1}/{total}] {si['stock_name']}({c}) "
                          f"耗时 {dur:.1f}s")
                    if progress_callback:
                        try:
                            progress_callback(i + 1, total, c, si['stock_name'], 'SUCCESS', dur, r)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"   ❌ [{i+1}/{total}] {si['stock_name']}({c}): {e}")
                    results[c] = _fail_result(c, si, e)
                    if progress_callback:
                        try:
                            progress_callback(i + 1, total, c, si['stock_name'], 'FAILED', time.time() - ts, results[c])
                        except Exception:
                            pass
        else:
            def _task(c):
                si = valid_data_pool[c]
                ts = time.time()
                try:
                    res = self.run_single_stock(
                        stock_info=si, data_dir=data_dir,
                        pred_dir=pred_dir,
                        backtest_batch_dir=backtest_batch_dir, **kwargs)
                    return c, res, None, time.time() - ts
                except Exception as ex:
                    return c, None, str(ex), time.time() - ts

            with ThreadPoolExecutor(max_workers=parallel_stocks) as pool:
                futs = {pool.submit(_task, c): c for c in codes}
                done = 0
                for fut in as_completed(futs):
                    c, res, err, dur = fut.result()
                    done += 1
                    si = valid_data_pool[c]
                    if res:
                        results[c] = res
                        print(f"   ✅ [{done}/{total}] {si['stock_name']}({c}) 耗时 {dur:.1f}s")
                        if progress_callback:
                            try:
                                progress_callback(done, total, c, si['stock_name'], 'SUCCESS', dur, res)
                            except Exception:
                                pass
                    else:
                        print(f"   ❌ [{done}/{total}] {si['stock_name']}({c}): {err}")
                        results[c] = _fail_result(c, si, err)
                        if progress_callback:
                            try:
                                progress_callback(done, total, c, si['stock_name'], 'FAILED', dur, results[c])
                            except Exception:
                                pass

        ok = sum(1 for r in results.values() if r.get('status') == 'SUCCESS')
        print(f"\n🎉 批量回测完毕: 成功 {ok}, 失败 {total-ok}, "
              f"总耗时 {time.time()-t0:.1f}s")
        return results

    # ---- 步骤 4: 排行榜 ----

    def generate_ranking_summary(self, results, output_dir):
        print("\n" + "=" * 75)
        print("🏆 [3/4] 生成量化回测绩效排行榜...")
        print("=" * 75)
        ensure_output_directory(output_dir)

        rows = []
        for code, item in results.items():
            name = item.get('stock_name', code)
            market = item.get('market', 'CN')
            currency = item.get('currency', 'CNY')
            cs = item.get('currency_symbol', '¥')
            status = item.get('status', 'FAILED')
            m = item.get('metrics', {})
            sk = item.get('skill', {})

            if status != 'SUCCESS' or not m:
                rows.append({
                    '股票代码': code, '股票名称': name, '市场': market,
                    '结算币种': currency, '期末资产': 'N/A',
                    '回测状态': '失败',
                    '累计收益率(%)': np.nan, '年化收益率(%)': np.nan,
                    '夏普比率': np.nan, '卡玛比率': np.nan,
                    '最大回撤(%)': np.nan, '胜率(%)': np.nan,
                    '盈亏比': np.nan, '交易笔数': 0,
                    '单笔均收益(%)': np.nan, '技能得分': 0.0,
                    '体检建议': f"失败: {item.get('error', '未知')}",
                })
                continue

            tradable = sk.get('tradable', False)
            final_cap = m.get('期末总资产', 0.0)
            tf_val = m.get('timeframe', '1d')
            rows.append({
                '股票代码': code, '股票名称': name, '市场': market,
                '周期': tf_val,
                '结算币种': currency,
                '期末资产': f"{cs}{final_cap:,.2f}",
                '回测状态': '成功',
                '累计收益率(%)': round(m.get('累计收益率', 0) * 100, 2),
                '年化收益率(%)': round(m.get('年化收益率', 0) * 100, 2),
                '夏普比率': round(m.get('夏普比率', 0), 2),
                '卡玛比率': round(m.get('卡玛比率', 0), 2),
                '最大回撤(%)': round(m.get('最大回撤', 0) * 100, 2),
                '胜率(%)': round(m.get('胜率', 0) * 100, 2),
                '盈亏比': round(m.get('盈亏比', 0), 2),
                '交易笔数': m.get('总交易笔数', 0),
                '单笔均收益(%)': round(m.get('单笔平均盈亏', 0) * 100, 2),
                '技能得分': round(sk.get('skill_score', 0), 1),
                '体检建议': "可实盘 🟢" if tradable else "需谨慎 🟡",
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('年化收益率(%)', ascending=False,
                            na_position='last').reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = '排名'

        out_file = os.path.join(output_dir, "batch_backtest_ranking.csv")
        df.to_csv(out_file, encoding='utf-8-sig')

        print("\n" + "=" * 135)
        print("🏆 批量量化回测 — 绩效排行榜 (年化收益率降序)")
        print("=" * 135)
        cols = ['股票代码', '股票名称', '市场', '结算币种', '期末资产', '累计收益率(%)',
                '年化收益率(%)', '夏普比率', '卡玛比率', '最大回撤(%)',
                '胜率(%)', '交易笔数', '体检建议']
        print(df[cols].to_string())
        print("=" * 135)
        print(f"💾 排行榜已保存: {out_file}")

        # 多市场分盘汇总统计
        if not df.empty and '市场' in df.columns:
            print("\n🌐 多市场分盘绩效汇总统计:")
            for mkt in ['CN', 'HK', 'US']:
                sub_df = df[(df['市场'] == mkt) & (df['回测状态'] == '成功')]
                if not sub_df.empty:
                    mkt_name = {'CN': 'A股', 'HK': '港股', 'US': '美股'}.get(mkt, mkt)
                    avg_cum = sub_df['累计收益率(%)'].mean()
                    avg_ann = sub_df['年化收益率(%)'].mean()
                    avg_dd = sub_df['最大回撤(%)'].mean()
                    avg_wr = sub_df['胜率(%)'].mean()
                    total_trades = sub_df['交易笔数'].sum()
                    print(f"  [{mkt_name}] 标的数: {len(sub_df)}只 | 平均累计: {avg_cum:+.2f}% | "
                          f"平均年化: {avg_ann:+.2f}% | 平均最大回撤: {avg_dd:.2f}% | "
                          f"平均胜率: {avg_wr:.1f}% | 总交易: {int(total_trades)}笔")
        return df

    def export_visualizations_summary(self, results, output_dir):
        print("\n" + "=" * 75)
        print("🎨 [4/4] 回测图表与明细汇总")
        print("=" * 75)
        for code, item in results.items():
            if item.get('status') == 'SUCCESS':
                print(f"   📊 [{item.get('market')}] "
                      f"{item.get('stock_name')}({code}) -> "
                      f"{item.get('stock_out_dir', '')}")


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_BATCH_CONFIG = {
    "stock_pool": [
        {"stock_code": "300418", "stock_name": "昆仑万维", "market": "CN"},
        {"stock_code": "600036", "stock_name": "招商银行", "market": "CN"},
        {"stock_code": "00700",  "stock_name": "腾讯控股", "market": "HK"},
        {"stock_code": "09988",  "stock_name": "阿里巴巴-W", "market": "HK"},
        {"stock_code": "AAPL",   "stock_name": "苹果公司", "market": "US"},
        {"stock_code": "TSLA",   "stock_name": "特斯拉",   "market": "US"},
    ],
    "lookback_bars": 90, "test_bars": 120,
    "lookback": 90, "test_days": 120, "step_size": 1,
    "samples": 35, "history_days": 700, "horizon": "auto",
    "max_auto_horizon": 10, "force_refresh": False,
    "capital": None, "z_threshold": 0.55,
    "hard_stop": 0.04, "trail_stop": 0.025, "max_hold": 15,
    "min_lot_guarantee": True, "entry_percentile": 0.80, "calib_window": 60,
    "dynamic_calib": True, "calib_window_min": 20,
    "risk_per_trade": 0.01, "max_position": 0.80,
    "require_skill": False,
    "max_workers": 6, "parallel_stocks": 1,
}


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def cli_main():
    parser = argparse.ArgumentParser(
        description="Kronos 批量股票量化回测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 股票池
    parser.add_argument("stocks", nargs="*", default=None,
                        help="股票代码 (例: 300418 600036 AAPL 00700)")
    parser.add_argument("--stock-file", type=str, default=None,
                        help="股票池文件 (.csv/.txt/.json)")
    parser.add_argument("--stocks-inline", type=str, default=None,
                        help="内联格式: '300418:昆仑万维:CN,00700:腾讯控股:HK,AAPL:苹果公司:US'")
    # 模式
    parser.add_argument("--auto", action="store_true", default=True,
                        help="自动 Walk-Forward 预测+回测 (默认)")
    parser.add_argument("--no-auto", action="store_false", dest="auto")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅预检行情数据，不加载模型")
    # 预测参数 (以 K 线根数 Bar 数量为绝对核心标准)
    parser.add_argument("--test-bars", "--test-days", dest="test_bars", type=int, default=None,
                        help="回测验证窗口 K线根数 (默认日线120，30m自适应200Bar)")
    parser.add_argument("--lookback-bars", "--lookback", dest="lookback_bars", type=int, default=None,
                        help="预测输入历史窗口 K线根数 (默认日线90，30m自适应160Bar)")
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--samples", type=int, default=35)
    parser.add_argument("--history-days", type=int, default=None)
    parser.add_argument("--horizon", type=str, default="auto")
    parser.add_argument("--max-auto-horizon", type=int, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    # 策略参数
    parser.add_argument("--capital", type=float, default=None,
                        help="初始资金（缺省根据股票所在市场自动匹配：CN=10万, HK=30万, US=5万）")
    parser.add_argument("--z-threshold", type=float, default=None,
                        help="入场信噪比门槛（默认按周期自适应：1d=0.55, 30m=0.75, 1m=1.05）")
    parser.add_argument("--hard-stop", type=float, default=None,
                        help="基准硬止损线（默认按周期自适应：1d=4.0%, 30m=2.2%, 1m=0.8%）")
    parser.add_argument("--trail-stop", type=float, default=None,
                        help="移动止盈回撤（默认按周期自适应：1d=2.5%, 30m=1.6%, 1m=0.5%）")
    parser.add_argument("--max-hold", type=int, default=None,
                        help="最大持仓Bar数（默认按周期自适应：1d=15天, 30m=24Bar, 1m=45Bar）")
    parser.add_argument("--entry-percentile", type=float, default=None,
                        help="入场自身分位阈值（默认按周期自适应：1d=0.80, 30m=0.82, 1m=0.90）")
    parser.add_argument("--calib-window", type=int, default=None,
                        help="基准滚动校准窗口（默认按周期自适应：1d=60, 30m=20, 1m=5）")
    parser.add_argument("--no-dynamic-calib", action="store_true",
                        help="关闭波动率自适应校准窗口（默认开启）")
    parser.add_argument("--calib-window-min", type=int, default=None,
                        help="极端高波动率下校准窗口收缩下限（默认按周期自适应）")
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position", type=float, default=0.80)
    parser.add_argument("--no-min-lot-guarantee", action="store_true",
                        help="关闭一手保底机制（默认开启，港美股高单价标的自适应）")
    parser.add_argument("--no-intraday-stop", action="store_true",
                        help="关闭盘中止损（默认开启，遵守市场 T+N 合规）")
    parser.add_argument("--require-skill", action="store_true")
    # 高频与微观结构
    parser.add_argument("--timeframe", type=str, default="1d",
                        choices=["1d", "60m", "30m", "15m", "5m", "1m"],
                        help="回测K线颗粒度 (1d/60m/30m/15m/5m/1m，默认1d)")
    parser.add_argument("--volume-cap", type=float, default=None,
                        help="单 Bar 最大成交量占比上限 (如 0.10 表示 10%%)")
    parser.add_argument("--slippage-model", type=str, default=None,
                        choices=["fixed", "dynamic"],
                        help="滑点模型 (默认按周期自适应: 1d/60m=fixed, 30m/1m=dynamic)")
    parser.add_argument("--intraday-exit-eod", action="store_true", default=None,
                        help="日内收盘强平 (不留隔夜仓，T+0市场小周期默认自适应启用)")
    parser.add_argument("--no-mtf-filter", action="store_true",
                        help="关闭跨周期日线宏观趋势滤网 (默认开启)")
    parser.add_argument("--no-auto-adapt", action="store_true", default=False,
                        help="关闭周期最佳参数自适应 (强制使用日线固定参数)")
    # 并发与目录
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--parallel-stocks", type=int, default=1,
                        help="股票回测并发数 (默认1=串行)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--data-dir", type=str,
                        default=resolve_data_dir(),
                        help="行情数据目录（默认优先 backtest/data，回退 examples/data）")
    parser.add_argument("--pred-dir", type=str,
                        default=resolve_pred_dir(),
                        help="预测数据目录（默认优先 backtest/yuce，回退 examples/yuce）")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(CURRENT_DIR, "batch"),
                        help="回测结果输出目录（默认 backtest/batch）")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备 (auto/mps/cuda/cpu，默认自动检测，Mac苹果芯片优先mps)")
    parser.add_argument("--no-log", action="store_true",
                        help="关闭本地文件日志落盘（仅控制台输出，不生成 .log 文件）")
    parser.add_argument("--clear-logs", action="store_true",
                        help="清空批量回测日志文件并退出")

    args = parser.parse_args()

    if args.clear_logs:
        cleared = clear_logs(log_dir=args.output_dir, clear_all=True)
        print(f"🧹 已清空批量回测日志 (共 {len(cleared)} 个文件):")
        for p in cleared:
            print(f"   - {p}")
        sys.exit(0)

    if args.no_log:
        set_logging_state(enabled=True, file_logging=False, log_dir=args.output_dir)
        print("📝 日志文件落盘已关闭 (--no-log)，仅终端输出控制台信息。")
    else:
        set_logging_state(enabled=True, file_logging=True, log_dir=args.output_dir)

    # 确定股票池
    pool = []
    if args.stock_file:
        print(f"📖 从文件加载股票池: {args.stock_file}")
        pool = load_stock_pool_from_file(args.stock_file)
    elif args.stocks_inline:
        pool = [parse_stock_item(it) for it in args.stocks_inline.split(',')
                if it.strip()]
    elif args.stocks:
        pool = [parse_stock_item(s) for s in args.stocks]
    else:
        print("💡 使用默认推荐股票池...")
        pool = DEFAULT_BATCH_CONFIG["stock_pool"]

    if not pool:
        print("❌ 未解析出有效标的"); sys.exit(1)

    horizon = args.horizon
    if isinstance(horizon, str) and horizon.isdigit():
        horizon = int(horizon)

    print("=" * 75)
    print("  Kronos 批量量化回测系统 (Batch Engine)")
    print("=" * 75)
    print(f"  标的数  : {len(pool)} 只")
    print(f"  模式    : {'DRY RUN 预检' if args.dry_run else ('AUTO 预测+回测' if args.auto else 'MANUAL 现有文件回测')}")
    print(f"  时界    : {horizon}")
    # 周期参数自适应解析
    tf_resolved = resolve_timeframe_params(
        timeframe=args.timeframe,
        market='CN',
        overrides={
            'test_bars': args.test_bars,
            'lookback_bars': args.lookback_bars,
            'history_days': args.history_days,
            'step_size': args.step_size,
            'max_auto_horizon': args.max_auto_horizon,
            'z_threshold': args.z_threshold,
            'hard_stop': args.hard_stop,
            'trail_stop': args.trail_stop,
            'max_hold': args.max_hold,
            'entry_percentile': args.entry_percentile,
            'calib_window': args.calib_window,
            'calib_window_min': args.calib_window_min,
            'volume_cap': args.volume_cap,
            'slippage_model': args.slippage_model,
            'intraday_exit_eod': True if args.intraday_exit_eod else None,
            'no_mtf_filter': args.no_mtf_filter,
        },
        auto_adapt=not args.no_auto_adapt,
    )
    tf_obj = TimeFrame.from_str(args.timeframe)
    test_bars = tf_resolved['test_bars']
    lookback_bars = tf_resolved['lookback_bars']
    history_days = tf_resolved['history_days']
    max_auto_horizon = tf_resolved['max_auto_horizon']
    step_size = tf_resolved.get('step_size', 1)

    print("=" * 75)
    print("  Kronos 批量量化回测系统 (Batch Engine)")
    print("=" * 75)
    print(f"  标的数  : {len(pool)} 只")
    print(f"  周期    : {tf_obj.label} ({args.timeframe})")
    print(f"  模式    : {'DRY RUN 预检' if args.dry_run else ('AUTO 预测+回测' if args.auto else 'MANUAL 现有文件回测')}")
    print(f"  时界    : {horizon} (最大搜索: {max_auto_horizon} 根 Bar)")
    print(f"  回测窗口: {test_bars} 根 Bar | 预测输入: {lookback_bars} 根 Bar | 历史拉取: {history_days} 天")
    print(f"  输出    : {args.output_dir}")
    print("=" * 75)

    if not args.no_auto_adapt and args.timeframe != "1d":
        print(f"🤖 [周期自适应] 已为 {tf_obj.label} ({args.timeframe}) 自动配置量化最佳实践参数:")
        print(f"   止损: {tf_resolved['hard_stop']:.2%} | 移动止盈: {tf_resolved['trail_stop']:.2%} | 持仓: {tf_resolved['max_hold']} Bar")
        print(f"   信噪比: {tf_resolved['z_threshold']} | 分位: {tf_resolved['entry_percentile']:.2f} | 滑点模型: {tf_resolved['slippage_model']} | MTF共振: {tf_resolved['use_mtf_trend_filter']}")
        print(f"   K线窗口: 回测={test_bars} 根 Bar | 预测输入={lookback_bars} 根 Bar | 历史获取={history_days} 天 | 推演={max_auto_horizon} Bar")

    engine = BatchBacktester(log_dir=args.output_dir, device=args.device)

    valid = engine.prepare_data_pool(
        stock_list=pool, data_dir=args.data_dir,
        history_days=history_days, lookback=lookback_bars,
        test_days=test_bars, force_refresh=args.force_refresh,
        max_workers=args.max_workers,
        timeframe=args.timeframe,
        lookback_bars=lookback_bars, test_bars=test_bars,
    )
    if not valid:
        print("❌ 无标的通过预检"); sys.exit(1)
    if args.dry_run:
        print("\n🔍 [DRY RUN] 预检完成，退出。"); return

    results = engine.run_batch_backtest(
        valid_data_pool=valid, data_dir=args.data_dir,
        pred_dir=args.pred_dir, backtest_batch_dir=args.output_dir,
        parallel_stocks=args.parallel_stocks,
        auto_predict=args.auto, test_days=test_bars,
        lookback=lookback_bars, step_size=step_size,
        sample_count=args.samples, forecast_horizon=horizon,
        max_auto_horizon=max_auto_horizon,
        force_refresh=args.force_refresh,
        test_bars=test_bars, lookback_bars=lookback_bars,
        initial_capital=args.capital, z_threshold=args.z_threshold,
        hard_stop=args.hard_stop, trail_stop=args.trail_stop,
        max_hold=args.max_hold,
        min_lot_guarantee=not args.no_min_lot_guarantee,
        entry_percentile=args.entry_percentile,
        calib_window=args.calib_window,
        dynamic_calib=not args.no_dynamic_calib,
        calib_window_min=args.calib_window_min,
        risk_per_trade=args.risk_per_trade,
        max_position=args.max_position,
        intraday_stop=not args.no_intraday_stop,
        require_skill=args.require_skill,
        generate_plots=not args.no_plots,
        enable_log=not args.no_log,
        timeframe=args.timeframe,
        volume_participation_cap=args.volume_cap,
        slippage_model=args.slippage_model,
        intraday_exit_at_eod=args.intraday_exit_eod,
        use_mtf_trend_filter=not args.no_mtf_filter,
        auto_adapt_params=not args.no_auto_adapt,
    )

    engine.generate_ranking_summary(results, args.output_dir)
    engine.export_visualizations_summary(results, args.output_dir)


def main():
    """自适应主入口：自动区分 Streamlit / CLI 环境"""
    try:
        from streamlit.runtime import exists as streamlit_exists
        in_streamlit = streamlit_exists()
    except Exception:
        in_streamlit = False

    if in_streamlit:
        from app_backtest_dashboard import streamlit_batch_main
        streamlit_batch_main()
    else:
        cli_main()


if __name__ == "__main__":
    main()
