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
   - 预测缓存共享 examples/yuce/，与单股 --auto 模式完全复用；
   - 回测输出隔离于 examples/backtest_batch/{stock_code}/。
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

socket.setdefaulttimeout(10.0)
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# 行情与预测缓存仍在 examples/，与预测脚本共用，不随回测脚本搬走
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")
for p in [PROJECT_ROOT, CURRENT_DIR, EXAMPLES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_backtest_kronos_pro import (
    KronosDataFetcher,
    KronosBacktesterPro,
    MarketProfile,
    SymbolResolver,
    setup_logger,
)

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
                          lookback=120, test_days=120, force_refresh=False,
                          max_workers=6, progress_callback=None):
        print("\n" + "=" * 75)
        print(f"📡 [1/4] 并发准备历史行情 (目标: {len(stock_list)} 只, 线程: {max_workers})")
        print("=" * 75)
        ensure_output_directory(data_dir)
        fetcher = self._get_fetcher()
        min_days = lookback + test_days

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
                )
                if hist_df is None or len(hist_df) == 0:
                    return norm_code, name, None, det_mkt, "行情为空"
                if len(hist_df) < min_days:
                    return norm_code, name, None, det_mkt, \
                        f"日K线({len(hist_df)}日)不足{min_days}日"
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
                         test_days=120, lookback=120, step_size=1,
                         sample_count=20, forecast_horizon="auto",
                         max_auto_horizon=10, force_refresh=False,
                         min_down_quantile=0.10,
                         initial_capital=None, z_threshold=0.70,
                         hard_stop=0.04, trail_stop=0.025, max_hold=15,
                         min_lot_guarantee=True, entry_percentile=0.80,
                         calib_window=60, risk_per_trade=0.01,
                         max_position=0.80, intraday_stop=True,
                         require_skill=False, generate_plots=True):
        code = stock_info['stock_code']
        name = stock_info['stock_name']
        market = stock_info.get('market')
        hist_df = stock_info.get('hist_df')

        stock_out = os.path.join(backtest_batch_dir, code)
        ensure_output_directory(stock_out)

        prof = MarketProfile.for_stock(code, market=market, stock_name=name)
        cap = initial_capital if initial_capital is not None else prof.default_capital

        print(f"\n{'='*70}")
        print(f"🚀 回测: [{prof.market}] {name} ({code}) | 本金: {prof.currency_symbol}{cap:,.2f} ({prof.currency})")
        print(f"   规则: {prof.describe()}")
        print(f"   输出: {stock_out}")
        print(f"{'='*70}")

        fetcher = self._get_fetcher()
        pred_df = None

        # 预测阶段
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
            )

        # 回测引擎
        bt = KronosBacktesterPro(
            data_dir=data_dir, model_dir=pred_dir,
            initial_capital=cap,
        )
        bt.set_market(code, market=market, stock_name=name)

        if hist_df is None or pred_df is None:
            hist_df, pred_df = bt.load_data(code)

        signal_df = bt.align_and_generate_signals(
            hist_df, pred_df, z_score_threshold=z_threshold,
            forecast_horizon=forecast_horizon,
            max_auto_horizon=max_auto_horizon,
            hard_stop_loss=hard_stop,
            entry_percentile=entry_percentile,
            calib_window=calib_window,
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
        )

        metrics = bt.evaluate_metrics(backtest_df, trades)
        metrics['_skill'] = skill

        # 隔离落盘
        if trades:
            pd.DataFrame(trades).to_csv(
                os.path.join(stock_out, f"{code}_trades.csv"),
                index=False, encoding='utf-8-sig')
        backtest_df.to_csv(
            os.path.join(stock_out, f"{code}_daily_backtest.csv"),
            encoding='utf-8-sig')

        m_clean = {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v)
                   for k, v in metrics.items() if k != '_skill'}
        m_clean['skill_score'] = skill.get('skill_score', 0.0)
        m_clean['tradable'] = skill.get('tradable', False)
        m_clean['market'] = bt.profile.market
        m_clean['currency'] = bt.profile.currency
        with open(os.path.join(stock_out, f"{code}_metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(m_clean, f, ensure_ascii=False, indent=2)

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

    # ---- 步骤 3: 批量调度 ----

    def run_batch_backtest(self, valid_data_pool, data_dir, pred_dir,
                           backtest_batch_dir, parallel_stocks=1,
                           progress_callback=None, **kwargs):
        codes = list(valid_data_pool.keys())
        total = len(codes)
        print("\n" + "=" * 75)
        print(f"📈 [2/4] 批量回测 (标的: {total} 只, 并发: {parallel_stocks})")
        print("=" * 75)

        ensure_output_directory(backtest_batch_dir)
        ensure_output_directory(kwargs.get('pred_dir', pred_dir))
        results = {}
        t0 = time.time()

        def _fail_result(c, si, err):
            return {
                'stock_code': c,
                'stock_name': si['stock_name'],
                'market': si.get('market', 'UNKNOWN'),
                'currency': 'CNY', 'status': 'FAILED',
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
            rows.append({
                '股票代码': code, '股票名称': name, '市场': market,
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
    "lookback": 120, "test_days": 120, "step_size": 1,
    "samples": 20, "history_days": 700, "horizon": "auto",
    "max_auto_horizon": 10, "force_refresh": False,
    "capital": None, "z_threshold": 0.70,
    "hard_stop": 0.04, "trail_stop": 0.025, "max_hold": 15,
    "min_lot_guarantee": True, "entry_percentile": 0.80, "calib_window": 60,
    "risk_per_trade": 0.01, "max_position": 0.80,
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
    # 预测参数
    parser.add_argument("--test-days", type=int, default=120)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--history-days", type=int, default=700)
    parser.add_argument("--horizon", type=str, default="auto")
    parser.add_argument("--max-auto-horizon", type=int, default=10)
    parser.add_argument("--force-refresh", action="store_true")
    # 策略参数
    parser.add_argument("--capital", type=float, default=None,
                        help="初始资金（缺省根据股票所在市场自动匹配：CN=10万, HK=30万, US=5万）")
    parser.add_argument("--z-threshold", type=float, default=0.70)
    parser.add_argument("--hard-stop", type=float, default=0.04)
    parser.add_argument("--trail-stop", type=float, default=0.025)
    parser.add_argument("--max-hold", type=int, default=15)
    parser.add_argument("--entry-percentile", type=float, default=0.80)
    parser.add_argument("--calib-window", type=int, default=60)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position", type=float, default=0.80)
    parser.add_argument("--no-min-lot-guarantee", action="store_true",
                        help="关闭一手保底机制（默认开启，港美股高单价标的自适应）")
    parser.add_argument("--no-intraday-stop", action="store_true",
                        help="关闭盘中止损（默认开启，遵守市场 T+N 合规）")
    parser.add_argument("--require-skill", action="store_true")
    # 并发与目录
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--parallel-stocks", type=int, default=1,
                        help="股票回测并发数 (默认1=串行)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(EXAMPLES_DIR, "data"))
    parser.add_argument("--pred-dir", type=str,
                        default=os.path.join(EXAMPLES_DIR, "yuce"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(EXAMPLES_DIR, "backtest_batch"))

    args = parser.parse_args()

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
    print(f"  回测    : {args.test_days} 交易日 | 回看: {args.lookback} 交易日")
    print(f"  输出    : {args.output_dir}")
    print("=" * 75)

    engine = BatchBacktester(log_dir=args.output_dir)

    valid = engine.prepare_data_pool(
        stock_list=pool, data_dir=args.data_dir,
        history_days=args.history_days, lookback=args.lookback,
        test_days=args.test_days, force_refresh=args.force_refresh,
        max_workers=args.max_workers,
    )
    if not valid:
        print("❌ 无标的通过预检"); sys.exit(1)
    if args.dry_run:
        print("\n🔍 [DRY RUN] 预检完成，退出。"); return

    results = engine.run_batch_backtest(
        valid_data_pool=valid, data_dir=args.data_dir,
        pred_dir=args.pred_dir, backtest_batch_dir=args.output_dir,
        parallel_stocks=args.parallel_stocks,
        auto_predict=args.auto, test_days=args.test_days,
        lookback=args.lookback, step_size=args.step_size,
        sample_count=args.samples, forecast_horizon=horizon,
        max_auto_horizon=args.max_auto_horizon,
        force_refresh=args.force_refresh,
        initial_capital=args.capital, z_threshold=args.z_threshold,
        hard_stop=args.hard_stop, trail_stop=args.trail_stop,
        max_hold=args.max_hold,
        min_lot_guarantee=not args.no_min_lot_guarantee,
        entry_percentile=args.entry_percentile,
        calib_window=args.calib_window,
        risk_per_trade=args.risk_per_trade,
        max_position=args.max_position,
        intraday_stop=not args.no_intraday_stop,
        require_skill=args.require_skill,
        generate_plots=not args.no_plots,
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
