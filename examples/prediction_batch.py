"""
Kronos 批量池化股票预测系统 (Batch Stock Prediction Engine)
===========================================================
基于 Kronos 基础大模型与国内多源行情接口（AKShare/新浪/东方财富/腾讯），
实现面向多标的股票池的批量、高并发、GPU批推理预测。

核心特性：
1. 模型单例常驻：只在启动时加载一次模型与分词器，显存高度复用；
2. 多线程并发 I/O：多源行情并发拉取，配合时效感知缓存实现秒级命中；
3. 严格时序预检与对齐：统一截取固定 lookback 历史序列与统一未来交易日历，自动过滤次新股与停牌异常；
4. GPU Mini-Batch 批推理：利用 KronosPredictor.predict_batch 实现张量级并行，支持蒙特卡洛置信区间并行解码；
5. 量化选股排行榜：汇总各股票预期收益率、极值收益率与波动风险，按预期涨跌幅降序输出跨股票大盘总览 CSV。
"""

import os
import sys
import time
import socket
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# 全局设置底层网络超时
socket.setdefaulttimeout(8.0)
warnings.filterwarnings('ignore')

# 添加项目路径
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
for p in [PROJECT_ROOT, CURRENT_SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.append(p)

from model import Kronos, KronosTokenizer, KronosPredictor

# 复用成熟的单股数据获取、清洗与绘图辅助模块
try:
    from prediction_akshare_2024_2025_copy import (
        fetch_or_load_stock_data,
        prepare_stock_data,
        generate_future_dates_with_holidays,
        plot_prediction_with_details,
        resolve_date_range,
        ensure_output_directory,
        detect_market_type,
        validate_stock_code,
        MARKET_CURRENCY_MAP
    )
except ImportError:
    # 兼容直接从 prediction_akshare_2024_2025 导入
    from examples.prediction_akshare_2024_2025_copy import (
        fetch_or_load_stock_data,
        prepare_stock_data,
        generate_future_dates_with_holidays,
        plot_prediction_with_details,
        resolve_date_range,
        ensure_output_directory,
        detect_market_type,
        validate_stock_code,
        MARKET_CURRENCY_MAP
    )


class BatchStockPredictor:
    """
    批量股票池化预测引擎
    """

    def __init__(self, model_name="NeoQuasar/Kronos-base",
                 tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                 device=None, max_context=512):
        """
        初始化预测器，加载模型至 GPU/MPS/CPU 并常驻
        """
        print("\n" + "=" * 70)
        print("🧠 [1/5] 初始化 Kronos 大模型引擎 (单例常驻模式)...")
        start_time = time.time()
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        self.model = Kronos.from_pretrained(model_name)
        self.predictor = KronosPredictor(self.model, self.tokenizer, device=device, max_context=max_context)
        print(f"✅ 模型与分词器加载完成 (设备: {self.predictor.device}, 耗时: {time.time() - start_time:.2f}s)")
        print("=" * 70)

    def prepare_data_pool(self, stock_list, data_dir, end_date=None, start_date=None,
                          history_days=365, lookback=120, pred_days=60,
                          force_refresh=False, max_workers=6):
        """
        多线程并发获取股票池行情数据（支持 A股/港股/美股多市场），进行时效感知缓存判定并对齐时序
        """
        print(f"\n📡 [2/5] 多线程并发准备股票行情 (目标标的: {len(stock_list)} 只, 并发线程: {max_workers})...")
        ensure_output_directory(data_dir)

        start_date_str, end_date_str, is_latest_mode = resolve_date_range(
            end_date=end_date, start_date=start_date, history_days=history_days, pred_days=pred_days
        )
        mode_text = "最新收盘行情模式" if is_latest_mode else f"历史回测模式 ({start_date_str} ~ {end_date_str})"
        print(f"   运行时间基准: {mode_text}")

        def _load_single(item):
            raw_code = item.get('stock_code', '')
            raw_market = item.get('market', None)
            name = item.get('stock_name', str(raw_code))

            # 1. 股票代码格式与所属市场合规性前置校验
            is_valid, err_msg, normalized_code, detected_market = validate_stock_code(raw_code, raw_market)
            if not is_valid:
                return str(raw_code), name, None, detected_market or 'UNKNOWN', f"代码校验未通过: {err_msg}"

            code = normalized_code
            try:
                raw_df, _ = fetch_or_load_stock_data(
                    stock_code=code,
                    data_dir=data_dir,
                    start_date=start_date,
                    end_date=end_date,
                    history_days=history_days,
                    pred_days=pred_days,
                    force_refresh=force_refresh,
                    market=detected_market,
                    allow_fallback=False
                )
                if raw_df is None or len(raw_df) == 0:
                    return code, name, None, detected_market, "未获取到行情数据 (代码可能不存在、已退市或网络不可达)"

                df = prepare_stock_data(raw_df, code)
                # 严格历史截断，防止未来信息泄露
                if not is_latest_mode and end_date_str:
                    df = df[df['timestamps'] <= pd.to_datetime(end_date_str)].reset_index(drop=True)
                if start_date_str:
                    df = df[df['timestamps'] >= pd.to_datetime(start_date_str)].reset_index(drop=True)

                if len(df) < lookback:
                    return code, name, None, detected_market, f"历史有效交易日({len(df)}天)不足回看窗口({lookback}天)"

                return code, name, df, detected_market, "OK"
            except Exception as e:
                return code, name, None, detected_market, str(e)

        valid_data_pool = {}
        failed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {executor.submit(_load_single, item): item for item in stock_list}
            for future in as_completed(future_to_stock):
                code, name, df, detected_market, status = future.result()
                if df is not None:
                    # 统一截取最近固定 lookback 个交易日
                    x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
                    x_ts = df.iloc[-lookback:]['timestamps'].reset_index(drop=True)
                    valid_data_pool[code] = {
                        'name': name,
                        'market': detected_market,
                        'full_df': df,
                        'x_df': x_df,
                        'x_timestamp': x_ts,
                        'last_date': df['timestamps'].iloc[-1],
                        'current_price': df['close'].iloc[-1]
                    }
                else:
                    failed_count += 1
                    market_label = {'CN': 'A股', 'HK': '港股', 'US': '美股'}.get(detected_market, '')
                    print(f"   ⚠️ 剔除标的 [{market_label}]{name}({code}): {status}")

        # 汇总各市场标的数量
        market_counts = {}
        for item in valid_data_pool.values():
            m = item['market']
            market_counts[m] = market_counts.get(m, 0) + 1
        market_summary = ', '.join([f"{'A股' if m=='CN' else '港股' if m=='HK' else '美股'}: {c}只" for m, c in market_counts.items()])
        print(f"✅ 数据准备就绪: 有效合格标的 {len(valid_data_pool)} 只 ({market_summary})，过滤剔除 {failed_count} 只")
        return valid_data_pool

    def run_batch_inference(self, valid_data_pool, pred_len=60, batch_size=8,
                            sample_count=5, temperature=1.0, top_p=0.9):
        """
        分块执行 GPU/MPS 批推理，支持蒙特卡洛多路径采样与置信区间聚合。
        各市场股票使用各自独立的交易日历。
        """
        stock_codes = list(valid_data_pool.keys())
        if not stock_codes:
            print("❌ 没有有效股票可供预测")
            return {}, []

        # 按市场类型分组生成未来交易日（各市场节假日不同）
        market_future_dates = {}
        for code in stock_codes:
            item = valid_data_pool[code]
            market = item.get('market', 'CN')
            if market not in market_future_dates:
                benchmark_date = item['last_date']
                market_future_dates[market] = generate_future_dates_with_holidays(benchmark_date, pred_len, market=market)

        results = {}
        total_stocks = len(stock_codes)
        print(f"\n🚀 [3/5] 开始 GPU/MPS 分块批推理 (共 {total_stocks} 只标的, 单批容量: {batch_size} 只, 蒙特卡洛采样: {sample_count} 条/股)...")

        total_batches = (total_stocks + batch_size - 1) // batch_size

        for b_idx in range(total_batches):
            chunk_codes = stock_codes[b_idx * batch_size:(b_idx + 1) * batch_size]
            b_start_time = time.time()

            dfs = []
            x_stamps = []
            y_stamps = []

            # 针对每只股票展开 sample_count 条并行采样路径
            for code in chunk_codes:
                item = valid_data_pool[code]
                market = item.get('market', 'CN')
                future_dates = market_future_dates[market]
                future_series = pd.Series(future_dates)
                for _ in range(sample_count):
                    dfs.append(item['x_df'])
                    x_stamps.append(item['x_timestamp'])
                    y_stamps.append(future_series)

            # 单次调用底层 predict_batch 进行张量并行推理
            pred_paths = self.predictor.predict_batch(
                df_list=dfs,
                x_timestamp_list=x_stamps,
                y_timestamp_list=y_stamps,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=1,
                verbose=False
            )

            # 按股票重塑并聚合采样轨迹
            for i, code in enumerate(chunk_codes):
                stock_paths = pred_paths[i * sample_count:(i + 1) * sample_count]
                market = valid_data_pool[code].get('market', 'CN')
                future_dates = market_future_dates[market]

                close_matrix = np.stack([p['close'].values for p in stock_paths], axis=0)
                vol_matrix = np.stack([p['volume'].values for p in stock_paths], axis=0)
                open_matrix = np.stack([p['open'].values for p in stock_paths], axis=0)
                high_matrix = np.stack([p['high'].values for p in stock_paths], axis=0)
                low_matrix = np.stack([p['low'].values for p in stock_paths], axis=0)
                amt_matrix = np.stack([p['amount'].values for p in stock_paths], axis=0)

                # 计算 90% 乐观上限与 10% 悲观下限置信区间
                c_upper = np.percentile(close_matrix, 90, axis=0) if sample_count >= 5 else np.max(close_matrix, axis=0)
                c_lower = np.percentile(close_matrix, 10, axis=0) if sample_count >= 5 else np.min(close_matrix, axis=0)

                pred_df = pd.DataFrame({
                    'open': np.mean(open_matrix, axis=0),
                    'high': np.mean(high_matrix, axis=0),
                    'low': np.mean(low_matrix, axis=0),
                    'close': np.mean(close_matrix, axis=0),
                    'volume': np.mean(vol_matrix, axis=0),
                    'amount': np.mean(amt_matrix, axis=0),
                    'close_upper': c_upper,
                    'close_lower': c_lower,
                    'close_std': np.std(close_matrix, axis=0)
                }, index=pd.Series(future_dates))

                results[code] = {
                    'stock_code': code,
                    'stock_name': valid_data_pool[code]['name'],
                    'market': market,
                    'full_df': valid_data_pool[code]['full_df'],
                    'current_price': valid_data_pool[code]['current_price'],
                    'pred_df': pred_df,
                    'sample_paths': stock_paths,
                    'future_dates': future_dates
                }

            elapsed = time.time() - b_start_time
            done_count = min((b_idx + 1) * batch_size, total_stocks)
            print(f"   ⚡ Batch [{b_idx + 1}/{total_batches}] 完成 ({len(chunk_codes)} 只股票, 耗时: {elapsed:.2f}s, 累计进度: {done_count}/{total_stocks})")

        return results, market_future_dates

    def generate_ranking_summary(self, results, output_dir):
        """
        生成量化选股大盘排行榜 (batch_stock_ranking.csv)，支持多市场标识与货币单位
        """
        print(f"\n📊 [4/5] 汇总多标的预测数据并生成量化选股排行榜...")
        ensure_output_directory(output_dir)

        summary_list = []
        for code, item in results.items():
            name = item['stock_name']
            market = item.get('market', 'CN')
            curr_p = item['current_price']
            pred_df = item['pred_df']
            currency_code, currency_unit = MARKET_CURRENCY_MAP.get(market, ('CNY', '元'))

            pred_end_p = pred_df['close'].iloc[-1]
            change_pct = ((pred_end_p / curr_p) - 1) * 100
            pred_max_p = pred_df['high'].max()
            pred_min_p = pred_df['low'].min()
            max_upside = ((pred_max_p / curr_p) - 1) * 100
            max_drawdown = ((pred_min_p / curr_p) - 1) * 100

            upper_p = pred_df['close_upper'].iloc[-1]
            lower_p = pred_df['close_lower'].iloc[-1]

            # 投资吸引力评级
            if change_pct >= 15:
                signal = "强烈看涨 🚀"
            elif change_pct >= 5:
                signal = "温和看涨 📈"
            elif change_pct >= -5:
                signal = "窄幅震荡 ↔️"
            elif change_pct >= -15:
                signal = "温和看跌 📉"
            else:
                signal = "大幅看跌 🔻"

            market_label = {'CN': 'A股', 'HK': '港股', 'US': '美股'}.get(market, market)

            summary_list.append({
                '股票代码': code,
                '股票名称': name,
                '市场': market_label,
                '货币': currency_code,
                '当前基准价': round(curr_p, 2),
                '预测期末价': round(pred_end_p, 2),
                '预期收益率(%)': round(change_pct, 2),
                '置信区间下限': round(lower_p, 2),
                '置信区间上限': round(upper_p, 2),
                '期间最高价': round(pred_max_p, 2),
                '期间最低价': round(pred_min_p, 2),
                '最大潜在涨幅(%)': round(max_upside, 2),
                '最大潜在回撤(%)': round(max_drawdown, 2),
                '模型信号': signal
            })

            # 保存单股票的详细预测 CSV
            market_prefix = f"{market}_" if market in ('HK', 'US') else ""
            stock_detail_file = os.path.join(output_dir, f"{market_prefix}{code}_detailed_predictions.csv")
            pred_detail_export = pred_df.copy()
            pred_detail_export.index.name = '日期'
            pred_detail_export.to_csv(stock_detail_file, encoding='utf-8-sig')

        summary_df = pd.DataFrame(summary_list)
        # 按预期收益率从高到低排序
        summary_df = summary_df.sort_values(by='预期收益率(%)', ascending=False).reset_index(drop=True)
        summary_df.index = summary_df.index + 1
        summary_df.index.name = '排名'

        ranking_file = os.path.join(output_dir, "batch_stock_ranking.csv")
        summary_df.to_csv(ranking_file, encoding='utf-8-sig')

        print("\n" + "=" * 110)
        print("🏆 批量股票预测 — 量化选股排行榜 (按预期收益率降序排序)")
        print("=" * 110)
        display_cols = ['股票代码', '股票名称', '市场', '货币', '当前基准价', '预测期末价', '预期收益率(%)', '最大潜在涨幅(%)', '最大潜在回撤(%)', '模型信号']
        print(summary_df[display_cols].to_string())
        print("=" * 110)
        print(f"💾 大盘选股排行榜已保存至: {ranking_file}")
        return summary_df

    def export_visualizations(self, results, output_dir, top_n=3, generate_all=False):
        """
        生成可视化图表：可仅对 Top N 重点关注股票画图，或全量生成
        """
        print(f"\n🎨 [5/5] 生成走势图表...")
        ensure_output_directory(output_dir)

        # 提取按预期收益率排序的股票代码
        sorted_codes = sorted(
            results.keys(),
            key=lambda c: (results[c]['pred_df']['close'].iloc[-1] / results[c]['current_price'] - 1),
            reverse=True
        )

        target_codes = sorted_codes if generate_all else sorted_codes[:top_n]
        if not generate_all:
            print(f"   (性能优化模式: 仅为收益率 Top {len(target_codes)} 优势标的生成高清走势图表)")

        for code in target_codes:
            item = results[code]
            kline_df = item['full_df'].iloc[-120:].reset_index(drop=True)
            try:
                plot_prediction_with_details(
                    kline_df=kline_df,
                    pred_df=item['pred_df'],
                    future_dates=item['future_dates'],
                    stock_code=code,
                    stock_name=item['stock_name'],
                    pred_len=len(item['pred_df']),
                    output_dir=output_dir,
                    sample_paths=item['sample_paths'],
                    show_plot=False
                )
                print(f"   📊 已保存图表: {code}_{item['stock_name']}_prediction_chart.png")
            except Exception as e:
                print(f"   ⚠️ 图表生成失败 {code}: {e}")

        print("🎉 全部任务已顺利执行完毕！\n")


def main():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ==================== 批量预测配置区 ====================
    BATCH_CONFIG = {
        # 1. 股票池列表配置（支持 A股/港股/美股混合配置，market 字段可选: 'CN'(A股,默认)/'HK'(港股)/'US'(美股)）
        "stock_pool": [
            # --- A股标的 ---
            {"stock_code": "300418", "stock_name": "昆仑万维", "market": "CN"},  # 人工智能
            {"stock_code": "600036", "stock_name": "招商银行", "market": "CN"},  # 权重白马

            # --- 港股标的 ---
            {"stock_code": "00700", "stock_name": "腾讯控股",   "market": "HK"},  # 港股互联网龙头
            {"stock_code": "09988", "stock_name": "阿里巴巴-W", "market": "HK"},  # 港股电商
            {"stock_code": "01810", "stock_name": "小米集团-W", "market": "HK"},  # 港股智能硬件/造车龙头
            {"stock_code": "03308", "stock_name": "中际旭创",   "market": "HK"}, 
            {"stock_code": "03750", "stock_name": "宁德时代",   "market": "HK"},  
            # --- 美股标的 ---
            {"stock_code": "AAPL",  "stock_name": "苹果公司",   "market": "US"},  # 美股科技巨头
            {"stock_code": "TSLA",  "stock_name": "特斯拉",     "market": "US"},  # 美股新能源车
        ],
        # 2. 预测时序参数
        "lookback": 120,          # 统一历史回看交易日数 (Kronos 输入长度)
        "pred_days": 60,          # 预测未来自然日天数 (模型自动折算为交易日)
        "history_days": 365,      # 历史数据抓取天数 (确保有充分交易日)
        "end_date": None,         # 预测基准日 (None: 最新实时行情；可填如 '2024-12-31' 进行历史回测)
        "force_refresh": False,   # 是否强制从网络刷新 (False: 自动时效感知，新鲜直接复用本地缓存)

        # 3. 硬件与推理参数
        "batch_size": 4,          # GPU/MPS 单批处理股票数 (根据显存大小调整，4~16 适中)
        "sample_count": 5,        # 蒙特卡洛采样路径数 (>=2 自动推导 10%~90% 扇形置信区间)
        "temperature": 1.0,       # 采样温度
        "top_p": 0.9,             # 核采样截断阈值

        # 4. 目录与报表控制
        "data_dir": os.path.join(CURRENT_DIR, "data"),
        "output_dir": os.path.join(CURRENT_DIR, "yuce_batch"),
        "top_n_plots": 3,         # 仅为收益率前 N 名的股票生成高清图表 (0 为不生成图表)
        "generate_all_plots": False  # 是否为所有股票都生成图表
    }
    # ========================================================

    predictor_engine = BatchStockPredictor(max_context=512)

    # 步骤 1: 多线程并发准备行情数据
    valid_data = predictor_engine.prepare_data_pool(
        stock_list=BATCH_CONFIG["stock_pool"],
        data_dir=BATCH_CONFIG["data_dir"],
        end_date=BATCH_CONFIG["end_date"],
        history_days=BATCH_CONFIG["history_days"],
        lookback=BATCH_CONFIG["lookback"],
        pred_days=BATCH_CONFIG["pred_days"],
        force_refresh=BATCH_CONFIG["force_refresh"],
        max_workers=6
    )

    if not valid_data:
        print("❌ 未能准备好有效的股票数据，流程终止。")
        return

    # 步骤 2: GPU/MPS 分块批推理与蒙特卡洛采样
    # 计算实际交易日步长 (约占自然日的 70%)
    pred_len = int(BATCH_CONFIG["pred_days"] * 5 / 7)
    results, future_dates = predictor_engine.run_batch_inference(
        valid_data_pool=valid_data,
        pred_len=pred_len,
        batch_size=BATCH_CONFIG["batch_size"],
        sample_count=BATCH_CONFIG["sample_count"],
        temperature=BATCH_CONFIG["temperature"],
        top_p=BATCH_CONFIG["top_p"]
    )

    # 步骤 3: 汇总生成量化选股大盘排行榜与单股明细 CSV
    predictor_engine.generate_ranking_summary(results, BATCH_CONFIG["output_dir"])

    # 步骤 4: 可选导出走势图
    if BATCH_CONFIG["top_n_plots"] > 0 or BATCH_CONFIG["generate_all_plots"]:
        predictor_engine.export_visualizations(
            results,
            BATCH_CONFIG["output_dir"],
            top_n=BATCH_CONFIG["top_n_plots"],
            generate_all=BATCH_CONFIG["generate_all_plots"]
        )


if __name__ == "__main__":
    main()
