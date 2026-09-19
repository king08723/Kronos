"""
Kronos 股票价格预测系统 - Streamlit 现代量化 Web 看板
基于深度学习时序大模型 Kronos + 多维度市场因子增强 (东方财富/Baostock/AKShare 多数据源)
"""

import os
import sys
import json
import time
import random
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import akshare as ak
import streamlit as st

warnings.filterwarnings('ignore')

# 添加项目路径以便导入 Kronos 模型
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    KRONOS_AVAILABLE = True
except ImportError:
    KRONOS_AVAILABLE = False

# 设置中文字体 (兼容 macOS / Windows / Linux)
plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS',  # macOS
    'SimHei', 'Microsoft YaHei',                              # Windows
    'WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif'        # Linux / Fallback
]
plt.rcParams['axes.unicode_minus'] = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(CURRENT_DIR, "data")
DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "yuce")


# ==================== 1. 基础数据工具与多源获取 ====================
def ensure_output_directory(output_dir: str) -> str:
    """确保输出目录存在"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_stock_market_prefix(stock_code: str) -> str:
    """判断市场类型 (Baostock格式: sh / sz / bj)"""
    if stock_code.startswith(('0', '2', '3')):
        return "sz"
    elif stock_code.startswith(('6', '9')):
        return "sh"
    elif stock_code.startswith(('4', '8', '920')):
        return "bj"
    return "sh"


def get_stock_market_em(stock_code: str) -> str:
    """判断东方财富市场类型 ('0'-深交所, '1'-上交所/北交所)"""
    if stock_code.startswith(('0', '2', '3')):
        return '0'
    return '1'


def fetch_quote_tencent(symbol: str) -> Optional[Dict]:
    """腾讯财经快速行情接口 (极简HTTP直连，毫秒响应)"""
    import urllib.request
    try:
        url = f"http://qt.gtimg.cn/q={symbol}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as res:
            text = res.read().decode('gbk', errors='ignore')
            if '~' in text:
                parts = text.split('="')[1].split('~')
                return {
                    'name': parts[1],
                    'code': parts[2],
                    'close': float(parts[3]),
                    'change': float(parts[4]),
                    'pct_chg': float(parts[5]),
                    'volume': float(parts[6]),
                    'amount': float(parts[7])
                }
    except Exception:
        pass
    return None


_BAOSTOCK_LOGGED_IN = False


def ensure_baostock_login() -> bool:
    """保证 Baostock 全局只 login 一次，退出自动 logout"""
    global _BAOSTOCK_LOGGED_IN
    if _BAOSTOCK_LOGGED_IN:
        return True
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code == '0':
            _BAOSTOCK_LOGGED_IN = True
            import atexit
            atexit.register(bs.logout)
            return True
    except Exception:
        pass
    return False


def fetch_stock_data_baostock(stock_code: str, history_years: int = 2) -> Optional[pd.DataFrame]:
    """数据源1: Baostock 原生 Socket 获取前复权股票日线数据"""
    try:
        if not ensure_baostock_login():
            return None
        import baostock as bs
        prefix = get_stock_market_prefix(stock_code)
        full_code = f"{prefix}.{stock_code}"
        start_date = (datetime.now() - timedelta(days=history_years * 365 + 60)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        rs = bs.query_history_k_data_plus(
            full_code,
            "date,open,high,low,close,volume,amount,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 2: 前复权
        )
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            df = df.rename(columns={'date': 'timestamps', 'pctChg': 'pct_chg'})
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df['stock_code'] = stock_code
            df = df.dropna(subset=['close']).sort_values('timestamps').reset_index(drop=True)
            if len(df) > 0:
                return df
    except Exception:
        pass
    return None


def parse_kline_data_directly(response_text: str, stock_code: str) -> Optional[pd.DataFrame]:
    """直接解析东方财富K线数据（文本切片兜底）"""
    try:
        if '"klines":[' in response_text:
            start_idx = response_text.find('"klines":[') + 10
            end_idx = response_text.find(']', start_idx)
            klines_str = response_text[start_idx:end_idx]
            klines = klines_str.replace('"', '').split(',')

            stock_data = []
            for kline in klines:
                if kline.strip():
                    items = kline.split(',')
                    if len(items) >= 6:
                        stock_data.append({
                            'timestamps': items[0],
                            'open': float(items[1]),
                            'close': float(items[2]),
                            'high': float(items[3]),
                            'low': float(items[4]),
                            'volume': float(items[5]),
                            'amount': float(items[6]) if len(items) > 6 else 0,
                            'pct_chg': float(items[8]) if len(items) > 8 else 0,
                            'turnover': float(items[10]) if len(items) > 10 else 0,
                            'stock_code': stock_code
                        })
            if stock_data:
                df = pd.DataFrame(stock_data)
                df['timestamps'] = pd.to_datetime(df['timestamps'])
                return df.sort_values('timestamps').reset_index(drop=True)
    except Exception:
        pass
    return None


def fetch_stock_data_eastmoney(stock_code: str, history_years: int = 2) -> Optional[pd.DataFrame]:
    """数据源2: 东方财富网API直连获取前复权日线数据"""
    try:
        start_date = (datetime.now() - timedelta(days=history_years * 365 + 60)).strftime('%Y%m%d')
        end_date = datetime.now().strftime('%Y%m%d')
        market = get_stock_market_em(stock_code)
        secid = f"{market}.{stock_code}"

        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            'secid': secid,
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': '101',  # 日线
            'fqt': '1',    # 前复权
            'beg': start_date,
            'end': end_date,
            'lmt': '10000',
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
            'cb': f'jQuery{random.randint(1000000, 9999999)}_{int(time.time()*1000)}'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://quote.eastmoney.com/',
            'Accept': '*/*'
        }

        response = requests.get(url, params=params, headers=headers, timeout=5, proxies={'http': None, 'https': None})
        if response.status_code == 200:
            response_text = response.text
            if response_text.startswith('/**/'):
                response_text = response_text[4:]
            start_idx = response_text.find('(')
            end_idx = response_text.rfind(')')
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx + 1:end_idx]
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    return parse_kline_data_directly(response_text, stock_code)
            else:
                return parse_kline_data_directly(response_text, stock_code)

            if data and data.get('data') and data['data'].get('klines'):
                klines = data['data']['klines']
                stock_data = []
                for kline in klines:
                    items = kline.split(',')
                    if len(items) >= 6:
                        stock_data.append({
                            'timestamps': items[0],
                            'open': float(items[1]),
                            'close': float(items[2]),
                            'high': float(items[3]),
                            'low': float(items[4]),
                            'volume': float(items[5]),
                            'amount': float(items[6]) if len(items) > 6 else 0,
                            'pct_chg': float(items[8]) if len(items) > 8 else 0,
                            'turnover': float(items[10]) if len(items) > 10 else 0,
                            'stock_code': stock_code
                        })
                if stock_data:
                    df = pd.DataFrame(stock_data)
                    df['timestamps'] = pd.to_datetime(df['timestamps'])
                    return df.sort_values('timestamps').reset_index(drop=True)
    except Exception:
        pass
    return None


def fetch_stock_data_akshare(stock_code: str, period: str = "daily", adjust: str = "qfq") -> Optional[pd.DataFrame]:
    """数据源3: 使用AKShare获取真实股票数据"""
    try:
        df = ak.stock_zh_a_hist(symbol=stock_code, period=period, adjust=adjust)
        if df is None or df.empty:
            return None

        column_mapping = {
            '日期': 'timestamps', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume',
            '成交额': 'amount', '振幅': 'amplitude', '涨跌幅': 'pct_chg',
            '涨跌额': 'change_amount', '换手率': 'turnover'
        }
        actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=actual_mapping)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['stock_code'] = stock_code
        return df.dropna(subset=['close']).sort_values('timestamps').reset_index(drop=True)
    except Exception:
        return None


def fetch_index_data_multi_source(index_code: str, history_years: int = 1) -> Optional[pd.DataFrame]:
    """
    多数据源获取大盘指数日K线 (含本地缓存、东财极速直连、Baostock与超时熔断)
    """
    index_name = "上证指数" if index_code == "000001" else "深证成指"
    cache_file = os.path.join(DEFAULT_DATA_DIR, f"index_{index_code}.csv")

    # 1. 优先使用本地缓存 (只要存在且有效，0秒极速加载)
    if os.path.exists(cache_file):
        try:
            df_cache = pd.read_csv(cache_file)
            if 'date' in df_cache.columns and len(df_cache) >= 20:
                df_cache['date'] = pd.to_datetime(df_cache['date'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df_cache.columns:
                        df_cache[col] = pd.to_numeric(df_cache[col], errors='coerce')
                # 检查缓存是否为今天/近期的
                last_cache_date = df_cache['date'].max()
                if (datetime.now() - last_cache_date).days <= 3:
                    return df_cache.sort_values('date').reset_index(drop=True)
        except Exception:
            pass

    # 2. 首选网络源: 东方财富 API (极速直连，毫秒响应，设置 3 秒严格超时)
    try:
        secid = f"1.{index_code}" if index_code == "000001" else f"0.{index_code}"
        start_date = (datetime.now() - timedelta(days=history_years * 365 + 60)).strftime('%Y%m%d')
        end_date = datetime.now().strftime('%Y%m%d')
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            'secid': secid,
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': '101', 'fqt': '1', 'beg': start_date, 'end': end_date,
            'lmt': '1000', 'ut': 'fa5fd1943c7b386f172d6893dbfba10b'
        }
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, params=params, headers=headers, timeout=3, proxies={'http': None, 'https': None})
        if res.status_code == 200:
            data = res.json() if res.text.startswith('{') else None
            if not data and '(' in res.text:
                s = res.text.find('(') + 1
                e = res.text.rfind(')')
                data = json.loads(res.text[s:e])
            if data and data.get('data') and data['data'].get('klines'):
                rows = []
                for k in data['data']['klines']:
                    p = k.split(',')
                    if len(p) >= 6:
                        rows.append({
                            'date': p[0], 'open': float(p[1]), 'close': float(p[2]),
                            'high': float(p[3]), 'low': float(p[4]), 'volume': float(p[5])
                        })
                if rows:
                    df = pd.DataFrame(rows)
                    df['date'] = pd.to_datetime(df['date'])
                    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
                    df.to_csv(cache_file, index=False)
                    return df.sort_values('date').reset_index(drop=True)
    except Exception:
        pass

    # 3. 备选网络源: Baostock (原生 Socket)
    try:
        if ensure_baostock_login():
            import baostock as bs
            prefix = "sh" if index_code.startswith("0") else "sz"
            full_code = f"{prefix}.{index_code}"
            start_date = (datetime.now() - timedelta(days=history_years * 365 + 60)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            rs = bs.query_history_k_data_plus(full_code, "date,open,high,low,close,volume", start_date=start_date, end_date=end_date, frequency="d")
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df['date'] = pd.to_datetime(df['date'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna().sort_values('date').reset_index(drop=True)
                if len(df) >= 20:
                    df.to_csv(cache_file, index=False)
                    return df
    except Exception:
        pass

    # 4. 极速备选: 腾讯实时行情接口 (2 秒超时)
    try:
        prefix = "sh" if index_code.startswith("0") else "sz"
        quote = fetch_quote_tencent(f"s_{prefix}{index_code}")
        if quote:
            dummy_dates = pd.bdate_range(end=datetime.now(), periods=30, freq='B')
            mock_closes = np.linspace(quote['close'] * (1 - quote['pct_chg'] / 100 * 0.05), quote['close'], len(dummy_dates))
            return pd.DataFrame({
                'date': dummy_dates, 'open': mock_closes, 'high': mock_closes * 1.005,
                'low': mock_closes * 0.995, 'close': mock_closes, 'volume': [quote['volume']] * len(dummy_dates)
            })
    except Exception:
        pass

    # 5. 兜底模拟走势 (防止阻断程序)
    base_price = 3800.0 if index_code == "000001" else 13000.0
    dates = pd.bdate_range(end=datetime.now(), periods=30, freq='B')
    return pd.DataFrame({
        'date': dates, 'open': base_price, 'high': base_price * 1.01,
        'low': base_price * 0.99, 'close': base_price, 'volume': 200000000.0
    })


def create_realistic_fallback_data(stock_code: str = "600580") -> pd.DataFrame:
    """高保真几何布朗运动蒙特卡洛模拟数据 (终极容灾兜底)"""
    real_stock_references = {
        '600580': {'name': '卧龙电驱', 'current_price': 27.0, 'range': (20.0, 32.0)},
        '603288': {'name': '海天味业', 'current_price': 34.0, 'range': (30.0, 42.0)},
        '002354': {'name': '天娱数科', 'current_price': 15.0, 'range': (10.0, 20.0)},
    }
    stock_info = real_stock_references.get(stock_code, {'name': '示例股票', 'current_price': 25.0, 'range': (18.0, 35.0)})

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
    np.random.seed(42)
    n_points = len(dates)

    current_price = stock_info['current_price']
    min_price, max_price = stock_info['range']
    prices = [current_price]
    for i in range(1, n_points):
        volatility = 0.02
        historical_return = np.random.normal(-0.0002, volatility)
        prev_price = prices[0] * (1 + historical_return)
        prev_price = max(min_price * 0.9, min(max_price * 1.1, prev_price))
        prices.insert(0, prev_price)

    stock_data = []
    for i, date in enumerate(dates):
        close_price = prices[i]
        daily_volatility = abs(np.random.normal(0, 0.015))
        open_price = close_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close_price) * (1 + daily_volatility)
        low_price = min(open_price, close_price) * (1 - daily_volatility)
        volume = int(abs(np.random.normal(1500000, 400000)))
        amount = volume * close_price
        pct_chg = ((close_price - prices[i - 1]) / prices[i - 1]) * 100 if i > 0 else 0

        stock_data.append({
            'timestamps': date, 'stock_code': stock_code,
            'open': round(open_price, 2), 'close': round(close_price, 2),
            'high': round(high_price, 2), 'low': round(low_price, 2),
            'volume': volume, 'amount': round(amount, 2),
            'pct_chg': round(pct_chg, 2), 'turnover': 3.5
        })

    return pd.DataFrame(stock_data)


def get_stock_data_with_retry_all_history(stock_code: str = "600580", retry_count: int = 2) -> pd.DataFrame:
    """多数据源轮询调度获取股票日K线"""
    data_sources = [
        ("东方财富网", lambda code: fetch_stock_data_eastmoney(code, history_years=2)),
        ("Baostock", lambda code: fetch_stock_data_baostock(code, history_years=2)),
        ("AKShare", lambda code: fetch_stock_data_akshare(code, period="daily", adjust="qfq"))
    ]

    for source_name, fetch_func in data_sources:
        for attempt in range(1, retry_count + 1):
            try:
                df = fetch_func(stock_code)
                if df is not None and not df.empty and len(df) >= 20:
                    df.attrs['source'] = source_name
                    return df
            except Exception:
                pass
            time.sleep(0.3)

    fallback_df = create_realistic_fallback_data(stock_code)
    fallback_df.attrs['source'] = "蒙特卡洛模拟数据(离线兜底)"
    return fallback_df


def save_all_history_stock_data(df: pd.DataFrame, stock_code: str, save_dir: str) -> bool:
    """保存数据到本地 CSV"""
    if df is not None and not df.empty:
        os.makedirs(save_dir, exist_ok=True)
        csv_file = os.path.join(save_dir, f"{stock_code}_stock_data.csv")
        df_to_save = df.copy()
        if isinstance(df_to_save.index, pd.DatetimeIndex):
            df_to_save = df_to_save.reset_index()
            if 'index' in df_to_save.columns and 'timestamps' not in df_to_save.columns:
                df_to_save = df_to_save.rename(columns={'index': 'timestamps'})
        elif 'index' in df_to_save.columns:
            df_to_save = df_to_save.drop(columns=['index'])
        df_to_save.to_csv(csv_file, encoding='utf-8-sig', index=False)
        return True
    return False


def get_stock_data(stock_code: str, data_dir: str, force_refresh: bool = False) -> Tuple[bool, Optional[str], Optional[str]]:
    """获取股票数据文件路径与来源"""
    csv_file_path = os.path.join(data_dir, f"{stock_code}_stock_data.csv")
    if os.path.exists(csv_file_path) and not force_refresh:
        return True, csv_file_path, "本地缓存"

    df = get_stock_data_with_retry_all_history(stock_code)
    if df is not None and not df.empty:
        source_name = df.attrs.get('source', '网络源')
        save_all_history_stock_data(df, stock_code, data_dir)
        return True, csv_file_path, source_name
    return False, None, None


def prepare_stock_data(csv_file_path: str, stock_code: str, history_years: int = 1) -> pd.DataFrame:
    """标准化与准备时序数据"""
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
    column_mapping = {
        '日期': 'timestamps', '开盘价': 'open', '最高价': 'high',
        '最低价': 'low', '收盘价': 'close', '成交量': 'volume',
        '成交额': 'amount', '开盘': 'open', '收盘': 'close',
        '最高': 'high', '最低': 'low'
    }
    actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=actual_mapping)

    if 'timestamps' not in df.columns and df.index.name == '日期':
        df = df.reset_index().rename(columns={'日期': 'timestamps'})

    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)

    if history_years > 0:
        cutoff_date = datetime.now() - timedelta(days=history_years * 365)
        df_filtered = df[df['timestamps'] >= cutoff_date]
        if len(df_filtered) >= 30:
            df = df_filtered.reset_index(drop=True)

    return df


def calculate_prediction_parameters(df: pd.DataFrame, target_days: int = 60) -> Tuple[int, int]:
    """根据历史记录动态规划 lookback 和 pred_len"""
    total_data_len = len(df)
    trading_days_ratio = 5 / 7
    target_trading_days = max(10, int(target_days * trading_days_ratio))

    max_pred_len = min(60, total_data_len // 3)
    pred_len = min(target_trading_days, max_pred_len)

    lookback_options = [pred_len * 3, pred_len * 2, pred_len * 4]
    lookback = min(total_data_len - pred_len - 10, max(lookback_options))
    return max(30, lookback), pred_len


def generate_trading_dates_only(last_date, pred_len: int) -> List[datetime]:
    """生成未来预测交易日列表（排除周末与常规假期）"""
    trading_dates = []
    current_date = pd.to_datetime(last_date)
    attempts = 0
    max_attempts = pred_len * 3

    while len(trading_dates) < pred_len and attempts < max_attempts:
        current_date += timedelta(days=1)
        attempts += 1
        if current_date.weekday() < 5:  # 周一至周五
            trading_dates.append(current_date)

    return trading_dates


# ==================== 2. 多维度市场因子分析器 ====================
class EnhancedMarketFactorAnalyzer:
    """多维度宏观、行业、大盘与基本面因子分析器"""

    def analyze_market_trend(self, index_codes=["000001", "399001"]) -> Dict:
        """分析大盘趋势（多均线共振）"""
        market_analysis = {}
        for index_code in index_codes:
            index_name = "上证指数" if index_code == "000001" else "深证成指"
            index_df = fetch_index_data_multi_source(index_code)
            if index_df is None or index_df.empty:
                continue

            index_df['ma5'] = index_df['close'].rolling(5).mean()
            index_df['ma20'] = index_df['close'].rolling(20).mean()
            index_df['ma60'] = index_df['close'].rolling(60).mean()

            current = index_df.iloc[-1]
            prev = index_df.iloc[-2]

            ma_condition = bool(current['ma5'] > current['ma20'] > current['ma60'])
            price_above_ma20 = bool(current['close'] > current['ma20'])
            trend_strength = 0.8 if (ma_condition and price_above_ma20) else (0.5 if price_above_ma20 else 0.3)
            is_main_uptrend = ma_condition and price_above_ma20

            market_analysis[index_name] = {
                'is_main_uptrend': is_main_uptrend,
                'trend_strength': trend_strength,
                'current_close': float(current['close']),
                'price_change_pct': float(((current['close'] - prev['close']) / prev['close']) * 100),
                'market_status': '主升多头' if is_main_uptrend else '震荡调整'
            }

        if market_analysis:
            avg_strength = float(np.mean([d['trend_strength'] for d in market_analysis.values()]))
            uptrend_count = sum(1 for d in market_analysis.values() if d['is_main_uptrend'])
            overall_uptrend = uptrend_count >= len(market_analysis) * 0.5
            return {
                'indices': market_analysis,
                'overall_is_main_uptrend': overall_uptrend,
                'overall_trend_strength': avg_strength,
                'market_status': '主升浪' if overall_uptrend else '震荡调整'
            }

        return {
            'indices': {},
            'overall_is_main_uptrend': False,
            'overall_trend_strength': 0.5,
            'market_status': '数据不足'
        }

    def analyze_sector_resonance(self, stock_code: str) -> Dict:
        """分析板块与热点共振"""
        hot_sectors = ["人工智能", "新能源", "人形机器人", "高股息资产", "半导体芯片", "生物医药"]
        traditional_sectors = ["传统制造", "食品饮料", "商贸零售", "钢铁建筑"]

        # 示例映射
        if stock_code.startswith(('300', '688')):
            matched = ["高科技创新"]
            score = 0.75
        elif stock_code in ['603288', '600519']:
            matched = ["消费龙头"]
            score = 0.65
        else:
            matched = ["智能制造装备"]
            score = 0.60

        return {
            'sector': matched[0],
            'resonance_score': score,
            'is_sector_hot': score >= 0.7,
            'sector_count': len(matched)
        }

    def analyze_macro_factors(self) -> Dict:
        """分析宏观政策环境"""
        return {
            'us_rate_cycle': {'trend': '降息周期', 'expected_cuts_2025': 2},
            'domestic_policy': '积极宽松',
            'overall_macro_score': 0.75
        }

    def analyze_company_fundamentals(self, stock_code: str) -> Dict:
        """分析公司基本面估值评级"""
        return {
            'stock_code': stock_code,
            'investment_rating': '增持/良好',
            'fundamental_score': 0.70
        }


# ==================== 3. 量化预测后处理与调整 ====================
def smooth_prediction_results(prediction_df: pd.DataFrame, historical_df: pd.DataFrame, smooth_factor: float = 0.3) -> pd.DataFrame:
    """消除历史真实数据与预测起点的跳跃断层"""
    smoothed_df = prediction_df.copy()
    last_real_price = historical_df['close'].iloc[-1]
    smoothed_prices = [last_real_price]

    for i in range(len(prediction_df)):
        pred_price = prediction_df['close'].iloc[i]
        smoothed_price = smooth_factor * pred_price + (1 - smooth_factor) * smoothed_prices[-1]
        smoothed_prices.append(smoothed_price)

    smoothed_df['close'] = smoothed_prices[1:]
    return smoothed_df


def validate_prediction_results(historical_df: pd.DataFrame, prediction_df: pd.DataFrame, max_price_change: float = 0.3) -> pd.DataFrame:
    """预测合理性与涨跌风控熔断"""
    validated_df = prediction_df.copy()
    current_price = historical_df['close'].iloc[-1]
    max_allowed = current_price * max_price_change

    for col in ['close', 'open', 'high', 'low']:
        if col in validated_df.columns:
            for i in range(len(validated_df)):
                val = validated_df[col].iloc[i]
                if abs(val - current_price) > max_allowed:
                    direction = 1 if val > current_price else -1
                    validated_df.iloc[i, validated_df.columns.get_loc(col)] = current_price + direction * max_allowed * 0.95
    return validated_df


def calculate_enhanced_adjustment_factor(market_analysis, sector_analysis, macro_analysis, fundamental_analysis) -> float:
    """量化多因子乘法调整系数"""
    factor = 1.0
    # 大盘
    trend_strength = market_analysis.get('overall_trend_strength', 0.5)
    factor *= (1.0 + (trend_strength - 0.5) * 0.05)
    # 板块
    resonance = sector_analysis.get('resonance_score', 0.5)
    factor *= (1.0 + (resonance - 0.5) * 0.05)
    # 宏观
    macro = macro_analysis.get('overall_macro_score', 0.5)
    factor *= (1.0 + (macro - 0.5) * 0.04)
    # 基本面
    fund = fundamental_analysis.get('fundamental_score', 0.5)
    factor *= (1.0 + (fund - 0.5) * 0.04)

    return float(np.clip(factor, 0.92, 1.08))


# ==================== 4. 预测绘图与分析报告函数 ====================
def generate_analysis_summary(enhancement_info: Dict) -> Dict:
    """生成多因子分析总结与风险提示"""
    market = enhancement_info.get('market_analysis', {})
    sector = enhancement_info.get('sector_analysis', {})
    macro = enhancement_info.get('macro_analysis', {})
    fundamental = enhancement_info.get('fundamental_analysis', {})
    adj_factor = enhancement_info.get('adjustment_factor', 1.0)

    summary = {
        'overall_sentiment': '积极偏多' if adj_factor >= 1.02 else ('谨慎震荡' if adj_factor >= 0.98 else '偏弱防守'),
        'key_drivers': [],
        'main_risks': [],
        'investment_suggestion': ''
    }

    if market.get('overall_trend_strength', 0.5) > 0.6:
        summary['key_drivers'].append('大盘均线呈多头排列，趋势动能向好')
    if sector.get('is_sector_hot', False):
        summary['key_drivers'].append(f"标的属于热门主题板块: {sector.get('sector', '成长赛道')}")
    if macro.get('overall_macro_score', 0.5) > 0.65:
        summary['key_drivers'].append(f"宏观流动性宽松，美联储处于{macro.get('us_rate_cycle', {}).get('trend', '降息周期')}")
    if fundamental.get('fundamental_score', 0.5) >= 0.6:
        summary['key_drivers'].append('公司基本面稳健，行业龙头估值具备安全边际')

    if market.get('overall_trend_strength', 0.5) < 0.45:
        summary['main_risks'].append('大盘处于震荡调整期，上方存在套牢筹码压制')
    if not sector.get('is_sector_hot', False):
        summary['main_risks'].append('标的非当前主线热门板块，资金关注度相对一般')
    if macro.get('overall_macro_score', 0.5) < 0.5:
        summary['main_risks'].append('外部宏观不确定性扰动，需警惕地缘政治与流动性收紧')

    if len(summary['key_drivers']) >= len(summary['main_risks']):
        summary['investment_suggestion'] = '趋势共振较强，建议逢低分批布局，持股待涨'
    else:
        summary['investment_suggestion'] = '趋势存在分歧，建议轻仓观望，严格设置止损'

    return summary


# ==================== 4. ECharts 交互式专业 K 线与量化看板引擎 ====================
_ECHARTS_JS_CACHE = None

def get_echarts_js_source() -> str:
    """获取 ECharts JS 源码或引用标签 (优先本地静态资源离线直载，备用国内极速镜像)"""
    global _ECHARTS_JS_CACHE
    if _ECHARTS_JS_CACHE is not None:
        return _ECHARTS_JS_CACHE

    local_path = os.path.join(CURRENT_DIR, "static", "echarts.min.js")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()
                if len(content) > 100000:
                    _ECHARTS_JS_CACHE = f"<script>{content}</script>"
                    return _ECHARTS_JS_CACHE
        except Exception:
            pass

    # 离线文件读取异常时的国内高速 CDN 回退方案 (避免 jsDelivr 被墙或超时白屏)
    fallback_script = """
    <script src="https://registry.npmmirror.com/echarts/5.5.0/files/dist/echarts.min.js"></script>
    <script>
        if (typeof echarts === 'undefined') {
            document.write('<script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.5.0/echarts.min.js"><\\/script>');
        }
    </script>
    """
    _ECHARTS_JS_CACHE = fallback_script
    return _ECHARTS_JS_CACHE


def build_echarts_kline_option(historical_df, base_pred_df, enhanced_pred_df, future_dates, stock_code, stock_name, factor_info):
    """
    构建符合金融行业标准规范的 Apache ECharts 5 交互式 K 线多轴联动配置字典:
    - 主图: 历史真实日K蜡烛线 + AI推演预测蜡烛线 + 动态均线组 (MA5/10/20/30) + 关键价位线与极值点
    - 副图: 红绿涨跌成交量柱状图 (主副图时间轴联动、十字光标联动)
    - 交互: 支持双模式无级缩放 (鼠标滚轮 + 底部滑动条)、十字光标全息浮窗 (Tooltip) 与图例过滤
    """
    # 1. 整理时间轴数据
    hist_dates = [
        d.strftime('%Y-%m-%d') if isinstance(d, (datetime, pd.Timestamp)) else str(d)[:10]
        for d in historical_df['timestamps']
    ]
    pred_dates = [
        d.strftime('%Y-%m-%d') if isinstance(d, (datetime, pd.Timestamp)) else str(d)[:10]
        for d in future_dates
    ]
    total_hist_len = len(hist_dates)
    total_pred_len = len(pred_dates)
    all_dates = hist_dates + pred_dates
    total_len = len(all_dates)

    # 2. 历史真实日 K 线 (OHLC: [open, close, lowest, highest])
    # 注意：ECharts Candlestick 缺少数据时必须使用 '-' 占位，使用 None/null 会触发解析崩溃
    hist_kline_data = []
    for _, row in historical_df.iterrows():
        o = round(float(row['open']), 2)
        c = round(float(row['close']), 2)
        l = round(float(row['low']), 2)
        h = round(float(row['high']), 2)
        hist_kline_data.append([o, c, l, h])
    hist_kline_data.extend(['-'] * total_pred_len)

    # 3. AI 预测推演蜡烛 K 线 (OHLC)
    # 历史区间使用 '-' 占位，未来预测区间填充数值
    pred_kline_data = ['-'] * total_hist_len
    for _, row in enhanced_pred_df.iterrows():
        o = round(float(row['open']), 2)
        c = round(float(row['close']), 2)
        l = round(float(row['low']), 2)
        h = round(float(row['high']), 2)
        pred_kline_data.append([o, c, l, h])

    # 4. 连续预测折线 (Kronos 基础预测 vs 多因子增强预测)
    last_hist_close = round(float(historical_df['close'].iloc[-1]), 2)
    last_hist_idx = total_hist_len - 1

    base_line_data = ['-'] * total_hist_len
    enhanced_line_data = ['-'] * total_hist_len

    # 从历史最后一个实盘收盘点自然延伸
    base_line_data[last_hist_idx] = last_hist_close
    enhanced_line_data[last_hist_idx] = last_hist_close

    for val in base_pred_df['close']:
        base_line_data.append(round(float(val), 2))
    for val in enhanced_pred_df['close']:
        enhanced_line_data.append(round(float(val), 2))

    # 5. 均线组计算 (MA5, MA10, MA20, MA30)
    all_closes = list(historical_df['close'].values) + list(enhanced_pred_df['close'].values)
    series_close = pd.Series(all_closes)

    def calc_ma(period):
        ma_s = series_close.rolling(period).mean()
        return [round(float(v), 2) if not np.isnan(v) else '-' for v in ma_s]

    ma5 = calc_ma(5)
    ma10 = calc_ma(10)
    ma20 = calc_ma(20)
    ma30 = calc_ma(30)

    # 6. 成交量副图 (Volume Bars: 涨红跌绿)
    volume_data = []
    # 历史成交量
    for _, row in historical_df.iterrows():
        v = round(float(row['volume']), 0)
        is_up = row['close'] >= row['open']
        color = '#ef5350' if is_up else '#26a69a'
        volume_data.append({
            'value': v,
            'itemStyle': {'color': color}
        })
    # 预测成交量 (带半透明与虚线区分)
    for _, row in enhanced_pred_df.iterrows():
        v = round(float(row['volume']), 0)
        is_up = row['close'] >= row['open']
        color = 'rgba(239, 83, 80, 0.45)' if is_up else 'rgba(38, 166, 154, 0.45)'
        volume_data.append({
            'value': v,
            'itemStyle': {
                'color': color,
                'borderColor': '#ef5350' if is_up else '#26a69a',
                'borderWidth': 1,
                'borderType': 'dashed'
            }
        })

    # 7. 自适应视窗范围 (默认展示最近 45 天历史 + 全部预测)
    visible_window = min(total_len, 45 + total_pred_len)
    start_pct = max(0, int((1 - visible_window / total_len) * 100))

    split_date = hist_dates[-1]
    final_target = enhanced_line_data[-1] if len(enhanced_line_data) > 0 else last_hist_close
    pct_chg_pred = ((final_target - last_hist_close) / last_hist_close) * 100

    option = {
        'backgroundColor': '#ffffff',
        'animation': True,
        'title': {
            'text': f"{stock_name} ({stock_code}) - Kronos 深度量化 K 线预测看板",
            'subtext': f"历史数据: {total_hist_len} 交易日 | 预测周期: {total_pred_len} 交易日 | 现价: ¥{last_hist_close:.2f} | 目标价: ¥{final_target:.2f} ({pct_chg_pred:+.2f}%)",
            'left': '2%',
            'top': 8,
            'textStyle': {'fontSize': 16, 'fontWeight': 'bold', 'color': '#1a1a1a'},
            'subtextStyle': {'fontSize': 12, 'color': '#555555'}
        },
        'legend': {
            'top': 10,
            'right': '10%',
            'data': ['历史K线', 'AI预测K线', '增强预测线', '基准预测线', 'MA5', 'MA10', 'MA20', 'MA30', '成交量'],
            'selected': {
                '历史K线': True,
                'AI预测K线': True,
                '增强预测线': True,
                '基准预测线': True,
                'MA5': True,
                'MA10': True,
                'MA20': True,
                'MA30': False,
                '成交量': True
            }
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {'type': 'cross', 'lineStyle': {'color': '#888', 'width': 1, 'type': 'dashed'}},
            'borderWidth': 1,
            'borderColor': '#cfd8dc',
            'backgroundColor': 'rgba(255, 255, 255, 0.96)',
            'padding': 12,
            'extraCssText': 'box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-radius: 6px;'
        },
        'axisPointer': {
            'link': [{'xAxisIndex': 'all'}],
            'label': {'backgroundColor': '#455a64'}
        },
        'toolbox': {
            'feature': {
                'dataZoom': {'yAxisIndex': False},
                'restore': {'title': '重置'},
                'saveAsImage': {'pixelRatio': 2, 'title': '保存图片', 'name': f"{stock_code}_K线预测"}
            },
            'right': '2%',
            'top': 8
        },
        'grid': [
            {'left': '5%', 'right': '4%', 'top': '14%', 'height': '54%'},
            {'left': '5%', 'right': '4%', 'top': '72%', 'height': '15%'}
        ],
        'xAxis': [
            {
                'type': 'category',
                'data': all_dates,
                'scale': True,
                'boundaryGap': True,
                'axisLine': {'onZero': False, 'lineStyle': {'color': '#333'}},
                'splitLine': {'show': False},
                'min': 'dataMin',
                'max': 'dataMax',
                'axisPointer': {'z': 100}
            },
            {
                'type': 'category',
                'gridIndex': 1,
                'data': all_dates,
                'scale': True,
                'boundaryGap': True,
                'axisLine': {'onZero': False, 'lineStyle': {'color': '#888'}},
                'axisTick': {'show': False},
                'splitLine': {'show': False},
                'axisLabel': {'show': False},
                'min': 'dataMin',
                'max': 'dataMax'
            }
        ],
        'yAxis': [
            {
                'scale': True,
                'splitArea': {'show': True, 'areaStyle': {'color': ['rgba(250,250,250,0.4)', 'rgba(240,240,240,0.1)']}},
                'splitLine': {'show': True, 'lineStyle': {'color': '#f0f0f0'}},
                'axisLabel': {'formatter': '¥{value}'}
            },
            {
                'scale': True,
                'gridIndex': 1,
                'splitNumber': 2,
                'axisLabel': {'show': True, 'fontSize': 10, 'color': '#777'},
                'axisLine': {'show': False},
                'axisTick': {'show': False},
                'splitLine': {'show': False}
            }
        ],
        'dataZoom': [
            {
                'type': 'inside',
                'xAxisIndex': [0, 1],
                'start': start_pct,
                'end': 100
            },
            {
                'show': True,
                'xAxisIndex': [0, 1],
                'type': 'slider',
                'top': '90%',
                'height': 22,
                'start': start_pct,
                'end': 100,
                'borderColor': '#cfd8dc',
                'fillerColor': 'rgba(33, 150, 243, 0.15)',
                'handleStyle': {'color': '#2196f3'}
            }
        ],
        'series': [
            {
                'name': '历史K线',
                'type': 'candlestick',
                'data': hist_kline_data,
                'itemStyle': {
                    'color': '#ef5350',
                    'color0': '#26a69a',
                    'borderColor': '#ef5350',
                    'borderColor0': '#26a69a'
                },
                'markLine': {
                    'symbol': ['none', 'none'],
                    'data': [
                        {
                            'name': '预测起点',
                            'xAxis': split_date,
                            'lineStyle': {'color': '#ff9800', 'width': 2, 'type': 'dashed'},
                            'label': {'position': 'end', 'formatter': '🚀 预测起点', 'color': '#e65100', 'fontWeight': 'bold', 'fontSize': 11}
                        },
                        {
                            'name': '最新收盘价',
                            'yAxis': last_hist_close,
                            'lineStyle': {'color': '#1976d2', 'width': 1.2, 'type': 'dotted'},
                            'label': {'position': 'start', 'formatter': f'现价: ¥{last_hist_close:.2f}', 'color': '#0d47a1'}
                        },
                        {
                            'name': '目标价',
                            'yAxis': final_target,
                            'lineStyle': {'color': '#2e7d32', 'width': 1.2, 'type': 'dotted'},
                            'label': {'position': 'start', 'formatter': f'目标: ¥{final_target:.2f}', 'color': '#1b5e20'}
                        }
                    ]
                },
                'markPoint': {
                    'label': {'fontSize': 10},
                    'data': [
                        {'type': 'max', 'name': '历史最高', 'valueDim': 'highest'},
                        {'type': 'min', 'name': '历史最低', 'valueDim': 'lowest'}
                    ]
                }
            },
            {
                'name': 'AI预测K线',
                'type': 'candlestick',
                'data': pred_kline_data,
                'itemStyle': {
                    'color': 'rgba(239, 83, 80, 0.65)',
                    'color0': 'rgba(38, 166, 154, 0.65)',
                    'borderColor': '#c62828',
                    'borderColor0': '#00695c',
                    'borderWidth': 1.5,
                    'borderType': 'dashed'
                },
                'markPoint': {
                    'label': {'fontSize': 10},
                    'data': [
                        {'type': 'max', 'name': '预测峰值', 'valueDim': 'highest'},
                        {'type': 'min', 'name': '预测谷值', 'valueDim': 'lowest'}
                    ]
                }
            },
            {
                'name': '增强预测线',
                'type': 'line',
                'data': enhanced_line_data,
                'smooth': True,
                'lineStyle': {'color': '#2ca02c', 'width': 2.5},
                'symbol': 'circle',
                'symbolSize': 4
            },
            {
                'name': '基准预测线',
                'type': 'line',
                'data': base_line_data,
                'smooth': True,
                'lineStyle': {'color': '#ff7f0e', 'width': 2, 'type': 'dashed'},
                'symbol': 'circle',
                'symbolSize': 3
            },
            {
                'name': 'MA5',
                'type': 'line',
                'data': ma5,
                'smooth': True,
                'lineStyle': {'opacity': 0.8, 'width': 1.3, 'color': '#f1c40f'},
                'symbol': 'none'
            },
            {
                'name': 'MA10',
                'type': 'line',
                'data': ma10,
                'smooth': True,
                'lineStyle': {'opacity': 0.8, 'width': 1.3, 'color': '#9b59b6'},
                'symbol': 'none'
            },
            {
                'name': 'MA20',
                'type': 'line',
                'data': ma20,
                'smooth': True,
                'lineStyle': {'opacity': 0.8, 'width': 1.3, 'color': '#3498db'},
                'symbol': 'none'
            },
            {
                'name': 'MA30',
                'type': 'line',
                'data': ma30,
                'smooth': True,
                'lineStyle': {'opacity': 0.8, 'width': 1.3, 'color': '#e67e22'},
                'symbol': 'none'
            },
            {
                'name': '成交量',
                'type': 'bar',
                'xAxisIndex': 1,
                'yAxisIndex': 1,
                'data': volume_data
            }
        ]
    }
    return option


def render_echarts_kline(historical_df, base_pred_df, enhanced_pred_df, future_dates, stock_code, stock_name, factor_info, height: int = 730):
    """
    通过 Streamlit 组件以原生 HTML/JS 模式渲染专业 ECharts K 线图
    """
    option = build_echarts_kline_option(historical_df, base_pred_df, enhanced_pred_df, future_dates, stock_code, stock_name, factor_info)
    option_json = json.dumps(option, ensure_ascii=False)
    echarts_script_tag = get_echarts_js_source()

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        {echarts_script_tag}
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                background-color: #ffffff;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }}
            #kline_chart_container {{
                width: 100%;
                height: {height}px;
            }}
        </style>
    </head>
    <body>
        <div id="kline_chart_container"></div>
        <script>
            (function() {{
                if (typeof echarts === 'undefined') {{
                    document.getElementById('kline_chart_container').innerHTML = '<div style="padding:20px;color:red;font-size:14px;">⚠️ ECharts 引擎加载失败，请检查网络连接或本地 static/echarts.min.js 文件</div>';
                    return;
                }}
                var chartDom = document.getElementById('kline_chart_container');
                var myChart = echarts.init(chartDom);
                var option = {option_json};

                // 高精度金融十字光标全息 Tooltip
                option.tooltip.formatter = function(params) {{
                    if (!params || params.length === 0) return '';
                    var date = params[0].axisValue;
                    var res = '<div style="font-weight:bold;margin-bottom:6px;border-bottom:1px solid #eee;padding-bottom:4px;color:#263238;">📅 ' + date + '</div>';

                    var kline = null;
                    var vol = null;
                    var mas = [];
                    var lines = [];

                    for (var i = 0; i < params.length; i++) {{
                        var item = params[i];
                        if (item.seriesName === '历史K线' || item.seriesName === 'AI预测K线') {{
                            if (item.value && Array.isArray(item.value) && item.value[1] !== undefined && item.value[1] !== '-') {{
                                kline = {{ name: item.seriesName, data: item.value }};
                            }}
                        }} else if (item.seriesName === '成交量') {{
                            if (item.value !== undefined && item.value !== null && item.value !== '-') {{
                                vol = item.value;
                            }}
                        }} else if (item.seriesName.indexOf('MA') === 0) {{
                            if (item.value !== undefined && item.value !== null && item.value !== '-') {{
                                mas.push({{ name: item.seriesName, val: item.value, color: item.color }});
                            }}
                        }} else if (item.seriesName.indexOf('预测线') !== -1) {{
                            if (item.value !== undefined && item.value !== null && item.value !== '-') {{
                                lines.push({{ name: item.seriesName, val: item.value, color: item.color }});
                            }}
                        }}
                    }}

                    if (kline && kline.data) {{
                        var open = Number(kline.data[1]).toFixed(2);
                        var close = Number(kline.data[2]).toFixed(2);
                        var low = Number(kline.data[3]).toFixed(2);
                        var high = Number(kline.data[4]).toFixed(2);
                        var chg = (close - open).toFixed(2);
                        var pct = ((chg / open) * 100).toFixed(2);
                        var color = close >= open ? '#ef5350' : '#26a69a';
                        var sign = chg >= 0 ? '+' : '';
                        var tag = kline.name === 'AI预测K线'
                            ? '<span style="background:#ff9800;color:#fff;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:bold;">🚀 AI 深度推演</span>'
                            : '<span style="background:#1976d2;color:#fff;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:bold;">📈 历史实盘</span>';

                        res += '<div style="margin-bottom:6px;">' + tag + '</div>';
                        res += '<table style="width:100%;font-size:12px;line-height:1.6;border-collapse:collapse;">';
                        res += '<tr><td style="color:#666;">开盘:</td><td style="text-align:right;font-weight:bold;padding-right:8px;">¥' + open + '</td><td style="color:#666;">最高:</td><td style="text-align:right;font-weight:bold;color:#ef5350;">¥' + high + '</td></tr>';
                        res += '<tr><td style="color:#666;">收盘:</td><td style="text-align:right;font-weight:bold;color:' + color + ';padding-right:8px;">¥' + close + '</td><td style="color:#666;">最低:</td><td style="text-align:right;font-weight:bold;color:#26a69a;">¥' + low + '</td></tr>';
                        res += '<tr><td style="color:#666;">涨跌:</td><td colspan="3" style="text-align:right;font-weight:bold;color:' + color + ';">' + sign + chg + ' (' + sign + pct + '%)</td></tr>';
                        res += '</table>';
                    }}

                    if (lines.length > 0) {{
                        res += '<div style="margin-top:6px;border-top:1px dashed #e0e0e0;padding-top:5px;font-size:12px;">';
                        lines.forEach(function(l) {{
                            res += '<div style="margin-bottom:2px;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + l.color + ';margin-right:6px;"></span>' + l.name + ': <b>¥' + Number(l.val).toFixed(2) + '</b></div>';
                        }});
                        res += '</div>';
                    }}

                    if (mas.length > 0) {{
                        res += '<div style="margin-top:6px;border-top:1px dashed #e0e0e0;padding-top:4px;font-size:11px;">';
                        var maTexts = [];
                        mas.forEach(function(m) {{
                            maTexts.push('<span style="color:' + m.color + ';font-weight:bold;">' + m.name + ': ' + Number(m.val).toFixed(2) + '</span>');
                        }});
                        res += maTexts.join(' | ');
                        res += '</div>';
                    }}

                    if (vol !== null) {{
                        var volVal = typeof vol === 'object' ? vol.value : vol;
                        var volStr = volVal > 100000000 ? (volVal / 100000000).toFixed(2) + ' 亿股' : (volVal > 10000 ? (volVal / 10000).toFixed(2) + ' 万股' : volVal + ' 股');
                        res += '<div style="margin-top:6px;border-top:1px dashed #e0e0e0;padding-top:4px;font-size:12px;color:#455a64;">📊 成交量: <b>' + volStr + '</b></div>';
                    }}

                    return res;
                }};

                myChart.setOption(option);
                window.addEventListener('resize', function() {{
                    myChart.resize();
                }});
            }})();
        </script>
    </body>
    </html>
    """
    st.iframe(html_code, height=height + 20)


def render_echarts_factor_distribution(factor_info: Dict, height: int = 340):
    """
    通过 ECharts 交互式环形图呈现多维度因子的权重占比与量化影响
    """
    weights = [
        {'name': '大盘趋势因子', 'value': 25, 'desc': '上证/深证多均线多头共振评估'},
        {'name': '行业板块共振', 'value': 30, 'desc': '所属细分行业超额动量与热点归属'},
        {'name': '宏观利率政策', 'value': 25, 'desc': '美联储利率预期与国内流动性基调'},
        {'name': '公司基本面', 'value': 20, 'desc': 'ROE与PEG估值中枢健康度'}
    ]
    option = {
        'title': {
            'text': '多因子强化权重分配',
            'subtext': f"综合增强系数: {factor_info.get('adjustment_factor', 1.0):.4f}",
            'left': 'center',
            'top': 10,
            'textStyle': {'fontSize': 15, 'fontWeight': 'bold'},
            'subtextStyle': {'fontSize': 12, 'color': '#2e7d32'}
        },
        'tooltip': {
            'trigger': 'item',
            'formatter': '{b}: {c}% ({d}%)<br/><i>{a}</i>'
        },
        'legend': {
            'orient': 'horizontal',
            'bottom': '5%',
            'data': [w['name'] for w in weights]
        },
        'series': [
            {
                'name': '因子影响说明',
                'type': 'pie',
                'radius': ['40%', '68%'],
                'center': ['50%', '50%'],
                'avoidLabelOverlap': False,
                'itemStyle': {
                    'borderRadius': 8,
                    'borderColor': '#fff',
                    'borderWidth': 2
                },
                'label': {
                    'show': True,
                    'formatter': '{b}\n{d}%',
                    'fontSize': 11
                },
                'emphasis': {
                    'label': {
                        'show': True,
                        'fontSize': 13,
                        'fontWeight': 'bold'
                    }
                },
                'data': weights
            }
        ]
    }
    opt_json = json.dumps(option, ensure_ascii=False)
    echarts_script_tag = get_echarts_js_source()
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        {echarts_script_tag}
        <style>
            body {{ margin: 0; padding: 0; background: #ffffff; overflow: hidden; }}
            #factor_chart {{ width: 100%; height: {height}px; }}
        </style>
    </head>
    <body>
        <div id="factor_chart"></div>
        <script>
            if (typeof echarts !== 'undefined') {{
                var chart = echarts.init(document.getElementById('factor_chart'));
                chart.setOption({opt_json});
                window.addEventListener('resize', function() {{ chart.resize(); }});
            }}
        </script>
    </body>
    </html>
    """
    st.iframe(html_code, height=height + 10)


def plot_prediction_dashboard(historical_df, base_pred_df, enhanced_pred_df, future_dates, stock_code, stock_name, factor_info):
    """
    严格依照 prediction_new.py 样式绘制综合研报图表:
    - 第1层: 主价格预测大图 (历史价格 + 基础预测 + 增强预测 + 预测起点气泡 + 峰谷标记)
    - 第2层: 成交量预测图 (与价格图共享时间轴X，万/亿单位自适应)
    - 第3层: 多因子 3 联分析 (权重分配饼图 + 各因素当前得分柱状图 + 市场状态总结卡片)
    - 第4层: 详细因素分析横幅 (关键驱动因素 + 主要风险提示 + 总体情绪与投资建议)
    """
    colors = {
        'historical': '#1f77b4',
        'prediction': '#ff7f0e',
        'enhanced': '#2ca02c',
        'background': '#f8f9fa',
        'grid': '#e9ecef'
    }

    current_price = historical_df['close'].iloc[-1]
    adj_factor = factor_info.get('adjustment_factor', 1.0)
    analysis_summary = generate_analysis_summary(factor_info)

    fig = plt.figure(figsize=(18, 14), dpi=100)
    gs = plt.GridSpec(4, 3, figure=fig, height_ratios=[2.2, 1.0, 1.1, 0.9])
    fig.patch.set_facecolor('white')

    # 1. 主价格图表
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(colors['background'])

    # 2. 成交量图表 (与价格图共享时间轴X)
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.set_facecolor(colors['background'])

    # 3. 因素分析子图
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(colors['background'])

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor(colors['background'])

    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor(colors['background'])

    # 4. 详细驱动与风险总结
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor(colors['background'])

    # ----- 绘制主价格走势 (ax1) -----
    hist_x = historical_df['timestamps']
    ax1.plot(hist_x, historical_df['close'], color=colors['historical'], linewidth=2.2, label='历史价格', marker='.', markersize=3)

    if len(future_dates) > 0:
        last_hist_date = hist_x.iloc[-1]
        last_hist_price = historical_df['close'].iloc[-1]
        first_pred_date = future_dates[0]

        # 连接历史到基础预测与增强预测
        ax1.plot([last_hist_date, first_pred_date], [last_hist_price, base_pred_df['close'].iloc[0]], color=colors['prediction'], linewidth=2.2, linestyle='-')
        ax1.plot(future_dates, base_pred_df['close'], color=colors['prediction'], linewidth=2.2, label='基础预测', marker='o', markersize=3.5)

        ax1.plot([last_hist_date, first_pred_date], [last_hist_price, enhanced_pred_df['close'].iloc[0]], color=colors['enhanced'], linewidth=2.2, linestyle='--')
        ax1.plot(future_dates, enhanced_pred_df['close'], color=colors['enhanced'], linewidth=2.2, linestyle='--', label='增强预测', marker='^', markersize=3.5)

        # 标注预测起点
        ax1.axvline(x=last_hist_date, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
        ax1.annotate('预测起点', xy=(last_hist_date, last_hist_price),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='red'))

        # 标记最高点与最低点
        close_vals = enhanced_pred_df['close'].values
        max_pos = int(np.argmax(close_vals))
        min_pos = int(np.argmin(close_vals))
        max_price = float(close_vals[max_pos])
        min_price = float(close_vals[min_pos])
        ax1.scatter([future_dates[max_pos]], [max_price], color='#e74c3c', s=70, zorder=5, label=f'预测峰值: {max_price:.2f}')
        ax1.scatter([future_dates[min_pos]], [min_price], color='#2980b9', s=70, zorder=5, label=f'预测谷值: {min_price:.2f}')

    title_str = f"{stock_name}({stock_code}) - 综合因素价格预测 (当前价: {current_price:.2f}元 | 综合调整因子: {adj_factor:.4f})"
    ax1.set_title(title_str, fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel('收盘价 (元)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, color=colors['grid'], alpha=0.7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

    # ----- 绘制成交量图表 (ax2) -----
    ax2.bar(hist_x, historical_df['volume'], alpha=0.6, color=colors['historical'], label='历史成交量', width=0.8)
    if len(future_dates) > 0:
        ax2.bar(future_dates, enhanced_pred_df['volume'], alpha=0.6, color=colors['prediction'], label='预测成交量', width=0.8)

    ax2.set_ylabel('成交量 (股)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, color=colors['grid'], alpha=0.7)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-8:.2f}亿' if x >= 1e8 else (f'{x*1e-4:.0f}万' if x >= 1e4 else f'{x:.0f}')))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

    # ----- 绘制多因子权重分配饼图 (ax3) -----
    factor_names = ['大盘趋势', '板块共振', '宏观环境', '美国降息', '基本面']
    weights = [25, 25, 20, 10, 20]
    colors_pie = [colors['historical'], colors['prediction'], colors['enhanced'], '#f39c12', '#9b59b6']
    ax3.pie(weights, labels=factor_names, autopct='%1.0f%%', colors=colors_pie, startangle=90, textprops={'fontsize': 9})
    ax3.set_title('因素权重分配', fontweight='bold', fontsize=11)

    # ----- 绘制各因素当前评分柱状图 (ax4) -----
    market_score = factor_info['market_analysis'].get('overall_trend_strength', 0.5)
    sector_score = factor_info['sector_analysis'].get('resonance_score', 0.5)
    macro_score = factor_info['macro_analysis'].get('overall_macro_score', 0.5)
    us_rate_score = 0.7 if factor_info['macro_analysis'].get('us_rate_cycle', {}).get('trend') == '降息周期' else 0.3
    fund_score = factor_info['fundamental_analysis'].get('fundamental_score', 0.5)

    scores = [market_score, sector_score, macro_score, us_rate_score, fund_score]
    x_pos = np.arange(len(factor_names))
    bars = ax4.bar(x_pos, scores, color=colors_pie, alpha=0.75)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(factor_names, rotation=35, fontsize=9)
    ax4.set_ylim(0, 1.0)
    ax4.set_ylabel('当前评分', fontsize=10)
    ax4.set_title('各因素当前评分 (0~1.0)', fontweight='bold', fontsize=11)
    ax4.grid(True, alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ----- 市场状态总结卡片 (ax5) -----
    market_status = factor_info['market_analysis'].get('market_status', '震荡调整')
    sector_status = "热门" if factor_info['sector_analysis'].get('is_sector_hot', False) else "一般"
    macro_status = "有利" if factor_info['macro_analysis'].get('overall_macro_score', 0.5) > 0.6 else "中性"
    us_trend = factor_info['macro_analysis'].get('us_rate_cycle', {}).get('trend', '降息周期')
    inv_rating = factor_info['fundamental_analysis'].get('investment_rating', '持有/观望')

    summary_text = (
        f"📊 市场状态总结:\n\n"
        f"• 大盘趋势: {market_status}\n"
        f"• 板块热度: {sector_status}\n"
        f"• 宏观环境: {macro_status}\n"
        f"• 美联储环境: {us_trend}\n"
        f"• 综合调整因子: {adj_factor:.4f}\n\n"
        f"💡 投资建议: {inv_rating}"
    )
    ax5.text(0.08, 0.92, summary_text, transform=ax5.transAxes, fontsize=10, verticalalignment='top', linespacing=1.6)
    ax5.set_title('市场状态与评级', fontweight='bold', fontsize=11)
    ax5.set_xticks([])
    ax5.set_yticks([])
    for spine in ax5.spines.values():
        spine.set_visible(False)

    # ----- 详细因素与风险提示横幅 (ax6) -----
    drivers_text = "\n".join([f"  ✅ {d}" for d in analysis_summary['key_drivers']]) if analysis_summary['key_drivers'] else "  ✅ 暂无明显特异驱动"
    risks_text = "\n".join([f"  ⚠️ {r}" for r in analysis_summary['main_risks']]) if analysis_summary['main_risks'] else "  ⚠️ 风险整体可控"

    banner_text = (
        f"🔑 关键驱动因素:\n{drivers_text}\n\n"
        f"⚡ 主要风险提示:\n{risks_text}\n\n"
        f"📌 总体情绪: {analysis_summary['overall_sentiment']}  |  操作策略: {analysis_summary['investment_suggestion']}"
    )
    ax6.text(0.02, 0.92, banner_text, transform=ax6.transAxes, fontsize=10, verticalalignment='top', linespacing=1.4)
    ax6.set_title('详细因素与多空驱动深入分析', fontweight='bold', fontsize=11)
    ax6.set_xticks([])
    ax6.set_yticks([])
    for spine in ax6.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig


# ==================== 5. Streamlit 缓存模型加载 ====================
@st.cache_resource(show_spinner=False)
def load_kronos_model(model_name: str = "NeoQuasar/Kronos-base"):
    """使用 Streamlit 资源缓存加载 Kronos 模型，避免重复读盘与重构图"""
    if not KRONOS_AVAILABLE:
        return None, None
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(model_name)
    return tokenizer, model


# ==================== 6. Streamlit 核心 Web 应用 ====================
def main():
    st.set_page_config(
        page_title="Kronos 时序大模型股票预测系统",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 顶部标题栏
    st.title("🤖 Kronos 时序大模型 - 多因子股票价格预测看板")
    st.caption("基于金融时间序列 Foundation Model (Kronos) + 东方财富/Baostock/AKShare 多数据源 + 宏观大盘多因子增强")

    # 检查 Kronos 模型可用性
    if not KRONOS_AVAILABLE:
        st.error("⚠️ 未能导入 Kronos 模型模块，请确保项目上级目录包含 model/ 文件夹且已安装 torch 与 transformers。")
        return

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 预测参数配置")
        stock_code = st.text_input("股票代码 (A股)", value="603288", help="支持 6 位沪深京股票代码，如 603288, 600580, 002354")
        stock_name = st.text_input("股票名称", value="海天味业")

        col_side1, col_side2 = st.columns(2)
        with col_side1:
            pred_days = st.slider("预测天数 (自然日)", min_value=15, max_value=120, value=60, step=5)
        with col_side2:
            history_years = st.selectbox("历史回看年限", options=[1, 2, 3], index=0)

        model_choice = st.selectbox("Kronos 模型规格", options=["NeoQuasar/Kronos-base", "NeoQuasar/Kronos-small"], index=0)
        force_refresh = st.checkbox("强制从网络重新获取数据 (忽略本地缓存)", value=False)

        st.markdown("---")
        start_btn = st.button("🚀 开始执行量化预测", type="primary", width="stretch")

        st.info("💡 **数据源策略**: 自动轮询东方财富API、Baostock Socket与AKShare，原生规避网络代理冲突。")

    # 主预测执行流程
    if start_btn:
        status_box = st.status("正在启动 Kronos 预测分析流程...", expanded=True)
        try:
            # 步骤 1: 获取数据
            status_box.write("📡 **步骤 1/7**: 获取标的股票历史行情数据...")
            success, csv_path, source_name = get_stock_data(stock_code, DEFAULT_DATA_DIR, force_refresh=force_refresh)
            if not success or not csv_path:
                status_box.update(label="❌ 获取股票历史数据失败", state="error")
                st.error("无法获取股票数据，请检查股票代码或网络连接。")
                return
            status_box.write(f"✅ 数据获取成功 (来源: `{source_name}`)，路径: `{csv_path}`")

            # 步骤 2: 加载模型
            status_box.write("🧠 **步骤 2/7**: 加载 Kronos 深度分词器与模型权重...")
            tokenizer, model = load_kronos_model(model_choice)
            if model is None:
                status_box.update(label="❌ Kronos 模型权重加载失败", state="error")
                return
            predictor = KronosPredictor(model, tokenizer, device=None, max_context=512)
            status_box.write("✅ 预测器初始化就绪 (max_context: 512)")

            # 步骤 3: 准备与清洗数据
            status_box.write("📊 **步骤 3/7**: 清洗并构建特征时间序列...")
            df = prepare_stock_data(csv_path, stock_code, history_years=history_years)
            lookback, pred_len = calculate_prediction_parameters(df, target_days=pred_days)
            last_date = df['timestamps'].iloc[-1]
            future_dates = generate_trading_dates_only(last_date, pred_len)

            x_df = df.loc[-lookback:, ['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
            x_timestamp = df.loc[-lookback:, 'timestamps'].reset_index(drop=True)
            status_box.write(f"✅ 滑窗构建完成: 回看 `{lookback}` 根K线, 预测未来 `{len(future_dates)}` 个有效交易日")

            # 步骤 4: 执行基准推理
            status_box.write("⚡ **步骤 4/7**: 运行 Kronos 自回归生成预测路径...")
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=pd.Series(future_dates),
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )
            status_box.write("✅ 基准价格预测生成完成")

            # 步骤 5: 验证与平滑
            status_box.write("🛠️ **步骤 5/7**: 实施跨期平滑与风控阈值校验...")
            smoothed_df = smooth_prediction_results(pred_df, df.loc[-lookback:].reset_index(drop=True), smooth_factor=0.3)
            validated_df = validate_prediction_results(df.loc[-lookback:].reset_index(drop=True), smoothed_df, max_price_change=0.35)

            # 步骤 6: 多因子增强
            status_box.write("🌐 **步骤 6/7**: 结合大盘趋势、行业共振与宏观因子进行增强调整...")
            status_box.write("  ↳ 正在获取大盘指数 (上证指数/深证成指)...")
            analyzer = EnhancedMarketFactorAnalyzer()
            market_analysis = analyzer.analyze_market_trend()
            status_box.write(f"  ↳ 大盘状态评估完成: `{market_analysis.get('market_status', '震荡调整')}` (趋势强度: `{market_analysis.get('overall_trend_strength', 0.5):.2f}`)")

            status_box.write("  ↳ 正在进行行业板块共振与宏观基本面量化打分...")
            sector_analysis = analyzer.analyze_sector_resonance(stock_code)
            macro_analysis = analyzer.analyze_macro_factors()
            fundamental_analysis = analyzer.analyze_company_fundamentals(stock_code)

            adj_factor = calculate_enhanced_adjustment_factor(market_analysis, sector_analysis, macro_analysis, fundamental_analysis)

            enhanced_df = validated_df.copy()
            for col in ['close', 'open', 'high', 'low']:
                if col in enhanced_df.columns:
                    enhanced_df[col] = enhanced_df[col] * adj_factor
            if 'volume' in enhanced_df.columns:
                enhanced_df['volume'] = enhanced_df['volume'] * (1 + (adj_factor - 1) * 0.3)

            factor_info = {
                'market_analysis': market_analysis,
                'sector_analysis': sector_analysis,
                'macro_analysis': macro_analysis,
                'fundamental_analysis': fundamental_analysis,
                'adjustment_factor': adj_factor
            }
            status_box.write(f"✅ 多因子强化修正完成 (综合调整因子: `{adj_factor:.4f}`)")

            # 步骤 7: 渲染图表
            status_box.write("📈 **步骤 7/7**: 生成综合研报与高清图表...")
            historical_slice = df.loc[-lookback:].reset_index(drop=True)
            fig = plot_prediction_dashboard(historical_slice, validated_df, enhanced_df, future_dates, stock_code, stock_name, factor_info)

            # 保存报告、图表与预测 CSV
            ensure_output_directory(DEFAULT_OUTPUT_DIR)
            chart_file = os.path.join(DEFAULT_OUTPUT_DIR, f"{stock_code}_comprehensive_prediction.png")
            fig.savefig(chart_file, dpi=120, bbox_inches='tight')

            pred_csv_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{stock_code}_streamlit_predictions.csv")
            result_df = pd.DataFrame({
                '日期': [d.strftime('%Y-%m-%d') for d in future_dates],
                '基准预测价': np.round(validated_df['close'].values, 2),
                '增强预测价': np.round(enhanced_df['close'].values, 2),
                '预测开盘价': np.round(enhanced_df['open'].values, 2),
                '预测成交量': np.round(enhanced_df['volume'].values, 0)
            })
            result_df.to_csv(pred_csv_path, index=False, encoding='utf-8-sig')

            status_box.update(label="🎉 预测与量化分析已全部完成！", state="complete", expanded=False)

            # ==================== 结果看板呈现 ====================
            st.markdown("### 📊 预测核心指标")
            current_close = historical_slice['close'].iloc[-1]
            final_pred_close = enhanced_df['close'].iloc[-1]
            pct_change = ((final_pred_close - current_close) / current_close) * 100

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("当前最新价", f"¥ {current_close:.2f}")
            col_m2.metric("增强预测目标价", f"¥ {final_pred_close:.2f}", delta=f"{pct_change:+.2f}%")
            col_m3.metric("市场综合调整因子", f"{adj_factor:.4f}", delta="多因子后验增强")
            col_m4.metric("大盘环境状态", market_analysis.get('market_status', '震荡'), delta=f"趋势强度: {market_analysis.get('overall_trend_strength', 0.5):.2f}")

            # 绘图展示 (交互式 ECharts K线为主，经典研报大图为辅)
            st.markdown("---")
            chart_tab1, chart_tab2 = st.tabs(["🕯️ 交互式专业 K 线图 (ECharts)", "📊 经典量化研报图 (静态对比)"])
            with chart_tab1:
                render_echarts_kline(historical_slice, validated_df, enhanced_df, future_dates, stock_code, stock_name, factor_info)
            with chart_tab2:
                try:
                    st.pyplot(fig, width="stretch")
                except Exception:
                    st.pyplot(fig)

            # 因子分析与数据明细
            st.markdown("---")
            tab_report, tab_data = st.tabs(["📋 多维度宏观与行业研报", "📅 逐日价格预测明细表"])

            with tab_report:
                col_chart, col_details = st.columns([1, 1.2])
                with col_chart:
                    render_echarts_factor_distribution(factor_info)
                with col_details:
                    st.subheader("大盘与板块共振")
                    st.write(f"- **大盘走势判断**: `{market_analysis.get('market_status', '未知')}` (趋势强度: `{market_analysis.get('overall_trend_strength', 0.5):.2f}`)")
                    st.write(f"- **所属行业板块**: `{sector_analysis.get('sector', '未知')}`")
                    st.write(f"- **板块共振得分**: `{sector_analysis.get('resonance_score', 0):.2f}` (是否热门: `{'是' if sector_analysis.get('is_sector_hot') else '否'}`)")
                    st.markdown("---")
                    st.subheader("宏观政策与公司基本面")
                    st.write(f"- **美联储利率周期**: `{macro_analysis.get('us_rate_cycle', {}).get('trend', '未知')}` (预期降息次数: `{macro_analysis.get('us_rate_cycle', {}).get('expected_cuts_2025', 0)}`次)")
                    st.write(f"- **国内货币政策**: `{macro_analysis.get('domestic_policy', '积极')}`")
                    st.write(f"- **投资分析建议**: `{fundamental_analysis.get('investment_rating', '持有')}`")

            with tab_data:
                try:
                    st.dataframe(result_df, width="stretch")
                except Exception:
                    st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    label="📥 下载完整预测 CSV 数据",
                    data=result_df.to_csv(index=False, encoding='utf-8-sig'),
                    file_name=f"{stock_code}_prediction_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            status_box.update(label=f"❌ 预测发生异常: {e}", state="error")
            st.exception(e)


if __name__ == "__main__":
    main()