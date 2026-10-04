import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sys
import os
import re
import requests
import json
import time
import random
import contextlib
import concurrent.futures
import socket
from datetime import datetime, timedelta
import warnings

# 全局设置底层网络超时，彻底防止任何三方库卡死
socket.setdefaulttimeout(8.0)
warnings.filterwarnings('ignore')

# 添加项目路径以便导入自定义模块
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
for p in [PROJECT_ROOT, CURRENT_SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.append(p)

from model import Kronos, KronosTokenizer, KronosPredictor

# 设置中文字体（跨平台兼容 macOS、Windows、Linux）
plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'STHeiti',  # macOS
    'SimHei', 'Microsoft YaHei',  # Windows
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC',  # Linux
    'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def ensure_output_directory(output_dir):
    """确保输出目录存在，如果不存在则创建"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    return output_dir


# ==================== 优化的多源数据自动获取模块 ====================

def get_stock_market(stock_code):
    """
    根据股票代码判断市场类型
    返回: 市场前缀 '0'-深交所/北交所, '1'-上交所
    """
    if stock_code.startswith(('0', '2', '3')):
        return '0'  # 深交所
    elif stock_code.startswith(('6', '9')):
        return '1'  # 上交所
    elif stock_code.startswith(('4', '8')):
        return '0'  # 北交所
    else:
        return '1'  # 默认上交所


def detect_market_type(stock_code, market=None):
    """
    智能检测股票所属市场类型。
    若显式指定了 market 参数则直接使用；否则根据 stock_code 格式自动推断。

    参数:
        stock_code: 股票代码
        market: 显式市场标识 ('CN'/'HK'/'US')，若为 None 则自动推断

    返回:
        'CN' (A股) / 'HK' (港股) / 'US' (美股)
    """
    if market and str(market).strip().upper() in ('CN', 'HK', 'US'):
        return str(market).strip().upper()
    code_clean = str(stock_code).strip().upper()
    # 剥离市场前缀后再推断
    if code_clean.startswith('HK_'):
        return 'HK'
    if code_clean.startswith('US_'):
        return 'US'
    if code_clean.startswith(('SH_', 'SZ_', 'CN_')):
        return 'CN'
    if code_clean.endswith('.HK') or code_clean.startswith('HK'):
        return 'HK'
    if code_clean.endswith(('.SS', '.SH', '.SZ', '.BJ')):
        return 'CN'
    # 美股：纯字母，或字母.字母 / 字母-字母（BRK.B / BRK-B）
    bare = code_clean
    if re.fullmatch(r'[A-Z]{1,5}[.\-][A-Z]{1,2}', bare) or (
        bare.isalpha() and 1 <= len(bare) <= 6
    ):
        return 'US'
    if code_clean.isdigit() and len(code_clean) <= 5:
        return 'HK'
    return 'CN'


def validate_stock_code(stock_code, market=None):
    """
    对用户输入的股票代码进行格式校验与市场规则匹配，提前拦截明显非法输入。

    返回:
        (is_valid: bool, error_msg: str, normalized_code: str, detected_market: str)
    """
    if stock_code is None or not str(stock_code).strip():
        return False, "股票代码不能为空", "", ""

    raw_code = str(stock_code).strip()
    code_str = raw_code.upper()

    # 解析市场与代码后缀 / 前缀（与 SymbolResolver 对齐）
    detected_market = str(market).strip().upper() if market else None

    if code_str.startswith('HK_'):
        detected_market = detected_market or 'HK'
        code_str = code_str[3:].strip()
    elif code_str.startswith('US_'):
        detected_market = detected_market or 'US'
        code_str = code_str[3:].strip()
    elif code_str.startswith(('SH_', 'SZ_', 'CN_')):
        detected_market = detected_market or 'CN'
        code_str = code_str[3:].strip()

    if code_str.endswith('.HK'):
        detected_market = detected_market or 'HK'
        code_str = code_str[:-3].strip()
    elif code_str.startswith('HK') and code_str[2:].isdigit():
        detected_market = detected_market or 'HK'
        code_str = code_str[2:].strip()
    elif code_str.endswith(('.SS', '.SH', '.SZ', '.BJ')):
        detected_market = detected_market or 'CN'
        code_str = code_str.split('.')[0].strip()

    if not detected_market:
        detected_market = detect_market_type(raw_code, market)

    # 1. A股市场规则校验
    if detected_market == 'CN':
        if not code_str.isdigit():
            return False, f"A股代码格式非法: [{raw_code}] 必须为6位纯数字", code_str, detected_market
        if len(code_str) != 6:
            return False, f"A股代码长度错误: [{raw_code}] 应为6位数字(当前为{len(code_str)}位)", code_str, detected_market
        # 常见A股号段校验 (60/68/00/30/4/8/92)
        if not code_str.startswith(('00', '30', '60', '68', '4', '8', '92')):
            return False, f"A股代码号段未知: [{raw_code}] 不是常规沪深北交所股票代码", code_str, detected_market
        return True, "OK", code_str, detected_market

    # 2. 港股市场规则校验
    elif detected_market == 'HK':
        if not code_str.isdigit():
            return False, f"港股代码格式非法: [{raw_code}] 必须为1~5位数字代码", code_str, detected_market
        if len(code_str) > 5 or len(code_str) == 0:
            return False, f"港股代码长度错误: [{raw_code}] 应为1~5位数字", code_str, detected_market
        # 自动补全为5位标准代码 (如 700 -> 00700, 1810 -> 01810)
        normalized_code = code_str.zfill(5)
        return True, "OK", normalized_code, detected_market

    # 3. 美股市场规则校验
    elif detected_market == 'US':
        # 纯字母，或字母.字母 / 字母-字母（BRK.B / BRK-B）
        code_str = code_str.replace('-', '.')
        code_check = code_str.replace('.', '')
        if not code_check.isalpha():
            return False, f"美股代码格式非法: [{raw_code}] 必须为英文字母代码（可含一个.或-）", code_str, detected_market
        if len(code_check) > 6:
            return False, f"美股代码过长: [{raw_code}] 异常长度", code_str, detected_market
        if code_str.count('.') > 1:
            return False, f"美股代码格式非法: [{raw_code}] 至多含一个小数点分段", code_str, detected_market
        return True, "OK", code_str, detected_market

    return False, f"未知或不支持的市场类型: [{detected_market}]", code_str, detected_market


def build_yfinance_ticker(stock_code, market):
    """
    根据市场类型构建 yfinance 的 ticker 符号

    参数:
        stock_code: 原始股票代码
        market: 市场类型 ('CN'/'HK'/'US')

    返回:
        yfinance ticker 字符串
    """
    market = market.upper()
    if market == 'HK':
        # 港股: 数字代码 + .HK, 如 00700 -> 0700.HK
        code = str(stock_code).lstrip('0') or '0'
        # 港股 ticker 至少4位数字
        code = code.zfill(4)
        return f"{code}.HK"
    elif market == 'US':
        # 美股: 直接使用字母 ticker
        return str(stock_code).upper()
    else:
        # A股
        code = str(stock_code).zfill(6)
        if code.startswith(('60', '68')):
            return f"{code}.SS"
        elif code.startswith(('00', '30')):
            return f"{code}.SZ"
        elif code.startswith(('4', '8', '92')):
            return f"{code}.BJ"
        return code


# ==================== 多市场节假日集合 ====================

# A股法定休市节假日 (2024~2027)
CN_MARKET_HOLIDAYS = {
    # 2024
    "2024-01-01", "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14",
    "2024-02-15", "2024-02-16", "2024-04-04", "2024-04-05", "2024-05-01",
    "2024-05-02", "2024-05-03", "2024-06-10", "2024-09-16", "2024-09-17",
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-07",
    # 2025
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30",
    "2025-01-31", "2025-02-03", "2025-02-04", "2025-04-04", "2025-05-01",
    "2025-05-02", "2025-05-05", "2025-06-02", "2025-10-01", "2025-10-02",
    "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08",
    # 2026
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19",
    "2026-02-20", "2026-04-06", "2026-05-01", "2026-05-04", "2026-05-05",
    "2026-06-19", "2026-09-25", "2026-10-01", "2026-10-02", "2026-10-05",
    "2026-10-06", "2026-10-07",
    # 2027
    "2027-01-01", "2027-02-05", "2027-02-08", "2027-02-09", "2027-02-10",
    "2027-02-11", "2027-02-12", "2027-04-05", "2027-05-03", "2027-06-09",
    "2027-10-01", "2027-10-04", "2027-10-05", "2027-10-06", "2027-10-07"
}

# 港股 (HKEX) 法定休市节假日 (2024~2027)
HK_MARKET_HOLIDAYS = {
    # 2024
    "2024-01-01", "2024-02-12", "2024-02-13",  # 元旦, 春节
    "2024-03-29", "2024-04-01",  # 耶稣受难节, 复活节星期一
    "2024-04-04",  # 清明节
    "2024-05-01", "2024-05-15",  # 劳动节, 佛诞
    "2024-06-10",  # 端午节
    "2024-07-01",  # 回归日
    "2024-09-18",  # 中秋节翌日
    "2024-10-01", "2024-10-11",  # 国庆, 重阳节
    "2024-12-25", "2024-12-26",  # 圣诞节
    # 2025
    "2025-01-01",  # 元旦
    "2025-01-29", "2025-01-30", "2025-01-31",  # 春节
    "2025-04-04",  # 清明节
    "2025-04-18", "2025-04-21",  # 耶稣受难节, 复活节星期一
    "2025-05-01", "2025-05-05",  # 劳动节, 佛诞
    "2025-05-31",  # 端午节
    "2025-07-01",  # 回归日
    "2025-10-01", "2025-10-07",  # 国庆, 重阳节
    "2025-10-29",  # 中秋节翌日（特殊年份）
    "2025-12-25", "2025-12-26",  # 圣诞节
    # 2026
    "2026-01-01",  # 元旦
    "2026-02-17", "2026-02-18", "2026-02-19",  # 春节
    "2026-04-03", "2026-04-06", "2026-04-07",  # 耶稣受难节, 复活节, 清明节
    "2026-05-01", "2026-05-25",  # 劳动节, 佛诞
    "2026-06-19",  # 端午节
    "2026-07-01",  # 回归日
    "2026-09-26",  # 中秋节翌日
    "2026-10-01", "2026-10-19",  # 国庆, 重阳节
    "2026-12-25",  # 圣诞节
    # 2027
    "2027-01-01",  # 元旦
    "2027-02-08", "2027-02-09", "2027-02-10",  # 春节
    "2027-03-26", "2027-03-29",  # 耶稣受难节, 复活节星期一
    "2027-04-05",  # 清明节
    "2027-05-13",  # 佛诞
    "2027-06-09",  # 端午节
    "2027-07-01",  # 回归日
    "2027-09-16",  # 中秋节翌日
    "2027-10-01", "2027-10-08",  # 国庆, 重阳节
    "2027-12-25", "2027-12-27",  # 圣诞节
}

# 美股 (NYSE/NASDAQ) 法定休市节假日 (2024~2027)
US_MARKET_HOLIDAYS = {
    # 2024
    "2024-01-01", "2024-01-15",  # 元旦, 马丁·路德·金日
    "2024-02-19",  # 总统日
    "2024-03-29",  # 耶稣受难日
    "2024-05-27",  # 阵亡将士纪念日
    "2024-06-19",  # 六月节
    "2024-07-04",  # 独立日
    "2024-09-02",  # 劳动节
    "2024-11-28",  # 感恩节
    "2024-12-25",  # 圣诞节
    # 2025
    "2025-01-01", "2025-01-20",  # 元旦, 马丁·路德·金日
    "2025-02-17",  # 总统日
    "2025-04-18",  # 耶稣受难日
    "2025-05-26",  # 阵亡将士纪念日
    "2025-06-19",  # 六月节
    "2025-07-04",  # 独立日
    "2025-09-01",  # 劳动节
    "2025-11-27",  # 感恩节
    "2025-12-25",  # 圣诞节
    # 2026
    "2026-01-01", "2026-01-19",  # 元旦, 马丁·路德·金日
    "2026-02-16",  # 总统日
    "2026-04-03",  # 耶稣受难日
    "2026-05-25",  # 阵亡将士纪念日
    "2026-06-19",  # 六月节
    "2026-07-03",  # 独立日 (7/4 为周六, 周五观察)
    "2026-09-07",  # 劳动节
    "2026-11-26",  # 感恩节
    "2026-12-25",  # 圣诞节
    # 2027
    "2027-01-01", "2027-01-18",  # 元旦, 马丁·路德·金日
    "2027-02-15",  # 总统日
    "2027-03-26",  # 耶稣受难日
    "2027-05-31",  # 阵亡将士纪念日
    "2027-06-18",  # 六月节 (6/19 为周六, 周五观察)
    "2027-07-05",  # 独立日 (7/4 为周日, 周一观察)
    "2027-09-06",  # 劳动节
    "2027-11-25",  # 感恩节
    "2027-12-24",  # 圣诞节 (12/25 为周六, 周五观察)
}

# 市场节假日快速查找映射
MARKET_HOLIDAYS_MAP = {
    'CN': CN_MARKET_HOLIDAYS,
    'HK': HK_MARKET_HOLIDAYS,
    'US': US_MARKET_HOLIDAYS,
}

# 市场货币单位映射
MARKET_CURRENCY_MAP = {
    'CN': ('CNY', '元'),
    'HK': ('HKD', '港元'),
    'US': ('USD', '美元'),
}


def get_hk_us_stock_data(stock_code, market, start_date=None, end_date=None, start_year=None, end_year=None):
    """
    使用 yfinance 获取港股或美股历史行情数据。
    统一输出格式与 A 股数据源一致（中文列名，DatetimeIndex='日期'）。

    参数:
        stock_code: 股票代码 (港股如 '00700', 美股如 'AAPL')
        market: 市场类型 ('HK' 或 'US')
    """
    try:
        import yfinance as yf
    except ImportError:
        print("⚠️ 未安装 yfinance 库，请执行: pip install yfinance")
        return None

    try:
        start_date, end_date, _ = resolve_date_range(end_date=end_date, start_date=start_date,
                                                      start_year=start_year, end_year=end_year)
        ticker_symbol = build_yfinance_ticker(stock_code, market)
        market_label = '港股' if market.upper() == 'HK' else '美股'
        print(f"📡 正在从 yfinance 获取{market_label} {stock_code} ({ticker_symbol}) 行情数据 ({start_date} ~ {end_date})...")

        yf_end = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        # auto_adjust=True：前复权 OHLC，与腾讯 qfq 口径对齐
        df = yf.download(
            ticker_symbol, start=start_date, end=yf_end,
            progress=False, auto_adjust=True,
        )

        if df is None or df.empty:
            print(f"⚠️ yfinance 未返回 {ticker_symbol} 的行情数据")
            return None

        # 处理 MultiIndex 列
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 统一时区
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = '日期'

        # 统一列名为中文
        rename_map = {
            'Open': '开盘价',
            'High': '最高价',
            'Low': '最低价',
            'Close': '收盘价',
            'Volume': '成交量'
        }
        df = df.rename(columns=rename_map)

        df['股票代码'] = stock_code
        if '成交额' not in df.columns:
            df['成交额'] = df['成交量'] * df['收盘价']

        if len(df) > 1:
            df['涨跌幅'] = df['收盘价'].pct_change() * 100
            df['涨跌额'] = df['收盘价'].diff()
            df['振幅'] = (df['最高价'] - df['最低价']) / df['收盘价'].shift(1) * 100

        df = df.sort_index()
        df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        _, currency_unit = MARKET_CURRENCY_MAP.get(market.upper(), ('', ''))
        print(f"✅ yfinance 成功获取{market_label} {len(df)} 条有效行情数据"
              f" ({df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')})")
        df.attrs['data_source'] = f'yfinance ({market_label})'
        return df
    except Exception as e:
        print(f"⚠️ yfinance 获取{market.upper()}市场数据异常: {e}")
        return None


@contextlib.contextmanager
def disable_proxy():
    """临时禁用系统/终端代理，确保国内财经数据接口直连不超时"""
    proxy_keys = ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']
    old_values = {k: os.environ.get(k) for k in proxy_keys}
    for k in proxy_keys:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in old_values.items():
            if v is not None:
                os.environ[k] = v


def resolve_date_range(end_date=None, start_date=None, history_days=365, pred_days=100, start_year=None, end_year=None):
    """
    智能推导时间范围：
    - 若未指定 end_date，且提供了 end_year，取 end_year-12-31；否则默认取当前系统日期（最新交易日）
    - 若未指定 start_date，且提供了 start_year，取 start_year-01-01；否则往前推算 max(history_days, pred_days * 3) 天
    返回: (start_date_str, end_date_str, is_latest_mode)
    """
    is_latest_mode = False
    if end_date is not None:
        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date.replace('/', '-'), '%Y-%m-%d')
        else:
            end_dt = end_date
    elif end_year is not None:
        end_dt = datetime(int(end_year), 12, 31)
    else:
        end_dt = datetime.now()
        is_latest_mode = True

    if start_date is not None:
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date.replace('/', '-'), '%Y-%m-%d')
        else:
            start_dt = start_date
    elif start_year is not None:
        start_dt = datetime(int(start_year), 1, 1)
    else:
        effective_days = max(int(history_days), int(pred_days) * 3)
        start_dt = end_dt - timedelta(days=effective_days)

    start_date_str = start_dt.strftime('%Y-%m-%d')
    end_date_str = end_dt.strftime('%Y-%m-%d')
    return start_date_str, end_date_str, is_latest_mode


def is_cache_fresh(df, is_latest_mode, target_start_date, target_end_date):
    """
    判断本地缓存数据是否新鲜有效：
    1. 若指定了历史回测区间，检查缓存是否完整覆盖了 target_start_date 到 target_end_date；
    2. 若是最新模式 (is_latest_mode=True):
       - 检查缓存最新一条数据的日期 last_date；
       - 若 last_date 就是今天，直接新鲜命中；
       - 若今天是工作日（周一至周五）且当前时间已过 15:00（收盘），如果 last_date < 今天，则需要刷新；
       - 若今天是交易时间内（9:30-15:00），昨日收盘数据视为有效新鲜；
       - 若今天是周末，只要包含周五收盘数据即为新鲜有效。
    """
    try:
        if '日期' in df.columns:
            dt_series = pd.to_datetime(df['日期'])
        elif 'timestamps' in df.columns:
            dt_series = pd.to_datetime(df['timestamps'])
        elif isinstance(df.index, pd.DatetimeIndex):
            dt_series = df.index
        else:
            return False

        first_date = dt_series.min().date()
        last_date = dt_series.max().date()

        now = datetime.now()
        today = now.date()

        # 历史回测模式：必须双向覆盖目标区间
        if not is_latest_mode and target_end_date:
            target_end = pd.to_datetime(target_end_date).date()
            target_start = pd.to_datetime(target_start_date).date() if target_start_date else None
            if target_start and first_date > target_start:
                return False
            return last_date >= target_end

        # 最新模式
        if last_date >= today:
            return True

        days_diff = (today - last_date).days

        # 相隔超过 4 天，必然跨过了交易日，缓存已落后
        if days_diff > 4:
            return False

        # 工作日判断
        if today.weekday() < 5:
            # 超过 15:00 收盘时间，缓存却未更新到今天 -> 需要刷新
            if now.hour >= 15:
                return False
            # 尚未收盘，如果包含了昨日数据，视为新鲜
            elif days_diff <= 1 and last_date == (today - timedelta(days=1)):
                return True

        # 周末判断
        if today.weekday() >= 5:
            friday = today - timedelta(days=(today.weekday() - 4))
            if last_date >= friday:
                return True

        return False
    except Exception:
        return False


def get_stock_data_tencent(stock_code="002354", start_date=None, end_date=None, start_year=None, end_year=None, market=None):
    """
    使用腾讯财经API获取股票K线（支持A股、港股与美股，直连极速、无反爬风控、支持前复权、自动支持动态日期）
    """
    try:
        detected_market = detect_market_type(stock_code, market)
        start_date, end_date, _ = resolve_date_range(end_date=end_date, start_date=start_date,
                                                     start_year=start_year, end_year=end_year)
        market_label = {'CN': 'A股', 'HK': '港股', 'US': '美股'}.get(detected_market, '股票')
        print(f"📡 正在从腾讯财经获取{market_label}股票 {stock_code} 的行情数据 ({start_date} ~ {end_date})...")

        # 1. 美股处理逻辑 (探测 .OQ / .N 等后缀)
        if detected_market == 'US':
            ticker_upper = str(stock_code).strip().upper()
            candidate_symbols = [f"us{ticker_upper}.OQ", f"us{ticker_upper}.N", f"us{ticker_upper}"]
            klines = []
            matched_symbol = None
            for sym in candidate_symbols:
                url = f"https://web.ifzq.gtimg.cn/appstock/app/usfqkline/get?param={sym},day,{start_date},{end_date},1200,qfq"
                try:
                    res = requests.get(url, proxies={'http': None, 'https': None}, timeout=8)
                    if res.status_code == 200:
                        data = res.json()
                        dict_data = data.get('data', {}).get(sym, {})
                        k_list = dict_data.get('qfqday', dict_data.get('day', []))
                        if len(k_list) > 2:
                            klines = k_list
                            matched_symbol = sym
                            break
                except Exception:
                    continue

            if not klines:
                return None

            records = []
            for item in klines:
                if len(item) >= 6:
                    close_p = float(item[2])
                    vol = float(item[5])
                    records.append({
                        '日期': item[0],
                        '股票代码': ticker_upper,
                        '开盘价': float(item[1]),
                        '收盘价': close_p,
                        '最高价': float(item[3]),
                        '最低价': float(item[4]),
                        '成交量': vol,
                        '成交额': close_p * vol
                    })

            df = pd.DataFrame(records)
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            df = df.sort_index()
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            if len(df) > 1:
                df['涨跌幅'] = df['收盘价'].pct_change() * 100
                df['涨跌额'] = df['收盘价'].diff()
                df['振幅'] = (df['最高价'] - df['最低价']) / df['收盘价'].shift(1) * 100

            print(f"✅ 腾讯财经成功获取美股 {len(df)} 条有效行情数据 ({df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')})")
            df.attrs['data_source'] = f'腾讯财经 (美股/{matched_symbol})'
            return df

        # 2. 港股与A股处理逻辑
        if detected_market == 'HK':
            symbol = f"hk{str(stock_code).zfill(5)}"
        elif str(stock_code).startswith(('60', '68')):
            symbol = f"sh{stock_code}"
        elif str(stock_code).startswith(('4', '8', '92')):
            symbol = f"bj{stock_code}"
        else:
            symbol = f"sz{stock_code}"

        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,{start_date},{end_date},1200,qfq"

        res = requests.get(url, proxies={'http': None, 'https': None}, timeout=8)
        if res.status_code != 200:
            return None

        data = res.json()
        stock_dict = data.get('data', {}).get(symbol, {})
        klines = stock_dict.get('qfqday', stock_dict.get('day', []))
        if not klines:
            return None

        records = []
        for item in klines:
            if len(item) >= 6:
                close_p = float(item[2])
                # 港股成交量单位已为股，A股单位为手需乘100
                vol = float(item[5]) if symbol.startswith('hk') else float(item[5]) * 100.0
                records.append({
                    '日期': item[0],
                    '股票代码': str(stock_code),
                    '开盘价': float(item[1]),
                    '收盘价': close_p,
                    '最高价': float(item[3]),
                    '最低价': float(item[4]),
                    '成交量': vol,
                    '成交额': close_p * vol
                })

        if not records:
            return None

        df = pd.DataFrame(records)
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        df = df.sort_index()
        df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

        if len(df) > 1:
            df['涨跌幅'] = df['收盘价'].pct_change() * 100
            df['涨跌额'] = df['收盘价'].diff()
            df['振幅'] = (df['最高价'] - df['最低价']) / df['收盘价'].shift(1) * 100

        print(f"✅ 腾讯财经成功获取{market_label} {len(df)} 条有效行情数据 ({df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')})")
        df.attrs['data_source'] = f'腾讯财经 ({market_label})'
        return df
    except Exception as e:
        print(f"⚠️ 腾讯财经接口异常: {e}")
        return None


def get_stock_data_eastmoney(stock_code="002354", start_date=None, end_date=None, start_year=None, end_year=None):
    """
    使用东方财富网API获取股票数据（速度极快，无需额外安装第三方依赖）
    """
    try:
        start_date, end_date, _ = resolve_date_range(end_date=end_date, start_date=start_date,
                                                     start_year=start_year, end_year=end_year)
        print(f"📡 正在从东方财富网获取股票 {stock_code} 的行情数据 ({start_date} ~ {end_date})...")

        beg_date = start_date.replace('-', '')
        end_date_clean = end_date.replace('-', '')

        market = get_stock_market(stock_code)
        secid = f"{market}.{stock_code}"

        url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            'secid': secid,
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': '101',  # 日线
            'fqt': '1',    # 前复权
            'beg': beg_date,
            'end': end_date_clean,
            'lmt': '10000',
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
            'cb': f'jQuery{random.randint(1000000, 9999999)}_{int(time.time()*1000)}'
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            'Referer': 'https://quote.eastmoney.com/',
            'Accept': '*/*',
        }

        response = requests.get(url, params=params, headers=headers, timeout=10, proxies={'http': None, 'https': None})

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
                    return parse_kline_data_directly(response_text, stock_code, start_date, end_date)
            else:
                return None

            if data and data.get('data') is not None:
                klines = data['data'].get('klines', [])
                if not klines:
                    print("⚠️ 东方财富返回K线数据为空")
                    return None

                stock_data = []
                for kline in klines:
                    try:
                        items = kline.split(',')
                        if len(items) >= 6:
                            stock_data.append({
                                '日期': items[0],
                                '股票代码': stock_code,
                                '开盘价': float(items[1]),
                                '收盘价': float(items[2]),
                                '最高价': float(items[3]),
                                '最低价': float(items[4]),
                                '成交量': float(items[5]),
                                '成交额': float(items[6]) if len(items) > 6 else 0.0,
                                '振幅': float(items[7]) if len(items) > 7 else 0.0,
                                '涨跌幅': float(items[8]) if len(items) > 8 else 0.0,
                                '涨跌额': float(items[9]) if len(items) > 9 else 0.0,
                                '换手率': float(items[10]) if len(items) > 10 else 0.0
                            })
                    except (ValueError, IndexError):
                        continue

                if not stock_data:
                    return None

                df = pd.DataFrame(stock_data)
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                df = df.sort_index()

                df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
                print(f"✅ 东方财富接口成功获取 {len(df)} 条有效行情数据 ({df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')})")
                df.attrs['data_source'] = '东方财富'
                return df
        return None
    except Exception as e:
        print(f"⚠️ 东方财富接口请求异常: {e}")
        return None


def parse_kline_data_directly(response_text, stock_code, start_date, end_date):
    """
    直接解析K线数据（当JSONP解析失败时备用）
    """
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
                            '日期': items[0],
                            '股票代码': stock_code,
                            '开盘价': float(items[1]),
                            '收盘价': float(items[2]),
                            '最高价': float(items[3]),
                            '最低价': float(items[4]),
                            '成交量': float(items[5]),
                            '成交额': float(items[6]) if len(items) > 6 else 0.0,
                        })

            if stock_data:
                df = pd.DataFrame(stock_data)
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                df = df.sort_index()
                df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
                df.attrs['data_source'] = '东方财富(直接解析)'
                return df
    except Exception as e:
        print(f"❌ 直接解析失败: {e}")
    return None


def get_stock_data_akshare(stock_code="002354", start_date=None, end_date=None, start_year=None, end_year=None):
    """
    使用AKShare作为备选数据源（动态导入）
    """
    try:
        import akshare as ak
        start_date, end_date, _ = resolve_date_range(end_date=end_date, start_date=start_date,
                                                     start_year=start_year, end_year=end_year)
        print(f"📡 尝试使用 AKShare 获取股票 {stock_code} 数据 ({start_date} ~ {end_date})...")
        start_date_clean = start_date.replace('-', '')
        end_date_clean = end_date.replace('-', '')

        with disable_proxy():
            df = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                                   start_date=start_date_clean, end_date=end_date_clean,
                                   adjust="qfq")

        if df is not None and not df.empty:
            column_mapping = {
                '日期': '日期',
                '开盘': '开盘价',
                '收盘': '收盘价',
                '最高': '最高价',
                '最低': '最低价',
                '成交量': '成交量',
                '成交额': '成交额',
                '振幅': '振幅',
                '涨跌幅': '涨跌幅',
                '涨跌额': '涨跌额',
                '换手率': '换手率'
            }
            actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=actual_mapping)

            df['股票代码'] = stock_code
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            df = df.sort_index()

            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            print(f"✅ AKShare成功获取 {len(df)} 条数据")
            df.attrs['data_source'] = 'AKShare'
            return df
        return None
    except ImportError:
        return None
    except Exception as e:
        print(f"⚠️ AKShare 获取数据异常: {e}")
        return None


def get_stock_data_baostock(stock_code="002354", start_date=None, end_date=None, start_year=None, end_year=None):
    """
    使用Baostock作为备选数据源（动态导入）
    """
    try:
        import baostock as bs
        start_date, end_date, _ = resolve_date_range(end_date=end_date, start_date=start_date,
                                                     start_year=start_year, end_year=end_year)
        print(f"📡 尝试使用 Baostock 获取股票 {stock_code} 数据 ({start_date} ~ {end_date})...")
        lg = bs.login()

        market = get_stock_market(stock_code)
        full_code = f"sz.{stock_code}" if market == '0' else f"sh.{stock_code}"

        rs = bs.query_history_k_data_plus(
            full_code,
            "date,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 前复权
        )

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()

        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            df['date'] = pd.to_datetime(df['date'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['amount'] = pd.to_numeric(df['amount'])

            df = df.rename(columns={
                'date': '日期',
                'open': '开盘价',
                'high': '最高价',
                'low': '最低价',
                'close': '收盘价',
                'volume': '成交量',
                'amount': '成交额',
            })
            df['股票代码'] = stock_code
            df.set_index('日期', inplace=True)
            df = df.sort_index()
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            print(f"✅ Baostock成功获取 {len(df)} 条数据")
            df.attrs['data_source'] = 'Baostock'
            return df
        return None
    except ImportError:
        return None
    except Exception as e:
        print(f"⚠️ Baostock 获取数据异常: {e}")
        return None



def get_stock_data_yfinance(stock_code="002354", start_date=None, end_date=None, start_year=None, end_year=None):
    """
    使用 yfinance 作为高可靠海外/代理环境数据源（仅用于 A 股 fallback）
    """
    try:
        import yfinance as yf
        start_date, end_date, _ = resolve_date_range(end_date=end_date, start_date=start_date,
                                                     start_year=start_year, end_year=end_year)
        print(f"📡 尝试使用 yfinance 获取股票 {stock_code} 数据 ({start_date} ~ {end_date})...")

        if stock_code.startswith(('60', '68')):
            ticker_symbol = f"{stock_code}.SS"
        elif stock_code.startswith(('00', '30')):
            ticker_symbol = f"{stock_code}.SZ"
        elif stock_code.startswith(('4', '8', '92')):
            ticker_symbol = f"{stock_code}.BJ"
        else:
            ticker_symbol = stock_code

        yf_end = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        # A 股备源同样使用前复权，与主源腾讯 qfq 对齐
        df = yf.download(
            ticker_symbol, start=start_date, end=yf_end,
            progress=False, auto_adjust=True,
        )

        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = '日期'

            rename_map = {
                'Open': '开盘价',
                'High': '最高价',
                'Low': '最低价',
                'Close': '收盘价',
                'Volume': '成交量'
            }
            df = df.rename(columns=rename_map)

            df['股票代码'] = stock_code
            if '成交额' not in df.columns:
                df['成交额'] = df['成交量'] * df['收盘价']

            if len(df) > 1:
                df['涨跌幅'] = df['收盘价'].pct_change() * 100
                df['涨跌额'] = df['收盘价'].diff()
                df['振幅'] = (df['最高价'] - df['最低价']) / df['收盘价'].shift(1) * 100

            df = df.sort_index()
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            print(f"✅ yfinance 成功获取 {len(df)} 条有效行情数据 ({df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')})")
            df.attrs['data_source'] = 'yfinance'
            return df
        return None
    except ImportError:
        print("⚠️ 未安装 yfinance 库")
        return None
    except Exception as e:
        print(f"⚠️ yfinance 获取数据异常: {e}")
        return None


def create_sample_fallback_data(stock_code="002354", start_date=None, end_date=None, start_year=None, end_year=None):
    """
    当网络完全不可用时的兜底模拟数据生成器
    """
    start_date, end_date, _ = resolve_date_range(end_date=end_date, start_date=start_date,
                                                 start_year=start_year, end_year=end_year)
    print(f"📊 创建 {start_date} ~ {end_date} 的高拟真兜底数据...")
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    trading_dates = pd.bdate_range(start=start_dt, end=end_dt, freq='B')

    np.random.seed(42)
    base_price = 10.0
    current_price = base_price
    stock_data = []

    for i, date in enumerate(trading_dates):
        volatility = 0.018
        if i > 0:
            daily_return = np.random.normal(0.0002, volatility)
            current_price = current_price * (1 + daily_return)
            current_price = max(base_price * 0.4, min(base_price * 3.0, current_price))

        open_var = np.random.normal(0, volatility * 0.3)
        open_price = current_price * (1 + open_var)
        daily_range = abs(np.random.normal(volatility * 0.8, volatility * 0.3))
        high_price = max(open_price, current_price) * (1 + daily_range)
        low_price = min(open_price, current_price) * (1 - daily_range)
        close_price = current_price

        volume = int(max(100000, np.random.normal(1200000, 300000)))
        amount = volume * close_price

        stock_data.append({
            '日期': date,
            '股票代码': stock_code,
            '开盘价': round(open_price, 2),
            '收盘价': round(close_price, 2),
            '最高价': round(high_price, 2),
            '最低价': round(low_price, 2),
            '成交量': volume,
            '成交额': round(amount, 2)
        })

    df = pd.DataFrame(stock_data)
    df.set_index('日期', inplace=True)
    df.attrs['data_source'] = '模拟兜底数据'
    return df


def get_stock_data_with_retry(stock_code="002354", start_date=None, end_date=None, start_year=None, end_year=None, market=None, allow_fallback=False):
    """
    优化的多数据源智能获取：
    - A股优先级：1. 腾讯财经 (极速直连) -> 2. 东方财富 API -> 3. yfinance -> 4. AKShare -> 5. Baostock
    - 港股/美股：直接使用 yfinance
    增加线程池超时保护，防止单数据源无限卡住；默认 allow_fallback=False 防止错误股票代码产生虚假模拟数据。
    """
    detected_market = detect_market_type(stock_code, market)
    start_date, end_date, _ = resolve_date_range(end_date=end_date, start_date=start_date,
                                                  start_year=start_year, end_year=end_year)

    # 港股：优先走腾讯财经直连极速接口(无频控)，备选走 yfinance
    if detected_market == 'HK':
        # 1. 优先尝试腾讯财经直连
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_stock_data_tencent, stock_code,
                                         start_date=start_date, end_date=end_date, market='HK')
                data = future.result(timeout=10)
            if data is not None and not data.empty:
                print(f"🎉 成功从 [腾讯财经] 取得港股真实行情数据！")
                return data
        except Exception as e:
            print(f"⚠️ 腾讯财经港股接口异常: {e}")

        # 2. 备选尝试 yfinance
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_hk_us_stock_data, stock_code, 'HK',
                                         start_date=start_date, end_date=end_date)
                data = future.result(timeout=15)
            if data is not None and not data.empty:
                print(f"🎉 成功从 [yfinance] 取得港股真实行情数据！")
                return data
        except concurrent.futures.TimeoutError:
            print(f"⚠️ yfinance 港股请求超时 (超过15秒)")
        except Exception as e:
            print(f"⚠️ yfinance 港股获取异常: {e}")

        if allow_fallback:
            print(f"⚠️ 港股数据获取失败，采用模拟兜底数据以保证程序运行...")
            return create_sample_fallback_data(stock_code, start_date=start_date, end_date=end_date)
        print(f"❌ 港股真实行情获取失败 (代码可能不存在、已退市或网络受限)")
        return None

    # 美股：优先走腾讯财经直连极速接口，备选走 yfinance
    elif detected_market == 'US':
        # 1. 优先尝试腾讯财经直连
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_stock_data_tencent, stock_code,
                                         start_date=start_date, end_date=end_date, market='US')
                data = future.result(timeout=10)
            if data is not None and not data.empty:
                print(f"🎉 成功从 [腾讯财经] 取得美股真实行情数据！")
                return data
        except Exception as e:
            print(f"⚠️ 腾讯财经美股接口异常: {e}")

        # 2. 备选尝试 yfinance
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_hk_us_stock_data, stock_code, 'US',
                                         start_date=start_date, end_date=end_date)
                data = future.result(timeout=15)
            if data is not None and not data.empty:
                print(f"🎉 成功从 [yfinance] 取得美股真实行情数据！")
                return data
        except concurrent.futures.TimeoutError:
            print(f"⚠️ yfinance 美股请求超时 (超过15秒)")
        except Exception as e:
            print(f"⚠️ yfinance 美股获取异常: {e}")

        if allow_fallback:
            print(f"⚠️ 美股数据获取失败，采用模拟兜底数据以保证程序运行...")
            return create_sample_fallback_data(stock_code, start_date=start_date, end_date=end_date)
        print(f"❌ 美股真实行情获取失败 (代码可能不存在、已退市或网络受限)")
        return None

    # A股：多数据源智能切换
    data_sources = [
        ("腾讯财经", get_stock_data_tencent),
        ("东方财富网", get_stock_data_eastmoney),
        ("yfinance", get_stock_data_yfinance),
        ("AKShare", get_stock_data_akshare),
        ("Baostock", get_stock_data_baostock),
    ]

    for source_name, data_func in data_sources:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(data_func, stock_code, start_date=start_date, end_date=end_date)
                data = future.result(timeout=8)

            if data is not None and not data.empty:
                print(f"🎉 成功从 [{source_name}] 取得真实行情数据！")
                return data
        except concurrent.futures.TimeoutError:
            print(f"⚠️ 数据源 [{source_name}] 请求超时 (超过8秒)，自动切换下一个源...")
        except Exception as e:
            print(f"⚠️ 数据源 [{source_name}] 获取异常: {e}")

    if allow_fallback:
        print("❌ 所有外部真实数据源请求失败，采用兜底数据以保证程序运行...")
        return create_sample_fallback_data(stock_code, start_date=start_date, end_date=end_date)
    print("❌ 所有外部真实数据源请求失败 (代码可能不存在或网络不可达)")
    return None



def save_stock_data_to_csv(df, stock_code, save_dir):
    """
    将抓取到的股票数据保存/缓存到指定 CSV 目录
    """
    if df is not None and not df.empty:
        os.makedirs(save_dir, exist_ok=True)
        csv_file = os.path.join(save_dir, f"{stock_code}_stock_data.csv")
        df_to_save = df.copy()
        if '日期' not in df_to_save.columns and df_to_save.index.name == '日期':
            df_to_save = df_to_save.reset_index()
        elif 'timestamps' in df_to_save.columns:
            df_to_save = df_to_save.rename(columns={'timestamps': '日期'})
        df_to_save.to_csv(csv_file, encoding='utf-8-sig', index=False)
        print(f"💾 数据已自动缓存落盘: {csv_file}")
        return csv_file
    return None


def fetch_or_load_stock_data(stock_code, data_dir, start_date=None, end_date=None,
                            history_days=365, pred_days=100, start_year=None, end_year=None,
                            force_refresh=False, market=None, allow_fallback=False):
    """
    智能时效调度器（支持多市场：CN/HK/US）：
    - 若指定了 force_refresh=True 或本地无缓存：从网络抓取并落盘；
    - 若本地有缓存：智能检测缓存最新记录的时间时效性；
      若缓存已包含最新收盘行情，则秒级命中缓存；若缓存已过期，则自动联网拉取最新数据覆盖落盘。
    缓存文件名按市场区分: {market}_{stock_code}_stock_data.csv
    """
    detected_market = detect_market_type(stock_code, market)
    ensure_output_directory(data_dir)

    # 缓存文件名按市场前缀区分，避免不同市场同代码冲突
    if detected_market in ('HK', 'US'):
        cache_filename = f"{detected_market}_{stock_code}_stock_data.csv"
    else:
        cache_filename = f"{stock_code}_stock_data.csv"  # A股向后兼容
    csv_file_path = os.path.join(data_dir, cache_filename)

    start_date_str, end_date_str, is_latest_mode = resolve_date_range(
        end_date=end_date, start_date=start_date, history_days=history_days,
        pred_days=pred_days, start_year=start_year, end_year=end_year
    )

    need_fetch = force_refresh or (not os.path.exists(csv_file_path))

    if not need_fetch:
        try:
            existing_df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
            if len(existing_df) < 30:
                print(f"⚠️ 本地数据量过少 ({len(existing_df)} 行)，将重新拉取...")
                need_fetch = True
            elif not is_cache_fresh(existing_df, is_latest_mode, start_date_str, end_date_str):
                last_dt_str = "未知"
                if '日期' in existing_df.columns:
                    last_dt_str = str(existing_df['日期'].max())[:10]
                print(f"🔄 检测到本地缓存数据已过时 (最后记录: {last_dt_str})，自动从网络获取最新数据...")
                need_fetch = True
            else:
                last_dt_str = "最新"
                if '日期' in existing_df.columns:
                    last_dt_str = str(existing_df['日期'].max())[:10]
                print(f"📁 命中最新本地缓存数据: {csv_file_path} (最新日期: {last_dt_str}，共 {len(existing_df)} 行)")
                return existing_df, csv_file_path
        except Exception:
            need_fetch = True

    if need_fetch:
        mode_desc = "最新实时行情" if is_latest_mode else f"{start_date_str} ~ {end_date_str}"
        market_label = {'CN': 'A股', 'HK': '港股', 'US': '美股'}.get(detected_market, '')
        print(f"🌐 自动从数据源拉取{market_label}股票 {stock_code} ({mode_desc}) 行情数据...")
        df = get_stock_data_with_retry(stock_code, start_date=start_date_str, end_date=end_date_str,
                                      market=detected_market, allow_fallback=allow_fallback)
        if df is not None and not df.empty:
            # 使用带市场前缀的文件名保存
            saved_path = save_stock_data_to_csv(df, cache_filename.replace('_stock_data.csv', ''), data_dir)
            return df, saved_path
        else:
            if os.path.exists(csv_file_path):
                cached_df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
                if len(cached_df) >= 30:
                    print(f"⚠️ 网络获取失败，退回使用本地现有有效缓存文件: {csv_file_path}")
                    return cached_df, csv_file_path
            return None, None


# ==================== 数据预处理与参数计算 ====================

def prepare_stock_data(data_input, stock_code):
    """
    准备股票数据，转换为Kronos模型需要的格式

    参数:
    data_input: 可以是 CSV 文件路径 (str)，也可以是已加载的 DataFrame
    stock_code: 股票代码，用于显示信息

    返回:
    df: 处理后的DataFrame
    """
    print(f"正在加载和预处理股票 {stock_code} 数据...")

    if isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        df = pd.read_csv(data_input, encoding='utf-8-sig')

    # 检查数据列名并重命名为标准格式
    column_mapping = {
        '日期': 'timestamps',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '成交额': 'amount'
    }

    # 只重命名存在的列
    actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=actual_mapping)

    # 确保时间戳列存在并转换为datetime格式
    if 'timestamps' not in df.columns:
        if df.index.name == '日期' or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'timestamps'})

    df['timestamps'] = pd.to_datetime(df['timestamps'])

    # 按时间排序（由旧到新升序排列）
    df = df.sort_values('timestamps').reset_index(drop=True)

    # 确保数值列格式正确
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    else:
        df['amount'] = df['close'] * df['volume']

    # 去除空值行
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)

    print(f"✅ 数据加载与预处理完成，共 {len(df)} 条有效记录")
    print(f"时间范围: {df['timestamps'].min().strftime('%Y-%m-%d')} 到 {df['timestamps'].max().strftime('%Y-%m-%d')}")
    print(f"数据列: {df.columns.tolist()}")

    return df


def calculate_prediction_parameters(df, target_days=100):
    """
    根据目标预测天数计算合适的参数

    参数:
    df: 股票数据DataFrame
    target_days: 目标预测天数（自然日）

    返回:
    lookback: 回看期数
    pred_len: 预测期数
    """
    # 计算平均交易日数量（考虑节假日）
    # 【示例】total_days = (2024-12-31 - 2024-01-02).days ≈ 364 天（自然日跨度）
    total_days = (df['timestamps'].max() - df['timestamps'].min()).days
    # 【示例】trading_days = 242（A股实际开市交易的天数）
    trading_days = len(df)
    # 【示例】trading_ratio = 242 / 364 ≈ 0.665（交易日/自然日比例，通常在 0.68 左右）
    trading_ratio = trading_days / total_days if total_days > 0 else 0.7  # 交易日比例

    # 计算目标预测的交易日数量
    # 【示例】target_days=100 时，pred_trading_days = int(100 * 0.665) = 66 天
    pred_trading_days = int(target_days * trading_ratio)

    # 设置回看期数为预测期数的2-3倍，但不超过数据总量的70%
    # 【示例】max_lookback = int(242 * 0.7) = 169
    # 【示例】lookback = min(66 * 2, 169, 242 - 66) = min(132, 169, 176) = 132（回看历史 132 个交易日）
    # 【示例】pred_len = min(66, 242 - 132) = 66（预测未来 66 个交易日）
    max_lookback = int(len(df) * 0.7)
    lookback = min(pred_trading_days * 2, max_lookback, len(df) - pred_trading_days)
    pred_len = min(pred_trading_days, len(df) - lookback)

    print(f"📊 参数计算:")
    print(f"  目标预测天数: {target_days} 天（自然日）")
    print(f"  预计交易日数量: {pred_trading_days} 天")
    print(f"  回看期数 (lookback): {lookback}")
    print(f"  预测期数 (pred_len): {pred_len}")

    return lookback, pred_len


def generate_future_dates_with_holidays(last_date, pred_len, market='CN'):
    """
    生成未来的交易日日期，智能排除周末与对应市场法定休市节假日（涵盖 2024~2027 年）。
    支持 A股(CN)、港股(HK)、美股(US) 三大市场。

    参数:
    last_date: 最后一个历史数据的日期
    pred_len: 预测期数
    market: 市场类型 ('CN'/'HK'/'US')，默认 'CN'

    返回:
    future_dates: 未来的交易日日期列表
    """
    market = market.upper() if market else 'CN'
    market_holidays = MARKET_HOLIDAYS_MAP.get(market, CN_MARKET_HOLIDAYS)
    market_label = {'CN': 'A股', 'HK': '港股', 'US': '美股'}.get(market, market)

    future_dates = []
    current_date = pd.to_datetime(last_date) + timedelta(days=1)
    holidays_encountered = 0

    while len(future_dates) < pred_len:
        date_str = current_date.strftime('%Y-%m-%d')
        # 周一至周五且非休市节假日
        if current_date.weekday() < 5:
            if date_str not in market_holidays:
                future_dates.append(current_date)
            else:
                holidays_encountered += 1
        current_date += timedelta(days=1)

    print(f"📅 生成的{market_label}未来交易日: 共 {len(future_dates)} 天")
    print(f"   起始日期: {future_dates[0].strftime('%Y-%m-%d')}")
    print(f"   结束日期: {future_dates[-1].strftime('%Y-%m-%d')}")
    print(f"   期间排除{market_label}休市节假日: {holidays_encountered} 天")

    return future_dates[:pred_len]


def plot_prediction_with_details(kline_df, pred_df, future_dates, stock_code="002354", stock_name="股票", pred_len=100,
                                 output_dir=".", sample_paths=None, show_plot=False):
    """
    绘制详细的预测结果图表 - 支持蒙特卡洛多路径与扇形置信区间 (Confidence Ribbon)
    """
    # 确保输出目录存在
    ensure_output_directory(output_dir)

    # 确保数据长度一致
    min_len = min(len(pred_df), len(future_dates))
    pred_df = pred_df.iloc[:min_len]
    future_dates = future_dates[:min_len]

    # 设置预测数据的索引为未来日期
    pred_df.index = future_dates

    # 准备价格数据
    # 【示例】sr_close: 历史收盘价序列 (Series，索引为 timestamps, 值为 float)
    # 【示例】sr_pred_close: 预测收盘价序列 (Series，索引为 future_dates, 值为 float)
    sr_close = kline_df.set_index('timestamps')['close']
    sr_pred_close = pred_df['close']
    sr_close.name = '历史数据'
    sr_pred_close.name = "预测数据"

    # 准备成交量数据
    sr_volume = kline_df.set_index('timestamps')['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = '历史数据'
    sr_pred_volume.name = "预测数据"

    # 合并数据
    # 【示例】close_df 包含两列 ['历史数据', '预测数据']，形状为 (lookback + pred_len, 2)
    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    hist_close = close_df['历史数据'].dropna()
    pred_close = close_df['预测数据'].dropna()
    hist_volume = volume_df['历史数据'].dropna()
    pred_volumes = volume_df['预测数据'].dropna()

    # 准备完整的OHLCV数据集（按时间排序），用于鼠标点击时精确定位与数据提取
    # 【示例】hist_records 结构: 历史 K 线的完整 OHLCV 记录
    hist_records = kline_df.copy()
    hist_records['timestamps'] = pd.to_datetime(hist_records['timestamps'])
    hist_records = hist_records.set_index('timestamps')
    hist_records['data_type'] = '历史'

    # 【示例】pred_records 结构: 预测未来的 OHLCV 及置信区间记录
    pred_records = pred_df.copy()
    pred_records.index = pd.to_datetime(pred_records.index)
    pred_records['data_type'] = '预测'

    # 汇总完整OHLCV记录并去重
    # 【示例】combined_ohlcv: 包含历史+预测所有交易日的完整行情总表，形状如 (132 + 66, 10)
    combined_ohlcv = pd.concat([hist_records, pred_records]).sort_index()
    combined_ohlcv = combined_ohlcv[~combined_ohlcv.index.duplicated(keep='last')]
    all_dates = combined_ohlcv.index
    all_x_nums = mdates.date2num(all_dates.to_pydatetime())

    # 创建更大的图表
    fig = plt.figure(figsize=(18, 14))

    # 使用GridSpec创建更灵活的布局
    gs = plt.GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1])

    ax1 = fig.add_subplot(gs[0])  # 价格图表
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # 成交量图表（与价格图共享X轴时间）
    ax3 = fig.add_subplot(gs[2])  # 价格变动图表

    # 1. 价格图表 - 更大更清晰
    # 只显示最近200个交易日的历史数据，避免图表过于拥挤
    recent_history = hist_close.iloc[-min(200, len(hist_close)):]
    line_hist = ax1.plot(recent_history.index, recent_history.values, label='历史价格', color='#1f77b4', linewidth=2.5, alpha=0.9, marker='.', markersize=4)

    # 绘制蒙特卡洛多路径独立采样轨迹（浅橙细虚线）
    if sample_paths is not None and len(sample_paths) > 1:
        for p_idx, sp in enumerate(sample_paths):
            sp_c = sp['close'].iloc[:min_len]
            lbl = f'各采样路径 ({len(sample_paths)}条)' if p_idx == 0 else ""
            ax1.plot(pred_close.index, sp_c.values, color='#ff9f43',
                     linewidth=0.9, linestyle='--', alpha=0.4, label=lbl)

    # 绘制扇形置信区间 (Confidence Ribbon)
    if 'close_upper' in pred_df.columns and 'close_lower' in pred_df.columns:
        valid_l = min(len(pred_close), len(pred_df['close_upper']))
        upper_v = pred_df['close_upper'].iloc[:valid_l].values
        lower_v = pred_df['close_lower'].iloc[:valid_l].values
        ax1.fill_between(pred_close.index[:valid_l], lower_v, upper_v,
                         color='#ff7f0e', alpha=0.25, label='置信区间 (Confidence Ribbon)')

    # 绘制预测均值折线
    line_pred = ax1.plot(pred_close.index, pred_close.values, label='预测走势 (均值)',
                         color='#ff7f0e', linewidth=2.5, linestyle='-', marker='o', markersize=4)

    # 添加预测起始点的标记
    prediction_start_date = pred_close.index[0] if len(pred_close) > 0 else hist_close.index[-1]
    prediction_start_price = hist_close.iloc[-1]
    ax1.axvline(x=prediction_start_date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.annotate('预测起点', xy=(prediction_start_date, prediction_start_price),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.set_ylabel('收盘价 (元)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{stock_name}({stock_code}) 股票价格预测 - 未来{pred_len}个交易日\n[💡 滚轮: 缩放时间轴 | 按住左键拖动: 左右平移 | 单击: 查看OHLCV/成交量/置信区间 | 双击: 重置视野 | 右键: 清除提示]',
                  fontsize=14, fontweight='bold', pad=15)

    # 设置x轴日期格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 设置y轴格式
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    # 2. 成交量图表 - 显示历史与预测成交量
    recent_hist_vol = hist_volume.iloc[-min(200, len(hist_volume)):]
    if len(recent_hist_vol) > 0:
        ax2.bar(recent_hist_vol.index, recent_hist_vol.values,
                alpha=0.6, color='#1f77b4', label='历史成交量', width=0.8)
    if len(pred_volumes) > 0:
        ax2.bar(pred_volumes.index, pred_volumes.values,
                alpha=0.7, color='#ff7f0e', label='预测成交量', width=0.8)

    ax2.set_ylabel('成交量 (股)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-4:.0f}万' if x >= 1e4 else f'{x:.0f}'))

    # 3. 价格变动图表 - 优化显示
    if len(pred_close) > 0 and len(hist_close) > 0:
        price_change = pred_close - hist_close.iloc[-1]
        colors = ['green' if x >= 0 else 'red' for x in price_change]

        # 每5个交易日显示一个标签，避免过于拥挤
        bars = ax3.bar(range(len(price_change)), price_change.values, alpha=0.8, color=colors)

        # 在关键点添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if i % 10 == 0 or i == len(bars) - 1 or abs(height) > price_change.std():  # 每10天或最后一天或显著波动
                ax3.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:+.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=8, fontweight='bold')

        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    ax3.set_ylabel('价格变动 (元)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('交易日', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 设置x轴标签
    if len(pred_close) > 0:
        # 每10个交易日显示一个标签
        xticks_positions = list(range(0, len(pred_close), max(1, len(pred_close) // 10)))
        if len(pred_close) - 1 not in xticks_positions:
            xticks_positions.append(len(pred_close) - 1)
        ax3.set_xticks(xticks_positions)
        ax3.set_xticklabels([f'D{i + 1}' for i in xticks_positions])

    # 添加详细的统计信息框
    if len(pred_close) > 0 and len(hist_close) > 0:
        last_hist_val = hist_close.iloc[-1]
        last_pred_val = pred_close.iloc[-1]
        change_ratio = (last_pred_val / last_hist_val - 1) * 100
        pred_stats = {
            '股票代码': stock_code,
            '股票名称': stock_name,
            '当前价格': f"{last_hist_val:.2f} 元",
            '预测结束价格': f"{last_pred_val:.2f} 元",
            '预测涨跌幅': f"{change_ratio:+.2f}%",
            '预测期间最高价': f"{pred_close.max():.2f} 元",
            '预测期间最低价': f"{pred_close.min():.2f} 元",
            '预测波动率': f"{pred_close.std():.2f} 元",
            '预测起始日期': f"{pred_close.index[0].strftime('%Y-%m-%d')}",
            '预测结束日期': f"{pred_close.index[-1].strftime('%Y-%m-%d')}",
            '预测交易日数': f"{len(pred_close)} 天"
        }
        if 'close_upper' in pred_df.columns and 'close_lower' in pred_df.columns:
            last_idx = len(pred_close) - 1
            pred_stats['期末置信区间'] = f"{pred_df['close_lower'].iloc[last_idx]:.2f} ~ {pred_df['close_upper'].iloc[last_idx]:.2f} 元"

        stats_text = "\n".join([f"{k}: {v}" for k, v in pred_stats.items()])
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                 verticalalignment='bottom')

    # 记录默认初始视野，便于双击一键复位
    default_xlim = ax1.get_xlim()
    default_ylim = ax1.get_ylim()

    # ==================== 交互功能：滚轮缩放、鼠标拖动与点击查看 ====================
    # 1. 价格图浮窗 (OHLCV)
    annot_ax1 = ax1.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round4,pad=0.6", fc="#FFFFF2", ec="#2B4C7E", lw=1.5, alpha=0.96),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="#2B4C7E", lw=1.5),
        fontsize=10,
        zorder=30
    )
    annot_ax1.set_visible(False)
    highlight_ax1 = ax1.scatter([], [], s=140, facecolors='none', edgecolors='#FF0000', linewidths=2.5, zorder=35)

    # 2. 成交量图浮窗 (Volume)
    annot_ax2 = ax2.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round4,pad=0.6", fc="#F0F8FF", ec="#E65100", lw=1.5, alpha=0.96),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="#E65100", lw=1.5),
        fontsize=10,
        zorder=30
    )
    annot_ax2.set_visible(False)
    highlight_ax2 = ax2.scatter([], [], s=120, facecolors='none', edgecolors='#E65100', linewidths=2.5, zorder=35)

    # 3. 联动垂直指示线
    vline_ax1 = ax1.axvline(x=np.nan, color='#888888', linestyle=':', alpha=0.7, linewidth=1.2, visible=False)
    vline_ax2 = ax2.axvline(x=np.nan, color='#888888', linestyle=':', alpha=0.7, linewidth=1.2, visible=False)

    # 拖拽状态记录
    drag_state = {
        'is_pressed': False,
        'is_dragging': False,
        'press_x': None,
        'press_xlim': None,
    }

    def auto_adjust_axes():
        """根据当前可见时间窗口，自动自适应价格轴和成交量轴的Y范围"""
        cur_xlim = ax1.get_xlim()
        mask = (all_x_nums >= cur_xlim[0]) & (all_x_nums <= cur_xlim[1])
        if np.any(mask):
            sub_df = combined_ohlcv.iloc[mask]
            # 价格Y轴自适应
            vis_high = sub_df['high'].max() if 'high' in sub_df.columns else sub_df['close'].max()
            vis_low = sub_df['low'].min() if 'low' in sub_df.columns else sub_df['close'].min()
            if pd.notnull(vis_high) and pd.notnull(vis_low) and vis_high > vis_low:
                p_margin = (vis_high - vis_low) * 0.1
                ax1.set_ylim([vis_low - p_margin, vis_high + p_margin])

            # 成交量Y轴自适应
            vis_vol = sub_df['volume'].max() if 'volume' in sub_df.columns else None
            if pd.notnull(vis_vol) and vis_vol > 0:
                ax2.set_ylim([0, vis_vol * 1.15])

    def on_scroll(event):
        """鼠标滚轮缩放事件：以鼠标所在时间为中心等比放大/缩小"""
        if event.inaxes not in [ax1, ax2]:
            return
        if event.xdata is None:
            return

        cur_xlim = ax1.get_xlim()
        cur_xrange = cur_xlim[1] - cur_xlim[0]

        # 向上滚放大(缩小跨度)，向下滚缩小(放大跨度)
        is_zoom_in = False
        if hasattr(event, 'step') and event.step != 0:
            is_zoom_in = event.step > 0
        elif hasattr(event, 'button'):
            is_zoom_in = (event.button == 'up')

        scale_factor = 0.8 if is_zoom_in else 1.25
        new_range = cur_xrange * scale_factor

        # 限制最大缩放倍数（最少 3 天，最多不超过全部数据的 2 倍）
        total_data_range = max(all_x_nums) - min(all_x_nums)
        if new_range < 3 or new_range > total_data_range * 2.5:
            return

        # 以鼠标所在的 X 坐标为锚点进行等比缩放
        xdata = event.xdata
        ratio = (xdata - cur_xlim[0]) / cur_xrange
        new_xmin = xdata - new_range * ratio
        new_xmax = xdata + new_range * (1.0 - ratio)

        ax1.set_xlim([new_xmin, new_xmax])
        auto_adjust_axes()
        fig.canvas.draw_idle()

    def on_press(event):
        """鼠标按下：记录起始点，准备拖拽或点击"""
        if event.inaxes not in [ax1, ax2]:
            return

        if event.button == 1:  # 左键按下
            drag_state['is_pressed'] = True
            drag_state['is_dragging'] = False
            drag_state['press_x'] = event.x
            drag_state['press_xlim'] = ax1.get_xlim()

        elif event.button == 3:  # 右键点击清除所有浮窗提示
            annot_ax1.set_visible(False)
            annot_ax2.set_visible(False)
            highlight_ax1.set_offsets(np.empty((0, 2)))
            highlight_ax2.set_offsets(np.empty((0, 2)))
            vline_ax1.set_visible(False)
            vline_ax2.set_visible(False)
            fig.canvas.draw_idle()

    def on_motion(event):
        """鼠标移动：按住左键移动时左右拖拽平移图表"""
        if not drag_state['is_pressed'] or drag_state['press_x'] is None:
            return
        if event.x is None:
            return

        dx_pixels = event.x - drag_state['press_x']
        # 移动超过 3 个像素触发拖动模式
        if abs(dx_pixels) > 3:
            drag_state['is_dragging'] = True

        if drag_state['is_dragging']:
            # 将像素差转换为数据坐标差
            inv = ax1.transData.inverted()
            x0_data = inv.transform((drag_state['press_x'], 0))[0]
            x1_data = inv.transform((event.x, 0))[0]
            dx_data = x1_data - x0_data

            orig_xmin, orig_xmax = drag_state['press_xlim']
            new_xmin = orig_xmin - dx_data
            new_xmax = orig_xmax - dx_data

            ax1.set_xlim([new_xmin, new_xmax])
            auto_adjust_axes()
            fig.canvas.draw_idle()

    def handle_click_info(event):
        """处理单击查看 OHLCV 或 Volume 数据"""
        if event.xdata is None or event.inaxes not in [ax1, ax2]:
            return

        dist = np.abs(all_x_nums - event.xdata)
        idx = int(np.argmin(dist))
        nearest_x = all_x_nums[idx]
        row = combined_ohlcv.iloc[idx]
        dt = all_dates[idx]

        # 更新纵向参考虚线
        vline_ax1.set_xdata([nearest_x, nearest_x])
        vline_ax1.set_visible(True)
        vline_ax2.set_xdata([nearest_x, nearest_x])
        vline_ax2.set_visible(True)

        # 点击价格图 ax1 -> 展示 OHLCV 详情
        if event.inaxes == ax1:
            curr_open = row.get('open', np.nan)
            curr_high = row.get('high', np.nan)
            curr_low = row.get('low', np.nan)
            curr_close = row.get('close', np.nan)
            curr_vol = row.get('volume', np.nan)
            curr_amt = row.get('amount', np.nan)
            dtype = row.get('data_type', '数据')

            prev_close = combined_ohlcv.iloc[idx - 1]['close'] if idx > 0 else curr_close
            chg = curr_close - prev_close
            chg_pct = (chg / prev_close) * 100 if prev_close else 0.0
            color_mark = "🔴" if chg >= 0 else "🟢"
            chg_str = f"{color_mark} {chg:+.2f} ({chg_pct:+.2f}%)"

            vol_hands = curr_vol / 100 if pd.notnull(curr_vol) else 0
            if pd.notnull(curr_amt) and curr_amt > 0:
                amt_str = f"{curr_amt / 1e8:.2f} 亿元" if curr_amt >= 1e8 else f"{curr_amt / 1e4:.2f} 万元"
            else:
                amt_str = "无"

            # 蒙特卡洛置信区间展示
            conf_info = ""
            if dtype == '预测' and 'close_upper' in row and 'close_lower' in row:
                c_up = row['close_upper']
                c_low = row['close_lower']
                if pd.notnull(c_up) and pd.notnull(c_low):
                    span = c_up - c_low
                    span_pct = (span / curr_close) * 100 if curr_close else 0.0
                    conf_info = (
                        f"\n──────────────────────\n"
                        f"🔮 蒙特卡洛置信区间:\n"
                        f"乐观上限 (90%):  {c_up:.2f} 元\n"
                        f"悲观下限 (10%):  {c_low:.2f} 元\n"
                        f"区间跨度:        {span:.2f} 元 (±{span_pct/2:.1f}%)"
                    )

            ohlcv_text = (
                f"📅 {dt.strftime('%Y-%m-%d')} 【{dtype}行情】\n"
                f"──────────────────────\n"
                f"开盘价 (Open):   {curr_open:.2f} 元\n"
                f"最高价 (High):   {curr_high:.2f} 元\n"
                f"最低价 (Low):    {curr_low:.2f} 元\n"
                f"收盘价 (Close):  {curr_close:.2f} 元\n"
                f"涨跌幅 (Change): {chg_str}\n"
                f"成交量 (Volume): {curr_vol:,.0f} 股 ({vol_hands:,.0f} 手)\n"
                f"成交额 (Amount): {amt_str}"
                f"{conf_info}"
            )

            annot_ax1.xy = (nearest_x, curr_close)
            annot_ax1.set_text(ohlcv_text)
            annot_ax1.set_visible(True)
            annot_ax2.set_visible(False)
            highlight_ax1.set_offsets([[nearest_x, curr_close]])
            highlight_ax2.set_offsets(np.empty((0, 2)))
            fig.canvas.draw_idle()

        # 点击成交量图 ax2 -> 展示 Volume 详情
        elif event.inaxes == ax2:
            curr_vol = row.get('volume', 0)
            curr_amt = row.get('amount', 0)
            curr_close = row.get('close', np.nan)
            dtype = row.get('data_type', '数据')

            vol_hands = curr_vol / 100 if pd.notnull(curr_vol) else 0
            if curr_vol >= 1e8:
                vol_scale = f"{curr_vol / 1e8:.2f} 亿股"
            elif curr_vol >= 1e4:
                vol_scale = f"{curr_vol / 1e4:.2f} 万股"
            else:
                vol_scale = f"{curr_vol:,.0f} 股"

            if pd.notnull(curr_amt) and curr_amt > 0:
                amt_str = f"{curr_amt / 1e8:.2f} 亿元" if curr_amt >= 1e8 else f"{curr_amt / 1e4:.2f} 万元"
            else:
                amt_str = "无"

            vol_text = (
                f"📊 {dt.strftime('%Y-%m-%d')} 【{dtype}成交量】\n"
                f"──────────────────────\n"
                f"成交量 (股):  {curr_vol:,.0f} 股 ({vol_scale})\n"
                f"成交量 (手):  {vol_hands:,.0f} 手\n"
                f"成交额:       {amt_str}\n"
                f"当日收盘价:   {curr_close:.2f} 元"
            )

            annot_ax2.xy = (nearest_x, curr_vol)
            annot_ax2.set_text(vol_text)
            annot_ax2.set_visible(True)
            annot_ax1.set_visible(False)
            highlight_ax2.set_offsets([[nearest_x, curr_vol]])
            highlight_ax1.set_offsets(np.empty((0, 2)))
            fig.canvas.draw_idle()

    def on_release(event):
        """鼠标松开：若未拖拽则执行点击查看或双击复位"""
        if not drag_state['is_pressed']:
            return

        was_dragging = drag_state['is_dragging']
        drag_state['is_pressed'] = False
        drag_state['is_dragging'] = False
        drag_state['press_x'] = None
        drag_state['press_xlim'] = None

        # 如果刚才处于拖拽平移，松开时不触发点击弹窗
        if was_dragging:
            return

        # 双击左键：一键重置恢复初始视野
        if event.dblclick:
            ax1.set_xlim(default_xlim)
            ax1.set_ylim(default_ylim)
            annot_ax1.set_visible(False)
            annot_ax2.set_visible(False)
            highlight_ax1.set_offsets(np.empty((0, 2)))
            highlight_ax2.set_offsets(np.empty((0, 2)))
            vline_ax1.set_visible(False)
            vline_ax2.set_visible(False)
            auto_adjust_axes()
            fig.canvas.draw_idle()
            return

        # 普通单击：展示浮窗
        handle_click_info(event)

    # 注册事件监听：滚轮缩放、拖拽平移、单击展示、双击复位
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.tight_layout()

    # 保存高分辨率图片到指定目录（保存时浮窗处于隐藏状态，保证图片整洁）
    chart_filename = os.path.join(output_dir, f'{stock_code}_prediction_chart.png')
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 预测图表已保存: {chart_filename}")

    if show_plot:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

    return close_df, volume_df


def generate_prediction_report(close_df, volume_df, pred_df, future_dates, stock_code="002354", stock_name="股票",
                               output_dir="."):
    """
    生成预测报告
    """
    # 确保输出目录存在
    ensure_output_directory(output_dir)

    print(f"\n{'=' * 70}")
    print(f"📊 {stock_name}({stock_code}) 股票预测报告")
    print(f"{'=' * 70}")

    hist_close = close_df['历史数据'].dropna()
    pred_close = close_df['预测数据'].dropna()
    vol_pred = volume_df['预测数据'].dropna()

    if len(pred_close) == 0 or len(hist_close) == 0:
        print("❌ 没有有效的预测数据可生成报告")
        return

    # 确保所有数组长度一致
    min_len = min(len(pred_close), len(vol_pred), len(future_dates))
    valid_pred_close = pred_close.iloc[:min_len]
    valid_vol_pred = vol_pred.iloc[:min_len]
    valid_future_dates = future_dates[:min_len]

    # 基本统计
    historical_close = hist_close.iloc[-1]
    predicted_close = valid_pred_close.iloc[-1]
    price_change_pct = (predicted_close / historical_close - 1) * 100

    print(f"🔮 预测概览:")
    print(f"   当前价格: {historical_close:.2f} 元")
    if 'close_upper' in pred_df.columns and 'close_lower' in pred_df.columns:
        last_up = pred_df['close_upper'].iloc[min_len - 1]
        last_low = pred_df['close_lower'].iloc[min_len - 1]
        print(f"   预测结束价格: {predicted_close:.2f} 元 (置信区间: {last_low:.2f} ~ {last_up:.2f} 元)")
    else:
        print(f"   预测结束价格: {predicted_close:.2f} 元")
    print(f"   预测涨跌幅: {price_change_pct:+.2f}%")
    print(f"   预测期间: {min_len} 个交易日")
    print(
        f"   预测时间范围: {valid_future_dates[0].strftime('%Y-%m-%d')} 到 {valid_future_dates[-1].strftime('%Y-%m-%d')}")

    print(f"\n📈 价格预测统计:")
    print(f"   预测最高价: {valid_pred_close.max():.2f} 元")
    print(f"   预测最低价: {valid_pred_close.min():.2f} 元")
    print(f"   预测平均价: {valid_pred_close.mean():.2f} 元")
    print(f"   价格波动率: {valid_pred_close.std():.2f} 元")

    print(f"\n📊 成交量预测统计:")
    print(f"   预测平均成交量: {valid_vol_pred.mean():,.0f} 手")
    print(f"   预测最大成交量: {valid_vol_pred.max():,.0f} 手")
    print(f"   预测最小成交量: {valid_vol_pred.min():,.0f} 手")

    # 保存详细预测数据到指定目录 - 确保所有数组长度一致
    detail_dict = {
        '日期': valid_future_dates,
        '预测收盘价_均值': valid_pred_close.values,
    }
    if 'close_upper' in pred_df.columns and 'close_lower' in pred_df.columns:
        detail_dict['预测收盘价_上限(90%)'] = pred_df['close_upper'].iloc[:min_len].values
        detail_dict['预测收盘价_下限(10%)'] = pred_df['close_lower'].iloc[:min_len].values
    detail_dict.update({
        '预测成交量': valid_vol_pred.values,
        '价格变动(元)': (valid_pred_close.values - historical_close),
        '价格变动(%)': ((valid_pred_close.values / historical_close - 1) * 100)
    })

    prediction_details = pd.DataFrame(detail_dict)
    prediction_file = os.path.join(output_dir, f'{stock_code}_detailed_predictions.csv')
    prediction_details.to_csv(prediction_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 详细预测数据已保存: {prediction_file}")


def main(stock_code="002354", stock_name="天娱数科", data_dir="./data", pred_days=100,
         end_date=None, start_date=None, history_days=365,
         start_year=None, end_year=None, force_refresh=False,
         output_dir="./output", sample_count=5, show_plot=False):
    """
    主函数：执行股票价格预测

    参数:
    stock_code: 股票代码
    stock_name: 股票名称
    data_dir: 数据文件目录
    pred_days: 预测天数（自然日）
    end_date: 预测基准结束日期 (默认 None: 自动拉取最新真实行情并向后预测；可指定如 '2024-12-31' 进行历史回测)
    start_date: 历史数据起始日期 (默认 None: 自动推导)
    history_days: 历史数据回溯天数 (默认 365 自然日，确保有充沛数据支撑模型上下文)
    start_year: 兼容旧版参数：起始年份
    end_year: 兼容旧版参数：结束年份
    force_refresh: 是否强制从网络刷新最新数据（False: 缓存新鲜则秒级复用，过期自动刷新）
    output_dir: 输出文件目录
    sample_count: 蒙特卡洛采样路径数（>=2 时自动启用扇形置信区间）
    """
    start_date_str, end_date_str, is_latest_mode = resolve_date_range(
        end_date=end_date, start_date=start_date, history_days=history_days,
        pred_days=pred_days, start_year=start_year, end_year=end_year
    )

    mode_text = "最新实时行情模式 (自动对齐最新交易日)" if is_latest_mode else f"历史指定范围模式 ({start_date_str} ~ {end_date_str})"

    print(f"🎯 开始 {stock_name}({stock_code}) 股票价格预测")
    print("=" * 70)
    print(f"股票代码: {stock_code} ({stock_name})")
    print(f"运行模式: {mode_text}")
    print(f"数据跨度: {start_date_str} ~ {end_date_str} (强制刷新: {force_refresh})")
    print(f"数据目录: {data_dir}")
    print(f"预测天数: {pred_days} 天（自然日）")
    print(f"采样路径: {sample_count} 条蒙特卡洛集成")
    print(f"输出目录: {output_dir}")

    # 确保输出目录存在
    ensure_output_directory(output_dir)

    try:
        # 步骤 0: 智能获取或加载数据（优化的多源自动获取与缓存落盘）
        print("\n步骤0: 智能获取/加载股票行情数据...")
        raw_df, csv_file_path = fetch_or_load_stock_data(
            stock_code=stock_code,
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
            history_days=history_days,
            pred_days=pred_days,
            start_year=start_year,
            end_year=end_year,
            force_refresh=force_refresh
        )

        if raw_df is None or len(raw_df) == 0:
            print(f"❌ 无法获取股票 {stock_code} 的有效行情数据，请检查网络或股票代码是否正确")
            return

        # 1. 加载模型和分词器
        print("\n步骤1: 加载Kronos模型和分词器...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        print("✅ 模型加载完成")

        # 2. 实例化预测器
        print("步骤2: 初始化预测器...")
        predictor = KronosPredictor(model, tokenizer, device=None, max_context=512)
        print("✅ 预测器初始化完成")

        # 3. 准备数据
        print("步骤3: 准备股票数据...")
        df = prepare_stock_data(raw_df, stock_code)

        # 历史回测模式下严格截断至 end_date_str 以前，防范未来信息泄露 (Data Leakage)
        if not is_latest_mode and end_date_str:
            df = df[df['timestamps'] <= pd.to_datetime(end_date_str)].reset_index(drop=True)
        if start_date_str:
            df = df[df['timestamps'] >= pd.to_datetime(start_date_str)].reset_index(drop=True)

        # 4. 计算预测参数
        print("步骤4: 计算预测参数...")
        lookback, pred_len = calculate_prediction_parameters(df, target_days=pred_days)

        if pred_len <= 0:
            print("❌ 数据量不足，无法进行预测")
            return

        print(f"✅ 最终参数 - 回看期: {lookback}, 预测期: {pred_len}")

        # 5. 准备输入数据
        print("步骤5: 准备输入数据...")
        # 使用最新的数据作为输入
        # 【示例数据: x_df 为纯 OHLCV 数值矩阵，形状如 (lookback=132, 6)】
        #       open   high    low  close     volume        amount
        # 0    34.20  34.80  33.90  34.50  1500000.0  5.175000e+07
        # ...
        # 131  38.10  39.50  37.80  39.20  2100000.0  8.190000e+07
        x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        # 【示例数据: x_timestamp 为历史对应时间戳序列，形状如 (132,)】
        # 0      2024-06-20
        # ...
        # 131    2024-12-31
        x_timestamp = df.iloc[-lookback:]['timestamps'].reset_index(drop=True)

        # 生成未来日期（考虑节假日）
        # 【示例数据: future_dates 为未来预测交易日列表，长度如 pred_len=66】
        # [Timestamp('2025-01-02'), Timestamp('2025-01-03'), ..., Timestamp('2025-04-10')]
        last_historical_date = df['timestamps'].iloc[-1]
        future_dates = generate_future_dates_with_holidays(last_historical_date, pred_len)

        print(f"输入数据形状: {x_df.shape}")
        print(f"历史数据时间范围: {x_timestamp.iloc[0]} 到 {x_timestamp.iloc[-1]}")
        print(f"预测时间范围: {future_dates[0]} 到 {future_dates[-1]}")

        # 6. 执行预测（支持蒙特卡洛多路径集成与扇形置信区间）
        if sample_count > 1:
            print(f"步骤6: 执行蒙特卡洛多路径集成预测 (采样路径数: {sample_count})...")
            # 【示例数据: sample_paths 是长度为 sample_count(如 5) 的列表】
            # 每个元素是一个独立预测的 DataFrame，形状为 (pred_len=66, 6)，代表一种可能的未来平行走势
            sample_paths = predictor.predict_batch(
                df_list=[x_df] * sample_count,
                x_timestamp_list=[x_timestamp] * sample_count,
                y_timestamp_list=[pd.Series(future_dates)] * sample_count,
                pred_len=pred_len,
                T=1.0,         # 采样温度：控制波动性大小 (T=1.0 为自然波动，T<1 保守，T>1 激进)
                top_p=0.9,     # 核采样截断阈值：过滤掉累计概率超过 90% 的低概率尾部极端异常 Token
                sample_count=1,
                verbose=True
            )

            # 聚合各采样轨迹矩阵
            # 【示例数据: close_matrix 堆叠矩阵，形状为 (sample_count, pred_len)，如 (5, 66)】
            # 每行是一条独立采样路径在未来 66 天的收盘价数值
            # [
            #   [39.5, 40.1, 39.8, ..., 45.2],  <- 路径 1 (中度看涨)
            #   [39.1, 38.6, 38.2, ..., 41.0],  <- 路径 2 (震荡回调)
            #   ...
            #   [39.8, 41.2, 42.5, ..., 48.0],  <- 路径 5 (强力突破)
            # ]
            close_matrix = np.stack([p['close'].values for p in sample_paths], axis=0)
            vol_matrix = np.stack([p['volume'].values for p in sample_paths], axis=0)
            open_matrix = np.stack([p['open'].values for p in sample_paths], axis=0)
            high_matrix = np.stack([p['high'].values for p in sample_paths], axis=0)
            low_matrix = np.stack([p['low'].values for p in sample_paths], axis=0)
            amt_matrix = np.stack([p['amount'].values for p in sample_paths], axis=0)

            # 【示例数据: 置信区间分位数，形状均为 (pred_len=66,)】
            # c_upper: 90% 分位数（未来每一步的乐观走势上限），如 [39.7, 41.0, ..., 47.5]
            # c_lower: 10% 分位数（未来每一步的悲观走势下限），如 [38.9, 38.2, ..., 40.5]
            c_upper = np.percentile(close_matrix, 90, axis=0) if sample_count >= 5 else np.max(close_matrix, axis=0)
            c_lower = np.percentile(close_matrix, 10, axis=0) if sample_count >= 5 else np.min(close_matrix, axis=0)

            # 【示例数据: 最终预测汇总表 pred_df，形状为 (66, 9)】
            # 包含开高低收均值、成交量/额均值、置信上下限及标准差，索引为 future_dates
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
            print(f"✅ 完成 {sample_count} 条蒙特卡洛轨迹集成与扇形置信区间推导")
        else:
            print("步骤6: 执行单路径价格预测...")
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=pd.Series(future_dates),
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=True
            )
            sample_paths = None
            print("✅ 预测完成")

        # 7. 显示预测结果
        print("\n步骤7: 显示预测结果...")
        print("预测数据前5行:")
        # 确保预测数据长度与未来日期一致
        min_len = min(len(pred_df), len(future_dates))
        pred_df = pred_df.iloc[:min_len]
        pred_df.index = future_dates[:min_len]
        print(pred_df[['open', 'high', 'low', 'close', 'volume']].head())

        # 8. 可视化结果（绘制扇形置信区间与独立采样轨迹）
        print("步骤8: 生成可视化图表...")
        # 使用最后一部分历史数据和预测数据
        kline_df = df.iloc[-lookback:].reset_index(drop=True)
        close_df, volume_df = plot_prediction_with_details(kline_df, pred_df, future_dates, stock_code, stock_name,
                                                           pred_len, output_dir, sample_paths=sample_paths,
                                                           show_plot=show_plot)

        # 9. 生成预测报告
        print("步骤9: 生成预测报告...")
        generate_prediction_report(close_df, volume_df, pred_df, future_dates, stock_code, stock_name, output_dir)

        print(f"\n🎉 {stock_name}({stock_code}) 股票预测完成!")
        print("生成的文件:")
        print(f"  📊 {os.path.join(output_dir, stock_code + '_prediction_chart.png')} - 预测图表")
        print(f"  📋 {os.path.join(output_dir, stock_code + '_detailed_predictions.csv')} - 详细预测数据")

        # 显示预测总结
        hist_close_valid = close_df['历史数据'].dropna()
        pred_close_valid = close_df['预测数据'].dropna()
        if len(pred_close_valid) > 0 and len(hist_close_valid) > 0:
            print(f"\n📈 预测总结:")
            historical_price = hist_close_valid.iloc[-1]
            predicted_price = pred_close_valid.iloc[-1]
            change_pct = (predicted_price / historical_price - 1) * 100

            print(f"  当前价格: {historical_price:.2f} 元")
            print(f"  预测价格(均值): {predicted_price:.2f} 元")
            if 'close_upper' in pred_df.columns and 'close_lower' in pred_df.columns:
                print(f"  置信区间: {pred_df['close_lower'].iloc[min_len-1]:.2f} ~ {pred_df['close_upper'].iloc[min_len-1]:.2f} 元")
            print(f"  预期涨跌: {change_pct:+.2f}%")
            print(
                f"  预测时间: {future_dates[0].strftime('%Y-%m-%d')} 到 {future_dates[min_len - 1].strftime('%Y-%m-%d')}")

            if change_pct > 10:
                print(f"  🚀 模型预测未来{pred_len}个交易日大幅看涨 (+{change_pct:.1f}%)")
            elif change_pct > 5:
                print(f"  📈 模型预测未来{pred_len}个交易日看涨 (+{change_pct:.1f}%)")
            elif change_pct > 0:
                print(f"  ↗️ 模型预测未来{pred_len}个交易日微涨 (+{change_pct:.1f}%)")
            elif change_pct > -5:
                print(f"  ↘️ 模型预测未来{pred_len}个交易日微跌 ({change_pct:.1f}%)")
            elif change_pct > -10:
                print(f"  📉 模型预测未来{pred_len}个交易日看跌 ({change_pct:.1f}%)")
            else:
                print(f"  🔻 模型预测未来{pred_len}个交易日大幅看跌 ({change_pct:.1f}%)")

    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# 使用方法说明
if __name__ == "__main__":
    """
    股票预测工具 - 支持多股票预测

    使用方法：
    修改下面的 STOCK_CONFIG 来预测不同的股票
    """

    # ==================== 在这里修改股票配置 ====================
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    STOCK_CONFIG = {
        "stock_code": "300418",  # 股票代码
        "stock_name": "昆仑万维",  # 股票名称
        "data_dir": os.path.join(CURRENT_DIR, "data"),  # 数据文件目录（自动保存与缓存）
        "pred_days": 100,  # 预测未来的自然日天数
        "sample_count": 5,  # 蒙特卡洛采样路径数 (>=2 自动开启扇形置信区间)
        "output_dir": os.path.join(CURRENT_DIR, "yuce"),  # 输出文件目录
        # --- 时间范围控制 (默认自动预测最新未来，如需回测历史可指定 end_date) ---
        "end_date": None,  # 默认 None: 自动拉取最新真实行情向后预测；可指定如 "2024-12-31" 进行历史回测
        "history_days": 365,  # 历史数据回溯天数 (默认 365 天，保证有充分上下文支撑模型)
        "force_refresh": False  # 是否强制刷新 (False: 自动智能检测时效，缓存新鲜则秒级复用，过期自动刷新)
    }

    # 其他股票配置示例：
    # STOCK_CONFIG = {"stock_code": "000001", "stock_name": "平安银行", "data_dir": os.path.join(CURRENT_DIR, "data"), "pred_days": 100, "sample_count": 5, "output_dir": os.path.join(CURRENT_DIR, "yuce")}
    # STOCK_CONFIG = {"stock_code": "600036", "stock_name": "招商银行", "data_dir": os.path.join(CURRENT_DIR, "data"), "pred_days": 100, "sample_count": 5, "output_dir": os.path.join(CURRENT_DIR, "yuce")}
    # STOCK_CONFIG = {"stock_code": "300750", "stock_name": "宁德时代", "data_dir": os.path.join(CURRENT_DIR, "data"), "pred_days": 100, "sample_count": 5, "output_dir": os.path.join(CURRENT_DIR, "yuce")}
    # =========================================================

    print("🤖 智能股票预测工具 (支持全自动最新行情与时效感知缓存)")
    print("=" * 70)
    print(f"当前预测股票: {STOCK_CONFIG['stock_name']}({STOCK_CONFIG['stock_code']})")
    time_target = f"历史指定基准: {STOCK_CONFIG.get('end_date')}" if STOCK_CONFIG.get('end_date') else "默认自动对齐最新真实收盘行情"
    print(f"时间模式: {time_target}")
    print(f"数据目录: {STOCK_CONFIG['data_dir']}")
    print(f"预测天数: {STOCK_CONFIG['pred_days']} 天（自然日）")
    print(f"蒙特卡洛采样: {STOCK_CONFIG['sample_count']} 条路径 (扇形置信区间)")
    print(f"输出目录: {STOCK_CONFIG['output_dir']}")
    print()

    # 运行主程序
    main(**STOCK_CONFIG)

    print(f"\n💡 提示：要预测其他股票或指定历史回测基准，请修改代码中的 STOCK_CONFIG 变量")