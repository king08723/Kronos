"""
backtest/core/timeframes.py
多时间粒度 (Multi-Timeframe) 与交易所交易日历时段工具
支持日线 (1d) 以及 15分钟 (15m)、30分钟 (30m)、60分钟等日内高频 K 线的无缝处理与年化校准。
"""

from __future__ import annotations

from typing import Union, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

from .types import TimeFrame


# 交易所常规交易时段配置 (本地时间)
MARKET_TRADING_SESSIONS = {
    'CN': [('09:30', '11:30'), ('13:00', '15:00')],                     # 4.0小时 (240分钟)
    'HK': [('09:30', '12:00'), ('13:00', '16:00')],                     # 5.5小时 (330分钟)
    'US': [('09:30', '16:00')],                                         # 6.5小时 (390分钟, 美东时间)
}


def is_same_trading_day(t1: Any, t2: Any) -> bool:
    """检查两个时间戳是否属于同一交易自然日"""
    try:
        d1 = t1.date() if hasattr(t1, 'date') else pd.to_datetime(t1).date()
        d2 = t2.date() if hasattr(t2, 'date') else pd.to_datetime(t2).date()
        return d1 == d2
    except Exception:
        return False


def get_trading_days_passed(entry_time: Any, current_time: Any) -> int:
    """计算从建仓到当前时间戳跨越的自然日天数"""
    try:
        d1 = entry_time.date() if hasattr(entry_time, 'date') else pd.to_datetime(entry_time).date()
        d2 = current_time.date() if hasattr(current_time, 'date') else pd.to_datetime(current_time).date()
        return max(0, (d2 - d1).days)
    except Exception:
        return 0


def get_annualization_factor(
    timeframe: Union[str, TimeFrame] = TimeFrame.D1,
    market: str = 'CN',
    annual_trading_days: Optional[int] = None,
) -> float:
    """
    计算特定市场与时间粒度下的年化折算因子。
    公式: sqrt(annual_trading_days * bars_per_day)

    示例:
    - CN 日线: sqrt(244 * 1) ≈ 15.62
    - HK 日线: sqrt(246 * 1) ≈ 15.68
    - US 日线: sqrt(252 * 1) ≈ 15.87
    - HK 15分钟: 1日22根Bar, sqrt(246 * 22) = sqrt(5412) ≈ 73.57
    - US 15分钟: 1日26根Bar, sqrt(252 * 26) = sqrt(6552) ≈ 80.94
    - US 30分钟: 1日13根Bar, sqrt(252 * 13) = sqrt(3276) ≈ 57.24
    """
    tf = TimeFrame.from_str(timeframe)
    return tf.annual_factor(market=market, annual_trading_days=annual_trading_days)


def detect_timeframe(df: pd.DataFrame) -> TimeFrame:
    """
    通过 DataFrame 索引或时间列自动推断 K 线时间粒度。
    若时间差中位数为 1 天或以上返回 D1，若在 10~20 分钟返回 M15，20~40 分钟返回 M30 等。
    """
    if df.empty:
        return TimeFrame.D1

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        time_col = None
        for col in ('date', 'datetime', 'timestamp', 'time'):
            if col in df.columns:
                time_col = col
                break
        if time_col:
            idx = pd.to_datetime(df[time_col])
        else:
            return TimeFrame.D1

    if len(idx) < 2:
        return TimeFrame.D1

    # 过滤重复索引计算时间差中位数
    unique_idx = pd.Series(idx).drop_duplicates().sort_values()
    if len(unique_idx) < 2:
        return TimeFrame.D1

    diffs = unique_idx.diff().dropna()
    median_sec = diffs.median().total_seconds()

    if median_sec >= 86400 * 0.8:
        return TimeFrame.D1
    elif 3000 <= median_sec <= 4200:
        return TimeFrame.H1
    elif 1500 <= median_sec < 3000:
        return TimeFrame.M30
    elif 600 <= median_sec < 1500:
        return TimeFrame.M15
    elif 180 <= median_sec < 600:
        return TimeFrame.M5
    elif median_sec < 180:
        return TimeFrame.M1
    return TimeFrame.D1


def resample_bars(
    df: pd.DataFrame,
    target_timeframe: Union[str, TimeFrame],
    market: str = 'CN',
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: str = 'volume',
    amount_col: Optional[str] = 'amount',
) -> pd.DataFrame:
    """
    将低级别 K 线 (如 1m/5m/15m) 合成为高级别 K 线 (如 30m/60m/1d)。
    使用交易所交易时段感知的 SessionResampler，严格保持 OHLCV 物理正确性并消除午休与跨日断点。
    """
    from .resampler import SessionResampler
    return SessionResampler.resample_to_timeframe(
        df=df,
        target_timeframe=target_timeframe,
        market=market,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        volume_col=volume_col,
        amount_col=amount_col,
    )


# ---------------------------------------------------------------------------
# 不同 K 线时间周期的量化最佳实践基准参数矩阵 (Adaptive Best Practices)
# ---------------------------------------------------------------------------

TIMEFRAME_BEST_PRACTICES: Dict[str, Dict[str, Any]] = {
    '1d': {
        'max_hold': 15,                     # 15 根 Bar (15 个交易日，波段中线 2~3周)
        'hard_stop': 0.040,                 # 4.0%
        'trail_stop': 0.025,                # 2.5%
        'trailing_stop_min_profit': 0.020,  # 2.0%
        'z_threshold': 0.55,                # 经典信噪比门槛
        'entry_percentile': 0.80,           # 前 20%
        'calib_window': 60,                 # 60 根 Bar (~1季度)
        'calib_window_min': 20,             # 20 根 Bar
        'volume_participation_cap': None,   # 日线无需微观单Bar截断
        'slippage_model': 'fixed',          # 固定基准滑点
        'intraday_exit_at_eod': False,
        'overnight_gap_protection': True,
        'use_mtf_trend_filter': False,      # 自身即日线
        'test_bars': 120,                   # 回测验证窗口 (120 根 Bar / 约半年)
        'test_days': 120,                   # 兼容别名
        'lookback_bars': 90,                # 预测输入窗口 (90 根 Bar / 对齐官方先验)
        'lookback': 90,                     # 兼容别名
        'history_days': 700,                # 历史拉取天数 (自然日)
        'max_auto_horizon': 10,             # Auto 模式向前推演 10 根 Bar (交易日)
        'step_size': 1,                     # 滚动步长 (逐日 1 根 Bar)
        'min_overlap_bars': 15,             # 最小重叠 K 线根数
        'description': '日线标准波段趋势跟踪：宽止损、低门槛、无流动性截断、顺大势持仓数周',
    },
    '60m': {
        'max_hold': 20,                     # 20 根 Bar (约 3~5 个交易日)
        'hard_stop': 0.025,                 # 2.5%
        'trail_stop': 0.018,                # 1.8%
        'trailing_stop_min_profit': 0.015,  # 1.5%
        'z_threshold': 0.70,                # 提升信噪比过滤高频杂波
        'entry_percentile': 0.82,
        'calib_window': 60,                 # 60 根 Bar
        'calib_window_min': 20,
        'volume_participation_cap': 0.15,   # 单 Bar 15% 封顶
        'slippage_model': 'fixed',
        'intraday_exit_at_eod': False,      # 允许跨日
        'overnight_gap_protection': True,
        'use_mtf_trend_filter': True,       # 严格启用日线双均线共振
        'test_bars': 150,                   # 回测验证窗口 (150 根 60m Bar，约 23~37 个交易日)
        'test_days': 150,                   # 兼容别名
        'lookback_bars': 120,               # 预测输入窗口 (120 根 60m Bar，充分激活 Transformer 上下文)
        'lookback': 120,                    # 兼容别名
        'history_days': 120,                # 历史数据自然天数
        'max_auto_horizon': 8,              # 向前推演 8 根 60m Bar (约 1~2 交易日)
        'step_size': 1,
        'min_overlap_bars': 20,
        'description': '60分钟次级别波段：中等止损、提高信噪比、严格日线趋势共振',
    },
    '30m': {
        'max_hold': 24,                     # 24 根 Bar (约 2~3 个交易日)
        'hard_stop': 0.022,                 # 2.2%
        'trail_stop': 0.016,                # 1.6%
        'trailing_stop_min_profit': 0.012,  # 1.2%
        'z_threshold': 0.75,                # 适度收紧信号门槛
        'entry_percentile': 0.82,
        'calib_window': 80,                 # 80 根 30m Bar
        'calib_window_min': 24,
        'volume_participation_cap': 0.10,   # 单 Bar 10% 封顶
        'slippage_model': 'dynamic',        # 动态平方根冲击滑点
        'intraday_exit_at_eod': False,      # 允许跨日短波段
        'overnight_gap_protection': True,
        'use_mtf_trend_filter': True,       # 严格启用日线双均线共振
        'test_bars': 200,                   # 回测验证窗口 (200 根 30m Bar，约 15~25 个交易日，样本丰富)
        'test_days': 200,                   # 兼容别名
        'lookback_bars': 160,               # 预测输入窗口 (160 根 30m Bar，约 12~20 个交易日，极佳时序上下文)
        'lookback': 160,                    # 兼容别名
        'history_days': 90,                 # 历史拉取自然日天数
        'max_auto_horizon': 8,              # 向前推演 8 根 30m Bar (4小时动态波峰)
        'step_size': 1,
        'min_overlap_bars': 30,
        'description': '30分钟平衡摆动：动态冲击滑点、流动性10%封顶、日线双均线共振',
    },
    '15m': {
        'max_hold': 32,                     # 32 根 Bar (约 1.5~2 个交易日)
        'hard_stop': 0.015,                 # 1.5%
        'trail_stop': 0.010,                # 1.0%
        'trailing_stop_min_profit': 0.008,  # 0.8%
        'z_threshold': 0.85,                # 显著提高信噪比
        'entry_percentile': 0.85,           # 前 15%
        'calib_window': 100,
        'calib_window_min': 30,
        'volume_participation_cap': 0.08,   # 单 Bar 8% 封顶
        'slippage_model': 'dynamic',
        'intraday_exit_at_eod': False,      # 港美股自动适配为 True，A股保持 False
        'overnight_gap_protection': True,
        'use_mtf_trend_filter': True,
        'test_bars': 240,                   # 回测验证窗口 (240 根 15m Bar，约 9~15 个交易日)
        'test_days': 240,                   # 兼容别名
        'lookback_bars': 200,               # 预测输入窗口 (200 根 15m Bar)
        'lookback': 200,                    # 兼容别名
        'history_days': 60,                 # 历史拉取天数
        'max_auto_horizon': 12,             # 向前推演 12 根 15m Bar (3小时)
        'step_size': 1,
        'min_overlap_bars': 40,
        'description': '15分钟短线/日内主频：紧止损、高信噪比门槛、强流动性约束',
    },
    '5m': {
        'max_hold': 36,                     # 36 根 Bar (约 3 小时日内)
        'hard_stop': 0.010,                 # 1.0%
        'trail_stop': 0.007,                # 0.7%
        'trailing_stop_min_profit': 0.005,  # 0.5%
        'z_threshold': 0.95,
        'entry_percentile': 0.88,
        'calib_window': 120,
        'calib_window_min': 36,
        'volume_participation_cap': 0.05,   # 单 Bar 5% 封顶
        'slippage_model': 'dynamic',
        'intraday_exit_at_eod': True,       # 默认日内平仓
        'overnight_gap_protection': True,
        'use_mtf_trend_filter': True,
        'test_bars': 300,                   # 回测验证窗口 (300 根 5m Bar，约 4~6 个交易日)
        'test_days': 300,                   # 兼容别名
        'lookback_bars': 250,               # 预测输入窗口 (250 根 5m Bar)
        'lookback': 250,                    # 兼容别名
        'history_days': 30,                 # 历史拉取天数
        'max_auto_horizon': 16,             # 向前推演 16 根 5m Bar (80分钟)
        'step_size': 1,
        'min_overlap_bars': 50,
        'description': '5分钟高频日内：严格日内强平(T+0)、5%流动性封顶、杜绝隔夜风险',
    },
    '1m': {
        'max_hold': 45,                     # 45 根 Bar (45分钟日内冲销)
        'hard_stop': 0.008,                 # 0.8%
        'trail_stop': 0.005,                # 0.5%
        'trailing_stop_min_profit': 0.004,  # 0.4%
        'z_threshold': 1.05,                # 极高信噪比杜绝过度交易
        'entry_percentile': 0.90,           # 仅抓前 10% 极端强信号
        'calib_window': 150,
        'calib_window_min': 45,
        'volume_participation_cap': 0.03,   # 单 Bar 3% 封顶
        'slippage_model': 'dynamic',
        'intraday_exit_at_eod': True,       # 严格日内平仓
        'overnight_gap_protection': True,
        'use_mtf_trend_filter': True,
        'test_bars': 390,                   # 回测验证窗口 (390 根 1m Bar，美股 1 全天 / A股 1.6 天)
        'test_days': 390,                   # 兼容别名
        'lookback_bars': 300,               # 预测输入窗口 (300 根 1m Bar)
        'lookback': 300,                    # 兼容别名
        'history_days': 15,                 # 历史拉取天数
        'max_auto_horizon': 20,             # 向前推演 20 根 1m Bar (20分钟)
        'step_size': 1,
        'min_overlap_bars': 60,
        'description': '1分钟超高频微观结构：极紧止损、超高信噪比、严格平方根滑点与日内强平',
    },
}


def get_timeframe_defaults(
    timeframe: Union[str, TimeFrame] = '1d',
    market: str = 'CN',
) -> Dict[str, Any]:
    """
    获取指定 K 线时间周期与市场规则下的自适应量化最佳实践默认参数。

    合规保障：
    1. A 股 (CN) 严格遵守 T+1 交易制度，无论周期多小，严禁强制 intraday_exit_at_eod=True（当天无法卖出）。
    2. 港美股 (HK/US) 支持 T+0 日内回转，15m / 5m / 1m 默认启用日内收盘强平，杜绝隔夜跳空尾部黑天鹅。
    3. 同步规范化 hard_stop / hard_stop_loss、trail_stop / trailing_stop、max_hold / max_holding_days、
       z_threshold / z_score_threshold 等同义键名。
    """
    tf = TimeFrame.from_str(timeframe)
    tf_key = tf.value
    base_cfg = TIMEFRAME_BEST_PRACTICES.get(tf_key, TIMEFRAME_BEST_PRACTICES['1d']).copy()

    # 市场制度合规微调
    market_str = (market or 'CN').upper()
    if market_str == 'CN':
        # A 股必须严格遵循 T+1，日内绝不可强平
        base_cfg['intraday_exit_at_eod'] = False
    elif market_str in ('HK', 'US'):
        # 港美股支持 T+0：超短线 (15m, 5m, 1m) 推荐开启日内强平，规避隔夜无涨跌停黑天鹅
        if tf in (TimeFrame.M1, TimeFrame.M5, TimeFrame.M15):
            base_cfg['intraday_exit_at_eod'] = True

    # 注入标准别名键名，确保所有引擎和策略层平滑读取
    base_cfg['timeframe'] = tf_key
    base_cfg['timeframe_enum'] = tf
    base_cfg['test_bars'] = base_cfg.get('test_bars', base_cfg.get('test_days', 120))
    base_cfg['test_days'] = base_cfg['test_bars']
    base_cfg['lookback_bars'] = base_cfg.get('lookback_bars', base_cfg.get('lookback', 90))
    base_cfg['lookback'] = base_cfg['lookback_bars']
    base_cfg['min_overlap_bars'] = base_cfg.get('min_overlap_bars', base_cfg.get('min_overlap_days', 15))
    base_cfg['min_overlap_days'] = base_cfg['min_overlap_bars']
    base_cfg['hard_stop_loss'] = base_cfg['hard_stop']
    base_cfg['trailing_stop'] = base_cfg['trail_stop']
    base_cfg['max_holding_days'] = base_cfg['max_hold']
    base_cfg['z_score_threshold'] = base_cfg['z_threshold']
    base_cfg['intraday_exit_eod'] = base_cfg['intraday_exit_at_eod']
    base_cfg['no_gap_protection'] = not base_cfg['overnight_gap_protection']
    base_cfg['no_mtf_filter'] = not base_cfg['use_mtf_trend_filter']

    return base_cfg


def resolve_timeframe_params(
    timeframe: Union[str, TimeFrame] = '1d',
    market: str = 'CN',
    overrides: Optional[Dict[str, Any]] = None,
    auto_adapt: bool = True,
) -> Dict[str, Any]:
    """
    自适应参数解析器：
    以最佳实践参数为基准底座，若 auto_adapt=True 且用户未显式传入某参数（即值为 None 时），
    自动填充对应周期的最佳参数；若用户显式指定了某参数值，则以用户输入为最高优先级覆盖。
    """
    if auto_adapt:
        params = get_timeframe_defaults(timeframe=timeframe, market=market)
    else:
        params = get_timeframe_defaults(timeframe='1d', market=market)

    if not overrides:
        return params

    alias_pairs = [
        ('test_bars', 'test_days'),
        ('lookback_bars', 'lookback'),
        ('min_overlap_bars', 'min_overlap_days'),
        ('hard_stop', 'hard_stop_loss'),
        ('trail_stop', 'trailing_stop'),
        ('max_hold', 'max_holding_days'),
        ('z_threshold', 'z_score_threshold'),
        ('volume_cap', 'volume_participation_cap'),
        ('intraday_exit_eod', 'intraday_exit_at_eod'),
        ('no_gap_protection', 'overnight_gap_protection'),
        ('no_mtf_filter', 'use_mtf_trend_filter'),
    ]

    for k, v in overrides.items():
        if v is not None:
            params[k] = v
            for a1, a2 in alias_pairs:
                if k == a1:
                    params[a2] = (not v) if 'no_' in a1 or 'no_' in a2 else v
                elif k == a2:
                    params[a1] = (not v) if 'no_' in a1 or 'no_' in a2 else v

    return params

