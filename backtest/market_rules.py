"""
market_rules.py
多市场微观交易规则：代码解析、申报单位、涨跌停、税费、年化交易日。

覆盖 A 股 (CN) / 港股 (HK) / 美股 (US)。不做空、不建模盘前盘后与碎股。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 港股印花税下调生效日（双边由 0.13% 降至 0.10%）
HK_STAMP_DUTY_CUTOVER = date(2023, 11, 17)

# 常见港股一手股数（未收录代码默认 100，并打警告）
HK_COMMON_LOT_SIZES = {
    '00700': 100,   # 腾讯控股
    '09988': 100,   # 阿里巴巴-W
    '03690': 100,   # 美团-W
    '01810': 200,   # 小米集团-W
    '01211': 500,   # 比亚迪股份
    '00005': 400,   # 汇丰控股
    '02318': 500,   # 中国平安
    '00388': 100,   # 香港交易所
    '00981': 500,   # 中芯国际
    '09618': 50,    # 京东集团-SW
    '09999': 100,   # 网易-S
    '02269': 500,   # 药明生物
    '01024': 100,   # 快手-W
    '02015': 100,   # 理想汽车-W
    '09868': 100,   # 小鹏汽车-W
    '09866': 10,    # 蔚来-SW
    '00941': 500,   # 中国移动
    '00386': 2000,  # 中国石油化工股份
    '01398': 1000,  # 工商银行
    '00939': 1000,  # 建设银行
    '03988': 500,   # 中国银行
    '02388': 500,   # 中银香港
    '02628': 2000,  # 中国人寿
    '00688': 2000,  # 中国海外发展
    '01109': 500,   # 华润置地
    '01928': 500,   # 金沙中国有限公司
    '02020': 500,   # 安踏体育
    '02331': 500,   # 李宁
    '06618': 50,    # 京东健康
    '09626': 20,    # 哔哩哔哩-W
    '09888': 20,    # 百度集团-SW
}


def _to_date(trade_date) -> Optional[date]:
    """将任意日期输入规范为 date；无法解析则返回 None。"""
    if trade_date is None or (isinstance(trade_date, float) and np.isnan(trade_date)):
        return None
    if isinstance(trade_date, date) and not isinstance(trade_date, datetime):
        return trade_date
    try:
        return pd.Timestamp(trade_date).date()
    except Exception:
        return None


def _looks_like_us_ticker(code: str) -> bool:
    """美股：纯字母，或字母加一个 '.' / '-'（如 BRK.B、BRK-B）。"""
    if not code:
        return False
    upper = code.upper()
    # 纯字母
    if upper.isalpha() and 1 <= len(upper) <= 6:
        return True
    # 字母.字母 或 字母-字母
    if re.fullmatch(r'[A-Z]{1,5}[.\-][A-Z]{1,2}', upper):
        return True
    return False


def _is_st_name(stock_name: Optional[str]) -> bool:
    """名称含 ST / *ST / S*ST 等视为特别处理股。"""
    if not stock_name:
        return False
    name = str(stock_name).upper().replace(' ', '')
    return 'ST' in name


class SymbolResolver:
    """股票代码、市场与缓存文件双向标准化解析器"""

    @staticmethod
    def resolve(stock_code: str, market: str = None) -> tuple:
        """
        标准化股票代码与市场类型。
        返回: (clean_code: str, market: str, file_prefix: str)
        """
        raw = str(stock_code).strip()
        upper = raw.upper()

        det_market = str(market).strip().upper() if market else None
        clean = raw

        # 剥离已知前缀
        if upper.startswith('HK_'):
            det_market = det_market or 'HK'
            clean = raw[3:].strip()
        elif upper.startswith('US_'):
            det_market = det_market or 'US'
            clean = raw[3:].strip()
        elif upper.startswith(('SH_', 'SZ_', 'CN_')):
            det_market = det_market or 'CN'
            clean = raw[3:].strip()

        # 剥离已知后缀
        upper_clean = clean.upper()
        if upper_clean.endswith('.HK'):
            det_market = det_market or 'HK'
            clean = clean[:-3].strip()
        elif upper_clean.endswith(('.SS', '.SH', '.SZ', '.BJ')):
            det_market = det_market or 'CN'
            clean = clean.split('.')[0].strip()

        # 自动推断市场：美股优先识别带点/连字符 ticker
        if not det_market:
            if _looks_like_us_ticker(clean):
                det_market = 'US'
            elif clean.isdigit() and len(clean) <= 5:
                det_market = 'HK'
            else:
                det_market = 'CN'

        det_market = det_market.upper()
        if det_market == 'HK':
            clean = clean.zfill(5) if clean.isdigit() else clean.upper()
            prefix = f"HK_{clean}"
        elif det_market == 'US':
            # 统一为点号风格（BRK-B -> BRK.B），便于 yfinance
            clean = clean.upper().replace('-', '.')
            prefix = f"US_{clean.replace('.', '_')}"
        else:
            clean = clean.zfill(6) if clean.isdigit() else clean.upper()
            prefix = clean

        return clean, det_market, prefix


class MarketProfile:
    """
    多市场微观交易规则画像。

    刻画 A股 / 港股 / 美股 在申报单位、涨跌停、摩擦税费与无风险利率上的制度差异。
    """

    def __init__(
        self,
        market: str = 'CN',
        lot_size: int = 100,
        stamp_duty_sell: float = 0.0005,
        stamp_duty_buy: float = 0.0,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        settlement_days: int = 1,
        price_limit_pct: Optional[float] = 0.10,
        risk_free_rate: float = 0.025,
        currency: str = 'CNY',
        currency_symbol: str = '¥',
        extra_fee_rate: float = 0.0,
        default_capital: float = 100000.0,
        annual_trading_days: int = 244,
        default_slippage: float = 0.001,
        stock_code: str = '',
        board: str = 'main',
        transfer_fee_rate: float = 0.0,
        sell_only_extra_fee_rate: float = 0.0,
        share_rounding: str = 'lot_multiple',
        unknown_hk_lot: bool = False,
    ):
        self.market = market
        self.lot_size = lot_size
        self.stamp_duty_sell = stamp_duty_sell
        self.stamp_duty_buy = stamp_duty_buy
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.settlement_days = settlement_days
        self.price_limit_pct = price_limit_pct
        self.risk_free_rate = risk_free_rate
        self.currency = currency
        self.currency_symbol = currency_symbol
        self.extra_fee_rate = extra_fee_rate
        self.default_capital = default_capital
        self.annual_trading_days = annual_trading_days
        self.default_slippage = default_slippage
        self.stock_code = stock_code
        self.board = board  # main / chi_next / star / bse / hk / us
        # 沪市过户费等双边费率（深市为 0）
        self.transfer_fee_rate = transfer_fee_rate
        # 美股 SEC/TAF 等仅卖方规费
        self.sell_only_extra_fee_rate = sell_only_extra_fee_rate
        # lot_multiple | min_then_one | one_share
        self.share_rounding = share_rounding
        self.unknown_hk_lot = unknown_hk_lot

    @classmethod
    def for_stock(
        cls,
        stock_code: str,
        market: str = None,
        stock_name: str = None,
    ) -> 'MarketProfile':
        clean_code, det_market, _ = SymbolResolver.resolve(stock_code, market=market)
        return cls.for_market(det_market, stock_code=clean_code, stock_name=stock_name)

    @classmethod
    def for_market(
        cls,
        market: str,
        stock_code: str = '',
        stock_name: str = None,
    ) -> 'MarketProfile':
        market = (market or 'CN').upper()
        # 去掉可能残留的市场前缀，保留美股小数点代码
        raw = str(stock_code or '')
        if '_' in raw and not raw.replace('.', '').replace('_', '').isalnum():
            clean_code = raw.split('_')[-1].upper()
        elif raw.upper().startswith(('HK_', 'US_', 'CN_', 'SH_', 'SZ_')):
            clean_code = raw.split('_', 1)[-1].upper()
        else:
            clean_code = raw.upper()

        if market == 'US':
            return cls(
                market='US',
                lot_size=1,
                stamp_duty_sell=0.0,
                stamp_duty_buy=0.0,
                # 默认零佣金；可通过构造参数覆盖为万五/最低 $1
                commission_rate=0.0,
                min_commission=0.0,
                settlement_days=0,
                price_limit_pct=None,
                risk_free_rate=0.045,
                currency='USD',
                currency_symbol='$',
                extra_fee_rate=0.0,
                sell_only_extra_fee_rate=0.000028,
                default_capital=50000.0,
                annual_trading_days=252,
                default_slippage=0.0005,
                stock_code=clean_code,
                board='us',
                share_rounding='one_share',
            )

        if market == 'HK':
            known = clean_code in HK_COMMON_LOT_SIZES
            lot = HK_COMMON_LOT_SIZES.get(clean_code, 100)
            if not known and clean_code:
                logger.warning(
                    f"[港股手数] {clean_code} 未收录在 HK_COMMON_LOT_SIZES，"
                    f"暂按每手 {lot} 股；实盘每手可能不同，请核对后补表。"
                )
            return cls(
                market='HK',
                lot_size=lot,
                stamp_duty_sell=0.001,
                stamp_duty_buy=0.001,
                commission_rate=0.0008,
                min_commission=15.0,
                settlement_days=0,
                price_limit_pct=None,
                risk_free_rate=0.035,
                currency='HKD',
                currency_symbol='HK$',
                extra_fee_rate=0.000105,
                default_capital=300000.0,
                annual_trading_days=246,
                default_slippage=0.0015,
                stock_code=clean_code,
                board='hk',
                share_rounding='lot_multiple',
                unknown_hk_lot=not known,
            )

        # ---- A 股 ----
        is_st = _is_st_name(stock_name)
        if is_st:
            limit = 0.05
            board = 'st'
        elif clean_code.startswith(('4', '8')) or clean_code.startswith('92'):
            limit = 0.30
            board = 'bse'
        elif clean_code.startswith(('688', '689')):
            limit = 0.20
            board = 'star'
        elif clean_code.startswith(('300', '301')):
            limit = 0.20
            board = 'chi_next'
        else:
            limit = 0.10
            board = 'main'

        # 科创板申报起点 200；其余主板/创业/北交所起点 100
        lot = 200 if board == 'star' else 100
        # 主板必须整手；创业/科创/北交：最少 lot，超出按 1 股
        if board == 'main' or board == 'st':
            rounding = 'lot_multiple'
        else:
            rounding = 'min_then_one'

        # 过户费仅沪市（60/68）双边约 0.001%
        transfer = 0.00001 if clean_code.startswith(('60', '68')) else 0.0

        return cls(
            market='CN',
            lot_size=lot,
            stamp_duty_sell=0.0005,
            stamp_duty_buy=0.0,
            commission_rate=0.0003,
            min_commission=5.0,
            settlement_days=1,
            price_limit_pct=limit,
            risk_free_rate=0.025,
            currency='CNY',
            currency_symbol='¥',
            extra_fee_rate=0.0,
            transfer_fee_rate=transfer,
            default_capital=100000.0,
            annual_trading_days=244,
            default_slippage=0.001,
            stock_code=clean_code,
            board=board,
            share_rounding=rounding,
        )

    def _stamp_rates(self, trade_date=None) -> tuple:
        """按成交日返回 (买方印花税, 卖方印花税)。港股在下调日前后不同。"""
        buy, sell = self.stamp_duty_buy, self.stamp_duty_sell
        if self.market == 'HK':
            d = _to_date(trade_date)
            if d is not None and d < HK_STAMP_DUTY_CUTOVER:
                buy = sell = 0.0013
            else:
                buy = sell = 0.001
        return buy, sell

    def buy_cost(self, gross: float, trade_date=None) -> float:
        """买入侧总税费（佣金 + 买方印花税 + 过户/征费）。美股卖方规费不计入。"""
        stamp_buy, _ = self._stamp_rates(trade_date)
        if self.min_commission > 0 or self.commission_rate > 0:
            commission = max(self.min_commission, gross * self.commission_rate)
        else:
            commission = 0.0
        return commission + gross * (stamp_buy + self.extra_fee_rate + self.transfer_fee_rate)

    def sell_cost(self, gross: float, trade_date=None) -> float:
        """卖出侧总税费（佣金 + 卖方印花税 + 过户/征费 + 仅卖方规费）。"""
        _, stamp_sell = self._stamp_rates(trade_date)
        if self.min_commission > 0 or self.commission_rate > 0:
            commission = max(self.min_commission, gross * self.commission_rate)
        else:
            commission = 0.0
        return commission + gross * (
            stamp_sell + self.extra_fee_rate + self.transfer_fee_rate + self.sell_only_extra_fee_rate
        )

    def round_shares(self, shares: float) -> int:
        """
        按市场申报规则返回可申报股数。

        - lot_multiple: 主板整手（向下取整到 lot_size 倍数）
        - min_then_one: 最少 lot_size，超出按 1 股（创业/科创/北交）
        - one_share: 美股 1 股起
        """
        if shares is None or shares <= 0 or np.isnan(shares):
            return 0
        s = float(shares)
        if self.share_rounding == 'one_share':
            return int(s)  # 向下取整到股
        if self.share_rounding == 'min_then_one':
            if s < self.lot_size:
                return 0
            return int(s)  # 已满足最低申报，按 1 股递增
        # lot_multiple
        if s < self.lot_size:
            return 0
        return int(s // self.lot_size) * self.lot_size

    def limit_prices(self, prev_close: float) -> tuple:
        """
        由前收计算理论涨跌停价（四舍五入到分）。
        无涨跌停市场返回 (None, None)。
        """
        if self.price_limit_pct is None or prev_close is None or prev_close <= 0:
            return None, None
        up = round(prev_close * (1.0 + self.price_limit_pct), 2)
        down = round(prev_close * (1.0 - self.price_limit_pct), 2)
        return up, down

    def is_open_limit_up(self, open_price: float, prev_close: float, tol: float = 0.01) -> bool:
        """执行日开盘是否已封在涨停价（无法买入）。"""
        up, _ = self.limit_prices(prev_close)
        if up is None or open_price is None:
            return False
        return open_price >= up - tol

    def is_open_limit_down(self, open_price: float, prev_close: float, tol: float = 0.01) -> bool:
        """执行日开盘是否已封在跌停价（无法卖出）。"""
        _, down = self.limit_prices(prev_close)
        if down is None or open_price is None:
            return False
        return open_price <= down + tol

    def describe(self) -> str:
        limit = '无涨跌停' if self.price_limit_pct is None else f'涨跌停±{self.price_limit_pct:.0%}'
        round_label = {
            'lot_multiple': f'整手{self.lot_size}',
            'min_then_one': f'最少{self.lot_size}超出按1股',
            'one_share': '1股起',
        }.get(self.share_rounding, self.share_rounding)
        return (
            f"{self.market}/{self.board} ({self.currency}) | {round_label} | "
            f"佣金{self.commission_rate:.4%}(最低{self.currency_symbol}{self.min_commission:g}) | "
            f"卖方印花税{self.stamp_duty_sell:.3%} | T+{self.settlement_days} | "
            f"{limit} | 年化交易日{self.annual_trading_days}"
        )


def compute_open_limit_flags(
    df: pd.DataFrame,
    profile: MarketProfile,
    open_col: str = 'open',
    prev_close_col: str = 'close_prev',
) -> pd.DataFrame:
    """
    在信号/回测矩阵上写入开盘涨跌停标志（仅有涨跌停的市场）。

    列: open_limit_up, open_limit_down（0/1）
    """
    out = df.copy()
    if profile.price_limit_pct is None:
        out['open_limit_up'] = 0
        out['open_limit_down'] = 0
        return out

    opens = out[open_col].astype(float)
    prevs = out[prev_close_col].astype(float)
    ups = (prevs * (1.0 + profile.price_limit_pct)).round(2)
    downs = (prevs * (1.0 - profile.price_limit_pct)).round(2)
    out['open_limit_up'] = ((opens >= ups - 0.01) & prevs.notna() & opens.notna()).astype(int)
    out['open_limit_down'] = ((opens <= downs + 0.01) & prevs.notna() & opens.notna()).astype(int)
    return out
