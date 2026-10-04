"""
三市场交易规则与组合合成单测（无网络、不跑模型）。
可用: python -m unittest backtest.test_market_rules -v
或:   pytest backtest/test_market_rules.py -v
"""
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
for p in (str(ROOT), str(EXAMPLES), str(BACKTEST_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from market_rules import (  # noqa: E402
    SymbolResolver,
    MarketProfile,
    HK_STAMP_DUTY_CUTOVER,
    compute_open_limit_flags,
)
from prediction_akshare_2024_2025_copy import validate_stock_code  # noqa: E402


def _approx(a, b, rel=1e-6):
    return abs(a - b) <= max(abs(a), abs(b), 1.0) * rel


class TestSymbolResolve(unittest.TestCase):
    def test_us_class_shares(self):
        clean, mkt, _ = SymbolResolver.resolve('BRK.B')
        self.assertEqual(mkt, 'US')
        self.assertEqual(clean, 'BRK.B')
        clean2, mkt2, _ = SymbolResolver.resolve('BRK-B')
        self.assertEqual(mkt2, 'US')
        self.assertEqual(clean2, 'BRK.B')

    def test_prefixes(self):
        self.assertEqual(SymbolResolver.resolve('HK_00700')[0:2], ('00700', 'HK'))
        self.assertEqual(SymbolResolver.resolve('US_AAPL')[0:2], ('AAPL', 'US'))
        self.assertEqual(SymbolResolver.resolve('300418')[1], 'CN')

    def test_validate_aligned(self):
        ok, _, code, mkt = validate_stock_code('HK_00700')
        self.assertTrue(ok)
        self.assertEqual((code, mkt), ('00700', 'HK'))
        ok, _, code, mkt = validate_stock_code('BRK.B')
        self.assertTrue(ok)
        self.assertEqual(mkt, 'US')


class TestShareRounding(unittest.TestCase):
    def test_cn_boards(self):
        star = MarketProfile.for_stock('688001')
        self.assertEqual(star.lot_size, 200)
        self.assertEqual(star.round_shares(250), 250)
        self.assertEqual(star.round_shares(150), 0)

        chi = MarketProfile.for_stock('300418')
        self.assertEqual(chi.round_shares(150), 150)

        main = MarketProfile.for_stock('600036')
        self.assertEqual(main.round_shares(150), 100)
        self.assertEqual(main.round_shares(99), 0)

    def test_bse_and_st_limits(self):
        bse = MarketProfile.for_stock('830799')
        self.assertAlmostEqual(bse.price_limit_pct, 0.30)
        st = MarketProfile.for_stock('600000', stock_name='*ST示例')
        self.assertAlmostEqual(st.price_limit_pct, 0.05)
        normal = MarketProfile.for_stock('600000', stock_name='浦发银行')
        self.assertAlmostEqual(normal.price_limit_pct, 0.10)

    def test_hk_lots(self):
        self.assertEqual(MarketProfile.for_stock('00941').lot_size, 500)
        self.assertEqual(MarketProfile.for_stock('00700').lot_size, 100)
        self.assertEqual(MarketProfile.for_stock('00700').round_shares(250), 200)


class TestFees(unittest.TestCase):
    def test_hk_stamp_cutover(self):
        hk = MarketProfile.for_stock('00700')
        gross = 100_000.0
        before = date(2023, 11, 16)
        self.assertLess(before, HK_STAMP_DUTY_CUTOVER)
        buy_before = hk.buy_cost(gross, trade_date=before)
        buy_after = hk.buy_cost(gross, trade_date=date(2023, 11, 17))
        self.assertTrue(_approx(buy_before, 80 + 130 + 10.5))
        self.assertTrue(_approx(buy_after, 80 + 100 + 10.5))
        self.assertGreater(buy_before, buy_after)

    def test_us_sell_exceeds_buy(self):
        us = MarketProfile.for_stock('AAPL')
        gross = 10_000.0
        buy = us.buy_cost(gross, trade_date=date(2024, 6, 1))
        sell = us.sell_cost(gross, trade_date=date(2024, 6, 1))
        self.assertAlmostEqual(buy, 0.0)
        self.assertGreater(sell, buy)
        self.assertTrue(_approx(sell, gross * 0.000028))

    def test_cn_transfer_fee(self):
        sh = MarketProfile.for_stock('600036')
        sz = MarketProfile.for_stock('000001')
        self.assertGreater(sh.transfer_fee_rate, 0)
        self.assertEqual(sz.transfer_fee_rate, 0.0)
        self.assertTrue(_approx(sh.buy_cost(100_000.0), 31.0))
        self.assertTrue(_approx(sz.buy_cost(100_000.0), 30.0))


class TestOpenLimit(unittest.TestCase):
    def test_open_not_close(self):
        """开盘未封、收盘封板可买；开盘封涨停不可买。"""
        prof = MarketProfile.for_stock('600036')
        df = pd.DataFrame({
            'open': [10.5, 11.00],
            'high': [11.00, 11.00],
            'low': [10.4, 10.9],
            'close': [11.00, 11.00],
            'close_prev': [10.0, 10.0],
        }, index=pd.to_datetime(['2024-01-02', '2024-01-03']))
        out = compute_open_limit_flags(df, prof)
        self.assertEqual(int(out['open_limit_up'].iloc[0]), 0)
        self.assertEqual(int(out['open_limit_up'].iloc[1]), 1)
        self.assertTrue(prof.is_open_limit_up(11.00, 10.0))
        self.assertFalse(prof.is_open_limit_up(10.5, 10.0))

    def test_annual_days(self):
        self.assertEqual(MarketProfile.for_stock('600036').annual_trading_days, 244)
        self.assertEqual(MarketProfile.for_stock('00700').annual_trading_days, 246)
        self.assertEqual(MarketProfile.for_stock('AAPL').annual_trading_days, 252)


class TestPortfolioCurve(unittest.TestCase):
    def test_skips_closed_market_days(self):
        from app_backtest_dashboard import calculate_portfolio_equity_curve

        with tempfile.TemporaryDirectory() as td:
            cn = os.path.join(td, '600036')
            hk = os.path.join(td, '00700')
            os.makedirs(cn)
            os.makedirs(hk)
            pd.DataFrame({
                'date': ['2024-01-02', '2024-01-04'],
                'portfolio_value': [100.0, 110.0],
            }).to_csv(os.path.join(cn, '600036_daily_backtest.csv'), index=False)
            pd.DataFrame({
                'date': ['2024-01-02', '2024-01-03', '2024-01-04'],
                'portfolio_value': [100.0, 105.0, 107.1],
            }).to_csv(os.path.join(hk, '00700_daily_backtest.csv'), index=False)

            curve = calculate_portfolio_equity_curve(td)
            self.assertIsNotNone(curve)
            # 01-03 仅港股 +5%；01-04 等权 (10%+2%)/2 = 6%
            self.assertTrue(_approx(curve['Portfolio_Basket'].iloc[0], 1.05))
            self.assertTrue(_approx(curve['Portfolio_Basket'].iloc[1], 1.05 * 1.06))


if __name__ == '__main__':
    unittest.main()
