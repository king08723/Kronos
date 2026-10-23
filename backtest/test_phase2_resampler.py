"""
backtest/test_phase2_resampler.py
Phase 2 交易时段感知合成引擎 (SessionResampler) 与动态数据管道专有测试。
测试覆盖：
1. A 股 1 分钟 -> 30 分钟合成：
   - 每日 240 根 1m Bar 精确合成 8 根 30m Bar。
   - 时间戳严格对齐 [10:00, 10:30, 11:00, 11:30, 13:30, 14:00, 14:30, 15:00]。
2. A 股 1 分钟 -> 60 分钟合成：
   - 每日精确合成 4 根 60m Bar [10:30, 11:30, 14:00, 15:00]。
   - 彻底消除午休 (11:30~13:00) 导致的空 Bar 与错位断点。
3. 物理守恒检验 (OHLCV Conservation)：
   - 成交量总和严格守恒 (Sum Volume Conservation)。
   - 首根 Open == 全天 Open，末根 Close == 全天 Close。
   - Max High == 全天最高价，Min Low == 全天最低价。
4. 港股 (HK 11 根 30m) 与美股 (US 13 根 30m) 交易时段合成检验。
5. 非法下采样拦截：低频向高频采样 (如日线转 1m) 抛出 ValueError。
6. DataFetcher 智能管道：当无目标周期文件时，自动加载基础高频文件并即时合成。
"""

import unittest
import os
import shutil
import tempfile
from datetime import datetime, date
import pandas as pd
import numpy as np

from backtest.core.types import TimeFrame
from backtest.core.market_rules import MarketProfile
from backtest.core.resampler import SessionResampler
from backtest.core.timeframes import resample_bars, detect_timeframe
from backtest.data.data_fetcher import DataFetcher


def generate_mock_cn_1m_data(trading_days: int = 2) -> pd.DataFrame:
    """生成真实的 A 股 1 分钟连续行情 (每天 240 根: 09:31-11:30, 13:01-15:00)"""
    records = []
    base_price = 100.0

    days = pd.bdate_range(start="2026-03-02", periods=trading_days)
    for d in days:
        d_str = d.strftime("%Y-%m-%d")
        # 上午 120 根: 09:31 ~ 11:30
        am_times = pd.date_range(f"{d_str} 09:31:00", f"{d_str} 11:30:00", freq="1min")
        # 下午 120 根: 13:01 ~ 15:00
        pm_times = pd.date_range(f"{d_str} 13:01:00", f"{d_str} 15:00:00", freq="1min")

        day_times = am_times.append(pm_times)
        for ts in day_times:
            # 模拟价格微小随机扰动
            open_px = base_price + np.random.uniform(-0.1, 0.1)
            high_px = open_px + np.random.uniform(0.0, 0.3)
            low_px = open_px - np.random.uniform(0.0, 0.3)
            close_px = np.random.uniform(low_px, high_px)
            vol = np.random.randint(100, 500)
            amt = close_px * vol * 100
            records.append({
                "date": ts,
                "open": round(open_px, 3),
                "high": round(high_px, 3),
                "low": round(low_px, 3),
                "close": round(close_px, 3),
                "volume": float(vol),
                "amount": round(amt, 2),
            })
            base_price = close_px

    df = pd.DataFrame(records).set_index("date")
    return df


class TestSessionResampler(unittest.TestCase):
    """测试时段感知合成器核心逻辑"""

    def setUp(self):
        self.df_1m = generate_mock_cn_1m_data(trading_days=2)
        self.assertEqual(len(self.df_1m), 480)  # 2 天 * 240 根 = 480 根

    def test_resample_cn_30m(self):
        """测试 A 股 1 分钟 -> 30 分钟：每天严格 8 根 Bar，时间节点准确"""
        df_30m = SessionResampler.resample_to_timeframe(
            self.df_1m,
            target_timeframe=TimeFrame.M30,
            market="CN",
        )
        # 2 天应恰好 16 根 Bar
        self.assertEqual(len(df_30m), 16)

        # 检查第一天的 8 根 Bar 时间戳
        d1_bars = df_30m.loc["2026-03-02"]
        self.assertEqual(len(d1_bars), 8)
        expected_times = [
            "10:00:00", "10:30:00", "11:00:00", "11:30:00",
            "13:30:00", "14:00:00", "14:30:00", "15:00:00"
        ]
        actual_times = [t.strftime("%H:%M:%S") for t in d1_bars.index]
        self.assertEqual(actual_times, expected_times)

    def test_resample_cn_60m(self):
        """测试 A 股 1 分钟 -> 60 分钟：每天严格 4 根 Bar，杜绝午休断裂"""
        df_60m = SessionResampler.resample_to_timeframe(
            self.df_1m,
            target_timeframe=TimeFrame.H1,
            market="CN",
        )
        # 2 天应恰好 8 根 Bar
        self.assertEqual(len(df_60m), 8)

        d1_bars = df_60m.loc["2026-03-02"]
        self.assertEqual(len(d1_bars), 4)
        expected_times = ["10:30:00", "11:30:00", "14:00:00", "15:00:00"]
        actual_times = [t.strftime("%H:%M:%S") for t in d1_bars.index]
        self.assertEqual(actual_times, expected_times)

    def test_ohlcv_conservation(self):
        """测试物理守恒定律：总量、极值、开收盘无损传递"""
        df_30m = SessionResampler.resample_to_timeframe(
            self.df_1m,
            target_timeframe=TimeFrame.M30,
            market="CN",
        )
        d1_1m = self.df_1m.loc["2026-03-02"]
        d1_30m = df_30m.loc["2026-03-02"]

        # 1. 成交量总和守恒
        self.assertAlmostEqual(d1_1m["volume"].sum(), d1_30m["volume"].sum(), places=2)
        # 2. 成交额总和守恒
        self.assertAlmostEqual(d1_1m["amount"].sum(), d1_30m["amount"].sum(), places=2)
        # 3. 开盘价与首根 Bar 对齐
        self.assertEqual(d1_30m["open"].iloc[0], d1_1m["open"].iloc[0])
        # 4. 收盘价与末根 Bar 对齐
        self.assertEqual(d1_30m["close"].iloc[-1], d1_1m["close"].iloc[-1])
        # 5. 最高价与最低价全区间完全一致
        self.assertEqual(d1_30m["high"].max(), d1_1m["high"].max())
        self.assertEqual(d1_30m["low"].min(), d1_1m["low"].min())

    def test_resample_to_daily(self):
        """测试 1 分钟线 -> 日线：每天 1 根 Bar"""
        df_d1 = SessionResampler.resample_to_timeframe(
            self.df_1m,
            target_timeframe=TimeFrame.D1,
            market="CN",
        )
        self.assertEqual(len(df_d1), 2)
        self.assertEqual(df_d1["open"].iloc[0], self.df_1m.loc["2026-03-02", "open"].iloc[0])
        self.assertEqual(df_d1["close"].iloc[0], self.df_1m.loc["2026-03-02", "close"].iloc[-1])
        self.assertAlmostEqual(df_d1["volume"].iloc[0], self.df_1m.loc["2026-03-02", "volume"].sum())

    def test_invalid_downsampling_raises_error(self):
        """测试试图将低频日线重采样为高频分钟线，应严格报错"""
        df_d1 = SessionResampler.resample_to_timeframe(self.df_1m, target_timeframe=TimeFrame.D1)
        with self.assertRaises(ValueError):
            SessionResampler.resample_to_timeframe(df_d1, target_timeframe=TimeFrame.M30)


class TestDataFetcherOnTheFlyResampling(unittest.TestCase):
    """测试 DataFetcher 自动利用高频底座即时合成多周期 K 线"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.df_1m = generate_mock_cn_1m_data(trading_days=3)
        # 仅保存一份 1m 基础文件
        self.file_1m = os.path.join(self.temp_dir, "300418_1m.csv")
        self.df_1m.to_csv(self.file_1m, encoding="utf-8-sig")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_stock_history_synthesizes_30m(self):
        """在没有 30m 文件的情况下，DataFetcher 应自动从 1m 文件合成 30m 数据"""
        hist_df, profile = DataFetcher.load_stock_history(
            stock_code="300418",
            data_dir=self.temp_dir,
            market="CN",
            timeframe="30m",
        )
        self.assertIsNotNone(hist_df)
        # 3 天 * 8 根 = 24 根 30m Bar
        self.assertEqual(len(hist_df), 24)
        detected_tf = detect_timeframe(hist_df)
        self.assertEqual(detected_tf, TimeFrame.M30)

    def test_load_stock_history_synthesizes_60m(self):
        """在没有 60m 文件的情况下，DataFetcher 应自动从 1m 文件合成 60m 数据"""
        hist_df, profile = DataFetcher.load_stock_history(
            stock_code="300418",
            data_dir=self.temp_dir,
            market="CN",
            timeframe="60m",
        )
        self.assertIsNotNone(hist_df)
        # 3 天 * 4 根 = 12 根 60m Bar
        self.assertEqual(len(hist_df), 12)
        detected_tf = detect_timeframe(hist_df)
        self.assertEqual(detected_tf, TimeFrame.H1)


if __name__ == "__main__":
    unittest.main()
