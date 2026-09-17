# -*- coding: utf-8 -*-
"""
prediction_cn_markets_day.py

Description:
    Predicts future daily K-line (1D) data for A-share markets using Kronos model and akshare.
    The script automatically downloads the latest historical data, cleans it, and runs model inference.

Usage:
    python prediction_cn_markets_day.py --symbol 000001

Arguments:
    --symbol     Stock code (e.g. 002594 for BYD, 000001 for SSE Index)

Output:
    - Saves the prediction results to ./outputs/pred_<symbol>_data.csv and ./outputs/pred_<symbol>_chart.png
    - Logs and progress are printed to console

Example:
    bash> python prediction_cn_markets_day.py --symbol 000001
    python3 prediction_cn_markets_day.py --symbol 002594
"""

import os
import argparse
import time
import pandas as pd
import yfinance as yf
import baostock as bs
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosTokenizer, KronosPredictor

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(save_dir, exist_ok=True)

# Setting
TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PRETRAINED = "NeoQuasar/Kronos-base"
DEVICE = "cpu"  # "cuda:0"
MAX_CONTEXT = 512
LOOKBACK = 400
PRED_LEN = 120
T = 0.3
TOP_P = 0.8
SAMPLE_COUNT = 1

def load_data(symbol: str) -> pd.DataFrame:
    """
    拉取 A 股日线历史数据。
    支持 yfinance（代码格式如 000001.SZ 或 600000.SS），
    若 yfinance 触发 Yahoo 限流 (429 Rate Limit) 或下载失败，自动降级至 baostock 直连国内通道。
    """
    df = None
    clean_sym = symbol.strip().upper()
    
    # 标准化 yfinance 股票代码
    if "." in clean_sym:
        parts = clean_sym.split(".")
        if parts[0] in ["SH", "SZ", "SS"]:
            yf_symbol = f"{parts[1]}.{'SS' if parts[0] in ['SH', 'SS'] else 'SZ'}"
        else:
            yf_symbol = f"{parts[0]}.{'SS' if parts[1] in ['SH', 'SS'] else 'SZ'}"
    else:
        if clean_sym.startswith(("60", "68", "90")):
            yf_symbol = f"{clean_sym}.SS"
        else:
            yf_symbol = f"{clean_sym}.SZ"

    print(f"📥 Fetching {clean_sym} daily data from yfinance (ticker: {yf_symbol}, 前复权) ...")
    try:
        yf_df = yf.download(yf_symbol, period="3y", auto_adjust=True, progress=False)
        if yf_df is not None and not yf_df.empty:
            # 兼容 MultiIndex columns
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            yf_df = yf_df.reset_index()
            yf_df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)
            if "date" in yf_df.columns and len(yf_df) > 0:
                df = yf_df
                print(f"✅ Successfully fetched data from yfinance (前复权): {len(df)} records")
    except Exception as e:
        print(f"⚠️ yfinance fetch failed: {e}")

    # 若 yfinance 限流 (Rate Limit) 或无数据，降级至 baostock 直连国内获取
    if df is None or df.empty or len(df) < LOOKBACK:
        print(f"🔄 Switching to baostock for reliable domestic data (前复权) ...")
        # 标准化 baostock 股票代码 (sh.600000 / sz.000001)
        raw_code = clean_sym.split(".")[0] if not clean_sym.startswith(("SH", "SZ")) else clean_sym.split(".")[1]
        bs_code = f"sh.{raw_code}" if raw_code.startswith(("60", "68", "90")) else f"sz.{raw_code}"
        
        lg = bs.login()
        if lg.error_code == '0':
            # adjustflag="2" 表示前复权 (1: 后复权, 2: 前复权, 3: 不复权)
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date="2022-01-01",
                frequency="d",
                adjustflag="2"
            )
            if rs.error_code == '0':
                df = rs.get_data()
            bs.logout()

    if df is None or df.empty:
        print(f"❌ Failed to fetch data for {symbol}. Exiting.")
        sys.exit(1)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 统一转换数值列
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    else:
        df["amount"] = df["close"] * df["volume"]

    # 补充无效数据
    open_bad = (df["open"] == 0) | (df["open"].isna())
    if open_bad.any():
        df.loc[open_bad, "open"] = df["close"].shift(1)
        df["open"].fillna(df["close"], inplace=True)

    if df["amount"].isna().all() or (df["amount"] == 0).all():
        df["amount"] = df["close"] * df["volume"]

    print(f"✅ Data ready: {len(df)} rows, range: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
    return df


def prepare_inputs(df):
    x_df = df.iloc[-LOOKBACK:][["open","high","low","close","volume","amount"]]
    x_timestamp = df.iloc[-LOOKBACK:]["date"]
    y_timestamp = pd.bdate_range(start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=PRED_LEN)
    return x_df, pd.Series(x_timestamp), pd.Series(y_timestamp)

def apply_price_limits(pred_df, last_close, limit_rate=0.1):
    print(f"🔒 Applying ±{limit_rate*100:.0f}% price limit ...")

    # Ensure integer index
    pred_df = pred_df.reset_index(drop=True)

    # Ensure float64 dtype for safe assignment
    cols = ["open", "high", "low", "close"]
    pred_df[cols] = pred_df[cols].astype("float64")

    for i in range(len(pred_df)):
        limit_up = last_close * (1 + limit_rate)
        limit_down = last_close * (1 - limit_rate)

        for col in cols:
            value = pred_df.at[i, col]
            if pd.notna(value):
                clipped = max(min(value, limit_up), limit_down)
                pred_df.at[i, col] = float(clipped)

        last_close = float(pred_df.at[i, "close"])  # ensure float type

    return pred_df


def plot_result(df_hist, df_pred, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(df_hist["date"], df_hist["close"], label="Historical", color="blue")
    plt.plot(df_pred["date"], df_pred["close"], label="Predicted", color="red", linestyle="--")
    plt.title(f"Kronos Prediction for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"pred_{symbol.replace('.', '_')}_chart.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    print(f"📊 Chart saved: {plot_path}")


def predict_future(symbol):
    print(f"🚀 Loading Kronos tokenizer:{TOKENIZER_PRETRAINED} model:{MODEL_PRETRAINED} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PRETRAINED)
    model = Kronos.from_pretrained(MODEL_PRETRAINED)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)

    df = load_data(symbol)
    x_df, x_timestamp, y_timestamp = prepare_inputs(df)

    print("🔮 Generating predictions ...")

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=T,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
    )

    pred_df["date"] = y_timestamp.values

    # Apply ±10% price limit
    last_close = df["close"].iloc[-1]
    pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

    # Merge historical and predicted data
    df_out = pd.concat([
        df[["date", "open", "high", "low", "close", "volume", "amount"]],
        pred_df[["date", "open", "high", "low", "close", "volume", "amount"]]
    ]).reset_index(drop=True)

    # Save CSV
    out_file = os.path.join(save_dir, f"pred_{symbol.replace('.', '_')}_data.csv")
    df_out.to_csv(out_file, index=False)
    print(f"✅ Prediction completed and saved: {out_file}")

    # Plot
    plot_result(df, pred_df, symbol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos stock prediction script")
    parser.add_argument("--symbol", type=str, default="002594.SZ", help="Stock code")
    args = parser.parse_args()

    predict_future(
        symbol=args.symbol,
    )
