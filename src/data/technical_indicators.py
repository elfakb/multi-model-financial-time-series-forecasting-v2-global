import pandas as pd
import ta
import os
from src.data.fetch_data import ASSETS, PERIODS
from src.data.preprocess import load_processed


def add_indicators(df):
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    df["RSI"]        = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["MACD"]       = ta.trend.MACD(close=close).macd()
    df["MACD_signal"]= ta.trend.MACD(close=close).macd_signal()
    df["BB_upper"]   = ta.volatility.BollingerBands(close=close).bollinger_hband()
    df["BB_lower"]   = ta.volatility.BollingerBands(close=close).bollinger_lband()
    df["BB_mid"]     = ta.volatility.BollingerBands(close=close).bollinger_mavg()
    df["MA_20"]      = close.rolling(20).mean()
    df["MA_50"]      = close.rolling(50).mean()
    df["EMA_20"]     = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df["EMA_50"]     = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    df["ATR"]        = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range()
    df["OBV"]        = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    return df.dropna()


def process_and_save():
    for period in PERIODS:
        for name in ASSETS:
            df = load_processed(name, period=period)
            df = add_indicators(df)
            df.to_csv(f"data/processed/{period}/{name}_features.csv")
            print(f"{name} [{period}] kaydedildi: {len(df)} satır, {len(df.columns)} kolon")


def load_features(name, period="5y"):
    return pd.read_csv(f"data/processed/{period}/{name}_features.csv", index_col="Date", parse_dates=True)


if __name__ == "__main__":
    process_and_save()