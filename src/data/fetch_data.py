import yfinance as yf
import pandas as pd
import os
from datetime import date, timedelta

ASSETS = {
    "AAPL"  : "AAPL",
    "TSLA"  : "TSLA",
    "NVDA"  : "NVDA",
    "GOOGL" : "GOOGL",
    "MSFT"  : "MSFT",
    "AMZN"  : "AMZN",
    "META"  : "META",
    "BTC"   : "BTC-USD",
    "ETH"   : "ETH-USD",
}

END_DATE = date.today().strftime("%Y-%m-%d")

PERIODS = {
    "6m": (date.today() - timedelta(days=180)).strftime("%Y-%m-%d"),
    "1y": (date.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
    "5y": (date.today() - timedelta(days=5*365)).strftime("%Y-%m-%d"),
}

FORECAST_HORIZONS = {
    "6m": [5, 20, 60],
    "1y": [5, 20, 60],
    "5y": [30, 60, 90],
}


def fetch_and_save():
    for period, start_date in PERIODS.items():
        raw_dir = f"data/raw/{period}"
        os.makedirs(raw_dir, exist_ok=True)

        for name, ticker in ASSETS.items():
            print(f"Çekiliyor: {name} [{period}]")
            df = yf.download(ticker, start=start_date, end=END_DATE, auto_adjust=True, progress=False)

            if df.empty:
                print(f"  HATA: {name} verisi gelmedi.")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.to_csv(f"{raw_dir}/{name}.csv")
            print(f"  Kaydedildi: {len(df)} satır")


def load_raw(name, period="5y"):
    return pd.read_csv(f"data/raw/{period}/{name}.csv", index_col="Date", parse_dates=True)


if __name__ == "__main__":
    fetch_and_save()