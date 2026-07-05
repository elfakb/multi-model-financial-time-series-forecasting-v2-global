import pandas as pd
import os
from src.data.fetch_data import load_raw, ASSETS, PERIODS


def clean(df):
    df = df.interpolate(method="linear", limit=2)
    df = df.ffill()
    df = df.dropna()
    return df


def split(df, test_ratio=0.2):
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def process_and_save():
    for period in PERIODS:
        os.makedirs(f"data/processed/{period}", exist_ok=True)

        for name in ASSETS:
            df = load_raw(name, period=period)
            df = clean(df)
            df.to_csv(f"data/processed/{period}/{name}_processed.csv")
            print(f"{name} [{period}] kaydedildi: {len(df)} satır")


def load_processed(name, period="5y"):
    return pd.read_csv(f"data/processed/{period}/{name}_processed.csv", index_col="Date", parse_dates=True)


if __name__ == "__main__":
    process_and_save()