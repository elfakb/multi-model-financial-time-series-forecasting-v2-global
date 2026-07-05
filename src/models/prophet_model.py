import pandas as pd
import numpy as np
import joblib
import mlflow
import warnings
import os
from prophet import Prophet
from src.data.preprocess import load_processed, split
from src.data.fetch_data import ASSETS, PERIODS, FORECAST_HORIZONS
from src.evaluation.metrics import evaluate

warnings.filterwarnings("ignore")

SAVE_DIR = "models/saved"


def train(asset_name, period="5y"):
    os.makedirs(f"{SAVE_DIR}/{period}", exist_ok=True)

    df = load_processed(asset_name, period=period)
    train_df, test_df = split(df)

    train_prophet = train_df[["Close"]].reset_index()
    train_prophet.columns = ["ds", "y"]

    print(f"[Prophet] {asset_name} [{period}] eğitiliyor...")

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
    model.fit(train_prophet)

    future   = model.make_future_dataframe(periods=len(test_df), freq="B")
    forecast = model.predict(future)

    preds      = forecast["yhat"].values[-len(test_df):]
    test_close = test_df["Close"].values
    results    = evaluate(test_close, preds, model_name="Prophet")

    joblib.dump(model, f"{SAVE_DIR}/{period}/{asset_name}_prophet.pkl")

    with mlflow.start_run(run_name=f"{asset_name}_{period}_Prophet"):
        mlflow.log_param("asset",  asset_name)
        mlflow.log_param("period", period)
        mlflow.log_metric("RMSE",  results["RMSE"])
        mlflow.log_metric("MAE",   results["MAE"])
        mlflow.log_metric("MAPE",  results["MAPE"])

    return model, preds, results


def predict(asset_name, period="5y", n_days=30):
    model    = joblib.load(f"{SAVE_DIR}/{period}/{asset_name}_prophet.pkl")
    future   = model.make_future_dataframe(periods=n_days, freq="B")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(n_days)


if __name__ == "__main__":
    for period in PERIODS:
        for asset in ASSETS:
            model, preds, results = train(asset, period=period)
            print(f"{asset} [{period}] → {results}")

            for n_days in FORECAST_HORIZONS[period]:
                forecast = predict(asset, period=period, n_days=n_days)
                print(f"  {n_days} günlük tahmin (ilk 3):\n{forecast.head(3)}")
        print()