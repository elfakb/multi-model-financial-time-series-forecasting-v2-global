import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import warnings
import os
from pmdarima import auto_arima
from src.data.preprocess import load_processed, split
from src.data.fetch_data import ASSETS, PERIODS, FORECAST_HORIZONS
from src.evaluation.metrics import evaluate

warnings.filterwarnings("ignore")

SAVE_DIR = "models/saved"


def train(asset_name, period="5y"):
    os.makedirs(f"{SAVE_DIR}/{period}", exist_ok=True)

    df          = load_processed(asset_name, period=period)
    train_df, test_df = split(df)
    train_close = train_df["Close"].values
    test_close  = test_df["Close"].values

    print(f"[ARIMA] {asset_name} [{period}] eğitiliyor...")

    model = auto_arima(train_close, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
    preds   = model.predict(n_periods=len(test_close))
    results = evaluate(test_close, preds, model_name="ARIMA")

    joblib.dump(model, f"{SAVE_DIR}/{period}/{asset_name}_arima.pkl")

    with mlflow.start_run(run_name=f"{asset_name}_{period}_ARIMA"):
        mlflow.log_param("asset",  asset_name)
        mlflow.log_param("period", period)
        mlflow.log_param("order",  str(model.order))
        mlflow.log_metric("RMSE",  results["RMSE"])
        mlflow.log_metric("MAE",   results["MAE"])
        mlflow.log_metric("MAPE",  results["MAPE"])
        mlflow.sklearn.log_model(model, "arima_model", serialization_format="cloudpickle")

    return model, preds, results


def predict(asset_name, period="5y", n_days=30):
    model = joblib.load(f"{SAVE_DIR}/{period}/{asset_name}_arima.pkl")
    return model.predict(n_periods=n_days)


if __name__ == "__main__":
    for period in PERIODS:
        for asset in ASSETS:
            model, preds, results = train(asset, period=period)
            print(f"{asset} [{period}] → {results}")

            for n_days in FORECAST_HORIZONS[period]:
                forecast = predict(asset, period=period, n_days=n_days)
                print(f"  {n_days} günlük tahmin (ilk 3): {forecast[:3]}")
        print()