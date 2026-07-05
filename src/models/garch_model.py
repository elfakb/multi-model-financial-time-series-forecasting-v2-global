import numpy as np
import pandas as pd
import joblib
import mlflow
import os
from arch import arch_model
from src.data.preprocess import load_processed, split
from src.data.fetch_data import ASSETS, PERIODS, FORECAST_HORIZONS
from src.evaluation.metrics import evaluate

SAVE_DIR = "models/saved"


def get_returns(close):
    return 100 * pd.Series(close).pct_change().dropna()


def train(asset_name, period="5y"):
    os.makedirs(f"{SAVE_DIR}/{period}", exist_ok=True)

    df = load_processed(asset_name, period=period)
    train_df, test_df = split(df)

    train_returns = get_returns(train_df["Close"].values)
    test_returns  = get_returns(test_df["Close"].values)

    print(f"[GARCH] {asset_name} [{period}] eğitiliyor...")

    model  = arch_model(train_returns, vol="Garch", p=1, q=1, dist="normal")
    result = model.fit(disp="off")

    forecasts = result.forecast(horizon=len(test_returns), reindex=False)
    pred_vol  = np.sqrt(forecasts.variance.values[-1])
    real_vol  = test_returns.values ** 2

    min_len  = min(len(pred_vol), len(real_vol))
    results  = evaluate(real_vol[:min_len], pred_vol[:min_len], model_name="GARCH")

    joblib.dump(result, f"{SAVE_DIR}/{period}/{asset_name}_garch.pkl")
    print(f"  Kaydedildi | AIC: {result.aic:.4f}")

    with mlflow.start_run(run_name=f"{asset_name}_{period}_GARCH"):
        mlflow.log_param("asset",  asset_name)
        mlflow.log_param("period", period)
        mlflow.log_param("aic",    round(result.aic, 4))
        mlflow.log_metric("RMSE",  results["RMSE"])
        mlflow.log_metric("MAE",   results["MAE"])
        mlflow.log_metric("MAPE",  results["MAPE"])

    return result, pred_vol, results


def predict_volatility_band(asset_name, last_price, period="5y", n_days=30):
    result     = joblib.load(f"{SAVE_DIR}/{period}/{asset_name}_garch.pkl")
    forecasts  = result.forecast(horizon=n_days, reindex=False)
    vol        = np.sqrt(forecasts.variance.values[-1])
    last_price = np.array(last_price)
    upper      = last_price * (1 + vol / 100)
    lower      = last_price * (1 - vol / 100)
    return upper, lower, vol


if __name__ == "__main__":
    for period in PERIODS:
        for asset in ASSETS:
            result, pred_vol, results = train(asset, period=period)
            print(f"{asset} [{period}] → {results}")

            df = load_processed(asset, period=period)
            for n_days in FORECAST_HORIZONS[period]:
                dummy        = np.full(n_days, df["Close"].iloc[-1])
                upper, lower, vol = predict_volatility_band(asset, dummy, period=period, n_days=n_days)
                print(f"  {n_days} gün | Vol: {vol[:3]}")
        print()