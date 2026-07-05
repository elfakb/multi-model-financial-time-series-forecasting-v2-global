import numpy as np
import pandas as pd
import joblib
import mlflow
import os
import xgboost as xgb
from src.data.preprocess import load_processed, split
from src.data.fetch_data import ASSETS, PERIODS, FORECAST_HORIZONS
from src.evaluation.metrics import evaluate

SAVE_DIR = "models/saved"
SEQ_LEN  = 60


def create_features(df):
    df = df.copy()
    for i in range(1, SEQ_LEN + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    return df.dropna()


def train(asset_name, period="5y"):
    os.makedirs(f"{SAVE_DIR}/{period}", exist_ok=True)

    df = load_processed(asset_name, period=period)
    train_df, test_df = split(df)

    train_feat = create_features(train_df)
    test_feat  = create_features(test_df)

    X_train = train_feat[[f"lag_{i}" for i in range(1, SEQ_LEN + 1)]]
    y_train = train_feat["Close"]
    X_test  = test_feat[[f"lag_{i}" for i in range(1, SEQ_LEN + 1)]]
    y_test  = test_feat["Close"]

    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, verbosity=0)

    print(f"[XGBoost] {asset_name} [{period}] eğitiliyor...")
    model.fit(X_train, y_train)

    preds   = model.predict(X_test)
    results = evaluate(y_test.values, preds, model_name="XGBoost")

    joblib.dump(model, f"{SAVE_DIR}/{period}/{asset_name}_xgb.pkl")

    with mlflow.start_run(run_name=f"{asset_name}_{period}_XGBoost"):
        mlflow.log_param("asset",   asset_name)
        mlflow.log_param("period",  period)
        mlflow.log_metric("RMSE",   results["RMSE"])
        mlflow.log_metric("MAE",    results["MAE"])
        mlflow.log_metric("MAPE",   results["MAPE"])

    return model, preds, results


def predict(asset_name, period="5y", n_days=30):
    model      = joblib.load(f"{SAVE_DIR}/{period}/{asset_name}_xgb.pkl")
    df         = load_processed(asset_name, period=period)
    last_price = df["Close"].iloc[-1]
    df_feat    = create_features(df)
    last       = df_feat[[f"lag_{i}" for i in range(1, SEQ_LEN + 1)]].iloc[-1].values.copy()

    preds = []
    for _ in range(n_days):
        pred = model.predict(last.reshape(1, -1))[0]
        preds.append(pred)
        last = np.roll(last, 1)
        last[0] = pred

    preds = np.array(preds)
    preds = preds * (last_price / preds[0])
    return preds


if __name__ == "__main__":
    for period in PERIODS:
        for asset in ASSETS:
            model, preds, results = train(asset, period=period)
            print(f"{asset} [{period}] → {results}")

            for n_days in FORECAST_HORIZONS[period]:
                forecast = predict(asset, period=period, n_days=n_days)
                print(f"  {n_days} günlük tahmin (ilk 3): {forecast[:3]}")
        print()