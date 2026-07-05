import numpy as np
import pandas as pd
import torch
import mlflow
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from lightning.pytorch import Trainer
from src.data.preprocess import load_processed, split
from src.data.fetch_data import ASSETS
from src.evaluation.metrics import evaluate

SAVE_DIR          = "models/saved"
SEQ_LEN           = 60
EPOCHS            = 30
BATCH             = 64
FORECAST_HORIZONS = [30, 60, 90]


def prepare_dataset(df, seq_len):
    df = df[["Close"]].copy()
    df["time_idx"] = range(len(df))
    df["group"]    = "asset"
    df["Close"]    = df["Close"].astype(float)
    return df


def train(asset_name):
    os.makedirs(SAVE_DIR, exist_ok=True)

    df = load_processed(asset_name)
    train_df, test_df = split(df)

    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["Close"] = scaler.fit_transform(train_df[["Close"]])
    test_df["Close"]  = scaler.transform(test_df[["Close"]])

    dataset = TimeSeriesDataSet(
        prepare_dataset(train_df, SEQ_LEN),
        time_idx="time_idx",
        target="Close",
        group_ids=["group"],
        min_encoder_length=SEQ_LEN // 2,
        max_encoder_length=SEQ_LEN,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_unknown_reals=["Close"],
        add_relative_time_idx=True,
        add_target_scales=True,
    )

    train_loader = dataset.to_dataloader(train=True, batch_size=BATCH, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=0.001,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        loss=MAE(),
    )

    print(f"[TFT] {asset_name} eğitiliyor...")
    trainer = Trainer(max_epochs=EPOCHS, enable_progress_bar=True, logger=False)
    trainer.fit(model, train_dataloaders=train_loader)

    test_loader = dataset.to_dataloader(train=False, batch_size=BATCH, num_workers=0)
    preds_raw   = model.predict(test_loader).numpy().flatten()
    preds       = scaler.inverse_transform(preds_raw.reshape(-1, 1)).flatten()
    test_close  = scaler.inverse_transform(test_df["Close"].values.reshape(-1, 1)).flatten()

    min_len    = min(len(preds), len(test_close))
    preds      = preds[:min_len]
    test_close = test_close[:min_len]

    results = evaluate(test_close, preds, model_name="TFT")

    torch.save(model.state_dict(), f"{SAVE_DIR}/{asset_name}_tft.pt")
    joblib.dump(scaler, f"{SAVE_DIR}/{asset_name}_tft_scaler.pkl")
    print(f"  Model kaydedildi: {SAVE_DIR}/{asset_name}_tft.pt")

    with mlflow.start_run(run_name=f"{asset_name}_TFT"):
        mlflow.log_param("asset",   asset_name)
        mlflow.log_param("seq_len", SEQ_LEN)
        mlflow.log_param("epochs",  EPOCHS)
        mlflow.log_metric("RMSE", results["RMSE"])
        mlflow.log_metric("MAE",  results["MAE"])
        mlflow.log_metric("MAPE", results["MAPE"])

    return model, scaler, preds, results


def predict(asset_name, n_days=30):
    scaler = joblib.load(f"{SAVE_DIR}/{asset_name}_tft_scaler.pkl")
    df     = load_processed(asset_name)
    scaled = scaler.transform(df[["Close"]].values)
    seq    = list(scaled[-SEQ_LEN:].flatten())

    preds = []
    for _ in range(n_days):
        next_val = seq[-1]
        preds.append(next_val)
        seq.append(next_val)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


if __name__ == "__main__":
    for asset in ASSETS:
        model, scaler, preds, results = train(asset)
        print(f"{asset} → {results}")

        for n_days in FORECAST_HORIZONS:
            forecast = predict(asset, n_days=n_days)
            print(f"  {n_days} günlük tahmin (ilk 3): {forecast[:3]}")
        print()