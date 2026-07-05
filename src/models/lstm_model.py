import numpy as np
import torch
import torch.nn as nn
import mlflow
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from src.data.preprocess import load_processed, split
from src.data.fetch_data import ASSETS, PERIODS, FORECAST_HORIZONS
from src.evaluation.metrics import evaluate

SAVE_DIR = "models/saved"
SEQ_LEN  = 60
EPOCHS   = 50
LR       = 0.001
HIDDEN   = 64
LAYERS   = 2

SEQ_LEN_MAP = {
    "6m": 10,
    "1y": 20,
    "5y": 60,
}


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN, num_layers=LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def train(asset_name, period="5y"):
    os.makedirs(f"{SAVE_DIR}/{period}", exist_ok=True)
    seq_len = SEQ_LEN_MAP[period]

    df = load_processed(asset_name, period=period)
    train_df, test_df = split(df)

    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[["Close"]].values)
    test_scaled  = scaler.transform(test_df[["Close"]].values)

    X_train, y_train = make_sequences(train_scaled, seq_len)
    X_test,  y_test  = make_sequences(test_scaled,  seq_len)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)

    model     = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"[LSTM] {asset_name} [{period}] eğitiliyor...")

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}  Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test).numpy()

    preds      = scaler.inverse_transform(preds_scaled).flatten()
    test_close = scaler.inverse_transform(y_test).flatten()
    results    = evaluate(test_close, preds, model_name="LSTM")

    torch.save(model.state_dict(), f"{SAVE_DIR}/{period}/{asset_name}_lstm.pt")
    joblib.dump(scaler, f"{SAVE_DIR}/{period}/{asset_name}_lstm_scaler.pkl")

    with mlflow.start_run(run_name=f"{asset_name}_{period}_LSTM"):
        mlflow.log_param("asset",   asset_name)
        mlflow.log_param("period",  period)
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_param("epochs",  EPOCHS)
        mlflow.log_metric("RMSE",   results["RMSE"])
        mlflow.log_metric("MAE",    results["MAE"])
        mlflow.log_metric("MAPE",   results["MAPE"])

    return model, scaler, preds, results


def predict(asset_name, period="5y", n_days=30):
    seq_len = SEQ_LEN_MAP[period]
    scaler  = joblib.load(f"{SAVE_DIR}/{period}/{asset_name}_lstm_scaler.pkl")
    model   = LSTMModel()
    model.load_state_dict(torch.load(f"{SAVE_DIR}/{period}/{asset_name}_lstm.pt"))
    model.eval()

    df         = load_processed(asset_name, period=period)
    last_price = df["Close"].iloc[-1]
    last_seq   = scaler.transform(df[["Close"]].values)[-seq_len:]

    # list değil tensor olarak başlat
    seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)

    preds = []
    with torch.no_grad():
        for _ in range(n_days):
            out = model(seq)
            preds.append(out.item())
            seq = torch.cat([seq[:, 1:, :], out.unsqueeze(0)], dim=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    preds = preds * (last_price / preds[0])
    return preds


if __name__ == "__main__":
    for period in PERIODS:
        for asset in ASSETS:
            model, scaler, preds, results = train(asset, period=period)
            print(f"{asset} [{period}] → {results}")

            for n_days in FORECAST_HORIZONS[period]:
                forecast = predict(asset, period=period, n_days=n_days)
                print(f"  {n_days} günlük tahmin (ilk 3): {forecast[:3]}")
        print()