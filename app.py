import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath("."))

from src.data.fetch_data import ASSETS, FORECAST_HORIZONS
from src.data.preprocess import load_processed
from src.models.arima_model import predict as arima_predict
from src.models.sarima_model import predict as sarima_predict
from src.models.prophet_model import predict as prophet_predict
from src.models.lstm_model import predict as lstm_predict
from src.models.gru_model import predict as gru_predict
from src.models.xgboost_model import predict as xgb_predict
from src.models.garch_model import predict_volatility_band
from src.ensemble.weighted_average import weighted_average

st.set_page_config(page_title="Finansal Zaman Serisi Tahmini", layout="wide")
st.title("📈 Çok Modelli Finansal Fiyat Tahmin Sistemi")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Ayarlar")

asset  = st.sidebar.selectbox("Varlık Seç", list(ASSETS.keys()))
period = st.sidebar.selectbox("Veri Penceresi", ["6m", "1y", "5y"])
n_days = st.sidebar.selectbox("Tahmin Ufku (Gün)", FORECAST_HORIZONS[period])
models = st.sidebar.multiselect(
    "Model Seç",
    ["ARIMA", "SARIMA", "Prophet", "LSTM", "GRU", "XGBoost", "Ensemble"],
    default=["LSTM", "GRU", "XGBoost", "Ensemble"]
)
show_band = st.sidebar.checkbox("GARCH Volatilite Bandı", value=True)

# ── Geçmiş veri ──────────────────────────────────────────────────────────────
st.subheader(f"{asset} — Geçmiş Fiyat Verisi [{period}]")

df = load_processed(asset, period=period)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=df.index, y=df["Close"],
    mode="lines", name="Kapanış Fiyatı", line=dict(color="royalblue")
))
fig_hist.update_layout(xaxis_title="Tarih", yaxis_title="Fiyat", height=350)
st.plotly_chart(fig_hist, use_container_width=True)

# ── Tahminler ────────────────────────────────────────────────────────────────
st.subheader(f"{n_days} Günlük Tahminler")

if st.button("Tahmin Üret"):
    predictions_dict = {}
    future_dates     = pd.bdate_range(start=df.index[-1], periods=n_days + 1)[1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[-60:], y=df["Close"].values[-60:],
        mode="lines", name="Gerçek", line=dict(color="white", width=2)
    ))

    with st.spinner("Tahminler hesaplanıyor..."):

        if "ARIMA" in models:
            preds = arima_predict(asset, period=period, n_days=n_days)
            predictions_dict["ARIMA"] = preds
            fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines", name="ARIMA"))

        if "SARIMA" in models:
            preds = sarima_predict(asset, period=period, n_days=n_days)
            predictions_dict["SARIMA"] = preds
            fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines", name="SARIMA"))

        if "Prophet" in models:
            forecast = prophet_predict(asset, period=period, n_days=n_days)
            preds    = forecast["yhat"].values
            predictions_dict["Prophet"] = preds
            fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines", name="Prophet"))

        if "LSTM" in models:
            preds = lstm_predict(asset, period=period, n_days=n_days)
            predictions_dict["LSTM"] = preds
            fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines", name="LSTM"))

        if "GRU" in models:
            preds = gru_predict(asset, period=period, n_days=n_days)
            predictions_dict["GRU"] = preds
            fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines", name="GRU"))

        if "XGBoost" in models:
            preds = xgb_predict(asset, period=period, n_days=n_days)
            predictions_dict["XGBoost"] = preds
            fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines", name="XGBoost"))

        if "Ensemble" in models and len(predictions_dict) > 0:
            ensemble_preds = weighted_average(predictions_dict)
            predictions_dict["Ensemble"] = ensemble_preds
            fig.add_trace(go.Scatter(
                x=future_dates, y=ensemble_preds,
                mode="lines", name="Ensemble",
                line=dict(color="red", width=2, dash="dash")
            ))

        if show_band and "Ensemble" in predictions_dict:
            upper, lower, _ = predict_volatility_band(asset, predictions_dict["Ensemble"], period=period, n_days=n_days)
            fig.add_trace(go.Scatter(
                x=future_dates, y=upper,
                mode="lines", name="Üst Band",
                line=dict(color="rgba(255,0,0,0.3)")
            ))
            fig.add_trace(go.Scatter(
                x=future_dates, y=lower,
                mode="lines", name="Alt Band",
                fill="tonexty",
                line=dict(color="rgba(255,0,0,0.3)"),
                fillcolor="rgba(255,0,0,0.05)"
            ))

    fig.update_layout(
        xaxis_title="Tarih",
        yaxis_title="Fiyat",
        height=500,
        yaxis=dict(
            range=[
                df["Close"].iloc[-60:].min() * 0.85,
                df["Close"].iloc[-60:].max() * 1.15
            ]
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tahmin Değerleri")
    pred_df = pd.DataFrame(predictions_dict, index=future_dates)
    st.dataframe(pred_df.round(2), use_container_width=True)

    csv = pred_df.to_csv().encode("utf-8")
    st.download_button("Tahminleri CSV İndir", csv, f"{asset}_{period}_{n_days}gun_tahmin.csv", "text/csv")