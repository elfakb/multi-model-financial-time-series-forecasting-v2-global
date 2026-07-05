import pandas as pd
import matplotlib.pyplot as plt
import os

FIGURES_DIR      = "reports/figures"
COMPARISONS_DIR  = "reports/model_comparisons"


def compare(results_list, asset_name):
    """
    results_list: her modelin evaluate() çıktısı olan dict'lerin listesi
    """
    os.makedirs(COMPARISONS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.DataFrame(results_list)
    df = df.sort_values("RMSE").reset_index(drop=True)

    print(f"\n── {asset_name} Model Karşılaştırması ──")
    print(df.to_string(index=False))

    df.to_csv(f"{COMPARISONS_DIR}/{asset_name}_comparison.csv", index=False)

    return df


def plot_predictions(dates, actual, predictions_dict, asset_name):
    """
    predictions_dict: {"ARIMA": [...], "LSTM": [...], ...}
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual, label="Gerçek", color="black", linewidth=2)

    for model_name, preds in predictions_dict.items():
        plt.plot(dates, preds, label=model_name, linewidth=1.5)

    plt.title(f"{asset_name} - Model Tahmin Karşılaştırması")
    plt.xlabel("Tarih")
    plt.ylabel("Fiyat")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{asset_name}_comparison.png", dpi=150)
    plt.close()
    print(f"  Grafik kaydedildi: {FIGURES_DIR}/{asset_name}_comparison.png")


def plot_metrics(df, asset_name):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, metric in zip(axes, ["RMSE", "MAE", "MAPE"]):
        ax.bar(df["Model"], df[metric], color="steelblue")
        ax.set_title(metric)
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"{asset_name} - Metrik Karşılaştırması")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{asset_name}_metrics.png", dpi=150)
    plt.close()
    print(f"  Grafik kaydedildi: {FIGURES_DIR}/{asset_name}_metrics.png")