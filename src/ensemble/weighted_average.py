import numpy as np
from src.data.fetch_data import ASSETS
from src.evaluation.metrics import evaluate

FORECAST_HORIZONS = [30, 60, 90]


def inverse_rmse_weights(results_list):
    """
    Her modelin RMSE'sine göre ağırlık üretir.
    Düşük RMSE → yüksek ağırlık
    """
    weights = {r["Model"]: 1 / r["RMSE"] for r in results_list}
    total   = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def weighted_average(predictions_dict, weights=None):
    """
    predictions_dict: {"ARIMA": array, "SARIMA": array, "Prophet": array, "LSTM": array, "TFT": array}
    weights: {"ARIMA": 0.2, ...} — None ise eşit ağırlık
    """
    models = list(predictions_dict.keys())
    preds  = list(predictions_dict.values())

    if weights is None:
        w = np.ones(len(models)) / len(models)
    else:
        w = np.array([weights[m] for m in models])
        w = w / w.sum()

    ensemble = np.zeros(len(preds[0]))
    for i, pred in enumerate(preds):
        ensemble += w[i] * np.array(pred)

    return ensemble


def run_ensemble(asset_name, test_close, predictions_dict, results_list):
    """
    Hem eşit ağırlıklı hem RMSE ağırlıklı ensemble üretir ve karşılaştırır.
    """
    print(f"\n── {asset_name} Ensemble ──")

    # Eşit ağırlıklı
    equal_preds   = weighted_average(predictions_dict)
    equal_results = evaluate(test_close, equal_preds, model_name="Ensemble (Eşit)")

    # RMSE ağırlıklı
    weights       = inverse_rmse_weights(results_list)
    rmse_preds    = weighted_average(predictions_dict, weights=weights)
    rmse_results  = evaluate(test_close, rmse_preds, model_name="Ensemble (RMSE)")

    print(f"  Ağırlıklar: {weights}")

    return {
        "equal" : {"preds": equal_preds, "results": equal_results},
        "rmse"  : {"preds": rmse_preds,  "results": rmse_results},
    }


if __name__ == "__main__":
    import numpy as np

    # Örnek: sahte tahminlerle test
    asset = "BTC"
    n     = 100
    actual = np.random.rand(n) * 1000 + 30000

    predictions_dict = {
        "ARIMA"  : actual + np.random.randn(n) * 100,
        "SARIMA" : actual + np.random.randn(n) * 120,
        "Prophet": actual + np.random.randn(n) * 90,
        "LSTM"   : actual + np.random.randn(n) * 80,
        "TFT"    : actual + np.random.randn(n) * 70,
    }

    results_list = [
        {"Model": "ARIMA",   "RMSE": 100, "MAE": 80, "MAPE": 0.3},
        {"Model": "SARIMA",  "RMSE": 120, "MAE": 95, "MAPE": 0.4},
        {"Model": "Prophet", "RMSE": 90,  "MAE": 72, "MAPE": 0.28},
        {"Model": "LSTM",    "RMSE": 80,  "MAE": 65, "MAPE": 0.25},
        {"Model": "TFT",     "RMSE": 70,  "MAE": 58, "MAPE": 0.22},
    ]

    results = run_ensemble(asset, actual, predictions_dict, results_list)