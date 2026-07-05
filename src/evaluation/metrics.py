import numpy as np


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def evaluate(actual, predicted, model_name="Model"):
    results = {
        "Model": model_name,
        "RMSE" : round(rmse(actual, predicted), 4),
        "MAE"  : round(mae(actual, predicted), 4),
        "MAPE" : round(mape(actual, predicted), 4),
    }
    print(f"{model_name} → RMSE: {results['RMSE']}  MAE: {results['MAE']}  MAPE: {results['MAPE']}%")
    return results