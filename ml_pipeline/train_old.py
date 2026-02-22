import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from database import load_all_data

import warnings
warnings.filterwarnings('ignore')

from config import MODEL_SAVE_PATH

def safe_filename(text):
    """convert text to safe filename (replace spaces with underscores)"""
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_")

def calculate_mape(actual, predicted):
    """Calculate MAPE"""
    mask = actual != 0
    if mask.sum() == 0:
        return np.sum
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    ape = np.abs((actual_filtered - predicted_filtered) / actual_filtered)
    return np.mean(ape) * 100

def train_model_for_ward(hospital, ward, ts):
    """train SARIMAX model for a ward"""
    model = SARIMAX(
        ts,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,         
        enforce_invertibility=False         
    )

    return model.fit(disp=False, maxiter = 100)



def forecast_sarima(results, ts, forecast_days):
    forecast = results.get_forecast(steps=14)
    forecast_df = forecast.summary_frame()

    forecast_df.index = pd.date_range(
        start=ts.index.max() + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    forecast_df = forecast_df.rename(columns={
        "mean": "forecast_occupied_beds",
        "mean_ci_lower": "lower_ci",
        "mean_ci_upper": "upper_ci"
    })

    return forecast_df.reset_index().rename(columns={"index": "date"})


def evaluate_sarima(ts, results, test_days):
    train = ts[:-test_days]
    test = ts[-test_days:]

    forecast = results.get_forecast(steps=test_days)
    pred = forecast.predicted_mean

    mae = mean_absolute_error(test, pred)
    mape = np.mean(np.abs((test - pred) / test)) * 100

    return {
        "test_mae": mae,
        "test_mape": mape,
        "train_size": len(train),
        "test_size": len(test),
        "aic": results.aic
    }


    
