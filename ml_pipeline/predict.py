import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from config import MODEL_SAVE_PATH, FORECAST_START_DATE
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Helpers
# -----------------------------
def safe_filename(text):
    """Sanitize strings for filenames"""
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_")

def calculate_mape(actual, predicted):
    """Calculate MAPE safely"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    ape = np.abs((actual[mask] - predicted[mask]) / actual[mask])
    return np.mean(ape) * 100

def get_model_path(hospital, ward):
    safe_hospital = safe_filename(hospital)
    safe_ward = safe_filename(ward)
    filename = f"{safe_hospital}_{safe_ward}_model.pkl"
    return os.path.join(MODEL_SAVE_PATH, filename)

# -----------------------------
# Load trained model
# -----------------------------
def load_trained_model(hospital, ward):
    path = get_model_path(hospital, ward)
    if not os.path.exists(path):
        print(f"No model found for {path}")
        return None, None
    try:
        saved = joblib.load(path)
        model = saved.get("model")
        metadata = saved.get("metadata", {})
        return model, metadata
    except Exception as e:
        print(f"Failed to load model for {path}: {e}")
        return None, None

# -----------------------------
# Generate forecast
# -----------------------------
def generate_forecast(hospital, ward, ts, forecast_days=14):
    """
    Generate forecast with metrics from trained SARIMAX
    Returns: forecast_df, mape, mae
    """

    # Minimum data check
    if len(ts) < 30:
        print(f"Not enough data to forecast {hospital} | {ward}")
        empty_df = pd.DataFrame({"date": [], "forecast": [], "lower_ci": [], "upper_ci": []})
        return empty_df, np.nan, np.nan

    ts = ts.asfreq("D").ffill().bfill()
    model, metadata = load_trained_model(hospital, ward)

    if model is None:
        print(f"No trained model found for {hospital} | {ward}")
        empty_df = pd.DataFrame({"date": [], "forecast": [], "lower_ci": [], "upper_ci": []})
        return empty_df, np.nan, np.nan

    try:
        # Use model to forecast
        forecast_result = model.get_forecast(steps=forecast_days)
        forecast_df = forecast_result.summary_frame().reset_index().rename(columns={
            "mean": "forecast",
            "mean_ci_lower": "lower_ci",
            "mean_ci_upper": "upper_ci",
            "index": "date"
        })

        # Ensure only needed columns
        forecast_df = forecast_df[["date", "forecast", "lower_ci", "upper_ci"]]

        # Load metrics from metadata if available
        mape = metadata.get("mape", np.nan)
        mae = metadata.get("mae", np.nan)

        # Calculate fallback metrics if not available
        if np.isnan(mape) or np.isnan(mae):
            # Use last test_days for fallback
            test_days = min(len(ts), forecast_days)
            test = ts[-test_days:]
            preds = forecast_df["forecast"][:test_days]
            if len(preds) == len(test):
                mae = mean_absolute_error(test.values, preds.values)
                mape = calculate_mape(test.values, preds.values)

        return forecast_df, mape, mae

    except Exception as e:
        print(f"Forecast failed for {hospital} | {ward}: {e}")
        empty_df = pd.DataFrame({"date": [], "forecast": [], "lower_ci": [], "upper_ci": []})
        return empty_df, np.nan, np.nan
