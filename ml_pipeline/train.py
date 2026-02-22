import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from database import load_all_data
from config import MODEL_SAVE_PATH

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def safe_filename(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_")


def calculate_mape(actual, predicted):
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


# --------------------------------------------------
# Model Training
# --------------------------------------------------

def train_model_for_a_ward(hospital, ward, ts, test_days=28):
    """
    Train + evaluate + persist model for one ward
    Using datetime-based train/test split
    """

    ts = ts.asfreq("D").ffill().bfill()

    if len(ts) < test_days + 30:  # ensure minimum data
        print(f"Skipping {hospital} | {ward} — not enough data")
        return None

    # -------------------------
    # Split by datetime
    # -------------------------
    train_cutoff = ts.index.max() - pd.Timedelta(days=test_days)
    train = ts[ts.index <= train_cutoff]
    test = ts[ts.index > train_cutoff]

    if len(train) < 30:
        print(f"Skipping {hospital} | {ward} — training set too small")
        return None

    # -------------------------
    # Train SARIMA
    # -------------------------
    model = SARIMAX(
        train,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False, maxiter=200)

    # -------------------------
    # Evaluate
    # -------------------------
    forecast_result = results.get_forecast(steps=len(test))
    preds = forecast_result.predicted_mean

    mae = mean_absolute_error(test, preds)
    mape = calculate_mape(test.values, preds.values)

    # -------------------------
    # Metadata
    # -------------------------
    metadata = {
        "hospital": hospital,
        "ward": ward,
        "trained_at": datetime.utcnow(),
        "aic": results.aic,
        "mae": float(mae),
        "mape": float(mape),
        "train_size": len(train),
        "test_size": len(test),
    }

    # -------------------------
    # Save model
    # -------------------------
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model_path = get_model_path(hospital, ward)
    with open(model_path, "wb") as f:
        joblib.dump({"model": results, "metadata": metadata}, f)

    print(f"Saved model → {model_path}")
    return metadata



# --------------------------------------------------
# Retrain Logic
# --------------------------------------------------

def should_retrain_models(model_path, max_age_days=14, mape_threshold=50):
    if not os.path.exists(model_path):
        return True, "model_not_found"

    saved = joblib.load(model_path)
    metadata = saved.get("metadata", {})

    trained_at = metadata.get("trained_at")
    mape = metadata.get("mape")

    if trained_at is None or mape is None:
        return True, "missing_metadata"

    age_days = (datetime.utcnow() - trained_at).days

    if age_days >= max_age_days:
        return True, "model_too_old"

    if mape >= mape_threshold:
        return True, "performance_degraded"

    return False, "model_healthy"


# --------------------------------------------------
# Train All Models
# --------------------------------------------------

def train_all_models(test_days=28, max_age_days=14, force_retrain=False):
    df = load_all_data()
    training_log = []

    for hospital in df["hospital"].unique():
        df_hospital = df[df["hospital"] == hospital]

        for ward in df_hospital["ward"].unique():
            print(f"\nProcessing: {hospital} | {ward}")

            ward_df = df_hospital[df_hospital["ward"] == ward].copy()
            ward_df = ward_df.sort_values("datetime")

            ts = ward_df.set_index("datetime")["occupied_beds"].asfreq("D").ffill().bfill()

            # ---------- Guard for minimal data ----------
            if len(ts) < test_days + 30:
                print("Not enough data — skipping")
                continue

            model_path = get_model_path(hospital, ward)

            retrain, reason = should_retrain_models(model_path, max_age_days=max_age_days)
            if force_retrain:
                retrain = True
                reason = "forced"

            if retrain:
                metadata = train_model_for_a_ward(hospital, ward, ts, test_days)
                if metadata:
                    metadata["retrain_reason"] = reason
                    training_log.append(metadata)
            else:
                print("Model healthy — skipping")

    return pd.DataFrame(training_log)
