import pandas as pd
import streamlit as st
from ml_pipeline.train import train_all_models
from ml_pipeline.predict import generate_forecast
from database import load_all_data, get_hospitals, get_hospital_wards
from utils.plot import plot_hospital_forecasts, plot_ward

st.set_page_config(page_title="SARIMA Bed Forecast", layout="wide")
st.title("🏥 Horizon Health Bed Forecast Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Model Training", "Forecast"])

# -------------------------
# Section 1: Model Training
# -------------------------
if section == "Model Training":
    st.header("Model Training")
    force_retrain = st.checkbox("Force retrain models")
    if st.button("Train / Refresh Models"):
        with st.spinner("Training models..."):
            result_df = train_all_models(force_retrain=force_retrain)
        st.success(f"Trained {len(result_df)} models")
        st.dataframe(result_df)

# -------------------------
# Section 2: Forecast
# -------------------------
elif section == "Forecast":
    st.header("Generate Forecast")
    hospitals = get_hospitals()
    hospital = st.selectbox("Select Hospital", hospitals)
    wards = get_hospital_wards(hospital)
    ward = st.selectbox("Select Ward", wards)

    forecast_days = st.number_input("Forecast Days", min_value=1, max_value=60, value=14)

    if st.button("Generate Forecast"):
        df = load_all_data()
        ward_df = df[(df["hospital"] == hospital) & (df["ward"] == ward)].copy()
        if ward_df.empty:
            st.error("Ward not found")
            st.stop()

        ts = ward_df.sort_values("datetime").set_index("datetime")["occupied_beds"].asfreq("D").ffill().bfill()
        with st.spinner("Generating forecast..."):
            forecast_df, mape, mae = generate_forecast(hospital, ward, ts, forecast_days)

        if forecast_df is None:
            st.error("Model not found — train first")
            st.stop()

        col1, col2 = st.columns(2)
        col1.metric("MAPE", f"{mape:.2f}%")
        col2.metric("MAE", f"{mae:.2f} beds")

        st.subheader("Forecast Table")
        st.dataframe(forecast_df)

        st.subheader("Forecast Chart")
        st.line_chart(forecast_df["forecast"])

