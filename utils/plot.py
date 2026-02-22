# plot.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import math

# -----------------------------
# Ward Plot Function
# -----------------------------
def plot_ward(ax, ward_name, forecast_df, daily_beds, metrics=None, history_days=60):
    """
    Plot historical occupancy + forecast + CI for one ward
    forecast_df: DataFrame with columns: 'date', 'forecast', 'lower_ci', 'upper_ci'
    daily_beds: historical DataFrame with 'datetime', 'ward', 'occupied_beds'
    """
    # Historical data
    ward_hist = daily_beds[daily_beds["ward"] == ward_name].copy()
    ward_hist["datetime"] = pd.to_datetime(ward_hist["datetime"])
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    ward_hist = ward_hist[
        ward_hist["datetime"] > ward_hist["datetime"].max() - pd.Timedelta(days=history_days)
    ]

    if ward_hist.empty:
        return  # nothing to plot

    last_date = ward_hist["datetime"].iloc[-1]
    last_value = ward_hist["occupied_beds"].iloc[-1]

    # Historical
    ax.plot(
        ward_hist["datetime"],
        ward_hist["occupied_beds"],
        label="Historical",
        linewidth=2
    )

    # Forecast
    ax.plot(
        [last_date] + list(forecast_df["date"]),
        [last_value] + list(forecast_df["forecast"]),
        "--",
        linewidth=2,
        label="Forecast"
    )

    # Confidence interval
    if "lower_ci" in forecast_df.columns and "upper_ci" in forecast_df.columns:
        ax.fill_between(
            forecast_df["date"],
            forecast_df["lower_ci"],
            forecast_df["upper_ci"],
            alpha=0.2,
            label="95% CI"
        )

    # Metrics box
    if metrics:
        ax.text(
            0.02, 0.98,
            f"MAPE: {metrics.get('mape', 0):.1f}% | MAE: {metrics.get('mae', 0):.1f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.8)
        )

    # Forecast start marker
    ax.axvline(last_date, linestyle=":", alpha=0.7)

    ax.set_title(ward_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Occupied Beds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=45)

# -----------------------------
# Main Plot Function
# -----------------------------
def plot_hospital_forecasts(forecasts, daily_beds, single_ward=None, single_forecast_df=None):
    """
    Plot forecasts for one or multiple wards.

    Parameters:
    - forecasts: dict of {ward_name: forecast_df} (for multiple wards)
    - daily_beds: historical bed occupancy DataFrame
    - single_ward: optional, ward name if plotting only one ward
    - single_forecast_df: optional, forecast DataFrame if plotting only one ward
    """

    # Prepare dictionary of ward forecasts
    if single_ward and single_forecast_df is not None:
        forecasts = {single_ward: single_forecast_df}

    if not forecasts:
        print("No forecasts to plot.")
        return

    n = len(forecasts)
    rows = max(1, math.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(15, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (ward, forecast_df) in enumerate(forecasts.items()):
        plot_ward(
            ax=axes[i],
            ward_name=ward,
            forecast_df=forecast_df,
            daily_beds=daily_beds,
            metrics={"mape": forecast_df.get("mape", [None])[0],
                     "mae": forecast_df.get("mae", [None])[0]}
        )

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(
        f"Bed Occupancy Forecasts",
        fontsize=15,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()
    plt.show()
