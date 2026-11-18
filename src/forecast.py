"""
forecast.py - Multi-step electricity consumption forecasting using Random Forest
with uncertainty intervals from tree variance and baseline comparison.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


def make_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct predictive lag, calendar, and weather features for energy forecasting.

    Features generated:
      - lag_24h: Consumption 24 hours prior (daily cycle).
      - lag_168h: Consumption 168 hours prior (weekly cycle - same hour last week).
      - rolling_24h_mean: Moving average of prior 24 hours.
      - rolling_168h_mean: Moving average of prior 7 days.
      - hour: Hour of day [0-23].
      - dayofweek: Day of week [0-6].
      - month: Month of year [1-12].
      - is_holiday: UK bank holiday flag [0 or 1].
      - temperature: Outdoor temperature in Celsius.

    Args:
        df: Enriched DataFrame with ['timestamp', 'kwh', 'temperature', 'is_holiday'].

    Returns:
        DataFrame containing forecast features and target 'kwh'.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = getattr(df["timestamp"].dt, "day_of_week", getattr(df["timestamp"].dt, "dayofweek", 0))
    df["month"] = df["timestamp"].dt.month

    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0
    if "temperature" not in df.columns:
        df["temperature"] = 12.0

    # Lags (strictly historical to prevent leakage)
    df["lag_24h"] = df["kwh"].shift(24)
    df["lag_168h"] = df["kwh"].shift(168)

    # Rolling means of prior hours
    shifted = df["kwh"].shift(1)
    df["rolling_24h_mean"] = shifted.rolling(window=24, min_periods=6).mean()
    df["rolling_168h_mean"] = shifted.rolling(window=168, min_periods=24).mean()

    # Clean initial missing values from lag creation
    df["lag_24h"] = df["lag_24h"].bfill().fillna(df["kwh"].mean())
    df["lag_168h"] = df["lag_168h"].bfill().fillna(df["kwh"].mean())
    df["rolling_24h_mean"] = df["rolling_24h_mean"].bfill().fillna(df["kwh"].mean())
    df["rolling_168h_mean"] = df["rolling_168h_mean"].bfill().fillna(df["kwh"].mean())

    return df


def get_forecast_feature_cols() -> List[str]:
    """Feature columns consumed by the Random Forest forecasting model."""
    return [
        "lag_24h",
        "lag_168h",
        "rolling_24h_mean",
        "rolling_168h_mean",
        "hour",
        "dayofweek",
        "month",
        "is_holiday",
        "temperature",
    ]


def train_rf(
    train_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = "models/rf_forecast.joblib",
    n_estimators: int = 120,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor to forecast hourly electricity consumption.

    Random Forest is chosen because:
      1. It effectively models complex non-linear interactions between hour, weekday, and temperature.
      2. It requires no feature scaling and is robust to remaining noise or extreme spikes.
      3. The ensemble of independent trees naturally yields predictive variance, providing an empirical
         uncertainty confidence interval without external parametric assumptions.

    Args:
        train_df: Prepared training DataFrame containing forecast features and target 'kwh'.
        save_path: Location to persist the trained model.
        n_estimators: Number of trees in the forest.
        random_state: Seed for reproducibility.

    Returns:
        Fitted RandomForestRegressor model.
    """
    feature_cols = get_forecast_feature_cols()
    X_train = train_df[feature_cols].copy().fillna(0.0)
    y_train = train_df["kwh"].values

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_p)

    return model


def predict(
    model: RandomForestRegressor,
    history: pd.DataFrame,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """
    Generate multi-step hourly consumption forecast over the requested horizon.
    Computes lower (10th percentile) and upper (90th percentile) confidence bounds
    from the individual prediction spread across all decision trees in the forest.

    Args:
        model: Trained RandomForestRegressor.
        history: Recent consumption history (at least 7-14 days required to seed lags).
        horizon_days: Forecast horizon in days (e.g. 7, 14, 30 days).

    Returns:
        DataFrame with ['timestamp', 'predicted_kwh', 'lower_bound', 'upper_bound'].
    """
    feature_cols = get_forecast_feature_cols()
    hist_clean = history.sort_values("timestamp").copy()
    last_timestamp = pd.to_datetime(hist_clean["timestamp"].iloc[-1])

    # Generate future hourly timestamps
    total_hours = horizon_days * 24
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=total_hours,
        freq="1h",
    )

    # Initialize simulation series with recent kwh history
    # We maintain a working buffer of kwh values to recursively generate autoregressive lags
    kwh_buffer = list(hist_clean["kwh"].values)
    temp_profile = hist_clean["temperature"].mean() if "temperature" in hist_clean else 12.0

    forecast_records = []

    for ts in future_timestamps:
        hour = ts.hour
        dayofweek = getattr(ts, "day_of_week", getattr(ts, "dayofweek", 0))
        month = ts.month
        is_holiday = 1 if (dayofweek >= 5) else 0  # simplified or UK holiday check

        # Seasonal temperature fluctuation
        day_of_yr = getattr(ts, "day_of_year", getattr(ts, "dayofyear", 1))
        temp_val = 12.0 - 7.0 * np.cos(2 * np.pi * (day_of_yr - 15) / 365.25) + 2.5 * np.sin(2 * np.pi * (hour - 8) / 24.0)

        # Autoregressive lags from the buffer
        lag_24h = kwh_buffer[-24] if len(kwh_buffer) >= 24 else kwh_buffer[-1]
        lag_168h = kwh_buffer[-168] if len(kwh_buffer) >= 168 else lag_24h
        rolling_24 = np.mean(kwh_buffer[-24:]) if len(kwh_buffer) >= 24 else np.mean(kwh_buffer)
        rolling_168 = np.mean(kwh_buffer[-168:]) if len(kwh_buffer) >= 168 else rolling_24

        x_row = np.array([[
            lag_24h,
            lag_168h,
            rolling_24,
            rolling_168,
            hour,
            dayofweek,
            month,
            is_holiday,
            temp_val,
        ]])

        # Extract predictions from every individual tree to quantify ensemble uncertainty
        tree_preds = np.array([tree.predict(x_row)[0] for tree in model.estimators_])

        mean_pred = float(np.mean(tree_preds))
        lower_pred = float(np.percentile(tree_preds, 10))
        upper_pred = float(np.percentile(tree_preds, 90))

        # Ensure energy is non-negative
        mean_pred = max(0.01, mean_pred)
        lower_pred = max(0.005, lower_pred)
        upper_pred = max(mean_pred, upper_pred)

        # Append predicted value to rolling buffer for subsequent autoregressive steps
        kwh_buffer.append(mean_pred)

        forecast_records.append({
            "timestamp": ts,
            "predicted_kwh": round(mean_pred, 4),
            "lower_bound": round(lower_pred, 4),
            "upper_bound": round(upper_pred, 4),
        })

    return pd.DataFrame(forecast_records)


def naive_baseline_forecast(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """
    Standard Naive Baseline: 'Same day last week' (lag 168h).
    For any hour t in the test set, predicted consumption is consumption at t - 168 hours.
    """
    combined = pd.concat([history_df, test_df], ignore_index=True).sort_values("timestamp")
    combined["naive_pred"] = combined["kwh"].shift(168).bfill()
    test_preds = combined.loc[combined["timestamp"].isin(test_df["timestamp"]), "naive_pred"]
    return test_preds.values


def evaluate_forecasting(
    hourly_df: pd.DataFrame,
    test_days: int = 30,
    reports_dir: Union[str, Path] = "reports",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate Random Forest Forecaster against the Naive Baseline (Same Day Last Week)
    using a strict chronological time-based split (no random shuffling).

    Computes:
      - MAE (Mean Absolute Error, in kWh)
      - RMSE (Root Mean Squared Error, in kWh)
      - MAPE (Mean Absolute Percentage Error, in %)
    """
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    (rep_path / "figures").mkdir(parents=True, exist_ok=True)

    # Use single representative household or aggregate for benchmark evaluation
    h_id = hourly_df["household_id"].iloc[0]
    h_df = hourly_df[hourly_df["household_id"] == h_id].sort_values("timestamp").reset_index(drop=True)

    # Time-based split: last `test_days` for evaluation
    split_cutoff = h_df["timestamp"].max() - pd.Timedelta(days=test_days)
    train_raw = h_df[h_df["timestamp"] < split_cutoff].copy()
    test_raw = h_df[h_df["timestamp"] >= split_cutoff].copy()

    # Feature engineering
    full_featured = make_forecast_features(h_df)
    train_feat = full_featured[full_featured["timestamp"] < split_cutoff]
    test_feat = full_featured[full_featured["timestamp"] >= split_cutoff]

    # Train Random Forest
    rf_model = train_rf(train_feat, save_path="models/rf_forecast.joblib")

    # Predict on test set
    feature_cols = get_forecast_feature_cols()
    X_test = test_feat[feature_cols].copy().fillna(0.0)
    rf_preds = rf_model.predict(X_test)
    y_true = test_feat["kwh"].values

    # Naive Baseline: same hour last week
    naive_preds = test_feat["lag_168h"].values

    # Metrics
    mae_rf = mean_absolute_error(y_true, rf_preds)
    rmse_rf = np.sqrt(mean_squared_error(y_true, rf_preds))
    mape_rf = mean_absolute_percentage_error(y_true, np.clip(rf_preds, 0.01, None)) * 100

    mae_naive = mean_absolute_error(y_true, naive_preds)
    rmse_naive = np.sqrt(mean_squared_error(y_true, naive_preds))
    mape_naive = mean_absolute_percentage_error(y_true, np.clip(naive_preds, 0.01, None)) * 100

    metrics_df = pd.DataFrame({
        "Model": ["Random Forest Forecaster", "Naive Baseline (Same Day Last Week)"],
        "MAE (kWh)": [round(mae_rf, 4), round(mae_naive, 4)],
        "RMSE (kWh)": [round(rmse_rf, 4), round(rmse_naive, 4)],
        "MAPE (%)": [round(mape_rf, 2), round(mape_naive, 2)],
    })

    metrics_df.to_csv(rep_path / "forecast_metrics.csv", index=False)

    # Visualization: Actual vs Predicted over a sample week
    sample_window = 168  # 1 week of hours
    plt.figure(figsize=(14, 5))
    ts_slice = test_feat["timestamp"].iloc[:sample_window]
    plt.plot(ts_slice, y_true[:sample_window], label="Actual Consumption", color="#2ca02c", lw=2)
    plt.plot(ts_slice, rf_preds[:sample_window], label="Random Forest Forecast", color="#1f77b4", linestyle="--", lw=1.8)
    plt.plot(ts_slice, naive_preds[:sample_window], label="Naive Baseline (Last Week)", color="#7f7f7f", linestyle=":", alpha=0.8)

    plt.title(f"WattWise AI: Electricity Forecast vs Actuals (Household {h_id})", fontsize=13, fontweight="bold")
    plt.xlabel("Timestamp", fontsize=11)
    plt.ylabel("Electricity Usage (kWh)", fontsize=11)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(rep_path / "figures" / "forecast_actual_vs_predicted.png", dpi=200)
    plt.close()

    return metrics_df, {
        "train_size": len(train_raw),
        "test_size": len(test_raw),
        "household_evaluated": h_id,
    }
