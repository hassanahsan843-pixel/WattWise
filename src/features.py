"""
features.py - Feature engineering for WattWise AI electricity consumption models.

Constructs temporal, meteorological, autoregressive (lag), rolling statistical,
and household-normalized baseline features required by both Isolation Forest
(anomaly detection) and Random Forest (forecasting).
"""

from typing import List, Tuple
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct rich feature representations from hourly electricity consumption data.

    Features generated per household:
      - hour: Integer [0-23], captures diurnal consumption patterns.
      - dayofweek: Integer [0-6], captures weekly rhythm (Mon=0, Sun=6).
      - is_weekend: Binary flag (1 for Saturday/Sunday, 0 for weekdays).
      - is_holiday: Binary flag (1 for UK bank holidays, 0 otherwise).
      - temperature: Ambient Celsius temperature (strong driver of heating/cooling).
      - rolling_24h_mean: Mean kWh over past 24 hours (smooths daily baseline).
      - rolling_24h_std: Volatility over past 24 hours (distinguishes steady vs noisy loads).
      - rolling_7d_mean: Weekly average consumption level (identifies macro trends).
      - rolling_7d_std: Weekly consumption variance.
      - lag_24h: Energy consumption exactly 24 hours prior (daily seasonality).
      - lag_168h: Energy consumption exactly 168 hours (7 days) prior (weekly seasonality).
      - ratio_to_household_profile: Actual consumption divided by the household's median
        consumption for that exact hour-of-day and day-of-week. A ratio >> 1 indicates abnormal spike;
        a ratio << 1 indicates abnormal drop.

    Args:
        df: Enriched hourly DataFrame containing ['household_id', 'timestamp', 'kwh', 'temperature', 'is_holiday'].

    Returns:
        DataFrame enriched with all engineered features, sorted by household and timestamp.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["household_id", "timestamp"]).reset_index(drop=True)

    # 1. Calendar / Temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = getattr(df["timestamp"].dt, "day_of_week", getattr(df["timestamp"].dt, "dayofweek", 0))
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    if "temperature" not in df.columns:
        df["temperature"] = 12.0

    featured_groups: List[pd.DataFrame] = []

    for _, group in df.groupby("household_id", as_index=False):
        grp = group.copy().sort_values("timestamp")
        grp = grp.set_index("timestamp")

        # Autoregressive lag features (24 hours and 7 days = 168 hours)
        # Using shift(1) at 24h/168h intervals ensures strictly causal (historical) inputs
        grp["lag_24h"] = grp["kwh"].shift(24)
        grp["lag_168h"] = grp["kwh"].shift(168)

        # Rolling statistics over past 24 hours (closed='left' to prevent data leakage)
        # Shift by 1 period so current hour is not included in historical rolling metrics
        shifted_kwh = grp["kwh"].shift(1)
        grp["rolling_24h_mean"] = shifted_kwh.rolling(window=24, min_periods=6).mean()
        grp["rolling_24h_std"] = shifted_kwh.rolling(window=24, min_periods=6).std().fillna(0.0)

        # Rolling statistics over past 7 days (168 hours)
        grp["rolling_7d_mean"] = shifted_kwh.rolling(window=168, min_periods=24).mean()
        grp["rolling_7d_std"] = shifted_kwh.rolling(window=168, min_periods=24).std().fillna(0.0)

        # Reset index to calculate median household profile by (dayofweek, hour)
        grp = grp.reset_index()

        profile = grp.groupby(["dayofweek", "hour"])["kwh"].median().rename("profile_median").reset_index()
        grp = pd.merge(grp, profile, on=["dayofweek", "hour"], how="left")

        # Ratio to normal profile: protect against division by zero with small epsilon (0.01 kWh)
        safe_median = grp["profile_median"].clip(lower=0.02)
        grp["ratio_to_household_profile"] = np.round(grp["kwh"] / safe_median, 4)

        # Impute early lag / rolling values where history is insufficient (backfill with earliest available)
        grp["lag_24h"] = grp["lag_24h"].bfill().fillna(grp["profile_median"])
        grp["lag_168h"] = grp["lag_168h"].bfill().fillna(grp["profile_median"])
        grp["rolling_24h_mean"] = grp["rolling_24h_mean"].bfill().fillna(grp["profile_median"])
        grp["rolling_24h_std"] = grp["rolling_24h_std"].bfill().fillna(0.05)
        grp["rolling_7d_mean"] = grp["rolling_7d_mean"].bfill().fillna(grp["rolling_24h_mean"])
        grp["rolling_7d_std"] = grp["rolling_7d_std"].bfill().fillna(grp["rolling_24h_std"])

        featured_groups.append(grp)

    featured_df = pd.concat(featured_groups, ignore_index=True)
    return featured_df


def get_feature_columns() -> List[str]:
    """List of numeric features fed into the Isolation Forest anomaly detector."""
    return [
        "hour",
        "dayofweek",
        "is_weekend",
        "is_holiday",
        "temperature",
        "rolling_24h_mean",
        "rolling_24h_std",
        "rolling_7d_mean",
        "rolling_7d_std",
        "lag_24h",
        "lag_168h",
        "ratio_to_household_profile",
    ]
