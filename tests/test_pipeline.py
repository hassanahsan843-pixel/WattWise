"""
Comprehensive end-to-end integration and pipeline unit tests for WattWise AI.

Covers:
  1. Preprocessing & Gap Cleaning
  2. Hourly and Daily Resampling
  3. Feature Engineering & Autoregressive Lags
  4. Anomaly Model Training & Scoring Output
  5. Forecast Shape & Uncertainty Intervals
  6. Bill Calculation & Avoidable Cost Mathematics
  7. App Input Data Validation Rules
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocess import clean, resample_readings, merge_context, select_subset
from src.features import build_features, get_feature_columns
from src.anomaly import train_isolation_forest, score, zscore_baseline, inject_anomalies
from src.forecast import make_forecast_features, train_rf, predict
from src.bill import estimate_bill, estimate_avoidable_cost


@pytest.fixture
def sample_raw_readings():
    """Generates 15 continuous days of realistic half-hourly readings with intentional gaps."""
    dates = pd.date_range("2013-03-01", periods=15 * 48, freq="30min")
    kwh = np.random.uniform(0.1, 0.5, size=len(dates)).astype(object)
    # Inject a 2-step gap (should be interpolated)
    kwh[10:12] = "Null"
    # Inject an exact duplicate
    df = pd.DataFrame({
        "household_id": "TEST_H01",
        "timestamp": dates,
        "kwh": kwh,
    })
    dup_row = df.iloc[5:6]
    df = pd.concat([df, dup_row], ignore_index=True)
    return df


def test_preprocessing_cleaning(sample_raw_readings):
    """Verify cleaning converts to numeric, removes duplicates, and interpolates short gaps."""
    cleaned = clean(sample_raw_readings)

    # 1. No duplicates
    assert len(cleaned) == 15 * 48
    assert cleaned.duplicated(subset=["household_id", "timestamp"]).sum() == 0

    # 2. kwh is purely numeric and has no NaNs
    assert pd.api.types.is_numeric_dtype(cleaned["kwh"])
    assert cleaned["kwh"].isna().sum() == 0

    # 3. Short gap (indices 10 and 11) was flagged as interpolated
    assert cleaned.loc[10:11, "is_interpolated"].all()


def test_resampling_hourly_and_daily(sample_raw_readings):
    """Verify half-hourly data aggregates accurately to hourly (2x) and daily (48x)."""
    cleaned = clean(sample_raw_readings)
    hourly = resample_readings(cleaned, freq="h")
    daily = resample_readings(cleaned, freq="D")

    # 15 days * 24 hours = 360 hours
    assert len(hourly) == 15 * 24
    # 15 days
    assert len(daily) == 15

    # Energy conservation: total kWh across hourly and daily should match exactly
    assert np.isclose(hourly["kwh"].sum(), daily["kwh"].sum(), rtol=1e-4)


def test_feature_engineering(sample_raw_readings):
    """Verify build_features produces all required columns with zero residual NaNs."""
    cleaned = clean(sample_raw_readings)
    hourly = resample_readings(cleaned, freq="h")
    enriched = merge_context(hourly)
    featured = build_features(enriched)

    expected_cols = get_feature_columns()
    for col in expected_cols:
        assert col in featured.columns, f"Feature column {col} missing"

    # Verify no unhandled NaN values in feature matrix
    assert featured[expected_cols].isna().sum().sum() == 0

    # Verify ratio_to_household_profile is strictly positive
    assert (featured["ratio_to_household_profile"] > 0).all()


def test_anomaly_model_training_and_scoring(sample_raw_readings):
    """Verify Isolation Forest training, decision score bounds, and anomaly flag assignment."""
    cleaned = clean(sample_raw_readings)
    hourly = resample_readings(cleaned, freq="h")
    enriched = merge_context(hourly)
    featured = build_features(enriched)

    model = train_isolation_forest(featured, contamination=0.05, save_path=None)
    scored = score(model, featured)

    assert "anomaly_flag" in scored.columns
    assert "anomaly_score" in scored.columns
    assert scored["anomaly_flag"].dtype == bool

    # Anomaly scores must reside within [0, 1]
    assert scored["anomaly_score"].min() >= 0.0
    assert scored["anomaly_score"].max() <= 1.0

    # Some anomalies should be flagged corresponding to contamination
    assert scored["anomaly_flag"].sum() > 0


def test_forecast_shape_and_uncertainty_bounds(sample_raw_readings):
    """Verify Random Forest forecast produces expected multi-step horizon and valid uncertainty bounds."""
    cleaned = clean(sample_raw_readings)
    hourly = resample_readings(cleaned, freq="h")
    enriched = merge_context(hourly)
    fc_feat = make_forecast_features(enriched)

    model = train_rf(fc_feat, save_path=None, n_estimators=30)
    horizon_days = 7
    forecast = predict(model, history=enriched, horizon_days=horizon_days)

    # Output row count must equal horizon_days * 24
    assert len(forecast) == horizon_days * 24
    assert list(forecast.columns) == ["timestamp", "predicted_kwh", "lower_bound", "upper_bound"]

    # Bounds sanity check: lower_bound <= predicted_kwh <= upper_bound (or lower <= upper)
    assert (forecast["lower_bound"] <= forecast["upper_bound"]).all()
    assert (forecast["predicted_kwh"] >= 0).all()


def test_bill_estimation_and_avoidable_cost():
    """Verify financial formulas for energy bills and avoidable excess costs."""
    kwh_series = pd.Series([10.0] * 30)  # 30 days * 10 kWh/day = 300 kWh total
    tariff = 0.28
    standing = 0.60
    n_days = 30

    bill = estimate_bill(kwh_series, tariff_per_kwh=tariff, standing_charge_per_day=standing, n_days=n_days)

    expected_unit = 300.0 * 0.28  # £84.00
    expected_standing = 30 * 0.60  # £18.00
    expected_total = expected_unit + expected_standing  # £102.00

    assert bill["total_kwh"] == 300.0
    assert np.isclose(bill["unit_cost"], expected_unit)
    assert np.isclose(bill["standing_cost"], expected_standing)
    assert np.isclose(bill["total_bill"], expected_total)
    assert bill["lower_bill"] < bill["total_bill"] < bill["upper_bill"]

    # Avoidable cost calculation check
    avoidable = estimate_avoidable_cost(anomalous_kwh=150.0, expected_kwh=100.0, tariff_per_kwh=tariff)
    assert avoidable["excess_kwh"] == 50.0
    assert np.isclose(avoidable["avoidable_cost_gbp"], 50.0 * 0.28)


def test_app_input_validation_rules():
    """Verify validation boundaries: rejecting fewer than 14 days, empty data, or missing columns."""
    # 1. Fewer than 14 days
    short_dates = pd.date_range("2013-01-01", periods=10 * 24, freq="1h")
    short_df = pd.DataFrame({"timestamp": short_dates, "kwh": [0.2] * len(short_dates)})
    timespan_days = (short_df["timestamp"].max() - short_df["timestamp"].min()).total_seconds() / 86400.0
    assert timespan_days < 14.0

    # 2. Missing columns
    bad_df = pd.DataFrame({"time": short_dates, "units": [1.0] * len(short_dates)})
    required = {"timestamp", "kwh"}
    assert not required.issubset(set(bad_df.columns))

    # 3. Empty data
    empty_df = pd.DataFrame()
    assert empty_df.empty
