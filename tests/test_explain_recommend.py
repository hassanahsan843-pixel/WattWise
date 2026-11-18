"""
Unit tests for explain.py and recommend.py covering all 5 anomaly pattern types
and a normal row verification.
"""

import pytest
import pandas as pd
from src.explain import explain_anomaly
from src.recommend import recommend


@pytest.fixture
def dummy_household_profile():
    records = []
    for dow in range(7):
        for h in range(24):
            records.append({
                "dayofweek": dow,
                "hour": h,
                "profile_median": 0.20,
            })
    return pd.DataFrame(records)


def test_night_spike_explanation_and_recommendation(dummy_household_profile):
    """Case 1: Sudden spike at 3:00 AM (e.g. 1.20 kWh vs 0.20 kWh normal)."""
    row = {
        "timestamp": "2013-05-10 03:00:00",
        "hour": 3,
        "dayofweek": 4,
        "kwh": 1.20,
        "temperature": 11.0,
        "rolling_24h_mean": 0.22,
        "rolling_7d_mean": 0.21,
    }
    exp = explain_anomaly(row, household_profile=dummy_household_profile)
    assert exp["pattern_type"] == "night_spike"
    assert exp["multiplier_vs_normal"] >= 4.0
    assert "overnight" in exp["explanation_text"].lower() or "spike" in exp["explanation_text"].lower()

    recs = recommend(exp["pattern_type"], magnitude=exp["actual_kwh"] - exp["normal_kwh"], tariff_per_kwh=0.28)
    assert len(recs) >= 1
    assert recs[0]["estimated_monthly_saving_kwh"] > 0
    assert recs[0]["estimated_monthly_saving_gbp"] > 0


def test_sustained_baseload_explanation_and_recommendation(dummy_household_profile):
    """Case 2: Elevated rolling overnight baseload at 2:00 AM."""
    row = {
        "timestamp": "2013-06-15 02:00:00",
        "hour": 2,
        "dayofweek": 5,
        "kwh": 0.70,
        "temperature": 14.0,
        "rolling_24h_mean": 0.65,  # high continuous background
        "rolling_7d_mean": 0.60,
    }
    exp = explain_anomaly(row, household_profile=dummy_household_profile)
    assert exp["pattern_type"] == "sustained_baseload"
    assert exp["multiplier_vs_normal"] >= 3.0

    recs = recommend(exp["pattern_type"], magnitude=0.50, tariff_per_kwh=0.28)
    assert len(recs) >= 1
    assert "vampire" in recs[0]["description"].lower() or "baseload" in recs[0]["action"].lower() or "appliances" in recs[0]["action"].lower()


def test_evening_peak_explanation_and_recommendation(dummy_household_profile):
    """Case 3: Evening surge at 19:00 (e.g. 2.50 kWh vs 0.30 kWh normal)."""
    row = {
        "timestamp": "2013-04-12 19:00:00",
        "hour": 19,
        "dayofweek": 4,
        "kwh": 2.50,
        "temperature": 12.0,
        "rolling_24h_mean": 0.45,
        "rolling_7d_mean": 0.40,
        "profile_median": 0.30,
    }
    exp = explain_anomaly(row, household_profile=dummy_household_profile)
    assert exp["pattern_type"] == "evening_peak"
    assert exp["multiplier_vs_normal"] >= 5.0

    recs = recommend(exp["pattern_type"], magnitude=2.20, tariff_per_kwh=0.28)
    assert len(recs) >= 1
    assert any("peak" in r["action"].lower() or "shift" in r["action"].lower() for r in recs)


def test_weather_related_explanation_and_recommendation(dummy_household_profile):
    """Case 4: Freezing day weather surge (0.5°C, high heating load)."""
    row = {
        "timestamp": "2013-01-20 14:00:00",
        "hour": 14,
        "dayofweek": 6,
        "kwh": 1.10,
        "temperature": 0.5,
        "rolling_24h_mean": 0.70,
        "rolling_7d_mean": 0.65,
        "profile_median": 0.25,
    }
    exp = explain_anomaly(row, household_profile=dummy_household_profile)
    assert exp["pattern_type"] == "weather_related"
    assert "freezing" in exp["explanation_text"].lower() or "weather" in exp["explanation_text"].lower()

    recs = recommend(exp["pattern_type"], magnitude=0.85, tariff_per_kwh=0.28)
    assert len(recs) >= 1
    assert any("thermostat" in r["action"].lower() or "radiator" in r["action"].lower() for r in recs)


def test_sudden_drop_explanation_and_recommendation(dummy_household_profile):
    """Case 5: Sudden collapse to 0.01 kWh during daytime."""
    row = {
        "timestamp": "2013-08-05 12:00:00",
        "hour": 12,
        "dayofweek": 0,
        "kwh": 0.01,
        "temperature": 21.0,
        "rolling_24h_mean": 0.35,
        "rolling_7d_mean": 0.30,
        "profile_median": 0.35,
    }
    exp = explain_anomaly(row, household_profile=dummy_household_profile)
    assert exp["pattern_type"] == "sudden_drop"
    assert exp["multiplier_vs_normal"] <= 0.1

    recs = recommend(exp["pattern_type"], magnitude=0.0, tariff_per_kwh=0.28)
    assert len(recs) >= 1
    assert any("breaker" in r["action"].lower() or "meter" in r["action"].lower() for r in recs)


def test_normal_row_returns_no_recommendations(dummy_household_profile):
    """Normal behavior check: must return 'normal' pattern and zero recommendations."""
    row = {
        "timestamp": "2013-09-10 11:00:00",
        "hour": 11,
        "dayofweek": 1,
        "kwh": 0.21,
        "temperature": 16.0,
        "rolling_24h_mean": 0.20,
        "rolling_7d_mean": 0.20,
        "profile_median": 0.20,
    }
    exp = explain_anomaly(row, household_profile=dummy_household_profile)
    assert exp["pattern_type"] == "normal"

    recs = recommend(exp["pattern_type"], magnitude=0.01, tariff_per_kwh=0.28)
    assert recs == []  # MUST return no recommendation for normal rows!
