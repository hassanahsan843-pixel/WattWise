"""
explain.py - Explaining electricity anomalies with contextual breakdown,
feature attribution, and natural language descriptions.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd


def explain_anomaly(
    row: Union[pd.Series, dict],
    household_profile: Optional[pd.DataFrame] = None,
    shap_values: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate actionable, human-interpretable explanations for an anomalous electricity reading.

    Computes:
      - multiplier vs normal: Actual kWh / Expected normal kWh for that hour & day
      - pattern_type: One of 'night_spike', 'sustained_baseload', 'evening_peak', 'sudden_drop', 'weather_related'
      - top_3_contributing_features: Key statistical/environmental features driving the anomaly
      - explanation_text: Clear plain-English sentence summarizing the finding.

    Args:
        row: Series or dict representing an hourly reading (must include timestamp, kwh, and features).
        household_profile: Optional DataFrame with ['dayofweek', 'hour', 'profile_median'].
        shap_values: Optional SHAP values array or object for local attribution.
        feature_names: Optional list of feature names corresponding to shap_values.

    Returns:
        Dictionary containing explanation metrics and descriptive text.
    """
    row_dict = row.to_dict() if isinstance(row, pd.Series) else dict(row)

    ts = pd.to_datetime(row_dict.get("timestamp", "2013-01-01 00:00:00"))
    hour = int(row_dict.get("hour", ts.hour))
    dayofweek = int(row_dict.get("dayofweek", getattr(ts, "day_of_week", getattr(ts, "dayofweek", 0))))
    actual_kwh = float(row_dict.get("kwh", 0.0))
    temp = float(row_dict.get("temperature", 12.0))
    rolling_24 = float(row_dict.get("rolling_24h_mean", actual_kwh))
    rolling_7d = float(row_dict.get("rolling_7d_mean", rolling_24))

    # Determine baseline normal consumption for this hour
    normal_kwh = 0.20  # default UK domestic fallback
    if "profile_median" in row_dict and pd.notna(row_dict["profile_median"]) and row_dict["profile_median"] > 0:
        normal_kwh = float(row_dict["profile_median"])
    elif household_profile is not None and not household_profile.empty:
        match = household_profile[
            (household_profile["hour"] == hour) & (household_profile["dayofweek"] == dayofweek)
        ]
        if not match.empty and "profile_median" in match.columns:
            normal_kwh = float(match["profile_median"].iloc[0])

    safe_normal = max(0.02, normal_kwh)
    multiplier = round(actual_kwh / safe_normal, 2)

    # Classify pattern type
    pattern_type = "unclassified"
    explanation_text = ""

    if multiplier <= 0.25 and normal_kwh > 0.15:
        pattern_type = "sudden_drop"
        explanation_text = (
            f"Consumption dropped to {actual_kwh:.2f} kWh ({multiplier:.1f}x normal) at {hour:02d}:00, "
            f"significantly below your expected {normal_kwh:.2f} kWh baseline."
        )
    elif hour in [0, 1, 2, 3, 4, 5] and multiplier >= 2.0:
        if rolling_24 > (1.8 * safe_normal):
            pattern_type = "sustained_baseload"
            explanation_text = (
                f"Sustained high overnight baseline detected at {hour:02d}:00: "
                f"Usage remained at {actual_kwh:.2f} kWh ({multiplier:.1f}x normal) throughout the night."
            )
        else:
            pattern_type = "night_spike"
            explanation_text = (
                f"Abnormal overnight spike: Usage reached {actual_kwh:.2f} kWh ({multiplier:.1f}x normal) at {hour:02d}:00 "
                f"while the home is typically inactive."
            )
    elif temp <= 3.0 and multiplier >= 1.5:
        pattern_type = "weather_related"
        explanation_text = (
            f"Weather-driven high consumption: Usage rose to {actual_kwh:.2f} kWh ({multiplier:.1f}x normal) "
            f"correlated with freezing outdoor temperatures ({temp:.1f}°C)."
        )
    elif hour in [17, 18, 19, 20, 21, 22] and multiplier >= 2.0:
        pattern_type = "evening_peak"
        explanation_text = (
            f"Extreme evening peak: Usage surged to {actual_kwh:.2f} kWh ({multiplier:.1f}x normal) at {hour:02d}:00, "
            f"exceeding typical dinner/active hours."
        )
    else:
        # Default fallback categorization based on time and level
        if hour in [0, 1, 2, 3, 4, 5]:
            pattern_type = "night_spike"
            explanation_text = f"Unusual overnight energy draw of {actual_kwh:.2f} kWh ({multiplier:.1f}x normal) at {hour:02d}:00."
        elif multiplier >= 1.5:
            pattern_type = "evening_peak" if hour >= 12 else "sustained_baseload"
            explanation_text = f"Elevated consumption of {actual_kwh:.2f} kWh ({multiplier:.1f}x normal) recorded at {hour:02d}:00."
        else:
            pattern_type = "normal"
            explanation_text = f"Normal consumption of {actual_kwh:.2f} kWh at {hour:02d}:00 consistent with household profile."

    # Top contributing features
    top_3_features = []
    if shap_values is not None and feature_names is not None:
        try:
            # Sort by absolute SHAP attribution
            indexed_shaps = list(zip(feature_names, np.abs(shap_values)))
            indexed_shaps.sort(key=lambda x: x[1], reverse=True)
            top_3_features = [f"{name} (impact: {val:.3f})" for name, val in indexed_shaps[:3]]
        except Exception:
            top_3_features = []

    if not top_3_features:
        # Statistical deviation ranking fallback
        feature_deviations = [
            ("ratio_to_household_profile", abs(multiplier - 1.0)),
            ("rolling_24h_mean", abs(actual_kwh - rolling_24)),
            ("temperature_effect", abs(15.0 - temp) / 10.0),
            ("lag_24h_delta", abs(actual_kwh - float(row_dict.get("lag_24h", actual_kwh)))),
        ]
        feature_deviations.sort(key=lambda x: x[1], reverse=True)
        top_3_features = [f[0] for f in feature_deviations[:3]]

    return {
        "timestamp": str(ts),
        "hour": hour,
        "actual_kwh": round(actual_kwh, 3),
        "normal_kwh": round(normal_kwh, 3),
        "multiplier_vs_normal": multiplier,
        "pattern_type": pattern_type,
        "top_3_contributing_features": top_3_features,
        "explanation_text": explanation_text,
    }
