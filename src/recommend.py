"""
recommend.py - Actionable energy efficiency recommendations mapped from
detected anomaly patterns and estimated financial/kWh savings.
"""

from typing import List, Dict, Any
from src.bill import DEFAULT_TARIFF_PER_KWH


RECOMMENDATION_CATALOG = {
    "night_spike": [
        {
            "action": "Check timer settings on immersion water heaters and electric storage heaters.",
            "description": "High draw during sleeping hours often indicates an unoptimized heating or water tank timer.",
            "kwh_fraction": 0.40,
        },
        {
            "action": "Inspect automated appliances or EV chargers configured to start overnight.",
            "description": "Ensure electric vehicle or dehumidifier chargers are properly scheduled during off-peak windows.",
            "kwh_fraction": 0.35,
        },
    ],
    "sustained_baseload": [
        {
            "action": "Audit vampire appliances and continuous background electrical draws.",
            "description": "Desktop PCs left running, media servers, second refrigerators, or continuous pumps elevate your baseline 24/7.",
            "kwh_fraction": 0.30,
        },
        {
            "action": "Install smart plugs with automated standby shut-off.",
            "description": "Automatically cut phantom power to entertainment centers and home office peripherals when not in use.",
            "kwh_fraction": 0.20,
        },
    ],
    "evening_peak": [
        {
            "action": "Shift high-draw laundry and dishwasher cycles outside peak hours (17:00 - 20:00).",
            "description": "Running washing machines, dryers, and dishwashers during morning or later evening reduces peak demand.",
            "kwh_fraction": 0.25,
        },
        {
            "action": "Stagger major electric cooking and space heating usage.",
            "description": "Avoid simultaneous operation of electric oven, hob, kettle, and supplementary electric heaters.",
            "kwh_fraction": 0.20,
        },
    ],
    "weather_related": [
        {
            "action": "Lower thermostat temperature setting by 1°C and inspect draft proofing.",
            "description": "Each degree reduction in thermostat temperature typically yields up to 10% reduction in heating energy.",
            "kwh_fraction": 0.20,
        },
        {
            "action": "Bleed radiators and verify thermostatic radiator valves (TRVs).",
            "description": "Ensure heating zones in unoccupied rooms are reduced during sub-zero temperature spells.",
            "kwh_fraction": 0.15,
        },
    ],
    "sudden_drop": [
        {
            "action": "Verify safety circuit breakers and smart meter communications.",
            "description": "An unexpected collapse in electricity consumption may indicate a tripped RCD fuse or smart meter HAN connectivity drop.",
            "kwh_fraction": 0.0,
        },
    ],
}


def recommend(
    pattern_type: str,
    magnitude: float = 1.0,
    tariff_per_kwh: float = DEFAULT_TARIFF_PER_KWH,
) -> List[Dict[str, Any]]:
    """
    Generate actionable energy-saving recommendations based on the detected anomaly pattern.

    Only triggered for instances classified as anomalous. If pattern is 'normal'
    or has zero excess magnitude, returns an empty list (no recommendation).

    Args:
        pattern_type: Anomaly pattern classification ('night_spike', 'sustained_baseload',
                      'evening_peak', 'weather_related', 'sudden_drop', or 'normal').
        magnitude: Estimated excess consumption in kWh (e.g. difference between actual and normal).
        tariff_per_kwh: Configurable electricity unit rate in £ per kWh.

    Returns:
        List of recommendation dictionaries, each specifying action, description,
        estimated monthly kWh savings, and estimated monthly GBP savings.
    """
    if pattern_type == "normal" or pattern_type not in RECOMMENDATION_CATALOG:
        return []

    # If magnitude is non-positive or near zero (e.g. drops), treat excess kWh as 0
    excess_kwh = max(0.0, float(magnitude))

    # Calculate conservative monthly projected impact (assuming similar pattern recurs across 30 days)
    # E.g. recurring weekly or daily pattern
    monthly_excess_kwh = excess_kwh * 20.0  # conservative recurring multiplier

    items = RECOMMENDATION_CATALOG[pattern_type]
    recommendations = []

    for item in items:
        kwh_fraction = item["kwh_fraction"]
        monthly_saving_kwh = round(monthly_excess_kwh * kwh_fraction, 1)
        monthly_saving_gbp = round(monthly_saving_kwh * tariff_per_kwh, 2)

        recommendations.append({
            "pattern_type": pattern_type,
            "action": item["action"],
            "description": item["description"],
            "estimated_monthly_saving_kwh": monthly_saving_kwh,
            "estimated_monthly_saving_gbp": monthly_saving_gbp,
        })

    return recommendations
