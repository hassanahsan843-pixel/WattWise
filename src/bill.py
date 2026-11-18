"""
bill.py - Electricity bill estimation and tariff calculations for WattWise AI.

Converts predicted and actual kilowatt-hour consumption into GBP currency (£)
using realistic UK energy tariffs (variable unit rate + daily standing charge).
"""

from typing import Dict, Any, Union, Optional
import pandas as pd
import numpy as np


# Standard default tariff based on published UK Ofgem Energy Price Cap benchmarks:
# - Unit rate: £0.28 per kWh
# - Standing charge: £0.60 per day
DEFAULT_TARIFF_PER_KWH = 0.28
DEFAULT_STANDING_CHARGE_PER_DAY = 0.60


def estimate_bill(
    kwh_series: Union[pd.Series, np.ndarray, list, pd.DataFrame],
    tariff_per_kwh: float = DEFAULT_TARIFF_PER_KWH,
    standing_charge_per_day: float = DEFAULT_STANDING_CHARGE_PER_DAY,
    lower_kwh_series: Optional[Union[pd.Series, np.ndarray, list]] = None,
    upper_kwh_series: Optional[Union[pd.Series, np.ndarray, list]] = None,
    n_days: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate estimated electricity bill in GBP (£) with uncertainty range.

    The total bill consists of two components:
      1. Consumption Charge = Total kWh * Unit Tariff (£/kWh)
      2. Standing Charge = Days * Daily Standing Charge (£/day)

    If lower/upper bound series are provided (from forecasting model tree variance),
    lower and upper bill estimates are also calculated.

    Args:
        kwh_series: Consumption readings (hourly or daily kWh).
        tariff_per_kwh: Configurable unit rate in £ per kWh (default £0.28).
        standing_charge_per_day: Fixed daily access charge in £ per day (default £0.60).
        lower_kwh_series: Optional lower confidence bound kWh series.
        upper_kwh_series: Optional upper confidence bound kWh series.
        n_days: Optional explicit number of days. If None, inferred from series length (assuming hourly or daily).

    Returns:
        Dictionary containing:
          - 'total_kwh': Total predicted/measured energy in kWh
          - 'unit_cost': Cost of consumed energy (£)
          - 'standing_cost': Fixed standing charge (£)
          - 'total_bill': Sum of unit cost and standing charge (£)
          - 'lower_bill': Lower bound estimate (£)
          - 'upper_bill': Upper bound estimate (£)
          - 'tariff_per_kwh': Tariff used (£)
          - 'standing_charge_per_day': Daily charge used (£)
    """
    if isinstance(kwh_series, pd.DataFrame):
        if "predicted_kwh" in kwh_series.columns:
            kwh_vals = kwh_series["predicted_kwh"].values
            if lower_kwh_series is None and "lower_bound" in kwh_series.columns:
                lower_kwh_series = kwh_series["lower_bound"].values
            if upper_kwh_series is None and "upper_bound" in kwh_series.columns:
                upper_kwh_series = kwh_series["upper_bound"].values
        elif "kwh" in kwh_series.columns:
            kwh_vals = kwh_series["kwh"].values
        else:
            kwh_vals = kwh_series.iloc[:, 0].values
    else:
        kwh_vals = np.asarray(kwh_series, dtype=float)

    total_kwh = float(np.nansum(kwh_vals))

    # Infer duration in days if not provided
    if n_days is None:
        # If length is roughly divisible by 24, assume hourly; otherwise assume daily
        length = len(kwh_vals)
        if length > 48:
            n_days = max(1.0, length / 24.0)
        else:
            n_days = max(1.0, float(length))

    unit_cost = total_kwh * tariff_per_kwh
    standing_cost = n_days * standing_charge_per_day
    total_bill = unit_cost + standing_cost

    # Calculate uncertainty bounds
    if lower_kwh_series is not None:
        lower_kwh_total = float(np.nansum(np.asarray(lower_kwh_series, dtype=float)))
        lower_bill = (lower_kwh_total * tariff_per_kwh) + standing_cost
    else:
        lower_bill = total_bill * 0.90  # fallback 10% lower bound

    if upper_kwh_series is not None:
        upper_kwh_total = float(np.nansum(np.asarray(upper_kwh_series, dtype=float)))
        upper_bill = (upper_kwh_total * tariff_per_kwh) + standing_cost
    else:
        upper_bill = total_bill * 1.10  # fallback 10% upper bound

    return {
        "total_kwh": round(total_kwh, 2),
        "n_days": round(float(n_days), 1),
        "unit_cost": round(unit_cost, 2),
        "standing_cost": round(standing_cost, 2),
        "total_bill": round(total_bill, 2),
        "lower_bill": round(lower_bill, 2),
        "upper_bill": round(upper_bill, 2),
        "tariff_per_kwh": tariff_per_kwh,
        "standing_charge_per_day": standing_charge_per_day,
    }


def estimate_avoidable_cost(
    anomalous_kwh: float,
    expected_kwh: float,
    tariff_per_kwh: float = DEFAULT_TARIFF_PER_KWH,
) -> Dict[str, float]:
    """
    Calculate the avoidable financial loss resulting from anomalous consumption
    exceeding normal baseline behavior.

    Args:
        anomalous_kwh: Total energy consumed during anomalous periods.
        expected_kwh: Normal expected baseline energy for those same periods.
        tariff_per_kwh: Configurable unit tariff.

    Returns:
        Dictionary with excess kWh and financial avoidable loss (£).
    """
    excess_kwh = max(0.0, anomalous_kwh - expected_kwh)
    financial_loss = excess_kwh * tariff_per_kwh
    return {
        "excess_kwh": round(excess_kwh, 2),
        "avoidable_cost_gbp": round(financial_loss, 2),
    }
