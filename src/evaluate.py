"""
evaluate.py - Unified evaluation engine for WattWise AI.

Produces consolidated metric tables, baseline comparisons, and diagnostic outputs
for both the Isolation Forest anomaly detector and the Random Forest forecaster.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

from src.anomaly import evaluate_anomaly_detection
from src.forecast import evaluate_forecasting


def evaluate_all(
    hourly_parquet_path: str = "data/processed/hourly.parquet",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Run full quantitative evaluation for all WattWise AI machine learning components.

    Outputs generated in reports/:
      - anomaly_metrics.csv
      - forecast_metrics.csv
      - figures/anomaly_detection_timeline.png
      - figures/anomaly_score_distribution.png
      - figures/forecast_actual_vs_predicted.png

    Args:
        hourly_parquet_path: Location of processed hourly data.
        reports_dir: Destination folder for metrics tables and diagnostic figures.

    Returns:
        Dictionary containing metric summaries and data splits.
    """
    hourly_p = Path(hourly_parquet_path)
    if not hourly_p.exists():
        raise FileNotFoundError(f"Processed hourly dataset not found at {hourly_p}")

    df_hourly = pd.read_parquet(hourly_p)
    rep_p = Path(reports_dir)
    rep_p.mkdir(parents=True, exist_ok=True)

    print("[Evaluation] Evaluating Anomaly Detection against Baseline...")
    anom_metrics, anom_details = evaluate_anomaly_detection(
        df_hourly,
        contamination=0.03,
        train_split_ratio=0.75,
        reports_dir=rep_p,
    )

    print("[Evaluation] Evaluating Forecaster against Naive Baseline...")
    fc_metrics, fc_details = evaluate_forecasting(
        df_hourly,
        test_days=30,
        reports_dir=rep_p,
    )

    return {
        "anomaly_metrics": anom_metrics.to_dict(orient="records"),
        "forecast_metrics": fc_metrics.to_dict(orient="records"),
        "details": {
            "anomaly": anom_details,
            "forecast": fc_details,
        },
    }


if __name__ == "__main__":
    evaluate_all()
