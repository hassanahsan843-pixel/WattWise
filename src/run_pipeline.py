"""
run_pipeline.py - Orchestrates the full WattWise AI data processing and model training pipeline.

Executes:
  1. Ingestion of raw London smart meter half-hourly data
  2. Subsetting top 50 households with highest data completeness
  3. Cleaning: type conversion, deduplication, interpolation of gaps <= 3 half-hours
  4. Resampling to hourly ('h') and daily ('D') consumption Parquet files
  5. Contextual merging of London temperature and UK bank holidays
  6. Anomaly detection training (Isolation Forest) and evaluation against Z-score baseline
  7. Forecasting model training (Random Forest) and evaluation against Naive Baseline
  8. Generation of performance metrics CSVs and figures for technical report and Streamlit app
"""

from pathlib import Path
import sys
import time
import warnings

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
try:
    pd.set_option("io.parquet.engine", "fastparquet")
except Exception:
    pass
warnings.filterwarnings("ignore")

from src.preprocess import (
    load_raw,
    select_subset,
    clean,
    resample_readings,
    merge_context,
    generate_data_quality_summary,
    format_summary_markdown,
)
from src.features import build_features
from src.anomaly import evaluate_anomaly_detection, train_isolation_forest
from src.forecast import evaluate_forecasting, train_rf, make_forecast_features


def run_full_pipeline():
    start_time = time.time()
    print("=" * 65)
    print("WATTWISE AI: END-TO-END DATA & MODEL TRAINING PIPELINE")
    print("=" * 65)

    raw_path = Path("data/raw/london_smart_meters_sample.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Run generate_sample_data.py first.")

    weather_path = Path("data/raw/weather_hourly_darksky.csv")
    holidays_path = Path("data/raw/uk_bank_holidays.csv")
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path("reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Raw Data
    print("\n[Step 1.1] Loading raw half-hourly readings...")
    df_raw = load_raw(raw_path)
    print(f"Loaded {len(df_raw):,} half-hourly records across {df_raw['household_id'].nunique()} households.")

    # 2. Select Subset
    print("\n[Step 1.2] Subsetting top complete households (n=50)...")
    df_sub = select_subset(df_raw, n_households=50)

    # 3. Clean
    print("\n[Step 1.3] Cleaning: sorting, deduplication, and gap interpolation...")
    df_clean = clean(df_sub)

    # 4. Resample Hourly & Daily
    print("\n[Step 1.4] Resampling half-hourly readings to hourly and daily aggregations...")
    df_hourly = resample_readings(df_clean, freq="h")
    df_daily = resample_readings(df_clean, freq="D")

    # 5. Merge Weather and Bank Holidays Context
    print("\n[Step 1.5] Merging London temperature and UK bank holidays...")
    df_hourly = merge_context(
        df_hourly,
        weather_path=weather_path if weather_path.exists() else None,
        holidays_path=holidays_path if holidays_path.exists() else None,
    )
    df_daily = merge_context(
        df_daily,
        weather_path=weather_path if weather_path.exists() else None,
        holidays_path=holidays_path if holidays_path.exists() else None,
    )

    # Save to Parquet
    hourly_parquet = proc_dir / "hourly.parquet"
    daily_parquet = proc_dir / "daily.parquet"
    df_hourly.to_parquet(hourly_parquet, index=False)
    df_daily.to_parquet(daily_parquet, index=False)
    print(f"Saved: {hourly_parquet} ({len(df_hourly):,} rows)")
    print(f"Saved: {daily_parquet} ({len(df_daily):,} rows)")

    # Data Quality Summary
    summary = generate_data_quality_summary(df_hourly)
    summary_md = format_summary_markdown(summary)
    with open(rep_dir / "data_quality_summary.md", "w") as f:
        f.write(summary_md)
    print("\nData Quality Summary:\n", summary_md)

    # Step 2: Anomaly Detection Model & Evaluation
    print("\n[Step 2] Training Isolation Forest & Evaluating against Z-Score Baseline...")
    anomaly_metrics_df, anom_details = evaluate_anomaly_detection(
        df_hourly,
        contamination=0.03,
        train_split_ratio=0.75,
        reports_dir=rep_dir,
    )
    print("\nAnomaly Detection Results:\n", anomaly_metrics_df.to_string(index=False))

    # Also train and save global model on all available feature data
    print("Fitting production Isolation Forest on full dataset...")
    full_features = build_features(df_hourly)
    train_isolation_forest(full_features, contamination=0.03, save_path=models_dir / "isolation_forest.joblib")

    # Step 3: Forecasting Model & Evaluation
    print("\n[Step 3] Training Random Forest Regressor & Evaluating against Naive Baseline...")
    forecast_metrics_df, fc_details = evaluate_forecasting(
        df_hourly,
        test_days=30,
        reports_dir=rep_dir,
    )
    print("\nForecasting Results:\n", forecast_metrics_df.to_string(index=False))

    # Train production Random Forest Forecaster on full dataset
    print("Fitting production Random Forest Forecaster on full dataset...")
    full_fc_features = make_forecast_features(df_hourly)
    train_rf(full_fc_features, save_path=models_dir / "rf_forecast.joblib")

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 65)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed}s")
    print(f"Models saved: {list(models_dir.glob('*.joblib'))}")
    print(f"Reports generated: {list(rep_dir.glob('*.csv')) + list(rep_dir.glob('*.md'))}")
    print("=" * 65)


if __name__ == "__main__":
    run_full_pipeline()
