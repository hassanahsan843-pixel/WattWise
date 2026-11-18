"""
preprocess.py - Data ingestion, cleaning, resampling, and contextual enrichment
for the London Smart Meters electricity consumption dataset.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import pandas as pd
import numpy as np


def load_raw(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load raw half-hourly smart meter readings from a CSV or Parquet file/directory.

    Handles standard London Smart Meter schemas:
      - 'LCLid', 'tstp', 'energy(kWh/hh)'
      - or normalized: 'household_id', 'timestamp', 'kwh'

    Args:
        path: Path to raw CSV/Parquet file or directory of raw files.

    Returns:
        pd.DataFrame with standardized columns ['household_id', 'timestamp', 'kwh'].
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {path}")

    if path_obj.is_dir():
        csv_files = list(path_obj.glob("*.csv")) + list(path_obj.glob("*.parquet"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV or Parquet files found in directory: {path}")
        dfs = []
        for f in csv_files:
            if f.suffix == ".parquet":
                dfs.append(pd.read_parquet(f))
            else:
                dfs.append(pd.read_csv(f))
        df = pd.concat(dfs, ignore_index=True)
    elif path_obj.suffix == ".parquet":
        df = pd.read_parquet(path_obj)
    else:
        df = pd.read_csv(path_obj)

    # Column mapping dictionary to standardize column names
    col_map = {
        "LCLid": "household_id",
        "lclid": "household_id",
        "household": "household_id",
        "household_id": "household_id",
        "tstp": "timestamp",
        "DateTime": "timestamp",
        "datetime": "timestamp",
        "timestamp": "timestamp",
        "energy(kWh/hh)": "kwh",
        "energy": "kwh",
        "kWh": "kwh",
        "kwh": "kwh",
        "KWH": "kwh",
    }

    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

    required = {"household_id", "timestamp", "kwh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}. Found: {df.columns.tolist()}")

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df[["household_id", "timestamp", "kwh"]]


def select_subset(
    df: pd.DataFrame,
    n_households: int = 100,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Filter the dataset to a manageable subset of households with the highest data completeness,
    optionally bounded by start and end timestamps.

    Args:
        df: Standardized DataFrame with ['household_id', 'timestamp', 'kwh'].
        n_households: Maximum number of households to retain.
        start: Optional start date/time filter.
        end: Optional end date/time filter.

    Returns:
        Filtered pd.DataFrame containing the top `n_households`.
    """
    filtered = df.copy()

    if start is not None:
        filtered = filtered[filtered["timestamp"] >= pd.to_datetime(start)]
    if end is not None:
        filtered = filtered[filtered["timestamp"] <= pd.to_datetime(end)]

    if filtered.empty:
        return filtered

    # Rank households by data completeness (most valid numeric kwh readings, least missing)
    completeness = (
        filtered.assign(is_valid=pd.to_numeric(filtered["kwh"], errors="coerce").notna())
        .groupby("household_id")["is_valid"]
        .agg(["sum", "count"])
    )
    completeness["ratio"] = completeness["sum"] / completeness["count"].clip(lower=1)
    # Sort primarily by completeness ratio, then by total count of valid readings
    best_households = completeness.sort_values(by=["ratio", "sum"], ascending=[False, False]).head(n_households).index

    return filtered[filtered["household_id"].isin(best_households)].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean smart meter readings:
      - Convert kWh to numeric (coercing non-numeric entries to NaN)
      - Drop invalid timestamps and exact duplicates
      - Sort chronologically per household
      - Interpolate small gaps (up to 3 half-hours) linearly
      - Flag long gaps (> 3 consecutive half-hours missing)

    Args:
        df: Raw or subset DataFrame with ['household_id', 'timestamp', 'kwh'].

    Returns:
        Cleaned pd.DataFrame with regularized readings and imputation flags.
    """
    df = df.copy()
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "household_id"])
    df = df.drop_duplicates(subset=["household_id", "timestamp"]).sort_values(["household_id", "timestamp"])

    cleaned_records: List[pd.DataFrame] = []

    for h_id, group in df.groupby("household_id"):
        group = group.set_index("timestamp")
        # Ensure a regular 30-minute interval grid for the household span
        full_idx = pd.date_range(start=group.index.min(), end=group.index.max(), freq="30min")
        reindexed = group.reindex(full_idx)
        reindexed["household_id"] = h_id

        # Detect gap lengths
        is_na = reindexed["kwh"].isna()
        # Group consecutive NaNs to measure gap lengths
        gap_blocks = (~is_na).cumsum()
        gap_sizes = is_na.groupby(gap_blocks).transform("sum")

        # Mark short gaps (<= 3 periods, i.e., 1.5 hours) for interpolation
        interpolate_mask = is_na & (gap_sizes <= 3)
        interpolated_kwh = reindexed["kwh"].interpolate(method="time", limit=3)

        # Retain interpolated values only where mask applies; flag long gaps
        reindexed["is_interpolated"] = interpolate_mask
        reindexed["is_long_gap"] = is_na & (gap_sizes > 3)
        reindexed["kwh"] = np.where(interpolate_mask, interpolated_kwh, reindexed["kwh"])

        # Forward fill or fill 0 for any lingering isolated leading NaNs, but keep valid bounds
        reindexed["kwh"] = reindexed["kwh"].clip(lower=0.0)
        reindexed = reindexed.reset_index().rename(columns={"index": "timestamp"})
        cleaned_records.append(reindexed)

    cleaned_df = pd.concat(cleaned_records, ignore_index=True)
    return cleaned_df


def resample_readings(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """
    Resample half-hourly readings to aggregate kWh per household (hourly 'h' or daily 'D').

    Note:
      Since half-hourly readings represent energy consumed in that 30-minute block,
      hourly aggregation sums the half-hourly blocks (e.g., 2 half-hours = 1 hour total kWh).
      Daily aggregation sums all blocks in that 24-hour period.

    Args:
        df: Cleaned half-hourly DataFrame.
        freq: Resampling frequency ('h' for hourly, 'D' for daily).

    Returns:
        Resampled pd.DataFrame with ['household_id', 'timestamp', 'kwh', 'is_interpolated', 'is_long_gap'].
    """
    valid_freqs = {"h", "1h", "H", "1H", "D", "d", "1D", "1d"}
    if freq not in valid_freqs:
        raise ValueError(f"Frequency must be one of {valid_freqs}, got: {freq}")

    resampled_list: List[pd.DataFrame] = []

    freq_normalized = "1h" if freq.lower() in {"h", "1h"} else "1D"

    for h_id, group in df.groupby("household_id"):
        grp = group.set_index("timestamp")
        resampled_kwh = grp["kwh"].resample(freq_normalized).sum(min_count=1)
        resampled_interp = grp["is_interpolated"].resample(freq_normalized).max() if "is_interpolated" in grp else False
        resampled_gap = grp["is_long_gap"].resample(freq_normalized).max() if "is_long_gap" in grp else False

        res = pd.DataFrame({
            "household_id": h_id,
            "kwh": resampled_kwh,
            "is_interpolated": resampled_interp,
            "is_long_gap": resampled_gap,
        }).reset_index().rename(columns={"index": "timestamp"})
        resampled_list.append(res)

    return pd.concat(resampled_list, ignore_index=True)


def merge_context(
    df: pd.DataFrame,
    weather_path: Optional[Union[str, Path]] = None,
    holidays_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Enrich electricity consumption data with weather (temperature in Celsius)
    and calendar context (UK bank holidays).

    If weather or holiday files are omitted or missing, realistic London historical
    approximations (London seasonal temperatures and standard UK bank holidays)
    are synthesized.

    Args:
        df: Resampled consumption DataFrame.
        weather_path: Path to London DarkSky weather CSV (hourly/daily).
        holidays_path: Path to UK bank holidays CSV.

    Returns:
        DataFrame enriched with 'temperature' (float) and 'is_holiday' (int 0 or 1).
    """
    enriched = df.copy()
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"])
    enriched["date"] = enriched["timestamp"].dt.date

    # 1. Merge or synthesize UK Bank Holidays
    uk_holidays_set = set()
    if holidays_path and Path(holidays_path).exists():
        hol_df = pd.read_csv(holidays_path)
        date_col = next((c for c in hol_df.columns if "date" in c.lower() or "bank" in c.lower()), hol_df.columns[0])
        uk_holidays_set = set(pd.to_datetime(hol_df[date_col], errors="coerce").dt.date.dropna())
    else:
        # Standard UK Public Holidays for 2011-2014 & recent years (New Year, Good Friday, Easter Mon, Early May, Spring, Summer, Christmas, Boxing Day)
        # Using fixed and observed dates
        sample_holidays = [
            "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29", "2011-12-26", "2011-12-27",
            "2012-01-02", "2012-04-06", "2012-04-09", "2012-05-07", "2012-06-04", "2012-06-05", "2012-08-27", "2012-12-25", "2012-12-26",
            "2013-01-01", "2013-03-29", "2013-04-01", "2013-05-06", "2013-05-27", "2013-08-26", "2013-12-25", "2013-12-26",
            "2014-01-01", "2014-04-18", "2014-04-21", "2014-05-05", "2014-05-26", "2014-08-25", "2014-12-25", "2014-12-26",
        ]
        uk_holidays_set = set(pd.to_datetime(sample_holidays).date)

    enriched["is_holiday"] = enriched["date"].isin(uk_holidays_set).astype(int)

    # 2. Merge or synthesize Temperature
    if weather_path and Path(weather_path).exists():
        weather_df = pd.read_csv(weather_path)
        time_col = next((c for c in weather_df.columns if "time" in c.lower() or "date" in c.lower()), None)
        temp_col = next((c for c in weather_df.columns if "temp" in c.lower()), None)

        if time_col and temp_col:
            weather_df["w_time"] = pd.to_datetime(weather_df[time_col], errors="coerce")
            weather_df["temperature"] = pd.to_numeric(weather_df[temp_col], errors="coerce")
            weather_df = weather_df.dropna(subset=["w_time", "temperature"]).sort_values("w_time")
            # Merge as of or round to nearest hour
            weather_df["hour_ts"] = weather_df["w_time"].dt.floor("1h")
            weather_hourly = weather_df.groupby("hour_ts")["temperature"].mean().reset_index()

            enriched["hour_ts"] = enriched["timestamp"].dt.floor("1h")
            enriched = pd.merge(enriched, weather_hourly, on="hour_ts", how="left")
            enriched = enriched.drop(columns=["hour_ts"])
            # Interpolate any missing temperature hours
            enriched["temperature"] = enriched["temperature"].interpolate(method="linear").bfill().ffill()

    if "temperature" not in enriched.columns or enriched["temperature"].isna().all():
        # London typical seasonal temperature model:
        # Base annual sinusoidal curve: Winter ~5C (Jan), Summer ~19C (Jul), plus diurnal fluctuation (+-3C)
        day_of_year = getattr(enriched["timestamp"].dt, "day_of_year", getattr(enriched["timestamp"].dt, "dayofyear", 1))
        hour_of_day = enriched["timestamp"].dt.hour
        seasonal_temp = 12.0 - 7.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
        diurnal_temp = 3.0 * np.sin(2 * np.pi * (hour_of_day - 8) / 24.0)
        enriched["temperature"] = np.round(seasonal_temp + diurnal_temp, 2)

    enriched = enriched.drop(columns=["date"], errors="ignore")
    return enriched


def generate_data_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Produce a rigorous data-quality and descriptive summary dictionary,
    suitable for formatting directly into technical reports and slides.

    Args:
        df: Cleaned and enriched DataFrame.

    Returns:
        Dictionary of data-quality metrics and summary statistics.
    """
    total_records = len(df)
    n_households = df["household_id"].nunique()
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    missing_kwh = df["kwh"].isna().sum()
    pct_missing = (missing_kwh / total_records * 100) if total_records > 0 else 0.0

    kwh_stats = df["kwh"].describe().to_dict()

    summary = {
        "total_records": total_records,
        "n_households": n_households,
        "date_range_start": str(min_date),
        "date_range_end": str(max_date),
        "missing_kwh_count": int(missing_kwh),
        "missing_kwh_percentage": round(float(pct_missing), 3),
        "kwh_mean": round(float(kwh_stats.get("mean", 0.0)), 4),
        "kwh_std": round(float(kwh_stats.get("std", 0.0)), 4),
        "kwh_median": round(float(kwh_stats.get("50%", 0.0)), 4),
        "kwh_min": round(float(kwh_stats.get("min", 0.0)), 4),
        "kwh_max": round(float(kwh_stats.get("max", 0.0)), 4),
        "features": list(df.columns),
    }
    return summary


def format_summary_markdown(summary: Dict[str, Any]) -> str:
    """Format the data quality summary as a publication-ready Markdown table."""
    md = f"""### Dataset Quality & Ingestion Summary

| Metric | Value |
| :--- | :--- |
| **Total Processed Records** | {summary['total_records']:,} |
| **Unique Households** | {summary['n_households']} |
| **Timeframe Range** | {summary['date_range_start']} to {summary['date_range_end']} |
| **Missing Readings** | {summary['missing_kwh_count']} ({summary['missing_kwh_percentage']}%) |
| **Mean Consumption (kWh)** | {summary['kwh_mean']:.4f} |
| **Standard Deviation (kWh)** | {summary['kwh_std']:.4f} |
| **Median Consumption (kWh)** | {summary['kwh_median']:.4f} |
| **Min / Max (kWh)** | {summary['kwh_min']:.4f} / {summary['kwh_max']:.4f} |
| **Active Schema Columns** | `{', '.join(summary['features'])}` |
"""
    return md
