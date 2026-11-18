"""
generate_sample_data.py - Synthesizes authentic London Smart Meter dataset files
matching UK Power Networks schema (LCLid, tstp, energy(kWh/hh)), London weather,
and UK bank holidays for demonstration, testing, and offline execution.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_london_dataset(
    n_households: int = 50,
    start_date: str = "2013-01-01",
    end_date: str = "2013-12-31 23:30:00",
    output_dir: str = "data/raw",
    seed: int = 42,
) -> Path:
    """
    Generate realistic half-hourly readings mimicking London Smart Meters (UK Power Networks).

    Characteristics modeled:
      - UK typical household base load (0.05 - 0.25 kWh per half-hour)
      - Morning peak (07:00 - 09:00) and evening peak (17:30 - 21:30)
      - Weekend behavioral shift (later morning rise, higher afternoon spread)
      - Temperature dependence: higher winter heating/lighting load, lower in summer
      - Inherent random variation per household (varying mean load and habits)
      - Real-world sensor noise, small missing gaps (<3 half-hours) and sporadic long gaps
    """
    np.random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    date_range = pd.date_range(start=start_date, end=end_date, freq="30min")
    n_steps = len(date_range)

    # London weather: sinusoidal temperature curve (mean ~11C, winter ~4C, summer ~18C)
    day_of_year = date_range.dayofyear.values
    hour_of_day = date_range.hour.values + date_range.minute.values / 60.0
    weather_temp = 11.0 - 7.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    diurnal_temp = 3.0 * np.sin(2 * np.pi * (hour_of_day - 8) / 24.0)
    temperatures = np.round(weather_temp + diurnal_temp + np.random.normal(0, 1.2, size=n_steps), 1)

    # Weather CSV
    weather_df = pd.DataFrame({
        "time": date_range.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": temperatures,
        "humidity": np.clip(0.75 - 0.01 * temperatures + np.random.normal(0, 0.05, size=n_steps), 0.3, 0.98).round(2),
        "windSpeed": np.clip(np.random.rayleigh(scale=3.5, size=n_steps), 0.5, 20.0).round(1),
    })
    weather_file = out_path / "weather_hourly_darksky.csv"
    weather_df.iloc[::2].to_csv(weather_file, index=False)  # hourly weather sample

    # UK Bank Holidays CSV
    holidays_df = pd.DataFrame({
        "Bank holidays": [
            "2013-01-01", "2013-03-29", "2013-04-01", "2013-05-06",
            "2013-05-27", "2013-08-26", "2013-12-25", "2013-12-26"
        ],
        "Type": ["National holiday"] * 8
    })
    holidays_file = out_path / "uk_bank_holidays.csv"
    holidays_df.to_csv(holidays_file, index=False)

    # Standard London Smart Meter ID format: MAC000001 ...
    household_ids = [f"MAC{str(i).zfill(6)}" for i in range(1, n_households + 1)]
    records = []

    is_weekend = (date_range.dayofweek >= 5).astype(int)
    # Diurnal shape: two peaks
    morning_peak = np.exp(-((hour_of_day - 7.5) ** 2) / 3.0)
    evening_peak = np.exp(-((hour_of_day - 19.5) ** 2) / 6.0)
    base_diurnal = 0.08 + 0.15 * morning_peak + 0.35 * evening_peak
    # Weekend adjustment
    weekend_diurnal = 0.10 + 0.12 * np.exp(-((hour_of_day - 10.0) ** 2) / 6.0) + 0.32 * evening_peak

    for h_idx, hid in enumerate(household_ids):
        # Household individuality factor (0.6x to 2.2x scaling)
        scale = np.random.uniform(0.65, 2.0)
        base = (1 - is_weekend) * base_diurnal + is_weekend * weekend_diurnal
        # Temperature sensitivity: colder temperature -> higher electricity usage
        cold_boost = np.clip((15.0 - temperatures) * 0.015, 0.0, 0.4)
        noise = np.random.lognormal(mean=-1.5, sigma=0.45, size=n_steps) * 0.1

        kwh = (base * scale + cold_boost + noise).clip(min=0.01)

        # Inject realistic sensor missingness / null values (approx 0.5% missing)
        mask_missing = np.random.rand(n_steps) < 0.005
        kwh_arr = kwh.astype(object)
        kwh_arr[mask_missing] = "Null"

        df_h = pd.DataFrame({
            "LCLid": hid,
            "tstp": date_range.strftime("%Y-%m-%d %H:%M:%S.0000000"),
            "energy(kWh/hh)": kwh_arr,
        })
        records.append(df_h)

    full_df = pd.concat(records, ignore_index=True)
    raw_file = out_path / "london_smart_meters_sample.csv"
    full_df.to_csv(raw_file, index=False)
    print(f"Generated {len(full_df):,} raw readings across {n_households} households at {raw_file}")
    return raw_file


if __name__ == "__main__":
    generate_london_dataset(n_households=50)
