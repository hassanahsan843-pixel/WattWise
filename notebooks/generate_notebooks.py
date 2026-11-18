"""
generate_notebooks.py - Programmatically creates the 3 mandatory Jupyter notebooks
for WattWise AI:
  1. 01_eda.ipynb
  2. 02_anomaly.ipynb
  3. 03_forecast.ipynb
"""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def markdown_cell(source):
    lines = [s + "\n" for s in source.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }


def code_cell(source):
    lines = [s + "\n" for s in source.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


def build_01_eda():
    cells = [
        markdown_cell(
            "# WattWise AI - Step 1: Exploratory Data Analysis (London Smart Meters)\n"
            "This notebook documents the ingestion, data quality verification, and consumption characteristics "
            "of the London Smart Meters dataset (UK Power Networks).\n\n"
            "**Key Objectives:**\n"
            "1. Missing-value summary and data-quality assessment\n"
            "2. Average daily profile (diurnal shape, morning and evening peaks)\n"
            "3. Weekday versus weekend behavioral comparisons\n"
            "4. Electricity consumption versus ambient temperature (heating sensitivity)\n"
            "5. Distribution of consumption across households"
        ),
        code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "sns.set_theme(style='whitegrid', palette='muted')\n"
            "plt.rcParams['figure.figsize'] = (12, 5)"
        ),
        markdown_cell("## 1. Load Processed Hourly and Daily Datasets"),
        code_cell(
            "df_hourly = pd.read_parquet('../data/processed/hourly.parquet')\n"
            "df_daily = pd.read_parquet('../data/processed/daily.parquet')\n\n"
            "print(f'Hourly records: {len(df_hourly):,}, Households: {df_hourly[\"household_id\"].nunique()}')\n"
            "print(f'Daily records:  {len(df_daily):,}, Households: {df_daily[\"household_id\"].nunique()}')\n"
            "df_hourly.head()"
        ),
        markdown_cell("## 2. Missing-Value and Data-Quality Summary"),
        code_cell(
            "from src.preprocess import generate_data_quality_summary\n"
            "summary = generate_data_quality_summary(df_hourly)\n"
            "for k, v in summary.items():\n"
            "    print(f'{k}: {v}')"
        ),
        markdown_cell("## 3. Average Daily Profile (Diurnal Curve)"),
        code_cell(
            "df_hourly['hour'] = pd.to_datetime(df_hourly['timestamp']).dt.hour\n"
            "hourly_profile = df_hourly.groupby('hour')['kwh'].agg(['mean', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)])\n"
            "hourly_profile.columns = ['mean', 'median', 'p25', 'p75']\n\n"
            "plt.figure(figsize=(11, 4.5))\n"
            "plt.plot(hourly_profile.index, hourly_profile['mean'], label='Mean Consumption', color='#1f77b4', lw=2.5)\n"
            "plt.plot(hourly_profile.index, hourly_profile['median'], label='Median Consumption', color='#2ca02c', lw=2, linestyle='--')\n"
            "plt.fill_between(hourly_profile.index, hourly_profile['p25'], hourly_profile['p75'], color='#1f77b4', alpha=0.15, label='IQR (25th - 75th percentile)')\n"
            "plt.title('London Smart Meters: Average Diurnal Electricity Profile', fontsize=13, fontweight='bold')\n"
            "plt.xlabel('Hour of Day', fontsize=11)\n"
            "plt.ylabel('Hourly Electricity (kWh)', fontsize=11)\n"
            "plt.xticks(range(0, 24))\n"
            "plt.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        markdown_cell("## 4. Weekday vs Weekend Comparison"),
        code_cell(
            "df_hourly['dayofweek'] = pd.to_datetime(df_hourly['timestamp']).dt.dayofweek\n"
            "df_hourly['day_type'] = np.where(df_hourly['dayofweek'] >= 5, 'Weekend', 'Weekday')\n\n"
            "day_profile = df_hourly.groupby(['day_type', 'hour'])['kwh'].mean().unstack(level=0)\n\n"
            "plt.figure(figsize=(11, 4.5))\n"
            "plt.plot(day_profile.index, day_profile['Weekday'], label='Weekday (Mon-Fri)', color='#1f77b4', lw=2.2)\n"
            "plt.plot(day_profile.index, day_profile['Weekend'], label='Weekend (Sat-Sun)', color='#ff7f0e', lw=2.2, linestyle='-')\n"
            "plt.title('Weekday vs Weekend Electricity Usage Comparison', fontsize=13, fontweight='bold')\n"
            "plt.xlabel('Hour of Day', fontsize=11)\n"
            "plt.ylabel('Average Hourly Electricity (kWh)', fontsize=11)\n"
            "plt.xticks(range(0, 24))\n"
            "plt.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        markdown_cell("## 5. Consumption vs Outdoor Temperature"),
        code_cell(
            "plt.figure(figsize=(9, 5))\n"
            "sns.regplot(data=df_daily.sample(min(len(df_daily), 5000), random_state=42), x='temperature', y='kwh', scatter_kws={'alpha': 0.25, 'color': '#3182bd'}, line_kws={'color': '#de2d26', 'lw': 2})\n"
            "plt.title('Daily Electricity Consumption vs Ambient Temperature', fontsize=13, fontweight='bold')\n"
            "plt.xlabel('Ambient Temperature (°C)', fontsize=11)\n"
            "plt.ylabel('Daily Electricity Consumption (kWh)', fontsize=11)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        markdown_cell("## 6. Distribution Across Households"),
        code_cell(
            "household_daily_mean = df_daily.groupby('household_id')['kwh'].mean()\n"
            "plt.figure(figsize=(9, 4.5))\n"
            "sns.histplot(household_daily_mean, bins=25, kde=True, color='#2b8cbe')\n"
            "plt.title('Distribution of Average Daily Consumption Across Households', fontsize=13, fontweight='bold')\n"
            "plt.xlabel('Mean Daily Electricity (kWh)', fontsize=11)\n"
            "plt.ylabel('Household Count', fontsize=11)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    ]
    return make_notebook(cells)


def build_02_anomaly():
    cells = [
        markdown_cell(
            "# WattWise AI - Step 2: Anomaly Detection with Isolation Forest\n"
            "This notebook details the training, threshold tuning, synthetic anomaly injection, "
            "and baseline benchmark comparison of the WattWise AI Anomaly Detector."
        ),
        code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import joblib\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "from src.features import build_features\n"
            "from src.anomaly import train_isolation_forest, score, inject_anomalies, zscore_baseline, evaluate_anomaly_detection"
        ),
        markdown_cell("## 1. Load Processed Hourly Data & Compute Engineered Features"),
        code_cell(
            "df_hourly = pd.read_parquet('../data/processed/hourly.parquet')\n"
            "df_features = build_features(df_hourly.head(5000))\n"
            "df_features.head()"
        ),
        markdown_cell("## 2. Quantitative Evaluation: Isolation Forest vs Z-Score Baseline"),
        code_cell(
            "metrics_df, details = evaluate_anomaly_detection(df_hourly, contamination=0.03, train_split_ratio=0.75, reports_dir='../reports')\n"
            "print(metrics_df.to_string(index=False))"
        ),
        markdown_cell("## 3. Visualizing Detected Anomalies on Sample Timeline"),
        code_cell(
            "model = joblib.load('../models/isolation_forest.joblib')\n"
            "scored_df = score(model, df_features)\n"
            "h_id = scored_df['household_id'].iloc[0]\n"
            "sample = scored_df[scored_df['household_id'] == h_id].iloc[:168]\n\n"
            "plt.figure(figsize=(13, 4.5))\n"
            "plt.plot(sample['timestamp'], sample['kwh'], label='Hourly Consumption', color='#1f77b4', lw=1.5)\n"
            "anomalies = sample[sample['anomaly_flag']]\n"
            "plt.scatter(anomalies['timestamp'], anomalies['kwh'], color='red', s=50, label='Flagged Anomaly', zorder=5)\n"
            "plt.title(f'Isolation Forest Detected Anomalies - Household {h_id}', fontsize=12, fontweight='bold')\n"
            "plt.xlabel('Timestamp')\n"
            "plt.ylabel('kWh')\n"
            "plt.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    ]
    return make_notebook(cells)


def build_03_forecast():
    cells = [
        markdown_cell(
            "# WattWise AI - Step 3: Multi-Step Forecasting & Bill Estimation\n"
            "This notebook details the Random Forest autoregressive forecasting architecture, "
            "uncertainty quantification, naive benchmark evaluation, and tariff bill estimation."
        ),
        code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import joblib\n"
            "import matplotlib.pyplot as plt\n\n"
            "from src.forecast import make_forecast_features, train_rf, predict, evaluate_forecasting\n"
            "from src.bill import estimate_bill"
        ),
        markdown_cell("## 1. Load Hourly Dataset & Evaluate Forecaster"),
        code_cell(
            "df_hourly = pd.read_parquet('../data/processed/hourly.parquet')\n"
            "metrics_df, details = evaluate_forecasting(df_hourly, test_days=30, reports_dir='../reports')\n"
            "print(metrics_df.to_string(index=False))"
        ),
        markdown_cell("## 2. 30-Day Multi-Step Prediction with Confidence Spread"),
        code_cell(
            "rf_model = joblib.load('../models/rf_forecast.joblib')\n"
            "h_id = df_hourly['household_id'].iloc[0]\n"
            "h_data = df_hourly[df_hourly['household_id'] == h_id].tail(30 * 24)\n"
            "forecast_30d = predict(rf_model, h_data, horizon_days=30)\n\n"
            "plt.figure(figsize=(14, 5))\n"
            "plt.plot(forecast_30d['timestamp'], forecast_30d['predicted_kwh'], label='RF Forecast (Mean)', color='#2ca02c')\n"
            "plt.fill_between(forecast_30d['timestamp'], forecast_30d['lower_bound'], forecast_30d['upper_bound'], color='#2ca02c', alpha=0.2, label='10th - 90th Percentile Confidence Spread')\n"
            "plt.title('WattWise AI: 30-Day Ahead Electricity Forecast with Uncertainty Bounds', fontsize=13, fontweight='bold')\n"
            "plt.xlabel('Timestamp')\n"
            "plt.ylabel('Predicted Hourly kWh')\n"
            "plt.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        markdown_cell("## 3. Bill Estimation with Published UK Tariffs"),
        code_cell(
            "bill = estimate_bill(forecast_30d, tariff_per_kwh=0.28, standing_charge_per_day=0.60)\n"
            "for k, v in bill.items():\n"
            "    print(f'{k}: {v}')"
        ),
    ]
    return make_notebook(cells)


def generate_all_notebooks():
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    nb1 = out_dir / "01_eda.ipynb"
    nb2 = out_dir / "02_anomaly.ipynb"
    nb3 = out_dir / "03_forecast.ipynb"

    with open(nb1, "w", encoding="utf-8") as f:
        json.dump(build_01_eda(), f, indent=2)
    with open(nb2, "w", encoding="utf-8") as f:
        json.dump(build_02_anomaly(), f, indent=2)
    with open(nb3, "w", encoding="utf-8") as f:
        json.dump(build_03_forecast(), f, indent=2)

    print(f"Generated notebooks:\n - {nb1}\n - {nb2}\n - {nb3}")


if __name__ == "__main__":
    generate_all_notebooks()
