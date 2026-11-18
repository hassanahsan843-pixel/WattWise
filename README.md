# WattWise AI: Intelligent Electricity Anomaly Detection & Bill Forecasting

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.67-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**WattWise AI** is an end-to-end Machine Learning software platform designed to ingest smart meter time series, detect abnormal energy consumption patterns using unsupervised learning, project future energy usage and electricity bills with quantified uncertainty, and deliver transparent energy-saving recommendations.

Developed strictly following **Path A: Software-Based Project** requirements with mandatory **Streamlit** front-end execution.

---

## ⚡ Key Capabilities

1. **Unsupervised Anomaly Detection (`IsolationForest`):**
   - Automatically isolates abnormal spikes, sustained overnight base loads, sudden drops, and weather-driven surges across multi-dimensional feature space.
   - Evaluated against a rolling Z-score baseline using synthetic ground-truth anomaly injection on chronologically held-out test splits.

2. **Autoregressive Multi-Step Forecasting (`RandomForestRegressor`):**
   - Generates 7 to 30 days of hourly electricity demand predictions using autoregressive lags (`lag_24h`, `lag_168h`), moving averages, calendar attributes, and London ambient temperatures.
   - Computes empirical uncertainty confidence bounds (10th to 90th percentile) derived from the prediction variance across decision trees.
   - Evaluated against a rigorous Naive Baseline (*Same day last week*).

3. **Contextual Explainability & Actionable Recommendations:**
   - Deconstructs flagged anomalies by comparing consumption against the household's diurnal baseline (`ratio_to_household_profile`).
   - Translates technical anomalies into plain-English root causes and maps them to transparent efficiency interventions with estimated monthly £ and kWh savings.

4. **Interactive Full-Flow Streamlit Application:**
   - Household selector or drag-and-drop CSV upload (supporting both standard London Smart Meter half-hourly format and normalized CSVs).
   - Real-time parameter tuning (anomaly contamination slider, tariff rates, forecast horizon).
   - Interactive Plotly visualizations and live academic model governance dashboard.

---

## 📁 Repository Structure

```text
wattwise/
├── app/
│   └── app.py                     # Streamlit web application
├── data/
│   ├── raw/                       # London Smart Meters raw samples, weather, holidays
│   └── processed/                 # Cleaned hourly.parquet and daily.parquet datasets
├── models/
│   ├── isolation_forest.joblib    # Serialized Isolation Forest model
│   └── rf_forecast.joblib         # Serialized Random Forest Forecaster model
├── notebooks/
│   ├── 01_eda.ipynb               # Ingestion, data-quality & diurnal profile analysis
│   ├── 02_anomaly.ipynb           # Isolation Forest training, tuning & Z-score benchmark
│   └── 03_forecast.ipynb          # Forecaster training, uncertainty & bill projections
├── reports/
│   ├── figures/                   # Generated evaluation plots (PNG)
│   ├── anomaly_metrics.csv        # Quantitative comparison table (IF vs Baseline)
│   ├── forecast_metrics.csv       # Quantitative comparison table (RF vs Naive)
│   ├── WattWise_AI_Report.md      # Comprehensive academic project report
│   └── WattWise_AI_Slides.md      # Presentation slide deck outline
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # Ingestion, cleaning, gap interpolation & resampling
│   ├── features.py                # Temporal, rolling, lag & profile ratio engineering
│   ├── anomaly.py                 # Isolation Forest, Z-score baseline & synthetic injection
│   ├── forecast.py                # Random Forest multi-step forecaster & naive baseline
│   ├── bill.py                    # Tariff models, bill estimation & avoidable costs
│   ├── explain.py                 # Multiplier calculation & plain-English explanation
│   ├── recommend.py               # Pattern-to-action recommendation catalog
│   └── evaluate.py                # Consolidated evaluation runner
├── tests/
│   ├── test_explain_recommend.py  # Unit tests for explanation and recommendation rules
│   └── test_pipeline.py           # End-to-end integration tests (clean, feat, models, app)
├── requirements.txt               # Pinned package dependencies
└── README.md                      # Project documentation and setup guide
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.11 or 3.12 installed.
- (Recommended) `uv` package manager or standard `pip`.

### 2. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd Project

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset Ingestion & Pipeline Execution

The system utilizes the **London Smart Meters Dataset (UK Power Networks, Kaggle)**:
- Half-hourly domestic consumption readings (`energy(kWh/hh)`).
- London DarkSky hourly weather records (`temperature` in °C).
- Official UK Bank Holidays calendar.

To execute the entire data pipeline (clean, resample to Parquet, engineer features, train models, and generate evaluation reports):

```bash
python -m src.run_pipeline
```

---

## 🖥️ Running the Streamlit Application

Launch the interactive web platform:

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.
1. Select a household (e.g. `MAC000001`) from the sidebar or upload a CSV file.
2. Adjust your tariff (default: £0.28/kWh unit rate + £0.60/day standing charge).
3. Click **🚀 Run Full Analysis**.
4. Explore detected anomalies, bill projections, energy-saving actions, and academic model metrics across the four interactive tabs.

---

## 🧪 Running Unit & Integration Tests

Execute the comprehensive test suite covering all pipeline stages:

```bash
pytest tests/ -v
```

---

## 📈 Quantitative Evaluation Summary

### 1. Anomaly Detection (Time-Based Split with Synthetic Injection)
| Model | Precision | Recall | F1-Score | True Positives | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest (WattWise)** | **0.842** | **0.865** | **0.853** | 1,820 | 341 | 284 |
| **Rolling Z-Score Baseline** | 0.512 | 0.548 | 0.529 | 1,153 | 1,099 | 951 |

*Interpretation:* Isolation Forest achieves an F1-score of 0.853, outperforming the rolling 3-sigma Z-score benchmark (0.529) by **+61.2%**. The baseline suffers from high false-positive rates during naturally volatile evening dinner hours, whereas Isolation Forest's multi-dimensional features correctly contextualize time-of-day and temperature.

### 2. Multi-Step Forecasting (30-Day Chronological Test Horizon)
| Model | MAE (kWh) | RMSE (kWh) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **Random Forest Forecaster (WattWise)** | **0.054** | **0.081** | **14.2%** |
| **Naive Baseline (Same Day Last Week)** | 0.118 | 0.174 | 29.8% |

*Interpretation:* Random Forest halves the forecast error compared to the seasonal naive baseline (MAE 0.054 kWh vs 0.118 kWh), translating to less than **£1.20 bill error** over a 30-day projection cycle.

---

## ⚠️ Known Limitations & Academic Disclosures

1. **Synthetic Anomaly Validation:** Because smart meter datasets do not include verified ground-truth labels for appliance malfunctions, validation relies on synthetically injected domain-accurate anomalies.
2. **Contextual Scope:** The dataset originates from London (2011–2014); modern smart home energy consumption may differ due to increased heat pump and electric vehicle adoption.
3. **Statistical vs. Causal Explanations:** Explanations represent statistical deviations from historical medians rather than telemetry from physical circuit-level sub-meters.
4. **Tariff Assumptions:** Financial projections utilize configurable default tariffs modeled after published UK Ofgem Energy Price Cap figures (£0.28/kWh, £0.60/day standing charge).
