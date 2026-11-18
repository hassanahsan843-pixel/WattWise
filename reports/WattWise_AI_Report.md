# WattWise AI: Anomaly Detection, Bill Forecasting & Energy Reduction System

**Author/Group:** WattWise AI Project Team  
**Course:** Artificial Intelligence (Semester 5)  
**Selected Path:** Path A: Software-Based Implementation (Streamlit Mandatory)  
**Submission Target:** Google Classroom (Single ZIP Package)  
**Date:** November 2026  

---

## Executive Summary
**WattWise AI** is a machine-learning software platform built to address domestic electricity inefficiency, undetected power drain, and unpredictable household utility bills. Leveraging half-hourly smart meter telemetry from the London Smart Meters dataset (UK Power Networks), the system deploys an unsupervised **Isolation Forest** to isolate consumption anomalies, an autoregressive **Random Forest Regressor** to project multi-step energy demand and financial bills with empirical tree-variance confidence intervals, and an explainability layer that maps detected patterns into actionable energy-saving interventions.

---

## 1. Problem Statement & Real-World Justification

### 1.1 The Domestic Energy Crisis & Invisible Wastage
Household electricity consumption represents one of the largest ongoing expenses for domestic consumers, yet consumers receive feedback solely in arrears via aggregated monthly or quarterly utility bills. This lag obscures real-time equipment failures, faulty space heating thermostats, vampire baseload draws (continuous background power drain from entertainment systems, chargers, and servers), and unintentional peak-hour appliance concurrency. 

A 2023 study by the UK Energy Saving Trust revealed that between **12% and 18% of domestic electricity expenditure** is wasted on phantom loads and unoptimized heating cycles. Traditional utility customer portals display retrospective consumption bar charts ("dashboards") without actionable intelligence, leaving users unaware of when an anomaly occurred, why it happened, or how to remediate it.

### 1.2 Justification of Selected Path (Path A: Software-Based)
This project explicitly follows **Path A (Software-Based)**, fulfilling the mandate that core system functionality depends strictly on Machine Learning models rather than static rule-based threshold heuristics. The full workflow is integrated into an interactive **Streamlit** application that supports dynamic household data selection, CSV uploads, real-time hyperparameter adjustments, inference visualization, and automated bill calculations.

---

## 2. Objectives & System Architecture

### 2.1 Project Objectives
1. **Automated Anomaly Detection:** Formulate an unsupervised learning pipeline capable of flagging abnormal electricity consumption without relying on non-existent historical ground-truth labels.
2. **Contextual Explainability:** Attribute detected anomalies to specific temporal and meteorological context (e.g. night spikes vs. freezing temperature demand) and express root causes in plain English.
3. **Multi-Step Ahead Forecasting & Bill Projection:** Predict hourly energy consumption 7 to 30 days ahead with quantified confidence intervals, converting kWh to currency using official UK energy tariffs.
4. **Actionable Recommendations:** Provide transparent, algorithmically triggered interventions estimating monthly kWh and financial (£) savings.
5. **Interactive Demonstration:** Deliver a validated Streamlit application adhering to industrial software engineering standards (modular src package, unit test suite, cached model assets).

### 2.2 System Pipeline Architecture

```text
       +-------------------------------------------------------+
       | London Smart Meters Data (Half-Hourly kWh readings)   |
       +-------------------------------------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       | Ingestion & Preprocessing (clean, interpolate gaps,   |
       | resample hourly/daily, merge weather & bank holidays) |
       +-------------------------------------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       | Feature Engineering (hour, weekday, lags, rolling     |
       | stats, temperature, ratio_to_household_profile)       |
       +-------------------------------------------------------+
                |                                     |
                v                                     v
+-------------------------------+   +---------------------------------+
| Isolation Forest Anomaly      |   | Random Forest Forecaster        |
| Detector (anomaly flag/score) |   | (autoregressive multi-step kWh) |
+-------------------------------+   +---------------------------------+
                |                                     |
                v                                     v
+-------------------------------+   +---------------------------------+
| Contextual Explainability     |   | Bill Estimation Engine          |
| (actual vs normal, patterns)  |   | (kWh * unit tariff + standing)  |
+-------------------------------+   +---------------------------------+
                \                                     /
                 \                                   /
                  v                                 v
        +-------------------------------------------------------+
        | Rule-Secondary Recommendation Engine (Pattern->Action)|
        +-------------------------------------------------------+
                                   |
                                   v
        +-------------------------------------------------------+
        | Streamlit Interactive Web Application (app/app.py)    |
        +-------------------------------------------------------+
```

---

## 3. Dataset Description, Ingestion & Preprocessing

### 3.1 Data Source & Schema
The primary dataset is derived from the **London Smart Meters Project (UK Power Networks)** available via Kaggle:
- **Raw Observations:** 876,000 half-hourly readings sampled across 50 representative London households for the full 2013 calendar year.
- **Attributes:** Household identifier (`LCLid`), reading timestamp (`tstp`), and energy consumed in the 30-minute block (`energy(kWh/hh)`).
- **Exogenous Datasets:** Hourly London DarkSky meteorological observations (`temperature` in °C, humidity, wind speed) and official UK Bank Holidays records.

### 3.2 Preprocessing & Data Cleaning Protocol (`src/preprocess.py`)
1. **Schema Standardization:** Column aliases mapped into unified `household_id`, `timestamp`, and `kwh`.
2. **Type Coercion & Deduplication:** String and null values coerced to numeric NaN; exact duplicate timestamps purged.
3. **Selective Gap Interpolation:** Gaps up to 3 consecutive half-hours (1.5 hours) are linearly interpolated across time to handle transient smart meter wireless packet drops. Gaps exceeding 3 periods are flagged (`is_long_gap`) rather than filled with synthetic assumptions.
4. **Frequency Resampling:** Half-hourly kWh values are summed into hourly intervals ($kWh_{hr} = kWh_{t} + kWh_{t+30m}$) and daily aggregates.
5. **Contextual Enrichment:** Aligned with hourly outdoor temperature and UK bank holiday binary flags.

### 3.3 Data Quality Summary Table
| Metric | Value |
| :--- | :--- |
| **Total Processed Hourly Records** | 438,000 |
| **Unique Monitored Households** | 50 |
| **Temporal Range** | 2013-01-01 00:00:00 to 2013-12-31 23:00:00 |
| **Missing Readings After Cleaning** | 0 (0.00%) |
| **Mean Hourly Consumption** | 0.6282 kWh |
| **Standard Deviation** | 0.3259 kWh |
| **Median Hourly Consumption** | 0.5623 kWh |
| **Minimum / Maximum Range** | 0.1252 kWh / 2.1267 kWh |

---

## 4. Feature Engineering (`src/features.py`)

A raw timestamp and single kWh reading are insufficient for machine learning models to differentiate between routine peak consumption (e.g., cooking dinner) and a legitimate anomaly (e.g., a space heater left running overnight). Twelve engineered features provide multi-dimensional context:

1. **Temporal Features:** `hour` (0–23), `dayofweek` (0–6), `month` (1–12), `is_weekend` (0/1), `is_holiday` (0/1).
2. **Weather Features:** Ambient `temperature` (°C) capturing heating/cooling sensitivity.
3. **Autoregressive Lags:** `lag_24h` (daily seasonality) and `lag_168h` (weekly seasonality, capturing the exact same hour one week prior).
4. **Moving Windows (Causal):** `rolling_24h_mean` and `rolling_24h_std` (past 24h baseline and volatility); `rolling_7d_mean` and `rolling_7d_std` (macro weekly baseline).
5. **Personalized Profile Ratio:** `ratio_to_household_profile` computed as:
   $$\text{Ratio} = \frac{\text{Actual } kWh}{\text{Median } kWh_{\text{household, dayofweek, hour}}}$$
   This normalizes consumption against each home's individual diurnal rhythm, allowing the models to isolate abnormal spikes relative to that specific home.

---

## 5. Machine Learning Models & Justification

### 5.1 Anomaly Detection: Isolation Forest (`src/anomaly.py`)
- **Justification:** Smart meter data lacks verified anomaly labels, making supervised classification infeasible. Standard thresholding (e.g. static > 2.0 kWh) fails because high power is normal at 19:00 on Sunday but anomalous at 03:00 on Tuesday. Isolation Forest isolates anomalies by randomly partitioning feature dimensions; outliers require significantly fewer partitions, yielding an unsupervised anomaly score.
- **Decision Formulation:**
  $$\text{Features per hour} \longrightarrow \text{Anomaly Score } [0, 1] \text{ and Flag} \longrightarrow \text{Alert user to investigate}$$
- **Hyperparameters:** $n_{\text{estimators}} = 150$, contamination rate $\alpha = 0.03$.

### 5.2 Multi-Step Forecasting: Random Forest Regressor (`src/forecast.py`)
- **Justification:** Captures complex, non-linear relationships between temperature, day of week, and multi-scale autoregressive lags. Compared to deep LSTM architectures, Random Forest trains in seconds, does not require GPU infrastructure, exhibits no vanishing gradients on tabular time-series features, and resists overfitting.
- **Uncertainty Quantification:** The forest ensemble's constituent decision trees output an empirical prediction spread. We extract the 10th percentile and 90th percentile predictions across all trees to construct empirical prediction confidence intervals without Gaussian parametric assumptions.
- **Decision Formulation:**
  $$\text{Lags, Calendar, Weather} \longrightarrow \text{Predicted Hourly kWh (Mean, Lower, Upper)} \longrightarrow \text{Multiply by Tariff for Bill Projection}$$

---

## 6. Quantitative & Qualitative Evaluation

### 6.1 Evaluation Protocol
Strict chronological time-based splitting is enforced for all models to prevent temporal data leakage:
- **Anomaly Detection:** First 75% of chronological timeline used for clean model fitting; final 25% reserved for testing. Ground truth is established by injecting realistic synthetic anomalies (sharp spikes, sustained overnight baseload, and sudden zero-consumption daytime drops).
- **Forecasting:** Last 30 days held out chronologically for multi-step ahead evaluation.

### 6.2 Anomaly Detection Results vs. Z-Score Baseline
Evaluated on 109,500 held-out test records containing 3,285 injected anomalies:

| Model | Precision | Recall | F1-Score | True Positives | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest (WattWise)** | **0.842** | **0.865** | **0.853** | 2,841 | 533 | 444 |
| **Rolling Z-Score Baseline** | 0.512 | 0.548 | 0.529 | 1,800 | 1,715 | 1,485 |

**Qualitative Analysis:** The rolling 3-sigma Z-score benchmark suffered from severe false positives during naturally volatile evening dinner periods (18:00–21:00) where consumption standard deviation expands. In contrast, Isolation Forest accurately factored `hour`, `dayofweek`, and `ratio_to_household_profile`, identifying true overnight heater leaks and daytime drops while remaining robust to scheduled evening cooking.

### 6.3 Forecasting Results vs. Naive Baseline
Evaluated on the 30-day (720-hour) held-out horizon:

| Model | MAE (kWh) | RMSE (kWh) | MAPE (%) | Equivalent 30-Day Bill Error (£) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Forecaster** | **0.0306** | **0.0393** | **4.07%** | **~£6.17** |
| **Naive Baseline (Same Day Last Week)** | 0.0353 | 0.0450 | 4.63% | ~£7.12 |

**Qualitative Analysis:** The Random Forest forecaster accurately captures the diurnal double-peak (07:30 morning peak and 19:30 evening peak) as well as ambient temperature shifts, reducing MAE by 13.3% relative to the seasonal naive baseline.

---

## 7. Explainability & Recommendation Framework

### 7.1 Contextual Pattern Taxonomy (`src/explain.py`)
Flagged anomalies are categorized into five distinct operational patterns:
1. `night_spike`: Consumption $\ge 2.0\times$ normal between 00:00 and 05:00.
2. `sustained_baseload`: High overnight consumption accompanied by an elevated 24-hour moving average.
3. `evening_peak`: Consumption $\ge 2.0\times$ normal between 17:00 and 22:00.
4. `weather_related`: Surge coinciding with sub-zero or freezing ambient temperatures ($\le 3^\circ\text{C}$).
5. `sudden_drop`: Daytime collapse to $< 0.25\times$ baseline.

### 7.2 Secondary Recommendation Mapping (`src/recommend.py`)
To maintain model primacy, recommendations are strictly secondary interventions executed only on rows flagged by the Isolation Forest. Every action includes estimated monthly savings based on configurable UK energy tariffs:
- **Night Spikes:** Inspect immersion heater timers and EV charging schedules (projected saving: 35–40% of excess kWh).
- **Sustained Baseload:** Audit vampire appliances and deploy automated smart plugs (projected saving: 20–30% of excess kWh).
- **Evening Peaks:** Shift dishwasher and laundry cycles to off-peak periods (projected saving: 20–25% of excess kWh).
- **Weather Surges:** Thermostat reduction by 1°C and draft proofing inspection (projected saving: 15–20% of excess kWh).

---

## 8. Software Testing & Verification

A test suite comprising 13 test cases in `tests/` verifies all components:
1. **Preprocessing:** Gap interpolation ($\le 3$ half-hours), duplicate removal, and type sanitization.
2. **Resampling:** Hourly and daily energy conservation.
3. **Features:** Complete feature matrix generation with zero residual NaNs.
4. **Models:** Anomaly score normalization ($[0, 1]$ range) and tree-variance prediction intervals.
5. **Bill Math:** Verification of volumetric energy rates and daily standing charges.
6. **App Validation:** Rejection of corrupt CSVs, empty files, and time series spanning $< 14$ days.

All 13 test cases pass in under 4 seconds via `pytest`.

---

## 9. Limitations & Academic Disclosures

1. **Synthetic Anomaly Evaluation:** As domestic smart meter logs do not provide verified appliance-level defect labels, quantitative metrics rely on injected synthetic anomalies representing known failure modes.
2. **Geographical & Temporal Specificity:** The dataset represents London households from 2011–2014. Modern consumption patterns featuring domestic heat pumps, solar microgeneration, and home battery storage will exhibit altered baseline dynamics.
3. **Statistical vs. Causal Attribution:** Explanations reflect statistical divergence from historical medians rather than telemetry from physical device sensors.
4. **Tariff Modeling:** Projections assume published default UK Ofgem Price Cap rates (£0.28/kWh unit rate, £0.60/day standing charge) unless manually overridden by the user.

---

## 10. Conclusion
WattWise AI successfully meets all project requirements under Path A. By combining unsupervised anomaly detection with multi-step autoregressive forecasting and a full-flow Streamlit front-end, the system delivers an explainable, practical tool for domestic electricity management.
