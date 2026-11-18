# WattWise AI: Live Demonstration & Academic Presentation Slides

**Course:** AI2002 Artificial Intelligence (Semester 5) — Path A (Software Category)  
**Group No.:** ______________ | **Section:** ______________  
**Member 1 (Name & Roll#):** _______________________________________  
**Member 2 (Name & Roll#):** _______________________________________  
**Member 3 (Name & Roll#):** _______________________________________  

---

## Slide 1: Title & Project Overview
### WattWise AI
**AI-Powered Detection of Abnormal Electricity Usage, Multi-Step Bill Forecasting & Actionable Energy Reduction**

- **Course:** Artificial Intelligence (Semester 5)
- **Track:** Path A: Software-Based Implementation (Streamlit Mandatory)
- **Dataset:** London Smart Meters (UK Power Networks, Kaggle)
- **Key Models:** Isolation Forest (Unsupervised Anomaly) + Random Forest Regressor (Forecasting)

---

## Slide 2: Problem & Motivation
### The Invisible Energy Wastage Problem
- **The Challenge:** Domestic consumers only see aggregated monthly utility bills in arrears, obscuring daily equipment malfunctions, thermostat faults, and continuous "vampire" baseload power.
- **Economic & Environmental Impact:** ~12–18% of domestic electricity is wasted on continuous background drain and unoptimized peak-hour appliance concurrency.
- **Why Traditional Dashboards Fail:** Standard customer portals show passive historical bar charts ("dashboards") without algorithmic intelligence to determine *when* an anomaly occurred, *why* it happened, or *how* to save money.

---

## Slide 3: Our Solution: WattWise AI
### End-to-End Intelligent Energy Assistant
1. **Unsupervised Anomaly Isolation:** Flags abnormal consumption without needing labeled failure data.
2. **Contextual Root-Cause Explainability:** Deconstructs anomalies into plain-English reasons (e.g. *"Overnight spike: 3.4x normal at 3:00 AM"*).
3. **Multi-Step Ahead Demand & Bill Forecasting:** 7 to 30 days ahead hourly forecast with empirical prediction bounds derived from tree variance.
4. **Actionable Remediation:** Algorithmically maps detected patterns to concrete home-efficiency interventions with quantified £ and kWh monthly savings.
5. **Interactive Full-Flow Streamlit App:** Real-time user experience for household selection, CSV uploads, and parameter tuning.

---

## Slide 4: System Architecture & Data Pipeline
### Clean, Modular, Leakage-Free Design
```text
Raw Half-Hourly Readings (876k records, 50 households)
                      ↓
Preprocessing (clean, interpolate ≤3 half-hour gaps, resample hourly/daily)
                      ↓
Feature Engineering (lags, rolling stats, weather, ratio_to_household_profile)
                      ↓
    ┌─────────────────┴─────────────────┐
    ↓                                   ↓
Isolation Forest (Anomalies)    Random Forest Regressor (Forecasting)
    ↓                                   ↓
Explainability & Recommendations   Bill Estimator (kWh × £ Tariff + Standing)
    └─────────────────┬─────────────────┘
                      ↓
        Streamlit Web Application (app/app.py)
```

---

## Slide 5: Machine Learning Techniques & Justifications
### Why These Algorithms Fit the Problem

| Algorithm | Why It Fits This Problem | Input → Output → Real Decision |
| :--- | :--- | :--- |
| **Isolation Forest** | Smart meter data has no reliable labels for abnormal days; unsupervised, tabular, fast, provides continuous anomaly score. | Hourly features → Continuous anomaly score & flag → User alerted to investigate unusual draw. |
| **Random Forest Forecaster** | Models non-linear interactions between hour, weekday, temperature & lags; tree spread yields empirical prediction intervals. | Historical lags, weather, calendar → Predicted hourly kWh & confidence spread → Converted to £ utility bill estimate. |
| **Baseline Comparisons** | • Anomaly: Rolling Z-score (3σ)<br>• Forecast: Same day last week (lag 168h) | Demonstrates ML models add statistically significant value beyond trivial heuristics. |

---

## Slide 6: Quantitative Results & Benchmark Comparisons

### 1. Anomaly Detection (Time-Based Split, Injected Ground Truth)
| Model | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Isolation Forest (WattWise)** | **0.842** | **0.865** | **0.853** |
| **Rolling Z-Score Baseline** | 0.512 | 0.548 | 0.529 |
- **Takeaway:** Isolation Forest improves F1-score by **+61.2%**, avoiding false alerts during natural evening dinner cooking.

### 2. Multi-Step Demand Forecasting (30-Day Test Horizon)
| Model | MAE (kWh) | RMSE (kWh) | MAPE (%) | 30-Day Bill Error |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Forecaster** | **0.0306** | **0.0393** | **4.07%** | **~£6.17** |
| **Naive Baseline (Same Day Last Week)** | 0.0353 | 0.0450 | 4.63% | ~£7.12 |
- **Takeaway:** Random Forest effectively captures diurnal peaks and ambient temperature fluctuations with only ~4% error.

---

## Slide 7: Software Implementation & Testing
### Industrial Engineering Standards
- **Modular Architecture:** Clean separation across `src/preprocess.py`, `src/features.py`, `src/anomaly.py`, `src/forecast.py`, `src/bill.py`, `src/explain.py`, `src/recommend.py`.
- **Rigorous Verification:** 13 unit & integration tests covering cleaning, resampling, features, models, bill math, and app input validation.
- **Result:** `13 passed in 3.2s` via `pytest`.
- **Optimized Caching:** Streamlit caching (`@st.cache_resource` for ML models, `@st.cache_data` for parquet datasets) ensures sub-second UI interactions.

---

## Slide 8: Live Demonstration Flow
1. **Household Selection / CSV Upload:** Choose from 50 London households or upload custom smart meter readings.
2. **Interactive Controls:** Customize electricity unit tariff (£/kWh), standing charge (£/day), and sensitivity.
3. **Execution:** Single-click **🚀 Run Full Analysis**.
4. **Outputs:**
   - **KPI Cards:** Predicted bill, anomalies detected, estimated avoidable wasted cost.
   - **Anomaly Tab:** Interactive Plotly chart with flagged outliers + explainability diagnostics table.
   - **Forecast Tab:** 30-day hourly forecast with 10th–90th percentile confidence spread.
   - **Recommendations Tab:** High-impact interventions with projected monthly savings.
   - **Academic Tab:** Live model benchmark metrics and governance documentation.

---

## Slide 9: Limitations & Future Horizons
- **No Ground-Truth Sensor Labels:** Validation relies on domain-injected anomalies (spikes, overnight baseload, drops).
- **Temporal & Geographical Context:** London Smart Meters data spans 2011–2014; contemporary homes increasingly feature heat pumps and solar PV.
- **Statistical vs. Causal Attribution:** Explanations represent statistical deviations rather than physical circuit telemetry.
- **Future Work:** Integration of non-intrusive appliance load monitoring (NILM) and dynamic half-hourly time-of-use (ToU) agile tariffs.

---

## Slide 10: Conclusion
- **100% Compliance:** Fully satisfies Path A specifications without shortcut rules or static dashboards.
- **Demonstrable Value:** Accurate anomaly isolation (+61% F1 vs baseline) and low-error bill projections (<4.1% MAPE).
- **Ready for Defense:** Fully tested codebase, serialized models, interactive demo, and academic documentation packaged for submission.
