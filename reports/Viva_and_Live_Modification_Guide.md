# WattWise AI: Viva Examination & Live Modification Guide

**Course:** AI2002 Artificial Intelligence  
**Target Rubric Items:**
- **Criterion 8:** Individual Viva (10 Marks)
  - Explains algorithm and its role (3 marks)
  - Traces input through preprocessing and prediction (3 marks)
  - Answers technical questions (4 marks)
- **Criterion 9:** Live Modification (10 Marks)
  - Independent coding ability. Runtime changes.
- **Criterion 11:** Overall Technical Demonstration (10 Marks)
  - Successfully demonstrates actual working ML solution (5 marks)
  - Independently explains key implementation decisions during demonstration (5 marks)

---

## Part 1: Individual Viva - Algorithm Role & Code Trace

### 1.1 The Role of Each Algorithm in WattWise AI

#### Q: Why does WattWise AI use two distinct Machine Learning models?
> **Answer:** Smart meter energy intelligence requires two fundamentally different analytical capabilities:
> 1. **Isolation Forest (Unsupervised Anomaly Detection):** Identifies historical abnormal consumption events (e.g., equipment faults, thermostat malfunctions, forgotten space heaters) where ground-truth labels do not exist.
> 2. **Random Forest Regressor (Supervised Autoregressive Forecaster):** Projects future hourly electricity demand ($t+1$ to $t+720$ hours) using historical lag features, moving averages, and weather dependencies to forecast utility bills.
> 
> *Key principle:* The application is NOT a rule-based dashboard. The anomalies and forecast outputs originate directly from trained ML estimators.

---

### 1.2 End-to-End Code Trace (Input → Preprocessing → Model → Output)

When an evaluator asks: *"Trace a single raw reading through your entire pipeline:"*

```text
Step 1: Ingestion (src/preprocess.py -> load_raw)
  Raw Reading: {"LCLid": "MAC000001", "tstp": "2013-05-10 03:00:00", "energy(kWh/hh)": "1.20"}
  Action: Standardized to household_id, timestamp (datetime64), kwh (float64).

Step 2: Cleaning & Interpolation (src/preprocess.py -> clean)
  Action: Checks for duplicates and missing timestamps. If missing for ≤3 periods (1.5h),
  it linearly interpolates; if longer, it flags is_long_gap.

Step 3: Hourly Resampling (src/preprocess.py -> resample_readings)
  Action: Aggregates two 30-min half-hours into 1 hourly consumption total (kWh).

Step 4: Contextual Enrichment (src/preprocess.py -> merge_context)
  Action: Merges London DarkSky ambient temperature (e.g. 11.0°C) and UK Bank Holiday flag (0).

Step 5: Feature Engineering (src/features.py -> build_features)
  Action: Computes:
    - hour = 3, dayofweek = 4 (Friday), is_weekend = 0
    - lag_24h = consumption yesterday at 03:00 (e.g. 0.18 kWh)
    - lag_168h = consumption last Friday at 03:00 (e.g. 0.20 kWh)
    - rolling_24h_mean = shifted 24h moving average (closed='left' to prevent leakage!)
    - ratio_to_household_profile = actual kWh (1.20) / median for Friday 03:00 (0.20) = 6.0x

Step 6: Anomaly Inference (src/anomaly.py -> score)
  Action: Model evaluates decision_function(X). Because a 6.0x draw at 03:00 isolates in very few
  tree splits, decision_function is negative. Normalized anomaly_score = 0.88, anomaly_flag = True.

Step 7: Explanation & Recommendation (src/explain.py & src/recommend.py)
  Action: Classified as 'night_spike'. Diagnoses: "Abnormal overnight spike: Usage reached 1.20 kWh (6.0x normal) at 03:00."
  Recommendation triggered: "Check timer settings on immersion water heaters." Estimated monthly saving: ~12.5 kWh (£3.50).

Step 8: Forecasting & Bill Projection (src/forecast.py & src/bill.py)
  Action: Random Forest recursive predictor predicts next 30 days hourly kWh.
  Tree spread (10th to 90th percentile) provides confidence bounds.
  Bill Estimator computes: Total Bill = (Total kWh × £0.28) + (30 days × £0.60) = £102.40.
```

---

## Part 2: Top 10 Technical Viva Questions & Model Answers

### Q1: Why use Isolation Forest instead of One-Class SVM or an Autoencoder?
> **Answer:** 
> 1. Isolation Forest scales linearly $\mathcal{O}(n)$, training on 400,000+ tabular records in seconds, whereas One-Class SVM has quadratic/cubic complexity $\mathcal{O}(n^2)$ and is computationally prohibitive.
> 2. Autoencoders require heavy deep learning frameworks, hyperparameter tuning of latent dimensions, and extensive calibration.
> 3. Isolation Forest specifically exploits isolation rather than distance measures, making it immune to distance distortion in high dimensions and robust to tabular smart meter features.

### Q2: How did you prevent Data Leakage between Train and Test sets?
> **Answer:**
> 1. **Time-based Chronological Split:** We strictly split data by time (e.g., first 75% for training, last 25% for testing). We NEVER perform random shuffling (`train_test_split(shuffle=False)`), because random shuffling allows future data to leak into the past.
> 2. **Causal Rolling Statistics:** All rolling means (`rolling_24h_mean`, `rolling_7d_mean`) use `shift(1)` (closed on left). This guarantees the current hour $t$ is NEVER included in its own historical rolling window.

### Q3: How do you evaluate Anomaly Detection without ground truth labels?
> **Answer:** Because smart meter datasets do not come with verified labels for faulty appliances, we use **synthetic anomaly injection** into the held-out test split. We inject three domain-verified failure modes:
> - Sharp spikes (faulty heaters, EV chargers)
> - Sustained overnight baseload (continuous phantom drain)
> - Sudden daytime collapses (meter drops / tripped breakers)
> We then measure **Precision, Recall, and F1-Score** against ground truth and compare against a rolling 3-sigma Z-score benchmark.

### Q4: Why does Isolation Forest outperform the Rolling Z-Score Baseline?
> **Answer:** A Z-score only considers 1-dimensional variance over a rolling window. It triggers numerous false alarms during naturally volatile evening dinner periods (18:00–21:00) where consumption variance naturally spikes. Isolation Forest evaluates multi-dimensional space, taking into account `hour`, `dayofweek`, `temperature`, and `ratio_to_household_profile` simultaneously.

### Q5: How do you obtain uncertainty bounds from Random Forest?
> **Answer:** Random Forest is an ensemble of $B$ independent decision trees (e.g. $B=120$). For any feature vector $x$, each tree $b$ outputs an individual prediction $\hat{y}_b(x)$. Rather than only computing the mean $\frac{1}{B}\sum \hat{y}_b(x)$, we extract the distribution of tree predictions: the $10^{\text{th}}$ percentile forms the lower bound, and the $90^{\text{th}}$ percentile forms the upper bound.

---

## Part 3: Live Modification Exercises (Criterion 9 - 10 Marks)

During the viva/demo, the evaluator may say: *"Modify the code right now to do X and show me the result."* Here are the exact solutions:

---

### Modification Task 1: "Change the anomaly contamination sensitivity or threshold in code"
**File to modify:** `src/anomaly.py` or `app/app.py`
- In `src/anomaly.py` line 25:
  Change `contamination: float = 0.03` to `0.05` or `0.01`.
- In `app/app.py` line 215:
  Adjust slider default: `value=0.05`.
- **Explanation to evaluator:** *"Increasing contamination from 0.03 to 0.05 instructs the model that 5% of observations are expected to be anomalous, broadening the isolation tree cutoff depth and increasing sensitivity."*

---

### Modification Task 2: "Add a new feature: 48-Hour Autoregressive Lag (`lag_48h`)"
**Files to modify:** `src/features.py` and `src/forecast.py`
1. In `src/features.py` inside `build_features()`:
   ```python
   # Add after grp["lag_24h"] = grp["kwh"].shift(24)
   grp["lag_48h"] = grp["kwh"].shift(48).bfill().fillna(grp["profile_median"])
   ```
2. In `src/features.py` inside `get_feature_columns()`:
   ```python
   return [
       "hour", "dayofweek", "is_weekend", "is_holiday", "temperature",
       "rolling_24h_mean", "rolling_24h_std", "rolling_7d_mean", "rolling_7d_std",
       "lag_24h", "lag_48h", "lag_168h", "ratio_to_household_profile"
   ]
   ```
3. Run `pytest tests/test_pipeline.py` to demonstrate test passes with the new feature!

---

### Modification Task 3: "Implement a Two-Tier Day / Night Economy-7 Tariff"
**File to modify:** `src/bill.py`
Replace `estimate_bill()` computation:
```python
def estimate_economy7_bill(hourly_kwh, timestamps, day_rate=0.32, night_rate=0.18, standing=0.60):
    total_cost = 0.0
    for kwh, ts in zip(hourly_kwh, timestamps):
        hour = pd.to_datetime(ts).hour
        # Night rate between 00:00 and 07:00
        rate = night_rate if (0 <= hour < 7) else day_rate
        total_cost += kwh * rate
    days = max(1.0, len(hourly_kwh) / 24.0)
    total_bill = total_cost + (days * standing)
    return {"total_bill": round(total_bill, 2), "unit_cost": round(total_cost, 2)}
```

---

### Modification Task 4: "Add a New Anomaly Pattern: `afternoon_surge`"
**File to modify:** `src/explain.py`
In `explain_anomaly()` after line 70:
```python
elif hour in [13, 14, 15, 16] and multiplier >= 2.5:
    pattern_type = "afternoon_surge"
    explanation_text = f"Unusual afternoon energy surge: Usage surged to {actual_kwh:.2f} kWh ({multiplier:.1f}x normal) at {hour:02d}:00."
```
In `src/recommend.py` add to `RECOMMENDATION_CATALOG`:
```python
"afternoon_surge": [
    {
        "action": "Check solar air conditioning or heavy midday appliances.",
        "description": "Midday spikes often occur when cooling units cycle rapidly during peak temperature hours.",
        "kwh_fraction": 0.25,
    }
]
```

---

### Modification Task 5: "Change Forecast Horizon from 30 Days to 14 Days"
**In Streamlit UI (`app/app.py`):**
Simply drag the sidebar slider **Forecast Horizon (Days)** from 30 to 14, or change the default in line 224: `value=14`.
The model immediately re-predicts for $14 \times 24 = 336$ hours and updates the bill projection table.
