# WattWise AI: Testing Evidence & Reliability Documentation

**Evaluation Criterion 7:** Testing & Reliability (12 Marks)  
- Tests multiple representative inputs (3 marks)
- Tests invalid / boundary cases (2 marks)
- Provides testing evidence (5 marks)
- Consistent outputs (2 marks)

---

## 1. Test Suite Overview
The WattWise AI test suite is automated using `pytest` and contains **14 automated test functions** covering unit, integration, boundary, and regression cases.

### Test Matrix

| Test Module | Test Name | Target Function | Category | Status |
| :--- | :--- | :--- | :--- | :---: |
| `test_explain_recommend.py` | `test_night_spike_explanation_and_recommendation` | `explain_anomaly`, `recommend` | Representative (Spike) | **PASSED** |
| `test_explain_recommend.py` | `test_sustained_baseload_explanation_and_recommendation` | `explain_anomaly`, `recommend` | Representative (Baseload) | **PASSED** |
| `test_explain_recommend.py` | `test_evening_peak_explanation_and_recommendation` | `explain_anomaly`, `recommend` | Representative (Peak) | **PASSED** |
| `test_explain_recommend.py` | `test_weather_related_explanation_and_recommendation` | `explain_anomaly`, `recommend` | Representative (Freezing) | **PASSED** |
| `test_explain_recommend.py` | `test_sudden_drop_explanation_and_recommendation` | `explain_anomaly`, `recommend` | Representative (Drop) | **PASSED** |
| `test_explain_recommend.py` | `test_normal_row_returns_no_recommendations` | `recommend` | Baseline (Zero false alarms) | **PASSED** |
| `test_pipeline.py` | `test_preprocessing_cleaning` | `clean` | Data Integrity (Gaps/Dups) | **PASSED** |
| `test_pipeline.py` | `test_resampling_hourly_and_daily` | `resample_readings` | Energy Conservation | **PASSED** |
| `test_pipeline.py` | `test_feature_engineering` | `build_features` | Lags & Profile Ratio | **PASSED** |
| `test_pipeline.py` | `test_anomaly_model_training_and_scoring` | `score` | Bounded $[0, 1]$ Output | **PASSED** |
| `test_pipeline.py` | `test_forecast_shape_and_uncertainty_bounds` | `predict` | Horizon & Uncertainty | **PASSED** |
| `test_pipeline.py` | `test_bill_estimation_and_avoidable_cost` | `estimate_bill` | Financial Formula Math | **PASSED** |
| `test_pipeline.py` | `test_app_input_validation_rules` | App Validation | Boundary Rejections | **PASSED** |
| `test_pipeline.py` | `test_sample_fixtures_representative_and_boundaries` | File Fixtures | Fixture File Integrity | **PASSED** |

---

## 2. Representative and Boundary Test Fixtures
Located in `data/sample_test_files/`:

1. **`representative_valid_household_30d.csv`**:
   - 30 days (1,440 half-hourly readings) of genuine domestic electricity consumption for London home `MAC000001`.
   - Realistic diurnal curve: morning peak (07:30) and evening cooking peak (19:30).
   - Expected Output: App loads cleanly, flags ~2.5% normal minor outliers, forecasts 30-day bill.

2. **`representative_anomalous_household.csv`**:
   - 30 days of data containing injected extreme anomalies: sharp 3.1 kWh spike (e.g. faulty space heater) and 3 consecutive nights of elevated 1.65 kWh baseload.
   - Expected Output: Isolation Forest flags these specific timestamps; Explainability layer diagnoses `night_spike` and `sustained_baseload`; Recommendation layer outputs vampire power audits.

3. **`boundary_invalid_too_short_5d.csv`**:
   - Boundary condition: spans only 5 days ($< 14$ days required to construct 168-hour weekly lag).
   - Expected Output: Streamlit app displays a clear error notification and halts inference without crashing:
     `❌ Validation Error: Dataset span is only 5.0 days. WattWise AI requires at least 14 continuous days...`

4. **`boundary_invalid_missing_columns.csv`**:
   - Boundary condition: missing required `kwh` or `energy(kWh/hh)` column.
   - Expected Output: App catches missing schema and reports exact missing columns.

5. **`boundary_invalid_empty.csv`**:
   - Boundary condition: 0-byte or 0-row empty file.
   - Expected Output: Catches empty data and prompts user to upload a valid file.

---

## 3. Terminal Execution Evidence (`pytest`)

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0 -- C:\semester 5\AI\Project\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\semester 5\AI\Project
plugins: anyio-4.16.0
collecting ... collected 14 items

tests/test_explain_recommend.py::test_night_spike_explanation_and_recommendation PASSED [  7%]
tests/test_explain_recommend.py::test_sustained_baseload_explanation_and_recommendation PASSED [ 14%]
tests/test_explain_recommend.py::test_evening_peak_explanation_and_recommendation PASSED [ 21%]
tests/test_explain_recommend.py::test_weather_related_explanation_and_recommendation PASSED [ 28%]
tests/test_explain_recommend.py::test_sudden_drop_explanation_and_recommendation PASSED [ 35%]
tests/test_explain_recommend.py::test_normal_row_returns_no_recommendations PASSED [ 42%]
tests/test_pipeline.py::test_preprocessing_cleaning PASSED               [ 50%]
tests/test_pipeline.py::test_resampling_hourly_and_daily PASSED          [ 57%]
tests/test_pipeline.py::test_feature_engineering PASSED                  [ 64%]
tests/test_pipeline.py::test_anomaly_model_training_and_scoring PASSED   [ 71%]
tests/test_pipeline.py::test_forecast_shape_and_uncertainty_bounds PASSED [ 78%]
tests/test_pipeline.py::test_bill_estimation_and_avoidable_cost PASSED   [ 85%]
tests/test_pipeline.py::test_app_input_validation_rules PASSED           [ 92%]
tests/test_pipeline.py::test_sample_fixtures_representative_and_boundaries PASSED [100%]

====================== 14 passed in 3.56s ======================
```

---

## 4. Consistency & Determinism
- Random seeds (`random_state=42`) are fixed across Isolation Forest partitioning and Random Forest tree construction.
- Time-based splits prevent random-seed variance on training and evaluation boundaries.
- Repeated executions yield identical anomaly scores and forecast figures down to 4 decimal places.
