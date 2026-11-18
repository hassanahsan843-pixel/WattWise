# Streamlit Community Cloud Deployment Guide

**Evaluation Criterion 6:** Deployment on Streamlit Community Cloud (5 Marks)

This repository is pre-configured for direct, 1-click deployment on **Streamlit Community Cloud** (`share.streamlit.io`).

---

## 🚀 Step-by-Step Deployment Instructions

### Step 1: Push Repository to GitHub
Ensure all code and configurations are pushed to your GitHub repository:
```bash
git remote add origin https://github.com/<your-username>/WattWise-AI.git
git branch -M main
git push -u origin main
```

### Step 2: Open Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click the **"New app"** button.

### Step 3: Configure Application Settings
- **Repository:** `<your-username>/WattWise-AI`
- **Branch:** `main` (or `master`)
- **Main file path:** `app/app.py`
- **App URL:** (Optional) Customize your app subdomain (e.g. `wattwise-ai.streamlit.app`)

### Step 4: Deploy!
Click **"Deploy!"**.
Streamlit Community Cloud automatically:
1. Detects `requirements.txt` and provisions the Python 3.11/3.12 environment.
2. Applies `.streamlit/config.toml` for branding and performance settings.
3. Launches the live interactive web app accessible worldwide.

---

## 📦 Cloud Compatibility Checklist
- [x] Pre-trained models (`models/isolation_forest.joblib` and `models/rf_forecast.joblib`) are included.
- [x] Processed dataset (`data/processed/hourly.parquet`) is compressed and ready.
- [x] `requirements.txt` specifies tested, stable dependency versions.
- [x] CSV file uploader and sample download button are active so users can test immediately.
- [x] Caching decorators (`@st.cache_resource`, `@st.cache_data`) guarantee smooth cloud response times.
