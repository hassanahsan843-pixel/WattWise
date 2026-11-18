"""
WattWise AI - Streamlit Web Application
End-to-End Smart Meter Anomaly Detection, Bill Forecasting & Energy Recommendations.
"""

from pathlib import Path
import sys

# Ensure root directory is accessible for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import warnings
import pandas as pd
try:
    pd.set_option("io.parquet.engine", "fastparquet")
except Exception:
    pass
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.anomaly import score, train_isolation_forest
from src.bill import estimate_avoidable_cost, estimate_bill
from src.explain import explain_anomaly
from src.features import build_features
from src.forecast import make_forecast_features, predict, train_rf
from src.preprocess import clean, merge_context, resample_readings
from src.recommend import recommend

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="WattWise AI - Smart Meter Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #10B981;
    }
    .rec-box {
        border-left: 4px solid #3B82F6;
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Caching: Models & Datasets
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Load pre-trained Isolation Forest and Random Forest models."""
    if_path = Path("models/isolation_forest.joblib")
    rf_path = Path("models/rf_forecast.joblib")

    if_model = joblib.load(if_path) if if_path.exists() else None
    rf_model = joblib.load(rf_path) if rf_path.exists() else None
    return if_model, rf_model


@st.cache_data(show_spinner=False)
def load_reference_dataset():
    """Load processed London Smart Meters hourly data."""
    hourly_p = Path("data/processed/hourly.parquet")
    if hourly_p.exists():
        return pd.read_parquet(hourly_p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_report_metrics():
    """Load saved evaluation metrics from reports directory."""
    anom_p = Path("reports/anomaly_metrics.csv")
    fc_p = Path("reports/forecast_metrics.csv")
    anom_df = pd.read_csv(anom_p) if anom_p.exists() else None
    fc_df = pd.read_csv(fc_p) if fc_p.exists() else None
    return anom_df, fc_df


# ---------------------------------------------------------
# Sidebar Inputs & Controls
# ---------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/electricity.png", width=64
)
st.sidebar.title("WattWise AI")
st.sidebar.markdown("**Intelligent Energy Analytics Platform**")

data_source = st.sidebar.radio(
    "Data Source Mode",
    options=["Select Household from Dataset", "Upload Custom Meter CSV"],
    index=0,
)

df_ref = load_reference_dataset()
selected_df = pd.DataFrame()

if data_source == "Select Household from Dataset":
    if not df_ref.empty:
        households = sorted(df_ref["household_id"].unique().tolist())
        selected_household = st.sidebar.selectbox(
            "Select Household ID", households, index=0
        )
        selected_df = df_ref[
            df_ref["household_id"] == selected_household
        ].copy()
    else:
        st.sidebar.warning(
            "No pre-processed dataset found. Please run the data pipeline or upload a CSV."
        )
else:
    # Option to download sample valid file for easy testing
    sample_file_path = Path("data/sample_test_files/representative_valid_household_30d.csv")
    if sample_file_path.exists():
        with open(sample_file_path, "rb") as fp:
            st.sidebar.download_button(
                label="📥 Download Sample Meter CSV",
                data=fp,
                file_name="representative_valid_household_30d.csv",
                mime="text/csv",
                help="Click to download a valid 30-day smart meter CSV for testing upload."
            )

    uploaded_file = st.sidebar.file_uploader(
        "Upload Smart Meter CSV", type=["csv", "parquet"]
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".parquet"):
                raw_up = pd.read_parquet(uploaded_file)
            else:
                raw_up = pd.read_csv(uploaded_file)

            # Standardize columns
            col_map = {
                "LCLid": "household_id",
                "lclid": "household_id",
                "household_id": "household_id",
                "tstp": "timestamp",
                "DateTime": "timestamp",
                "timestamp": "timestamp",
                "energy(kWh/hh)": "kwh",
                "energy": "kwh",
                "kWh": "kwh",
                "kwh": "kwh",
            }
            raw_up = raw_up.rename(
                columns={c: col_map[c] for c in raw_up.columns if c in col_map}
            )

            if "household_id" not in raw_up.columns:
                raw_up["household_id"] = "UPLOADED_HOME"

            # Clean and resample if half-hourly
            cleaned = clean(raw_up)
            hourly_up = resample_readings(cleaned, freq="h")
            enriched_up = merge_context(hourly_up)
            selected_df = enriched_up
            st.sidebar.success(
                f"Successfully loaded {len(selected_df):,} hourly readings!"
            )
        except Exception as e:
            st.sidebar.error(f"Error parsing uploaded file: {str(e)}")

# Date Range Filtering
if not selected_df.empty:
    selected_df["timestamp"] = pd.to_datetime(selected_df["timestamp"])
    min_date = selected_df["timestamp"].min().date()
    max_date = selected_df["timestamp"].max().date()

    date_range = st.sidebar.date_input(
        "Analysis Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
else:
    date_range = None

# Tariff & Model Settings
st.sidebar.subheader("Tariff Configuration (£)")
tariff_per_kwh = st.sidebar.number_input(
    "Unit Electricity Rate (£/kWh)",
    min_value=0.05,
    max_value=1.50,
    value=0.28,
    step=0.01,
    help="Default based on UK Ofgem Price Cap (£0.28/kWh).",
)
standing_charge_per_day = st.sidebar.number_input(
    "Daily Standing Charge (£/day)",
    min_value=0.00,
    max_value=3.00,
    value=0.60,
    step=0.05,
    help="Fixed daily utility connection charge (£0.60/day).",
)

st.sidebar.subheader("Model Hyperparameters")
contamination = st.sidebar.slider(
    "Anomaly Contamination Sensitivity",
    min_value=0.01,
    max_value=0.10,
    value=0.03,
    step=0.005,
    help="Expected percentage of consumption anomalies in the series.",
)
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=7,
    max_value=30,
    value=30,
    step=1,
    help="Number of future days to forecast.",
)

# ---------------------------------------------------------
# Main Page Header & Tabs
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">WattWise AI: Smart Energy Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Unsupervised Anomaly Detection • Autoregressive Multi-Step Bill Forecasting • Actionable Energy Reduction</div>',
    unsafe_allow_html=True,
)

tab_analysis, tab_forecast, tab_recommend, tab_about = st.tabs([
    "📊 Anomaly Detection & Diagnostics",
    "📈 Bill Forecasting & Projection",
    "💡 Energy Reduction Actions",
    "ℹ️ Model Performance & Methodology",
])


# ---------------------------------------------------------
# Run Analysis Trigger & Validation
# ---------------------------------------------------------
run_button = st.sidebar.button(
    "🚀 Run Full Analysis", type="primary", use_container_width=True
)

if not run_button and "analysis_performed" not in st.session_state:
    st.info(
        "👋 Welcome to **WattWise AI**! Select a household or upload a smart meter CSV in the sidebar, "
        "configure your parameters, and click **🚀 Run Full Analysis** to execute the ML pipeline."
    )
    # Preview sample data
    if not selected_df.empty:
        st.subheader("Data Preview")
        st.dataframe(selected_df.head(10), use_container_width=True)
else:
    # Validate inputs
    if selected_df.empty:
        st.error(
            "❌ Validation Error: No smart meter data is available. Please select a household or upload a file."
        )
        st.stop()

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_ts = pd.to_datetime(date_range[0])
        end_ts = pd.to_datetime(date_range[1]) + pd.Timedelta(
            days=1
        ) - pd.Timedelta(seconds=1)
        active_df = selected_df[
            (selected_df["timestamp"] >= start_ts)
            & (selected_df["timestamp"] <= end_ts)
        ].copy()
    else:
        active_df = selected_df.copy()

    # Required validation: at least 14 days of data
    timespan_days = (
        active_df["timestamp"].max() - active_df["timestamp"].min()
    ).total_seconds() / 86400.0
    if timespan_days < 14:
        st.error(
            f"❌ Validation Error: Dataset span is only {timespan_days:.1f} days. "
            f"WattWise AI requires at least 14 continuous days of smart meter readings to construct autoregressive lags (lag_24h, lag_168h)."
        )
        st.stop()

    required_cols = {"timestamp", "kwh"}
    if not required_cols.issubset(set(active_df.columns)):
        st.error(
            f"❌ Validation Error: Missing required columns: {required_cols - set(active_df.columns)}"
        )
        st.stop()

    with st.spinner("Executing Feature Engineering and Model Inference..."):
        # 1. Feature Engineering
        featured_df = build_features(active_df)

        # 2. Anomaly Detection
        if_model, rf_model = load_models()
        if (
            if_model is None
            or abs(if_model.contamination - contamination) > 0.005
        ):
            # Retrain if contamination changed or model not yet saved
            if_model = train_isolation_forest(
                featured_df, contamination=contamination, save_path=None
            )

        scored_df = score(if_model, featured_df)

        # 3. Explanations & Recommendations
        profile_df = (
            featured_df.groupby(["dayofweek", "hour"])["profile_median"]
            .mean()
            .reset_index()
        )
        anomalies_only = scored_df[scored_df["anomaly_flag"]].copy()

        explanations = []
        all_recommendations = []

        for _, row in anomalies_only.iterrows():
            exp = explain_anomaly(row, household_profile=profile_df)
            explanations.append(exp)
            excess = max(0.0, exp["actual_kwh"] - exp["normal_kwh"])
            recs = recommend(
                exp["pattern_type"],
                magnitude=excess,
                tariff_per_kwh=tariff_per_kwh,
            )
            all_recommendations.extend(recs)

        anom_report_df = pd.DataFrame(explanations)

        # 4. Forecasting next horizon_days
        if rf_model is None:
            fc_train = make_forecast_features(active_df)
            rf_model = train_rf(fc_train, save_path=None)

        forecast_df = predict(
            rf_model, history=active_df, horizon_days=forecast_horizon
        )

        # 5. Bill Calculations
        bill_metrics = estimate_bill(
            kwh_series=forecast_df,
            tariff_per_kwh=tariff_per_kwh,
            standing_charge_per_day=standing_charge_per_day,
            n_days=forecast_horizon,
        )

        # Avoidable cost calculation
        total_anom_kwh = (
            anomalies_only["kwh"].sum() if not anomalies_only.empty else 0.0
        )
        expected_anom_kwh = (
            anom_report_df["normal_kwh"].sum()
            if not anom_report_df.empty
            else 0.0
        )
        avoidable_cost = estimate_avoidable_cost(
            total_anom_kwh, expected_anom_kwh, tariff_per_kwh
        )

        st.session_state["analysis_performed"] = True

    # ---------------------------------------------------------
    # KPI Metric Cards
    # ---------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Predicted {forecast_horizon}-Day Bill</div>
            <div class="metric-value">£{bill_metrics['total_bill']:.2f}</div>
            <div class="metric-sub">Range: £{bill_metrics['lower_bill']:.2f} – £{bill_metrics['upper_bill']:.2f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with m2:
        n_anom = len(anomalies_only)
        pct_anom = (n_anom / len(scored_df) * 100) if len(scored_df) > 0 else 0
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Detected Anomalies</div>
            <div class="metric-value" style="color: {'#DC2626' if n_anom > 0 else '#10B981'};">{n_anom}</div>
            <div class="metric-sub">{pct_anom:.2f}% of monitored hours flagged</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Avoidable Cost</div>
            <div class="metric-value" style="color: #D97706;">£{avoidable_cost['avoidable_cost_gbp']:.2f}</div>
            <div class="metric-sub">{avoidable_cost['excess_kwh']:.1f} excess kWh wasted</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with m4:
        avg_daily = (
            active_df["kwh"].sum() / timespan_days if timespan_days > 0 else 0
        )
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Historical Mean Daily</div>
            <div class="metric-value">{avg_daily:.2f} <span style="font-size: 1rem; color: #6B7280;">kWh/day</span></div>
            <div class="metric-sub">£{avg_daily * tariff_per_kwh + standing_charge_per_day:.2f} per day</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 1: Anomaly Detection Visuals & Table
    # ---------------------------------------------------------
    with tab_analysis:
        st.subheader("Interactive Electricity Consumption & Detected Anomalies")

        # Plotly consumption time series with red anomaly markers
        fig_anom = go.Figure()
        fig_anom.add_trace(
            go.Scatter(
                x=scored_df["timestamp"],
                y=scored_df["kwh"],
                mode="lines",
                name="Hourly Consumption (kWh)",
                line=dict(color="#2563EB", width=1.5),
            )
        )

        if not anomalies_only.empty:
            fig_anom.add_trace(
                go.Scatter(
                    x=anomalies_only["timestamp"],
                    y=anomalies_only["kwh"],
                    mode="markers",
                    name="Flagged Anomaly",
                    marker=dict(
                        color="#DC2626",
                        size=8,
                        symbol="circle",
                        line=dict(width=1, color="#7F1D1D"),
                    ),
                    text=[
                        f"Score: {s:.3f}<br>Ratio: {r:.2f}x"
                        for s, r in zip(
                            anomalies_only["anomaly_score"],
                            anomalies_only["ratio_to_household_profile"],
                        )
                    ],
                    hoverinfo="x+y+text+name",
                )
            )

        fig_anom.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Timeline",
            yaxis_title="Electricity Usage (kWh)",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig_anom, use_container_width=True)

        st.subheader("Detailed Anomaly Diagnostics & Explainability")
        if not anom_report_df.empty:
            display_cols = [
                "timestamp",
                "actual_kwh",
                "normal_kwh",
                "multiplier_vs_normal",
                "pattern_type",
                "explanation_text",
            ]
            formatted_table = anom_report_df[display_cols].rename(
                columns={
                    "timestamp": "Timestamp",
                    "actual_kwh": "Measured (kWh)",
                    "normal_kwh": "Baseline (kWh)",
                    "multiplier_vs_normal": "Multiplier",
                    "pattern_type": "Pattern Classification",
                    "explanation_text": "Plain-English Contextual Explanation",
                }
            )
            st.dataframe(
                formatted_table, use_container_width=True, hide_index=True
            )
        else:
            st.success(
                "✅ No abnormal consumption spikes or drops detected within this timeframe!"
            )

    # ---------------------------------------------------------
    # TAB 2: Multi-Step Forecasting & Bill Projections
    # ---------------------------------------------------------
    with tab_forecast:
        st.subheader(
            f"Multi-Step Ahead Consumption Forecast ({forecast_horizon} Days)"
        )

        fig_fc = go.Figure()
        # Historical context (last 7 days)
        recent_hist = active_df.tail(7 * 24)
        fig_fc.add_trace(
            go.Scatter(
                x=recent_hist["timestamp"],
                y=recent_hist["kwh"],
                mode="lines",
                name="Recent History",
                line=dict(color="#64748B", width=1.5),
            )
        )

        # Upper bound
        fig_fc.add_trace(
            go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["upper_bound"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                name="Upper Bound (90th percentile)",
            )
        )
        # Lower bound with fill
        fig_fc.add_trace(
            go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["lower_bound"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.18)",
                name="Prediction Spread (10th - 90th percentile)",
            )
        )
        # Predicted Mean
        fig_fc.add_trace(
            go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["predicted_kwh"],
                mode="lines",
                name="Random Forest Forecast",
                line=dict(color="#059669", width=2.2),
            )
        )

        fig_fc.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Date / Time",
            yaxis_title="Electricity Consumption (kWh)",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Bill Breakdown Table
        st.subheader("Financial Bill Projection Breakdown")
        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            bill_table = pd.DataFrame({
                "Component": [
                    "Forecast Energy Usage",
                    "Unit Electricity Rate",
                    "Energy Consumption Cost",
                    "Standing Access Charge",
                    "Total Estimated Bill",
                    "Conservative Lower Bound",
                    "Upper Bound (Cold / Peak)",
                ],
                "Value": [
                    f"{bill_metrics['total_kwh']:.2f} kWh",
                    f"£{tariff_per_kwh:.4f} / kWh",
                    f"£{bill_metrics['unit_cost']:.2f}",
                    f"£{bill_metrics['standing_cost']:.2f} (£{standing_charge_per_day:.2f}/day)",
                    f"£{bill_metrics['total_bill']:.2f}",
                    f"£{bill_metrics['lower_bill']:.2f}",
                    f"£{bill_metrics['upper_bill']:.2f}",
                ],
            })
            st.table(bill_table)

        with col_b2:
            # Donut chart of bill composition
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=["Unit Energy Usage", "Daily Standing Charge"],
                        values=[
                            bill_metrics["unit_cost"],
                            bill_metrics["standing_cost"],
                        ],
                        hole=0.6,
                        marker=dict(colors=["#0284C7", "#94A3B8"]),
                    )
                ]
            )
            fig_donut.update_layout(
                title="Bill Composition",
                height=260,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: Recommendations & Energy Savings
    # ---------------------------------------------------------
    with tab_recommend:
        st.subheader("Personalized Energy Reduction Recommendations")
        st.write(
            "These targeted energy-efficiency actions are algorithmically mapped from your detected anomaly patterns:"
        )

        if all_recommendations:
            # Deduplicate by action text
            seen_actions = set()
            unique_recs = []
            for r in all_recommendations:
                if r["action"] not in seen_actions:
                    seen_actions.add(r["action"])
                    unique_recs.append(r)

            total_potential_kwh = sum(
                r["estimated_monthly_saving_kwh"] for r in unique_recs
            )
            total_potential_gbp = sum(
                r["estimated_monthly_saving_gbp"] for r in unique_recs
            )

            st.success(
                f"🌱 **Total Potential Savings: ~{total_potential_kwh:.1f} kWh/month (saving ~£{total_potential_gbp:.2f}/month)**"
            )

            for rec in unique_recs:
                st.markdown(
                    f"""
                <div class="rec-box">
                    <div style="font-weight: 700; font-size: 1.05rem; color: #1E40AF;">{rec['action']}</div>
                    <div style="color: #374151; font-size: 0.92rem; margin-top: 0.2rem;">{rec['description']}</div>
                    <div style="margin-top: 0.5rem; font-weight: 600; color: #047857; font-size: 0.88rem;">
                        Estimated Monthly Savings: ~{rec['estimated_monthly_saving_kwh']} kWh (£{rec['estimated_monthly_saving_gbp']:.2f})
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "🌟 Excellent efficiency! No major anomalous wastage detected in the current timeframe."
            )

    # ---------------------------------------------------------
    # TAB 4: Model Governance & Academic Documentation
    # ---------------------------------------------------------
    with tab_about:
        st.subheader("Model Performance & Academic Governance")

        anom_metrics_df, fc_metrics_df = load_report_metrics()

        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown(
                "#### Anomaly Detection: Isolation Forest vs Z-Score Baseline"
            )
            if anom_metrics_df is not None:
                st.dataframe(anom_metrics_df, hide_index=True)
            else:
                st.write(
                    "Run `src/run_pipeline.py` to populate performance metrics."
                )

        with c_m2:
            st.markdown(
                "#### Forecasting: Random Forest vs Naive Baseline (Same Day Last Week)"
            )
            if fc_metrics_df is not None:
                st.dataframe(fc_metrics_df, hide_index=True)
            else:
                st.write(
                    "Run `src/run_pipeline.py` to populate performance metrics."
                )

        st.markdown("---")
        st.markdown("### System Architecture & Compliance Details")
        st.markdown("""
        - **Unsupervised Anomaly Isolation:** Isolation Forest recursively isolates multi-dimensional feature coordinates. Outliers require significantly fewer tree cuts to isolate.
        - **Feature Space:** Engineered features include `hour`, `dayofweek`, `temperature`, `rolling_24h_mean/std`, `rolling_7d_mean/std`, `lag_24h`, `lag_168h`, and `ratio_to_household_profile`.
        - **Time-Based Evaluation:** Evaluated on chronologically held-out test splits without data leakage. Synthetic anomalies (overnight load, spikes, drops) benchmark ground truth.
        - **Limitations:** Data is UK-based (2011–2014). Explanations are statistical associations rather than physical telemetry sensors.
        """)
