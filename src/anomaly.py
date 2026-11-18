"""
anomaly.py - Isolation Forest anomaly detection, baseline models, synthetic anomaly injection,
and rigorous time-based evaluation for WattWise AI.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import get_feature_columns, build_features


def train_isolation_forest(
    features_df: pd.DataFrame,
    contamination: float = 0.03,
    save_path: Optional[Union[str, Path]] = "models/isolation_forest.joblib",
    household_id: Optional[str] = None,
    n_estimators: int = 150,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train an Isolation Forest model to isolate electricity consumption anomalies.

    Isolation Forest works by recursively partitioning feature space with random splits.
    Because anomalous usage instances require fewer partitions to isolate (shallow tree depth),
    it provides an unsupervised continuous anomaly score without needing true labels.

    Args:
        features_df: DataFrame containing engineered features.
        contamination: Expected proportion of anomalies (typically 0.01 to 0.05).
        save_path: Optional file destination to persist trained model via joblib.
        household_id: Optional filter to train a per-household model. If None, trains cluster/cohort model.
        n_estimators: Number of isolation trees (150 offers stability and speed).
        random_state: Seed for reproducibility.

    Returns:
        Fitted IsolationForest instance.
    """
    df = features_df.copy()
    if household_id is not None:
        df = df[df["household_id"] == household_id]
        if df.empty:
            raise ValueError(f"No records found for household_id={household_id}")

    feature_cols = get_feature_columns()
    X = df[feature_cols].copy().fillna(0.0)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
        bootstrap=False,
    )
    model.fit(X)

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_p)

    return model


def score(
    model: IsolationForest,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score readings using a fitted Isolation Forest model.

    Args:
        model: Trained IsolationForest instance.
        features_df: DataFrame containing feature columns.

    Returns:
        DataFrame with added columns:
          - anomaly_score: float in [0, 1] where 1 is highly anomalous.
            Computed from decision_function: lower decision_function -> higher anomaly score.
          - anomaly_flag: boolean True (1) if flagged as an anomaly, False (0) otherwise.
    """
    df = features_df.copy()
    feature_cols = get_feature_columns()
    X = df[feature_cols].copy().fillna(0.0)

    # decision_function yields signed anomaly score: negative for anomalies, positive for normal
    raw_scores = model.decision_function(X)
    preds = model.predict(X)  # -1 for anomaly, 1 for normal

    # Invert and normalize to a intuitive [0, 1] range: higher = more anomalous
    # Decision function values are typically centered near 0.1 to -0.2
    normalized_score = 1.0 / (1.0 + np.exp(raw_scores * 12.0))

    df["anomaly_score"] = np.round(normalized_score, 4)
    df["anomaly_flag"] = (preds == -1)

    return df


def inject_anomalies(
    df: pd.DataFrame,
    seed: int = 42,
    anomaly_rate: float = 0.03,
) -> pd.DataFrame:
    """
    Inject synthetic ground-truth anomalies into held-out smart meter data
    to rigorously evaluate detection accuracy (Precision, Recall, F1).

    Three realistic domestic anomaly types are injected:
      1. Spikes: Sudden sharp spikes (e.g., faulty space heater, EV fast charging on wrong circuit, 3.5x-6x).
      2. Sustained overnight usage: High continuous base load between 01:00 and 05:00 (e.g., HVAC stuck on, immersion heater leak).
      3. Sudden drops: Zero/near-zero consumption during normal daytime waking hours (e.g., meter disconnect, power trip).

    Args:
        df: Cleaned consumption DataFrame.
        seed: Random seed for repeatability.
        anomaly_rate: Target proportion of anomalous points to inject.

    Returns:
        DataFrame with an added 'true_anomaly' column (0 for normal, 1 for synthetic anomaly),
        and modified 'kwh' values where anomalies were injected.
    """
    np.random.seed(seed)
    df_injected = df.copy().reset_index(drop=True)
    df_injected["true_anomaly"] = 0
    df_injected["anomaly_type"] = "normal"

    n_rows = len(df_injected)
    target_count = int(n_rows * anomaly_rate)
    count_per_type = max(1, target_count // 3)

    # 1. Spikes: randomly select points throughout the timeline
    spike_candidates = np.random.choice(n_rows, size=count_per_type * 2, replace=False)
    spike_count = 0
    for idx in spike_candidates:
        if spike_count >= count_per_type:
            break
        orig = df_injected.at[idx, "kwh"]
        multiplier = np.random.uniform(3.5, 6.0)
        df_injected.at[idx, "kwh"] = round(orig * multiplier + np.random.uniform(1.5, 3.5), 3)
        df_injected.at[idx, "true_anomaly"] = 1
        df_injected.at[idx, "anomaly_type"] = "spike"
        spike_count += 1

    # 2. Sustained overnight usage: find nighttime hours (01:00 - 05:00)
    timestamps = pd.to_datetime(df_injected["timestamp"])
    overnight_indices = np.where(timestamps.dt.hour.isin([1, 2, 3, 4]))[0]
    overnight_indices = [i for i in overnight_indices if df_injected.at[i, "true_anomaly"] == 0]
    if len(overnight_indices) > count_per_type:
        chosen_overnight = np.random.choice(overnight_indices, size=count_per_type, replace=False)
        for idx in chosen_overnight:
            df_injected.at[idx, "kwh"] = round(df_injected.at[idx, "kwh"] + np.random.uniform(1.2, 2.5), 3)
            df_injected.at[idx, "true_anomaly"] = 1
            df_injected.at[idx, "anomaly_type"] = "sustained_overnight"

    # 3. Sudden drops: find daytime hours (10:00 - 18:00) where consumption was normally > 0.3 kWh
    daytime_indices = np.where(timestamps.dt.hour.between(10, 18))[0]
    daytime_indices = [i for i in daytime_indices if df_injected.at[i, "true_anomaly"] == 0 and df_injected.at[i, "kwh"] > 0.25]
    if len(daytime_indices) > count_per_type:
        chosen_drops = np.random.choice(daytime_indices, size=count_per_type, replace=False)
        for idx in chosen_drops:
            df_injected.at[idx, "kwh"] = round(np.random.uniform(0.00, 0.02), 3)
            df_injected.at[idx, "true_anomaly"] = 1
            df_injected.at[idx, "anomaly_type"] = "sudden_drop"

    return df_injected


def zscore_baseline(
    df: pd.DataFrame,
    threshold: float = 3.0,
    window_hours: int = 24,
) -> pd.DataFrame:
    """
    Simple benchmark baseline: flags anomalies using a rolling Z-score.

    Z = (kwh - rolling_mean) / rolling_std
    Flags as anomaly if |Z| > threshold.

    Args:
        df: Hourly DataFrame.
        threshold: Z-score cutoff (standard is 3.0 standard deviations).
        window_hours: Rolling historical window.

    Returns:
        DataFrame with 'baseline_flag' (boolean) and 'zscore' (float).
    """
    df_out = df.copy()
    results = []

    for _, group in df_out.groupby("household_id"):
        grp = group.sort_values("timestamp").copy()
        rolling_mean = grp["kwh"].shift(1).rolling(window=window_hours, min_periods=6).mean()
        rolling_std = grp["kwh"].shift(1).rolling(window=window_hours, min_periods=6).std().replace(0, np.nan)
        z = (grp["kwh"] - rolling_mean) / rolling_std
        grp["zscore"] = z.fillna(0.0).round(3)
        grp["baseline_flag"] = grp["zscore"].abs() > threshold
        results.append(grp)

    return pd.concat(results, ignore_index=True)


def evaluate_anomaly_detection(
    hourly_df: pd.DataFrame,
    contamination: float = 0.03,
    train_split_ratio: float = 0.75,
    reports_dir: Union[str, Path] = "reports",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Conduct rigorous time-based split evaluation of Isolation Forest vs Rolling Z-Score Baseline.

    1. Splits dataset chronologically (e.g. earliest 75% for training, latest 25% for test).
    2. Injects synthetic anomalies exclusively into the test split.
    3. Fits Isolation Forest on clean training data.
    4. Evaluates both Isolation Forest and Rolling Z-score on the injected test split.
    5. Computes Precision, Recall, F1, and False Positive Rate.
    6. Outputs reports/anomaly_metrics.csv and evaluation plots.

    Returns:
        Tuple of (metrics_summary_dataframe, detailed_metrics_dict).
    """
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    (rep_path / "figures").mkdir(parents=True, exist_ok=True)

    df_sorted = hourly_df.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(df_sorted["timestamp"])
    split_idx = int(len(df_sorted) * train_split_ratio)
    split_timestamp = timestamps.iloc[split_idx]

    train_raw = df_sorted[df_sorted["timestamp"] < split_timestamp].copy()
    test_raw = df_sorted[df_sorted["timestamp"] >= split_timestamp].copy()

    # Build features on clean training data
    train_featured = build_features(train_raw)

    # Train Isolation Forest on train set
    model = train_isolation_forest(
        train_featured,
        contamination=contamination,
        save_path="models/isolation_forest.joblib",
    )

    # Inject anomalies into test split
    test_injected = inject_anomalies(test_raw, seed=42, anomaly_rate=contamination)

    # Build features on test split with injected anomalies
    test_featured = build_features(test_injected)

    # Score with Isolation Forest
    scored_if = score(model, test_featured)

    # Score with Rolling Z-score Baseline
    scored_both = zscore_baseline(scored_if, threshold=3.0)

    y_true = scored_both["true_anomaly"].values
    y_pred_if = scored_both["anomaly_flag"].astype(int).values
    y_pred_base = scored_both["baseline_flag"].astype(int).values

    # Compute metrics
    metrics = {
        "Model": ["Isolation Forest (WattWise)", "Rolling Z-Score Baseline"],
        "Precision": [
            precision_score(y_true, y_pred_if, zero_division=0),
            precision_score(y_true, y_pred_base, zero_division=0),
        ],
        "Recall": [
            recall_score(y_true, y_pred_if, zero_division=0),
            recall_score(y_true, y_pred_base, zero_division=0),
        ],
        "F1-Score": [
            f1_score(y_true, y_pred_if, zero_division=0),
            f1_score(y_true, y_pred_base, zero_division=0),
        ],
        "True Positives": [
            int(confusion_matrix(y_true, y_pred_if).ravel()[3]),
            int(confusion_matrix(y_true, y_pred_base).ravel()[3]),
        ],
        "False Positives": [
            int(confusion_matrix(y_true, y_pred_if).ravel()[1]),
            int(confusion_matrix(y_true, y_pred_base).ravel()[1]),
        ],
        "False Negatives": [
            int(confusion_matrix(y_true, y_pred_if).ravel()[2]),
            int(confusion_matrix(y_true, y_pred_base).ravel()[2]),
        ],
    }

    metrics_df = pd.DataFrame(metrics).round(4)
    csv_path = rep_path / "anomaly_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)

    # Generate Publication Plots
    # Plot 1: Anomaly timeline on a sample household
    sample_h = scored_both["household_id"].iloc[0]
    sample_slice = scored_both[scored_both["household_id"] == sample_h].iloc[:250]

    plt.figure(figsize=(14, 5))
    plt.plot(sample_slice["timestamp"], sample_slice["kwh"], label="Actual kWh", color="#1f77b4", lw=1.5)
    anom_points = sample_slice[sample_slice["anomaly_flag"]]
    plt.scatter(
        anom_points["timestamp"],
        anom_points["kwh"],
        color="#d62728",
        s=45,
        zorder=5,
        label=f"Isolation Forest Anomaly ({len(anom_points)})",
    )
    plt.title(f"WattWise AI: Anomaly Detection Timeline (Household {sample_h})", fontsize=13, fontweight="bold")
    plt.xlabel("Timestamp", fontsize=11)
    plt.ylabel("Electricity Usage (kWh)", fontsize=11)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(rep_path / "figures" / "anomaly_detection_timeline.png", dpi=200)
    plt.close()

    # Plot 2: Anomaly Score Distribution
    plt.figure(figsize=(9, 4.5))
    sns.histplot(
        data=scored_both,
        x="anomaly_score",
        hue="anomaly_flag",
        bins=40,
        palette={False: "#4575b4", True: "#d73027"},
        kde=True,
    )
    plt.title("Distribution of WattWise Isolation Forest Anomaly Scores", fontsize=12, fontweight="bold")
    plt.xlabel("Anomaly Score (0 = Normal, 1 = Severe Anomaly)", fontsize=11)
    plt.ylabel("Reading Count", fontsize=11)
    plt.tight_layout()
    plt.savefig(rep_path / "figures" / "anomaly_score_distribution.png", dpi=200)
    plt.close()

    return metrics_df, {
        "split_timestamp": str(split_timestamp),
        "test_records": len(test_raw),
        "metrics": metrics,
    }
