from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modules.ops_store import log_monitoring_results

matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parents[2]
MONITORING_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "monitoring"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference_clean = pd.Series(reference).dropna().astype(float)
    current_clean = pd.Series(current).dropna().astype(float)

    if reference_clean.empty or current_clean.empty:
        return 0.0

    breakpoints = np.unique(
        np.quantile(reference_clean, np.linspace(0, 1, bins + 1))
    )
    if len(breakpoints) < 3:
        return 0.0

    reference_hist, _ = np.histogram(reference_clean, bins=breakpoints)
    current_hist, _ = np.histogram(current_clean, bins=breakpoints)

    reference_ratio = np.clip(reference_hist / max(reference_hist.sum(), 1), 1e-6, None)
    current_ratio = np.clip(current_hist / max(current_hist.sum(), 1), 1e-6, None)

    return float(np.sum((current_ratio - reference_ratio) * np.log(current_ratio / reference_ratio)))


def classify_drift(psi: float) -> str:
    if psi >= 0.20:
        return "high"
    if psi >= 0.10:
        return "moderate"
    return "low"


def compute_monitoring_snapshot(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    dataset_label: str,
    run_id: str | None = None,
) -> pd.DataFrame:
    observed_at_utc = _utc_now_iso()
    resolved_run_id = run_id or str(uuid4())
    rows: list[dict[str, object]] = []

    for column in reference_df.columns:
        psi = calculate_psi(reference_df[column], current_df[column])
        rows.append(
            {
                "run_id": resolved_run_id,
                "observed_at_utc": observed_at_utc,
                "dataset_label": dataset_label,
                "feature_name": column,
                "reference_mean": float(reference_df[column].mean()),
                "current_mean": float(current_df[column].mean()),
                "reference_std": float(reference_df[column].std()),
                "current_std": float(current_df[column].std()),
                "reference_missing_rate": float(reference_df[column].isna().mean()),
                "current_missing_rate": float(current_df[column].isna().mean()),
                "psi": float(psi),
                "drift_severity": classify_drift(psi),
            }
        )

    return pd.DataFrame(rows)


def build_drift_alerts(
    snapshot_df: pd.DataFrame,
    *,
    prediction_rate_reference: float | None = None,
    prediction_rate_current: float | None = None,
) -> pd.DataFrame:
    alert_rows: list[dict[str, object]] = []
    alert_timestamp = _utc_now_iso()

    for _, row in snapshot_df.iterrows():
        if row["drift_severity"] == "low":
            continue
        alert_rows.append(
            {
                "run_id": row["run_id"],
                "alert_timestamp_utc": alert_timestamp,
                "feature_name": row["feature_name"],
                "psi": float(row["psi"]),
                "severity": row["drift_severity"],
                "message": (
                    f"Feature '{row['feature_name']}' shows {row['drift_severity']} drift "
                    f"with PSI={row['psi']:.4f}."
                ),
                "recommended_action": (
                    "Review recent scoring batches and assess whether a retraining cycle should be prepared."
                ),
            }
        )

    if prediction_rate_reference is not None and prediction_rate_current is not None:
        delta = prediction_rate_current - prediction_rate_reference
        if abs(delta) >= 0.08:
            alert_rows.append(
                {
                    "run_id": snapshot_df["run_id"].iloc[0],
                    "alert_timestamp_utc": alert_timestamp,
                    "feature_name": "predicted_churn_rate",
                    "psi": None,
                    "severity": "high" if abs(delta) >= 0.15 else "moderate",
                    "message": (
                        "Predicted churn rate changed materially between the reference and the current batch. "
                        f"Delta={delta:.4f}."
                    ),
                    "recommended_action": (
                        "Investigate operational changes, validate drift severity, and prepare a retraining review."
                    ),
                }
            )

    return pd.DataFrame(alert_rows)


def plot_drift_summary(snapshot_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = snapshot_df.sort_values("psi", ascending=True).copy()
    severity_palette = {
        "low": "#7fb069",
        "moderate": "#f2c14e",
        "high": "#d95d39",
    }
    colors = plot_df["drift_severity"].map(severity_palette)

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature_name"], plot_df["psi"], color=colors)
    plt.axvline(0.10, color="#f2c14e", linestyle="--", linewidth=1.2, label="PSI moderado")
    plt.axvline(0.20, color="#d95d39", linestyle="--", linewidth=1.2, label="PSI alto")
    plt.xlabel("Population Stability Index (PSI)")
    plt.ylabel("")
    plt.title("Resumo visual do drift por feature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_prediction_rate_shift(
    prediction_rate_reference: float | None,
    prediction_rate_current: float | None,
    output_path: Path,
) -> None:
    if prediction_rate_reference is None or prediction_rate_current is None:
        return

    rate_df = pd.DataFrame(
        {
            "dataset": ["baseline", "drifted"],
            "predicted_churn_rate": [prediction_rate_reference, prediction_rate_current],
        }
    )

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=rate_df,
        x="dataset",
        y="predicted_churn_rate",
        hue="dataset",
        palette={"baseline": "#4c78a8", "drifted": "#d95d39"},
        dodge=False,
        legend=False,
    )
    ax.bar_label(ax.containers[0], fmt="%.2f")
    plt.ylim(0, max(rate_df["predicted_churn_rate"].max() * 1.2, 0.1))
    plt.xlabel("")
    plt.ylabel("Predicted churn rate")
    plt.title("Mudanca da taxa prevista de churn")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_alerted_feature_distributions(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    output_path: Path,
) -> None:
    alerted_features = snapshot_df.loc[
        snapshot_df["drift_severity"] != "low", "feature_name"
    ].tolist()[:4]
    if not alerted_features:
        return

    n_features = len(alerted_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, max(3.5 * n_features, 4)), squeeze=False)

    for ax, feature_name in zip(axes.flatten(), alerted_features):
        sns.kdeplot(reference_df[feature_name], ax=ax, label="baseline", fill=True, alpha=0.25, color="#4c78a8")
        sns.kdeplot(current_df[feature_name], ax=ax, label="drifted", fill=True, alpha=0.25, color="#d95d39")
        feature_psi = snapshot_df.loc[snapshot_df["feature_name"] == feature_name, "psi"].iloc[0]
        ax.set_title(f"{feature_name} | PSI={feature_psi:.3f}")
        ax.set_xlabel("")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_monitoring(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    dataset_label: str,
    prediction_rate_reference: float | None = None,
    prediction_rate_current: float | None = None,
    output_dir: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, object]:
    snapshot_df = compute_monitoring_snapshot(
        reference_df,
        current_df,
        dataset_label=dataset_label,
    )
    alerts_df = build_drift_alerts(
        snapshot_df,
        prediction_rate_reference=prediction_rate_reference,
        prediction_rate_current=prediction_rate_current,
    )

    resolved_output_dir = output_dir or MONITORING_ARTIFACTS_DIR
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = resolved_output_dir / "drift_summary.csv"
    alerts_path = resolved_output_dir / "drift_alerts.csv"
    report_path = resolved_output_dir / "drift_report.json"
    drift_overview_plot_path = resolved_output_dir / "drift_overview.png"
    prediction_shift_plot_path = resolved_output_dir / "predicted_churn_rate_shift.png"
    alerted_features_plot_path = resolved_output_dir / "alerted_feature_distributions.png"

    snapshot_df.to_csv(snapshot_path, index=False)
    alerts_df.to_csv(alerts_path, index=False)
    report = {
        "run_id": snapshot_df["run_id"].iloc[0],
        "dataset_label": dataset_label,
        "observed_at_utc": snapshot_df["observed_at_utc"].iloc[0],
        "features_with_moderate_or_high_drift": snapshot_df.loc[
            snapshot_df["drift_severity"] != "low", "feature_name"
        ].tolist(),
        "alert_count": int(len(alerts_df)),
        "prediction_rate_reference": prediction_rate_reference,
        "prediction_rate_current": prediction_rate_current,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    plot_drift_summary(snapshot_df, drift_overview_plot_path)
    plot_prediction_rate_shift(
        prediction_rate_reference,
        prediction_rate_current,
        prediction_shift_plot_path,
    )
    plot_alerted_feature_distributions(
        reference_df,
        current_df,
        snapshot_df,
        alerted_features_plot_path,
    )

    log_monitoring_results(snapshot_df, alerts_df, db_path=db_path)
    return {
        "snapshot_df": snapshot_df,
        "alerts_df": alerts_df,
        "snapshot_path": snapshot_path,
        "alerts_path": alerts_path,
        "report_path": report_path,
        "drift_overview_plot_path": drift_overview_plot_path,
        "prediction_shift_plot_path": prediction_shift_plot_path,
        "alerted_features_plot_path": alerted_features_plot_path,
    }
