from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.modules.monitoring import run_monitoring
from src.modules.predict import RAW_INPUT_COLUMNS, predict_dataframe
from src.modules.synthetic_data import generate_synthetic_baseline, generate_synthetic_drift


def test_monitoring_detects_drift(tmp_path, trained_artifacts) -> None:
    raw_df = pd.read_parquet(Path("data/trusted/train.parquet")).drop(columns=["churn"])
    baseline_df = generate_synthetic_baseline(raw_df, n_samples=200, random_state=42)[
        RAW_INPUT_COLUMNS
    ]
    drift_df = generate_synthetic_drift(baseline_df, random_state=42)[RAW_INPUT_COLUMNS]

    baseline_predictions, _ = predict_dataframe(
        baseline_df,
        model_bundle=trained_artifacts["bundle"],
        source="baseline",
    )
    drift_predictions, _ = predict_dataframe(
        drift_df,
        model_bundle=trained_artifacts["bundle"],
        source="drift",
    )

    monitoring_result = run_monitoring(
        baseline_df,
        drift_df,
        dataset_label="pytest_drift",
        prediction_rate_reference=float(baseline_predictions["predicted_label"].mean()),
        prediction_rate_current=float(drift_predictions["predicted_label"].mean()),
        output_dir=tmp_path / "monitoring",
        db_path=tmp_path / "ops.duckdb",
    )

    snapshot_df = monitoring_result["snapshot_df"]
    alerts_df = monitoring_result["alerts_df"]

    assert not snapshot_df.empty
    assert (snapshot_df["drift_severity"] != "low").any()
    assert not alerts_df.empty
    assert monitoring_result["drift_overview_plot_path"].exists()
    assert monitoring_result["prediction_shift_plot_path"].exists()
    assert monitoring_result["alerted_features_plot_path"].exists()
