from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modules.monitoring import run_monitoring
from src.modules.ops_store import OPS_DB_PATH, log_predictions
from src.modules.predict import RAW_INPUT_COLUMNS, predict_dataframe
from src.modules.synthetic_data import generate_and_save_synthetic_datasets


def main() -> None:
    synthetic_paths = generate_and_save_synthetic_datasets()
    baseline_df = pd.read_parquet(synthetic_paths["baseline_path"]).loc[:, RAW_INPUT_COLUMNS]
    drift_df = pd.read_parquet(synthetic_paths["drift_path"]).loc[:, RAW_INPUT_COLUMNS]

    baseline_predictions, _ = predict_dataframe(baseline_df, source="synthetic_baseline")
    drift_predictions, _ = predict_dataframe(drift_df, source="synthetic_drift")

    log_predictions(baseline_predictions, baseline_df, db_path=OPS_DB_PATH)
    log_predictions(drift_predictions, drift_df, db_path=OPS_DB_PATH)

    monitoring_result = run_monitoring(
        baseline_df,
        drift_df,
        dataset_label="synthetic_api_drift_demo",
        prediction_rate_reference=float(baseline_predictions["predicted_label"].mean()),
        prediction_rate_current=float(drift_predictions["predicted_label"].mean()),
    )

    summary = {
        "baseline_path": str(synthetic_paths["baseline_path"]),
        "drift_path": str(synthetic_paths["drift_path"]),
        "monitoring_report_path": str(monitoring_result["report_path"]),
        "drift_overview_plot_path": str(monitoring_result["drift_overview_plot_path"]),
        "prediction_shift_plot_path": str(monitoring_result["prediction_shift_plot_path"]),
        "alerted_features_plot_path": str(monitoring_result["alerted_features_plot_path"]),
        "drift_alert_count": int(len(monitoring_result["alerts_df"])),
        "features_with_alerts": monitoring_result["alerts_df"]["feature_name"].tolist(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
