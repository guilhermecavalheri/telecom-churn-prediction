from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATABASE_DIR = ROOT_DIR / "artifacts" / "database"
OPS_DB_PATH = DATABASE_DIR / "ml_ops.duckdb"


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str)


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    resolved_path = db_path or OPS_DB_PATH
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(resolved_path))


def initialize_ops_store(db_path: Path | None = None) -> Path:
    connection = get_connection(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            model_version VARCHAR,
            model_name VARCHAR,
            created_at_utc TIMESTAMP,
            threshold DOUBLE,
            bundle_path VARCHAR,
            metadata_path VARCHAR,
            metrics_json VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            prediction_timestamp_utc TIMESTAMP,
            request_id VARCHAR,
            source VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            score DOUBLE,
            predicted_label INTEGER,
            threshold DOUBLE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_inputs_sample (
            prediction_timestamp_utc TIMESTAMP,
            request_id VARCHAR,
            payload_json VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS monitoring_snapshots (
            run_id VARCHAR,
            observed_at_utc TIMESTAMP,
            dataset_label VARCHAR,
            feature_name VARCHAR,
            reference_mean DOUBLE,
            current_mean DOUBLE,
            reference_std DOUBLE,
            current_std DOUBLE,
            reference_missing_rate DOUBLE,
            current_missing_rate DOUBLE,
            psi DOUBLE,
            drift_severity VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS drift_alerts (
            run_id VARCHAR,
            alert_timestamp_utc TIMESTAMP,
            feature_name VARCHAR,
            psi DOUBLE,
            severity VARCHAR,
            message VARCHAR,
            recommended_action VARCHAR
        )
        """
    )
    connection.close()
    return db_path or OPS_DB_PATH


def log_model_registration(
    metadata: dict[str, Any],
    *,
    bundle_path: Path,
    metadata_path: Path,
    db_path: Path | None = None,
) -> None:
    initialize_ops_store(db_path)
    row = pd.DataFrame(
        [
            {
                "model_version": metadata["model_version"],
                "model_name": metadata["model_name"],
                "created_at_utc": metadata["created_at_utc"],
                "threshold": metadata["threshold"],
                "bundle_path": str(bundle_path),
                "metadata_path": str(metadata_path),
                "metrics_json": _json_dump(metadata["holdout_metrics"]),
            }
        ]
    )
    connection = get_connection(db_path)
    connection.register("model_registry_df", row)
    connection.execute("INSERT INTO model_registry SELECT * FROM model_registry_df")
    connection.close()


def log_predictions(
    predictions_df: pd.DataFrame,
    input_df: pd.DataFrame,
    *,
    db_path: Path | None = None,
    sample_size: int = 20,
) -> None:
    initialize_ops_store(db_path)
    connection = get_connection(db_path)
    connection.register(
        "prediction_logs_df",
        predictions_df[
            [
                "prediction_timestamp_utc",
                "request_id",
                "source",
                "model_name",
                "model_version",
                "score",
                "predicted_label",
                "threshold",
            ]
        ],
    )
    connection.execute("INSERT INTO prediction_logs SELECT * FROM prediction_logs_df")

    payload_rows = input_df.head(sample_size).copy()
    payload_rows["request_id"] = predictions_df["request_id"].head(sample_size).tolist()
    payload_rows["prediction_timestamp_utc"] = predictions_df["prediction_timestamp_utc"].head(
        sample_size
    ).tolist()
    payload_rows["payload_json"] = payload_rows.drop(
        columns=["request_id", "prediction_timestamp_utc"]
    ).apply(lambda row: _json_dump(row.to_dict()), axis=1)
    payload_rows = payload_rows[["prediction_timestamp_utc", "request_id", "payload_json"]]
    connection.register("prediction_inputs_df", payload_rows)
    connection.execute("INSERT INTO prediction_inputs_sample SELECT * FROM prediction_inputs_df")
    connection.close()


def log_monitoring_results(
    snapshot_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    *,
    db_path: Path | None = None,
) -> None:
    initialize_ops_store(db_path)
    connection = get_connection(db_path)

    if not snapshot_df.empty:
        connection.register("monitoring_snapshot_df", snapshot_df)
        connection.execute("INSERT INTO monitoring_snapshots SELECT * FROM monitoring_snapshot_df")

    if not alerts_df.empty:
        connection.register("drift_alerts_df", alerts_df)
        connection.execute("INSERT INTO drift_alerts SELECT * FROM drift_alerts_df")

    connection.close()


def fetch_table(table_name: str, *, db_path: Path | None = None) -> pd.DataFrame:
    initialize_ops_store(db_path)
    connection = get_connection(db_path)
    result = connection.execute(f"SELECT * FROM {table_name}").fetchdf()
    connection.close()
    return result
