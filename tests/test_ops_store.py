from __future__ import annotations

from pathlib import Path

from src.modules.ops_store import fetch_table, initialize_ops_store, log_model_registration, log_predictions
from src.modules.predict import predict_dataframe


def test_ops_store_logs_model_and_predictions(tmp_path, trained_artifacts, raw_input_df) -> None:
    db_path = tmp_path / "ml_ops.duckdb"
    initialize_ops_store(db_path)

    metadata = trained_artifacts["metadata"]
    log_model_registration(
        metadata,
        bundle_path=Path(metadata["bundle_path"]),
        metadata_path=tmp_path / "best_model_metadata.json",
        db_path=db_path,
    )

    prediction_df, _ = predict_dataframe(
        raw_input_df.head(3),
        model_bundle=trained_artifacts["bundle"],
        source="pytest",
    )
    log_predictions(prediction_df, raw_input_df.head(3), db_path=db_path)

    model_registry_df = fetch_table("model_registry", db_path=db_path)
    prediction_logs_df = fetch_table("prediction_logs", db_path=db_path)
    prediction_inputs_df = fetch_table("prediction_inputs_sample", db_path=db_path)

    assert len(model_registry_df) == 1
    assert len(prediction_logs_df) == 3
    assert len(prediction_inputs_df) == 3
