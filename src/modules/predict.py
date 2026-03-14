from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.churn_modeling import engineer_features
from src.modules.model_io import DEFAULT_MODEL_BUNDLE_PATH, DEFAULT_RAW_INPUT_COLUMNS, load_model_bundle
from src.modules.schema import SchemaValidationError, validate_input_dataframe

RAW_INPUT_COLUMNS = DEFAULT_RAW_INPUT_COLUMNS


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def prepare_inference_features(
    raw_input_df: pd.DataFrame,
    model_bundle: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    expected_raw_columns = model_bundle.get("raw_input_columns", RAW_INPUT_COLUMNS)
    validation = validate_input_dataframe(
        raw_input_df,
        expected_columns=expected_raw_columns,
        numeric_columns=expected_raw_columns,
        non_negative_columns=expected_raw_columns,
    )

    engineered_df = engineer_features(validation.dataframe)
    expected_feature_columns = model_bundle["feature_columns"]
    missing_feature_columns = [
        column for column in expected_feature_columns if column not in engineered_df.columns
    ]
    if missing_feature_columns:
        raise SchemaValidationError(
            f"Engineered dataframe is missing columns required by the model: {', '.join(missing_feature_columns)}"
        )

    return engineered_df.loc[:, expected_feature_columns].copy(), validation.warnings


def predict_dataframe(
    raw_input_df: pd.DataFrame,
    *,
    model_bundle: dict[str, Any] | None = None,
    bundle_path: Path | None = None,
    source: str = "batch",
    request_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    bundle = model_bundle or load_model_bundle(bundle_path or DEFAULT_MODEL_BUNDLE_PATH)
    feature_df, warnings = prepare_inference_features(raw_input_df, bundle)

    threshold = float(bundle.get("threshold", 0.50))
    probabilities = bundle["pipeline"].predict_proba(feature_df)[:, 1]
    predicted_labels = (probabilities >= threshold).astype(int)

    resolved_request_ids = request_ids or [str(uuid4()) for _ in range(len(raw_input_df))]
    prediction_timestamp = _utc_now_iso()
    prediction_df = pd.DataFrame(
        {
            "prediction_timestamp_utc": [prediction_timestamp] * len(feature_df),
            "request_id": resolved_request_ids,
            "source": [source] * len(feature_df),
            "model_name": [bundle["model_name"]] * len(feature_df),
            "model_version": [bundle["model_version"]] * len(feature_df),
            "score": probabilities,
            "predicted_label": predicted_labels,
            "threshold": [threshold] * len(feature_df),
        }
    )

    return prediction_df, warnings


def predict_records(
    records: list[dict[str, Any]],
    *,
    model_bundle: dict[str, Any] | None = None,
    bundle_path: Path | None = None,
    source: str = "api",
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_input_df = pd.DataFrame(records)
    prediction_df, warnings = predict_dataframe(
        raw_input_df,
        model_bundle=model_bundle,
        bundle_path=bundle_path,
        source=source,
    )
    return prediction_df.to_dict(orient="records"), warnings
