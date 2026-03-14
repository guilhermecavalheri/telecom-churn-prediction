from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "models"
DEFAULT_MODEL_BUNDLE_PATH = MODEL_ARTIFACTS_DIR / "best_churn_pipeline.joblib"
DEFAULT_MODEL_METADATA_PATH = MODEL_ARTIFACTS_DIR / "best_churn_pipeline_metadata.json"
DEFAULT_RAW_INPUT_COLUMNS = [
    "call_failure",
    "complains",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age_group",
    "tariff_plan",
    "status",
    "age",
    "customer_value",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify_model_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def _to_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_serializable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def build_model_bundle(
    *,
    model_name: str,
    pipeline: Any,
    raw_input_columns: list[str],
    feature_columns: list[str],
    target_column: str,
    threshold: float,
    holdout_metrics: dict[str, Any],
    project_summary: dict[str, Any],
    model_version: str | None = None,
) -> dict[str, Any]:
    created_at_utc = _utc_now_iso()
    resolved_model_version = model_version or f"{_slugify_model_name(model_name)}_{created_at_utc}"

    return {
        "model_name": model_name,
        "model_version": resolved_model_version,
        "created_at_utc": created_at_utc,
        "pipeline": pipeline,
        "raw_input_columns": raw_input_columns,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "threshold": float(threshold),
        "holdout_metrics": _to_json_serializable(holdout_metrics),
        "project_summary": _to_json_serializable(project_summary),
    }


def save_best_model(
    *,
    model_name: str,
    pipeline: Any,
    raw_input_columns: list[str],
    feature_columns: list[str],
    target_column: str,
    threshold: float,
    holdout_metrics: dict[str, Any],
    project_summary: dict[str, Any],
    bundle_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    resolved_bundle_path = bundle_path or DEFAULT_MODEL_BUNDLE_PATH
    resolved_metadata_path = metadata_path or DEFAULT_MODEL_METADATA_PATH

    bundle = build_model_bundle(
        model_name=model_name,
        pipeline=pipeline,
        raw_input_columns=raw_input_columns,
        feature_columns=feature_columns,
        target_column=target_column,
        threshold=threshold,
        holdout_metrics=holdout_metrics,
        project_summary=project_summary,
    )
    joblib.dump(bundle, resolved_bundle_path)

    metadata = {
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
        "created_at_utc": bundle["created_at_utc"],
        "raw_input_columns": bundle["raw_input_columns"],
        "feature_columns": bundle["feature_columns"],
        "target_column": bundle["target_column"],
        "threshold": bundle["threshold"],
        "holdout_metrics": bundle["holdout_metrics"],
        "project_summary": bundle["project_summary"],
        "bundle_path": str(resolved_bundle_path),
    }
    resolved_metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return {
        "bundle": bundle,
        "bundle_path": resolved_bundle_path,
        "metadata": metadata,
        "metadata_path": resolved_metadata_path,
    }


def load_model_bundle(bundle_path: Path | None = None) -> dict[str, Any]:
    resolved_bundle_path = bundle_path or DEFAULT_MODEL_BUNDLE_PATH
    bundle = joblib.load(resolved_bundle_path)
    bundle.setdefault("raw_input_columns", DEFAULT_RAW_INPUT_COLUMNS)
    bundle.setdefault("model_version", f"{_slugify_model_name(bundle['model_name'])}_legacy")
    bundle.setdefault("created_at_utc", _utc_now_iso())
    bundle.setdefault("threshold", 0.50)
    return bundle


def load_model_metadata(metadata_path: Path | None = None) -> dict[str, Any]:
    resolved_metadata_path = metadata_path or DEFAULT_MODEL_METADATA_PATH
    metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    metadata.setdefault("raw_input_columns", DEFAULT_RAW_INPUT_COLUMNS)
    metadata.setdefault("model_version", f"{_slugify_model_name(metadata['model_name'])}_legacy")
    metadata.setdefault("created_at_utc", _utc_now_iso())
    metadata.setdefault("threshold", 0.50)
    return metadata
