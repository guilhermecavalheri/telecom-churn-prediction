from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modules.model_io import (
    DEFAULT_MODEL_METADATA_PATH,
    load_model_bundle,
    load_model_metadata,
)
from src.modules.ops_store import OPS_DB_PATH, get_connection, initialize_ops_store, log_predictions
from src.modules.predict import RAW_INPUT_COLUMNS, predict_dataframe
from src.modules.schema import SchemaValidationError

REQUIRED_OPS_TABLES = {
    "model_registry",
    "prediction_logs",
    "prediction_inputs_sample",
    "monitoring_snapshots",
    "drift_alerts",
}


class TelecomFeatures(BaseModel):
    call_failure: int = Field(ge=0)
    complains: int = Field(ge=0)
    subscription_length: int = Field(ge=0)
    charge_amount: int = Field(ge=0)
    seconds_of_use: int = Field(ge=0)
    frequency_of_use: int = Field(ge=0)
    frequency_of_sms: int = Field(ge=0)
    distinct_called_numbers: int = Field(ge=0)
    age_group: int = Field(ge=0)
    tariff_plan: int = Field(ge=0)
    status: int = Field(ge=0)
    age: int = Field(ge=0)
    customer_value: float = Field(ge=0.0)

    model_config = ConfigDict(extra="forbid")


class BatchPredictionRequest(BaseModel):
    records: list[TelecomFeatures]
    source: str = "api_batch"


def _check_database_readiness(db_path: Path) -> dict[str, Any]:
    connection = get_connection(db_path)
    try:
        tables_df = connection.execute("SHOW TABLES").fetchdf()
        existing_tables = set(tables_df["name"].tolist())
    finally:
        connection.close()

    missing_tables = sorted(REQUIRED_OPS_TABLES - existing_tables)
    return {
        "database_accessible": True,
        "database_path": str(db_path),
        "existing_tables": sorted(existing_tables),
        "missing_tables": missing_tables,
        "required_tables_available": not missing_tables,
    }


def create_app(
    *,
    bundle_path: Path | None = None,
    metadata_path: Path | None = None,
    db_path: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Telecom Churn API", version="1.0.0")
    resolved_db_path = db_path or OPS_DB_PATH
    initialize_ops_store(resolved_db_path)

    app.state.model_bundle = load_model_bundle(bundle_path)
    app.state.model_metadata = load_model_metadata(metadata_path or DEFAULT_MODEL_METADATA_PATH)
    app.state.db_path = resolved_db_path

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "alive",
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        model_loaded = app.state.model_bundle is not None
        metadata_loaded = app.state.model_metadata is not None
        db_checks = _check_database_readiness(app.state.db_path)

        ready_status = (
            model_loaded
            and metadata_loaded
            and db_checks["database_accessible"]
            and db_checks["required_tables_available"]
        )

        return {
            "status": "ready" if ready_status else "not_ready",
            "checks": {
                "model_bundle_loaded": model_loaded,
                "model_metadata_loaded": metadata_loaded,
                "model_version": app.state.model_metadata.get("model_version"),
                "threshold": app.state.model_metadata.get("threshold"),
                "expected_input_columns": RAW_INPUT_COLUMNS,
                **db_checks,
            },
        }

    @app.get("/model/info")
    def model_info() -> dict[str, Any]:
        return {
            "model_name": app.state.model_metadata["model_name"],
            "model_version": app.state.model_metadata["model_version"],
            "threshold": app.state.model_metadata["threshold"],
            "feature_columns": app.state.model_metadata["feature_columns"],
            "raw_input_columns": app.state.model_metadata["raw_input_columns"],
        }

    @app.post("/predict")
    def predict(payload: TelecomFeatures) -> dict[str, Any]:
        raw_df = pd.DataFrame([payload.model_dump()])
        try:
            prediction_df, warnings = predict_dataframe(
                raw_df,
                model_bundle=app.state.model_bundle,
                source="api_single",
            )
        except SchemaValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        log_predictions(prediction_df, raw_df, db_path=app.state.db_path)
        return {
            "prediction": prediction_df.iloc[0].to_dict(),
            "warnings": warnings,
        }

    @app.post("/predict/batch")
    def predict_batch(payload: BatchPredictionRequest) -> dict[str, Any]:
        raw_df = pd.DataFrame([record.model_dump() for record in payload.records])
        try:
            prediction_df, warnings = predict_dataframe(
                raw_df,
                model_bundle=app.state.model_bundle,
                source=payload.source,
            )
        except SchemaValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        log_predictions(prediction_df, raw_df, db_path=app.state.db_path)
        return {
            "predictions": prediction_df.to_dict(orient="records"),
            "warnings": warnings,
            "count": int(len(prediction_df)),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False)
