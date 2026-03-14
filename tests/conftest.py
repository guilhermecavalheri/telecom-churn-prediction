from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.churn_modeling import run_pipeline
from src.modules.model_io import load_model_bundle, load_model_metadata

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_METADATA_PATH = ROOT_DIR / "artifacts" / "models" / "best_churn_pipeline_metadata.json"
TRUSTED_DATASET_PATH = ROOT_DIR / "data" / "trusted" / "train.parquet"


@pytest.fixture(scope="session")
def trained_artifacts() -> dict[str, object]:
    needs_refresh = True
    if MODEL_METADATA_PATH.exists():
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        needs_refresh = not {"raw_input_columns", "model_version"}.issubset(metadata)

    if needs_refresh:
        run_pipeline()

    return {
        "bundle": load_model_bundle(),
        "metadata": load_model_metadata(),
    }


@pytest.fixture(scope="session")
def raw_input_df() -> pd.DataFrame:
    return (
        pd.read_parquet(TRUSTED_DATASET_PATH)
        .drop(columns=["churn"])
        .head(25)
        .reset_index(drop=True)
    )
