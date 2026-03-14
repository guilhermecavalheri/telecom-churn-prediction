from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATASET_PATH = ROOT_DIR / "data" / "raw" / "iranian_churn_telecom.parquet"
SYNTHETIC_DATA_DIR = ROOT_DIR / "data" / "synthetic"

RAW_INPUT_COLUMNS = [
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
INTEGER_COLUMNS = [
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
]


def _normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = (
        normalized.columns.str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace(
            "__",
            "_",
            regex=False,
        )
    )
    return normalized


def _clip_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in INTEGER_COLUMNS:
        cleaned[column] = cleaned[column].round().clip(lower=0).astype(int)
    cleaned["customer_value"] = cleaned["customer_value"].clip(lower=0.0).round(4)
    return cleaned


def generate_synthetic_baseline(
    raw_df: pd.DataFrame,
    *,
    n_samples: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    sampled = raw_df.sample(n=n_samples, replace=True, random_state=random_state).reset_index(
        drop=True
    )

    synthetic_df = sampled.copy()
    synthetic_df["seconds_of_use"] = synthetic_df["seconds_of_use"] * rng.normal(1.0, 0.12, n_samples)
    synthetic_df["frequency_of_use"] = synthetic_df["frequency_of_use"] * rng.normal(1.0, 0.10, n_samples)
    synthetic_df["frequency_of_sms"] = synthetic_df["frequency_of_sms"] * rng.normal(1.0, 0.18, n_samples)
    synthetic_df["distinct_called_numbers"] = synthetic_df["distinct_called_numbers"] * rng.normal(
        1.0,
        0.10,
        n_samples,
    )
    synthetic_df["customer_value"] = synthetic_df["customer_value"] * rng.normal(1.0, 0.08, n_samples)
    synthetic_df["charge_amount"] = synthetic_df["charge_amount"] * rng.normal(1.0, 0.08, n_samples)
    synthetic_df["call_failure"] = synthetic_df["call_failure"] + rng.poisson(0.2, n_samples)
    synthetic_df["complains"] = np.where(
        rng.random(n_samples) < 0.06,
        np.minimum(synthetic_df["complains"] + 1, 1),
        synthetic_df["complains"],
    )

    return _clip_and_cast(synthetic_df)


def generate_synthetic_drift(
    baseline_df: pd.DataFrame,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state + 7)
    drift_df = baseline_df.copy()

    # This drift scenario simulates worsening service quality and lower engagement.
    drift_df["call_failure"] = drift_df["call_failure"] * rng.normal(1.8, 0.18, len(drift_df)) + rng.poisson(
        2.0,
        len(drift_df),
    )
    drift_df["complains"] = np.where(rng.random(len(drift_df)) < 0.35, 1, drift_df["complains"])
    drift_df["frequency_of_use"] = drift_df["frequency_of_use"] * rng.normal(0.72, 0.12, len(drift_df))
    drift_df["seconds_of_use"] = drift_df["seconds_of_use"] * rng.normal(0.78, 0.10, len(drift_df))
    drift_df["frequency_of_sms"] = drift_df["frequency_of_sms"] * rng.normal(0.84, 0.12, len(drift_df))
    drift_df["customer_value"] = drift_df["customer_value"] * rng.normal(0.85, 0.06, len(drift_df))
    drift_df["charge_amount"] = drift_df["charge_amount"] * rng.normal(1.12, 0.05, len(drift_df))
    drift_df["subscription_length"] = drift_df["subscription_length"] * rng.normal(
        0.88,
        0.08,
        len(drift_df),
    )

    return _clip_and_cast(drift_df)


def generate_and_save_synthetic_datasets(
    *,
    n_samples: int = 500,
    random_state: int = 42,
) -> dict[str, Path]:
    raw_df = _normalize_raw_columns(pd.read_parquet(RAW_DATASET_PATH))
    baseline_df = generate_synthetic_baseline(raw_df, n_samples=n_samples, random_state=random_state)
    drift_df = generate_synthetic_drift(baseline_df, random_state=random_state)

    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = SYNTHETIC_DATA_DIR / "api_test_baseline.parquet"
    drift_path = SYNTHETIC_DATA_DIR / "api_test_drifted.parquet"
    baseline_df.to_parquet(baseline_path, index=False)
    drift_df.to_parquet(drift_path, index=False)

    return {"baseline_path": baseline_path, "drift_path": drift_path}
