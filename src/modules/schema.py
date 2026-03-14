from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when inference input does not satisfy the expected schema."""


@dataclass(slots=True)
class ValidationResult:
    dataframe: pd.DataFrame
    warnings: list[str]


def _missing_columns(df: pd.DataFrame, expected_columns: Iterable[str]) -> list[str]:
    return [column for column in expected_columns if column not in df.columns]


def _unexpected_columns(df: pd.DataFrame, expected_columns: Iterable[str]) -> list[str]:
    expected = set(expected_columns)
    return [column for column in df.columns if column not in expected]


def validate_input_dataframe(
    df: pd.DataFrame,
    *,
    expected_columns: list[str],
    numeric_columns: list[str] | None = None,
    non_negative_columns: list[str] | None = None,
    allow_extra_columns: bool = False,
) -> ValidationResult:
    if df.empty:
        raise SchemaValidationError("The input dataframe is empty.")

    missing_columns = _missing_columns(df, expected_columns)
    if missing_columns:
        raise SchemaValidationError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    unexpected_columns = _unexpected_columns(df, expected_columns)
    if unexpected_columns and not allow_extra_columns:
        raise SchemaValidationError(
            f"Unexpected columns received: {', '.join(unexpected_columns)}"
        )

    cleaned_df = df.loc[:, expected_columns].copy()

    if cleaned_df[expected_columns].isnull().any().any():
        null_columns = cleaned_df.columns[cleaned_df.isnull().any()].tolist()
        raise SchemaValidationError(
            f"Null values detected in required columns: {', '.join(null_columns)}"
        )

    resolved_numeric_columns = numeric_columns or expected_columns
    for column in resolved_numeric_columns:
        try:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="raise")
        except Exception as exc:  # pragma: no cover - pandas error text is enough
            raise SchemaValidationError(
                f"Column '{column}' cannot be converted to a numeric dtype."
            ) from exc

    warnings: list[str] = []
    for column in non_negative_columns or []:
        if (cleaned_df[column] < 0).any():
            raise SchemaValidationError(
                f"Column '{column}' contains negative values, which are invalid for this model."
            )
        if cleaned_df[column].eq(0).all():
            warnings.append(
                f"Column '{column}' is entirely zero in this input batch. Review whether that is expected."
            )

    return ValidationResult(dataframe=cleaned_df, warnings=warnings)
