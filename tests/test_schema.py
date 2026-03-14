from __future__ import annotations

import pytest

from src.modules.predict import RAW_INPUT_COLUMNS
from src.modules.schema import SchemaValidationError, validate_input_dataframe


def test_validate_input_dataframe_accepts_valid_batch(raw_input_df) -> None:
    validation = validate_input_dataframe(
        raw_input_df,
        expected_columns=RAW_INPUT_COLUMNS,
        numeric_columns=RAW_INPUT_COLUMNS,
        non_negative_columns=RAW_INPUT_COLUMNS,
    )

    assert validation.dataframe.columns.tolist() == RAW_INPUT_COLUMNS
    assert isinstance(validation.warnings, list)


def test_validate_input_dataframe_rejects_missing_column(raw_input_df) -> None:
    invalid_df = raw_input_df.drop(columns=["customer_value"])

    with pytest.raises(SchemaValidationError, match="Missing required columns"):
        validate_input_dataframe(
            invalid_df,
            expected_columns=RAW_INPUT_COLUMNS,
            numeric_columns=RAW_INPUT_COLUMNS,
            non_negative_columns=RAW_INPUT_COLUMNS,
        )


def test_validate_input_dataframe_rejects_negative_value(raw_input_df) -> None:
    invalid_df = raw_input_df.copy()
    invalid_df.loc[0, "age"] = -1

    with pytest.raises(SchemaValidationError, match="contains negative values"):
        validate_input_dataframe(
            invalid_df,
            expected_columns=RAW_INPUT_COLUMNS,
            numeric_columns=RAW_INPUT_COLUMNS,
            non_negative_columns=RAW_INPUT_COLUMNS,
        )
