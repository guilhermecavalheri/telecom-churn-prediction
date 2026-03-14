from __future__ import annotations

from src.modules.predict import predict_dataframe


def test_predict_dataframe_returns_scores(trained_artifacts, raw_input_df) -> None:
    prediction_df, warnings = predict_dataframe(
        raw_input_df.head(10),
        model_bundle=trained_artifacts["bundle"],
        source="pytest",
    )

    assert len(prediction_df) == 10
    assert prediction_df["score"].between(0.0, 1.0).all()
    assert set(prediction_df["predicted_label"].unique()).issubset({0, 1})
    assert prediction_df["model_version"].nunique() == 1
    assert isinstance(warnings, list)
