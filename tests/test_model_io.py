from __future__ import annotations


def test_saved_bundle_contains_operational_metadata(trained_artifacts: dict[str, object]) -> None:
    bundle = trained_artifacts["bundle"]

    assert bundle["model_name"]
    assert bundle["model_version"]
    assert bundle["raw_input_columns"]
    assert bundle["feature_columns"]
    assert len(bundle["feature_columns"]) > len(bundle["raw_input_columns"])
    assert "pipeline" in bundle


def test_saved_metadata_matches_bundle(trained_artifacts: dict[str, object]) -> None:
    bundle = trained_artifacts["bundle"]
    metadata = trained_artifacts["metadata"]

    assert metadata["model_name"] == bundle["model_name"]
    assert metadata["model_version"] == bundle["model_version"]
    assert metadata["threshold"] == bundle["threshold"]
