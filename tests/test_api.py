from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import create_app


def test_health_endpoint_returns_ok(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "ops.duckdb"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "alive"


def test_ready_endpoint_returns_operational_state(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "ops.duckdb"))

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["model_bundle_loaded"] is True
    assert body["checks"]["model_metadata_loaded"] is True
    assert body["checks"]["database_accessible"] is True
    assert body["checks"]["required_tables_available"] is True
    assert "prediction_logs" in body["checks"]["existing_tables"]


def test_predict_endpoint_returns_prediction(tmp_path, raw_input_df) -> None:
    client = TestClient(create_app(db_path=tmp_path / "ops.duckdb"))

    response = client.post("/predict", json=raw_input_df.iloc[0].to_dict())

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert 0.0 <= body["prediction"]["score"] <= 1.0
