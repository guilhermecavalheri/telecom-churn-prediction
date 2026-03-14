from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import create_app


def test_health_endpoint_returns_ok(tmp_path) -> None:
    client = TestClient(create_app(db_path=tmp_path / "ops.duckdb"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_prediction(tmp_path, raw_input_df) -> None:
    client = TestClient(create_app(db_path=tmp_path / "ops.duckdb"))

    response = client.post("/predict", json=raw_input_df.iloc[0].to_dict())

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert 0.0 <= body["prediction"]["score"] <= 1.0
