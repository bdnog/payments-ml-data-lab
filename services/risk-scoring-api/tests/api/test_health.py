from fastapi.testclient import TestClient

from risk_scoring.api.main import create_app
from risk_scoring.config import Settings


def test_health_returns_ok_payload(tmp_path):
    # No model at all: /health is liveness and must not care.
    with TestClient(create_app(Settings(app_version="9.9.9", model_dir=tmp_path))) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "risk-scoring-api", "version": "9.9.9"}
