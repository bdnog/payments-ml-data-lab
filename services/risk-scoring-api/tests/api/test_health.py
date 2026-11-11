from fastapi.testclient import TestClient

from risk_scoring.api.main import create_app
from risk_scoring.config import Settings


def test_health_returns_ok_payload():
    client = TestClient(create_app(Settings(app_version="9.9.9")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "risk-scoring-api", "version": "9.9.9"}
