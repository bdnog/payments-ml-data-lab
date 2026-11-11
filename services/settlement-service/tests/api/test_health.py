from fastapi.testclient import TestClient

from settlement.api.main import create_app
from settlement.config import Settings


def test_health_returns_ok_payload():
    client = TestClient(create_app(Settings(app_version="9.9.9")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "settlement-service", "version": "9.9.9"}
