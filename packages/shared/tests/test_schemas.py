from payments_shared import HealthResponse


def test_health_response_serializes_fields():
    payload = HealthResponse(status="ok", service="demo", version="1.2.3")

    assert payload.model_dump() == {"status": "ok", "service": "demo", "version": "1.2.3"}
