"""Behavior of the real endpoints: scoring, idempotency, batch, readiness, errors."""

import json
import re
from copy import deepcopy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from payments_shared.schemas import ProblemDetails
from risk_scoring.api.main import create_app
from risk_scoring.api.schemas import ScoreResponse
from risk_scoring.config import Settings
from risk_scoring.dataset_mapping import row_to_payload

from .conftest import EXAMPLES


def assert_problem(response, status: int, slug: str | None = None) -> dict[str, Any]:
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert ProblemDetails.model_validate(body).status == status
    assert re.fullmatch(r"urn:problem:[a-z-]+", body["type"]), body["type"]
    if slug:
        assert body["type"] == f"urn:problem:{slug}"
    return body


def payload(request_example, transaction_id: str, **patch) -> dict[str, Any]:
    body = deepcopy(request_example)
    body["transaction_id"] = transaction_id
    body.update(patch)
    return body


# --- readiness and model info -----------------------------------------------------------------


def test_ready_is_200_when_the_model_is_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_version": "mlp-test-0.1"}


def test_ready_is_503_when_the_model_is_missing(tmp_path):
    with TestClient(create_app(Settings(model_dir=tmp_path))) as client:
        body = assert_problem(client.get("/ready"), 503, "not-ready")
        assert "metadata.json" in body["detail"]
        assert client.get("/health").status_code == 200  # liveness is independent


def test_ready_is_503_when_the_artifact_is_tampered_with(model_dir, tmp_path):
    import shutil

    copy = tmp_path / "model"
    shutil.copytree(model_dir, copy)
    (copy / "model.pt").write_bytes(b"not the model")
    with TestClient(create_app(Settings(model_dir=copy))) as client:
        assert "does not match the sha256" in assert_problem(client.get("/ready"), 503)["detail"]


def test_scoring_routes_are_503_without_a_model(tmp_path, request_example):
    with TestClient(create_app(Settings(model_dir=tmp_path))) as client:
        assert_problem(client.post("/v1/score", json=request_example), 503, "model-unavailable")
        assert_problem(client.post("/v1/score:batch", json={"items": [request_example]}), 503)
        assert_problem(client.get("/v1/model"), 503, "model-unavailable")


def test_model_info_comes_from_the_artifact(client):
    body = client.get("/v1/model").json()

    assert body["model_version"] == "mlp-test-0.1"
    assert body["trained_at"] == "2026-02-01T03:00:00Z"
    assert set(body["thresholds"]) == {"review", "decline"}
    assert body["thresholds"]["review"] <= body["thresholds"]["decline"]
    assert body["metrics"] == {"test.pr_auc": 0.5, "test.recall_at_fpr_1pct": 0.9}


def test_threshold_overrides_reach_the_model_endpoint(model_dir):
    settings = Settings(model_dir=model_dir, threshold_review=0.25, threshold_decline=0.75)
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/model").json()["thresholds"] == {"review": 0.25, "decline": 0.75}


# --- /v1/score --------------------------------------------------------------------------------


def test_the_contract_example_is_scored_with_a_valid_response(client, request_example):
    response = client.post("/v1/score", json=request_example)

    assert response.status_code == 200
    body = ScoreResponse.model_validate(response.json())
    assert body.transaction_id == request_example["transaction_id"]
    assert body.model_version == "mlp-test-0.1"
    assert 0 <= body.score <= 1 and body.latency_ms > 0
    example = json.loads((EXAMPLES / "score.response.json").read_text())
    assert set(response.json()) == set(example)


def test_a_risky_request_is_flagged_with_reasons_and_a_calm_one_is_not(client, request_example):
    risky = payload(
        request_example,
        "txn_000000000001",
        device={**request_example["device"], "is_proxy": True, "is_new_for_account": True},
    )
    calm = payload(
        request_example,
        "txn_000000000002",
        amount_minor=2_000,
        device={**request_example["device"], "is_proxy": False, "is_new_for_account": False,
                "ip_country": "BR"},
        card={**request_example["card"], "issuer_country": "BR", "is_new_for_merchant": False},
        behavior={"txn_count_1h": 0, "txn_count_24h": 1, "avg_amount_30d_minor": 2_000,
                  "seconds_since_last_txn": 86_000},
        account={**request_example["account"], "age_days": 500},
        merchant={**request_example["merchant"], "mcc": "5411"},
    )  # fmt: skip

    risky_body = client.post("/v1/score", json=risky).json()
    calm_body = client.post("/v1/score", json=calm).json()

    assert risky_body["score"] > calm_body["score"]
    assert calm_body["decision"] == "APPROVE" and calm_body["reason_codes"] == []
    if risky_body["decision"] != "APPROVE":
        assert "PROXY_DETECTED" in risky_body["reason_codes"]
        assert len(risky_body["reason_codes"]) <= 5


def test_every_sample_row_can_be_scored(client, sample_transactions):
    for row in sample_transactions.head(25).iter_rows(named=True):
        assert (
            client.post(
                "/v1/score", json=json.loads(json.dumps(row_to_payload(row), default=str))
            ).status_code
            == 200
        )


def test_the_decision_is_logged(client, repository, request_example):
    client.post("/v1/score", json=request_example)

    stored = repository.get(request_example["transaction_id"])
    assert stored is not None
    assert stored.request["amount_minor"] == request_example["amount_minor"]
    assert stored.model_version == "mlp-test-0.1"


def test_a_decision_is_logged_as_a_structured_line(client, request_example, caplog):
    with caplog.at_level("INFO", logger="risk_scoring.decisions"):
        client.post("/v1/score", json=request_example)

    line = next(r.getMessage() for r in caplog.records if r.name == "risk_scoring.decisions")
    for key in ("transaction_id=", "decision=", "score=", "latency_ms=", "model_version="):
        assert key in line


# --- idempotency ------------------------------------------------------------------------------


def test_replaying_a_request_returns_the_same_body(client, request_example):
    first = client.post("/v1/score", json=request_example)
    second = client.post("/v1/score", json=request_example)

    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()  # including latency_ms: it is the stored one


def test_key_order_does_not_matter_for_a_replay(client, request_example):
    first = client.post("/v1/score", json=request_example)
    reordered = dict(reversed(list(request_example.items())))
    assert client.post("/v1/score", json=reordered).json() == first.json()


def test_same_id_with_a_different_payload_is_a_409_problem(client, request_example, repository):
    client.post("/v1/score", json=request_example)
    changed = {**request_example, "amount_minor": request_example["amount_minor"] + 1}

    body = assert_problem(client.post("/v1/score", json=changed), 409, "idempotency-conflict")

    assert request_example["transaction_id"] in body["detail"]
    assert body["instance"] == "/v1/score"
    assert len(repository) == 1


def test_a_conflict_does_not_overwrite_the_original_decision(client, request_example):
    original = client.post("/v1/score", json=request_example).json()
    client.post("/v1/score", json={**request_example, "amount_minor": 1_000_000})
    assert client.post("/v1/score", json=request_example).json() == original


# --- validation -------------------------------------------------------------------------------


def test_invalid_body_is_a_422_problem_that_names_the_field(client, request_example):
    body = assert_problem(
        client.post("/v1/score", json={**request_example, "amount_minor": 0}),
        422,
        "validation-error",
    )
    assert any("amount_minor" in error["loc"] for error in body["errors"])


def test_a_naive_timestamp_is_a_422(client, request_example):
    response = client.post(
        "/v1/score", json={**request_example, "occurred_at": "2026-01-12T22:41:07"}
    )
    assert_problem(response, 422)


def test_a_422_is_never_logged_as_a_decision(client, request_example, repository):
    client.post("/v1/score", json={**request_example, "installments": 99})
    assert len(repository) == 0


# --- /v1/score:batch --------------------------------------------------------------------------


def test_batch_scores_every_item_in_order(client, request_example):
    items = [payload(request_example, f"txn_0000000000{i:02x}") for i in range(5)]
    response = client.post("/v1/score:batch", json={"items": items})

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["transaction_id"] for r in results] == [i["transaction_id"] for i in items]
    assert all("result" in r and r.get("error") is None for r in results)


def test_batch_results_equal_single_scoring(client, request_example):
    items = [
        payload(request_example, f"txn_0000000001{i:02x}", amount_minor=1_000 + 7_000 * i)
        for i in range(4)
    ]
    batch = client.post("/v1/score:batch", json={"items": items}).json()["results"]

    for i, (item, result) in enumerate(zip(items, batch, strict=True)):
        single_id = f"txn_0000000003{i:02x}"  # a fresh id, so it is scored again
        single = client.post("/v1/score", json={**item, "transaction_id": single_id}).json()
        assert result["result"]["score"] == pytest.approx(single["score"], abs=1e-6)
        assert result["result"]["decision"] == single["decision"]


def test_batch_with_valid_and_invalid_items_returns_a_result_or_an_error_for_each(
    client, request_example
):
    good = payload(request_example, "txn_0000000000a1")
    bad = payload(request_example, "txn_0000000000a2", amount_minor=0, installments=99)
    items = [good, bad, payload(request_example, "txn_0000000000a3")]

    response = client.post("/v1/score:batch", json={"items": items})

    assert response.status_code == 200
    first, second, third = response.json()["results"]
    assert first["result"]["transaction_id"] == "txn_0000000000a1"
    assert third["result"]["transaction_id"] == "txn_0000000000a3"
    assert second["transaction_id"] == "txn_0000000000a2" and "result" not in second
    problem = second["error"]
    assert (problem["status"], problem["type"]) == (422, "urn:problem:validation-error")
    fields = {e["loc"][0] for e in problem["errors"]}
    assert fields == {"amount_minor", "installments"}


def test_batch_item_without_a_transaction_id_fails_the_whole_request(client, request_example):
    broken = {k: v for k, v in request_example.items() if k != "transaction_id"}
    response = client.post("/v1/score:batch", json={"items": [request_example, broken]})
    body = assert_problem(response, 422, "validation-error")
    assert "item 1" in json.dumps(body["errors"])


def test_batch_conflicts_are_per_item_errors(client, request_example):
    client.post("/v1/score", json=request_example)
    other = payload(request_example, "txn_0000000000b2")
    changed = {**request_example, "amount_minor": 12_345}

    results = client.post("/v1/score:batch", json={"items": [changed, other]}).json()["results"]

    assert results[0]["error"]["status"] == 409
    assert results[0]["error"]["type"] == "urn:problem:idempotency-conflict"
    assert results[1]["result"]["transaction_id"] == "txn_0000000000b2"


def test_batch_replay_returns_stored_results(client, request_example):
    items = [payload(request_example, f"txn_0000000002{i:02x}") for i in range(3)]
    first = client.post("/v1/score:batch", json={"items": items}).json()
    second = client.post("/v1/score:batch", json={"items": items}).json()
    assert first == second


def test_the_same_transaction_twice_in_one_batch_is_a_replay_not_an_error(client, request_example):
    results = client.post(
        "/v1/score:batch", json={"items": [request_example, request_example]}
    ).json()["results"]
    assert results[0] == results[1] and "result" in results[0]


def test_the_same_id_with_two_payloads_in_one_batch_conflicts_on_the_second(
    client, request_example
):
    changed = {**request_example, "amount_minor": 1}
    results = client.post("/v1/score:batch", json={"items": [request_example, changed]}).json()[
        "results"
    ]
    assert "result" in results[0] and results[1]["error"]["status"] == 409


@pytest.mark.parametrize("size", [0, 101])
def test_batch_size_is_limited_to_1_to_100(client, request_example, size):
    assert_problem(client.post("/v1/score:batch", json={"items": [request_example] * size}), 422)


def test_a_batch_of_exactly_100_is_scored(client, request_example):
    items = [payload(request_example, f"txn_{i:012x}") for i in range(100)]
    response = client.post("/v1/score:batch", json={"items": items})
    assert response.status_code == 200 and len(response.json()["results"]) == 100


# --- failure of the decision log --------------------------------------------------------------


def test_a_database_failure_is_a_503_problem(model_dir, request_example):
    from sqlalchemy.exc import OperationalError

    class Broken:
        def save_or_get(self, record):
            raise OperationalError("INSERT", {}, Exception("connection refused"))

        def get(self, transaction_id):
            raise OperationalError("SELECT", {}, Exception("connection refused"))

    def factory():
        yield Broken()  # type: ignore[misc]

    app = create_app(Settings(model_dir=model_dir), repository_factory=factory)
    with TestClient(app, raise_server_exceptions=False) as client:
        body = assert_problem(
            client.post("/v1/score", json=request_example), 503, "decision-log-unavailable"
        )
        assert "retry" in body["detail"]
        assert_problem(client.post("/v1/score:batch", json={"items": [request_example]}), 503)
