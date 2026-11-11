from typing import Any, get_args, get_origin

import polars as pl
import pytest
from pydantic import BaseModel, ValidationError

from payments_shared.data_contracts import Role, load_contract
from payments_shared.paths import contracts_dir, repo_root
from payments_shared.schemas import ProblemDetails
from risk_scoring.api.schemas import (
    FEATURE_PATHS,
    IDENTITY_PATHS,
    BatchScoreItemResult,
    BatchScoreResponse,
    BehaviorBlock,
    Decision,
    ModelInfo,
    ReasonCode,
    ScoreRequest,
    ScoreResponse,
)
from risk_scoring.dataset_mapping import iter_requests, row_to_request

EXAMPLES = contracts_dir() / "examples" / "risk-scoring"
SAMPLE_TRANSACTIONS = repo_root() / "data" / "samples" / "risk" / "transactions.parquet"


def assert_problem(response, status: int) -> dict[str, Any]:
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/problem+json")
    body = ProblemDetails.model_validate(response.json())
    assert body.status == status
    return response.json()


# --- examples ---------------------------------------------------------------------------------


def test_request_example_validates():
    ScoreRequest.model_validate_json((EXAMPLES / "score.request.json").read_text())


def test_response_example_validates_and_matches_the_request_example(request_example):
    response = ScoreResponse.model_validate_json((EXAMPLES / "score.response.json").read_text())
    assert response.transaction_id == request_example["transaction_id"]


def test_problem_example_validates():
    ProblemDetails.model_validate_json((EXAMPLES / "problem.json").read_text())


# --- model rules ------------------------------------------------------------------------------


def test_naive_timestamp_is_rejected(request_example):
    with pytest.raises(ValidationError, match="timezone"):
        ScoreRequest.model_validate({**request_example, "occurred_at": "2026-01-12T22:41:07"})


@pytest.mark.parametrize(
    "patch",
    [
        {"installments": 13},
        {"installments": 0},
        {"currency": "USD"},
        {"channel": "fax"},
        {"transaction_id": "txn_XYZ"},
        {"amount_minor": 12.5},
    ],
)
def test_invalid_top_level_values_are_rejected(request_example, patch):
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate({**request_example, **patch})


def test_velocity_must_be_monotonic():
    with pytest.raises(ValidationError, match="txn_count_1h cannot exceed"):
        BehaviorBlock(
            txn_count_1h=5, txn_count_24h=2, avg_amount_30d_minor=None, seconds_since_last_txn=None
        )


def test_first_transaction_has_null_history():
    block = BehaviorBlock(
        txn_count_1h=0, txn_count_24h=0, avg_amount_30d_minor=None, seconds_since_last_txn=None
    )
    assert block.avg_amount_30d_minor is None


def test_a_batch_item_has_exactly_one_outcome():
    problem = ProblemDetails(title="Bad", status=422)
    with pytest.raises(ValidationError):
        BatchScoreItemResult(transaction_id="txn_0a1b2c3d4e5f")
    with pytest.raises(ValidationError):
        response = ScoreResponse.model_validate_json((EXAMPLES / "score.response.json").read_text())
        BatchScoreItemResult(transaction_id="txn_0a1b2c3d4e5f", result=response, error=problem)
    assert BatchScoreItemResult(transaction_id="txn_0a1b2c3d4e5f", error=problem).error == problem


def test_batch_response_can_mix_results_and_errors():
    response = ScoreResponse.model_validate_json((EXAMPLES / "score.response.json").read_text())
    ok = BatchScoreItemResult(transaction_id=response.transaction_id, result=response)
    bad = BatchScoreItemResult(
        transaction_id="txn_ffffffffffff", error=ProblemDetails(title="Bad", status=422)
    )
    assert len(BatchScoreResponse(results=[ok, bad]).results) == 2


def test_reason_codes_and_decisions_are_the_agreed_sets():
    assert {c.value for c in ReasonCode} == {
        "NEW_DEVICE",
        "PROXY_DETECTED",
        "COUNTRY_MISMATCH",
        "AMOUNT_ABOVE_NORMAL",
        "HIGH_VELOCITY_1H",
        "HIGH_VELOCITY_24H",
        "NEW_ACCOUNT",
        "HIGH_RISK_MCC",
        "NEW_CARD_FOR_MERCHANT",
    }
    assert {d.value for d in Decision} == {"APPROVE", "REVIEW", "DECLINE"}


def test_model_info_requires_ordered_thresholds():
    base = {
        "model_version": "x",
        "trained_at": "2026-02-01T03:00:00Z",
        "metrics": {"pr_auc": 0.4},
    }
    ModelInfo.model_validate({**base, "thresholds": {"review": 0.5, "decline": 0.85}})
    with pytest.raises(ValidationError):
        ModelInfo.model_validate({**base, "thresholds": {"review": 0.9, "decline": 0.5}})


# --- alignment with the data contract ---------------------------------------------------------


def field_exists(model: type[BaseModel], path: str) -> bool:
    """Whether a dotted path such as ``device.ip_country`` leads to a field of the model."""
    head, _, rest = path.partition(".")
    field = model.model_fields.get(head)
    if field is None:
        return False
    if not rest:
        return True
    nested = [a for a in (field.annotation, *get_args(field.annotation)) if isinstance(a, type)]
    nested = [a for a in nested if issubclass(a, BaseModel)]
    if get_origin(field.annotation) is None and isinstance(field.annotation, type):
        nested = [field.annotation] if issubclass(field.annotation, BaseModel) else []
    return any(field_exists(model_, rest) for model_ in nested)


def test_every_feature_column_of_the_contract_is_represented_in_the_request():
    contract = load_contract("transactions")
    features = {c.name for c in contract.columns_with_role(Role.FEATURE)}

    assert set(FEATURE_PATHS) == features, "the request and the data contract have drifted apart"
    for column, path in {**FEATURE_PATHS, **IDENTITY_PATHS}.items():
        assert field_exists(ScoreRequest, path), f"{column} -> {path} is not a request field"


def test_occurred_at_is_a_model_input_and_the_only_one_that_is_a_timestamp():
    """Phase 3 decision: time-of-day features need ``occurred_at``, so it has role ``feature``."""
    contract = load_contract("transactions")
    assert contract.column("occurred_at").role is Role.FEATURE
    assert FEATURE_PATHS["occurred_at"] == "occurred_at"
    assert "occurred_at" not in IDENTITY_PATHS


def test_request_model_has_no_field_that_the_contract_does_not_know():
    """The reverse direction: nothing in the request is unmapped (no silent extra input)."""

    def leaf_paths(model: type[BaseModel], prefix: str = "") -> set[str]:
        paths: set[str] = set()
        for name, field in model.model_fields.items():
            annotation = field.annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                paths |= leaf_paths(annotation, f"{prefix}{name}.")
            else:
                paths.add(f"{prefix}{name}")
        return paths

    assert leaf_paths(ScoreRequest) == set(FEATURE_PATHS.values()) | set(IDENTITY_PATHS.values())


def test_every_row_of_the_committed_sample_is_a_valid_score_request():
    transactions = pl.read_parquet(SAMPLE_TRANSACTIONS)
    assert transactions.height > 1_000

    assert sum(1 for _ in iter_requests(transactions)) == transactions.height


def test_the_request_carries_the_values_of_the_row_unchanged():
    row = pl.read_parquet(SAMPLE_TRANSACTIONS).row(100, named=True)
    dumped = row_to_request(row).model_dump(mode="python")

    for column, path in FEATURE_PATHS.items():
        value: Any = dumped
        for part in path.split("."):
            value = value[part]
        assert str(value) == str(row[column]), column


# --- OpenAPI ----------------------------------------------------------------------------------


def test_openapi_describes_every_route_and_documents_problems(client):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {
        "/health",
        "/ready",
        "/v1/score",
        "/v1/score:batch",
        "/v1/model",
    }

    score = schema["paths"]["/v1/score"]["post"]
    assert {"200", "409", "422", "503"} <= set(score["responses"])
    assert "501" not in score["responses"]
    assert "Idempotent by `transaction_id`" in score["description"]
    assert "not a calibrated probability" in score["description"]
    for status in ("409", "422", "503"):
        content = score["responses"][status]["content"]
        assert list(content) == ["application/problem+json"]
        assert content["application/problem+json"]["schema"]["$ref"].endswith("/ProblemDetails")

    ready = schema["paths"]["/ready"]["get"]["responses"]
    assert set(ready) == {"200", "503"}
    assert list(ready["503"]["content"]) == ["application/problem+json"]

    schemas = schema["components"]["schemas"]
    assert set(schemas["ReasonCode"]["enum"]) == {c.value for c in ReasonCode}
    assert schemas["ScoreRequest"]["properties"]["amount_minor"]["exclusiveMinimum"] == 0
    assert "Not a calibrated" in schemas["ScoreResponse"]["properties"]["score"]["description"]


def test_openapi_still_documents_the_batch_items_as_plain_score_requests(client):
    """The lenient runtime validation must not change what the contract says."""
    items = client.get("/openapi.json").json()["components"]["schemas"]["BatchScoreRequest"][
        "properties"
    ]["items"]
    assert items["maxItems"] == 100 and items["minItems"] == 1
    assert items["items"]["$ref"].endswith("/ScoreRequest")
