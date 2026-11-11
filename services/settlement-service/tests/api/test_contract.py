import datetime as dt
import json
from types import UnionType
from typing import Any, Union, get_args, get_origin

import polars as pl
import pytest
from fastapi.testclient import TestClient
from pydantic import AwareDatetime, BaseModel, ValidationError

from payments_shared.data_contracts import (
    Column,
    ColumnType,
    list_contracts,
    load_contract,
    validate_dataframe,
)
from payments_shared.paths import contracts_dir
from payments_shared.schemas import ProblemDetails
from settlement.api.main import create_app
from settlement.api.schemas import (
    BreaksPage,
    BreakType,
    DailySettlement,
    Mart,
    ReconciliationBreak,
    Run,
    RunRequest,
    RunStatus,
    SettlementsPage,
)
from settlement.config import Settings

EXAMPLES = contracts_dir() / "examples" / "settlement"
RUN_ID = "run_4f9c2a7d10be"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(Settings(app_version="1.0.0")))


def assert_problem(response, status: int) -> dict[str, Any]:
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/problem+json")
    assert ProblemDetails.model_validate(response.json()).status == status
    return response.json()


def example(name: str) -> dict[str, Any]:
    return json.loads((EXAMPLES / name).read_text())


# --- examples ---------------------------------------------------------------------------------


def test_examples_validate_against_the_models():
    SettlementsPage.model_validate(example("settlements.page.json"))
    BreaksPage.model_validate(example("breaks.page.json"))
    Run.model_validate(example("run.json"))


@pytest.mark.parametrize(
    "file, page_model, contract",
    [
        ("settlements.page.json", SettlementsPage, "daily_settlement_by_merchant"),
        ("breaks.page.json", BreaksPage, "reconciliation_breaks"),
    ],
)
def test_example_rows_honor_the_mart_data_contract(file, page_model, contract):
    """The examples are valid *data*, not only valid JSON: they satisfy the mart's rules."""
    spec = load_contract(contract)
    page = page_model.model_validate(example(file))
    rows = [item.model_dump(mode="python") for item in page.items]
    rows = [{k: getattr(v, "value", v) for k, v in row.items()} for row in rows]

    frame = pl.DataFrame(rows, schema=spec.schema)

    assert validate_dataframe(frame, spec) == []


# --- stub routes ------------------------------------------------------------------------------


def test_health_still_works(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "service": "settlement-service",
        "version": "1.0.0",
    }


@pytest.mark.parametrize(
    "method, url, body",
    [
        ("GET", "/v1/settlements", None),
        (
            "GET",
            "/v1/settlements?merchant_id=mer_5d7a1e&from=2026-01-01&to=2026-01-31&limit=50",
            None,
        ),
        ("GET", "/v1/reconciliation/breaks", None),
        ("GET", "/v1/reconciliation/breaks?date=2026-01-15&type=FEE_MISMATCH&limit=10", None),
        ("GET", "/v1/exports/fee_summary/2026-01-15", None),
        ("POST", "/v1/runs", {"run_date": "2026-01-15"}),
        ("GET", f"/v1/runs/{RUN_ID}", None),
    ],
)
def test_stub_routes_answer_501_problems(client, method, url, body):
    response = client.request(method, url, json=body)
    problem = assert_problem(response, 501)
    assert problem["instance"] == url.split("?")[0]


@pytest.mark.parametrize(
    "url",
    [
        "/v1/settlements?limit=0",
        f"/v1/settlements?limit={1_001}",
        "/v1/settlements?from=15-01-2026",
        "/v1/settlements?merchant_id=nope",
        "/v1/reconciliation/breaks?type=NOT_A_TYPE",
        "/v1/exports/not_a_mart/2026-01-15",
        "/v1/exports/fee_summary/yesterday",
        "/v1/runs/not-a-run",
    ],
)
def test_invalid_parameters_are_422_problems(client, url):
    assert_problem(client.get(url), 422)


def test_run_request_needs_a_valid_date(client):
    assert_problem(client.post("/v1/runs", json={"run_date": "2026-13-01"}), 422)
    assert_problem(client.post("/v1/runs", json={}), 422)


# --- model rules ------------------------------------------------------------------------------


def run(**patch):
    return {**example("run.json"), **patch}


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "DONE", "finished_at": None},
        {"status": "RUNNING", "finished_at": "2026-01-16T02:01:12Z"},
        {"status": "PENDING", "started_at": "2026-01-16T02:00:03Z", "finished_at": None},
        {"status": "FAILED", "error": None},
        {"status": "DONE", "error": "boom"},
        {"requested_at": "2026-01-16T02:00:00"},
    ],
)
def test_inconsistent_runs_are_rejected(patch):
    with pytest.raises(ValidationError):
        Run.model_validate(run(**patch))


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "PENDING", "started_at": None, "finished_at": None},
        {"status": "RUNNING", "finished_at": None},
        {"status": "FAILED", "error": "PSP file missing"},
    ],
)
def test_consistent_runs_are_accepted(patch):
    Run.model_validate(run(**patch))


def test_run_statuses_and_run_request():
    assert {s.value for s in RunStatus} == {"PENDING", "RUNNING", "DONE", "FAILED"}
    assert RunRequest(run_date=dt.date(2026, 1, 15)).run_date == dt.date(2026, 1, 15)


# --- alignment with the mart data contracts ---------------------------------------------------

EXPECTED_TYPES: dict[ColumnType, type] = {
    ColumnType.STRING: str,
    ColumnType.INT64: int,
    ColumnType.FLOAT64: float,
    ColumnType.BOOL: bool,
    ColumnType.DATE: dt.date,
    ColumnType.TIMESTAMP_UTC: dt.datetime,
}


def split_optional(annotation: Any) -> tuple[Any, bool]:
    """``X | None`` becomes ``(X, True)``; anything else becomes ``(annotation, False)``."""
    if get_origin(annotation) in (UnionType, Union):
        members = [a for a in get_args(annotation) if a is not type(None)]
        return members[0], len(members) < len(get_args(annotation))
    return annotation, False


def metadata_value(field, attribute: str) -> Any:
    for item in field.metadata:
        if hasattr(item, attribute):
            return getattr(item, attribute)
    return None


@pytest.mark.parametrize(
    "model, contract_name",
    [
        (DailySettlement, "daily_settlement_by_merchant"),
        (ReconciliationBreak, "reconciliation_breaks"),
    ],
)
def test_item_model_matches_its_mart_contract(model: type[BaseModel], contract_name: str):
    contract = load_contract(contract_name)

    assert list(model.model_fields) == contract.column_names, "field names or order differ"
    for column in contract.columns:
        field = model.model_fields[column.name]
        base, optional = split_optional(field.annotation)
        assert optional == column.nullable, f"{column.name}: nullability differs"
        assert issubclass(base, EXPECTED_TYPES[column.type]), f"{column.name}: type differs"
        assert_constraints_match(column, field, base)


def assert_constraints_match(column: Column, field, base: type) -> None:
    c = column.constraints
    if c.min is not None:
        assert metadata_value(field, "ge") == c.min, f"{column.name}: min differs"
    else:
        assert metadata_value(field, "ge") is None, f"{column.name}: model has an extra minimum"
    if c.pattern is not None:
        assert metadata_value(field, "pattern") == c.pattern, f"{column.name}: pattern differs"
    if c.accepted_values is not None:
        assert {m.value for m in base} == set(c.accepted_values), f"{column.name}: values differ"


def test_pagination_response_items_are_the_mart_models():
    assert SettlementsPage.model_fields["items"].annotation == list[DailySettlement]
    assert BreaksPage.model_fields["items"].annotation == list[ReconciliationBreak]


def test_break_types_match_the_data_contracts():
    for name in ("injected_breaks", "reconciliation_breaks"):
        accepted = load_contract(name).column("break_type").constraints.accepted_values
        assert {b.value for b in BreakType} == set(accepted)


def test_mart_enum_is_exactly_the_mart_contracts():
    assert {m.value for m in Mart} == {
        "daily_settlement_by_merchant",
        "reconciliation_breaks",
        "fee_summary",
    }
    assert {m.value for m in Mart} <= set(list_contracts())


def test_timestamps_in_runs_must_carry_a_timezone():
    assert split_optional(Run.model_fields["started_at"].annotation)[0] is AwareDatetime


# --- OpenAPI ----------------------------------------------------------------------------------


def test_openapi_describes_the_v1_contract(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert set(paths) == {
        "/health",
        "/v1/settlements",
        "/v1/reconciliation/breaks",
        "/v1/exports/{mart}/{date}",
        "/v1/runs",
        "/v1/runs/{run_id}",
    }

    params = {p["name"] for p in paths["/v1/settlements"]["get"]["parameters"]}
    assert params == {"merchant_id", "from", "to", "cursor", "limit"}
    limit = next(p for p in paths["/v1/settlements"]["get"]["parameters"] if p["name"] == "limit")
    assert (limit["schema"]["minimum"], limit["schema"]["maximum"]) == (1, 1000)

    export = paths["/v1/exports/{mart}/{date}"]["get"]["responses"]
    assert list(export["200"]["content"]) == ["application/vnd.apache.parquet"]

    runs = paths["/v1/runs"]["post"]["responses"]
    assert {"200", "202", "422", "501"} <= set(runs)
    assert "Idempotent by `run_date`" in paths["/v1/runs"]["post"]["description"]

    for operation in (paths["/v1/settlements"]["get"], paths["/v1/runs/{run_id}"]["get"]):
        content = operation["responses"]["501"]["content"]
        assert list(content) == ["application/problem+json"]

    components = schema["components"]["schemas"]
    assert set(components["Mart"]["enum"]) == {m.value for m in Mart}
    assert "next_cursor" in components["SettlementsPage"]["properties"]
