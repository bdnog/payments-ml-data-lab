"""Behavior of the real endpoints, on the marts of a real run over the sample."""

import base64
import io
import json
import re
import shutil
import threading
from datetime import date

import polars as pl
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from payments_shared.schemas import ProblemDetails
from settlement.api.main import create_app
from settlement.api.schemas import Run
from settlement.batch.pipeline import run_dates
from settlement.config import Settings
from settlement.persistence.repository import InMemoryRunRepository
from settlement.storage.cursors import encode_break_cursor, encode_settlement_cursor
from settlement.storage.marts import MartStore


def assert_problem(response, status: int, slug: str | None = None) -> dict:
    assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert ProblemDetails.model_validate(body).status == status
    assert re.fullmatch(r"urn:problem:[a-z-]+", body["type"])
    if slug:
        assert body["type"] == f"urn:problem:{slug}"
    return body


def walk(client: TestClient, url: str, limit: int) -> list[dict]:
    """Follow ``next_cursor`` until it is null, returning every item in order."""
    items: list[dict] = []
    cursor = None
    separator = "&" if "?" in url else "?"
    for _ in range(10_000):
        suffix = f"{separator}limit={limit}" + (f"&cursor={cursor}" if cursor else "")
        page = client.get(url + suffix).json()
        items += page["items"]
        cursor = page["next_cursor"]
        if cursor is None:
            return items
    raise AssertionError("pagination did not terminate")


@pytest.fixture
def store(marts_dir) -> MartStore:
    return MartStore(marts_dir)


# --- settlements ------------------------------------------------------------------------------


def test_walking_all_pages_yields_every_row_exactly_once_in_stable_order(client, store):
    expected = (
        store.scan("daily_settlement_by_merchant")
        .collect()
        .sort(["settlement_date", "merchant_id"])
    )

    for limit in (1, 7, 100, 1000):
        items = walk(client, "/v1/settlements", limit)
        assert len(items) == expected.height, limit
        keys = [(i["settlement_date"], i["merchant_id"]) for i in items]
        assert keys == sorted(keys) and len(set(keys)) == len(keys), limit
        assert (
            pl.DataFrame(items)
            .with_columns(pl.col("settlement_date").str.to_date())
            .drop("currency")
            .equals(expected.drop("currency"))
        )


def test_default_page_size_is_100_and_the_last_page_has_no_cursor(client):
    page = client.get("/v1/settlements").json()
    assert len(page["items"]) == 100 and page["next_cursor"]
    everything = client.get("/v1/settlements?limit=1000").json()
    assert len(everything["items"]) > 100


def test_a_page_that_ends_exactly_at_the_data_has_no_cursor(client, store):
    total = store.scan("daily_settlement_by_merchant").collect().height
    page = client.get(f"/v1/settlements?limit={total}").json()
    assert len(page["items"]) == total and page["next_cursor"] is None


def test_settlement_rows_have_the_contract_fields(client):
    row = client.get("/v1/settlements?limit=1").json()["items"][0]
    assert list(row) == [
        "settlement_date", "merchant_id", "currency", "payment_count", "gross_minor",
        "fee_minor", "refunds_minor", "chargebacks_minor", "net_minor",
    ]  # fmt: skip
    assert row["net_minor"] == (
        row["gross_minor"] - row["fee_minor"] - row["refunds_minor"] - row["chargebacks_minor"]
    )


def test_filter_by_merchant(client, store):
    daily = store.scan("daily_settlement_by_merchant").collect()
    merchant = daily["merchant_id"].value_counts().sort("count")["merchant_id"][-1]

    items = walk(client, f"/v1/settlements?merchant_id={merchant}", 5)

    assert items and {i["merchant_id"] for i in items} == {merchant}
    assert len(items) == daily.filter(pl.col("merchant_id") == merchant).height


def test_filter_by_date_range_is_inclusive(client):
    items = walk(client, "/v1/settlements?from=2026-01-10&to=2026-01-12", 50)
    assert {i["settlement_date"] for i in items} == {"2026-01-10", "2026-01-11", "2026-01-12"}


def test_a_range_without_data_or_an_unknown_merchant_is_an_empty_page(client):
    for url in (
        "/v1/settlements?from=2030-01-01",
        "/v1/settlements?merchant_id=mer_ffffff",
        "/v1/settlements?to=2025-12-31",
    ):
        assert client.get(url).json() == {"items": [], "next_cursor": None}


def test_from_after_to_is_a_422_problem(client):
    body = assert_problem(
        client.get("/v1/settlements?from=2026-01-12&to=2026-01-10"), 422, "validation-error"
    )
    assert "from" in body["detail"]


def test_filters_and_cursor_compose(client):
    first = client.get("/v1/settlements?from=2026-01-05&to=2026-01-20&limit=10").json()
    second = client.get(
        f"/v1/settlements?from=2026-01-05&to=2026-01-20&limit=10&cursor={first['next_cursor']}"
    ).json()
    assert second["items"][0] != first["items"][-1]
    last = (first["items"][-1]["settlement_date"], first["items"][-1]["merchant_id"])
    assert (second["items"][0]["settlement_date"], second["items"][0]["merchant_id"]) > last


def test_a_cursor_past_the_end_gives_an_empty_page(client):
    cursor = encode_settlement_cursor(date(2030, 1, 1), "mer_000000")
    assert client.get(f"/v1/settlements?cursor={cursor}").json() == {
        "items": [],
        "next_cursor": None,
    }


# --- breaks -----------------------------------------------------------------------------------


def test_walking_all_break_pages_yields_every_row_once_in_order(client, store):
    expected = store.scan("reconciliation_breaks").collect()
    items = walk(client, "/v1/reconciliation/breaks", 13)

    assert len(items) == expected.height == 160
    keys = [(i["settlement_date"], i["break_id"]) for i in items]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)


def test_filter_breaks_by_type(client):
    items = walk(client, "/v1/reconciliation/breaks?type=FEE_MISMATCH", 8)
    assert len(items) == 21 and {i["break_type"] for i in items} == {"FEE_MISMATCH"}


def test_filter_breaks_by_run_date(client):
    everything = walk(client, "/v1/reconciliation/breaks", 50)
    day = everything[0]["settlement_date"]

    items = walk(client, f"/v1/reconciliation/breaks?date={day}", 50)

    assert items and {i["run_date"] for i in items} == {day} == {
        i["settlement_date"] for i in items
    }
    assert len(items) == sum(1 for i in everything if i["settlement_date"] == day)


def test_filter_by_date_and_type_together(client):
    items = walk(client, "/v1/reconciliation/breaks?date=2026-01-15&type=MISSING_IN_PSP", 5)
    assert all(i["break_type"] == "MISSING_IN_PSP" and i["run_date"] == "2026-01-15" for i in items)


def test_a_date_that_was_never_run_is_an_empty_page(client):
    assert client.get("/v1/reconciliation/breaks?date=2031-01-01").json() == {
        "items": [],
        "next_cursor": None,
    }


def test_break_items_follow_the_taxonomy_amounts(client):
    items = walk(client, "/v1/reconciliation/breaks", 200)
    for item in items:
        ledger, psp = item["ledger_amount_minor"], item["psp_amount_minor"]
        match item["break_type"]:
            case "MISSING_IN_PSP":
                assert psp is None and item["diff_minor"] == -ledger
            case "MISSING_IN_LEDGER" | "CHARGEBACK_NOT_IN_LEDGER" | "DUPLICATE_IN_PSP":
                assert ledger is None and item["diff_minor"] == psp
            case _:
                assert item["diff_minor"] == psp - ledger


# --- cursors ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cursor",
    ["garbage", "e30", "AAAA", "%20", "eyJkIjogIjIwMjYtMDEtMTQifQ", "x" * 300],
)
@pytest.mark.parametrize("url", ["/v1/settlements", "/v1/reconciliation/breaks"])
def test_a_malformed_cursor_is_a_400_problem(client, url, cursor):
    body = assert_problem(client.get(f"{url}?cursor={cursor}"), 400, "invalid-cursor")
    assert "next_cursor" in body["detail"]


def test_a_cursor_of_the_other_listing_is_rejected(client):
    break_cursor = encode_break_cursor(date(2026, 1, 10), "brk_50140001")
    settlement_cursor = encode_settlement_cursor(date(2026, 1, 10), "mer_000001")
    assert_problem(client.get(f"/v1/settlements?cursor={break_cursor}"), 400, "invalid-cursor")
    assert_problem(
        client.get(f"/v1/reconciliation/breaks?cursor={settlement_cursor}"), 400, "invalid-cursor"
    )


def test_a_hand_edited_cursor_with_a_bad_id_is_rejected(client):
    forged = base64.urlsafe_b64encode(json.dumps({"d": "2026-01-10", "m": "x'; --"}).encode())
    assert_problem(
        client.get(f"/v1/settlements?cursor={forged.decode().rstrip('=')}"), 400, "invalid-cursor"
    )


@pytest.mark.parametrize("limit", [0, -1, 1001, "abc"])
def test_limit_bounds(client, limit):
    assert_problem(client.get(f"/v1/settlements?limit={limit}"), 422, "validation-error")


def test_the_maximum_limit_is_accepted(client):
    assert client.get("/v1/settlements?limit=1000").status_code == 200


# --- exports ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mart", ["daily_settlement_by_merchant", "reconciliation_breaks", "fee_summary"]
)
def test_an_export_is_the_partition_file(client, store, mart):
    response = client.get(f"/v1/exports/{mart}/2026-01-15")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.apache.parquet"
    assert f"{mart}-2026-01-15.parquet" in response.headers["content-disposition"]
    frame = pl.read_parquet(io.BytesIO(response.content))
    assert frame.equals(store.read_partition(mart, date(2026, 1, 15)))
    assert set(frame["settlement_date"]) == {date(2026, 1, 15)}


def test_a_date_without_a_partition_is_a_404_problem(client):
    body = assert_problem(
        client.get("/v1/exports/fee_summary/2031-01-01"), 404, "partition-not-found"
    )
    assert "2031-01-01" in body["detail"]


def test_an_empty_partition_is_an_empty_file_not_a_404(tmp_path, sample_dir):
    marts = tmp_path / "marts"
    run_dates([date(2031, 1, 1)], sample_dir, marts)  # a date with nothing to settle
    app = create_app(Settings(marts_dir=marts), repository=InMemoryRunRepository())
    with TestClient(app) as client:
        response = client.get("/v1/exports/reconciliation_breaks/2031-01-01")

    assert response.status_code == 200
    assert pl.read_parquet(io.BytesIO(response.content)).height == 0


@pytest.mark.parametrize(
    "url",
    [
        "/v1/exports/not_a_mart/2026-01-15",
        "/v1/exports/..%2F..%2Fetc%2Fpasswd/2026-01-15",
        "/v1/exports/fee_summary/..%2F..%2Fetc%2Fpasswd",
        "/v1/exports/fee_summary/2026-01-15%2F..%2F..%2F..%2Fsecret",
        "/v1/exports/fee_summary/20260115",
        "/v1/exports/fee_summary/2026-13-45",
        "/v1/exports/%2e%2e/%2e%2e/2026-01-15",
    ],
)
def test_path_traversal_and_bad_parameters_are_rejected(client, url):
    response = client.get(url)
    assert response.status_code in (404, 422), (url, response.status_code)
    assert response.headers["content-type"].split(";")[0] != "application/vnd.apache.parquet"


def test_the_staging_directory_is_not_exportable(client, marts_dir):
    (marts_dir / ".staging").mkdir(exist_ok=True)
    assert client.get("/v1/exports/.staging/2026-01-15").status_code in (404, 422)


# --- runs -------------------------------------------------------------------------------------


def test_post_a_run_is_accepted_with_a_location_header(client):
    response = client.post("/v1/runs", json={"run_date": "2026-01-15"})

    assert response.status_code == 202
    run = Run.model_validate(response.json())
    assert run.status == "PENDING" and run.run_date == date(2026, 1, 15)
    assert response.headers["location"] == f"/v1/runs/{run.run_id}"


def test_posting_the_same_date_again_returns_the_same_run_with_200(client):
    first = client.post("/v1/runs", json={"run_date": "2026-01-15"})
    second = client.post("/v1/runs", json={"run_date": "2026-01-15"})

    assert (first.status_code, second.status_code) == (202, 200)
    assert second.json() == first.json()
    assert "location" not in second.headers


def test_a_different_date_is_a_different_run(client):
    a = client.post("/v1/runs", json={"run_date": "2026-01-15"}).json()
    b = client.post("/v1/runs", json={"run_date": "2026-01-16"}).json()
    assert a["run_id"] != b["run_id"]


def test_get_a_run(client):
    created = client.post("/v1/runs", json={"run_date": "2026-01-15"}).json()
    assert client.get(f"/v1/runs/{created['run_id']}").json() == created


def test_get_an_unknown_run_is_a_404_problem(client):
    assert_problem(client.get("/v1/runs/run_ffffffffffff"), 404, "run-not-found")


def test_a_run_goes_through_its_lifecycle_in_the_api(client, repository):
    created = client.post("/v1/runs", json={"run_date": "2026-01-15"}).json()
    run = repository.claim_next()
    assert client.get(f"/v1/runs/{created['run_id']}").json()["status"] == "RUNNING"
    repository.complete(run.run_id, run.attempts, {"breaks": 1})

    body = client.get(f"/v1/runs/{created['run_id']}").json()

    assert body["status"] == "DONE" and body["finished_at"] and body["error"] is None
    Run.model_validate(body)
    assert "summary" not in body and "attempts" not in body, "internal columns are not exposed"


def test_a_failed_run_does_not_block_a_new_request(client, repository):
    first = client.post("/v1/runs", json={"run_date": "2026-01-15"}).json()
    run = repository.claim_next()
    repository.fail(run.run_id, run.attempts, "boom")
    assert client.get(f"/v1/runs/{first['run_id']}").json()["error"] == "boom"

    again = client.post("/v1/runs", json={"run_date": "2026-01-15"})

    assert again.status_code == 202 and again.json()["run_id"] != first["run_id"]


@pytest.mark.parametrize("body", [{}, {"run_date": "2026-13-01"}, {"run_date": "yesterday"}])
def test_a_bad_run_request_is_a_422_problem(client, body):
    assert_problem(client.post("/v1/runs", json=body), 422, "validation-error")


def test_a_database_failure_is_a_503_problem(marts_dir):
    class Broken:
        def create(self, *args, **kwargs):
            raise OperationalError("INSERT", {}, Exception("connection refused"))

        def get(self, *args, **kwargs):
            raise OperationalError("SELECT", {}, Exception("connection refused"))

    app = create_app(Settings(marts_dir=marts_dir), repository=Broken())  # type: ignore[arg-type]
    with TestClient(app, raise_server_exceptions=False) as client:
        assert_problem(
            client.post("/v1/runs", json={"run_date": "2026-01-15"}), 503, "database-unavailable"
        )
        assert_problem(client.get("/v1/runs/run_000000000001"), 503)
        assert client.get("/v1/settlements?limit=1").status_code == 200, "marts do not need the DB"


# --- readiness --------------------------------------------------------------------------------


def test_ready_is_200_when_the_database_answers_and_the_marts_are_readable(client):
    response = client.get("/ready")
    assert response.status_code == 200 and response.json() == {"status": "ready"}


def test_ready_is_503_when_the_database_is_down(make_client):
    def down():
        raise ConnectionError("no route to host")

    (client,) = make_client(database_check=down)
    body = assert_problem(client.get("/ready"), 503, "not-ready")
    assert "database" in body["detail"]
    assert client.get("/health").status_code == 200, "liveness does not depend on it"


def test_ready_is_503_when_the_marts_directory_is_unreadable(make_client, tmp_path):
    (client,) = make_client(marts_dir=tmp_path / "does-not-exist")
    body = assert_problem(client.get("/ready"), 503, "not-ready")
    assert "marts directory" in body["detail"]


def test_an_api_over_empty_marts_answers_empty_pages(tmp_path):
    marts = tmp_path / "marts"
    marts.mkdir()
    app = create_app(Settings(marts_dir=marts), repository=InMemoryRunRepository())
    with TestClient(app) as client:
        assert client.get("/v1/settlements").json() == {"items": [], "next_cursor": None}
        assert client.get("/v1/reconciliation/breaks").json() == {"items": [], "next_cursor": None}
        assert client.get("/ready").status_code == 200


def test_reads_survive_a_rewrite_of_the_partition_they_are_reading(marts_dir, tmp_path):
    """The worker may overwrite a partition while the API serves it (see ADR 0011)."""
    marts = tmp_path / "marts"
    shutil.copytree(marts_dir, marts)
    store = MartStore(marts)
    stop = threading.Event()
    day = date(2026, 1, 15)
    frame = store.read_partition("reconciliation_breaks", day)

    def rewriter():
        while not stop.is_set():
            store.write_partition("reconciliation_breaks", day, frame)

    app = create_app(Settings(marts_dir=marts), repository=InMemoryRunRepository())
    thread = threading.Thread(target=rewriter)
    thread.start()
    try:
        with TestClient(app) as client:
            for _ in range(60):
                response = client.get(f"/v1/reconciliation/breaks?date={day}&limit=1000")
                assert response.status_code == 200
                assert len(response.json()["items"]) == frame.height
    finally:
        stop.set()
        thread.join(timeout=10)
