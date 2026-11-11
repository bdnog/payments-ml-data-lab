"""Routes of the settlement-service API v1.

Plain ``def`` handlers: reading Parquet with Polars is blocking, CPU-bound work, so FastAPI runs
the handler in its threadpool instead of on the event loop (same reasoning as ADR 0009).

Reads scan the Hive-partitioned marts lazily with the date filter pushed down to the partitions,
sort by the key, and fetch ``limit + 1`` rows to know whether another page exists (keyset
pagination, see ``storage/cursors.py``).
"""

import datetime as dt
from typing import Annotated

import polars as pl
from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from payments_shared.problem import problem_response, problem_responses, problem_type
from settlement.api.deps import get_repository, get_store
from settlement.api.errors import InvalidParameterError, PartitionNotFoundError
from settlement.api.schemas import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    BreaksPage,
    BreakType,
    DailySettlement,
    Mart,
    ReconciliationBreak,
    Run,
    RunRequest,
    SettlementsPage,
)
from settlement.persistence.repository import RunRecord, RunRepository
from settlement.storage.cursors import (
    decode_break_cursor,
    decode_settlement_cursor,
    encode_break_cursor,
    encode_settlement_cursor,
)
from settlement.storage.marts import PARTITION_COLUMN, MartStore, collect_with_retry

router = APIRouter(prefix="/v1")

Store = Annotated[MartStore, Depends(get_store)]
Repository = Annotated[RunRepository, Depends(get_repository)]

Cursor = Annotated[
    str | None,
    Query(description="Opaque cursor from a previous page's `next_cursor`."),
]
Limit = Annotated[
    int,
    Query(ge=1, le=MAX_PAGE_SIZE, description="Maximum number of items in the page."),
]

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
DAILY = Mart.DAILY_SETTLEMENT_BY_MERCHANT.value
BREAKS = Mart.RECONCILIATION_BREAKS.value
PAGE_PROBLEMS = problem_responses(400, 422)
PAGE_PROBLEMS[400] = {**PAGE_PROBLEMS[400], "description": "Invalid cursor"}

SETTLEMENT_COLUMNS = list(DailySettlement.model_fields)
BREAK_COLUMNS = list(ReconciliationBreak.model_fields)


def _after(date_column: str, id_column: str, last_date: dt.date, last_id: str) -> pl.Expr:
    """Keyset condition: rows strictly after ``(last_date, last_id)`` in sort order."""
    day, key = pl.col(date_column), pl.col(id_column)
    return (day > last_date) | ((day == last_date) & (key > last_id))


@router.get(
    "/settlements",
    operation_id="list_settlements",
    summary="Daily settlement per merchant",
    description=(
        "Rows of the `daily_settlement_by_merchant` mart, ordered by `settlement_date`, "
        "`merchant_id`: the **ledger-expected payout** for each day, not what the PSP confirmed. "
        "Cursor pagination: pass `next_cursor` back as `cursor` until it is null. An invalid "
        "cursor is a `400`."
    ),
    response_model=SettlementsPage,
    responses=PAGE_PROBLEMS,
    tags=["settlements"],
)
def list_settlements(
    store: Store,
    merchant_id: Annotated[str | None, Query(pattern=r"^mer_[0-9a-f]{6}$")] = None,
    from_date: Annotated[
        dt.date | None, Query(alias="from", description="First settlement date, inclusive.")
    ] = None,
    to_date: Annotated[
        dt.date | None, Query(alias="to", description="Last settlement date, inclusive.")
    ] = None,
    cursor: Cursor = None,
    limit: Limit = DEFAULT_PAGE_SIZE,
) -> SettlementsPage:
    _check_range(from_date, to_date)
    last = decode_settlement_cursor(cursor) if cursor else None

    def query() -> pl.LazyFrame:
        # A cursor also raises the lower date bound, so earlier partitions are never opened.
        lower = _latest(from_date, last[0] if last else None)
        lazy = store.scan(DAILY, lower, to_date)
        if merchant_id:
            lazy = lazy.filter(pl.col("merchant_id") == merchant_id)
        if last:
            lazy = lazy.filter(_after(PARTITION_COLUMN, "merchant_id", *last))
        return (
            lazy.select(SETTLEMENT_COLUMNS)
            .sort(["settlement_date", "merchant_id", "currency"])
            .head(limit + 1)
        )

    rows = collect_with_retry(query).to_dicts()
    items, more = rows[:limit], len(rows) > limit
    next_cursor = (
        encode_settlement_cursor(items[-1]["settlement_date"], items[-1]["merchant_id"])
        if more
        else None
    )
    return SettlementsPage(
        items=[DailySettlement.model_validate(r) for r in items], next_cursor=next_cursor
    )


@router.get(
    "/reconciliation/breaks",
    operation_id="list_reconciliation_breaks",
    summary="Reconciliation breaks",
    description=(
        "Rows of the `reconciliation_breaks` mart, ordered by `settlement_date`, `break_id`. "
        "`date` is the business date of the run that found the break (`run_date`), which equals "
        "the settlement date of every break of that run. Cursor pagination, as for settlements; "
        "an invalid cursor is a `400`."
    ),
    response_model=BreaksPage,
    responses=PAGE_PROBLEMS,
    tags=["reconciliation"],
)
def list_breaks(
    store: Store,
    date: Annotated[dt.date | None, Query(description="Run date (`run_date`) to list.")] = None,
    type: Annotated[BreakType | None, Query(description="Only breaks of this type.")] = None,
    cursor: Cursor = None,
    limit: Limit = DEFAULT_PAGE_SIZE,
) -> BreaksPage:
    last = decode_break_cursor(cursor) if cursor else None

    def query() -> pl.LazyFrame:
        lower = _latest(date, last[0] if last else None)
        lazy = store.scan(BREAKS, lower, date)
        if type:
            lazy = lazy.filter(pl.col("break_type") == type.value)
        if last:
            lazy = lazy.filter(_after(PARTITION_COLUMN, "break_id", *last))
        return lazy.select(BREAK_COLUMNS).sort(["settlement_date", "break_id"]).head(limit + 1)

    rows = collect_with_retry(query).to_dicts()
    items, more = rows[:limit], len(rows) > limit
    next_cursor = (
        encode_break_cursor(items[-1]["settlement_date"], items[-1]["break_id"]) if more else None
    )
    return BreaksPage(
        items=[ReconciliationBreak.model_validate(r) for r in items], next_cursor=next_cursor
    )


@router.get(
    "/exports/{mart}/{date}",
    operation_id="export_mart",
    summary="Download a mart as Parquet",
    description=(
        "The mart produced by the run of the given business date, as a Parquet file (zstd). The "
        "format is carried by the media type, so the URL has no `.parquet` suffix. A date with no "
        "partition (the run never happened) is a `404`; a run that found nothing is an empty file "
        "with the right schema."
    ),
    response_class=FileResponse,
    responses={
        200: {
            "description": "The mart as a Parquet file.",
            "content": {PARQUET_MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}},
        },
        **problem_responses(404, 422),
    },
    tags=["exports"],
)
def export_mart(
    store: Store,
    mart: Annotated[Mart, Path(description="Which mart to export.")],
    date: Annotated[dt.date, Path(description="Run date (`run_date`) of the mart.")],
):
    # Both path parameters are parsed (an enum and a date), never concatenated from user text.
    if not store.has_partition(mart.value, date):
        raise PartitionNotFoundError(f"no {mart.value} partition for {date.isoformat()}")
    return FileResponse(
        store.partition_file(mart.value, date),
        media_type=PARQUET_MEDIA_TYPE,
        filename=f"{mart.value}-{date.isoformat()}.parquet",
    )


@router.post(
    "/runs",
    operation_id="create_run",
    summary="Request a settlement run",
    description=(
        "Idempotent by `run_date`. `202` with a `Location` header when a new run is accepted (a "
        "worker will execute it); `200` with the existing run when that date already has a run "
        "that has not failed. A `FAILED` run does not block a new request."
    ),
    status_code=202,
    response_model=Run,
    responses={
        202: {
            "description": "A new run was accepted and is PENDING.",
            "headers": {
                "Location": {
                    "description": "URL of the new run, `/v1/runs/{run_id}`.",
                    "schema": {"type": "string"},
                }
            },
        },
        200: {"model": Run, "description": "A non-failed run for this date already exists."},
        **problem_responses(422, 503),
    },
    tags=["runs"],
)
def create_run(body: RunRequest, repository: Repository) -> JSONResponse:
    result = repository.create(body.run_date)
    run = _to_run(result.run)
    headers = {"Location": f"/v1/runs/{run.run_id}"} if result.created else None
    return JSONResponse(
        run.model_dump(mode="json"), status_code=202 if result.created else 200, headers=headers
    )


@router.get(
    "/runs/{run_id}",
    operation_id="get_run",
    summary="Get a settlement run",
    response_model=Run,
    responses=problem_responses(404, 422, 503),
    tags=["runs"],
)
def get_run(
    request: Request,
    repository: Repository,
    run_id: Annotated[str, Path(pattern=r"^run_[0-9a-f]{12}$")],
):
    record = repository.get(run_id)
    if record is None:
        return problem_response(
            request,
            404,
            "Run not found",
            f"There is no run {run_id}.",
            type=problem_type("run-not-found"),
        )
    return _to_run(record)


def _to_run(record: RunRecord) -> Run:
    return Run(
        run_id=record.run_id,
        run_date=record.run_date,
        status=record.status,
        requested_at=record.requested_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        error=record.error,
    )


def _latest(*dates: dt.date | None) -> dt.date | None:
    present = [d for d in dates if d is not None]
    return max(present) if present else None


def _check_range(from_date: dt.date | None, to_date: dt.date | None) -> None:
    if from_date and to_date and from_date > to_date:
        raise InvalidParameterError("`from` must not be after `to`")
