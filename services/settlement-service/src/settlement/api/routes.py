"""Routes of the settlement-service API v1.

Only ``/health`` (in ``main.py``) does real work. The ``/v1`` routes are the *contract*: they
declare parameters, models and every status code, and answer ``501`` until phase 4.
"""

import datetime as dt
from typing import Annotated

from fastapi import APIRouter, Path, Query, Request, Response

from payments_shared.problem import not_implemented, problem_responses
from settlement.api.schemas import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    BreaksPage,
    BreakType,
    Mart,
    Run,
    RunRequest,
    SettlementsPage,
)

router = APIRouter(prefix="/v1")

Cursor = Annotated[
    str | None,
    Query(description="Opaque cursor from a previous page's `next_cursor`."),
]
Limit = Annotated[
    int,
    Query(ge=1, le=MAX_PAGE_SIZE, description="Maximum number of items in the page."),
]

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"


@router.get(
    "/settlements",
    operation_id="list_settlements",
    summary="Daily settlement per merchant",
    description=(
        "Rows of the `daily_settlement_by_merchant` mart, ordered by `settlement_date`, "
        "`merchant_id`. Cursor pagination: pass `next_cursor` back as `cursor` until it is null."
    ),
    response_model=SettlementsPage,
    responses=problem_responses(422, 501),
    tags=["settlements"],
)
def list_settlements(
    request: Request,
    merchant_id: Annotated[str | None, Query(pattern=r"^mer_[0-9a-f]{6}$")] = None,
    from_date: Annotated[
        dt.date | None, Query(alias="from", description="First settlement date, inclusive.")
    ] = None,
    to_date: Annotated[
        dt.date | None, Query(alias="to", description="Last settlement date, inclusive.")
    ] = None,
    cursor: Cursor = None,
    limit: Limit = DEFAULT_PAGE_SIZE,
):
    return not_implemented(request)


@router.get(
    "/reconciliation/breaks",
    operation_id="list_reconciliation_breaks",
    summary="Reconciliation breaks",
    description=(
        "Rows of the `reconciliation_breaks` mart. `date` is the business date of the run that "
        "found the break (`run_date`). Cursor pagination, as for settlements."
    ),
    response_model=BreaksPage,
    responses=problem_responses(422, 501),
    tags=["reconciliation"],
)
def list_breaks(
    request: Request,
    date: Annotated[dt.date | None, Query(description="Run date (`run_date`) to list.")] = None,
    type: Annotated[BreakType | None, Query(description="Only breaks of this type.")] = None,
    cursor: Cursor = None,
    limit: Limit = DEFAULT_PAGE_SIZE,
):
    return not_implemented(request)


@router.get(
    "/exports/{mart}/{date}",
    operation_id="export_mart",
    summary="Download a mart as Parquet",
    description=(
        "The mart produced by the run of the given business date, as a Parquet file. The format is "
        "carried by the media type, so the URL has no `.parquet` suffix."
    ),
    response_class=Response,
    responses={
        200: {
            "description": "The mart as a Parquet file.",
            "content": {PARQUET_MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}},
        },
        **problem_responses(404, 422, 501),
    },
    tags=["exports"],
)
def export_mart(
    request: Request,
    mart: Annotated[Mart, Path(description="Which mart to export.")],
    date: Annotated[dt.date, Path(description="Run date (`run_date`) of the mart.")],
):
    return not_implemented(request)


@router.post(
    "/runs",
    operation_id="create_run",
    summary="Request a settlement run",
    description=(
        "Idempotent by `run_date`. `202` when a new run is accepted; `200` with the existing run "
        "when that date already has a run that has not failed."
    ),
    status_code=202,
    response_model=Run,
    responses={200: {"model": Run, "description": "A non-failed run for this date already exists."}}
    | problem_responses(422, 501),
    tags=["runs"],
)
def create_run(body: RunRequest, request: Request):
    return not_implemented(request)


@router.get(
    "/runs/{run_id}",
    operation_id="get_run",
    summary="Get a settlement run",
    response_model=Run,
    responses=problem_responses(404, 422, 501),
    tags=["runs"],
)
def get_run(request: Request, run_id: Annotated[str, Path(pattern=r"^run_[0-9a-f]{12}$")]):
    return not_implemented(request)
