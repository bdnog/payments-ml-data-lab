"""Conversions between a ``transactions`` row, a ``ScoreRequest`` and a feature frame.

The API receives nested JSON; training reads flat Parquet rows. This module is the only place that
knows how the two line up (via ``FEATURE_PATHS`` and ``IDENTITY_PATHS``, which a contract test keeps
in sync with the data contract). Tests and scripts reuse it to replay dataset rows through the API.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import polars as pl

from payments_shared.data_contracts import load_contract
from risk_scoring.api.schemas import FEATURE_PATHS, IDENTITY_PATHS, ScoreRequest
from risk_scoring.training.features import required_columns


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    for parent in parents:
        target = target.setdefault(parent, {})
    target[leaf] = value


def _get_path(source: Any, path: str) -> Any:
    for part in path.split("."):
        source = getattr(source, part)
    return source


def row_to_payload(row: dict[str, Any]) -> dict[str, Any]:
    """A ``transactions`` row (as a dict) as the nested JSON-like body of ``POST /v1/score``."""
    payload: dict[str, Any] = {}
    for column, path in {**IDENTITY_PATHS, **FEATURE_PATHS}.items():
        _set_path(payload, path, row[column])
    return payload


def row_to_request(row: dict[str, Any]) -> ScoreRequest:
    return ScoreRequest.model_validate(row_to_payload(row))


def request_to_feature_row(request: ScoreRequest) -> dict[str, Any]:
    """The raw feature columns of a request, keyed by data-contract column name."""
    return {column: _get_path(request, FEATURE_PATHS[column]) for column in required_columns()}


def requests_to_frame(requests: Sequence[ScoreRequest]) -> pl.DataFrame:
    """A frame with exactly the contract's feature columns and dtypes: the serving input."""
    contract = load_contract("transactions")
    schema = {name: contract.schema[name] for name in required_columns()}
    rows = [request_to_feature_row(r) for r in requests]
    return pl.DataFrame(rows, schema=schema)


def feature_frame(transactions: pl.DataFrame) -> pl.DataFrame:
    """The training input: the contract's feature columns of a ``transactions`` frame."""
    return transactions.select(required_columns())


def iter_requests(transactions: pl.DataFrame) -> Iterable[ScoreRequest]:
    for row in transactions.iter_rows(named=True):
        yield row_to_request(row)
