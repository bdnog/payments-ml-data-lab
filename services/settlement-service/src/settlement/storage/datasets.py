"""The pipeline's inputs: lazy readers for the Phase 2 dataset layout, plus validation.

The layout is the output directory of ``payments_datagen generate``::

    <input_dir>/reference/merchants.parquet
    <input_dir>/settlement/{payments,refunds,chargebacks,psp_settlement}.parquet

``injected_breaks.parquet`` lives next to them but is the generator's answer key: the pipeline does
not read it (only ``settlement.batch.verification`` and tests do), and nothing here can load it.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from payments_shared.data_contracts import (
    ContractViolationError,
    Violation,
    check_foreign_key,
    load_contract,
    validate_dataframe,
)

# dataset (= data contract name) -> path relative to the input directory
INPUT_PATHS: dict[str, str] = {
    "merchants": "reference/merchants.parquet",
    "payments": "settlement/payments.parquet",
    "refunds": "settlement/refunds.parquet",
    "chargebacks": "settlement/chargebacks.parquet",
    "psp_settlement": "settlement/psp_settlement.parquet",
}

# (child dataset, child column, parent dataset, parent column)
FOREIGN_KEYS: tuple[tuple[str, str, str, str], ...] = (
    ("payments", "merchant_id", "merchants", "merchant_id"),
    ("refunds", "payment_id", "payments", "payment_id"),
    ("chargebacks", "payment_id", "payments", "payment_id"),
    ("psp_settlement", "merchant_id", "merchants", "merchant_id"),
)


class InputValidationError(ContractViolationError):
    """The input datasets break their data contracts or a foreign key."""


@dataclass(frozen=True)
class InputDatasets:
    merchants: pl.LazyFrame
    payments: pl.LazyFrame
    refunds: pl.LazyFrame
    chargebacks: pl.LazyFrame
    psp: pl.LazyFrame


def input_file(input_dir: Path, dataset: str) -> Path:
    return Path(input_dir) / INPUT_PATHS[dataset]


def scan_inputs(input_dir: Path) -> InputDatasets:
    """Lazy readers for the five input datasets; fails early and clearly if one is missing."""
    missing = [
        str(input_file(input_dir, name))
        for name in INPUT_PATHS
        if not input_file(input_dir, name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"input dataset(s) not found: {', '.join(missing)}. Generate a dataset first, e.g. "
            "`uv run python -m payments_datagen generate --profile sample --out DIR`"
        )
    scan = {name: pl.scan_parquet(input_file(input_dir, name)) for name in INPUT_PATHS}
    return InputDatasets(
        merchants=scan["merchants"],
        payments=scan["payments"],
        refunds=scan["refunds"],
        chargebacks=scan["chargebacks"],
        psp=scan["psp_settlement"],
    )


def validate_inputs(input_dir: Path) -> None:
    """Check each input against its data contract, then the foreign keys. Raises on any violation.

    Reads the datasets eagerly (about a second for 500k payments): a run must not start on data
    that is not what the contracts say, because every later number would be wrong without a crash.
    """
    frames = {name: pl.read_parquet(input_file(input_dir, name)) for name in INPUT_PATHS}
    violations: list[Violation] = []
    for name, frame in frames.items():
        violations += validate_dataframe(frame, load_contract(name))
    for child, child_col, parent, parent_col in FOREIGN_KEYS:
        violations += check_foreign_key(frames[child], child_col, frames[parent], parent_col)
    if violations:
        raise InputValidationError(violations)


def psp_date_range(input_dir: Path) -> tuple[date, date]:
    """First and last ``settlement_date`` of the PSP file: the dates a full run covers."""
    bounds = (
        pl.scan_parquet(input_file(input_dir, "psp_settlement"))
        .select(
            pl.col("settlement_date").min().alias("first"),
            pl.col("settlement_date").max().alias("last"),
        )
        .collect()
    )
    first, last = bounds["first"][0], bounds["last"][0]
    if first is None or last is None:
        raise ValueError("the PSP file is empty: there is nothing to settle")
    return first, last
