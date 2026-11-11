"""Fixtures shared by the risk-scoring tests."""

import polars as pl
import pytest

from payments_shared.paths import repo_root

SAMPLE_TRANSACTIONS = repo_root() / "data" / "samples" / "risk" / "transactions.parquet"


@pytest.fixture(scope="session")
def sample_transactions() -> pl.DataFrame:
    """The committed Phase 2 sample (3,000 rows): real rows, cheap to load."""
    return pl.read_parquet(SAMPLE_TRANSACTIONS)
