"""Fixtures shared by the risk-scoring tests."""

import polars as pl
import pytest

from payments_datagen.generate import generate_world
from payments_datagen.profiles import Profile
from payments_shared.paths import repo_root

SAMPLE_TRANSACTIONS = repo_root() / "data" / "samples" / "risk" / "transactions.parquet"


@pytest.fixture(scope="session")
def sample_transactions() -> pl.DataFrame:
    """The committed Phase 2 sample (3,000 rows): real rows, cheap to load."""
    return pl.read_parquet(SAMPLE_TRANSACTIONS)


# --- a small, synthetic world with a high fraud rate, for training tests --------------------------

# About 15,000 transactions with 5% fraud: enough positives to learn from, still fast to generate.
TRAINING_PROFILE = Profile(
    name="risk-test",
    n_accounts=500,
    n_merchants=40,
    days=120,
    target_transactions=15_000,
    fraud_rate=0.05,
    break_rate=0.02,
    seed=7,
)


@pytest.fixture(scope="session")
def training_transactions() -> pl.DataFrame:
    """Transactions of a small generated world (session-scoped: generated once)."""
    return generate_world(TRAINING_PROFILE)["transactions"]


@pytest.fixture(scope="session")
def prepared(training_transactions):
    """Split, labeled and encoded training world (the preprocessing is under test elsewhere)."""
    from risk_scoring.training.dataset import prepare_data

    return prepare_data(training_transactions, min_category_count=5)


@pytest.fixture(scope="session")
def trained(prepared):
    """A model trained on the small world, shared by every test that needs a real one."""
    from risk_scoring.training.trainer import TrainConfig, train_model

    train, validation = prepared.splits["train"], prepared.splits["validation"]
    return train_model(
        train.encoded,
        train.labels,
        validation.encoded,
        validation.labels,
        prepared.preprocessor.vocab_sizes,
        TrainConfig(max_epochs=15, patience=4),
    )
