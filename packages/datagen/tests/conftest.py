"""Shared fixtures. Worlds are built once per session: generation is deterministic and read-only."""

import pytest

from payments_datagen.generate import generate_world
from payments_datagen.profiles import SAMPLE, Profile

# Larger than the committed sample so statistical checks are tight, still well under a second.
STATS = Profile(
    name="stats",
    n_accounts=1_500,
    n_merchants=40,
    days=60,
    target_transactions=40_000,
    fraud_rate=0.02,
    break_rate=0.05,
)


@pytest.fixture(scope="session")
def sample_world():
    return generate_world(SAMPLE)


@pytest.fixture(scope="session")
def stats_world():
    return generate_world(STATS)


@pytest.fixture(scope="session")
def stats_world_without_breaks():
    return generate_world(STATS.with_overrides(break_rate=0.0))
