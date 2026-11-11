"""Fixtures shared by the settlement tests."""

from pathlib import Path

import pytest

from payments_datagen.generate import generate_world
from payments_datagen.profiles import Profile
from payments_datagen.storage import write_world
from payments_shared.paths import repo_root

SAMPLE_DIR = repo_root() / "data" / "samples"
SEEDS = (11, 22, 33)


def small_profile(seed: int) -> Profile:
    return Profile(
        name=f"settle-test-{seed}",
        n_accounts=160,
        n_merchants=14,
        days=45,
        target_transactions=3_000,
        fraud_rate=0.03,
        break_rate=0.08,
        seed=seed,
    )


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    """The committed Phase 2 sample (30 days, about 3,000 transactions)."""
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def seeded_datasets(tmp_path_factory) -> dict[int, Path]:
    """Three small generated datasets, one per seed, written once for the whole session."""
    datasets: dict[int, Path] = {}
    for seed in SEEDS:
        directory = tmp_path_factory.mktemp(f"dataset-{seed}")
        write_world(generate_world(small_profile(seed)), directory)
        datasets[seed] = directory
    return datasets


@pytest.fixture(scope="session", params=SEEDS, ids=[f"seed{s}" for s in SEEDS])
def seeded_dataset(request, seeded_datasets) -> Path:
    """Each seeded dataset in turn: tests that take this run once per seed."""
    return seeded_datasets[request.param]
