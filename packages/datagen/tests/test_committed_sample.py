"""The sample committed under ``data/samples`` must stay valid.

It is refreshed with ``make data-sample``. These tests deliberately do not compare it with a fresh
generation: a library upgrade may change the random stream, and that must not fail the build.
"""

from payments_datagen.generate import DATASET_PATHS
from payments_datagen.manifest import MANIFEST_NAME, read_manifest
from payments_datagen.validate import validate_directory
from payments_shared.paths import repo_root

SAMPLE_DIR = repo_root() / "data" / "samples"
MAX_BYTES = 1_000_000


def test_committed_sample_passes_validation():
    report = validate_directory(SAMPLE_DIR, determinism=False)
    assert report.ok, report.render()


def test_committed_sample_describes_the_sample_profile():
    manifest = read_manifest(SAMPLE_DIR / MANIFEST_NAME)
    assert manifest["profile"] == "sample"
    assert manifest["seed"] == 42


def test_committed_sample_is_small_and_complete():
    files = [SAMPLE_DIR / relative for relative in DATASET_PATHS.values()]
    files.append(SAMPLE_DIR / MANIFEST_NAME)
    assert all(path.is_file() for path in files)
    assert sum(path.stat().st_size for path in SAMPLE_DIR.rglob("*") if path.is_file()) < MAX_BYTES
