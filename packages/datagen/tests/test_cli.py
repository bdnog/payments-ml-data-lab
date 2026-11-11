import json
import shutil

import pytest

from payments_datagen.__main__ import main
from payments_datagen.generate import DATASET_PATHS, generate_world
from payments_datagen.manifest import MANIFEST_NAME, build_manifest, read_manifest
from payments_datagen.profiles import SAMPLE
from payments_datagen.storage import read_datasets, write_world

SMALL = [
    "--profile",
    "sample",
    "--days",
    "10",
    "--target-transactions",
    "700",
    "--n-accounts",
    "80",
    "--n-merchants",
    "20",
]


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("universe")
    assert main(["generate", *SMALL, "--out", str(out)]) == 0
    return out


def test_generate_writes_one_parquet_file_per_dataset_and_a_manifest(generated):
    for relative in DATASET_PATHS.values():
        assert (generated / relative).is_file()
    assert (generated / MANIFEST_NAME).is_file()


def test_validate_passes_on_a_generated_directory(generated, capsys):
    assert main(["validate", str(generated)]) == 0
    output = capsys.readouterr().out
    assert "RESULT: PASSED" in output
    assert "FAIL" not in output


def test_validate_fails_with_a_readable_report_when_a_file_is_missing(generated, tmp_path, capsys):
    broken = tmp_path / "broken"
    broken.mkdir()
    for path in generated.rglob("*"):
        target = broken / path.relative_to(generated)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.write_bytes(path.read_bytes())
    (broken / DATASET_PATHS["refunds"]).unlink()

    assert main(["validate", str(broken), "--skip-determinism"]) == 1
    output = capsys.readouterr().out
    assert "[FAIL] files" in output
    assert "refunds" in output
    assert "RESULT: FAILED" in output


def test_validate_fails_when_the_data_violates_a_contract(generated, tmp_path, capsys):
    broken = tmp_path / "bad_data"
    shutil.copytree(generated, broken)
    path = broken / DATASET_PATHS["payments"]
    frame = read_datasets(broken)["payments"]
    frame.with_columns(amount_minor=frame["amount_minor"] * 0).write_parquet(path)

    assert main(["validate", str(broken), "--skip-determinism"]) == 1
    assert "[FAIL] data contracts" in capsys.readouterr().out


def test_validate_fails_when_the_manifest_row_counts_do_not_match(generated, tmp_path, capsys):
    broken = tmp_path / "bad_manifest"
    shutil.copytree(generated, broken)
    manifest = read_manifest(broken / MANIFEST_NAME)
    manifest["row_counts"]["transactions"] += 1
    (broken / MANIFEST_NAME).write_text(json.dumps(manifest))

    assert main(["validate", str(broken), "--skip-determinism"]) == 1
    assert "[FAIL] manifest" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["generate", "--profile", "huge", "--out", "x"],
        ["generate", "--profile", "sample"],
        ["generate", "--profile", "sample", "--out", "x", "--start-date", "01/01/2026"],
        [],
    ],
)
def test_bad_arguments_exit_non_zero(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code != 0


def test_invalid_overrides_are_rejected():
    with pytest.raises(ValueError):
        main(["generate", "--profile", "sample", "--out", "x", "--fraud-rate", "0.9"])


def test_parquet_files_round_trip_without_losing_types(sample_world, tmp_path):
    write_world(sample_world, tmp_path)
    back = read_datasets(tmp_path)
    for name, frame in sample_world.frames.items():
        assert back[name].equals(frame), name
        assert back[name].schema == frame.schema, name


def test_same_inputs_give_identical_files(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    write_world(generate_world(SAMPLE), first)
    write_world(generate_world(SAMPLE), second)
    for relative in [*DATASET_PATHS.values(), MANIFEST_NAME]:
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), relative


def test_manifest_has_the_inputs_the_counts_and_no_timestamps(sample_world):
    manifest = build_manifest(sample_world)
    assert manifest["seed"] == 42
    assert manifest["profile"] == "sample"
    assert manifest["parameters"]["start_date"] == "2026-01-01"
    assert manifest["row_counts"]["transactions"] == sample_world["transactions"].height
    realized = manifest["realized"]
    assert realized["fraud_rate"] == pytest.approx(0.03, abs=0.005)
    assert set(realized["breaks_by_type"]) == {
        "AMOUNT_MISMATCH",
        "MISSING_IN_PSP",
        "MISSING_IN_LEDGER",
        "DUPLICATE_IN_PSP",
        "FEE_MISMATCH",
        "CHARGEBACK_NOT_IN_LEDGER",
    }
    text = json.dumps(manifest)
    for forbidden in ("generated_at", "created", "timestamp", "version_info"):
        assert forbidden not in text.replace("manifest_version", "")
