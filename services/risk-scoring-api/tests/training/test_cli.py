import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from risk_scoring.training.__main__ import main
from risk_scoring.training.artifacts import load_bundle
from risk_scoring.training.pipeline import read_transactions

STAMP = "2026-11-01T12:00:00+00:00"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory, training_transactions) -> Path:
    directory = tmp_path_factory.mktemp("data")
    (directory / "risk").mkdir()
    training_transactions.write_parquet(directory / "risk" / "transactions.parquet")
    (directory / "manifest.json").write_text(
        json.dumps({"profile": "risk-test", "parameters": {"seed": 7}, "realized": {"n": 1}})
    )
    return directory


def train(data_dir: Path, out: Path):
    return main(
        [
            "train",
            "--data",
            str(data_dir),
            "--out",
            str(out),
            "--model-version",
            "mlp-test-1",
            "--max-epochs",
            "3",
            "--trained-at",
            STAMP,
        ]
    )


def test_train_writes_a_loadable_bundle_and_a_report(data_dir, tmp_path, capsys):
    assert train(data_dir, tmp_path / "run") == 0

    files = {p.name for p in (tmp_path / "run").iterdir()}
    assert files == {"model.pt", "preprocessor.json", "metadata.json", "report.md", "metrics.json"}
    bundle = load_bundle(tmp_path / "run")
    assert bundle.model_version == "mlp-test-1"
    assert bundle.metadata["trained_at"] == datetime.fromisoformat(STAMP).isoformat()
    assert bundle.metadata["data_manifest"]["parameters"] == {"seed": 7}
    assert bundle.metadata["label_as_of"] > bundle.metadata["split"]["train_end"]
    assert "test.pr_auc" in bundle.metadata["metrics"]
    assert "epoch  1" in capsys.readouterr().out


def test_report_contains_the_required_sections_and_the_disclaimer(data_dir, tmp_path):
    train(data_dir, tmp_path / "run")
    report = (tmp_path / "run" / "report.md").read_text()

    for heading in (
        "The data is synthetic",
        "## Splits and prevalence",
        "## Ranking metrics",
        "## Decision bands",
        "## Recall by fraud pattern",
        "## Label coverage",
        "## Permutation importance",
        "## Score deciles (test)",
        "## Training curve",
        "## Sanity guards",
        "not a calibrated probability",
        "recall @ FPR 0.1%",
        "amount recall",
    ):
        assert heading in report, heading
    assert "risk-test" in report  # the data manifest is embedded


def test_metrics_json_is_valid_json(data_dir, tmp_path):
    train(data_dir, tmp_path / "run")
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert set(metrics) >= {"splits", "bands", "patterns", "label_coverage", "sanity"}


def test_two_runs_with_the_same_seed_write_the_same_files(data_dir, tmp_path):
    train(data_dir, tmp_path / "a")
    train(data_dir, tmp_path / "b")

    for name in ("model.pt", "preprocessor.json", "metadata.json", "report.md", "metrics.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_evaluate_reuses_the_artifacts_preprocessor_and_thresholds(data_dir, tmp_path, capsys):
    train(data_dir, tmp_path / "run")
    capsys.readouterr()

    assert main(["evaluate", "--model-dir", str(tmp_path / "run"), "--data", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "validation" in out and "test" in out and "sanity" in out


def test_missing_data_explains_how_to_generate_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="payments_datagen generate"):
        read_transactions(tmp_path)


def test_naive_trained_at_is_read_as_utc(data_dir, tmp_path):
    main(
        [
            "train", "--data", str(data_dir), "--out", str(tmp_path / "run"),
            "--model-version", "v", "--max-epochs", "1", "--trained-at", "2026-11-01T12:00:00",
        ]
    )  # fmt: skip
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert metadata["trained_at"] == datetime(2026, 11, 1, 12, tzinfo=UTC).isoformat()
