import json
import math
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from risk_scoring.serving.reasons import ReasonRules
from risk_scoring.training import artifacts
from risk_scoring.training.artifacts import ArtifactError, load_bundle, save_bundle, sha256_of
from risk_scoring.training.evaluate import FRAUD_PATTERNS, evaluate, flatten_metrics
from risk_scoring.training.features import FEATURE_GROUPS
from risk_scoring.training.thresholds import choose_thresholds
from risk_scoring.training.trainer import TrainConfig, predict_scores

TRAINED_AT = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def thresholds(prepared, trained):
    validation = prepared.splits["validation"]
    return choose_thresholds(
        validation.labels, predict_scores(trained.model, validation.encoded), 0.02, 0.002
    )


@pytest.fixture(scope="module")
def evaluation(prepared, trained, thresholds):
    return evaluate(prepared, trained.model, thresholds, permutation_repeats=2)


# --- evaluation -------------------------------------------------------------------------------


def test_every_split_reports_prevalence_as_the_reference_line(prepared, evaluation):
    for name, split in prepared.splits.items():
        reported = evaluation["splits"][name]
        assert reported["rows"] == split.rows
        assert reported["prevalence"] == pytest.approx(split.labels.mean())
        assert reported["pr_auc_lift"] == pytest.approx(reported["pr_auc"] / reported["prevalence"])


def test_the_model_learns_something_on_held_out_data(evaluation):
    test = evaluation["splits"]["test"]
    assert test["pr_auc_lift"] >= 3
    assert 0.5 < test["roc_auc"] <= 1
    assert evaluation["sanity"]["at_least_3x_prevalence"]
    assert not evaluation["sanity"]["suspiciously_perfect"]


def test_bands_partition_the_traffic(evaluation):
    for split in ("validation", "test"):
        bands = evaluation["bands"][split]
        assert [b["band"] for b in bands] == ["APPROVE", "REVIEW", "DECLINE"]
        assert sum(b["traffic_share"] for b in bands) == pytest.approx(1.0)
        assert sum(b["recall"] for b in bands if not math.isnan(b["recall"])) == pytest.approx(1.0)


def test_review_queue_is_a_small_share_of_traffic(evaluation):
    approve, review, decline = evaluation["bands"]["validation"]
    assert review["traffic_share"] + decline["traffic_share"] < 0.1


def test_patterns_are_reported_against_ground_truth(prepared, evaluation):
    test = prepared.splits["test"].frame
    for row in evaluation["patterns"]["test"]:
        assert row["pattern"] in FRAUD_PATTERNS
        expected = int((test["fraud_pattern"] == row["pattern"]).sum())
        assert row["rows"] == expected
        if expected:
            assert 0 <= row["recall_at_decline"] <= row["recall_at_review"] <= 1


def test_label_coverage_shows_fraud_without_a_label(evaluation):
    for name, coverage in evaluation["label_coverage"].items():
        assert (
            coverage["labeled_fraud_rows"] + coverage["unlabeled_fraud_rows"]
            == coverage["fraud_rows"]
        )
        assert 0 < coverage["coverage"] < 1, name
        assert coverage["labels_that_are_not_fraud"] == 0, "only fraud can be charged back"


def test_permutation_importance_covers_every_group_and_is_sorted(evaluation):
    rows = evaluation["permutation_importance"]
    assert {r["group"] for r in rows} == set(FEATURE_GROUPS)
    drops = [r["pr_auc_drop"] for r in rows]
    assert drops == sorted(drops, reverse=True)
    assert drops[0] > 0.01, "shuffling the most important group must hurt"


def test_flat_metrics_are_numbers_without_nan(evaluation):
    flat = flatten_metrics(evaluation)
    assert "test.pr_auc" in flat and "test.prevalence" in flat and "validation.roc_auc" in flat
    assert all(isinstance(v, float) and not math.isnan(v) for v in flat.values())


# --- artifacts --------------------------------------------------------------------------------


@pytest.fixture
def bundle_dir(tmp_path, prepared, trained, thresholds) -> Path:
    save_bundle(
        tmp_path,
        model=trained.model,
        preprocessor=prepared.preprocessor,
        thresholds=thresholds,
        reason_rules=ReasonRules(),
        train_config=TrainConfig(),
        model_version="mlp-test",
        trained_at=TRAINED_AT,
        flat_metrics={"test.pr_auc": 0.5},
    )
    return tmp_path


def test_round_trip_gives_identical_scores(bundle_dir, prepared, trained, thresholds):
    loaded = load_bundle(bundle_dir)
    encoded = prepared.splits["test"].encoded

    np.testing.assert_array_equal(
        predict_scores(loaded.model, encoded), predict_scores(trained.model, encoded)
    )
    assert loaded.preprocessor == prepared.preprocessor
    assert loaded.thresholds == thresholds
    assert loaded.reason_rules == ReasonRules()
    assert loaded.model_version == "mlp-test"


def test_metadata_records_hashes_and_the_score_semantics(bundle_dir):
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    for name in ("model.pt", "preprocessor.json"):
        assert metadata["files"][name]["sha256"] == sha256_of(bundle_dir / name)
    assert "NOT a calibrated probability" in metadata["score_semantics"]
    assert metadata["trained_at"] == TRAINED_AT.isoformat()


@pytest.mark.parametrize("name", ["model.pt", "preprocessor.json"])
def test_a_tampered_file_is_detected(bundle_dir, name):
    path = bundle_dir / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ArtifactError, match=f"{name} does not match"):
        load_bundle(bundle_dir)


def test_a_missing_file_is_reported(bundle_dir):
    (bundle_dir / "model.pt").unlink()
    with pytest.raises(ArtifactError, match="missing artifact file model.pt"):
        load_bundle(bundle_dir)


def test_missing_metadata_is_reported(tmp_path):
    with pytest.raises(ArtifactError, match="metadata.json"):
        load_bundle(tmp_path)


def rehash(bundle_dir: Path) -> None:
    """Make the metadata agree with the files again (what an attacker with write access does)."""
    path = bundle_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    for name in artifacts.HASHED_FILES:
        metadata["files"][name]["sha256"] = sha256_of(bundle_dir / name)
    path.write_text(json.dumps(metadata))


def test_loader_rejects_an_artifact_trained_on_other_feature_columns(bundle_dir):
    path = bundle_dir / "preprocessor.json"
    data = json.loads(path.read_text())
    data["input_columns"] = [c for c in data["input_columns"] if c != "mcc"]
    path.write_text(json.dumps(data))
    rehash(bundle_dir)

    with pytest.raises(ArtifactError, match="different feature columns"):
        load_bundle(bundle_dir)


@pytest.mark.filterwarnings("ignore:Detected pickle protocol")
def test_weights_are_loaded_with_weights_only(bundle_dir):
    """Even with matching hashes, a pickle that is not plain tensors must not be executed."""

    class NotATensor:
        def __reduce__(self):
            return (print, ("this must never run",))

    (bundle_dir / "model.pt").write_bytes(pickle.dumps(NotATensor()))
    rehash(bundle_dir)

    with pytest.raises(Exception, match="(?i)weights|pickle|zip|PytorchStreamReader"):
        load_bundle(bundle_dir)


def test_unsupported_artifact_version_is_rejected(bundle_dir):
    path = bundle_dir / "metadata.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "schema_version": 99}))
    with pytest.raises(ArtifactError, match="schema"):
        load_bundle(bundle_dir)


def test_trained_at_needs_a_timezone(tmp_path, prepared, trained, thresholds):
    with pytest.raises(ValueError, match="timezone"):
        save_bundle(
            tmp_path,
            model=trained.model,
            preprocessor=prepared.preprocessor,
            thresholds=thresholds,
            reason_rules=ReasonRules(),
            train_config=TrainConfig(),
            model_version="x",
            trained_at=datetime(2026, 1, 1),
            flat_metrics={},
        )


def test_reason_rules_round_trip():
    rules = ReasonRules(high_risk_mccs=("1111",), max_reasons=3)
    assert ReasonRules.from_dict(rules.to_dict()) == rules
