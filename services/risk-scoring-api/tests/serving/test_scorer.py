import numpy as np
import pytest

from risk_scoring.api.schemas import Decision, ReasonCode
from risk_scoring.config import Settings
from risk_scoring.dataset_mapping import row_to_request
from risk_scoring.serving.scorer import Scorer, thresholds_with_overrides
from risk_scoring.training.artifacts import ArtifactError
from risk_scoring.training.thresholds import BandThresholds


@pytest.fixture(scope="module")
def requests(sample_transactions):
    rows = sample_transactions.gather_every(15).head(200)
    return [row_to_request(r) for r in rows.iter_rows(named=True)], rows


def test_scores_are_in_unit_interval_and_decisions_are_bands(scorer, requests):
    reqs, _ = requests
    results = scorer.score_many(reqs)

    assert len(results) == len(reqs)
    t = scorer.thresholds
    for r in results:
        assert 0.0 <= r.score <= 1.0
        expected = (
            Decision.DECLINE
            if r.score >= t.decline
            else Decision.REVIEW
            if r.score >= t.review
            else Decision.APPROVE
        )
        assert r.decision == expected


def test_batch_equals_single(scorer, requests):
    reqs, _ = requests
    batch = scorer.score_many(reqs)
    for request, expected in zip(reqs[:40], batch[:40], strict=True):
        single = scorer.score_one(request)
        assert single.score == pytest.approx(expected.score, abs=1e-6)
        assert single.decision == expected.decision
        assert single.reason_codes == expected.reason_codes


def test_scoring_is_deterministic(scorer, requests):
    reqs, _ = requests
    assert scorer.score_many(reqs[:20]) == scorer.score_many(reqs[:20])


def test_approved_transactions_have_no_reasons(scorer, requests):
    reqs, _ = requests
    for r in scorer.score_many(reqs):
        if r.decision == Decision.APPROVE:
            assert r.reason_codes == []
        else:
            assert len(r.reason_codes) <= 5


def test_fraud_scores_clearly_above_legitimate(scorer, sample_transactions):
    rows = sample_transactions.head(1_500)
    scores = np.array(
        [
            s.score
            for s in scorer.score_many([row_to_request(r) for r in rows.iter_rows(named=True)])
        ]
    )
    fraud = rows["is_fraud"].to_numpy()

    assert scores[fraud].mean() > 3 * scores[~fraud].mean()


def test_empty_batch_is_empty(scorer):
    assert scorer.score_many([]) == []


def test_timed_reports_a_per_item_latency(scorer, requests):
    reqs, _ = requests
    timed = scorer.timed(reqs[:50])
    assert len(timed.results) == 50
    assert 0 < timed.latency_ms_per_item < 1_000


def test_lowering_the_thresholds_moves_decisions_between_bands(scorer, requests):
    reqs, _ = requests
    base = [s.decision for s in scorer.score_many(reqs)]

    strict = Scorer(scorer.bundle, BandThresholds(review=0.0, decline=0.0))
    assert {s.decision for s in strict.score_many(reqs)} == {Decision.DECLINE}

    lax = Scorer(scorer.bundle, BandThresholds(review=1.0, decline=1.0))
    assert all(s.decision != Decision.REVIEW for s in lax.score_many(reqs))
    assert base != [s.decision for s in strict.score_many(reqs)]


def test_threshold_overrides_from_settings(scorer):
    base = scorer.bundle.thresholds
    assert thresholds_with_overrides(base, Settings()) is base

    only_review = thresholds_with_overrides(base, Settings(threshold_review=0.01))
    assert (only_review.review, only_review.decline) == (0.01, base.decline)

    both = thresholds_with_overrides(base, Settings(threshold_review=0.2, threshold_decline=0.4))
    assert (both.review, both.decline) == (0.2, 0.4)


def test_inconsistent_override_is_rejected(scorer):
    with pytest.raises(ValueError, match="cannot exceed"):
        thresholds_with_overrides(
            scorer.bundle.thresholds, Settings(threshold_review=0.9, threshold_decline=0.1)
        )


def test_load_applies_the_override_and_exposes_the_artifact(model_dir):
    scorer = Scorer.load(model_dir, Settings(threshold_review=0.3, threshold_decline=0.6))

    assert (scorer.thresholds.review, scorer.thresholds.decline) == (0.3, 0.6)
    assert scorer.model_version == "mlp-test-0.1"
    assert scorer.trained_at.isoformat() == "2026-02-01T03:00:00+00:00"
    assert scorer.metrics["test.pr_auc"] == 0.5
    assert scorer.info()["thresholds"] == {"review": 0.3, "decline": 0.6}


def test_load_fails_for_a_missing_model(tmp_path):
    with pytest.raises(ArtifactError):
        Scorer.load(tmp_path)


def test_every_reason_code_is_a_contract_code(scorer, requests):
    reqs, _ = requests
    seen = {c for s in scorer.score_many(reqs) for c in s.reason_codes}
    assert seen <= set(ReasonCode)
