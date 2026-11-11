"""The scorer: request in, score + decision + reason codes out.

This is the online half of the model. It owns the loaded artifact and applies *the same* feature
pipeline as training (``Preprocessor.prepare``): there is no second implementation of the features.

Scoring is a pure, deterministic function of the request and the artifact, which is what lets the
decision log be idempotent (two racing identical requests compute the same answer, ADR 0009).
It is blocking CPU work, so callers run it in a thread (FastAPI does that for plain ``def`` routes).
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from risk_scoring.api.schemas import Decision, ReasonCode, ScoreRequest
from risk_scoring.config import Settings
from risk_scoring.dataset_mapping import requests_to_frame
from risk_scoring.serving.reasons import ReasonRules, reason_codes
from risk_scoring.training.artifacts import Bundle, load_bundle
from risk_scoring.training.thresholds import BandThresholds, assign_bands
from risk_scoring.training.trainer import predict_scores


@dataclass(frozen=True)
class Scored:
    score: float
    decision: Decision
    reason_codes: list[ReasonCode]


@dataclass(frozen=True)
class Timed:
    """Results plus the time spent. For a batch, the time of one item is the batch time over n."""

    results: list[Scored]
    latency_ms_per_item: float


class Scorer:
    def __init__(self, bundle: Bundle, thresholds: BandThresholds | None = None) -> None:
        self.bundle = bundle
        self.thresholds = thresholds or bundle.thresholds

    # --- construction ---------------------------------------------------------------------------

    @classmethod
    def load(cls, model_dir: Path, settings: Settings | None = None) -> "Scorer":
        """Load and verify the artifact; apply threshold overrides and the torch thread setting."""
        settings = settings or Settings()
        bundle = load_bundle(model_dir)
        torch.set_num_threads(settings.torch_threads)  # process-wide, by design (see ADR 0009)
        return cls(bundle, thresholds_with_overrides(bundle.thresholds, settings))

    # --- what the artifact says about itself ----------------------------------------------------

    @property
    def model_version(self) -> str:
        return self.bundle.model_version

    @property
    def trained_at(self) -> datetime:
        return datetime.fromisoformat(self.bundle.metadata["trained_at"])

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.bundle.metadata["metrics"])

    @property
    def reason_rules(self) -> ReasonRules:
        return self.bundle.reason_rules

    # --- scoring --------------------------------------------------------------------------------

    def score_many(self, requests: Sequence[ScoreRequest]) -> list[Scored]:
        """One batched forward pass for all requests; results in request order."""
        if not requests:
            return []
        encoded = self.bundle.preprocessor.prepare(requests_to_frame(requests))
        scores = predict_scores(self.bundle.model, encoded)
        bands = assign_bands(scores, self.thresholds)
        return [
            Scored(
                score=float(score),
                decision=Decision(str(band)),
                reason_codes=reason_codes(
                    request, self.reason_rules, flagged=band != Decision.APPROVE
                ),
            )
            for request, score, band in zip(requests, scores, bands, strict=True)
        ]

    def score_one(self, request: ScoreRequest) -> Scored:
        return self.score_many([request])[0]

    def timed(self, requests: Sequence[ScoreRequest]) -> Timed:
        """Score and measure feature computation plus inference (not validation, not the DB)."""
        started = time.perf_counter()
        results = self.score_many(requests)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return Timed(results, elapsed_ms / max(len(requests), 1))

    def info(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "thresholds": {"review": self.thresholds.review, "decline": self.thresholds.decline},
            "metrics": self.metrics,
        }


def thresholds_with_overrides(base: BandThresholds, settings: Settings) -> BandThresholds:
    """The artifact's thresholds, with ``RISK_THRESHOLD_REVIEW/DECLINE`` applied when set."""
    review = base.review if settings.threshold_review is None else settings.threshold_review
    decline = base.decline if settings.threshold_decline is None else settings.threshold_decline
    if review == base.review and decline == base.decline:
        return base
    return BandThresholds(review, decline, base.review_fpr, base.decline_fpr)
