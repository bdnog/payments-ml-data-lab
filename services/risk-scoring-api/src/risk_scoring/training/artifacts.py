"""The model artifact bundle: everything serving needs, in four small files.

    model.pt            torch ``state_dict`` (weights only, loaded with ``weights_only=True``)
    preprocessor.json   vocabularies and scaling statistics
    metadata.json       version, hyperparameters, thresholds, reason rules, metrics, sha256 of the
                        two files above
    report.md           human-readable evaluation report (not loaded by the service)

``weights_only=True`` makes ``torch.load`` refuse to unpickle arbitrary objects, and the loader
checks the sha256 recorded in ``metadata.json`` before touching a file: a tampered or swapped file
fails loudly instead of being executed or served. (A small supply-chain control, not a signature:
whoever can edit ``metadata.json`` can edit the hashes too.)
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from risk_scoring.serving.reasons import ReasonRules
from risk_scoring.training.model import FraudMLP
from risk_scoring.training.preprocessing import Preprocessor
from risk_scoring.training.thresholds import BandThresholds
from risk_scoring.training.trainer import TrainConfig

ARTIFACT_SCHEMA_VERSION = 1
MODEL_FILE = "model.pt"
PREPROCESSOR_FILE = "preprocessor.json"
METADATA_FILE = "metadata.json"
REPORT_FILE = "report.md"
METRICS_FILE = "metrics.json"
HASHED_FILES = (MODEL_FILE, PREPROCESSOR_FILE)
SCORE_SEMANTICS = (
    "ranking score in [0, 1] (sigmoid of the logit); NOT a calibrated probability, because the "
    "loss uses pos_weight"
)


class ArtifactError(Exception):
    """The artifact is missing, tampered with, or incompatible with this code."""


@dataclass(frozen=True)
class Bundle:
    model: FraudMLP
    preprocessor: Preprocessor
    thresholds: BandThresholds
    reason_rules: ReasonRules
    train_config: TrainConfig
    metadata: dict[str, Any]

    @property
    def model_version(self) -> str:
        return self.metadata["model_version"]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_bundle(
    directory: Path,
    *,
    model: FraudMLP,
    preprocessor: Preprocessor,
    thresholds: BandThresholds,
    reason_rules: ReasonRules,
    train_config: TrainConfig,
    model_version: str,
    trained_at: datetime,
    flat_metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``model.pt``, ``preprocessor.json`` and ``metadata.json``; return the metadata."""
    if trained_at.tzinfo is None:
        raise ValueError("trained_at must carry a timezone")
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / MODEL_FILE)
    (directory / PREPROCESSOR_FILE).write_text(preprocessor.to_json(), encoding="utf-8")

    metadata: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": model_version,
        "trained_at": trained_at.isoformat(),
        "torch_version": torch.__version__,
        "score_semantics": SCORE_SEMANTICS,
        "files": {name: {"sha256": sha256_of(directory / name)} for name in HASHED_FILES},
        "n_dense": len(preprocessor.dense_columns),
        "vocab_sizes": preprocessor.vocab_sizes,
        "train_config": train_config.to_dict(),
        "thresholds": thresholds.to_dict(),
        "reason_rules": reason_rules.to_dict(),
        "metrics": flat_metrics,
        **(extra or {}),
    }
    (directory / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_metadata(directory: Path) -> dict[str, Any]:
    path = directory / METADATA_FILE
    if not path.is_file():
        raise ArtifactError(f"no {METADATA_FILE} in {directory}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(f"unsupported artifact schema {metadata.get('schema_version')}")
    return metadata


def verify_hashes(directory: Path, metadata: dict[str, Any]) -> None:
    for name in HASHED_FILES:
        path = directory / name
        if not path.is_file():
            raise ArtifactError(f"missing artifact file {name}")
        expected = metadata["files"][name]["sha256"]
        if sha256_of(path) != expected:
            raise ArtifactError(f"{name} does not match the sha256 recorded in the metadata")


def load_bundle(directory: Path) -> Bundle:
    """Load and verify an artifact: hashes first, then feature columns, then the weights."""
    directory = Path(directory)
    metadata = load_metadata(directory)
    verify_hashes(directory, metadata)

    preprocessor = Preprocessor.from_json((directory / PREPROCESSOR_FILE).read_text("utf-8"))
    try:
        preprocessor.check_against_contract()
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc

    train_config = TrainConfig.from_dict(metadata["train_config"])
    model = FraudMLP(metadata["n_dense"], preprocessor.vocab_sizes, train_config.model)
    state = torch.load(directory / MODEL_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return Bundle(
        model=model,
        preprocessor=preprocessor,
        thresholds=BandThresholds.from_dict(metadata["thresholds"]),
        reason_rules=ReasonRules.from_dict(metadata["reason_rules"]),
        train_config=train_config,
        metadata=metadata,
    )
