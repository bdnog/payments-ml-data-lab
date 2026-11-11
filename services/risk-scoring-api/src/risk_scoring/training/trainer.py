"""The training loop: weighted BCE, AdamW, early stopping on validation PR-AUC, fixed seeds.

Reproducibility is on purpose and has limits: with the same seed, data and hyperparameters the
weights are identical *on the same platform* (CPU, same torch build and thread count). Across
platforms, floating-point summation order may differ in the last bits.
"""

import copy
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from risk_scoring.training.metrics import average_precision
from risk_scoring.training.model import FraudMLP, ModelConfig
from risk_scoring.training.preprocessing import Encoded


@dataclass(frozen=True)
class TrainConfig:
    """All hyperparameters. Stored in the model artifact next to the weights."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 30
    patience: int = 5
    pos_weight_cap: float = 50.0
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        return cls(**{**data, "model": ModelConfig.from_dict(data["model"])})


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation_pr_auc: float


@dataclass(frozen=True)
class TrainResult:
    model: FraudMLP
    history: list[EpochRecord]
    best_epoch: int
    pos_weight: float
    stopped_early: bool


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, and make torch refuse non-deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def to_tensors(encoded: Encoded) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(encoded.dense), torch.from_numpy(encoded.categorical)


def predict_scores(model: FraudMLP, encoded: Encoded, batch_size: int = 65_536) -> np.ndarray:
    """Ranking scores in [0, 1] (sigmoid of the logit). Not calibrated probabilities."""
    dense, categorical = to_tensors(encoded)
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            logits = model(
                dense[start : start + batch_size], categorical[start : start + batch_size]
            )
            parts.append(torch.sigmoid(logits).numpy())
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def train_model(
    train: Encoded,
    y_train: np.ndarray,
    validation: Encoded,
    y_validation: np.ndarray,
    vocab_sizes: dict[str, int],
    config: TrainConfig | None = None,
    on_epoch: Callable[[EpochRecord], None] | None = None,
) -> TrainResult:
    config = config or TrainConfig()
    seed_everything(config.seed)

    n_pos = int(np.sum(y_train))
    n_neg = len(y_train) - n_pos
    if n_pos == 0 or int(np.sum(y_validation)) == 0:
        raise ValueError("training and validation sets both need at least one positive label")
    # Fraud is rare: without a weight the loss is dominated by legitimate rows. The cap keeps a
    # very rare label from exploding the gradients. It distorts calibration: scores only rank.
    pos_weight = min(n_neg / n_pos, config.pos_weight_cap)

    model = FraudMLP(train.dense.shape[1], vocab_sizes, config.model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))

    dense, categorical = to_tensors(train)
    target = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    shuffler = torch.Generator().manual_seed(config.seed)  # manual shuffling, seeded explicitly

    history: list[EpochRecord] = []
    best_pr_auc, best_epoch, best_state = -1.0, 0, copy.deepcopy(model.state_dict())

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(len(target), generator=shuffler)
        total_loss = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(dense[batch], categorical[batch]), target[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        pr_auc = average_precision(y_validation, predict_scores(model, validation))
        record = EpochRecord(epoch, total_loss / len(target), pr_auc)
        history.append(record)
        if on_epoch:
            on_epoch(record)

        if pr_auc > best_pr_auc:
            best_pr_auc, best_epoch = pr_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        pos_weight=pos_weight,
        stopped_early=len(history) < config.max_epochs,
    )
