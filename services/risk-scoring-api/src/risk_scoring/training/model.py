"""The network: embeddings for the categorical columns, concatenated with the dense block, then
a small MLP that outputs one logit.

    [embeddings (mcc 8, card_bin 8, channel 3, issuer_country 4, ip_country 4)] + [dense block]
        -> Linear(in, 64) -> ReLU -> Dropout(0.1) -> Linear(64, 32) -> ReLU -> Linear(32, 1)

For a Java reader: ``nn.Module`` is a class whose ``forward`` is the "method" that is called when
you apply the object like a function; the layers declared in ``__init__`` are its parameters.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Self

import torch
from torch import nn

from risk_scoring.training.features import CATEGORICAL_COLUMNS


def _default_embedding_dims() -> dict[str, int]:
    return {"mcc": 8, "card_bin": 8, "channel": 3, "issuer_country": 4, "ip_country": 4}


@dataclass(frozen=True)
class ModelConfig:
    embedding_dims: dict[str, int] = field(default_factory=_default_embedding_dims)
    hidden_1: int = 64
    hidden_2: int = 32
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if set(self.embedding_dims) != set(CATEGORICAL_COLUMNS):
            raise ValueError(f"embedding_dims must cover exactly {CATEGORICAL_COLUMNS}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)


class FraudMLP(nn.Module):
    def __init__(self, n_dense: int, vocab_sizes: dict[str, int], config: ModelConfig) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(vocab_sizes[column], config.embedding_dims[column])
            for column in CATEGORICAL_COLUMNS
        )
        n_inputs = n_dense + sum(config.embedding_dims[c] for c in CATEGORICAL_COLUMNS)
        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, config.hidden_1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.ReLU(),
            nn.Linear(config.hidden_2, 1),
        )

    def forward(self, dense: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """``dense``: (n, d) floats; ``categorical``: (n, c) indices. Returns logits, shape (n,)."""
        embedded = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.mlp(torch.cat([dense, *embedded], dim=1)).squeeze(1)
