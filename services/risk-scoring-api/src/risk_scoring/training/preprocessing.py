"""Fit and apply the preprocessing that sits between the derived features and the network.

``Preprocessor.fit`` learns, from the **training split only**: the vocabulary of each categorical
column (categories seen fewer than ``min_category_count`` times become ``__unknown__``) and the mean
and standard deviation of each continuous column. ``transform`` is then a pure function of the
derived frame. The whole object round-trips through ``preprocessor.json`` and is stored in the
model artifact, so serving applies exactly the statistics that training used.
"""

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import polars as pl

from risk_scoring.training.features import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    derive_features,
    required_columns,
)

SCHEMA_VERSION = 1
UNKNOWN = "__unknown__"
MISSING = "__missing__"
CLIP = 5.0
MIN_STD = 1e-8
# Only ip_country can be null (in-person purchases): it gets its own index instead of "unknown".
NULLABLE_CATEGORICALS = frozenset({"ip_country"})


@dataclass(frozen=True)
class Encoded:
    """What the network consumes: a dense block and one integer index per categorical column."""

    dense: np.ndarray  # float32, shape (n, len(NUMERIC_COLUMNS) + len(BINARY_COLUMNS))
    categorical: np.ndarray  # int64, shape (n, len(CATEGORICAL_COLUMNS))

    def __len__(self) -> int:
        return self.dense.shape[0]

    def take(self, index: np.ndarray) -> "Encoded":
        return Encoded(self.dense[index], self.categorical[index])


@dataclass(frozen=True)
class Preprocessor:
    input_columns: tuple[str, ...]
    vocabularies: dict[str, tuple[str, ...]]  # position in the tuple == embedding index
    means: dict[str, float]
    stds: dict[str, float]
    clip: float = CLIP
    min_category_count: int = 20
    schema_version: int = SCHEMA_VERSION

    # --- fitting --------------------------------------------------------------------------------

    @classmethod
    def fit(cls, derived: pl.DataFrame, *, min_category_count: int = 20) -> Self:
        """Learn vocabularies and scaling statistics from the (training) derived frame."""
        if derived.height == 0:
            raise ValueError("cannot fit a preprocessor on an empty frame")
        vocabularies = {
            column: _build_vocabulary(column, derived[column], min_category_count)
            for column in CATEGORICAL_COLUMNS
        }
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for column in NUMERIC_COLUMNS:
            values = derived[column]
            means[column] = float(values.mean())  # type: ignore[arg-type]
            std = float(values.std(ddof=0))  # type: ignore[arg-type]
            stds[column] = std if std > MIN_STD else 1.0
        return cls(
            input_columns=required_columns(),
            vocabularies=vocabularies,
            means=means,
            stds=stds,
            min_category_count=min_category_count,
        )

    # --- applying -------------------------------------------------------------------------------

    @property
    def dense_columns(self) -> tuple[str, ...]:
        return (*NUMERIC_COLUMNS, *BINARY_COLUMNS)

    @property
    def vocab_sizes(self) -> dict[str, int]:
        return {column: len(self.vocabularies[column]) for column in CATEGORICAL_COLUMNS}

    def transform(self, derived: pl.DataFrame) -> Encoded:
        """Standardize + clip the continuous columns, pass flags through, index the categories."""
        numeric = np.column_stack(
            [
                np.clip(
                    (derived[c].to_numpy() - self.means[c]) / self.stds[c], -self.clip, self.clip
                )
                for c in NUMERIC_COLUMNS
            ]
        )
        binary = np.column_stack([derived[c].to_numpy() for c in BINARY_COLUMNS])
        dense = np.concatenate([numeric, binary], axis=1).astype(np.float32)
        categorical = np.column_stack(
            [self._index(column, derived[column]) for column in CATEGORICAL_COLUMNS]
        ).astype(np.int64)
        return Encoded(dense, categorical)

    def prepare(self, raw: pl.DataFrame) -> Encoded:
        """Raw feature columns to network input: the one path used by training and serving."""
        return self.transform(derive_features(raw))

    def _index(self, column: str, values: pl.Series) -> np.ndarray:
        vocabulary = self.vocabularies[column]
        mapping = {category: i for i, category in enumerate(vocabulary)}
        # Nulls become "__missing__"; for columns without that entry it is simply unknown.
        index = values.fill_null(MISSING).replace_strict(
            mapping, default=mapping[UNKNOWN], return_dtype=pl.Int64
        )
        return index.to_numpy()

    # --- serialization --------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_columns": list(self.input_columns),
            "numeric_columns": list(NUMERIC_COLUMNS),
            "binary_columns": list(BINARY_COLUMNS),
            "categorical_columns": list(CATEGORICAL_COLUMNS),
            "min_category_count": self.min_category_count,
            "clip": self.clip,
            "vocabularies": {k: list(v) for k, v in self.vocabularies.items()},
            "means": self.means,
            "stds": self.stds,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported preprocessor schema {data['schema_version']}")
        for key, expected in (
            ("numeric_columns", NUMERIC_COLUMNS),
            ("binary_columns", BINARY_COLUMNS),
            ("categorical_columns", CATEGORICAL_COLUMNS),
        ):
            if tuple(data[key]) != expected:
                raise ValueError(f"preprocessor {key} differ from the code: {data[key]}")
        return cls(
            input_columns=tuple(data["input_columns"]),
            vocabularies={k: tuple(v) for k, v in data["vocabularies"].items()},
            means={k: float(v) for k, v in data["means"].items()},
            stds={k: float(v) for k, v in data["stds"].items()},
            clip=float(data["clip"]),
            min_category_count=int(data["min_category_count"]),
        )

    @classmethod
    def from_json(cls, text: str) -> Self:
        return cls.from_dict(json.loads(text))

    def check_against_contract(self) -> None:
        """Fail fast when the artifact was trained on other columns than the current contract."""
        if self.input_columns != required_columns():
            raise ValueError(
                "the preprocessor was fitted on different feature columns than the current "
                f"data contract: {list(self.input_columns)} != {list(required_columns())}"
            )


def _build_vocabulary(column: str, values: pl.Series, min_count: int) -> tuple[str, ...]:
    counts = Counter(v for v in values.to_list() if v is not None)
    kept = sorted(category for category, n in counts.items() if n >= min_count)
    head = [UNKNOWN, MISSING] if column in NULLABLE_CATEGORICALS else [UNKNOWN]
    return (*head, *kept)
