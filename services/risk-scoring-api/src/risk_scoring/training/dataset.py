"""From a ``transactions`` frame to model-ready splits: the glue between the training modules.

The order matters and is the whole leakage story:

1. split by time;
2. label the training split with a mature ``as_of`` (guarded), the other two with eventual labels;
3. derive features and **fit the preprocessor on the training split only**;
4. encode all three splits with that preprocessor.

``is_fraud`` and ``fraud_pattern`` stay in ``SplitData.frame`` for the evaluation report, but only
``encoded`` (built from the contract's feature columns) is ever shown to the model.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from risk_scoring.dataset_mapping import feature_frame
from risk_scoring.training.features import derive_features
from risk_scoring.training.labels import (
    DEFAULT_LABEL_MATURITY_DAYS,
    derive_label,
    training_as_of,
    training_label,
)
from risk_scoring.training.preprocessing import Encoded, Preprocessor
from risk_scoring.training.splits import DEFAULT_FRACTIONS, SplitPlan, plan_split


@dataclass(frozen=True)
class SplitData:
    name: str
    frame: pl.DataFrame  # full transactions rows, including analysis-only columns (reports only)
    encoded: Encoded
    labels: np.ndarray  # bool
    amounts: np.ndarray  # amount_minor, int64

    @property
    def rows(self) -> int:
        return self.frame.height


@dataclass(frozen=True)
class PreparedData:
    plan: SplitPlan
    label_as_of: datetime
    label_maturity_days: int
    preprocessor: Preprocessor
    splits: dict[str, SplitData]


def prepare_data(
    transactions: pl.DataFrame,
    *,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    label_maturity_days: int = DEFAULT_LABEL_MATURITY_DAYS,
    min_category_count: int = 20,
    label_as_of: datetime | None = None,
    preprocessor: Preprocessor | None = None,
) -> PreparedData:
    """Split, label, fit the preprocessor on train and encode everything.

    ``label_as_of`` exists to make the maturity guard testable; leave it unset to use the correct
    ``train_end + label_maturity_days``. Pass ``preprocessor`` to evaluate an existing model: it is
    then used as it is instead of being fitted again.
    """
    plan = plan_split(transactions, fractions)
    as_of = label_as_of or training_as_of(plan.train_end, label_maturity_days)
    frames = plan.apply(transactions)

    if preprocessor is None:
        train_derived = derive_features(feature_frame(frames["train"]))
        preprocessor = Preprocessor.fit(train_derived, min_category_count=min_category_count)

    splits: dict[str, SplitData] = {}
    for name, frame in frames.items():
        labels = (
            training_label(frame, plan.train_end, as_of, label_maturity_days)
            if name == "train"
            else derive_label(frame)  # eventual labels: an after-the-fact offline evaluation
        )
        splits[name] = SplitData(
            name=name,
            frame=frame,
            encoded=preprocessor.prepare(feature_frame(frame)),
            labels=labels.to_numpy(),
            amounts=frame["amount_minor"].to_numpy(),
        )
    return PreparedData(plan, as_of, label_maturity_days, preprocessor, splits)
