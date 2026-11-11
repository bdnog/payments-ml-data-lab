"""Label derivation: what "fraud" means for training, and when that is knowable.

The ground truth (``is_fraud``) is *not* available in production: what the business learns is that
a cardholder disputed a payment, some days later (a chargeback). So the training label is

    label = chargeback_reported_at IS NOT NULL AND chargeback_reported_at <= as_of

where ``as_of`` is the date at which the labels are read. A transaction that is fraud but whose
chargeback has not arrived yet at ``as_of`` is labeled 0: a *false negative in the labels*. The
longer after the transaction we wait, the fewer of those there are. In this dataset chargebacks
arrive at most 45 days after the transaction (ADR 0008), so training labels are only trustworthy
for transactions at least 45 days older than ``as_of``. ``training_label`` enforces that.
"""

from datetime import datetime, timedelta

import polars as pl

DEFAULT_LABEL_MATURITY_DAYS = 45
LABEL_COLUMN = "label"


class ImmatureLabelsError(ValueError):
    """The labels are read too soon after the end of the training period to be trusted."""


def derive_label(transactions: pl.DataFrame, as_of: datetime | None = None) -> pl.Series:
    """Boolean label per row. ``as_of=None`` means *eventual*: any chargeback ever reported.

    Eventual labels are what an offline evaluation done after the fact would use. They are not
    available at training time in production, which is why training uses ``training_label``.
    """
    reported = pl.col("chargeback_reported_at")
    condition = reported.is_not_null()
    if as_of is not None:
        condition = condition & (reported <= pl.lit(as_of, dtype=pl.Datetime("us", "UTC")))
    return transactions.select(condition.alias(LABEL_COLUMN))[LABEL_COLUMN]


def training_as_of(train_end: datetime, label_maturity_days: int) -> datetime:
    return train_end + timedelta(days=label_maturity_days)


def check_label_maturity(
    train_end: datetime, as_of: datetime, label_maturity_days: int = DEFAULT_LABEL_MATURITY_DAYS
) -> None:
    """Refuse a training set whose labels were read less than ``label_maturity_days`` after it."""
    waited = as_of - train_end
    if waited < timedelta(days=label_maturity_days):
        raise ImmatureLabelsError(
            f"labels read {waited.days} days after the end of the training period, but "
            f"chargebacks can arrive up to {label_maturity_days} days after a transaction: "
            "recent training transactions would be labeled legitimate by mistake"
        )


def training_label(
    transactions: pl.DataFrame,
    train_end: datetime,
    as_of: datetime,
    label_maturity_days: int = DEFAULT_LABEL_MATURITY_DAYS,
) -> pl.Series:
    """The label of a training set: guarded, so an immature ``as_of`` cannot be used by mistake."""
    check_label_maturity(train_end, as_of, label_maturity_days)
    return derive_label(transactions, as_of)
