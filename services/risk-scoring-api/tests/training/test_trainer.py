from datetime import timedelta

import numpy as np
import polars as pl
import pytest
import torch

from risk_scoring.dataset_mapping import feature_frame
from risk_scoring.training.dataset import prepare_data
from risk_scoring.training.features import BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from risk_scoring.training.labels import ImmatureLabelsError
from risk_scoring.training.metrics import average_precision, prevalence
from risk_scoring.training.model import FraudMLP, ModelConfig
from risk_scoring.training.trainer import (
    TrainConfig,
    predict_scores,
    train_model,
)

N_DENSE = len(NUMERIC_COLUMNS) + len(BINARY_COLUMNS)


def small_inputs(prepared):
    train, val = prepared.splits["train"], prepared.splits["validation"]
    return (
        train.encoded,
        train.labels,
        val.encoded,
        val.labels,
        prepared.preprocessor.vocab_sizes,
    )


# --- model ------------------------------------------------------------------------------------


def test_model_has_the_documented_architecture(prepared):
    vocab = prepared.preprocessor.vocab_sizes
    model = FraudMLP(N_DENSE, vocab, ModelConfig())

    dims = {c: e.embedding_dim for c, e in zip(CATEGORICAL_COLUMNS, model.embeddings, strict=True)}
    assert dims == {"mcc": 8, "card_bin": 8, "channel": 3, "issuer_country": 4, "ip_country": 4}
    linears = [m for m in model.mlp if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [
        (N_DENSE + 27, 64),
        (64, 32),
        (32, 1),
    ]
    assert [m.p for m in model.mlp if isinstance(m, torch.nn.Dropout)] == [0.1]


def test_forward_returns_one_logit_per_row(prepared):
    train = prepared.splits["train"].encoded.take(np.arange(7))
    model = FraudMLP(N_DENSE, prepared.preprocessor.vocab_sizes, ModelConfig())
    out = model(torch.from_numpy(train.dense), torch.from_numpy(train.categorical))
    assert out.shape == (7,)


def test_embedding_dims_must_cover_the_categoricals():
    with pytest.raises(ValueError, match="embedding_dims"):
        ModelConfig(embedding_dims={"mcc": 8})


# --- training ---------------------------------------------------------------------------------


def test_model_beats_prevalence_by_a_clear_margin(prepared, trained):
    test = prepared.splits["test"]
    scores = predict_scores(trained.model, test.encoded)

    pr_auc = average_precision(test.labels, scores)
    base = prevalence(test.labels)

    assert pr_auc >= 3 * base, f"PR-AUC {pr_auc:.3f} is not 3x the prevalence {base:.3f}"
    assert pr_auc < 0.98, "suspiciously perfect: look for leakage"


def test_scores_are_in_unit_interval(prepared, trained):
    scores = predict_scores(trained.model, prepared.splits["test"].encoded)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_training_is_deterministic_for_a_seed(prepared):
    args = small_inputs(prepared)
    config = TrainConfig(max_epochs=3)

    a = train_model(*args, config)
    b = train_model(*args, config)

    for (name, wa), (_, wb) in zip(
        a.model.state_dict().items(), b.model.state_dict().items(), strict=True
    ):
        assert torch.equal(wa, wb), f"{name} differs between identical runs"
    assert [r.validation_pr_auc for r in a.history] == [r.validation_pr_auc for r in b.history]


def test_another_seed_gives_other_weights(prepared):
    args = small_inputs(prepared)
    a = train_model(*args, TrainConfig(max_epochs=1, seed=1))
    b = train_model(*args, TrainConfig(max_epochs=1, seed=2))
    assert not torch.equal(a.model.mlp[0].weight, b.model.mlp[0].weight)


def test_best_weights_are_restored_and_early_stopping_is_reported(prepared, trained):
    best = max(trained.history, key=lambda r: r.validation_pr_auc)
    assert trained.best_epoch == best.epoch
    val = prepared.splits["validation"]
    restored = average_precision(val.labels, predict_scores(trained.model, val.encoded))
    assert restored == pytest.approx(best.validation_pr_auc, abs=1e-6)


def test_patience_stops_training_early(prepared):
    result = train_model(*small_inputs(prepared), TrainConfig(max_epochs=30, patience=1))
    assert result.stopped_early
    assert len(result.history) < 30


def test_pos_weight_is_capped(prepared):
    capped = train_model(*small_inputs(prepared), TrainConfig(max_epochs=1, pos_weight_cap=2.0))
    assert capped.pos_weight == 2.0
    uncapped = train_model(*small_inputs(prepared), TrainConfig(max_epochs=1, pos_weight_cap=1e9))
    train = prepared.splits["train"]
    assert uncapped.pos_weight == pytest.approx((~train.labels).sum() / train.labels.sum())


def test_training_needs_positive_labels(prepared):
    train, _, val, y_val, vocab = small_inputs(prepared)
    with pytest.raises(ValueError, match="at least one positive"):
        train_model(train, np.zeros(len(train)), val, y_val, vocab)


def test_config_round_trips_through_a_dict():
    config = TrainConfig(max_epochs=7, model=ModelConfig(hidden_1=16))
    assert TrainConfig.from_dict(config.to_dict()) == config


# --- the dataset glue (leakage guards) ----------------------------------------------------------


def test_preprocessor_is_fitted_on_the_training_split_only(prepared, training_transactions):
    train_rows = prepared.splits["train"].frame
    assert train_rows["occurred_at"].max() < prepared.plan.train_end
    # An mcc that only appears after train_end must be "unknown" for the model.
    later = prepared.splits["test"].frame["mcc"].unique().to_list()
    train_mccs = set(train_rows["mcc"].to_list())
    for mcc in set(later) - train_mccs:
        assert mcc not in prepared.preprocessor.vocabularies["mcc"]


def test_training_labels_are_read_after_the_maturity_period(prepared):
    assert prepared.label_as_of - prepared.plan.train_end == timedelta(days=45)


def test_an_immature_as_of_is_refused_by_the_dataset_builder(training_transactions):
    from risk_scoring.training.splits import plan_split

    plan = plan_split(training_transactions)
    with pytest.raises(ImmatureLabelsError):
        prepare_data(training_transactions, label_as_of=plan.train_end + timedelta(days=10))


def test_validation_and_test_use_eventual_labels(prepared):
    for name in ("validation", "test"):
        split = prepared.splits[name]
        assert split.labels.sum() == split.frame["chargeback_reported_at"].is_not_null().sum()


def test_model_input_does_not_depend_on_the_ground_truth(prepared):
    """Flipping every label and analysis column must not change a single model input."""
    split = prepared.splits["validation"]
    flipped = split.frame.with_columns(
        (~pl.col("is_fraud")).alias("is_fraud"),
        pl.lit("CARD_TESTING").alias("fraud_pattern"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("chargeback_reported_at"),
    )
    again = prepared.preprocessor.prepare(feature_frame(flipped))

    np.testing.assert_array_equal(again.dense, split.encoded.dense)
    np.testing.assert_array_equal(again.categorical, split.encoded.categorical)
