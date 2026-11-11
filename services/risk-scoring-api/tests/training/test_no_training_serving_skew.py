"""The most important test of the service: training and serving must see the same numbers.

Training reads a Parquet frame and calls the pipeline on all rows at once. Serving receives
nested JSON per request, maps it to a frame and calls the same pipeline. If anything in between
(the mapping, the types, the timestamp precision, the null handling) changed a value, the model
in production would see inputs it was never trained on, and nothing would crash.
"""

import numpy as np
import polars as pl
import pytest

from risk_scoring.dataset_mapping import (
    feature_frame,
    request_to_feature_row,
    requests_to_frame,
    row_to_request,
)
from risk_scoring.training.features import derive_features, required_columns
from risk_scoring.training.preprocessing import Preprocessor

N_ROWS = 400


@pytest.fixture(scope="module")
def rows(sample_transactions) -> pl.DataFrame:
    # Spread over the whole window, so every channel, null pattern and fraud pattern is covered.
    return sample_transactions.gather_every(len(sample_transactions) // N_ROWS).head(N_ROWS)


@pytest.fixture(scope="module")
def preprocessor(sample_transactions) -> Preprocessor:
    return Preprocessor.fit(
        derive_features(feature_frame(sample_transactions)), min_category_count=5
    )


def serving_frame(rows: pl.DataFrame) -> pl.DataFrame:
    """Rows as the API sees them: JSON-shaped request, validated, mapped back to a frame."""
    return requests_to_frame([row_to_request(r) for r in rows.iter_rows(named=True)])


def test_the_request_round_trip_returns_the_raw_features_unchanged(rows):
    training = feature_frame(rows)
    serving = serving_frame(rows)

    assert serving.columns == training.columns == list(required_columns())
    assert serving.schema == training.schema
    assert serving.equals(training)


def test_batch_serving_encodes_exactly_like_training(rows, preprocessor):
    training = preprocessor.prepare(feature_frame(rows))
    serving = preprocessor.prepare(serving_frame(rows))

    np.testing.assert_allclose(serving.dense, training.dense, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(serving.categorical, training.categorical)


def test_single_request_serving_encodes_exactly_like_training(rows, preprocessor):
    training = preprocessor.prepare(feature_frame(rows))

    for i, row in enumerate(rows.iter_rows(named=True)):
        single = preprocessor.prepare(requests_to_frame([row_to_request(row)]))
        np.testing.assert_allclose(single.dense[0], training.dense[i], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(single.categorical[0], training.categorical[i])


def test_the_sample_covers_the_hard_cases_the_parity_test_relies_on(rows):
    assert rows["ip_country"].null_count() > 0  # in-person
    assert rows["avg_amount_30d_minor"].null_count() > 0  # first transactions
    assert rows["channel"].n_unique() == 3
    assert rows["is_proxy"].any()


def test_feature_row_of_a_request_has_the_contract_columns(rows):
    request = row_to_request(rows.row(0, named=True))
    assert list(request_to_feature_row(request)) == list(required_columns())


# --- the final scores, through the real serving class -------------------------------------------


def test_final_scores_through_the_scorer_equal_the_training_path(rows, scorer):
    """Same model, same rows: the batch (training) path and the request (serving) path agree."""
    from risk_scoring.training.trainer import predict_scores

    training_scores = predict_scores(
        scorer.bundle.model, scorer.bundle.preprocessor.prepare(feature_frame(rows))
    )
    requests = [row_to_request(r) for r in rows.iter_rows(named=True)]

    batch_scores = np.array([s.score for s in scorer.score_many(requests)])
    single_scores = np.array([scorer.score_one(r).score for r in requests])

    np.testing.assert_allclose(batch_scores, training_scores, rtol=0, atol=1e-6)
    np.testing.assert_allclose(single_scores, training_scores, rtol=0, atol=1e-6)
