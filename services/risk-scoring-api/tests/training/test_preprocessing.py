import numpy as np
import polars as pl
import pytest

from risk_scoring.training.features import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    derive_features,
    required_columns,
)
from risk_scoring.training.preprocessing import CLIP, MISSING, UNKNOWN, Preprocessor

N_DENSE = len(NUMERIC_COLUMNS) + len(BINARY_COLUMNS)


@pytest.fixture(scope="module")
def derived(sample_transactions) -> pl.DataFrame:
    return derive_features(sample_transactions)


@pytest.fixture(scope="module")
def fitted(derived) -> Preprocessor:
    return Preprocessor.fit(derived, min_category_count=5)


def test_continuous_columns_are_standardized_with_the_fitted_statistics(derived, fitted):
    encoded = fitted.transform(derived)

    assert encoded.dense.shape == (derived.height, N_DENSE)
    assert encoded.dense.dtype == np.float32
    assert encoded.categorical.shape == (derived.height, len(CATEGORICAL_COLUMNS))
    assert encoded.categorical.dtype == np.int64
    numeric = encoded.dense[:, : len(NUMERIC_COLUMNS)]
    # Clipping at 5 sigma moves the mean a little, so the tolerance is loose on purpose.
    assert np.abs(numeric.mean(axis=0)).max() < 0.1
    assert np.abs(numeric).max() <= CLIP + 1e-6


def test_flags_are_passed_through_unscaled(derived, fitted):
    flags = fitted.transform(derived).dense[:, len(NUMERIC_COLUMNS) :]
    assert set(np.unique(flags)) <= {0.0, 1.0}


def test_a_constant_column_does_not_divide_by_zero(derived):
    constant = derived.with_columns(pl.lit(4.0).alias("installments"))
    pre = Preprocessor.fit(constant, min_category_count=5)

    assert pre.stds["installments"] == 1.0
    assert np.isfinite(pre.transform(constant).dense).all()


def test_vocabularies_reserve_unknown_and_only_ip_country_has_a_missing_slot(fitted):
    for column, vocabulary in fitted.vocabularies.items():
        assert vocabulary[0] == UNKNOWN
        assert (MISSING in vocabulary) == (column == "ip_country")
        assert len(set(vocabulary)) == len(vocabulary)


def test_rare_categories_map_to_unknown(derived):
    counts = derived["mcc"].value_counts()
    threshold = int(counts["count"].median()) + 1  # about half of the mccs are rarer than this
    rare = counts.filter(pl.col("count") < threshold)["mcc"].to_list()
    assert rare and len(rare) < counts.height

    pre = Preprocessor.fit(derived, min_category_count=threshold)

    assert not set(rare) & set(pre.vocabularies["mcc"])
    rare_rows = derived.filter(pl.col("mcc").is_in(pl.Series(rare).implode()))
    assert (pre.transform(rare_rows).categorical[:, 0] == 0).all()
    common_rows = derived.filter(~pl.col("mcc").is_in(pl.Series(rare).implode()))
    assert (pre.transform(common_rows).categorical[:, 0] > 0).all()


def test_unseen_categories_and_null_ip_country_get_their_own_indices(derived, fitted):
    odd = derived.head(2).with_columns(
        pl.Series("mcc", ["9999", "9999"]),
        pl.Series("ip_country", [None, "ZZ"], dtype=pl.String),
    )
    encoded = fitted.transform(odd).categorical
    column = {c: i for i, c in enumerate(CATEGORICAL_COLUMNS)}

    assert encoded[0, column["mcc"]] == encoded[1, column["mcc"]] == 0  # __unknown__
    assert encoded[0, column["ip_country"]] == fitted.vocabularies["ip_country"].index(MISSING)
    assert encoded[1, column["ip_country"]] == 0  # unseen country is unknown, not missing


def test_json_round_trip_preserves_the_transformation(derived, fitted):
    restored = Preprocessor.from_json(fitted.to_json())

    assert restored == fitted
    a, b = fitted.transform(derived), restored.transform(derived)
    np.testing.assert_array_equal(a.dense, b.dense)
    np.testing.assert_array_equal(a.categorical, b.categorical)


def test_serialized_form_is_stable_and_documents_its_inputs(fitted):
    data = fitted.to_dict()
    assert data["schema_version"] == 1
    assert data["input_columns"] == list(required_columns())
    assert fitted.to_json() == Preprocessor.from_json(fitted.to_json()).to_json()


def test_unsupported_schema_version_is_rejected(fitted):
    with pytest.raises(ValueError, match="schema"):
        Preprocessor.from_dict({**fitted.to_dict(), "schema_version": 99})


def test_fitted_on_other_columns_than_the_contract_fails_fast(fitted):
    stale = Preprocessor.from_dict({**fitted.to_dict(), "input_columns": ["amount_minor"]})
    with pytest.raises(ValueError, match="different feature columns"):
        stale.check_against_contract()
    fitted.check_against_contract()  # the real one is fine


def test_cannot_fit_on_nothing(derived):
    with pytest.raises(ValueError, match="empty"):
        Preprocessor.fit(derived.head(0))
