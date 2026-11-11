import shutil
from datetime import date

import polars as pl
import pytest

from payments_shared.data_contracts import ContractViolationError
from settlement.storage.datasets import (
    INPUT_PATHS,
    InputValidationError,
    input_file,
    psp_date_range,
    scan_inputs,
    validate_inputs,
)


def test_inputs_are_lazy_and_cover_the_five_datasets(sample_dir):
    inputs = scan_inputs(sample_dir)

    for frame in (
        inputs.merchants,
        inputs.payments,
        inputs.refunds,
        inputs.chargebacks,
        inputs.psp,
    ):
        assert isinstance(frame, pl.LazyFrame)
    assert inputs.psp.select(pl.len()).collect().item() > 1_000


def test_the_answer_key_is_not_an_input():
    assert not any("injected" in path for path in INPUT_PATHS.values())


def test_the_sample_is_valid(sample_dir):
    validate_inputs(sample_dir)


def test_every_seeded_dataset_is_valid(seeded_datasets):
    for directory in seeded_datasets.values():
        validate_inputs(directory)


def test_a_missing_dataset_says_how_to_generate_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="payments_datagen generate"):
        scan_inputs(tmp_path)


@pytest.fixture
def copy_of_sample(tmp_path, sample_dir):
    target = tmp_path / "input"
    for name in INPUT_PATHS:
        destination = input_file(target, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(input_file(sample_dir, name), destination)
    return target


def test_a_contract_violation_in_an_input_fails_validation(copy_of_sample):
    path = input_file(copy_of_sample, "payments")
    frame = pl.read_parquet(path)
    frame.with_columns(amount_minor=pl.lit(-1)).write_parquet(path)

    with pytest.raises(ContractViolationError, match="amount_minor"):
        validate_inputs(copy_of_sample)


def test_a_dangling_foreign_key_fails_validation(copy_of_sample):
    path = input_file(copy_of_sample, "refunds")
    frame = pl.read_parquet(path)
    frame.with_columns(payment_id=pl.lit("pay_ffffffffffff")).write_parquet(path)

    with pytest.raises(InputValidationError, match="payment_id -> payment_id"):
        validate_inputs(copy_of_sample)


def test_psp_date_range_is_the_span_of_the_file(sample_dir):
    first, last = psp_date_range(sample_dir)
    assert first == date(2026, 1, 1) and last == date(2026, 1, 30)


def test_psp_date_range_of_an_empty_file_is_an_error(copy_of_sample):
    path = input_file(copy_of_sample, "psp_settlement")
    pl.read_parquet(path).head(0).write_parquet(path)
    with pytest.raises(ValueError, match="empty"):
        psp_date_range(copy_of_sample)
