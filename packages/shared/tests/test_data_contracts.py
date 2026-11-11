from datetime import UTC, date, datetime

import polars as pl
import pytest

from payments_shared.data_contracts import (
    Contract,
    ContractViolationError,
    assert_valid,
    check_foreign_key,
    load_contract,
    validate_dataframe,
)
from payments_shared.paths import CONTRACTS_DIR_ENV, contracts_dir, repo_root

CONTRACT = Contract.model_validate(
    {
        "name": "demo",
        "version": 1,
        "description": "demo",
        "primary_key": ["id"],
        "columns": [
            {
                "name": "id",
                "type": "string",
                "nullable": False,
                "role": "id",
                "constraints": {"pattern": "^x_[0-9a-f]{4}$"},
            },
            {
                "name": "amount",
                "type": "int64",
                "nullable": False,
                "role": "feature",
                "constraints": {"min": 1, "max": 100},
            },
            {
                "name": "kind",
                "type": "string",
                "nullable": True,
                "constraints": {"accepted_values": ["A", "B"]},
            },
            {"name": "day", "type": "date", "nullable": False},
            {"name": "at", "type": "timestamp_utc", "nullable": False},
        ],
        "rules": [
            {"name": "a_needs_big_amount", "expr": "kind IS NULL OR kind <> 'A' OR amount > 10"}
        ],
    }
)


def good_frame() -> pl.DataFrame:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "id": ["x_0001", "x_0002"],
            "amount": [50, 20],
            "kind": ["A", None],
            "day": [date(2026, 1, 1)] * 2,
            "at": [ts, ts],
        },
        schema={
            "id": pl.String,
            "amount": pl.Int64,
            "kind": pl.String,
            "day": pl.Date,
            "at": pl.Datetime("us", "UTC"),
        },
    )


def kinds(df: pl.DataFrame) -> set[tuple[str, str]]:
    return {(v.kind, v.target) for v in validate_dataframe(df, CONTRACT)}


def test_valid_frame_has_no_violations():
    assert validate_dataframe(good_frame(), CONTRACT) == []
    assert_valid(good_frame(), CONTRACT)


def test_wrong_dtype_is_caught():
    df = good_frame().with_columns(pl.col("amount").cast(pl.Float64))
    assert ("dtype", "amount") in kinds(df)


def test_wrong_timestamp_timezone_is_caught():
    df = good_frame().with_columns(pl.col("at").dt.replace_time_zone(None))
    assert ("dtype", "at") in kinds(df)


def test_unexpected_and_missing_columns_are_caught():
    df = good_frame().drop("kind").with_columns(extra=pl.lit(1))
    assert {("missing_column", "kind"), ("unexpected_column", "extra")} <= kinds(df)


def test_null_in_non_nullable_column_is_caught():
    df = good_frame().with_columns(amount=pl.Series([None, 20], dtype=pl.Int64))
    assert ("null", "amount") in kinds(df)


def test_null_in_nullable_column_is_fine():
    assert validate_dataframe(good_frame(), CONTRACT) == []


def test_duplicate_primary_key_is_caught():
    df = good_frame().with_columns(id=pl.lit("x_0001"))
    violations = [v for v in validate_dataframe(df, CONTRACT) if v.kind == "primary_key"]
    assert len(violations) == 1
    assert violations[0].count == 2
    assert violations[0].examples == ["x_0001"]


def test_out_of_range_values_are_caught():
    df = good_frame().with_columns(amount=pl.Series([0, 101]))
    found = kinds(df)
    assert ("min", "amount") in found
    assert ("max", "amount") in found


def test_bad_pattern_is_caught():
    df = good_frame().with_columns(id=pl.Series(["x_0001", "nope"]))
    violations = [v for v in validate_dataframe(df, CONTRACT) if v.kind == "pattern"]
    assert violations[0].examples == ["nope"]


def test_value_not_accepted_is_caught():
    df = good_frame().with_columns(kind=pl.Series(["A", "Z"]))
    assert ("accepted_values", "kind") in kinds(df)


def test_failing_rule_is_caught_and_reports_primary_key_examples():
    df = good_frame().with_columns(amount=pl.Series([5, 20]))  # kind A with amount 5
    violations = [v for v in validate_dataframe(df, CONTRACT) if v.kind == "rule"]
    assert [(v.target, v.count, v.examples) for v in violations] == [
        ("a_needs_big_amount", 1, ["x_0001"])
    ]


def test_assert_valid_raises_with_readable_message():
    df = good_frame().with_columns(amount=pl.Series([0, 20]))
    with pytest.raises(ContractViolationError, match=r"\[demo\] min on amount: 1 row"):
        assert_valid(df, CONTRACT)


def test_broken_rule_is_reported_instead_of_crashing():
    broken = Contract.model_validate(
        {**CONTRACT.model_dump(), "rules": [{"name": "bad", "expr": "nope_col > 1"}]}
    )
    violations = validate_dataframe(good_frame(), broken)
    assert [v.kind for v in violations] == ["rule_error"]


def test_foreign_key_detects_orphans_and_ignores_nulls():
    parent = pl.DataFrame({"id": ["a", "b"]})
    child = pl.DataFrame({"ref": ["a", "zzz", None, "zzz"]})
    [violation] = check_foreign_key(child, "ref", parent, "id")
    assert violation.count == 2
    assert violation.examples == ["zzz", "zzz"]
    assert check_foreign_key(pl.DataFrame({"ref": ["a", None]}), "ref", parent, "id") == []


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"primary_key": ["nope"]}, "primary key columns not defined"),
        ({"primary_key": ["kind"]}, "must not be nullable"),
        ({"primary_key": []}, "must not be empty"),
        ({"columns": CONTRACT.model_dump()["columns"] * 2}, "duplicate column names"),
        ({"surprise": 1}, "Extra inputs are not permitted"),
    ],
)
def test_contract_definition_is_checked_for_consistency(patch, message):
    data = {**CONTRACT.model_dump(), **patch}
    with pytest.raises(ValueError, match=message):
        Contract.model_validate(data)


def test_contracts_dir_honors_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv(CONTRACTS_DIR_ENV, str(tmp_path))
    assert contracts_dir() == tmp_path


def test_contracts_dir_defaults_to_repo_contracts(monkeypatch):
    monkeypatch.delenv(CONTRACTS_DIR_ENV, raising=False)
    assert contracts_dir() == repo_root() / "contracts"


def test_load_contract_reads_yaml_and_checks_the_file_name(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    yaml_text = """
name: tiny
version: 2
description: tiny contract
primary_key: [id]
columns:
  - {name: id, type: string, nullable: false, role: id}
"""
    (tmp_path / "data" / "tiny.yaml").write_text(yaml_text)
    (tmp_path / "data" / "other.yaml").write_text(yaml_text)
    monkeypatch.setenv(CONTRACTS_DIR_ENV, str(tmp_path))

    assert load_contract("tiny").version == 2
    with pytest.raises(ValueError, match="does not match file name"):
        load_contract("other")
    with pytest.raises(FileNotFoundError):
        load_contract("missing")
