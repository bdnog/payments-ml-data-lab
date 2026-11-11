import json

import pytest

from settlement.batch.__main__ import main
from settlement.storage.marts import MartStore


def paths(sample_dir, tmp_path):
    return ["--input-dir", str(sample_dir), "--marts-dir", str(tmp_path / "marts")]


def test_run_settles_one_date_and_prints_a_summary(sample_dir, tmp_path, capsys):
    assert main(["run", "--date", "2026-01-15", *paths(sample_dir, tmp_path)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["date_count"] == 1 and summary["first_date"] == "2026-01-15"
    assert MartStore(tmp_path / "marts").partition_dates("fee_summary").__len__() == 1


def test_run_range_covers_every_date_of_the_range(sample_dir, tmp_path, capsys):
    args = ["run-range", "--from", "2026-01-10", "--to", "2026-01-14", *paths(sample_dir, tmp_path)]
    assert main(args) == 0

    assert json.loads(capsys.readouterr().out)["date_count"] == 5


def test_run_range_all_reads_the_dates_from_the_psp_file(sample_dir, tmp_path, capsys):
    assert main(["run-range", "--all", *paths(sample_dir, tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["first_date"], summary["last_date"], summary["date_count"]) == (
        "2026-01-01",
        "2026-01-30",
        30,
    )


def test_verify_exits_zero_on_a_correct_run(sample_dir, tmp_path, capsys):
    main(["run-range", "--all", *paths(sample_dir, tmp_path)])
    capsys.readouterr()

    assert main(["verify", *paths(sample_dir, tmp_path)]) == 0
    assert "100% precision and recall" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["2026-13-01", "10/11/2026", "20261110", "not-a-date"])
def test_invalid_dates_exit_nonzero(bad):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--date", bad])
    assert exc.value.code != 0


@pytest.mark.parametrize(
    "argv",
    [
        ["run"],
        ["run-range"],
        ["run-range", "--from", "2026-01-01"],
        ["run-range", "--from", "2026-01-05", "--to", "2026-01-01"],
        ["run-range", "--all", "--from", "2026-01-01"],
    ],
)
def test_incomplete_or_inconsistent_arguments_are_usage_errors(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code != 0


def test_skipping_input_validation_does_not_skip_output_validation(sample_dir, tmp_path):
    """With bad inputs the run fails at input validation; skipping that, the marts still refuse."""
    import shutil

    import polars as pl

    broken = tmp_path / "input"
    shutil.copytree(sample_dir, broken)
    path = broken / "settlement" / "payments.parquet"
    pl.read_parquet(path).with_columns(amount_minor=pl.lit(-5, dtype=pl.Int64)).write_parquet(path)
    common = ["--input-dir", str(broken), "--marts-dir", str(tmp_path / "marts")]

    with pytest.raises(AssertionError, match=r"\[payments\].*amount_minor"):
        main(["run", "--date", "2026-01-15", *common])
    with pytest.raises(AssertionError, match=r"\[daily_settlement_by_merchant\].*gross_minor"):
        main(["run", "--date", "2026-01-15", "--skip-input-validation", *common])


def test_a_failed_validation_writes_no_partition(sample_dir, tmp_path):
    import shutil

    import polars as pl

    broken = tmp_path / "input"
    shutil.copytree(sample_dir, broken)
    path = broken / "settlement" / "payments.parquet"
    pl.read_parquet(path).with_columns(amount_minor=pl.lit(-5, dtype=pl.Int64)).write_parquet(path)

    with pytest.raises(AssertionError):
        main(
            [
                "run",
                "--date",
                "2026-01-15",
                "--input-dir",
                str(broken),
                "--marts-dir",
                str(tmp_path / "marts"),
            ]
        )
    assert not (tmp_path / "marts").exists()
