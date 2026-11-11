"""The reconciliation against the generator's answer key, on real generated data."""

import ast
import shutil
from pathlib import Path

import duckdb
import polars as pl

from settlement.batch.__main__ import main
from settlement.batch.verification import INJECTED_PATH, verify
from settlement.domain import BreakType
from settlement.storage.marts import MartStore


def settle_everything(input_dir: Path, marts_dir: Path) -> None:
    assert (
        main(["run-range", "--all", "--input-dir", str(input_dir), "--marts-dir", str(marts_dir)])
        == 0
    )


def assert_perfect(input_dir: Path, marts_dir: Path):
    outcome = verify(input_dir, marts_dir)
    assert outcome.ok, outcome.render()
    assert {r.break_type for r in outcome.per_type} == {b.value for b in BreakType}
    for result in outcome.per_type:
        assert result.precision == 1.0 and result.recall == 1.0, result
        assert result.injected > 0, f"the dataset must exercise {result.break_type}"
    return outcome


def test_the_committed_sample_reconciles_with_100_percent_precision_and_recall(
    sample_dir, tmp_path
):
    settle_everything(sample_dir, tmp_path / "marts")
    assert_perfect(sample_dir, tmp_path / "marts")


def test_datasets_of_other_seeds_reconcile_perfectly_too(seeded_dataset, tmp_path):
    settle_everything(seeded_dataset, tmp_path / "marts")
    assert_perfect(seeded_dataset, tmp_path / "marts")


def test_the_pipeline_works_without_the_answer_key(seeded_datasets, tmp_path):
    """The pipeline must never read injected_breaks: delete the file and run it."""
    source = next(iter(seeded_datasets.values()))
    copy = tmp_path / "input"
    shutil.copytree(source, copy)
    (copy / INJECTED_PATH).unlink()
    assert not (copy / INJECTED_PATH).exists()

    settle_everything(copy, tmp_path / "marts_without_key")
    settle_everything(source, tmp_path / "marts_with_key")

    for mart in ("daily_settlement_by_merchant", "reconciliation_breaks", "fee_summary"):
        without = MartStore(tmp_path / "marts_without_key").scan(mart).collect()
        with_key = MartStore(tmp_path / "marts_with_key").scan(mart).collect()
        assert without.sort(without.columns).equals(with_key.sort(with_key.columns)), mart
    # ...and only then does verification need the file again:
    shutil.copy(source / INJECTED_PATH, copy / INJECTED_PATH)
    assert verify(copy, tmp_path / "marts_without_key").ok


def code_strings(path: Path) -> list[str]:
    """String literals of a module, minus docstrings (prose may mention the answer key)."""
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]


def test_only_the_verification_module_refers_to_the_answer_key():
    """A static guard next to the behavioral one: no other module can name the file."""
    package = Path(__file__).resolve().parents[2] / "src" / "settlement"
    offenders = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if any("injected_breaks" in value for value in code_strings(path))
    ]
    assert offenders == ["batch/verification.py"]


def test_verification_fails_when_the_mart_loses_a_break(sample_dir, tmp_path):
    marts = tmp_path / "marts"
    settle_everything(sample_dir, marts)
    store = MartStore(marts)
    day = next(
        d
        for d in store.partition_dates("reconciliation_breaks")
        if store.read_partition("reconciliation_breaks", d).height
    )
    frame = store.read_partition("reconciliation_breaks", day)
    store.write_partition("reconciliation_breaks", day, frame.slice(1))  # drop one row

    outcome = verify(sample_dir, marts)

    assert not outcome.ok and len(outcome.missed) == 1
    assert "FAILED" in outcome.render()
    assert main(["verify", "--input-dir", str(sample_dir), "--marts-dir", str(marts)]) == 1


def test_verification_fails_when_the_mart_invents_a_break(sample_dir, tmp_path):
    marts = tmp_path / "marts"
    settle_everything(sample_dir, marts)
    store = MartStore(marts)
    day = store.partition_dates("reconciliation_breaks")[0]
    frame = store.read_partition("reconciliation_breaks", day)
    invented = frame.head(1).with_columns(
        break_id=pl.lit("brk_ffffffff"),
        payment_id=pl.lit("pay_ffffffffffff"),
        break_type=pl.lit("MISSING_IN_PSP"),
        psp_record_id=pl.lit(None, dtype=pl.String),
    )
    store.write_partition("reconciliation_breaks", day, pl.concat([frame, invented]))

    outcome = verify(sample_dir, marts)

    assert (
        not outcome.ok
        and outcome.unexpected
        and outcome.unexpected[0]["payment_id"] == "pay_ffffffffffff"
    )


def test_verification_catches_a_wrong_difference(sample_dir, tmp_path):
    marts = tmp_path / "marts"
    settle_everything(sample_dir, marts)
    store = MartStore(marts)
    day = next(
        d
        for d in store.partition_dates("reconciliation_breaks")
        if store.read_partition("reconciliation_breaks", d).height
    )
    frame = store.read_partition("reconciliation_breaks", day)
    store.write_partition(
        "reconciliation_breaks", day, frame.with_columns(diff_minor=pl.col("diff_minor") + 1)
    )

    assert not verify(sample_dir, marts).ok


def test_verification_of_an_empty_marts_directory_is_a_failure(sample_dir, tmp_path):
    outcome = verify(sample_dir, tmp_path / "nothing")
    assert not outcome.ok and "nothing to verify" in outcome.render()


def test_verification_only_looks_at_the_dates_the_marts_cover(sample_dir, tmp_path):
    marts = tmp_path / "marts"
    assert (
        main(
            [
                "run-range",
                "--from",
                "2026-01-10",
                "--to",
                "2026-01-12",
                "--input-dir",
                str(sample_dir),
                "--marts-dir",
                str(marts),
            ]
        )
        == 0
    )
    outcome = verify(sample_dir, marts)
    assert outcome.ok and len(outcome.dates) == 3
    assert sum(r.injected for r in outcome.per_type) < 160


# --- a BI tool querying the result ------------------------------------------------------------


def bi_query(root: Path, mart: str, sql: str):
    """Run SQL over a mart exactly as a BI user would: a hive-partitioned Parquet glob."""
    source = f"read_parquet('{root}/{mart}/**/*.parquet', hive_partitioning=true)"
    return duckdb.sql(sql.format(t=source)).fetchall()


def test_duckdb_can_query_the_marts_of_a_whole_run(sample_dir, tmp_path):
    settle_everything(sample_dir, tmp_path / "marts")
    root = tmp_path / "marts"

    by_type = dict(
        bi_query(root, "reconciliation_breaks", "select break_type, count(*) from {t} group by 1")
    )
    payout = bi_query(root, "daily_settlement_by_merchant", "select sum(net_minor) from {t}")
    pruned = bi_query(
        root,
        "fee_summary",
        "select count(distinct settlement_date) from {t} "
        "where settlement_date between date '2026-01-05' and date '2026-01-07'",
    )

    assert by_type == {
        "AMOUNT_MISMATCH": 39,
        "MISSING_IN_PSP": 40,
        "MISSING_IN_LEDGER": 25,
        "DUPLICATE_IN_PSP": 20,
        "FEE_MISMATCH": 21,
        "CHARGEBACK_NOT_IN_LEDGER": 15,
    }
    assert payout == [(36_515_515,)]
    assert pruned == [(3,)]
