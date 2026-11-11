"""Validating Polars DataFrames against a data contract.

Checks, in order: exact column set and dtypes, nullability, primary key uniqueness, per-column
constraints, row-level rules. A column with the wrong dtype is not checked further, because the
later checks would only produce noise.
"""

from dataclasses import dataclass, field

import polars as pl

from payments_shared.data_contracts.models import Column, Contract

MAX_EXAMPLES = 3


@dataclass(frozen=True)
class Violation:
    contract: str
    kind: str  # missing_column, unexpected_column, dtype, null, primary_key, min, max, ...
    target: str  # column name, rule name or "a -> b" for foreign keys
    count: int
    examples: list[str] = field(default_factory=list)
    detail: str = ""

    def __str__(self) -> str:
        examples = f" e.g. {', '.join(self.examples)}" if self.examples else ""
        detail = f" ({self.detail})" if self.detail else ""
        return (
            f"[{self.contract}] {self.kind} on {self.target}: {self.count} row(s){detail}{examples}"
        )


class ContractViolationError(AssertionError):
    def __init__(self, violations: list[Violation]):
        self.violations = violations
        super().__init__("data contract violated:\n  " + "\n  ".join(str(v) for v in violations))


def _examples(values: pl.Series) -> list[str]:
    return [str(v) for v in values.head(MAX_EXAMPLES).to_list()]


def _check_schema(df: pl.DataFrame, contract: Contract) -> tuple[list[Violation], set[str]]:
    """Return schema violations and the set of columns that are safe to check further."""
    violations: list[Violation] = []
    expected = set(contract.column_names)
    present = set(df.columns)
    for name in contract.column_names:
        if name not in present:
            violations.append(Violation(contract.name, "missing_column", name, 0))
    for name in df.columns:
        if name not in expected:
            violations.append(Violation(contract.name, "unexpected_column", name, 0))
    usable: set[str] = set()
    for col in contract.columns:
        if col.name not in present:
            continue
        actual = df.schema[col.name]
        if actual != col.dtype:
            detail = f"expected {col.dtype}, got {actual}"
            violations.append(Violation(contract.name, "dtype", col.name, 0, detail=detail))
        else:
            usable.add(col.name)
    return violations, usable


def _check_constraints(df: pl.DataFrame, contract: Contract, col: Column) -> list[Violation]:
    values = pl.col(col.name)
    present = df.filter(values.is_not_null())
    c = col.constraints
    checks: list[tuple[str, pl.Expr, str]] = []
    if c.min is not None:
        checks.append(("min", values < c.min, f"minimum is {c.min}"))
    if c.max is not None:
        checks.append(("max", values > c.max, f"maximum is {c.max}"))
    if c.accepted_values is not None:
        checks.append(
            ("accepted_values", ~values.is_in(c.accepted_values), f"accepted: {c.accepted_values}")
        )
    if c.pattern is not None:
        checks.append(("pattern", ~values.str.contains(c.pattern), f"pattern {c.pattern}"))
    violations = []
    for kind, bad_expr, detail in checks:
        bad = present.filter(bad_expr)[col.name]
        if len(bad):
            violations.append(
                Violation(contract.name, kind, col.name, len(bad), _examples(bad), detail)
            )
    return violations


def _check_rule(df: pl.DataFrame, contract: Contract, name: str, expr: str) -> Violation | None:
    try:
        ok = df.select(pl.sql_expr(expr).alias("ok"))["ok"]
    except Exception as exc:  # a broken rule must be reported, not crash the whole validation
        return Violation(
            contract.name, "rule_error", name, 0, detail=f"{type(exc).__name__}: {exc}"
        )
    # SQL three-valued logic: only an explicit false is a violation; null is not.
    bad_mask = ok.fill_null(True).not_()
    count = int(bad_mask.sum())
    if count == 0:
        return None
    key = contract.primary_key if all(k in df.columns for k in contract.primary_key) else None
    offending = df.filter(bad_mask)
    examples = (
        [", ".join(str(v) for v in row) for row in offending.select(key).head(MAX_EXAMPLES).rows()]
        if key
        else []
    )
    return Violation(contract.name, "rule", name, count, examples, detail=expr)


def validate_dataframe(df: pl.DataFrame, contract: Contract) -> list[Violation]:
    """Return every violation found; an empty list means the frame honors the contract."""
    violations, usable = _check_schema(df, contract)

    for col in contract.columns:
        if col.name not in usable:
            continue
        if not col.nullable:
            nulls = df[col.name].null_count()
            if nulls:
                violations.append(Violation(contract.name, "null", col.name, nulls))
        violations.extend(_check_constraints(df, contract, col))

    if all(k in usable for k in contract.primary_key):
        dup_mask = df.select(contract.primary_key).is_duplicated()
        dup_count = int(dup_mask.sum())
        if dup_count:
            dup_keys = df.filter(dup_mask).select(contract.primary_key).unique(maintain_order=True)
            rows = dup_keys.head(MAX_EXAMPLES).rows()
            examples = [", ".join(str(v) for v in row) for row in rows]
            target = ", ".join(contract.primary_key)
            violations.append(Violation(contract.name, "primary_key", target, dup_count, examples))

    # Rules need every column they might reference; skip them when the schema itself is broken.
    if len(usable) == len(contract.columns) and len(df.columns) == len(contract.columns):
        for rule in contract.rules:
            violation = _check_rule(df, contract, rule.name, rule.expr)
            if violation:
                violations.append(violation)
    return violations


def assert_valid(df: pl.DataFrame, contract: Contract) -> None:
    violations = validate_dataframe(df, contract)
    if violations:
        raise ContractViolationError(violations)


def check_foreign_key(
    child_df: pl.DataFrame, child_col: str, parent_df: pl.DataFrame, parent_col: str
) -> list[Violation]:
    """Every non-null ``child_col`` value must exist in ``parent_col``."""
    orphans = (
        child_df.filter(pl.col(child_col).is_not_null())
        .select(pl.col(child_col).alias("v"))
        .join(parent_df.select(pl.col(parent_col).alias("v")).unique(), on="v", how="anti")["v"]
    )
    if not len(orphans):
        return []
    return [
        Violation(
            "foreign_key",
            "foreign_key",
            f"{child_col} -> {parent_col}",
            len(orphans),
            _examples(orphans),
        )
    ]
