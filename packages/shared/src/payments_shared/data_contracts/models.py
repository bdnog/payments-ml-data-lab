"""The data-contract model: what a YAML file under ``contracts/data/`` may contain."""

from enum import StrEnum
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, model_validator


class ColumnType(StrEnum):
    STRING = "string"
    INT64 = "int64"
    FLOAT64 = "float64"
    BOOL = "bool"
    DATE = "date"
    TIMESTAMP_UTC = "timestamp_utc"


class Role(StrEnum):
    """What a column is for. Later phases derive feature lists from ``role: feature``."""

    ID = "id"
    FEATURE = "feature"
    METADATA = "metadata"
    LABEL_SOURCE = "label_source"
    ANALYSIS_ONLY = "analysis_only"


POLARS_DTYPES: dict[ColumnType, pl.DataType] = {
    ColumnType.STRING: pl.String(),
    ColumnType.INT64: pl.Int64(),
    ColumnType.FLOAT64: pl.Float64(),
    ColumnType.BOOL: pl.Boolean(),
    ColumnType.DATE: pl.Date(),
    ColumnType.TIMESTAMP_UTC: pl.Datetime("us", "UTC"),
}


class _Strict(BaseModel):
    # Typos in a YAML contract must fail loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Constraints(_Strict):
    min: float | None = None
    max: float | None = None
    accepted_values: list[Any] | None = None
    pattern: str | None = None


class Column(_Strict):
    name: str
    type: ColumnType
    nullable: bool
    role: Role = Role.METADATA
    description: str = ""
    constraints: Constraints = Constraints()

    @property
    def dtype(self) -> pl.DataType:
        return POLARS_DTYPES[self.type]


class Rule(_Strict):
    """A row-level rule: a SQL boolean expression. A row violates it when the result is false."""

    name: str
    expr: str


class Contract(_Strict):
    name: str
    version: int
    description: str
    primary_key: list[str]
    columns: list[Column]
    rules: list[Rule] = []

    @model_validator(mode="after")
    def _check_consistency(self) -> "Contract":
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate column names")
        if not self.primary_key:
            raise ValueError(f"{self.name}: primary_key must not be empty")
        unknown = [k for k in self.primary_key if k not in names]
        if unknown:
            raise ValueError(f"{self.name}: primary key columns not defined: {unknown}")
        by_name = {c.name: c for c in self.columns}
        nullable_keys = [k for k in self.primary_key if by_name[k].nullable]
        if nullable_keys:
            raise ValueError(
                f"{self.name}: primary key columns must not be nullable: {nullable_keys}"
            )
        rule_names = [r.name for r in self.rules]
        if len(set(rule_names)) != len(rule_names):
            raise ValueError(f"{self.name}: duplicate rule names")
        return self

    def column(self, name: str) -> Column:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def columns_with_role(self, role: Role) -> list[Column]:
        return [c for c in self.columns if c.role == role]

    @property
    def schema(self) -> dict[str, pl.DataType]:
        """Polars schema, in contract column order."""
        return {c.name: c.dtype for c in self.columns}
