"""The manifest: what was generated and what the data actually looks like.

It holds the seed, the profile parameters, row counts and *realized* rates. There is deliberately
no timestamp and no library version in it: regenerating with the same inputs gives the same file.
"""

import json
from pathlib import Path

import polars as pl

from payments_datagen.generate import World

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def realized_statistics(frames: dict[str, pl.DataFrame]) -> dict:
    """Rates measured on the data itself (no hidden state), so they can be recomputed from files."""
    txns, payments = frames["transactions"], frames["payments"]
    fraud = txns.filter(pl.col("is_fraud"))
    captured = payments.filter(pl.col("status") == "CAPTURED")
    approved_fraud = fraud.join(captured.select("transaction_id"), on="transaction_id", how="inner")
    breaks = frames["injected_breaks"]

    def counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
        found = frame.group_by(column).len().sort(column)
        return {row[column]: row["len"] for row in found.to_dicts()}

    return {
        "fraud_transactions": fraud.height,
        "fraud_rate": _rate(fraud.height, txns.height),
        "fraud_by_pattern": counts(fraud, "fraud_pattern"),
        "fraud_with_chargeback_share": _rate(
            fraud["chargeback_reported_at"].is_not_null().sum(), fraud.height
        ),
        "approved_fraud_with_chargeback_share": _rate(
            approved_fraud["chargeback_reported_at"].is_not_null().sum(), approved_fraud.height
        ),
        "approval_rate": _rate(payments.height, txns.height),
        "voided_rate": _rate(payments.height - captured.height, payments.height),
        "refunded_payment_share": _rate(
            frames["refunds"]["payment_id"].n_unique(), captured.height
        ),
        "break_rate": _rate(breaks.height, captured.height),
        "breaks_by_type": counts(breaks, "break_type"),
        "psp_records_by_type": counts(frames["psp_settlement"], "record_type"),
    }


def build_manifest(world: World) -> dict:
    profile = world.universe.profile
    return {
        "manifest_version": MANIFEST_VERSION,
        "seed": profile.seed,
        "profile": profile.name,
        "parameters": profile.to_dict(),
        "row_counts": {name: frame.height for name, frame in world.frames.items()},
        "realized": realized_statistics(world.frames),
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
