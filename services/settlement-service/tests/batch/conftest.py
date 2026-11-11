"""A tiny builder of hand-made input datasets: each test states exactly the situation it checks."""

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from settlement.batch.pipeline import RunSummary, run_dates
from settlement.storage.marts import MartStore

MERCHANT = "mer_000001"


def pay(n: int) -> str:
    return f"pay_{n:012x}"


def psp_id(n: int) -> str:
    return f"psp_{n:012x}"


def at(day: str, hour: int = 10) -> datetime:
    return datetime.fromisoformat(f"{day}T{hour:02d}:00:00").replace(tzinfo=UTC)


class Scenario:
    """Collects ledger and PSP rows, writes them as the dataset layout, and runs the pipeline.

    Defaults: one merchant at 100 bps settling D+1, so a 10,000 sale captured on 2026-03-10 is due
    on 2026-03-11 with a contracted fee of 100.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.merchants = [(MERCHANT, 100, 1)]
        self.payments: list[dict] = []
        self.refunds: list[dict] = []
        self.chargebacks: list[dict] = []
        self.psp: list[dict] = []

    # --- ledger ---------------------------------------------------------------------------------

    def merchant(self, merchant_id: str, mdr_bps: int, settlement_days: int) -> None:
        self.merchants.append((merchant_id, mdr_bps, settlement_days))

    def payment(
        self,
        n: int,
        amount: int = 10_000,
        captured: str = "2026-03-10",
        merchant: str = MERCHANT,
        status: str = "CAPTURED",
    ) -> str:
        self.payments.append(
            {
                "payment_id": pay(n),
                "merchant_id": merchant,
                "currency": "BRL",
                "status": status,
                "amount_minor": amount,
                "captured_at": at(captured) if status == "CAPTURED" else None,
            }
        )
        return pay(n)

    def refund(self, n: int, payment: int, amount: int, on: str = "2026-03-10") -> None:
        self.refunds.append(
            {
                "refund_id": f"ref_{n:012x}",
                "payment_id": pay(payment),
                "amount_minor": amount,
                "created_at": at(on),
            }
        )

    def chargeback(self, n: int, payment: int, amount: int, on: str = "2026-03-10") -> None:
        self.chargebacks.append(
            {
                "chargeback_id": f"cbk_{n:012x}",
                "payment_id": pay(payment),
                "amount_minor": amount,
                "opened_at": at(on),
            }
        )

    # --- PSP ------------------------------------------------------------------------------------

    def psp_record(
        self,
        n: int,
        reference: int | str | None,
        record_type: str = "SALE",
        gross: int = 10_000,
        fee: int | None = None,
        on: str = "2026-03-11",
        merchant: str = MERCHANT,
    ) -> None:
        if fee is None:
            fee = {"SALE": gross * 100 // 10_000, "REFUND": 0, "CHARGEBACK": 1_500}[record_type]
        self.psp.append(
            {
                "psp_record_id": psp_id(n),
                "merchant_reference": pay(reference) if isinstance(reference, int) else reference,
                "record_type": record_type,
                "merchant_id": merchant,
                "gross_amount_minor": gross,
                "fee_minor": fee,
                "net_amount_minor": 0,  # not used by the pipeline
                "currency": "BRL",
                "settlement_date": date.fromisoformat(on),
            }
        )

    # --- running --------------------------------------------------------------------------------

    def write(self) -> Path:
        (self.root / "reference").mkdir(parents=True, exist_ok=True)
        (self.root / "settlement").mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            self.merchants, schema=["merchant_id", "mdr_bps", "settlement_days"], orient="row"
        ).write_parquet(self.root / "reference" / "merchants.parquet")

        def frame(rows: list[dict], schema: dict) -> pl.DataFrame:
            return pl.DataFrame(rows, schema=schema)

        ts = pl.Datetime("us", "UTC")
        frame(
            self.payments,
            {
                "payment_id": pl.String,
                "merchant_id": pl.String,
                "currency": pl.String,
                "status": pl.String,
                "amount_minor": pl.Int64,
                "captured_at": ts,
            },
        ).write_parquet(self.root / "settlement" / "payments.parquet")
        frame(
            self.refunds,
            {
                "refund_id": pl.String,
                "payment_id": pl.String,
                "amount_minor": pl.Int64,
                "created_at": ts,
            },
        ).write_parquet(self.root / "settlement" / "refunds.parquet")
        frame(
            self.chargebacks,
            {
                "chargeback_id": pl.String,
                "payment_id": pl.String,
                "amount_minor": pl.Int64,
                "opened_at": ts,
            },
        ).write_parquet(self.root / "settlement" / "chargebacks.parquet")
        frame(
            self.psp,
            {
                "psp_record_id": pl.String,
                "merchant_reference": pl.String,
                "record_type": pl.String,
                "merchant_id": pl.String,
                "gross_amount_minor": pl.Int64,
                "fee_minor": pl.Int64,
                "net_amount_minor": pl.Int64,
                "currency": pl.String,
                "settlement_date": pl.Date,
            },
        ).write_parquet(self.root / "settlement" / "psp_settlement.parquet")
        return self.root

    def run(self, *days: str) -> RunSummary:
        self.write()
        self.store = MartStore(self.root / "marts")
        dates = [date.fromisoformat(d) for d in days]
        return run_dates(dates, self.root, self.root / "marts", validate=False)

    def mart(self, name: str) -> pl.DataFrame:
        return self.store.scan(name).collect().sort(self.store.scan(name).collect_schema().names())

    def breaks(self) -> pl.DataFrame:
        return self.store.scan("reconciliation_breaks").collect().sort("break_id")


@pytest.fixture
def scenario(tmp_path) -> Scenario:
    return Scenario(tmp_path / "scenario")
