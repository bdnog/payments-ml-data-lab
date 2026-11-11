"""The whole pipeline: profile in, one coherent universe of datasets out."""

from dataclasses import dataclass

import polars as pl

from payments_datagen.fraud import apply_outcomes, generate_fraud
from payments_datagen.profiles import Profile
from payments_datagen.settlement import build_settlement
from payments_datagen.transactions import assemble_transactions, generate_legit
from payments_datagen.universe import Universe, build_universe

# Dataset name -> path relative to the output directory.
DATASET_PATHS = {
    "merchants": "reference/merchants.parquet",
    "accounts": "reference/accounts.parquet",
    "transactions": "risk/transactions.parquet",
    "payments": "settlement/payments.parquet",
    "refunds": "settlement/refunds.parquet",
    "chargebacks": "settlement/chargebacks.parquet",
    "psp_settlement": "settlement/psp_settlement.parquet",
    "injected_breaks": "settlement/injected_breaks.parquet",
}


@dataclass(frozen=True)
class World:
    """Everything one seed produces.

    ``frames`` are the datasets that get written to disk. ``outcomes`` (which transactions were
    declined or voided) is hidden state used to derive the settlement data; it is deliberately not
    written, because exposing it would leak the label.
    """

    universe: Universe
    frames: dict[str, pl.DataFrame]
    outcomes: pl.DataFrame

    def __getitem__(self, name: str) -> pl.DataFrame:
        return self.frames[name]


def generate_world(profile: Profile) -> World:
    universe = build_universe(profile)
    legit = generate_legit(universe)
    fraud = generate_fraud(universe, legit)
    transactions, outcomes = apply_outcomes(universe, assemble_transactions(universe, legit, fraud))
    settlement = build_settlement(universe, transactions, outcomes)
    frames = {
        "merchants": universe.merchants,
        "accounts": universe.accounts,
        "transactions": transactions,
        "payments": settlement.payments,
        "refunds": settlement.refunds,
        "chargebacks": settlement.chargebacks,
        "psp_settlement": settlement.psp_settlement,
        "injected_breaks": settlement.injected_breaks,
    }
    return World(universe, frames, outcomes)
