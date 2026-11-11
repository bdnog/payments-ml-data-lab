"""The static part of the universe: merchants, accounts and what they own.

Besides the two reference datasets that are written to disk (``merchants``, ``accounts``), the
universe carries *latent traits* used by later stages (how active an account is, which cards and
devices it owns, whether it travels). Latent traits are never written out: they are the hidden
state that the observable data is generated from.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from payments_datagen.catalog import (
    BINS_PER_FOREIGN_COUNTRY,
    BR_BIN_COUNT,
    FOREIGN_IP_COUNTRIES,
    HOME_COUNTRIES,
    HOME_COUNTRY_WEIGHTS,
    MCC_TABLE,
    NAME_PREFIXES,
)
from payments_datagen.profiles import Profile
from payments_datagen.rng import stream
from payments_datagen.util import US_PER_DAY, epoch_us, hex_ids, ts_series

NEW_ACCOUNT_SHARE = 0.15  # accounts opened during the window
TRAVELER_SHARE = 0.05  # accounts that spend some days abroad
SECOND_CARD_SHARE = 0.25


@dataclass(frozen=True)
class Universe:
    profile: Profile
    merchants: pl.DataFrame  # contract: merchants
    accounts: pl.DataFrame  # contract: accounts
    traits: pl.DataFrame  # latent, one row per account
    devices: pl.DataFrame  # latent: account_id, device_id, rank (0 is the primary device)
    terminals: pl.DataFrame  # latent: merchant_id, device_id (pos terminals)
    bins: pl.DataFrame  # latent: card_bin, issuer_country


def build_bins(seed: int) -> pl.DataFrame:
    """40 synthetic BINs: 30 Brazilian and 2 for each of 5 other countries.

    BINs are drawn from 9xxxxx, a range not assigned to real card networks.
    """
    rng = stream(seed, "bins")
    countries = ["BR"] * BR_BIN_COUNT
    for country in HOME_COUNTRIES:
        if country != "BR":
            countries += [country] * BINS_PER_FOREIGN_COUNTRY
    numbers = rng.choice(100_000, size=len(countries), replace=False) + 900_000
    return pl.DataFrame({"card_bin": [f"{n:06d}" for n in numbers], "issuer_country": countries})


def build_merchants(profile: Profile) -> pl.DataFrame:
    rng = stream(profile.seed, "merchants")
    n = profile.n_merchants
    codes = MCC_TABLE["mcc"].to_numpy()
    weights = MCC_TABLE["weight"].to_numpy()
    covered = min(n, len(codes))
    # The first merchants cover every category once, so every category exists in small profiles too.
    mcc = np.concatenate(
        [
            rng.permutation(codes)[:covered],
            rng.choice(codes, size=n - covered, p=weights / weights.sum()),
        ]
    )
    created_us = (
        epoch_us(profile.window_start)
        - rng.integers(30, 1500, size=n) * US_PER_DAY
        - rng.integers(0, US_PER_DAY, size=n)
    )
    prefix = np.array(NAME_PREFIXES)[rng.integers(0, len(NAME_PREFIXES), size=n)]
    country = rng.choice(["BR", "AR", "PT"], size=n, p=[0.97, 0.02, 0.01])
    mdr_bps = rng.integers(50, 501, size=n)
    settlement_days = rng.choice([1, 2, 30], size=n, p=[0.35, 0.50, 0.15])

    frame = pl.DataFrame(
        {
            "mcc": mcc,
            "country": country,
            "mdr_bps": mdr_bps,
            "settlement_days": settlement_days,
            "created_at": ts_series("created_at", created_us),
            "prefix": prefix,
        }
    ).sort(["created_at", "mcc"])
    ids = hex_ids(rng, "mer_", 6, n, "merchant_id")
    return (
        frame.with_columns(ids, i=pl.int_range(1, n + 1))
        .join(MCC_TABLE.select("mcc", "label"), on="mcc", how="left", maintain_order="left")
        .with_columns(
            name=pl.format(
                "{} {} {}",
                pl.col("prefix"),
                pl.col("label"),
                pl.col("i").cast(pl.String).str.zfill(4),
            )
        )
        .select("merchant_id", "name", "mcc", "country", "mdr_bps", "settlement_days", "created_at")
        .with_columns(pl.col("mdr_bps").cast(pl.Int64), pl.col("settlement_days").cast(pl.Int64))
    )


def build_accounts(profile: Profile, bins: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(accounts, traits)``, both sorted by account creation time."""
    rng = stream(profile.seed, "accounts")
    n = profile.n_accounts
    window_us = profile.days * US_PER_DAY

    is_new = rng.random(n) < NEW_ACCOUNT_SHARE
    age_days = np.clip(rng.lognormal(np.log(400), 0.9, size=n), 1, 2000)
    created_us = np.where(
        is_new,
        epoch_us(profile.window_start) + rng.random(n) * 0.9 * window_us,
        epoch_us(profile.window_start) - age_days * US_PER_DAY,
    ).astype(np.int64)
    home = rng.choice(HOME_COUNTRIES, size=n, p=HOME_COUNTRY_WEIGHTS)
    activity = rng.lognormal(0.0, 0.8, size=n)
    ticket_mult = rng.lognormal(0.0, 0.4, size=n)

    # Cards: every account owns a primary card, a quarter also a second one, from its home country.
    card_1 = np.empty(n, dtype=object)
    card_2 = np.full(n, None, dtype=object)
    for country in HOME_COUNTRIES:
        pool = bins.filter(pl.col("issuer_country") == country)["card_bin"].to_numpy()
        rows = np.flatnonzero(home == country)
        card_1[rows] = rng.choice(pool, size=len(rows))
        second = rng.choice(pool, size=len(rows))
        has_second = (rng.random(len(rows)) < SECOND_CARD_SHARE) & (second != card_1[rows])
        card_2[rows[has_second]] = second[has_second]

    # Trips: a few accounts spend 3 to 10 days abroad during the window.
    is_traveler = rng.random(n) < TRAVELER_SHARE
    trip_start_us = epoch_us(profile.window_start) + (rng.random(n) * window_us).astype(np.int64)
    trip_end_us = trip_start_us + (rng.uniform(3, 10, size=n) * US_PER_DAY).astype(np.int64)
    trip_country = rng.choice(FOREIGN_IP_COUNTRIES, size=n)
    trip_country = np.where(trip_country == home, "ES", trip_country)

    order = np.argsort(created_us, kind="stable")
    ids = hex_ids(rng, "acc_", 8, n, "account_id")
    accounts = pl.DataFrame(
        {
            "account_id": ids,
            "created_at": ts_series("created_at", created_us[order]),
            "home_country": home[order],
        }
    )
    traits = pl.DataFrame(
        {
            "account_id": ids,
            "activity": activity[order],
            "ticket_mult": ticket_mult[order],
            "card_bin": pl.Series(card_1[order].tolist(), dtype=pl.String),
            "card_bin_2": pl.Series(card_2[order].tolist(), dtype=pl.String),
            "trip_start": ts_series("trip_start", trip_start_us[order]),
            "trip_end": ts_series("trip_end", trip_end_us[order]),
            "trip_country": pl.Series(trip_country[order].tolist(), dtype=pl.String),
            "is_traveler": is_traveler[order],
        }
    )
    # Only travelers have trip dates; everyone else gets nulls.
    traits = traits.with_columns(
        [
            pl.when("is_traveler").then(pl.col(c)).alias(c)
            for c in ("trip_start", "trip_end", "trip_country")
        ]
    ).drop("is_traveler")
    return accounts, traits


def build_universe(profile: Profile) -> Universe:
    bins = build_bins(profile.seed)
    merchants = build_merchants(profile)
    accounts, traits = build_accounts(profile, bins)

    # Devices and terminals share one id space, so a device id is never both.
    rng_ids = stream(profile.seed, "ids")
    rng_shape = stream(profile.seed, "shape")
    n_terminals = rng_shape.integers(1, 5, size=profile.n_merchants)
    n_devices = rng_shape.choice([1, 2, 3], size=profile.n_accounts, p=[0.60, 0.30, 0.10])
    all_ids = hex_ids(rng_ids, "dev_", 10, int(n_terminals.sum() + n_devices.sum()), "device_id")
    terminal_ids = all_ids.head(int(n_terminals.sum()))
    device_ids = all_ids.tail(int(n_devices.sum()))

    terminals = pl.DataFrame(
        {
            "merchant_id": merchants["merchant_id"].gather(
                np.repeat(np.arange(profile.n_merchants), n_terminals)
            ),
            "device_id": terminal_ids,
        }
    )
    devices = pl.DataFrame(
        {
            "account_id": accounts["account_id"].gather(
                np.repeat(np.arange(profile.n_accounts), n_devices)
            ),
            "device_id": device_ids,
        }
    ).with_columns(rank=pl.int_range(pl.len()).over("account_id"))
    return Universe(profile, merchants, accounts, traits, devices, terminals, bins)
