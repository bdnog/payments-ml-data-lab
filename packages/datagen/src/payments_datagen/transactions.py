"""Legitimate traffic, and the assembly of the final ``transactions`` dataset.

Two stages live here:

* ``generate_legit`` draws legitimate traffic from the latent traits of the universe, including
  *hard negatives*: legitimate behavior that looks suspicious (see ``docs/fraud-patterns.md``).
* ``assemble_transactions`` merges legitimate and fraudulent rows, applies the in-person (pos)
  rules, computes the **point-in-time** features and hands out ids in time order.

Point-in-time means: a feature of a transaction only uses data strictly before it. The current
transaction is excluded, and so is anything after it. This is exactly what a scoring service can
know at request time, and getting it wrong is the classic source of training/serving skew.
"""

import numpy as np
import polars as pl

from payments_datagen.catalog import FOREIGN_IP_COUNTRIES, MCC_TABLE
from payments_datagen.rng import stream
from payments_datagen.universe import Universe
from payments_datagen.util import (
    US_PER_DAY,
    US_PER_HOUR,
    US_PER_MINUTE,
    epoch_us,
    epoch_us_of,
    hex_ids,
    ts_series,
)

CHANNELS = ["web", "app", "pos"]
ONLINE_SPLIT_WEB = 0.55  # of non-pos purchases
WEEKEND_FACTOR = 1.15
# Relative traffic per hour of day, UTC. Quiet at night, an evening peak around 21-23 UTC.
HOUR_WEIGHTS = np.array(
    [0.5, 0.4, 0.3, 0.3, 0.3, 0.4, 0.6, 0.9, 1.1, 1.2, 1.2, 1.3,
     1.3, 1.3, 1.3, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.2, 1.4]
)  # fmt: skip
HOUR_P = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()
PREFERRED_MERCHANTS = 5
PREFERRED_SLOT_P = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
PREFERRED_SHARE = 0.70
SECOND_CARD_USE = 0.20
PRIMARY_DEVICE_SHARE = 0.70
LEGIT_PROXY_RATE = 0.02  # ordinary people use VPNs
LEGIT_FOREIGN_IP_RATE = 0.01  # noise: IP geolocation is imperfect
# Hard negatives
BIG_TICKET_RATE = 0.006
BURST_SEED_SHARE = 0.005
BURST_MEAN_EXTRA = 3.5  # extra transactions per burst seed (2 to 5)
SCENARIOS = ["normal", "traveler", "big_ticket", "burst"]

RAW_COLUMNS = [
    "occurred_at",
    "account_id",
    "merchant_id",
    "amount_minor",
    "channel",
    "installments",
    "card_bin",
    "device_id",
    "ip_country",
    "is_proxy",
    "is_fraud",
    "fraud_pattern",
    "scenario",
]


def _pick(mask: np.ndarray, if_true: pl.Series, if_false: pl.Series) -> pl.Series:
    return if_true.zip_with(pl.Series(mask), if_false)


def draw_installments(rng: np.random.Generator, amount: np.ndarray) -> np.ndarray:
    """Bigger tickets are more often split into installments (2 to 12, shorter more likely)."""
    p_split = np.clip((amount - 8_000) / 200_000, 0.0, 0.7)
    options = np.arange(2, 13)
    chosen = rng.choice(options, size=len(amount), p=(1 / options) / (1 / options).sum())
    return np.where(rng.random(len(amount)) < p_split, chosen, 1).astype(np.int64)


def generate_legit(universe: Universe) -> pl.DataFrame:
    """Legitimate transactions, in no particular order, with raw (pre-feature) columns."""
    profile = universe.profile
    rng = stream(profile.seed, "legit")
    n_days = profile.days
    window_start_us = epoch_us(profile.window_start)
    window_end_us = window_start_us + n_days * US_PER_DAY
    accounts, traits, merchants = universe.accounts, universe.traits, universe.merchants
    n_accounts, n_merchants = accounts.height, merchants.height

    # --- when: each account is active from its creation (or the window start) to the window end.
    created_us = epoch_us_of(accounts["created_at"])
    active_from_day = (np.maximum(created_us, window_start_us) - window_start_us) / US_PER_DAY
    weekday0 = profile.start_date.weekday()  # Monday is 0
    day_weight = np.where((weekday0 + np.arange(n_days)) % 7 >= 5, WEEKEND_FACTOR, 1.0)
    cum_weight = np.concatenate([[0.0], np.cumsum(day_weight)])  # traffic mass up to each day start
    whole = np.minimum(np.floor(active_from_day).astype(int), n_days - 1)
    mass_before = cum_weight[whole] + day_weight[whole] * (active_from_day - whole)
    mass = cum_weight[-1] - mass_before  # traffic mass in the account's active period

    n_target = profile.target_transactions * (1 - profile.fraud_rate)
    n_target /= 1 + BURST_SEED_SHARE * BURST_MEAN_EXTRA
    raw_rate = traits["activity"].to_numpy() * mass
    counts = rng.poisson(raw_rate * n_target / raw_rate.sum())
    acct = np.repeat(np.arange(n_accounts), counts)
    n = len(acct)

    # Inverse-CDF sampling of the day (so weekends weigh a bit more), then the hour of day.
    u = mass_before[acct] + rng.random(n) * mass[acct]
    day = np.minimum(np.searchsorted(cum_weight, u, side="right") - 1, n_days - 1)
    hour = rng.choice(24, size=n, p=HOUR_P)
    occurred = (
        window_start_us + day * US_PER_DAY + hour * US_PER_HOUR + rng.integers(0, US_PER_HOUR, n)
    )
    earliest = created_us[acct] + rng.integers(US_PER_MINUTE, 6 * US_PER_HOUR, size=n)
    occurred = np.minimum(np.maximum(occurred, earliest), window_end_us - 1)

    # --- where: preferred merchants (Zipf-like popularity) most of the time.
    mcc_info = merchants.join(MCC_TABLE, on="mcc", how="left", maintain_order="left")
    p_pos = mcc_info["p_pos"].to_numpy()
    median_ticket = mcc_info["median_ticket_minor"].to_numpy()
    popularity = 1.0 / (rng.permutation(n_merchants) + 1.0)
    popularity /= popularity.sum()
    k = min(PREFERRED_MERCHANTS, n_merchants)
    scores = np.log(popularity)[None, :] + rng.gumbel(size=(n_accounts, n_merchants))
    preferred = np.argsort(-scores, axis=1)[:, :k]
    slot_p = PREFERRED_SLOT_P[:k] / PREFERRED_SLOT_P[:k].sum()
    slot = rng.choice(k, size=n, p=slot_p)
    merchant = np.where(
        rng.random(n) < PREFERRED_SHARE,
        preferred[acct, slot],
        rng.choice(n_merchants, size=n, p=popularity),
    )

    # --- how: channel follows the merchant category; the amount follows category and account.
    is_pos = rng.random(n) < p_pos[merchant]
    channel_code = np.where(is_pos, 2, np.where(rng.random(n) < ONLINE_SPLIT_WEB, 0, 1))
    ticket = median_ticket[merchant] * traits["ticket_mult"].to_numpy()[acct]
    amount = np.rint(ticket * rng.lognormal(0.0, 0.6, size=n))
    scenario = np.zeros(n, dtype=np.int64)

    # Hard negative: a big-ticket purchase (4x to 8x the usual) on the account's primary device.
    big = rng.random(n) < BIG_TICKET_RATE
    amount = np.where(big, amount * rng.uniform(4, 8, size=n), amount)
    scenario[big] = 2
    amount = np.clip(np.rint(amount), 50, 5_000_000).astype(np.int64)

    # --- who: card and device.
    has_second = traits["card_bin_2"].is_not_null().to_numpy()
    use_second = (rng.random(n) < SECOND_CARD_USE) & has_second[acct]
    card_bin = _pick(use_second, traits["card_bin_2"].gather(acct), traits["card_bin"].gather(acct))

    n_devices = universe.devices.group_by("account_id", maintain_order=True).len()["len"].to_numpy()
    first_device = np.cumsum(n_devices) - n_devices
    rank = np.where(
        rng.random(n) < PRIMARY_DEVICE_SHARE, 0, (rng.random(n) * n_devices[acct]).astype(np.int64)
    )
    rank[big] = 0
    device_id = universe.devices["device_id"].gather(first_device[acct] + rank)

    # --- network signals: home country mostly; travelers abroad show a foreign IP without a proxy.
    home = accounts["home_country"].gather(acct)
    foreign_pool = pl.Series(FOREIGN_IP_COUNTRIES)
    noise_country = foreign_pool.gather(rng.integers(0, len(FOREIGN_IP_COUNTRIES), size=n))
    ip_country = _pick(rng.random(n) < LEGIT_FOREIGN_IP_RATE, noise_country, home)
    is_proxy = rng.random(n) < LEGIT_PROXY_RATE
    trip_start = traits["trip_start"].dt.epoch("us").fill_null(-1).to_numpy()
    trip_end = traits["trip_end"].dt.epoch("us").fill_null(-1).to_numpy()
    on_trip = (trip_start[acct] >= 0) & (occurred >= trip_start[acct]) & (occurred < trip_end[acct])
    on_trip &= ~is_pos
    ip_country = _pick(on_trip, traits["trip_country"].gather(acct), ip_country)
    is_proxy &= ~on_trip
    scenario[on_trip] = 1

    frame = pl.DataFrame(
        {
            "occurred_at": ts_series("occurred_at", occurred),
            "account_id": accounts["account_id"].gather(acct),
            "merchant_id": merchants["merchant_id"].gather(merchant),
            "amount_minor": amount,
            "channel": pl.Series(CHANNELS).gather(channel_code),
            "installments": draw_installments(rng, amount),
            "card_bin": card_bin,
            "device_id": device_id,
            "ip_country": ip_country,
            "is_proxy": is_proxy,
            "scenario": pl.Series(SCENARIOS).gather(scenario),
        }
    )

    # Hard negative: a legitimate burst. 3 to 6 purchases within 30 minutes at one merchant,
    # on a known device, with ordinary amounts.
    seeds = np.flatnonzero(rng.random(n) < BURST_SEED_SHARE)
    extras_per_seed = rng.integers(2, 6, size=len(seeds))
    source = np.repeat(seeds, extras_per_seed)
    extras = frame.gather(source)
    delay = rng.integers(1, 30 * US_PER_MINUTE, size=len(source))
    extra_amount = np.clip(
        np.rint(amount[source] * rng.lognormal(0.0, 0.3, len(source))), 50, 5_000_000
    )
    extras = extras.with_columns(
        ts_series("occurred_at", occurred[source] + delay),
        pl.Series("amount_minor", extra_amount.astype(np.int64)),
        pl.lit(1, pl.Int64).alias("installments"),
        pl.lit("burst").alias("scenario"),
    )
    extras = extras.filter(pl.col("occurred_at") < profile.window_end)
    burst_seed = np.zeros(n, dtype=bool)
    burst_seed[seeds] = True
    frame = frame.with_columns(
        pl.when(pl.Series(burst_seed))
        .then(pl.lit("burst"))
        .otherwise(pl.col("scenario"))
        .alias("scenario")
    )
    frame = pl.concat([frame, extras])
    return frame.with_columns(is_fraud=pl.lit(False), fraud_pattern=pl.lit(None, pl.String)).select(
        RAW_COLUMNS
    )


def add_point_in_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add the behavioral features. ``df`` needs ``account_id``, ``occurred_at``, ``amount_minor``,
    ``device_id``, ``card_bin``, ``merchant_id`` and ``created_at``; the result is sorted by
    account and time.

    The windows are half-open, ``[t - window, t)``, so the current row is never part of its own
    features. The frame is processed per account, in time order.
    """
    order = ["account_id", "occurred_at", "merchant_id", "amount_minor", "device_id"]
    by_account = pl.col("account_id")

    def rolling(column: str, window: str) -> pl.Expr:
        return (
            pl.col(column)
            .rolling_sum_by("occurred_at", window_size=window, closed="left", min_samples=0)
            .over(by_account)
        )

    out = df.sort(order).with_columns(ones=pl.lit(1, pl.Int64))
    out = out.with_columns(
        txn_count_1h=rolling("ones", "1h"),
        txn_count_24h=rolling("ones", "24h"),
        _count_30d=rolling("ones", "30d"),
        _sum_30d=rolling("amount_minor", "30d"),
        seconds_since_last_txn=(
            (
                pl.col("occurred_at") - pl.col("occurred_at").shift(1).over(by_account)
            ).dt.total_seconds()
        ),
        device_is_new_for_account=pl.int_range(pl.len()).over(["account_id", "device_id"]) == 0,
        card_is_new_for_merchant=pl.int_range(pl.len()).over(
            ["account_id", "card_bin", "merchant_id"]
        )
        == 0,
        account_age_days=(pl.col("occurred_at") - pl.col("created_at")).dt.total_days(),
    )
    # Mean of the prior 30 days, in integer arithmetic and rounded half up: no floats involved.
    return out.with_columns(
        avg_amount_30d_minor=pl.when(pl.col("_count_30d") > 0)
        .then((2 * pl.col("_sum_30d") + pl.col("_count_30d")) // (2 * pl.col("_count_30d")))
        .otherwise(None)
        .cast(pl.Int64)
    ).drop("ones", "_count_30d", "_sum_30d")


def _apply_pos_rules(raw: pl.DataFrame, universe: Universe) -> pl.DataFrame:
    """In-person purchases have a terminal instead of a device, and no IP signals."""
    rng = stream(universe.profile.seed, "normalize")
    merchants = universe.merchants.with_row_index("m_idx").select("merchant_id", "m_idx")
    terminals = universe.terminals.group_by("merchant_id", maintain_order=True).len()
    terminals = merchants.join(terminals, on="merchant_id", how="left", maintain_order="left")
    n_terminals = terminals["len"].to_numpy()
    first_terminal = np.cumsum(n_terminals) - n_terminals

    m_idx = raw.join(merchants, on="merchant_id", how="left", maintain_order="left")[
        "m_idx"
    ].to_numpy()
    pick = first_terminal[m_idx] + (rng.random(raw.height) * n_terminals[m_idx]).astype(np.int64)
    terminal = universe.terminals["device_id"].gather(pick)
    is_pos = (raw["channel"] == "pos").to_numpy()
    return raw.with_columns(
        device_id=_pick(is_pos, terminal, raw["device_id"]),
        ip_country=pl.when(pl.Series(is_pos)).then(None).otherwise(pl.col("ip_country")),
        is_proxy=pl.col("is_proxy") & ~pl.Series(is_pos),
    )


def assemble_transactions(
    universe: Universe, legit: pl.DataFrame, fraud: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Merge raw legitimate and fraudulent rows into the ``transactions`` contract.

    ``chargeback_reported_at`` is left null here; the label rules fill it in (``fraud.py``).
    """
    raw = pl.concat([legit, fraud]) if fraud is not None else legit
    raw = _apply_pos_rules(raw, universe)
    raw = (
        raw.join(
            universe.merchants.select("merchant_id", "mcc"),
            on="merchant_id",
            how="left",
            maintain_order="left",
        )
        .join(universe.bins, on="card_bin", how="left", maintain_order="left")
        .join(
            universe.accounts.select("account_id", "created_at"),
            on="account_id",
            how="left",
            maintain_order="left",
        )
    )
    featured = add_point_in_time_features(raw)
    # The device of an in-person purchase is a terminal, so "new device" carries no meaning there.
    featured = featured.with_columns(
        device_is_new_for_account=pl.col("device_is_new_for_account") & (pl.col("channel") != "pos")
    )

    final = featured.sort(["occurred_at", "account_id", "merchant_id", "amount_minor", "device_id"])
    ids = hex_ids(
        stream(universe.profile.seed, "txn_ids"), "txn_", 12, final.height, "transaction_id"
    )
    return final.with_columns(
        ids,
        currency=pl.lit("BRL"),
        chargeback_reported_at=pl.lit(None, pl.Datetime("us", "UTC")),
        installments=pl.col("installments").cast(pl.Int64),
    ).select(
        "transaction_id",
        "occurred_at",
        "account_id",
        "merchant_id",
        "amount_minor",
        "currency",
        "mcc",
        "channel",
        "installments",
        "card_bin",
        "issuer_country",
        "device_id",
        "ip_country",
        "is_proxy",
        "device_is_new_for_account",
        "card_is_new_for_merchant",
        "account_age_days",
        "txn_count_1h",
        "txn_count_24h",
        "avg_amount_30d_minor",
        "seconds_since_last_txn",
        "is_fraud",
        "fraud_pattern",
        "chargeback_reported_at",
    )
