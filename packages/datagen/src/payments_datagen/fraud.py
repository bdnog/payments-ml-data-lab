"""Fraud episodes and the label rules.

Fraud is generated as *episodes*: a burst of related transactions that share a purpose (testing a
stolen card, taking over an account, ...). Every transaction of an episode is labeled fraud.
``docs/fraud-patterns.md`` describes the patterns, the hard negatives that resemble them, and why
the label that a model can train on is *not* the ground truth.

Episode sizes and attributes are drawn with vectorized NumPy calls; nothing loops over rows.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from payments_datagen.catalog import FOREIGN_IP_COUNTRIES, MCC_TABLE
from payments_datagen.rng import stream
from payments_datagen.transactions import CHANNELS, RAW_COLUMNS, draw_installments
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

# Share of the fraud *transactions* that belongs to each pattern.
PATTERN_SHARES = {
    "CARD_TESTING": 0.30,
    "ACCOUNT_TAKEOVER": 0.30,
    "GEO_MISMATCH": 0.25,
    "VELOCITY_BURST": 0.15,
}
STEALTH_SHARE = 0.15  # episodes that hide their signals (domestic IP, no proxy)

# Label rules.
LEGIT_DECLINE_RATE = 0.06
FRAUD_DECLINE_RATE = 0.40
VOID_RATE = 0.03
CHARGEBACK_REPORT_RATE = 0.85
CHARGEBACK_DELAY_DAYS = (7, 45)


@dataclass
class _Context:
    universe: Universe
    rng: np.random.Generator
    created_us: np.ndarray
    home: pl.Series
    window_start_us: int
    window_end_us: int
    account_mean_amount: np.ndarray  # mean legit amount per account (fallback: latent ticket)
    median_ticket: np.ndarray  # per merchant
    online_pool: np.ndarray  # merchant indices usable online
    risk_pool: np.ndarray  # high-risk categories, usable online
    in_person_pool: np.ndarray

    @property
    def n_accounts(self) -> int:
        return self.universe.accounts.height


def _context(universe: Universe, legit: pl.DataFrame) -> _Context:
    profile = universe.profile
    info = universe.merchants.join(MCC_TABLE, on="mcc", how="left", maintain_order="left")
    p_pos = info["p_pos"].to_numpy()
    high_risk = info["high_risk"].to_numpy()
    everyone = np.arange(info.height)

    def pool(mask: np.ndarray) -> np.ndarray:
        found = np.flatnonzero(mask)
        return found if len(found) else everyone

    means = legit.group_by("account_id").agg(pl.col("amount_minor").mean().alias("mean_amount"))
    per_account = universe.accounts.join(
        means, on="account_id", how="left", maintain_order="left"
    ).with_columns(universe.traits["ticket_mult"].alias("ticket_mult"))
    fallback = per_account["ticket_mult"] * 10_000
    mean_amount = per_account["mean_amount"].fill_null(fallback).to_numpy()

    window_start_us = epoch_us(profile.window_start)
    return _Context(
        universe=universe,
        rng=stream(profile.seed, "fraud"),
        created_us=epoch_us_of(universe.accounts["created_at"]),
        home=universe.accounts["home_country"],
        window_start_us=window_start_us,
        window_end_us=window_start_us + profile.days * US_PER_DAY,
        account_mean_amount=mean_amount,
        median_ticket=info["median_ticket_minor"].to_numpy(),
        online_pool=pool(p_pos < 0.9),
        risk_pool=pool(high_risk & (p_pos < 0.9)),
        in_person_pool=pool(p_pos >= 0.5),
    )


def _episode_sizes(rng: np.random.Generator, budget: int, low: int, high: int) -> np.ndarray:
    """Draw episode sizes until they add up to ``budget``; the last one is cut to fit."""
    sizes = rng.integers(low, high + 1, size=budget // low + 1)
    cumulative = np.cumsum(sizes)
    keep = int(np.searchsorted(cumulative, budget, side="left")) + 1
    sizes = sizes[:keep].copy()
    sizes[-1] -= cumulative[keep - 1] - budget
    return sizes


def _episode_starts(
    ctx: _Context, low: np.ndarray, duration_us: int, high: np.ndarray | None = None
) -> np.ndarray:
    """Start of each episode, uniform in ``[low, high]`` and early enough to fit its duration."""
    latest = ctx.window_end_us - duration_us - 1
    high = latest if high is None else np.minimum(high, latest)
    high = np.maximum(high, low)
    return (low + ctx.rng.random(len(low)) * (high - low)).astype(np.int64)


def _foreign_country(ctx: _Context, home: pl.Series) -> pl.Series:
    """A random foreign country for each account (never the account's own)."""
    pool = pl.Series(FOREIGN_IP_COUNTRIES)
    picked = pool.gather(ctx.rng.integers(0, len(pool), size=len(home)))
    return pl.Series(np.where(picked.to_numpy() == home.to_numpy(), "ES", picked.to_numpy()))


def _frame(
    ctx: _Context,
    pattern: str,
    episode: np.ndarray,
    account: np.ndarray,
    occurred: np.ndarray,
    merchant: np.ndarray,
    amount: np.ndarray,
    channel_code: np.ndarray,
    device_id: pl.Series,
    ip_country: pl.Series,
    is_proxy: np.ndarray,
) -> pl.DataFrame:
    universe = ctx.universe
    n = len(account)
    amount = np.clip(np.rint(amount), 50, 5_000_000).astype(np.int64)
    return pl.DataFrame(
        {
            "occurred_at": ts_series("occurred_at", np.minimum(occurred, ctx.window_end_us - 1)),
            "account_id": universe.accounts["account_id"].gather(account),
            "merchant_id": universe.merchants["merchant_id"].gather(merchant),
            "amount_minor": amount,
            "channel": pl.Series(CHANNELS).gather(channel_code),
            "installments": draw_installments(ctx.rng, amount),
            "card_bin": universe.traits["card_bin"].gather(account),  # the stolen card
            "device_id": device_id,
            "ip_country": ip_country,
            "is_proxy": is_proxy,
            "is_fraud": np.ones(n, dtype=bool),
            "fraud_pattern": pl.Series([pattern] * n, dtype=pl.String),
            "scenario": pl.Series(["fraud"] * n, dtype=pl.String),
        }
    ).select(RAW_COLUMNS)


def _fresh_devices(ctx: _Context, episodes: int) -> pl.Series:
    return hex_ids(ctx.rng, "dev_", 10, episodes, "device_id")


def _card_testing(ctx: _Context, budget: int) -> pl.DataFrame:
    """5 to 30 tiny charges in 15 minutes from one device, sometimes ending with a larger one."""
    rng = ctx.rng
    sizes = _episode_sizes(rng, budget, 5, 30)
    n_episodes = len(sizes)
    episode = np.repeat(np.arange(n_episodes), sizes)
    n = len(episode)

    account_e = rng.integers(0, ctx.n_accounts, size=n_episodes)
    low = ctx.created_us[account_e] + US_PER_HOUR
    start = _episode_starts(ctx, np.maximum(low, ctx.window_start_us), 15 * US_PER_MINUTE)
    offset = (rng.random(n) * 15 * US_PER_MINUTE).astype(np.int64)

    amount = rng.integers(50, 501, size=n).astype(np.float64)
    last_in_episode = np.lexsort((offset, episode))[np.cumsum(sizes) - 1]
    ends_big = (rng.random(n_episodes) < 0.30) & (sizes >= 5)
    amount[last_in_episode[ends_big]] = rng.integers(5_000, 60_001, size=int(ends_big.sum()))

    home_e = ctx.home.gather(account_e)
    abroad = rng.random(n_episodes) < 0.70
    ip_e = pl.Series(np.where(abroad, _foreign_country(ctx, home_e).to_numpy(), home_e.to_numpy()))
    return _frame(
        ctx,
        "CARD_TESTING",
        episode,
        account_e[episode],
        start[episode] + offset,
        ctx.online_pool[rng.integers(0, len(ctx.online_pool), size=n)],
        amount,
        np.where(rng.random(n_episodes) < 0.85, 0, 1)[episode],
        _fresh_devices(ctx, n_episodes).gather(episode),
        ip_e.gather(episode),
        (abroad & (rng.random(n_episodes) < 0.5))[episode],
    )


def _account_takeover(ctx: _Context, budget: int) -> pl.DataFrame:
    """1 to 5 purchases in 2 hours from a new device: 3x to 10x the usual ticket, high-risk MCCs."""
    rng = ctx.rng
    sizes = _episode_sizes(rng, budget, 1, 5)
    n_episodes = len(sizes)
    episode = np.repeat(np.arange(n_episodes), sizes)
    n = len(episode)

    old = np.flatnonzero(ctx.created_us <= ctx.window_start_us - 90 * US_PER_DAY)
    old = old if len(old) else np.arange(ctx.n_accounts)
    account_e = old[rng.integers(0, len(old), size=n_episodes)]
    quiet_days = min(3 * US_PER_DAY, (ctx.window_end_us - ctx.window_start_us) // 5)
    low = np.full(n_episodes, ctx.window_start_us + quiet_days, dtype=np.int64)
    start = _episode_starts(ctx, low, 2 * US_PER_HOUR)
    offset = (rng.random(n) * 2 * US_PER_HOUR).astype(np.int64)

    stealth = rng.random(n_episodes) < STEALTH_SHARE
    home_e = ctx.home.gather(account_e)
    ip_e = pl.Series(np.where(stealth, home_e.to_numpy(), _foreign_country(ctx, home_e).to_numpy()))
    proxy_e = ~stealth & (rng.random(n_episodes) < 0.70)
    amount = ctx.account_mean_amount[account_e][episode] * rng.uniform(3, 10, size=n)
    return _frame(
        ctx,
        "ACCOUNT_TAKEOVER",
        episode,
        account_e[episode],
        start[episode] + offset,
        ctx.risk_pool[rng.integers(0, len(ctx.risk_pool), size=n)],
        amount,
        np.where(rng.random(n_episodes) < 0.70, 0, 1)[episode],
        _fresh_devices(ctx, n_episodes).gather(episode),
        ip_e.gather(episode),
        proxy_e[episode],
    )


def _geo_mismatch(ctx: _Context, budget: int) -> pl.DataFrame:
    """A Brazilian card used from abroad through a proxy on an account under 30 days old."""
    rng = ctx.rng
    sizes = _episode_sizes(rng, budget, 1, 4)
    n_episodes = len(sizes)
    episode = np.repeat(np.arange(n_episodes), sizes)
    n = len(episode)

    brazilian = (ctx.home.to_numpy() == "BR") & (
        ctx.universe.traits["card_bin"].is_not_null().to_numpy()
    )
    newcomers = np.flatnonzero(brazilian & (ctx.created_us >= ctx.window_start_us))
    candidates = newcomers if len(newcomers) else np.flatnonzero(brazilian)
    account_e = candidates[rng.integers(0, len(candidates), size=n_episodes)]
    low = np.maximum(ctx.created_us[account_e] + US_PER_HOUR, ctx.window_start_us)
    start = _episode_starts(
        ctx, low, 3 * US_PER_HOUR, high=ctx.created_us[account_e] + 29 * US_PER_DAY
    )
    offset = (rng.random(n) * 3 * US_PER_HOUR).astype(np.int64)

    stealth = rng.random(n_episodes) < STEALTH_SHARE
    home_e = ctx.home.gather(account_e)
    ip_e = pl.Series(np.where(stealth, home_e.to_numpy(), _foreign_country(ctx, home_e).to_numpy()))
    merchant = ctx.online_pool[rng.integers(0, len(ctx.online_pool), size=n)]
    amount = ctx.median_ticket[merchant] * rng.uniform(1.0, 3.0, size=n)  # moderate
    return _frame(
        ctx,
        "GEO_MISMATCH",
        episode,
        account_e[episode],
        start[episode] + offset,
        merchant,
        amount,
        np.where(rng.random(n_episodes) < 0.60, 0, 1)[episode],
        _fresh_devices(ctx, n_episodes).gather(episode),
        ip_e.gather(episode),
        (~stealth)[episode],
    )


def _velocity_burst(ctx: _Context, budget: int) -> pl.DataFrame:
    """8 to 20 ordinary-looking purchases in an hour, spread over many merchants."""
    rng = ctx.rng
    sizes = _episode_sizes(rng, budget, 8, 20)
    n_episodes = len(sizes)
    episode = np.repeat(np.arange(n_episodes), sizes)
    n = len(episode)

    account_e = rng.integers(0, ctx.n_accounts, size=n_episodes)
    low = np.maximum(ctx.created_us[account_e] + US_PER_HOUR, ctx.window_start_us)
    start = _episode_starts(ctx, low, US_PER_HOUR)
    offset = (rng.random(n) * US_PER_HOUR).astype(np.int64)

    in_person = rng.random(n) < 0.25  # a cloned card also gets used in shops
    merchant = np.where(
        in_person,
        ctx.in_person_pool[rng.integers(0, len(ctx.in_person_pool), size=n)],
        ctx.online_pool[rng.integers(0, len(ctx.online_pool), size=n)],
    )
    ticket = ctx.universe.traits["ticket_mult"].to_numpy()[account_e][episode]
    amount = ctx.median_ticket[merchant] * ticket * rng.lognormal(0.0, 0.5, size=n)

    home_e = ctx.home.gather(account_e)
    abroad = rng.random(n_episodes) < 0.20
    ip_e = pl.Series(np.where(abroad, _foreign_country(ctx, home_e).to_numpy(), home_e.to_numpy()))
    channel = np.where(in_person, 2, np.where(rng.random(n) < 0.55, 0, 1))
    return _frame(
        ctx,
        "VELOCITY_BURST",
        episode,
        account_e[episode],
        start[episode] + offset,
        merchant,
        amount,
        channel,
        _fresh_devices(ctx, n_episodes).gather(episode),
        ip_e.gather(episode),
        (abroad & (rng.random(n_episodes) < 0.7))[episode],
    )


_GENERATORS = {
    "CARD_TESTING": _card_testing,
    "ACCOUNT_TAKEOVER": _account_takeover,
    "GEO_MISMATCH": _geo_mismatch,
    "VELOCITY_BURST": _velocity_burst,
}


def fraud_budgets(profile_target: int, fraud_rate: float) -> dict[str, int]:
    """Fraud transactions per pattern. Every pattern gets at least one when fraud is enabled."""
    if fraud_rate <= 0:
        return {name: 0 for name in PATTERN_SHARES}
    total = round(profile_target * fraud_rate)
    return {name: max(1, round(total * share)) for name, share in PATTERN_SHARES.items()}


def generate_fraud(universe: Universe, legit: pl.DataFrame) -> pl.DataFrame:
    """Raw fraudulent rows (same columns as ``generate_legit``), every one labeled ``is_fraud``."""
    profile = universe.profile
    budgets = fraud_budgets(profile.target_transactions, profile.fraud_rate)
    ctx = _context(universe, legit)
    frames = [_GENERATORS[name](ctx, budget) for name, budget in budgets.items() if budget > 0]
    if not frames:
        return legit.clear()
    return pl.concat(frames)


def apply_outcomes(universe: Universe, txns: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply the label rules to assembled transactions.

    Returns ``(transactions, outcomes)``. ``transactions`` gets ``chargeback_reported_at``.
    ``outcomes`` (transaction_id, declined, voided) is *hidden* state: declines never become
    payments and are not exposed in ``transactions``, because they would leak the label.

    * About 6% of legitimate and 40% of fraudulent transactions are declined upstream.
    * About 3% of approved ones are voided before capture (no money moves).
    * Only approved, captured fraud can be charged back, and only about 85% of it is reported, 7 to
      45 days later: possibly after the end of the data window, and sometimes never.
    """
    rng = stream(universe.profile.seed, "outcomes")
    n = txns.height
    is_fraud = txns["is_fraud"].to_numpy()

    declined = rng.random(n) < np.where(is_fraud, FRAUD_DECLINE_RATE, LEGIT_DECLINE_RATE)
    voided = ~declined & (rng.random(n) < VOID_RATE)
    reported = is_fraud & ~declined & ~voided & (rng.random(n) < CHARGEBACK_REPORT_RATE)
    delay_days = rng.uniform(*CHARGEBACK_DELAY_DAYS, size=n)
    occurred = epoch_us_of(txns["occurred_at"])
    reported_us = occurred + (delay_days * US_PER_DAY).astype(np.int64)

    reported_at = pl.when(pl.Series(reported)).then(ts_series("x", reported_us)).otherwise(None)
    labeled = txns.with_columns(reported_at.alias("chargeback_reported_at"))
    outcomes = pl.DataFrame(
        {"transaction_id": txns["transaction_id"], "declined": declined, "voided": voided}
    )
    return labeled, outcomes
