# 0006. v1 scoring takes precomputed behavior features; v2 computes them server-side

- Status: accepted
- Date: 2026-11-11

## Context

Several of the most useful fraud features are *velocity* features: how many transactions this account
made in the last hour or day, the average of the last 30 days, the time since the previous one. Serving
them needs a store of recent history per account, with low-latency reads, which is real infrastructure
(Redis or Postgres, state, backfill). Phase 3 should be about the model, not about that store.

The real risk is *training/serving skew*: if the training features are computed one way and the
serving features another, offline results will not hold in production.

## Decision

- **v1 of the scoring contract: the caller sends the `behavior` block** (`txn_count_1h`,
  `txn_count_24h`, `avg_amount_30d_minor`, `seconds_since_last_txn`) already computed. The service
  scores what it is given.
- The data generator computes those features **point-in-time**, with one precise definition: for a
  transaction at time `t` and an account, only the account's earlier transactions count; the
  windows are half-open `[t - window, t)`; the current transaction and anything later is excluded;
  the 30-day mean is an integer rounded half up. A test recomputes them naively row by row and another
  proves that removing the future does not change the past.
- **v2 will compute `behavior` server-side** from history. It must reproduce exactly the definition
  above; the generated data is the reference to replay against to detect skew.

## Consequences

- Phase 3 can train and serve a model without building a feature store.
- v1 trusts the caller: a client that sends wrong velocity numbers gets wrong scores. This is
  acceptable for a lab and unacceptable for production, which is the reason v2 exists.
- Moving to v2 is a compatible change: `behavior` can become optional and be computed when absent.
- The feature definitions are now a contract in their own right (see `contracts/data/transactions.yaml`
  and `docs/data-model.md`); changing a window or a rounding rule is a breaking change for any model
  trained on the old data.
