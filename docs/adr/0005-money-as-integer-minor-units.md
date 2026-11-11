# 0005. Money is an integer in minor units

- Status: accepted
- Date: 2026-11-11

## Context

Payments code that stores money as floating point eventually loses a cent: `0.1 + 0.2 != 0.3`, and
sums of many rounded values drift. Reconciliation is exactly the job of finding cents that do not
add up, so a representation that creates its own discrepancies would make the whole exercise
meaningless. Fees are percentages of amounts, so rounding has to be defined too.

## Decision

- Every amount, everywhere (datasets, APIs, code, tests), is an **integer in minor units** (for BRL,
  centavos). Columns and fields end in `_minor`; the type is `int64` in Parquet and `int` in the API.
  There are no floats or decimals for money.
- **Fees are computed with integer arithmetic and a documented rounding rule: half up.**
  `fee = (gross * mdr_bps + 5000) // 10000`, which is `round_half_up(gross * mdr_bps / 10000)`.
  Signs are explicit: PSP `gross_amount_minor` is always positive and the direction lives in
  `net_amount_minor` (SALE `gross - fee`, REFUND `-gross`, CHARGEBACK `-(gross + fee)`).
- The `currency` column exists on every money dataset but only `BRL` is accepted for now. Supporting
  another currency means adding its minor-unit exponent (BRL has 2, JPY has 0) instead of assuming 2.
- Statistical summaries such as `avg_amount_30d_minor` are integers too, rounded half up from integer
  sums, so features are reproducible bit for bit.

## Consequences

- Sums, differences and comparisons are exact; a reconciliation difference is a real difference.
- Presentation has to divide by 100; anything shown to a person is formatted at the edge.
- JSON numbers are safe up to 2^53, far above any amount in this system (the largest ticket is
  50,000.00 BRL). Aggregates stay inside `int64`.
- A model that wants a roughly log-scaled amount computes it from the integer; the integer stays the
  record of truth.
