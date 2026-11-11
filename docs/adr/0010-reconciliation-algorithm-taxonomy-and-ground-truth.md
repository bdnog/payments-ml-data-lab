# 0010. Reconciliation: one pass, a complete taxonomy, and a ground truth kept apart

- Status: accepted
- Date: 2026-11-11

## Context

The settlement service compares what the platform expects the PSP to settle (from the ledger) with
what the PSP says it settled. The differences must be classified so someone can act on them, and the
classification must be *demonstrably right*, not just plausible. The generator already knows every
discrepancy it injected (`injected_breaks`). The risk is that a pipeline that can see the answer
proves nothing.

## Decision

**Run semantics.** A run for date `D` reconciles everything *due on `D`* (`settlement_date = D`).
The pipeline functions are date-agnostic: they group by `settlement_date`, so one pass handles many
dates (`run-range`), and `run --date D` is a list of one. The answer for a date does not depend on its
neighbours (a test checks it).

**Matching key** `(settlement_date, payment_id, record_type)`, where the PSP's `merchant_reference` is
the payment id. A PSP record with a null reference gets a surrogate key nobody else has, so orphans
are never mistaken for duplicates of each other.

**A complete taxonomy**, with no "unclassified" state: `DUPLICATE_IN_PSP`, `MISSING_IN_PSP`,
`CHARGEBACK_NOT_IN_LEDGER`, `MISSING_IN_LEDGER`, `AMOUNT_MISMATCH`, `FEE_MISMATCH`, with the amounts and
sign convention (PSP minus ledger) in `docs/settlement.md`. `AMOUNT_MISMATCH` takes precedence over
`FEE_MISMATCH`: a wrong amount makes the fee wrong too, and the amount is the cause.

**Duplicates and pairing.** The PSP records of a key are numbered by `psp_record_id`; the ledger
expects `n` of them (one for a SALE). The first `max(n, 1)` are originals, the rest are duplicates; the
originals are then paired with the expected records by rank of amount. For a SALE this is exactly "the
lowest id is the original". The refinement exists because a payment can have two partial refunds that
settle on the same day (133 of 20,806 refunds in the `full` dataset): with a naive rule the second
legitimate refund would be flagged as a duplicate. The generator needed no change, because it gives
the extra copy of a duplicated sale the *higher* id.

**Generator is the reference for ambiguous conventions**, and the pipeline mirrors it: a break's
`payment_id` is the PSP reference only if it is a known payment (else null, as for orphans);
`psp_record_id` is null when the PSP has no record; refunds and chargebacks settle the day after the
event. These are written down in `docs/settlement.md`.

**The ground truth is kept separate.** `settlement/batch/verification.py` is the only production
module that reads `injected_breaks`; the pipeline never imports it. A test deletes the file from a
copy of the input and the pipeline output is identical; another checks statically (an AST scan of
string literals) that no other module names the file. Verification matches on
`(break_type, payment_id, psp_record_id)` with nulls equal *and* compares the difference, and reports
precision and recall per type; it exits non-zero unless all are 100%. The vocabulary is a single
source: the domain enums, which tests compare with the data-contract `accepted_values`.

**Money is integer minor units**, and the fee rule is `(gross * mdr_bps + 5000) // 10000`, implemented
twice (a Python function in `domain/`, a Polars expression in `batch/`) with a parity test over a grid.

## Consequences

- The result is checked against an independent answer key: 100% precision and recall on the sample,
  on three other seeds, and on the `full` dataset (9,072 breaks).
- Because the generator and the pipeline share conventions, a change on one side shows up as a
  verification failure. That is the point, and also a coupling: the conventions are documented rather
  than hidden.
- Known limitations are explicit: no cross-date matching (one real gap shows as two breaks), pairing
  by rank for same-day multiple refunds, a single currency, static merchant terms.
- With real data there is no `injected_breaks`; verification would be replaced by a review of the
  breaks themselves. The pipeline does not depend on it, which is why it can be run on any input.
