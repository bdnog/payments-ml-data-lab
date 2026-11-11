# 0003. `payments-datagen` is a workspace package, and risk and settlement share one universe

- Status: accepted
- Date: 2026-11-11

## Context

Phase 2 needs synthetic data for both services: labeled transactions for fraud scoring, and ledger
plus PSP files for reconciliation. Two questions came up.

1. **Where does the generator live?** As scripts under `data/generators/`, or as a package.
2. **Is the data for the two services generated independently, or together?**

Scripts are hard to test and cannot be imported by the tests that need data. Independent datasets
would have unrelated ids, so nothing could be said about, for example, "this fraudulent transaction
was charged back and shows up in the PSP file".

## Decision

- The generator is a uv workspace member, `packages/datagen` (distribution `payments-datagen`,
  import `payments_datagen`), with a CLI: `python -m payments_datagen generate|validate`. The
  `data/` directory only holds *outputs* (`data/samples/` committed, `data/out/` ignored).
- It generates **one universe** from one seed: accounts and merchants, then transactions with fraud
  labels, then the settlement data *derived from those same transactions* (approved transactions
  become payments, payments get refunds and chargebacks, and the PSP file is produced from the
  ledger). Risk and settlement therefore share ids and labels.
- Determinism is a feature: a fixed seed, no wall-clock time, independent named random streams per
  stage (adding a draw in one stage does not change the numbers of another), ids and rows in a
  deterministic order. The manifest records seed, parameters and realized rates, with no timestamp.
- Hidden state (which transactions were declined or voided) is used to derive the settlement data
  but is never written, because exposing it would leak the label.
- `payments-datagen` depends on `payments-shared` and validates its output with the same data-contract
  code the services will use.

## Consequences

- The generator is unit-tested like any other code, and the tests can build a universe in well under
  a second. The `full` profile (500k transactions) takes a couple of seconds.
- Cross-dataset facts can be asserted: the invariants check that every foreign key resolves, that
  fraud chargebacks equal the labels, and that every difference between ledger and PSP file is an
  injected break.
- All data comes from one stream of decisions, so changing one stage's logic changes the numbers of
  the stages after it. Regenerating is cheap, but the committed sample must be refreshed
  (`make data-sample`) when generation changes, and a library upgrade can change the random stream.
  For that reason tests validate the committed sample but never compare it with a regeneration.
- `payments-shared` now depends on Polars and PyYAML (the contract validator needs them), so both
  service images include Polars even though `risk-scoring-api` does not use it yet.
