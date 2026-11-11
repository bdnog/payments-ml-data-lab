# settlement-service: rules, reconciliation and marts

> **All data is synthetic.** The rules below are the ones the data generator
> (`packages/datagen/src/payments_datagen/settlement.py`) encodes. When a convention is ambiguous the
> generator is the reference for what the data means, and the pipeline mirrors it.

This page is written in the order the work was done: first the rules read out of the generator, then
(as the pipeline grows) the reconciliation, the marts and the run lifecycle.

## Rules extracted from the generator

### Settlement dates

| Record | Settles on | Where it comes from |
| --- | --- | --- |
| `SALE` | UTC calendar date of `captured_at` + the merchant's `settlement_days` (1, 2 or 30) | `payments` (status `CAPTURED`) joined to `merchants` |
| `REFUND` | UTC date of the refund's `created_at` + 1 day | `refunds` |
| `CHARGEBACK` | UTC date of the chargeback's `opened_at` + 1 day | `chargebacks` |

Only the *date* of the timestamp matters: 23:59:59 and 00:00:01 UTC are different days. The Phase 2
text (`psp_settlement` contract and `docs/data-model.md`) already says this; the generator agrees.
Voided payments (`VOIDED`) never settle: they moved no money.

### Fee

The contracted SALE fee is `round_half_up(gross * mdr_bps / 10000)`, computed as
`(gross * mdr_bps + 5000) // 10000` in integers. A gross of 1050 at 100 bps is exactly 10.5, so the
fee is 11; 1049 gives 10. REFUND records carry no fee. A CHARGEBACK record carries a flat fee of 1500
that the PSP charges the platform. The merchant's terms (`mdr_bps`, `settlement_days`) are static.

### Signs

The PSP file stores `gross_amount_minor` as a positive magnitude and puts the sign in
`net_amount_minor`: `SALE` net = gross - fee, `REFUND` net = -gross, `CHARGEBACK` net =
-(gross + fee). `diff_minor` in a break is always **PSP minus ledger**.

### Due-date cutoff

The PSP file has no record settling after the last day of the data window, and a captured payment that
is not due yet is *not* a break. The pipeline reconciles only the dates it is asked to, and "due on D"
means `settlement_date = D`, so it never compares something before its due date. For a dataset, the
dates to run are those the PSP file covers (`run-range --all` reads them from the file).

### References

The PSP's `merchant_reference` is the `payment_id` the merchant sent. It is null for some orphan
records and, for others, a well-formed id that no payment owns. The matching key is
`(settlement_date, payment_id, record_type)`.

### Conventions the generator fixes (and the pipeline mirrors)

- **Duplicates.** The extra copy of a duplicated SALE gets the *higher* `psp_record_id`: ids are
  handed out after sorting and the original sorts first. So "the lowest `psp_record_id` is the
  original, the others are duplicates" reproduces the ground truth.
- **`payment_id` of a break.** The ground truth has it null for `MISSING_IN_LEDGER` (the PSP record
  has no payment), the payment id for every other type. The pipeline sets it to the PSP reference
  *only if that reference is a known payment*, and to null otherwise, which gives the same answer.
- **`psp_record_id` of a break.** Null for `MISSING_IN_PSP` (nothing to point at); for
  `DUPLICATE_IN_PSP` it is the *duplicate's* id.
- **Several refunds of one payment.** A payment may have two partial refunds, and they can fall on
  the same settlement date (133 of 20,806 refunds in the `full` dataset). The key
  `(date, payment, type)` then has two ledger records and two PSP records, and the PSP file has no
  refund id to tell them apart. The pipeline pairs them by rank (see the reconciliation section)
  instead of calling the second PSP refund a duplicate. This is the one place where the stated rule
  "more than one PSP record per key means duplicates" needed refining; no generator change is needed.

### Differences from the Phase 2 text

- The mart contract `daily_settlement_by_merchant` was described as "aggregated from the PSP
  records". The Phase 4 decision is that it is the **ledger-expected payout** ("what we owe the
  merchant"): the description is corrected. Its `chargebacks_minor` is the chargeback *amounts* only;
  the flat 1500 fee is absorbed by the platform and not deducted from the merchant. The description
  said "plus their flat fees": corrected.
- The `date` of `GET /v1/reconciliation/breaks` is the run date, which equals the settlement date of
  every row of that run.
