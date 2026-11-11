# 0008. Label maturity, temporal split and selection bias

- Status: accepted
- Date: 2026-11-11

## Context

In production nobody knows whether a transaction was fraud. What exists is a report: a cardholder
disputes a payment and a chargeback arrives some days later (here 7 to 45 days; about 85% of approved
fraud is reported at all). A model trained on "reported so far" learns that recent transactions are
clean simply because their reports have not arrived. Separately, only *approved* transactions can be
charged back, so the training data is shaped by whatever decided before the model.

## Decision

**Label.** `label = chargeback_reported_at IS NOT NULL AND chargeback_reported_at <= as_of`. The
ground truth `is_fraud` is never a training label; it is only used in evaluation reports.

**Maturity guard.** Training labels are read at `as_of = train_end + label_maturity_days`, with
`label_maturity_days = 45`, the longest delay the generator produces. `training_label` raises
`ImmatureLabelsError` for any `as_of - train_end < label_maturity_days`, so an immature label set
cannot be used by accident. A test shows that reading the labels at `train_end` loses a large
share of the fraud labels.

**Temporal split.** By `occurred_at`, as fractions of the data window: train 55%, validation 15%,
test 30% (for the `full` profile about days 0 to 99, 99 to 126, 126 to 180). The preprocessor, the
class weight and early stopping only see the past. Validation and test use **eventual labels**: that
is an offline evaluation done after the fact, and says so in the report. Choosing thresholds on
validation labels read the same way keeps the bands consistent with what is measured on test.

**No account-level split.** Production scores accounts it has already seen, so holding accounts out
would measure something else. The consequence is that an account's behavior leaks across the
boundary exactly as it would in production.

**Thresholds** are chosen on validation by target false-positive rates on label-negatives: review at
2%, decline at 0.2% (overridable by environment variables).

**Selection bias is reported, not fixed.** The report states how much ground-truth fraud has no label
(declined upstream or never reported: about half on the `full` data) and shows precision against the
labels and against the ground truth side by side. We do not attempt reject inference or
re-weighting in this phase.

## Consequences

- The model is trained on what a real team could know at the time, and the guard makes the common
  mistake (reading labels too soon) fail loudly.
- Reported PR-AUC and recall at low FPR are *pessimistic* for a model that detects fraud the labels
  miss: unlabeled fraud ranked high counts as a false positive. The model card explains this.
- A model trained this way inherits the blind spots of the earlier decision system. Only a
  randomized hold-out (let a small share of declines through) could measure them, which is out of
  scope.
- Training on `as_of = train_end + 45 days` uses chargebacks reported while the validation period is
  already running. That is how a model retrained today would be built; it is not a leak into the
  evaluation, because validation and test rows are never in the training split.
