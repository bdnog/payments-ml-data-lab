# Model card: `mlp-2026.11.1`

> **The data is synthetic.** Every number below describes how well this model separates the fraud
> patterns that this repository's own generator produced. None of it says anything about real-world
> fraud performance. The generator is probably easier than reality (clean, stationary patterns) and
> harder in places (hand-made overlap); nobody can say by how much.

The full, machine-generated evaluation is in
[`services/risk-scoring-api/models/current/report.md`](../services/risk-scoring-api/models/current/report.md).
The decisions behind the setup are in ADR 0007 (model and features), 0008 (labels and splits) and
0009 (serving).

## Intended use

- **Purpose:** rank card transactions by fraud risk at authorization time and map the score to an
  `APPROVE` / `REVIEW` / `DECLINE` decision, with heuristic reason codes.
- **Users:** a payments core calling `POST /v1/score`, and the engineers learning from this repo.
- **Not for:** real payments, credit decisions, or anything involving real people. There is no real
  customer data anywhere in the repository.

## Model

| | |
| --- | --- |
| Architecture | Embeddings (mcc 8, card_bin 8, channel 3, issuer_country 4, ip_country 4) concatenated with a 21-column dense block, then `Linear(48, 64)` `ReLU` `Dropout(0.1)` `Linear(64, 32)` `ReLU` `Linear(32, 1)` |
| Output | One logit; the service returns `sigmoid(logit)` as `score` |
| Loss | `BCEWithLogitsLoss` with `pos_weight = min(neg/pos, 50)` (here the cap of 50 applies) |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, batch size 1024, up to 30 epochs |
| Early stopping | Validation PR-AUC, patience 5, best weights restored (best epoch 7, stopped after 12) |
| Parameters | 5,797 (548 in the embeddings); the weights file is 27 KB |
| Framework | PyTorch (CPU), seeded, deterministic algorithms on |

**The score is a ranking score in [0, 1], not a calibrated probability.** The weighted loss
distorts calibration on purpose. Do not read 0.9 as "90% chance of fraud"; read it as "ranks above
almost everything else". Use the thresholds.

**Inputs** are exactly the `role: feature` columns of the `transactions` data contract: amount,
`occurred_at` (only as hour and weekday in UTC), mcc, channel, installments, card BIN, issuer
country, IP country, proxy flag, new-device and new-card flags, account age and four precomputed
behavior features (v1 contract, ADR 0006). Account, merchant and device ids, `is_fraud`,
`fraud_pattern` and the chargeback date are never inputs; a test flips them all and checks that the
model input does not change.

## Data and protocol

- **Dataset:** the `full` profile (seed 42): 499,666 transactions of 5,000 accounts and 200
  merchants over 180 days, 1% ground-truth fraud.
- **Label:** a chargeback reported on or before a cutoff. About half of the real fraud has no label
  (declined upstream, or never reported).
- **Split by time**, 55% / 15% / 30% of the window: train until 2026-04-10, validation until
  2026-05-07, test until 2026-06-30. No account-level split, because production scores accounts it
  has already seen.
- **Training labels** are read 45 days after the end of training (2026-05-25), the longest
  chargeback delay; validation and test use eventual labels (an offline, after-the-fact evaluation).

| split | rows | labeled fraud | label rate | PR-AUC | PR-AUC / prevalence |
| --- | --- | --- | --- | --- | --- |
| train | 263,054 | 1,276 | 0.485% | 0.506 | 104x |
| validation | 77,210 | 334 | 0.433% | 0.449 | 104x |
| test | 159,402 | 858 | 0.538% | 0.464 | 86x |

The **prevalence** (the label rate) is the PR-AUC of a random classifier, and is the only reference
line in this phase: there is no baseline model on purpose. Gradient boosting is the usual strong
baseline for tabular fraud data, and adding one is an intentional future extension (ADR 0007).

## Test results

| metric | value |
| --- | --- |
| PR-AUC | 0.464 (prevalence 0.538%) |
| ROC-AUC | 0.995 |
| Recall at FPR 1% | 95.0% |
| Recall at FPR 0.1% | 16.7% |
| Amount-weighted recall at the review threshold | 98.5% |
| Amount-weighted recall at the decline threshold | 29.1% |
| Brier score (not meaningful as calibration) | 0.0078 |

Bands on the test set (thresholds chosen on validation: review at FPR 2%, decline at FPR 0.2%):

| band | share of traffic | precision against labels | precision against ground truth | recall against labels |
| --- | --- | --- | --- | --- |
| APPROVE | 97.55% | 0.01% | 0.03% | 2.6% |
| REVIEW | 2.06% | 15.9% | 32.8% | 61.1% |
| DECLINE | 0.39% | 50.2% | 99.8% | 36.4% |

Recall by fraud pattern on the test set, against the ground truth (analysis-only columns):

| pattern | fraud rows | recall at review | recall at decline |
| --- | --- | --- | --- |
| CARD_TESTING | 609 | 98.7% | 51.9% |
| ACCOUNT_TAKEOVER | 492 | 98.6% | 22.6% |
| GEO_MISMATCH | 354 | 98.3% | 46.6% |
| VELOCITY_BURST | 292 | 90.4% | 9.6% |

Permutation importance on validation (PR-AUC drop when a group is shuffled): velocity 0.152,
device and card 0.119, geo 0.097, history 0.020, merchant and channel 0.009, time of day 0.002,
amount 0.002.

### Reading these numbers honestly

- **PR-AUC of 0.46 next to a ROC-AUC of 0.995 is the label bias, not a weak model.** About half of
  the real fraud has no chargeback label, and the model scores it as high as the labeled half
  (it looks the same). Those rows count as false positives. In the decline band, precision is
  50% against the labels and 99.8% against the ground truth. The same effect makes "recall at FPR
  0.1%" (16.7%) look poor: it is capped by unlabeled fraud ranked above legitimate traffic.
- **Selection bias.** Only approved transactions can be charged back, so the model never sees
  labels for what the previous system declined. A model trained this way inherits the blind spots
  of whoever decided before it, and the metrics above cannot show them.
- **The review band is where the value is.** 2% of traffic goes to review and catches 61% of the
  labeled fraud; declining 0.39% catches another 36%. The manual-review volume is a business
  parameter (the 2% FPR target), not a model property.
- **Velocity dominates** because two of the four fraud patterns are defined by it. That says how
  the generator works as much as how the model works.
- **VELOCITY_BURST is the weakest pattern** at the decline threshold (9.6%): its transactions are
  individually ordinary, and legitimate bursts are a deliberate hard negative.
- **Guards held.** The test PR-AUC is 86 times the prevalence (guard: at least 3x) and below 0.98
  (guard: suspiciously perfect). They are for investigation, not for tuning.

## Limitations

- Synthetic, stationary, single-currency (BRL) data; no seasonality, no adversary that adapts.
- v1 takes the velocity features from the caller. Computing them server-side is a v2 concern
  (ADR 0006), and the caller must compute them point-in-time (past data only) or the model sees
  inputs it was never trained on.
- Reason codes are **rules over the request, not attributions of the model**. They explain why a
  score is plausibly high, not why this network produced it. SHAP or integrated gradients would be
  the faithful alternative (ADR 0007).
- Categories seen fewer than 20 times in training map to `__unknown__`; a new card BIN or MCC is
  scored on the unknown embedding.
- No drift monitoring, no recalibration, no retraining schedule (out of scope).
- Reproducibility is guaranteed with the same seed, data and library build on the same platform;
  across platforms the last floating-point bits may differ.

## Ethics and misuse

The data describes no real person. The features are deliberately limited to transaction and device
signals; there are no demographic attributes. In a real deployment a fraud model decides who can
pay, so it needs: a documented appeal path for declined customers, monitoring of false-positive
rates by segment (country, channel, account age) for disparate impact, human review for the
`REVIEW` band, and a fallback policy when the service is unavailable. None of that is built here.
