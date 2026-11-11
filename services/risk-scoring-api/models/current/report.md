# Model report: `mlp-2026.11.1`

**The data is synthetic.** These numbers describe how well a model separates the fraud patterns that *our own generator* produced. They say nothing about real-world fraud performance: the generator probably makes the problem easier (clean, stationary patterns) and harder (hand-made overlap) than reality in ways nobody can quantify.

## Setup

- Split by `occurred_at` (fractions of the window): train `2026-01-01T00:00:00+00:00` to `2026-04-10T00:00:00+00:00`, validation to `2026-05-07T00:00:00+00:00`, test to `2026-06-30T00:00:00+00:00`. No account-level split (production scores known accounts).
- Training labels read as of `2026-05-25T00:00:00+00:00` (train end + 45 days, the maximum chargeback delay). Validation and test use eventual labels (an offline evaluation after the fact).
- Seed 42; best epoch 7 of 12 (early stopping triggered); pos_weight 50.00.
- The `score` is a **ranking score in [0, 1], not a calibrated probability** (the loss is weighted). Use the thresholds below, not the raw value as a risk percentage.

Data manifest of the dataset used:

```json
{
  "parameters": {
    "break_rate": 0.02,
    "days": 180,
    "fraud_rate": 0.01,
    "n_accounts": 5000,
    "n_merchants": 200,
    "name": "full",
    "seed": 42,
    "start_date": "2026-01-01",
    "target_transactions": 500000
  },
  "profile": "full",
  "realized": {
    "approval_rate": 0.935869,
    "approved_fraud_with_chargeback_share": 0.849862,
    "break_rate": 0.02,
    "breaks_by_type": {
      "AMOUNT_MISMATCH": 2228,
      "CHARGEBACK_NOT_IN_LEDGER": 914,
      "DUPLICATE_IN_PSP": 1395,
      "FEE_MISMATCH": 1357,
      "MISSING_IN_LEDGER": 1425,
      "MISSING_IN_PSP": 1753
    },
    "fraud_by_pattern": {
      "ACCOUNT_TAKEOVER": 1500,
      "CARD_TESTING": 1500,
      "GEO_MISMATCH": 1250,
      "VELOCITY_BURST": 750
    },
    "fraud_rate": 0.010007,
    "fraud_transactions": 5000,
    "fraud_with_chargeback_share": 0.4936,
    "psp_records_by_type": {
      "CHARGEBACK": 3681,
      "REFUND": 19184,
      "SALE": 441894
    },
    "refunded_payment_share": 0.039947,
    "voided_rate": 0.029992
  }
}
```

Hyperparameters:

```json
{
  "batch_size": 1024,
  "learning_rate": 0.001,
  "max_epochs": 30,
  "model": {
    "dropout": 0.1,
    "embedding_dims": {
      "card_bin": 8,
      "channel": 3,
      "ip_country": 4,
      "issuer_country": 4,
      "mcc": 8
    },
    "hidden_1": 64,
    "hidden_2": 32
  },
  "patience": 5,
  "pos_weight_cap": 50.0,
  "seed": 42,
  "weight_decay": 0.0001
}
```

## Splits and prevalence

| split | rows | positives | label rate | PR-AUC | lift over prevalence |
| --- | --- | --- | --- | --- | --- |
| train | 263,054 | 1,276 | 0.485% | 0.5057 | 104.3x |
| validation | 77,210 | 334 | 0.433% | 0.4488 | 103.7x |
| test | 159,402 | 858 | 0.538% | 0.4642 | 86.2x |

The **prevalence** (label rate) is the PR-AUC of a random classifier: the reference line. There is no baseline model in this phase on purpose (ADR 0007).

## Ranking metrics

| split | PR-AUC | ROC-AUC | recall @ FPR 0.1% | recall @ FPR 1% | amount recall @ review | amount recall @ decline | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0.5057 | 0.9971 | 24.29% | 98.12% | 99.28% | 32.36% | 0.00900 |
| validation | 0.4488 | 0.9961 | 19.46% | 95.51% | 98.78% | 35.04% | 0.00781 |
| test | 0.4642 | 0.9951 | 16.67% | 94.99% | 98.51% | 29.12% | 0.00783 |

Recall is measured against the *labels* (reported chargebacks). Amount-weighted recall is the share of the labeled fraud *money* (`amount_minor`) at or above the threshold.

## Decision bands

Thresholds were chosen on the validation set: review at FPR 2.0%, decline at FPR 0.2% of label-negatives. `review` = 0.109590, `decline` = 0.986345.

**validation**

| band | rows | share of traffic | labeled fraud | precision (labels) | precision (ground truth) | recall (labels) |
| --- | --- | --- | --- | --- | --- | --- |
| APPROVE | 75,347 | 97.59% | 8 | 0.01% | 0.02% | 2.40% |
| REVIEW | 1,574 | 2.04% | 190 | 12.07% | 25.03% | 56.89% |
| DECLINE | 289 | 0.37% | 136 | 47.06% | 100.00% | 40.72% |

**test**

| band | rows | share of traffic | labeled fraud | precision (labels) | precision (ground truth) | recall (labels) |
| --- | --- | --- | --- | --- | --- | --- |
| APPROVE | 155,492 | 97.55% | 22 | 0.01% | 0.03% | 2.56% |
| REVIEW | 3,289 | 2.06% | 524 | 15.93% | 32.78% | 61.07% |
| DECLINE | 621 | 0.39% | 312 | 50.24% | 99.84% | 36.36% |

## Recall by fraud pattern (ground truth, analysis only)

**validation**

| pattern | fraud rows | recall at review | recall at decline |
| --- | --- | --- | --- |
| CARD_TESTING | 229 | 100.00% | 69.87% |
| ACCOUNT_TAKEOVER | 205 | 100.00% | 27.32% |
| GEO_MISMATCH | 196 | 94.39% | 37.24% |
| VELOCITY_BURST | 70 | 91.43% | 0.00% |

**test**

| pattern | fraud rows | recall at review | recall at decline |
| --- | --- | --- | --- |
| CARD_TESTING | 609 | 98.69% | 51.89% |
| ACCOUNT_TAKEOVER | 492 | 98.58% | 22.56% |
| GEO_MISMATCH | 354 | 98.31% | 46.61% |
| VELOCITY_BURST | 292 | 90.41% | 9.59% |

## Label coverage (selection bias)

Ground-truth fraud that has **no label** was declined upstream (and so never charged back) or was never reported. The model cannot learn from it, and the metrics above do not count it.

| split | fraud rows | labeled | unlabeled | coverage |
| --- | --- | --- | --- | --- |
| train | 2,553 | 1,276 | 1,277 | 49.98% |
| validation | 700 | 334 | 366 | 47.71% |
| test | 1,747 | 858 | 889 | 49.11% |

## Permutation importance by feature group (validation)

| feature group | PR-AUC drop when shuffled |
| --- | --- |
| velocity | 0.1516 |
| device_and_card | 0.1193 |
| geo | 0.0965 |
| history | 0.0201 |
| merchant_and_channel | 0.0087 |
| time_of_day | 0.0025 |
| amount | 0.0018 |

Baseline validation PR-AUC: 0.4488. Groups are shuffled jointly; the drop is the mean of repeated shuffles.

## Score deciles (test)

| decile (1 = highest) | rows | labeled fraud | rate | min | max |
| --- | --- | --- | --- | --- | --- |
| 1 | 15,941 | 853 | 5.35% | 0.0040 | 0.9974 |
| 2 | 15,941 | 4 | 0.03% | 0.0007 | 0.0040 |
| 3 | 15,940 | 0 | 0.00% | 0.0002 | 0.0007 |
| 4 | 15,940 | 1 | 0.01% | 0.0001 | 0.0002 |
| 5 | 15,940 | 0 | 0.00% | 0.0000 | 0.0001 |
| 6 | 15,940 | 0 | 0.00% | 0.0000 | 0.0000 |
| 7 | 15,940 | 0 | 0.00% | 0.0000 | 0.0000 |
| 8 | 15,940 | 0 | 0.00% | 0.0000 | 0.0000 |
| 9 | 15,940 | 0 | 0.00% | 0.0000 | 0.0000 |
| 10 | 15,940 | 0 | 0.00% | 0.0000 | 0.0000 |

## Training curve

| epoch | train loss | validation PR-AUC |
| --- | --- | --- |
| 1 | 0.2215 | 0.4208 |
| 2 | 0.0867 | 0.4367 |
| 3 | 0.0745 | 0.4365 |
| 4 | 0.0694 | 0.4426 |
| 5 | 0.0655 | 0.4435 |
| 6 | 0.0620 | 0.4412 |
| 7 | 0.0581 | 0.4488 |
| 8 | 0.0572 | 0.4378 |
| 9 | 0.0540 | 0.4460 |
| 10 | 0.0504 | 0.4456 |
| 11 | 0.0502 | 0.4455 |
| 12 | 0.0463 | 0.4470 |

## Sanity guards

- Test PR-AUC is 86.2x the prevalence (guard: at least 3x): held.
- Test PR-AUC above 0.98 (guard: suspiciously perfect, check for leakage): no.

These are for investigation, not for tuning. Vocabulary sizes: {'mcc': 16, 'card_bin': 41, 'channel': 4, 'issuer_country': 7, 'ip_country': 13}.
