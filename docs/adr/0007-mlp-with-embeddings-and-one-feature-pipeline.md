# 0007. An MLP with embeddings, one shared feature pipeline, and no baseline model yet

- Status: accepted
- Date: 2026-11-11

## Context

Phase 3 turns `risk-scoring-api` into a working service with a PyTorch model. Three things can go
wrong quietly in a fraud model: the features differ between training and serving (skew), a column
that encodes the answer sneaks in as an input (leakage), and the numbers are read as more than they
are (calibration, synthetic data).

## Decision

**Model.** A small MLP: embeddings for the five categorical columns (`mcc` 8, `card_bin` 8, `channel`
3, `issuer_country` 4, `ip_country` 4) concatenated with a 21-column numeric and flag block, then
`Linear(48, 64) ReLU Dropout(0.1) Linear(64, 32) ReLU Linear(32, 1)`. Categories seen fewer than 20
times in training map to `__unknown__`; a null `ip_country` (in-person purchase) has its own
`__missing__` index. Loss is `BCEWithLogitsLoss` with `pos_weight = min(neg/pos, 50)`, AdamW, early
stopping on validation PR-AUC. Hyperparameters live in a frozen dataclass and are stored in the
artifact.

**One feature pipeline.** `training/features.py::derive_features` is a pure Polars function from the
contract's feature columns to the derived columns. Training calls it on a whole frame; serving
builds a one-row (or batch) frame from the request and calls the same function, then the same
`Preprocessor.transform`. There is no second implementation to drift. The test
`test_no_training_serving_skew` checks the values and the final scores both ways.

**Inputs are the contract.** The required raw columns are read from the `role: feature` columns of
`contracts/data/transactions.yaml`, and a test asserts the sets are equal. Ids, `is_fraud`,
`fraud_pattern` and the chargeback date are not inputs, and a test flips all of them and checks the
model input does not change. `occurred_at` became a feature for the hour and weekday (UTC); the
raw timestamp is never fed to the network.

**Scaling.** Continuous columns are standardized with training-split statistics and clipped to
[-5, 5]. Flags are passed through as 0/1 (standardizing a 1% flag only makes a z-score of 10).

**`score` is a ranking score.** `pos_weight` distorts calibration on purpose, so `score` is not a
probability. This is stated in the OpenAPI description and the model card. Decisions use thresholds
(ADR 0008), not the raw value.

**Reason codes are heuristics.** They are rules over the request values with documented
thresholds stored in the artifact, in a fixed priority order, at most five, and only for
transactions that are not approved. They are *not* attributions of the network.

**No baseline model in this phase.** We do not train LightGBM or any other baseline. We report the
**prevalence** (the PR-AUC of a random classifier) as the only reference line. Gradient boosting is
the usual strong baseline for tabular fraud data, and comparing against it is an intentional
future extension point; the training code is split so that another model only needs a trainer and
the same `Preprocessor` (or its own encoding of the derived frame). Until then, no claim is made
about whether a neural network is the right tool here.

## Consequences

- Skew is prevented by construction (one code path) and checked by a test, not by discipline.
- A new feature needs a change in the contract, the pipeline and the request model; the alignment
  tests fail until all three agree.
- The numbers have no external reference except prevalence. They show the pipeline works and the
  model learns the generator's patterns; they do not show it is the best choice.
- Explaining a score faithfully would need SHAP or integrated gradients; that is a future option,
  and the heuristic reason codes must not be presented as such.
- Reproducibility is guaranteed for the same seed, data and library build on the same platform
  (CPU, same thread count); across platforms the last bits of floating point may differ.
