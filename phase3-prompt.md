# Prompt: Phase 3 of `payments-ml-data-lab` (risk-scoring-api for real, PyTorch)

## Role and context

You are continuing work on the **public portfolio monorepo** `payments-ml-data-lab`. Phases 1 and 2 are done: uv workspace, `packages/shared` (data-contract loader and validator), `packages/datagen` (synthetic "universe"), YAML data contracts in `contracts/data/`, committed sample data in `data/samples/`, and the **contract-first** `risk-scoring-api` (Pydantic schemas, stub routes returning `501`, exported OpenAPI). I am a senior Java/Spring Boot engineer learning Python, so keep the code idiomatic, typed and readable, and explain non-obvious choices in ADRs or short docstrings.

**Phase 3 turns `risk-scoring-api` into a working service:**
1. **Offline training** (`risk_scoring/training/`): label derivation, temporal splits, a shared feature pipeline, a PyTorch MLP, evaluation, and a versioned artifact bundle.
2. **Online serving** (`risk_scoring/serving/` and `api/`): load the artifact, score requests, apply decision thresholds, return reason codes, and keep an idempotent **decision log** in Postgres (SQLAlchemy 2.0 + Alembic).

**Deliberate decision: no LightGBM or other baseline model in this phase.** Do not add one. Report the **prevalence** (a random classifier's PR-AUC) as the reference line, and write in the ADR that gradient boosting is the usual strong baseline for tabular fraud data and is an intentional future extension point.

**Out of scope:** server-side velocity features (v2), Redis, the settlement pipeline, ONNX, MLflow or any experiment tracker, GPU support, drift monitoring, Prometheus metrics.

## Ground rules

- Read the repo first (`README.md`, `Makefile`, root and service `pyproject.toml`, ADRs, `contracts/`, `packages/*`) and **follow existing conventions**. If something differs from what this prompt assumes, adapt and mention it in the final report.
- All repo content in **English**. Do not ask me questions; decide, and record non-obvious decisions in ADRs.
- **The Phase 2 API contract is the source of truth.** Implement it exactly. The only intended contract changes are the additions listed in this prompt (`GET /ready`, and the role change of `occurred_at` below). `make contracts-check` must pass after regenerating, and every contract change must be listed in the final report.
- **No training/serving skew:** one feature pipeline, used by both paths.
- **No leakage:** `is_fraud`, `fraud_pattern` and any `analysis_only` column are never model inputs. They may only appear in evaluation reports.
- Allowed runtime dependencies: `torch`, `numpy`, `sqlalchemy>=2`, `alembic`, `psycopg[binary]` (v3). Dev-only: `scikit-learn` (used **only in tests** to cross-check my own metric implementations). Not allowed: lightgbm, xgboost, pandas, mlflow, matplotlib.
- One Conventional Commit per step. Do not push. Never claim something works without running it; if a check cannot run, say so in the final report.
- Be honest in reports: the data is synthetic, so metrics say nothing about real-world fraud performance. State that in the README, the model card and the report.

## Decisions already made

- **Model**: a small MLP with **embeddings** for categorical features (`mcc`, `card_bin`, `channel`, `issuer_country`, `ip_country`, each with an `__unknown__` index, and `__missing__` for null `ip_country`) concatenated with a numeric and boolean block: `Linear(in, 64) -> ReLU -> Dropout(0.1) -> Linear(64, 32) -> ReLU -> Linear(32, 1)`, producing a logit. Embedding sizes: mcc 8, card_bin 8, channel 3, countries 4. Rare categories (fewer than 20 train occurrences) map to `__unknown__`. Hyperparameters live in a frozen dataclass and are stored in the artifact.
- **Model inputs == the contract's feature columns, exactly.** The preprocessor derives its required raw columns from `contracts/data/transactions.yaml` (`role: feature`) and a test asserts the sets are equal. Ids (`account_id`, `merchant_id`, `device_id`) are not inputs. `occurred_at` is used for time-of-day features, so **change its role in the `transactions` contract from `metadata` to `feature`** (it is available at request time) and update the contract tests.
- **Label and split protocol** (document in an ADR and in `docs/risk-scoring.md`):
  - Label = `chargeback_reported_at IS NOT NULL AND chargeback_reported_at <= as_of`.
  - Temporal split by `occurred_at`, as fractions of the window: **train 55%, validation 15%, test 30%** (for the `full` profile about days 0 to 99, 99 to 126, 126 to 180). No account-level split (production scores known accounts), and say so.
  - Training labels use `as_of = train_end + label_maturity_days` (default **45**, the generator's maximum chargeback delay), with a **guard that refuses** any training set whose `as_of - train_end < label_maturity_days`. Validation and test use matured ("eventual") labels, as an offline evaluation done after the fact. Add a test showing that a too-early `as_of` loses fraud labels.
- **Loss and optimization**: `BCEWithLogitsLoss` with `pos_weight = min(neg/pos, 50)`, AdamW (lr 1e-3, weight decay 1e-4), batch size 1024, up to 30 epochs, early stopping on **validation PR-AUC** (patience 5, restore the best weights). CPU by default; fixed seeds for Python, NumPy and torch; `torch.use_deterministic_algorithms(True)`; manual shuffling with a seeded generator.
- **`score` is a ranking score in [0, 1], not a calibrated probability** (pos_weight distorts calibration). Say so in the OpenAPI description and the model card.
- **Decision bands**: `APPROVE` if `score < review`, `REVIEW` if `review <= score < decline`, else `DECLINE`. Thresholds are chosen on the validation set by target false-positive rates on label-negatives (defaults: review at FPR 2%, decline at FPR 0.2%), stored in the artifact, and overridable by env vars.
- **Reason codes are heuristic explanations, not model attributions.** Rule-based on the request values, using the enum from the Phase 2 contract, with documented thresholds stored in the artifact metadata (for example: `PROXY_DETECTED` if `is_proxy`; `NEW_DEVICE`; `COUNTRY_MISMATCH` if `ip_country` differs from `issuer_country`; `AMOUNT_ABOVE_NORMAL` if amount is at least 3x `avg_amount_30d`; `HIGH_VELOCITY_1H` at 5 or more; `HIGH_VELOCITY_24H`; `NEW_ACCOUNT` under 30 days; `HIGH_RISK_MCC` from a configured set; `NEW_CARD_FOR_MERCHANT`). Return at most 5, in a fixed priority order. Mention SHAP or integrated gradients as a future option in the ADR.
- **Serving**: endpoints are plain `def` (not `async def`) because inference is CPU-bound and blocking; FastAPI runs them in its threadpool. `torch.set_num_threads` from settings (default 1), `torch.inference_mode()`, scale with `uvicorn` workers (`WEB_CONCURRENCY`). Explain this trade-off in the ADR (a classic interview topic).
- **Persistence**: ports and adapters. A `DecisionRepository` protocol with an `InMemoryDecisionRepository` (unit tests) and a `SqlDecisionRepository` (Postgres). Sync SQLAlchemy 2.0 `Session` per request through `Depends`. Alembic migrations in `services/risk-scoring-api/migrations/`.
- **Idempotency**: insert-first with `INSERT ... ON CONFLICT (transaction_id) DO NOTHING`; if the row already existed, load it and compare `request_hash` (sha256 of canonical JSON with sorted keys): same hash returns the stored response (`200`), different hash returns `409` problem+json. Scoring is deterministic, so racing identical requests converge.
- **Model files**: weights saved as a `state_dict` and loaded with `torch.load(..., weights_only=True)`. The artifact records sha256 of each file and the loader verifies them (small supply-chain talking point).
- **Promoted model**: training outputs go to `services/risk-scoring-api/models/runs/<version>/` (gitignored). `make promote-model VERSION=...` copies one run into `services/risk-scoring-api/models/current/` (**committed**, it is only a few KB), so the repo works right after cloning. `model_version` is passed explicitly (`--model-version mlp-2026.11.1` for the promoted one); no wall-clock time in the version.

## Target structure (additions and changes)

```
services/risk-scoring-api/
├── alembic.ini
├── migrations/versions/0001_create_decisions.py
├── models/
│   └── current/                       # committed: model.pt, preprocessor.json, metadata.json, report.md
├── src/risk_scoring/
│   ├── config.py                      # + database_url, model_dir, thresholds overrides, torch_threads
│   ├── dataset_mapping.py             # transactions row -> ScoreRequest (reused by tests and scripts)
│   ├── training/
│   │   ├── __main__.py                # CLI: train / evaluate
│   │   ├── labels.py                  # derive_label(df, as_of), maturity guard
│   │   ├── splits.py                  # temporal split by fractions
│   │   ├── features.py                # THE feature pipeline (pure, Polars, batch and single-row)
│   │   ├── preprocessing.py           # fit/transform, vocabularies, scaling stats, JSON (de)serialization
│   │   ├── model.py                   # MLP with embeddings
│   │   ├── trainer.py                 # training loop, early stopping, determinism
│   │   ├── metrics.py                 # PR-AUC, ROC-AUC, recall at FPR, amount-weighted recall, Brier (NumPy)
│   │   ├── thresholds.py              # choose review/decline thresholds by target FPR
│   │   ├── evaluate.py                # per split, per band, per fraud pattern, permutation importance
│   │   ├── report.py                  # report.md and metrics.json
│   │   └── artifacts.py               # save/load bundle, sha256 verification
│   ├── serving/
│   │   ├── scorer.py                  # Scorer: load artifact, score one or many
│   │   └── reasons.py                 # heuristic reason codes
│   ├── persistence/
│   │   ├── models.py                  # SQLAlchemy 2.0 declarative: DecisionRow
│   │   ├── repository.py              # protocol + InMemory + Sql adapters
│   │   └── session.py                 # engine/session factory
│   └── api/                           # lifespan, deps, errors (problem+json), routes
└── tests/                             # unit, api, persistence, parity
scripts/score_sample.py                # sends sample rows to a running API and prints a summary
docker/postgres/init/01-create-databases.sql   # creates risk_scoring and settlement databases
docs/
├── risk-scoring.md                    # online flow (Mermaid sequence), skew prevention, idempotency, bands, failure behavior
├── model-card.md                      # intended use, data, metrics, limitations, ethics
└── adr/0007..0009
```

## Feature pipeline (`features.py`)

One pure function that takes a Polars DataFrame with the contract's feature columns and returns the **derived frame** the preprocessor consumes. It must work identically on a 500k-row frame and a 1-row frame. Derived features (all documented in the docstring):
- `log1p` of `amount_minor`, `account_age_days`, `txn_count_1h`, `txn_count_24h`, `avg_amount_30d_minor`, `seconds_since_last_txn`, with **missing indicators** for the two nullable behavior fields (fill with 0 after the indicator).
- `amount_to_avg_ratio` (clipped to [0, 20]; 1.0 when the average is missing).
- Time features from `occurred_at` in UTC: hour as sin/cos, day-of-week as sin/cos.
- `country_mismatch` (`ip_country` not null and different from `issuer_country`), `ip_foreign`, `issuer_foreign`, `ip_country_missing`.
- Booleans as 0/1 floats: `is_proxy`, `device_is_new_for_account`, `card_is_new_for_merchant`; `installments` as numeric.
- Categoricals kept as strings for the embedding indices: `mcc`, `card_bin`, `channel`, `issuer_country`, `ip_country`.

Numeric standardization uses **train-only** statistics, then clips to [-5, 5]. The preprocessor serializes to `preprocessor.json` (schema version, input columns, vocabularies, means/stds, clip bounds) and the loader fails fast if its input columns differ from the current contract's features.

## Evaluation and report

Compute per split (train, validation, test): rows, label rate, and the **prevalence reference** PR-AUC. Metrics (implemented in NumPy, ties handled by grouping equal scores): PR-AUC (average precision, step-wise like scikit-learn), ROC-AUC, recall at FPR 0.1% and 1%, **amount-weighted recall** (sum of `amount_minor` of caught labeled fraud over total labeled fraud amount), Brier score, and a score-decile table. Also: per-band table on validation and test (share of traffic, precision, recall, so the manual-review volume is visible); **per-fraud-pattern recall** at the review and decline thresholds using the `analysis_only` columns (never as features); label coverage versus ground truth (how much real fraud has no label because it was declined upstream or never reported, which is the selection bias to discuss); and **permutation importance by feature group** on validation (PR-AUC drop). `report.md` also contains the training curve (loss and validation PR-AUC per epoch as a table), hyperparameters, seed, split dates, `label_as_of`, and the data manifest (seed, profile, parameters) of the dataset used.

Sanity guards, to report and not to tune away: if test PR-AUC is not clearly above the prevalence (at least 3x), investigate for bugs; if it is suspiciously perfect (above 0.98), investigate leakage before celebrating. Say in the report that the synthetic generator likely makes the problem easier or harder than reality.

## API behavior

- `GET /ready`: `200` when the model and thresholds are loaded, else `503` problem+json. `/health` stays a pure liveness probe. Add `/ready` to the OpenAPI.
- `POST /v1/score`: validate, compute features and score through `Scorer`, apply bands, derive reason codes, persist through the repository with the idempotency rules above, return the contract response (`model_version`, `latency_ms` as the contract defines it, measured around feature computation plus inference).
- `POST /v1/score:batch` (max 100): one batched forward pass; per-item result or per-item error exactly as the Phase 2 contract defines; idempotency per item.
- `GET /v1/model`: `model_version`, `trained_at`, `thresholds`, flattened test metrics (for example `test.pr_auc`), read from the artifact metadata.
- Errors: a handler that turns `RequestValidationError` into `application/problem+json` (`422`, with an `errors` extension), and `409` for idempotency conflicts. Problem `type` values use `urn:problem:<slug>`.
- Structured log line per decision (`transaction_id`, `decision`, `score`, `latency_ms`, `model_version`), using `payments_shared.logging`.
- Startup via FastAPI `lifespan`: load the artifact (verify hashes and feature columns) once and keep the `Scorer` in `app.state`. If loading fails, the app still starts but `/ready` returns `503`.

### Decision log table (`decisions`)
`transaction_id` (PK), `request_hash`, `request` (JSONB), `score`, `decision`, `model_version`, `reason_codes` (JSONB), `latency_ms`, `created_at` (server default `now()`). Index on `created_at`. Use timezone-aware timestamps.

## Infrastructure changes

- **Torch CPU wheels on Linux** (avoid multi-GB CUDA downloads in CI and Docker). In the root `pyproject.toml`:
  ```toml
  [tool.uv.sources]
  torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true
  ```
  Verify that `uv lock` and `uv sync` work (if workspace-level sources do not apply to the member, move them to the service `pyproject.toml`). Use the latest stable `torch` 2.x supporting Python 3.12. Add `numpy` explicitly.
- `datagen` and `scikit-learn` are **dev** dependencies of `risk-scoring-api` (tests generate a small dataset via the `payments_datagen` API; training itself only reads Parquet).
- **Dockerfile**: the service image now includes `models/current/`, `migrations/` and `alembic.ini`; keep the multi-stage build, non-root user, `WEB_CONCURRENCY` support; report the final image size.
- **Compose**: init script creating the `risk_scoring` and `settlement` databases (document that an existing Postgres volume needs a one-time `docker compose down -v`); a one-shot `risk-migrate` service running `alembic upgrade head`; `risk-scoring-api` depends on it with `condition: service_completed_successfully`; its healthcheck uses `/ready`.
- **CI**: add a Postgres service to the test job and set `RISK_TEST_DATABASE_URL`; the SQL repository tests run there. Keep the job fast. Also run `make contracts-check`.
- **Makefile** targets: `train-risk` (`VERSION=`, `DATA=data/out/full`), `eval-risk`, `promote-model` (`VERSION=`), `migrate-risk`, `test-db` (brings up Postgres and runs the SQL tests locally), `bench-risk` (in-process: 1,000 single-request scorings, prints p50 and p95), `demo-score` (runs `scripts/score_sample.py`).

## Testing requirements

- **Features**: golden-value tests for each derived feature; null handling; the single-row path equals the batch path.
- **`test_no_training_serving_skew`** (critical): for N sample rows, features and final scores computed through the batch (training) path equal those through the single-request serving path (`allclose`). The preprocessor's required columns equal the contract's `role: feature` set exactly.
- **Labels and splits**: maturity guard; a too-early `as_of` loses fraud labels; splits are disjoint, ordered in time, and cover the window.
- **Metrics**: PR-AUC, ROC-AUC and recall-at-FPR compared with scikit-learn on random data with ties (tolerance 1e-9); amount-weighted recall on a hand-computed example.
- **Training**: determinism (same seed gives equal weights on CPU, with a note that this is guaranteed only on the same platform); a session-scoped fixture generates about 15,000 transactions at a 5% fraud rate through `payments_datagen`, and the model must beat prevalence by a clear margin (at least 3x PR-AUC); the whole training test suite stays under about 30 seconds.
- **Artifacts**: save/load round trip, hash verification detects a tampered file, loader rejects mismatched feature columns.
- **Scorer and reasons**: reason code rules, thresholds (monkeypatch) move decisions between bands, batch equals single.
- **API** (`TestClient`, in-memory repository, a small model trained into a tmp dir): valid response for every Phase 2 example, `422` problem+json, idempotent replay returns the same body, same id with a different payload returns `409`, batch with mixed valid and invalid items, `/ready` is `503` when the model is missing, `/v1/model` contents.
- **Repository contract tests** parametrized over both adapters (the SQL one skipped with an explicit reason unless `RISK_TEST_DATABASE_URL` is set), including a **concurrency test**: several threads insert the same `transaction_id`, exactly one row exists and every caller gets the same result.
- Phase 2 tests (contract alignment, OpenAPI check) keep passing.

## Step-by-step plan (one commit per step)

1. **Dependencies and tooling**: torch CPU index, numpy, sqlalchemy, alembic, psycopg; dev deps; verify `uv sync`, Docker build and CI still work.
2. **Contract tweak**: `occurred_at` role to `feature`; update contract tests and regenerate artifacts that depend on it.
3. **Features and preprocessor** with serialization, golden tests and the skew-parity test (using sample data).
4. **Labels and splits** with the maturity guard and tests.
5. **Metrics** (NumPy) with scikit-learn parity tests.
6. **Model and trainer** with determinism and learning tests.
7. **Thresholds, evaluation, report, artifacts, CLI** (`train`, `evaluate`) with tests.
8. **Train on the full dataset**: ensure `data/out/full` exists (generate it with the Phase 2 command if not), run `make train-risk VERSION=mlp-2026.11.1`, review the report against the sanity guards, then `make promote-model`. Commit `models/current/` and write `docs/model-card.md` from the real numbers.
9. **Scorer and reason codes** with tests (including parity and `make bench-risk`).
10. **Persistence**: SQLAlchemy model, repository protocol and adapters, Alembic migration, contract tests for both adapters including concurrency.
11. **API**: lifespan, `/ready`, real endpoints, problem+json handlers, tests; regenerate OpenAPI (`make contracts`) and review the diff.
12. **Docker, Compose, CI**: migrate service, init script, healthchecks, CI Postgres service.
13. **DX and docs**: Makefile targets, `scripts/score_sample.py`, `docs/risk-scoring.md` (Mermaid sequence diagram of the online path, skew prevention, idempotency flow, bands, and a short section on what a caller such as a payments core should do on timeout or `5xx`: bounded timeout, fallback policy, never block the payment indefinitely), ADRs 0007 (model, features and why no baseline yet), 0008 (label maturity, temporal split and selection bias), 0009 (serving concurrency model, idempotent decision log, ports and adapters), README (training and serving quickstart, a results table with the real numbers and the synthetic-data disclaimer, roadmap with Phase 3 checked).

## Definition of done

Run these yourself and show the results:

1. `uv sync --all-packages`, `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest` pass; the SQL tests also pass with `make test-db`.
2. `make train-risk VERSION=mlp-2026.11.1` completes on the `full` dataset. Report the wall-clock time, split sizes and label rates, test PR-AUC versus prevalence, ROC-AUC, recall at FPR 0.1% and 1%, amount-weighted recall, the share of traffic in each band, per-pattern recall, and the permutation-importance ranking. State whether the sanity guards held.
3. The skew-parity test passes, and `make contracts-check` passes (list every contract change).
4. `docker compose up -d` brings up Postgres, the migration and `risk-scoring-api` (`/ready` returns 200). Then `make demo-score` shows: fraud rows get clearly higher scores than legitimate rows (report the means), a batch call works, replaying a request returns the same body, and the same `transaction_id` with a changed payload returns `409`.
5. `make bench-risk` reports p50 and p95 latency (target: p95 under 20 ms per request on CPU; if not met, report it and the likely cause instead of hiding it).
6. `docker compose build` works; report the `risk-scoring-api` image size.
7. `git log --oneline` shows small conventional commits, and `git status` is clean.

## Final report format

Keep it short: what was created (tree), the results of the seven checks (pass, fail or not run, with reasons), the key metrics table, any contract changes, any decision not obvious from this prompt, and the suggested next step (Phase 4: settlement-service pipeline with Polars, reconciliation validated against `injected_breaks`, marts and the read API).
