# Prompt: Phase 2 of `payments-ml-data-lab` (data generators and contracts)

## Role and context

You are continuing work on the **public portfolio monorepo** `payments-ml-data-lab` (Phase 1, the skeleton, is done: uv workspace, `packages/shared`, `services/risk-scoring-api`, `services/settlement-service`, Docker, Compose, CI, Makefile, ADRs). I am a senior Java/Spring Boot engineer learning Python, so keep the code idiomatic, typed and readable, and explain non-obvious choices in ADRs or short docstrings.

**Phase 2 delivers two things:**
1. **Contracts**: API contracts (contract-first, as Pydantic models plus stub routes plus exported OpenAPI) and **data contracts** (YAML, validated in code) for every dataset in the lab.
2. **Synthetic data**: a deterministic generator that produces one coherent "universe" (accounts, merchants, transactions with fraud labels, and the settlement inputs derived from those transactions), plus a small committed sample.

**Out of scope:** model training, scoring logic, the settlement pipeline, database access, Alembic. Stub routes return `501`.

## Ground rules

- Start by reading the repo (`README.md`, root `pyproject.toml`, `Makefile`, `docs/adr/`, both services) and **follow the conventions already there**. If Phase 1 differs from what this prompt assumes, adapt and mention it in the final report.
- All repo content in **English**. Do not ask me questions; decide, and record non-obvious decisions in ADRs.
- Everything is **synthetic**. No real brands (use generic merchant names), no real data, no full card numbers (only 6-digit BINs). Generators must be **deterministic** (fixed seed, no wall-clock time, no unordered iteration affecting output).
- No new heavy dependencies: **no** PyTorch, LightGBM, scikit-learn, pandas, DuckDB, Faker. Allowed: `polars`, `numpy`, `pyyaml`.
- **Money is always an integer in minor units** (`amount_minor`, centavos), never a float. Record this in an ADR.
- Vectorize with Polars/NumPy. No per-row Python loops, except per fraud *episode* if needed. The `full` profile must generate in well under a minute on a laptop.
- One Conventional Commit per step. Do not push. Never claim something works without running it; if a check cannot run, say so in the final report.

## Decisions already made

- **v1 of the scoring contract: the caller sends the `behavior` block** (velocity features) already computed. Server-side computation from history is a future v2. The generator computes these features **point-in-time** (only past data, current transaction excluded), which is exactly the logic v2 will need to reproduce at serving time (training/serving skew).
- Data generators live in a **new workspace package** `packages/datagen` (distribution `payments-datagen`, import `payments_datagen`), not in `data/generators/`. Remove `data/generators/` and fix any docs that mention it.
- Data contracts live as **YAML in `contracts/data/`** (source of truth, human-readable). `packages/shared` provides the loader and validator. OpenAPI JSON in `contracts/openapi/` is **generated** from the apps and checked in CI.
- Settlement data is **derived from the same transactions** as the risk data, so both services share one universe of ids.

## Target structure (additions and changes)

```
contracts/
├── README.md                          # update: explain the three kinds of contract
├── data/                              # 11 YAML data contracts (see below)
├── openapi/
│   ├── risk-scoring-api.v1.json       # generated
│   └── settlement-service.v1.json     # generated
└── examples/
    ├── risk-scoring/{score.request.json, score.response.json, problem.json}
    └── settlement/{settlements.page.json, breaks.page.json, run.json}
packages/
├── shared/src/payments_shared/
│   ├── data_contracts/                # loader + validator (new)
│   └── paths.py                       # repo_root(), contracts_dir() honoring $CONTRACTS_DIR
└── datagen/                           # new workspace member
    ├── pyproject.toml
    ├── src/payments_datagen/
    │   ├── __main__.py                # CLI: generate / validate
    │   ├── profiles.py                # sample / full parameters
    │   ├── universe.py                # accounts, merchants
    │   ├── transactions.py            # legit traffic + point-in-time features
    │   ├── fraud.py                   # fraud episodes + hard negatives + chargeback labels
    │   ├── settlement.py              # payments, refunds, chargebacks, PSP file, injected breaks
    │   ├── invariants.py              # cross-dataset checks
    │   └── manifest.py
    └── tests/
data/
├── samples/                           # committed output of the `sample` profile (+ manifest.json)
└── out/                               # gitignored output of local runs
scripts/export_openapi.py              # writes contracts/openapi/*.json deterministically
docs/
├── data-model.md                      # Mermaid ER diagram of all datasets + notes
├── fraud-patterns.md                  # patterns, hard negatives, label delay and bias
└── adr/0003..0006
```

Dataset layout on disk (`--out DIR`): `reference/{merchants,accounts}.parquet`, `risk/transactions.parquet`, `settlement/{payments,refunds,chargebacks,psp_settlement,injected_breaks}.parquet`, `manifest.json`. One Parquet file per dataset (zstd), rows sorted deterministically (by time or primary key). The manifest holds seed, profile, parameters, row counts and realized rates, with **no timestamps of the run**.

## Data contract format (YAML)

```yaml
name: transactions
version: 1
description: One row per card transaction attempt, with point-in-time behavioral features and fraud labels.
primary_key: [transaction_id]
columns:
  - name: transaction_id
    type: string                 # string | int64 | float64 | bool | date | timestamp_utc
    nullable: false
    role: id                     # id | feature | metadata | label_source | analysis_only
    description: Unique transaction id.
    constraints: { pattern: "^txn_[0-9a-f]{12}$" }
  - name: amount_minor
    type: int64
    nullable: false
    role: feature
    constraints: { min: 50, max: 5000000 }
rules:                           # row-level rules as SQL expressions (pl.sql_expr); a row violates when the expr is false
  - { name: velocity_monotonic, expr: "txn_count_1h <= txn_count_24h" }
```

`constraints` support `min`, `max`, `accepted_values`, `pattern`. `role` lets later phases derive the feature list from the contract (`role: feature`) instead of hardcoding it.

**Validator API** (`payments_shared.data_contracts`): `load_contract(name)`, `validate_dataframe(df, contract) -> list[Violation]`, `assert_valid(df, contract)`, `check_foreign_key(child_df, child_col, parent_df, parent_col)`. It checks: exact column set and dtypes, nullability, primary key uniqueness, constraints, rules. Violations include contract name, column/rule, count and a few example values.

## Dataset specifications

Id formats (all lowercase hex, assigned **after sorting by time/creation** so ids grow with time): `acc_` + 8, `mer_` + 6, `dev_` + 10, `txn_` + 12, `pay_` + 12, `ref_` + 12, `cbk_` + 12, `psp_` + 12, `brk_` + 8. Currency is `BRL` only for now (the column exists; constraint `accepted_values: [BRL]`, note it in the contract). Country codes are ISO 3166-1 alpha-2. Timestamps are UTC with microsecond precision.

### `merchants` (PK `merchant_id`)
`merchant_id`, `name` (generic, e.g. "Northwind Grocer 0042"-style, no real brands), `mcc` (4 digits), `country`, `mdr_bps` (int, 50 to 500, contracted fee in basis points), `settlement_days` (accepted 1, 2, 30), `created_at`.

### `accounts` (PK `account_id`)
`account_id`, `created_at` (some before the window start, some during it, so `account_age_days` varies and new accounts exist), `home_country` (about 95% BR).

### `transactions` (PK `transaction_id`)

| Column | Type | Null | Role | Notes |
|---|---|---|---|---|
| transaction_id | string | no | id | |
| occurred_at | timestamp_utc | no | metadata | |
| account_id, merchant_id | string | no | id | FKs |
| amount_minor | int64 | no | feature | 50 to 5,000,000 |
| currency | string | no | metadata | BRL |
| mcc | string | no | feature | copied from merchant (available at request time) |
| channel | string | no | feature | web, app, pos |
| installments | int64 | no | feature | 1 to 12 |
| card_bin | string | no | feature | 6 digits, from a pool of about 40 synthetic BINs |
| issuer_country | string | no | feature | |
| device_id | string | no | id | for `pos` it is the terminal id |
| ip_country | string | yes | feature | null for `pos` |
| is_proxy | bool | no | feature | false for `pos` |
| device_is_new_for_account | bool | no | feature | point-in-time |
| card_is_new_for_merchant | bool | no | feature | point-in-time |
| account_age_days | int64 | no | feature | at `occurred_at`, >= 0 |
| txn_count_1h, txn_count_24h | int64 | no | feature | prior transactions of the account, current excluded |
| avg_amount_30d_minor | int64 | yes | feature | mean of prior 30 days; null if none |
| seconds_since_last_txn | int64 | yes | feature | null if first transaction |
| is_fraud | bool | no | analysis_only | ground truth, **never a feature** |
| fraud_pattern | string | yes | analysis_only | CARD_TESTING, ACCOUNT_TAKEOVER, GEO_MISMATCH, VELOCITY_BURST |
| chargeback_reported_at | timestamp_utc | yes | label_source | see label rules |

Rules: `txn_count_1h <= txn_count_24h`; `is_fraud OR (fraud_pattern IS NULL AND chargeback_reported_at IS NULL)`; `chargeback_reported_at IS NULL OR chargeback_reported_at > occurred_at`. Compute point-in-time features with Polars rolling-by-time operations (window closed so the current row is excluded) over rows sorted by account and time.

### `payments` (PK `payment_id`)
`payment_id`, `transaction_id` (FK), `merchant_id` (FK), `account_id` (FK), `amount_minor`, `currency`, `status` (CAPTURED, VOIDED), `authorized_at`, `captured_at` (null iff VOIDED). Rule: `captured_at IS NULL OR captured_at >= authorized_at`. Capture happens 0 to 2 days after authorization; about 3% of payments are VOIDED. Refunds are separate rows; a refunded payment stays CAPTURED.

### `refunds` (PK `refund_id`)
`refund_id`, `payment_id` (FK), `amount_minor` (> 0 and <= payment amount, partial refunds allowed, sum per payment <= payment amount), `created_at` (after capture), `reason` (CUSTOMER_REQUEST, DUPLICATE, PRODUCT_ISSUE). About 4% of captured payments.

### `chargebacks` (PK `chargeback_id`)
`chargeback_id`, `payment_id` (FK), `amount_minor` (full payment amount), `opened_at`, `reason_code` (FRAUD, PRODUCT_NOT_RECEIVED, DUPLICATE_CHARGE). For FRAUD chargebacks, `opened_at` **equals** the transaction's `chargeback_reported_at`. Add a few non-fraud chargebacks (friendly fraud) on legitimate payments.

### `psp_settlement` (PK `psp_record_id`), the file received from the PSP
`psp_record_id`, `merchant_reference` (the `payment_id`; nullable, for orphans), `record_type` (SALE, REFUND, CHARGEBACK), `merchant_id`, `gross_amount_minor` (positive), `fee_minor` (>= 0), `net_amount_minor` (signed), `currency`, `settlement_date` (date).
Semantics: SALE `net = gross - fee`; REFUND `fee = 0`, `net = -gross`; CHARGEBACK `fee = 1500` (flat), `net = -(gross + fee)`. Expected SALE fee = `round_half_up(gross * mdr_bps / 10000)` (document the rounding rule as part of the domain). `settlement_date` = capture date + merchant `settlement_days`. Records exist only when `settlement_date <= last day of the window`: payments not yet due have **no** PSP record and are **not** breaks (document this "due date cutoff", reconciliation in Phase 4 must respect it).

### `injected_breaks` (PK `break_id`), ground truth for testing Phase 4
`break_id`, `break_type`, `payment_id` (nullable), `psp_record_id` (nullable), `expected_diff_minor` (signed, PSP minus ledger).

| break_type | What the generator does | expected_diff_minor |
|---|---|---|
| AMOUNT_MISMATCH | PSP gross differs from ledger amount | psp gross - ledger amount |
| MISSING_IN_PSP | due SALE record omitted from the PSP file | -ledger amount |
| MISSING_IN_LEDGER | PSP SALE record whose reference is unknown or null | +psp gross |
| DUPLICATE_IN_PSP | same SALE emitted twice (new `psp_record_id`) | +duplicate gross |
| FEE_MISMATCH | PSP fee differs from contracted fee | psp fee - expected fee |
| CHARGEBACK_NOT_IN_LEDGER | PSP CHARGEBACK with no ledger chargeback | +psp gross |

At most **one break per payment**. Break rate is a profile parameter.

### Mart contracts (YAML only in this phase; produced in Phase 4)
- `daily_settlement_by_merchant` (PK `settlement_date, merchant_id, currency`): `payment_count`, `gross_minor`, `fee_minor`, `refunds_minor`, `chargebacks_minor`, `net_minor`.
- `reconciliation_breaks` (PK `break_id`): `run_date`, `settlement_date` (nullable), `break_type`, `payment_id`, `psp_record_id`, `ledger_amount_minor` (nullable), `psp_amount_minor` (nullable), `diff_minor`.
- `fee_summary` (PK `settlement_date, merchant_id, currency`): `gross_minor`, `expected_fee_minor`, `actual_fee_minor`, `fee_diff_minor`.

## Generation design

**Profiles** (CLI flags override any value; `--seed` default 42, `--start-date` default 2026-01-01):

| | sample | full |
|---|---|---|
| accounts / merchants | 300 / 25 | 5,000 / 200 |
| days | 30 | 180 |
| target transactions | about 3,000 | about 500,000 |
| fraud rate | 3% | 1% |
| break rate (of captured payments) | 6% | 2% |

**Legitimate traffic.** Per-account latent traits: activity level (log-normal), typical ticket (log-normal), 1 to 3 devices, preferred merchants (Zipf-like), home country. Hour-of-day with an evening peak, mild weekend effect, `pos` mostly in-person MCCs (groceries, restaurants, gas), `web` and `app` for the rest. Use roughly 15 MCCs with different typical tickets and risk (for example 5411, 5812, 5814, 5541, 5311, 4111, 4900, 5732, 5944, 5816, 5967, 4829).

**Fraud episodes** (a fraud-rate share of transactions; every transaction in an episode is labeled fraud). Mix: 30% CARD_TESTING, 30% ACCOUNT_TAKEOVER, 25% GEO_MISMATCH, 15% VELOCITY_BURST.

| Pattern | Shape |
|---|---|
| CARD_TESTING | 5 to 30 transactions within 15 minutes, tiny amounts (50 to 500 minor), same device, mostly web, `card_is_new_for_merchant` true, sometimes ends with one larger charge |
| ACCOUNT_TAKEOVER | account older than 90 days, new device, foreign `ip_country`, proxy likely, amount 3x to 10x `avg_amount_30d`, high-risk MCCs, 1 to 5 transactions within 2 hours |
| GEO_MISMATCH | `issuer_country` BR but foreign `ip_country` with proxy, new account (under 30 days), moderate amounts |
| VELOCITY_BURST | 8 to 20 transactions within 1 hour across many merchants, ordinary amounts |

**Hard negatives** (legitimate traffic that looks suspicious, so the problem is not trivially separable): travelers (foreign `ip_country`, no proxy), big-ticket purchases on a known device (amount 5x average), legitimate bursts (3 to 6 transactions in 30 minutes at one merchant, known device), brand-new accounts with normal behavior. Add mild noise to features so classes overlap. Document all of this in `docs/fraud-patterns.md`.

**Label rules (realistic delays and bias).** About 6% of legitimate and about 40% of fraudulent transactions are declined upstream: they never become payments, and the decision is **not** exposed as a column (it would leak). Only approved fraud can produce a chargeback; about 85% of approved fraud gets `chargeback_reported_at` between 7 and 45 days after `occurred_at` (possibly after the end of the window; that is intended), the rest is never reported. So `is_fraud` is ground truth for analysis only, and a training label must be derived as "chargeback reported on or before a cutoff date". Explain label delay, unreported fraud and selection bias in `docs/fraud-patterns.md` as interview talking points.

**Settlement derivation.** Build `payments` from non-declined transactions, then refunds, chargebacks (FRAUD ones from the label rules), the PSP file with the semantics above, and finally inject breaks into the PSP file and record them in `injected_breaks`.

## API contracts

Use Pydantic v2 models in `api/schemas.py` of each service, with `Field` constraints, `examples`, and enums (`StrEnum`). Register **stub routes** returning `501` with a `ProblemDetails` body (RFC 7807 style: `type`, `title`, `status`, `detail`, `instance`; media type `application/problem+json`) and declare `responses=` so the OpenAPI is complete. Only `/health` really works.

### risk-scoring-api v1
- `POST /v1/score`: request mirrors the feature columns of `transactions` (nested blocks: `merchant {id, mcc}`, `card {bin, issuer_country, is_new_for_merchant}`, `device {id, ip_country (nullable), is_proxy, is_new_for_account}`, `account {id, age_days}`, `behavior {txn_count_1h, txn_count_24h, avg_amount_30d_minor (nullable), seconds_since_last_txn (nullable)}`, plus `transaction_id`, `occurred_at` (timezone-aware), `amount_minor` (int, > 0), `currency`, `channel`, `installments`). Response: `transaction_id`, `score` (0 to 1), `decision` (APPROVE, REVIEW, DECLINE), `model_version`, `reason_codes` (enum list, positive codes only: `NEW_DEVICE`, `PROXY_DETECTED`, `COUNTRY_MISMATCH`, `AMOUNT_ABOVE_NORMAL`, `HIGH_VELOCITY_1H`, `HIGH_VELOCITY_24H`, `NEW_ACCOUNT`, `HIGH_RISK_MCC`, `NEW_CARD_FOR_MERCHANT`), `latency_ms`. Document in the OpenAPI description: idempotent by `transaction_id` (same payload returns the same decision; same id with a different payload returns `409`), and `422` for validation errors.
- `POST /v1/score:batch`: up to 100 items, per-item result or error.
- `GET /v1/model`: `model_version`, `trained_at`, `thresholds {review, decline}`, `metrics` (free-form numeric map).
- `GET /health` (already exists).

### settlement-service v1
- `GET /v1/settlements?merchant_id=&from=&to=&cursor=&limit=`: cursor pagination, `{items: [DailySettlement], next_cursor}`, fields match the `daily_settlement_by_merchant` mart.
- `GET /v1/reconciliation/breaks?date=&type=&cursor=&limit=`: fields match the `reconciliation_breaks` mart.
- `GET /v1/exports/{mart}/{date}`: returns a Parquet file (`application/vnd.apache.parquet`); `mart` is an enum of the three marts; the format is carried by the media type (no `.parquet` suffix).
- `POST /v1/runs` body `{run_date}`: `202` for a new run, `200` with the existing run if that date already has a non-failed run (idempotent by date). `GET /v1/runs/{run_id}`: `Run {run_id, run_date, status (PENDING, RUNNING, DONE, FAILED), requested_at, started_at, finished_at, error}`.

Keep the API field names aligned with the data contracts (same names, same units).

## Step-by-step plan (one commit per step)

1. **shared: data-contract framework**: `paths.py`, loader, validator, FK helper, with tests that prove each violation type is caught (wrong dtype, unexpected or missing column, null in non-nullable, duplicate key, out of range, bad pattern, value not accepted, failing rule). Add `polars` and `pyyaml` to `payments-shared`.
2. **Data contracts**: the 11 YAML files in `contracts/data/`, plus a test that every file loads and is internally consistent (primary key columns exist, accepted types, roles valid).
3. **datagen package**: workspace member, profiles, `universe.py` (accounts, merchants), tests.
4. **Legitimate transactions with point-in-time features**, including hard negatives, tests (contract-valid, features really point-in-time: recompute naively on a small frame and compare).
5. **Fraud episodes and labels**: patterns, label delay and decline rules, tests (realized fraud rate within tolerance, every pattern present in the sample profile, chargeback timing rules).
6. **Settlement inputs**: payments, refunds, chargebacks, PSP file, injected breaks, tests of the fee rule, sign semantics, due-date cutoff and one break per payment.
7. **CLI, manifest, invariants**: `python -m payments_datagen generate --profile sample|full --out DIR [--seed --start-date --days --fraud-rate --break-rate]` and `python -m payments_datagen validate DIR` (contracts, foreign keys, cross-dataset invariants, readable report with a non-zero exit code on failure). Invariants to implement: every FK resolves; FRAUD chargebacks match `chargeback_reported_at`; refunds never exceed payment amounts; every due, captured payment without a PSP SALE record has a MISSING_IN_PSP break; every injected break is explainable from the data; determinism (same seed gives equal frames, different seed differs).
8. **Sample data**: generate `data/samples/` with the `sample` profile, commit it (the whole folder should stay well under 1 MB), add `.gitignore` rules (`data/out/` ignored, `data/samples/` tracked), and a test that the committed sample passes `validate`. Do **not** assert byte equality or frame equality with a regeneration (library versions may change the stream); document that the sample is refreshed with `make data-sample`.
9. **risk-scoring-api contract**: schemas, stub routes, examples in `contracts/examples/risk-scoring/`, tests: examples validate against the models; stub routes return `501` with `application/problem+json`; **alignment test**: take a row from the committed sample `transactions`, map it to a `ScoreRequest` and validate it, and assert that every `role: feature` column of the `transactions` contract is represented in the request model (so the two contracts cannot drift apart silently).
10. **settlement-service contract**: schemas, stub routes, examples, tests (same style; plus a test that the page item models have the same field names and types as the corresponding mart contracts).
11. **OpenAPI export and CI**: `scripts/export_openapi.py` (sorted keys, stable formatting) writing `contracts/openapi/*.json`; Makefile targets `contracts` (export) and `contracts-check` (export, then `git diff --exit-code contracts/`); add a CI step for `contracts-check` and a step that generates the `sample` profile into a temp dir and runs `validate` on it.
12. **Docs and DX**: Makefile targets `data-sample`, `data-full`, `data-validate`; README (quickstart for data, roadmap with Phase 2 checked); `docs/data-model.md` (Mermaid ER diagram plus notes on keys, units, due-date cutoff); `docs/fraud-patterns.md`; update `contracts/README.md`; ADRs: 0003 `payments-datagen` as a workspace package and a single shared universe, 0004 contract-first with YAML data contracts and generated OpenAPI (include the trade-off that contracts are not packaged into Docker images yet), 0005 money as integer minor units, 0006 v1 scoring takes precomputed behavior features and v2 will compute them server-side.

## Definition of done

Run these yourself and show the results:

1. `uv sync --all-packages` succeeds; `uv.lock` committed.
2. `uv run ruff check .` and `uv run ruff format --check .` pass.
3. `uv run pytest` passes for every member, including the new tests, and the whole suite stays fast (the `full` profile is never used in tests; use `sample` and a smaller internal test profile).
4. `make data-sample` is idempotent: running it twice leaves `git status` clean (same environment).
5. `uv run python -m payments_datagen generate --profile full --out data/out/full` completes, and `uv run python -m payments_datagen validate data/out/full` passes. Report the wall-clock time, file sizes, realized fraud rate, break counts by type, and the share of fraud with chargebacks.
6. `make contracts-check` passes, and `docker compose config` and `docker compose build` still work (the images must still build and answer `/health`).
7. `git log --oneline` shows small conventional commits, and `git status` is clean.

## Final report format

Keep it short: what was created (tree), results of the seven checks (pass, fail or not run, with reasons), the realized statistics from the `full` run, any decision not obvious from this prompt, and the suggested next step (Phase 3: risk-scoring-api with a LightGBM baseline, a PyTorch MLP, and a comparison).
