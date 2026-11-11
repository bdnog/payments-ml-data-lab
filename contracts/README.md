# Contracts

The **interfaces between components**, written down so they can be reviewed, diffed and enforced.
Nothing here is documentation that can drift: each kind of contract is checked by code.

```
contracts/
├── data/       11 YAML data contracts (source of truth)
├── openapi/    OpenAPI JSON exported from each app (generated)
└── examples/   example request, response and page payloads (validated by tests)
```

## The three kinds

### 1. Data contracts: `data/*.yaml`

One file per dataset, hand-written, human-readable. They cover the eight datasets the generator
produces (`merchants`, `accounts`, `transactions`, `payments`, `refunds`, `chargebacks`,
`psp_settlement`, `injected_breaks`) and the three marts the phase 4 batch will produce
(`daily_settlement_by_merchant`, `reconciliation_breaks`, `fee_summary`).

```yaml
name: transactions
version: 1
description: One row per card transaction attempt, ...
primary_key: [transaction_id]
columns:
  - name: amount_minor
    type: int64            # string | int64 | float64 | bool | date | timestamp_utc
    nullable: false
    role: feature          # id | feature | metadata | label_source | analysis_only
    description: Amount in minor units (centavos).
    constraints: { min: 50, max: 5000000 }   # min | max | accepted_values | pattern
rules:                     # row-level rules, SQL expressions; a row violates when false
  - { name: velocity_monotonic, expr: "txn_count_1h <= txn_count_24h" }
```

`payments_shared.data_contracts` loads and enforces them: `load_contract(name)`,
`validate_dataframe(df, contract)`, `assert_valid(df, contract)`, `check_foreign_key(...)`. It checks
the exact column set and dtypes, nullability, primary key uniqueness, constraints and rules. The `role`
lets code derive the feature list (`role: feature`) instead of hardcoding it.

### 2. API contracts: `openapi/*.v1.json`

The OpenAPI document of each service, **generated from the apps** by `scripts/export_openapi.py`
(`make contracts`). Never edit them by hand. Each service declares its models in `api/schemas.py` and
its routes in `api/routes.py`; routes not implemented yet answer `501` as `application/problem+json`.

CI runs `make contracts-check`: it regenerates the files and fails if they differ from what is
committed (or if a generated file is not committed). Change a model, run `make contracts`, commit the diff.

### 3. Examples: `examples/**`

Concrete payloads for people to read and for tests to use:

- `examples/risk-scoring/`: `score.request.json`, `score.response.json`, `problem.json`
- `examples/settlement/`: `settlements.page.json`, `breaks.page.json`, `run.json`

Tests validate every example against its Pydantic model, and the settlement examples also against the
mart data contracts (as rows), so they cannot go stale.

## How they stay consistent

| What | Enforced by |
| --- | --- |
| Generated data honors its contract | `payments_datagen validate` and the datagen tests |
| Every `role: feature` column is a scoring request field (and the reverse) | alignment test in `risk-scoring-api` |
| Every row of the committed sample is a valid scoring request | same test |
| Settlement page items match the mart contracts (names, types, nullability, bounds) | alignment test in `settlement-service` |
| `openapi/*.json` matches the code | `make contracts-check` in CI |

## Conventions

- **Synthetic only.** No real data, no real card numbers (only 6-digit BINs), no secrets.
- **Money is an integer in minor units**, named `*_minor` ([ADR 0005](../docs/adr/0005-money-as-integer-minor-units.md)).
- **Field names are shared**: the API and the data contract use the same name for the same thing, in the same unit.
- **Versioned.** A data contract states its `version`; the OpenAPI files carry the API version (`v1` in the file
  name). Adding a nullable column or an optional field is compatible. Removing, renaming, retyping or
  changing the meaning of one is breaking: bump the version and say so in the PR.
- **One grain per dataset**, stated in its description.
- Why contracts are not in the Docker images yet: [ADR 0004](../docs/adr/0004-contract-first-with-yaml-data-contracts-and-generated-openapi.md).
