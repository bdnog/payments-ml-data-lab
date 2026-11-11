# 0004. Contract-first: YAML data contracts and generated OpenAPI

- Status: accepted
- Date: 2026-11-11

## Context

The two services exchange information with the outside world in two ways: HTTP APIs and datasets
(Parquet marts, generated files). Both are interfaces that other people (BI users, the scoring
caller) depend on, so changes to them must be visible and reviewable. We also want the API to be
usable by a client before it is implemented, and we want training data, the scoring request and the
data generator to agree on what a feature is.

## Decision

- **Data contracts are YAML**, in `contracts/data/`, one file per dataset: columns with type,
  nullability, `role` (id, feature, metadata, label_source, analysis_only), description and
  constraints (`min`, `max`, `accepted_values`, `pattern`), plus row-level `rules` written as SQL
  expressions. YAML is the source of truth because it is readable by non-Python people. The loader and
  validator live in `payments_shared.data_contracts`.
- **API contracts are Pydantic models plus stub routes.** Each service declares its v1 request and
  response models in `api/schemas.py` and its routes in `api/routes.py`; routes return `501`
  (`application/problem+json`) until implemented, and declare every status code so the OpenAPI is
  complete from day one.
- **OpenAPI JSON is generated, never edited**: `scripts/export_openapi.py` writes
  `contracts/openapi/*.v1.json` deterministically, and CI runs `make contracts-check`, which fails
  when the committed files differ from what the code produces. The API version is a constant in the
  script, independent of the build version.
- **The contracts are tied together by tests**, not by convention: every `role: feature` column of the
  `transactions` contract must be a field of the scoring request (and the reverse), every row of the
  committed sample must be a valid scoring request, and the settlement page items must have the same
  names, types, nullability and bounds as the mart contracts.
- The `role` of a column is how later phases derive the feature list, instead of hardcoding it.

## Consequences

- A change to a dataset or an API shows up as a diff in `contracts/` and fails CI until it is
  committed, so interface changes cannot slip in unnoticed.
- The source of truth is split: YAML for data, Python models for APIs. The alignment tests are what
  keeps the two sides honest.
- **Contracts are not packaged into the Docker images yet.** The loader finds `contracts/` by walking
  up from the installed package or through `$CONTRACTS_DIR`; inside an image built today neither
  exists. This is fine while the services do not read contracts at runtime. When the settlement service
  starts validating its marts (phase 4), the contracts must be copied into the image or shipped as
  package data, and this decision should be revisited.
- Rules are SQL expressions evaluated with Polars, so they are tied to Polars' SQL dialect.
