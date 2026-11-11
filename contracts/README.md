# Contracts

This directory will hold the **interfaces between components**, written down so they can be reviewed
and diffed. It is empty for now; content arrives in phase 2 (data generators and contracts).

## What will live here

| Path (planned) | Content |
| --- | --- |
| `contracts/openapi/<service>.json` | OpenAPI document exported from each FastAPI app (`app.openapi()`). |
| `contracts/data/<mart>.md` | Data contract for each Parquet mart: columns, types, nullability, grain, keys, partitioning, freshness. |
| `contracts/examples/<name>.json` | Example request/response payloads, used in docs and tests. |

## Conventions

- **Synthetic only.** Examples and samples never contain real data, real card numbers or secrets.
- **Generated, not hand-edited.** OpenAPI files are exported from the code by a script (and checked in
  CI that the committed file is up to date); hand-written edits would drift.
- **Data contracts are versioned.** A mart's contract states its version. Adding a nullable column is a
  minor change; removing, renaming or changing the type or meaning of a column is a breaking change and
  needs a version bump and a note in the PR.
- **One grain per mart,** stated explicitly (for example "one row per settled transaction per business
  date").
- **Examples are tested.** Payload examples should be validated against the Pydantic models so they
  cannot go stale.
- **Parquet and OpenAPI are the source of truth at the boundary,** not the internal Python classes.
