# 0002. settlement-service has `batch` and `api` modules around a shared `domain`

- Status: accepted
- Date: 2026-11-11

## Context

Settlement and reconciliation is naturally a batch workload: it processes a business day of
transactions and produces results. Those results must also be available to BI tools and to people
who query them, which is a read-mostly, online concern. The two have different lifecycles: the batch
runs on a schedule and can use heavy columnar processing (Polars); the API must stay up, be light,
and answer quickly.

We want one place for the settlement rules, but we do not want the API process to depend on the
batch machinery, or the batch to depend on a web framework.

## Decision

`settlement-service` is one distributable with three layers, in one package (`settlement`):

- `domain`: pure settlement and reconciliation rules. It must not import Polars, FastAPI or
  storage code. It is the shared core.
- `batch`: the batch execution mode, a CLI (`python -m settlement.batch run --date YYYY-MM-DD`)
  that reads inputs, applies `domain` rules (using Polars for the heavy lifting) and writes results.
- `api`: the read-mostly FastAPI application that exposes results. It reads them; it does not
  compute them.
- `storage`: persistence of results, with Parquet marts as the format.

The Parquet marts are the contract between `batch` and `api`: the batch writes them, the API and BI
tools read them. The two modes never call each other. Both modes ship in one Docker image; the default
command starts the API, and the batch is run by overriding the command (in Compose, via the `batch`
profile).

## Consequences

- Business rules are testable without Polars, HTTP or files, and are reused by both modes.
- The batch and the API can be deployed, scaled and failed independently; a failed batch run does
  not take the API down.
- The marts' schema becomes a public interface and needs documented, versioned data contracts
  (see `contracts/`). Changing a column is a breaking change for BI consumers.
- The API can show data that is up to one batch run old; it is not a real-time view.
- One image carries Polars even when it only runs the API. We accept the extra size to keep a single
  build and a single set of dependencies; splitting the image is possible later if it matters.
- The `domain` import rule is enforced by convention and review for now; an import-linter check
  could enforce it automatically later.
