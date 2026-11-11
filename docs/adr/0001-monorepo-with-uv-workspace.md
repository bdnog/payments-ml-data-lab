# 0001. Monorepo with a uv workspace

- Status: accepted
- Date: 2026-11-11

## Context

The project models two sides of a payments system: online fraud scoring (`risk-scoring-api`) and
batch settlement (`settlement-service`). Both share schemas (for example the health payload),
logging setup, synthetic data generators and data contracts. The repository is a public portfolio
piece, so a reviewer should be able to clone it, run one command and see everything working.

Options considered:

1. One repository per service, shared code published as a package.
2. One repository, one Python project with all code in a single package.
3. One repository, several independently packaged members in a uv workspace.

## Decision

We will use a single repository organised as a uv workspace (option 3):

- `packages/*` holds libraries (`payments-shared`), `services/*` holds deployable services.
- There is one `uv.lock` at the root, committed, so every member resolves to the same versions.
- The root is a virtual project (`package = false`) that only carries shared tooling config
  (ruff, pytest) and the `dev` dependency group.
- Members use the `src/` layout with `hatchling` as the build backend.
- Members depend on each other through `[tool.uv.sources] ... = { workspace = true }`.
- Docker builds use the repository root as build context and
  `uv sync --frozen --no-dev --package <service>`, so each image contains only its own
  dependency closure.

## Consequences

- One command (`uv sync --all-packages`) sets up everything; lint, tests and CI run on one lockfile.
- Cross-cutting changes (a shared schema, a data contract) land in one commit and one review.
- Heavy dependencies added later (PyTorch, LightGBM) stay scoped to the member that needs them and
  do not leak into the other service's image.
- A single lockfile forces compatible version ranges across members; a conflict must be resolved
  rather than worked around.
- Docker build context is the whole repo, so `.dockerignore` matters and builds copy more files than
  a per-service repo would. Layer caching is coarser (any change under `packages/` or `services/`
  invalidates the install layer).
- Services are not independently versioned or released; if that becomes necessary, this decision
  should be revisited.
