# Repository layout

This document maps important repository paths to their current
responsibilities. Use it alongside the user's guide, developer's guide, and
design document when deciding where a change belongs.

## Top-level directories

- `chutoro-core/` contains the library crate: clustering pipeline logic,
  session state, HNSW graph code, public domain traits, and core test support.
- `chutoro-cli/` contains the command-line interface and CLI-facing integration
  tests.
- `chutoro-providers/` contains optional data providers. `dense/` owns numeric
  dense-vector providers and kernels; `text/` owns text ingestion and distance
  support.
- `chutoro-benches/` contains benchmark harnesses, benchmark data-source
  helpers, and benchmark report generation.
- `chutoro-test-support/` contains shared test and CI helper code that is not
  part of the production library surface.
- `docs/` contains user-facing, maintainer-facing, design, roadmap, ADR, and
  ExecPlan documentation. `docs/execplans/` holds living implementation plans.
- `scripts/` contains repository automation that does not naturally belong in
  a Makefile target, including proof-runner wrappers.
- `verus/` contains Verus proof files and proof-facing models for invariants
  that are checked outside normal Cargo builds.
- `.github/` contains GitHub Actions workflows and repository automation.
- `target/` is generated Cargo build output and must not be edited by hand.
- `.verus/` is generated Verus/tooling output and must not be edited by hand.

## Test locations

- Unit tests usually live next to the code under each crate's `src/` tree.
- Behavioural and integration tests live under crate-local `tests/`
  directories, such as `chutoro-core/tests/`.
- Feature files for BDD tests live under `tests/features/` in the crate that
  owns the behaviour.
- Compile-surface tests use dedicated integration-test binaries under the
  relevant crate's `tests/` directory.

## Core session module

Session code is split by responsibility under `chutoro-core/src/session/`:

- `mod.rs` owns the `ClusteringSession` struct, lightweight read-only
  accessors, public re-exports, and the high-level Rustdoc contract.
- `config.rs` owns `SessionRefreshPolicy` and `SessionConfig`, the small value
  types carried by each session.
- `session_impl.rs` owns construction, `append`, HNSW error mapping, and the
  edge-harvesting write path.
- `core_distance.rs` owns the pure core-distance helpers and the recompute
  workflow. The pure helpers must not depend on HNSW adapter internals beyond
  the public `Neighbour` value.
- `clock.rs` is compiled only with the `metrics` feature and owns the
  monotonic-clock support used for deterministic latency tests.
