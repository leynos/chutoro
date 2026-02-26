# Execution Plan (ExecPlan): add optional Gaussian clustering quality tracking to CPU benchmarks

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds.

Status: COMPLETE

PLANS.md is not present in this repository, so no additional plan constraints
apply.

## Purpose / big picture

Implement roadmap item `2.1.6` so benchmark runs can optionally emit clustering
quality metrics for synthetic Gaussian data in addition to timing metrics.
Quality is measured against known ground-truth cluster labels using Adjusted
Rand Index (ARI) and Normalized Mutual Information (NMI).

Success is observable when:

- Benchmark timing groups still run under Criterion with no behavioural
  regression.
- An optional quality-report pass computes ARI/NMI for Gaussian synthetic data
  and writes a machine-readable CSV report under `target/benchmarks/`.
- Quality report rows include the timing context (`point_count`, `M`,
  `ef_construction`, and elapsed build time) so quality can be interpreted
  alongside performance.
- Unit tests (including parameterized `rstest` cases) validate metric math,
  report writing, and Gaussian ground-truth handling across happy, unhappy, and
  edge paths.
- Design decisions are documented in `docs/chutoro-design.md`.
- `docs/roadmap.md` marks `2.1.6` as done after implementation is complete.
- `make check-fmt`, `make lint`, and `make test` pass.

## Constraints

- Keep all Rust source files under 400 lines. If a file would exceed 400 lines,
  split into a new module.
- Prefer bench-local integration in `chutoro-benches`; if shared metric logic
  is required, keep `chutoro-core` additions additive and backwards-compatible.
- Keep benchmark timing behaviour stable; quality tracking must be optional and
  must not add overhead to Criterion iteration closures.
- Use deterministic synthetic generation (fixed seed) so quality changes are
  attributable to code changes, not randomness.
- Use `#[expect(..., reason = "...")]` only when strictly necessary and keep
  suppressions tightly scoped.
- Add broad unit-test coverage with `rstest` parameterized cases where repeated
  scenarios exist.
- Use en-GB-oxendict spelling in docs and comments.
- Do not add external dependencies unless no practical in-repo alternative
  exists.

## Tolerances (exception triggers)

- Scope: if implementation requires changes to more than 10 files or more than
  700 net lines, stop and escalate.
- Interface: if a public `chutoro-core` API must change to satisfy this item,
  stop and escalate with alternatives.
- Dependencies: if any new crate dependency is required, stop and escalate.
- Iterations: if `make lint` or `make test` fails after 3 repair attempts,
  stop and escalate with failure logs.
- Ambiguity: if roadmap wording supports materially different interpretations
  (for example, per-case quality versus aggregate quality), stop and present
  options with trade-offs.
- Runtime: if optional quality reporting pushes benchmark setup into unstable
  durations for normal local runs, gate it behind explicit environment
  variables and document defaults.

## Risks

- Risk: ARI/NMI formulas are easy to implement incorrectly, especially for edge
  partitions. Severity: high. Likelihood: medium. Mitigation: add parameterized
  oracle-style unit tests for identity, permutation invariance, finite bounds,
  and degenerate partitions.

- Risk: Gaussian synthetic data currently does not expose explicit labels,
  encouraging brittle implicit assumptions. Severity: medium. Likelihood: high.
  Mitigation: add an explicit labelled-generation helper in benchmark source
  code and test its label contract.

- Risk: benchmark files near the 400-line cap can overflow during integration.
  Severity: medium. Likelihood: medium. Mitigation: place ARI/NMI and reporting
  logic in a dedicated support module instead of growing bench binaries.

- Risk: optional quality pass may run during `--list`/`--exact` discovery and
  slow nextest/bench harness workflows. Severity: medium. Likelihood: medium.
  Mitigation: mirror the existing gating pattern used by memory/recall
  reporting (`--list`, `--exact`, and explicit env var overrides).

## Progress

- [x] (2026-02-25) Drafted ExecPlan for roadmap item `2.1.6`.
- [x] (2026-02-25) Stage A complete: added
  `chutoro-benches/src/clustering_quality.rs` with ARI/NMI computation, typed
  report rows, and CSV writer.
- [x] (2026-02-25) Stage B complete: added
  `SyntheticSource::generate_gaussian_blobs_with_labels(...)` and covered label
  contracts in `chutoro-benches/src/source/tests.rs`.
- [x] (2026-02-25) Stage C complete: integrated optional ARI/NMI reporting into
  `chutoro-benches/benches/hnsw_ef_sweep.rs` as a one-shot setup pass outside
  Criterion timing closures.
- [x] (2026-02-25) Stage D complete: added/updated unit tests, documented
  design decisions in `docs/chutoro-design.md`, marked roadmap item `2.1.6`
  done in `docs/roadmap.md`, and ran required quality gates.

## Surprises & Discoveries

- Observation: ARI/NMI logic was originally duplicated between
  `chutoro-benches/src/clustering_quality.rs` and
  `chutoro-core/tests/functional_ari_nmi.rs`. Evidence: both modules contained
  private implementations. Impact: follow-up review work extracted shared
  metric helpers into `chutoro-core/src/clustering_quality.rs` and rewired both
  call sites.

- Observation: `chutoro-benches/benches/hnsw.rs` and
  `chutoro-benches/src/recall.rs` are already close to the 400-line policy.
  Evidence: current line counts are 344 and 357 respectively. Impact: new
  quality logic should live in a new module.

- Observation: pure Hierarchical Navigable Small World (HNSW) search labels
  are not available as a direct benchmark output in this sweep. Evidence: the
  existing benchmark only measures build timing and optional recall. Impact:
  quality reporting now runs a dedicated setup-only pipeline
  (`build_with_edges` -> mutual reachability MST -> label extraction) so timing
  loops remain unchanged.

## Decision log

- Decision: implement ARI/NMI in a dedicated benchmark-support module (planned
  `chutoro-benches/src/clustering_quality.rs`) and keep bench binaries as thin
  orchestrators. Rationale: this protects file-size limits, improves
  testability, and mirrors the existing `recall` module pattern. Date/Author:
  2026-02-25 (Codex)

- Decision: quality reporting will be optional, executed as a one-shot setup
  pass (not a Criterion measurement target). Rationale: ARI/NMI are
  correctness/quality signals, not variance-sensitive timing metrics; this
  avoids distorting benchmark timing measurements. Date/Author: 2026-02-25
  (Codex)

- Decision: Gaussian generators will expose ground truth explicitly through a
  labelled generation helper instead of relying on hidden assumptions in bench
  code. Rationale: keeps the quality contract explicit and robust against
  future generator changes. Date/Author: 2026-02-25 (Codex)

- Decision: benchmark quality reporting uses explicit env vars
  `CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT` and
  `CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT_PATH`, and defaults to skipping
  report generation when `--list`/`--exact` is present. Rationale: avoids
  surprise overhead during benchmark discovery and nextest list-mode flows.
  Date/Author: 2026-02-25 (Codex)

- Decision: Gaussian ground-truth labels are assigned in round-robin centroid
  order and returned directly from source generation. Rationale: deterministic
  labels provide stable ARI/NMI baselines across runs with fixed seeds.
  Date/Author: 2026-02-25 (Codex)

- Decision: shared ARI/NMI logic now lives in
  `chutoro-core/src/clustering_quality.rs`, with benches and core functional
  tests delegating to that module. Rationale: removes formula drift risk and
  enables one-pass ARI+NMI computation in `clustering_quality_score`.
  Date/Author: 2026-02-26 (Codex)

## Outcomes & retrospective

Implemented outcomes:

- Added optional ARI/NMI benchmark quality reporting for Gaussian synthetic
  data via `chutoro-benches/src/clustering_quality.rs` and
  `chutoro-benches/benches/hnsw_ef_sweep.rs`.
- Extracted shared ARI/NMI computation into
  `chutoro-core/src/clustering_quality.rs`; both benchmark quality helpers and
  `chutoro-core/tests/functional_ari_nmi.rs` now use this shared implementation.
- Added deterministic labelled Gaussian generation and unit tests covering
  shape, determinism, and round-robin label assignment.
- Added parameterized `rstest` coverage for metric happy-path and degenerate
  scenarios, mismatch/unhappy path checks, bounded score checks, and CSV output
  checks.
- Updated `docs/chutoro-design.md` (§11.5) with rationale, configuration, and
  workflow details; marked roadmap item `2.1.6` done in `docs/roadmap.md`.
- Quality gates passed:
  `make check-fmt`, `make lint`, and `make test` (`790 passed, 1 skipped` in
  the final workspace run).

Retrospective:

- Consolidating quality scoring in shared `chutoro-core` helpers eliminated
  duplicated formulas and reduced future drift risk.
- Running quality extraction as setup-only work maintained benchmark timing
  integrity and met the “secondary metrics” requirement cleanly.

## Context and orientation

Benchmark infrastructure is split between:

- `chutoro-benches/benches/` for Criterion entrypoints and timing groups.
- `chutoro-benches/src/` for reusable benchmark support logic.

Relevant existing files:

- `chutoro-benches/benches/hnsw_ef_sweep.rs` already combines timing with an
  optional one-shot recall report pass; this is the closest integration pattern
  for optional quality tracking.
- `chutoro-benches/src/recall.rs` already demonstrates a report model, CSV
  writer, and `rstest` coverage pattern.
- `chutoro-benches/src/source/numeric/mod.rs` generates Gaussian synthetic
  inputs but currently returns only vectors.
- `chutoro-core/tests/functional_ari_nmi.rs` provides proven ARI/NMI formulas
  and behavioural expectations to mirror in benchmark support code.

Roadmap requirement (`docs/roadmap.md`, item `2.1.6`) asks for optional
clustering quality metrics for synthetic Gaussian data as secondary signals
next to timing, guarding against quality regressions during performance tuning.

## Plan of work

### Stage A: add clustering-quality support module

Create a new support module in `chutoro-benches/src/` that contains:

- ARI/NMI calculators with explicit error handling for invalid inputs (for
  example, label-length mismatch).
- A typed record structure for report rows containing timing context and
  quality metrics.
- A CSV writer helper that creates parent directories and writes deterministic
  headers/rows.

Add focused unit tests in the module with `rstest` parameterization.

Go/no-go: stop if clean API boundaries cannot be maintained without changing
`chutoro-core` public APIs.

### Stage B: add Gaussian ground-truth label support

Extend synthetic Gaussian generation support so benchmark code can obtain both:

- the generated `SyntheticSource`, and
- deterministic ground-truth labels for each generated point.

Implementation should preserve the current unlabelled API by delegating to the
new helper so existing benchmarks remain source-compatible.

Add tests for:

- label count equals `point_count`,
- label values are within expected cluster range,
- deterministic labels across repeated runs with the same config,
- unhappy paths inherited from Gaussian config validation.

Go/no-go: stop if this change requires broad rewrites of benchmark callers.

### Stage C: integrate optional quality-report pass into benchmarks

Use the existing optional-report pattern from `hnsw_ef_sweep`:

- add env-var-controlled enabling/disabling for clustering quality reporting,
- skip reporting during `--list` and `--exact` discovery unless explicitly
  enabled,
- write report to `target/benchmarks/` with override env var for output path.

Quality pass should run outside Criterion timed closures and include timing
context from the evaluated benchmark configuration so ARI/NMI are recorded as
secondary metrics alongside performance.

Go/no-go: stop if integration pushes a benchmark source file past 400 lines;
extract integration glue into a helper module.

### Stage D: tests, documentation, roadmap, and quality gates

- Expand unit-test coverage in `chutoro-benches` using `rstest` for broad case
  matrices.
- Update `docs/chutoro-design.md` with a new section under §11 recording:
  - why quality reporting is optional,
  - how ARI/NMI are computed and reported,
  - configuration knobs and expected workflow.
- Mark roadmap item `2.1.6` as done in `docs/roadmap.md` after implementation.
- Run required quality gates and confirm success.

## Concrete steps

From repository root:

```sh
set -o pipefail
make check-fmt 2>&1 | tee /tmp/execplan-2-1-6-check-fmt.log
make lint 2>&1 | tee /tmp/execplan-2-1-6-lint.log
make test 2>&1 | tee /tmp/execplan-2-1-6-test.log
```

Implementation-time benchmark verification commands:

```sh
set -o pipefail
cargo bench -p chutoro-benches --bench hnsw_ef_sweep -- --noplot 2>&1 | tee /tmp/execplan-2-1-6-bench.log
```

Expected observable artefact after feature implementation:

```plaintext
A CSV report under target/benchmarks/ containing ARI/NMI columns and timing
context columns for Gaussian synthetic benchmark configurations.
```

## Validation and acceptance

Done means all of the following are true:

- Benchmark quality report is optional and documented, with deterministic
  defaults.
- ARI/NMI are computed against Gaussian ground truth and emitted with timing
  context.
- Unit tests cover metric happy paths, mismatch/unhappy paths, and edge cases
  (identity/permutation/degenerate partitions) using `rstest` where appropriate.
- Existing benchmark timing behaviour remains intact.
- `docs/chutoro-design.md` records design decisions and usage guidance for this
  feature.
- `docs/roadmap.md` marks item `2.1.6` as completed.
- `make check-fmt`, `make lint`, and `make test` all pass.

## Idempotence and recovery

- Report writing is deterministic and safe to rerun; existing report files are
  overwritten atomically by a full rewrite.
- Optional quality pass can be disabled via env var if a benchmark diagnosis
  needs timing-only runs.
- If quality-gate commands fail, fix only reported issues and rerun failed
  gates until all pass.

## Interfaces and dependencies

Implemented interfaces:

- `chutoro_benches::clustering_quality::adjusted_rand_index(...)`
- `chutoro_benches::clustering_quality::normalized_mutual_information(...)`
- `chutoro_benches::clustering_quality::clustering_quality_score(...)`
- `chutoro_benches::clustering_quality::write_clustering_quality_report(...)`
- `SyntheticSource::generate_gaussian_blobs_with_labels(...)`

Implemented environment controls:

- `CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT`
- `CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT_PATH`

Dependencies: no new crates were added.
