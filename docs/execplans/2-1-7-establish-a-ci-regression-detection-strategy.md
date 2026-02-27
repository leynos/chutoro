# Execution Plan (ExecPlan): establish a benchmark CI regression detection strategy (roadmap 2.1.7)

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds.

Status: COMPLETE

PLANS.md is not present in this repository, so no additional plan constraints
apply.

## Purpose / big picture

Implement roadmap item `2.1.7` by adding an explicit Continuous Integration
(CI) benchmark regression strategy using Criterion baseline comparison, and by
documenting both CI behaviour and the local developer workflow.

Success is observable when:

- Benchmark regression detection strategy is explicit and implemented in CI:
  either per pull request (PR) or scheduled nightly/weekly, with rationale.
- If full baseline comparison is too slow for PR gating, a scheduled workflow
  is present and operational, and the PR path remains fast.
- Developer documentation includes exact commands for saving a baseline and
  running comparison locally.
- Unit tests (with `rstest` parameterization where appropriate) cover new
  regression-policy logic across happy, unhappy, and edge paths.
- Design decisions are recorded in `docs/chutoro-design.md` (§11 extension).
- `docs/roadmap.md` marks item `2.1.7` as done when implementation completes.
- Required quality gates pass: `make check-fmt`, `make lint`, and `make test`.

## Constraints

- Keep all Rust source files under 400 lines; extract helper modules when
  needed.
- Preserve existing CI quality gates in `.github/workflows/ci.yml`.
- Do not make full Criterion timing comparisons a hard PR gate unless runtime
  and noise are shown to be stable within existing CI budgets.
- Use Criterion baseline comparison (`--save-baseline` / `--baseline`) as the
  comparison mechanism rather than inventing a separate measurement format.
- Keep dependency set unchanged unless a new dependency is required and agreed.
- Add unit tests for any new policy/parsing logic; prefer `rstest` for
  parameterized case sets.
- Keep documentation in en-GB-oxendict spelling, with wrapped paragraphs.
- Use Makefile targets where available for validation.

## Tolerances (exception triggers)

- Scope: if implementation requires changes to more than 9 files or more than
  600 net lines, stop and escalate.
- Interface: if a public `chutoro-core` API change is needed, stop and
  escalate.
- Dependencies: if any new crate is required, stop and escalate with options.
- CI runtime: if scheduled benchmark job exceeds 90 minutes consistently on
  GitHub-hosted runners, stop and propose a reduced matrix.
- Iterations: if `make lint` or `make test` fails after 3 repair attempts,
  stop and escalate with logs.
- Ambiguity: if Criterion baseline semantics in CI are unclear enough to permit
  materially different implementations, stop and present alternatives.

## Risks

- Risk: Criterion timing variance on shared CI runners creates false positive
  regression signals. Severity: high. Likelihood: medium. Mitigation: run
  regression detection on schedule, use fixed matrix/seed, and keep PR checks
  to fast correctness/smoke validation.

- Risk: Benchmark job cost becomes too high and reduces CI signal quality.
  Severity: medium. Likelihood: medium. Mitigation: path-filter PR checks,
  scheduled full comparison, and time-budgeted benchmark subsets.

- Risk: Documentation drift between workflow YAML, design doc, and developer
  guide creates confusion. Severity: medium. Likelihood: medium. Mitigation:
  update all three in one atomic change and include acceptance checks that
  reference each file.

- Risk: Regression-policy helper logic is under-tested.
  Severity: medium. Likelihood: low. Mitigation: add `rstest` parameterized
  unit tests for event/policy parsing, default behaviour, invalid inputs, and
  edge values.

## Progress

- [x] (2026-02-27) Drafted ExecPlan for roadmap item `2.1.7`.
- [x] (2026-02-27) Stage A complete: added benchmark CI policy helper module
  in `chutoro-test-support` with `rstest` unit coverage and a gate binary for
  workflow outputs.
- [x] (2026-02-27) Stage B complete: added
  `.github/workflows/benchmark-regressions.yml` with PR smoke checks and
  scheduled/manual baseline comparison jobs.
- [x] (2026-02-27) Stage C complete: updated `docs/chutoro-design.md`
  (§11.6), `docs/developers-guide.md`, and `docs/roadmap.md` (2.1.7 done).
- [x] (2026-02-27) Stage D complete: ran `make fmt`,
  `make markdownlint`, `make nixie`, `make check-fmt`, `make lint`, and
  `make test` (`830 passed, 1 skipped`).

## Surprises & Discoveries

- Observation: benchmark binaries accept `--list` quickly for discovery, but
  `--help` did not produce usable Criterion flag output in this repository's
  bench invocation path. Evidence: local `cargo bench ... -- --list` succeeds;
  `--help` returned "unexpected argument found". Impact: implementation should
  use known-working benchmark invocations in CI and docs, and avoid relying on
  `--help` output shape.

- Observation: current CI already uses a two-tier model for property tests
  (PR lightweight plus weekly exhaustive). Evidence:
  `.github/workflows/property-tests.yml` has PR matrix with constrained budgets
  and a scheduled weekly deep run. Impact: benchmark regression detection
  should mirror this proven pattern.

## Decision log

- Decision: prefer scheduled Criterion baseline comparison (weekly and
  `workflow_dispatch`) over full PR gating. Rationale: full benchmark suites
  are expensive and noise-prone on shared runners, and roadmap text explicitly
  allows scheduled strategy when PR gating is too slow. Date/Author: 2026-02-27
  (Codex)

- Decision: keep PR benchmark validation lightweight and deterministic (build
  and discovery-mode checks only, optionally path-filtered), while reserving
  baseline regression detection for scheduled runs. Rationale: preserves fast
  developer feedback without discarding benchmark CI coverage. Date/Author:
  2026-02-27 (Codex)

- Decision: introduce a small, testable benchmark-CI policy helper module in
  `chutoro-test-support` and cover it with `rstest` instead of encoding all
  policy logic only in YAML. Rationale: satisfies unit-test requirements and
  keeps policy behaviour reviewable and regression-safe. Date/Author:
  2026-02-27 (Codex)

## Outcomes & retrospective

Implemented outcomes:

- Added a shared benchmark regression profile parser at
  `chutoro-test-support/src/ci/benchmark_regression_profile.rs` with
  parameterized unit tests and a new binary gate
  (`src/bin/benchmark_regression_gate.rs`).
- Added behavioural CLI coverage in
  `chutoro-test-support/tests/benchmark_regression_gate_cli.rs`.
- Added `.github/workflows/benchmark-regressions.yml` implementing
  path-filtered PR smoke checks and scheduled/manual baseline comparison.
- Updated benchmark strategy docs in `docs/chutoro-design.md` (§11.6) and
  `docs/developers-guide.md`; marked roadmap item `2.1.7` done.
- Quality gates all passed with logs captured under `/tmp/2-1-7-continue-*`:
  - `make check-fmt`
  - `make lint`
  - `make test` (`830 tests run: 830 passed, 1 skipped`)

## Context and orientation

Current state relevant to this task:

- `.github/workflows/ci.yml` runs format, markdown lint, clippy, tests, and
  coverage, but does not perform benchmark regression comparison.
- `.github/workflows/property-tests.yml` already demonstrates a successful
  PR-light + weekly-deep CI pattern.
- `Makefile` exposes `make bench` (`cargo bench -p chutoro-benches`) and
  standard quality gates (`make check-fmt`, `make lint`, `make test`).
- `docs/chutoro-design.md` now documents the CI regression strategy in §11.6.
- `docs/developers-guide.md` now includes a baseline save/compare developer
  workflow subsection.
- `docs/roadmap.md` now marks item `2.1.7` as done.

Terms used in this plan:

- Criterion baseline comparison: running a benchmark once to save a named
  baseline and then re-running with comparison against that baseline.
- PR-light benchmark validation: quick, deterministic benchmark checks that
  validate harness health without full timing regression gating.
- Scheduled deep benchmark run: nightly/weekly CI job that runs full baseline
  comparison and surfaces regressions outside the PR critical path.

## Plan of work

### Stage A: introduce testable benchmark CI policy logic

Add a new helper module in `chutoro-test-support/src/ci/` (implemented as
`benchmark_regression_profile.rs`) containing pure functions for:

- selecting run mode from CI context (`pull_request`, `schedule`,
  `workflow_dispatch`),
- resolving whether full baseline comparison is required,
- validating supported policy strings and defaults.

Add unit tests in the same module with `rstest` parameterized cases covering:

- happy paths for each event type,
- unhappy paths for unsupported policy values,
- edge paths (empty strings, mixed case, explicit override combinations).

Go/no-go: stop if policy logic cannot be represented as pure functions without
introducing a new dependency.

### Stage B: add benchmark regression workflow

Create `.github/workflows/benchmark-regressions.yml` with:

- scheduled trigger (weekly) and `workflow_dispatch`,
- optional PR path-filtered lightweight job (non-regression comparison),
- full scheduled job that runs baseline save + compare for benchmark suites.

Use deterministic environment configuration (`CARGO_TERM_COLOR`, fixed seeds,
and existing benchmark env controls) and upload benchmark artifacts/logs.

Prefer commands that are already known to work in this repository's bench
harness invocation path.

Go/no-go: stop if CI runtime exceeds tolerance budget or if workflow design
requires weakening existing required checks.

### Stage C: document strategy and developer workflow

Update docs:

1. `docs/chutoro-design.md`
   Add new section `11.6` describing the chosen CI regression strategy,
   rationale, schedule, and operational constraints.

2. `docs/developers-guide.md`
   Add a concrete "Benchmark regression workflow" subsection with baseline
   save/compare commands and expected outputs.

3. `docs/roadmap.md`
   Mark `2.1.7` as done only after workflow, tests, and docs are complete.

Document design decisions explicitly so roadmap text, design notes, and
operational guide remain aligned.

Go/no-go: stop if documentation statements diverge from implemented workflow.

### Stage D: validation and close-out

Run all quality gates and verify benchmark workflow syntax plus local command
behaviour. Capture logs using `tee` and `set -o pipefail`.

If all checks pass, finalize plan sections (`Progress`, `Decision Log`,
`Outcomes & Retrospective`) with exact results and timestamps.

## Concrete steps

Run from repository root (`/home/user/project`):

```sh
set -o pipefail; make check-fmt 2>&1 | tee /tmp/2-1-7-check-fmt.log
set -o pipefail; make lint 2>&1 | tee /tmp/2-1-7-lint.log
set -o pipefail; make test 2>&1 | tee /tmp/2-1-7-test.log
```

Expected transcript (abbreviated):

```plaintext
... cargo fmt --all -- --check
... cargo clippy --all-targets --all-features -- -D warnings
... cargo nextest run --profile ... --all-targets --all-features
... test result: ok
```

Benchmark workflow smoke check commands (local):

```sh
set -o pipefail; cargo bench -p chutoro-benches --bench hnsw_ef_sweep -- --list \
  2>&1 | tee /tmp/2-1-7-bench-list.log
set -o pipefail; cargo bench -p chutoro-benches --bench hnsw_ef_sweep -- \
  --save-baseline ci-reference --noplot 2>&1 | tee /tmp/2-1-7-bench-save.log
set -o pipefail; cargo bench -p chutoro-benches --bench hnsw_ef_sweep -- \
  --baseline ci-reference --noplot 2>&1 | tee /tmp/2-1-7-bench-compare.log
```

Expected transcript (abbreviated):

```plaintext
... Running benches/hnsw_ef_sweep.rs
... hnsw_build_ef_sweep/...: benchmark
... baseline saved (first run)
... change/percent comparison output (comparison run)
```

## Validation and acceptance

Done means all of the following are true:

- CI strategy is explicit:
  scheduled benchmark baseline comparison is implemented and documented, with
  PR behaviour clearly defined.
- Unit-test coverage exists for new benchmark-CI policy logic:
  `rstest` parameterized tests include happy/unhappy/edge paths.
- Developer workflow is reproducible locally with documented baseline save and
  comparison commands.
- Design decisions are recorded in `docs/chutoro-design.md`.
- Roadmap item `2.1.7` is marked done.
- Quality gates pass:
  `make check-fmt`, `make lint`, `make test`.

Quality method:

- Local gate commands listed in "Concrete steps".
- CI workflow YAML validation through GitHub Actions execution.
- Bench logs/artifacts retained for scheduled runs.

## Idempotence and recovery

- Re-running formatting, lint, and tests is safe and idempotent.
- Re-running benchmark save/compare with the same baseline name overwrites the
  local baseline; this is acceptable for iterative development.
- If baseline data is suspected stale, remove local benchmark output under
  `target/criterion/` and regenerate baseline.
- If scheduled CI run fails due runner noise, rerun via `workflow_dispatch`
  before changing thresholds.

## Artifacts and notes

Expected artifacts after implementation:

- `.github/workflows/benchmark-regressions.yml`
- policy helper module and tests under `chutoro-test-support/src/` and
  `chutoro-test-support/tests/`
- updated docs sections in:
  - `docs/chutoro-design.md` (§11.6),
  - `docs/developers-guide.md`,
  - `docs/roadmap.md` (`2.1.7` checked).
- log files under `/tmp/2-1-7-*.log` during local validation.

## Interfaces and dependencies

Implemented interfaces (new, in `chutoro-test-support`):

```rust
pub enum BenchmarkCiEvent {
    PullRequest,
    Schedule,
    WorkflowDispatch,
    Other,
}

pub enum BenchmarkRegressionMode {
    Disabled,
    DiscoveryOnly,
    BaselineCompare,
}

pub fn resolve_regression_mode(
    event: BenchmarkCiEvent,
    policy: BenchmarkCiPolicy,
) -> BenchmarkRegressionMode;
```

No new external dependencies are planned. Use existing workspace crates,
Criterion CLI options, and GitHub Actions primitives.

## Revision note

- 2026-02-27: Initial draft created for roadmap item `2.1.7`, including
  scheduled-baseline CI strategy, test expectations, and documentation/roadmap
  completion steps.
- 2026-02-27: Implemented Stage A-C edits (policy module + tests, workflow,
  docs, roadmap) and marked plan status as `IN PROGRESS` pending final
  quality-gate run results.
- 2026-02-27: Completed Stage D validation and marked plan status
  `COMPLETE` after all gates passed.
