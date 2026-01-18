# Phase 1: Nightly slow continuous integration (CI) job for Kani full runs

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds.

Status: COMPLETE

PLANS.md is not present in this repository, so no additional plan constraints
apply.

## Purpose / Big Picture

Provide a nightly GitHub Actions job that runs `make kani-full` only when the
`main` branch has new commits in the last 24 hours (Coordinated Universal Time
(UTC)), while keeping normal `make test` usage unchanged so Kani remains opt-in
for developers. The job must be manually triggerable for verification. Success
is observable when the scheduled workflow skips on days with no recent commits,
runs `make kani-full` when there are commits within the last 24 hours (UTC),
and a manual dispatch runs the job regardless of commit freshness.

## Constraints

- Do not change the behaviour of `make test` or any default developer loop.
- Do not add Kani to the existing PR CI flow in `.github/workflows/ci.yml`.
- Keep `make kani-full` as the single source of truth for the full Kani run.
- Follow workspace lint rules (no new warnings, no clippy suppressions).
- Every new module must start with a `//!` module doc comment.
- Documentation edits must follow the Markdown rules and be wrapped at 80
  columns.

## Tolerances (Exception Triggers)

- Scope: if this requires changes to more than 10 files or more than 400 net
  lines of code, stop and ask for confirmation.
- Interfaces: if a public API must change in an existing crate, stop and
  confirm the intended surface.
- Dependencies: adding a new external crate beyond adding `rstest` as a
  dev-dependency is not allowed without explicit approval.
- CI design: if a GitHub Actions change requires more than one new workflow
  file or alters the existing `ci.yml` job graph, stop and confirm.
- Tests: if `make test` still fails after two fix attempts, stop and ask for
  guidance.

## Risks

- Risk: The definition of "last 24 hours" can be ambiguous across time zones.
  Severity: medium. Likelihood: medium. Mitigation: define the window using
  Coordinated Universal Time (UTC) epoch seconds and document the choice in the
  design document.
- Risk: The nightly workflow may skip even though a commit landed just before
  midnight local time. Severity: low. Likelihood: medium. Mitigation: use UTC
  and document the behaviour; allow manual override.
- Risk: Kani runs could exceed the default GitHub Actions timeout.
  Severity: medium. Likelihood: low. Mitigation: set an explicit timeout and
  capture logs for analysis.

## Progress

- [x] (2026-01-17 13:10Z) Reviewed CI workflow and Kani targets.
- [x] (2026-01-17 13:35Z) Implemented nightly gate helper with `rstest`
  coverage.
- [x] (2026-01-17 13:55Z) Added nightly schedule + manual workflow.
- [x] (2026-01-17 14:05Z) Updated design documentation and roadmap entry.
- [x] (2026-01-17 14:40Z) Ran `make fmt`, `make markdownlint`, `make nixie`,
  `make check-fmt`, `make lint`, and `make test` with logs in `/tmp`.

## Surprises & Discoveries

- Observation: `cargo nextest` was not installed, so `make test` failed.
  Evidence: `cargo nextest` missing command error. Impact: installed
  `cargo-nextest` v0.9.114 (Rust 1.88 compatible) before rerunning `make test`.

## Decision Log

- Decision: Create a dedicated nightly workflow instead of modifying
  `.github/workflows/ci.yml`. Rationale: isolates slow Kani runs from PR CI and
  keeps existing flow unchanged. Date/Author: 2026-01-16 (Codex)
- Decision: Implement gating logic as a Rust helper in
  `chutoro-test-support`, with unit tests using `rstest`. Rationale: enables
  deterministic, parameterized tests and keeps CI logic versioned with the
  repo. Date/Author: 2026-01-16 (Codex)
- Decision: Gate on commits within the last 24 hours (UTC) instead of strict
  UTC day boundaries. Rationale: avoids skipping commits that land after the
  cron trigger. Date/Author: 2026-01-18 (Codex)
- Decision: Default manual `workflow_dispatch` runs to `force_run = true`.
  Rationale: ensures manual verification runs always execute without requiring
  a recent commit. Date/Author: 2026-01-17 (Codex)

## Outcomes & Retrospective

Delivered a gated nightly Kani workflow, a Rust-based gate helper with
parameterized tests, and updated documentation and roadmap entries. The
required formatting, lint, and test gates passed. Next time, preinstall
`cargo-nextest` or document the required version alongside the toolchain.

## Context and Orientation

The current CI workflow lives in `.github/workflows/ci.yml` and runs format,
lint, tests, and coverage on pull requests and manual dispatch. The Makefile
already defines `make kani-full` as the full Kani verification command. The
roadmap entry for this task is in `docs/roadmap.md` under Phase 1. Design
updates should be recorded in `docs/chutoro-design.md`. Test patterns using
`rstest` are documented in `docs/rust-testing-with-rstest-fixtures.md`, and
Markdown rules are defined in `AGENTS.md` and `docs/rust-doctest-dry-guide.md`.

## Plan of Work

Stage A: Validate the current CI and Kani entry points. Confirm how
`.github/workflows/ci.yml` is structured, verify the existing `make kani-full`
command, and locate the Phase 1 roadmap entry. No code changes in this stage.
Stop if a dedicated workflow conflicts with existing CI conventions.

Stage B: Add a small, testable Rust helper that decides whether the nightly run
should proceed. The helper should accept the commit timestamp and the current
time as inputs so `rstest` can cover within-window, outside-window, and
future-commit cases, as well as manual override behaviour. A thin binary
wrapper should expose the decision to GitHub Actions, writing `should_run` and
`reason` to `$GITHUB_OUTPUT` when present. Validation at this stage is
`cargo test -p chutoro-test-support` and the new unit tests must fail before
and pass after the implementation.

Stage C: Create a new workflow (for example,
`.github/workflows/nightly-kani.yml`) triggered by `schedule` and
`workflow_dispatch`. The workflow should check out `main`, run the nightly
helper, and only run `make kani-full` when the helper says to proceed or when a
manual dispatch explicitly forces the run. Keep permissions minimal and add a
job timeout suitable for Kani. Validation is a dry run via `workflow_dispatch`
with a forced run and confirmation that the workflow skips when the helper
returns `should_run=false`.

Stage D: Update `docs/chutoro-design.md` with an implementation update noting
the nightly Kani job, the rolling 24-hour rule, and the manual override. Mark
the roadmap entry in `docs/roadmap.md` as done. Run `make fmt`,
`make markdownlint`, `make nixie`, `make check-fmt`, `make lint`, and
`make test` with logged output.

## Concrete Steps

1. Read `.github/workflows/ci.yml`, `Makefile`, and the Phase 1 entry in
   `docs/roadmap.md` to confirm the integration points.
2. Add a new module under `chutoro-test-support/src/ci/` with a pure function
   such as:

   The function signature should be:

       fn should_run_kani_full(
           commit_epoch: u64,
           now_epoch: u64,
           force: bool,
       ) -> Result<NightlyDecision, NightlyGateError>

   The function should compare a rolling 24-hour window using a 86,400-second
   cutoff derived from Unix epoch seconds and return a decision plus a reason
   string.
3. Add `rstest`-based unit tests in the same module covering:

   - commit time equals now time (run)
   - commit time outside the last 24 hours (skip)
   - commit time is in the future (error)
   - force override true regardless of time
   - boundary cases around the 24-hour cutoff

4. Add a small binary under `chutoro-test-support/src/bin/` that reads
   the HEAD commit epoch via `git log -1 --format=%ct`, gets the current epoch
   via `SystemTime`, applies the helper, and writes outputs for GitHub Actions.
   Avoid `unwrap`/`expect`; propagate errors via `Result`.
5. Add a new workflow file for the nightly job that:

   - triggers on a daily cron (pick a UTC time) and `workflow_dispatch`
   - checks out `main`
   - runs the helper and sets outputs
   - runs `make kani-full` only when `should_run=true`
   - supports an input like `force_run: true` to override the gate

6. Update `docs/chutoro-design.md` with a dated implementation update
   describing the nightly Kani job, the rolling 24-hour rule, and the manual
   override.
7. Update `docs/roadmap.md` to mark the nightly Kani entry as done.
8. Run the required formatting, lint, and test commands with `tee` logging.

Use the following command pattern to preserve exit codes and capture logs:

    set -o pipefail
    make fmt 2>&1 | tee /tmp/make-fmt.log

    set -o pipefail
    make markdownlint 2>&1 | tee /tmp/make-markdownlint.log

    set -o pipefail
    make nixie 2>&1 | tee /tmp/make-nixie.log

    set -o pipefail
    make check-fmt 2>&1 | tee /tmp/make-check-fmt.log

    set -o pipefail
    make lint 2>&1 | tee /tmp/make-lint.log

    set -o pipefail
    make test 2>&1 | tee /tmp/make-test.log

## Validation and Acceptance

The change is complete when all of the following are true:

- The nightly workflow runs `make kani-full` only when the helper indicates a
  commit within the last 24 hours, and manual dispatch can force the run.
- The helper has `rstest` unit coverage for happy paths, unhappy paths, and
  edge cases.
- `make check-fmt`, `make lint`, and `make test` succeed.
- Markdown checks (`make fmt`, `make markdownlint`, `make nixie`) pass.
- The design document records the gating decision.
- The Phase 1 roadmap entry is marked as done.

## Idempotence and Recovery

All steps are safe to rerun. If the nightly workflow fails, rerun the helper
locally to inspect the decision output. If any quality gate fails, fix the
reported issues and rerun the specific command with the same `pipefail` + `tee`
pattern.

## Artifacts and Notes

Keep `/tmp/make-*.log` files until the change is accepted. Capture the helper
stdout and the GitHub Actions job summary as evidence that the gate is working
as intended.

## Interfaces and Dependencies

- New helper module: `chutoro_test_support::ci::nightly_gate` (exact module
  name to be finalized), exposing `should_run_kani_full` and a
  `NightlyDecision` struct containing `should_run: bool` and `reason: String`.
- New binary: `chutoro-test-support/src/bin/kani_nightly_gate.rs` invoked from
  the workflow, writing `should_run` and `reason` to `GITHUB_OUTPUT` when
  present.
- Workflow inputs: `force_run` boolean (manual only) mapped to an environment
  variable such as `CHUTORO_KANI_FORCE`.
- No new dependencies beyond adding `rstest` as a dev-dependency for
  `chutoro-test-support`.

## Revision note

2026-01-17: Updated status to COMPLETE, recorded progress, discoveries, and
validation results, and added the `cargo-nextest` installation note. No
remaining work is pending.

2026-01-17: Expanded acronyms for CI and UTC to follow documentation rules.
2026-01-18: Adjusted the plan for review feedback covering workflow
concurrency, gate skew handling, the rolling 24-hour gate, and behavioural test
coverage.
