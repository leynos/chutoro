# Phase 1: Property-test CI integration for CPU HNSW, candidate edge harvest, and parallel Kruskal MST

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds.

Status: COMPLETE

PLANS.md is not present in this repository, so no additional plan constraints
apply.

## Purpose / Big Picture

Integrate property-based test suites into Continuous Integration (CI) with a
path-filtered pull request (PR) job and a weekly deep-coverage job, as required
by `docs/roadmap.md` Phase 1 and `docs/property-testing-design.md` §5. The PR
job must run with 250 cases and a 10-minute timeout. The weekly job must run
with 25,000 cases, `fork = true`, and `PROGTEST_CASES` provided by the workflow
environment.

The implementation must also tune CI guardrails by setting a concrete
`CHUTORO_HNSW_PBT_MIN_RECALL` floor and by preparing broader `max_connections`
guardrails beyond the current `max_connections >= 16` search assumption.
Success is observable when CI executes the targeted property suites with the
configured run profile, unit tests cover happy and unhappy paths plus edge
cases using `rstest`, design decisions are recorded in
`docs/chutoro-design.md`, and the relevant roadmap item is marked done.

## Constraints

- Keep existing default CI flow in `.github/workflows/ci.yml` intact for fmt,
  lint, coverage, and Verus proofs.
- Use Makefile targets for gates and preserve command exit status with
  `set -o pipefail` and `tee`.
- Keep property-test tuning deterministic and explicit via environment-backed
  configuration.
- Add/extend unit tests with broad `rstest` parameterization for configuration
  parsing and guardrail predicates, covering happy and unhappy paths and edge
  cases.
- Record implementation decisions in `docs/chutoro-design.md`.
- Do not mark the roadmap task complete until all acceptance criteria in this
  plan are met.

## Tolerances (Exception Triggers)

- Scope: if delivery needs more than 12 files or more than 700 net lines,
  stop and confirm.
- Interfaces: if public non-test APIs in `chutoro-core` must change, stop and
  confirm.
- Dependencies: if a new crate is required, stop and confirm before adding it.
- CI runtime: if the PR property job cannot stay within 10 minutes after two
  optimization passes, stop and present options.
- Flakiness: if guardrail thresholds fail nondeterministically across three
  reruns with identical configuration, stop and escalate with evidence.

## Risks

- Risk: `PROGTEST_CASES` is not currently wired into all property suites.
  Severity: high. Likelihood: high. Mitigation: centralize suite profile
  parsing and route all relevant proptest configurations through it.
- Risk: A strict recall floor causes noisy PR failures.
  Severity: medium. Likelihood: medium. Mitigation: baseline the current recall
  distribution, set an initial CI floor, and document the uplift trigger.
- Risk: Weekly 25,000-case runs exceed timeout.
  Severity: medium. Likelihood: medium. Mitigation: run only targeted suites,
  use `fork = true`, and capture failure artifacts for replay.
- Risk: Expanding `max_connections` coverage exposes existing connectivity gaps.
  Severity: medium. Likelihood: medium. Mitigation: keep a documented staged
  policy and add explicit unit coverage for guard predicates.

## Progress

- [x] (2026-02-09 20:25Z) Drafted ExecPlan with required sections, file
  targets, and acceptance criteria.
- [x] (2026-02-10 00:25Z) Finalised CI guardrails:
  `CHUTORO_HNSW_PBT_MIN_RECALL=0.60` and
  `CHUTORO_HNSW_PBT_MIN_MAX_CONNECTIONS=12`.
- [x] (2026-02-10 00:35Z) Implemented shared property-run profile support for
  `PROGTEST_CASES` and fork mode in `chutoro-test-support`.
- [x] (2026-02-10 00:45Z) Integrated path-filtered PR property workflow
  (250 cases, 10-minute timeout).
- [x] (2026-02-10 00:45Z) Integrated weekly scheduled property workflow
  (25,000 cases, `fork=true`) with failure artifact upload.
- [x] (2026-02-10 00:55Z) Added and passed `rstest` coverage for profile and
  guardrail configuration happy/unhappy/edge cases.
- [x] (2026-02-10 01:05Z) Updated design and roadmap docs.
- [x] (2026-02-10 01:15Z) Ran full quality gates and verified logs in `/tmp`.

## Surprises & Discoveries

- Observation: `.github/workflows/ci.yml` has no dedicated property-test job
  and no path-filtered property execution. Evidence: direct workflow
  inspection. Impact: requires a new workflow for this feature.
- Observation: `PROGTEST_CASES` is referenced in design docs and roadmap but is
  not consumed by current property suites. Evidence: repository search found no
  code usage. Impact: implementation must add explicit parsing/plumbing before
  CI wiring.
- Observation: HNSW search property currently filters to
  `max_connections >= 16`. Evidence:
  `chutoro-core/src/hnsw/tests/property/search_property.rs`. Impact: guardrail
  expansion must be staged and documented.

## Decision Log

- Decision: Add a dedicated workflow file
  `.github/workflows/property-tests.yml` instead of mutating
  `.github/workflows/ci.yml`. Rationale: isolates expensive property runs while
  preserving existing CI behaviour. Date/Author: 2026-02-09 (Codex)
- Decision: Introduce a shared proptest runtime profile reader that maps
  `PROGTEST_CASES` and fork mode into suite configs used by HNSW, edge-harvest,
  and MST property tests. Rationale: prevents drift and ensures weekly/PR jobs
  use one policy surface. Date/Author: 2026-02-09 (Codex)
- Decision: Set initial CI recall floor to `CHUTORO_HNSW_PBT_MIN_RECALL=0.60`,
  with an explicit follow-up to raise it after high-fan-out search work lands.
  Rationale: strengthens enforcement above the current default while remaining
  practical for PR stability. Date/Author: 2026-02-09 (Codex)

## Outcomes & Retrospective

Implemented the planned property CI integration end-to-end:

- Added `.github/workflows/property-tests.yml` with path-filtered PR runs and
  weekly deep coverage runs.
- Centralised property run-profile parsing in
  `chutoro-test-support/src/ci/property_test_profile.rs`.
- Wired HNSW, candidate edge harvest, and MST suites to use
  `PROGTEST_CASES` and fork-mode overrides.
- Added configurable HNSW max-connection guardrails via
  `CHUTORO_HNSW_PBT_MIN_MAX_CONNECTIONS` and recorded guardrail decisions.
- Updated `docs/chutoro-design.md` and marked the roadmap entry done in
  `docs/roadmap.md`.

All required gates passed:

- `make fmt`
- `make markdownlint`
- `make nixie`
- `make check-fmt`
- `make lint`
- `make test`

## Context and Orientation

The current repository has two workflows: `.github/workflows/ci.yml` and
`.github/workflows/nightly-kani.yml`. The property suites already exist in
`chutoro-core`, but they use a mix of fixed case counts and local config, and
CI does not yet run them as first-class jobs.

Key code and doc anchors:

- `.github/workflows/ci.yml`
- `.github/workflows/nightly-kani.yml`
- `chutoro-core/src/hnsw/tests/property/tests.rs`
- `chutoro-core/src/hnsw/tests/property/search_config.rs`
- `chutoro-core/src/hnsw/tests/property/search_property.rs`
- `chutoro-core/src/hnsw/tests/property/edge_harvest_output/mod.rs`
- `chutoro-core/src/hnsw/tests/property/edge_harvest_suite/connectivity.rs`
- `chutoro-core/src/hnsw/tests/property/edge_harvest_suite/degree_ceiling.rs`
- `chutoro-core/src/hnsw/tests/property/edge_harvest_suite/rnn_uplift.rs`
- `chutoro-core/src/mst/property/tests.rs`
- `docs/chutoro-design.md`
- `docs/property-testing-design.md`
- `docs/roadmap.md`

Terminology used in this plan:

- Path-filtered PR job: a PR-triggered workflow that runs only when relevant
  files change.
- Weekly deep job: a scheduled workflow with a much larger proptest case budget
  and process forking enabled.
- Guardrails: CI-enforced thresholds such as minimum recall and valid
  max-connection ranges.

## Plan of Work

Stage A: confirm baseline and finalize guardrails (no code changes). Measure
current runtime and recall behaviour for existing property suites and finalize
CI values for recall floor and max-connection policy. Go/no-go: do not start
code edits until thresholds are explicit and documented in this plan.

Stage B: add shared configuration plumbing and tests. Implement a shared
test-profile helper that reads `PROGTEST_CASES` and fork mode, then route HNSW,
edge-harvest, and MST property runners through it. Add `rstest` unit coverage
for parser happy/unhappy/edge cases and HNSW guardrail predicates. Go/no-go:
all new/changed unit tests pass locally before CI edits.

Stage C: integrate CI workflows for PR and weekly schedules. Add a
path-filtered PR job and a scheduled weekly job in
`.github/workflows/property-tests.yml` using the agreed profiles: PR = 250
cases, 10-minute timeout; weekly = 25,000 cases, `fork=true`,
`PROGTEST_CASES=25000`. Include failure artifact upload for proptest
regressions and logs. Go/no-go: workflow syntax validates and targeted local
commands match workflow commands.

Stage D: documentation, roadmap update, and final gates. Record design
decisions and guardrail values in `docs/chutoro-design.md`, then mark the
relevant Phase 1 roadmap entry done in `docs/roadmap.md` only after all gates
pass. Run `make fmt`, `make markdownlint`, `make nixie`, `make check-fmt`,
`make lint`, and `make test` with logged output.

## Concrete Steps

1. Add shared profile support in test-support code and expose stable helpers.
   Candidate interface:

    pub struct ProptestRunProfile {
        pub cases: u32,
        pub fork: bool,
    }

    pub fn load_profile(default_cases: u32, default_fork: bool)
        -> ProptestRunProfile

2. Wire profile usage into property suites that currently hardcode case counts:
   `chutoro-core/src/hnsw/tests/property/tests.rs`,
   `chutoro-core/src/hnsw/tests/property/edge_harvest_output/mod.rs`,
   `chutoro-core/src/hnsw/tests/property/edge_harvest_suite/connectivity.rs`,
   `chutoro-core/src/hnsw/tests/property/edge_harvest_suite/degree_ceiling.rs`,
   `chutoro-core/src/hnsw/tests/property/edge_harvest_suite/rnn_uplift.rs`,
   `chutoro-core/src/mst/property/tests.rs`.
3. Extend HNSW search guardrail configuration tests in
   `chutoro-core/src/hnsw/tests/property/search_config.rs` and
   `chutoro-core/src/hnsw/tests/property/search_property.rs` with `rstest`
   cases for: happy paths (valid recall and max-connection policies), unhappy
   paths (invalid env values), and edge conditions (boundary values).
4. Add `.github/workflows/property-tests.yml` with:
   PR trigger with path filters, 10-minute timeout, and `PROGTEST_CASES=250`,
   `CHUTORO_HNSW_PBT_MIN_RECALL=0.60`; weekly schedule trigger with
   `PROGTEST_CASES=25000` and `fork=true`; artifact upload on failure for
   `**/proptest-regressions/**`.
5. Update `docs/chutoro-design.md` with an implementation update capturing:
   chosen recall floor, weekly profile, and staged max-connections policy.
6. After all validations pass, mark the roadmap item at
   `docs/roadmap.md` lines 168-176 as done.
7. Run validation commands with logging:

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

Expected signals of success:

- Property workflow appears on PRs that touch targeted paths.
- PR workflow completes within 10 minutes and runs with 250 cases.
- Weekly workflow runs with `PROGTEST_CASES=25000` and forking enabled.
- Roadmap checkbox is updated only at feature completion.

## Validation and Acceptance

The feature is complete only when all statements below are true:

- Path-filtered PR property job exists and uses 250 cases with a 10-minute
  timeout.
- Weekly scheduled property job exists and uses 25,000 cases with
  `fork = true` and `PROGTEST_CASES`.
- CI enforces a concrete `CHUTORO_HNSW_PBT_MIN_RECALL` value and documents the
  future uplift trigger.
- Max-connections guardrail policy is broadened and documented, with test
  coverage for configured behaviour.
- New/changed unit tests use `rstest` and cover happy/unhappy/edge cases.
- Design decisions are recorded in `docs/chutoro-design.md`.
- `docs/roadmap.md` Phase 1 property-test CI item is marked done.
- `make check-fmt`, `make lint`, and `make test` pass.
- Markdown gates (`make fmt`, `make markdownlint`, `make nixie`) pass.

## Idempotence and Recovery

All changes are additive and safe to rerun. If a workflow or test step fails,
fix the failure in place and rerun the same command/workflow with identical
environment variables. Keep `/tmp/make-*.log` files until the feature is
accepted so failure context remains available.

## Artifacts and Notes

- Preserve `/tmp/make-*.log` outputs for all quality gates.
- On CI failures, collect `proptest-regressions` artifacts and include failing
  seeds in the implementation notes.
- Capture the final workflow run links and include them in the completion
  update for traceability.

## Interfaces and Dependencies

- New workflow: `.github/workflows/property-tests.yml`.
- Shared test-profile helper in test-support, used by HNSW and MST property
  suites.
- Environment keys:
  `PROGTEST_CASES`, `CHUTORO_HNSW_PBT_MIN_RECALL`, and fork-mode control.
- No new external Rust crate dependencies expected.

## Revision note

2026-02-09: Initial draft created to scope Phase 1 CI integration for property
tests, including staged implementation, guardrails, and acceptance criteria.

2026-02-10: Completed implementation. Updated status/progress, recorded the
final guardrail values, and captured successful validation gates.
