# Expose a public edge-harvesting HNSW insertion API

This ExecPlan (execution plan) is a living document. The sections
`Constraints`, `Tolerances`, `Risks`, `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be
kept up to date as work proceeds.

Status: COMPLETE

`PLANS.md` was not found in the repository root at the time of writing,
so no additional plan-governance file applies.

## Purpose / big picture

Deliver roadmap item `11.1.1` by exposing the edge-harvesting HNSW
insertion path as a public `chutoro-core` API. After this change,
external callers will be able to insert a point into `CpuHnsw` and
receive the harvested `Vec<CandidateEdge>` values that are currently
discarded by `NoopCollector`.

Success is visible when a caller can use the new public API from
outside
`chutoro-core`, the behaviour is covered by unit tests and Rustdoc, the
design document records the final API choice, roadmap item `11.1.1` is
marked done, and `make check-fmt`, `make lint`, and `make test`
succeed.

## Constraints

- Keep scope limited to the HNSW CPU insertion surface in
  `chutoro-core` plus the minimum required design-document, roadmap, and
  ExecPlan updates for roadmap item `11.1.1`.
- Do not change the behaviour or signature of the existing
  `CpuHnsw::insert()` method. It must continue to discard harvested
  edges through `NoopCollector`.
- Reuse the existing insertion pipeline. Do not duplicate insertion
  logic when a thin wrapper or visibility change can reuse
  `insert_with_edges()` and `insert_with_collector()`.
- Do not add new dependencies, new feature flags, or new public data
  types unless escalation is approved.
- Keep file sizes below 400 lines. Split tests or helpers rather than
  growing an existing file into a “Bumpy Road”.
- Add unit coverage with `rstest` where parameterization improves
  coverage and readability.
- Any new public Rustdoc must follow
  `docs/rust-doctest-dry-guide.md`, including hidden setup where
  appropriate and no `unwrap()` in examples outside tests.
- Documentation updates must use en-GB spelling and pass the repository
  Markdown validators.

## Tolerances (exception triggers)

- Scope: if implementation requires changes to more than 8 files or more
  than 300 net lines of code, stop and escalate.
- Interface: if satisfying `11.1.1` requires changing any existing
  public API other than adding exactly one new public insertion method
  (or promoting the existing harvesting method without adding a second
  alias), stop and escalate.
- Dependencies: if a new crate, Cargo feature, or test-only dependency
  is required, stop and escalate.
- Behaviour: if the public API cannot reuse the existing collector path
  and instead requires a second insertion implementation, stop and
  escalate.
- Validation: if `make lint` or `make test` still fails after two repair
  iterations caused by this change, stop and escalate with the failing
  logs.
- Ambiguity: if the expected behaviour for bootstrap insertion, duplicate
  insertion, or edge-return ordering is unclear from existing code and
  design notes, stop and present the alternatives before proceeding.

## Risks

- Risk: the public method name is ambiguous between promoting
  `insert_with_edges` and adding a clearer wrapper such as
  `insert_harvesting`.
  Severity: medium
  Likelihood: medium
  Mitigation: inspect existing public naming conventions, choose one
  public name only, and record the rationale in the design document and
  `Decision Log`.

- Risk: tests that compare harvested edges too literally may become
  brittle because HNSW insertion depends on graph state and random-level
  assignment.
  Severity: medium
  Likelihood: medium
  Mitigation: prefer graph-state parity and edge-invariant checks over
  comparing unrelated builds, and only compare exact edge vectors when
  the setup is fully controlled and deterministic.

- Risk: documentation may drift from the shipped API name if the naming
  decision changes during implementation.
  Severity: low
  Likelihood: medium
  Mitigation: update `docs/chutoro-design.md`, `docs/roadmap.md`, and
  Rustdoc only after the final public signature is settled.

## Progress

- [x] (2026-04-07 15:53Z) Reviewed `docs/roadmap.md`,
  `docs/chutoro-design.md` §12.4, the testing guidance documents,
  and the existing HNSW API and test surface.
- [x] (2026-04-07 16:30Z) Added public `insert_harvesting()` method to
  `CpuHnsw` with full Rustdoc and example.
- [x] (2026-04-07 16:45Z) Added unit tests with `rstest` covering happy
  paths, unhappy paths, and bootstrap edge cases.
- [x] (2026-04-07 17:00Z) Updated `docs/chutoro-design.md` §12.4 and
  marked roadmap item `11.1.1` as done.
- [x] (2026-04-07 17:15Z) All validators pass: `make fmt`,
  `make check-fmt`, `make lint`, `make nixie`, and `cargo test`.

## Surprises & Discoveries

- Observation: the required return type is already public.
  Evidence: `CandidateEdge` is already re-exported from
  `chutoro-core/src/hnsw/mod.rs` and `chutoro-core/src/lib.rs`.
  Impact: the task is an API-surface exposure change, not a type-design
  change.

- Observation: the harvesting path already exists and is used in
  production code.
  Evidence: `CpuHnsw::build_with_edges()` calls
  `EdgeHarvest::from_parallel_inserts()`, which in turn calls the
  private `insert_with_edges()` method.
  Impact: implementation should stay small and should not require a new
  harvesting algorithm.

## Decision Log

- Decision: plan around a thin public harvesting method on `CpuHnsw`
  that returns `Vec<CandidateEdge>` while preserving
  `CpuHnsw::insert()` as-is.
  Rationale: this matches roadmap item `11.1.1`, preserves existing
  caller behaviour, and keeps the implementation aligned with the
  existing collector abstraction.
  Date/Author: 2026-04-07 / assistant

- Decision: prefer a new public method name,
  `insert_harvesting(...) -> Result<Vec<CandidateEdge>, HnswError>`,
  unless code review shows that promoting `insert_with_edges` directly
  is materially cleaner.
  Rationale: `insert_harvesting` is clearer to external callers than a
  helper-style `*_with_edges` name, but the plan keeps one narrow
  fallback to avoid over-committing before implementation.
  Date/Author: 2026-04-07 / assistant

## Outcomes & Retrospective

Work has not started yet. The intended completed state is a minimal
public API addition with broad automated coverage, updated design notes,
and all required validators passing without widening the task into
session-level incremental clustering.

## Context and orientation

The relevant implementation lives in
`chutoro-core/src/hnsw/cpu/mod.rs`. Today that file exposes:

1. `CpuHnsw::insert(...) -> Result<(), HnswError>`, which routes through
   `insert_with_collector(..., &mut NoopCollector)` and intentionally
   discards harvested edges.
2. `CpuHnsw::build_with_edges(...) -> Result<(CpuHnsw, EdgeHarvest),
   HnswError>`, which already uses the private harvesting path during
   batch build.
3. A private `insert_with_edges(...) -> Result<Vec<CandidateEdge>,
   HnswError>` helper that creates a `VecCollector`, calls
   `insert_with_collector`, and returns the harvested edges.

The supporting types are already public:

- `CpuHnsw`
- `CandidateEdge`
- `HnswError`
- `EdgeHarvest`
- `HnswParams`

Those are re-exported from `chutoro-core/src/hnsw/mod.rs` and then from
`chutoro-core/src/lib.rs`, so roadmap item `11.1.1` does not require new
re-export plumbing unless the final method lands behind a different type
or module, which this plan does not expect.

Current test coverage is spread across:

- `chutoro-core/src/hnsw/tests/build.rs`, which already covers duplicate
  insertion and other direct `insert()` behaviour.
- `chutoro-core/src/hnsw/tests/edge_harvest.rs`, which already covers
  `build_with_edges()` and `CandidateEdge` invariants.
- `chutoro-core/src/hnsw/tests/property/`, which covers broader HNSW
  invariants and idempotency.

The design guidance for this task lives in
`docs/chutoro-design.md` §12.4. That section explicitly describes the
current gap: `insert()` uses `NoopCollector`, while
`insert_with_edges()` already returns the candidate edges needed by
future incremental clustering work.

## Plan of work

Stage A: choose the exact public API and lock down red tests.

Inspect the existing public naming style around `CpuHnsw` and then pick a
single public insertion entry point. The default plan is:

```rust
impl CpuHnsw {
    pub fn insert_harvesting<D: DataSource + Sync>(
        &self,
        node: usize,
        source: &D,
    ) -> Result<Vec<CandidateEdge>, HnswError>;
}
```

This method should remain a thin shim over the existing private helper.
If promoting `insert_with_edges` directly is materially simpler and no
less clear, switch to that instead, but do not expose both names in v1.

Before implementation, add or adjust tests so they fail for the missing
public API and clearly describe the desired behaviour:

1. the first inserted node returns an empty harvested-edge vector;
2. later inserts return only in-bounds, finite, non-self candidate
   edges;
3. duplicate insertion returns `HnswError::DuplicateNode`;
4. the harvesting path leaves the graph in the same state as the
   existing `insert()` path when run on identically seeded twin indices;
5. source-level failures such as non-finite distances continue to
   propagate unchanged.

Stage B: implement the smallest possible API exposure.

Change only `chutoro-core/src/hnsw/cpu/mod.rs` unless tests force a
small supporting adjustment elsewhere. Reuse `VecCollector` and the
existing `insert_with_collector()` path. Avoid touching search, graph,
planner, or distance-cache logic.

Add Rustdoc for the new public method with a small example that shows an
external caller creating an index, inserting a point, and receiving
harvested edges. Keep the example focused and hide setup with `#` lines
where that improves readability.

Stage C: harden the behaviour with focused tests.

Extend the direct HNSW test surface instead of creating a sprawling new
test harness. The likely homes are:

- `chutoro-core/src/hnsw/tests/build.rs` for direct per-insert behaviour;
- `chutoro-core/src/hnsw/tests/edge_harvest.rs` for harvested-edge
  invariants and parameterized cases;
- Rustdoc on the new public method for external-API coverage.

Use `rstest` where the same assertions should run across multiple
parameter sets such as seeds, graph sizes, or insertion positions.
Prefer helpers over copy-pasted setup.

Stage D: update design records and close the roadmap item.

Once the method name is final and tests pass, update
`docs/chutoro-design.md` §12.4 to record the chosen public surface and
why it was selected. Then mark roadmap item `11.1.1` as done in
`docs/roadmap.md`. Keep this ExecPlan current by updating `Progress`,
`Decision Log`, `Surprises & Discoveries`, and `Outcomes & Retrospective`.

## Concrete steps

All commands below are run from the repository root:
`/home/leynos/Projects/chutoro.worktrees/11-1-1-make-edge-harvesting-hnsw-insertion-path-public`.

1. Review the implementation and test files before editing.

   ```bash
   leta show chutoro-core/src/hnsw/cpu/mod.rs:CpuHnsw.insert
   leta show chutoro-core/src/hnsw/cpu/mod.rs:CpuHnsw.insert_with_edges
   leta refs CpuHnsw.insert
   ```

2. Add the new public method and Rustdoc in
   `chutoro-core/src/hnsw/cpu/mod.rs`.

3. Add or update direct tests in:

   - `chutoro-core/src/hnsw/tests/build.rs`
   - `chutoro-core/src/hnsw/tests/edge_harvest.rs`

   Expected new test coverage names can be close to:

   - `insert_harvesting_initial_insert_returns_empty_edges`
   - `insert_harvesting_returns_valid_edges`
   - `insert_harvesting_duplicate_insert_is_rejected`
   - `insert_harvesting_matches_insert_graph_state`
   - `insert_harvesting_propagates_distance_errors`

4. Run quick targeted checks during iteration.

   ```bash
   cargo test -p chutoro-core edge_harvest -- --nocapture
   cargo test -p chutoro-core duplicate_insert_is_rejected -- --nocapture
   cargo test -p chutoro-core insert_harvesting -- --nocapture
   ```

5. Update `docs/chutoro-design.md` and `docs/roadmap.md` after the code
   and tests are settled.

6. Run the required validators sequentially and keep logs.

   ```bash
   set -o pipefail && make fmt 2>&1 | tee /tmp/11-1-1-fmt.log
   set -o pipefail && make check-fmt 2>&1 | tee /tmp/11-1-1-check-fmt.log
   set -o pipefail && make markdownlint 2>&1 | tee /tmp/11-1-1-markdownlint.log
   set -o pipefail && make nixie 2>&1 | tee /tmp/11-1-1-nixie.log
   set -o pipefail && make lint 2>&1 | tee /tmp/11-1-1-lint.log
   set -o pipefail && make test 2>&1 | tee /tmp/11-1-1-test.log
   ```

## Validation and acceptance

The change is done only when all of the following are true:

- External callers can invoke the new `CpuHnsw` harvesting method and
  receive a `Vec<CandidateEdge>`.
- `CpuHnsw::insert()` still returns `Result<(), HnswError>` and still
  discards edges.
- Unit tests cover happy paths, duplicate inserts, bootstrap behaviour,
  and propagated error cases.
- The new Rustdoc example compiles and passes.
- `docs/chutoro-design.md` records the chosen API and `docs/roadmap.md`
  marks `11.1.1` as done.
- `make check-fmt`, `make lint`, and `make test` pass.
- Because this task also changes Markdown, `make fmt`,
  `make markdownlint`, and `make nixie` pass as well.

Observable acceptance criteria:

1. A targeted test for the new method passes and shows that a later
   insert returns at least one candidate edge on a tiny dataset.
2. A parity test shows the harvesting method mutates the graph exactly as
   the existing `insert()` method does on the same insertion sequence.
3. A duplicate-insert test confirms the new method returns the same
   `HnswError::DuplicateNode` failure as `insert()`.

## Idempotence and recovery

All implementation and validation steps are safe to re-run. If
formatting changes files, rerun the downstream validators in the same
order. If a targeted test fails, fix the issue locally first, rerun the
targeted check, and only then rerun the full validator chain. Keep the
logs in `/tmp/11-1-1-*.log` for any escalation.

## Artifacts and notes

- Primary code file:
  `chutoro-core/src/hnsw/cpu/mod.rs`
- Primary direct test files:
  `chutoro-core/src/hnsw/tests/build.rs` and
  `chutoro-core/src/hnsw/tests/edge_harvest.rs`
- Required design records:
  `docs/chutoro-design.md` and `docs/roadmap.md`
- Execution log files:
  `/tmp/11-1-1-fmt.log`,
  `/tmp/11-1-1-check-fmt.log`,
  `/tmp/11-1-1-markdownlint.log`,
  `/tmp/11-1-1-nixie.log`,
  `/tmp/11-1-1-lint.log`,
  `/tmp/11-1-1-test.log`

## Interfaces and dependencies

The planned public surface is one of the following, with the first option
preferred:

```rust
pub fn insert_harvesting<D: DataSource + Sync>(
    &self,
    node: usize,
    source: &D,
) -> Result<Vec<CandidateEdge>, HnswError>
```

or, if that proves materially clearer after implementation review:

```rust
pub fn insert_with_edges<D: DataSource + Sync>(
    &self,
    node: usize,
    source: &D,
) -> Result<Vec<CandidateEdge>, HnswError>
```

No new dependencies are planned. The implementation should continue to
rely on:

- `CandidateEdge`
- `HnswError`
- `VecCollector`
- `NoopCollector`
- `insert_with_collector`

## Revision note (2026-04-07)

Initial draft created from roadmap item `11.1.1`, the current HNSW CPU
implementation, and the repository's testing and documentation guidance.
It proposes a minimal public harvesting method, records the main risks and
tolerances, and establishes the approval gate before any implementation
work begins.
