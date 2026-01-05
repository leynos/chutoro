# Replace 3-Node Bidirectionality Harness With Commit-Path Harness

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

PLANS.md is not present in this repository (checked with
`rg --files -g 'PLANS.md'`).

## Purpose / Big Picture

The goal is for the formal verification harness for bidirectional Hierarchical
Navigable Small World (HNSW) edges to follow real production behaviour, not a
simplified path. The current 3-node harness manually enforces reciprocity; the
new harness should run the same commit path that production code uses,
including deferred scrubs and reconciliation. A successful change means the
Kani harness proves the invariant using
`CommitApplicator::apply_neighbour_updates`, unit tests cover happy/unhappy
paths for this commit flow, and the Phase 1 roadmap entry is marked as done.

## Progress

- [x] (2026-01-03 00:00Z) Draft ExecPlan with required sections and commands.
- [x] (2026-01-03 01:05Z) Replace the 3-node Kani harness with a commit-path
  harness.
- [x] (2026-01-03 01:10Z) Add Kani-only helper(s) to drive
  `CommitApplicator::apply_neighbour_updates`.
- [x] (2026-01-03 01:20Z) Add unit tests with `rstest` covering
  happy/unhappy/edge cases.
- [x] (2026-01-03 01:25Z) Update design documentation with decisions taken.
- [x] (2026-01-03 01:28Z) Update `docs/roadmap.md` to mark the entry as done.
- [x] (2026-01-03 01:40Z) Run formatting, linting, and tests with logging.

## Surprises & Discoveries

- Observation: The initial `make test` run hit the default 120s tool timeout,
  so it was rerun with a longer timeout and completed successfully. Evidence:
  `/tmp/make-test.log`.

## Decision Log

- Decision: Use level 1 with `max_connections = 1` in the commit-path harness
  so eviction and deferred scrubs occur in a 3-node graph. Rationale: Level 0
  doubles capacity, preventing eviction with only three nodes. Date/Author:
  2026-01-03 (Codex)
- Decision: Add `apply_commit_updates_for_kani` to drive the full commit path,
  including `apply_new_node_neighbours`, and gate inputs with `kani::assume`.
  Rationale: Keeps Kani aligned with production preconditions while exercising
  deferred scrub logic. Date/Author: 2026-01-03 (Codex)
- Decision: Add a dedicated commit-path test module under
  `chutoro-core/src/hnsw/insert/commit/`. Rationale: Lets tests access
  `CommitApplicator` directly without inflating the executor test module.
  Date/Author: 2026-01-03 (Codex)

## Outcomes & Retrospective

- The 3-node bidirectionality harness now exercises the commit path and
  deferred scrubs, Kani helper wiring is in place, and unit tests cover happy,
  eviction, and error paths.
- Documentation and the Phase 1 roadmap entry were updated to reflect the new
  harness.

## Context and Orientation

The HNSW insertion commit flow lives under `chutoro-core/src/hnsw/insert/`.
`CommitApplicator::apply_neighbour_updates` in
`chutoro-core/src/hnsw/insert/commit.rs` applies trimmed neighbour lists,
reconciles reverse edges, and runs deferred scrubs that clean up one-way edges.
The Kani harnesses are in `chutoro-core/src/hnsw/kani_proofs.rs` and currently
include a 3-node harness that enforces reciprocity manually. A “commit-path
harness” means the harness sets up a small graph, constructs neighbour updates,
then calls the same commit function used in production to drive reconciliation
and deferred scrub logic. Because `CommitApplicator` and its input types are
not public, a small Kani-only wrapper in `chutoro-core/src/hnsw/insert/mod.rs`
is needed to expose the commit path to the harness.

The tests for insertion commit logic live in
`chutoro-core/src/hnsw/insert/executor/tests/mod.rs`. A new test module may be
added under `chutoro-core/src/hnsw/insert/commit/` if it keeps files under the
400-line limit. Use `rstest` fixtures and parameterized cases as described in
`docs/rust-testing-with-rstest-fixtures.md`. Keep functions small and avoid
complex conditionals per
`docs/complexity-antipatterns-and-refactoring-strategies.md`. If new Rustdoc
examples are added, follow `docs/rust-doctest-dry-guide.md` to avoid brittle
doctests.

## Plan of Work

Start by designing a Kani-only helper in `chutoro-core/src/hnsw/insert/mod.rs`
that wraps `CommitApplicator::apply_neighbour_updates`. This helper should take
simple, public inputs (for example, a list of update specs that name the node,
level, and neighbour list) and internally build the `StagedUpdate` and
`FinalisedUpdate` values required by the commit path. Add `kani::assume` guards
for preconditions (nodes exist, levels are valid, neighbour lists are
deduplicated) to keep the state space bounded and consistent with production
assumptions.

Next, replace the existing 3-node harness in
`chutoro-core/src/hnsw/kani_proofs.rs` with a new commit-path harness that:

- Builds a 3-node, two-level graph (level 1) so eviction can occur at
  `max_connections = 1`.
- Seeds neighbour lists so at least one update forces an eviction and thus
  schedules a deferred scrub.
- Applies the commit-path helper to run reconciliation and deferred scrubs via
  `CommitApplicator::apply_neighbour_updates`.
- Asserts the bidirectional invariant using
  `crate::hnsw::invariants::is_bidirectional`.

Then add unit tests for the commit path. Prefer a dedicated test module in
`chutoro-core/src/hnsw/insert/commit/` so the tests can target
`CommitApplicator` directly. Use `rstest` to cover:

- A happy path where updates add reciprocal edges without evictions.
- An edge case where a reverse edge insertion evicts a neighbour and the
  deferred scrub removes the evicted node’s forward edge.
- An unhappy path where an update references a missing node or invalid level
  and returns `HnswError::GraphInvariantViolation`.

Document any design decision in
`docs/adr-002-adoption-of-kani-formal-verification.md` (or
`docs/chutoro-design.md` if it affects architecture). Finally, mark the
relevant Phase 1 roadmap entry as done in `docs/roadmap.md` once the feature is
complete and tests pass.

## Concrete Steps

1. Inspect the current harness and commit path code to identify integration
   points:

    rg -n "verify_bidirectional_links_commit_path_3_nodes|CommitApplicator" -S \
      chutoro-core/src/hnsw

2. Add a Kani-only commit helper in `chutoro-core/src/hnsw/insert/mod.rs`. Keep
   the API minimal and add `kani::assume` guards for preconditions.

3. Replace the 3-node harness in `chutoro-core/src/hnsw/kani_proofs.rs` with a
   commit-path harness and update any helper usage.

4. Add commit-path unit tests using `rstest` (new test module or expand existing
   tests), ensuring each test has clear assertions about reciprocity and
   deferred scrubs.

5. Update design documentation with decisions made in Step 2–4.

6. Update `docs/roadmap.md` to mark the Phase 1 Kani harness bullet as done.

7. Run formatting, linting, and tests with log capture:

    set -o pipefail
    make check-fmt 2>&1 | tee /tmp/make-check-fmt.log

    set -o pipefail
    make lint 2>&1 | tee /tmp/make-lint.log

    set -o pipefail
    make test 2>&1 | tee /tmp/make-test.log

   If documentation changes are made, also run:

    set -o pipefail
    make fmt 2>&1 | tee /tmp/make-fmt.log

    set -o pipefail
    make markdownlint 2>&1 | tee /tmp/make-markdownlint.log

    set -o pipefail
    make nixie 2>&1 | tee /tmp/make-nixie.log

## Validation and Acceptance

- The new Kani harness calls the commit-path helper, which in turn calls
  `CommitApplicator::apply_neighbour_updates`, and `is_bidirectional` asserts
  pass.
- Unit tests using `rstest` cover the happy path, deferred-scrub eviction path,
  and at least one failure path with a meaningful error assertion.
- `make check-fmt`, `make lint`, and `make test` succeed (logs in `/tmp/`).
- Design decision(s) are recorded in the design document.
- The Phase 1 Kani roadmap item in `docs/roadmap.md` is marked done.

## Idempotence and Recovery

All steps are safe to rerun. If a test or lint step fails, fix the reported
issue and rerun the specific command with the same `set -o pipefail | tee`
pattern. If a Kani harness becomes too slow, reduce bounds or add
`kani::assume` constraints and record the decision in the design document.

## Artifacts and Notes

Keep any log files created by the commands in `/tmp/` until the change is
accepted. When citing results in the design document, include the harness name
and the command used to run it.

## Interfaces and Dependencies

- `chutoro-core/src/hnsw/insert/commit.rs`:
  `CommitApplicator::apply_neighbour_updates( final_updates, max_connections, new_node)`
   remains the single commit-path entry point.
- `chutoro-core/src/hnsw/insert/mod.rs`:
  add a `#[cfg(kani)]` helper (for example, `apply_commit_updates_for_kani`)
  that accepts a list of simple update specs, calls
  `CommitApplicator::apply_neighbour_updates`, and applies new-node neighbours.
- `chutoro-core/src/hnsw/kani_proofs.rs`:
  replace `verify_bidirectional_links_3_nodes_1_layer` with
  `verify_bidirectional_links_commit_path_3_nodes` to use the new commit-path
  helper and assert `is_bidirectional`.
- Tests should live under `chutoro-core/src/hnsw/insert/commit/` or existing
  insert executor tests, using `rstest` fixtures and cases.

## Revision note

Updated progress, decisions, and outcomes to reflect the completed
implementation, and clarified the commit-path harness configuration. This
completes the remaining work in the plan.
