# Phase 1: Verus Proofs for Edge Harvest Primitives

This execution plan (ExecPlan) is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

## Purpose / Big Picture

Add Verus proofs for the edge harvest helper invariants listed in
`docs/property-testing-design.md` Appendix A. The focus is on deterministic,
helper-level proofs only: `extract_candidate_edges`,
`CandidateEdge::canonicalise`, and `EdgeHarvest::from_unsorted`. Proofs must
avoid `assume` shortcuts, keep scope limited to helper invariants (no
concurrency or planner proofs), and run in CI with a pinned Verus toolchain.
Unit tests (parameterized with `rstest`) will validate happy and unhappy paths
around the helper logic. On completion, the relevant Phase 1 roadmap entry in
`docs/roadmap.md` is marked as done.

## Progress

- [ ] Draft ExecPlan with required sections and commands.
- [ ] Confirm helper signatures are stable and document any constraints.
- [ ] Pin and document the Verus toolchain for contributors.
- [ ] Add Verus proof harnesses for the Appendix A invariants.
- [ ] Add unit tests with `rstest` covering happy, unhappy, and edge cases.
- [ ] Update design documentation with decisions made.
- [ ] Ensure CI runs Verus proofs with the pinned toolchain.
- [ ] Update `docs/roadmap.md` to mark the entry as done.
- [ ] Run formatting, linting, and tests with logging.

## Surprises & Discoveries

No surprises encountered during planning.

## Decision Log

| Decision                                                                   | Rationale                                                                 | Date/Author        |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------ |
| Gate Verus proofs behind `cfg(verus)` in a dedicated `verus_proofs` module | Keep production code untouched while isolating verification-specific code | 2026-02-02 (Codex) |
| Use a verified, local sort helper for Verus ordering proofs                | Avoid `assume` shortcuts for `EdgeHarvest::from_unsorted` ordering        | 2026-02-02 (Codex) |

## Outcomes & Retrospective

Pending implementation.

## Context and Orientation

### Key Files

| File                                                            | Purpose                                            |
| --------------------------------------------------------------- | -------------------------------------------------- |
| `chutoro-core/src/hnsw/insert/mod.rs`                           | `extract_candidate_edges` helper to verify         |
| `chutoro-core/src/hnsw/types.rs`                                | `CandidateEdge` and `EdgeHarvest` implementations  |
| `chutoro-core/src/hnsw/tests/edge_harvest.rs`                   | Existing edge-harvest unit tests                   |
| `chutoro-core/src/hnsw/tests/property/edge_harvest_property.rs` | Property-test references for ordering expectations |
| `docs/property-testing-design.md`                               | Appendix A invariant definitions                   |
| `docs/chutoro-design.md`                                        | Design decisions for formal verification scope     |
| `.github/workflows/ci.yml`                                      | CI job definitions                                 |
| `Makefile`                                                      | Test, lint, and verification entry points          |

### Appendix A Invariants (Verus Targets)

- `extract_candidate_edges`:
  - All edges have `source == source_node`.
  - No self-edges are emitted (`target != source_node`).
  - The `sequence` field is preserved (`sequence == source_sequence`).
  - Edge count equals total neighbours across layers minus self-neighbours.
- `CandidateEdge::canonicalise`:
  - Returned edge satisfies `source <= target`.
  - `distance` and `sequence` fields are preserved.
- `EdgeHarvest::from_unsorted`:
  - Output is a permutation of the input.
  - Output is sorted by `(sequence, Ord)`.

## Plan of Work

Start by confirming that helper signatures in `insert/mod.rs` and `types.rs`
are stable so proofs will not churn. Pin the Verus toolchain in contributor
documentation and ensure CI can install that exact version. Implement Verus
proof harnesses in a dedicated module gated by `cfg(verus)`, leaning on a
verified local sort routine (or an approved Verus library equivalent) to avoid
`assume` shortcuts. Add unit tests with `rstest` that exercise happy paths,
self-edge filtering, duplicate neighbours, and ordering edge cases. Record any
verification-specific decisions in `docs/chutoro-design.md` and mark the Phase
1 roadmap entry as done once proofs and tests pass.

## Concrete Steps

1. Inspect helper definitions and existing tests for edge harvest invariants:

   rg -n "extract_candidate_edges|canonicalise|from_unsorted" -S \
     chutoro-core/src/hnsw

2. Confirm helper signatures are stable; if changes are required, document the
   constraint in the design document and adjust the proof plan accordingly.

3. Pin the Verus toolchain and document contributor setup (e.g., add a
   `docs/verus-toolchain.md` section or extend `docs/chutoro-design.md` with
   the pinned version and installation notes).

4. Add a Verus proof module (e.g., `chutoro-core/src/hnsw/verus_proofs.rs`)
   gated behind `cfg(verus)` and include it in `hnsw/mod.rs`. Provide:

   - A proof harness for `extract_candidate_edges` that asserts source/target,
     sequence, and count invariants.
   - A proof harness for `CandidateEdge::canonicalise` that asserts ordering
     and field preservation.
   - A proof harness for `EdgeHarvest::from_unsorted` that asserts permutation
     and ordering, using a verified local sort helper to avoid `assume`.

5. Add unit tests with `rstest` to cover:

   - Happy paths for edges across multiple layers.
   - Self-edge filtering and duplicate neighbour handling.
   - Canonicalise preserving distance/sequence and handling already-ordered
     edges.
   - `from_unsorted` ordering with mixed sequences and equal distances.
   - Unhappy paths such as empty plans and zero neighbours.

6. Update design documentation with the decisions in this plan (toolchain
   pinning, sort strategy, scope limits).

7. Ensure CI runs Verus proofs using the pinned toolchain (new CI job or
   extension of `ci.yml`), and add a Makefile target if needed for a standard
   `make verus` entry point.

8. Update `docs/roadmap.md` to mark the Phase 1 Verus proofs entry as done once
   all proofs and tests pass.

9. Run formatting, linting, and tests with log capture:

   set -o pipefail make check-fmt 2>&1 | tee /tmp/make-check-fmt.log

   set -o pipefail make lint 2>&1 | tee /tmp/make-lint.log

   set -o pipefail make test 2>&1 | tee /tmp/make-test.log

   If documentation changes are made, also run:

   set -o pipefail make fmt 2>&1 | tee /tmp/make-fmt.log

   set -o pipefail make markdownlint 2>&1 | tee /tmp/make-markdownlint.log

   set -o pipefail make nixie 2>&1 | tee /tmp/make-nixie.log

## Validation and Acceptance

- Verus proofs cover the Appendix A invariants without `assume` shortcuts.
- Proof harnesses pass in CI for the pinned Verus toolchain.
- Unit tests using `rstest` cover happy, unhappy, and edge cases.
- `make check-fmt`, `make lint`, and `make test` succeed (logs in `/tmp/`).
- Design decisions are recorded in the design document.
- The Phase 1 roadmap entry is marked as done after completion.

## Idempotence and Recovery

All steps are safe to rerun. If proofs time out or become too complex, reduce
bounds or simplify helper inputs, then record the decision in the design
document. Re-run failed `make` targets with the same `set -o pipefail | tee`
pattern after each fix.

## Artifacts and Notes

Keep log files in `/tmp/` until the change is accepted. Record proof harness
names and commands alongside any verification notes in documentation.

## Interfaces and Dependencies

- `extract_candidate_edges(source_node, source_sequence, plan)` in
  `chutoro-core/src/hnsw/insert/mod.rs`.
- `CandidateEdge::canonicalise` in `chutoro-core/src/hnsw/types.rs`.
- `EdgeHarvest::from_unsorted` and ordering helpers in
  `chutoro-core/src/hnsw/types.rs`.
- CI configuration in `.github/workflows/ci.yml` and Makefile entry points.
- Appendix A in `docs/property-testing-design.md` for invariant definitions.
