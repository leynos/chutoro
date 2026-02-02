# ExecPlan: phase 1 - Verus proofs for edge harvest primitives

This execution plan (ExecPlan) is a living document. The sections "Progress",
"Surprises & Discoveries", "Decision Log", and "Outcomes & Retrospective" must
be kept up to date as work proceeds.

**Status**: Planned **Issue**: See `docs/roadmap.md` Phase 1 **Branch**: TBD

## Purpose / Big Picture

Add Verus proofs for the edge harvest primitives listed in
`docs/property-testing-design.md` Appendix A, plus unit tests with broad
parameterized coverage using `rstest`. The proofs must cover the invariants
without `assume` shortcuts, run in CI using a pinned Verus toolchain, and stay
scoped to helper invariants (no concurrency or planner proofs). When complete,
record design decisions in the design document and mark the roadmap entry as
"done".

## Progress

- [ ] Confirm edge harvest helper signatures are stable and document any
      required constraints.
- [ ] Pin and document the Verus toolchain for contributors.
- [ ] Add Verus proof harnesses for Appendix A invariants.
- [ ] Add unit tests with `rstest` covering happy/unhappy paths and edge cases.
- [ ] Update design documentation with decisions taken.
- [ ] Mark the roadmap entry as done.
- [ ] Run formatting, linting, and tests with logging.

## Surprises & Discoveries

None so far.

## Decision Log

| Decision | Rationale | Date/Author |
| -------- | --------- | ----------- |
| TBD      | TBD       | TBD         |

## Outcomes & Retrospective

Pending completion.

## Context and Orientation

### Key Files

| File                                          | Purpose                           |
| --------------------------------------------- | --------------------------------- |
| `chutoro-core/src/hnsw/insert/mod.rs`         | `extract_candidate_edges` helper  |
| `chutoro-core/src/hnsw/types.rs`              | `CandidateEdge` and `EdgeHarvest` |
| `chutoro-core/src/hnsw/tests/edge_harvest.rs` | Existing edge harvest tests       |
| `docs/property-testing-design.md`             | Appendix A invariants             |
| `docs/chutoro-design.md`                      | Design decisions to update        |
| `docs/roadmap.md`                             | Phase 1 entry to mark done        |
| `Makefile`                                    | Add Verus proof target            |
| `.github/workflows/ci.yml`                    | CI integration for Verus proofs   |

### Existing Test Patterns

Use the `rstest` patterns described in
`docs/rust-testing-with-rstest-fixtures.md` and avoid doctest duplication per
`docs/rust-doctest-dry-guide.md`.

## Invariants to Prove (Appendix A)

1. `extract_candidate_edges`:
   - `source == source_node` for all edges.
   - `target != source_node` (no self edges).
   - `sequence == source_sequence` for all edges.
   - Edge count equals total neighbours across layers minus self neighbours.
2. `CandidateEdge::canonicalise`:
   - `source <= target` in the result.
   - `distance` and `sequence` preserved.
3. `EdgeHarvest::from_unsorted`:
   - Output is a permutation of the input.
   - Output sorted by `(sequence, Ord)`.

## Plan of Work

### Step 1: Confirm helper stability and constraints

- Review `extract_candidate_edges`, `CandidateEdge::canonicalise`, and
  `EdgeHarvest::from_unsorted` signatures to confirm they are stable.
- Capture any implicit preconditions needed by proofs (for example, sorting
  comparator totality) and record them in the design document.

### Step 2: Pin and document the Verus toolchain

- Decide how to pin Verus (for example, a dedicated toolchain file or a
  `tools/verus` version manifest).
- Document the pinned version and install steps for contributors.
- Add CI setup instructions and any required environment variables.

### Step 3: Add Verus proof scaffolding

- Introduce a Verus module for edge harvest proofs (for example
  `chutoro-core/src/hnsw/verus_proofs.rs`) with a module-level `//!` comment.
- Gate with `#[cfg(verus)]` to avoid affecting normal builds.
- Add spec helpers for sequences, ordering predicates, and permutation checks.

### Step 4: Prove `extract_candidate_edges` invariants

- Model the insertion plan data as a Verus `Seq` of layers and neighbours.
- Prove source, target, and sequence fields are preserved for each edge.
- Prove the count formula for total neighbours minus self neighbours.

### Step 5: Prove `CandidateEdge::canonicalise` properties

- Prove the canonicalised edge preserves `distance` and `sequence`.
- Prove `source <= target` and field swapping is the only change.

### Step 6: Prove `EdgeHarvest::from_unsorted` ordering

- Specify sorting order by `(sequence, Ord)` and define a predicate for sorted
  output.
- Choose a proof strategy that avoids `assume` shortcuts, such as:
  - A verified sort implementation in Verus for the proof harness, or
  - A trusted, specified wrapper with proofs for permutation and ordering.
- Record the chosen strategy in the design document.

### Step 7: Add unit tests with `rstest`

Add or extend tests in `chutoro-core/src/hnsw/tests/edge_harvest.rs`:

- `extract_candidate_edges`:
  - Happy paths for multiple layers and neighbours.
  - Unhappy paths for self neighbours and empty layers.
  - Edge count assertions for mixed layers.
- `CandidateEdge::canonicalise`:
  - Already canonical edges and reversed edges.
  - Self edge behaviour.
- `EdgeHarvest::from_unsorted`:
  - Unsorted inputs with duplicate distances and sequences.
  - Ordering by `(sequence, Ord)` and permutation preservation.

Prefer `rstest` fixtures and parameterized cases for broad coverage.

### Step 8: Update documentation

- Record design decisions in `docs/chutoro-design.md`.
- Update `docs/roadmap.md` to mark the Verus proof entry as done.
- Run Markdown formatting and linting after documentation changes.

### Step 9: Run quality gates

Run all required commands with logging via `tee`:

```bash
set -o pipefail && make check-fmt 2>&1 | tee /tmp/check-fmt.log
set -o pipefail && make lint 2>&1 | tee /tmp/lint.log
set -o pipefail && make test 2>&1 | tee /tmp/test.log
```

If Verus adds new make targets, run them and capture logs similarly.

## Validation and Acceptance

- [ ] Verus proofs cover Appendix A invariants without `assume` shortcuts.
- [ ] Proof harnesses pass in CI using the pinned Verus toolchain.
- [ ] Unit tests cover happy/unhappy paths and edge cases with `rstest`.
- [ ] `make check-fmt`, `make lint`, and `make test` succeed with logs saved.
- [ ] Design decisions recorded in `docs/chutoro-design.md`.
- [ ] Roadmap entry marked as done.

## Idempotence and Recovery

All steps are safe to rerun. If a proof or test fails, fix the issue and rerun
only the failed command using the same `set -o pipefail | tee` pattern.

## References

- `docs/roadmap.md` (Phase 1, Verus proofs entry)
- `docs/property-testing-design.md` Appendix A
- `docs/chutoro-design.md`
- `docs/complexity-antipatterns-and-refactoring-strategies.md`
- `docs/rust-testing-with-rstest-fixtures.md`
- `docs/rust-doctest-dry-guide.md`
