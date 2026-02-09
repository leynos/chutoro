# ExecPlan: phase 1 - parallel Kruskal property suite

This execution plan (ExecPlan) is a living document. The sections "Progress",
"Surprises & Discoveries", "Decision Log", and "Outcomes & Retrospective" must
be kept up to date as work proceeds.

**Status**: Complete **Issue**: See `docs/roadmap.md` Phase 1 **Branch**:
`parallel-kruskal-mst-edge-harvest-gxbsrh`

## Purpose / big picture

Add a property-based testing suite for the parallel Kruskal Minimum Spanning
Tree (MST) implementation, as specified in `docs/property-testing-design.md`
Section 4. The suite verifies three properties:

1. **Equivalence with a sequential oracle** — total weight matches a trusted
   sequential Kruskal.
2. **Structural invariant verification** — acyclicity, connectivity, edge
   count, canonical form.
3. **Concurrency safety** — repeated runs on the same input produce identical
   total weights and edge lists.

Graph generation strategies produce pathological inputs (unique weights, many
identical weights, sparse, dense, disconnected) to stress the parallel
implementation. When complete, the final Phase 1 roadmap entry ("Build parallel
Kruskal property suite") will be marked done.

## Constraints and tolerances

- Maximum 400 lines per file (`AGENTS.md`).
- Existing `mst/tests.rs` is at 416 lines — new property tests must go in
  separate files.
- Use `rstest` for parameterized cases, `proptest` for generative testing.
- en-GB-oxendict spelling in comments and documentation.
- Oracle sort order must exactly match `MstEdge::Ord`:
  `(weight.total_cmp, source, target, sequence)` after canonicalization and
  deduplication.
- Total weight comparison uses `f64` accumulation to avoid order-dependent
  `f32` rounding differences.
- Concurrency repetitions default to 5, configurable via env var
  `CHUTORO_MST_PBT_CONCURRENCY_REPS`.

## Progress

- [x] Create `mst/property/` module structure with `mod.rs`.
- [x] Implement fixture types in `property/types.rs`.
- [x] Implement graph generation strategies in `property/strategies.rs`.
- [x] Implement sequential Kruskal oracle in `property/oracle.rs`.
- [x] Implement Property 1: oracle equivalence in `property/equivalence.rs`.
- [x] Implement Property 2: structural invariants in `property/structural.rs`.
- [x] Implement Property 3: concurrency safety in `property/concurrency.rs`.
- [x] Implement test runners and rstest cases in `property/tests.rs`.
- [x] Wire `property` module into `mst/mod.rs`.
- [x] Mark roadmap entry as done.
- [x] Run formatting, linting, and tests with logging.

## Surprises & discoveries

- Clippy's `excessive_nesting` lint triggered on the disconnected graph
  generator's nested `for`/`for`/`if` loop inside an `impl` block. Resolved by
  extracting an `all_pairs()` helper to flatten the nested iteration.
- Clippy's `too_many_arguments` lint triggered on the extracted
  `generate_component` helper. Resolved by introducing an `EdgeBuilder` struct
  to group the mutable edge accumulator and sequence counter.

## Decision log

| Decision                                                             | Rationale                                                                                                      | Date/Author         |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------- |
| Separate `property/` submodule under `mst/`                          | `tests.rs` at 416 lines; follows established HNSW property test pattern                                        | 2026-02-06 (Claude) |
| MST-specific graph generators (not reusing HNSW topology generators) | Avoids cross-module visibility issues; MST needs weighted `CandidateEdge` lists, not `GraphFixture` structures | 2026-02-06 (Claude) |
| `f64` weight accumulation for oracle comparison                      | Avoids order-dependent `f32` rounding differences between parallel and sequential summation                    | 2026-02-06 (Claude) |
| Concurrency repetitions default to 5                                 | Lower end of the 5-10 range from design doc; configurable via `CHUTORO_MST_PBT_CONCURRENCY_REPS`               | 2026-02-06 (Claude) |
| `EdgeBuilder` struct for disconnected generator                      | Avoids `too_many_arguments` lint while keeping edge accumulation and sequence numbering co-located             | 2026-02-06 (Claude) |

## Outcomes & retrospective

Completed all three property tests plus the sequential oracle, rstest
parameterized cases (11 per property across 5 distributions and multiple
seeds), and 9 oracle unit tests. All 549 tests pass (including 42 new MST
property tests). All quality gates (`make check-fmt`, `make lint`, `make test`,
`make markdownlint`) pass. Roadmap entry marked as done.

## Context and orientation

### Key files

| File                                    | Purpose                                                |
| --------------------------------------- | ------------------------------------------------------ |
| `chutoro-core/src/mst/mod.rs`           | `parallel_kruskal`, `MstEdge`, `MinimumSpanningForest` |
| `chutoro-core/src/mst/tests.rs`         | Existing unit tests (at 400-line limit)                |
| `chutoro-core/src/mst/union_find.rs`    | `ConcurrentUnionFind`                                  |
| `chutoro-core/src/hnsw/types.rs`        | `CandidateEdge`, `EdgeHarvest`                         |
| `chutoro-core/src/hnsw/tests/property/` | HNSW property test patterns (template)                 |
| `docs/property-testing-design.md`       | Design specification (Section 4)                       |
| `docs/roadmap.md`                       | Phase 1 entry to mark done                             |

### Existing test patterns

The Hierarchical Navigable Small World (HNSW) property test suite under
`chutoro-core/src/hnsw/tests/property/` is the authoritative template:

- Strategies in `strategies.rs` — builder functions returning
  `impl Strategy<Value = T>`.
- Property functions follow
  `run_<property>_property(fixture) -> TestCaseResult`.
- Test runners wire proptest via `proptest!` macro and use `rstest` for
  parameterized cases.
- Each file has a `//!` module comment and stays under 400 lines.
- Uses `rand::SmallRng` seeded from proptest-generated `u64` values.

## Properties to implement (Section 4)

### 4.3.1 Property 1: equivalence with sequential oracle

For any generated input graph, compute MST via both the parallel implementation
and a simple sequential Kruskal oracle. Assert that the total weight is
identical (compared as `f64`). Also compare edge count and component count.

### 4.3.2 Property 2: structural invariant verification

For any MST/forest produced by the parallel algorithm, verify directly:

- **Acyclicity**: no cycles (union-find based detection).
- **Connectivity**: connected input produces connected output.
- **Edge count**: `V - C` edges for `C` connected components.
- **No self-loops**: `source != target` for all edges.
- **Canonical form**: `source < target` for all edges.
- **Finite weights**: all edge weights are finite.

### 4.3.3 Property 3: concurrency safety

Run parallel Kruskal on the same input graph multiple times (default 5). Assert
that total weight, edge count, component count, and the exact edge list are
identical across all runs.

## Plan of work

### Step 1: Create module structure

Create `chutoro-core/src/mst/property/` directory with `mod.rs` declaring
submodules. Add `#[cfg(test)] mod property;` to `mst/mod.rs`.

### Step 2: Implement fixture types

Create `property/types.rs` with:

- `WeightDistribution` enum: `Unique`, `ManyIdentical`, `Sparse`, `Dense`,
  `Disconnected`.
- `MstFixture` struct: `node_count`, `edges: Vec<CandidateEdge>`,
  `distribution`.
- `ConcurrencyConfig`: loads repetition count from env var.

### Step 3: Implement graph generation strategies

Create `property/strategies.rs` with:

- `mst_fixture_strategy()` using `prop_oneof!` weighted across all five
  distributions.
- Generator functions for each distribution type, each taking `SmallRng`.
- All generators produce `CandidateEdge` with monotonic sequence numbers.

### Step 4: Implement sequential Kruskal oracle

Create `property/oracle.rs` with:

- `sequential_kruskal(node_count, edges) -> SequentialMstResult`.
- Canonicalize, skip self-loops, deduplicate, sort by `MstEdge::Ord` order.
- Simple array union-find with path compression and rank.

### Step 5: Implement Property 1 — oracle equivalence

Create `property/equivalence.rs` with
`run_oracle_equivalence_property(fixture) -> TestCaseResult`.

### Step 6: Implement Property 2 — structural invariants

Create `property/structural.rs` with
`run_structural_invariants_property(fixture) -> TestCaseResult`.

### Step 7: Implement Property 3 — concurrency safety

Create `property/concurrency.rs` with
`run_concurrency_safety_property(fixture) -> TestCaseResult`.

### Step 8: Implement test runners

Create `property/tests.rs` with:

- `proptest!` runners for all three properties.
- `rstest` parameterized cases covering all distributions and multiple seeds.
- Unit tests for the sequential oracle itself.

### Step 9: Update roadmap

Mark the parallel Kruskal property suite entry as done in `docs/roadmap.md`.

### Step 10: Run quality gates

```bash
set -o pipefail && make check-fmt 2>&1 | tee /tmp/check-fmt.log
set -o pipefail && make lint 2>&1 | tee /tmp/lint.log
set -o pipefail && make test 2>&1 | tee /tmp/test.log
```

## Validation and acceptance

- [x] Property 1 (oracle equivalence) passes for all five weight
  distributions.
- [x] Property 2 (structural invariants) passes for all five distributions.
- [x] Property 3 (concurrency safety) passes with 5 repetitions per input.
- [x] rstest parameterized cases cover all five distributions with multiple
  seeds.
- [x] Sequential oracle has dedicated unit tests.
- [x] No file exceeds 400 lines.
- [x] All files have `//!` module-level comments.
- [x] `make check-fmt`, `make lint`, and `make test` succeed with logs saved.
- [x] Roadmap entry for parallel Kruskal property suite marked as done.

## Idempotence and recovery

All steps are safe to rerun. The property module is purely additive; no
existing tests are modified. If any step fails, fix the issue and rerun only
the failed command using the same `set -o pipefail | tee` pattern.

## References

- `docs/property-testing-design.md` Section 4 (specification)
- `docs/roadmap.md` Phase 1 (tracking)
- `docs/chutoro-design.md` (design decisions)
- `docs/complexity-antipatterns-and-refactoring-strategies.md`
- `docs/rust-testing-with-rstest-fixtures.md`
- `docs/rust-doctest-dry-guide.md`
- `chutoro-core/src/hnsw/tests/property/` (pattern template)
- `chutoro-core/src/mst/mod.rs` (implementation under test)
- `chutoro-core/src/mst/tests.rs` (existing unit tests)
