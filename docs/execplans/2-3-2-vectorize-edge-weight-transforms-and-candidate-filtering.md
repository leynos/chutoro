# Vectorize edge-weight transforms and candidate filtering (2.3.2)

This ExecPlan (execution plan) is a living document. The sections `Constraints`,
`Tolerances`, `Risks`, `Progress`, `Surprises & discoveries`, `Decision log`,
and `Outcomes & retrospective` must be kept up to date as work proceeds.

Status: DRAFT

Roadmap item: 2.3.2 (Phase 2, "Hot-path optimizations"). See `docs/roadmap.md`
lines 400-402 and `docs/chutoro-design.md` §6.3 "SIMD utilization", lines
904-908.

## Purpose / big picture

Chutoro clusters data by building an approximate nearest-neighbour graph, then
converting the harvested candidate edges into a mutual-reachability minimum
spanning forest (MSF) with a parallel Kruskal implementation, then extracting a
cluster hierarchy from that forest. This plan targets the middle stage: the
work that happens between harvesting candidate edges and feeding them to the
union-find data structure.

Today that middle stage is slower than it needs to be for reasons that are
visible in the source and do not require speculation:

1. The mutual-reachability weight transform in
   `chutoro-core/src/cpu_pipeline.rs` lines 79-88 is a single-threaded
   `.iter().map()` that allocates a fresh 32-byte-per-edge `Vec<CandidateEdge>`.
2. The result is immediately wrapped in `EdgeHarvest::new`
   (`chutoro-core/src/cpu_pipeline.rs` line 89), which sorts the entire edge
   list by insertion sequence (`chutoro-core/src/hnsw/types.rs` lines 268-272).
   That sort is then thrown away, because `prepare_edge_list` immediately
   re-sorts by weight (`chutoro-core/src/mst/mod.rs` line 283). One of the two
   full sorts is pure waste.
3. `prepare_edge_list` materializes a second intermediate,
   `Vec<&CandidateEdge>` (`chutoro-core/src/mst/mod.rs` line 269), purely to
   obtain a Rayon parallel iterator.
4. The per-edge validation and canonicalization then runs through a
   `try_fold` / `try_reduce` pair whose reduction step is `left.extend(right)`
   (`chutoro-core/src/mst/mod.rs` lines 270-281). This allocates one `Vec` per
   Rayon task and copies every surviving edge once per level of the reduction
   tree.
5. The sort comparator (`chutoro-core/src/mst/mod.rs` lines 134-142) performs
   an `f32::total_cmp` followed by up to three integer comparisons, on 32-byte
   records.
6. Nothing filters candidate edges against union-find state. With HNSW
   `M = 16`, harvest produces roughly `16n` candidate edges of which only
   `n - 1` can be accepted; the remaining ~94% are discovered to be cycles only
   after two path-compressing `find` walks over atomic memory, inside a
   mutex-protected `try_union`.
7. The union-find allocates `node_count` mutexes plus two `AtomicUsize` arrays
   (`chutoro-core/src/mst/union_find.rs` lines 19-43), so its resident working
   set is 8 bytes per node for parents and 8 more for ranks, even though ranks
   never exceed `log2(node_count)`.

After this change, a developer can observe the following.

- `cargo bench -p chutoro-benches --bench mst` reports a measurably lower time
  for the `parallel_kruskal` group at every point count, and a new
  `mst_prepare` group reports the cost of each pre-union-find stage separately
  so future regressions are attributable.
- `scripts/bench-mst-prepare.sh` runs the same benchmark binary through
  `hyperfine` and corroborates the Criterion result at whole-binary scope.
- The clustering output is unchanged. `parallel_kruskal` returns exactly the
  same `MinimumSpanningForest` — same edges, in the same order, with the same
  `component_count` — for every input it accepts today. A new property test
  asserts exact edge-list equality against the sequential oracle, which the
  current suite does not do.
- Error reporting becomes deterministic. When an edge list contains more than
  one invalid edge, the reported `MstError` is the one belonging to the lowest
  input index, on every run and on every thread count.

Success is therefore observable as "same answers, deterministic errors, less
time", measured by Criterion with `hyperfine` as corroboration, and locked in
by property tests, bounded model checking, and a deductive proof.

## Relevant documentation and skills

Read these before starting. They are the source of truth for this repository's
conventions and this plan does not restate them in full.

- `AGENTS.md` — commit discipline, quality gates, the 400-line file cap, the
  abstraction/port/helper sweep policy, en-GB Oxford spelling for comments, and
  the `tee`-to-`/tmp` logging convention for long commands.
- `docs/roadmap.md` §2.3 — the roadmap item being delivered.
- `docs/chutoro-design.md` §3.2 (MST construction survey), §6.2 (parallel
  Kruskal algorithmic sketch, lines 804-886), §6.3 (SIMD utilization, lines
  887-938, and the implementation-update log that follows it).
- `docs/adr-003-soa-prefetch-adapter-boundary.md` — the precedent for how a
  §6.3 hot-path boundary decision is recorded, and for gating structural change
  on measured evidence rather than on plausibility.
- `docs/property-testing-design.md` — property-suite structure and naming.
- `docs/rust-testing-with-rstest-fixtures.md` — fixture and case conventions.
- `docs/rust-doctest-dry-guide.md` — doctest style for new public items.
- `docs/complexity-antipatterns-and-refactoring-strategies.md` — the
  refactoring heuristics `AGENTS.md` requires after each functional commit.
- `docs/documentation-style-guide.md` — 80-column prose wrap, 120-column code
  wrap, sentence-case headings, captioned tables and diagrams, ADR template.
- `docs/developers-guide.md` — "Benchmarks" (benchmark architecture, the
  regression workflow, and "Adding a new benchmark"), "Dense SIMD parity
  suite", "Dense SIMD Kani harnesses", "Verus proofs".
- `docs/users-guide.md` — "Error handling" and "Feature flags and execution
  strategies" are the sections this work may touch.
- `docs/contents.md` — the documentation index; every new document must be
  registered here.

Skills to load: `leta` (semantic navigation; load first and add the worktree as
a workspace), `rust-router` then `rust-performance-and-layout` and
`rust-unit-testing`, `hexagonal-architecture`, `proptest`, `kani`, `verus`,
`nextest`, `execplans`, `commit-message`, `pr-creation`, and `en-gb-oxendict`.
Use `firecrawl` for any further external lookup.

## Constraints

Hard invariants. Violation requires escalation, not a workaround.

1. **Output equivalence.** For every input that `parallel_kruskal` accepts
   today, it must return an identical `MinimumSpanningForest`: the same edge
   sequence (including each edge's `sequence` value) and the same
   `component_count`. This is the acceptance bar for the whole plan.
2. **Determinism.** Repeated runs on identical input, at any Rayon thread
   count, must produce byte-identical results. The existing property
   `run_concurrency_safety_property`
   (`chutoro-core/src/mst/property/concurrency.rs`) already asserts this and
   must keep passing.
3. **No `unsafe` outside SIMD adapter modules.** The domain, port, scalar
   kernel, and driver code must contain no `unsafe`. Any `unsafe` introduced by
   an intrinsics adapter must be confined to that adapter's module, be
   accompanied by a documented invariant list, and be covered by the parity
   suite.
4. **Public API additions only, no removals or signature changes.**
   `parallel_kruskal`, `MstEdge`, `MinimumSpanningForest`, `CandidateEdge`, and
   `EdgeHarvest` keep their current signatures and observable semantics.
   `MstError` is `#[non_exhaustive]`, so adding a variant is permitted;
   removing or renaming one is not.
5. **Minimum supported Rust version stays 1.89.0** (`Cargo.toml`
   `workspace.package.rust-version`). No nightly-only feature may be required
   for the default build. A nightly-gated backend, if added, follows the
   existing `nightly_portable_simd` pattern in `chutoro-providers/dense`.
6. **No new runtime dependency** in `chutoro-core`. SIMD work uses
   `core::arch` intrinsics from the standard library.
7. **File-size cap.** No source file may exceed 400 lines (`AGENTS.md`,
   enforced by Whitaker's `module_max_lines`).
8. **Dependency direction.** `chutoro-core` must not depend on
   `chutoro-providers-dense`. The dense provider's `simd` module is
   crate-private and is not reusable here; it may be mirrored in shape but not
   imported.
9. **Do not modify** `chutoro-providers/dense/src/simd/**`,
   `chutoro-core/src/hnsw/**` (beyond additive documentation), or
   `chutoro-core/src/hierarchy/**`.

## Tolerances (exception triggers)

Stop and escalate when any of these is reached. Do not work around them.

1. **Scope.** More than 22 files changed, or more than 1800 net lines added
   across production and test code.
2. **Interface.** Any change to an existing public signature, or any need to
   make an existing private item public other than through the documented
   `bench_internals` seam described in Milestone 1.
3. **Dependencies.** Any new entry in `[dependencies]` for any crate.
4. **Iterations.** A gate still failing after four consecutive fix attempts.
5. **Evidence.** A milestone's pre-registered benchmark threshold is not met.
   Record the null result and escalate rather than keeping the change.
6. **Ambiguity.** Any place where this plan and the code disagree about
   current behaviour.
7. **Verification cost.** A Kani harness exceeding 15 minutes, or a Verus
   proof exceeding 5 minutes, on the 24-core development machine.
8. **Time.** More than 6 hours of wall-clock work on a single milestone.

## Risks

- Risk: The pre-union-find stages turn out to be a small fraction of
  `parallel_kruskal` wall time, so vectorizing them cannot pay for itself.
  Severity: high. Likelihood: medium. Mitigation: Milestone 1 measures the
  split before any production change and carries an explicit go/no-go. Roadmap
  item 2.3.1 and ADR-003 set the precedent for recording a null result instead
  of shipping a speculative rewrite.

- Risk: Hardware gather instructions do not help the mutual-reachability
  transform. `core_distances[left]` and `core_distances[right]` are random
  accesses keyed by node id, and published measurements repeatedly find
  `vgatherdps` no faster than scalar loads because the bottleneck is memory,
  not issue width. Severity: medium. Likelihood: high. Mitigation: treat gather
  as explicitly out of scope for the committed work. The transform's win comes
  from parallelizing it, from deleting the redundant sort, and from writing
  straight into a structure-of-arrays (SoA) buffer. Gather is only considered
  in Milestone 5 and only behind a measurement.

- Risk: A SIMD compaction adapter reorders surviving edges relative to the
  scalar kernel, silently changing MST tie-breaking. Severity: high.
  Likelihood: medium. Mitigation: the compaction contract is order-preserving
  by construction (both `_mm512_mask_compressstoreu_epi32` and the AVX2
  permute-plus-lookup approach preserve lane order). The parity property suite
  asserts index-wise equality against the scalar kernel, not merely set
  equality, and the exact-edge-list differential property covers the end-to-end
  effect.

- Risk: Narrowing union-find storage to 32-bit node ids introduces a silent
  truncation for very large graphs. Severity: high. Likelihood: low.
  Mitigation: an explicit guard returns a new `MstError` variant when
  `node_count` exceeds `u32::MAX`, with a unit test and a Kani-checked
  boundary. Clippy's `cast_possible_truncation` is already denied
  workspace-wide, so an unchecked cast will not compile.

- Risk: The candidate pre-filter drops an edge that the union-find would have
  accepted, corrupting the forest. Severity: high. Likelihood: low. Mitigation:
  the filter is a one-sided under-approximation justified by component
  monotonicity, proved deductively in Verus and checked by a bounded Kani
  harness and a property test that compares filtered and unfiltered runs.

- Risk: Kani harnesses over symbolic `f32` values are slow or fail to
  converge. Severity: medium. Likelihood: medium. Mitigation: keep the float
  harness to a single symbolic pair with no loops, and try
  `#[kani::solver(kissat)]` before widening bounds. If it still fails,
  downgrade to an exhaustive `proptest` over structured bit patterns and record
  the decision.

- Risk: `googletest`, `pretty_assertions`, and `insta` are named in
  `AGENTS.md` but are entirely absent from this workspace, so adopting them
  here would introduce three dependencies with no in-repo precedent. Severity:
  low. Likelihood: high. Mitigation: see the decision log. This plan follows
  the observed house style (plain `assert!`/`assert_eq!` plus named helper
  assertions) and records the divergence explicitly.

## Progress

- [ ] Milestone 0: orientation and workspace setup.
- [ ] Milestone 1: measurement baseline and go/no-go (no production change).
- [ ] Milestone 2: red tests — exact-equivalence property, error-determinism
      characterization, and the BDD feature specification.
- [ ] Milestone 3: domain policy, SoA staging, order-preserving key, and the
      scalar kernel port.
- [ ] Milestone 4: cache-friendly union-find and the Filter-Kruskal candidate
      pre-filter, with Kani and Verus verification.
- [ ] Milestone 5: SIMD adapters (evidence-gated; may end as a null result).
- [ ] Milestone 6: documentation, ADR-005, roadmap update, and final gates.

## Surprises & discoveries

- Observation: the mutual-reachability edge-weight transform that roadmap
  2.3.2 names does not live in `chutoro-core/src/mst/` at all. Evidence:
  `chutoro-core/src/cpu_pipeline.rs` lines 79-88 hold the
  `dist.max(core_distances[left]).max(core_distances[right])` computation; the
  `mst` module only sees the already-transformed weights. Impact: the plan must
  move or re-home that transform so it can share the SoA staging buffer with
  the validation and filtering stages. This is what makes the redundant
  `EdgeHarvest::new` sort visible and removable.

- Observation: the incremental session refresh path does not call
  `parallel_kruskal` yet. Evidence: `chutoro-core/src/session/mod.rs` line 91
  declares `_mst_edges: Vec<MstEdge>` and line 93
  `pending_edges: Vec<CandidateEdge>`, both unused placeholders for roadmap
  item 11.1.4; no `session` module references `mst::`. Impact:
  `chutoro-core/src/cpu_pipeline.rs` line 91 is the only production call site.
  Scope is narrower than it first appears, and no session-side regression risk
  exists.

- Observation: the existing MST oracle-equivalence property compares total
  weight, edge count, and component count, but not the edge list itself.
  Evidence: `chutoro-core/src/mst/property/equivalence.rs`
  `run_oracle_equivalence_property`. Impact: a reordering regression would pass
  today. Milestone 2 closes this gap before any production change, which is
  what makes the rest of the plan safe.

- Observation: `make kani` does not run the MST harnesses.
  Evidence: `Makefile` lines 95-99 name four harnesses explicitly, none from
  `chutoro-core/src/mst/kani_harness.rs`; only `make kani-full` (lines 101-103,
  nightly-gated) runs every proof. Impact: new MST harnesses default to
  nightly-only coverage unless deliberately added to the fast path.

- Observation: `googletest`, `pretty_assertions`, and `insta` do not appear in
  `Cargo.lock` or in any source file. Evidence: repository-wide search for the
  crate names and for `assert_that!` / `expect_that!` / `insta::` returns no
  hits outside prose. Impact: recorded as a decision rather than silently
  adopted.

## Decision log

- Decision: measure before changing, with an explicit go/no-go gate.
  Rationale: roadmap 2.3.1 was resolved as an evidence-backed verification
  decision rather than a speculative rewrite, and ADR-003 instructs later items
  to record null results beside the roadmap entry rather than widening
  interfaces on plausibility. Applying the same discipline here is house style,
  not caution for its own sake. Date/Author: 2026-08-16, planning agent.

- Decision: follow the repository's actual assertion style (plain `assert!`,
  `assert_eq!`, and named helpers such as `check_forest_invariants`) rather
  than introducing `googletest` and `pretty_assertions`. Rationale: neither
  crate appears anywhere in this workspace. Introducing two test-framework
  dependencies as a side effect of a hot-path optimization would be a
  cross-cutting change with its own review surface, and would breach this
  plan's dependency tolerance. If the user wants them adopted, that is a
  separate, workspace-wide change. Date/Author: 2026-08-16, planning agent.

- Decision: no `insta` snapshot tests.
  Rationale: `AGENTS.md` scopes snapshot testing to "multivariant output format
  consistency". This work produces a `Vec<MstEdge>` and an `MstError`, not
  formatted output, and exact structural equality is already the stronger
  assertion. `insta` is also absent from the workspace. Date/Author:
  2026-08-16, planning agent.

- Decision: the candidate pre-filter is modelled on Filter-Kruskal (Osipov,
  Sanders and Singler, ALENEX 2009), which discards edges whose endpoints
  already share a component before they reach the union-find, rather than on an
  ad hoc heuristic. Rationale: it is the established algorithm for exactly this
  step, its soundness argument (component membership is monotone under union)
  is simple enough to prove deductively, and it filters on root identity rather
  than on a weaker parent-equality hint. Date/Author: 2026-08-16, planning
  agent.

- Decision: hardware gather (`_mm256_i32gather_ps` and equivalents) is out of
  scope for the committed work. Rationale: the core-distance lookups are
  random-access and published measurements consistently find AVX2 gather no
  faster than scalar loads on memory-bound workloads. Committing to gather
  would be a plausibility-driven choice, which is what ADR-003 exists to
  prevent. Date/Author: 2026-08-16, planning agent.

- Decision: keep the global sort in Rayon, and make it cheaper by sorting on a
  precomputed order-preserving integer key rather than by replacing the sort.
  Rationale: `docs/chutoro-design.md` §6.3 line 904 says explicitly "Keep the
  global sort in Rayon". Precomputing the key is a lane-parallel operation in
  the transform stage that costs nothing extra and turns a `total_cmp`-led
  four-key comparator into an integer comparison chain. Date/Author:
  2026-08-16, planning agent.

## Context and orientation

This section assumes no prior knowledge of the repository.

### Terms

- **HNSW** — Hierarchical Navigable Small World, the approximate
  nearest-neighbour index Chutoro builds first. Implementation:
  `chutoro-core/src/hnsw/`.
- **Candidate edge** — a `(source, target, distance, sequence)` record
  produced while inserting a point into the HNSW graph. Type: `CandidateEdge` in
  `chutoro-core/src/hnsw/types.rs` lines 146-152. The `sequence` field is a
  monotonically increasing insertion counter used purely for deterministic
  tie-breaking.
- **Edge harvest** — `EdgeHarvest`, a newtype over `Vec<CandidateEdge>`
  (`chutoro-core/src/hnsw/types.rs` line 248) whose constructors always sort by
  `(sequence, natural Ord)`.
- **Core distance** — for a point `p` and a minimum cluster size `k`, the
  distance from `p` to its `k`-th nearest neighbour. Computed in
  `chutoro-core/src/cpu_pipeline.rs` lines 65-77.
- **Mutual reachability** — the FISHDBC edge weight
  `max(d(u, v), core(u), core(v))`. Computed in
  `chutoro-core/src/cpu_pipeline.rs` line 85.
- **MSF / MST** — minimum spanning forest and its connected special case, the
  minimum spanning tree. Built by
  `chutoro-core/src/mst/mod.rs::parallel_kruskal`.
- **Union-find** — the disjoint-set structure that answers "are these two
  nodes already connected?". Implementation:
  `chutoro-core/src/mst/union_find.rs`.
- **SoA** — structure of arrays: one array per field, so that a vector
  instruction can load sixteen consecutive `source` values in one go. The
  opposite, AoS (array of structures), interleaves fields and defeats
  vectorization. `CandidateEdge` is AoS.
- **Stream compaction** — writing only the elements selected by a lane mask,
  contiguously, preserving their order. AVX-512 has a single instruction for it
  (`_mm512_mask_compressstoreu_epi32`); AVX2 emulates it with a mask-indexed
  permutation table.

### Current control flow

`run_cpu_pipeline_with_len` (`chutoro-core/src/cpu_pipeline.rs` lines 47-106)
runs five steps: build the HNSW index and harvest edges; compute core
distances; apply the mutual-reachability transform; build the MSF; extract
labels.

`parallel_kruskal` (`chutoro-core/src/mst/mod.rs` lines 188-193) delegates to
`parallel_kruskal_from_edges` (lines 290-335), which calls `prepare_edge_list`
(lines 265-288) and then walks the sorted edge list in equal-weight groups,
handing each group to `process_weight_group` (lines 241-255), which calls
`ConcurrentUnionFind::try_union` sequentially.

Note that the union-find is exercised strictly sequentially on this path. Its
striped-lock design exists for a concurrent future and is exercised by
`chutoro-core/src/mst/property/concurrency.rs`; this plan preserves that
capability rather than removing it.

### Where the plan intervenes

The plan introduces one new module tree, `chutoro-core/src/mst/prepare/`, which
owns everything between "harvested candidate edges plus core distances" and "a
sorted, deduplicated, filtered edge list ready for union-find". The
`cpu_pipeline` transform moves into it, `prepare_edge_list` is rewritten in
terms of it, and the union-find gains a root-snapshot accessor so the candidate
filter can consult it.

## Interfaces and dependencies

At the end of Milestone 4, the following items must exist with these
signatures. All are crate-internal unless marked otherwise.

In `chutoro-core/src/mst/prepare/policy.rs`, the domain object that fixes the
contract in one place, mirroring how
`chutoro-providers/dense/src/simd/semantics.rs` fixes the distance contract:

```rust
/// Fixes the edge-preparation contract shared by every backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EdgePreparePolicy {
    node_count: usize,
}

impl EdgePreparePolicy {
    pub(crate) const fn new(node_count: usize) -> Self;

    /// Mutual-reachability weight: `max(distance, core(source), core(target))`.
    pub(crate) fn mutual_reachability(self, distance: f32, source_core: f32, target_core: f32) -> f32;

    /// Classifies one edge into keep, drop, or reject.
    pub(crate) fn classify(self, source: usize, target: usize, weight: f32) -> EdgeVerdict;

    /// Canonical undirected form: `(min(source, target), max(source, target))`.
    pub(crate) const fn canonicalize(self, source: u32, target: u32) -> (u32, u32);
}

/// The three outcomes of edge classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EdgeVerdict {
    /// The edge is valid and participates in the forest.
    Keep,
    /// The edge is a self-loop and is silently discarded.
    Drop,
    /// The edge is invalid; preparation fails with this error.
    Reject(MstError),
}
```

In `chutoro-core/src/mst/prepare/keys.rs`, the order-preserving bijection that
makes the sort comparator integral:

```rust
/// Maps an `f32` to a `u32` whose unsigned order matches `f32::total_cmp`.
///
/// Flips the sign bit for non-negative values and every bit for negative
/// values, the standard IEEE 754 radix-sort key transform.
pub(crate) const fn weight_key(weight: f32) -> u32;

/// Inverse of [`weight_key`].
pub(crate) const fn weight_from_key(key: u32) -> f32;
```

In `chutoro-core/src/mst/prepare/soa.rs`, the staging buffers:

```rust
/// Structure-of-arrays staging buffer for candidate edges.
///
/// All arrays share a length and are padded to a lane multiple so vector
/// loads never read past the logical end. Padding lanes hold `u32::MAX`
/// endpoints and `f32::INFINITY` weights so they can never be selected.
pub(crate) struct EdgeSoa {
    sources: Vec<u32>,
    targets: Vec<u32>,
    weights: Vec<f32>,
    keys: Vec<u32>,
    sequences: Vec<u64>,
    len: usize,
}

impl EdgeSoa {
    pub(crate) fn with_capacity(capacity: usize) -> Self;
    pub(crate) fn len(&self) -> usize;
    pub(crate) fn padded_len(&self) -> usize;
}
```

In `chutoro-core/src/mst/prepare/kernel.rs`, the port. Each kernel is a plain
function pointer so dispatch is a one-time `OnceLock` patch and hot loops stay
branch-free, exactly as `chutoro-providers/dense/src/simd/dispatch.rs` does:

```rust
/// Backend-selectable kernels for the edge-preparation stages.
pub(crate) struct EdgePrepareKernels {
    /// Writes `max(distance, core[source], core[target])` into `out`.
    pub(crate) mutual_reachability: fn(&EdgeSoa, &[f32], &mut [f32]),
    /// Writes an order-preserving sort key per edge.
    pub(crate) weight_keys: fn(&[f32], &mut [u32]),
    /// Sets one mask byte per edge from the policy verdict.
    pub(crate) classify: fn(&EdgeSoa, EdgePreparePolicy, &mut [u8]) -> Option<usize>,
    /// Order-preserving stream compaction driven by the mask.
    pub(crate) compact: fn(&EdgeSoa, &[u8], &mut EdgeSoa),
    /// Clears the mask for edges whose endpoints already share a root.
    pub(crate) cycle_filter: fn(&EdgeSoa, &[u32], &mut [u8]),
}

/// Returns the one-time-selected kernel table.
pub(crate) fn kernels() -> &'static EdgePrepareKernels;
```

In `chutoro-core/src/mst/prepare/dispatch.rs`, mirroring the dense provider's
enum and priority so the two are recognizably the same pattern:

```rust
/// Backends available for edge preparation, in selection priority order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EdgePrepareBackend {
    Avx512,
    Avx2,
    Neon,
    Scalar,
}

pub(crate) fn edge_prepare_backend() -> EdgePrepareBackend;
```

In `chutoro-core/src/mst/union_find.rs`, the additions that let the filter see
component state without disturbing the striped-lock protocol:

```rust
impl ConcurrentUnionFind {
    /// Overwrites `out` with the current root of every node.
    ///
    /// The snapshot is a sound over-approximation of connectivity: two nodes
    /// sharing a root at snapshot time share a root for ever after, because
    /// `union` only merges components.
    pub(super) fn refresh_root_snapshot(&self, out: &mut Vec<u32>);
}
```

In `chutoro-core/src/mst/mod.rs`, one new error variant:

```rust
/// The graph has more nodes than the packed 32-bit node id can address.
#[error("node_count {node_count} exceeds the maximum supported {limit}")]
NodeCountTooLarge {
    /// The requested node count.
    node_count: usize,
    /// The largest supported node count.
    limit: usize,
},
```

with a matching `MstErrorCode::NodeCountTooLarge` mapping to
`"NODE_COUNT_TOO_LARGE"`.

## Plan of work

### Milestone 0: orientation and workspace setup

No code changes. Confirm the tree is clean, the branch is
`2-3-2-vectorize-edge-weight-transforms-and-candidate-filtering`, and the gates
pass before any edit, so later failures are attributable.

```sh
set -o pipefail
git branch --show-current
git status --short
make check-fmt 2>&1 | tee /tmp/check-fmt-chutoro-2-3-2-baseline.out
make lint      2>&1 | tee /tmp/lint-chutoro-2-3-2-baseline.out
make test      2>&1 | tee /tmp/test-chutoro-2-3-2-baseline.out
```

Expect all three to succeed. Record the test count in `Progress`.

### Milestone 1: measurement baseline and go/no-go

This milestone changes no production behaviour. It adds the instrument.

Add a non-default Cargo feature to `chutoro-core/Cargo.toml`:

```toml
[features]
bench_internals = []
```

Under that feature, expose a `#[doc(hidden)]` re-export module so
`chutoro-benches` can time individual stages without widening the real public
API:

```rust
/// Internal seams exposed solely for the benchmark crate.
///
/// Not part of the public API and not covered by semantic versioning.
#[cfg(feature = "bench_internals")]
#[doc(hidden)]
pub mod bench_internals {
    pub use crate::mst::prepare_edge_list_for_bench;
}
```

Add `chutoro-benches/benches/mst_prepare.rs` with `harness = false`, following
the fallible-`_impl`-plus-thin-wrapper pattern documented in
`docs/developers-guide.md` "Adding a new benchmark" and used by
`chutoro-benches/benches/mst.rs`. Reuse that file's constants (`SEED = 42`,
`DIMENSIONS = 16`, `M = 16`) and extend the point counts to
`[100, 500, 1_000, 10_000]` so the largest case reflects the profile recorded
for roadmap 2.3.1. Time four groups separately:

1. `mutual_reachability` — the `cpu_pipeline` transform in isolation.
2. `harvest_resort` — the `EdgeHarvest::new` sort that this plan deletes.
3. `prepare_edge_list` — validation, canonicalization, sort, dedup.
4. `union_find_loop` — the weight-group walk and `try_union` calls.

Add `scripts/bench-mst-prepare.sh`, a copy of
`scripts/bench-neighbour-scoring.sh` retargeted at the `mst_prepare` binary.
Keep its structure identical: `require_command` guards for `cargo`, `jq`, and
`hyperfine`; `cargo bench --no-run --message-format=json` to locate the binary;
`hyperfine --shell bash --warmup 1 --runs 10` with output `tee`d to a
branch-named log under `${TMPDIR:-/tmp}`.

Register the script in `docs/developers-guide.md` beside the neighbour-scoring
entry, keeping that section's stated policy that `hyperfine` is corroboration
and Criterion is the primary signal.

**Go/no-go.** Run the benchmark at `point_count = 10_000` and record the share
of total `parallel_kruskal` time spent in groups 1 to 3. Proceed only if that
share is at least 20%. Below 20%, stop, write the measurement into
`Outcomes & retrospective` and into `docs/chutoro-design.md` §6.3 as a null
result beside the roadmap entry, and escalate. This threshold is pre-registered
here precisely so it cannot be rationalized after the fact.

### Milestone 2: red tests

Nothing in this milestone touches production code. Every test added here must
be run and its outcome recorded before Milestone 3 begins.

First, close the equivalence gap. In
`chutoro-core/src/mst/property/equivalence.rs`, extend
`run_oracle_equivalence_property` to assert exact edge-list equality against
`sequential_kruskal`, not just aggregate weight and counts. Expect this to pass
immediately; it is a *guard*, not a red test, and it is what makes the rest of
the work safe. Record it as such.

Second, characterize error selection. Add a test that builds an edge list with
two distinct invalid edges — one at index 0, one at the final index — and runs
`parallel_kruskal` 64 times, asserting the reported `MstError` is always the
index-0 edge's. Run it under a forced multi-threaded pool
(`RAYON_NUM_THREADS=8`).

This test may pass or fail on the current code; `rayon`'s `try_reduce` does not
document which error wins. Run it first and record the actual observed
behaviour in `Surprises & discoveries`. If it passes, keep it as a regression
guard on behaviour the rewrite must preserve. If it fails, it is the red test
for the determinism improvement and must go green in Milestone 3.

Third, add the filter-soundness property to `chutoro-core/src/mst/property/`,
as a new `filtering.rs` module wired into `mod.rs` alongside the existing four
properties and added to the `parameterised_property_test!` case list in
`property/tests.rs` (remember `test_cases_count_matches_macro_expectations`
guards that list). The property runs `parallel_kruskal` with the candidate
filter disabled and enabled and asserts identical output. Until Milestone 4
introduces the toggle, the property is written against a `PrepareOptions` value
that does not yet exist, so this file is added at the start of Milestone 4
rather than here; note that sequencing in `Progress`.

Fourth, add the behavioural specification. Create
`chutoro-core/tests/features/mst_edge_preparation.feature`:

```gherkin
Feature: Minimum spanning forest edge preparation

  Scenario: Mutual-reachability weights drive forest selection
    Given a graph with 6 nodes and 9 candidate edges
    And core distances that raise the weight of edge 3
    When I build the minimum spanning forest
    Then the forest has 5 edges
    And the forest excludes edge 3

  Scenario: Duplicate candidate edges are collapsed
    Given a graph with 4 nodes and 8 candidate edges
    And 4 of the candidate edges are exact duplicates
    When I build the minimum spanning forest
    Then the forest has 3 edges
    And every forest edge has source less than target

  Scenario: Self-loops are discarded without failing
    Given a graph with 5 nodes and 7 candidate edges
    And 2 of the candidate edges are self-loops
    When I build the minimum spanning forest
    Then the forest has 4 edges

  Scenario: The first invalid edge is reported
    Given a graph with 4 nodes and 6 candidate edges
    And candidate edge 1 references node 9
    And candidate edge 5 has a non-finite weight
    When I build the minimum spanning forest
    Then forest construction fails with error code "INVALID_NODE_ID"

  Scenario: A disconnected graph yields a forest, not a tree
    Given a graph with 6 nodes and 4 candidate edges
    And the candidate edges span only nodes 0 to 3
    When I build the minimum spanning forest
    Then the forest reports 3 components
    And the forest is not a tree
```

Implement the step glue in `chutoro-core/tests/mst_edge_preparation_bdd.rs`,
following `chutoro-core/tests/session_append_bdd.rs` exactly: a `World` struct,
a `#[fixture] fn world()`, `#[given]`/`#[when]`/`#[then]` steps returning
`rstest_bdd::StepResult<(), BddStepError>` with a local error enum, and one
`#[scenario(path = ..., name = ...)]` function per scenario.

Run the suite and confirm every scenario passes against the current
implementation. These scenarios describe behaviour the change must preserve, so
they are green from the start by design; the fourth scenario is the one that
may be red, depending on the error-selection characterization above.

```sh
set -o pipefail
RAYON_NUM_THREADS=8 cargo nextest run -p chutoro-core \
  -E 'test(/mst_edge_preparation/) or test(/mst::/)' \
  2>&1 | tee /tmp/test-chutoro-2-3-2-red.out
```

### Milestone 3: domain policy, SoA staging, and the scalar kernel

Create `chutoro-core/src/mst/prepare/` with the modules listed in *Interfaces
and dependencies*. Each file carries a `//!` module comment and stays under 400
lines; split `scalar.rs` if it approaches the cap.

Before writing `policy.rs`, perform the `AGENTS.md` abstraction sweep: confirm
no existing helper already encodes these rules. The sweep will find
`CandidateEdge::canonicalise` (`chutoro-core/src/hnsw/types.rs` line 191),
`validate_and_canonicalize_edge` (`chutoro-core/src/mst/mod.rs` line 195),
`property/helpers.rs::is_invalid_edge`, and
`chutoro-providers/dense/src/simd/semantics.rs::DistanceSemantics`. Record in
the decision log that `EdgePreparePolicy` supersedes the first three for the
MST path and mirrors the fourth's role, and document its ownership boundary: it
is the single definition of the MST edge contract, callable only from
`mst::prepare`, and every backend must agree with its scalar realization.

Implement the scalar kernels branch-free over slices, with no early exit, so
LLVM has the best chance of auto-vectorizing them without intrinsics.
Concretely: `classify` writes `0` or `1` per lane from bitwise combinations of
comparisons rather than from `if`; the non-finite test uses the exponent-mask
form `(bits & 0x7F80_0000) != 0x7F80_0000` rather than `is_finite()` inside a
branch; `compact` runs a mask popcount prefix pass followed by a scatter pass.

Then rewire the two call sites.

In `chutoro-core/src/cpu_pipeline.rs`, replace lines 79-89 with a call that
builds the `EdgeSoa` directly from `harvested` and `core_distances`,
parallelized with Rayon over chunks. Delete the `EdgeHarvest::new` round trip.
The transform and the key computation happen in the same pass, so each edge is
touched once.

In `chutoro-core/src/mst/mod.rs`, rewrite `prepare_edge_list` to:

1. Reject `node_count > u32::MAX` with the new error variant.
2. Borrow the harvest's slice directly, deleting the `Vec<&CandidateEdge>`
   intermediate.
3. Run `classify` in parallel over chunks, each chunk writing into its own
   slice of a single pre-sized mask buffer, and reducing the rejection index by
   *minimum*, so the reported error is the lowest input index regardless of
   scheduling. This is what makes error selection deterministic.
4. Run `compact` into a single pre-sized `EdgeSoa`, replacing the
   `try_fold`/`try_reduce` allocation chain.
5. Sort with `par_sort_unstable` as today, but on records ordered by
   `(key, source, target, sequence)` with `key` the precomputed `u32`. The
   comparator becomes an integer chain.
6. Deduplicate with the vectorizable adjacent-equality mask instead of
   `dedup_by`.

Keep `MstEdge`'s public accessors and `Ord` semantics unchanged: `Ord` may be
implemented in terms of the key internally, but must remain observationally
identical to the current `total_cmp`-led ordering. The Kani harness in
Milestone 4 is what proves this.

Add `rstest` unit tests covering: `weight_key` on `-inf`, `-1.0`, `-0.0`, `0.0`,
`1.0`, `+inf`, and `NaN`; `weight_key` round-tripping through
`weight_from_key`; `EdgeSoa` padding at logical lengths 0, 1, 15, 16, and 17,
mirroring the dense provider's `DensePointView` padding tests; policy
classification for in-bounds, out-of-bounds, self-loop, and non-finite edges;
and the `node_count > u32::MAX` guard.

Validate and commit:

```sh
set -o pipefail
cargo nextest run -p chutoro-core -E 'test(/mst::/)' 2>&1 \
  | tee /tmp/test-chutoro-2-3-2-m3.out
make check-fmt 2>&1 | tee /tmp/check-fmt-chutoro-2-3-2-m3.out
make lint      2>&1 | tee /tmp/lint-chutoro-2-3-2-m3.out
make test      2>&1 | tee /tmp/test-chutoro-2-3-2-m3-full.out
cargo bench -p chutoro-benches --bench mst -- --baseline local-reference --noplot
```

### Milestone 4: cache-friendly union-find and the candidate pre-filter

Narrow `ConcurrentUnionFind` storage: `parents: Vec<AtomicU32>` and
`ranks: Vec<AtomicU8>`. Ranks are bounded by `log2(node_count) <= 32` under
union by rank with 32-bit ids, so `u8` cannot overflow; assert this in a unit
test and state it in the module comment. Keep the striped-lock protocol, the
retry loop, and `lock_order` unchanged — the concurrency property suite is the
guard.

Add `refresh_root_snapshot`, which fills a caller-owned `Vec<u32>` with
`find(i)` for every node.

Add the Filter-Kruskal pre-filter to `parallel_kruskal_from_edges`. Filtering
is amortized, not per-edge: refresh the root snapshot and sweep the remaining
edge list only when the component count has fallen by at least a factor of two
since the last sweep, and only while at least `FILTER_MIN_REMAINING` edges
remain. Both constants live in `prepare/policy.rs` with a comment recording
that they are benchmark-tuned, and both are surfaced through a `PrepareOptions`
value so the property suite can disable filtering entirely.

The soundness argument, which must appear in the module comment: `union` only
merges components, so the partition induced by the union-find only ever
coarsens. If two nodes shared a root when the snapshot was taken, they share a
root now. Therefore an edge the filter discards would have been rejected by
`try_union` anyway, and the filter can only remove work, never change the
result.

Now add `chutoro-core/src/mst/property/filtering.rs` as specified in Milestone
2, wire it into `property/mod.rs`, and add its case to
`parameterised_property_test!` in `property/tests.rs`, updating
`test_cases_count_matches_macro_expectations`.

Add two Kani harnesses to `chutoro-core/src/mst/kani_harness.rs` (splitting the
file if it approaches 400 lines):

```rust
/// Proves the packed sort key induces exactly `f32::total_cmp` order.
#[kani::proof]
fn verify_weight_key_matches_total_cmp() {
    let left = kani::any::<f32>();
    let right = kani::any::<f32>();
    kani::assert(
        weight_key(left).cmp(&weight_key(right)) == left.total_cmp(&right),
        "packed weight key must reproduce total_cmp ordering",
    );
}

/// Proves the candidate filter never discards an edge union-find would accept.
#[kani::proof]
#[kani::unwind(7)]
fn verify_cycle_filter_is_sound_4_nodes() { /* bounded 4-node graph */ }
```

Add `verify_weight_key_matches_total_cmp` to the `make kani` fast path in the
`Makefile`; it is loop-free and should complete in seconds. Leave the 4-node
graph harness to `make kani-full`, consistent with the existing MST harnesses.
Validate both by deliberate mutation: break `weight_key`'s negative branch and
confirm the harness fails with the stated message, then restore. A harness that
still passes after mutation is not testing what it claims.

Add the Verus proof in `verus/mst_filter_soundness.rs` and register it in the
`PROOF_FILES` array in `scripts/run-verus.sh`. Model the union-find abstractly,
as a partition of node ids into disjoint sets, rather than as a parent array —
this avoids proving termination of root-finding and keeps the proof about the
property that actually matters:

```rust
/// The set of nodes connected to `node` under partition `p`.
open spec fn component(p: Partition, node: nat) -> Set<nat>;

/// Merging two components never separates nodes that were already together.
proof fn lemma_union_is_monotone(p: Partition, a: nat, b: nat, u: nat, v: nat)
    requires component(p, u).contains(v)
    ensures component(union(p, a, b), u).contains(v)
{ /* ... */ }

/// A filter decision taken against an earlier partition stays valid.
proof fn lemma_filter_decision_is_stable(before: Partition, after: Partition, u: nat, v: nat)
    requires
        coarsens(before, after),
        component(before, u).contains(v),
    ensures component(after, u).contains(v)
{ /* ... */ }
```

The proof must derive stability from monotonicity, not assume it. If the
obligation reduces to restating the precondition, the proof is not substantive
and the milestone has failed its own bar.

Re-run the full gates, the property suite at the weekly case count, and the
benchmark comparison:

```sh
set -o pipefail
PROPTEST_CASES=25000 cargo nextest run -p chutoro-core \
  -E 'test(/mst::property::/)' 2>&1 | tee /tmp/proptest-chutoro-2-3-2-m4.out
make kani  2>&1 | tee /tmp/kani-chutoro-2-3-2-m4.out
make verus 2>&1 | tee /tmp/verus-chutoro-2-3-2-m4.out
cargo bench -p chutoro-benches --bench mst_prepare -- --baseline local-reference --noplot
scripts/bench-mst-prepare.sh
```

### Milestone 5: SIMD adapters (evidence-gated)

Enter this milestone only if, after Milestones 3 and 4, the `classify`,
`compact`, and `cycle_filter` groups together still account for at least 10% of
`parallel_kruskal` time at `point_count = 10_000`. Otherwise the scalar kernels
have already captured the win; record that in the decision log and in
`docs/chutoro-design.md` §6.3 as a null result, and skip to Milestone 6.

Before writing intrinsics, inspect whether LLVM already vectorized the scalar
kernels:

```sh
set -o pipefail
RUSTFLAGS="-C target-cpu=native" cargo asm -p chutoro-core \
  'chutoro_core::mst::prepare::scalar::classify' 2>&1 \
  | tee /tmp/asm-chutoro-2-3-2-classify.out
```

If the output already contains `vpcmpgt`/`vpmovmskb`-class instructions across
the loop body, handwritten intrinsics are unlikely to help; record that and
stop.

Otherwise add `simd_avx2`, `simd_avx512`, and `simd_neon` features to
`chutoro-core/Cargo.toml`, defaulting on, exactly as
`chutoro-providers/dense/Cargo.toml` does, and implement `prepare/simd/avx2.rs`,
`prepare/simd/avx512.rs`, and `prepare/simd/neon.rs`.

For compaction, AVX-512 uses `_mm512_mask_compressstoreu_epi32` directly. AVX2
has no compaction instruction; use the established mask-to-permutation-table
approach — compute an 8-bit keeper mask with `_mm256_movemask_ps`, index a
256-entry `__m256i` lookup table, and permute with
`_mm256_permutevar8x32_epi32`. Both are order-preserving, which Constraint 1
requires.

Each `unsafe` block carries a comment naming the invariant it relies on
(alignment, in-bounds length, `target_feature` availability). Add
`prepare/dispatch.rs` with the same `OnceLock` one-time-patch structure and the
same `Avx512 > Avx2 > Neon > Scalar` priority as the dense provider.

Add a parity property suite under `chutoro-core/src/mst/prepare/tests/parity/`,
mirroring `chutoro-providers/dense/src/simd/tests/parity/`. It enumerates
compiled and runtime-supported backends and compares each against the scalar
kernel index-wise, over generated lengths straddling lane boundaries (15, 16,
17, 31, 32, 33), all-keep and all-drop masks, duplicate edges, and non-finite
weights. Set equality is not sufficient; the assertion must be positional.

Add the suite to `.github/workflows/property-tests.yml`'s matrix beside the
existing `mst` entry.

### Milestone 6: documentation, ADR, and roadmap

Write `docs/adr-005-mst-edge-preparation-boundary.md`, following
`docs/adr-003-soa-prefetch-adapter-boundary.md`'s exact section order: Status,
Date, Context and problem statement, Decision drivers, Y-Statement, Options
considered, Decision outcome, Consequences, Known risks and limitations. The
decision it records is the boundary: the edge-preparation policy, staging
layout, and backend dispatch are private to `chutoro-core::mst::prepare`;
`chutoro-core` does not depend on the dense provider's SIMD module and does not
export a vectorization surface; and adoption of intrinsics adapters is gated on
the pre-registered measurement in Milestone 5.

Append an `_Implementation update (<date>)._` paragraph to
`docs/chutoro-design.md` §6.3, in the same register as the existing 2.2.x and
2.3.1 entries, recording what shipped, what was measured, and what was declined
with its null result.

Update `docs/developers-guide.md` with a new section on the MST edge
preparation kernels, covering the policy object as the single source of the
contract, the parity-suite seam, how to add a backend, and how to run
`scripts/bench-mst-prepare.sh`.

Update `docs/users-guide.md` "Error handling" with the new `NodeCountTooLarge`
variant and its `NODE_COUNT_TOO_LARGE` code, and note the now-deterministic
error selection. If Milestone 5 shipped feature flags, add them to "Feature
flags and execution strategies".

Register the ADR and this ExecPlan in `docs/contents.md`.

Mark roadmap item 2.3.2 `[x]` in `docs/roadmap.md`.

Run every gate:

```sh
set -o pipefail
make check-fmt    2>&1 | tee /tmp/check-fmt-chutoro-2-3-2-final.out
make lint         2>&1 | tee /tmp/lint-chutoro-2-3-2-final.out
make test         2>&1 | tee /tmp/test-chutoro-2-3-2-final.out
make markdownlint 2>&1 | tee /tmp/markdownlint-chutoro-2-3-2-final.out
make nixie        2>&1 | tee /tmp/nixie-chutoro-2-3-2-final.out
make kani         2>&1 | tee /tmp/kani-chutoro-2-3-2-final.out
make verus        2>&1 | tee /tmp/verus-chutoro-2-3-2-final.out
```

## Concrete steps

Work milestone by milestone. Commit after each milestone, and after each
separate refactor, using the `commit-message` skill. Do not batch milestones
into one commit; the point of frequent commits here is that a failed evidence
gate can be rolled back to a known-good measurement.

Suggested commit sequence:

1. `Add MST edge preparation benchmark and hyperfine script`
2. `Assert exact MST edge list equality against the oracle`
3. `Specify MST edge preparation behaviour with BDD scenarios`
4. `Introduce SoA edge preparation policy and scalar kernels`
5. `Pack union-find storage into 32-bit parents and 8-bit ranks`
6. `Filter cycle-forming candidate edges before union-find`
7. `Prove packed weight key reproduces total_cmp ordering`
8. `Prove union-find filter decisions stay valid under merging`
9. `Add SIMD backends for edge classification and compaction` (conditional)
10. `Record MST edge preparation boundary in ADR-005`

## Validation and acceptance

Acceptance is behavioural, not structural.

**Equivalence.** `cargo nextest run -p chutoro-core -E 'test(/mst::/)'` passes,
including the strengthened `mst_oracle_equivalence` property asserting exact
edge-list equality. Under `PROPTEST_CASES=25000`, the oracle-equivalence,
structural-invariant, concurrency-safety, and filter-soundness properties all
pass.

**Determinism.** The error-selection test passes 64 consecutive runs under
`RAYON_NUM_THREADS=8`, always reporting the lowest-index invalid edge.

**Behaviour.** All five scenarios in
`chutoro-core/tests/features/mst_edge_preparation.feature` pass.

**End to end.** Running the CLI over a fixed synthetic source produces
identical cluster assignments before and after the change:

```sh
set -o pipefail
cargo run --release --bin chutoro-cli -- --help
```

Capture the assignment output for one fixed input on the pre-change commit and
on the post-change commit and diff them; expect no differences.

**Performance.** Criterion reports an improvement on the `parallel_kruskal`
group at `point_count = 1_000` and `10_000`, with no group regressing by more
than 3%. `scripts/bench-mst-prepare.sh` corroborates at whole-binary scope. Per
`docs/developers-guide.md`, Criterion is the primary signal and `hyperfine` is
corroboration, not the other way round.

**Verification.** `make kani` passes including
`verify_weight_key_matches_total_cmp`. `make kani-full` passes including
`verify_cycle_filter_is_sound_4_nodes`. `make verus` passes including
`mst_filter_soundness.rs`. Each new harness and proof has been validated by
deliberate mutation, and that validation is recorded in `Progress`.

**Gates.** `make check-fmt`, `make lint`, `make test`, `make markdownlint`, and
`make nixie` all succeed.

Red-Green-Refactor evidence to record in `Progress`:

- Red: the error-selection test under `RAYON_NUM_THREADS=8`, with its observed
  pre-change outcome quoted verbatim.
- Green: the same command after Milestone 3, passing.
- Refactor: `make lint` and `make test` after each extraction, passing.

## Idempotence and recovery

Every step is re-runnable. Benchmarks are read-only with respect to the source
tree; Criterion baselines live under `target/criterion/` and can be deleted
freely. `make kani` and `make verus` are read-only.

If a milestone's gate fails, read the `tee`d log named in the command rather
than re-running the gate; re-run only after applying a fix. If a milestone must
be abandoned, `git revert` its commit — the commit sequence above is designed
so each milestone is independently revertible, with the sole exception that
Milestone 4's filter depends on Milestone 3's staging buffers.

Delete nothing under `/tmp` that other agents may be using; the log names above
are branch-scoped to avoid collisions.

## Artefacts and notes

Prior art consulted while planning:

- Osipov, Sanders and Singler, "The Filter-Kruskal Minimum Spanning Tree
  Algorithm", ALENEX 2009. The `filter(E)` step — discarding edges whose
  endpoints already share a component before they reach the union-find — is the
  direct ancestor of Milestone 4's pre-filter.
- Quickwit, "Filtering a Vector with SIMD Instructions (AVX-2 and AVX-512)".
  A worked Rust treatment of order-preserving stream compaction with
  `_mm512_mask_compressstoreu_epi32` and the AVX2 `_mm256_permutevar8x32_epi32`
  lookup-table alternative.
- Giesen, "Order-preserving bijections", and the standard IEEE 754 radix-sort
  key transform: flip the sign bit for non-negative values, flip every bit for
  negative values.
- Published AVX2 gather measurements showing `vgatherdps` frequently matches
  or loses to scalar loads on memory-bound access patterns, which is why gather
  is excluded from the committed scope.

## Outcomes & retrospective

To be completed at the end of each milestone and at completion. Compare the
delivered result against the purpose stated at the top: same answers,
deterministic errors, less time. Record the measured before-and-after numbers,
anything that was declined with its null result, and what would be done
differently.
