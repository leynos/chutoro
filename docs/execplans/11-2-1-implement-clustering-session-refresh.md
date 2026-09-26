# Implement `ClusteringSession::refresh`

This ExecPlan (execution plan) is a living document. The sections
`Constraints`, `Tolerances`, `Risks`, `Progress`, `Surprises & Discoveries`,
`Decision Log`, `Outcomes & Retrospective`, `Conformance Basis`, and
`Verification Plan` must be kept up to date as work proceeds.

Status: DRAFT

This plan delivers roadmap item `11.2.1` (see `docs/roadmap.md` §11.2). It must
be approved by the user before any implementation begins. It was revised after
a df12 Logisphere community-of-experts review and a user gap review; see the
Decision Log and Surprises sections for the findings that reshaped it.

## Purpose / big picture

Deliver roadmap item `11.2.1` by giving `ClusteringSession` its first
**clustering output**: a published flat-label snapshot produced by an
incremental minimum spanning tree (MST) refresh. Today a session can build an
empty Hierarchical Navigable Small World (HNSW) index, `append` point indices
(harvesting candidate edges into `pending_edges`), and compute per-point core
distances. It cannot yet turn those ingredients into cluster labels. This work
adds that final step.

After this work lands, a caller can:

1. Build a session with `ChutoroBuilder::build_session`.
2. `append` a contiguous range of point indices.
3. Call the new `session.refresh()`.
4. Read the resulting labels with the new `session.labels()`, receiving an
   `Arc<Vec<usize>>` whose length equals the total number of appended points.

Success is observable in three concrete ways:

1. After building with `with_min_cluster_size(2)`, appending `&[0, 1, 2, 3]`,
   and calling `session.refresh()`, `session.labels().len() == 4` and
   `session.snapshot_version()` has advanced by exactly one.
2. A differential unit test shows the incremental labels induce the **same
   partition** as a full batch `Chutoro::run` on the identical contiguous
   dataset (Adjusted Rand Index (ARI) `== 1.0` on small deterministic inputs,
   for a **single** refresh over a dataset of at least `min_cluster_size`
   points).
3. A `proptest` property shows that across arbitrary append/refresh sequences,
   `snapshot_version` increases by exactly one per `refresh` call, `labels()`
   length always equals the live point count, and `pending_edges` is empty
   after every `refresh`.

`refresh` reuses the exact batch primitives so incremental and batch results
stay aligned: it reweights the merged **raw** candidate edge set with the
mutual-reachability formula using current core distances, feeds the result to
the Kruskal machinery via the crate-internal `parallel_kruskal_from_edges`
entry point (see Decision Log — this subsumes the roadmap's "construct a fresh
`EdgeHarvest`" step without changing observable behaviour), and extracts labels
with `extract_labels_from_mst`.

### What this plan does and does not authorize

This plan authorizes:

- `ClusteringSession::refresh`.
- A minimal non-blocking `ClusteringSession::labels` accessor sufficient to
  observe the published snapshot.
- One shared `mutual_reachability_weight` helper, homed in the batch pipeline
  layer (`cpu_pipeline.rs`) and called *down* into by the session.
- **Raw-distance retention of the MST backbone.** The session retains the MST
  edges it selected as *raw* `CandidateEdge` values (not weighted `MstEdge`),
  so every refresh reweights from raw distances and current core distances.
  This is a small, correctness-critical refinement of design Figure 3 (see
  Decision Log and ADR-005); without it, reweighting a previously weighted
  edge ratchets weights upward across refreshes.
- Promoting `map_cpu_mst_error` and `map_cpu_hierarchy_error` from private to
  `pub(crate)` so `refresh` can reuse them.
- Adding `googletest`, `pretty_assertions`, and `insta` as dev-dependencies
  (first adoption in the workspace; see Decision Log).

This plan does **not** authorize:

- `historical_edges` population, the retention cap, or heaviest-first eviction
  of non-MST edges (roadmap `11.2.5`). `historical_edges` is **read** into the
  merge (it is always empty here) but never written.
- Count-triggered automatic refresh (`refresh_every_n`) (roadmap `11.2.3`).
- `refresh_full`, ARI/Normalized Mutual Information (NMI) drift triggers,
  baseline caching, staleness/overlap gates, or `set_baseline` (roadmap
  `11.2.4`, `11.2.6`).
- Seeded / empty-session constructors `from_source` / `new_empty` (roadmap
  `11.3`).
- Point deletion, in-place edits, compaction, or stable cluster identity
  across snapshots (design §12.2, §12.8, §13).

## Constraints

Hard invariants. Violation requires escalation, not a workaround.

- Keep scope to roadmap item `11.2.1` as bounded above.
- Reuse the batch CPU primitives without forking their behaviour: the Kruskal
  machinery (via `crate::mst::parallel_kruskal_from_edges`, which the public
  `parallel_kruskal` delegates to), `crate::extract_labels_from_mst`,
  `crate::HierarchyConfig`, `crate::MstEdge`, and `crate::CandidateEdge`. The
  incremental mutual-reachability weighting must be numerically identical to
  the batch path in `chutoro-core/src/cpu_pipeline.rs`, and must always be
  computed from **raw** distances so it does not depend on refresh history.
- Home the shared `mutual_reachability_weight` helper in the lower layer
  (`cpu_pipeline.rs`, `pub(crate)`); the session calls *down* to it. Do not
  have the batch pipeline import from the session module (the session already
  depends down on the pipeline via `map_cpu_hnsw_error`).
- Preserve existing public behaviour of `CpuHnsw::insert`,
  `CpuHnsw::insert_harvesting`, `CpuHnsw::search`, `ChutoroBuilder::build`,
  `ChutoroBuilder::build_session`, `ClusteringSession::append`,
  `ClusteringSession::recompute_core_distances`,
  `ClusteringSession::recompute_core_distances_full`, and `Chutoro::run`.
- Route every HNSW query through `CpuHnsw::search`. Do not reach into private
  HNSW adapter internals.
- Keep the session CPU-only behind the existing `cpu` feature gate. Add no GPU
  path.
- Add no new production dependency and no **new public** `ChutoroError`
  variant. Degenerate and invalid-input paths reuse the existing
  `EmptySource`, `InsufficientItems`, and `CpuMstFailure` variants (see
  Decision Log). The approved assertion crates `googletest`,
  `pretty_assertions`, and `insta` enter the workspace as **dev-dependencies
  only** (see Decision Log); they must not appear in any `[dependencies]`
  table.
- `refresh` is all-or-nothing at the snapshot level: no observable session
  state (`labels`, `snapshot_version`, `pending_edges`, `mst_backbone`) is
  mutated unless the whole refresh succeeds.
- Keep every touched source file below 400 lines. The pure rebuild plus
  helpers live in `session/refresh.rs`; the `refresh` inherent method and its
  metrics live in `session_impl.rs`. Split further if either would exceed the
  cap.
- Domain/policy logic (edge merge, mutual-reachability reweighting, Kruskal
  and label-extraction orchestration, raw-backbone recovery) must be a **pure
  function** with no HNSW, input/output, or clock dependency, so it is unit-,
  property-, and proof-testable in isolation. The impure adapter step
  (core-distance recompute via HNSW search) stays in the session methods.
  This is the hexagonal boundary this plan protects.

## Tolerances (exception triggers)

- Scope: if implementation requires touching more than 14 files or a net
  +1000 lines of code, stop and escalate.
- Interface: if a public API signature beyond `ClusteringSession::refresh`,
  `ClusteringSession::labels`, and one new `pub(crate)` helper must change,
  stop and escalate. Changing the internal `mst_edges` field type from
  `Vec<MstEdge>` to `Vec<CandidateEdge>` is authorized by this plan. Promoting
  `map_cpu_mst_error`/`map_cpu_hierarchy_error` to `pub(crate)` is authorized.
  Adding any **new public** `ChutoroError` variant is not — stop and escalate.
- Dependencies: if any new external dependency is required beyond the three
  authorized dev-dependencies (`googletest`, `pretty_assertions`, `insta`),
  stop and escalate.
- Iterations: if the deterministic commit gates still fail after 3 fix
  attempts on a single milestone, stop and escalate.
- Verification: if the Verus lemma cannot be discharged without an `assume`
  shortcut after 3 attempts, stop and record the blocker; do not ship an
  `assume`-weakened proof.
- Ambiguity: if the contiguous-point-id assumption (below) proves false for a
  required caller, stop and escalate rather than silently mislabelling
  output.

## Risks

- Risk: **Point-id contiguity.** HNSW node ids equal the source indices passed
  to `append` (`CpuHnsw::insert_with_edges(node, source)` uses `node`
  verbatim), and `core_distances`/`extract_labels_from_mst` are indexed by
  that id. `refresh` passes `node_count = point_count()`. A non-contiguous
  append (for example `&[0, 1, 5]`) makes `point_count()` 3 while an edge
  endpoint reaches id 5, so the reweight step would index `core_distances[5]`
  out of range.
  Severity: high. Likelihood: low (all shipped paths append contiguous
  prefixes; the batch pipeline has the same assumption).
  Mitigation: the pure rebuild validates every endpoint `< node_count` and
  `< core_distances.len()` and returns `ChutoroError::CpuMstFailure` with a
  descriptive message (no panic) — the same variant `parallel_kruskal` itself
  would produce for an invalid node id. Document the contiguity requirement
  on `append` and `refresh` and in ADR-005.

- Risk: **Cross-refresh weight drift (mitigated by raw retention).** If the
  MST backbone were retained as weighted `MstEdge` and reweighted again, the
  mutual-reachability `max` would ratchet weights upward whenever a core
  distance later fell (core distances are monotonically non-increasing after
  a point saturates its neighbourhood). This plan retains the backbone as
  **raw** `CandidateEdge`, so every refresh recomputes
  `max(raw, core_u, core_v)` from scratch and the result is independent of
  refresh count.
  Severity: was high; reduced to low by raw retention.
  Likelihood: low. Mitigation: raw-backbone retention plus a multi-refresh
  differential property test (append, refresh, append, refresh) asserting the
  incremental partition still matches batch within ARI ≥ 0.95.

- Risk: **Approximate-MST candidate coverage.** Even with raw retention, the
  merged candidate set (`mst_backbone` + `historical_edges` +
  `pending_edges`) is a subset of all pairwise mutual-reachability edges.
  With `historical_edges` empty until `11.2.5`, a non-MST old-old edge that a
  core-distance shift would promote is not yet retained.
  Severity: medium. Likelihood: low for the single- and few-refresh tests
  here. Mitigation: retained-non-MST coverage is the explicit job of
  `11.2.5`; the full-refresh reset is `11.2.4`; the large differential
  harness is `11.4`. Record the boundary; do not claim exactness beyond what
  is tested.

- Risk: **Degenerate small sessions.** `extract_labels_from_mst` returns
  `HierarchyError::MinClusterSizeTooLarge` when
  `min_cluster_size > node_count`, and the Kruskal machinery errors on
  `node_count == 0`.
  Severity: medium. Likelihood: high (tests and warm-up sessions hit it).
  Mitigation: `refresh` short-circuits before those calls — an empty session
  publishes an empty snapshot, and a session with
  `0 < point_count < min_cluster_size` publishes an all-noise snapshot
  (`vec![0; point_count]`). Both advance `snapshot_version`. This diverges
  intentionally from batch `run` (which errors with `InsufficientItems`)
  because a streaming session must make progress during warm-up; see
  Decision Log.

- Risk: **Label-length contract.** Verified: `extract_flat_labels` returns
  `vec![_; node_count]` (`chutoro-core/src/hierarchy/single_linkage/mod.rs`);
  noise adds a label *value*, not an entry. Severity: low. Mitigation: a
  direct `labels().len()` assertion (obligation OBL-3).

## Progress

- [ ] EP-M1: Pure domain layer — `mutual_reachability_weight` in
  `cpu_pipeline.rs`, `rebuild_mst_labels` (with raw-backbone recovery) in
  `chutoro-core/src/session/refresh.rs`, dev-dependency adoption, and the
  pure-function unit tests (red first, then green).
- [ ] EP-M2: Session wiring — `refresh` and `labels` methods, field renames,
  refresh metrics, all-or-nothing publish; unit, error, and BDD suites (red
  first, then green).
- [ ] EP-M3: Property suites — `proptest` invariants including the
  multi-refresh differential property.
- [ ] EP-M4: Formal verification — Verus lemmas OBL-1 and OBL-2; optional
  bounded Kani harness (go/no-go).
- [ ] EP-M5: Documentation — design §12.5, users' guide, developers' guide,
  ADR-005; mark roadmap `11.2.1` done.

## Surprises & discoveries

- Observation: the `MstEdge -> CandidateEdge` round-trip is field-lossless but
  **semantically lossy**.
  Evidence: `MstEdge.weight()` is a mutual-reachability weight;
  `CandidateEdge::new(.., distance, ..)` expects a raw distance
  (`chutoro-core/src/mst/mod.rs`, `cpu_pipeline.rs`).
  Impact: retaining the backbone as weighted `MstEdge` and reweighting again
  ratchets weights upward across refreshes. This plan retains the backbone as
  raw `CandidateEdge` instead. The `mst_edges` field therefore becomes
  `mst_backbone: Vec<CandidateEdge>`.

- Observation: mutual-reachability weighting in the batch path is an inline
  `.map()` closure, not a named function; `map_cpu_mst_error` and
  `map_cpu_hierarchy_error` are private.
  Evidence: `chutoro-core/src/cpu_pipeline.rs`.
  Impact: extract a shared `pub(crate) mutual_reachability_weight` in
  `cpu_pipeline.rs` and promote the two mappers to `pub(crate)`.

- Observation: the default `min_cluster_size` is 5.
  Evidence: `chutoro-core/src/builder.rs`; `session_builder()` is a bare
  `ChutoroBuilder::new()`.
  Impact: tests and examples must set `with_min_cluster_size` explicitly (or
  rely on the degenerate all-noise path) so a 4-point session does not error.

## Decision log

- Decision: Retain the MST backbone as raw `CandidateEdge` (change `mst_edges`
  to `mst_backbone: Vec<CandidateEdge>`), reweighting from raw distances every
  refresh.
  Rationale: reweighting an already-weighted edge ratchets weights upward when
  core distances fall, drifting the partition across refreshes and
  contradicting the cited FISHDBC approach (which refeeds raw triples). Raw
  retention makes the reweight refresh-count-independent and numerically
  identical to batch. The cost is one extra recovery pass matching Kruskal
  output back to raw inputs by `(source, target, sequence)`. Refines design
  Figure 3 (see Conformance Basis: DES-FIG3 deviation, accepted by the plan
  approver at approval time).
  Date/Author: 2026-07-22, planning (post expert review).

- Decision: Graceful degenerate handling instead of mirroring batch `run`.
  Rationale: `run` errors on `items == 0` (`EmptySource`) and
  `items < min_cluster_size` (`InsufficientItems`). A streaming session must
  make progress during warm-up and the `11.2.1` acceptance criterion requires
  `labels().len() == point_count` for any size. So `refresh` publishes an
  empty snapshot when `point_count == 0` and an all-noise snapshot when
  `0 < point_count < min_cluster_size`, both advancing the version. This also
  makes the 4-point flagship example pass under the default
  `min_cluster_size`.
  Date/Author: 2026-07-22, planning (post expert review).

- Decision: `refresh` auto-recomputes core distances (calls
  `recompute_core_distances()`); it does not refuse while dirty.
  Rationale: design §12.3 phase 3 says refresh "recompute[s] core distances
  for new points". The predecessor plan `11-1-4` speculated `refresh` would
  "refuse while any dirty bit is set"; that assumption is overridden here in
  favour of the friendlier, design-aligned auto-recompute. §12.4 is updated
  to match. The core-distance step is best-effort-then-abort: on failure it
  returns before any snapshot mutation, and a retry re-processes the
  remaining dirty set.
  Date/Author: 2026-07-22, planning (post expert review).

- Decision: Reuse existing `ChutoroError` variants; add none.
  Rationale: `EmptySource`/`InsufficientItems` cover degenerate sizes (if a
  future decision reverts to erroring), and `CpuMstFailure` covers a
  contiguity violation (an invalid MST node id, exactly what the Kruskal
  machinery reports). This keeps the interface tolerance intact.
  Date/Author: 2026-07-22, planning (post expert review).

- Decision: Expose a minimal `ClusteringSession::labels` accessor (returning
  `Arc<Vec<usize>>`) in this plan, ahead of roadmap `11.2.2`.
  Rationale: the `11.2.1` acceptance criterion is phrased in terms of
  `session.labels()`. A `&self` accessor returning `Arc::clone(&self.labels)`
  is inherently non-blocking; the `self.labels = Arc::new(...)` publish is
  the atomic `Arc` swap design §12.6 already promises for a single writer.
  Roadmap `11.2.2` formalizes the multi-reader guarantees without changing
  this signature.
  Date/Author: 2026-07-22, planning.

- Decision: `refresh` computes `node_count = self.point_count()` and requires
  appended indices to form a contiguous prefix `[0, point_count)`.
  Rationale: HNSW node id equals the source index, and both `core_distances`
  and `extract_labels_from_mst` are indexed by that id; this matches the
  batch pipeline's assumption. The pure rebuild guards it and returns
  `CpuMstFailure` rather than panicking on an out-of-range endpoint.
  Date/Author: 2026-07-22, planning.

- Decision: `historical_edges` is read into the merge but not populated here.
  Rationale: population, the `2×` cap, and heaviest-first eviction are
  roadmap `11.2.5` (`requires 11.2.1`). Reading the (empty) buffer now keeps
  the merge signature stable.
  Date/Author: 2026-07-22, planning.

- Decision: `snapshot_version` is a per-refresh-call counter.
  Rationale: it advances once per `refresh` call, including empty and no-op
  refreshes. This is a call counter, not a content-change token; `11.2.2`
  formalizes the reader contract. Stated so consumers do not treat every
  increment as a content change.
  Date/Author: 2026-07-22, planning (post expert review).

- Decision: Call `parallel_kruskal_from_edges` directly instead of the
  roadmap's "construct a fresh `EdgeHarvest`" wording.
  Rationale: the `pub(crate)` entry point at `chutoro-core/src/mst/mod.rs`
  accepts `impl IntoIterator<Item = &CandidateEdge>` and is what the public
  `parallel_kruskal` itself delegates to. `EdgeHarvest::from_unsorted` exists
  to guarantee harvest ordering, but `prepare_edge_list` re-sorts by weight
  regardless, so routing through `EdgeHarvest` adds a redundant O(E log E)
  sort and an intermediate allocation purely to satisfy the wrapper type.
  The observable result is identical (same validation, canonicalization,
  sort, dedup, and union-find). The roadmap's stated intent — "run
  `parallel_kruskal` over the combined set" — is met via its own internal
  entry point; recorded here because it deviates from the item's literal
  wording (see Conformance Basis).
  Date/Author: 2026-08-16, planning (gap review).

- Decision: Dedup of the merged candidate set relies on
  `mst::prepare_edge_list`, not on any pre-merge filtering.
  Rationale: the three merged buffers (`mst_backbone`, `historical_edges`,
  `pending_edges`) can legitimately overlap — an edge selected into the MST
  may also be re-harvested by a later insertion. `prepare_edge_list` dedups
  identical `(weight, source, target)` triples after canonicalization, and
  duplicates of the same raw edge reweight identically under the same core
  distances, so overlap collapses inside Kruskal at no correctness cost.
  `EdgeHarvest` never dedups, so nothing is lost by bypassing it. Obligation
  OBL-7 asserts this.
  Date/Author: 2026-08-16, planning (gap review).

- Decision: Adopt `googletest`, `pretty_assertions`, and `insta` as
  dev-dependencies in this milestone.
  Rationale: the `rust-unit-testing` brief names them as the approved rich
  assertion stack, yet none is used anywhere in the workspace; deferring
  adoption "until they are used" is circular. This is the first new test
  suite since their approval, so it starts the practice: readable diffs for
  vector equality, matcher-based structural claims, and one snapshot for the
  deterministic fixture. Dev-dependencies only; the
  no-new-production-dependency constraint is unchanged.
  Date/Author: 2026-08-16, planning (gap review).

- Decision: Prior-art alignment with FISHDBC (Dell'Amico, 2019,
  arXiv:1910.07283).
  Rationale: FISHDBC maintains an *approximate* MST updated incrementally by
  feeding batches of raw candidate triples back through Kruskal, updating
  mutual-reachability distances as the `max` of endpoint core distances and
  the raw distance. This plan mirrors that exactly (merge raw candidates →
  reweight by `max` → Kruskal), confirming the design's §12.5 strategy.
  Date/Author: 2026-07-22, planning.

## Outcomes & retrospective

To be completed at delivery. Note: each refresh rebuilds and re-sorts the
whole combined candidate set (O(E log E)) even for a no-op refresh; the
cut-based incremental update sketched in design §12.5 is the future
optimization target (out of scope here). Compare the shipped `refresh` against
the three observable success criteria in Purpose. Before marking this plan
`COMPLETE`, reconcile every discovery with the Conformance Basis artefacts:
the DES-FIG3 raw-backbone refinement and the §12.4 auto-recompute wording
must have landed in `docs/chutoro-design.md`, and ADR-005 must exist and be
referenced from the design document.

## Context and orientation

The reader needs no prior plans. Relevant facts, with full paths:

- The session lives in `chutoro-core/src/session/`:
  - `mod.rs` — the `ClusteringSession<D>` struct and read-only accessors
    (`config`, `point_count`, `snapshot_version`, `core_distance`). The
    struct declares placeholder fields `_mst_edges: Vec<MstEdge>`,
    `_historical_edges: Vec<CandidateEdge>`, `_labels: Arc<Vec<usize>>`, and
    `_last_refresh_len: usize`, awaiting this item.
  - `session_impl.rs` — construction and `append` (calls
    `CpuHnsw::insert_harvesting`, extends `pending_edges`, marks
    core-distance dirty). It already reuses
    `crate::cpu_pipeline::map_cpu_hnsw_error` via `map_hnsw_error`.
  - `core_distance.rs` — `recompute_core_distances` (incremental) and
    `recompute_core_distances_full` (batch-equivalent), plus pure helpers.
  - `config.rs` — `SessionConfig` (`min_cluster_size() -> NonZeroUsize`,
    `hnsw_params() -> &HnswParams`, `refresh_policy()`).
  - `tests.rs` registers the child test modules; `tests/common.rs` provides
    `SessionTestSource`, `#[fixture] session_builder`, `make_session`, and
    `harvest_expected_edges`; siblings `append.rs`, `core_distance.rs`,
    `core_distance_errors.rs`, `properties.rs`, `concurrency.rs`.
- The batch pipeline `chutoro-core/src/cpu_pipeline.rs` shows the target
  sequence: build+harvest, per-point core distance, inline mutual-reachability
  reweighting, `parallel_kruskal(items, &harvest)`, then
  `extract_labels_from_mst(items, forest.edges(),
  HierarchyConfig::new(mcs))`. It also defines the crate-internal error
  mapper `map_cpu_hnsw_error` (`pub(crate)`) plus `map_cpu_mst_error` and
  `map_cpu_hierarchy_error`, which are currently **private** and must be
  promoted to `pub(crate)` for `refresh` to reuse them.
- Primitive signatures (verified), each defined in the noted module:

  ```text
  // chutoro-core/src/mst/mod.rs
  parallel_kruskal(node_count: usize, edges: &EdgeHarvest)
      -> Result<MinimumSpanningForest, MstError>
  // pub(crate) sibling; parallel_kruskal delegates to it. prepare_edge_list
  // validates, canonicalizes (source <= target), sorts by weight, and dedups
  // identical (weight, source, target) triples before union-find. Note the
  // dedup lives HERE, not in EdgeHarvest, which never dedups.
  parallel_kruskal_from_edges(node_count: usize,
      edges: impl IntoIterator<Item = &CandidateEdge>)
      -> Result<MinimumSpanningForest, MstError>
  MinimumSpanningForest::edges() -> &[MstEdge]

  // chutoro-core/src/hierarchy/mod.rs  (returns a node_count-length vector)
  extract_labels_from_mst(node_count: usize, edges: &[MstEdge],
      config: HierarchyConfig) -> Result<Vec<usize>, HierarchyError>

  // chutoro-core/src/hnsw/types.rs
  CandidateEdge::new(source, target, distance, sequence)
  CandidateEdge { source(), target(), distance(), sequence() }
  MstEdge { source(), target(), weight(), sequence() }  // no raw distance
  ```

- Public re-exports for all of the above are in `chutoro-core/src/lib.rs`
  behind `#[cfg(feature = "cpu")]`. `adjusted_rand_index(&[usize], &[usize])`
  is public in `crate::clustering_quality`.
- Definitions of terms:
  - **Core distance** of point `p` at `m = min_cluster_size`: the distance
    from `p` to its `m`-th nearest neighbour (self excluded).
  - **Mutual-reachability weight** of a raw edge `(u, v)` with raw distance
    `d`: `max(d, core[u], core[v])`.
  - **MST backbone**: the raw `CandidateEdge` values whose reweighted forms
    were selected into the MST by the previous refresh; retained for the next
    merge.
  - **Snapshot version**: a monotonically increasing `u64` bumped once per
    `refresh` call.

### Documentation and skills to consult

- Design: `docs/chutoro-design.md` §12.3 (session architecture), §12.4 (edge
  harvesting and core distances), §12.5 (incremental MST refresh strategy —
  the primary spec), §12.6 (concurrency model), §12.7 (differential testing).
- Roadmap: `docs/roadmap.md` §11.2 (this item and siblings
  `11.2.2`–`11.2.6`, and `11.4`).
- Predecessor execplans: `docs/execplans/11-1-3-clustering-session-append.md`
  and `docs/execplans/11-1-4-incremental-core-distance-computation.md`.
- Testing docs: `docs/property-testing-design.md`,
  `docs/rust-testing-with-rstest-fixtures.md`,
  `docs/rust-doctest-dry-guide.md`,
  `docs/complexity-antipatterns-and-refactoring-strategies.md`.
- Verification: `docs/verus-toolchain.md`,
  `docs/adr-002-adoption-of-kani-formal-verification.md`.
- Style: `docs/documentation-style-guide.md`.
- Skills: `execplans` (this document), `rust-router` →
  `rust-unit-testing` / `rust-errors` / `arch-crate-design`, `proptest`,
  `verus`, `kani`, `hexagonal-architecture`, `leta`, `nextest`,
  `commit-message`, `pr-creation`.

## Conformance basis

No Terms of Reference document exists for this repository; the upstream
artefacts are the roadmap and the technical design document. Traced items use
these identifiers:

- **RM-11.2.1** — roadmap item 11.2.1 in `docs/roadmap.md` (revision at
  branch point: commit `2aafc3e`): implement `ClusteringSession::refresh`
  with the stated acceptance criteria (label length equals total points;
  `snapshot_version` increments by exactly one per refresh).
- **DES-12.5** — `docs/chutoro-design.md` §12.5 (same revision): the
  nine-step incremental MST refresh strategy (merge, reweight, Kruskal,
  extract, publish).
- **DES-12.3 / DES-12.4 / DES-12.6 / DES-12.7** — design §12.3 (session
  architecture and lifecycle), §12.4 (core distances and
  mutual-reachability), §12.6 (single-writer/multi-reader model), §12.7
  (differential testing thresholds).
- **DES-FIG3** — design Figure 3, the `ClusteringSession` state sketch
  (`mst_edges: Vec<MstEdge>`).
- ADRs: ADR-002 (Kani adoption) and ADR-003 (adapter boundaries) govern
  verification and layering conventions. ADR-005 is authored by this plan.

Trace links (upstream → milestone → acceptance evidence):

```plaintext
RM-11.2.1 -> DES-12.5 -> EP-M2
  -> session::tests::refresh::refresh_publishes_labels_of_point_count_length
  -> session::tests::refresh::refresh_advances_snapshot_version_by_one
RM-11.2.1 -> DES-12.7 -> EP-M2/EP-M3
  -> session::tests::refresh::refresh_matches_batch_partition
  -> refresh_properties::multi_refresh_matches_batch_within_ari_bound
DES-12.4 (mutual reachability) -> EP-M1 -> OBL-1 (Verus, EP-M4)
DES-12.5 step 8 (partition output) -> EP-M1
  -> refresh::tests::backbone_recovery_matches_forest
DES-12.6 (atomic snapshot publish) -> EP-M2
  -> refresh_errors::failed_refresh_preserves_snapshot_and_pending
```

Known deviations carried by this plan (each recorded in the Decision Log and
requiring the plan approver's acceptance at the approval gate):

- **DES-FIG3 deviation**: `mst_edges: Vec<MstEdge>` becomes
  `mst_backbone: Vec<CandidateEdge>` (raw retention). Upstream impact:
  Figure 3 and §12.5 step 8 wording must be updated in EP-M5; ADR-005 records
  the rationale. Downstream impact: roadmap `11.2.5` retains raw non-MST
  edges against the same representation.
- **RM-11.2.1 literal-wording deviation**: "construct a fresh `EdgeHarvest`"
  is realized via `parallel_kruskal_from_edges`, which the public wrapper
  delegates to. No behavioural difference; roadmap text is satisfied in
  intent. No upstream edit required.
- **Degenerate-size behaviour**: batch `run` errors below
  `min_cluster_size`; `refresh` publishes an all-noise snapshot instead.
  Upstream impact: §12.5 gains a sentence in EP-M5 documenting warm-up
  behaviour.

When any traced item changes during implementation, update this section and
the affected links before continuing.

## Verification plan

Verification was co-designed with the implementation structure: the refresh
pipeline is factored so that everything between "buffers in" and "labels +
backbone out" is the pure function `rebuild_mst_labels`, making every
obligation below testable or provable without HNSW, clocks, or input/output.

Non-trivial axioms this plan relies on (not re-verified here):

- **AXM-1 Kruskal correctness.** `parallel_kruskal_from_edges` produces the
  MST/forest of its input edge set. Already verified upstream by the Kani
  harnesses and the parallel-Kruskal property suite (roadmap 1.5.1, 1.6.5).
- **AXM-2 Label extraction contract.** `extract_labels_from_mst` returns a
  `node_count`-length vector of contiguous labels (verified by reading
  `extract_flat_labels`; re-asserted cheaply by OBL-3 at the session level).
- **AXM-3 `f32::max` semantics.** IEEE-754 `max` on finite inputs is
  commutative, associative, and idempotent; core distances and raw distances
  entering the reweight are finite (HNSW rejects non-finite distances;
  `core_distance` filters non-finite cells).
- **AXM-4 Deterministic fixtures.** Under the fixed proptest seed and the
  deterministic HNSW tie-break rules, identical append sequences produce
  identical harvests, so snapshot-based evidence is stable.
- **AXM-5 Third-party dev-dependencies.** `rstest`, `rstest-bdd`, `proptest`,
  `googletest`, `pretty_assertions`, and `insta` behave per their documented
  interfaces; no attempt is made to verify their internals.

Obligations. Each lists method, rationale, domain, artefact, evidence, and a
non-vacuity argument recorded beside it.

- Obligation: **OBL-1 reweight faithfulness (lemma).**
  `mutual_reachability_weight(d, cs, ct)` equals the pointwise maximum of its
  three arguments; it is commutative in `cs`/`ct`, `>=` each argument, and —
  the substantive part — depends only on the raw distance and the two current
  core distances, never on any previously stored weight. Hence two sessions
  reaching the same `(raw edge set, core distances)` produce identical
  weighted edges regardless of refresh history.
  Method: formal proof (Verus), per the `rust-verification` selection rules —
  the guarantee must hold for all admissible inputs, not a sampled range.
  Domain: all finite `f32` triples (modelled as reals in spec; finiteness is
  AXM-3's precondition, mirrored as a `requires` clause).
  Artefact: `verus/session_refresh.rs`, run by `make verus`.
  Evidence: `prover-tools verus run --proof-file verus/session_refresh.rs`
  exits 0 with no `assume`; before implementation the proof file's lemma is
  stated with an open body and fails, which is the red state.
  Non-vacuity: the `requires` clause (finiteness) is inhabited — witness
  `(1.0, 2.0, 0.5)` is asserted concretely in the proof body; the
  history-independence claim is exactly the property the double-weighting
  bug violated, so a variant lemma modelling the buggy fold
  (`max(prev_weight, cs, ct)` with `prev_weight > max(d, cs, ct)`) is shown
  to yield a different result via a concrete counterexample assertion,
  proving the lemma distinguishes correct from buggy semantics.

- Obligation: **OBL-2 merge multiset preservation (lemma).** The combined
  candidate sequence has length
  `mst_backbone.len() + historical_edges.len() + pending_edges.len()`, and
  reweighting is index-preserving: element `i` of the output has the same
  `source`, `target`, and `sequence` as element `i` of the concatenated
  input, differing only in weight.
  Method: formal proof (Verus) over unbounded sequences (`Seq<EdgeSpec>`),
  by induction; a bounded check would not cover all lengths.
  Domain: all sequences of spec edges.
  Artefact: `verus/session_refresh.rs` (same proof file, separate lemmas).
  Evidence: as OBL-1; discharge condition is a clean `make verus` run with
  no `assume`.
  Non-vacuity: the inductive step is exercised by non-empty witnesses
  (concrete 2-element and 3-buffer instantiations asserted in the proof);
  the length equality fails on any implementation that filters or dedups
  during merge, which is a real candidate mistake (dedup belongs to
  `prepare_edge_list`, not the merge — see Decision Log).

- Obligation: **OBL-3 label-length invariant.** After every successful
  `refresh`, `session.labels().len() == session.point_count()`.
  Method: property test (generated append/refresh interleavings) plus a
  parameterized unit test for the explicit partitions (empty session,
  below-`min_cluster_size` session, exactly-`min_cluster_size` session,
  larger session).
  Rationale: the domain is a range of operation sequences, not a finite
  enumeration, so examples alone are insufficient; a proof is
  disproportionate because the invariant composes AXM-1/AXM-2 with
  short-circuit branches that the property test exercises directly.
  Domain: append batches of 0–32 indices over sources of 1–64 points,
  1–4 refreshes, `min_cluster_size` in 1–6.
  Artefact: `chutoro-core/src/session/tests/refresh_properties.rs`
  (proptest, `suite_proptest_config`) and
  `chutoro-core/src/session/tests/refresh.rs` (rstest `#[case]`s).
  Evidence: red — tests fail to compile before `refresh`/`labels` exist;
  green — `cargo nextest run -p chutoro-core -E 'test(refresh)'` passes.
  Non-vacuity: the generator's acceptance is total (no `prop_filter`), and
  the four explicit partitions each have a named `#[case]` witness; a seeded
  fault (temporarily truncating the label vector by one in
  `rebuild_mst_labels`) must fail both the property and the unit case — this
  mutation check is performed once during EP-M3 and recorded in Artefacts.

- Obligation: **OBL-4 snapshot-version discipline.** `snapshot_version`
  increases by exactly one per successful `refresh` call and is otherwise
  unchanged.
  Method: property test over interleavings (shares the OBL-3 harness) plus a
  unit test asserting the delta across two consecutive refreshes.
  Domain: as OBL-3.
  Artefact: `refresh_properties.rs`; `refresh.rs`
  (`refresh_advances_snapshot_version_by_one`).
  Evidence: as OBL-3.
  Non-vacuity: sequences with ≥ 2 refreshes are generated (the refresh count
  strategy starts at 1 and the unit test pins the 2-refresh case), so the
  "exactly one" claim is distinguished from "at least one"; a seeded
  double-increment fault must fail the unit case.

- Obligation: **OBL-5 all-or-nothing publish.** A failed `refresh` leaves
  `labels`, `snapshot_version`, `pending_edges`, and `mst_backbone`
  unchanged; a retry after the failure cause is removed succeeds.
  Method: parameterized error-path tests (the failure modes form a finite
  partition: contiguity violation; injected `DataSource` failure during
  core-distance recompute).
  Rationale: each failure mode is an explicit branch; exhaustive enumeration
  is practical, so parameterized tests are proportionate.
  Domain: the two named failure modes, each with a state-snapshot
  before/after comparison.
  Artefact: `chutoro-core/src/session/tests/refresh_errors.rs`, using the
  existing `FailableSource` pattern from `core_distance_errors.rs`.
  Evidence: red — tests fail while `refresh` mutates state before its
  fallible calls (or does not exist); green — pass once the commit block is
  ordered after all fallible calls.
  Non-vacuity: the contiguity witness (`append(&[0, 1, 3])` against a
  4-point source, index 2 skipped — accepted by `append`, invalid for the
  rebuild) demonstrably reaches the guard because the same input passes once
  index 2 is appended; moving `snapshot_version += 1` above the fallible
  calls is the representative mutation the test must reject.

- Obligation: **OBL-6 batch partition equivalence.** For a single refresh
  over a contiguous dataset of at least `min_cluster_size` points, the
  incremental labels induce the same partition as `Chutoro::run` on the
  identical dataset (`adjusted_rand_index == 1.0`); across multiple
  append/refresh rounds, ARI ≥ 0.95 (DES-12.7's threshold).
  Method: differential unit test (single refresh) plus differential property
  test (multi-refresh), both against the real batch pipeline — a faithful
  contract-level boundary is unnecessary because the real oracle is cheap at
  test sizes.
  Domain: deterministic small fixture (unit); generated datasets of 8–64
  points with 2–4 separated value clusters, split into 2–3 append rounds
  (property).
  Artefact: `refresh.rs` (`refresh_matches_batch_partition`, plus an
  `insta::assert_debug_snapshot!` of the fixture's label vector);
  `refresh_properties.rs`
  (`multi_refresh_matches_batch_within_ari_bound`).
  Evidence: red — fails to compile before the API exists; green — passes;
  the snapshot file is committed beside the test.
  Non-vacuity: fixtures contain ≥ 2 genuine clusters, so ARI is not
  trivially 1.0 for degenerate labellings (an all-noise or single-cluster
  output scores < 1.0 against the batch oracle); a seeded fault replacing
  `max` with `min` in `mutual_reachability_weight` must drive ARI below the
  bound — this mutation check is performed once during EP-M3 and recorded in
  Artefacts.

- Obligation: **OBL-7 overlap tolerance.** Duplicating any input edge across
  two of the three merge buffers leaves the output labels and backbone
  unchanged.
  Method: parameterized unit test on `rebuild_mst_labels` (pure, no session
  needed).
  Domain: the three buffer-pair combinations, each with a duplicated edge.
  Artefact: `chutoro-core/src/session/refresh.rs` `#[cfg(test)]` module.
  Evidence: red-then-green with the rest of EP-M1's pure-fn tests.
  Non-vacuity: the duplicated edge is chosen to be MST-selected in the
  deduplicated baseline, so a merge that double-counts it would either alter
  the backbone (two copies recovered) or, with distinct sequence numbers,
  survive dedup and change the candidate count — the test asserts both the
  labels and the recovered backbone match the no-duplicate baseline exactly.

- Obligation: **OBL-8 bounded rebuild safety (optional, go/no-go).** Within
  a 2–3 node bound, `rebuild_mst_labels` never panics: it returns
  `Ok` with `labels.len() == node_count` or `Err` for out-of-range
  endpoints.
  Method: bounded model check (Kani), `#[cfg(kani)]`, tight
  `#[kani::unwind]`.
  Rationale: complements OBL-3/OBL-5 with exhaustive small-bound coverage of
  panic freedom (index arithmetic), which sampling cannot guarantee; kept
  optional because heap-`Vec` harnesses hit a combinatorial cliff and MST
  structural safety is already covered upstream (AXM-1).
  Domain: `node_count ≤ 3`, symbolic edge endpoints and finite weights.
  Artefact: harness in `chutoro-core/src/session/refresh.rs` behind
  `#[cfg(kani)]`; run manually via `cargo kani`, not added to `make test` or
  the default `make kani` tier without a follow-up decision.
  Evidence: `cargo kani --harness verify_rebuild_bounded` completes with all
  checks passed within a few minutes; otherwise record the cost in the
  Decision Log and drop the harness (Verus plus the test suites remain the
  verification of record).
  Non-vacuity: the symbolic endpoint domain includes values `>= node_count`,
  so the error branch is reachable; a `kani::cover!` on the `Ok` branch with
  a non-empty edge set proves the success path is exercised too.

No other non-trivial invariants are introduced: the metrics emission and
`tracing` instrumentation are observability side channels with no behavioural
contract, covered by the existing metrics test pattern only if EP-M2 adds
counters (mirroring `session/tests/metrics.rs`).

## Plan of work

Stages within each milestone follow red → green → refactor; the milestones
themselves are the plateaus listed in the next section. Each milestone ends
with the deterministic commit gates (`make check-fmt`, `make lint`,
`make test`, plus `make verus` from EP-M4 onward) and, once those pass, a
`coderabbit review --agent` pass whose concerns are cleared before moving on.
Delegate full gate runs to the `scrutineer` subagent.

### Stage A — understand and propose (no code changes)

Completed during planning and confirmed by expert review; recorded in Context
and Conformance Basis. No edits.

### EP-M1 — pure domain layer

1. Add `googletest`, `pretty_assertions`, and `insta` to `chutoro-core`'s
   `[dev-dependencies]` (via the workspace dependency table if one exists;
   crate-level otherwise). Usage assignments, chosen for expressiveness
   rather than blanket replacement:
   - `pretty_assertions::assert_eq!` wherever whole label vectors or edge
     lists are compared, so failures render a readable diff.
   - `googletest` matchers (`assert_that!` with `len`, `each`, `eq`,
     `matches_pattern!`) for structural claims — "every label is below the
     cluster-count bound", "the error is `CpuMstFailure` with this code".
   - `insta::assert_debug_snapshot!` for the deterministic single-refresh
     label vector of the fixed fixture (OBL-6); review with
     `cargo insta review`; commit the `.snap` file.
   Plain `assert!`/`assert_eq!` remains fine for trivial scalar checks;
   `prop_assert!` continues inside `proptest` blocks.
2. Red: write the pure-function tests in
   `chutoro-core/src/session/refresh.rs` `#[cfg(test)]` (they fail to
   compile until the functions exist): `mutual_reachability_weight`
   commutativity/lower-bound/idempotence cases, and `rebuild_mst_labels`
   label length, empty and all-noise short-circuits, out-of-range endpoint
   error, raw-backbone recovery, and OBL-7 overlap tolerance.
3. Green: in `chutoro-core/src/cpu_pipeline.rs`, add a `pub(crate)`
   `mutual_reachability_weight(distance, core_source, core_target) -> f32`
   returning `distance.max(core_source).max(core_target)`; refactor the
   batch path's inline `.map()` to call it (behaviour unchanged; batch tests
   stay green); promote `map_cpu_mst_error` and `map_cpu_hierarchy_error` to
   `pub(crate)`. Then implement the pure rebuild in
   `chutoro-core/src/session/refresh.rs`:

   ```rust
   pub(super) struct RefreshOutcome {
       pub labels: Vec<usize>,
       pub mst_backbone: Vec<CandidateEdge>, // raw edges selected into MST
   }

   pub(super) fn rebuild_mst_labels(
       node_count: usize,
       mst_backbone: &[CandidateEdge],   // raw, from the prior refresh
       historical_edges: &[CandidateEdge],
       pending_edges: &[CandidateEdge],
       core_distances: &[f32],
       min_cluster_size: std::num::NonZeroUsize,
   ) -> crate::Result<RefreshOutcome>;
   ```

   `rebuild_mst_labels`:
   - Short-circuits `node_count == 0` to an empty `RefreshOutcome`.
   - Short-circuits `node_count < min_cluster_size` to all-noise labels
     (`vec![0; node_count]`) and an empty backbone (a warm-up session keeps
     no structure yet).
   - Builds the combined raw `Vec<CandidateEdge>` = `mst_backbone` ++
     `historical_edges` ++ `pending_edges` (all raw).
   - Validates every endpoint `< node_count` and `< core_distances.len()`,
     returning `ChutoroError::CpuMstFailure` on violation (no panic).
   - Reweights each edge with `mutual_reachability_weight`, preserving
     `(source, target, sequence)` (OBL-2's index-preservation contract).
   - `parallel_kruskal_from_edges(node_count, reweighted.iter())` (no
     `EdgeHarvest` construction; see Decision Log) mapped via
     `map_cpu_mst_error`, then `extract_labels_from_mst(node_count,
     forest.edges(), HierarchyConfig::new(min_cluster_size))` mapped via
     `map_cpu_hierarchy_error`.
   - Overlaps between the three merged buffers need no pre-filtering:
     `mst::prepare_edge_list` canonicalizes and dedups identical
     `(weight, source, target)` triples inside Kruskal, and duplicates of
     the same raw edge reweight identically under the same core distances,
     so they collapse there.
   - Recovers the raw backbone: build a set of `(source, target, sequence)`
     keys from `forest.edges()` (canonicalized), then filter the raw
     combined set to those keys. Returns labels plus that raw backbone.
4. Refactor and validate: focused tests pass; run the milestone gates.

### EP-M2 — session wiring

1. Red: unit tests in a new `chutoro-core/src/session/tests/refresh.rs`
   (registered from `session/tests.rs`), using the `session_builder` fixture
   and `make_session`; every test expecting real clusters builds with
   `with_min_cluster_size(2)` (or smaller) and enough points:
   - `refresh_publishes_labels_of_point_count_length` — build with
     `with_min_cluster_size(2)`; append `&[0,1,2,3]`; `refresh()`; assert
     `labels().len() == 4`.
   - `refresh_advances_snapshot_version_by_one` — delta exactly one per call
     across two refreshes (OBL-4).
   - `refresh_clears_pending_edges` — `pending_edges` empty after refresh
     (child-module private-field access, as `append.rs` uses).
   - `refresh_on_empty_session_publishes_empty_snapshot` — no appends;
     `refresh()`; `labels().is_empty()` and version advanced by one.
   - `refresh_below_min_cluster_size_publishes_all_noise` —
     `with_min_cluster_size(5)`, append `&[0,1,2,3]`, `refresh()`; assert
     `labels() == [0,0,0,0]` and version advanced by one.
   - `refresh_matches_batch_partition` — differential (OBL-6): run
     `Chutoro::run`, map its `assignments()` (`&[ClusterId]`) to
     `Vec<usize>` via `ClusterId::get() as usize`, and assert
     `adjusted_rand_index(&incremental, &batch) == 1.0`; snapshot the label
     vector with `insta`.
   Error-path tests in `chutoro-core/src/session/tests/refresh_errors.rs`
   (OBL-5): `failed_refresh_preserves_snapshot_and_pending` and
   `contiguity_violation_reports_cpu_mst_failure`.
   BDD: **reuse the existing session harness**
   (`chutoro-core/tests/features/session_append.feature` and
   `chutoro-core/tests/session_append_bdd.rs`), as `11-1-4` did. Add the
   scenarios below plus `When I refresh the session` /
   `Then the label snapshot has length {count:usize}` /
   `Then the snapshot version is {version:u64}` steps.

   ```gherkin
     Scenario: Refreshing a populated session publishes labels for every point
       Given a clustering session over 4 points with min cluster size 2
       When I append points "0,1,2,3"
       And I refresh the session
       Then the label snapshot has length 4
       And the snapshot version is 1

     Scenario: Refreshing an empty session bumps the version, empty labels
       When I refresh the session
       Then the label snapshot is empty
       And the snapshot version is 1
   ```

   Confirm the red state: the suite fails to compile because
   `refresh`/`labels` do not exist (the expected red reason), converting to
   assertion failures as the API lands.
2. Green: in `session/mod.rs`, rename `_mst_edges: Vec<MstEdge>` to
   `mst_backbone: Vec<CandidateEdge>` and `_historical_edges ->
   historical_edges`, `_labels -> labels`, `_last_refresh_len ->
   last_refresh_len`; update `session_impl.rs` construction; drop the unused
   `MstEdge` import if nothing else needs it. Add
   `pub fn labels(&self) -> Arc<Vec<usize>>` returning
   `Arc::clone(&self.labels)`. Register the refresh metrics in the
   construction `describe_*` block and implement
   `pub fn refresh(&mut self) -> Result<()>` in `session_impl.rs`:
   - Call `self.recompute_core_distances()?` (impure adapter step).
   - Set `node_count = self.point_count()` and call the pure rebuild:

   ```rust
   let outcome = rebuild_mst_labels(
       node_count,
       &self.mst_backbone,
       &self.historical_edges,
       &self.pending_edges,
       &self.core_distances,
       self.config.min_cluster_size(),
   )?;
   ```

   - Publish only after every fallible call succeeds, as one commit block,
     so `refresh` is all-or-nothing (OBL-5):

   ```rust
   // commit block — all-or-nothing snapshot publish
   self.labels = Arc::new(outcome.labels);
   self.mst_backbone = outcome.mst_backbone;
   self.pending_edges.clear();
   self.last_refresh_len = node_count;
   self.snapshot_version += 1;
   ```

   - Record metrics under the existing `#[cfg(feature = "metrics")]`
     pattern: `chutoro.session.refresh.seconds` (duration histogram),
     `chutoro.session.refresh.errors_total` (counter labelled by reason),
     `chutoro.session.refresh.candidate_edges` (merged edge-count
     histogram), and `chutoro.session.refresh.cluster_count`
     (distinct-label histogram), each `describe_*`-registered in
     construction.
   - Document errors: `ChutoroError::DataSource`/`CpuHnswFailure` from the
     core-distance recompute, and `CpuMstFailure`/`CpuHierarchyFailure` from
     the pure rebuild.
3. Refactor and validate: unit, error, and BDD suites green; milestone
   gates; record the EP-M2 conformance check.

### EP-M3 — property suites

Add `chutoro-core/src/session/tests/refresh_properties.rs` (proptest,
`suite_proptest_config`) covering OBL-3, OBL-4, determinism (identical
append/refresh sequences under the fixed seed produce identical label
vectors), `pending_edges` emptiness after every refresh, and OBL-6's
multi-refresh differential property (append, refresh, append, refresh;
ARI ≥ 0.95 against batch on the final dataset). Perform the two seeded-fault
mutation checks recorded in the Verification Plan (truncated labels for
OBL-3; `min` substitution for OBL-6), confirm the suites reject them, restore
the code, and paste the failing output into Artefacts.

### EP-M4 — formal verification

Author `verus/session_refresh.rs` with spec mirrors of the edge types (the
repository's established pattern from the roadmap 1.6.4 proofs) and discharge
OBL-1 and OBL-2, including their non-vacuity witnesses, with no `assume`.
Register the proof file so `make verus` runs it. Attempt OBL-8 (bounded Kani
harness); apply its go/no-go rule and record the outcome either way.

### EP-M5 — documentation

- `docs/chutoro-design.md` §12.5: mark `11.2.1` implemented, worded so the
  reader knows `historical_edges` retention/eviction is still deferred to
  `11.2.5` (only the empty read path exists). Record the raw-backbone
  retention (DES-FIG3 refinement), the `node_count = point_count` contiguity
  invariant, the degenerate all-noise behaviour and its intentional
  divergence from batch `run`, the auto-recompute behaviour (update §12.4's
  dirty-set note), and the early minimal `labels()`.
- `docs/users-guide.md`: document the `append -> refresh -> labels` workflow
  and the two new public methods, with a runnable doctest that sets
  `with_min_cluster_size` explicitly; state the warm-up (all-noise)
  behaviour.
- `docs/developers-guide.md`: document the hexagonal split (pure
  `rebuild_mst_labels` vs the HNSW-backed core-distance adapter), the shared
  lower-layer `mutual_reachability_weight` helper, the raw-backbone
  retention rationale, the dedup-in-`prepare_edge_list` fact, and the
  contiguity invariant with its `CpuMstFailure` guard.
- `docs/adr-005-incremental-refresh-domain-boundary.md`: a Y-Statement ADR
  capturing (a) the pure-domain refresh boundary, (b) raw-backbone retention
  and why weighted retention is unsound across refreshes, (c) the contiguous
  point-id invariant and its `CpuMstFailure` guard, (d) graceful degenerate
  handling diverging from batch `run`, and (e) exposing `labels()` ahead of
  `11.2.2`. Reference it from the design doc.
- `docs/roadmap.md`: tick `11.2.1` to `[x]` once all gates pass.

## Milestones and plateaus

No compatibility machinery is prescribed anywhere in this plan: the session
API is pre-1.0 with no external consumers, `mst_backbone` is a private field,
and every rename is applied atomically with its callers. Compatibility
decision for all milestones: **none required** (nothing deployed, no wire or
persisted format exists yet — session checkpointing arrives in §13).

- **EP-M1 — pure domain layer.**
  Outcome: `mutual_reachability_weight` shared by batch and (soon) session;
  `rebuild_mst_labels` implemented and unit-tested; dev-dependencies
  adopted; session behaviour otherwise unchanged.
  Requirements: advances RM-11.2.1; discharges DES-12.4's reweight formula
  sharing; OBL-7 evidence lands here.
  Acceptance evidence: `cargo nextest run -p chutoro-core -E
  'test(refresh)'` green for the pure-fn suite; batch pipeline tests remain
  green after the DRY refactor.
  Conformance check: no public interface changed; no production dependency
  added; helper homed in `cpu_pipeline` (layering constraint); trace links
  current.
  Recovery: revert the milestone commit; nothing else depends on it yet.
  Remaining gaps: session cannot yet call the rebuild.

- **EP-M2 — session wiring.**
  Outcome: `refresh` and `labels` live; placeholder fields renamed;
  all-or-nothing publish; metrics registered; unit, error, and BDD suites
  green. This is the plateau at which RM-11.2.1's acceptance criteria are
  first observable.
  Requirements: discharges RM-11.2.1 acceptance criteria; DES-12.5 steps
  1–9 (with step 2 reading an empty buffer and step 8 storing the raw
  backbone); DES-12.6 atomic publish (OBL-5).
  Acceptance evidence: the six unit tests and two error tests named in the
  Plan of work; the two BDD scenarios pass via the existing harness.
  Conformance check: public surface grew by exactly `refresh` and `labels`;
  no new error variant; DES-FIG3 deviation implemented as approved; trace
  links updated to point at real test names.
  Recovery: revert the milestone commit; EP-M1's layer remains valid.
  Remaining gaps: property-level and formal evidence outstanding.

- **EP-M3 — property suites.**
  Outcome: OBL-3/OBL-4/OBL-6 property evidence plus the two recorded
  mutation checks.
  Requirements: DES-12.7 thresholds exercised incrementally.
  Acceptance evidence: `make test` green including the new property file;
  mutation-check transcripts pasted into Artefacts.
  Conformance check: proptest configuration uses the shared
  `suite_proptest_config` (CI reproducibility convention); no silent
  reduction of case counts.
  Recovery: revert the test-only commit; production code untouched.
  Remaining gaps: lemmas unproven.

- **EP-M4 — formal verification.**
  Outcome: OBL-1 and OBL-2 discharged in Verus; OBL-8 attempted with its
  go/no-go recorded.
  Requirements: verification-plan obligations; ADR-002's tiering (Kani out
  of `make test`).
  Acceptance evidence: `make verus` green with the new proof file listed in
  its run set; Decision Log entry if OBL-8 is dropped.
  Conformance check: no `assume` in shipped proofs; proof spec mirrors the
  production types faithfully (reviewed side by side).
  Recovery: proofs are additive; revert freely.
  Remaining gaps: documentation.

- **EP-M5 — documentation.**
  Outcome: design §12.4/§12.5/Figure 3 updated; users' and developers'
  guides updated; ADR-005 authored and referenced; roadmap ticked.
  Requirements: closes the DES-FIG3 and degenerate-behaviour deviations by
  landing their upstream edits; completes RM-11.2.1.
  Acceptance evidence: `make markdownlint` and `make nixie` green; roadmap
  shows `[x] 11.2.1`.
  Conformance check: every deviation recorded in Conformance Basis now has
  its upstream edit landed or an accepted Decision Log entry; no unrecorded
  drift remains (prerequisite for `COMPLETE`).
  Recovery: docs-only; revert freely.
  Remaining gaps: none for this roadmap item; `11.2.2`–`11.2.6` follow.

## Concrete steps

Run from the repository root. Prefer Makefile targets; capture long output
with `tee` to a `/tmp` log per `CLAUDE.md`.

```bash
# Focused red test (expect compile failure before implementation):
cargo nextest run -p chutoro-core --all-features -E 'test(refresh)' 2>&1 \
  | tee /tmp/test-chutoro-$(git branch --show-current).out

# Full deterministic gates (delegate to the scrutineer subagent):
make check-fmt
make lint
make test

# Verification (EP-M4 onward):
make verus
cargo kani --harness verify_rebuild_bounded   # only while evaluating OBL-8

# Snapshot review when the insta snapshot changes intentionally:
cargo insta review

# CodeRabbit, after deterministic gates are green:
coderabbit review --agent
```

Expected end state: `make test` passes with the new refresh unit, error,
property, and BDD tests green; the single-refresh differential reports ARI
`== 1.0` and the multi-refresh property stays ≥ 0.95; `make verus`
discharges OBL-1 and OBL-2.

## Validation and acceptance

Red-Green-Refactor evidence to record here as work proceeds:

- Red: `cargo nextest run -p chutoro-core -E 'test(refresh)'` fails before
  implementation (compile error: `refresh`/`labels` missing — the expected
  reason). The Verus proof file's lemmas stand open before EP-M4.
- Green: the same command passes after EP-M2; `make verus` passes after
  EP-M4.
- Refactor: `make check-fmt && make lint && make test` all pass after
  EP-M1's `cpu_pipeline` DRY refactor and after each subsequent milestone.

Acceptance (behaviour a human can verify), matching RM-11.2.1:

1. After `session.append(&[0..N])` then `session.refresh()`,
   `session.labels()` returns a vector of length `N` (`= point_count`).
2. `session.snapshot_version()` increases by exactly one per `refresh` call.
3. Incremental labels are partition-equivalent to `Chutoro::run` on the same
   contiguous dataset (ARI `== 1.0` on the deterministic single-refresh
   test).
4. `pending_edges` is empty after `refresh`; a failed `refresh` leaves
   `snapshot_version` and `pending_edges` unchanged.

Verification obligations: record, per obligation, the initial red state, the
passing command, bounds explored (OBL-8), axioms relied on (AXM-1..AXM-5), and
any obligation left undischarged. An implementation change that requires an
unplanned invariant, lemma, or axiom returns to the Verification Plan before
further elaboration. At each milestone boundary, record the conformance check
from Milestones and plateaus; if evidence requires an unapproved design
departure, record the proposed deviation in the Decision Log, set Status to
BLOCKED, and await acceptance.

Quality criteria (what "done" means):

- Tests: new unit + error + `proptest` + `rstest-bdd` suites pass under
  `make test`; batch `cpu_pipeline` tests remain green after the DRY
  refactor.
- Verification: OBL-1 and OBL-2 discharged by `make verus` with no `assume`;
  OBL-3..OBL-7 discharged by the test suites with their recorded
  non-vacuity/mutation evidence; OBL-8 discharged or explicitly dropped with
  a Decision Log entry.
- Lint/typecheck: `make check-fmt` and `make lint` clean (warnings denied).
- Review: `coderabbit review --agent` reports no unresolved concerns per
  milestone.

Quality method: the `scrutineer` subagent runs the gates sequentially and
returns a bounded report; failures are fixed forward within the iteration
tolerance.

## Idempotence and recovery

All edits are additive, field renames, or a single field-type change; steps
are re-runnable. If a milestone's gate fails, fix forward within the
iteration tolerance or revert the milestone's commit (work is committed per
milestone). The pure `rebuild_mst_labels` is deterministic, so reruns produce
identical snapshots. A failed `refresh` mutates no snapshot state; a retry
re-processes the remaining dirty core-distance cells and republishes.
Snapshot files (`.snap`) regenerate deterministically under AXM-4; delete and
re-accept via `cargo insta review` if corrupted.

## Artefacts and notes

To be populated during implementation with: the red-state transcript, the
green nextest summary, the two mutation-check failure transcripts (OBL-3
truncation, OBL-6 `min` substitution), the `make verus` output for OBL-1 and
OBL-2, and the OBL-8 outcome (runtime or drop decision).

## Interfaces and dependencies

At the end of this work the following must exist in `chutoro-core`:

```rust
// chutoro-core/src/cpu_pipeline.rs
pub(crate) fn mutual_reachability_weight(
    distance: f32, core_source: f32, core_target: f32,
) -> f32;
// map_cpu_mst_error and map_cpu_hierarchy_error promoted from private to
// pub(crate) so session/refresh.rs can reuse them:
pub(crate) fn map_cpu_mst_error(error: MstError) -> ChutoroError;
pub(crate) fn map_cpu_hierarchy_error(
    error: HierarchyError,
) -> ChutoroError;

// chutoro-core/src/session/refresh.rs
pub(super) struct RefreshOutcome {
    pub labels: Vec<usize>,
    pub mst_backbone: Vec<CandidateEdge>,
}
pub(super) fn rebuild_mst_labels(
    node_count: usize,
    mst_backbone: &[CandidateEdge],
    historical_edges: &[CandidateEdge],
    pending_edges: &[CandidateEdge],
    core_distances: &[f32],
    min_cluster_size: std::num::NonZeroUsize,
) -> crate::Result<RefreshOutcome>;

// chutoro-core/src/session/{mod.rs,session_impl.rs}
impl<D: DataSource + Send + Sync> ClusteringSession<D> {
    pub fn refresh(&mut self) -> crate::Result<()>;
    pub fn labels(&self) -> std::sync::Arc<Vec<usize>>;
}
```

Reused (unchanged) symbols: `crate::mst::parallel_kruskal_from_edges`
(`pub(crate)`, the refresh path's Kruskal entry point),
`crate::HierarchyConfig`, `crate::extract_labels_from_mst`, `crate::MstEdge`,
`crate::CandidateEdge`, `crate::adjusted_rand_index`. Internal change: the
session's `mst_edges` field becomes `mst_backbone: Vec<CandidateEdge>`. New
dev-dependencies (test-only): `googletest`, `pretty_assertions`, `insta`. No
new production dependency; no new public error variant.

## Revision note

Revised 2026-07-22 after the df12 Logisphere community-of-experts review.
Changes from the first draft: (1) adopt **raw-backbone retention** (store MST
edges as raw `CandidateEdge`, reweight from raw each refresh) to remove the
cross-refresh weight ratchet the first draft would have shipped — the field
`mst_edges` becomes `mst_backbone: Vec<CandidateEdge>`; (2) add **graceful
degenerate handling** (empty and all-noise snapshots) so small sessions do
not error and the flagship 4-point criterion holds; (3) home
`mutual_reachability_weight` in the lower `cpu_pipeline` layer to avoid a
dependency inversion; (4) reuse existing `ChutoroError` variants
(`CpuMstFailure` for contiguity violations) rather than add one;
(5) enumerate concrete `chutoro.session.refresh.*` metrics with `describe_*`
registration; (6) add a failed-refresh atomicity test, a multi-refresh
differential property, and reuse the existing BDD harness; (7) reframe the
Verus lemma around raw-retention reweight faithfulness; (8) record the
auto-recompute and per-call-counter semantics and update §12.4 wording.
Awaiting user approval before implementation.

Revised 2026-08-16 after a user gap review. Changes: (1) adopt `googletest`,
`pretty_assertions`, and `insta` as dev-dependencies with concrete usage
assignments in the new suites (first adoption in the workspace); (2) switch
the pure rebuild from `EdgeHarvest::from_unsorted` + `parallel_kruskal` to
the crate-internal `parallel_kruskal_from_edges` entry point, removing a
redundant sort and allocation (recorded as a deviation from the roadmap
item's literal wording); (3) document that dedup of the overlapping merged
buffers happens in `mst::prepare_edge_list` (not `EdgeHarvest`) and add an
overlap-tolerance unit test. Remaining work is unaffected; the plan still
awaits approval.

Redrafted 2026-08-23 to match the current execplans skill template. Changes:
(1) added the mandatory `Conformance Basis` section with stable upstream
identifiers (RM-11.2.1, DES-12.x, DES-FIG3), trace links from requirements
through milestones to named acceptance tests, and the three recorded
deviations (raw-backbone Figure 3 refinement, `parallel_kruskal_from_edges`
wording, degenerate-size behaviour); (2) restructured verification into the
mandatory `Verification Plan` with named obligations OBL-1..OBL-8, explicit
axioms AXM-1..AXM-5, per-obligation method/domain/artefact/evidence, and
non-vacuity arguments including two planned seeded-fault mutation checks;
(3) replaced the standalone red-tests milestone with red → green → refactor
stages *inside* each milestone so every milestone ends in a validated
plateau, renumbering to EP-M1..EP-M5 with per-milestone conformance checks,
recovery paths, and an explicit no-compatibility-machinery decision;
(4) added the `Artefacts and notes` section as the destination for mutation
and proof transcripts. No design or scope decisions changed; the plan still
awaits approval before implementation.
