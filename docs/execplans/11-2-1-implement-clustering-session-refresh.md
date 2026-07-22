# Implement `ClusteringSession::refresh`

This ExecPlan (execution plan) is a living document. The sections `Constraints`,
`Tolerances`, `Risks`, `Progress`, `Surprises & Discoveries`, `Decision Log`,
and `Outcomes & Retrospective` must be kept up to date as work proceeds.

Status: DRAFT

This plan delivers roadmap item `11.2.1` (see `docs/roadmap.md` §11.2). It must
be approved by the user before any implementation begins. It was revised after a
df12 Logisphere community-of-experts review; see the Decision Log and Surprises
sections for the findings that reshaped it.

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
mutual-reachability formula using current core distances, constructs a fresh
`EdgeHarvest`, runs `parallel_kruskal`, and extracts labels with
`extract_labels_from_mst`.

### What this plan does and does not authorize

This plan authorizes:

- `ClusteringSession::refresh`.
- A minimal non-blocking `ClusteringSession::labels` accessor sufficient to
  observe the published snapshot.
- One shared `mutual_reachability_weight` helper, homed in the batch pipeline
  layer (`cpu_pipeline.rs`) and called *down* into by the session.
- **Raw-distance retention of the MST backbone.** The session retains the MST
  edges it selected as *raw* `CandidateEdge` values (not weighted `MstEdge`), so
  every refresh reweights from raw distances and current core distances. This is
  a small, correctness-critical refinement of design Figure 3 (see Decision
  Log and ADR-005); without it, reweighting a previously weighted edge ratchets
  weights upward across refreshes.
- Promoting `map_cpu_mst_error` and `map_cpu_hierarchy_error` from private to
  `pub(crate)` so `refresh` can reuse them.

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
- Reuse the batch CPU primitives without forking their behaviour:
  `crate::parallel_kruskal`, `crate::extract_labels_from_mst`,
  `crate::HierarchyConfig`, `crate::EdgeHarvest`, `crate::MstEdge`, and
  `crate::CandidateEdge`. The incremental mutual-reachability weighting must be
  numerically identical to the batch path in
  `chutoro-core/src/cpu_pipeline.rs`, and must always be computed from **raw**
  distances so it does not depend on refresh history.
- Home the shared `mutual_reachability_weight` helper in the lower layer
  (`cpu_pipeline.rs`, `pub(crate)`); the session calls *down* to it. Do not have
  the batch pipeline import from the session module (the session already depends
  down on the pipeline via `map_cpu_hnsw_error`).
- Preserve existing public behaviour of `CpuHnsw::insert`,
  `CpuHnsw::insert_harvesting`, `CpuHnsw::search`, `ChutoroBuilder::build`,
  `ChutoroBuilder::build_session`, `ClusteringSession::append`,
  `ClusteringSession::recompute_core_distances`,
  `ClusteringSession::recompute_core_distances_full`, and `Chutoro::run`.
- Route every HNSW query through `CpuHnsw::search`. Do not reach into private
  HNSW adapter internals.
- Keep the session CPU-only behind the existing `cpu` feature gate. Add no GPU
  path.
- Add no new production dependency and no **new public** `ChutoroError` variant.
  Degenerate and invalid-input paths reuse the existing `EmptySource`,
  `InsufficientItems`, and `CpuMstFailure` variants (see Decision Log).
- `refresh` is all-or-nothing at the snapshot level: no observable session state
  (`labels`, `snapshot_version`, `pending_edges`, `mst_edges`) is mutated unless
  the whole refresh succeeds.
- Keep every touched source file below 400 lines. The pure rebuild plus helpers
  live in `session/refresh.rs`; the `refresh` inherent method and its metrics
  live in `session_impl.rs`. Split further if either would exceed the cap.
- Domain/policy logic (edge merge, mutual-reachability reweighting, Kruskal and
  label-extraction orchestration, raw-backbone recovery) must be a **pure
  function** with no HNSW, input/output, or clock dependency, so it is unit-,
  property-, and proof-testable in isolation. The impure adapter step
  (core-distance recompute via HNSW search) stays in the session methods. This
  is the hexagonal boundary this plan protects.

## Tolerances (exception triggers)

- Scope: if implementation requires touching more than 14 files or a net
  +1000 lines of code, stop and escalate.
- Interface: if a public API signature beyond `ClusteringSession::refresh`,
  `ClusteringSession::labels`, and one new `pub(crate)` helper must change, stop
  and escalate. Changing the internal `mst_edges` field type from
  `Vec<MstEdge>` to `Vec<CandidateEdge>` is authorized by this plan. Promoting
  `map_cpu_mst_error`/`map_cpu_hierarchy_error` to `pub(crate)` is authorized.
  Adding any **new public** `ChutoroError` variant is not — stop and escalate.
- Dependencies: if any new external dependency is required, stop and escalate.
- Iterations: if the deterministic commit gates still fail after 3 fix attempts
  on a single milestone, stop and escalate.
- Verification: if the Verus lemma cannot be discharged without an `assume`
  shortcut after 3 attempts, stop and record the blocker; do not ship an
  `assume`-weakened proof.
- Ambiguity: if the contiguous-point-id assumption (below) proves false for a
  required caller, stop and escalate rather than silently mislabelling output.

## Risks

- Risk: **Point-id contiguity.** HNSW node ids equal the source indices passed
  to `append` (`CpuHnsw::insert_with_edges(node, source)` uses `node`
  verbatim), and `core_distances`/`extract_labels_from_mst` are indexed by that
  id. `refresh` passes `node_count = point_count()`. A non-contiguous append
  (for example `&[0, 1, 5]`) makes `point_count()` 3 while an edge endpoint
  reaches id 5, so the reweight step would index `core_distances[5]` out of
  range.
  Severity: high. Likelihood: low (all shipped paths append contiguous
  prefixes; the batch pipeline has the same assumption).
  Mitigation: the pure rebuild validates every endpoint `< node_count` and
  `< core_distances.len()` and returns `ChutoroError::CpuMstFailure` with a
  descriptive message (no panic) — the same variant `parallel_kruskal` itself
  would produce for an invalid node id. Document the contiguity requirement on
  `append` and `refresh` and in ADR-005.

- Risk: **Cross-refresh weight drift (mitigated by raw retention).** If the MST
  backbone were retained as weighted `MstEdge` and reweighted again, the
  mutual-reachability `max` would ratchet weights upward whenever a core
  distance later fell (core distances are monotonically non-increasing after a
  point saturates its neighbourhood). This plan retains the backbone as **raw**
  `CandidateEdge`, so every refresh recomputes `max(raw, core_u, core_v)` from
  scratch and the result is independent of refresh count.
  Severity: was high; reduced to low by raw retention.
  Likelihood: low. Mitigation: raw-backbone retention plus a multi-refresh
  differential property test (append, refresh, append, refresh) asserting the
  incremental partition still matches batch within ARI ≥ 0.95.

- Risk: **Approximate-MST candidate coverage.** Even with raw retention, the
  merged candidate set (`mst_edges` + `historical_edges` + `pending_edges`) is a
  subset of all pairwise mutual-reachability edges. With `historical_edges`
  empty until `11.2.5`, a non-MST old-old edge that a core-distance shift would
  promote is not yet retained.
  Severity: medium. Likelihood: low for the single- and few-refresh tests here.
  Mitigation: retained-non-MST coverage is the explicit job of `11.2.5`; the
  full-refresh reset is `11.2.4`; the large differential harness is `11.4`.
  Record the boundary; do not claim exactness beyond what is tested.

- Risk: **Degenerate small sessions.** `extract_labels_from_mst` returns
  `HierarchyError::MinClusterSizeTooLarge` when `min_cluster_size > node_count`,
  and `parallel_kruskal` errors on `node_count == 0`.
  Severity: medium. Likelihood: high (tests and warm-up sessions hit it).
  Mitigation: `refresh` short-circuits before those calls — an empty session
  publishes an empty snapshot, and a session with `0 < point_count <
  min_cluster_size` publishes an all-noise snapshot (`vec![0; point_count]`).
  Both advance `snapshot_version`. This diverges intentionally from batch `run`
  (which errors with `InsufficientItems`) because a streaming session must make
  progress during warm-up; see Decision Log.

- Risk: **Label-length contract.** Verified: `extract_flat_labels` returns
  `vec![_; node_count]` (`chutoro-core/src/hierarchy/single_linkage/mod.rs`);
  noise adds a label *value*, not an entry. Severity: low. Mitigation: a direct
  `labels().len()` assertion.

## Progress

- [ ] M0: Red tests and BDD scenarios specifying `refresh`/`labels` behaviour
  (fail for the expected reason before implementation).
- [ ] M1: Pure domain layer — `mutual_reachability_weight` in `cpu_pipeline.rs`
  and `rebuild_mst_labels` (with raw-backbone recovery) in
  `chutoro-core/src/session/refresh.rs`; refactor the batch path to call the
  shared helper; pure-function unit tests.
- [ ] M2: Wire `ClusteringSession::refresh` and `ClusteringSession::labels`;
  change `mst_edges` to `Vec<CandidateEdge>` (raw) and rename the other
  placeholder fields; add refresh metrics with `describe_*` registration;
  publish the snapshot all-or-nothing.
- [ ] M3: `proptest` property suite (including a multi-refresh differential
  property) and `rstest-bdd` scenarios added to the existing session harness.
- [ ] M4: Verus lemma for raw-retention reweight faithfulness and merge
  invariants; optional bounded Kani harness for the pure rebuild (go/no-go).
- [ ] M5: Documentation — design §12.5, users' guide, developers' guide, and
  ADR-005; mark roadmap `11.2.1` done.

## Surprises & discoveries

- Observation: the `MstEdge -> CandidateEdge` round-trip is field-lossless but
  **semantically lossy**.
  Evidence: `MstEdge.weight()` is a mutual-reachability weight;
  `CandidateEdge::new(.., distance, ..)` expects a raw distance
  (`chutoro-core/src/mst/mod.rs`, `cpu_pipeline.rs`).
  Impact: retaining the backbone as weighted `MstEdge` and reweighting again
  ratchets weights upward across refreshes. This plan retains the backbone as
  raw `CandidateEdge` instead. The `mst_edges` field type therefore changes from
  `Vec<MstEdge>` to `Vec<CandidateEdge>`.

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
  to `Vec<CandidateEdge>`), reweighting from raw distances every refresh.
  Rationale: reweighting an already-weighted edge ratchets weights upward when
  core distances fall, drifting the partition across refreshes and contradicting
  the cited FISHDBC approach (which refeeds raw triples). Raw retention makes
  the reweight refresh-count-independent and numerically identical to batch. The
  cost is one extra recovery pass matching Kruskal output back to raw inputs by
  `(source, target, sequence)`. Refines design Figure 3.
  Date/Author: 2026-07-22, planning (post expert review).

- Decision: Graceful degenerate handling instead of mirroring batch `run`.
  Rationale: `run` errors on `items == 0` (`EmptySource`) and
  `items < min_cluster_size` (`InsufficientItems`). A streaming session must
  make progress during warm-up and the `11.2.1` acceptance criterion requires
  `labels().len() == point_count` for any size. So `refresh` publishes an empty
  snapshot when `point_count == 0` and an all-noise snapshot when
  `0 < point_count < min_cluster_size`, both advancing the version. This also
  makes the 4-point flagship example pass under the default `min_cluster_size`.
  Date/Author: 2026-07-22, planning (post expert review).

- Decision: `refresh` auto-recomputes core distances (calls
  `recompute_core_distances()`); it does not refuse while dirty.
  Rationale: design §12.3 phase 3 says refresh "recompute[s] core distances for
  new points". The predecessor plan `11-1-4` speculated `refresh` would "refuse
  while any dirty bit is set"; that assumption is overridden here in favour of
  the friendlier, design-aligned auto-recompute. §12.4 is updated to match. The
  core-distance step is best-effort-then-abort: on failure it returns before any
  snapshot mutation, and a retry re-processes the remaining dirty set.
  Date/Author: 2026-07-22, planning (post expert review).

- Decision: Reuse existing `ChutoroError` variants; add none.
  Rationale: `EmptySource`/`InsufficientItems` cover degenerate sizes (if a
  future decision reverts to erroring), and `CpuMstFailure` covers a contiguity
  violation (an invalid MST node id, exactly what `parallel_kruskal` reports).
  This keeps the interface tolerance intact.
  Date/Author: 2026-07-22, planning (post expert review).

- Decision: Expose a minimal `ClusteringSession::labels` accessor (returning
  `Arc<Vec<usize>>`) in this plan, ahead of roadmap `11.2.2`.
  Rationale: the `11.2.1` acceptance criterion is phrased in terms of
  `session.labels()`. A `&self` accessor returning `Arc::clone(&self.labels)` is
  inherently non-blocking; the `self.labels = Arc::new(...)` publish is the
  atomic `Arc` swap design §12.6 already promises for a single writer. Roadmap
  `11.2.2`
  formalises the multi-reader guarantees without changing this signature.
  Date/Author: 2026-07-22, planning.

- Decision: `refresh` computes `node_count = self.point_count()` and requires
  appended indices to form a contiguous prefix `[0, point_count)`.
  Rationale: HNSW node id equals the source index, and both `core_distances` and
  `extract_labels_from_mst` are indexed by that id; this matches the batch
  pipeline's assumption. The pure rebuild guards it and returns `CpuMstFailure`
  rather than panicking on an out-of-range endpoint.
  Date/Author: 2026-07-22, planning.

- Decision: `historical_edges` is read into the merge but not populated here.
  Rationale: population, the `2×` cap, and heaviest-first eviction are roadmap
  `11.2.5` (`requires 11.2.1`). Reading the (empty) buffer now keeps the merge
  signature stable.
  Date/Author: 2026-07-22, planning.

- Decision: `snapshot_version` is a per-refresh-call counter.
  Rationale: it advances once per `refresh` call, including empty and no-op
  refreshes. This is a call counter, not a content-change token; `11.2.2`
  formalises the reader contract. Stated so consumers do not treat every
  increment as a content change.
  Date/Author: 2026-07-22, planning (post expert review).

- Decision: Prior-art alignment with FISHDBC (Dell'Amico, 2019,
  arXiv:1910.07283).
  Rationale: FISHDBC maintains an *approximate* MST updated incrementally by
  feeding batches of raw candidate triples back through Kruskal, updating
  mutual-reachability distances as the `max` of endpoint core distances
  and the raw distance. This plan mirrors that exactly (merge raw candidates →
  reweight by `max` → Kruskal), confirming the design's §12.5 strategy.
  Date/Author: 2026-07-22, planning.

## Outcomes & retrospective

To be completed at delivery. Note: each refresh rebuilds and re-sorts the whole
combined candidate set (O(E log E)) even for a no-op refresh; the cut-based
incremental update sketched in design §12.5 is the future optimisation target
(out of scope here). Compare the shipped `refresh` against the three observable
success criteria in Purpose.

## Context and orientation

The reader needs no prior plans. Relevant facts, with full paths:

- The session lives in `chutoro-core/src/session/`:
  - `mod.rs` — the `ClusteringSession<D>` struct and read-only accessors
    (`config`, `point_count`, `snapshot_version`, `core_distance`). The struct
    declares placeholder fields `_mst_edges: Vec<MstEdge>`,
    `_historical_edges: Vec<CandidateEdge>`, `_labels: Arc<Vec<usize>>`, and
    `_last_refresh_len: usize`, awaiting this item.
  - `session_impl.rs` — construction and `append` (calls
    `CpuHnsw::insert_harvesting`, extends `pending_edges`, marks core-distance
    dirty). It already reuses `crate::cpu_pipeline::map_cpu_hnsw_error` via
    `map_hnsw_error`.
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
  reweighting into `EdgeHarvest::new`, `parallel_kruskal(items, &harvest)`, then
  `extract_labels_from_mst(items, forest.edges(), HierarchyConfig::new(mcs))`.
  It also defines the crate-internal error mapper `map_cpu_hnsw_error`
  (`pub(crate)`) plus `map_cpu_mst_error` and `map_cpu_hierarchy_error`, which
  are currently **private** and must be promoted to `pub(crate)` for `refresh`
  to reuse them (see the Decision Log).
- Primitive signatures (verified), each defined in the noted module:

  ```text
  // chutoro-core/src/mst/mod.rs
  parallel_kruskal(node_count: usize, edges: &EdgeHarvest)
      -> Result<MinimumSpanningForest, MstError>
  MinimumSpanningForest::edges() -> &[MstEdge]

  // chutoro-core/src/hierarchy/mod.rs  (returns a node_count-length vector)
  extract_labels_from_mst(node_count: usize, edges: &[MstEdge],
      config: HierarchyConfig) -> Result<Vec<usize>, HierarchyError>

  // chutoro-core/src/hnsw/types.rs  (from_unsorted aliases new; sorts + orders)
  EdgeHarvest::from_unsorted(Vec<CandidateEdge>) -> EdgeHarvest
  CandidateEdge::new(source, target, distance, sequence)
  CandidateEdge { source(), target(), distance(), sequence() }
  MstEdge { source(), target(), weight(), sequence() }  // no raw distance kept
  ```

- Public re-exports for all of the above are in `chutoro-core/src/lib.rs` behind
  `#[cfg(feature = "cpu")]`. `adjusted_rand_index(&[usize], &[usize])` is public
  in `crate::clustering_quality`.
- Definitions of terms:
  - **Core distance** of point `p` at `m = min_cluster_size`: the distance from
    `p` to its `m`-th nearest neighbour (self excluded).
  - **Mutual-reachability weight** of a raw edge `(u, v)` with raw distance `d`:
    `max(d, core[u], core[v])`.
  - **MST backbone**: the raw `CandidateEdge` values whose reweighted forms were
    selected into the MST by the previous refresh; retained for the next merge.
  - **Snapshot version**: a monotonically increasing `u64` bumped once per
    `refresh` call.

### Documentation and skills to consult

- Design: `docs/chutoro-design.md` §12.3 (session architecture), §12.4 (edge
  harvesting and core distances), §12.5 (incremental MST refresh strategy — the
  primary spec), §12.6 (concurrency model), §12.7 (differential testing).
- Roadmap: `docs/roadmap.md` §11.2 (this item and siblings `11.2.2`–`11.2.6`,
  and `11.4`).
- Predecessor execplans: `docs/execplans/11-1-3-clustering-session-append.md`
  and `docs/execplans/11-1-4-incremental-core-distance-computation.md`.
- Testing docs: `docs/property-testing-design.md`,
  `docs/rust-testing-with-rstest-fixtures.md`, `docs/rust-doctest-dry-guide.md`,
  `docs/complexity-antipatterns-and-refactoring-strategies.md`.
- Verification: `docs/verus-toolchain.md`,
  `docs/adr-002-adoption-of-kani-formal-verification.md`.
- Style: `docs/documentation-style-guide.md`.
- Skills: `execplans` (this document), `rust-router` →
  `rust-unit-testing` / `rust-errors` / `arch-crate-design`, `proptest`,
  `verus`, `kani`, `hexagonal-architecture`, `leta`, `nextest`,
  `commit-message`, `pr-creation`.

## Plan of work

Stages map to the milestones in Progress. Each stage ends with the deterministic
commit gates (`make check-fmt`, `make lint`, `make test`) and, once those pass,
a `coderabbit review --agent` pass whose concerns are cleared before moving on.
Delegate full gate runs to the `scrutineer` subagent.

### Stage A — understand and propose (no code changes)

Completed during planning and confirmed by expert review. Recorded above. No
edits.

### Stage B — Red (M0)

Add the smallest failing specifications before any production change.

1. Unit tests in a new `chutoro-core/src/session/tests/refresh.rs` (registered
   from `session/tests.rs`), using the `session_builder` fixture and
   `make_session` helper. Every test that expects real clusters builds with
   `with_min_cluster_size(2)` (or smaller) and enough points:
   - `refresh_publishes_labels_of_point_count_length` — build with
     `with_min_cluster_size(2)`; append `&[0,1,2,3]`; `refresh()`; assert
     `labels().len() == 4`.
   - `refresh_advances_snapshot_version_by_one` — assert the version delta is
     exactly one per call across two refreshes.
   - `refresh_clears_pending_edges` — assert `pending_edges` empty after refresh
     (child-module private-field access, as `append.rs` uses).
   - `refresh_on_empty_session_publishes_empty_snapshot` — no appends;
     `refresh()`; assert `labels().is_empty()` and version advanced by one.
   - `refresh_below_min_cluster_size_publishes_all_noise` —
     `with_min_cluster_size(5)`, append `&[0,1,2,3]`, `refresh()`; assert
     `labels() == [0,0,0,0]` and version advanced by one.
   - `refresh_matches_batch_partition` — differential: build the same contiguous
     dataset with `min_cluster_size` ≤ point count, run `Chutoro::run`, map its
     `assignments()` (`&[ClusterId]`) to `Vec<usize>` via `ClusterId::get() as
     usize`, and assert `adjusted_rand_index(&incremental, &batch) == 1.0` for a
     single refresh.
2. Error-path tests in `chutoro-core/src/session/tests/refresh_errors.rs`
   (registered from `session/tests.rs`, matching the existing
   `core_distance_errors` split):
   - `failed_refresh_preserves_snapshot_and_pending` — drive a refresh failure
     (a contiguity violation, or an injected out-of-range endpoint via the pure
     rebuild) and assert `snapshot_version` and `pending_edges` are unchanged,
     mirroring `append_failure_preserves_prior_successes`.
   - `contiguity_violation_reports_cpu_mst_failure` — assert the error is
     `ChutoroError::CpuMstFailure` with the documented error code.
3. Behaviour-driven development (BDD): **reuse the existing session harness**
   (`chutoro-core/tests/features/session_append.feature` and
   `chutoro-core/tests/session_append_bdd.rs`), as `11-1-4` did, rather than a
   new `World`. Add scenarios and the `When I refresh the session` /
   `Then the label snapshot has length {count:usize}` /
   `Then the snapshot version is {version:u64}` steps to the existing files.

Run the focused tests and confirm they fail because `refresh`/`labels` do not
yet exist (a compile error is the expected red state; convert to assertion
failures as the API lands).

Feature scenarios to add to `session_append.feature`:

```gherkin
  Scenario: Refreshing a populated session publishes labels for every point
    Given a clustering session over 4 points with min cluster size 2
    When I append points "0,1,2,3"
    And I refresh the session
    Then the label snapshot has length 4
    And the snapshot version is 1

  Scenario: Refreshing an empty session advances the version with empty labels
    When I refresh the session
    Then the label snapshot is empty
    And the snapshot version is 1
```

### Stage C — Green

M1 (pure domain layer):

1. In `chutoro-core/src/cpu_pipeline.rs`, add a `pub(crate)`
   `mutual_reachability_weight(distance, core_source, core_target) -> f32` that
   returns `distance.max(core_source).max(core_target)`, and refactor the batch
   path's inline `.map()` to call it. Behaviour is
   unchanged; batch tests stay green. Promote `map_cpu_mst_error` and
   `map_cpu_hierarchy_error` to `pub(crate)`.
2. Add `chutoro-core/src/session/refresh.rs` with the pure rebuild:

   ```rust
   pub(super) struct RefreshOutcome {
       pub labels: Vec<usize>,
       pub mst_backbone: Vec<CandidateEdge>, // raw edges selected into the MST
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
     (`vec![0; node_count]`) and an empty backbone (a warm-up session keeps no
     structure yet).
   - Builds the combined raw `Vec<CandidateEdge>` = `mst_backbone` ++
     `historical_edges` ++ `pending_edges` (all raw).
   - Validates every endpoint `< node_count` and `< core_distances.len()`,
     returning `ChutoroError::CpuMstFailure` on violation (no panic).
   - Reweights each edge with `mutual_reachability_weight`, preserving
     `(source, target, sequence)`.
   - `EdgeHarvest::from_unsorted(reweighted)`, then
     `parallel_kruskal(node_count, &harvest)` mapped via `map_cpu_mst_error`,
     then `extract_labels_from_mst(node_count, forest.edges(),
     HierarchyConfig::new(min_cluster_size))` mapped via
     `map_cpu_hierarchy_error`.
   - Recovers the raw backbone: build a set of `(source, target, sequence)` keys
     from `forest.edges()` (canonicalised), then filter the raw combined set to
     those keys. Returns labels plus that raw backbone.
3. Add pure-function unit tests in `refresh.rs` `#[cfg(test)]` for
   `mutual_reachability_weight` (commutativity in the two cores, `>= max` of the
   three inputs, idempotence when all equal) and `rebuild_mst_labels` (label
   length, empty and all-noise short-circuits, out-of-range endpoint error, and
   raw-backbone recovery correctness).

M2 (session wiring):

1. In `session/mod.rs`, change `_mst_edges: Vec<MstEdge>` to
   `mst_backbone: Vec<CandidateEdge>` and rename `_historical_edges ->
   historical_edges`, `_labels -> labels`, `_last_refresh_len ->
   last_refresh_len`. Update `session_impl.rs` construction accordingly and drop
   the now-unused `MstEdge` import if nothing else needs it.
2. Add `pub fn labels(&self) -> Arc<Vec<usize>>` returning
   `Arc::clone(&self.labels)`.
3. Register the refresh metrics in the construction `describe_*` block and
   implement `pub fn refresh(&mut self) -> Result<()>` in `session_impl.rs`:
   - Call `self.recompute_core_distances()?` (incremental adapter step).
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

   - Publish only after every fallible call succeeds, as one commit block, so
     `refresh` is all-or-nothing (on any error, no state is mutated):

   ```rust
   // commit block — all-or-nothing snapshot publish
   self.labels = Arc::new(outcome.labels);
   self.mst_backbone = outcome.mst_backbone;
   self.pending_edges.clear();
   self.last_refresh_len = node_count;
   self.snapshot_version += 1;
   ```

   - Record metrics under the existing `#[cfg(feature = "metrics")]` pattern:
     `chutoro.session.refresh.seconds` (duration histogram),
     `chutoro.session.refresh.errors_total` (counter labelled by reason),
     `chutoro.session.refresh.candidate_edges` (merged edge-count histogram),
     and `chutoro.session.refresh.cluster_count` (distinct-label histogram),
     each `describe_*`-registered in construction.
   - Document errors: `ChutoroError::DataSource`/`CpuHnswFailure` from the
     core-distance recompute, and `CpuMstFailure`/`CpuHierarchyFailure` from the
     pure rebuild.

Run the focused unit, error, and BDD tests; they now pass.

### Stage D — Refactor, verify, document

M3 (property + BDD): add `proptest` properties (a new
`session/tests/refresh_properties.rs`, using `suite_proptest_config`):

- `snapshot_version` increases by exactly one per `refresh`, monotone across
  random append/refresh interleavings.
- `labels().len() == point_count()` after every `refresh`.
- `pending_edges` empty after every `refresh`.
- Determinism: identical append/refresh sequences under the fixed proptest seed
  produce identical label vectors.
- Multi-refresh differential: append, refresh, append, refresh over a contiguous
  dataset of at least `min_cluster_size` points; assert the incremental labels
  match a batch `Chutoro::run` on the final dataset within ARI ≥ 0.95. This is
  the property that guards against the reweight ratchet the raw-backbone
  retention fixes.

M4 (formal verification):

- **Verus (required, substantive).** In `verus/session_refresh.rs`, mirror the
  spec types (as the repo does for edge-harvest proofs under `1.6.4`) and prove:
  1. Raw-retention reweight faithfulness: reweighting a raw edge is exactly
     `max(distance, core_source, core_target)` and depends only on the raw
     distance and the two current core distances — never on any previously
     stored weight. Therefore two sessions that reach the same
     `(raw edge set, core distances)` produce identical weighted edges
     regardless of refresh history. This is the invariant the double-weighting
     bug violated, so it is substantive rather than a restatement.
  2. Merge multiset preservation: the combined candidate length equals the sum
     of the three input lengths, and reweighting is a bijection that preserves
     `source`, `target`, and `sequence` (changing only the weight).
  No `assume` shortcuts. Register the proof so `make verus` runs it.
- **Kani (optional, go/no-go).** A bounded `#[cfg(kani)]` harness over
  `rebuild_mst_labels` with a fixed 2–3 node combined edge set proving output
  `labels.len() == node_count` and that an out-of-range endpoint yields `Err`
  (not a panic), with a tight `#[kani::unwind]`. Because heap-`Vec` harnesses
  scale poorly and MST structural correctness is already Kani-verified under
  `1.5.1`, keep node count ≤ 3; if compile/solve exceeds a few minutes, record
  the cost and drop the harness (Verus plus proptest remain the verification of
  record). Do not add Kani to `make test`.

M5 (documentation):

- `docs/chutoro-design.md` §12.5: mark `11.2.1` implemented, but word it so the
  reader knows `historical_edges` **retention/eviction is still deferred to
  `11.2.5`** (only the empty read path exists). Record the raw-backbone
  retention (Figure 3 refinement: `mst_edges` retained as raw `CandidateEdge`),
  the `node_count = point_count` contiguity invariant, the degenerate all-noise
  behaviour (and its intentional divergence from batch `run`), the
  auto-recompute behaviour (update §12.4's dirty-set note), and the early
  minimal `labels()`.
- `docs/users-guide.md`: document the `append -> refresh -> labels` workflow and
  the two new public methods, with a runnable doctest that sets
  `with_min_cluster_size` explicitly. State the warm-up (all-noise) behaviour.
- `docs/developers-guide.md`: document the hexagonal split (pure
  `rebuild_mst_labels` vs the HNSW-backed core-distance adapter), the shared
  lower-layer `mutual_reachability_weight` helper, the raw-backbone retention
  rationale, and the contiguity invariant with its `CpuMstFailure` guard.
- `docs/adr-005-incremental-refresh-domain-boundary.md`: a Y-Statement ADR
  capturing (a) the pure-domain refresh boundary, (b) raw-backbone retention and
  why weighted retention is unsound across refreshes, (c) the contiguous
  point-id invariant and its `CpuMstFailure` guard, (d) graceful degenerate
  handling diverging from batch `run`, and (e) exposing `labels()` ahead of
  `11.2.2`. Reference it from the design doc.
- `docs/roadmap.md`: tick `11.2.1` to `[x]` once all gates pass.

## Concrete steps

Run from the repository root. Prefer Makefile targets; capture long output with
`tee` to a `/tmp` log per `CLAUDE.md`.

```bash
# Focused red test (expect failure / compile error before implementation):
cargo nextest run -p chutoro-core --all-features -E 'test(refresh)' 2>&1 \
  | tee /tmp/test-chutoro-$(git branch --show-current).out

# Full deterministic gates (delegate to the scrutineer subagent):
make check-fmt
make lint
make test

# Verification:
make verus
make kani   # only if the optional Kani harness is kept

# CodeRabbit, after deterministic gates are green:
coderabbit review --agent
```

Expected end state: `make test` passes with the new `refresh` unit, error,
property, and BDD tests green; the single-refresh differential reports ARI
`== 1.0` and the multi-refresh property stays ≥ 0.95; `make verus` discharges
the new lemma.

## Validation and acceptance

Red-Green-Refactor evidence to record here as work proceeds:

- Red: `cargo nextest run -p chutoro-core -E 'test(refresh)'` fails before
  implementation (missing `refresh`/`labels`).
- Green: the same command passes after M2.
- Refactor: `make check-fmt && make lint && make test` all pass after M1's
  `cpu_pipeline` DRY refactor and after each subsequent milestone.

Acceptance (behaviour a human can verify), matching the roadmap criteria:

1. After `session.append(&[0..N])` then `session.refresh()`, `session.labels()`
   returns a vector of length `N` (`= point_count`).
2. `session.snapshot_version()` increases by exactly one per `refresh` call.
3. Incremental labels are partition-equivalent to `Chutoro::run` on the same
   contiguous dataset (ARI `== 1.0` on the deterministic single-refresh test).
4. `pending_edges` is empty after `refresh`; a failed `refresh` leaves
   `snapshot_version` and `pending_edges` unchanged.

Quality criteria for "done":

- Tests: new unit + error + `proptest` + `rstest-bdd` suites pass under
  `make test`; the batch `cpu_pipeline` tests remain green after the DRY
  refactor.
- Lint/typecheck: `make check-fmt` and `make lint` clean (warnings denied).
- Verification: `make verus` passes the new lemma with no `assume`.
- Review: `coderabbit review --agent` reports no unresolved concerns per
  milestone.

## Idempotence and recovery

All edits are additive, field renames, or a single field-type change; steps are
re-runnable. If a milestone's gate fails, fix forward within the iteration
tolerance or revert the milestone's commit (work is committed per milestone).
The pure `rebuild_mst_labels` is deterministic, so reruns produce identical
snapshots. A failed `refresh` mutates no snapshot state; a retry re-processes
the remaining dirty core-distance cells and republishes.

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

Reused (unchanged) symbols: `crate::parallel_kruskal`, `crate::HierarchyConfig`,
`crate::extract_labels_from_mst`, `crate::EdgeHarvest`, `crate::MstEdge`,
`crate::CandidateEdge`, `crate::adjusted_rand_index`. Internal
change: the session's `mst_edges` field becomes `mst_backbone:
Vec<CandidateEdge>`. No new external dependency; no new public error variant.

## Revision note

Revised 2026-07-22 after the df12 Logisphere community-of-experts review.
Changes from the first draft: (1) adopt **raw-backbone retention** (store MST
edges as raw `CandidateEdge`, reweight from raw each refresh) to remove the
cross-refresh weight ratchet the first draft would have shipped — the field
`mst_edges` becomes `mst_backbone: Vec<CandidateEdge>`; (2) add **graceful
degenerate handling** (empty and all-noise snapshots) so small sessions do not
error and the flagship 4-point criterion holds; (3) home
`mutual_reachability_weight` in the lower `cpu_pipeline` layer to avoid a
dependency inversion; (4) reuse existing `ChutoroError` variants
(`CpuMstFailure` for contiguity violations) rather than add one; (5) enumerate
concrete `chutoro.session.refresh.*` metrics with `describe_*` registration;
(6) add a failed-refresh atomicity test, a multi-refresh differential property,
and reuse the existing BDD harness; (7) reframe the Verus lemma around
raw-retention reweight faithfulness; (8) record the auto-recompute and
per-call-counter semantics and update §12.4 wording. Awaiting user approval
before implementation.
