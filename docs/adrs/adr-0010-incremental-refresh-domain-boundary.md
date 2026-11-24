# Architecture Decision Record (ADR) 010: incremental refresh domain boundary

## Status

Accepted - 2026-11-23: `ClusteringSession::refresh` runs a pure
merge-reweight-extract pipeline over a raw retained MST backbone, with the
HNSW-backed core-distance recompute kept outside it.

## Date

2026-11-23

## Context and problem statement

Roadmap item 11.2.1 gives `ClusteringSession` its first clustering output: a
published flat-label snapshot produced by an incremental minimum spanning tree
(MST) refresh (design §12.5). The session already appends points through the
HNSW edge-harvesting path, buffers candidate edges in `pending_edges`, and
maintains per-point core distances.

The design's Figure 3 sketch retained the MST as `Vec<MstEdge>`. `MstEdge`
stores a **mutual-reachability weight**, not a raw distance. A refresh that
reweighted those stored edges with the formula
`max(distance, core[source], core[target])` would feed a previous weight back
into the maximum. Whenever a core distance later falls — which closer arriving
points cause, and which approximate HNSW recomputation permits in either
direction (#395) — the stale, higher weight would survive and the partition
would drift by an amount that depends on refresh history rather than on data.

The question this decision answers is how to structure the first refresh so it
is correct across repeated refreshes, remains numerically aligned with the
batch pipeline, and stays testable and provable without an HNSW index.

## Decision drivers

- Repeated refreshes must not drift by construction.
- Incremental and batch results must stay aligned; the differential tests
  compare them on identical datasets (design §12.7).
- The domain logic must be testable and formally provable without HNSW, I/O,
  or a clock.
- No new public `ChutoroError` variant, and no new production dependency.
- Degenerate warm-up sessions must make progress rather than erroring.
- The batch pipeline's own primitives must be reused, not forked.

## Options considered

- **Retain weighted `MstEdge` values and reweight them.** Rejected: the maximum
  ratchets upward when a core distance falls, so results depend on refresh
  count and diverge from batch.
- **Retain raw `CandidateEdge` values and reweight from raw.** Accepted: the
  formula becomes a pure function of the raw distance and the current core
  distances, so a given edge's weight is independent of how many times it was
  reweighted. The candidate set a refresh sees can still depend on earlier
  refreshes, and batch equivalence is bounded by the differential suite rather
  than guaranteed.
- **Mirror batch `run`'s error behaviour for empty and undersized sessions.**
  Rejected: batch `run` returns `EmptySource` or `InsufficientItems`, but a
  streaming session must publish something during warm-up.
- **Add a new `ChutoroError` variant for a non-contiguous append.** Rejected:
  the highest appended id is already a node id at or above `node_count`, which
  is exactly `MstError::InvalidNodeId`, already classified as caller input.
- **Put the refresh orchestration in the impure session method.** Rejected: it
  would need an HNSW index to test, and the Verus lemmas would have nothing
  small to target.
- **Copy the mutual-reachability formula into the session.** Rejected: two
  copies would drift. The batch helper is generalized to a slice and reused.

## Decision outcome

The refresh is split at a hexagonal seam:

- The **pure function** `rebuild_mst_labels` in `session/refresh.rs` merges the
  three raw buffers, reweights them with the batch pipeline's own
  `mutual_reachability_edges` helper, calls the crate-internal
  `parallel_kruskal_from_edges` entry point, extracts labels with
  `extract_labels_from_mst`, and recovers the raw inputs of the accepted forest
  edges as the new backbone.
- The **impure adapter** in `session/refresh_impl.rs` recomputes dirty core
  distances through `CpuHnsw::search`, times the call, and performs one
  all-or-nothing publish.

The session retains the backbone as raw `CandidateEdge` values and matches
accepted edges back to raw inputs by canonical `(source, target, sequence)`
key. Labels, version, pending edges, and backbone are committed together only
after every fallible call succeeds.

Degenerate sessions publish an empty snapshot when no points are appended and
an all-noise snapshot when the point count is below `min_cluster_size`; both
advance `snapshot_version`. A warm-up refresh selects no tree but retains the
merged raw candidates as the backbone, because HNSW never re-harvests an
old-old edge and discarding the pool would lose those connections. A
non-contiguous append sequence is reported as `CpuMstFailure` with
`INVALID_NODE_ID` before any reweight work.

The minimal `labels()` accessor lands here, ahead of roadmap item 11.2.2,
because the 11.2.1 acceptance criteria are phrased in terms of it. It returns
`Arc<Vec<usize>>` and never blocks a writer; 11.2.2 formalizes the multi-reader
contract without changing the signature.

### Y-Statement

In the context of incremental clustering sessions that republish labels after
each append batch, facing the risk that reweighting an already weighted MST
edge ratchets weights upward across refreshes, we decided for a pure
merge-reweight-extract function over a raw retained MST backbone, and against
weighted-edge retention, duplicated weighting logic, and batch-style errors for
degenerate warm-up sessions, to achieve weights that depend only on the raw
distance and the current core distances rather than on refresh count — with
batch agreement as evidence on the differential suite's fixture domain, not a
guarantee — accepting one extra recovery pass that maps Kruskal output back to
raw inputs by `(source, target, sequence)`.

## Consequences

- The private session field changes from `mst_edges: Vec<MstEdge>` to
  `mst_backbone: Vec<CandidateEdge>`. Design Figure 3 and §12.5 record it.
- `historical_edges` is read into the merge but not yet populated; the cap and
  heaviest-first eviction remain roadmap item 11.2.5.
- `refresh` auto-recomputes dirty core distances rather than refusing; the
  dirty set drains during refresh, and a failure leaves the remaining cells
  dirty for a retry.
- Duplicated raw edges across buffers need no pre-merge filtering: Kruskal's
  `prepare_edge_list` removes only **adjacent** identical
  `(weight, source, target)` triples after sorting, and backbone recovery
  matches each accepted key at most once. Duplicates that sort apart can both
  survive, which changes nothing observable because they reweight identically.
- The public surface grows by exactly `refresh()` and `labels()`; no error
  variant and no production dependency is added.
- Each refresh rebuilds and re-sorts the whole combined candidate set
  (O(E log E)) even when nothing changed. The cut-based update sketched in
  design §12.5 remains a future optimization.

## Validation evidence

- Unit tests in `chutoro-core/src/session/refresh/tests.rs` cover the pure
  rebuild: empty and all-noise short-circuits, the non-contiguous error, raw
  backbone recovery, and overlap tolerance across the three buffers.
- Session tests in `chutoro-core/src/session/tests/refresh.rs` cover the
  acceptance criteria and a differential single-refresh ARI comparison against
  batch `run`. `refresh_errors.rs` covers both failure partitions and their
  all-or-nothing state preservation.
- Property tests in `refresh_properties.rs` cover version discipline, label
  length, pending-edge clearing, determinism, and a multi-round ARI bound.
- `verus/session_refresh.rs` discharges reweight faithfulness and merge
  multiset preservation; `make verus` runs it.
- A BDD scenario exercises `append -> refresh -> labels` through the public
  surface.

## References

- [Design document §12.3-§12.7](../chutoro-design.md)
- [Roadmap item 11.2.1](../roadmap.md)
- [ExecPlan 11.2.1](../execplans/11-2-1-implement-clustering-session-refresh.md)
- [FISHDBC, Dell'Amico 2019, arXiv:1910.07283](https://arxiv.org/abs/1910.07283)
