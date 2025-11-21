# Architecture Decision Record (ADR) 001: Post-insertion bidirectional enforcement

## Status

Accepted

## Context

- Property test `hnsw_mutations_preserve_invariants_proptest` still fails with
  missing reverse links after staged insertions are committed.
- `commit` currently writes trimmed neighbour lists but does not guarantee that
  every retained edge has a back-link, especially when trimming evicts the new
  node from existing lists.
- Reconfigurations in tests were leaving the graph params unchanged; a small
  fix now propagates updates via `Graph::set_params`.
- Initial population calculation was capped to half the fixture size to avoid
  overshooting insert capacity.

## Decision

- Localize bidirectional enforcement inside `commit`: derive reciprocal links
  per layer from the trimmed updates, filter the new node's neighbours to those
  reciprocated candidates, and, when reciprocity disappears, add a reverse edge
  to the earliest planned neighbour (evicting the weakest entry if needed)
  instead of scanning the whole graph.
- Guarantee each layer retains at least one neighbour with a confirmed
  back-link while respecting degree bounds by using the level-specific
  connection limit (doubling at layer 0 when space is available).
- Enable test-only deletions via `Graph::delete_node` and
  `CpuHnsw::delete_node_for_test`, scrubbing references, reconnecting former
  neighbours, recomputing the entry point, and decrementing the public length
  counter so mutation properties exercise real delete semantics.

## Consequences

- Property tests exercise add, delete, and reconfigure sequences with
  bidirectional links enforced only for the touched nodes, avoiding a
  full-graph sweep while keeping invariants intact.
- Delete operations now mutate the graph, so reachability and bidirectional
  checks cover removal paths as well as insertion.
- The graph parameters stay in sync during test reconfigures, reducing false
  positives from parameter drift.
- Initial population seeding remains bounded to half the fixture size,
  preventing early overfill edge cases.

## Findings to date

- `hnsw_mutations_preserve_invariants_proptest` still fails intermittently. The
  most recent run aborted with a stack overflow after reporting
  `proptest: FileFailurePersistence::SourceParallel set, but no source file
  known`, suggesting uncontrolled recursion while attempting to heal the graph
  after staged insertions.
- Another failing seed flagged a missing reverse edge after the bootstrap
  insert stage (`edge exists 13 -> 0 at level 0 but no reverse edge`), which
  indicates the local reciprocity fallback may still drop back-links when
  trim-induced evictions and base-layer degree limits collide.
- Test-only reachability healing and bidirectional enforcement now run inside
  `commit`, but they are not yet sufficient to stabilise the mutation property
  across seeds. Additional guardrails or instrumentation are needed.
- After tightening reachability healing to avoid entry-centric eviction churn,
  a failing seed exposed a missing reverse link at layer 1 (`edge 2 -> 4`)
  caused by evicting an existing neighbour while adding a reverse edge. We now
  scrub the evicted node's forward edge during reverse-link insertion.
- A later seed showed a bootstrap failure with `edge 11 -> 8` missing a reverse
  link at layer 0. Hypothesis: one-way edges can survive when a reverse-link
  insertion evicts a neighbour but the new node keeps the forward edge.
  Mitigation: after every commit we now run a reciprocity pass over all touched
  nodes (`ensure_reciprocity_for_touched`) that either adds the missing
  back-link (evicting and scrubbing as needed) or removes the forward edge. New
  unit tests cover both reverse-edge eviction scrubbing and healing an existing
  one-way edge.

- Intermittent failure (`edge 4 -> 0` missing backlink at layer 0) was
  reproduced on a small clustered fixture when the forward edge belonged to a
  prior insertion, not the new node. The touched-node reciprocity pass addresses
  this broader case; repeated property runs now pass locally.
- Another intermittent bootstrap failure reported node 11 unreachable. Hypothesis:
  base-layer healing refused to link unreachable nodes when all reachable nodes
  were at capacity, leaving isolated vertices. Healing now tries capacity-first
  and then forces a link via any reachable node (allowing eviction) to restore
  reachability.
- The current `commit` path invokes `enforce_bidirectional` without a test-only
  guard; it walks every edge (`collect_all_edges`/`ensure_reverse_edge`) after
  each insertion. This introduces an `O(E)` production cost and can rewrite
  unrelated adjacency lists, regressing from the prior local update strategy as
  the graph grows.

## Next steps

- Add targeted unit tests where insertion trimming evicts the new node and the
  fallback path must evict a neighbour to restore reciprocity.
- Benchmark the localized commit path against the previous post-pass to confirm
  insertion latency improvements and to guard against degree-bound regressions.
- Expand deletion coverage with adversarial cases to validate the lightweight
  reconnection heuristic under high churn.
