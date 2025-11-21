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

## Next steps

- Add targeted unit tests where insertion trimming evicts the new node and the
  fallback path must evict a neighbour to restore reciprocity.
- Benchmark the localized commit path against the previous post-pass to confirm
  insertion latency improvements and to guard against degree-bound regressions.
- Expand deletion coverage with adversarial cases to validate the lightweight
  reconnection heuristic under high churn.
