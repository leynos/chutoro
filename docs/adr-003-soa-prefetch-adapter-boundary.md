# Architecture Decision Record (ADR) 003: SoA and prefetch adapter boundary

## Status

Accepted. Keep SoA and prefetch policy private to dense adapters, and defer
core HNSW structural changes until separate benchmark evidence justifies them.

## Date

2026-06-25

## Context and problem statement

Roadmap item `2.3.1` asked for HNSW neighbour evaluation to use packed indices,
Structure of Arrays (SoA) coordinates, prefetching, and scoring outside the
graph write lock. Implementation review showed that prior `2.2.x` work already
provides the structural boundary needed for most of that request:

- HNSW candidate scoring calls `DataSource::batch_distances(query, candidates)`.
- The dense provider keeps `DensePointView<'a>` and SIMD dispatch private to
  the dense adapter.
- HNSW insert planning and trim scoring compute distances before acquiring the
  graph write lock; the C1 guard now tests that invariant.

Milestone 0 measurements for this execplan also showed that accumulated batch
scoring accounted for 8.30% of the 10k-point synthetic build and 3.61% of the
100k-point synthetic build. That evidence does not justify widening core HNSW
interfaces only to expose dense-provider layout details.

## Decision drivers

- Preserve the `DataSource` adapter boundary so future GPU and alternate
  providers can choose their own layouts.
- Avoid duplicating dense vectors into a second dimension-major store unless a
  measured workload proves the memory cost pays for itself.
- Keep HNSW graph policy independent of dense SIMD packing and prefetch
  policy.
- Require each structural optimization to clear its own benchmark gate before
  becoming roadmap work.

## Y-Statement

In the context of HNSW neighbour scoring for dense vectors, facing pressure to
move packed-index, SoA, prefetch, and scratch-buffer details into core HNSW, we
decided for a private dense-adapter implementation boundary, and against
cross-node beam scoring in core, a secondary dimension-major SoA copy, and a
public `batch_distances_into` buffer-reuse API, to achieve adapter independence
and evidence-led optimization, accepting that each future structural win needs a
separate benchmark-backed roadmap item.

## Options considered

### Keep SoA and prefetch private to dense adapters

The dense provider owns `DensePointView<'a>`, aligned packing, SIMD dispatch,
and any future prefetch experiment. Core HNSW continues to pass query and
candidate indices through `DataSource`.

This preserves the hexagonal boundary and lets non-dense providers avoid
irrelevant layout requirements.

### Add cross-node beam scoring to core HNSW

Core HNSW could aggregate candidates from several insertion or search jobs into
larger scoring batches. This may improve SIMD occupancy, but it couples graph
traversal scheduling to dense-provider batch policy and complicates Rayon
ownership, ordering, and cache behaviour.

This option is deferred until a prototype shows a measured win that survives
determinism checks.

### Store a secondary dimension-major SoA copy

The dense provider could maintain a persistent dimension-major copy in addition
to the current row-major data. This would avoid repeated packing for query
batches, but it increases resident memory and update complexity for every
dense dataset, including workloads where HNSW scoring is not the dominant cost.

This option is deferred until measurements prove the memory trade-off is
worthwhile.

### Add `batch_distances_into` to `DataSource`

Core could expose an output-buffer-reuse method so callers provide scratch
storage. This may reduce allocation pressure, but it expands the public trait
contract with length-equality and output-atomicity obligations for every
provider.

This option is deferred until allocation profiling shows it is a real
bottleneck and the trait contract can be specified cleanly.

## Decision outcome

Keep packed coordinate layout, SIMD dispatch, and prefetch experiments inside
`chutoro-providers-dense`. Core HNSW should continue to express candidate
scoring as query-plus-candidate indices through `DataSource`.

Create separate evidence-gated roadmap items for the three deferred structural
levers:

- cross-node beam scoring;
- persistent dimension-major SoA storage;
- `batch_distances_into` output-buffer reuse.

Each item must bring its own benchmark harness, correctness gate, and public
API justification before implementation.

## Consequences

- Core HNSW keeps a stable adapter boundary while the dense provider remains
  free to evolve its private layout and prefetch policy.
- The current `2.3.1` work records a verified invariant and measurement
  baseline rather than landing speculative structural churn.
- Future optimization proposals have clearer evidence gates and can be reviewed
  independently.

## Known risks and limitations

- Very large candidate batches may still benefit from cross-node aggregation,
  but that remains unproven for the measured HNSW profiles.
- Repacking costs may become visible on a different dataset or hardware profile.
  The deferred roadmap items intentionally keep that path open.
- Keeping `batch_distances_into` deferred means allocation improvements remain
  local until profiling proves a shared trait method is needed.
