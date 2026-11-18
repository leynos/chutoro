# Migration notes

This document records compatibility changes for upcoming Chutoro releases.

## 0.2.0

### `DataSource::distance_batch` takes `PointPair` slices

`DataSource::distance_batch` no longer accepts `&[(usize, usize)]`. A pair is
now the semantic `PointPair` type, whose ordered `left` and `right` accessors
name the direction of the computation.

Callers that built tuple pairs replace them with the equivalent constructor:

```rust
use chutoro_core::PointPair;

let pairs = [PointPair::new(0, 1), PointPair::new(2, 0)];
```

`PointPair` implements `From<(usize, usize)>`, so an existing collection can be
mapped rather than rebuilt:

```rust
use chutoro_core::PointPair;

let raw_pairs = vec![(0, 1), (2, 0)];
let pairs: Vec<PointPair> = raw_pairs.into_iter().map(PointPair::from).collect();
```

Implementers that override `distance_batch` must change the parameter type to
`&[PointPair]` and read the endpoints with `pair.left()` and `pair.right()`. An
implementation that forwards to a tuple-oriented kernel converts inside its own
boundary; the ordering of each pair must be preserved.

### New metrics: trim scoring

Enabling the `metrics` feature now also emits
`chutoro.hnsw.trim.jobs_submitted_total` and
`chutoro.hnsw.trim.duration_seconds`. The counter adds the full batch size for
every non-empty trim batch the insertion boundary scores, including a batch
that fails during scoring. The histogram records one wall-time sample per
non-empty batch, including failed batches, so it covers the time insertion
spends holding its mutex while scoring. Neither instrument carries labels, so
their cardinality is fixed, and both are additive: builds without the feature
are unchanged.

### Additive API: dense-provider test fixtures

The dense provider exposes reusable Arrow and Parquet fixture builders through
its opt-in `testing` feature. Crates that use them in their own tests should
enable the feature only in `[dev-dependencies]`. This does not change runtime
defaults or require a migration for existing consumers. The
[user's guide](users-guide.md#reusing-dense-fixtures-in-dependent-tests) shows
how to enable the feature.

### Error-code enums are non-exhaustive and implement `Display`

`MstErrorCode`, `HierarchyErrorCode`, and `HnswErrorCode` are now
`#[non_exhaustive]`, matching the error enums they describe. A downstream
`match` that lists every variant without a catch-all arm no longer compiles;
add one, such as `_`. A match that already ends in a catch-all binding needs no
change:

```rust
# use chutoro_core::{MstError, MstErrorCode};
# fn describe(error: &MstError) -> &'static str {
match error.code() {
    MstErrorCode::EmptyGraph => "empty graph",
    _ => "other MST failure",
}
# }
```

All three also implement `Display`, which renders the same stable string as
`as_str()`, so either form can be logged without a separate match.

## 0.1.0

The public `ClusteringResult::from_assignments` constructor is no longer
available. Code that constructs results from externally supplied assignments
must use the fallible `ClusteringResult::try_from_assignments` constructor:

```rust
use chutoro_core::{ClusterId, ClusteringResult, NonContiguousClusterIds};

fn make_result(
    assignments: Vec<ClusterId>,
) -> Result<ClusteringResult, NonContiguousClusterIds> {
    ClusteringResult::try_from_assignments(assignments)
}
```

Callers must propagate or match `NonContiguousClusterIds` errors. The variants
identify missing zero, gaps, duplicates, and identifiers that exceed or reach
the host pointer-width limit. The former panicking helper remains an internal
CPU-pipeline convenience only.

### Additive API: `EdgeHarvest::as_slice`

`EdgeHarvest::as_slice` returns the harvested candidate edges as a
`&[CandidateEdge]`. The method is additive: existing callers of `iter`,
`into_iter`, and `into_inner` are unaffected and need no migration. Consumers
that fan work out with Rayon should prefer `as_slice`, because it borrows the
edge list directly instead of collecting an intermediate vector of references.

### Cluster selection keeps exact stability ties

Stable-cluster selection now keeps the child clusters when a parent cluster's
stability is exactly equal to the sum of its children's stabilities. A parent
is selected only when its stability strictly exceeds that total, matching the
excess-of-mass rule in the design document. Previously an exact tie selected
the parent.

Integer-valued metrics make exact ties routine, so `ClusteringResult` can
report a different `cluster_count` and different flat labels for the same input:

```text
words: apple apply apples appel banana bananas bandana banane
Levenshtein distance, min_cluster_size = 2

before: cluster_count = 1, assignments = [0 0 0 0 0 0 0 0]
after:  cluster_count = 2, assignments = [0 0 0 0 1 1 1 1]
```

Callers that store expected cluster counts, expected flat labels, or partitions
derived from them must regenerate those expectations for integer-valued
metrics. Continuous metrics rarely produce exact ties, so their results are
normally unchanged.
