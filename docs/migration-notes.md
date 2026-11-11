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

### Additive API: dense-provider test fixtures

The dense provider exposes reusable Arrow and Parquet fixture builders through
its opt-in `testing` feature. Crates that use them in their own tests should
enable the feature only in `[dev-dependencies]`. This does not change runtime
defaults or require a migration for existing consumers. The
[user's guide](users-guide.md#reusing-dense-fixtures-in-dependent-tests) shows
how to enable the feature.

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
