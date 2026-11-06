# Migration notes

This document records compatibility changes for upcoming Chutoro releases.

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
