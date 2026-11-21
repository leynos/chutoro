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

### `CacheKey` and `ObjectKey` are opaque string keys

`chutoro-bench-datasets` no longer exposes `camino::Utf8PathBuf` through its
storage keys. `CacheKey` and `ObjectKey` wrap an `Arc<str>`, and `new` accepts
any `impl Into<Arc<str>>`, such as `&str` or `String`. The path-returning
`to_path_buf` and `into_inner` accessors are removed; read the identifier with
`AsRef<str>` or `Display`.

```rust
use chutoro_bench_datasets::CacheKey;

let key = CacheKey::new("cache/mnist/raw.gz");
assert_eq!(key.as_ref(), "cache/mnist/raw.gz");
```

Adapters that need a filesystem path, object-store prefix, or other
backend-specific address resolve it inside their own boundary, for example
`Utf8PathBuf::from(key.as_ref())`. The `Storage` and `Publisher` port
signatures are unchanged, so recipe code needs no migration beyond replacing
path accessor calls.

### Memory guard models the multi-layer harvest

The pre-flight memory guard behind `ChutoroBuilder::with_max_bytes` now
estimates the candidate edges that CPU HNSW construction harvests. It decides
before construction starts, so it has no measured harvest to count: it derives
the count from the effective `ef_construction` search width and the level
sampler's expected layer count. Candidate, mutual-reachability,
validation-buffer and prepared-edge terms use `n × ef × layers` instead of the
single-layer `n × M` model. The estimate is therefore larger than before for
the same input, so a limit that used to admit a batch may now reject it with
`ChutoroError::MemoryLimitExceeded`.

The parameter-aware estimate derives its layer count from the configured level
multiplier, so `HnswParams::with_level_multiplier` changes the estimate; a
multiplier that stops no node counts all `max_level + 1` layers. The legacy
`estimate_peak_bytes(point_count, max_connections)` helper keeps its
two-argument signature. It uses `4 × M` as the harvest width and the default
level multiplier and cap, but it passes zero as the search-state width, so it
omits the search-state term that `estimate_peak_bytes_for_hnsw_params`
includes. Raise the configured limit, or lower `ef_construction` and
`max_level`, if a previously accepted batch no longer fits.

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

### Additive API: `ALL` on error-code enums

Every error-code enum exposes an `ALL` slice listing its variants in
declaration order. Diagnostics and tests can enumerate the taxonomy without a
hand-maintained list. The enums remain `#[non_exhaustive]`, so `ALL` can grow
in a minor release; do not treat its length as a stability guarantee.

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

### CLI diagnostics default to `warn`

The `chutoro-cli` binary now installs its `tracing` filter at the `warn` level
when `RUST_LOG` is unset or invalid, instead of `info`. This is a change in CLI
output rather than in the library API: a successful run writes only its summary
and no longer emits span lifecycle lines on `stderr`. Operators who relied on
the previous verbose output must request it explicitly:

```sh
RUST_LOG=info chutoro-cli run text words.txt
```

The `text` data source also defaults `--metric` to `levenshtein`, its only
supported value, so a first run need not name it. Invocations that pass the
flag explicitly continue to work unchanged.
