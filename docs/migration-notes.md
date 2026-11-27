# Migration notes

This document records compatibility changes for upcoming Chutoro releases.

## 0.2.0

### The `skeleton` feature is removed

`chutoro-core` no longer declares the `skeleton` feature. It had enabled
nothing since the CPU backend stopped depending on it, and it was kept only so
that older manifests would still resolve. Remove `"skeleton"` from any
`features` list that names it for `chutoro-core`; Cargo rejects a dependency
that requests an undeclared feature. No code or behaviour changes.

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

### `PartialState::orphaned_cache_key` is removed

`PartialState` no longer carries an `orphaned_cache_key` field. The driver
could never populate it: a recipe writes to the cache through
`RecipeContext::storage`, so `run_recipe` has no record of the keys it created,
and the field was always `None` under the driver.

A `cleanup` implementation that branched on `partial.orphaned_cache_key` never
took the populated branch, so delete that branch. Recipes that must remove
partial output should record the `CacheKey` values they write, or derive them
deterministically from their inputs, and remove them through their own
mechanism. `Storage` has no deletion operation, and the driver still reports
only the last completed phase, so a failing phase can leave writes behind even
when its phase is not named in the state.

```rust
// Before: this branch never ran under run_recipe.
if partial.orphaned_cache_key.is_some() {
    remove_written_entries()?;
}

// After: the recipe removes the keys it recorded when it wrote them.
for key in &self.written_keys {
    remove_cache_entry(key)?;
}
```

`PartialState::new`, `PartialState::highest_completed_phase`, and the
`DatasetRecipe::cleanup` signature are unchanged. Code that assigned the field
on a `PartialState` it owned should drop the assignment.

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

### New `core_distances` stage label

The `metrics` feature already emitted `chutoro.cpu_pipeline.stage_seconds` for
the `mutual_reachability` and `kruskal` stages. CPU batch runs now also record
a sample for the `core_distances` stage in the same histogram, so the batch
core-distance share is readable beside the rest of the pipeline. The label set
stays fixed and grows by exactly one value; existing labels keep their meaning,
a failing stage still records no sample, and builds without the feature are
unchanged.

### Additive API: `http` staging locks

The new `http` feature exposes `chutoro_bench_datasets::transfer`, which
re-exports `StagingLock`, `StagingLockGuard`, `LockPolicy`, `LockPolicyError`,
`LockStep`, `LockAttempt`, `DEFAULT_POLL_INTERVAL`, and `MIN_POLL_INTERVAL`. It
enables the optional `cap-std` and `rustix` dependencies; `default` and
`testing` are unchanged, so existing consumers see no behaviour change unless
they opt in.

Four `RecipeError` variants describe staging-lock failures:
`StagingLockInvalidName`, `StagingLockContended`, `StagingLockIo`, and
`StagingLockUnsupportedPlatform`. `RecipeError` remains `#[non_exhaustive]`, so
an exhaustive downstream `match` already needs a wildcard arm; add specific
arms only where the caller reacts to a lock failure. The
[user's guide](users-guide.md#staging-download-locks) shows the feature and the
API, and [ADR-009](adrs/adr-0009-staging-lock-via-rustix-flock.md) records the
design.

### Additive API: `ClusteringSession::refresh` and `labels`

`ClusteringSession::refresh()` publishes a flat cluster-label snapshot produced
by an incremental minimum spanning tree refresh, and
`ClusteringSession::labels()` returns that snapshot as `Arc<Vec<usize>>`. Both
are additive: existing session callers need no migration, and no error variant
or production dependency is added. A successful refresh advances
`snapshot_version` by exactly one and clears the pending edge buffer. Appended
point ids must form the contiguous prefix `[0, point_count)`; a non-contiguous
append makes `refresh` return `CpuMstFailure` with `INVALID_NODE_ID`. Sessions
below `min_cluster_size` publish an all-noise snapshot and retain their
harvested candidates for the first real refresh. The
[users' guide](users-guide.md#refreshing-labels) documents the workflow and the
error contract.

### Additive API: core-distance selection helper

`chutoro_core::core_distance_from_neighbours` exposes the core-distance
selection rule that the batch pipeline, incremental sessions, and the benchmark
harness use. It takes a point's sorted, non-self neighbours and the minimum
cluster size and returns the `m - 1`-th neighbour's distance, the farthest
supplied neighbour when the list is shorter, or `0.0` for an empty list. This
is a new function; no existing API changes. Adapters still own the HNSW search
and the removal of the query point from its own neighbour list.

### Batch core distances run in parallel

The CPU batch pipeline now searches the HNSW graph for all points through an
indexed Rayon parallel iterator instead of a serial loop. Output order and
error identity are unchanged: results stay in source-index order, and the
reported error belongs to the lowest source index at any thread count. Callers
that pinned behaviour to a single thread should pin `RAYON_NUM_THREADS` or
install an explicit Rayon pool. The incremental session path remains serial. The
[user's guide](users-guide.md#running-the-clustering-pipeline) documents the
contract.

Because those searches hold state at the same time, the `with_max_bytes`
estimate now counts one search state per Rayon worker rather than one per run. A
`max_bytes` that admitted a run under the former estimate can therefore be
rejected on a wide pool, which is the intended conservative behaviour: the
former estimate under-counted live search state.
`estimate_peak_bytes_for_hnsw_params` keeps its single-search meaning, and the
new `estimate_peak_bytes_for_hnsw_params_with_concurrency` takes the cluster
size and the pool width explicitly, so each worker's search state is sized to
the width the batch stage actually searches at rather than to the construction
breadth alone.

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

### `Checksum` is an inhabited value with a `sha256:<hex>` parser

`Checksum` was an uninhabited enum whose only variant sat behind
`#[cfg(any())]`, so `SourceSpec.checksum` could never hold a value. It is now
the struct `Checksum { algorithm: DigestAlgorithm, bytes: [u8; 32] }`, and
`SourceSpec.checksum` accepts a parsed value. `DigestAlgorithm` is a
non-exhaustive enum that currently names only SHA-256.

`Checksum::parse` accepts `sha256:<hex>` in either hexadecimal case, requires
64 payload bytes, and displays the canonical lowercase form:

```rust
use chutoro_bench_datasets::{Checksum, DigestAlgorithm};

let text = concat!(
    "sha256:",
    "0123456789abcdef0123456789abcdef",
    "0123456789abcdef0123456789abcdef",
);
let checksum = Checksum::parse(text)?;
assert_eq!(checksum.algorithm(), DigestAlgorithm::Sha256);
assert_eq!(checksum.to_string(), text);
# Ok::<(), chutoro_bench_datasets::RecipeError>(())
```

### `RecipeError::ChecksumUnsupported` is removed

Checksum support no longer returns a "deferred" placeholder. Parsing failures
now return `RecipeError::Integrity`, which boxes an `IntegrityFailure`; the
`RecipeError::integrity` constructor builds one. Replace a match on the removed
variant:

```rust
use chutoro_bench_datasets::{IntegrityFailure, RecipeError};

let error = RecipeError::integrity(IntegrityFailure::MissingAlgorithmPrefix);
assert!(matches!(error, RecipeError::Integrity(_)));
```

Code that matched `RecipeError::ChecksumUnsupported` should handle
`RecipeError::Integrity` and inspect the boxed `IntegrityFailure` for the
specific cause, such as `UnknownAlgorithm` or `InvalidDigestLength`.

### Additive API: compound cleanup failures

When a recipe phase and its cleanup both fail, `run_recipe` now returns a
`RecipeError::Cleanup` whose source is a `CleanupFailure`. That payload retains
both the original phase error and the cleanup error, so a caller can diagnose
the compound failure from the returned `Result` without a `tracing` subscriber.

The new public items are `CleanupFailure`,
`RecipeError::cleanup_with_phase_error`, `RecipeError::cleanup_failure`,
`CleanupFailure::phase_error`, and `CleanupFailure::cleanup_error`. The phase
error is the payload's source; the cleanup error is read through
`cleanup_error()`:

```rust
use std::error::Error;

use chutoro_bench_datasets::{Phase, PortName, RecipeError};

fn main() -> Result<(), Box<dyn Error>> {
    let error = RecipeError::cleanup_with_phase_error(
        Phase::Prepare,
        RecipeError::prepare("archive corrupt"),
        RecipeError::port(PortName::Storage, "cache locked"),
    );
    let cleanup = error
        .cleanup_failure()
        .ok_or_else(|| std::io::Error::other("cleanup failure should be retained"))?;
    assert!(matches!(cleanup.phase_error(), RecipeError::Prepare(_)));
    assert_eq!(
        cleanup.cleanup_error().to_string(),
        "port Storage failed: cache locked",
    );
    Ok(())
}
```

`RecipeError::cleanup(phase, source)` stores the supplied source without adding
a phase error. If that source is a `CleanupFailure`, it retains both errors and
`cleanup_failure()` returns `Some`; an ordinary cleanup error is stored on its
own and `cleanup_failure()` returns `None`. Existing matches on
`RecipeError::Cleanup { phase, .. }` need no change.

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
