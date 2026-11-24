# Migrating to v0.1.0

This guide covers the changes in the v0.1.0 release series, up to and including
v0.1.0-beta1, the first release published to crates.io.

New users can skip it: start with the [users' guide](users-guide.md) instead.
Read on if you depended on Chutoro through a Git or path dependency before
v0.1.0-beta1. Most applications need only the dependency change below; the
remaining sections apply only if you use the API or behaviour they name.

## Depend on the published prerelease

Replace the Git or path dependency with the published crate. A plain `"0.1.0"`
requirement does not match a prerelease, so name the beta explicitly:

<!-- tested-example: migration-dependency -->

```toml
[dependencies]
chutoro-core = "0.1.0-beta1"
```

The data-source crates and the command-line tool follow the same version:
`chutoro-providers-dense`, `chutoro-providers-text`, and `chutoro-cli`.

## Build batch distances from `PointPair`

`DataSource::distance_batch` no longer accepts `&[(usize, usize)]`. A pair is
now the semantic `PointPair` type, whose `left()` and `right()` accessors name
the direction of the computation. Build pairs with the constructor, or convert
an existing tuple collection with `PointPair::from`:

<!-- tested-example: migration-point-pair-construction -->

```rust
use chutoro_core::PointPair;

let pairs = [PointPair::new(0, 1), PointPair::new(2, 0)];

let raw_pairs = vec![(0, 1), (2, 0)];
let converted: Vec<PointPair> = raw_pairs.into_iter().map(PointPair::from).collect();

assert_eq!(converted, pairs);
assert_eq!((pairs[1].left(), pairs[1].right()), (2, 0));
```

Implementations that override `distance_batch` change the parameter type to
`&[PointPair]` and read each endpoint through its accessor. A kernel that still
expects tuples converts at its own boundary, preserving each pair's order:

<!-- tested-example: migration-point-pair-override -->

```rust
use chutoro_core::{DataSource, DataSourceError, PointPair};

struct Line(Vec<f32>);

impl DataSource for Line {
    fn len(&self) -> usize { self.0.len() }
    fn name(&self) -> &str { "line" }
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
    }
    fn distance_batch(&self, pairs: &[PointPair], out: &mut [f32]) -> Result<(), DataSourceError> {
        if pairs.len() != out.len() {
            return Err(DataSourceError::OutputLengthMismatch {
                out: out.len(),
                expected: pairs.len(),
            });
        }
        for (slot, pair) in out.iter_mut().zip(pairs) {
            *slot = self.distance(pair.left(), pair.right())?;
        }
        Ok(())
    }
}

fn main() -> Result<(), DataSourceError> {
    let pairs = [PointPair::new(0, 1), PointPair::new(2, 0)];
    let mut out = [0.0; 2];
    Line(vec![1.0, 4.0, 9.0]).distance_batch(&pairs, &mut out)?;
    assert_eq!(out, [3.0, 8.0]);
    Ok(())
}
```

## Construct results with `try_from_assignments`

The public `ClusteringResult::from_assignments` constructor has been removed.
Code that builds a result from externally supplied assignments uses the fallible
`ClusteringResult::try_from_assignments` and handles `NonContiguousClusterIds`:

<!-- tested-example: migration-try-from-assignments -->

```rust
use chutoro_core::{ClusterId, ClusteringResult, NonContiguousClusterIds};

fn main() -> Result<(), NonContiguousClusterIds> {
    let result = ClusteringResult::try_from_assignments(vec![
        ClusterId::new(0),
        ClusterId::new(1),
        ClusterId::new(0),
    ])?;
    assert_eq!(result.cluster_count(), 2);

    let gap = ClusteringResult::try_from_assignments(vec![ClusterId::new(0), ClusterId::new(2)]);
    assert!(gap.is_err(), "identifiers must be contiguous from zero");
    Ok(())
}
```

The error's variants identify a missing zero, gaps, and identifiers that exceed
or reach the host pointer-width limit.

## Run batches through `Chutoro::run`

The public `run_cpu_pipeline` entry point has been removed. Build a `Chutoro`
from `ChutoroBuilder` and call `run`, which applies the same validated
configuration, preconditions, memory guard, and instrumentation:

<!-- tested-example: migration-run-pipeline -->

```rust
use chutoro_core::{ChutoroBuilder, ChutoroError, DataSource, DataSourceError};

struct Line(Vec<f32>);

impl DataSource for Line {
    fn len(&self) -> usize { self.0.len() }
    fn name(&self) -> &str { "line" }
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
    }
}

fn main() -> Result<(), ChutoroError> {
    let source = Line(vec![0.0, 0.1, 0.2, 0.3, 9.0, 9.1, 9.2, 9.3]);
    let result = ChutoroBuilder::new().with_min_cluster_size(3).build()?.run(&source)?;
    assert_eq!(result.cluster_count(), 2);
    Ok(())
}
```

## Allow for new error codes

`ChutoroErrorCode`, `DataSourceErrorCode`, `HnswErrorCode`, `MstErrorCode`, and
`HierarchyErrorCode` are all `#[non_exhaustive]`. A `match` that lists every
variant without a catch-all arm no longer compiles; add one. Each code enum
also implements `Display`, rendering the same string as `as_str()`, and lists
its variants in an `ALL` slice, which may grow in a minor release:

<!-- tested-example: migration-error-code-match -->

```rust
use chutoro_core::{EdgeHarvest, MstError, MstErrorCode, parallel_kruskal};

fn describe(error: &MstError) -> &'static str {
    match error.code() {
        MstErrorCode::EmptyGraph => "empty graph",
        _ => "other MST failure",
    }
}

fn main() {
    let error = parallel_kruskal(0, &EdgeHarvest::from_unsorted(Vec::new())).expect_err("there are no nodes");
    assert_eq!(describe(&error), "empty graph");
    assert_eq!(error.code().to_string(), "EMPTY_GRAPH");
    assert!(MstErrorCode::ALL.contains(&MstErrorCode::EmptyGraph));
}
```

`parallel_kruskal` now validates edges in a fixed order, so when several edges
are invalid it always reports the first. The users' guide lists
[every subsystem code](users-guide.md#subsystem-error-codes).

## Expect HNSW parameters to apply to batch runs

`ChutoroBuilder::with_hnsw_params` now configures one-shot `run` calls as well
as sessions. Earlier snapshots built the batch index with default parameters,
so a run that set non-default `HnswParams` may now take a different amount of
time and memory.

## Expect larger memory estimates

The estimate behind `with_max_bytes` now counts more of what a run really
allocates, so a limit that admitted a run before may now reject it with
`ChutoroError::MemoryLimitExceeded`:

- the configured distance-cache capacity;
- the candidate edges harvested from every expected index layer, which grow
  with `ef_construction` and `max_level`; and
- one search state per thread of the Rayon pool that runs the batch, sized by
  `min_cluster_size` as well as by `ef_construction`.

Raise the limit, or lower `ef_construction`, `max_level`, or the distance-cache
capacity. `estimate_peak_bytes_for_hnsw_params_with_concurrency` returns the
figure `run` checks, and `batch_search_width` the width each search uses.
`estimate_peak_bytes_for_hnsw_params` and `estimate_peak_bytes` keep their
signatures and their single-search meaning. See
[Bound memory use](users-guide.md#bound-memory-use).

## Expect parallel core-distance searches

Batch runs now search for every item's neighbours in parallel on the calling
thread's Rayon pool. Results and reported errors do not depend on the thread
count, but a run uses more threads than before. Set `RAYON_NUM_THREADS`, or run
inside an explicit pool, to restrict it. See
[Control parallelism](users-guide.md#control-parallelism).

## Expect separate clusters on exact stability ties

Cluster selection now keeps the child clusters when a parent's stability
exactly equals the sum of its children's, as the
[design document](chutoro-design.md) specifies. Earlier snapshots kept the
parent, which merged well-separated groups whenever an integer-valued metric,
such as Levenshtein distance, produced an exact tie. Some datasets therefore
report more clusters than before: the eight words in the
[README example](../README.md#try-it-from-the-command-line) used to form one
cluster and now form two. Regenerate any stored cluster counts or labels for
integer-valued metrics; continuous metrics rarely tie.

## Note the quieter command-line defaults

`chutoro-cli` no longer prints a spurious "structured logging already
configured elsewhere" warning on every invocation. Two defaults also changed:

- `--metric` on the `text` source is optional and defaults to `levenshtein`,
  its only value. Scripts that pass it keep working.
- Diagnostics default to the `warn` level, so a successful run writes only its
  summary. Set `RUST_LOG` to restore the earlier span trace:

<!-- tested-example: migration-cli-verbose -->

```sh
RUST_LOG=info chutoro-cli run --min-cluster-size 2 text words.txt
```

Embedders of the `chutoro_cli` library crate should note that
`LoggingError::InstallFailed` now carries a
`tracing::subscriber::SetGlobalDefaultError` rather than a
`tracing_subscriber::util::TryInitError`.

## Prefer `EdgeHarvest::as_slice` for parallel consumers

This change is additive. `EdgeHarvest::as_slice` returns the harvested
candidate edges as a `&[CandidateEdge]`, so a Rayon consumer can split the list
without first collecting a vector of references. Existing callers of `iter`,
`into_iter`, and `into_inner` need no changes.

## Drop the `skeleton` feature

`chutoro-core` no longer declares the `skeleton` feature, which had no effect.
Cargo rejects a dependency that requests an undeclared feature, so remove
`"skeleton"` from any `features` list for `chutoro-core`. Nothing else changes.

## Use the new helpers and metrics

These changes are additive and need no migration:

- `core_distance_from_neighbours` exposes the rule that `run` and sessions use
  to pick a core distance from a sorted neighbour list.
- `CandidateEdge::canonicalize` is the Oxford spelling of `canonicalise`, which
  remains as an alias.
- The dense provider's `testing` feature exposes Arrow and Parquet fixture
  builders for tests; enable it only under `[dev-dependencies]`.
- With the `metrics` feature, `chutoro.cpu_pipeline.stage_seconds` gains the
  `core_distances` stage, and HNSW insertion records
  `chutoro.hnsw.trim.jobs_submitted_total` and
  `chutoro.hnsw.trim.duration_seconds`. None adds an unbounded label.

The users' guide describes each one in context.

## Update benchmark dataset recipes

`chutoro-bench-datasets` is contributor tooling and is not published, but
recipes written against an earlier snapshot need these changes:

- `CacheKey` and `ObjectKey` are opaque string identifiers. `new` accepts any
  `impl Into<Arc<str>>`, and `to_path_buf` and `into_inner` are removed. Read a
  key with `as_ref()` or `Display`, and let a storage adapter derive a path
  from it, for example with `Utf8PathBuf::from(key.as_ref())`.
- `Checksum` now holds a value. `Checksum::parse` reads `sha256:<hex>` in either
  case and displays the lowercase form.
- `RecipeError::ChecksumUnsupported` is removed. A malformed checksum returns
  `RecipeError::Integrity`, which boxes an `IntegrityFailure` naming the cause.
- The optional `http` feature adds `StagingLock` and its policy types, with
  four new `RecipeError` variants for lock failures. `RecipeError` is
  `#[non_exhaustive]`, so existing matches already have a catch-all arm.

<!-- tested-example: migration-bench-datasets -->

```rust
use chutoro_bench_datasets::{CacheKey, Checksum, DigestAlgorithm, IntegrityFailure, RecipeError};

fn main() -> Result<(), RecipeError> {
    let key = CacheKey::new("cache/mnist/raw.gz");
    assert_eq!(key.as_ref(), "cache/mnist/raw.gz");

    let text = concat!("sha256:", "0123456789abcdef0123456789abcdef", "0123456789abcdef0123456789abcdef");
    let checksum = Checksum::parse(text)?;
    assert_eq!(checksum.algorithm(), DigestAlgorithm::Sha256);
    assert_eq!(checksum.to_string(), text);

    let error = RecipeError::integrity(IntegrityFailure::MissingAlgorithmPrefix);
    assert!(matches!(error, RecipeError::Integrity(_)));
    Ok(())
}
```

The [developers' guide](developers-guide.md#benchmark-dataset-recipes) covers
the recipe model and the staging lock.
