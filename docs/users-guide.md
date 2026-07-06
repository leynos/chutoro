# Chutoro user guide

Integration of the `chutoro-core` crate into Rust applications is described
below. The guide focuses on the APIs exposed to downstream crates and the
behaviour of the CPU FISHDBC implementation.

## Audience and scope

The guide targets developers embedding Chutoro inside their services. It
assumes familiarity with Rust, asynchronous job runners, and dependency
management via Cargo.

## Adding the crate

The project's `Cargo.toml` must include `chutoro-core`. The default feature set
enables the CPU backend; the `gpu` feature prepares the orchestration surface
for a future GPU backend.

```toml
[dependencies]
chutoro-core = "0.1.0"
```

The crate exports all public entry points from the root module, so required
types can be imported directly from `chutoro_core`.

## Running the clustering pipeline

A `Chutoro` instance is constructed with `ChutoroBuilder`, followed by
invocation of `run` with a `DataSource` implementation.

```rust
use chutoro_core::{ChutoroBuilder, DataSource, DataSourceError, ExecutionStrategy};

struct Dummy(Vec<f32>);

impl DataSource for Dummy {
    fn len(&self) -> usize { self.0.len() }
    fn name(&self) -> &str { "dummy" }
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
    }
}

let chutoro = ChutoroBuilder::new()
    .with_min_cluster_size(8)
    .with_execution_strategy(ExecutionStrategy::CpuOnly)
    .build()?;
let result = chutoro.run(&Dummy(vec![1.0, 2.0, 4.0, 8.0]))?;
assert_eq!(result.cluster_count(), 1);
# Ok::<(), chutoro_core::ChutoroError>(())
```

`ExecutionStrategy::Auto` runs the CPU backend. The `gpu` feature prepares the
orchestration surface for a future accelerator backend; requesting
`ExecutionStrategy::GpuPreferred` currently yields `BackendUnavailable`.

## Incremental clustering sessions

Prefer `build_session()` over `Chutoro::run()` when the application needs a
long-lived, incrementally updated index instead of a one-shot batch clustering
run.

The public session surface exposes three types. `SessionConfig` carries the
validated `min_cluster_size`, `HnswParams`, and `SessionRefreshPolicy`.
`SessionRefreshPolicy` represents either manual refresh or an append-threshold
trigger. `ClusteringSession<D>` owns the live session state.

Construct a session through `ChutoroBuilder`:

```rust
use std::sync::Arc;
use chutoro_core::{
    ChutoroBuilder, DataSource, DataSourceError, MetricDescriptor,
    SessionRefreshPolicy,
};

# struct Dummy(Vec<f32>);
# impl DataSource for Dummy {
#     fn len(&self) -> usize { self.0.len() }
#     fn name(&self) -> &str { "dummy" }
#     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
#         let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
#         let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
#         Ok((a - b).abs())
# }
```

Sessions are CPU-only, so `ExecutionStrategy::GpuPreferred` is rejected during
`build_session()`. Empty and undersized sources are accepted at construction
time because session creation does not seed HNSW or run the batch bootstrap
path.

After `append`, newly inserted points have dirty core distances. Calling
`core_distance(i)` before a recompute returns `None` for those points. Call
`recompute_core_distances()` after append batches to compute core distances for
new points and for existing points that appeared near those new points in HNSW.
Treat these recomputed values as provisional until each point has at least
`min_cluster_size` non-self neighbours. Before that neighbourhood saturation
point, the fallback core-distance rule can still increase; monotonic
non-increase only applies after saturation. Call
`recompute_core_distances_full()` when the session must re-establish parity
with a from-scratch batch core-distance pass; it searches every inserted point
and is more expensive than the incremental path.

Use `append(&[...])` to insert source indices that already exist in the backing
`DataSource`. The session does not copy or extend source storage; the caller
owns that storage contract. Each index is inserted into the live HNSW index
through the edge-harvesting path, and harvested candidate edges are kept
internally for the later refresh workflow.

`append` is fail-fast with partial progress. If a slice contains `[0, 1, bad]`
and the first two inserts succeed, those points remain in the session and their
harvested edges remain pending when the error for `bad` is returned.
Out-of-bounds indices surface as `ChutoroError::DataSource`; duplicate indices
and HNSW structural failures surface as `ChutoroError::CpuHnswFailure`.

#     fn metric_descriptor(&self) -> MetricDescriptor {
#         MetricDescriptor::new("abs")
# }
```

Sessions are CPU-only, so `ExecutionStrategy::GpuPreferred` is rejected during
`build_session()`. Empty and undersized sources are accepted at construction
time because session creation does not seed HNSW or run the batch bootstrap
path.

After `append`, newly inserted points have dirty core distances. Calling
`core_distance(i)` before a recompute returns `None` for those points. Call
`recompute_core_distances()` after append batches to compute core distances for
new points and for existing points that appeared near those new points in HNSW.
Treat these recomputed values as provisional until each point has at least
`min_cluster_size` non-self neighbours. Before that neighbourhood saturation
point, the fallback core-distance rule can still increase; monotonic
non-increase only applies after saturation. Call
`recompute_core_distances_full()` when the session must re-establish parity
with a from-scratch batch core-distance pass; it searches every inserted point
and is more expensive than the incremental path.

Use `append(&[...])` to insert source indices that already exist in the backing
`DataSource`. The session does not copy or extend source storage; the caller
owns that storage contract. Each index is inserted into the live HNSW index
through the edge-harvesting path, and harvested candidate edges are kept
internally for the later refresh workflow.

`append` is fail-fast with partial progress. If a slice contains `[0, 1, bad]`
and the first two inserts succeed, those points remain in the session and their
harvested edges remain pending when the error for `bad` is returned.
Out-of-bounds indices surface as `ChutoroError::DataSource`; duplicate indices
and HNSW structural failures surface as `ChutoroError::CpuHnswFailure`.

# }
```

Sessions are CPU-only, so `ExecutionStrategy::GpuPreferred` is rejected during
`build_session()`. Empty and undersized sources are accepted at construction
time because session creation does not seed HNSW or run the batch bootstrap
path.

After `append`, newly inserted points have dirty core distances. Calling
`core_distance(i)` before a recompute returns `None` for those points. Call
`recompute_core_distances()` after append batches to compute core distances for
new points and for existing points that appeared near those new points in HNSW.
Treat these recomputed values as provisional until each point has at least
`min_cluster_size` non-self neighbours. Before that neighbourhood saturation
point, the fallback core-distance rule can still increase; monotonic
non-increase only applies after saturation. Call
`recompute_core_distances_full()` when the session must re-establish parity
with a from-scratch batch core-distance pass; it searches every inserted point
and is more expensive than the incremental path.

Use `append(&[...])` to insert source indices that already exist in the backing
`DataSource`. The session does not copy or extend source storage; the caller
owns that storage contract. Each index is inserted into the live HNSW index
through the edge-harvesting path, and harvested candidate edges are kept
internally for the later refresh workflow.

`append` is fail-fast with partial progress. If a slice contains `[0, 1, bad]`
and the first two inserts succeed, those points remain in the session and their
harvested edges remain pending when the error for `bad` is returned.
Out-of-bounds indices surface as `ChutoroError::DataSource`; duplicate indices
and HNSW structural failures surface as `ChutoroError::CpuHnswFailure`.

# fn example(source: Arc<Dummy>) -> Result<(), chutoro_core::ChutoroError> {
let mut session = ChutoroBuilder::new()
    .with_min_cluster_size(10)
    .with_session_refresh_policy(SessionRefreshPolicy::manual())
    .build_session(source)?;

assert_eq!(session.point_count(), 0);
assert_eq!(session.snapshot_version(), 0);

session.append(&[0, 1])?;
session.recompute_core_distances()?;

assert_eq!(session.point_count(), 2);
assert!(session.core_distance(0).is_some());
assert_eq!(session.snapshot_version(), 0);
# Ok(())
# }
```

Sessions are CPU-only, so `ExecutionStrategy::GpuPreferred` is rejected during
`build_session()`. Empty and undersized sources are accepted at construction
time because session creation does not seed HNSW or run the batch bootstrap
path.

After `append`, newly inserted points have dirty core distances. Calling
`core_distance(i)` before a recompute returns `None` for those points. Call
`recompute_core_distances()` after append batches to compute core distances for
new points and for existing points that appeared near those new points in HNSW.
Treat these recomputed values as provisional until each point has at least
`min_cluster_size` non-self neighbours. Before that neighbourhood saturation
point, the fallback core-distance rule can still increase; monotonic
non-increase only applies after saturation. Call
`recompute_core_distances_full()` when the session must re-establish parity
with a from-scratch batch core-distance pass; it searches every inserted point
and is more expensive than the incremental path.

Use `append(&[...])` to insert source indices that already exist in the backing
`DataSource`. The session does not copy or extend source storage; the caller
owns that storage contract. Each index is inserted into the live HNSW index
through the edge-harvesting path, and harvested candidate edges are kept
internally for the later refresh workflow.

`append` is fail-fast with partial progress. If a slice contains `[0, 1, bad]`
and the first two inserts succeed, those points remain in the session and their
harvested edges remain pending when the error for `bad` is returned.
Out-of-bounds indices surface as `ChutoroError::DataSource`; duplicate indices
and HNSW structural failures surface as `ChutoroError::CpuHnswFailure`.

### Limitations

- The v1 incremental design is append-oriented; deletion and arbitrary
  in-place mutation are not part of the public session surface.
- Stable cluster identity across snapshots is not guaranteed until roadmap item
  `12.3.1` lands.
- Refresh is intended to operate as a micro-batched workflow rather than a
  per-item online relabelling path.
- A refresh can relabel existing points as well as newly appended points.

Refresh and full batch bootstrap are not yet available on the public session
surface. Those workflows remain future roadmap work. The `cpu` feature must be
enabled to access `build_session()`, `append(&[usize])`, `SessionRefreshPolicy`,
`SessionConfig`, and `ClusteringSession<D>`.

## Implementing data sources

`DataSource` abstracts item storage and distance calculations. Implementations
must provide three methods:

- `len` returns the number of items.
- `name` yields a human-readable identifier surfaced in telemetry and errors.
- `distance` computes a pairwise distance, returning `DataSourceError` on
  failure.

The default `distance_batch` helper uses `distance` to fill an output buffer
and keeps it unchanged if any pair fails. Override when the backend can compute
batches more efficiently.

The CPU backend performs parallel HNSW insertion, so `Chutoro::run` requires a
`DataSource + Sync`.

Empty inputs should be handled by returning `DataSourceError::EmptyData` or
`ZeroDimension` during ingestion. Chutoro rejects a `DataSource` with zero
items, or one with fewer than `min_cluster_size` items, before invoking the
backend.

## Working with `CpuHnsw` directly

Advanced integrations can build and query the Hierarchical Navigable Small
World (HNSW) index directly through `CpuHnsw` instead of calling
`Chutoro::run`. This is useful when an application needs to manage index
construction incrementally or consume harvested candidate edges while building
its own graph-derived structures.

`CpuHnsw::build(source, params)` constructs a complete index from every item in
the `DataSource`. `CpuHnsw::build_with_edges(source, params)` performs the same
build, but also returns an `EdgeHarvest` containing the candidate edges
discovered during insertion. `CpuHnsw::with_capacity(params, capacity)` creates
an empty index that can be populated manually with `insert` or
`insert_harvesting`.

Use `insert(node, source)` when only graph mutation is required. Use
`insert_harvesting(node, source)` when the insertion must also return the
candidate edges considered during planning. The first insertion into an empty
index returns an empty `Vec<CandidateEdge>` because there are no prior nodes to
connect to. Subsequent insertions return candidate edges that can be consumed
by incremental minimum spanning tree (MST) or auditing workflows.

`insert_harvesting` follows the same insertion rules as `insert` and returns
`Result<Vec<CandidateEdge>, HnswError>`. `HnswError::DuplicateNode` is returned
when the same node identifier is inserted twice. `HnswError::DataSource`
surfaces failures from `distance` or `distance_batch`, while `EmptyBuild`,
`InvalidParameters`, `NonFiniteDistance`, and lock-related errors report
invalid setup or inconsistent runtime state.

After insertion, `search(source, query, ef)` returns the `ef` nearest
neighbours currently reachable from the HNSW entry point. The query source must
implement `DataSource + Sync`, matching the requirement for parallel insertion.
For an end-to-end example, see the Rustdoc for
`chutoro_core::CpuHnsw::insert_harvesting`.

## Results and assignments

`Chutoro::run` returns a `ClusteringResult`, which exposes the per-item
`assignments` and a pre-computed `cluster_count`. The helper enforces that
cluster identifiers start at `0` and are contiguous.
`ClusteringResult::try_from_assignments` is available when validation of
identifiers is required before constructing a result manually. Errors are
reported via the `NonContiguousClusterIds` enum so actionable feedback can be
surfaced upstream.

Each assignment stores a `ClusterId`. The underlying value can be accessed with
`get()` when serializing or displaying results.

## Error handling

Builder validation returns `ChutoroError::InvalidMinClusterSize` when the
provided minimum cluster size is zero. `Chutoro::run` surfaces runtime failures
via `ChutoroError` variants:

- `EmptySource`: returned when a `DataSource` yields zero items.
- `InsufficientItems`: triggered if `len()` falls below `min_cluster_size`.
- `BackendUnavailable`: emitted when the requested `ExecutionStrategy` is not
  compiled into the binary.
- `DataSource`: raised when `distance` or `distance_batch` fails. Use
  `ChutoroError::data_source_code()` to recover the underlying
  `DataSourceErrorCode` and respond programmatically.
- `CpuHnswFailure`, `CpuMstFailure`, and `CpuHierarchyFailure`: raised when the
  CPU backend encounters internal failures in HNSW construction/search, MST
  construction, or hierarchy extraction.

`DataSourceError` distinguishes out-of-bounds indices, dimension mismatches,
and invalid buffers. Propagate these errors verbatim, so callers receive stable
error codes via `DataSourceError::code()`.

## Distance helpers

`chutoro-core` also ships scalar Euclidean and cosine distance helpers. Both
validate inputs for finite values, matching dimensions, and positive lengths
before computing the distance. The functions return the `Distance` newtype,
which dereferences to `f32` and exposes the raw value through `value()`.

Use `CosineNorms::from_vectors` or `CosineNorms::new` to pre-compute validated
L2 norms for cosine distance. Reuse the cached norms across repeated queries to
avoid redundant work; the cosine helper verifies cached values before it
performs the final calculation.

## Feature flags and execution strategies

The crate exposes the following feature flags:

- `cpu` includes the CPU backend in the default feature set.
- `metrics` exposes metrics emission from hot paths.
- `gpu` prepares the GPU execution path selection surface (the accelerator
  implementation is not yet available).
- `skeleton` is a legacy compatibility flag retained for early versions; it is
  no longer required by the CPU backend.

Choose an `ExecutionStrategy` that matches the compiled features. Allowing
`Auto` keeps behaviour stable across builds while seamlessly adopting GPU
support when available.

## Preparing benchmark datasets

The `chutoro-bench-datasets` crate provides a typed lifecycle for benchmark
dataset preparation. It is intended for benchmark and evaluation tooling that
needs to fetch source data, validate it, prepare benchmark-ready bytes, and
publish a final artefact through explicit infrastructure ports.

### Dataset audience and scope

Use this crate when writing benchmark dataset recipes or tests around dataset
adapters. It is not required for normal clustering with `chutoro-core`.

### Adding the dataset crate

Add `chutoro-bench-datasets` when the application owns dataset preparation.
Enable the `testing` feature in tests to use in-memory and filesystem adapters.

```toml
[dependencies]
chutoro-bench-datasets = "0.1.0"

[dev-dependencies]
chutoro-bench-datasets = { version = "0.1.0", features = ["testing"] }
```

### Dataset recipe usage pattern

Implement `DatasetRecipe` for each dataset. Its associated types model the four
ordered phases: `Fetched`, `Validated`, `Prepared`, and `Published`. Each phase
consumes the previous phase output, so callers cannot skip validation before
preparation at compile time.

Recipes access I/O through `RecipeContext` instead of storing infrastructure
state directly:

- `Fetcher` reads declared source bytes.
- `Storage` caches mutable intermediate artefacts.
- `Publisher` writes the final prepared artefact.

The `testing` feature exposes `InMemoryFetcher`, `InMemoryStorage`,
`InMemoryPublisher`, `FilesystemFetcher`, and `StubRecipe` for adapter contract
tests and lifecycle tests.

```rust
use bytes::Bytes;
use chutoro_bench_datasets::{
    PublishedArtefact, RecipeContext, SourceUrl, run_recipe,
    testing::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage, StubRecipe},
};

let source = SourceUrl::parse("https://example.test/dataset.bin")?;
let fetcher = InMemoryFetcher::new([(source.clone(), Bytes::from_static(b"abc"))]);
let storage = InMemoryStorage::default();
let publisher = InMemoryPublisher::default();
let ctx = RecipeContext::new(&fetcher, &storage, &publisher);
let recipe = StubRecipe::new("example", vec![source]);

let published = run_recipe(&recipe, &ctx)?;
assert_eq!(published.manifest_uri().as_str(), "manifests/example.json");
# Ok::<(), chutoro_bench_datasets::RecipeError>(())
```

For the design rationale behind the four-phase trait and port split, see
[ADR-003: Benchmark dataset recipe trait](adr-003-bench-dataset-recipe-trait.md).

## Benchmarking

The companion `chutoro-benches` crate ships Criterion benchmarks for the four
CPU pipeline stages: Hierarchical Navigable Small World (HNSW) index
construction, edge harvest, minimum spanning tree (MST) computation, and
hierarchy extraction. These benchmarks use a `SyntheticSource` — a `DataSource`
implementation that generates random Euclidean vectors from a seeded random
number generator (RNG) for reproducibility.

Run all benchmarks with:

```sh
make bench
```

Criterion writes HTML timing reports to `target/criterion/`. The benchmarks
cover multiple dataset sizes and parameter combinations so that scaling
behaviour is visible. Consumers integrating `chutoro-core` into their own
projects can use `chutoro-benches` as a reference for structuring performance
tests around the pipeline APIs.

### Neighbour-scoring diagnostics

This contributor-only benchmark is documented in
[Neighbour scoring measurements](./developers-guide.md#neighbour-scoring-measurements).
