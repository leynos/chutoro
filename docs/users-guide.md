# Chutoro user guide

This guide explains how to integrate the `chutoro-core` crate into your Rust
applications. It focuses on the APIs exposed to downstream crates and the
behaviour of the current walking skeleton implementation.

## Audience and scope

The guide targets developers embedding Chutoro inside their own services. It
assumes familiarity with Rust, asynchronous job runners, and dependency
management via Cargo.

## Adding the crate

Add `chutoro-core` to your project's `Cargo.toml`. Enable the `skeleton`
feature to use the CPU walking skeleton and, optionally, the `gpu` feature for
future GPU support.

```toml
[dependencies]
chutoro-core = { version = "0.1.0", features = ["skeleton"] }
```

The crate exports all public entry points from the root module, so import the
types you need directly from `chutoro_core`.

## Running the clustering pipeline

Construct a `Chutoro` instance with `ChutoroBuilder`, then call `run` with a
`DataSource` implementation.

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

`ExecutionStrategy::Auto` resolves to the CPU skeleton today. Once a GPU
backend ships, the strategy will prefer GPU execution when compiled with the
`gpu` feature.

The walking skeleton partitions input indices into contiguous buckets sized by
`min_cluster_size`. Use this behaviour for smoke testing orchestration only;
the algorithm will change once the full FISHDBC pipeline lands.

## Implementing data sources

`DataSource` abstracts item storage and distance calculations. Implementations
must provide three methods:

- `len` returns the number of items.
- `name` yields a human-readable identifier surfaced in telemetry and errors.
- `distance` computes a pairwise distance, returning `DataSourceError` on
  failure.

The default `distance_batch` helper uses `distance` to fill an output buffer
and keeps it unchanged if any pair fails. Override it when your backend can
compute batches more efficiently.

Guard against empty inputs by returning `DataSourceError::EmptyData` or
`ZeroDimension` during ingestion. Chutoro will reject a `DataSource` with zero
items, or one with fewer than `min_cluster_size` items, before invoking your
backend.

## Results and assignments

`Chutoro::run` returns a `ClusteringResult`, which exposes the per-item
`assignments` and a pre-computed `cluster_count`. The helper enforces that
cluster identifiers start at `0` and are contiguous. Use
`ClusteringResult::try_from_assignments` when you need to validate identifiers
before constructing a result manually. Errors are reported via the
`NonContiguousClusterIds` enum so you can surface actionable feedback upstream.

Each assignment stores a `ClusterId`. Access the underlying value with `get()`
when serialising or displaying results.

## Error handling

Builder validation returns `ChutoroError::InvalidMinClusterSize` when the
provided minimum cluster size is zero. `Chutoro::run` surfaces runtime failures
via `ChutoroError` variants:

- `EmptySource` when a `DataSource` yields zero items.
- `InsufficientItems` when `len()` is below `min_cluster_size`.
- `BackendUnavailable` when the requested `ExecutionStrategy` is not compiled
  into the binary.
- `DataSource` when `distance` or `distance_batch` fails. Use
  `ChutoroError::data_source_code()` to recover the underlying
  `DataSourceErrorCode` and respond programmatically.

`DataSourceError` distinguishes out-of-bounds indices, dimension mismatches,
and invalid buffers. Propagate these errors verbatim so callers receive stable
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

The crate exposes two opt-in features:

- `skeleton` enables the CPU walking skeleton implementation. Without it,
  `ExecutionStrategy::Auto` and `ExecutionStrategy::CpuOnly` return
  `BackendUnavailable`.
- `gpu` prepares the orchestration surface for the forthcoming GPU backend. In
  the walking skeleton build it reuses the CPU path. Without `skeleton`, GPU
  execution errors out until the real accelerator implementation arrives.

Choose an `ExecutionStrategy` that matches the compiled features. Allowing
`Auto` keeps behaviour stable across builds while seamlessly adopting GPU
support when available.
