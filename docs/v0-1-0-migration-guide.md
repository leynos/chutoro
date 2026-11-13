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

## Expect HNSW parameters to apply to batch runs

`ChutoroBuilder::with_hnsw_params` now configures one-shot `run` calls as well
as sessions. Earlier snapshots built the batch index with default parameters,
so a run that set non-default `HnswParams` may now take a different amount of
time and memory. The memory estimate used by `with_max_bytes` also counts the
configured distance-cache capacity; see
[Bound memory use](users-guide.md#bound-memory-use).

## Expect separate clusters on exact stability ties

Cluster selection now keeps the child clusters when a parent's stability
exactly equals the sum of its children's, as the
[design document](chutoro-design.md) specifies. Earlier snapshots kept the
parent, which merged well-separated groups whenever an integer-valued metric,
such as Levenshtein distance, produced an exact tie. Some datasets therefore
report more clusters than before.

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
