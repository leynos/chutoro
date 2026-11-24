# Chutoro users' guide

Chutoro groups items into clusters by how densely they sit together. It needs
only a way to measure the distance between two items and a sense of how small a
group can be before it stops counting as a cluster.

This guide starts with a first run and works towards the lower-level building
blocks. Each part builds on the one before, so newcomers can read from the top
and stop once they have what they need. Readers who already know their way
around can jump straight to a part:

1. [Getting started](#getting-started): install Chutoro and cluster something,
   from the command line or from Rust.
2. [How Chutoro clusters](#how-chutoro-clusters): what `min_cluster_size`
   means, how noise is labelled, and what to expect from small datasets.
3. [Bringing your own data](#bringing-your-own-data): the bundled data sources
   and the `DataSource` trait.
4. [Results and errors](#results-and-errors): reading assignments, scoring
   them, and handling failures.
5. [Tuning and operations](#tuning-and-operations): HNSW parameters, memory
   limits, parallelism, feature flags, and observability.
6. [Advanced building blocks](#advanced-building-blocks): incremental
   sessions, the HNSW index, and the tree and hierarchy stages.

Chutoro v0.1.0-beta1 is a preview release, so interfaces may still change
before v0.1.0. Anyone upgrading from a pre-release snapshot should also read the
[v0.1.0 migration guide](v0-1-0-migration-guide.md).

## Getting started

### Pick the crates you need

Chutoro is published as four crates. Every crate in the series shares the
version `0.1.0-beta1` and needs Rust 1.89 or later.

| Crate                     | Use it to                                              |
| ------------------------- | ------------------------------------------------------ |
| `chutoro-cli`             | Cluster Parquet vectors or a text file from the shell. |
| `chutoro-core`            | Run clustering from Rust over any `DataSource`.        |
| `chutoro-providers-dense` | Load dense `f32` vectors from Parquet or Arrow.        |
| `chutoro-providers-text`  | Compare lines of UTF-8 text by Levenshtein distance.   |

_Table 1: Published crates and their purpose._

### Cluster from the command line

The command-line tool is the quickest way to see Chutoro work. Install it with
Cargo:

<!-- tested-example: guide-cli-install -->

```sh
cargo install chutoro-cli@0.1.0-beta1
```

Save one item per line in a UTF-8 text file. This one is called `fruit.txt`:

<!-- tested-example: guide-cli-text-corpus -->

```text
pear
pears
peer
spear
melon
melons
lemon
felon
```

Run the `text` source over it. `--min-cluster-size` sets the smallest group
worth calling a cluster:

<!-- tested-example: guide-cli-text-run -->

```sh
chutoro-cli run --min-cluster-size 2 text fruit.txt
```

The tool prints the data source name, the number of clusters, and then each
line's zero-based index alongside the cluster it joined:

<!-- tested-example: guide-cli-text-output -->

```text
data source: fruit
clusters: 2
0	0
1	0
2	0
3	0
4	1
5	1
6	1
7	1
```

The `text` source compares lines by Levenshtein edit distance, its only metric
today. The `parquet` source reads a column of fixed-length `f32` vectors, as
written by most embedding pipelines, and compares rows by Euclidean distance:

<!-- tested-example: guide-cli-parquet-run -->

```sh
chutoro-cli run --min-cluster-size 5 --max-bytes 2G parquet embeddings.parquet --column embedding
```

The column must have the Arrow type `FixedSizeList<Float32, D>` with no null
rows. Every option is listed in Table 2.

| Option                   | Applies to           | Meaning                                                            |
| ------------------------ | -------------------- | ------------------------------------------------------------------ |
| `--min-cluster-size <N>` | `run`                | Smallest group reported as a cluster. Defaults to 5.               |
| `--max-bytes <SIZE>`     | `run`                | Refuse runs estimated to need more memory, such as `512M` or `2G`. |
| `--column <NAME>`        | `parquet`            | Column holding the vectors. Required.                              |
| `--metric levenshtein`   | `text`               | Distance between lines. Defaults to `levenshtein`.                 |
| `--name <NAME>`          | `parquet` and `text` | Data source name in the output. Defaults to the file stem.         |

_Table 2: `chutoro-cli run` options._

Byte sizes accept `K`, `M`, `G`, and `T` suffixes in binary units, so `2G` is 2
GiB. On success the tool exits with status 0 and writes only the summary to
standard output. A run that fails exits with status 1, and invalid arguments
exit with status 2; either way the cause is reported on standard error.
Diagnostics stay quiet by default; set `RUST_LOG=info` to see a trace of each
stage, and `CHUTORO_LOG_FORMAT=json` for machine-readable logs.

### Cluster from Rust

Add `chutoro-core` to `Cargo.toml`:

<!-- tested-example: guide-core-dependency -->

```toml
[dependencies]
chutoro-core = "0.1.0-beta1"
```

A run needs a `DataSource`, which tells Chutoro how many items there are and
how far apart any two of them are. Build a `Chutoro` with `ChutoroBuilder`,
then call `run`:

<!-- tested-example: guide-first-run -->

```rust
use chutoro_core::{ChutoroBuilder, ChutoroError, DataSource, DataSourceError};

/// Readings along one axis; distance is the gap between two readings.
struct Readings(Vec<f32>);

impl DataSource for Readings {
    fn len(&self) -> usize { self.0.len() }
    fn name(&self) -> &str { "readings" }
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
    }
}

fn main() -> Result<(), ChutoroError> {
    let readings = Readings(vec![1.0, 1.1, 1.2, 1.3, 8.0, 8.1, 8.2, 8.3]);

    let chutoro = ChutoroBuilder::new().with_min_cluster_size(3).build()?;
    let result = chutoro.run(&readings)?;

    let labels: Vec<u64> = result.assignments().iter().map(|id| id.get()).collect();
    assert_eq!(labels, [0, 0, 0, 0, 1, 1, 1, 1]);
    assert_eq!(result.cluster_count(), 2);
    Ok(())
}
```

`build` checks the configuration and `run` checks the data, so both return a
`Result`. [Results and errors](#results-and-errors) covers what each can report.

## How Chutoro clusters

Chutoro implements FISHDBC, a scalable approximation of HDBSCAN\*. It builds an
approximate nearest-neighbour graph over the data, links the items into a
minimum spanning tree weighted by local density, and then keeps the groups that
stay together across the widest range of density. The
[design document](chutoro-design.md) describes each stage in depth.

### Choose `min_cluster_size`

`min_cluster_size` is the one setting every run needs. It does two jobs:

- It is the smallest group reported as a cluster. Smaller groups are absorbed
  into a neighbour or reported as noise.
- It sets how density is measured. An item's _core distance_ is the distance
  to its `min_cluster_size`-th nearest neighbour, not counting itself.

Together these mean a group needs **at least `min_cluster_size + 1` members**
to stand out as its own cluster: only then does each member find enough
neighbours inside the group to look dense. The readings from the first example
form two groups of four, which separate with a minimum cluster size of 3 and
merge with a minimum of 4:

<!-- tested-example: guide-min-cluster-size -->

```rust
use chutoro_core::{ChutoroBuilder, ChutoroError, DataSource, DataSourceError};

struct Readings(Vec<f32>);

impl DataSource for Readings {
    fn len(&self) -> usize { self.0.len() }
    fn name(&self) -> &str { "readings" }
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
    }
}

fn clusters(min_cluster_size: usize) -> Result<usize, ChutoroError> {
    let readings = Readings(vec![1.0, 1.1, 1.2, 1.3, 8.0, 8.1, 8.2, 8.3]);
    let chutoro = ChutoroBuilder::new().with_min_cluster_size(min_cluster_size).build()?;
    Ok(chutoro.run(&readings)?.cluster_count())
}

fn main() -> Result<(), ChutoroError> {
    assert_eq!(clusters(3)?, 2, "each group of four has three neighbours of its own");
    assert_eq!(clusters(4)?, 1, "a fourth neighbour lies in the other group");
    Ok(())
}
```

Start with the smallest group you would act on, then raise the value if the
results are too fragmented. The builder defaults to 5. A value of 0 is rejected
by `build` with `ChutoroError::InvalidMinClusterSize`, and a source with fewer
items than `min_cluster_size` is rejected by `run` with
`ChutoroError::InsufficientItems`.

### Read noise labels

Items that never settle into a cluster are noise. Chutoro gives every noise
item the same extra label, numbered after the real clusters: with two clusters
and some noise, labels `0` and `1` are clusters and label `2` is noise.
`cluster_count()` counts that noise label too. When nothing qualifies as a
cluster, every item is noise and receives label `0`.

`ClusteringResult` does not yet report which label, if any, is noise. Until the
planned clustering snapshot lands (roadmap item 12.1), treat a label whose
members are scattered rather than close as likely noise.

### See how clusters are selected

Chutoro keeps a group of clusters when, together, they are at least as stable
as the larger cluster that contains them. A parent cluster replaces its
children only when its stability _strictly_ exceeds theirs combined; an exact
tie keeps the children. Integer-valued metrics, such as Levenshtein distance
over short words, produce exact ties often, which is why the two fruit families
in the command-line example come back as two clusters rather than one.

### Expect less from very small datasets

Cluster selection may choose the whole dataset as a single cluster when that
explains the data at least as well as any split. Large datasets rarely end up
there, but a handful of items can: one outlier added to the two word families
in the command-line example tips the selection towards a single cluster. Prefer
tens of items or more when judging results, and treat clusterings of a few
items as illustrations rather than benchmarks.

## Bringing your own data

### Use the bundled data sources

`chutoro-providers-text` holds one UTF-8 string per item and compares items by
Levenshtein distance:

<!-- tested-example: guide-text-provider -->

```rust
use chutoro_core::{ChutoroBuilder, DataSource};
use chutoro_providers_text::TextProvider;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lines = ["pear", "pears", "peer", "spear", "melon", "melons", "lemon", "felon"];
    let fruit = TextProvider::new("fruit", lines.map(String::from).to_vec())?;
    assert_eq!(fruit.distance(0, 1)?, 1.0);

    let result = ChutoroBuilder::new().with_min_cluster_size(2).build()?.run(&fruit)?;
    assert_eq!(result.cluster_count(), 2);
    Ok(())
}
```

`TextProvider::try_from_reader` reads the lines from any `BufRead` instead.

`chutoro-providers-dense` holds fixed-length `f32` vectors and compares them by
Euclidean distance, using SIMD kernels where the CPU supports them. Load a
Parquet column with `DenseMatrixProvider::try_from_parquet_path`, or wrap an
Arrow `FixedSizeListArray` that is already in memory:

<!-- tested-example: guide-dense-provider -->

```rust
use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array};
use arrow_schema::{DataType, Field};
use chutoro_core::{ChutoroBuilder, DataSource};
use chutoro_providers_dense::DenseMatrixProvider;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let values = Float32Array::from(vec![
        0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.1, 0.2, // near the origin
        5.0, 5.0, 5.1, 4.9, 5.0, 5.2, 5.2, 5.1, // near (5, 5)
    ]);
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let rows = FixedSizeListArray::try_new(item, 2, Arc::new(values), None)?;

    let points = DenseMatrixProvider::try_from_fixed_size_list("points", &rows)?;
    assert_eq!(points.len(), 8);

    let result = ChutoroBuilder::new().with_min_cluster_size(3).build()?.run(&points)?;
    assert_eq!(result.cluster_count(), 2);
    Ok(())
}
```

Both crates are added alongside `chutoro-core`. The dense provider's Arrow
types come from `arrow-array` and `arrow-schema` 60.

Tests that need ready-made Arrow arrays or Parquet files can borrow the dense
provider's fixture builders. Enable its `testing` feature as a development
dependency only, so the builders stay out of release builds:

<!-- tested-example: guide-dense-testing-feature -->

```toml
[dev-dependencies]
chutoro-providers-dense = { version = "0.1.0-beta1", features = ["testing"] }
```

The feature exposes builders for fixed-size list arrays, feature fields, and
in-memory or file-backed Parquet data; the
[testing module](../chutoro-providers/dense/src/testing/mod.rs) lists them.
Keeping the feature out of normal builds relies on Cargo's feature resolver 2,
the default since the 2021 edition.

### Implement `DataSource`

Any type can be clustered once it implements `DataSource`. Table 3 lists the
trait's methods.

| Method              | Required | Purpose                                                          |
| ------------------- | -------- | ---------------------------------------------------------------- |
| `len`               | Yes      | Number of items; items are addressed by index `0..len()`.        |
| `name`              | Yes      | Human-readable identifier used in errors.                        |
| `distance`          | Yes      | Distance between two items, or a `DataSourceError`.              |
| `is_empty`          | No       | Whether `len()` is zero.                                         |
| `metric_descriptor` | No       | A stable name for the metric; defaults to `"unknown"`.           |
| `distance_batch`    | No       | Fill an output buffer with the distance for each `PointPair`.    |
| `batch_distances`   | No       | Distances from one query item to a list of candidates, in order. |

_Table 3: `DataSource` methods._

Distances must be finite and non-negative, and should be symmetric:
`distance(i, j)` should equal `distance(j, i)`. A true metric gives the most
predictable clusters, but Chutoro also accepts dissimilarities that break the
triangle inequality. `Chutoro::run` builds its index in parallel, so the source
must also be `Sync`.

Report a bad index with `DataSourceError::OutOfBounds`, and vectors of
different lengths with `DimensionMismatch`. Use `EmptyData` and `ZeroDimension`
when loading fails because there is nothing to cluster.

`metric_descriptor` lets the distance cache tell apart sources that would
otherwise share a name. Override it when one type can be configured with
different metrics or preprocessing.

### Batch distance work

Chutoro asks for distances in batches during index construction. The default
`distance_batch` calls `distance` once per pair. Override it when the backing
store computes many distances faster together, for example with SIMD or a
vectorized database query:

<!-- tested-example: guide-distance-batch -->

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
}

fn main() -> Result<(), DataSourceError> {
    let line = Line(vec![1.0, 4.0, 9.0]);

    let pairs = [PointPair::new(0, 1), PointPair::new(2, 0)];
    let mut out = [0.0_f32; 2];
    line.distance_batch(&pairs, &mut out)?;
    assert_eq!(out, [3.0, 8.0]);

    assert_eq!(line.batch_distances(1, &[0, 2])?, vec![3.0, 5.0]);
    Ok(())
}
```

A `PointPair` keeps its `left` and `right` order, and the output holds the
distance from each pair's `left` item to its `right` item. The default fills
the buffer only when every pair succeeds, and reports
`DataSourceError::OutputLengthMismatch` when the buffer and the pair list
differ in length. An override must keep the same contract and must not swap a
pair's endpoints. A kernel that wants plain `(usize, usize)` tuples converts
each pair at its own boundary.

## Results and errors

### Read assignments

`run` returns a `ClusteringResult`. `assignments()` holds one `ClusterId` per
item, in item order, and `ClusterId::get()` returns the label as a `u64`.
Labels start at `0` and are contiguous; the last one may be the shared noise
label described in [Read noise labels](#read-noise-labels).

To build a `ClusteringResult` from labels produced elsewhere, use
`ClusteringResult::try_from_assignments`. It returns `NonContiguousClusterIds`
when the labels skip a value or do not start at zero.

### Score a clustering

When reference labels are available, `clustering_quality_score` compares them
with a clustering. It reports the Adjusted Rand Index (ARI), from −1 to 1, and
Normalized Mutual Information (NMI), from 0 to 1. Both reach 1 for a perfect
match and ignore how the labels are numbered:

<!-- tested-example: guide-quality-score -->

```rust
use chutoro_core::{ClusteringQualityError, clustering_quality_score};

fn main() -> Result<(), ClusteringQualityError> {
    let expected = [0, 0, 0, 1, 1, 1];
    let found = [1, 1, 1, 0, 0, 0];

    let score = clustering_quality_score(&expected, &found)?;
    assert_eq!((score.ari, score.nmi), (1.0, 1.0));
    Ok(())
}
```

`adjusted_rand_index` and `normalized_mutual_information` return either score
on its own. All three fail with `ClusteringQualityError::LabelLengthMismatch`
when the label lists differ in length.

### Handle errors

`build` and `run` return `ChutoroError`. Every variant has a stable,
machine-readable code from `code().as_str()`, suitable for logs, alerts, and
metrics. Table 4 lists them.

| Variant                 | Code                               | Reported by                     | Cause                                                   |
| ----------------------- | ---------------------------------- | ------------------------------- | ------------------------------------------------------- |
| `InvalidMinClusterSize` | `CHUTORO_INVALID_MIN_CLUSTER_SIZE` | `build`, `build_session`        | `min_cluster_size` is 0.                                |
| `BackendUnavailable`    | `CHUTORO_BACKEND_UNAVAILABLE`      | `build`, `build_session`, `run` | No compiled backend can serve the `ExecutionStrategy`.  |
| `EmptySource`           | `CHUTORO_EMPTY_SOURCE`             | `run`                           | The source has no items.                                |
| `InsufficientItems`     | `CHUTORO_INSUFFICIENT_ITEMS`       | `run`                           | The source has fewer items than `min_cluster_size`.     |
| `MemoryLimitExceeded`   | `CHUTORO_MEMORY_LIMIT_EXCEEDED`    | `run`                           | The memory estimate exceeds the `with_max_bytes` limit. |
| `DataSource`            | `CHUTORO_DATA_SOURCE_FAILURE`      | `run`, sessions                 | `distance` or `distance_batch` returned an error.       |
| `CpuHnswFailure`        | `CHUTORO_CPU_HNSW_FAILURE`         | `run`, sessions                 | Index construction or search failed.                    |
| `CpuMstFailure`         | `CHUTORO_CPU_MST_FAILURE`          | `run`                           | Minimum spanning tree construction failed.              |
| `CpuHierarchyFailure`   | `CHUTORO_CPU_HIERARCHY_FAILURE`    | `run`                           | Cluster extraction failed.                              |

_Table 4: `ChutoroError` variants and their stable codes._

The enum is `#[non_exhaustive]`, so a `match` needs a wildcard arm. When the
failure came from the data source, `data_source_code()` recovers the underlying
`DataSourceErrorCode` without matching on the variant:

<!-- tested-example: guide-error-codes -->

```rust
use chutoro_core::{ChutoroBuilder, ChutoroError, DataSource, DataSourceError};

/// A source whose second item cannot be read.
struct Broken;

impl DataSource for Broken {
    fn len(&self) -> usize { 4 }
    fn name(&self) -> &str { "broken" }
    fn distance(&self, _i: usize, j: usize) -> Result<f32, DataSourceError> {
        if j == 1 {
            return Err(DataSourceError::OutOfBounds { index: j });
        }
        Ok(1.0)
    }
}

fn main() -> Result<(), ChutoroError> {
    let error = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .build()?
        .run(&Broken)
        .expect_err("item 1 cannot be read");

    assert_eq!(error.code().as_str(), "CHUTORO_DATA_SOURCE_FAILURE");
    let cause = error.data_source_code().map(|code| code.as_str());
    assert_eq!(cause, Some("DATA_SOURCE_OUT_OF_BOUNDS"));
    Ok(())
}
```

`DataSourceError` codes are `DATA_SOURCE_OUT_OF_BOUNDS`,
`DATA_SOURCE_OUTPUT_LENGTH_MISMATCH`, `DATA_SOURCE_DIMENSION_MISMATCH`,
`DATA_SOURCE_EMPTY`, and `DATA_SOURCE_ZERO_DIMENSION`. A `DataSource`
implementation should return these errors unchanged rather than wrapping them,
so callers keep the stable codes.

`ChutoroError`, `DataSourceError`, and their code enums are always available.
The subsystem errors described next require the default `cpu` feature.

### Subsystem error codes

The CPU backend is built from three stages, and each reports a typed error with
its own stable code: `HnswError` from the `CpuHnsw` index, `MstError` from
`parallel_kruskal`, and `HierarchyError` from `extract_labels_from_mst`. A
direct caller of those APIs, described under
[Advanced building blocks](#advanced-building-blocks), receives these errors as
they are. `code().as_str()` returns the stable string listed in Tables 5, 6,
and 7; the error's own `Display` text, such as
`edge references node 3, but node_count is 3`, is for people and may change.

`run` and sessions do not pass the typed errors on. They wrap each one in
`CpuHnswFailure`, `CpuMstFailure`, or `CpuHierarchyFailure`, whose two fields
hold the stage's code string (`code`) and its message (`message`). The outer
`ChutoroErrorCode` names the stage, such as `CHUTORO_CPU_MST_FAILURE`, and the
inner `code` field names the cause, such as `INVALID_NODE_ID`. The `error_code`
metric label described in [Observe a run](#observe-a-run) carries the outer
code. One exception: an `HnswError::DataSource` becomes
`ChutoroError::DataSource`, so `data_source_code()` still reaches the underlying
`DataSourceErrorCode`.

Match on the code string rather than the message. Every code enum is
`#[non_exhaustive]` and may gain variants, so keep a catch-all arm and treat an
unknown code as an unknown failure. Each enum also implements `Display`, which
renders the same string as `as_str()`, and lists its variants in an `ALL`
slice. The tables classify each code in one of four ways:

- **Caller input**: an argument or configuration was invalid.
- **Data-source output**: the `DataSource` returned an unusable value.
- **Runtime fault**: an earlier panic, usually elsewhere, poisoned a lock.
- **Internal fault**: a Chutoro invariant failed; please report it as a bug.

#### HNSW errors

`CpuHnsw::build`, `build_with_edges`, `insert`, `insert_harvesting`, and
`search` return `HnswError`.

| `as_str()` value            | Variant                   | Raised when                                                                                                   | Classification                    |
| --------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `EMPTY_BUILD`               | `EmptyBuild`              | `build` or `build_with_edges` was given a source with no items.                                               | Caller input                      |
| `INVALID_PARAMETERS`        | `InvalidParameters`       | Invalid HNSW parameters or capacity; a batch response of the wrong length; or inconsistent cache bookkeeping. | Caller input, data source, or bug |
| `DUPLICATE_NODE`            | `DuplicateNode`           | The same item was inserted twice.                                                                             | Caller input                      |
| `GRAPH_EMPTY`               | `GraphEmpty`              | `search` ran before the first insertion into a `with_capacity` index.                                         | Caller input                      |
| `GRAPH_INVARIANT_VIOLATION` | `GraphInvariantViolation` | The graph is inconsistent, including a harvested edge with no core distance.                                  | Internal fault                    |
| `NON_FINITE_DISTANCE`       | `NonFiniteDistance`       | The source returned NaN or infinity.                                                                          | Data-source output                |
| `LOCK_POISONED`             | `LockPoisoned`            | A panic poisoned the graph, insertion, or random-number lock.                                                 | Runtime fault                     |
| `DATA_SOURCE`               | `DataSource`              | `distance`, `distance_batch`, or `batch_distances` failed.                                                    | Data-source output                |

_Table 5: `HnswError` codes._

`INVALID_PARAMETERS` covers three faults. Bad configuration is the caller's to
fix. A batch response whose length differs from the candidates requested comes
from the `DataSource`, so fix the provider. Inconsistent cache or trimming
bookkeeping should never happen; report it as a bug. In `run`, an empty source
is rejected as `EmptySource` before the index is built, so `EMPTY_BUILD` only
reaches direct callers.

#### MST errors

`parallel_kruskal` returns `MstError`.

| `as_str()` value      | Variant              | Raised when                                           | Classification |
| --------------------- | -------------------- | ----------------------------------------------------- | -------------- |
| `EMPTY_GRAPH`         | `EmptyGraph`         | `node_count` is zero.                                 | Caller input   |
| `INVALID_NODE_ID`     | `InvalidNodeId`      | An edge endpoint is not below `node_count`.           | Caller input   |
| `NON_FINITE_WEIGHT`   | `NonFiniteWeight`    | An edge weight is NaN or infinite.                    | Caller input   |
| `LOCK_POISONED`       | `LockPoisoned`       | A panic poisoned the union-find lock.                 | Runtime fault  |
| `INVARIANT_VIOLATION` | `InvariantViolation` | An internal union-find index fell outside its tables. | Internal fault |

_Table 6: `MstError` codes._

Validation is deterministic. `EMPTY_GRAPH` is reported before any edge is read;
edges are then checked in input order, source endpoint before target endpoint
before weight, and the first invalid edge is the one reported. Self-edges are
ignored, finite negative weights are accepted, and a disconnected graph yields
a spanning forest rather than an error. In `run`, earlier checks pre-empt most
of these: a non-finite distance surfaces as the HNSW `NON_FINITE_DISTANCE`
instead.

#### Hierarchy errors

`extract_labels_from_mst` returns `HierarchyError`.

| `as_str()` value             | Variant                   | Raised when                                               | Classification |
| ---------------------------- | ------------------------- | --------------------------------------------------------- | -------------- |
| `EMPTY_DATASET`              | `EmptyDataset`            | `node_count` is zero.                                     | Caller input   |
| `MIN_CLUSTER_SIZE_TOO_LARGE` | `MinClusterSizeTooLarge`  | `min_cluster_size` is larger than `node_count`.           | Caller input   |
| `INVALID_EDGE_WEIGHT`        | `InvalidEdgeWeight`       | An edge weight is negative or non-finite.                 | Caller input   |
| `INVALID_EDGE_ENDPOINT`      | `InvalidEdgeEndpoint`     | An edge endpoint lies outside the dataset.                | Caller input   |
| `INVALID_FOREST_REFERENCE`   | `InvalidForestReference`  | The linkage forest refers to a missing node.              | Internal fault |
| `INVALID_CLUSTER_REFERENCE`  | `InvalidClusterReference` | The condensed tree refers to a missing cluster.           | Internal fault |
| `INVALID_POINT_REFERENCE`    | `InvalidPointReference`   | The condensed tree refers to a point outside the dataset. | Internal fault |

_Table 7: `HierarchyError` codes._

Extraction checks the dataset size, then `min_cluster_size`, then each edge in
input order, endpoints before weight. Both `0.0` and `-0.0` are accepted
weights. The extractor does not check that the edges really form a minimum
spanning forest.

## Tuning and operations

### Choose an execution strategy

`ExecutionStrategy` selects the backend. `Auto`, the default, and `CpuOnly`
both run the CPU backend. A GPU backend is planned; until it exists,
`GpuPreferred` is rejected:

- without the `gpu` feature, `build` returns `BackendUnavailable`;
- with the `gpu` feature, `build` succeeds and `run` returns
  `BackendUnavailable`.

Incremental sessions are CPU-only, so `build_session` always rejects
`GpuPreferred`.

### Select feature flags

`chutoro-core` has three feature flags, listed in Table 8.

| Feature   | Default | Effect                                                                  |
| --------- | ------- | ----------------------------------------------------------------------- |
| `cpu`     | Yes     | The CPU backend, `CpuHnsw`, sessions, and the MST and hierarchy stages. |
| `metrics` | No      | Emits the metrics listed in [Observe a run](#observe-a-run).            |
| `gpu`     | No      | Lets `GpuPreferred` pass `build`; no GPU backend exists yet.            |

_Table 8: `chutoro-core` feature flags._

A crate that only implements `DataSource`, as the provider crates do, can turn
off default features and avoid compiling the CPU backend.

### Tune the HNSW index

Chutoro finds neighbours with a Hierarchical Navigable Small World (HNSW) index.
`HnswParams::new(max_connections, ef_construction)` sets its two main
trade-offs, and `with_hnsw_params` applies them to both `run` and sessions:

- `max_connections` (default 16) is how many neighbours each item links to.
  Higher values improve recall on high-dimensional data and cost memory.
- `ef_construction` (default 64) is how widely each insertion searches. Higher
  values give a better graph and take longer to build. It must be at least
  `max_connections`.

`HnswParams::new` rejects a zero `max_connections` and an `ef_construction`
below `max_connections` with an `HnswError`. The index is randomized with a
fixed default seed; `with_rng_seed` chooses another.
`with_distance_cache_max_entries` and `with_distance_cache_ttl` bound the
distance cache, which holds up to 1,048,576 entries by default.

### Bound memory use

`with_max_bytes` sets a memory limit for `run`. Before allocating anything,
Chutoro estimates the run's peak memory and returns
`ChutoroError::MemoryLimitExceeded` when the estimate exceeds the limit. Without
`with_max_bytes`, no limit applies.
`estimate_peak_bytes_for_hnsw_params_with_concurrency` returns the same
estimate that `run` checks, so a limit can be checked or sized in advance:

<!-- tested-example: guide-memory-limit -->

```rust
use std::num::NonZeroUsize;

use chutoro_core::{
    ChutoroBuilder, ChutoroError, HnswParams, estimate_peak_bytes_for_hnsw_params_with_concurrency, format_bytes,
};
use chutoro_providers_text::TextProvider;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lines = ["pear", "pears", "peer", "spear", "melon", "melons", "lemon", "felon"];
    let fruit = TextProvider::new("fruit", lines.map(String::from).to_vec())?;
    let params = HnswParams::new(32, 128)?;
    let min_cluster_size = NonZeroUsize::new(2).ok_or("the size must be non-zero")?;
    let threads = NonZeroUsize::new(rayon::current_num_threads()).ok_or("the pool has threads")?;

    let estimate = estimate_peak_bytes_for_hnsw_params_with_concurrency(8, min_cluster_size, &params, threads);
    println!("estimated peak memory: {}", format_bytes(estimate));

    let roomy = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .with_hnsw_params(params.clone())
        .with_max_bytes(estimate)
        .build()?;
    assert_eq!(roomy.run(&fruit)?.cluster_count(), 2);

    let cramped = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .with_hnsw_params(params)
        .with_max_bytes(estimate - 1)
        .build()?;
    let error = cramped.run(&fruit).expect_err("one byte below the estimate");
    assert!(matches!(error, ChutoroError::MemoryLimitExceeded { .. }));
    Ok(())
}
```

The estimate grows with the number of items, `max_connections`, the effective
construction width, and the distance-cache capacity. The default cache alone
accounts for roughly 120 MiB, so a limit below that rejects even tiny runs
unless the cache is made smaller with `with_distance_cache_max_entries`.

Two further terms make the estimate deliberately conservative:

- **The candidate-edge harvest spans several index layers.** The estimate
  multiplies the construction width by the expected number of layers, which
  follows the level multiplier and is capped at `max_level + 1`. It therefore
  grows with `ef_construction` rather than with `max_connections` alone.
- **Every Rayon thread holds its own search state.** The core-distance stage
  searches several items at once, as described in
  [Control parallelism](#control-parallelism). Each state is sized to
  `batch_search_width`: the larger of `min_cluster_size + 1` and the
  construction width, capped at the item count. Raising `min_cluster_size`, or
  running on a wider pool, raises the estimate.

The example passes the global pool's width, because `run` checks the limit
against the pool it is called from. Inside an explicit pool, pass that pool's
width instead. `estimate_peak_bytes_for_hnsw_params(point_count, &params)`
gives the single-search figure, and
`estimate_peak_bytes(point_count, max_connections)` gives a quick estimate with
default settings for everything else; both leave out the per-thread search
states.

### Control parallelism

`run` builds the index, searches for every item's neighbours, and builds the
minimum spanning tree on the Rayon thread pool it is called from. Set
`RAYON_NUM_THREADS`, or call `run` inside a `rayon::ThreadPoolBuilder` pool, to
choose the width. The thread count does not change the result: core distances
keep item order, and when several searches fail, the error reported is the one
for the lowest item index. A failing run may still search every item before it
reports the error. Incremental sessions search one item at a time.

### Observe a run

Chutoro logs through `tracing` and, with the `metrics` feature, records through
the `metrics` facade. As a library it never installs a subscriber or recorder;
the application chooses where diagnostics go. Batch metrics and batch decision
tracing never include source names, source paths, or item data.

Each `run` increments the counter `chutoro.batch.runs_total` once. Its labels
are `backend` (`cpu`, `gpu`, or `unavailable`), `outcome` (`success` or
`error`), and `error_code` (`none`, or a code from Table 4). Runs that pass the
source checks also record the histograms in Table 9, labelled only by `backend`.

| Metric                                    | Unit    | Records                                                     |
| ----------------------------------------- | ------- | ----------------------------------------------------------- |
| `chutoro.batch.max_connections`           | Count   | The configured `max_connections`.                           |
| `chutoro.batch.effective_ef_construction` | Count   | The construction width after bounding it by the item count. |
| `chutoro.batch.estimated_bytes`           | Bytes   | The peak-memory estimate.                                   |
| `chutoro.batch.memory_limit_bytes`        | Bytes   | The `with_max_bytes` limit, when one is set.                |
| `chutoro.cpu_pipeline.stage_seconds`      | Seconds | Stage duration, labelled `stage` (see below).               |

_Table 9: Batch histograms._

Byte histograms are exact up to 8 PiB. `chutoro.cpu_pipeline.stage_seconds`
records one sample for each CPU stage that succeeds, with the `stage` label set
to `core_distances`, `mutual_reachability`, or `kruskal`. A stage that fails
records nothing.

The HNSW index and its distance cache add the metrics in Table 10.

| Metric                                            | Kind      | Records                                                                |
| ------------------------------------------------- | --------- | ---------------------------------------------------------------------- |
| `distance_cache_hits`                             | Counter   | Cache lookups answered from the cache.                                 |
| `distance_cache_misses`                           | Counter   | Lookups that had to compute the distance, including expired entries.   |
| `distance_cache_evictions`                        | Counter   | Entries removed by the capacity limit or by expiry.                    |
| `distance_cache_lookup_latency_histogram`         | Histogram | Lookup latency in seconds, for hits and completed misses.              |
| `chutoro.hnsw.reconciliation.healed_nodes_total`  | Counter   | Items re-linked to the entry point after losing every base-layer link. |
| `chutoro.hnsw.reconciliation.orphan_scrubs_total` | Counter   | Removed orphaned edges, labelled `layer` = `base` or `upper`.          |
| `chutoro.hnsw.trim.jobs_submitted_total`          | Counter   | Neighbour-list trims scored during insertion, including failed ones.   |
| `chutoro.hnsw.trim.duration_seconds`              | Histogram | Seconds to score one non-empty batch of trims, including failed ones.  |

_Table 10: HNSW and distance-cache metrics._

Incremental sessions record their own metrics, listed in
[Use incremental sessions](#use-incremental-sessions).

## Advanced building blocks

### Use incremental sessions

`ClusteringSession` is a preview of incremental clustering: it keeps a live
HNSW index that grows as items are appended. Today a session maintains the
index and each item's core distance, but it does **not** yet produce cluster
labels. Refreshing labels from a session is the next step on the roadmap (item
11.2). Use `Chutoro::run` when labels are needed now.

Build a session from a shared source with `build_session`:

<!-- tested-example: guide-session -->

```rust
use std::sync::Arc;

use chutoro_core::{ChutoroBuilder, ChutoroError, SessionRefreshPolicy};
use chutoro_providers_text::TextProvider;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lines = ["pear", "pears", "peer", "spear", "melon", "melons", "lemon", "felon"];
    let fruit = Arc::new(TextProvider::new("fruit", lines.map(String::from).to_vec())?);

    let mut session = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .with_session_refresh_policy(SessionRefreshPolicy::manual())
        .build_session(Arc::clone(&fruit))?;
    assert_eq!(session.point_count(), 0);

    session.append(&[0, 1, 2, 3])?;
    assert_eq!(session.core_distance(0), None, "new items wait for a recompute");

    session.recompute_core_distances()?;
    assert!(session.core_distance(0).is_some());

    let error = session.append(&[99]).expect_err("index 99 is not in the source");
    assert!(matches!(error, ChutoroError::DataSource { .. }));
    assert_eq!(session.point_count(), 4);
    Ok(())
}
```

`build_session` takes the source as `Arc<D>`, where
`D: DataSource + Send + Sync`. It does not insert anything: `append` adds items
by their index in the source, and the session never copies or extends the
source's storage. An empty or undersized source is accepted at construction.

`append` stops at the first failure but keeps everything inserted before it. An
index beyond the source returns `ChutoroError::DataSource`. A duplicate index,
or one that was not yet in the source when the session was built, returns
`ChutoroError::CpuHnswFailure`, because the session sizes its index from
`source.len()` at construction. Size the source before building the session.

After `append`, the new items' core distances read as `None` until
`recompute_core_distances()` runs. That call also refreshes existing items that
the new ones landed near. Treat the values as provisional until each item has
at least `min_cluster_size` other items in the session.
`recompute_core_distances_full()` recomputes every item, matching what a fresh
batch run would compute, at a higher cost.

Further limits apply to this preview:

- Sessions grow by appending; items cannot be removed or changed in place.
- `SessionRefreshPolicy::with_refresh_every_n` stores an append threshold, but
  no refresh runs yet, so it has no effect.
- `snapshot_version()` stays at 0 until refresh exists.
- `with_max_bytes` does not apply to sessions.
- Cluster identities will not be stable across refreshes until roadmap item
  12.3.1 lands.

With the `metrics` feature, sessions record the metrics in Table 11.

| Metric                                                         | Kind      | Records                                                                  |
| -------------------------------------------------------------- | --------- | ------------------------------------------------------------------------ |
| `chutoro.session.append.errors_total`                          | Counter   | Append failures, labelled `reason` = `out_of_bounds` or `hnsw_failure`.  |
| `chutoro.session.append.point_seconds`                         | Histogram | Seconds to insert one item.                                              |
| `chutoro.session.harvested_edges`                              | Counter   | Candidate edges buffered for a future refresh.                           |
| `chutoro.session.core_distance.queries_total`                  | Counter   | Index searches made while recomputing core distances.                    |
| `chutoro.session.core_distance.recomputed_existing`            | Counter   | Existing items refreshed because new items landed near them.             |
| `chutoro.session.core_distance.appends_left_dirty_total`       | Counter   | Recomputes that found items still waiting for a core distance.           |
| `chutoro.session.core_distance.errors_total`                   | Counter   | Failures, labelled `reason` = `data_source`, `hnsw_failure`, or `other`. |
| `chutoro.session.core_distance.touched_existing_per_recompute` | Histogram | Existing items touched by one incremental recompute.                     |
| `chutoro.session.core_distance.recompute_seconds`              | Histogram | Seconds spent in one recompute.                                          |

_Table 11: Session metrics._

### Work with the HNSW index directly

`CpuHnsw` is the approximate nearest-neighbour index that `run` builds
internally. Use it directly to answer neighbour queries or to consume the
candidate edges found while the index is built:

<!-- tested-example: guide-cpu-hnsw -->

```rust
use std::num::NonZeroUsize;

use chutoro_core::{CpuHnsw, HnswParams};
use chutoro_providers_text::TextProvider;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lines = ["pear", "pears", "peer", "spear", "melon", "melons", "lemon", "felon"];
    let fruit = TextProvider::new("fruit", lines.map(String::from).to_vec())?;

    let (index, harvest) = CpuHnsw::build_with_edges(&fruit, HnswParams::default())?;
    assert_eq!(index.len(), 8);
    assert!(!harvest.is_empty());

    let ef = NonZeroUsize::new(3).ok_or("ef must be non-zero")?;
    let neighbours = index.search(&fruit, 4, ef)?;
    let ids: Vec<usize> = neighbours.iter().map(|neighbour| neighbour.id).collect();
    assert_eq!(ids.first(), Some(&4), "an item is its own nearest neighbour");
    assert!(ids.len() <= 3);
    Ok(())
}
```

`CpuHnsw::build` builds the same index without collecting edges.
`CpuHnsw::with_capacity(params, capacity)` creates an empty index that `insert`
or `insert_harvesting` fill one item at a time; item indices must be below the
capacity. `search(source, query, ef)` returns up to `ef` neighbours of a
`query` item, closest first. The query item is always included, because every
item is its own nearest neighbour.

Insertion keeps every link reciprocal and re-links an item that loses its last
base-layer neighbour, so no inserted item becomes unreachable. Search is still
approximate: a larger `ef` trades time for recall. Bounded Kani proofs and
property tests check these invariants, and `index.invariants()` exposes the
same checks for use in an application's own tests.

`EdgeHarvest` exposes the candidate edges through `as_slice`, `iter`, and
`into_inner`. `as_slice` suits a parallel consumer, because it lends the whole
list without first collecting references. The first `insert_harvesting` call on
an empty index returns no edges, because there is nothing to connect to yet.
`CandidateEdge::canonicalize()` orders an edge's two endpoints so that the same
undirected edge always compares equal; the older spelling `canonicalise()`
remains as an alias.

`HnswError` reports index failures, each with a stable `code()`.
`DuplicateNode` marks an item inserted twice, `DataSource` wraps a failed
`distance`, `distance_batch`, or `batch_distances` call, and `GraphEmpty` marks
a search before the first insertion. Table 5, under
[Subsystem error codes](#subsystem-error-codes), lists every code.

### Assemble the pipeline yourself

`run` chains the index, a minimum spanning tree (MST), and cluster extraction.
The last two stages are public, for applications that build their own edge
lists:

<!-- tested-example: guide-mst-hierarchy -->

```rust
use std::num::NonZeroUsize;

use chutoro_core::{
    CandidateEdge, EdgeHarvest, HierarchyConfig, extract_labels_from_mst, parallel_kruskal,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Two tight triangles, {0, 1, 2} and {3, 4, 5}, joined by one long edge.
    let edges = EdgeHarvest::from_unsorted(vec![
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 2, 1.0, 1),
        CandidateEdge::new(0, 2, 1.0, 2),
        CandidateEdge::new(3, 4, 1.0, 3),
        CandidateEdge::new(4, 5, 1.0, 4),
        CandidateEdge::new(3, 5, 1.0, 5),
        CandidateEdge::new(2, 3, 9.0, 6),
    ]);

    let forest = parallel_kruskal(6, &edges)?;
    assert!(forest.is_tree());

    let min_cluster_size = NonZeroUsize::new(2).ok_or("size must be non-zero")?;
    let labels = extract_labels_from_mst(6, forest.edges(), HierarchyConfig::new(min_cluster_size))?;
    assert_eq!(labels, [0, 0, 0, 1, 1, 1]);
    Ok(())
}
```

Edge weights passed to `extract_labels_from_mst` should already be
mutual-reachability distances: the larger of the two items' core distances and
the distance between them. `run` computes these from the HNSW index; an
application assembling the stages itself is responsible for them.
`parallel_kruskal` returns a `MinimumSpanningForest`, whose `component_count()`
reports how many disconnected parts the edges left. Failures are reported as
`MstError` and `HierarchyError`, each with a stable `code()`, such as
`EMPTY_DATASET` for an extraction over no items; Tables 6 and 7 list them all.

### Compute distances directly

`euclidean_distance` and `cosine_distance` compute a distance between two `f32`
slices and return the `Distance` newtype, which dereferences to `f32`. Both
reject empty, mismatched, and non-finite input with a `DistanceError`, and
cosine distance also rejects a zero-length vector:

<!-- tested-example: guide-distance-helpers -->

```rust
use chutoro_core::{CosineNorms, DistanceError, cosine_distance, euclidean_distance};

fn main() -> Result<(), DistanceError> {
    let left = [3.0, 4.0];
    let right = [0.0, 0.0];
    assert_eq!(euclidean_distance(&left, &right)?.value(), 5.0);

    let x = [1.0, 0.0];
    let y = [0.0, 2.0];
    let norms = CosineNorms::from_vectors(&x, &y)?;
    assert_eq!(*cosine_distance(&x, &y, Some(norms))?, 1.0);

    assert!(cosine_distance(&x, &right, None).is_err(), "zero-length vectors have no direction");
    Ok(())
}
```

`CosineNorms` caches the two vectors' lengths so repeated comparisons skip
their square roots. Chutoro checks that cached norms are finite and positive,
but not that they belong to the vectors passed in: reuse a `CosineNorms` only
with the vectors it was built from.

`core_distance_from_neighbours` applies the core-distance rule from
[Choose `min_cluster_size`](#choose-min_cluster_size) to a neighbour list that
an application has found itself. Pass the item's neighbours closest first,
without the item itself. It returns the distance of the `min_cluster_size`-th
neighbour, the farthest one when the list is shorter, and `0.0` when the list
is empty:

<!-- tested-example: guide-core-distance -->

```rust
use std::num::NonZeroUsize;

use chutoro_core::{Neighbour, core_distance_from_neighbours};

fn main() -> Result<(), &'static str> {
    let neighbours = [
        Neighbour { id: 1, distance: 0.5 },
        Neighbour { id: 2, distance: 1.5 },
        Neighbour { id: 3, distance: 2.5 },
    ];

    let two = NonZeroUsize::new(2).ok_or("two is non-zero")?;
    assert_eq!(core_distance_from_neighbours(&neighbours, two), 1.5);

    let five = NonZeroUsize::new(5).ok_or("five is non-zero")?;
    assert_eq!(core_distance_from_neighbours(&neighbours, five), 2.5, "a short list uses its farthest entry");
    Ok(())
}
```

`run`, incremental sessions, and the benchmarks all use this function, so an
application that searches `CpuHnsw` itself computes the same core distances.

### Benchmarks and dataset tooling

The Criterion benchmarks and the benchmark dataset recipes, including their
checksums, cache keys, and staging locks, are contributor tooling and are not
published. The developers' guide covers
[running the benchmarks](developers-guide.md#benchmarks) and
[writing dataset recipes](developers-guide.md#benchmark-dataset-recipes).
