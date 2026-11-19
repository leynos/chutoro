# Chutoro user guide

Integration of the `chutoro-core` crate into Rust applications is described
below. The guide focuses on the APIs exposed to downstream crates and the
behaviour of the CPU FISHDBC implementation.

## Audience and scope

The guide targets developers embedding Chutoro inside their services. It
assumes familiarity with Rust, asynchronous job runners, and dependency
management via Cargo. The section on the
[`chutoro-cli` binary](#running-the-chutoro-cli-binary) also applies to anyone
running the clustering pipeline from the command line.

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

## Reusing dense fixtures in dependent tests

Crate tests that need shared Arrow and Parquet inputs can enable the dense
provider's opt-in `testing` feature from a development dependency:

```toml
[dev-dependencies]
chutoro-providers-dense = { version = "0.1.0", features = ["testing"] }
```

The feature exposes builders for fixed-size list arrays, feature fields, and
in-memory or file-backed Parquet data. It is disabled by default; keeping this
feature isolated from normal dependencies requires Cargo resolver 2 or later.
See the [testing module API](../chutoro-providers/dense/src/testing/mod.rs) for
the builder surface and the
[developer's guide](developers-guide.md#support-module-boundaries) for its
ownership boundary.

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

On CPU builds, `with_hnsw_params` configures both one-shot batch runs and
incremental sessions. For a batch run, `max_connections` and `ef_construction`
are passed to CPU HNSW construction. The configured `max_connections` and the
effective, dataset-bounded construction-search width derived from
`ef_construction` are included when estimating the run's peak memory for a
limit set with `with_max_bytes`. The estimate counts the multi-layer
candidate-edge harvest using the expected layer count, so it grows with
`ef_construction` rather than with `max_connections` alone. That layer count is
an estimate of the expected harvest, not a guarantee that every permitted layer
is covered. The expected layer count follows the configured level multiplier.
The default multiplier's uncapped expectation is the geometric mean
`M / (M − 1)`; the estimate uses the finite sum capped at `max_level + 1`, so a
low configured level cap lowers it below that mean. `M = 1` reaches the level
cap because its default multiplier is unbounded, and a multiplier large enough
to stop no node also counts every permitted layer, up to `max_level + 1`:

```rust
use chutoro_core::{ChutoroBuilder, HnswParams};

let chutoro = ChutoroBuilder::new()
    .with_min_cluster_size(2)
    .with_hnsw_params(HnswParams::new(32, 128)?)
    .with_max_bytes(1_073_741_824)
    .build()?;
let result = chutoro.run(&Dummy(vec![1.0, 2.0, 4.0, 8.0]))?;
# Ok::<(), chutoro_core::ChutoroError>(())
```

When the estimate exceeds the configured limit, `run` returns
`ChutoroError::MemoryLimitExceeded` before allocating the pipeline. Omitting
`with_max_bytes` leaves this guard disabled.

When the `metrics` feature is enabled, one-shot runs emit bounded batch
metrics. The `chutoro.batch.runs_total` counter has the labels `backend`,
`outcome`, and `error_code`. `backend` is one of `cpu`, `gpu`, or `unavailable`;
`outcome` is `success` or `error`; and `error_code` is `none` for successful
runs or the stable `ChutoroErrorCode` string for failures. The resource
histograms use only the bounded `backend` label:

- `chutoro.batch.max_connections` (Count) records the configured HNSW
  connection width.
- `chutoro.batch.effective_ef_construction` (Count) records the dataset-
  bounded HNSW construction-search width.
- `chutoro.batch.estimated_bytes` (Bytes) records the estimated peak memory.
- `chutoro.batch.memory_limit_bytes` (Bytes) records the configured memory
  limit when one is present.

Both byte histograms record the exact byte count up to 2^53 bytes (8 PiB), so
an 8 GiB limit and a 64 GiB limit stay distinguishable; larger values saturate
at 2^53.

The stable `ChutoroErrorCode` vocabulary includes
`CHUTORO_INVALID_MIN_CLUSTER_SIZE`, `CHUTORO_EMPTY_SOURCE`,
`CHUTORO_INSUFFICIENT_ITEMS`, `CHUTORO_BACKEND_UNAVAILABLE`,
`CHUTORO_DATA_SOURCE_FAILURE`, `CHUTORO_CPU_HNSW_FAILURE`,
`CHUTORO_CPU_MST_FAILURE`, `CHUTORO_CPU_HIERARCHY_FAILURE`, and
`CHUTORO_MEMORY_LIMIT_EXCEEDED`. The `CHUTORO_INVALID_MIN_CLUSTER_SIZE` code is
reported during builder validation; the remaining applicable codes can be
reported as `run` outcomes. Neither these metrics nor the batch decision-point
tracing includes source names, source paths, or source payload data.

The public `run_cpu_pipeline` entry point has been removed. Migrate callers to
`ChutoroBuilder::build()?.run(&source)`, which applies the same validated
configuration and run preconditions as the supported batch API.

`ExecutionStrategy::Auto` runs the CPU backend. The `gpu` feature prepares the
orchestration surface for a future accelerator backend; requesting
`ExecutionStrategy::GpuPreferred` currently yields `BackendUnavailable`.

## Running the `chutoro-cli` binary

The `chutoro-cli` crate builds a binary of the same name for clustering data
held in a file. It exposes one `run` command with two data-source variants:

```sh
chutoro-cli run text words.txt
chutoro-cli run parquet features.parquet --column embedding
```

Both variants accept `--min-cluster-size`, which defaults to `5`, and `--name`,
which overrides the data-source name reported in the summary and diagnostics.
When `--name` is omitted the name is derived from the file name.

Results are written to `stdout` as human-readable text:

```text
data source: <name>
clusters: <count>
<index>	<cluster-id>
```

### Quiet diagnostics by default

Diagnostics are written to `stderr` through `tracing` and default to the `warn`
level, so a successful run prints only its summary. Span lifecycle events are
logged at `info`, so `RUST_LOG=info` restores the verbose trace:

```sh
RUST_LOG=info chutoro-cli run text words.txt
```

`RUST_LOG` accepts any `tracing-subscriber` filter directive. A malformed
directive falls back to the `warn` default rather than failing the run. Set
`CHUTORO_LOG_FORMAT=json` to receive the same diagnostics as structured JSON
records.

### Choosing a text metric

`--metric` accepts `levenshtein` and defaults to it, because it is the only
supported value. Naming it explicitly remains valid, so existing scripts
continue to work:

```sh
chutoro-cli run text words.txt --metric levenshtein
```

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
#     }
#     fn metric_descriptor(&self) -> MetricDescriptor {
#         MetricDescriptor::new("abs")
#     }
# }
#
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

A pair of items is expressed as `PointPair`, which carries an ordered `left` and
`right` index. `distance_batch` receives `&[PointPair]` and fills a
same-length output buffer with the distance from each pair's `left` item to its
`right` item:

```rust
use chutoro_core::{DataSource, DataSourceError, PointPair};

fn batch(source: &impl DataSource) -> Result<Vec<f32>, DataSourceError> {
    let pairs = [PointPair::new(0, 1), PointPair::new(2, 0)];
    let mut out = vec![0.0_f32; pairs.len()];
    source.distance_batch(&pairs, &mut out)?;
    Ok(out)
}
```

`batch_distances(query, candidates)` is a query-centric convenience over the
same contract: it builds one `PointPair` per candidate with the query as the
`left` endpoint, so the distance from the query to each candidate is returned
in candidate order.

The default `distance_batch` helper uses `distance` to fill an output buffer
and keeps it unchanged if any pair fails. Override when the backend can compute
batches more efficiently. An override must not reorder the endpoints of a pair:
implementations that call a kernel which expects a bare `(left, right)` tuple
convert each `PointPair` at that boundary.

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

An `EdgeHarvest` exposes its edges through `as_slice`, `iter`, `into_iter`, and
`into_inner`. Use `as_slice` when a consumer can work from a contiguous
`&[CandidateEdge]`: it borrows the harvested edges directly, so a parallel
transform can split the list across threads without first collecting an
intermediate vector. Use `iter` for straightforward single-threaded traversal
and `into_inner` when the harvest itself is no longer needed and ownership of
the edges should transfer.

Use `insert(node, source)` when only graph mutation is required. Use
`insert_harvesting(node, source)` when the insertion must also return the
candidate edges considered during planning. The first insertion into an empty
index returns an empty `Vec<CandidateEdge>` because there are no prior nodes to
connect to. Subsequent insertions return candidate edges that can be consumed
by incremental minimum spanning tree (MST) or auditing workflows. Call
`CandidateEdge::canonicalize()` to order an undirected edge's endpoints; the
legacy `CandidateEdge::canonicalise()` name remains available for existing
callers.

`insert_harvesting` follows the same insertion rules as `insert` and returns
`Result<Vec<CandidateEdge>, HnswError>`. `HnswError::DuplicateNode` is returned
when the same node identifier is inserted twice. `HnswError::DataSource`
surfaces failures from `distance`, `distance_batch`, or `batch_distances`, and
`HnswError::NonFiniteDistance` reports a non-finite value returned by any of
them. `HnswError::InvalidParameters` covers invalid setup, including a batch
response whose length does not match the candidates requested, while
lock-related errors report inconsistent runtime state. See
[HNSW errors](#hnsw-errors) for the complete code list.

After insertion, `search(source, query, ef)` returns the `ef` nearest
neighbours currently reachable from the HNSW entry point. The query source must
implement `DataSource + Sync` so that concurrent searches can share it. For an
end-to-end example, see the Rustdoc for
`chutoro_core::CpuHnsw::insert_harvesting`.

### Insertion guarantees

Insertion maintains reciprocal links: every neighbour edge the index stores has
a matching reverse edge at the same layer, and a point whose last base-layer
neighbour is displaced by later insertions is re-linked through the entry point
rather than left unreachable. Search therefore never strands an inserted point,
and `search` results reflect every point accepted by `insert` or
`insert_harvesting`. These guarantees are verified by bounded Kani proofs and
property tests; see the [developers' guide](developers-guide.md) for the
verification policy.

## Results and assignments

`Chutoro::run` returns a `ClusteringResult`, which exposes the per-item
`assignments` and a pre-computed `cluster_count`. To construct a result
manually, use the public fallible `ClusteringResult::try_from_assignments`. It
verifies that cluster identifiers start at `0` and are contiguous, reporting
invalid input through the `NonContiguousClusterIds` enum so actionable feedback
can be surfaced upstream. The panicking `from_assignments` convenience is
internal to `chutoro-core`, where the CPU pipeline guarantees the precondition.

Each assignment stores a `ClusterId`. The underlying value can be accessed with
`get()` when serializing or displaying results.

### How clusters are selected

Cluster selection follows the design's excess-of-mass rule. A parent cluster
replaces its descendants only when its stability strictly exceeds the summed
stability of its children; an exact tie keeps the child clusters.
Integer-valued metrics, such as Levenshtein distance over short words, produce
exact ties often, so `cluster_count` can be higher for those inputs than for a
continuous metric over the same points.

For example, the words `apple`, `apply`, `apples`, `appel`, `banana`, `bananas`,
`bandana` and `banane` are at most two edits apart within each word family and
at least four edits apart between the families. Their root cluster and their
two children score the same stability, so the strict rule reports two clusters
and assigns the first four words to one cluster and the last four to the other.

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

`ChutoroError`, `ChutoroErrorCode`, `DataSourceError`, and
`DataSourceErrorCode` are always available. The subsystem error types described
below, and their code enums, require the default `cpu` feature.

### Subsystem error codes

The CPU pipeline is built from three subsystems, and each publishes a typed
error with a stable, machine-readable code: `MstError` and `MstErrorCode` from
`parallel_kruskal`; `HnswError` and `HnswErrorCode` from `CpuHnsw`
construction, insertion, and search; and `HierarchyError` and
`HierarchyErrorCode` from `extract_labels_from_mst`. These are the errors a
direct caller of those APIs receives. Each error's `code()` method returns its
code enum, whose `as_str()` returns the stable string listed in the tables
below. The error's own `Display` renders a human-readable message such as
`edge references node 3, but node_count is 3`; only `code().as_str()` and the
code enum's `Display` are machine-readable.

`Chutoro::run` does not forward these typed errors. It maps them to
`ChutoroError::CpuMstFailure`, `ChutoroError::CpuHnswFailure`, or
`ChutoroError::CpuHierarchyFailure`, each carrying only two `Arc<str>` fields:
`code`, the subsystem's `as_str()` value, and `message`, its `Display` text.
The wrapper does not retain the typed error or its structured fields, so
recover detail by matching the `code` string and treat the message as
human-readable context. `HnswError::DataSource` is the exception; it becomes
`ChutoroError::DataSource`, as described under [HNSW errors](#hnsw-errors).

The `ChutoroErrorCode` value identifies the stage, not the cause. A
`CpuMstFailure` has outer code `CHUTORO_CPU_MST_FAILURE`, a `CpuHnswFailure` has
`CHUTORO_CPU_HNSW_FAILURE`, and a `CpuHierarchyFailure` has
`CHUTORO_CPU_HIERARCHY_FAILURE`, while the inner `code` field holds the
subsystem code such as `INVALID_NODE_ID`. The batch-metrics `error_code` label
described in
[Running the clustering pipeline](#running-the-clustering-pipeline) carries the
outer `ChutoroErrorCode` string, not the inner subsystem code.

Match the inner `code` string rather than the message. The message may change
between releases, and the code enums are `#[non_exhaustive]`, so treat an
unrecognized code as an unknown failure rather than assuming the set is closed.
`ChutoroErrorCode`, `DataSourceErrorCode`, `MstErrorCode`,
`HierarchyErrorCode`, and `HnswErrorCode` are all `#[non_exhaustive]`, implement
`Display`, and expose `as_str()`; `Display` renders the same stable string as
`as_str()`, while the owning error's `code()` returns the code value. A
downstream `match` must therefore keep a catch-all arm, such as `_`; see the
[0.2.0 migration notes](migration-notes.md#error-code-enums-are-non-exhaustive-and-implement-display)
for the migration, and the
[developers' guide](developers-guide.md#generated-error-code-enums) for why the
enums are generated by `define_error_codes!`.

The classification in each table uses four categories:

- **Caller input**: the caller supplied an invalid argument or configuration.
- **Data-source output**: the `DataSource` returned an unusable value.
- **Runtime fault**: a synchronization primitive was poisoned by an earlier
  panic, usually in unrelated code.
- **Internal fault**: a Chutoro invariant failed; report it as a defect.

#### MST errors

`parallel_kruskal` returns `MstError`, and `Chutoro::run` surfaces the same
failure as `ChutoroError::CpuMstFailure`.

| Code variant         | `as_str()` value      | Error variant and fields                                        | Producing condition                                                                                            | Classification |
| -------------------- | --------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------- |
| `EmptyGraph`         | `EMPTY_GRAPH`         | `MstError::EmptyGraph`                                          | `node_count` is zero; checked before any edge is inspected                                                     | Caller input   |
| `InvalidNodeId`      | `INVALID_NODE_ID`     | `MstError::InvalidNodeId { node, node_count }`                  | An edge endpoint is greater than or equal to `node_count`; the source bound is checked before the target bound | Caller input   |
| `NonFiniteWeight`    | `NON_FINITE_WEIGHT`   | `MstError::NonFiniteWeight { left, right }`                     | An edge weight is NaN or infinite; `left` and `right` keep the original orientation                            | Caller input   |
| `LockPoisoned`       | `LOCK_POISONED`       | `MstError::LockPoisoned { resource }`                           | The `"union-find root lock"` was poisoned by an earlier panic                                                  | Runtime fault  |
| `InvariantViolation` | `INVARIANT_VIOLATION` | `MstError::InvariantViolation { invariant, index, lock_count }` | An internal union-find index lies outside its lock, parent, or rank table                                      | Internal fault |

_Table 1: MST error codes produced by `parallel_kruskal`._

Kruskal validates a graph in a fixed order. It reports `EMPTY_GRAPH` before
inspecting any edge when `node_count` is zero. It then visits edges in
`EdgeHarvest::as_slice()` order and, for each edge, checks the source bound,
the target bound, and finally weight finiteness. When several edges are
invalid, the reported error is the one at the lowest input index; `EmptyGraph`
takes precedence over every edge-level error. Self-edges are ignored after
validation, finite negative weights are accepted, and an empty edge list or a
disconnected graph returns a minimum spanning forest rather than an error.

In the `Chutoro::run` pipeline, earlier stages normally pre-empt these errors.
An empty source returns `ChutoroError::EmptySource` and a source smaller than
`min_cluster_size` returns `ChutoroError::InsufficientItems`, so `EMPTY_GRAPH`
is reachable only through the direct API. Problems detected earlier in the
pipeline surface as HNSW failures rather than MST failures: a non-finite
distance becomes HNSW `NON_FINITE_DISTANCE`, and a harvested edge with no core
distance becomes HNSW `GRAPH_INVARIANT_VIOLATION`. Kruskal still validates the
transformed mutual-reachability weights, so a non-finite value that survives
the transform reports `NON_FINITE_WEIGHT`.

#### HNSW errors

`CpuHnsw` construction, `insert`, `insert_harvesting`, and `search` return
`HnswError`. `Chutoro::run` and incremental sessions use the same conversion:
every variant except `DataSource` becomes
`ChutoroError::CpuHnswFailure { code, message }`.

| Code variant              | `as_str()` value            | Error variant and fields                         | Producing condition                                                                                                                                                                                                       | Classification                     |
| ------------------------- | --------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `EmptyBuild`              | `EMPTY_BUILD`               | `HnswError::EmptyBuild`                          | `build` or `build_with_edges` was called on a source with no items                                                                                                                                                        | Caller input                       |
| `InvalidParameters`       | `INVALID_PARAMETERS`        | `HnswError::InvalidParameters { reason }`        | Invalid configuration, such as zero `max_connections`, `ef_construction` below `max_connections`, zero capacity, or a level above `max_level`; also a batch response whose length does not match the requested candidates | Caller input or data-source output |
| `DuplicateNode`           | `DUPLICATE_NODE`            | `HnswError::DuplicateNode { node }`              | The same node identifier was inserted twice                                                                                                                                                                               | Caller input                       |
| `GraphEmpty`              | `GRAPH_EMPTY`               | `HnswError::GraphEmpty`                          | Search or planning ran on an index with no entry point, such as a `with_capacity` index before its first insertion                                                                                                        | Caller input                       |
| `GraphInvariantViolation` | `GRAPH_INVARIANT_VIOLATION` | `HnswError::GraphInvariantViolation { message }` | The graph is inconsistent, including a harvested edge that has no core distance                                                                                                                                           | Internal fault                     |
| `NonFiniteDistance`       | `NON_FINITE_DISTANCE`       | `HnswError::NonFiniteDistance { left, right }`   | The source returned NaN or infinity for a distance                                                                                                                                                                        | Data-source output                 |
| `LockPoisoned`            | `LOCK_POISONED`             | `HnswError::LockPoisoned { resource }`           | A panic poisoned the `"graph"`, `"insert mutex"`, `"rng mutex"`, or `"worker rng mutex"` lock                                                                                                                             | Runtime fault                      |
| `DataSource`              | `DATA_SOURCE`               | `HnswError::DataSource(DataSourceError)`         | A `distance`, `distance_batch`, or `batch_distances` call failed                                                                                                                                                          | Data-source output                 |

_Table 2: HNSW error codes produced by the CPU backend._

`HnswError::DataSource` is the exception: it becomes
`ChutoroError::DataSource { data_source, error }`, whose outer code is
`CHUTORO_DATA_SOURCE_FAILURE` and whose inner `DataSourceErrorCode` is
available from `data_source_code()`. `DATA_SOURCE` is therefore never the
`code` field of a `CpuHnswFailure`; it is the `HnswErrorCode` for the wrapped
data-source failure at the typed API boundary.

`INVALID_PARAMETERS` covers two distinct faults. Invalid configuration is the
caller's responsibility, while a batch response whose length does not match the
requested candidates originates in the `DataSource`; repair the provider in the
latter case.

`GRAPH_EMPTY` reports caller-visible state rather than a defect. An index
created by `CpuHnsw::with_capacity` has no entry point until its first
insertion succeeds, so `search` returns `GRAPH_EMPTY` until then; insert a node
before searching.

In `Chutoro::run`, a stable empty source returns `ChutoroError::EmptySource`
before HNSW construction, so `EMPTY_BUILD` is reachable only through the direct
`CpuHnsw::build` and `CpuHnsw::build_with_edges` APIs. Sessions convert HNSW
failures with the same rule as batch runs. When the pipeline cannot find a core
distance for a harvested edge, it reports `GRAPH_INVARIANT_VIOLATION` directly
rather than through the MST stage.

#### Hierarchy errors

`extract_labels_from_mst` returns `HierarchyError`, and `Chutoro::run` maps it
to `ChutoroError::CpuHierarchyFailure { code, message }`.

| Code variant              | `as_str()` value             | Error variant and fields                                                  | Producing condition                                                            | Classification |
| ------------------------- | ---------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------- |
| `EmptyDataset`            | `EMPTY_DATASET`              | `HierarchyError::EmptyDataset`                                            | Extraction was called with `node_count` zero                                   | Caller input   |
| `MinClusterSizeTooLarge`  | `MIN_CLUSTER_SIZE_TOO_LARGE` | `HierarchyError::MinClusterSizeTooLarge { node_count, min_cluster_size }` | `min_cluster_size` is larger than `node_count`                                 | Caller input   |
| `InvalidEdgeWeight`       | `INVALID_EDGE_WEIGHT`        | `HierarchyError::InvalidEdgeWeight { left, right, weight }`               | An edge weight is negative or non-finite; `left` and `right` are canonicalized | Caller input   |
| `InvalidEdgeEndpoint`     | `INVALID_EDGE_ENDPOINT`      | `HierarchyError::InvalidEdgeEndpoint { endpoint, node_count }`            | An MST edge endpoint lies outside the dataset                                  | Caller input   |
| `InvalidForestReference`  | `INVALID_FOREST_REFERENCE`   | `HierarchyError::InvalidForestReference { node_id }`                      | The linkage forest references a missing node                                   | Internal fault |
| `InvalidClusterReference` | `INVALID_CLUSTER_REFERENCE`  | `HierarchyError::InvalidClusterReference { cluster_id }`                  | Condensation references a missing cluster                                      | Internal fault |
| `InvalidPointReference`   | `INVALID_POINT_REFERENCE`    | `HierarchyError::InvalidPointReference { point_id, node_count }`          | Condensation references a point outside the dataset                            | Internal fault |

_Table 3: hierarchy error codes produced by `extract_labels_from_mst`._

Extraction validates in a fixed order: an empty dataset with `EMPTY_DATASET`,
then a `min_cluster_size` larger than the dataset with
`MIN_CLUSTER_SIZE_TOO_LARGE`, then each edge in input order. For each edge,
both endpoints are checked before the weight. An endpoint outside the dataset
reports `INVALID_EDGE_ENDPOINT`; a negative or non-finite weight reports
`INVALID_EDGE_WEIGHT`. The values `0.0` and `-0.0` are accepted, and the error's
`left` and `right` fields are canonicalized to the smaller and larger
endpoint. The extractor does not verify that the edges form a minimum spanning
forest.

In the `Chutoro::run` pipeline, `EmptySource` and `InsufficientItems` pre-empt
`EMPTY_DATASET` and `MIN_CLUSTER_SIZE_TOO_LARGE`. Kruskal validates endpoints
before extraction, so `INVALID_EDGE_ENDPOINT` is reachable only through the
direct API. A negative weight reaches extraction only when negative data-source
distances survive the core-distance maximum applied by the mutual-reachability
transform. The three reference codes (`INVALID_FOREST_REFERENCE`,
`INVALID_CLUSTER_REFERENCE`, and `INVALID_POINT_REFERENCE`) describe internal
hierarchy state, so report them as defects.

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

When the `metrics` feature is enabled, the distance cache emits four metrics:
`distance_cache_hits` counts successful lookups, `distance_cache_misses` counts
lookups that require computation, and `distance_cache_evictions` counts LRU
entries removed from the cache. `distance_cache_lookup_latency_histogram`
records lookup latency for both hits and completed misses.

The HNSW reconciliation path also emits two counters when the `metrics` feature
is enabled. `chutoro.hnsw.reconciliation.healed_nodes_total` counts nodes that
removed-edge reconciliation isolated at the base layer and that were re-linked
to the entry node. `chutoro.hnsw.reconciliation.orphan_scrubs_total` increments
once for each deferred scrub that removes an orphaned forward edge; its `layer`
label is `base` for base-layer scrubs and `upper` for upper-layer scrubs.

The trim-scoring path emits one counter and one histogram when the `metrics`
feature is enabled. `chutoro.hnsw.trim.jobs_submitted_total` adds the full
batch size for each non-empty batch, including a batch that fails during
scoring, and `chutoro.hnsw.trim.duration_seconds` records one wall-time sample
per non-empty batch, including failed batches, on the thread holding the
insertion mutex. Neither carries labels, so their cardinality is fixed.

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
[ADR-004: Benchmark dataset recipe trait](adrs/adr-0004-bench-dataset-recipe-trait.md).

### Identifying cache and object keys

`CacheKey` and `ObjectKey` are opaque identifiers in the recipe model. Recipes
treat them as names rather than addresses. Adapters that need a filesystem path
or object-store prefix resolve it at their own boundary.

Construct a key from a string literal or an owned `String` with `new`, and read
the identifier back with `AsRef<str>` or `Display`:

```rust
use chutoro_bench_datasets::{CacheKey, ObjectKey};

let cache = CacheKey::new("cache/mnist/raw.gz");
let object = ObjectKey::new(String::from("manifests/mnist.json"));

assert_eq!(cache.as_ref(), "cache/mnist/raw.gz");
assert_eq!(object.to_string(), "manifests/mnist.json");
```

The path-returning `to_path_buf` and `into_inner` accessors are removed. An
adapter derives its own address from the identifier, for example
`Utf8PathBuf::from(key.as_ref())`. See the
[0.2.0 migration notes](migration-notes.md#020) for the full compatibility note.

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
