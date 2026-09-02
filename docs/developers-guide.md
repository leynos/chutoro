# Developers guide

This guide collects day-to-day practices for contributors working on the
Chutoro codebase. It complements the more specialized documents in `docs/` and
keeps operational guidance in one place.

## GitHub Actions runner profiles

The bounded nightly portable-SIMD verification job uses the shared uncached
`namespace-profile-default` runner (Ubuntu 22.04, amd64, 4 vCPU, and 16 GB).
Its cache volume is disabled, so this initial migration adds no cache owner or
cache-write policy.

The property-test jobs retain `ubicloud-standard-8` because their CPU-bound
HNSW, edge-harvest, MST, and SIMD suites were sized for that runner's eight-core
capacity. The `ci` nextest profile caps test concurrency at four; the runner's
larger capacity also preserves the existing build and workload headroom.
Benchmark regressions, Kani, coverage, and the broad CI test jobs remain
GitHub-hosted until each workload has been measured against an equivalent
Namespace capacity profile. Externally owned reusable workflows retain their
existing runner selection.

The private `_job` helper in
`tests/workflow_contracts/namespace_runners_test.py` is owned by that module.
Keep its call sites there for read-only workflow YAML parsing; do not reuse it
in production code or for external workflow control. For example,
`_job("property-tests.yml", "property-tests-pr")` returns that job's mapping,
including its `runs-on` entry.

## CPU HNSW public APIs

`chutoro-core` exposes `CpuHnsw` as the public CPU-resident Hierarchical
Navigable Small World (HNSW) index. The primary insertion and query entry
points are:

```rust
pub fn build<D: DataSource + Sync>(source: &D, params: HnswParams)
    -> Result<CpuHnsw, HnswError>;
pub fn build_with_edges<D: DataSource + Sync>(source: &D, params: HnswParams)
    -> Result<(CpuHnsw, EdgeHarvest), HnswError>;
pub fn with_capacity(params: HnswParams, capacity: usize)
    -> Result<CpuHnsw, HnswError>;
pub fn insert<D: DataSource + Sync>(&self, node: usize, source: &D)
    -> Result<(), HnswError>;
pub fn insert_harvesting<D: DataSource + Sync>(&self, node: usize, source: &D)
    -> Result<Vec<CandidateEdge>, HnswError>;
pub fn search<D: DataSource + Sync>(
    &self,
    source: &D,
    query: usize,
    ef: NonZeroUsize,
) -> Result<Vec<Neighbour>, HnswError>;
```

`build` and `build_with_edges` seed the entry point from node `0` and insert
the remaining nodes in parallel. `build_with_edges` is the preferred path when
the caller needs the deterministic `EdgeHarvest` used by the MST stage.
`with_capacity` is the entry point for manual or incremental index population.

`insert` mutates the graph without allocating harvested edge storage.
`insert_harvesting` performs the same planning and commit sequence, but also
returns the candidate edges identified during the read-phase search. The first
insertion into an empty index returns an empty vector, while later insertions
return edges ordered by insertion sequence. `DuplicateNode` indicates a repeat
insertion, and other `HnswError` variants cover invalid parameters, poisoned
locks, non-finite distances, graph invariant failures, and wrapped
`DataSourceError` values.

`search` is the observable way to compare graph state after different insertion
paths. The HNSW edge-harvesting tests treat equivalent `search` results as the
behavioural contract between `insert` and `insert_harvesting`.

Design rationale and deeper implementation notes live in
[the design document](./chutoro-design.md), the completed
[edge-harvesting ExecPlan](./execplans/11-1-1-make-edge-harvesting-hnsw-insertion-path-public.md),
and the completed
[incremental core-distance ExecPlan](./execplans/11-1-4-incremental-core-distance-computation.md).

## Session public APIs

The public session surface is CPU-only. `build_session` constructs an empty
`ClusteringSession` without seeding HNSW or running the batch bootstrap path.
The architectural rationale for that split lives in
[the design document](./chutoro-design.md).

```rust
// ChutoroBuilder (cpu feature required for session APIs)
pub fn with_hnsw_params(self, params: HnswParams) -> Self;
pub fn hnsw_params(&self) -> &HnswParams;
pub fn with_session_refresh_policy(self, policy: SessionRefreshPolicy) -> Self;
pub fn session_refresh_policy(&self) -> &SessionRefreshPolicy;
pub fn build_session<D: DataSource + Send + Sync>(self, source: Arc<D>)
    -> Result<ClusteringSession<D>>;
```

```rust
// SessionRefreshPolicy
pub fn manual() -> Self;
pub fn with_refresh_every_n(self, refresh_every_n: Option<NonZeroUsize>) -> Self;
pub fn refresh_every_n(&self) -> Option<NonZeroUsize>;

// SessionConfig
pub fn min_cluster_size(&self) -> NonZeroUsize;
pub fn hnsw_params(&self) -> &HnswParams;
pub fn refresh_policy(&self) -> &SessionRefreshPolicy;

// ClusteringSession<D: DataSource + Send + Sync>
pub fn config(&self) -> &SessionConfig;
pub fn append(&mut self, indices: &[usize]) -> Result<()>;
pub fn recompute_core_distances(&mut self) -> Result<()>;
pub fn recompute_core_distances_full(&mut self) -> Result<()>;
pub fn core_distance(&self, point: usize) -> Option<f32>;
pub fn point_count(&self) -> usize;
pub fn snapshot_version(&self) -> u64;
```

`build_session` validates `min_cluster_size > 0`, rejects
`ExecutionStrategy::GpuPreferred`, accepts empty and undersized sources, and
returns an inert session whose initial observable state is `point_count() == 0`
and `snapshot_version() == 0`.

The internal `ExecutionConfig` is the single validated source for
`min_cluster_size` and CPU `HnswParams`. `ChutoroBuilder` creates it after
validation and passes it unchanged to batch `Chutoro` or `SessionConfig`.
One-shot runs use these same HNSW settings for both CPU pipeline construction
and memory estimates. It is a composition boundary only: execution and session
code may read it, but must not create or revalidate alternate copies of those
settings.

`append` inserts source indices into the live HNSW index by calling
`CpuHnsw::insert_harvesting` for each index. It must not duplicate HNSW
insertion logic or inspect private HNSW adapter internals. The session stores
all returned `CandidateEdge` values in its internal `pending_edges` buffer for
future refresh work. The method is fail-fast and preserves partial progress:
insertions completed before the first error remain in the index, and their
harvested edges remain pending.

`recompute_core_distances` computes core distances for dirty newly inserted
source indices and for existing points that appear in those new points'
non-self HNSW neighbour lists. `recompute_core_distances_full` searches every
inserted point and mirrors the batch CPU core-distance loop. The
`core_distance(i)` accessor returns `None` for cells that are dirty, unset, or
outside the source-indexed storage, and it also treats non-finite cells as
unavailable. Callers that need to distinguish an out-of-range read from a dirty
inserted point should check their own source index bookkeeping; `point_count()`
reports the number of inserted points, not a maximum source index.

Session construction allocates HNSW capacity from `source.len().max(1)` while
still leaving the index empty. `append` prevalidates each requested index
against `source.len()` before insertion so early bootstrap cases return a
`ChutoroError::DataSource` for out-of-bounds indices even when HNSW would not
need a distance query for the first inserted node.

The v1 incremental clustering surface has these limitations:

- Ingestion is append-only; deletions and updates are not supported.
- Cluster identity is not stable across snapshots, and cluster IDs may change.
- Refreshes are micro-batched rather than applied per point.
- Existing points may be relabelled after a refresh.

### Session internal architecture

Keep the session responsibility split intact when adding behaviour. Domain
state should stay on `ClusteringSession`; configuration-only changes belong
with session configuration; append construction work belongs with the append
implementation; and core-distance work should stay in the core-distance
subsystem until it grows a clearer sub-boundary.

Core-distance storage is indexed by source index, not by dense insertion
ordinal. `core_distances: Vec<f32>` stores finite values after recompute and
uses `f32::INFINITY` only as an internal unset sentinel.
`dirty_core_distances: Vec<bool>` is authoritative: `true` means the
corresponding cell is stale or never computed, and `false` means a finite cell
may be read through `core_distance(i)`. The dirty state uses `Vec<bool>`
because the workspace does not already depend on `fixedbitset`, and roadmap
item 11.1.4 forbids new production dependencies.

`ClusteringSession::append` emits tracing through `#[tracing::instrument]` and
structured `warn!`/`debug!` events, but it must not install tracing
subscribers. Library code may emit metrics and tracing; application boundaries
remain responsible for recorder and subscriber installation.

Metrics support is entirely feature-gated behind `metrics`. Production builds
without that feature must not allocate the clock field or compile metric
emission code. When `metrics` is enabled, construction describes the append
error counter, per-point latency histogram, and harvested-edge counter. The
append path records:

- `chutoro.session.append.errors_total`, labelled by low-cardinality failure
  reason.
- `chutoro.session.append.point_seconds`, one histogram sample per inserted
  point.
- `chutoro.session.harvested_edges`, counting buffered candidate edges.

Core-distance recompute helpers record:

- `chutoro.session.core_distance.queries_total`, counting HNSW searches used
  for core-distance recompute.
- `chutoro.session.core_distance.recomputed_existing`, counting existing
  points recomputed after incremental neighbour discovery.
- `chutoro.session.core_distance.appends_left_dirty_total`, counting recompute
  calls that began with dirty core distances.
- `chutoro.session.core_distance.touched_existing_per_recompute`, recording
  incremental recompute fan-out.
- `chutoro.session.core_distance.recompute_seconds`, recording recompute
  duration.

The latency histogram reads time through the internal `MonotonicClock` trait.
`StdMonotonicClock` is the production implementation. Tests may replace it via
`with_clock_for_test` with `FixedMonotonicClock`, which is available only under
`#[cfg(all(feature = "metrics", test))]`. Do not expose this seam through the
public constructor or builder API; it exists solely to make metrics assertions
deterministic while preserving the public session contract.

## Workspace lint and check-cfg policy

The seven workspace crates inherit the root `[workspace.lints]` policy through
`[lints] workspace = true`: `chutoro-core`, `chutoro-cli`,
`chutoro-providers-dense`, `chutoro-providers-text`, `chutoro-test-support`,
`chutoro-benches`, and `chutoro-bench-datasets`. Keep that inheritance rather
than duplicating lint tables in individual manifests. The root policy denies
private-item documentation debt through Clippy's
`missing_docs_in_private_items` lint, in addition to the Rustdoc `missing_docs`
policy.

The workspace also declares the configuration names used by supported builds
and verification tools: `kani`, `coverage`, `nightly`, and `dylint_lib` (with
arbitrary values for the latter). Add a new `cfg` name to the root `check-cfg`
list when introducing a supported build mode so local Rustc diagnostics and the
commit gates agree.

Whitaker filesystem exceptions are boundary-specific entries in `dylint.toml`,
not crate-wide opt-outs. The current exceptions cover the CLI's ambient input
boundary, CLI file-backed test fixtures, the dense provider's parquet-path
adapter, benchmark report writers and Linux `/proc` sampling, MNIST cache
staging, and the separately compiled CI gate binaries. Keep the rest of each
workspace crate under `no_std_fs_operations` enforcement and add an explanatory
comment whenever a new boundary is unavoidable.

## Whitaker lint suite

Whitaker is a Dylint lint suite that runs as a commit gate alongside Clippy.
`make lint` runs `lint-clippy` (rustdoc plus Clippy) followed by
`lint-whitaker`, which invokes the `whitaker` wrapper with
`RUSTFLAGS="-D warnings"` over `--all-targets --all-features`. Individual lints
are referenced elsewhere in this guide where they apply: the
[fallible fixture policy](#fallible-fixture-policy) covers
`no_expect_outside_tests`, and the
[support-module boundaries](#support-module-boundaries) section covers the
400-line `module_max_lines` cap.

### Installing Whitaker locally

Install the `whitaker` wrapper the same way continuous integration (CI) does,
by installing `whitaker-installer` and letting it place the wrapper on your
`PATH`:

```shell
cargo binstall --no-confirm --locked whitaker-installer
# or, if cargo-binstall is unavailable:
cargo install --locked whitaker-installer

whitaker-installer
```

`whitaker-installer` installs the `whitaker` wrapper itself; the Makefile
invokes it by bare name, so it must resolve on `PATH`. Override the `WHITAKER`
make variable with an explicit path if you keep the binary somewhere else:

```shell
make lint WHITAKER=/path/to/whitaker
```

If `whitaker` is unavailable, run `make lint-clippy` for a Clippy-only pass.

**Agents must not install, upgrade, or downgrade Whitaker from this repository,
and must not otherwise modify the user's Whitaker installation.** If the
wrapper is missing, ask the user to install it. See `AGENTS.md` for the full
agent-facing rule.

### CI resolution and configuration

CI resolves the newest `whitaker-installer` release at run time — via
`gh api repos/leynos/whitaker/releases/latest` — rather than pinning a version,
then installs that release and runs it to obtain the `whitaker` wrapper.
Because the suite version is not pinned, a new Whitaker release can introduce
findings on code that has not otherwise changed; treat such findings as genuine
and fix them rather than pinning around them.

Per-lint configuration, including `no_std_fs_operations` crate exclusions with
rationale comments, lives in the root `dylint.toml`.

## Test fixture conventions

The house policy is that fixtures and helpers are not tests. It governs how
test-support code reports failure, distinct from how `#[test]`, `#[rstest]`, and
`proptest!` bodies consume that failure.

### Fallible fixture policy

Test helpers, fixtures, and support functions must not panic to report their
own invariants: no `.expect(...)`, `.unwrap()`, `panic!`, or `assert!` for
conditions the helper itself is checking. They return `Result` and propagate
failure with `?`. Panicking and assertion belong at the `#[test]` / `#[rstest]`
/ `proptest!` boundary, where a failure is attributable to the test rather than
to the helper it called. Whitaker's `no_expect_outside_tests` lint enforces the
`.expect(...)` half of this rule; the rest is convention.

Test bodies unwrap the `Result` a helper returns with `.expect("...")`, keeping
the message the helper's assertion used to carry, or they return `Result`
themselves and use `?`:

```rust
#[rstest]
fn trims_the_entry_node() -> Result<(), TrimmingFixtureError> {
    let graph = build_trimming_test_graph(&params, &[1, 2], 3)?;
    verify_post_trim_reciprocity(&graph, &params, 3, 1)?;
    Ok(())
}
```

### Typed error contracts

Fixtures return a named error enum rather than `Box<dyn Error>`, so the
underlying domain error stays intact and fixture-invariant breakage is
distinguishable from a genuine failure of the code under test. Both examples
below derive their `Error` implementation with `thiserror`:

- `TrimmingFixtureError` in
  `chutoro-core/src/hnsw/insert/executor/tests/trimming_fixtures.rs` wraps
  `HnswError` transparently via `#[from]`, and adds `MissingTrimJob`,
  `ExcessTrimJobs`, `UnexpectedTrimTarget`, `MissingNode`, and
  `ReciprocityViolated` variants for the fixture's own expectations.
- `FixtureError` in `chutoro-providers/dense/src/tests/support.rs` names the
  `RowLength`, `Dimension`, `Arrow`, and `Parquet` failure modes of the Arrow
  and Parquet builders.

### Test-only budget types

`chutoro-core/src/hnsw/tests/property/test_runner_support/budget_types.rs`
holds newtypes (`TestCases`, `StackSize`) that validate proptest runner
configuration. They expose fallible `try_new` constructors that return a named
error (`InvalidTestCasesError`, `InvalidStackSizeError`) rather than panicking
constructors, so a zero or otherwise invalid budget surfaces as an error to the
calling test. `budget_selection.rs` re-exports them, so consumers can import
from either path.

### Support-module boundaries

Test modules split their fixtures into dedicated support modules when the test
file approaches Whitaker's 400-line `module_max_lines` cap, keeping the test
file focused on the behaviour under test. Current examples:

- `chutoro-core/src/session/tests/common.rs` — shared session fixtures and the
  `SessionTestSource` data source.
- `chutoro-core/src/session/tests/core_distance_support.rs` — batch-oracle
  helpers for core-distance tests.
- `chutoro-core/src/hnsw/tests/property/test_runner_support/` — proptest
  runner configuration.
- `chutoro-providers/dense/src/tests/support.rs` — Arrow and Parquet fixture
  builders.

Support modules carry `//!` module docs and `///` item docs, with `# Errors`
sections on fallible helpers.

## Continuous integration

Property-test CI jobs (`property-tests-pr` and `property-tests-weekly`) run on
`ubicloud-standard-8`, an 8-core Ubicloud runner, rather than `ubuntu-latest`.

The PR job has a `timeout-minutes: 20` budget, sized to exceed the longest
`nextest` `slow-timeout` (600 s for HNSW idempotency) so earlier setup and
property phases do not consume the full budget. The weekly job retains a
`timeout-minutes: 120` budget for deeper test runs.

All test runs use the `nextest` CI profile (`--profile ci`). Benchmark targets
require `threads-required = 8`; see `.config/nextest.toml`.

Use `.github/workflows/property-tests.yml` and `.config/nextest.toml` for the
authoritative configuration, and `docs/property-testing-design.md` for the
architectural rationale.

### Workflow pins and Dependabot

Dependabot owns the upgrade of GitHub Actions and reusable workflows, including
calls into `leynos/shared-actions`. Contract tests that assert a caller's exact
commit SHA create a lockstep dependency: every time Dependabot opens a bump PR,
the test fails until a human edits the pinned constant to match. That defeats
the purpose of automated dependency updates and turns a routine bump into a
manual chore.

Contract tests may still verify the *shape* of a reusable-workflow caller. They
must not verify the specific SHA value.

- Do assert the workflow references the correct reusable workflow path.
- Do assert the ref is pinned to a full 40-character commit SHA, not a
  mutable branch such as `main` or `rolling`.
- Do assert the expected `on:` triggers, least-privilege `permissions:`, and
  the inputs the caller relies on.
- Do not hard-code the current SHA value as an expected string. Match it with
  a pattern instead.
- Do not fail a test purely because Dependabot bumped the pinned SHA.

```python
import re

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

def test_uses_pinned_full_sha(caller_step):
    ref = caller_step["uses"].split("@")[-1]
    assert SHA_RE.match(ref), f"expected a 40-hex commit SHA, got {ref!r}"
```

If a workflow's behaviour genuinely depends on a feature only present from a
particular commit onwards, express that as a comment or a changelog note, not
as a test assertion on the SHA string.

### Arrow and Parquet family updates

Keep the root workspace's `arrow-array`, `arrow-schema`, and `parquet`
requirements on the same version. The dense provider passes Arrow's
`RecordBatchReader`, `ArrowError`, and `DataType` across the Parquet bridge, so
mixing compatible-looking major-minor releases creates distinct Rust types and
prevents the workspace from compiling.

Dependabot's `arrow` group covers version updates, while `arrow-security`
covers security updates. Both match `arrow*` and `parquet`. Do not accept a
single-package update from either category: update all three direct workspace
requirements together, regenerate `Cargo.lock`, and run the Arrow/Parquet
compile-pass test before the full quality gates.

## Dense SIMD parity suite

Dense Euclidean backend parity tests live in
`chutoro-providers/dense/src/simd/tests/parity/`. The suite compares each
compiled and runtime-supported backend against the scalar oracle defined by the
test-only `DistanceSemantics` value object.

When adding a new dense SIMD backend or Euclidean kernel:

1. Add the backend to `dispatch.rs::enabled_backends` so tests can discover it
   only when it is both compiled and runtime-supported.
2. Wire pairwise and query-to-points entry functions in the test-only helpers
   in `kernels.rs`. Keep backend implementation modules private unless a
   production caller needs wider visibility.
3. Extend `tests/parity/strategies.rs` only when the new backend has a new
   layout, lane width, or input hazard that the existing generators do not
   cover.
4. Run `cargo nextest run -p chutoro-providers-dense simd::tests::parity::`
   before the full `make check-fmt`, `make lint`, `make typecheck`, and
   `make test` gates.

If proptest records a regression, keep the generated file under the relevant
`proptest-regressions/` directory. That file is the shrunk counterexample and
should be treated as a regression guard, not as disposable local output.

## Dense SIMD Kani harnesses

Dense SIMD Kani harnesses live in
`chutoro-providers/dense/src/simd/kani_proofs.rs` and are compiled only under
`#[cfg(kani)]`. They prove boundary policy for the safe SIMD seams, not raw
architecture intrinsics.

When changing dense SIMD tail padding, lane batching, or runtime backend
selection:

1. Keep reusable arithmetic in production-used helpers, then prove those
   helpers or their immediate call boundary. Avoid proof-only arithmetic that
   can drift away from the kernels.
2. Keep selector policy in `dispatch.rs::choose_euclidean_backend`; the Kani
   harness proves every compile-time and runtime support-mask combination.
3. Use `rstest` unit tests for concrete storage behaviour, especially
   `DensePointView<'a>` alignment, 16-lane padding, and zero-filled unused
   lanes.
4. Run the practical Kani suite before requesting review:

   ```sh
   set -o pipefail
   make kani 2>&1 | tee /tmp/kani-chutoro-$(git branch --show-current).out
   ```

`make kani-full` runs every Kani harness in `chutoro-core` and
`chutoro-providers-dense`. Keep new dense harnesses small enough for
`make kani` unless they are intentionally slow-lane proofs.

## Benchmarks

The `chutoro-benches` crate provides Criterion benchmarks for the four CPU
pipeline stages: Hierarchical Navigable Small World (HNSW) index construction,
edge harvest, minimum spanning tree (MST) computation, and hierarchy extraction.

### Running benchmarks

Run all benchmarks from the repository root:

```sh
make bench
```

Criterion writes HTML reports to `target/criterion/`. Open the report for a
specific group (for example `target/criterion/hnsw_build/report/index.html`) to
view timing distributions and comparisons against previous runs.

### Benchmark regression workflow

Benchmark regression detection follows a two-tier strategy:

- Pull request (PR) workflows run a fast benchmark smoke check using
  discovery mode (`--list`) to confirm that benchmark binaries still compile
  and enumerate benchmark cases.
- A scheduled weekly workflow (plus manual `workflow_dispatch`) runs
  Criterion baseline comparison by saving a reference baseline from `HEAD^` and
  comparing the current revision with `--baseline`.

Run the local baseline workflow for one benchmark from the repository root:

```sh
set -o pipefail
CHUTORO_BENCH_HNSW_MEMORY_PROFILE=0 \
CHUTORO_BENCH_HNSW_RECALL_REPORT=0 \
CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT=0 \
cargo bench -p chutoro-benches --bench hnsw_ef_sweep -- \
  --save-baseline local-reference \
  --noplot \
  2>&1 | tee /tmp/bench-hnsw-ef-sweep-save.log

set -o pipefail
CHUTORO_BENCH_HNSW_MEMORY_PROFILE=0 \
CHUTORO_BENCH_HNSW_RECALL_REPORT=0 \
CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT=0 \
cargo bench -p chutoro-benches --bench hnsw_ef_sweep -- \
  --baseline local-reference \
  --noplot \
  2>&1 | tee /tmp/bench-hnsw-ef-sweep-compare.log
```

Use `--list` for a quick discovery check only:

```sh
set -o pipefail
cargo bench -p chutoro-benches --bench hnsw_ef_sweep -- --list \
  2>&1 | tee /tmp/bench-hnsw-ef-sweep-list.log
```

### Neighbour scoring measurements

The `neighbour_scoring` Criterion benchmark isolates HNSW candidate scoring
from full graph construction. It reports realistic candidate buckets (`8`, `16`,
`24`, `32`, `48`) and diagnostic buckets (`256`, `1024`) for dense dimensions
`32`, `128`, and `768`.

Criterion targets use `harness = false`, so Cargo does not execute test
functions embedded in benchmark binaries. Keep benchmark files as thin
entrypoint wiring and place testable profiling, candidate-planning, and
argument-selection logic in `chutoro-benches/src/`, with unit tests beside the
library modules. This gives those tests Cargo's normal test harness and avoids
lint expectations for code that only appears unused in the harness-free build.

The library seam is owned by `chutoro-benches`: benchmark binaries may call it,
but production crates must not depend on it. Prefer pure helpers without
Criterion types for reusable selection and planning rules. Where a complete
runner must cross the separate-crate boundary between a benchmark target and
the support library, expose only a `#[doc(hidden)]` entrypoint; keep its
constituent helpers private to the support crate and compose them there.

Run the benchmark directly when comparing code changes:

```sh
set -o pipefail
cargo bench -p chutoro-benches --bench neighbour_scoring -- --save-baseline before \
  2>&1 | tee /tmp/bench-neighbour-scoring-save.log
```

Use the helper script for whole-binary corroboration with `hyperfine`:

```sh
scripts/bench-neighbour-scoring.sh
```

Set `CHUTORO_BENCH_NEIGHBOUR_PROFILE=1` to add the optional HNSW build profile.
Set `CHUTORO_BENCH_NEIGHBOUR_SHORT_MEASUREMENT` to a truthy value, using the
same parsing as `CHUTORO_BENCH_NEIGHBOUR_PROFILE`, to shorten Criterion's
warm-up and measurement durations for the `neighbour_scoring` group. Use this
mode for quick local iteration, not baseline comparisons. `neighbour_scoring`
uses this environment variable rather than `--exact` to select short
measurement mode.

By default, the reports are written to
`target/benchmarks/neighbour_scoring_build_profile.csv` and
`target/benchmarks/neighbour_scoring_lane_utilisation.csv`. When
`CARGO_TARGET_DIR` is set, the same filenames are written below its
`benchmarks/` directory. Treat `hyperfine` as corroboration; cycle-count and
Criterion evidence remain the primary signal for keeping a structural
optimization.

### Benchmark architecture

Benchmarks live in `chutoro-benches/benches/` as separate Criterion binaries.
Shared support code — the `SyntheticSource` data source, benchmark parameter
types, and the `BenchSetupError` error type — lives in the `chutoro-benches`
library (`chutoro-benches/src/`).

Each benchmark file follows this pattern:

1. A fallible `_impl` function performs all setup (data generation, HNSW
   build, MST computation) using `?` to propagate errors via `BenchSetupError`.
2. A thin wrapper registered with `criterion_group!` calls the `_impl`
   function and panics on failure.
3. The measured closure uses `b.iter()` or `b.iter_batched()` to time only
   the target operation.

### Lint policy for benchmarks

The `chutoro-benches` crate inherits the workspace lints through its
`[lints] workspace = true` manifest entry. Criterion's macro expansions
(`criterion_group!`, `criterion_main!`, and `bench_with_input` closures) can
still trigger strict denials — most notably `missing_docs`, `shadow_reuse`, and
`excessive_nesting` — so benchmark source files use tightly scoped
`#![expect(lint, reason = "…")]` attributes only where a macro expansion makes
the diagnostic unavoidable. Handwritten benchmark support remains subject to
the same root policy.

### Adding a new benchmark

1. Create a new file in `chutoro-benches/benches/`.
2. Add a corresponding `[[bench]]` entry in `chutoro-benches/Cargo.toml`.
3. Follow the fallible-setup pattern described above.
4. Use `#![expect(…)]` (not `#![allow(…)]`) for any Criterion-triggered lint
   suppressions, with a reason string.
5. Run `make bench` to verify the new benchmark appears in the output.

## HNSW scoring invariants

HNSW distance scoring must not run while the current thread holds the graph
write lock. The focused `hnsw::tests::write_lock` module enables a test-only
`CpuHnsw` write-graph marker, wraps a `DataSource`, and asserts every
`distance`, `batch_distances`, and `distance_batch` call happens outside the
write scope. Keep new insertion, trimming, and search code compatible with that
guard.

`DataSource::batch_distances(query, candidates)` has the following cache-layer
contracts:

- the returned distance vector length must equal `candidates.len()`;
- non-finite distances are rejected by validation before they enter HNSW
  scoring;
- `distance_batch(pairs, out)` implementations must leave `out` unmodified on
  error.

These contracts let `hnsw/validate.rs` and `hnsw/helpers.rs` merge cache hits
and misses without corrupting caller buffers after a provider error.

## Benchmark dataset recipes

The `chutoro-bench-datasets` crate defines the shared recipe surface for
benchmark dataset preparation. A `DatasetRecipe` fetches source inputs,
validates them, prepares benchmark-ready bytes, and publishes a manifest handle
through typed phase handoffs. The type returned by each phase is consumed by
the next one, so callers cannot publish fetched data that has not passed
validation and preparation.

Recipes receive infrastructure through `RecipeContext`. `Fetcher` reads source
bytes with a mandatory `max_bytes` cap, `Storage` is a mutable cache with
overwrite semantics, and `Publisher` is the final write-once sink. Network
download, archive extraction, object-store adapters, lockfiles, checksum
verification, and licence gates are deferred to later roadmap items.

Enable the crate's `testing` feature for adapter contract tests. It exposes
`InMemoryFetcher`, `InMemoryStorage`, `InMemoryPublisher`, `FilesystemFetcher`,
and `StubRecipe`. The behavioural tests in
`chutoro-bench-datasets/tests/recipe_bdd.rs` run the same fetcher contract
against both in-memory and filesystem-backed adapters.

The design rationale is recorded in
[`ADR-004`](adr-004-bench-dataset-recipe-trait.md). The broader dataset
pipeline is described in
[`benchmark-dataset-retrieval.md`](benchmark-dataset-retrieval.md) §3.1.

## Verus proofs

Verus is used for formal verification of edge harvest primitives. Run proofs via
`make verus`, which is idempotent and installs the pinned Verus release and
required Rust toolchain as needed.

### Quantifier trigger annotations

Verus prints warnings when it selects quantifier triggers automatically. Do not
ignore these warnings. Prefer explicit annotations so the prover behaviour is
stable and predictable:

- Use `#[trigger]` when a specific term should control instantiation.
- Use `#![auto]` only when the automatically chosen trigger is acceptable and
  the quantifier is straightforward.
- Avoid `--triggers-mode silent` in continuous integration (CI) because it
  hides trigger-selection changes.

Example:

```rust
assert forall|i: int| #![auto] 0 <= i < edges.len() implies edges[i].source == source_node;
```

## Spelling gate

Run `make spelling` to enforce en-GB-oxendict spelling in tracked Markdown
prose. The target checks `typos.toml` for drift, runs the consumer phrase
scanner, then runs the pinned `typos` release over tracked Markdown files.
`make markdownlint` depends on this gate, and `make all` runs it with the
repository's release build.

The generated configuration combines the shared estate dictionary with the
repository-specific `typos.local.toml` overlay. Do not edit `typos.toml` by
hand. Add only narrow identifier, API, proper-name, or immutable-fixture
exceptions to the local overlay; ordinary prose belongs in Oxford spelling.

The configuration builder is pinned to commit
`d6da92f02240a79a945c835f69bdd08a888da1d0`. Regenerate the configuration with:

```sh
TYPOS_CONFIG_BUILDER_COMMIT=d6da92f02240a79a945c835f69bdd08a888da1d0
uvx --python 3.14 \
  --from "git+https://github.com/leynos/typos-config-builder.git@${TYPOS_CONFIG_BUILDER_COMMIT}" \
  typos-config-builder
```

Use the same command with `--check` in quality gates to detect drift without
rewriting `typos.toml`. The builder refreshes the shared dictionary into the
untracked `.typos-oxendict-base.toml` cache only when the authority is newer,
records refresh metadata in `.typos-oxendict-base.json`, and reuses a valid
local cache when the authority is unavailable.

Typos splits hyphenated phrases into separate words. The consumer-owned
`scripts/typos_rollout_check.py` therefore reads phrase corrections from the
shared cache and local overlay, while taking ignore patterns and file
exclusions from generated `typos.toml`. It reports prohibited phrases without
duplicating the builder's validation, cache, merge or rendering behaviour.
