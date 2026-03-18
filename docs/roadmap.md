# chutoro — Phased implementation roadmap (CPU → GPU → plugins)

_This roadmap references sections of the revised design document (e.g., §6.2)
rather than external sources._

______________________________________________________________________

## 0. Walking skeleton (CPU-only, no dynamic plugins)

### 0.1. Workspace and core API

- [x] 0.1.1. Create Cargo workspace with crates: `chutoro-core`,
  `chutoro-cli`, `chutoro-providers/dense`, `chutoro-providers/text`. (See §4)
- [x] 0.1.2. Define public `DataSource` trait with `len()`, `name()`,
  `distance(i,j)`, and a defaulted `distance_batch(pairs, out)` for
  forward-compatibility. (See §10.2)
- [x] 0.1.3. Implement `ChutoroBuilder` and
  `Chutoro::run<D: DataSource>` orchestration API. (See §10.1)

### 0.2. Data providers and distance functions

- [x] 0.2.1. Implement `DenseMatrixProvider` that packs Parquet/Arrow
  `FixedSizeList<Float32,D>` into a contiguous row-major `Vec<f32>`; reject
  ragged/null rows. (See §5, §10.6)
- [x] 0.2.2. Implement `TextProvider` (one UTF-8 string per line) to
  exercise non-metric distances (e.g., Levenshtein). (See §1.3)
- [x] 0.2.3. Add `cosine` and `euclidean` distance implementations in
  `chutoro-core` with optional precomputed norms for cosine. (See §3.1)

### 0.3. Command-Line Interface (CLI) and operational infrastructure

- [x] 0.3.1. Ship a minimal CLI:
  `chutoro run parquet <path> --column features ...` and
  `chutoro run text <path> --metric levenshtein`. Shared flags include
  `--min-cluster-size <usize>` defaulting to `5` and `--name <string>`.
  `--name` defaults to the file stem via UTF-8 lossy conversion for visibility.
  (See §10)
- [x] 0.3.2. Add structured logging via `tracing` +
  `tracing-subscriber`; bridge the `log` crate via `tracing-log`; replace
  manual prints and initialize logging in the CLI (e.g.,
  `tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env())`,
  human and JSON formats, and span IDs). (See §10.5)
- [x] 0.3.3. Define error taxonomy; adopt `thiserror` for a unified
  `ChutoroError` in public APIs and use `anyhow` in binaries; return
  `Result<T, ChutoroError>` with stable, documented error codes. (See §10.6)
- [ ] 0.3.4. Establish Continuous Integration (CI) (fmt, clippy, test),
  feature gates (`cpu`, `gpu` off by default), and reproducible toolchain
  (`rust-toolchain.toml`). (See §11)

**Exit criteria:** compile+run on small datasets; end-to-end smoke test
produces a cluster assignment (even with stubbed internals).

______________________________________________________________________

## 1. CPU HNSW, candidate edge harvest, and parallel Kruskal MST

### 1.1. Hierarchical Navigable Small World (HNSW) graph construction

- [x] 1.1.1. Implement CPU HNSW insertion/search with Rayon; two-phase
  locking (`read` for search → `write` for insert) on a shared graph. (See
  §6.1, §6.2)
- [x] 1.1.2. Introduce a `DistanceCache` backed by `dashmap` to avoid
  recomputing distances across threads during HNSW insertion. (See §6.2, §10.6)
  - Key: normalize to `(min(i,j), max(i,j))`; encode metric and its
    parameters (e.g., cosine with/without pre-norms) in the key.
  - Value: `f32` distance; NaN policy: do not cache NaN; propagate an
    error and log at WARN.
  - Bounds: enforce `max_entries` via Least Recently Used (LRU) eviction
    (optional TTL); document defaults and configuration knobs.
  - Metrics: expose `distance_cache_hits`,
    `distance_cache_misses`, `distance_cache_evictions`, and
    `distance_cache_lookup_latency_histogram` via the `metrics` crate.
    The first three are monotonic counters; lookup latency is a
    histogram of seconds. These metrics are emitted only when the
    `metrics` feature flag is enabled and are documented for external
    consumption; emit `tracing` spans for hot paths.
  - Concurrency: require `Send + Sync`; forbid iteration on hot paths;
    avoid holding HNSW write locks while updating the cache.
  - Determinism: ensure neighbour selection is unchanged under fixed
    seeds by using a deterministic tie-break: on equal distances prefer
    the lower item id; when ids match fall back to the insertion
    sequence number. Eviction must be time-independent in tests. This
    rule is used in all builds and tests to guarantee stable outputs
    under fixed seeds.
- [x] 1.1.3. During insertion, capture candidate edges `(u,v,w)`
  discovered by HNSW; accumulate via Rayon `map` → `reduce` into a global edge
  list. (See §6.2)

### 1.2. Parallel Kruskal minimum spanning tree (MST) and hierarchy extraction

- [x] 1.2.1. Implement parallel Kruskal: parallel sort of edges,
  concurrent union-find for cycle checks. (See §3.2, §6.2)
- [x] 1.2.2. Implement hierarchy extraction from MST (sequential to
  start): build condensed tree and stability scoring; output flat labels. (See
  §1.2, §6.2)

### 1.3. Correctness testing

- [x] 1.3.1. Add deterministic tests on tiny datasets (exact
  mutual-reachability path) to sanity-check hierarchy logic. (See §11)
- [x] 1.3.2. Add functional tests comparing Adjusted Rand Index (ARI)
  and Normalized Mutual Information (NMI) with HDBSCAN/HNSW baselines on small
  public sets. (See §11)

### 1.4. HNSW property testing

- [x] 1.4.1. Introduce property-based generators for HNSW using
  `proptest` + `test-strategy`; cover uniform, clustered, manifold, and
  duplicate vectors plus configuration sampling. (See property-testing-design
  §2.1)
- [x] 1.4.2. Surface HNSW structural invariant checkers (layer
  consistency, degree bounds, reachability, bidirectional links) callable from
  properties. (See property-testing-design §2.2)
- [x] 1.4.3. Add HNSW search correctness property comparing index
  results against a brute-force oracle with a configurable recall threshold and
  timing capture. (See property-testing-design §2.3.1)
- [x] 1.4.4. Add stateful HNSW mutation property that exercises
  add/delete/reconfigure sequences and revalidates invariants after each
  operation. (See property-testing-design §2.3.2)
  - [x] Improve invariant logging and document how to reproduce/shrink
    property-test failures for debugging.
  - [x] Restore safe deletion semantics in tests: reintroduce delete
    handling without breaking reachability, or fail fast when deletion
    would disconnect the graph. (Follow-up to reciprocity fix.)
  - [x] Add a regression test where trimming evicts the new node from
    an existing neighbour, asserting the post-commit reciprocity pass
    either adds the reverse edge or removes the forward edge.
  - [x] Integrate reciprocity enforcement into the insertion executor
    to avoid the post-pass scan once correctness is validated.
- [x] 1.4.5. Add HNSW insertion idempotency property: repeated
  duplicate insertions leave graph state unchanged. (See
  property-testing-design §2.3.3)

### 1.5. Formal verification

- [x] 1.5.1. Formal verification (Kani) expansion for HNSW and
  pipeline invariants:
  - [x] Replace the 3-node bidirectionality harness with a commit-path
    harness that drives reconciliation and deferred scrubs via
    `CommitApplicator::apply_neighbour_updates`.
  - [x] Add an eviction/deferred-scrub scenario: pre-fill a target
    neighbour list to `max_connections`, force `ensure_reverse_edge` to
    evict, and assert reciprocity after `apply_deferred_scrubs`.
  - [x] Add bounded Kani harnesses for the following explicit
    invariants:
    - **No self-loops**: For every node `u` and layer `l`, `u` is not
      in `N_l(u)`.
    - **Neighbour list uniqueness**: For every node `u` and layer `l`,
      the list `N_l(u)` contains no duplicates (set semantics).
    - **Entry-point validity and maximality**: If the graph contains at
      least one node, the entry point is set to an existing node, and
      its level equals the maximum level present in the graph (for all
      nodes `v`, `level(entry) >= level(v)`).
    - **MST structural correctness (CPU/GPU)**: For any connected
      candidate graph on `n` nodes, the MST output has exactly `n-1`
      edges, is acyclic, and connects all nodes. For disconnected
      graphs with `c` components, the output is a forest with `n-c`
      edges.
    - **Distance kernel consistency (CPU/GPU)**: For any metric
      distance implementation, distances are symmetric and zero on
      identical inputs, and CPU and GPU implementations agree within a
      defined tolerance `epsilon` for the same inputs.
- [x] 1.5.2. Add a nightly "slow" CI job that runs `make kani-full`
  only when main has new commits in the last 24 hours (Coordinated Universal
  Time (UTC)); keep `make test` unchanged so Kani remains opt-in for normal
  development loops.

### 1.6. Edge harvest and MST property testing

- [x] 1.6.1. Implement composite graph strategies (random, scale-free,
  lattice, disconnected) for candidate edge harvest testing. (See
  property-testing-design §3.1)
- [x] 1.6.2. Add candidate edge harvest property suite covering
  determinism, degree ceilings, connectivity preservation, and reverse nearest
  neighbour uplift metrics. (See property-testing-design §3.2)
- [x] 1.6.3. Add candidate edge harvest algorithm property suite that
  validates the harvested output (edge validity, degree constraints,
  connectivity preservation or bounded destruction, and RNN uplift relative to
  the input) across generated topologies. (See property-testing-design §3.2
  additions)
  - Prerequisites:
    - Composite topology generators are complete. (See §3.1)
    - Candidate edge harvest topology suite is complete. (See §3.2)
  - Acceptance criteria:
    - Run at least 256 generated fixtures per topology in the proptest
      suite.
    - Edge validity passes in 100% of cases (no self-loops, in-bounds
      endpoints, finite distances).
    - Degree ceilings are respected in 100% of cases.
    - Connectivity is preserved for connected topologies in at least
      95% of cases, with any remaining cases limited to a +1 component
      increase.
    - RNN uplift median delta is ≥ 0.05 for
      lattice/random/disconnected inputs and ≥ 0.0 for scale-free
      inputs.
- [x] 1.6.4. Add Verus proofs for edge harvest primitives:
  - `extract_candidate_edges` invariants
    (source/target/sequence/count).
  - `CandidateEdge::canonicalise` preserves order and fields.
  - `EdgeHarvest::from_unsorted` permutation + ordering guarantees.
  - (See property-testing-design Appendix A)
  - Prerequisites:
    - Edge harvest helper signatures are stable.
    - Verus toolchain version is pinned and documented for
      contributors.
  - Acceptance criteria:
    - Verus proofs cover the Appendix A invariants without `assume`
      shortcuts.
    - Proof harnesses pass in CI for the pinned Verus toolchain.
    - Scope is limited to helper invariants (no concurrency or planner
      proofs).
- [x] 1.6.5. Build parallel Kruskal property suite: compare against
  sequential oracle, enforce acyclicity/connectivity/edge-count invariants, and
  rerun jobs to detect race-induced non-determinism. (See
  property-testing-design §4)

### 1.7. Continuous integration

- [x] 1.7.1. Integrate property-based suites into CI with a
  path-filtered pull request (PR) job (250 cases, 10-minute timeout) and
  scheduled weekly job (25,000 cases, `fork = true`, `PROGTEST_CASES` env var).
  (See property-testing-design §5)
  - [x] Define and enforce property-based CI guardrail thresholds
    (recall floor and max_connections minimums) with explicit values.
    - Decide on the recall floor that CI must enforce by setting
      `CHUTORO_HNSW_PBT_MIN_RECALL`; raise the recall floor once the
      high-fan-out search implementation improves.
    - Broaden max_connections guardrails once graph connectivity work
      lands so that the current `max_connections >= 16` guard can be
      tightened.

**Exit criteria:** 100k × D vectors complete in minutes on CPU; memory bounded
≈ `n*M` edges; ARI/NMI within acceptable band vs reference. (See §6.2)

______________________________________________________________________

## 2. Benchmarking and profiling (CPU)

### 2.1. Benchmarking infrastructure

- [x] 2.1.1. Create `chutoro-benches` with Criterion harness for HNSW
  build, edge harvest, MST, and extraction timings. (See §11)
- [x] 2.1.2. Extend `SyntheticSource` with diverse data generators:
  Gaussian blobs (configurable separation, cluster count, and anisotropy),
  ring/manifold patterns for non-linearly-separable data, and text strings
  exercising Levenshtein distance. Add MNIST (70k ×
  784) as a real-world baseline via a download-and-cache helper.
  Document which synthetic patterns stress which pipeline stages. (See §1.3)
- [x] 2.1.3. Track memory footprint alongside timing: use Criterion
  custom measurements or a separate profiler to record peak memory during HNSW
  build. Report memory/point and memory/edge metrics vs `M∈{8,12,16,24}`.
  Validate that memory scales as expected (≈ n×M edges). (See §11)
- [x] 2.1.4. Expand parameter coverage: add M=12 and M=24 to HNSW
  benchmarks (current sweep is {8, 16}), and vary `ef_construction` (e.g., 100,
  200, 400) to show build-time vs recall trade-offs. Document parameter choices
  and their performance/quality implications. (See §6.2)
- [x] 2.1.5. Implement memory guards (`--max-bytes`) in the CLI with
  clear failure messages when datasets exceed available memory. Ensure
  benchmarks fail fast on oversized datasets and document expected memory
  requirements per dataset size. (See §5)
- [x] 2.1.6. Add optional clustering quality tracking to benchmarks:
  for synthetic clustered data (Gaussians), compute Adjusted Rand Index (ARI)
  and Normalized Mutual Information (NMI) against ground truth as secondary
  metrics alongside timing. Guards against quality regressions during
  performance tuning. (See §11)
- [x] 2.1.7. Establish a CI regression detection strategy for
  benchmarks: document whether benchmarks run on every pull request (PR) or on
  a scheduled nightly/weekly job using Criterion's baseline comparison. If too
  slow for PR gating, add a scheduled job and document the developer-run
  workflow. (See §11)

### 2.2. Single Instruction, Multiple Data (SIMD) distance kernels

- [x] 2.2.1. Add CPU distance kernels using stable `core::arch` x86 intrinsics
  with AVX2/AVX-512 specializations; make `distance_batch` the default HNSW
  scoring path. Keep `std::simd` optional and nightly gated while
  `portable_simd` remains unstable (`rust-lang/rust#86656`); AVX-512 stable
  intrinsics are available from Rust `1.89.0` (`rust-lang/rust#111137`). (See
  §6.3)
- [x] 2.2.2. Introduce `DensePointView<'a>` for aligned Structure of
  Arrays (SoA) access with a scalar fallback. (See §6.3)
- [ ] 2.2.3. Gate SIMD backends behind `simd_avx2`, `simd_avx512`, and
  `simd_neon` features with CPUID runtime dispatch. (See §6.3)
  - Use `is_x86_feature_detected!`/platform equivalents and one-time
    function-pointer patching to avoid hot-path branching.
  - Define NaN and other non-finite treatment for reductions to ensure
    CPU/GPU parity.
  - Guarantee 64-byte alignment and lane-multiple padding for
    `DensePointView<'a>`; zero-pad tails.
- [ ] 2.2.4. Add an optional nightly only `std::simd` backend behind a
  non-default Cargo feature and nightly Continuous Integration (CI) job; keep
  stable `core::arch` implementation as default. Track `portable_simd`
  stabilization (`rust-lang/rust#86656`) and AVX-512 adjunct blockers
  (`rust-lang/rust#127356` and `rust-lang/rust#127213`). (See §6.3)
- [ ] 2.2.5. Implement portable-SIMD gating mechanics so stable and nightly
  paths can coexist safely:
  - add a non-default Cargo feature (for example, `nightly-portable-simd`);
  - gate crate-level `#![feature(portable_simd)]` with `cfg_attr`;
  - isolate nightly SIMD modules behind feature `cfg` guards;
  - add CI checks that verify stable builds with the feature disabled and
    nightly builds with the feature enabled. (See §6.3)

### 2.3. Hot-path optimizations

- [ ] 2.3.1. Restructure HNSW neighbour evaluation to use packed
  indices and an SoA layout, prefetching blocks and scoring outside the write
  lock. (See §6.3)
- [ ] 2.3.2. Vectorize edge-weight transforms and candidate filtering
  before union-find in parallel Kruskal; keep union-find cache-friendly. (See
  §6.3)

### 2.4. Performance validation

- [ ] 2.4.1. Benchmark Euclidean/cosine kernels, neighbour scoring, and
  batched `distance_batch` versus scalar `distance`; bucket candidate sizes to
  confirm SIMD gains. (See §6.3, §11)

**Exit criteria:** stable baseline numbers to compare against GPU phases;
documented CPU hot spots.

______________________________________________________________________

## 3. GPU MST offload (Borůvka via rust-cuda HAL)

### 3.1. GPU Hardware Abstraction Layer (HAL) and backends

- [ ] 3.1.1. Add GPU HAL to `chutoro-core` with an execution-path
  selector. (See §7.1)
- [ ] 3.1.2. Implement `chutoro-backend-cuda` using `cust`/`cudarc`,
  `cuda_std`, `rustc_codegen_nvvm`, `cuda_builder`; gate behind `backend-cuda`.
  (See §7.1)
- [ ] 3.1.3. Stub `chutoro-backend-cubecl` and optional
  `chutoro-backend-sycl` crates with features `backend-portable` and
  `backend-sycl`. (See §7.1)

### 3.2. Device data structures and kernels

- [ ] 3.2.1. Define device data structures: global edge list, Disjoint
  Set Union (DSU) parent array, MST output buffer. (See §8.2)
- [ ] 3.2.2. Implement Kernel 1: per-component (or per-vertex) min
  outgoing edge selection. (See §8.2)
- [ ] 3.2.3. Implement Kernel 2: parallel union (atomic
  Compare-And-Swap (CAS)) and component compaction; host loop rounds until one
  component remains. (See §8.2)

### 3.3. Host integration and validation

- [ ] 3.3.1. Add host-side adapter: copy candidate edges once to
  device, run Borůvka, copy MST back. (See §9.1)
- [ ] 3.3.2. Gate backends behind features (`backend-cuda`,
  `backend-portable`, `backend-sycl`); fall back cleanly to CPU when
  unavailable. (See §7.1, §11)
- [ ] 3.3.3. Verify correctness vs CPU Kruskal on random graphs;
  benchmark speedup. (See §11)

**Exit criteria:** MST wall-time reduced significantly vs CPU on ≥1e6 edges;
identical MST (or accepted tie-break equivalence) to CPU implementation.

______________________________________________________________________

## 4. Hybrid HNSW distance offload (CPU traversal, GPU distance batches)

### 4.1. GPU distance offload

- [ ] 4.1.1. Implement CUDA distance kernel operating on row-major
  `f32` matrix (one thread per candidate). (See §8.1)
- [ ] 4.1.2. Add host orchestration hook in HNSW insertion to batch
  candidate distance evaluations to GPU. (See §8.1)
- [ ] 4.1.3. Introduce a small pinned host/device ring buffer for
  batched copies. (See §9.2)

### 4.2. Validation

- [ ] 4.2.1. Measure build-time reduction vs CPU-only HNSW; ensure
  identical neighbour selections under deterministic seeds. (See §8.1, §11)

**Exit criteria:** measurable reduction in HNSW insertion time on large batches
without altering clustering materially.

______________________________________________________________________

## 5. Async orchestration and streams

### 5.1. Stream pipeline

- [ ] 5.1.1. Introduce multi-stream pipeline: Stream 1 (mem ops),
  Stream 2 (HNSW distance), Stream 3 (MST). (See §9.2)
- [ ] 5.1.2. Overlap dataset upload with initial compute; enforce
  dependencies via events/stream waits. (See §9.2)
- [ ] 5.1.3. Keep large buffers resident on device; copy back only
  final labels/hierarchy. (See §9.1)

### 5.2. Observability

- [ ] 5.2.1. Add tracing/logging around stream scheduling to validate
  overlap. (See §9.2)

**Exit criteria:** observed kernel/transfer overlap; reduced end-to-end
wall-time vs phase 4.

______________________________________________________________________

## 6. Dynamic plugin loader (C-ABI v-table handshake)

### 6.1. Application Binary Interface (ABI) definition

- [ ] 6.1.1. Freeze `chutoro_v1` `#[repr(C)]` v-table with
  `abi_version`, `caps`, `state`, and function pointers; include optional
  `distance_batch`. (See §5.3)
- [ ] 6.1.2. Add mandatory `destroy` callback to the v-table and ensure
  the `PluginManager` calls it when unloading plugins. (See §5.3)
  - ABI: `extern "C" fn destroy(state: *mut c_void)`; no panics across
    Foreign Function Interface (FFI).
  - Ordering: drop all host-side wrappers/handles; call `destroy`, wait
    for it to return; then unload the dynamic library.
  - Idempotency: `destroy` must be safe if called multiple times
    (idempotent); the host calls it at most once in normal operation.
    Document behaviour if the plugin reports errors.
  - Quiescence: ensure all in-flight callbacks complete and any
    plugin-spawned threads are joined/cancelled before calling
    `destroy`; forbid re-entry after unload.
  - Timeouts: enforce a configurable quiescence timeout (e.g., 30s) and
    log at WARN on expiry before proceeding to unload.
  - Safety: add logs at INFO with plugin name and timing.
  - Tests: load→use→quiesce→destroy→unload cycle under leak detectors
    and ASAN/UBSAN.

### 6.2. Plugin management

- [ ] 6.2.1. Implement `PluginManager` using `libloading` to locate
  `_plugin_create`, validate `abi_version`, wrap as safe `DataSource`. (See
  §5.2, §5.3)
- [ ] 6.2.2. Ship example plugin `chutoro-plugin-csv` and
  `chutoro-plugin-parquet`; document build and loading. (See §5)
- [ ] 6.2.3. Add capability flags (`HAS_DISTANCE_BATCH`,
  `HAS_DEVICE_VIEW`, `HAS_NATIVE_KERNELS`) for GPU-aware providers. (See §5.3,
  §7.1)

### 6.3. Safety and hardening

- [ ] 6.3.1. Fuzz and harden FFI boundaries (lengths, nulls, lifetime
  ownership); ensure host remains robust on plugin failure. (See §5.1)

**Exit criteria:** load/unload plugins at runtime; parity with statically
linked providers; safe failure semantics.

______________________________________________________________________

## 7. DataFusion and `object_store` provider (optional)

### 7.1. DataFusion provider and cloud storage

- [ ] 7.1.1. Implement `DataFusionProvider` to execute SQL
  predicates/projections and materialize to `DenseMatrixProvider`. (See §5,
  §10.6)
- [ ] 7.1.2. Support `s3://`, `gs://`, `azure://` via `object_store`;
  add sampling options for large tables. (See §5)

### 7.2. Documentation

- [ ] 7.2.1. Document schema contract (single `features`
  `FixedSizeList<Float32,D>` column) and error messages. (See §5)

**Exit criteria:** users can point at local/cloud Parquet and run queries that
materialize into chutoro's fast path.

______________________________________________________________________

## 8. Documentation, examples, and release

### 8.1. Documentation and examples

- [ ] 8.1.1. Write `README` with architecture diagram and quickstart;
  add API docs via `cargo doc`. (See §4, §10)
- [ ] 8.1.2. Provide end-to-end examples: numeric embeddings and string
  clustering. (See §1.3)

### 8.2. Release

- [ ] 8.2.1. Publish `v0.1.0` (CPU baseline), feature-gated GPU, and
  example plugin crates; tag and create release notes with benchmark tables.
  (See §11)

**Exit criteria:** documented, installable release; examples reproduce reported
benchmarks.

______________________________________________________________________

## 9. Hardening and enhancements (post-v0.1)

### 9.1. Performance and observability

- [ ] 9.1.1. Parallelize parts of hierarchy extraction if profiling
  justifies. (See §8.3)
- [ ] 9.1.2. Add metrics/telemetry hooks (timings, memory) behind a
  feature flag. (See §11)

### 9.2. Plugin ABI evaluation

- [ ] 9.2.1. Evaluate `abi_stable`/`stabby` as an alternative to C-ABI
  for Rust-to-Rust plugins. (See §5.2)

**Exit criteria:** measurable incremental gains; stable ABI story; clean
telemetry for future tuning.

______________________________________________________________________

## 10. Benchmark dataset operations and matrix execution

### 10.1. Shared dataset retrieval and preparation infrastructure

- [ ] 10.1.1. Introduce `chutoro-bench-datasets` with a `DatasetRecipe`
  trait for fetch, validate, prepare, and publish steps shared across all
  datasets. See `docs/benchmark-dataset-retrieval.md` §3.1.
- [ ] 10.1.2. Implement shared download primitives: pinned source URLs,
  checksum/signature verification, resumable transfers, and archive extraction
  (`.gz`, `.bz2`, `.xz`, `.tar`). See `docs/benchmark-dataset-retrieval.md`
  §3.2.
- [ ] 10.1.3. Define canonical prepared artefact contracts
  (`features`, `labels`, optional `ground_truth`) plus immutable
  `manifest.json` metadata containing source provenance, schema version, and
  preprocessing hash. See `docs/benchmark-dataset-retrieval.md` §3.1.
- [ ] 10.1.4. Add `object_store`-based cache adapters for S3-compatible
  endpoints (AWS S3, Scaleway Object Storage, DigitalOcean Spaces), including
  endpoint overrides, credentials, and server-side encryption settings. See
  `docs/benchmark-dataset-retrieval.md` §3.2.
- [ ] 10.1.5. Add local cache plus lockfile semantics so repeated matrix runs
  deduplicate downloads and avoid concurrent preparation races. See
  `docs/benchmark-dataset-retrieval.md` §3.2.
- [ ] 10.1.6. Add provenance/licence checks that fail fast when a dataset source
  terms change, require interactive access, or lack reproducible checksum
  coverage. See `docs/benchmark-dataset-retrieval.md` §3.3.

### 10.2. Matrix benchmark framework and result publication

- [ ] 10.2.1. Define a declarative matrix spec
  (`benchmarks/matrix.toml`) describing dataset, backend, metric, and parameter
  dimensions, with profile selectors (`smoke`, `cpu`, `scale`). See
  `docs/benchmark-dataset-retrieval.md` §4.1.
- [ ] 10.2.2. Implement `chutoro-bench-matrix` (Rust binary) to expand matrix
  entries and orchestrate Criterion benchmark invocations per tuple; preserve
  deterministic seeds and run metadata. See
  `docs/benchmark-dataset-retrieval.md` §4.1.
- [ ] 10.2.3. Add backend capability discovery so unsupported tuples (for
  example GPU-only backends on CPU hosts) are skipped with structured reasons
  instead of hard failures. See `docs/benchmark-dataset-retrieval.md` §4.1.
- [ ] 10.2.4. Standardize result artefacts (`results.jsonl`,
  `summary.parquet`, `report.md`) with tuple identifiers:
  dataset/version/backend/metric/profile/git SHA. See
  `docs/benchmark-dataset-retrieval.md` §4.2.
- [ ] 10.2.5. Add object-store write-back for benchmark results and baselines
  under immutable keys (for example `bench-results/<run-id>/...`) and maintain
  a mutable pointer to latest baseline per tuple. See
  `docs/benchmark-dataset-retrieval.md` §4.3.
- [ ] 10.2.6. Integrate scheduled CI jobs that run matrix profiles, publish
  artefacts, and compare against stored baselines for regression alerts.
  Requires 2.1.7. See `docs/benchmark-dataset-retrieval.md` §4.3.

### 10.3. Dataset-specific enablement backlog

- [ ] 10.3.1. `make_blobs`: add deterministic recipe catalogue (seeded
  separation/noise/imbalance profiles), emit labels, and publish smoke profile
  manifests. See `docs/benchmark-dataset-retrieval.md` §7.1.
- [ ] 10.3.2. MNIST digits: implement pinned IDX fetch + checksum validation,
  convert to canonical dense format, and register `smoke` and `cpu` matrix
  tuples with ARI/NMI scoring. See `docs/benchmark-dataset-retrieval.md` §7.2.
- [ ] 10.3.3. Fashion-MNIST: mirror MNIST pipeline with separate provenance and
  checksums, then register as a harder small-scale baseline in matrix profiles.
  See `docs/benchmark-dataset-retrieval.md` §7.3.
- [ ] 10.3.4. CIFAR-10/100: add fixed embedding pipeline (frozen model and
  versioned preprocessing), cache vectors + labels, and expose both coarse and
  fine class evaluation tuples. See `docs/benchmark-dataset-retrieval.md` §7.4.
- [ ] 10.3.5. 20 Newsgroups: add deterministic text normalization and embedding
  recipe, cache vectors + topic labels, and register CPU profile tuples with
  topic recovery metrics. See `docs/benchmark-dataset-retrieval.md` §7.5.
- [ ] 10.3.6. RCV1-v2: add multilabel fetch and normalization pipeline with
  sparse-to-dense projection recipe, publish reproducible train/test manifests,
  and enable long-running CPU matrix tuples. See
  `docs/benchmark-dataset-retrieval.md` §7.6.
- [ ] 10.3.7. SNAP com-Amazon: add graph ingest + node embedding recipe with
  pinned hyperparameters, cache vectors + community labels, and enable graph
  quality tuple scoring. See `docs/benchmark-dataset-retrieval.md` §7.7.
- [ ] 10.3.8. SNAP com-DBLP: implement parallel graph pipeline to com-Amazon
  with overlapping-community handling and dedicated matrix tuple definitions.
  See `docs/benchmark-dataset-retrieval.md` §7.8.
- [ ] 10.3.9. PBMC 68k (10x): define controlled-source ingestion path from an
  internally mirrored artefact, add PCA/normalization recipe checks, and gate
  matrix use behind provenance confirmation. See
  `docs/benchmark-dataset-retrieval.md` §7.9.
- [ ] 10.3.10. GloVe vectors: implement dimension-selectable ingestion
  (`25/50/100/200`), enforce tokenizer/projection invariants, and add angular
  distance tuples in `cpu` and `scale` profiles. See
  `docs/benchmark-dataset-retrieval.md` §7.10.
- [ ] 10.3.11. SIFT1M: integrate ANN-Benchmarks HDF5 ingestion, validate
  provided nearest-neighbour ground truth, and register recall@{1,10,100}
  matrix tuples. See `docs/benchmark-dataset-retrieval.md` §7.11.
- [ ] 10.3.12. GIST1M: add high-dimensional ingest validation plus memory guard
  assertions, and gate tuples to hosts that satisfy documented memory
  thresholds. See `docs/benchmark-dataset-retrieval.md` §7.12.
- [ ] 10.3.13. DEEP1B/BigANN: implement sharded subset ingestion
  (1M/10M/100M/1B), resumable object-store uploads, and profile-aware matrix
  partitioning so billion-scale jobs run only on dedicated scale runners. See
  `docs/benchmark-dataset-retrieval.md` §7.13.

______________________________________________________________________

## 11. Incremental clustering

### 11.1. Incremental insertion and edge harvesting

- [ ] 11.1.1. Expose the edge-harvesting HNSW insertion path as a public
  API. The internal `insert_with_edges` already returns `Vec<CandidateEdge>`;
  promote it (or add a public wrapper such as `insert_harvesting`) so that
  callers outside `chutoro-core` can insert points and receive candidate edges
  without discarding them via `NoopCollector`. (See `docs/chutoro-design.md`
  §12.4)
- [ ] 11.1.2. Define `SessionConfig` carrying clustering parameters
  (min cluster size, HNSW params, refresh policy) derived from
  `ChutoroBuilder`, and add `ChutoroBuilder::build_session` returning a
  `ClusteringSession`. (See `docs/chutoro-design.md` §12.3)
- [ ] 11.1.3. Implement `ClusteringSession::append` accepting a slice
  of new point indices. Each insertion calls the public edge-harvesting path
  and accumulates delta `CandidateEdge` values in the session's `pending_edges`
  buffer. (See `docs/chutoro-design.md` §12.3, §12.4)
  - Acceptance criteria: after appending N points the HNSW index
    contains all appended points and `pending_edges` includes every
    harvested edge produced by those insertions. Zero harvested edges
    is valid for datasets or stages (e.g. early bootstrap) where
    inserts produce no harvested edges.
- [ ] 11.1.4. Implement incremental core-distance computation for newly
  appended points: search the HNSW index for each new point's
  `min_cluster_size`-th nearest neighbour and record its core distance.
  Recompute core distances for existing points that appear as neighbours of new
  insertions. Requires 11.1.2 (`ClusteringSession`), 11.1.3 (`append` and
  `pending_edges`). (See `docs/chutoro-design.md` §12.4)

### 11.2. Incremental MST refresh and label extraction

- [ ] 11.2.1. Implement `ClusteringSession::refresh`: merge
  `pending_edges` with the existing `mst_edges` and retained non-MST historical
  edges (see §12.5), apply mutual-reachability weighting using updated core
  distances, construct a fresh `EdgeHarvest` from the combined set, run
  `parallel_kruskal`, and extract labels via `extract_labels_from_mst`. Publish
  the new label snapshot as `Arc<Vec<usize>>` and advance the
  `snapshot_version` counter. Requires 11.1.3 (`pending_edges`), 11.1.4 (core
  distances). (See `docs/chutoro-design.md` §12.5)
  - Acceptance criteria: after refresh, `session.labels()` returns a
    label vector whose length equals the total number of points
    (initial + appended). `snapshot_version` increments by exactly one
    per refresh call.
- [ ] 11.2.2. Implement `ClusteringSession::labels` returning the
  latest `Arc<Vec<usize>>` label snapshot without blocking the writer. Requires
  11.2.1 (`refresh` publishes the label snapshot). (See
  `docs/chutoro-design.md` §12.6)
- [ ] 11.2.3. Add count-triggered refresh policy: configure
  `SessionConfig` with an optional `refresh_every_n` threshold so that `append`
  automatically triggers `refresh` after every N accumulated points. Requires
  11.1.3 (`append`), 11.2.1 (`refresh`). (See `docs/chutoro-design.md` §12.6)
- [ ] 11.2.4. Implement `ClusteringSession::refresh_full`: perform a
  complete core-distance recomputation for all points before running the
  standard refresh path. Expose `SessionConfig` options for automatic
  full-refresh triggers: (a) cumulative appended point fraction exceeding a
  configurable threshold (default 25% of total dataset); (b) ARI/NMI
  degradation trigger—add `SessionConfig` fields `ari_threshold` (default
  0.92), `nmi_threshold` (default 0.92), and `enable_ari_nmi_trigger` (default
  `false`); when enabled, the refresh decision path computes current ARI/NMI
  against the baseline snapshot and invokes `refresh_full()` when either metric
  drops below its configured threshold; (c) caller request via explicit
  `refresh_full()` call. In addition, implement the following gating checks in
  the refresh decision path before computing ARI/NMI or invoking
  `refresh_full()`: (d) baseline staleness policy—track the baseline snapshot's
  `snapshot_version` and `dataset_size`; compute a staleness metric
  (`session.snapshot_version - baseline.snapshot_version`) and compare against
  `SessionConfig::baseline_max_age_refreshes` (default 50); skip trigger (b)
  when the baseline is stale or incompatible
  (`baseline.dataset_size > session.point_count()`); (e) shape
  compatibility—verify that the baseline label vector's dimensionality and
  feature schema are compatible with the current dataset before computing
  ARI/NMI; reject mismatched shapes with a diagnostic; (f) overlap gating—add
  `SessionConfig::minimum_overlap_fraction` (default 0.50); compute the overlap
  fraction between baseline and current dataset (shared point-id prefix size
  divided by current dataset size); only compute ARI/NMI or trigger
  `refresh_full()` when the overlap fraction ≥ `minimum_overlap_fraction`;
  otherwise treat the baseline as stale. Requires 11.2.1. (See
  `docs/chutoro-design.md` §12.4)
  - Acceptance criteria: after `refresh_full()`, ARI/NMI against a
    batch baseline is ≥ 0.98 on a dataset that has undergone ≥ 50
    incremental refresh cycles. Unit tests verify that the
    `ari_threshold`, `nmi_threshold`, `enable_ari_nmi_trigger`,
    `baseline_max_age_refreshes`, and `minimum_overlap_fraction`
    config fields default correctly. Unit tests assert that
    staleness, shape-mismatch, and insufficient overlap each
    prevent ARI/NMI-triggered refreshes, and that a compatible,
    fresh baseline with sufficient overlap allows trigger (b) to
    fire when degradation exceeds the configured threshold.
- [ ] 11.2.5. Implement bounded `historical_edges` retention: after
  each refresh, partition Kruskal output into MST and non-MST edges. Retain
  non-MST edges in `historical_edges` up to a configurable cap (default 2× MST
  edge count), evicting the heaviest edges first. Clear `pending_edges`.
  Requires 11.2.1. (See `docs/chutoro-design.md` §12.5)
  - Acceptance criteria: `historical_edges.len()` never exceeds the
    configured cap. After 100 append-refresh cycles, total session
    edge memory (MST + historical + pending) remains within 4× the
    MST edge count.

### 11.3. Seeding and batch bootstrap

- [ ] 11.3.1. Implement `ClusteringSession::from_source`: seed a session
  from an initial `DataSource` by running the full batch CPU pipeline (HNSW
  build with edge harvest, core distances, MST, label extraction) and
  populating all session state from the batch result. Requires 11.1.2. (See
  `docs/chutoro-design.md` §12.3)
  - Acceptance criteria: `session.labels()` matches `Chutoro::run()` on
    the same source (identical label vectors).
- [ ] 11.3.2. Implement `ClusteringSession::new_empty`: create an empty
  session with no initial data, ready to receive appends. Requires 11.1.2. (See
  `docs/chutoro-design.md` §12.3)

### 11.4. Differential testing and correctness validation

- [ ] 11.4.1. Build a differential test harness: seed a
  `ClusteringSession`, append a batch of new points, call `refresh()`, then run
  a full batch `Chutoro::run()` on the complete dataset and compare incremental
  labels against batch labels using ARI ≥ 0.95 and NMI ≥ 0.95 via the shared
  `chutoro_core::adjusted_rand_index` and
  `chutoro_core::normalized_mutual_information` helpers. Requires 11.2.1,
  11.3.1. (See `docs/chutoro-design.md` §12.7)
- [ ] 11.4.2. Add property-based differential tests using `proptest`:
  generate random append sequences across varied dataset sizes,
  dimensionalities, and cluster separations; verify that incremental results
  remain within quality bounds (ARI ≥ 0.90, NMI ≥ 0.90) for all generated
  fixtures. Requires 11.4.1. (See `docs/chutoro-design.md` §12.7)
- [ ] 11.4.3. Add regression benchmarks comparing incremental refresh
  wall-time against full batch `run()` for equivalent datasets using the
  `hnsw_ef_sweep` benchmark profile. Incremental refresh wall-time for a 1%
  append (relative to existing data) must be ≤ 0.7× the wall-time of a full
  batch run on the same final dataset. Requires 11.2.1, 11.3.1.
- [ ] 11.4.4. Add snapshot consistency tests: verify that
  `session.labels()` returned to concurrent readers during a refresh always
  returns a complete, internally consistent snapshot (correct length, valid
  label range) and never exposes partially updated state. Requires 11.2.2.

### 11.5. Documentation and API surface

- [ ] 11.5.1. Add Rustdoc documentation for `ClusteringSession`,
  `SessionConfig`, and all public methods including examples demonstrating the
  seed → append → refresh → read lifecycle. (See `docs/chutoro-design.md` §12.3)
- [ ] 11.5.2. Extend the CLI with an optional `--incremental` mode (or
  subcommand) that creates a `ClusteringSession`, reads an initial dataset,
  then accepts appended data from stdin or a secondary file and prints updated
  labels after each refresh. Requires 11.3.1. (See `docs/chutoro-design.md`
  §12.3, §12.6)
- [ ] 11.5.3. Document limitations of v1 incremental clustering in the
  design document: append-only, no stable cluster identity across snapshots,
  micro-batched rather than per-point, potential relabelling of existing points
  on refresh. (See `docs/chutoro-design.md` §12.2)

**Exit criteria:** incremental append + refresh produces clustering quality
(ARI/NMI ≥ 0.95 vs batch) on synthetic Gaussian datasets; incremental refresh
wall-time for a 1% append ≤ 0.7× full batch on the same final dataset (see
11.4.3); the differential test harness passes under property-based generation;
concurrent readers never observe partial snapshots.

______________________________________________________________________

## 12. Persistent snapshots and lineage

### 12.1. `ClusteringSnapshot` and probability metadata

- [ ] 12.1.1. Define `ClusteringSnapshot` struct carrying labels,
  optional per-point outlier/membership probabilities, per-cluster
  `ClusterStats`, snapshot version, and `SnapshotLineage` linking it to its
  predecessor. Gate the probabilities field behind a non-default
  `probabilities` Cargo feature. (See `docs/chutoro-design.md` §13.1)
  - Acceptance criteria: `ClusteringSnapshot` is constructible from the
    output of `extract_labels_from_mst`; snapshot version is monotonic;
    `labels.len()` equals the session's total point count.
- [ ] 12.1.2. Extend the hierarchy extraction pass (§6.2) to emit
  per-point stability-weighted membership scores when the `probabilities`
  feature is enabled. Propagate scores into `ClusteringSnapshot`. Requires
  12.1.1. (See `docs/chutoro-design.md` §13.1)
- [ ] 12.1.3. Compute `ClusterStats` for each cluster during snapshot
  construction: size, medoid (via `DataSource::distance`), up to k exemplars,
  cohesion (mean intra-cluster distance), separation (medoid distance to the
  nearest neighbouring cluster's medoid), noise ratio (requires
  `probabilities`), and nearest-cluster identifier. Requires 12.1.1. (See
  `docs/chutoro-design.md` §14.1)
  - Acceptance criteria: medoid is the point minimizing average
    intra-cluster distance; cohesion and separation are finite and
    non-negative for all non-singleton clusters.
- [ ] 12.1.4. Define a `VectorClusterStats` extension trait providing
  centroid computation for `DataSource` implementations that expose raw vectors
  via `row_slice()`. Keep the generic `ClusterStats` path medoid-only, avoiding
  Euclidean assumptions in non-metric contexts. Requires 12.1.3. (See
  `docs/chutoro-design.md` §14.1)

### 12.2. Checkpoint and restore

- [ ] 12.2.1. Define a self-describing binary checkpoint format with a
  version tag, capturing: HNSW index state (graph adjacency, entry point, level
  assignments, insertion sequence counter), MST edges, pending delta edges,
  core distances, current `ClusteringSnapshot`, and `SessionConfig`. (See
  `docs/chutoro-design.md` §13.2)
- [ ] 12.2.2. Implement `ClusteringSession::checkpoint` serializing
  session state to the binary format. Requires 12.2.1. (See
  `docs/chutoro-design.md` §13.2)
- [ ] 12.2.3. Implement `ClusteringSession::restore` deserializing from
  a checkpoint file and validating against a supplied `DataSource` (point
  count, metric descriptor). Return `SessionRestorationError` on mismatch.
  Requires 12.2.1. (See `docs/chutoro-design.md` §13.2)
  - Acceptance criteria: checkpoint → restore round-trip yields a session
    whose `labels()` output is identical to the pre-checkpoint state.
- [ ] 12.2.4. Add checkpoint integrity tests: corrupt each section of
  the binary format and verify that `restore` returns a specific error variant
  rather than panicking or producing silent corruption. Requires 12.2.3.

### 12.3. Stable cluster-identity matching

- [ ] 12.3.1. Implement bipartite Jaccard assignment between consecutive
  snapshots' cluster memberships. Clusters above a configurable overlap
  threshold (default 0.5) are matched and retain a persistent `ClusterId`.
  Unmatched new clusters receive fresh identifiers; unmatched old clusters are
  recorded as extinct. Gate behind an opt-in `SessionConfig` flag. (See
  `docs/chutoro-design.md` §13.3)
  - Acceptance criteria: on a no-change refresh (no new points), every
    cluster retains its previous `ClusterId`. On a small append (1%),
    ≥ 90% of clusters retain identity.
- [ ] 12.3.2. Add property-based tests verifying that identity matching
  is deterministic, that unmatched clusters receive monotonically increasing
  fresh identifiers, and that Jaccard scores agree with a brute-force oracle.
  Requires 12.3.1.

### 12.4. Structural diff API

- [ ] 12.4.1. Implement a `diff` function comparing two
  `ClusteringSnapshot` values and returning a `Vec<ClusterEvent>` carrying
  `Survive`, `Split`, `Merge`, `Birth`, and `Death` events. Reuse the bipartite
  Jaccard assignment from 12.3.1 for detection. Requires 12.3.1. (See
  `docs/chutoro-design.md` §13.4)
  - Acceptance criteria: a no-change refresh produces only `Survive`
    events. Synthetically removing half the points from one cluster and
    refreshing produces a `Split`. Merging two small clusters by
    inserting bridging points produces a `Merge`.
- [ ] 12.4.2. Add tests for `Birth` and `Death` events triggered by
  topic emergence and decay in a synthetic dataset with known breakpoints.
  Requires 12.4.1.

**Exit criteria:** snapshots carry labels, optional probabilities, and
per-cluster stats; checkpoint/restore round-trips succeed; stable cluster
identity survives a no-change refresh; structural diffs correctly classify
synthetic split, merge, birth, and death scenarios.

______________________________________________________________________

## 13. Local reclustering and diagnostics

### 13.1. Subset reclustering

- [ ] 13.1.1. Implement `recluster_subset` on `ClusteringSession`,
  accepting a slice of point indices and a `HierarchyConfig` and returning a
  `ClusteringSnapshot`. The method builds a local MST from edges incident on
  the specified indices (using the session's HNSW index for neighbour lookup)
  and runs hierarchy extraction. The global session state is not modified. (See
  `docs/chutoro-design.md` §14.2)
  - Acceptance criteria: `recluster_subset` on all points produces
    labels with ARI ≥ 0.95 against the session's current global labels.
    The global snapshot is unchanged after the call.
- [ ] 13.1.2. Add `recluster_cluster(cluster_id, config)` convenience
  wrapper resolving the cluster's member indices from the current snapshot and
  delegating to `recluster_subset`. Requires 13.1.1.
- [ ] 13.1.3. Add tests verifying that local reclustering of a known
  bimodal cluster produces two sub-clusters, and that local reclustering of a
  tight unimodal cluster returns a single cluster. Requires 13.1.1.

### 13.2. Graph and MST slice export

- [ ] 13.2.1. Implement `hnsw_neighbours` on `ClusteringSession`,
  accepting a point index and returning a `Vec<Neighbour>` containing a copy of
  the HNSW neighbours for that point at all layers. (See
  `docs/chutoro-design.md` §14.3)
- [ ] 13.2.2. Implement `mst_edges_for` on `ClusteringSession`,
  accepting a slice of point indices and returning a `Vec<MstEdge>` containing
  MST edges where at least one endpoint belongs to the specified index set.
  (See `docs/chutoro-design.md` §14.3)
- [ ] 13.2.3. Add tests verifying that exported slices are read-only
  copies (mutating the returned vectors does not affect session state) and that
  edge counts are consistent with the global MST. Requires 13.2.1, 13.2.2.

### 13.3. Diagnostic integration

- [ ] 13.3.1. Extend the CLI with a `diagnose` subcommand that loads a
  checkpoint (12.2.3) and prints per-cluster `ClusterStats` in a human-readable
  table, optionally formatted as JSON or CSV. Requires 12.1.3, 12.2.3.
- [ ] 13.3.2. Add Rustdoc documentation for `recluster_subset`,
  `recluster_cluster`, `hnsw_neighbours`, and `mst_edges_for` with examples
  demonstrating diagnostic workflows. Requires 13.1.1, 13.1.2, 13.2.1, 13.2.2.

**Exit criteria:** subset reclustering produces valid local snapshots without
mutating global state; graph and MST slice exports return consistent read-only
copies; the CLI `diagnose` command prints cluster statistics from a checkpoint.

______________________________________________________________________

## 14. Mutability and long-lived maintenance

### 14.1. Tombstone-based soft deletion

- [ ] 14.1.1. Add a `BitVec` (or equivalent compact set) to
  `ClusteringSession` tracking tombstoned point indices. Implement
  `ClusteringSession::delete(indices)` marking points as tombstoned. (See
  `docs/chutoro-design.md` §15.1)
  - Acceptance criteria: tombstoned points are excluded from the next
    snapshot's labels and `ClusterStats`. Tombstoned indices appear in
    the snapshot's lineage delta.
- [ ] 14.1.2. Extend the refresh path (§12.5) to handle tombstones:
  remove MST edges incident on tombstoned points before the Kruskal pass, mark
  neighbours' core distances as stale for recomputation, and produce a snapshot
  over the reduced (non-tombstoned) point set. Requires 14.1.1. (See
  `docs/chutoro-design.md` §15.2)
- [ ] 14.1.3. Add differential tests comparing tombstone-refresh
  results against a fresh batch `run()` on only the surviving points, using ARI
  ≥ 0.90 and NMI ≥ 0.90. Requires 14.1.2.

### 14.2. Compaction

- [ ] 14.2.1. Implement `compaction_recommended` on
  `ClusteringSession`, returning `true` when the tombstone ratio exceeds a
  configurable threshold (default 20%). (See `docs/chutoro-design.md` §15.3)
- [ ] 14.2.2. Implement `compact` on `ClusteringSession`: rebuild the
  HNSW index from scratch over surviving points with full edge harvesting,
  recompute all core distances, run a full `parallel_kruskal` and
  `extract_labels_from_mst` pass, and publish a new snapshot with a fresh
  lineage root. Requires 14.2.1. (See `docs/chutoro-design.md` §15.3)
  - Acceptance criteria: after compaction, `session.labels()` matches a
    fresh `Chutoro::run()` on the surviving points (identical labels).
    `tombstone_count` is zero. HNSW index contains only surviving
    points.
- [ ] 14.2.3. Add regression benchmarks comparing compaction wall-time
  against a full batch `run()` for the same surviving point count. The
  compaction path should complete within 1.2× of a fresh batch run. Requires
  14.2.2.

### 14.3. Memory-budget instrumentation

- [ ] 14.3.1. Expose session memory metrics behind the `metrics` feature
  flag: `session_point_count`, `session_live_point_count`,
  `session_tombstone_count`, `session_tombstone_ratio`,
  `session_mst_edge_count`, `session_pending_edge_count`,
  `session_snapshot_version`, `session_hnsw_memory_bytes`,
  `session_refresh_duration_seconds` (histogram), and
  `session_compaction_duration_seconds` (histogram). (See
  `docs/chutoro-design.md` §15.4)
- [ ] 14.3.2. Add integration tests verifying that metric values are
  consistent with session state after append, delete, refresh, and compact
  operations. Requires 14.3.1.

**Exit criteria:** tombstone deletion excludes points from snapshots without
immediate graph detachment; refresh after deletion produces ARI/NMI ≥ 0.90 vs a
fresh batch run on surviving points; compaction restores the index to a state
equivalent to a fresh build; memory metrics track all session lifecycle
operations.

______________________________________________________________________

## 15. Streaming text validation

### 15.1. Corpus recipe and embedding pipeline

- [ ] 15.1.1. Implement a deterministic synthetic text corpus generator
  producing documents with controlled reply-chain depth (1–10), configurable
  near-duplicate fraction (default 10%), periodic multi-topic digest messages,
  and topic-drift breakpoints. Seed and publish a manifest recording
  ground-truth topic labels and drift timestamps. (See `docs/chutoro-design.md`
  §16.1)
- [ ] 15.1.2. Define a reproducible text embedding recipe: pin a frozen
  Sentence-BERT checkpoint with a versioned model card, document preprocessing
  steps, and cache the resulting dense vectors alongside the corpus manifest.
  Alternatively, support a direct Levenshtein path over raw text for
  small-scale non-metric exercises. Requires 15.1.1. (See
  `docs/chutoro-design.md` §16.2)
- [ ] 15.1.3. Register the streaming text corpus as a benchmark dataset
  recipe in `chutoro-bench-datasets` (§10.1) with `smoke` and `cpu` matrix
  profiles. Requires 15.1.1, 15.1.2.

### 15.2. Streaming benchmark harness

- [ ] 15.2.1. Implement the streaming benchmark protocol: seed a
  `ClusteringSession` from an initial batch, append documents in micro-batches
  with periodic refreshes, and record per-refresh metrics. (See
  `docs/chutoro-design.md` §16.3)
- [ ] 15.2.2. Instrument the following per-refresh metrics: ARI/NMI
  against ground-truth topic labels, label churn (fraction of existing points
  whose assignment changed), append p95 latency, refresh wall-time, cluster
  stability (fraction of clusters surviving across consecutive snapshots via
  §13.3), and drift event quality (precision/recall of structural diff events
  against ground-truth drift breakpoints). Requires 12.3.1, 12.4.1. (See
  `docs/chutoro-design.md` §16.3)
- [ ] 15.2.3. Add CI integration running the streaming text benchmark
  on the `smoke` profile (1,000 initial documents, 200 appended) as a scheduled
  weekly job, publishing metric summaries alongside existing benchmark
  artefacts (§10.2). Requires 15.2.1, 15.2.2.

### 15.3. Acceptance and regression gating

- [ ] 15.3.1. Assert streaming-phase acceptance criteria: ARI ≥ 0.85
  and NMI ≥ 0.85 against ground-truth topic labels after the streaming phase
  completes; label churn ≤ 5% per refresh when no topic drift occurs in the
  corresponding append window; append p95 latency ≤ 2× mean single-point HNSW
  insertion time; structural diff events align with drift breakpoints at
  precision ≥ 0.7 and recall ≥ 0.7. Requires 15.2.1, 15.2.2, 15.2.3. (See
  `docs/chutoro-design.md` §16.4)
- [ ] 15.3.2. Add regression alerting: compare streaming benchmark
  metrics against stored baselines (§10.2.5) and surface regressions in CI.
  Requires 15.2.3.

**Exit criteria:** the streaming text corpus generator produces reproducible,
ground-truth-labelled documents; the streaming benchmark harness measures
quality, churn, latency, stability, and drift detection; acceptance criteria
are met on the `smoke` profile; CI publishes and compares metric summaries.

______________________________________________________________________

## Benchmark dataset suite

| Scale        | Dataset                      | Size / Dim.                                 | Labels / "clusters"                    | Why it's useful                                                                                                                            |
| ------------ | ---------------------------- | ------------------------------------------: | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Toy-Small    | `make_blobs` (synthetic)[^1] | configurable                                | exact cluster IDs                      | Sanity-check end-to-end behaviour; control separation, anisotropy, imbalance, and noise.                                                   |
| Small        | MNIST digits[^2]             | 70k × 784                                   | 10 classes                             | Easy, well-behaved baseline for Euclidean space; also available prepackaged for ANN-Benchmarks.                                            |
| Small        | Fashion-MNIST[^3]            | 70k × 784                                   | 10 classes                             | Slightly harder clusters than MNIST; drop-in replacement; HDF5 splits exist in ANN-Benchmarks.                                             |
| Small-Medium | CIFAR-10 / CIFAR-100[^4]     | 60k × 3×32×32                               | 10 / 100 classes                       | Labels enable testing both "coarse" and "fine" cluster granularity; embedding via a fixed CNN yields vectors.                              |
| Medium       | 20 Newsgroups[^5]            | 18,846 docs                                 | 20 topics                              | Text with clear topic labels; create vectors (e.g., TF-IDF, SBERT) then test cluster recovery.                                             |
| Medium-Large | RCV1-v2[^6]                  | 804,414 docs, 103 topics                    | multilabel topics                      | Large, messy, real text; multilabel enables probing overlapping clusters; scikit-learn fetcher provides canonical split.                   |
| Medium       | SNAP com-Amazon[^7]          | 334,863 nodes                               | product categories = communities       | Real graph with ground-truth communities for community/cluster evaluation after k-NN graph build.                                          |
| Medium       | SNAP com-DBLP[^7]            | 317,080 nodes                               | venue-based communities                | Overlapping scientific communities; good stress for cluster quality on graph structures.                                                   |
| Medium (bio) | PBMC 68k (10x Genomics)[^8]  | ~68k cells, ≫10k genes → 50d (PCA)          | cell-type labels                       | Classic scRNA-seq clustering benchmark with annotated cell types; strong test for high-dimensional distances.                              |
| Large (ANN)  | GloVe word vectors[^9]       | 1.18M × {25,50,100,200}                     | none (but lexical categories possible) | Covers angular distance regimes; turnkey HDF5 in ANN-Benchmarks.                                                                           |
| Large (ANN)  | SIFT1M[^10]                  | 1,000,000 × 128                             | nn ground truth (no classes)           | De-facto Euclidean Approximate Nearest Neighbour (ANN) baseline with exact k-NN ground truth; great for graph/MST quality via k-NN recall. |
| Large (ANN)  | GIST1M[^10]                  | 1,000,000 × 960                             | nn ground truth                        | Very high-dimensional Euclidean stress test; in ANN-Benchmarks.                                                                            |
| XL-Billion   | DEEP1B / BigANN[^10]         | up to 1B × 96 (Deep1B) / 128 (SIFT BigANN)  | nn ground truth                        | For scale limits, memory pressure, sharding; official subsets (1M/10M/100M) and GT available.                                              |

_Table 1: Benchmark datasets by scale, with dimensions, labels, and rationale._

Provenance notes:

- `make_blobs`: BSD-3 via scikit-learn.[^1]
- MNIST digits: Yann LeCun; permissive distribution.[^2]
- Fashion-MNIST: MIT licence from Zalando.[^3]
- CIFAR-10/100: MIT licence; provided by Krizhevsky et al.[^4]
- 20 Newsgroups: by Ken Lang; available via scikit-learn fetcher.[^5]
- RCV1-v2: Reuters licence; fetched with scikit-learn.[^6]
- SNAP com-Amazon: Stanford SNAP dataset under CC BY-SA.[^7]
- SNAP com-DBLP: Stanford SNAP dataset under CC BY-SA.[^7]
- PBMC 68k: 10x Genomics, CC BY 4.0.[^8]
- GloVe word vectors: Stanford NLP, public domain.[^9]
- SIFT1M: ANN-Benchmarks HDF5 package.[^10]
- GIST1M: ANN-Benchmarks HDF5 package.[^10]
- DEEP1B/BigANN: ANN-Benchmarks HDF5 package.[^10]

### How to use them (minimal ceremony)

- **With labels (MNIST/Fashion-MNIST/CIFAR/20NG/RCV1/SNAP/PBMC):**
  build k-NN/HNSW, run the MST/clusterer, and score with NMI/ARI vs. labels;
  also report k-NN recall@k vs. exact neighbours to measure graph fidelity.
  Recall@k is |P ∩ G_k|/k where G_k is the exact neighbour list (including
  ties) supplied with the dataset.
- **Without labels (SIFT/GIST/GloVe/DEEP1B):** rely on provided
  **exact neighbour ground truth** for recall@{1,10,100} and report graph
  metrics (conductance, connected-component purity) as proxies for cluster
  "shape." Conductance = cut(S, V−S)/min(vol(S), vol(V−S)) and
  connected-component purity is the dominant label count divided by the
  component size. ANN-Benchmarks HDF5 packs standardize splits and ground truth.

### Practical picks by "benchmark tier"

- **Smoke tests (minutes):** `make_blobs` (varied separation),
  MNIST/Fashion-MNIST.
- **Serious CPU micro/macro (hours):** 20 Newsgroups, RCV1-v2,
  SNAP com-Amazon/com-DBLP, SIFT1M.
- **Scale and memory (days):** GIST1M, GloVe-200d, DEEP10M/100M, and
  DEEP1B/BigANN for the most demanding scale.

[^1]: scikit-learn — make_blobs —
      <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html>
[^2]: MNIST database — <http://yann.lecun.com/exdb/mnist/>
[^3]: Fashion-MNIST — <https://github.com/zalandoresearch/fashion-mnist>
[^4]: CIFAR-10/100 — <https://www.cs.toronto.edu/~kriz/cifar.html>
[^5]: 20 Newsgroups —
      <https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset>
[^6]: RCV1-v2 —
      <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html>
[^7]: SNAP datasets — <https://snap.stanford.edu/data/>
[^8]: PBMC 68k —
      <https://support.10xgenomics.com/single-cell-gene-expression/datasets>
[^9]: GloVe vectors — <https://nlp.stanford.edu/projects/glove/>
[^10]: ANN-Benchmarks datasets —
       <https://github.com/erikbern/ann-benchmarks>
