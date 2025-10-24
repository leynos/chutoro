# chutoro — Phased Implementation Roadmap (CPU → GPU → Plugins)

_This roadmap references sections of the revised design document (e.g., §6.2)
rather than external sources._

______________________________________________________________________

## Phase 0 — Walking Skeleton (CPU-only, no dynamic plugins)

- [x] Create Cargo workspace with crates: `chutoro-core`, `chutoro-cli`,
  `chutoro-providers/dense`, `chutoro-providers/text`. (See §4)
- [x] Define public `DataSource` trait with `len()`, `name()`, `distance(i,j)`,
  and a defaulted `distance_batch(pairs, out)` for forward-compatibility. (See
  §10.2)
- [x] Implement `ChutoroBuilder` and `Chutoro::run<D: DataSource>`
  orchestration API. (See §10.1)
- [x] Implement `DenseMatrixProvider` that packs Parquet/Arrow
  `FixedSizeList<Float32,D>` into a contiguous row‑major `Vec<f32>`; reject
  ragged/null rows. (See §5, §10.6)
- [x] Implement `TextProvider` (one UTF‑8 string per line) to exercise
  non‑metric distances (e.g., Levenshtein). (See §1.3)
- [x] Add `cosine` and `euclidean` distance implementations in `chutoro-core`
  with optional precomputed norms for cosine. (See §3.1)
- [x] Ship a minimal CLI: `chutoro run parquet <path> --column features ...`
  and `chutoro run text <path> --metric levenshtein`. Shared flags include
  `--min-cluster-size <usize>` defaulting to `5` and `--name <string>`.
  `--name` defaults to the file stem via UTF-8 lossy conversion for visibility.
  (See §10)
- [x] Add structured logging via `tracing` + `tracing-subscriber`; bridge the
  `log` crate via `tracing-log`; replace manual prints and initialize logging
  in the CLI (e.g.,
  `tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env())`,
  human and JSON formats, and span IDs). (See §10.5)
- [x] Define error taxonomy; adopt `thiserror` for a unified `ChutoroError` in
  public APIs and use `anyhow` in binaries; return `Result<T, ChutoroError>`
  with stable, documented error codes. (See §10.6)
- [ ] Establish CI (fmt, clippy, test), feature gates (`cpu`, `gpu` off by
  default), and reproducible toolchain (`rust-toolchain.toml`). (See §11)

**Exit criteria:** compile+run on small datasets; end‑to‑end smoke test
produces a cluster assignment (even with stubbed internals).

______________________________________________________________________

## Phase 1 — CPU HNSW + Candidate Edge Harvest + Parallel Kruskal MST

- [ ] Implement CPU HNSW insertion/search with Rayon; two‑phase locking (`read`
  for search → `write` for insert) on a shared graph. (See §6.1, §6.2)
- [ ] Introduce a `DistanceCache` backed by `dashmap` to avoid recomputing
  distances across threads during HNSW insertion. (See §6.2, §10.6)
  - Key: normalise to `(min(i,j), max(i,j))`; encode metric and its parameters
    (e.g., cosine with/without pre-norms) in the key.
  - Value: `f32` distance; NaN policy: do not cache NaN; propagate an error and
    log at WARN.
  - Bounds: enforce `max_entries` via LRU eviction (optional TTL); document
    defaults and configuration knobs.
  - Metrics: expose `distance_cache_hits`, `distance_cache_misses`,
    `distance_cache_evictions`, and `distance_cache_lookup_latency_histogram`
    via the `metrics` crate. The first three are monotonic counters; lookup
    latency is a histogram of seconds. These metrics are emitted only when the
    `metrics` feature flag is enabled and are documented for external
    consumption; emit `tracing` spans for hot paths.
  - Concurrency: require `Send + Sync`; forbid iteration on hot paths; avoid
    holding HNSW write locks while updating the cache.
  - Determinism: ensure neighbour selection is unchanged under fixed seeds by
    using a deterministic tie-break: on equal distances prefer the lower item
    id; when ids match fall back to the insertion sequence number. Eviction
    must be time-independent in tests. This rule is used in all builds and
    tests to guarantee stable outputs under fixed seeds.
- [ ] During insertion, capture candidate edges `(u,v,w)` discovered by HNSW;
  accumulate via Rayon `map` → `reduce` into a global edge list. (See §6.2)
- [ ] Implement parallel Kruskal: parallel sort of edges, concurrent union‑find
  for cycle checks. (See §3.2, §6.2)
- [ ] Implement hierarchy extraction from MST (sequential to start): build
  condensed tree and stability scoring; output flat labels. (See §1.2, §6.2)
- [ ] Add deterministic tests on tiny datasets (exact mutual-reachability path)
  to sanity-check hierarchy logic. (See §11)
- [ ] Add functional tests comparing ARI/NMI with HDBSCAN/HNSW baselines on
  small public sets. (See §11)
- [ ] Introduce property-based generators for HNSW using `proptest` +
  `test-strategy`; cover uniform, clustered, manifold, and duplicate vectors
  plus configuration sampling. (See property-testing-design §2.1)
- [ ] Surface HNSW structural invariant checkers (layer consistency, degree
  bounds, reachability, bidirectional links) callable from properties. (See
  property-testing-design §2.2)
- [ ] Add HNSW search correctness property comparing index results against a
  brute-force oracle with a configurable recall threshold and timing capture.
  (See property-testing-design §2.3.1)
- [ ] Add stateful HNSW mutation property that exercises add/delete/
  reconfigure sequences and revalidates invariants after each operation. (See
  property-testing-design §2.3.2)
- [ ] Add insertion idempotency property for HNSW ensuring repeated duplicate
  insertions leave graph state unchanged. (See property-testing-design §2.3.3)
- [ ] Implement composite graph strategies (random, scale-free, lattice,
  disconnected) for candidate edge harvest testing. (See
  property-testing-design §3.1)
- [ ] Add candidate edge harvest property suite covering determinism, degree
  ceilings, connectivity preservation, and reverse nearest neighbour uplift
  metrics. (See property-testing-design §3.2)
- [ ] Build parallel Kruskal property suite: compare against sequential
  oracle, enforce acyclicity/connectivity/edge-count invariants, and rerun jobs
  to detect race-induced non-determinism. (See property-testing-design §4)
- [ ] Integrate property-based suites into CI with a path-filtered PR job (250
  cases, 10 minute timeout) and scheduled weekly job (25,000 cases,
  `fork = true`, `PROGTEST_CASES` env var). (See property-testing-design §5)

**Exit criteria:** 100k × D vectors complete in minutes on CPU; memory bounded
≈ `n*M` edges; ARI/NMI within acceptable band vs reference. (See §6.2)

______________________________________________________________________

## Phase 2 — Benchmarking & Profiling (CPU)

- [ ] Create `chutoro-benches` with Criterion harness for HNSW build, edge
  harvest, MST, and extraction timings. (See §11)
- [ ] Add dataset fixtures (synthetic Gaussians, rings; text strings) and
  record footprints vs `M∈{8,12,16,24}`. (See §1.3)
- [ ] Implement memory guards (`--max-bytes`) and clear failure messages. (See
  §5)
- [ ] Add CPU distance kernels using `std::simd` with AVX2/AVX-512
  specializations; make `distance_batch` the default HNSW scoring path. (See
  §6.3)
- [ ] Restructure HNSW neighbour evaluation to use packed indices and an SoA
  layout, prefetching blocks and scoring outside the write lock. (See §6.3)
- [ ] Vectorize edge-weight transforms and candidate filtering before
  union-find in parallel Kruskal; keep union-find cache-friendly. (See §6.3)
- [ ] Introduce `DensePointView<'a>` for aligned SoA access with a scalar
  fallback. (See §6.3)
- [ ] Gate SIMD backends behind `simd_avx2`, `simd_avx512`, and `simd_neon`
  features with CPUID runtime dispatch. (See §6.3)
  - Use `is_x86_feature_detected!`/platform equivalents and one-time
    function-pointer patching to avoid hot-path branching.
  - Define NaN and other non-finite treatment for reductions to ensure
    CPU/GPU parity.
  - Guarantee 64-byte alignment and lane-multiple padding for
    `DensePointView<'a>`; zero-pad tails.
- [ ] Benchmark Euclidean/cosine kernels, neighbour scoring, and batched
  `distance_batch` versus scalar `distance`; bucket candidate sizes to confirm
  SIMD gains. (See §6.3, §11)

**Exit criteria:** stable baseline numbers to compare against GPU phases;
documented CPU hot spots.

______________________________________________________________________

## Phase 3 — GPU MST Offload (Borůvka via rust‑cuda HAL)

- [ ] Add GPU HAL to `chutoro-core` with an execution‑path selector. (See §7.1)
- [ ] Implement `chutoro-backend-cuda` using `cust`/`cudarc`, `cuda_std`,
  `rustc_codegen_nvvm`, `cuda_builder`; gate behind `backend-cuda`. (See §7.1)
- [ ] Stub `chutoro-backend-cubecl` and optional `chutoro-backend-sycl` crates
  with features `backend-portable` and `backend-sycl`. (See §7.1)
- [ ] Define device data structures: global edge list, DSU parent array, MST
  output buffer. (See §8.2)
- [ ] Implement Kernel 1: per‑component (or per‑vertex) min outgoing edge
  selection. (See §8.2)
- [ ] Implement Kernel 2: parallel union (atomic CAS) and component compaction;
  host loop rounds until one component remains. (See §8.2)
- [ ] Add host‑side adapter: copy candidate edges once to device, run Borůvka,
  copy MST back. (See §9.1)
- [ ] Gate backends behind features (`backend-cuda`, `backend-portable`,
  `backend-sycl`); fall back cleanly to CPU when unavailable. (See §7.1, §11)
- [ ] Verify correctness vs CPU Kruskal on random graphs; benchmark speedup.
  (See §11)

**Exit criteria:** MST wall‑time reduced significantly vs CPU on ≥1e6 edges;
identical MST (or accepted tie‑break equivalence) to CPU implementation.

______________________________________________________________________

## Phase 4 — Hybrid HNSW Distance Offload (CPU traversal, GPU distance batches)

- [ ] Implement CUDA distance kernel operating on row‑major `f32` matrix (one
  thread per candidate). (See §8.1)
- [ ] Add host orchestration hook in HNSW insertion to batch candidate distance
  evaluations to GPU. (See §8.1)
- [ ] Introduce a small pinned host/device ring buffer for batched copies. (See
  §9.2)
- [ ] Measure build‑time reduction vs CPU‑only HNSW; ensure identical neighbour
  selections under deterministic seeds. (See §8.1, §11)

**Exit criteria:** measurable reduction in HNSW insertion time on large batches
without altering clustering materially.

______________________________________________________________________

## Phase 5 — Async Orchestration & Streams

- [ ] Introduce multi‑stream pipeline: Stream 1 (mem ops), Stream 2 (HNSW
  distance), Stream 3 (MST). (See §9.2)
- [ ] Overlap dataset upload with initial compute; enforce dependencies via
  events/stream waits. (See §9.2)
- [ ] Keep large buffers resident on device; copy back only final
  labels/hierarchy. (See §9.1)
- [ ] Add tracing/logging around stream scheduling to validate overlap. (See
  §9.2)

**Exit criteria:** observed kernel/transfer overlap; reduced end‑to‑end
wall‑time vs Phase 4.

______________________________________________________________________

## Phase 6 — Dynamic Plugin Loader (C‑ABI v‑table handshake)

- [ ] Freeze `chutoro_v1` #[repr(C)] v‑table with `abi_version`, `caps`,
  `state`, and function pointers; include optional `distance_batch`. (See §5.3)
- [ ] Add mandatory `destroy` callback to the v‑table and ensure the
  `PluginManager` calls it when unloading plugins. (See §5.3)
  - ABI: `extern "C" fn destroy(state: *mut c_void)`; no panics across FFI.
  - Ordering: drop all host‑side wrappers/handles; call `destroy`, wait for it
    to return; then unload the dynamic library.
  - Idempotency: `destroy` must be safe if called multiple times (idempotent);
    the host calls it at most once in normal operation. Document behaviour if
    the plugin reports errors.
  - Quiescence: ensure all in-flight callbacks complete and any plugin-spawned
    threads are joined/cancelled before calling `destroy`; forbid re-entry
    after unload.
  - Timeouts: enforce a configurable quiescence timeout (e.g., 30s) and log at
    WARN on expiry before proceeding to unload.
  - Safety: add logs at INFO with plugin name and timing.
  - Tests: load→use→quiesce→destroy→unload cycle under leak detectors and
    ASAN/UBSAN.
- [ ] Implement `PluginManager` using `libloading` to locate `_plugin_create`,
  validate `abi_version`, wrap as safe `DataSource`. (See §5.2, §5.3)
- [ ] Ship example plugin `chutoro-plugin-csv` and `chutoro-plugin-parquet`;
  document build and loading. (See §5)
- [ ] Add capability flags (`HAS_DISTANCE_BATCH`, `HAS_DEVICE_VIEW`,
  `HAS_NATIVE_KERNELS`) for GPU‑aware providers. (See §5.3, §7.1)
- [ ] Fuzz and harden FFI boundaries (lengths, nulls, lifetime ownership);
  ensure host remains robust on plugin failure. (See §5.1)

**Exit criteria:** load/unload plugins at runtime; parity with statically
linked providers; safe failure semantics.

______________________________________________________________________

## Phase 7 — DataFusion + object_store Provider (Optional)

- [ ] Implement `DataFusionProvider` to execute SQL predicates/projections and
  materialise to `DenseMatrixProvider`. (See §5, §10.6)
- [ ] Support `s3://`, `gs://`, `azure://` via `object_store`; add sampling
  options for large tables. (See §5)
- [ ] Document schema contract (single `features` `FixedSizeList<Float32,D>`
  column) and error messages. (See §5)

**Exit criteria:** users can point at local/cloud Parquet and run queries that
materialise into chutoro’s fast path.

______________________________________________________________________

## Phase 8 — Documentation, Examples, and Release

- [ ] Write `README` with architecture diagram and quickstart; add API docs via
  `cargo doc`. (See §4, §10)
- [ ] Provide end‑to‑end examples: numeric embeddings and string clustering.
  (See §1.3)
- [ ] Publish `v0.1.0` (CPU baseline), feature‑gated GPU, and example plugin
  crates; tag and create release notes with benchmark tables. (See §11)

**Exit criteria:** documented, installable release; examples reproduce reported
benchmarks.

______________________________________________________________________

## Phase 9 — Hardening & Nice‑to‑Haves (Post‑v0.1)

- [ ] Parallelise parts of hierarchy extraction if profiling justifies. (See
  §8.3)
- [ ] Add metrics/telemetry hooks (timings, memory) behind a feature flag. (See
  §11)
- [ ] Evaluate `abi_stable`/`stabby` as an alternative to C‑ABI for
  Rust‑to‑Rust plugins. (See §5.2)

**Exit criteria:** measurable incremental gains; stable ABI story; clean
telemetry for future tuning.

______________________________________________________________________

## Benchmark dataset suite

| Scale          | Dataset                      | Size / Dim.                                 | Labels / “clusters”                    | Why it’s useful                                                                                                          |
| -------------- | ---------------------------- | ------------------------------------------: | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Toy–Small      | `make_blobs` (synthetic)[^1] | configurable                                | exact cluster IDs                      | Sanity-check end-to-end behaviour; control separation, anisotropy, imbalance, and noise.                                 |
| Small          | MNIST digits[^2]             | 70k × 784                                   | 10 classes                             | Easy, well-behaved baseline for Euclidean space; also available prepackaged for ANN-Benchmarks.                          |
| Small          | Fashion-MNIST[^3]            | 70k × 784                                   | 10 classes                             | Slightly harder clusters than MNIST; drop-in replacement; HDF5 splits exist in ANN-Benchmarks.                           |
| Small–Medium   | CIFAR-10 / CIFAR-100[^4]     | 60k × 3×32×32                               | 10 / 100 classes                       | Labels enable testing both “coarse” and “fine” cluster granularity; embedding via a fixed CNN yields vectors.            |
| Medium         | 20 Newsgroups[^5]            | 18,846 docs                                 | 20 topics                              | Text with clear topic labels; create vectors (e.g., TF-IDF, SBERT) then test cluster recovery.                           |
| Medium–Large   | RCV1-v2[^6]                  | 804,414 docs, 103 topics                    | multilabel topics                      | Large, messy, real text; multilabel enables probing overlapping clusters; scikit-learn fetcher provides canonical split. |
| Medium         | SNAP com-Amazon[^7]          | 334,863 nodes                               | product categories = communities       | Real graph with ground-truth communities for community/cluster evaluation after k-NN graph build.                        |
| Medium         | SNAP com-DBLP[^7]            | 317,080 nodes                               | venue-based communities                | Overlapping scientific communities; good stress for cluster quality on graph structures.                                 |
| Medium (bio)   | PBMC 68k (10x Genomics)[^8]  | ~68k cells, ≫10k genes → 50d (PCA)          | cell-type labels                       | Classic scRNA-seq clustering benchmark with annotated cell types; strong test for high-dimensional distances.            |
| Large (ANN)    | GloVe word vectors[^9]       | 1.18M × {25,50,100,200}                     | none (but lexical categories possible) | Covers angular distance regimes; turnkey HDF5 in ANN-Benchmarks.                                                         |
| Large (ANN)    | SIFT1M[^10]                  | 1,000,000 × 128                             | nn ground truth (no classes)           | De-facto Euclidean ANN baseline with exact k-NN ground truth; great for graph/MST quality via k-NN recall.               |
| Large (ANN)    | GIST1M[^10]                  | 1,000,000 × 960                             | nn ground truth                        | Very high-dimensional Euclidean stress test; in ANN-Benchmarks.                                                          |
| XL–Billion     | DEEP1B / BigANN[^10]         | up to 1B × 96 (Deep1B) / 128 (SIFT BigANN)  | nn ground truth                        | For scale limits, memory pressure, sharding; official subsets (1M/10M/100M) and GT available.                            |

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

- **With labels (MNIST/Fashion-MNIST/CIFAR/20NG/RCV1/SNAP/PBMC):** build
  k-NN/HNSW, run the MST/clusterer, and score with NMI/ARI vs. labels; also
  report k-NN recall@k vs. exact neighbours to measure graph fidelity. Recall@k
  is |P ∩ G_k|/k where G_k is the exact neighbour list (including ties)
  supplied with the dataset.
- **Without labels (SIFT/GIST/GloVe/DEEP1B):** rely on provided **exact
  neighbour ground truth** for recall@{1,10,100} and report graph metrics
  (conductance, connected-component purity) as proxies for cluster “shape.”
  Conductance = cut(S, V−S)/min(vol(S), vol(V−S)) and connected-component
  purity is the dominant label count divided by the component size.
  ANN-Benchmarks HDF5 packs standardize splits and ground truth.

### Practical picks by “benchmark tier”

- **Smoke tests (minutes):** `make_blobs` (varied separation),
  MNIST/Fashion-MNIST.
- **Serious CPU micro/macro (hours):** 20 Newsgroups, RCV1-v2,
  SNAP com-Amazon/com-DBLP, SIFT1M.
- **Scale & memory (days):** GIST1M, GloVe-200d, DEEP10M/100M, and
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
[^10]: ANN-Benchmarks datasets — <https://github.com/erikbern/ann-benchmarks>
