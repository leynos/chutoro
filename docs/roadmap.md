# chutoro — Phased Implementation Roadmap (CPU → GPU → Plugins)

_This roadmap references sections of the revised design document (e.g., §6.2)
rather than external sources._

______________________________________________________________________

## Phase 0 — Walking Skeleton (CPU-only, no dynamic plugins)

- [ ] Create Cargo workspace with crates: `chutoro-core`, `chutoro-cli`,
  `chutoro-providers/dense`, `chutoro-providers/text`. (See §4)
- [ ] Define public `DataSource` trait with `len()`, `name()`, `distance(i,j)`,
  and a defaulted `distance_batch(pairs, out)` for forward-compatibility. (See
  §10.2)
- [ ] Implement `ChutoroBuilder` and `Chutoro::run<D: DataSource>`
  orchestration API. (See §10.1)
- [ ] Implement `DenseMatrixProvider` that packs Parquet/Arrow
  `FixedSizeList<Float32,D>` into a contiguous row‑major `Vec<f32>`; reject
  ragged/null rows. (See §5, §10.4)
- [ ] Implement `TextProvider` (one UTF‑8 string per line) to exercise
  non‑metric distances (e.g., Levenshtein). (See §1.3)
- [ ] Add `cosine` and `euclidean` distance implementations in `chutoro-core`
  with optional precomputed norms for cosine. (See §3.1)
- [ ] Ship a minimal CLI: `chutoro run parquet <path> --column features ...`
  and `chutoro run text <path> --metric levenshtein`. (See §10)
- [ ] Establish CI (fmt, clippy, test), feature gates (`cpu`, `gpu` off by
  default), and reproducible toolchain (`rust-toolchain.toml`). (See §11)

**Exit criteria:** compile+run on small datasets; end‑to‑end smoke test
produces a cluster assignment (even with stubbed internals).

______________________________________________________________________

## Phase 1 — CPU HNSW + Candidate Edge Harvest + Parallel Kruskal MST

- [ ] Implement CPU HNSW insertion/search with Rayon; two‑phase locking (`read`
  for search → `write` for insert) on a shared graph. (See §6.1, §6.2)
- [ ] During insertion, capture candidate edges `(u,v,w)` discovered by HNSW;
  accumulate via Rayon `map` → `reduce` into a global edge list. (See §6.2)
- [ ] Implement parallel Kruskal: parallel sort of edges, concurrent union‑find
  for cycle checks. (See §3.2, §6.2)
- [ ] Implement hierarchy extraction from MST (sequential to start): build
  condensed tree and stability scoring; output flat labels. (See §1.2, §6.2)
- [ ] Add deterministic tests on tiny datasets (exact mutual‑reachability path)
  to sanity‑check hierarchy logic. (See §11)
- [ ] Add functional tests comparing ARI/NMI with HDBSCAN/HNSW baselines on
  small public sets. (See §11)

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
- [ ] Optimise hot loops (SIMD, cache of norms, small‑vecs) guided by profiles.
  (See §6.1)

**Exit criteria:** stable baseline numbers to compare against GPU phases;
documented CPU hot spots.

______________________________________________________________________

## Phase 3 — GPU MST Offload (Borůvka on CUDA via rust‑cuda)

- [ ] Introduce `chutoro-gpu` crate with `cust`, `cuda_std`,
  `rustc_codegen_nvvm`, `cuda_builder`. (See §7)
- [ ] Define device data structures: global edge list, DSU parent array, MST
  output buffer. (See §8.2)
- [ ] Implement Kernel 1: per‑component (or per‑vertex) min outgoing edge
  selection. (See §8.2)
- [ ] Implement Kernel 2: parallel union (atomic CAS) and component compaction;
  host loop rounds until one component remains. (See §8.2)
- [ ] Add host‑side adapter: copy candidate edges once to device, run Borůvka,
  copy MST back. (See §9.1)
- [ ] Gate behind `--features gpu`; fall back cleanly to CPU when unavailable.
  (See §7, §11)
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
- [ ] Implement `PluginManager` using `libloading` to locate `_plugin_create`,
  validate `abi_version`, wrap as safe `DataSource`. (See §5.2, §5.3)
- [ ] Ship example plugin `chutoro-plugin-csv` and `chutoro-plugin-parquet`;
  document build and loading. (See §5)
- [ ] Add capability flags (`HAS_BATCH`, `HAS_DEVICE_VIEW`) for future
  GPU‑aware providers. (See §5.3)
- [ ] Fuzz and harden FFI boundaries (lengths, nulls, lifetime ownership);
  ensure host remains robust on plugin failure. (See §5.1)

**Exit criteria:** load/unload plugins at runtime; parity with statically
linked providers; safe failure semantics.

______________________________________________________________________

## Phase 7 — DataFusion + object_store Provider (Optional)

- [ ] Implement `DataFusionProvider` to execute SQL predicates/projections and
  materialise to `DenseMatrixProvider`. (See §5, §10.4)
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

- [ ] Add SIMD/AVX distance kernels on CPU and optional SoA transpose on device
  for coalesced loads. (See §9.1)
- [ ] Parallelise parts of hierarchy extraction if profiling justifies. (See
  §8.3)
- [ ] Add metrics/telemetry hooks (timings, memory) behind a feature flag. (See
  §11)
- [ ] Evaluate `abi_stable`/`stabby` as an alternative to C‑ABI for
  Rust‑to‑Rust plugins. (See §5.2)

**Exit criteria:** measurable incremental gains; stable ABI story; clean
telemetry for future tuning.
