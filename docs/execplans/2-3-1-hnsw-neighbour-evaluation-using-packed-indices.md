# Restructure HNSW neighbour evaluation: packed indices, SoA scoring, and prefetch (2.3.1)

This ExecPlan (execution plan) is a living document. The sections
`Constraints`, `Tolerances`, `Risks`, `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work
proceeds.

Status: DRAFT

Roadmap item: 2.3.1 (Phase 2.3, Hot-path optimizations). See `docs/roadmap.md`
lines 319-321 and `docs/chutoro-design.md` §6.3 (lines 887-963).

## Purpose / big picture

chutoro builds a CPU Hierarchical Navigable Small World (HNSW) graph to harvest
candidate edges for clustering. The hot path is *neighbour evaluation*: for each
node visited during search and insertion, the engine reads that node's
neighbour list and computes the distance from a query point to every neighbour,
then keeps the closest. Roadmap item 2.3.1 asks us to make that evaluation
cache-friendly and lock-friendly: operate on *packed* candidate indices and a
*structure-of-arrays* (SoA) coordinate layout, *prefetch* upcoming coordinate
blocks to hide memory latency, and compute scores *outside the write lock* so
concurrent readers are not blocked while distances are computed.

After this change a developer can:

1. Run `make bench` (or the focused Criterion harness `cargo bench -p
   chutoro-benches --bench neighbour_scoring`) and see a microbenchmark that
   buckets neighbour-set scoring by candidate-set size, with a documented
   before/after comparison captured via `critcmp`.
2. Run the `hyperfine`-based whole-binary comparison script
   (`scripts/bench-neighbour-scoring.sh`) and observe end-to-end HNSW build
   wall-time for the baseline binary versus the optimised binary on a fixed
   synthetic dataset, with statistical output.
3. Read `docs/chutoro-design.md` §6.3 and `docs/developers-guide.md` and find
   the SoA/packed-index/prefetch contract documented, including the invariant
   that distance scoring never runs while the graph write lock is held, and the
   regression test that proves it.

Crucially, clustering output is **unchanged**: the optimisation is
behaviour-preserving under fixed seeds. Success is *faster or equal* neighbour
scoring with *identical* neighbour selection, proven by the existing search
correctness, idempotency, and backend-parity suites plus a new write-lock
invariant guard.

### Honest scope note (read before estimating)

Much of 2.3.1's headline intent was already delivered by Phase 2.2 (SIMD
distance kernels) and the existing two-phase locking design. Specifically:

- **Scoring already happens outside the write lock.** `InsertionPlanner::plan`
  (`chutoro-core/src/hnsw/insert/planner.rs:78`) scores candidates under the
  *read* lock; `CpuHnsw::score_trim_jobs`
  (`chutoro-core/src/hnsw/cpu/trim.rs:65`) scores trim candidates with **no
  lock held**, in parallel via Rayon, between the two write-lock phases of
  `insert_with_collector` (`chutoro-core/src/hnsw/cpu/mod.rs:364-419`). No
  distance computation occurs under the write lock today.
- **SoA scoring already exists** as an adapter detail: `DenseMatrixProvider`
  packs query-centric candidate batches into a 64-byte-aligned, 16-lane-padded,
  dimension-major SoA `DensePointView`
  (`chutoro-providers/dense/src/simd/point_view.rs`) inside `batch_distances`,
  then runs AVX-512/AVX2/Neon/scalar kernels.
- **Candidate ids are already packed** contiguous `&[usize]` slices at the
  `DataSource::batch_distances(query, candidates)` port.

Therefore this plan does **not** re-architect locking or invent a new SoA type.
The genuine, measurable residual deltas are:

- **D1 — Provider-internal prefetch** of upcoming candidate coordinate blocks,
  `cfg`-gated to x86_64 and behind a non-default `simd_prefetch` feature, with
  a no-op fallback elsewhere.
- **D2 — SoA packing-buffer reuse** so `DensePointView::from_row_indices` stops
  allocating a fresh `PackedSoaStorage` per batch in the hot path.
- **D3 — Core-side hot-path allocation hygiene**: reuse scratch buffers for the
  per-node candidate list and distance output in `LayerSearcher::search_layer`
  and the trim path, keeping candidate ids packed and contiguous.
- **D4 — Write-lock invariant guard**: a deterministic regression test proving
  distance scoring never runs while the graph write lock is held, plus
  documentation of the invariant.
- **D5 — Benchmark evidence**: a Criterion neighbour-scoring microbenchmark
  bucketed by candidate-set size and a `hyperfine` whole-binary before/after
  comparison, proving the win — or honestly reporting its absence at a go/no-go
  gate.

If measurement at Milestone 0 shows a delta is not worth its complexity (very
plausible for D1 prefetch, per the research below), that delta is dropped and
the decision recorded, not forced through.

## Constraints

Hard invariants that must hold throughout implementation. Violation requires
escalation, not a workaround.

1. **Behaviour preservation.** Clustering output and HNSW neighbour selection
   must be byte-for-byte identical under fixed seeds, before and after. The
   deterministic tie-break (distance, then lower item id, then insertion
   sequence; see `docs/roadmap.md` 1.1.2 and `chutoro-core/src/hnsw/search.rs`
   `compare_neighbours`) must be preserved exactly.
2. **Hexagonal boundary.** The SoA coordinate layout, SIMD kernels, prefetch
   intrinsics, and `DensePointView` must remain an *adapter* detail private to
   `chutoro-providers/dense` (`pub(crate)`). They must **not** be exposed across
   the crate boundary into `chutoro-core`. Domain/policy logic (which
   candidates to score, packed-index iteration, deterministic ordering,
   lock discipline) stays in `chutoro-core`; coordinate packing, prefetch, and
   vectorisation stay behind the `DataSource` port. See the
   `hexagonal-architecture` skill.
3. **Public API stability.** The public `DataSource` trait
   (`chutoro-core/src/datasource.rs`), `CpuHnsw` public methods, and the CLI
   surface must remain source-compatible. Any new trait method must be a
   *defaulted* (non-breaking) addition, and adding one triggers the Tolerances
   escalation below.
4. **No scoring under the write lock.** Distance computation must never run
   while a `RwLockWriteGuard<Graph>` is held. This is both a constraint and a
   tested invariant (D4).
5. **MSRV and toolchain.** Stable Rust only on the default build path; MSRV
   `1.89.0`, pinned toolchain `1.93.1` (`rust-toolchain.toml`). `_mm_prefetch`
   is stable since 1.27 and safe to call, so D1 needs no `unsafe` for the call
   itself and no nightly. The optional nightly `std::simd` path is out of scope.
6. **Quality gates.** `make check-fmt`, `make lint` (clippy `-D warnings`), and
   `make test` must pass before every commit. Lints must not be silenced except
   as a tightly-scoped, justified last resort (see `AGENTS.md`).
7. **File size.** No source file may exceed 400 lines (`AGENTS.md`); extract
   helpers and colocate by feature.

## Tolerances (exception triggers)

Stop and escalate (do not work around) when:

1. **Scope.** A milestone requires changing more than ~6 source files or ~400
   net lines of code beyond what that milestone's plan names.
2. **Interface.** Any *non-defaulted* change to the public `DataSource` trait,
   `CpuHnsw`, or CLI is required. (Adding D2b's optional defaulted
   `batch_distances_into` is pre-authorised but must still be flagged in the
   `Decision Log`.)
3. **Dependencies.** Any new external crate is required. (None is anticipated;
   prefetch uses `core::arch`.)
4. **Behaviour drift.** Any existing test that asserts neighbour selection,
   recall, parity, or determinism changes its expected output. This must never
   happen; if it does, stop immediately.
5. **No measurable win.** If Milestone 0 plus a milestone's own benchmark shows
   that milestone produces no statistically significant improvement (Criterion
   reports the change within noise, or a regression), stop and present the
   go/no-go decision rather than merging speculative complexity.
6. **Iterations.** If a milestone's tests still fail after 3 focused attempts,
   stop and escalate.
7. **Prefetch regression.** If enabling `simd_prefetch` regresses any bucket of
   the neighbour-scoring microbenchmark on the target host, abandon D1 and
   record the result (the research predicts this is likely).

## Risks

1. Risk: **Prefetch yields no win or a regression.**
   Severity: medium. Likelihood: high.
   Mitigation: D1 is a prototyping milestone behind a non-default feature with
   an explicit go/no-go gate; the default build is unaffected if it is dropped.
   The research (Lemire; parallel-rust-cpp; Algorithmica) shows software
   prefetch "fails more often than not" against modern hardware prefetchers and
   deep out-of-order windows, and can perturb the optimizer (register
   spilling).
2. Risk: **The allocation deltas (D2/D3) are dwarfed by distance arithmetic** so
   removing them shows no wall-time change.
   Severity: medium. Likelihood: medium.
   Mitigation: Milestone 0 measures allocation cost first (via a Criterion
   microbenchmark and an allocation-count assertion); proceed only where the
   cost is real.
3. Risk: **A scratch-buffer reuse subtly changes ordering or reuses stale data**
   across iterations, breaking determinism.
   Severity: high. Likelihood: low.
   Mitigation: clear/resize scratch explicitly each use; gate every change
   behind the existing search-correctness, idempotency, mutation, and
   backend-parity property suites plus new equality tests.
4. Risk: **Per-node batches are too small to benefit from SoA/SIMD/prefetch.**
   The best-first search scores one node's neighbour list at a time (≤ `2M`
   ids), so the prefetch/SIMD window is bounded by `M`.
   Severity: medium. Likelihood: medium.
   Mitigation: bucket the microbenchmark by candidate-set size to find the
   crossover; document where SoA wins and where scalar fallback is retained
   (the provider already prefers scalar for tiny batches).
5. Risk: **Benchmark noise on a shared 6-core host** masks real effects (AVX
   frequency throttling, denormals, other agents' load).
   Severity: medium. Likelihood: medium.
   Mitigation: pin with `taskset`, flush-to-zero against denormals, use
   realistic data, prefer Criterion bootstrap CIs and `critcmp`, and cross-check
   with an instruction-count measure where feasible. Run on a quiet machine.
6. Risk: **DensePointView buffer reuse introduces aliasing/`unsafe`.**
   Severity: high. Likelihood: low.
   Mitigation: prefer a safe owned scratch buffer (`Vec<f32>` reused via
   `clear()` + `resize()`); document any `unsafe` with a SAFETY comment and add
   a Kani or Miri check if `unsafe` proves unavoidable.

## Progress

- [ ] (DRAFT) Plan authored and submitted for approval.
- [ ] Milestone 0: baseline measurement harness + go/no-go data.
- [ ] Milestone 1 (D3): core-side packed-index allocation hygiene.
- [ ] Milestone 2 (D2): provider SoA packing-buffer reuse.
- [ ] Milestone 3 (D1): provider-internal prefetch behind `simd_prefetch`
  (prototyping, go/no-go).
- [ ] Milestone 4 (D4): write-lock-free-scoring invariant guard + docs.
- [ ] Milestone 5 (D5): full benchmark evidence, doc updates, ADR, roadmap
  marked done.

## Surprises & discoveries

- Observation: 2.3.1's "score outside the write lock" is already realised.
  Evidence: `score_trim_jobs` (`chutoro-core/src/hnsw/cpu/trim.rs:65`) runs with
  no lock held; the planner scores under the read lock
  (`chutoro-core/src/hnsw/insert/planner.rs:78`). No distance call occurs inside
  either `write_graph` closure in `insert_with_collector`
  (`chutoro-core/src/hnsw/cpu/mod.rs:402-416`).
  Impact: the locking portion of 2.3.1 becomes *verification and
  documentation* (D4), not re-architecture.
- Observation: `DensePointView` is `pub(crate)` and never crosses the crate
  boundary; the dense provider already does query-centric SoA packing in
  `batch_distances`.
  Evidence: `chutoro-providers/dense/src/simd/mod.rs:19` re-exports
  `DensePointView` only within the crate; `should_pack_query_points`
  gates SoA on `dimension > 0 && candidate_count > 1 && backend != Scalar`.
  Impact: confirms the hexagonal boundary constraint; rejects the recon
  suggestion to expose `DensePointView` to core.
- Observation: `DensePointView::from_row_indices` allocates a fresh packed
  buffer per batch.
  Evidence: recon of `point_view.rs:68`.
  Impact: motivates D2 (packing-buffer reuse), measured at Milestone 0.

## Decision log

- Decision: Keep `DensePointView`/SoA/prefetch private to
  `chutoro-providers/dense`; do **not** expose to `chutoro-core`.
  Rationale: preserves the hexagonal boundary (Constraint 2), keeps future GPU
  backends free to choose their own layout, and matches the canonical pure-Rust
  HNSW design (instant-distance keeps coordinates in a separate immutable array
  behind the scoring call). Exposing it would couple the domain to one CPU
  layout.
  Date/Author: 2026-06-09, planning agent.
- Decision: Implement prefetch behind a **non-default** `simd_prefetch` Cargo
  feature on the dense crate, `cfg(target_arch = "x86_64")`-gated with a no-op
  elsewhere, rather than unconditionally.
  Rationale: the evidence strongly predicts prefetch is workload- and
  CPU-dependent and often a no-op or regression; a feature gate keeps the
  default build deterministic in performance, makes A/B benchmarking trivial,
  and matches the existing `simd_avx2`/`simd_avx512`/`simd_neon` convention.
  Date/Author: 2026-06-09, planning agent.
- Decision: Treat 2.3.1 as behaviour-preserving; validate with the *existing*
  correctness/parity/determinism suites plus equality and write-lock-guard
  tests, not new clustering-quality thresholds.
  Rationale: the change is a layout/locking optimisation, not an algorithm
  change. New quality thresholds would be the wrong acceptance signal.
  Date/Author: 2026-06-09, planning agent.

## Outcomes & retrospective

To be completed at milestones and at the end. Compare achieved neighbour-scoring
throughput and HNSW build wall-time against the Milestone 0 baseline; record
which deltas were kept, which were dropped at go/no-go, and the measured effect
of each.

## Context and orientation

This section assumes no prior knowledge of the repository.

### Crates and layout

- `chutoro-core` — domain logic: the `DataSource` trait, the CPU HNSW
  (`src/hnsw/`), MST, hierarchy extraction, and the clustering session.
- `chutoro-providers/dense` — the `DenseMatrixProvider` adapter: row-major
  `f32` storage, SoA packing (`src/simd/point_view.rs`), and SIMD distance
  kernels (`src/simd/`).
- `chutoro-providers/text` — a non-metric (Levenshtein) provider.
- `chutoro-benches` — Criterion benchmark harness and `SyntheticSource`
  generators.
- `chutoro-cli` — the `chutoro` binary.
- `chutoro-test-support` — shared test utilities.

### The neighbour-evaluation hot path (current state)

Insertion is orchestrated by `CpuHnsw::insert_with_collector`
(`chutoro-core/src/hnsw/cpu/mod.rs:364-419`). Its phases:

1. **Serialize**: take `insert_mutex` (line 370) so insertions are deterministic.
2. **Plan (read lock)**: `read_graph(|g| g.insertion_planner().plan(...))` (line
   389). `InsertionPlanner::plan` (`insert/planner.rs:78`) does greedy descent
   (`greedy_search_layer`) and per-layer best-first search (`search_layer`),
   scoring candidates via `DataSource::batch_distances`. Returns an
   `InsertionPlan` of `LayerPlan { level, neighbours: Vec<Neighbour> }`.
3. **Apply (write lock #1)**: `write_graph(|g| executor.apply(...))` (lines
   402-410) stages neighbour updates and produces `Vec<TrimJob>` of nodes whose
   neighbour lists exceed `max_connections`. *No distance computation here.*
4. **Score trim jobs (no lock)**: `score_trim_jobs(trim_jobs, source)` (line
   412; `cpu/trim.rs:65`) computes `batch_distances` in parallel via Rayon and
   ranks with a bounded `BinaryHeap` using the deterministic tie-break.
5. **Commit (write lock #2)**: `write_graph(|g| executor.commit(...))` (lines
   413-416) reconciles reciprocal edges and heals connectivity. *No distance
   computation here.*

Search (`CpuHnsw::search`, `cpu/mod.rs:444`) uses the same `LayerSearcher`
(`hnsw/search.rs`) under a read guard. `LayerSearcher::search_layer`
(`search.rs:313`) is the inner loop: it pops a candidate, collects that node's
*fresh* (not-yet-discovered) neighbour ids into a `Vec<usize>` (line 351),
scores them with `validate_batch` → `DataSource::batch_distances`, and enqueues.
The per-iteration `fresh` `Vec` and the returned distances `Vec` are the
core-side hot-path allocations targeted by D3.

`DataSource::batch_distances(query, candidates)`
(`chutoro-core/src/datasource.rs:161`) is the **port**. The dense adapter
overrides the scoring kernel; for query-centric batches it packs `candidates`
into a `DensePointView` (SoA) and runs the SIMD kernel, otherwise it falls back
to scalar pairwise.

### Key terms

- **SoA (structure-of-arrays)**: storing coordinate dimension `d` for all points
  contiguously, then dimension `d+1`, etc. (dimension-major), so a SIMD kernel
  streams one dimension across many points. Contrast array-of-structs (one
  point's coordinates contiguous).
- **Packed indices**: candidate node ids held in a single contiguous slice
  (`&[usize]`/`&[u32]`) rather than chased through per-node heap nodes.
- **Prefetch**: a CPU hint (`_mm_prefetch`) to begin loading a cache line before
  it is needed, to hide memory latency for hard-to-predict gathers.
- **Port / adapter**: in hexagonal architecture, the `DataSource` trait is the
  port; `DenseMatrixProvider` is an adapter. Domain code depends on the port,
  never on the adapter's internals.

### Documentation and skills to consult

- Design: `docs/chutoro-design.md` §6.3 (SIMD utilization; the 2.3.1 target).
- Roadmap: `docs/roadmap.md` 2.3.1, 2.4.1; determinism rule 1.1.2.
- Testing: `docs/property-testing-design.md` (§2.3.1 search correctness oracle),
  `docs/rust-testing-with-rstest-fixtures.md`, `docs/rust-doctest-dry-guide.md`,
  `docs/reliable-testing-in-rust-via-dependency-injection.md`.
- Refactoring: `docs/complexity-antipatterns-and-refactoring-strategies.md`.
- ADRs: `docs/adr-001-commit-post-processing.md`,
  `docs/adr-002-adoption-of-kani-formal-verification.md`; this plan adds
  `docs/adr-003-soa-prefetch-adapter-boundary.md`.
- Skills: `rust-router` → `rust-performance-and-layout`,
  `rust-memory-and-state`, `rust-unsafe-and-ffi` (only if prefetch needs
  `unsafe`), `rust-unit-testing`, `proptest`, `kani`, `nextest`,
  `hexagonal-architecture`, `arch-decision-records`, `execplans`.

### Prior art (from research; cite in the design doc update)

- **instant-distance** (pure-Rust HNSW): SoA split — `points: Vec<P>`
  (immutable coordinates) separate from fixed-width sentinel-padded neighbour
  arrays; per-node `RwLock`; read-snapshot of neighbour ids, lock-free scoring
  against immutable coordinates, narrow write lock only for topology. This is
  the canonical pattern and matches chutoro's existing design.
- **hnswlib** (C++ reference): flat fixed-stride level-0 record
  `[link-count | ids | optional inline vector | label]` so ids and coordinates
  sit in adjacent cache lines.
- **Prefetch**: `_mm_prefetch` stable since Rust 1.27, safe, x86_64-only;
  `_MM_HINT_T0`/`_MM_HINT_NTA`; tune look-ahead empirically (~20 elements in the
  parallel-rust-cpp study); expect frequent no-ops/regressions (Lemire;
  Algorithmica).
- **Benchmarking**: Criterion `benchmark_group` bucketed by candidate-set size
  with `Throughput::Elements`; `critcmp` for saved-baseline before/after;
  `hyperfine --warmup --runs --export-json` for whole-binary timing; beware AVX
  frequency throttling and denormals.

## Plan of work

Work proceeds milestone by milestone. Each milestone ends with its own
validation; do not proceed if validation fails. Stages within milestones follow
Red-Green-Refactor where a test framework applies.

### Milestone 0 — Baseline measurement and go/no-go data (prototyping)

Goal: establish the evidence base before changing hot-path code, so every later
delta is justified by measurement.

1. Add a Criterion microbenchmark `chutoro-benches/benches/neighbour_scoring.rs`
   that scores a single query against candidate sets of bucketed sizes
   straddling cache levels (e.g. 8, 16, 32, 64, 256, 1024, 4096 candidates) over
   a `DenseMatrixProvider`, across dimensions {32, 128, 768}, using
   `Throughput::Elements`, `criterion::black_box`, and a fixed seed. Register it
   in `chutoro-benches/Cargo.toml` as `[[bench]] name = "neighbour_scoring"`.
2. Add a focused allocation-count test (in `chutoro-core`, `#[cfg(test)]`) that
   records how many `Vec` allocations `search_layer` performs per inserted node
   on a small fixture, to quantify D3's target. Use a counting allocator shim
   scoped to the test, or instrument via a wrapper `DataSource` that counts
   `batch_distances` invocations and slice lengths.
3. Capture a Criterion baseline: `cargo bench -p chutoro-benches --bench
   neighbour_scoring -- --save-baseline before`.
4. Add `scripts/bench-neighbour-scoring.sh`: a `hyperfine` wrapper that builds
   the CLI in release and times an HNSW build over a fixed synthetic dataset,
   with `--warmup 3 --runs 20 --export-json`, `taskset`-pinned, documented in
   `docs/developers-guide.md`.

Validation: `make bench` builds; the new bench runs and emits per-bucket
numbers; the baseline is saved under `target/criterion`. Record the numbers and
the allocation counts in `Progress` and `Surprises & discoveries`.

Go/no-go: from these numbers, confirm which of D1/D2/D3 target a real cost. Drop
any that do not, recording the decision.

### Milestone 1 — Core-side packed-index allocation hygiene (D3)

Goal: remove per-iteration heap allocations in the neighbour-evaluation inner
loop while keeping candidate ids packed and contiguous and output identical.

Files: `chutoro-core/src/hnsw/search.rs` (`SearchState`, `search_layer`,
`find_better_neighbour`), and the trim path
(`chutoro-core/src/hnsw/cpu/trim.rs`, `chutoro-core/src/hnsw/helpers.rs`).

1. Red: add an rstest equality test asserting that a refactored
   scratch-reusing `search_layer` yields the *same* `Vec<Neighbour>` as the
   current implementation on a battery of fixtures (uniform, clustered,
   duplicate). Add a test asserting allocation count drops versus the Milestone
   0 measurement.
2. Green: give `SearchState` (or `LayerSearcher`) a reusable scratch
   `Vec<usize>` for the per-node `fresh` candidate list and a reusable
   `Vec<f32>` for distances, cleared (`clear()` + `reserve`) each iteration
   rather than freshly allocated. Keep the deterministic enqueue/tie-break
   identical. Mirror the same scratch reuse in `run_trim_job` where applicable.
3. Refactor: extract a small `ScratchBuffers` helper if it improves clarity;
   keep files under 400 lines.

Validation: the equality test passes; allocation-count test shows the drop;
existing HNSW unit, property (search correctness, idempotency, mutation), and
backend-parity suites pass unchanged; `neighbour_scoring` bench compared with
`critcmp before <new>` shows no regression (improvement expected at larger
buckets). Determinism preserved.

### Milestone 2 — Provider SoA packing-buffer reuse (D2)

Goal: stop allocating a fresh packed SoA buffer per `batch_distances` call.

Files: `chutoro-providers/dense/src/simd/point_view.rs`,
`chutoro-providers/dense/src/simd/mod.rs`,
`chutoro-providers/dense/src/provider.rs`.

1. Red: add a dense-crate unit test asserting packed output is identical
   whether produced fresh or via a reused buffer, and a test/bench showing the
   per-batch allocation is eliminated.
2. Green: introduce a reusable `PackedSoaStorage` scratch owned per scoring
   call site (e.g. a thread-local or an explicit scratch passed into
   `from_row_indices`/the kernel entry), reset (`clear()` + `resize`) per batch.
   Preserve 64-byte alignment, 16-lane padding, and deterministic `0.0` tail
   fill. Keep everything `pub(crate)` — no boundary change.
3. Optional D2b (managed by exception; pre-authorised but flag in Decision
   Log): if Milestone 0 shows the `Vec<f32>` returned by `batch_distances` is a
   material cost, add a *defaulted* `DataSource::batch_distances_into(query,
   candidates, out: &mut Vec<f32>)` so core can reuse the output buffer. The
   default impl delegates to `batch_distances`; the dense adapter overrides it.
   This is the only sanctioned port change and must keep all existing impls
   compiling.

Validation: dense unit tests and the backend-parity property suite
(`docs/chutoro-design.md` §6.3; 2.2.6) pass; the SoA tail-padding/dispatch Kani
harnesses (2.2.7) still pass under `make kani`; `neighbour_scoring` bench shows
no regression. Parity within epsilon preserved.

### Milestone 3 — Provider-internal prefetch behind `simd_prefetch` (D1, prototyping)

Goal: hide coordinate-gather latency by prefetching the next candidate's SoA
coordinate block, *only* where it measurably helps.

Files: `chutoro-providers/dense/Cargo.toml` (add `simd_prefetch` feature, off by
default), `chutoro-providers/dense/src/simd/kernels.rs` (and/or
`point_view.rs`), plus a small `prefetch` helper module.

1. Add a `prefetch` helper: `#[cfg(all(target_arch = "x86_64", feature =
   "simd_prefetch"))]` calls `core::arch::x86_64::_mm_prefetch(ptr as *const
   i8, _MM_HINT_T0)`; every other target/feature combination is an inlined
   no-op. Document with a comment that prefetch is semantically a no-op, so
   omitting it is always correct.
2. Inside the query-centric SoA kernel loop, prefetch the coordinate block of a
   tunable number of candidates ahead (start at the empirically-cited ~ a few
   blocks; expose the look-ahead as a small `const`). Keep hot-loop temporaries
   in scalar locals to avoid the optimizer regression the research observed.
3. Benchmark `simd_prefetch` on vs off with `critcmp` across all buckets.

Validation and go/no-go: if `simd_prefetch` shows a statistically significant
improvement on the target host with no bucket regressing, keep the feature
(off by default, documented). If it is within noise or regresses, **abandon D1**,
remove the kernel changes (retain only the no-op helper if cheap), and record
the result in `Decision Log` and `Outcomes`. Behaviour is identical either way.

### Milestone 4 — Write-lock-free-scoring invariant guard (D4)

Goal: turn the existing "no scoring under the write lock" property into a tested,
documented invariant.

1. Red/Green: add a deterministic regression test in `chutoro-core` using an
   instrumented `DataSource` decorator that, on every `distance`/
   `batch_distances` call, asserts the graph is **not** write-locked. The probe
   uses a test-only handle to the graph `RwLock` and asserts
   `graph.try_read().is_ok()` (a thread holding the write lock — or any
   writer — makes `try_read` fail). Wire a `#[cfg(test)]` (or `pub(crate)`
   test-only) accessor on `CpuHnsw` returning `Arc<RwLock<Graph>>` so the
   decorator can probe it. Build a small index through this decorator and assert
   no violation is recorded across planning, trim scoring, and commit.
2. Confirm the guard *fails* if scoring is deliberately moved under the write
   lock (one-off mutation check, then revert), proving the test has teeth.

Validation: the guard passes on the real code and fails on the deliberate
mutation. `make test` green.

### Milestone 5 — Benchmark evidence, documentation, ADR, roadmap (D5)

1. Re-run the Criterion `neighbour_scoring` bench and `critcmp before after`;
   run `scripts/bench-neighbour-scoring.sh` (hyperfine) for whole-binary
   before/after. Record both in `Outcomes & retrospective` and in a short
   results table appended to `docs/chutoro-design.md` §6.3 as an implementation
   update dated to the merge.
2. Update `docs/chutoro-design.md` §6.3 with an implementation update for 2.3.1:
   the packed-index/SoA scoring contract, the scoring-outside-write-lock
   invariant and its guard, the `simd_prefetch` feature (and whether it was
   kept), and the cited prior art.
3. Update `docs/developers-guide.md`: the neighbour-scoring scratch-reuse
   convention, the `simd_prefetch` feature, the benchmark/`hyperfine` workflow,
   and the write-lock invariant test.
4. Update `docs/users-guide.md` *only if* a user-visible change exists (a new
   non-default feature flag is the likely only candidate; default behaviour is
   unchanged).
5. Add `docs/adr-003-soa-prefetch-adapter-boundary.md` (Y-Statement; see the
   `arch-decision-records` skill) recording the decision to keep SoA/prefetch
   private to the dense adapter and to gate prefetch behind a non-default
   feature. Reference it from `docs/chutoro-design.md` §6.3.
6. Mark roadmap item 2.3.1 as done in `docs/roadmap.md`.

Validation: `make check-fmt`, `make lint`, `make test`, `make kani`,
`make markdownlint`, and `make nixie` (if any Mermaid changed) all pass.

## Concrete steps

Run from the repository root unless stated. Use `tee` to capture long output, per
the global agent instructions:

```bash
# chutoro-core/.. repository root
make check-fmt 2>&1 | tee "/tmp/check-fmt-chutoro-$(git branch --show-current).out"
make lint      2>&1 | tee "/tmp/lint-chutoro-$(git branch --show-current).out"
make test      2>&1 | tee "/tmp/test-chutoro-$(git branch --show-current).out"
```

Baseline and comparison:

```bash
# Save a Criterion baseline before changes, then compare after each milestone.
cargo bench -p chutoro-benches --bench neighbour_scoring -- --save-baseline before
# ... implement a milestone ...
cargo bench -p chutoro-benches --bench neighbour_scoring -- --save-baseline after
critcmp before after
```

Whole-binary timing (after a representative milestone):

```bash
bash scripts/bench-neighbour-scoring.sh   # wraps hyperfine; see developers-guide
```

Prefetch A/B (Milestone 3):

```bash
cargo bench -p chutoro-benches --bench neighbour_scoring -- --save-baseline noprefetch
cargo bench -p chutoro-benches --features chutoro-providers-dense/simd_prefetch \
  --bench neighbour_scoring -- --save-baseline prefetch
critcmp noprefetch prefetch
```

Kani (after Milestone 2):

```bash
make kani 2>&1 | tee "/tmp/kani-chutoro-$(git branch --show-current).out"
```

Expected: each gate prints a success summary (clippy `0 warnings`, tests
`... passed`, Kani `VERIFICATION:- SUCCESSFUL`). `critcmp` shows the optimised
baseline equal-or-faster per bucket with overlapping-or-better confidence
intervals.

## Validation and acceptance

Acceptance is behavioural and statistical:

1. **Identical clustering and neighbour selection** under fixed seeds: the
   existing search-correctness oracle property
   (`docs/property-testing-design.md` §2.3.1), idempotency, mutation, and
   backend-parity suites pass unchanged; new equality tests (Milestone 1/2)
   confirm refactors are output-preserving.
2. **Write-lock invariant**: the Milestone 4 guard passes on real code and fails
   on a deliberate "score under write lock" mutation.
3. **Performance**: `critcmp before after` shows the `neighbour_scoring`
   microbenchmark is equal-or-faster across buckets (improvement expected at
   mid/large candidate-set sizes); `hyperfine` shows HNSW build wall-time
   equal-or-faster. Any milestone with no measurable benefit is dropped at its
   go/no-go gate, recorded in the `Decision Log` — that is an accepted outcome,
   not a failure.
4. **Quality gates**: `make check-fmt`, `make lint`, `make test`, `make kani`,
   `make markdownlint`, and `make nixie` all pass.

Red-Green-Refactor evidence to record per milestone: the red command and its
expected failure (e.g. the equality test failing before the scratch refactor is
wired, or the allocation-count test failing before reuse), the green command and
its pass, and the refactor command sequence and pass.

Quality criteria ("done"):

- Tests: all suites above green; new equality, allocation-count, and write-lock
  guard tests added and passing.
- Lint/typecheck: `make lint` with `-D warnings` clean.
- Performance: documented `critcmp` and `hyperfine` evidence; no bucket
  regresses on the target host with the default feature set.
- Security: n/a (no new dependencies, no FFI, no `unsafe` unless justified with
  a SAFETY comment and a Miri/Kani check).

Quality method: run the gates locally with `tee`; run `coderabbit review
--agent` after each major milestone and clear all concerns before the next, but
only after the deterministic gates above are green.

## Idempotence and recovery

All steps are re-runnable. Criterion baselines are named (`before`/`after`/
`noprefetch`/`prefetch`) and can be re-saved. The `simd_prefetch` feature is
additive and off by default, so reverting Milestone 3 is removing a feature and
its kernel hunk. Commit after each milestone so any milestone can be rolled back
with `git revert`. No destructive or irreversible step is involved.

## Artifacts and notes

Record, as concise indented transcripts, at minimum: the Milestone 0 per-bucket
baseline, the `critcmp before after` table, the `hyperfine` JSON summary, and the
Kani success line. Keep evidence focused on proving the acceptance criteria.

## Interfaces and dependencies

No new external dependencies. Interfaces that must exist at completion:

- `chutoro-providers/dense` gains a non-default Cargo feature `simd_prefetch`
  and an internal `prefetch` helper:

```rust
// chutoro-providers/dense/src/simd/prefetch.rs
/// Hints the CPU to begin loading `ptr` into cache. A no-op where unsupported;
/// prefetch is semantically a hint, so omitting it is always correct.
#[inline]
pub(crate) fn prefetch_t0(ptr: *const f32) {
    #[cfg(all(target_arch = "x86_64", feature = "simd_prefetch"))]
    // SAFETY: `_mm_prefetch` never dereferences `ptr`; it is a safe hint.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
            ptr as *const i8,
        );
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "simd_prefetch")))]
    let _ = ptr;
}
```

(Note: `_mm_prefetch` is itself safe; the `unsafe` block above is only needed if
the chosen signature requires it — prefer the safe call form and drop the block
if clippy confirms it is unnecessary.)

- `chutoro-benches` gains `[[bench]] name = "neighbour_scoring"`.
- Optional, only if Milestone 0 justifies it (D2b), a *defaulted* port method:

```rust
// chutoro-core/src/datasource.rs (DataSource trait)
/// Scores `query` against `candidates`, writing into `out` (cleared first) to
/// let hot-path callers reuse the allocation. Defaults to `batch_distances`.
fn batch_distances_into(
    &self,
    query: usize,
    candidates: &[usize],
    out: &mut Vec<f32>,
) -> Result<(), DataSourceError> {
    out.clear();
    out.extend(self.batch_distances(query, candidates)?);
    Ok(())
}
```

- No change to `DataSource::distance`, `DataSource::batch_distances`,
  `DataSource::distance_batch`, `CpuHnsw`'s public methods, or the CLI.

## Signposted documentation and skills

- Plans/process: `execplans` skill (this document's format).
- Architecture: `hexagonal-architecture`, `arch-decision-records`,
  `arch-crate-design`; `docs/chutoro-design.md` §6.3.
- Rust: `rust-router` → `rust-performance-and-layout`,
  `rust-memory-and-state`, `rust-types-and-apis`, and `rust-unsafe-and-ffi`
  (only if prefetch needs `unsafe`).
- Testing/verification: `rust-unit-testing`, `proptest`, `kani`, `nextest`,
  `rust-verification`; `docs/property-testing-design.md`,
  `docs/rust-testing-with-rstest-fixtures.md`,
  `docs/rust-doctest-dry-guide.md`,
  `docs/reliable-testing-in-rust-via-dependency-injection.md`,
  `docs/complexity-antipatterns-and-refactoring-strategies.md`.

## Revision note

Initial draft (2026-06-09). Establishes the honest scope (most of 2.3.1's intent
already delivered by 2.2.x), the hexagonal boundary decision, the five
measured deltas (D1-D5), milestone structure with go/no-go gates, and the
validation strategy (behaviour-preserving; Criterion + `critcmp` + `hyperfine`).
Pending: community-of-experts review and any revisions arising from it, then
user approval before implementation.
