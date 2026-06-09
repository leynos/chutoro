# Restructure HNSW neighbour evaluation: packed indices, SoA scoring, and prefetch (2.3.1)

This ExecPlan (execution plan) is a living document. The sections
`Constraints`, `Tolerances`, `Risks`, `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work
proceeds.

Status: DRAFT (revised after community-of-experts review)

Roadmap item: 2.3.1 (Phase 2.3, Hot-path optimizations). See `docs/roadmap.md`
lines 319-321 and `docs/chutoro-design.md` §6.3 (lines 887-963).

## Purpose / big picture

chutoro builds a CPU Hierarchical Navigable Small World (HNSW) graph to harvest
candidate edges for clustering. The hot path is *neighbour evaluation*: for each
node visited during search and insertion, the engine reads that node's
neighbour list and computes the distance from a query point to every neighbour,
then keeps the closest. Roadmap item 2.3.1 asks that this evaluation use *packed*
candidate indices and a *structure-of-arrays* (SoA) coordinate layout, *prefetch*
upcoming coordinate blocks, and score *outside the write lock*.

Reconnaissance (recorded under `Surprises & discoveries`) shows that the
*structural* intent of 2.3.1 is already realised by Phase 2.2 (SIMD kernels) and
the existing two-phase locking design — but with one genuinely unrealised seam
(a pack→unpack→repack round-trip at the port) and a measurement question
(whether any residual hot-path win actually exists once the distance cache
fragments batches). This plan therefore reframes 2.3.1 around **evidence**, not
motion. It has two scopes:

1. **Committed scope (always delivered).** Prove and document the realised
   intent: a regression guard that distance scoring never runs under the write
   lock; an implementation-update to `docs/chutoro-design.md` §6.3 explaining
   how 2.2.x satisfies the layout/locking structure; an Architecture Decision
   Record (ADR) recording the adapter boundary and the deferred structural
   alternatives; and a measurement harness (a `neighbour_scoring` Criterion
   benchmark — also the canonical artefact for roadmap 2.4.1) that quantifies
   the real candidate-set-size distribution, the cache-miss-subset distribution,
   the share of build wall-time spent scoring, and distance-cache lock
   contention.
2. **Conditional scope (delivered only on evidence).** Three hot-path deltas —
   core-side allocation hygiene (E1), SoA packing-buffer reuse plus its required
   query-centric port override (E2), and packing-step prefetch (E3) — each
   implemented **only if** the committed-scope measurement clears a
   pre-registered threshold. If the evidence does not clear the bar, the correct
   and expected outcome is that the committed scope ships and E1-E3 do not. That
   is a success, not a shortfall.

After this change a developer can:

1. Run `cargo bench -p chutoro-benches --bench neighbour_scoring` and see
   neighbour-set scoring bucketed by the candidate-set sizes that *actually
   occur* in the HNSW hot path (≈ 8-48), with cycle-count and per-bucket SoA
   lane-utilisation data, not just wall-time.
2. Read `docs/chutoro-design.md` §6.3, `docs/developers-guide.md`, and
   `docs/adr-003-soa-prefetch-adapter-boundary.md` and find documented: how
   2.2.x satisfies 2.3.1's layout/locking structure, the cache-fragmentation
   reality, the invariant that scoring never runs under the write lock and the
   test that proves it, and the deferred structural levers with their
   trade-offs.
3. See clustering output **unchanged** under fixed seeds — every delta is
   behaviour-preserving, proven by the existing search-correctness, idempotency,
   mutation, and backend-parity suites plus the new write-lock guard.

### What is already realised vs genuinely residual (read before estimating)

Confirmed by reading the code (file:line cited in `Surprises & discoveries`):

- **Scoring already happens outside the write lock.** `InsertionPlanner::plan`
  (`insert/planner.rs:78`) scores under the *read* lock;
  `CpuHnsw::score_trim_jobs` (`cpu/trim.rs:65`) scores with **no lock held**, in
  parallel via Rayon; neither `write_graph` closure in `insert_with_collector`
  (`cpu/mod.rs:402-416`) computes a distance. This is the plan's strongest,
  most durable deliverable (committed scope C1).
- **SoA scoring already exists** as an adapter detail: `DenseMatrixProvider`
  reaches a 64-byte-aligned, 16-lane-padded, dimension-major SoA `DensePointView`
  (`dense/src/simd/point_view.rs`) and AVX-512/AVX2/Neon/scalar kernels.

Genuinely unrealised or unmeasured (the residual the panel surfaced):

- **The "packed indices" round-trip.** The dense provider does **not** override
  `DataSource::batch_distances` (it overrides only `distance` and
  `distance_batch`, `provider.rs:137,143`). So core's *default*
  `batch_distances` (`datasource.rs:161-177`) turns the packed `&[usize]` into a
  fresh `Vec<(usize,usize)>` of pairs, then `shared_query_candidates`
  (`dense/src/simd/mod.rs:195-213`) scans those pairs to *re-derive* the shared
  query and re-collects a fresh candidate `Vec`, before `from_row_indices`
  (`point_view.rs:68`) allocates `PackedSoaStorage`. The roadmap item is
  literally titled "use packed indices"; structurally the code packs, unpacks to
  pairs, then repacks every call. This is the clearest unrealised seam (E2's
  prerequisite).
- **The distance cache fragments batches.** The cache is present on *both* the
  insertion and search paths (`Some(cache)` at `cpu/mod.rs:394,456,466`). So the
  slice that actually reaches `batch_distances` is the cache-*miss* subset
  (`validate.rs:123-127`, `helpers.rs:134-151`), not the full ≤ `2M` neighbour
  list. As the cache warms, miss subsets routinely fall below
  `should_pack_query_points`'s `candidate_count > 1` threshold
  (`dense/src/simd/mod.rs:163-173`) and drop to the scalar path. D1/D2/D3 only
  matter where miss counts are routinely large enough to pack — which must be
  *measured*, not assumed.
- **Prefetch has no obvious target in the kernel.** The SoA layout is
  dimension-major (`point_view.rs:82`), so the kernel already streams contiguous
  per-dimension blocks (hardware-prefetcher territory). The only scattered
  gather is `matrix.row(index)` during *packing* (`point_view.rs:79-84`). So
  prefetch, if it helps anywhere, belongs in the pack step, and is likely a
  no-op at realistic batch sizes (E3 is conditional and structurally suspect).

### Deferred to a separate, evidence-first optimisation item (NOT built here)

The strongest measured wins are structural and out of scope for 2.3.1; they are
recorded in ADR-003 and proposed as a new roadmap item rather than smuggled in:

- **Cross-node "beam" scoring** — accumulate candidates from multiple popped
  nodes in `search_layer` into one packing+scoring call, widening the window
  from ≈ `2M` to hundreds. This touches core search *policy* and must preserve
  deterministic best-first visitation order; larger blast radius.
- **Secondary dimension-major (SoA) matrix copy** held once at provider
  construction, so `from_row_indices` becomes a strided gather with no per-call
  transpose and no packing buffer — trading ~1× memory to eliminate the path E2
  optimises incrementally.
- **`batch_distances_into` out-buffer reuse** — only meaningful if a
  query-centric override exists and the cache's indexed-scatter consumption is
  redesigned; otherwise it saves nothing on the cached path and penalises
  non-overriding providers (e.g. text).

## Constraints

Hard invariants. Violation requires escalation, not a workaround.

1. **Behaviour preservation.** Clustering output and HNSW neighbour selection
   must be identical under fixed seeds, before and after. The deterministic
   tie-break (distance, then lower item id, then insertion sequence;
   `search.rs` `compare_neighbours`; `docs/roadmap.md` 1.1.2) must be preserved
   exactly.
2. **Hexagonal boundary.** The SoA layout, SIMD kernels, prefetch intrinsics,
   and `DensePointView`/`PackedSoaStorage` remain an *adapter* detail private to
   `chutoro-providers/dense` (`pub(crate)`). Domain/policy logic (which
   candidates to score, packed-index iteration, deterministic ordering, lock
   discipline) stays in `chutoro-core` behind the `DataSource` port. The only
   sanctioned cross-boundary change is *adding* a defaulted query-centric port
   method (E2 prerequisite); the adapter's internal SoA types never leak into
   core. See the `hexagonal-architecture` skill.
3. **Public API stability.** Any new `DataSource` trait method must be
   *defaulted* and non-breaking; existing impls (dense, text, synthetic) must
   keep compiling unchanged. `CpuHnsw` public methods and the CLI surface are
   unchanged.
4. **No scoring under the write lock.** Distance computation must never run while
   a `RwLockWriteGuard<Graph>` is held *on the current thread*. This is both a
   constraint and a tested invariant (C1).
5. **MSRV and toolchain.** Stable Rust on the default build path; MSRV `1.89.0`,
   pinned toolchain `1.93.1`. `_mm_prefetch` is stable since 1.27 and safe to
   call. The optional nightly `std::simd` path is out of scope.
6. **Quality gates.** `make check-fmt`, `make lint` (clippy `-D warnings`), and
   `make test` pass before every commit; markdown changes also pass
   `make markdownlint` (and `make nixie` if Mermaid changes). Lints are not
   silenced except as a tightly-scoped, justified last resort.
7. **File size.** No source file exceeds 400 lines (`AGENTS.md`). `search.rs` is
   already 369 lines; any E1 work must first extract a sibling module
   (Constraint enforced as a milestone validation step).

## Tolerances (exception triggers)

Stop and escalate (do not work around) when:

1. **Scope.** A milestone requires changing more than ~6 source files or ~400
   net lines beyond what that milestone names.
2. **Interface.** Any *non-defaulted* change to the public `DataSource` trait,
   `CpuHnsw`, or CLI is required.
3. **Dependencies.** Any new external crate is required (none anticipated;
   prefetch uses `core::arch`; cycle counts use the existing toolchain or
   `perf`/`cachegrind`/`iai` which are dev-time tools, not crate deps — confirm
   `iai`/`iai-callgrind` policy before adding it as a dev-dependency and flag in
   the Decision Log if used).
4. **Behaviour drift.** Any existing test asserting neighbour selection, recall,
   parity, or determinism changes its expected output. This must never happen.
5. **Evidence gate not cleared.** If committed-scope measurement (C4) does not
   clear the pre-registered keep threshold for a conditional delta, that delta is
   **not implemented**; stop and record the null result (this is an accepted
   outcome, not an escalation, but it must be documented in `Outcomes`).
6. **Iterations.** If a milestone's tests still fail after 3 focused attempts,
   stop and escalate.
7. **Regression masquerading as noise.** If a conditional delta shows any
   regression on the cycle-count cross-check at the realistic buckets, drop it;
   never keep a delta on "within wall-time noise".

## Risks

1. Risk: **Closed with motion, not measurement** — a feature flag, ADR, bench,
   and scratch buffers ship but build wall-time is unchanged because batches were
   always cache-miss-subset-narrow and the distance-cache `Mutex` was the real
   bound. (Pre-mortem A.)
   Severity: high. Likelihood: medium.
   Mitigation: committed scope is C1-C4 only; E1-E3 gate on a pre-registered C4
   threshold derived from the *measured* miss-subset distribution; pre-author the
   null-result §6.3 update and Outcomes entry so "committed-scope-only" is a
   first-class success.
2. Risk: **Reuse buffer poisons a warm-cache build** — `PackedSoaStorage` reuse
   leaves NaN/Inf in the padded tail; a later smaller batch reads stale lanes;
   active-lane output is preserved most of the time, so a naive single-batch
   test passes and a non-reproducible determinism break ships. (Pre-mortem B.)
   Severity: high. Likelihood: low-medium.
   Mitigation: mandate the adversarial large-then-small reuse test with NaN/Inf
   trailing rows asserting bit-exact `0.0` tails; extend the 2.2.7 tail-padding
   Kani harness to the reused-buffer path; forbid shared scratch; `debug_assert`
   scratch is reset before refill.
3. Risk: **A real regression waved through as noise** on the contended 6-core
   host. (Pre-mortem C.)
   Severity: medium. Likelihood: medium.
   Mitigation: cycle/instruction count is the *primary* keep/drop signal;
   explicit minimum effect size; pin `RAYON_NUM_THREADS`; a documented
   load-average gate at capture time; "within noise" = drop-by-default for any
   code-adding delta.
4. Risk: **The optimised window is structurally too small** — per-node batches
   are ≈ `2M` (16-48), fragmented further to cache-miss subsets, with 16-lane
   padding wasting 30-47% at those sizes — so SoA packing overhead cancels the
   SIMD win and prefetch has nothing to hide.
   Severity: medium. Likelihood: high.
   Mitigation: this is exactly what C4 measures; E1-E3 proceed only if the data
   contradicts it; the real structural lever (cross-node beam) is deferred to a
   separate item.
5. Risk: **Benchmark noise / AVX throttling / denormals** mask effects.
   Severity: medium. Likelihood: medium.
   Mitigation: `taskset` pinning, flush-to-zero, realistic data, Criterion
   bootstrap CIs, `critcmp`, and the mandatory cycle-count cross-check; quiet
   host.
6. Risk: **Scratch reuse interacts badly with Rayon-parallel trim** — a
   shared `&self` `RefCell`/`Mutex` scratch serialises `score_trim_jobs` (erasing
   the 2.2 parallel-scoring win) or makes allocation-count assertions
   nondeterministic across thread counts.
   Severity: high. Likelihood: medium.
   Mitigation: forbid shared scratch in the Decision Log; require thread-local or
   explicit per-call/per-job scratch; reuse buffer is `Vec<AlignedBlock>`
   (64-byte aligned, 16-lane padded), not `Vec<f32>`; pin `RAYON_NUM_THREADS=1`
   for allocation-count tests if thread-local is chosen.

## Progress

- [x] (2026-06-09) Draft authored.
- [x] (2026-06-09) Community-of-experts (Logisphere) review: verdict REVISE;
  all 12 required revisions folded into this draft.
- [ ] User approval of the revised plan (required before implementation).
- [ ] Milestone 0 (C4): measurement harness + cache/scoring/contention data +
  pre-registered go/no-go thresholds.
- [ ] Milestone 1 (C1): write-lock-free-scoring invariant guard (same-thread
  marker).
- [ ] Milestone 2 (C2/C3): §6.3 implementation-update, developers-guide, ADR-003,
  roadmap mark — reflecting whatever the evidence supports.
- [ ] Milestone 3 (E1, conditional): core-side allocation hygiene.
- [ ] Milestone 4 (E2, conditional): query-centric port override + SoA
  packing-buffer reuse.
- [ ] Milestone 5 (E3, conditional): packing-step prefetch behind `simd_prefetch`.
- [ ] Milestone 6 (D5): final evidence, doc/ADR finalisation, roadmap done.

## Surprises & discoveries

- Observation: scoring already runs outside the write lock.
  Evidence: `cpu/trim.rs:65` (`score_trim_jobs`, no lock); `insert/planner.rs:78`
  (planner under read lock); no distance call inside either `write_graph` closure
  (`cpu/mod.rs:402-416`).
  Impact: the locking part of 2.3.1 is verification + documentation (C1), not
  re-architecture.
- Observation: the dense provider does **not** override `batch_distances`.
  Evidence: `dense/src/provider.rs:137,143` overrides only `distance` and
  `distance_batch`; core's default `batch_distances` (`datasource.rs:161-177`)
  builds a pairs `Vec`; `shared_query_candidates` (`dense/src/simd/mod.rs:195`)
  re-derives the query and re-collects candidates before
  `from_row_indices` (`point_view.rs:68`).
  Impact: "packed indices" is genuinely unrealised at the port; E2's prerequisite
  is a query-centric override that carries `(query, &[candidate])` intact.
- Observation: the distance cache is present on both insertion and search paths.
  Evidence: `Some(cache)` at `cpu/mod.rs:394` (planner), `456`/`466` (search);
  `validate_batch_distances` forwards only the *miss* subset
  (`validate.rs:123-127`; `helpers.rs:134-151`).
  Impact: batches reaching the adapter are warm-cache-narrow; SoA frequently
  disabled by `should_pack_query_points`. D1/D2/D3 benefit is unproven and
  must be measured.
- Observation: the reuse buffer must preserve alignment.
  Evidence: `PackedSoaStorage { blocks: Vec<AlignedBlock> }`,
  `AlignedBlock([f32; 16])` `repr(C, align(64))` (`point_view.rs:13-19`).
  Impact: scratch is `Vec<AlignedBlock>`, not `Vec<f32>`; corrects the original
  Risk 6 mitigation.
- Observation: `search.rs` is 369/400 lines.
  Evidence: `wc -l`.
  Impact: E1 must extract a sibling module before adding scratch.

## Decision log

- Decision: Re-scope 2.3.1 to committed C1-C4 (guard + docs + measurement) with
  E1-E3 conditional on pre-registered evidence; defer the structural levers
  (cross-node beam, secondary SoA copy, `batch_distances_into`) to a separate
  evidence-first roadmap item.
  Rationale: code reading shows the structural intent is satisfied by 2.2.x; the
  residual per-call deltas operate inside a window the cache fragments to near
  nothing; committing speculative complexity would close the item on motion. The
  Logisphere panel's verdict was REVISE on exactly this point.
  Date/Author: 2026-06-09, planning agent (post-review).
- Decision: Keep `DensePointView`/`PackedSoaStorage`/SoA/prefetch private to
  `chutoro-providers/dense`; the only boundary change permitted is *adding* a
  defaulted query-centric port method.
  Rationale: preserves the hexagonal boundary and future GPU/alternate layouts;
  matches the canonical pure-Rust HNSW design (instant-distance keeps coordinates
  in a separate immutable array behind the scoring call).
  Date/Author: 2026-06-09, planning agent.
- Decision: The keep/drop signal for any conditional delta is the
  cycle/instruction-count cross-check (primary), with `critcmp` wall-time and
  `hyperfine` as corroboration only; keep requires >5% median improvement AND
  non-overlapping CIs at the realistic (8-48) buckets; "within noise" drops.
  Rationale: per-call effects are sub-microsecond on a shared host; wall-time CIs
  overlap the effect; cycle counts are noise-immune.
  Date/Author: 2026-06-09, planning agent.
- Decision: Forbid shared `Mutex`/`RefCell` SoA scratch; require thread-local or
  explicit per-call/per-job `Vec<AlignedBlock>` scratch.
  Rationale: a shared scratch serialises the Rayon-parallel trim and destroys the
  2.2 parallel-scoring win; alignment/padding must be preserved.
  Date/Author: 2026-06-09, planning agent.
- Decision: The C1 guard asserts a *same-thread* "not holding a write guard"
  invariant via a marker set inside `write_graph`'s scope, not a global
  `try_read().is_ok()` probe.
  Rationale: a global probe spuriously fails when another Rayon/search thread
  legitimately holds the write lock; only the same-thread property is meaningful.
  Date/Author: 2026-06-09, planning agent (corrects the original draft).

## Outcomes & retrospective

To be completed at milestones and at the end. Must state, with cited numbers:
the measured candidate-set-size and cache-miss-subset distributions; the share of
build wall-time spent scoring vs graph mutation/MST/hierarchy; distance-cache
lock contention; and, for each of E1/E2/E3, the go/no-go decision with its
cycle-count evidence. A "committed scope only (C1-C4), E1-E3 dropped on evidence"
result is an explicitly successful outcome and must be recorded as such, with the
deferred structural levers carried into the proposed follow-up item.

## Context and orientation

Assumes no prior knowledge of the repository.

### Crates and layout

- `chutoro-core` — domain: the `DataSource` trait, CPU HNSW (`src/hnsw/`), MST,
  hierarchy extraction, clustering session.
- `chutoro-providers/dense` — the `DenseMatrixProvider` adapter: row-major `f32`
  storage, SoA packing (`src/simd/point_view.rs`), SIMD kernels (`src/simd/`).
- `chutoro-providers/text` — a non-metric (Levenshtein) provider (no SoA
  override; relevant to Constraint 3).
- `chutoro-benches` — Criterion harness and `SyntheticSource` generators.
- `chutoro-cli`, `chutoro-test-support` — binary and shared test utilities.

### The neighbour-evaluation hot path (current state)

Insertion: `CpuHnsw::insert_with_collector` (`cpu/mod.rs:364-419`):

1. Serialize via `insert_mutex` (line 370).
2. **Plan (read lock)** `read_graph(|g| g.insertion_planner().plan(...))` (line
   389): greedy descent + per-layer best-first search scoring via
   `DataSource::batch_distances`, **with the distance cache present**
   (`cache: Some(cache)`, line 394). Returns `InsertionPlan`.
3. **Apply (write lock #1)** (lines 402-410): stage updates, produce
   `Vec<TrimJob>`. *No distance computation.*
4. **Score trim jobs (no lock)** `score_trim_jobs` (line 412; `cpu/trim.rs:65`):
   parallel Rayon scoring of the cache-*miss* subset, deterministic bounded heap.
5. **Commit (write lock #2)** (lines 413-416): reconcile reciprocity, heal
   connectivity. *No distance computation.*

Search (`cpu/mod.rs:444`) uses the same `LayerSearcher` (`hnsw/search.rs`) under
a read guard, also with the cache present (lines 456,466).

### Cache interaction (critical to all measurement)

The cache sits between core and the port on every production path. On a cache
hit, no `batch_distances` call is made for that candidate; on a miss,
`validate.rs` (`CacheBatch::resolve`, lines 118-146) and `helpers.rs`
(`batch_distances_for_trim`, lines 122-168) forward **only the miss subset** to
`source.batch_distances`. As the graph warms, miss subsets shrink, frequently
below the SoA packing threshold (`candidate_count > 1`,
`dense/src/simd/mod.rs:172`). Therefore:

- The batch size the adapter sees is *not* `2M`; it is the per-call miss count,
  which Milestone 0 must measure (cold vs warm).
- The distance cache's own `Mutex<LruCache>` may be the dominant cost under
  parallel trim; Milestone 0 measures its contention so go/no-go attributes cost
  honestly.

### The packed-index round-trip (the unrealised seam)

`core &[usize]` → default `batch_distances` builds `Vec<(query, candidate)>`
(`datasource.rs:169-173`) → `distance_batch` → `euclidean_distance_batch_raw_pairs`
→ `shared_query_candidates` scans pairs to re-derive the shared query and
re-collect candidates (`dense/src/simd/mod.rs:195-213`) → `from_row_indices`
allocates `PackedSoaStorage` (`point_view.rs:68`). The "packed" ids are unpacked
to pairs then repacked. E2's prerequisite eliminates this by adding a defaulted
query-centric port method the dense adapter overrides.

### Key terms

- **SoA (structure-of-arrays)**: dimension-major coordinate storage so a SIMD
  kernel streams one dimension across many points.
- **Packed indices**: candidate ids in a single contiguous slice carried intact
  to the scoring kernel (the thing currently lost to the pairs round-trip).
- **Cache-miss subset**: the candidates for which no cached distance exists; the
  only ids that reach `batch_distances` on a warm path.
- **Port / adapter**: `DataSource` is the port; `DenseMatrixProvider` an adapter.
  Domain depends on the port, never on adapter internals.

### Documentation and skills to consult

- Design: `docs/chutoro-design.md` §6.3. Roadmap: `docs/roadmap.md` 2.3.1, 2.4.1;
  determinism 1.1.2.
- Testing: `docs/property-testing-design.md` (§2.3.1 search-correctness oracle),
  `docs/rust-testing-with-rstest-fixtures.md`, `docs/rust-doctest-dry-guide.md`,
  `docs/reliable-testing-in-rust-via-dependency-injection.md`,
  `docs/complexity-antipatterns-and-refactoring-strategies.md`.
- ADRs: `docs/adr-001-commit-post-processing.md`,
  `docs/adr-002-adoption-of-kani-formal-verification.md`; this plan adds
  `docs/adr-003-soa-prefetch-adapter-boundary.md`.
- Skills: `rust-router` → `rust-performance-and-layout`,
  `rust-memory-and-state`, `rust-unsafe-and-ffi` (only if prefetch needs
  `unsafe`), `rust-unit-testing`, `proptest`, `kani`, `nextest`,
  `hexagonal-architecture`, `arch-decision-records`, `execplans`.

### Prior art (cite in the §6.3 update and ADR-003)

- **instant-distance** (pure-Rust HNSW): coordinates in a separate immutable
  array, fixed-width sentinel-padded neighbour arrays, per-node `RwLock`,
  read-snapshot → lock-free scoring → narrow write lock. Matches chutoro's
  existing design and the deferred secondary-SoA-copy alternative.
- **hnswlib** (C++): flat fixed-stride records placing ids and coordinates in
  adjacent cache lines.
- **Prefetch**: `_mm_prefetch` stable since 1.27, safe, x86_64-only;
  `_MM_HINT_T0`/`_MM_HINT_NTA`; tune look-ahead empirically; expect frequent
  no-ops/regressions (Lemire; Algorithmica; parallel-rust-cpp).
- **Benchmarking**: Criterion `benchmark_group` with `Throughput::Elements`;
  `critcmp` baselines; `hyperfine`; cycle counts via `perf`/`cachegrind`/`iai`;
  beware AVX throttling and denormals.

## Plan of work

Milestone by milestone; each ends with its own validation. Stages follow
Red-Green-Refactor where a test framework applies.

### Milestone 0 — Measurement and pre-registered go/no-go (C4, prototyping)

Goal: produce the evidence that decides whether E1-E3 are built at all.

1. Add `chutoro-benches/benches/neighbour_scoring.rs` (registered in
   `chutoro-benches/Cargo.toml` as `[[bench]] name = "neighbour_scoring"`). This
   is also the canonical 2.4.1 artefact — cross-reference it in `docs/roadmap.md`
   2.4.1 so that item consumes rather than rebuilds it. Bucket by candidate-set
   sizes that *occur*: 8, 16, 24, 32, 48 (straddling `2M` for M∈{8,16,24}); add
   diagnostic-only buckets {256, 1024} labelled "does not occur in the HNSW hot
   path". Dimensions {32, 128, 768}. Use `Throughput::Elements`, `black_box`, a
   fixed seed. Record per-bucket effective SoA lane utilisation *including*
   16-lane padding waste, and SoA-vs-scalar crossover.
2. Instrument a real synthetic HNSW build (N ∈ {10k, 100k}) to capture, as data
   written under `target/benchmarks/`:
   - the per-call candidate-set-size histogram *after* cache-miss subsetting
     (cold vs warm graph);
   - total `batch_distances` call count and aggregate candidates scored per
     build;
   - the share of build wall-time in scoring vs graph mutation vs MST vs
     hierarchy;
   - distance-cache `Mutex<LruCache>` contention (acquisitions, lock-wait) under
     parallel trim.
3. Capture cycle/instruction counts for the `neighbour_scoring` buckets via
   `perf stat`/`cachegrind` (or `iai`/`iai-callgrind` if its dev-dependency
   policy is cleared — flag in Decision Log). Save a Criterion baseline
   (`--save-baseline before`). Pin with `taskset`; record CPU model,
   `target-cpu`, and 1-min load average at capture.
4. Add `scripts/bench-neighbour-scoring.sh` (a `hyperfine` whole-binary wrapper,
   corroboration only) documented in `docs/developers-guide.md`.

Pre-registered thresholds (record in the plan before implementing E1-E3): a
conditional delta is **built only if** (a) scoring is a non-trivial share of
build wall-time (e.g. ≥10%), AND (b) the realistic-bucket regime it targets
actually occurs in the measured histogram, AND (c) a prototype shows ≥5% median
cycle-count improvement with non-overlapping CIs at those buckets. Otherwise the
delta is dropped and the null result recorded.

Validation: `make bench` builds; the harness runs; all distributions and the
contention numbers are captured and written into `Surprises & discoveries` and
`Outcomes`. Go/no-go for E1/E2/E3 decided and recorded here.

### Milestone 1 — Write-lock-free-scoring invariant guard (C1, committed)

Goal: turn "no scoring under the write lock" into a tested, durable invariant.

1. Red/Green: add a same-thread marker to `CpuHnsw` — e.g. a thread-local or an
   `AtomicBool`-per-thread flag set true for the duration of each `write_graph`
   closure (`cpu/internal.rs:45-51`) and false after. Add an instrumented
   `DataSource` decorator (test-only) that, on each `distance`/`batch_distances`
   call, asserts the marker is false (the calling thread is not inside a
   `write_graph` scope). Build a small index single-threaded (no concurrent
   search) through the decorator and assert no violation across plan, trim, and
   commit.
2. Prove teeth: temporarily move a scoring call inside a `write_graph` closure,
   confirm the guard fails, then revert.

Validation: guard passes on real code, fails on the deliberate mutation;
`make test` green. (This milestone is independent of Milestone 0's outcome.)

### Milestone 2 — Documentation, ADR, roadmap reflecting the evidence (C2/C3, committed)

Goal: record what is true regardless of whether E1-E3 are built.

1. `docs/chutoro-design.md` §6.3 implementation-update for 2.3.1: how 2.2.x
   satisfies the packed-index/SoA/scoring-outside-lock *structure*; the
   cache-fragmentation reality; the scoring-outside-write-lock invariant and its
   guard; the measured candidate-size/scoring-share/contention numbers; and a
   null-result template if E1-E3 are dropped.
2. `docs/adr-003-soa-prefetch-adapter-boundary.md` (Y-Statement;
   `arch-decision-records` skill): the decision to keep SoA/prefetch private to
   the dense adapter; the cross-node-beam and secondary-SoA-copy and
   `batch_distances_into` alternatives with memory-vs-repack and policy-coupling
   trade-offs; a recommendation to spin measured-win work into a separate
   evidence-first roadmap item. Reference it from §6.3.
3. `docs/developers-guide.md`: the measurement harness and `hyperfine` workflow,
   the write-lock invariant test, and (if E1-E3 land) the scratch and
   `simd_prefetch` conventions. Document the `DataSource` length-equality and
   error-atomicity (output-unmodified-on-error) invariants the cache layer relies
   on (`validate.rs:129-137`, `helpers.rs:152-160`) directly in the trait docs.
4. Mark roadmap item 2.3.1 done once committed scope lands (the item's intent is
   satisfied; any deferred work is a new item).

Validation: `make markdownlint`, `make nixie` (if Mermaid changes), `make fmt`
clean.

### Milestone 3 — Core-side allocation hygiene (E1, conditional on Milestone 0)

Proceed only if Milestone 0 clears the threshold for core-side allocations.

Files: `chutoro-core/src/hnsw/search.rs` and a NEW sibling module (e.g.
`hnsw/search/scratch.rs` or `hnsw/search_state.rs`) extracted **first** to stay
under 400 lines; `cpu/trim.rs`, `helpers.rs`.

1. Extract the scratch machinery and `SearchState` scratch fields into the new
   module before wiring reuse; add a line-budget check to validation.
2. Red: a **proptest differential** test (reusing the §2.3.1 search-correctness
   oracle across randomised graphs and `ef` values, not three fixtures) asserting
   the scratch-reusing path yields identical `Vec<Neighbour>`; assert scratch
   length is 0 at the top of each iteration; assert the Milestone 0 allocation
   count drops.
3. Green: reuse a per-search `Vec<usize>` for the per-node `fresh` list and a
   `Vec<f32>` for distances, cleared each iteration. Retarget the actual cached
   path allocations: `missing` (`validate.rs:123-127`),
   `miss_candidates`/`miss_meta` (`helpers.rs:134-135`), and the `fresh` Vec —
   plus the `metric_descriptor()` `Arc` clone per batch (`helpers.rs:105,132`;
   `validate.rs:15`) so the baseline is attributed correctly.
4. Refactor; keep files < 400 lines.

Validation: differential equality passes; allocation-count drops; existing HNSW
unit/property/parity suites pass unchanged; cycle-count cross-check shows the
pre-registered improvement; determinism preserved.

### Milestone 4 — Query-centric port override + SoA buffer reuse (E2, conditional)

Proceed only if Milestone 0 clears the threshold AND E2's prerequisite is
justified.

Files: `chutoro-core/src/datasource.rs` (defaulted port method),
`chutoro-providers/dense/src/{provider.rs,simd/mod.rs,simd/point_view.rs}`.

1. Add a *defaulted* query-centric port method carrying `(query, &[candidate])`
   intact (e.g. `batch_distances` is already query-centric in signature — the
   change is for the **dense adapter to override it** directly, bypassing the
   pairs round-trip, and/or a new defaulted method if a buffer-out form is
   needed). The default delegates to the existing path so text/synthetic
   providers are unchanged.
2. Thread a reusable `PackedSoaStorage` scratch through packing:
   `from_row_indices(..., scratch: &mut PackedSoaStorage)` (update the
   `dense/src/simd/mod.rs:102` caller). Scratch is owned per-call or per-job
   (NOT a shared `Mutex`/`RefCell`); it is `Vec<AlignedBlock>` preserving 64-byte
   alignment and the 16-lane zero-padded tail.
3. Red: the adversarial reuse test — pack a large batch (≥17 points, NaN/Inf in
   trailing rows), then pack a 2-point batch into the SAME storage, asserting
   `coordinate_block(d)[point_count..padded_point_count]` is bit-exactly `0.0`
   for every dimension; plus length-equality and output-unmodified-on-error
   assertions. Extend the 2.2.7 tail-padding Kani harness to the reused-buffer
   path. `debug_assert` scratch is reset before refill.
4. Green/Refactor: implement reuse; keep everything `pub(crate)` (no SoA type
   leaks across the boundary).

Validation: dense unit tests, backend-parity suite (2.2.6), and the extended
tail-padding Kani harness (2.2.7) pass; cycle-count cross-check clears the
threshold; parity within epsilon preserved.

Note: `batch_distances_into` (out-buffer reuse) remains **deferred** — on the
cached path core never calls `batch_distances` directly, so a reusable `out`
saves nothing there, and the defaulted impl penalises non-overriding providers.
Record in ADR-003, do not build here.

### Milestone 5 — Packing-step prefetch behind `simd_prefetch` (E3, conditional)

Proceed only if Milestone 0 shows a large-batch regime that actually occurs and
a prototype clears the threshold. Given the dimension-major layout, prefetch (if
anywhere) targets the scattered `matrix.row(index)` gather in the *pack* step
(`point_view.rs:79-84`), not the kernel scan.

Files: `chutoro-providers/dense/Cargo.toml` (non-default `simd_prefetch`
feature), a `prefetch` helper module, `point_view.rs` (pack loop).

1. Add a `prefetch_t0` helper: `#[cfg(all(target_arch = "x86_64", feature =
   "simd_prefetch"))]` calls `core::arch::x86_64::_mm_prefetch::<_MM_HINT_T0>`;
   every other target/feature is an inlined no-op (prefetch is semantically a
   no-op, so omitting it is always correct).
2. In the pack loop, prefetch the next candidate's source row a tunable few rows
   ahead; keep hot-loop temporaries in scalar locals (avoids the optimizer
   regression the research observed).
3. A/B with `critcmp` and cycle counts across the realistic buckets.

Go/no-go: keep only on a statistically significant cycle-count win with no
bucket regressing; otherwise **revert the entire milestone** (feature flag,
helper, ADR prefetch clause) rather than leaving a dead flag and an ADR about a
no-op. If kept, document `simd_prefetch` as host-tuned, off by default, requiring
re-benchmarking per target, and record the exact CPU model and `target-cpu` in
Outcomes.

### Milestone 6 — Final evidence and close-out (D5)

1. Re-run `neighbour_scoring` + `critcmp before after` + cycle counts +
   `hyperfine`; append a results table (or a null-result statement) to §6.3 and
   `Outcomes`.
2. Finalise ADR-003 and `developers-guide`; confirm `users-guide.md` needs
   changes only if a user-visible feature flag was kept.
3. Confirm roadmap 2.3.1 marked done; ensure any deferred work is filed as a new
   item.

Validation: `make check-fmt`, `make lint`, `make test`, `make kani`,
`make markdownlint`, `make nixie` all pass.

## Concrete steps

From the repository root; use `tee` for long output (global agent instructions):

```bash
make check-fmt 2>&1 | tee "/tmp/check-fmt-chutoro-$(git branch --show-current).out"
make lint      2>&1 | tee "/tmp/lint-chutoro-$(git branch --show-current).out"
make test      2>&1 | tee "/tmp/test-chutoro-$(git branch --show-current).out"
```

Measurement and comparison (Milestone 0; cycle count primary):

```bash
# Realistic buckets only for keep/drop; >=256 are diagnostic.
taskset -c 0-3 cargo bench -p chutoro-benches --bench neighbour_scoring -- --save-baseline before
# Cycle/instruction counts (noise-immune primary signal):
perf stat -r 20 -- target/release/deps/neighbour_scoring-* --bench 16   # example bucket
critcmp before after   # corroboration only
bash scripts/bench-neighbour-scoring.sh   # hyperfine; corroboration only
```

Prefetch A/B (Milestone 5, only if reached):

```bash
cargo bench -p chutoro-benches --bench neighbour_scoring -- --save-baseline noprefetch
cargo bench -p chutoro-benches --features chutoro-providers-dense/simd_prefetch \
  --bench neighbour_scoring -- --save-baseline prefetch
critcmp noprefetch prefetch
```

Kani (after Milestone 4, if reached):

```bash
make kani 2>&1 | tee "/tmp/kani-chutoro-$(git branch --show-current).out"
```

Expected: clippy `0 warnings`; tests `... passed`; Kani
`VERIFICATION:- SUCCESSFUL`. Keep decisions cite cycle-count deltas, not
wall-time alone.

## Validation and acceptance

1. **Committed scope always lands**: C1 guard passes (and fails on a deliberate
   under-lock scoring mutation); C2/C3 docs + ADR-003 merged; C4 measurement
   captured. Roadmap 2.3.1 marked done on committed scope.
2. **Identical clustering/neighbour selection** under fixed seeds: existing
   search-correctness oracle, idempotency, mutation, and backend-parity suites
   pass unchanged; E1/E2 differential equality tests confirm output-preserving
   refactors.
3. **Conditional deltas are evidence-gated**: each of E1/E2/E3 is implemented
   only on a ≥5% median cycle-count win with non-overlapping CIs at the realistic
   buckets, on a pinned, low-load host; otherwise dropped with the null result
   recorded. A committed-scope-only outcome is acceptance, not failure.
4. **Quality gates**: `make check-fmt`, `make lint`, `make test`, `make kani`,
   `make markdownlint`, `make nixie` pass.

Red-Green-Refactor evidence per implemented milestone: the red command + expected
failure, the green command + pass, the refactor command + pass.

Quality criteria ("done"): committed scope merged; any conditional delta backed
by cycle-count evidence; no bucket regresses on the default feature set; no new
dependency without a flagged Decision Log entry; no `unsafe` without a SAFETY
comment plus a Miri/Kani check.

Quality method: local gates with `tee`; `coderabbit review --agent` after each
major milestone with all concerns cleared, run only after the deterministic gates
are green.

## Idempotence and recovery

All steps re-runnable. Criterion baselines are named. The `simd_prefetch` feature
is additive and reverted wholesale if dropped. Commit per milestone for clean
`git revert`. No destructive step.

## Artifacts and notes

Record, as concise indented transcripts: the Milestone 0 candidate-size and
miss-subset histograms, the scoring-share-of-build-time number, the cache
contention figures, the `critcmp` and `perf stat` tables, and the Kani success
line. If E1-E3 are dropped, record the null result and the deferred-item proposal.

## Interfaces and dependencies

No new runtime dependencies. `perf`/`cachegrind` are dev tools; an `iai` dev-dep,
if used, is flagged in the Decision Log. Interfaces at completion:

- A test-only same-thread "inside write_graph" marker on `CpuHnsw` and an
  instrumented `DataSource` decorator (C1).
- `chutoro-benches` gains `[[bench]] name = "neighbour_scoring"` (C4 / 2.4.1).
- If E2 lands: a *defaulted*, query-centric override on the dense adapter
  bypassing the pairs round-trip, and
  `from_row_indices(..., scratch: &mut PackedSoaStorage)` (internal). No SoA type
  crosses the crate boundary.
- If E3 lands: a non-default `simd_prefetch` feature and an internal `prefetch`
  helper:

```rust
// chutoro-providers/dense/src/simd/prefetch.rs
/// Hints the CPU to begin loading `ptr`. A no-op where unsupported; prefetch is
/// semantically a hint, so omitting it is always correct.
#[inline]
pub(crate) fn prefetch_t0(ptr: *const f32) {
    #[cfg(all(target_arch = "x86_64", feature = "simd_prefetch"))]
    core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    #[cfg(not(all(target_arch = "x86_64", feature = "simd_prefetch")))]
    let _ = ptr;
}
```

- No change to `DataSource::distance`/`distance_batch`, `CpuHnsw` public methods,
  or the CLI. `batch_distances_into` is **deferred** (ADR-003).

## Signposted documentation and skills

- Process: `execplans`. Architecture: `hexagonal-architecture`,
  `arch-decision-records`, `arch-crate-design`; `docs/chutoro-design.md` §6.3.
- Rust: `rust-router` → `rust-performance-and-layout`, `rust-memory-and-state`,
  `rust-types-and-apis`, `rust-unsafe-and-ffi` (only if prefetch needs `unsafe`).
- Testing/verification: `rust-unit-testing`, `proptest`, `kani`, `nextest`,
  `rust-verification`; `docs/property-testing-design.md`,
  `docs/rust-testing-with-rstest-fixtures.md`, `docs/rust-doctest-dry-guide.md`,
  `docs/reliable-testing-in-rust-via-dependency-injection.md`,
  `docs/complexity-antipatterns-and-refactoring-strategies.md`.

## Revision note

Revision 2 (2026-06-09): folded the Logisphere community-of-experts review
(verdict REVISE) — all 12 required revisions. Major changes: split into committed
scope (C1 guard, C2/C3 docs+ADR, C4 measurement) and evidence-gated conditional
scope (E1 allocation hygiene, E2 port override + `Vec<AlignedBlock>` scratch
reuse, E3 packing-step prefetch); named the pack→unpack→repack round-trip and
made a query-centric override E2's prerequisite; added the cache-fragmentation
reality and made Milestone 0 measure miss-subset distributions, scoring share of
build time, and cache-lock contention; re-centred benchmark buckets on the
realistic 8-48 regime with cycle-count as the primary keep/drop signal; corrected
the C1 guard to a same-thread marker; corrected the scratch type to
`Vec<AlignedBlock>` and forbade shared scratch; pre-committed `search.rs` module
extraction; deferred the structural levers (cross-node beam, secondary SoA copy,
`batch_distances_into`) to a separate evidence-first item recorded in ADR-003.
Revision 1 (2026-06-09): initial draft. Pending: user approval before
implementation.
