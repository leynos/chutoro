# Restructure HNSW neighbour evaluation: packed indices, SoA scoring, and prefetch (2.3.1)

This ExecPlan (execution plan) is a living document. The sections `Constraints`,
`Tolerances`, `Risks`, `Progress`, `Surprises & Discoveries`, `Decision Log`,
and `Outcomes & Retrospective` must be kept up to date as work proceeds.

Status: ACTIVE (implementation in progress)

Roadmap item: 2.3.1 (Phase 2.3, Hot-path optimizations). See `docs/roadmap.md`
lines 319-321 and `docs/chutoro-design.md` §6.3 (lines 887-963).

## Purpose / big picture

chutoro builds a CPU Hierarchical Navigable Small World (HNSW) graph to harvest
candidate edges for clustering. The hot path is *neighbour evaluation*: for
each node visited during search and insertion, the engine reads that node's
neighbour list and computes the distance from a query point to every neighbour,
then keeps the closest. Roadmap item 2.3.1 asks that this evaluation use
*packed* candidate indices and a *structure-of-arrays* (SoA) coordinate layout,
*prefetch* upcoming coordinate blocks, and score *outside the write lock*.

Reconnaissance (recorded under `Surprises & discoveries`) shows that the
*structural* intent of 2.3.1 is already realised by Phase 2.2 (SIMD kernels)
and the existing two-phase locking design — but with one genuinely unrealised
seam (a pack→unpack→repack round-trip at the port) and a measurement question
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
   the real candidate-set-size distribution, the cache-miss-subset
   distribution, accumulated batch-scoring time against build wall-clock, and
   distance-cache lock contention.
2. **Conditional scope (delivered only on evidence).** Three hot-path deltas —
   core-side allocation hygiene (E1), SoA packing-buffer reuse plus its
   required query-centric port override (E2), and packing-step prefetch (E3) —
   each implemented **only if** the committed-scope measurement clears a
   pre-registered threshold. If the evidence does not clear the bar, the
   correct and expected outcome is that the committed scope ships and E1-E3 do
   not. That is a success, not a shortfall.

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
   behaviour-preserving, proven by the existing search-correctness,
   idempotency, mutation, and backend-parity suites plus the new write-lock
   guard.

### What is already realised vs genuinely residual (read before estimating)

Confirmed by reading the code (file:line cited in `Surprises & discoveries`):

- **Scoring already happens outside the write lock.** `InsertionPlanner::plan`
  (`insert/planner.rs:78`) scores under the *read* lock;
  `CpuHnsw::score_trim_jobs` (`cpu/trim.rs:65`) scores with **no lock held**,
  in parallel via Rayon; neither `write_graph` closure in
  `insert_with_collector` (`cpu/mod.rs:402-416`) computes a distance. This is
  the plan's strongest, most durable deliverable (committed scope C1).
- **SoA scoring already exists** as an adapter detail: `DenseMatrixProvider`
  reaches a 64-byte-aligned, 16-lane-padded, dimension-major SoA
  `DensePointView` (`dense/src/simd/point_view.rs`) and
  AVX-512/AVX2/Neon/scalar kernels.

Genuinely unrealised or unmeasured (the residual the panel surfaced):

- **The "packed indices" round-trip.** The dense provider does **not** override
  `DataSource::batch_distances` (it overrides only `distance` and
  `distance_batch`, `provider.rs:137,143`). So core's *default*
  `batch_distances` (`datasource.rs:161-177`) turns the packed `&[usize]` into
  a fresh `Vec<(usize,usize)>` of pairs, then `shared_query_candidates`
  (`dense/src/simd/mod.rs:195-213`) scans those pairs to *re-derive* the shared
  query and re-collects a fresh candidate `Vec`, before `from_row_indices`
  (`point_view.rs:68`) allocates `PackedSoaStorage`. The roadmap item is
  literally titled "use packed indices"; structurally the code packs, unpacks
  to pairs, then repacks every call. This is the clearest unrealised seam (E2's
  prerequisite).
- **The distance cache fragments batches.** The cache is present on *both* the
  insertion and search paths (`Some(cache)` at `cpu/mod.rs:394,456,466`). So
  the slice that actually reaches `batch_distances` is the cache-*miss* subset
  (`validate.rs:123-127`, `helpers.rs:134-151`), not the full ≤ `2M` neighbour
  list. As the cache warms, miss subsets routinely fall below
  `should_pack_query_points`'s `candidate_count > 1` threshold
  (`dense/src/simd/mod.rs:163-173`) and drop to the scalar path. D1/D2/D3 only
  matter where miss counts are routinely large enough to pack — which must be
  *measured*, not assumed.
- **Prefetch has no obvious target in the kernel.** The SoA layout is
  dimension-major (`point_view.rs:82`), so the kernel already streams
  contiguous per-dimension blocks (hardware-prefetcher territory). The only
  scattered gather is `matrix.row(index)` during *packing*
  (`point_view.rs:79-84`). So prefetch, if it helps anywhere, belongs in the
  pack step, and is likely a no-op at realistic batch sizes (E3 is conditional
  and structurally suspect).

### Deferred to separate, evidence-first optimisation items (NOT built here)

The strongest measured wins are structural and out of scope for 2.3.1. They are
recorded in ADR-003 and **must be created as new roadmap items in
`docs/roadmap.md` during implementation of this plan** (a required step of
Milestone 2; see the Plan of work), rather than smuggled into 2.3.1. Creating
the roadmap entries is mandatory and is not contingent on the Milestone 0
go/no-go — the go/no-go decides whether the *conditional* deltas (E1-E3) of
2.3.1 are built, whereas these three structural levers are deferred work that
must be tracked openly whatever the measurement shows. File each as its own
checkbox item under Phase 2.3 (suggested numbers 2.3.3, 2.3.4, 2.3.5, after the
existing 2.3.1 and 2.3.2), each cross-referencing ADR-003 and this execplan,
and each marked as gated on its own evidence:

- **Cross-node "beam" scoring** (suggested roadmap item 2.3.3) — accumulate
  candidates from multiple popped nodes in `search_layer` into one
  packing+scoring call, widening the window from ≈ `2M` to hundreds. This
  touches core search *policy* and must preserve deterministic best-first
  visitation order; larger blast radius.
- **Secondary dimension-major (SoA) matrix copy** (suggested roadmap item
  2.3.4) — held once at provider construction, so `from_row_indices` becomes a
  strided gather with no per-call transpose and no packing buffer — trading ~1×
  memory to eliminate the path E2 optimises incrementally.
- **`batch_distances_into` out-buffer reuse** (suggested roadmap item 2.3.5) —
  only meaningful if a query-centric override exists and the cache's
  indexed-scatter consumption is redesigned; otherwise it saves nothing on the
  cached path and penalises non-overriding providers (e.g. text).

The suggested numbering is provisional: at implementation time, reconcile it
with the current state of `docs/roadmap.md` (renumber if 2.3.3+ are already
taken) and record the assigned numbers in this plan's `Decision Log`.

## Constraints

Hard invariants. Violation requires escalation, not a workaround.

1. **Behaviour preservation.** Clustering output and HNSW neighbour selection
   must be identical under fixed seeds, before and after. The deterministic
   tie-break (distance, then lower item id, then insertion sequence; `search.rs`
   `compare_neighbours`; `docs/roadmap.md` 1.1.2) must be preserved exactly.
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
4. **No scoring under the write lock.** Distance computation must never run
   while a `RwLockWriteGuard<Graph>` is held *on the current thread*. This is
   both a constraint and a tested invariant (C1).
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
   `perf` /`cachegrind`/`iai` which are dev-time tools, not crate deps — confirm
   `iai`/ `iai-callgrind` policy before adding it as a dev-dependency and flag
   in the Decision Log if used).
4. **Behaviour drift.** Any existing test asserting neighbour selection, recall,
   parity, or determinism changes its expected output. This must never happen.
5. **Evidence gate not cleared.** If committed-scope measurement (C4) does not
   clear the pre-registered keep threshold for a conditional delta, that delta
   is **not implemented**; stop and record the null result (this is an accepted
   outcome, not an escalation, but it must be documented in `Outcomes`).
6. **Iterations.** If a milestone's tests still fail after 3 focused attempts,
   stop and escalate.
7. **Regression masquerading as noise.** If a conditional delta shows any
   regression on the cycle-count cross-check at the realistic buckets, drop it;
   never keep a delta on "within wall-time noise".

## Risks

1. Risk: **Closed with motion, not measurement** — a feature flag, ADR, bench,
   and scratch buffers ship but build wall-time is unchanged because batches
   were always cache-miss-subset-narrow and the distance-cache `Mutex` was the
   real bound. (Pre-mortem A.) Severity: high. Likelihood: medium. Mitigation:
   committed scope is C1-C4 only; E1-E3 gate on a pre-registered C4 threshold
   derived from the *measured* miss-subset distribution; pre-author the
   null-result §6.3 update and Outcomes entry so "committed-scope-only" is a
   first-class success.
2. Risk: **Reuse buffer poisons a warm-cache build** — `PackedSoaStorage` reuse
   leaves NaN/Inf in the padded tail; a later smaller batch reads stale lanes;
   active-lane output is preserved most of the time, so a naive single-batch
   test passes and a non-reproducible determinism break ships. (Pre-mortem B.)
   Severity: high. Likelihood: low-medium. Mitigation: mandate the adversarial
   large-then-small reuse test with NaN/Inf trailing rows asserting bit-exact
   `0.0` tails; extend the 2.2.7 tail-padding Kani harness to the reused-buffer
   path; forbid shared scratch; `debug_assert` scratch is reset before refill.
3. Risk: **A real regression waved through as noise** on the contended 6-core
   host. (Pre-mortem C.) Severity: medium. Likelihood: medium. Mitigation:
   cycle/instruction count is the *primary* keep/drop signal; explicit minimum
   effect size; pin `RAYON_NUM_THREADS`; a documented load-average gate at
   capture time; "within noise" = drop-by-default for any code-adding delta.
4. Risk: **The optimised window is structurally too small** — per-node batches
   are ≈ `2M` (16-48), fragmented further to cache-miss subsets, with 16-lane
   padding wasting 30-47% at those sizes — so SoA packing overhead cancels the
   SIMD win and prefetch has nothing to hide. Severity: medium. Likelihood:
   high. Mitigation: this is exactly what C4 measures; E1-E3 proceed only if
   the data contradicts it; the real structural lever (cross-node beam) is
   deferred to a separate item.
5. Risk: **Benchmark noise / AVX throttling / denormals** mask effects.
   Severity: medium. Likelihood: medium. Mitigation: `taskset` pinning,
   flush-to-zero, realistic data, Criterion bootstrap CIs, `critcmp`, and the
   mandatory cycle-count cross-check; quiet host.
6. Risk: **Scratch reuse interacts badly with Rayon-parallel trim** — a
   shared `&self` `RefCell`/`Mutex` scratch serialises `score_trim_jobs`
   (erasing the 2.2 parallel-scoring win) or makes allocation-count assertions
   nondeterministic across thread counts. Severity: high. Likelihood: medium.
   Mitigation: forbid shared scratch in the Decision Log; require thread-local
   or explicit per-call/per-job scratch; reuse buffer is `Vec<AlignedBlock>`
   (64-byte aligned, 16-lane padded), not `Vec<f32>`; pin `RAYON_NUM_THREADS=1`
   for allocation-count tests if thread-local is chosen.

## Progress

- [x] (2026-06-09) Draft authored.
- [x] (2026-06-09) Community-of-experts (Logisphere) review: verdict REVISE;
  all 12 required revisions folded into this draft.
- [x] (2026-06-24) User approval received via implementation request; branch,
  PR title, upstream tracking, Lody session title, and PR reference link
  aligned.
- [x] Milestone 0 (C4): measurement harness + cache/scoring/contention data +
  pre-registered go/no-go thresholds.
  - [x] (2026-06-24) Red stage captured:
    `cargo bench -p chutoro-benches --bench neighbour_scoring -- --list`
    reported that no `neighbour_scoring` bench target existed.
  - [x] (2026-06-24) Added `chutoro-benches/benches/neighbour_scoring.rs`,
    `chutoro-benches/benches/neighbour_scoring/support.rs`, registered the
    benchmark in `chutoro-benches/Cargo.toml`, and added
    `scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-24) Green stage captured: the benchmark lists realistic
    buckets `{8, 16, 24, 32, 48}` and diagnostic buckets `{256, 1024}` for
    dimensions `{32, 128, 768}`.
  - [x] (2026-06-24) Targeted validation passed after splitting the benchmark
    support module to keep every Rust source file below 400 lines:
    `cargo clippy -p chutoro-benches --bench neighbour_scoring -- -D warnings`
    and `cargo bench -p chutoro-benches --bench neighbour_scoring -- --list`.
  - [x] (2026-06-24) Full `make test` exposed an existing Criterion exact-probe
    timeout in `chutoro-benches::bench/extraction`; the extraction benchmark now
    follows the HNSW benchmark pattern and shortens warm-up/measurement windows
    only when nextest invokes one exact benchmark case as a test.
  - [x] (2026-06-24) Deterministic gates for this milestone pass:
    `make check-fmt`, `make markdownlint`, `make lint`, and `make test` (1009
    passed, 1 skipped). `make fmt` was also run; it required one narrow
    Markdown lint suppression in the older 2.2.6 execplan after the formatter
    rewrote a long nightly command.
  - [x] (2026-06-24) First CodeRabbit review completed with five valid
    findings. Fixes applied: report writing now uses `cap_std::fs_utf8` and
    `camino`; branch names are sanitized before composing benchmark log paths;
    the measured Criterion loop borrows the fixture candidates instead of
    cloning them; zero-candidate lane utilisation returns `0`; and the
    diagnostic math now has library-level unit/property tests.
  - [x] (2026-06-24) Post-CodeRabbit deterministic gates pass:
    `make check-fmt`, `make markdownlint`, `make lint`, `make test` (1015
    passed, 1 skipped), plus `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-24) Second CodeRabbit review completed with three valid
    findings. Fixes applied: lane-utilisation arithmetic is total for extreme
    `usize` inputs, poisoned profiling mutexes no longer abort benchmark
    scoring calls, and the benchmark runner derives the exact executable from
    Cargo JSON output instead of scanning `target/release/deps`.
  - [x] (2026-06-24) Post-second-review deterministic gates pass:
    `make check-fmt`, `make markdownlint`, `make lint`, `make test` (1016
    passed, 1 skipped), plus targeted `cargo clippy -p chutoro-benches
    --all-targets -- -D warnings`, `cargo test -p chutoro-benches
    neighbour_scoring --lib`, `cargo bench -p chutoro-benches --bench
    neighbour_scoring -- --list`, and `shellcheck
    scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Third CodeRabbit review completed with one valid minor
    finding. The HNSW build profile now labels its timing metric as
    accumulated batch-scoring time compared with build wall-clock instead of a
    literal wall-time share, because parallel scoring can accumulate more CPU
    duration than elapsed build time.
  - [x] (2026-06-25) Post-third-review deterministic gates pass:
    `make check-fmt`, `make markdownlint`, `make lint`, `make test` (1016
    passed, 1 skipped), plus targeted `cargo clippy -p chutoro-benches
    --all-targets -- -D warnings`, `cargo test -p chutoro-benches
    neighbour_scoring --lib`, `cargo bench -p chutoro-benches --bench
    neighbour_scoring -- --list`, and `shellcheck
    scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Fourth CodeRabbit review completed with two valid minor
    findings. Fixes applied: the `hyperfine` wrapper now uses `mktemp` plus an
    exit trap for its tee log path, and the build-profile median calculation
    now averages the two central sorted values for even-length samples with
    regression coverage in `chutoro-benches::neighbour_scoring`.
  - [x] (2026-06-25) Post-fourth-review deterministic gates pass:
    `make check-fmt`, `make markdownlint`, `make lint`, `make test` (1020
    passed, 1 skipped), plus targeted `cargo clippy -p chutoro-benches
    --all-targets -- -D warnings`, `cargo test -p chutoro-benches
    neighbour_scoring --lib` (11 passed), `cargo bench -p chutoro-benches
    --bench neighbour_scoring -- --list`, and `shellcheck
    scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Fifth CodeRabbit review completed with seven findings.
    Fixes applied: the inline Criterion module has module-level docs; benchmark
    setup validates the fixed query/candidate scoring call before timing; the
    nextest-probe timing hook no longer overloads Criterion's `--exact`; bench
    support errors now preserve `DataSourceError` and `io::Error` semantics via
    a `thiserror` enum; and the shell wrapper now uses portable `mktemp`,
    explicit Cargo build failure handling, and `jq`-based Cargo JSON parsing.
  - [x] (2026-06-25) Post-fifth-review deterministic gates pass:
    `make check-fmt`, `make markdownlint`, `make lint`, and `make test` (1020
    passed, 1 skipped). A first full `make test` rerun exposed a
    non-reproducing core HNSW mutation proptest case; the targeted rerun passed
    and the full workspace rerun then passed. Targeted follow-up checks also
    pass: `cargo clippy -p chutoro-benches --all-targets -- -D warnings`,
    `cargo test -p chutoro-benches neighbour_scoring --lib` (11 passed),
    `cargo bench -p chutoro-benches --bench neighbour_scoring -- --list`, and
    `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Sixth CodeRabbit review completed after the required
    randomized rate-limit backoff. Fixes applied: benchmark iterations now fail
    fast if `score_candidates` returns an unexpected error, and
    `scripts/bench-neighbour-scoring.sh` checks for `cargo`, `jq`, and
    `hyperfine` before building or parsing benchmark artefacts.
  - [x] (2026-06-25) Post-sixth-review deterministic gates pass:
    `make check-fmt`, `make lint`, and `make test` (1020 passed, 1 skipped),
    plus `cargo bench -p chutoro-benches --bench neighbour_scoring -- --list`
    and `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Seventh CodeRabbit review completed after the required
    randomized rate-limit backoff. Fixes applied: `duration_basis_points` now
    uses integer nanosecond basis-point maths instead of floating-point
    seconds, and the simple diagnostic helper tests were collapsed into
    table-driven `rstest` cases while keeping the property-style checks
    standalone.
  - [x] (2026-06-25) Post-seventh-review deterministic gates pass:
    `make check-fmt`, `make lint`, and `make test` (1026 passed, 1 skipped),
    plus targeted `cargo clippy -p chutoro-benches --all-targets -- -D
    warnings`, `cargo test -p chutoro-benches neighbour_scoring --lib` (17
    passed), `cargo bench -p chutoro-benches --bench neighbour_scoring --
    --list`, and `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Eighth CodeRabbit review completed with three valid
    findings. Fixes applied: synthetic dense fixture generation now builds one
    flat row-major `Vec<f32>` instead of nested row vectors; build-profile
    accumulated scoring time now counts only batched `DataSource` calls, while
    scalar distance calls remain count-only; and CSV schemas now render through
    tested library helpers rather than untested bench-only formatting code.
  - [x] (2026-06-25) Post-eighth-review deterministic gates pass:
    `make check-fmt`, `make lint`, and `make test` (1028 passed, 1 skipped),
    plus targeted `cargo clippy -p chutoro-benches --all-targets -- -D
    warnings`, `cargo test -p chutoro-benches neighbour_scoring --lib` (19
    passed), `cargo bench -p chutoro-benches --bench neighbour_scoring --
    --list`, and `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Ninth CodeRabbit review completed after the required
    randomized rate-limit backoff. Three valid findings were fixed:
    report-rendering tests now use contextual `expect(...)` diagnostics,
    build-profile snapshots clone the guarded stats value directly, and the
    optional build-profile environment flag is normalised before truthy-value
    matching. One critical module-root finding was stale: the worktree contains
    only `chutoro-benches/src/neighbour_scoring/mod.rs` and its `report.rs`
    submodule, not a competing `src/neighbour_scoring.rs` root.
  - [x] (2026-06-25) Post-ninth-review deterministic gates pass:
    `make check-fmt`, `make lint`, and `make test` (1028 passed, 1 skipped),
    plus targeted `cargo clippy -p chutoro-benches --all-targets -- -D
    warnings`, `cargo test -p chutoro-benches neighbour_scoring --lib` (19
    passed), `cargo bench -p chutoro-benches --bench neighbour_scoring --
    --list`, and `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Tenth CodeRabbit review completed with three valid
    findings. Fixes applied: the lane-utilisation CSV writer now quotes and
    escapes string fields when needed; report helper coverage now uses
    `rstest` fixtures and parameterized cases, including escaped-field and
    header-only edge paths; and the `hyperfine` wrapper shell-escapes the
    benchmark binary before interpolating it into the command string.
  - [x] (2026-06-25) Post-tenth-review deterministic gates pass:
    `make check-fmt`, `make lint`, and `make test` (1030 passed, 1 skipped),
    plus targeted `cargo clippy -p chutoro-benches --all-targets -- -D
    warnings`, `cargo test -p chutoro-benches neighbour_scoring --lib` (21
    passed), `cargo bench -p chutoro-benches --bench neighbour_scoring --
    --list`, and `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Eleventh CodeRabbit review completed with two valid
    findings and one repeated stale finding. Fixes applied: report padded-lane
    rendering now uses checked padding and the same u128 overflow fallback as
    the diagnostic helper; benchmark support errors now preserve dense-provider,
    synthetic-source, HNSW-parameter, and HNSW-build source errors instead of
    stringifying them. The repeated duplicate-root finding remained stale:
    `git ls-files` and a filesystem check confirmed
    `chutoro-benches/src/neighbour_scoring.rs` does not exist.
  - [x] (2026-06-25) Post-eleventh-review deterministic gates pass:
    `make check-fmt`, `make lint`, and `make test` (1030 passed, 1 skipped),
    plus targeted `cargo clippy -p chutoro-benches --all-targets -- -D
    warnings`, `cargo test -p chutoro-benches neighbour_scoring --lib` (21
    passed), `cargo bench -p chutoro-benches --bench neighbour_scoring --
    --list`, and `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Twelfth CodeRabbit review completed with two valid
    findings. Fixes applied: CSV escaping now returns borrowed data on the
    no-escape path rather than allocating every label, and the optional
    build-profile report file is created only after the fallible HNSW profiling
    loop has successfully produced rows.
  - [x] (2026-06-25) Post-twelfth-review deterministic gates pass:
    `make check-fmt`, `make lint`, and `make test` (1030 passed, 1 skipped),
    plus targeted `cargo clippy -p chutoro-benches --all-targets -- -D
    warnings`, `cargo test -p chutoro-benches neighbour_scoring --lib` (21
    passed), `cargo bench -p chutoro-benches --bench neighbour_scoring --
    --list`, and `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Thirteenth CodeRabbit review completed with two valid
    report-helper findings and one repeated stale duplicate-root finding.
    Fixes applied: report rendering now derives padded and wasted lane counts
    from a single `padded_lane_count` source of truth, and the lane-utilisation
    CSV writer has property coverage for generated bucket labels and candidate
    counts, including CSV round-trip parsing of quoted fields.
  - [x] (2026-06-25) Post-thirteenth-review deterministic gates pass:
    `make check-fmt`, `make markdownlint`, `make lint`, and `make test` (1031
    passed, 1 skipped), plus targeted `cargo clippy -p chutoro-benches
    --all-targets -- -D warnings`, `cargo test -p chutoro-benches
    neighbour_scoring --lib` (22 passed), `cargo bench -p chutoro-benches
    --bench neighbour_scoring -- --list`, and `shellcheck
    scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Fourteenth CodeRabbit review completed with three valid
    findings and one repeated stale duplicate-root finding. Fixes applied:
    build-profile stats are moved out of the profiling mutex with `mem::take`
    instead of cloning the batch-size vector; median has generated sorted-input
    coverage for empty, odd, and even cases; and the neighbour-scoring
    benchmark's short-measurement mode now uses the supported
    `CHUTORO_BENCH_NEIGHBOUR_SHORT_MEASUREMENT` environment variable rather
    than an ad hoc Criterion CLI flag.
  - [x] (2026-06-25) Post-fourteenth-review deterministic gates pass:
    `make check-fmt`, `make lint`, and `make test` (1032 passed, 1 skipped),
    plus targeted `cargo clippy -p chutoro-benches --all-targets -- -D
    warnings`, `cargo test -p chutoro-benches neighbour_scoring --lib` (23
    passed), `cargo bench -p chutoro-benches --bench neighbour_scoring --
    --list`, and `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Fifteenth CodeRabbit review completed with three valid
    findings and one repeated stale duplicate-root finding. Fixes applied:
    build-profile stats now use a plain `Mutex` instead of `Arc<Mutex<_>>`,
    benchmark reports honour `CARGO_TARGET_DIR` when it is set, and the
    build-profile CSV row writer uses one contiguous format string.
  - [x] (2026-06-25) Post-fifteenth-review deterministic gates pass:
    `make check-fmt`, `make lint`, and `make test` (1032 passed, 1 skipped),
    plus targeted `cargo clippy -p chutoro-benches --all-targets -- -D
    warnings`, `cargo test -p chutoro-benches neighbour_scoring --lib` (23
    passed), `cargo bench -p chutoro-benches --bench neighbour_scoring --
    --list`, and `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Sixteenth CodeRabbit review completed with two valid
    findings. Fixes applied: build-profile CSV rendering now has generated
    duration-ratio coverage for zero, tiny, and large timings, and the
    benchmark wrapper asks Hyperfine to parse its `%q`-escaped benchmark
    command with Bash.
  - [x] (2026-06-25) Post-sixteenth-review deterministic gates pass:
    `make check-fmt`, `make lint`, and `make test` (1033 passed, 1 skipped),
    plus targeted `cargo clippy -p chutoro-benches --all-targets -- -D
    warnings`, `cargo test -p chutoro-benches neighbour_scoring --lib` (24
    passed), `cargo bench -p chutoro-benches --bench neighbour_scoring --
    --list`, and `shellcheck scripts/bench-neighbour-scoring.sh`.
  - [x] (2026-06-25) Current-tree validation before the next CodeRabbit review
    also passes after the execplan update: `make markdownlint`, `make
    check-fmt`, `make lint`, and `make test` (1033 passed, 1 skipped).
  - [x] (2026-06-25) Seventeenth CodeRabbit review completed with one valid
    trivial finding and two invalid findings. Fix applied: the test-only CSV
    parser now documents that it only supports the newline-terminated format
    emitted by these writers and is not a general-purpose CSV parser. Invalid
    findings: `CandidateCountConversion` is still constructed by
    `chutoro-benches/benches/neighbour_scoring.rs`, and the repeated duplicate
    `src/neighbour_scoring.rs` path is absent on disk.
  - [x] (2026-06-25) Post-seventeenth-review deterministic gates pass:
    `make check-fmt`, `make lint`, targeted `cargo test -p chutoro-benches
    neighbour_scoring --lib` (24 passed), `make markdownlint`, and `make test`
    (1033 passed, 1 skipped).
  - [x] (2026-06-25) Eighteenth CodeRabbit review completed with three valid
    findings and two repeated invalid findings. Fixes applied: profiling
    counters now use atomics for scalar calls, batch calls, total batch
    candidates, and elapsed batch-scoring nanoseconds while keeping the mutex
    only for exact batch-size collection; the lane-utilisation CSV writer uses
    inlined format arguments; and the Hyperfine wrapper writes to a stable
    branch-specific `/tmp` log rather than deleting its tee output on exit.
    Invalid findings: the repeated duplicate `src/neighbour_scoring.rs` path is
    absent, and `CandidateCountConversion` is still constructed by the
    benchmark entry point.
  - [x] (2026-06-25) Post-eighteenth-review deterministic gates pass:
    `make check-fmt`, `make lint`, targeted `cargo test -p chutoro-benches
    neighbour_scoring --lib` (24 passed), `cargo bench -p chutoro-benches
    --bench neighbour_scoring -- --list`, and `shellcheck
    scripts/bench-neighbour-scoring.sh`, `make markdownlint`, and `make test`
    (1033 passed, 1 skipped).
  - [x] (2026-06-25) Nineteenth CodeRabbit review completed with two valid
    minor findings and one repeated invalid finding. Fixes applied: the
    profiling wrapper moved into its own sibling bench module so both
    `support.rs` and the profiling wrapper stay comfortably below the 400-line
    source-file cap; the profiling duration/saturating-add helpers moved into
    `chutoro-benches::neighbour_scoring` with property tests that run under
    the normal library test harness. Invalid finding: the repeated duplicate
    `src/neighbour_scoring.rs` path is absent on disk.
  - [x] (2026-06-25) Post-nineteenth-review deterministic gates pass:
    `make check-fmt`, `make lint`, targeted `cargo test -p chutoro-benches
    neighbour_scoring --lib` (27 passed), `cargo bench -p chutoro-benches
    --bench neighbour_scoring -- --list`, `shellcheck
    scripts/bench-neighbour-scoring.sh`, `make markdownlint`, and `make test`
    (1036 passed, 1 skipped).
  - [x] (2026-06-25) Twentieth CodeRabbit review completed with one valid
    finding and two repeated invalid findings. Fix applied: the atomic
    profiling helper tests now include multithreaded `Arc`/`thread` coverage
    for both `saturating_add_usize` and `saturating_add_u64`, asserting the
    final accumulated value after all worker joins. Invalid findings:
    `CandidateCountConversion` is still constructed by the benchmark entry
    point, and the repeated duplicate `src/neighbour_scoring.rs` path is
    absent on disk.
  - [x] (2026-06-25) Post-twentieth-review deterministic gates pass:
    `make check-fmt`, `make lint`, targeted `cargo test -p chutoro-benches
    neighbour_scoring --lib` (29 passed), `cargo bench -p chutoro-benches
    --bench neighbour_scoring -- --list`, `shellcheck
    scripts/bench-neighbour-scoring.sh`, `make markdownlint`, and `make test`
    (1038 passed, 1 skipped).
  - [x] (2026-06-25) Twenty-first CodeRabbit review completed with three valid
    findings and one repeated invalid finding. Fixes applied:
    `record_batch` now pushes the batch size and updates profiling counters
    under the same successful batch-size lock so a single batch cannot be split
    across snapshots; the Hyperfine wrapper writes its stable branch-specific
    log under a user-owned private `/tmp/chutoro-benches-${UID}` directory;
    and env/report-path behaviour for the optional build-profile report now
    has integration coverage that runs under the normal test harness. Invalid
    finding: the repeated duplicate `src/neighbour_scoring.rs` path is absent.
  - [x] (2026-06-25) Post-twenty-first-review deterministic gates pass:
    `make check-fmt`, `make lint`, targeted `cargo test -p chutoro-benches
    --test neighbour_scoring_support` (10 passed), targeted `cargo test -p
    chutoro-benches neighbour_scoring --lib` (29 passed), `cargo bench -p
    chutoro-benches --bench neighbour_scoring -- --list`, `shellcheck
    scripts/bench-neighbour-scoring.sh`, and `make test` (1048 passed, 1
    skipped), followed by `make markdownlint` after this plan update.
  - [x] (2026-06-25) Twenty-second CodeRabbit review completed with two valid
    findings and one repeated invalid finding. Fixes applied: environment
    mutation tests now use a shared `chutoro-test-support::env::EnvVarGuard`
    instead of a local one-off helper, and build-profile environment/path
    behaviour moved behind a narrow `chutoro-benches::neighbour_scoring` seam
    so the integration test no longer includes the broad Criterion support
    module with dead-code allowances. Invalid finding: the repeated duplicate
    `src/neighbour_scoring.rs` path is absent on disk.
  - [x] (2026-06-25) Post-twenty-second-review deterministic gates pass:
    `shellcheck scripts/bench-neighbour-scoring.sh`, `make check-fmt`,
    `make lint`, and `make test` (1051 passed, 1 skipped). File-size audit also
    passes: touched Rust sources remain below the 400-line cap, with
    `chutoro-benches/src/neighbour_scoring/report.rs` currently the largest at
    398 lines.
  - [x] (2026-06-25) Twenty-third CodeRabbit review completed with one
    repeated invalid finding and no valid findings. Invalid finding: the
    repeated duplicate `chutoro-benches/src/neighbour_scoring.rs` root is absent
    on disk; `chutoro-benches/src/lib.rs` resolves `pub mod neighbour_scoring`
    to `chutoro-benches/src/neighbour_scoring/mod.rs`, and
    `median_matches_sorted_middle_values` already lives beside `median`,
    `lane_utilisation_basis_points`, and `duration_basis_points` there.
  - [x] (2026-06-25) Captured full Criterion baseline, optional HNSW build
    profile CSV, Hyperfine corroboration, and `perf stat` counters. Criterion
    baseline `before` was saved for all 21 cases; the lane-utilisation report
    was written to `target/benchmarks/neighbour_scoring_lane_utilisation.csv`;
    the build-profile report was written to
    `target/benchmarks/neighbour_scoring_build_profile.csv`; Hyperfine reported
    `30.776 s +/- 0.297 s`; and `perf stat -r 20` over the 128-dimensional
    realistic bucket group reported 31,819,791,060 cycles,
    138,569,605,046 instructions, IPC 4.35, and 0.17% cache misses.
  - [x] (2026-06-25) CodeRabbit review for the Milestone 0 evidence update
    completed with zero findings after `make markdownlint` passed.
- [ ] Milestone 1 (C1): write-lock-free-scoring invariant guard (same-thread
  marker).
- [ ] Milestone 2 (C2/C3): §6.3 implementation-update, developers-guide,
      ADR-003,
  roadmap mark — reflecting whatever the evidence supports.
- [ ] Milestone 3 (E1, conditional): core-side allocation hygiene.
- [ ] Milestone 4 (E2, conditional): query-centric port override + SoA
  packing-buffer reuse.
- [ ] Milestone 5 (E3, conditional): packing-step prefetch behind
      `simd_prefetch`.
- [ ] Milestone 6 (D5): final evidence, doc/ADR finalisation, roadmap done.

## Surprises & discoveries

- Observation: scoring already runs outside the write lock.
  Evidence: `cpu/trim.rs:65` (`score_trim_jobs`, no lock);
  `insert/planner.rs:78` (planner under read lock); no distance call inside
  either `write_graph` closure (`cpu/mod.rs:402-416`). Impact: the locking part
  of 2.3.1 is verification + documentation (C1), not re-architecture.
- Observation: the dense provider does **not** override `batch_distances`.
  Evidence: `dense/src/provider.rs:137,143` overrides only `distance` and
  `distance_batch`; core's default `batch_distances` (`datasource.rs:161-177`)
  builds a pairs `Vec`; `shared_query_candidates` (`dense/src/simd/mod.rs:195`)
  re-derives the query and re-collects candidates before `from_row_indices`
  (`point_view.rs:68`). Impact: "packed indices" is genuinely unrealised at the
  port; E2's prerequisite is a query-centric override that carries
  `(query, &[candidate])` intact.
- Observation: the distance cache is present on both insertion and search paths.
  Evidence: `Some(cache)` at `cpu/mod.rs:394` (planner), `456`/`466` (search);
  `validate_batch_distances` forwards only the *miss* subset
  (`validate.rs:123-127`; `helpers.rs:134-151`). Impact: batches reaching the
  adapter are warm-cache-narrow; SoA frequently disabled by
  `should_pack_query_points`. D1/D2/D3 benefit is unproven and must be measured.
- Observation: the reuse buffer must preserve alignment.
  Evidence: `PackedSoaStorage { blocks: Vec<AlignedBlock> }`,
  `AlignedBlock([f32; 16])` `repr(C, align(64))` (`point_view.rs:13-19`).
  Impact: scratch is `Vec<AlignedBlock>`, not `Vec<f32>`; corrects the original
  Risk 6 mitigation.
- Observation: `search.rs` is 369/400 lines.
  Evidence: `wc -l`. Impact: E1 must extract a sibling module before adding
  scratch.
- Observation: `docs/contents.md` and `docs/repository-layout.md`, referenced
  by the top-level `AGENTS.md`, do not exist in this worktree. Evidence: direct
  reads failed with `No such file or directory`; `rg --files docs` shows the
  live documentation set. Impact: orientation used the concrete docs named by
  this plan instead.
- Observation: the Milestone 0 lane-utilisation diagnostic is now generated at
  `target/benchmarks/neighbour_scoring_lane_utilisation.csv`. Evidence: after
  `cargo bench -p chutoro-benches --bench neighbour_scoring -- --list`,
  realistic buckets report 50.00% utilisation for 8 candidates, 75.00% for 24
  candidates, and 100.00% for 16, 32, and 48 candidates. Impact: the benchmark
  makes the small-batch padding cost visible before any optimization is
  attempted.
- Observation: exact distance-cache LRU-lock contention is not exposed through
  a public or benchmark-only snapshot API. Evidence: `DistanceCache` currently
  emits optional `metrics` counters and a lookup-latency histogram, but its LRU
  `Mutex` acquisition timing is internal and not surfaced to `chutoro-benches`.
  Impact: the Milestone 0 harness records cache-miss subset sizes and scoring
  time at the `DataSource` boundary without widening core APIs; direct
  lock-wait measurement still needs either a scoped metrics extension or an
  explicit core telemetry seam before go/no-go is final.
- Observation: nextest treats Criterion benchmark functions as exact tests, so
  long-running benchmark bodies can block the workspace `make test` gate even
  when no production logic changed. Evidence: `make test` timed out on the
  existing `extract_labels/n=100,min=5` benchmark case before Milestone 0 code
  could be committed; shortening Criterion warm-up and measurement time only
  when `--exact` is present made the exact probe pass in about 10 seconds.
  Impact: benchmark targets need explicit exact-probe settings if they contain
  large cases and are registered in the workspace test suite.
- Observation: `harness = false` Criterion targets do not expose ordinary
  `#[test]` functions from bench-only modules;
  `cargo test -p chutoro-benches --bench neighbour_scoring -- --list` only
  listed Criterion benchmark cases. Impact: property coverage for the
  diagnostic helper maths lives in the `chutoro-benches` library module
  `neighbour_scoring`, and the Criterion support code delegates to that tested
  implementation.
- Observation: a full `make test` rerun after the fifth CodeRabbit fixes hit a
  single non-reproducing `hnsw_mutations_preserve_invariants_proptest` failure
  in `chutoro-core`, then passed on both a targeted rerun and a subsequent full
  workspace rerun. Impact: the milestone records the transient as validation
  context, but no production or test expectation was changed because the
  deterministic reruns are clean.
- Observation: benchmark-only support modules should not own report schema
  formatting when the schema is part of the milestone evidence. Evidence:
  CodeRabbit's eighth review noted that ad hoc CSV formatting in
  `benches/neighbour_scoring/support.rs` was untested because Criterion bench
  modules use `harness = false`. Impact: lane-utilisation and build-profile CSV
  rendering now lives in `chutoro-benches::neighbour_scoring::report`, with
  exact header-and-row regression tests.
- Observation: CodeRabbit can report stale paths after a module move. Evidence:
  the ninth review flagged a critical duplicate-root problem for
  `chutoro-benches/src/neighbour_scoring.rs`, but `rg --files` shows only
  `chutoro-benches/src/neighbour_scoring/mod.rs` and
  `chutoro-benches/src/neighbour_scoring/report.rs`. Impact: no code change was
  needed for that finding; the single-root module layout remains intentional.
- Observation: even benchmark-only CSV report labels must be treated as real
  CSV string fields. Evidence: the tenth CodeRabbit review noted that raw
  `bucket_kind` output would break parsing for commas, quotes, or line breaks.
  Impact: the report helper now quotes only fields that need escaping and has a
  regression case for an embedded comma and quote.
- Observation: report rendering should share the diagnostic helper's overflow
  model, not repeat a narrower `usize` calculation. Evidence: the eleventh
  CodeRabbit review identified the only remaining `next_multiple_of` use in the
  report row writer. Impact: padded and wasted lane columns render from the
  same checked/u128 basis used by `lane_utilisation_basis_points`.
- Observation: report helpers should avoid avoidable work on the common path.
  Evidence: the twelfth CodeRabbit review noted that most benchmark bucket
  labels need no CSV escaping, but the previous helper still allocated a new
  `String`. Impact: `csv_escape` now returns `Cow<'_, str>`, borrowing
  unchanged labels while still allocating for quoted fields.
- Observation: optional report files should not be truncated before the
  expensive/fallible profiling work completes. Evidence: the twelfth CodeRabbit
  review identified that opening `neighbour_scoring_build_profile.csv` before
  the HNSW build loop could replace a previous valid report with an empty file
  if profiling failed. Impact: the benchmark now creates that report only after
  profiling rows are available.
- Observation: CSV report invariants are compact enough to property-test
  directly. Evidence: the thirteenth CodeRabbit review asked for generated
  coverage over bucket labels and candidate counts, and the resulting
  `lane_utilisation_report_round_trips_generated_rows` case checks CSV parsing,
  padding, wasted lanes, and utilisation bounds without exceeding the
  repository's file-size limit. Impact: the report schema now has example and
  generated coverage.
- Observation: Criterion benchmark configuration should use supported controls
  only. Evidence: the fourteenth CodeRabbit review noted that a custom
  benchmark CLI flag would be rejected before the benchmark body ran. Impact:
  the optional short-measurement mode is now controlled by
  `CHUTORO_BENCH_NEIGHBOUR_SHORT_MEASUREMENT`, leaving normal Criterion and
  nextest arguments untouched.
- Observation: build-profile reporting consumes stats once after each build.
  Evidence: the fourteenth CodeRabbit review identified that cloning
  `BuildProfileStats` duplicated the potentially large batch-size vector.
  Impact: the stats wrapper now takes the guarded value with `mem::take`,
  preserving poisoned-mutex handling while avoiding the clone.
- Observation: benchmark reports must follow Cargo's configured target
  directory. Evidence: the fifteenth CodeRabbit review noted that the harness
  always wrote under the workspace-relative `target/benchmarks` path. Impact:
  report path resolution now honours `CARGO_TARGET_DIR` when set and falls back
  to the existing workspace target directory otherwise.
- Observation: build-profile report ratios need generated coverage, not only
  fixed examples. Evidence: the sixteenth CodeRabbit review identified that
  `accumulated_batch_scoring_vs_wall_basis_points` was not property-tested even
  though duration ratio arithmetic is edge-case prone. Impact: generated
  zero-duration, tiny-duration, and large-duration cases now check that the CSV
  column matches `duration_basis_points`.
- Observation: benchmark command strings escaped with Bash `%q` should declare
  their shell to Hyperfine. Evidence: the sixteenth CodeRabbit review noted
  that Hyperfine otherwise delegates command parsing to `/bin/sh`, whose
  quoting rules need not match Bash's `%q` output. Impact: the wrapper now
  passes `--shell bash` with its escaped benchmark command.
- Observation: narrow test helpers should state their parser contract where
  they resemble general-purpose utilities. Evidence: the seventeenth CodeRabbit
  review noted that the generated-report CSV parser only needs the writers'
  newline-terminated output. Impact: the helper now says it relies on `\n`,
  treats bare `\r` as data, and may drop a trailing unterminated row.
- Observation: build-profile instrumentation must not serialize the measured
  scoring path on a single stats mutex. Evidence: the eighteenth CodeRabbit
  review noted that taking a mutex for every scalar and batch update skews the
  profile. Impact: counters and elapsed time now use relaxed atomics; only the
  exact batch-size vector remains mutex-protected.
- Observation: benchmark wrapper logs should survive successful script exit.
  Evidence: the eighteenth CodeRabbit review noted that writing Hyperfine
  output through `tee` to a `mktemp` path and then deleting that path made the
  log unavailable for inspection. Impact: the script now writes to a stable
  branch-specific `/tmp/hyperfine-neighbour-scoring-chutoro-*.out` path.
- Observation: property tests placed only in Criterion bench modules are not
  part of the normal unit-test harness when the bench is registered with
  `harness = false`. Evidence: after the nineteenth CodeRabbit request for
  generated coverage around profiling arithmetic, `cargo test -p
  chutoro-benches --bench neighbour_scoring profiling` built and ran the
  Criterion binary without executing those unit tests. Impact: the profiling
  arithmetic now lives in `chutoro-benches::neighbour_scoring`, where
  `cargo test -p chutoro-benches neighbour_scoring --lib` and `make test`
  execute its property tests.
- Observation: sequential atomic helper properties do not exercise
  `compare_exchange_weak` retry behaviour under contention. Evidence: the
  twentieth CodeRabbit review asked for concurrent `Arc`/`thread` tests around
  both profiling saturation helpers. Impact: the helper tests now hammer each
  atomic counter from multiple workers and assert the final accumulated value.
- Observation: executable coverage should target a narrow library seam rather
  than include broad Criterion support modules from an integration test.
  Evidence: the twenty-first and twenty-second CodeRabbit reviews asked for
  build-profile environment/report-path coverage, then identified that
  importing `benches/neighbour_scoring/support.rs` forced a broad
  `#[expect(dead_code)]` allowance. Impact: build-profile env/path helpers now
  live in `chutoro-benches::neighbour_scoring::build_profile`; the integration
  test covers those helpers directly, and mutable environment tests use the
  shared `chutoro-test-support::env::EnvVarGuard`.
- Observation: atomics plus a side-vector need an explicit snapshot boundary.
  Evidence: the twenty-first CodeRabbit review noted that updating atomics
  before pushing `batch_sizes` could split one batch across two snapshots.
  Impact: `record_batch` now performs all batch accounting while holding the
  batch-size lock.
- Observation: environment variable mutation in tests is shared infrastructure,
  not benchmark-specific behaviour. Evidence: the twenty-second CodeRabbit
  review noted that a local integration-test guard duplicated a reusable
  pattern and could drift from other environment tests. Impact:
  `chutoro-test-support::env::EnvVarGuard` serializes mutation process-wide,
  restores the previous value on drop, and centralizes the required unsafe
  environment calls behind documented safety comments.
- Observation: Milestone 0 shows the measured packed-index batch-scoring window
  is a small share of complete HNSW build time on the synthetic profile. Evidence:
  `target/benchmarks/neighbour_scoring_build_profile.csv` reports accumulated
  batch-scoring time at 1.678 seconds of a 20.208-second 10k-point build
  (8.30%) and 28.197 seconds of a 781.631-second 100k-point build (3.61%).
  Impact: any E1-E3 optimisation must still clear the pre-registered cycle
  threshold in realistic buckets; the profile does not justify speculative
  structural churn by itself.

## Decision log

- Decision: Re-scope 2.3.1 to committed C1-C4 (guard + docs + measurement) with
  E1-E3 conditional on pre-registered evidence; defer the structural levers
  (cross-node beam, secondary SoA copy, `batch_distances_into`) to a separate
  evidence-first roadmap item. Rationale: code reading shows the structural
  intent is satisfied by 2.2.x; the residual per-call deltas operate inside a
  window the cache fragments to near nothing; committing speculative complexity
  would close the item on motion. The Logisphere panel's verdict was REVISE on
  exactly this point. Date/Author: 2026-06-09, planning agent (post-review).
- Decision: Keep `DensePointView`/`PackedSoaStorage`/SoA/prefetch private to
  `chutoro-providers/dense`; the only boundary change permitted is *adding* a
  defaulted query-centric port method. Rationale: preserves the hexagonal
  boundary and future GPU/alternate layouts; matches the canonical pure-Rust
  HNSW design (instant-distance keeps coordinates in a separate immutable array
  behind the scoring call). Date/Author: 2026-06-09, planning agent.
- Decision: The keep/drop signal for any conditional delta is the
  cycle/instruction-count cross-check (primary), with `critcmp` wall-time and
  `hyperfine` as corroboration only; keep requires >5% median improvement AND
  non-overlapping CIs at the realistic (8-48) buckets; "within noise" drops.
  Rationale: per-call effects are sub-microsecond on a shared host; wall-time
  CIs overlap the effect; cycle counts are noise-immune. Date/Author:
  2026-06-09, planning agent.
- Decision: Forbid shared `Mutex`/`RefCell` SoA scratch; require thread-local or
  explicit per-call/per-job `Vec<AlignedBlock>` scratch. Rationale: a shared
  scratch serialises the Rayon-parallel trim and destroys the 2.2
  parallel-scoring win; alignment/padding must be preserved. Date/Author:
  2026-06-09, planning agent.
- Decision: The C1 guard asserts a *same-thread* "not holding a write guard"
  invariant via a marker set inside `write_graph`'s scope, not a global
  `try_read().is_ok()` probe. Rationale: a global probe spuriously fails when
  another Rayon/search thread legitimately holds the write lock; only the
  same-thread property is meaningful. Date/Author: 2026-06-09, planning agent
  (corrects the original draft).
- Decision: The first Milestone 0 artefact uses the public dense-provider Arrow
  ingestion path instead of adding a constructor solely for benchmarks.
  Rationale: this preserves the adapter boundary and avoids adding public API
  before the evidence says the dense adapter needs structural changes.
  Date/Author: 2026-06-24, implementation agent.
- Decision: Do not add a public `CpuHnsw`/`DistanceCache` telemetry API merely
  to satisfy the first measurement pass. Rationale: the plan's public-API
  stability constraint is stronger than the convenience of direct cache-lock
  snapshots; the remaining contention measurement should be handled by a scoped
  metrics extension or benchmark-only internal seam if the deterministic gates
  justify it. Date/Author: 2026-06-24, implementation agent.
- Decision: Keep the `neighbour_scoring` benchmark as a small harness plus a
  private support module instead of a single file. Rationale: the initial
  self-contained harness exceeded the repository's 400-line source-file limit;
  separating fixture/report/profile support preserves readability without
  changing the measured path. Date/Author: 2026-06-24, implementation agent.
- Decision: Put lane-utilisation, duration-ratio, and median diagnostic
  calculations in `chutoro-benches::neighbour_scoring`, not in the
  Criterion-only bench module. Rationale: Criterion targets use
  `harness = false`, so ordinary unit tests in bench-only modules are not part
  of the normal test list; the library module gives the helper maths real
  property-test coverage and keeps the benchmark support module focused on
  fixture/report orchestration. Date/Author: 2026-06-24, implementation agent.
- Decision: Keep benchmark profiling counter arithmetic in
  `chutoro-benches::neighbour_scoring` even though the profiling wrapper itself
  remains bench-only. Rationale: the arithmetic is small, deterministic, and
  easy to property-test in the normal library harness; the wrapper still owns
  the benchmark-only `DataSource` instrumentation and error plumbing.
  Date/Author: 2026-06-25, implementation agent.
- Decision: Keep `hyperfine` wrapper logs in a stable branch-specific file under
  a user-owned private `/tmp/chutoro-benches-${UID}` directory. Rationale: the
  wrapper's output must survive successful script exit for measurement review,
  while the private directory avoids cross-user collisions and symlink risk.
  Date/Author: 2026-06-25, implementation agent.
- Decision: The benchmark setup validates each fixed query/candidate scoring
  case before Criterion timing, while the timed loop still calls
  `batch_distances` so the benchmark measures scoring work rather than cached
  fixture output. Rationale: Criterion closures cannot return `Result`, and
  moving the scoring call entirely out of `b.iter` would invalidate the
  measurement. Date/Author: 2026-06-25, implementation agent.
- Decision: The build profile's accumulated scoring metric counts only batched
  `DataSource::batch_distances` elapsed time; scalar `distance` calls are
  counted but excluded from that duration. Rationale: the Milestone 0 profile
  is intended to measure the batch-scoring window relevant to packed indices,
  not all possible distance work. Date/Author: 2026-06-25, implementation agent.
- Decision: CSV evidence rendering belongs in the tested
  `chutoro-benches::neighbour_scoring` library module, not in the Criterion
  support module. Rationale: the report schemas are reviewable artefacts and
  need deterministic unit-test coverage, while `harness = false` bench modules
  do not expose ordinary tests. Date/Author: 2026-06-25, implementation agent.

## Outcomes & retrospective

To be completed at milestones and at the end. Must state, with cited numbers:
the measured candidate-set-size and cache-miss-subset distributions;
accumulated batch-scoring time compared with build wall-clock; distance-cache
lock contention; and, for each of E1/E2/E3, the go/no-go decision with its
cycle-count evidence. A "committed scope only (C1-C4), E1-E3 dropped on
evidence" result is an explicitly successful outcome and must be recorded as
such, with the deferred structural levers carried into the proposed follow-up
item.

Milestone 0 outcome (2026-06-25): the benchmark harness exists, is registered,
and has baseline evidence. `cargo bench -p chutoro-benches --bench
neighbour_scoring -- --list` reports 21 benchmark cases: realistic candidate
counts 8, 16, 24, 32, and 48 plus diagnostic counts 256 and 1024 for dimensions
32, 128, and 768. The lane-utilisation report shows the pre-registered
realistic buckets waste 8 of 16 lanes at candidate count 8, 8 of 32 lanes at
candidate count 24, and no lanes at 16, 32, or 48.

The full Criterion baseline was saved as `before`. For the central
128-dimensional realistic bucket group, Criterion reported medians of
906.45 ns (8 candidates), 1.2741 us (16), 1.9291 us (24), 2.2409 us (32), and
3.2663 us (48). Diagnostic 128-dimensional medians were 80.638 us at 256
candidates and 910.20 us at 1024 candidates. Hyperfine corroboration over the
full benchmark binary reported `30.776 s +/- 0.297 s` over ten runs.

The optional HNSW build profile reported:

- 10,000 points, dimension 128: build 20.208 seconds, accumulated batch scoring
  1.678 seconds (8.30%), 612,507 batch calls, 14,939,510 total batch candidates,
  min/max/median batch 1/33/27.
- 100,000 points, dimension 128: build 781.631 seconds, accumulated batch
  scoring 28.197 seconds (3.61%), 6,969,245 batch calls, 176,427,818 total
  batch candidates, min/max/median batch 1/33/29.

`perf stat -r 20` over the 128-dimensional realistic bucket group reported
31,819,791,060 cycles, 138,569,605,046 instructions, IPC 4.35,
927,935,251 cache references, 1,605,048 cache misses, a 0.17% cache-miss rate,
and 7.3746 +/- 0.0898 seconds elapsed. Exact distance-cache LRU lock-wait
contention remains unavailable without widening core telemetry, so Milestone 0
records batch miss-subset sizes and `DataSource` scoring time at the adapter
boundary instead. No E1/E2/E3 go/no-go decision has been made yet because no
candidate delta has been measured against this baseline.

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
   parallel Rayon scoring of the cache-*miss* subset, deterministic bounded
   heap.
5. **Commit (write lock #2)** (lines 413-416): reconcile reciprocity, heal
   connectivity. *No distance computation.*

Search (`cpu/mod.rs:444`) uses the same `LayerSearcher` (`hnsw/search.rs`)
under a read guard, also with the cache present (lines 456,466).

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
  parallel trim; Milestone 0 measures its contention so go/no-go attributes
  cost honestly.

### The packed-index round-trip (the unrealised seam)

`core &[usize]` → default `batch_distances` builds `Vec<(query, candidate)>`
(`datasource.rs:169-173`) → `distance_batch` →
`euclidean_distance_batch_raw_pairs` → `shared_query_candidates` scans pairs to
re-derive the shared query and re-collect candidates
(`dense/src/simd/mod.rs:195-213`) → `from_row_indices` allocates
`PackedSoaStorage` (`point_view.rs:68`). The "packed" ids are unpacked to pairs
then repacked. E2's prerequisite eliminates this by adding a defaulted
query-centric port method the dense adapter overrides.

### Key terms

- **SoA (structure-of-arrays)**: dimension-major coordinate storage so a SIMD
  kernel streams one dimension across many points.
- **Packed indices**: candidate ids in a single contiguous slice carried intact
  to the scoring kernel (the thing currently lost to the pairs round-trip).
- **Cache-miss subset**: the candidates for which no cached distance exists; the
  only ids that reach `batch_distances` on a warm path.
- **Port / adapter**: `DataSource` is the port; `DenseMatrixProvider` an
  adapter. Domain depends on the port, never on adapter internals.

### Documentation and skills to consult

- Design: `docs/chutoro-design.md` §6.3. Roadmap: `docs/roadmap.md` 2.3.1,
  2.4.1; determinism 1.1.2.
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
   `chutoro-benches/Cargo.toml` as `[[bench]] name = "neighbour_scoring"`).
   This is also the canonical 2.4.1 artefact — cross-reference it in
   `docs/roadmap.md` 2.4.1 so that item consumes rather than rebuilds it.
   Bucket by candidate-set sizes that *occur*: 8, 16, 24, 32, 48 (straddling
   `2M` for M∈{8,16,24}); add diagnostic-only buckets {256, 1024} labelled
   "does not occur in the HNSW hot path". Dimensions {32, 128, 768}. Use
   `Throughput::Elements`, `black_box`, a fixed seed. Record per-bucket
   effective SoA lane utilisation *including* 16-lane padding waste, and
   SoA-vs-scalar crossover.
2. Instrument a real synthetic HNSW build (N ∈ {10k, 100k}) to capture, as data
   written under `target/benchmarks/`:
   - the per-call candidate-set-size histogram *after* cache-miss subsetting
     (cold vs warm graph);
   - total `batch_distances` call count and aggregate candidates scored per
     build;
   - accumulated batch-scoring time compared with build wall-clock;
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
cycle-count improvement with non-overlapping CIs at those buckets. Otherwise
the delta is dropped and the null result recorded.

Validation: `make bench` builds; the harness runs; all distributions and the
contention numbers are captured and written into `Surprises & discoveries` and
`Outcomes`. Go/no-go for E1/E2/E3 decided and recorded here.

### Milestone 1 — Write-lock-free-scoring invariant guard (C1, committed)

Goal: turn "no scoring under the write lock" into a tested, durable invariant.

1. Red/Green: add a same-thread marker to `CpuHnsw` — e.g. a thread-local or an
   `AtomicBool`-per-thread flag set true for the duration of each `write_graph`
   closure (`cpu/internal.rs:45-51`) and false after. Add an instrumented
   `DataSource` decorator (test-only) that, on each `distance`/
   `batch_distances` call, asserts the marker is false (the calling thread is
   not inside a `write_graph` scope). Build a small index single-threaded (no
   concurrent search) through the decorator and assert no violation across
   plan, trim, and commit.
2. Prove teeth: temporarily move a scoring call inside a `write_graph` closure,
   confirm the guard fails, then revert.

Validation: guard passes on real code, fails on the deliberate mutation;
`make test` green. (This milestone is independent of Milestone 0's outcome.)

### Milestone 2 — Documentation, ADR, roadmap reflecting the evidence (C2/C3, committed)

Goal: record what is true regardless of whether E1-E3 are built.

1. `docs/chutoro-design.md` §6.3 implementation-update for 2.3.1: how 2.2.x
   satisfies the packed-index/SoA/scoring-outside-lock *structure*; the
   cache-fragmentation reality; the scoring-outside-write-lock invariant and
   its guard; the measured candidate-size/scoring-share/contention numbers; and
   a null-result template if E1-E3 are dropped.
2. `docs/adr-003-soa-prefetch-adapter-boundary.md` (Y-Statement;
   `arch-decision-records` skill): the decision to keep SoA/prefetch private to
   the dense adapter; the cross-node-beam and secondary-SoA-copy and
   `batch_distances_into` alternatives with memory-vs-repack and
   policy-coupling trade-offs; a recommendation to spin measured-win work into
   separate evidence-first roadmap items. Reference it from §6.3.
3. **Create the deferred roadmap items.** Add the three deferred structural
   levers (cross-node beam scoring, secondary dimension-major SoA copy,
   `batch_distances_into` out-buffer reuse) as explicit, unchecked items under
   Phase 2.3 in `docs/roadmap.md` (suggested 2.3.3-2.3.5; reconcile numbering
   with the live roadmap), each referencing ADR-003 and this execplan and each
   gated on its own evidence. Record the assigned numbers in the
   `Decision Log`. This step is mandatory and independent of the Milestone 0
   go/no-go outcome.
4. `docs/developers-guide.md`: the measurement harness and `hyperfine` workflow,
   the write-lock invariant test, and (if E1-E3 land) the scratch and
   `simd_prefetch` conventions. Document the `DataSource` length-equality and
   error-atomicity (output-unmodified-on-error) invariants the cache layer
   relies on (`validate.rs:129-137`, `helpers.rs:152-160`) directly in the
   trait docs.
5. Mark roadmap item 2.3.1 done once committed scope lands (the item's intent is
   satisfied; the deferred levers are now their own tracked roadmap items per
   step 3).

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
   oracle across randomised graphs and `ef` values, not three fixtures)
   asserting the scratch-reusing path yields identical `Vec<Neighbour>`; assert
   scratch length is 0 at the top of each iteration; assert the Milestone 0
   allocation count drops.
3. Green: reuse a per-search `Vec<usize>` for the per-node `fresh` list and a
   `Vec<f32>` for distances, cleared each iteration. Retarget the actual cached
   path allocations: `missing` (`validate.rs:123-127`), `miss_candidates`/
   `miss_meta` (`helpers.rs:134-135`), and the `fresh` Vec — plus the
   `metric_descriptor()` `Arc` clone per batch (`helpers.rs:105,132`;
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
   (NOT a shared `Mutex`/`RefCell`); it is `Vec<AlignedBlock>` preserving
   64-byte alignment and the 16-lane zero-padded tail.
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
a prototype clears the threshold. Given the dimension-major layout, prefetch
(if anywhere) targets the scattered `matrix.row(index)` gather in the *pack*
step (`point_view.rs:79-84`), not the kernel scan.

Files: `chutoro-providers/dense/Cargo.toml` (non-default `simd_prefetch`
feature), a `prefetch` helper module, `point_view.rs` (pack loop).

1. Add a `prefetch_t0` helper:
   `#[cfg(all(target_arch = "x86_64", feature = "simd_prefetch"))]` calls
   `core::arch::x86_64::_mm_prefetch::<_MM_HINT_T0>`; every other
   target/feature is an inlined no-op (prefetch is semantically a no-op, so
   omitting it is always correct).
2. In the pack loop, prefetch the next candidate's source row a tunable few rows
   ahead; keep hot-loop temporaries in scalar locals (avoids the optimizer
   regression the research observed).
3. A/B with `critcmp` and cycle counts across the realistic buckets.

Go/no-go: keep only on a statistically significant cycle-count win with no
bucket regressing; otherwise **revert the entire milestone** (feature flag,
helper, ADR prefetch clause) rather than leaving a dead flag and an ADR about a
no-op. If kept, document `simd_prefetch` as host-tuned, off by default,
requiring re-benchmarking per target, and record the exact CPU model and
`target-cpu` in Outcomes.

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
   only on a ≥5% median cycle-count win with non-overlapping CIs at the
   realistic buckets, on a pinned, low-load host; otherwise dropped with the
   null result recorded. A committed-scope-only outcome is acceptance, not
   failure.
4. **Quality gates**: `make check-fmt`, `make lint`, `make test`, `make kani`,
   `make markdownlint`, `make nixie` pass.

Red-Green-Refactor evidence per implemented milestone: the red command +
expected failure, the green command + pass, the refactor command + pass.

Quality criteria ("done"): committed scope merged; any conditional delta backed
by cycle-count evidence; no bucket regresses on the default feature set; no new
dependency without a flagged Decision Log entry; no `unsafe` without a SAFETY
comment plus a Miri/Kani check.

Quality method: local gates with `tee`; `coderabbit review --agent` after each
major milestone with all concerns cleared, run only after the deterministic
gates are green.

## Idempotence and recovery

All steps re-runnable. Criterion baselines are named. The `simd_prefetch`
feature is additive and reverted wholesale if dropped. Commit per milestone for
clean `git revert`. No destructive step.

## Artifacts and notes

Record, as concise indented transcripts: the Milestone 0 candidate-size and
miss-subset histograms, the scoring-share-of-build-time number, the cache
contention figures, the `critcmp` and `perf stat` tables, and the Kani success
line. If E1-E3 are dropped, record the null result and the deferred-item
proposal.

## Interfaces and dependencies

No new runtime dependencies. `perf`/`cachegrind` are dev tools; an `iai`
dev-dep, if used, is flagged in the Decision Log. Interfaces at completion:

- A test-only same-thread "inside write_graph" marker on `CpuHnsw` and an
  instrumented `DataSource` decorator (C1).
- `chutoro-benches` gains `[[bench]] name = "neighbour_scoring"` (C4 / 2.4.1).
- If E2 lands: a *defaulted*, query-centric override on the dense adapter
  bypassing the pairs round-trip, and
  `from_row_indices(..., scratch: &mut PackedSoaStorage)` (internal). No SoA
  type crosses the crate boundary.
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

- No change to `DataSource::distance`/`distance_batch`, `CpuHnsw` public
  methods, or the CLI. `batch_distances_into` is **deferred** (ADR-003).

## Signposted documentation and skills

- Process: `execplans`. Architecture: `hexagonal-architecture`,
  `arch-decision-records`, `arch-crate-design`; `docs/chutoro-design.md` §6.3.
- Rust: `rust-router` → `rust-performance-and-layout`, `rust-memory-and-state`,
  `rust-types-and-apis`, `rust-unsafe-and-ffi` (only if prefetch needs
  `unsafe`).
- Testing/verification: `rust-unit-testing`, `proptest`, `kani`, `nextest`,
  `rust-verification`; `docs/property-testing-design.md`,
  `docs/rust-testing-with-rstest-fixtures.md`, `docs/rust-doctest-dry-guide.md`,
  `docs/reliable-testing-in-rust-via-dependency-injection.md`,
  `docs/complexity-antipatterns-and-refactoring-strategies.md`.

## Revision note

Revision 3 (2026-06-09): made it explicit that the three deferred structural
levers must be **created as roadmap items in `docs/roadmap.md` during
implementation** (a mandatory Milestone 2 step, independent of the Milestone 0
go/no-go), with suggested numbers 2.3.3-2.3.5 to be reconciled with the live
roadmap and recorded in the Decision Log.

Revision 2 (2026-06-09): folded the Logisphere community-of-experts review
(verdict REVISE) — all 12 required revisions. Major changes: split into
committed scope (C1 guard, C2/C3 docs+ADR, C4 measurement) and evidence-gated
conditional scope (E1 allocation hygiene, E2 port override +
`Vec<AlignedBlock>` scratch reuse, E3 packing-step prefetch); named the
pack→unpack→repack round-trip and made a query-centric override E2's
prerequisite; added the cache-fragmentation reality and made Milestone 0
measure miss-subset distributions, scoring share of build time, and cache-lock
contention; re-centred benchmark buckets on the realistic 8-48 regime with
cycle-count as the primary keep/drop signal; corrected the C1 guard to a
same-thread marker; corrected the scratch type to `Vec<AlignedBlock>` and
forbade shared scratch; pre-committed `search.rs` module extraction; deferred
the structural levers (cross-node beam, secondary SoA copy,
`batch_distances_into`) to a separate evidence-first item recorded in ADR-003.
Revision 1 (2026-06-09): initial draft. Pending: user approval before
implementation.
