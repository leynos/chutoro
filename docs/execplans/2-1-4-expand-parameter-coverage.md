# Execution Plan: expand HNSW benchmark `ef_construction` parameter coverage

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds.

Status: COMPLETE

PLANS.md is not present in this repository, so no additional plan constraints
apply.

## Purpose / big picture

Roadmap item 2.1.4 extends the CPU HNSW benchmark harness so that
`ef_construction` is varied independently of `M`, revealing build-time versus
recall trade-offs. Today every benchmark configuration uses
`ef_construction = M * 2`, which is the cheapest viable setting. After this
change a user running `cargo bench -p chutoro-benches --bench hnsw` will see
additional Criterion timing results for
`ef_construction in {M*2, 100, 200, 400}` and a machine-readable CSV report at
`target/benchmarks/hnsw_recall_vs_ef.csv` showing mean recall@10 versus build
time for each `(M, ef_construction)` pair. Parameter choices and their
performance/quality implications will be documented in `docs/chutoro-design.md`
§11.3.

Success is observable when:

- A new Criterion group `hnsw_build_ef_sweep` benchmarks build time across
  2 dataset sizes × 2 M values × 4 ef_construction values = 16 cases.
- A recall measurement pass writes
  `target/benchmarks/hnsw_recall_vs_ef.csv` with 8 rows (2 M × 4 ef).
- New unit tests in `chutoro-benches` cover recall helpers and ef_construction
  parameter construction using parameterized `rstest` cases with happy,
  unhappy, and edge-case coverage.
- `docs/chutoro-design.md` §11.3 documents parameter rationale and trade-offs.
- `docs/roadmap.md` marks item 2.1.4 as done.
- `make check-fmt`, `make lint`, and `make test` all succeed.

## Constraints

- Files must not exceed 400 lines. Split modules where needed.
- Preserve existing `chutoro-core` public API; this is benchmark-only work.
- Do not weaken lint policy; any lint exceptions must be tightly scoped with
  `#[expect(..., reason = "...")]`.
- Keep benchmarks deterministic by reusing the existing explicit seed (42).
- Use en-GB-oxendict spelling in all comments and documentation.
- Avoid adding new crate dependencies; all required APIs (`CpuHnsw::search`,
  `Neighbour`, `DataSource`, `HnswParams`) are already public from
  `chutoro-core`, and `rand` is already a dependency of `chutoro-benches`.
- New behaviour must include unit tests using `rstest` parameterization where
  repetition would otherwise occur.
- The existing three benchmark groups (`hnsw_build`, `hnsw_build_with_edges`,
  `hnsw_build_diverse_sources`) must remain structurally unchanged.
- `HnswParams::new(m, ef)` requires `ef >= m`; all ef_construction values in
  the sweep satisfy this (minimum ef is M\*2, minimum M is 8, so minimum ef is
  16; all explicit values are >= 100).

## Tolerances (exception triggers)

- Scope: if implementation needs changes in more than 10 files or more than
  600 net lines, stop and escalate.
- Interface: if satisfying this item requires a breaking public API change in
  `chutoro-core`, stop and escalate.
- Dependencies: if any new crate is required, stop and escalate.
- Iterations: if `make lint` or `make test` fails after 3 repair attempts,
  stop and escalate with captured logs.
- Ambiguity: if recall measurement semantics cannot be defined without
  conflicting interpretations, stop and present options with trade-offs.
- File size: if `chutoro-benches/benches/hnsw.rs` exceeds 400 lines after
  edits, split the ef_sweep benchmark into a separate bench binary before
  proceeding.

## Risks

- Risk: combinatorial explosion of benchmark cases makes full runs too slow
  for practical use. Severity: medium. Likelihood: medium. Mitigation: the new
  ef_sweep group uses a reduced parameter matrix (2 n × 2 M × 4 ef = 16 cases)
  instead of the full (4 n × 4 M × 4 ef = 64 cases). Existing groups are
  unchanged.

- Risk: recall measurement adds non-trivial wall-clock time during benchmark
  registration, causing nextest timeouts (as seen during 2.1.3 memory profiling
  work). Severity: high. Likelihood: medium. Mitigation: gate recall
  measurement behind `should_collect_recall_report()` using the same
  `--list`/`--exact` guard pattern established in 2.1.3. Add env var
  `CHUTORO_BENCH_HNSW_RECALL_REPORT` for explicit enable/disable.

- Risk: brute-force oracle at n=1000 with Q=50 queries is O(50,000) distance
  calls per (M, ef) combination. At 8 combinations this is 400k distance calls
  total. Severity: low. Likelihood: low. Mitigation: n=1000 with 16-dimensional
  vectors is trivially fast (~milliseconds). No further optimisation needed.

- Risk: `float_arithmetic` lint deny prevents naive recall computation.
  Severity: medium. Likelihood: certain. Mitigation: use integer-only
  `RecallScore { hits: usize, total: usize }` struct internally; perform the
  float division only in CSV formatting with a tightly scoped
  `#[expect(clippy::float_arithmetic, ...)]`.

- Risk: benchmark-only code may regress under strict Clippy lints.
  Severity: medium. Likelihood: medium. Mitigation: keep helpers small,
  fallible, and covered by focused unit tests. Follow patterns already
  established in `chutoro-benches/src/profiling/mod.rs`.

## Progress

- [x] (2026-02-21 14:00Z) Draft ExecPlan for roadmap item 2.1.4.
- [x] (2026-02-21) Create `chutoro-benches/src/ef_sweep.rs` with constants and
      helpers.
- [x] (2026-02-21) Create `chutoro-benches/src/recall.rs` with oracle, scoring,
      and report.
- [x] (2026-02-21) Update `chutoro-benches/src/lib.rs` to export new modules.
- [x] (2026-02-21) Update `chutoro-benches/src/error.rs` with `DataSource`
      variant.
- [x] (2026-02-21) Split ef_sweep into separate bench binary
      (`benches/hnsw_ef_sweep.rs`) — `hnsw.rs` would have exceeded 400 lines.
- [x] (2026-02-21) Update `chutoro-benches/benches/hnsw.rs` to delegate
      `make_hnsw_params` to `make_hnsw_params_with_ef`.
- [x] (2026-02-21) Add §11.3 to `docs/chutoro-design.md`.
- [x] (2026-02-21) Mark 2.1.4 done in `docs/roadmap.md`.
- [x] (2026-02-21) Run formatting and documentation checks.
- [x] (2026-02-21) Run quality gates: `make check-fmt`, `make lint`,
      `make test` — all pass (717/717 tests).

## Surprises & discoveries

- Adding the ef_sweep group inline to `hnsw.rs` pushed the file to ~520 lines,
  well past the 400-line limit. The tolerance trigger fired and the benchmark
  was split into a separate binary (`benches/hnsw_ef_sweep.rs`) registered in
  `Cargo.toml`. This was smoother than expected — the benchmark is fully
  self-contained and has no shared state with the other groups.

- The strict lint policy required several scoped `#[expect]` annotations:
  `float_arithmetic` and `cast_precision_loss` for the recall fraction CSV
  column, `integer_division` for the evenly-spaced query index computation,
  and `excessive_nesting` for the triple-parameter Criterion loop. All were
  tightly scoped with reason strings.

- The `DataSource` trait method `name()` returns `&str` in the trait
  definition, but Clippy's `unnecessary_literal_bound` lint required the test
  `MatrixSource` implementation to return `&'static str` explicitly.

## Decision log

- Decision: use ef_construction values `{M*2, 100, 200, 400}` rather than
  only `{100, 200, 400}`. Rationale: M\*2 is the existing baseline and provides
  a meaningful low-end comparison point. The roadmap text suggests
  `{100, 200, 400}` as examples ("e.g."), not a fixed requirement. Including
  M\*2 preserves continuity with existing benchmark results while adding the
  three higher values. Date/Author: 2026-02-21 (DevBoxer)

- Decision: add a dedicated `hnsw_build_ef_sweep` Criterion group with a
  reduced parameter matrix (2 n × 2 M × 4 ef) instead of expanding the existing
  `hnsw_build` group to a full 3D matrix. Rationale: keeps existing benchmark
  baselines stable and comparable. A full matrix of 4 n × 4 M × 4 ef = 64 cases
  per group would double total benchmark time. The reduced matrix (n ∈ {500,
  5000}, M ∈ {8, 24}) captures the extremes of both dataset size and graph
  connectivity. Date/Author: 2026-02-21 (DevBoxer)

- Decision: implement recall measurement as a one-shot pass (like the memory
  profiler) rather than as a Criterion timed benchmark. Rationale: recall is a
  single deterministic quality number, not a timing-variance measurement.
  Criterion's statistical sampling would add overhead without improving the
  signal. The pattern is proven by the existing `profile_hnsw_memory_impl`
  function. Date/Author: 2026-02-21 (DevBoxer)

- Decision: reimplement `brute_force_top_k` and `recall_at_k` in
  `chutoro-benches` rather than exporting them from `chutoro-core`. Rationale:
  the existing implementations in
  `chutoro-core/src/hnsw/tests/property/search_property.rs` are `pub(super)`
  and `fn` (not `pub`), making them inaccessible outside the test module.
  Promoting them to public API would expand `chutoro-core`'s surface area for a
  benchmark-only need, violating the "no public API changes" constraint. The
  reimplementation is ~30 lines and follows the same algorithm. Date/Author:
  2026-02-21 (DevBoxer)

- Decision: use `RecallScore { hits, total }` integer struct to avoid
  `float_arithmetic` lint, with float conversion only in display/report code.
  Rationale: the crate denies `float_arithmetic` at the crate level. All
  existing metric helpers in `profiling/mod.rs` use integer arithmetic with
  scoped `#[expect]` for integer division. The same pattern applies here.
  Date/Author: 2026-02-21 (DevBoxer)

## Outcomes & retrospective

All acceptance criteria met:

- **Benchmark behaviour**: `hnsw_build_ef_sweep` group runs 16 cases across
  `(n, M, ef) = {500, 5000} × {8, 24} × {M*2, 100, 200, 400}`.
- **Recall reporting**: recall measurement writes 8-row CSV to
  `target/benchmarks/hnsw_recall_vs_ef.csv`.
- **Tests**: 717/717 tests pass including new parameterised `rstest` cases in
  `ef_sweep.rs` (8 cases) and `recall.rs` (13 cases).
- **Documentation**: §11.3 added to `docs/chutoro-design.md`.
- **Roadmap**: 2.1.4 marked `[x]` in `docs/roadmap.md`.
- **Quality gates**: `make check-fmt`, `make lint`, `make test` all pass.

Files changed (8 files, well within the 10-file tolerance):

1. `chutoro-benches/src/ef_sweep.rs` (new, ~123 lines)
2. `chutoro-benches/src/recall.rs` (new, ~352 lines)
3. `chutoro-benches/src/error.rs` (modified, +3 lines)
4. `chutoro-benches/src/lib.rs` (modified, +2 lines)
5. `chutoro-benches/benches/hnsw.rs` (modified, minor refactor)
6. `chutoro-benches/benches/hnsw_ef_sweep.rs` (new, ~215 lines)
7. `chutoro-benches/Cargo.toml` (modified, +4 lines)
8. `docs/chutoro-design.md` (modified, +47 lines)
9. `docs/roadmap.md` (modified, 1-character change)

The decision to split the ef_sweep into a separate bench binary proved
beneficial — it kept `hnsw.rs` clean and stable while the new benchmark
is independently runnable via `cargo bench --bench hnsw_ef_sweep`.

## Context and orientation

The HNSW (Hierarchical Navigable Small World) benchmarks live in the
`chutoro-benches` crate, a benchmark-support library separate from the core
`chutoro-core` library. The benchmark infrastructure was established in roadmap
items 2.1.1–2.1.3.

Key files and their roles:

- `chutoro-benches/benches/hnsw.rs` (361 lines): the Criterion benchmark
  binary. Contains three benchmark groups (`hnsw_build`,
  `hnsw_build_with_edges`, `hnsw_build_diverse_sources`), a memory profiling
  pass, and helper functions. Currently sweeps `M in {8, 12, 16, 24}` with
  `ef_construction = M * 2` hardcoded via `make_hnsw_params(m)`.

- `chutoro-benches/src/lib.rs` (11 lines): crate root exporting `error`,
  `params`, `profiling`, and `source` modules.

- `chutoro-benches/src/params.rs` (87 lines): `HnswBenchParams` display struct
  that formats as `n=1000,M=16,ef=32` for Criterion benchmark IDs.

- `chutoro-benches/src/profiling/mod.rs` (388 lines): memory profiling
  infrastructure including `HnswMemoryRecord`, `HnswMemoryInput`,
  `EdgeScalingBounds`, CSV report writer, and edge-scaling validation. Near the
  400-line limit — must not be expanded.

- `chutoro-benches/src/error.rs` (36 lines): `BenchSetupError` enum wrapping
  errors from synthetic sources, HNSW, MST, hierarchy, and profiling.

- `chutoro-core/src/hnsw/params.rs` (148 lines): `HnswParams` struct with
  builder pattern. `HnswParams::new(m, ef)` validates `ef >= m` and `m > 0`.

- `chutoro-core/src/hnsw/cpu/mod.rs`: `CpuHnsw` with `build`, `search`,
  `build_with_edges` public methods. `search` takes
  `(source, query_id, ef: NonZeroUsize)` and returns `Vec<Neighbour>`.

- `chutoro-core/src/hnsw/types.rs`:
  `Neighbour { id: usize, distance: f32 }` — the public type used for search
  results.

- `chutoro-core/src/hnsw/tests/property/search_property.rs` (383 lines):
  contains `brute_force_top_k` and `recall_at_k` as private (`fn`) helpers
  within the property test module. These cannot be imported but serve as
  reference implementations.

- `docs/chutoro-design.md`: design document with §11 "Concluding
  Recommendations", §11.1 (benchmark synthetic sources, roadmap 2.1.2), and
  §11.2 (memory footprint tracking, roadmap 2.1.3). New §11.3 will document
  ef_construction parameter coverage.

- `docs/roadmap.md` line 253: the unchecked `[ ] 2.1.4` entry.

Terminology:

- **M** (`max_connections`): the maximum number of bidirectional edges per node
  per HNSW layer. Controls graph density. Higher M → more memory, better
  recall, slower build.
- **ef_construction**: the search beam width used during index construction.
  More candidates evaluated per insertion → slower build, better graph quality
  and recall. Must be >= M.
- **recall@k**: the fraction of true k-nearest neighbours that HNSW search
  returns, compared against a brute-force oracle. Ranges from 0.0 to 1.0.

## Plan of work

### Stage A: new library modules (no benchmark changes)

Create two new modules in `chutoro-benches/src/`:

**`ef_sweep.rs`** (~60 lines): ef_construction sweep constants and parameter
construction helper.

- `EF_SWEEP_POINT_COUNTS`, `EF_SWEEP_MAX_CONNECTIONS`, and
  `EF_CONSTRUCTION_VALUES` constants.
- `resolve_ef_construction` sentinel resolver.
- `make_hnsw_params_with_ef` parameter builder.
- Unit tests via rstest.

**`recall.rs`** (~120 lines): brute-force oracle, recall scoring, and CSV
report writer for benchmark recall reporting.

- `brute_force_top_k` O(n) oracle.
- `RecallScore` integer-only recall representation.
- `recall_at_k` set-intersection scorer.
- `RecallMeasurement` report row and `write_recall_report` CSV writer.
- Unit tests via rstest.

Update `lib.rs` to export the new modules and `error.rs` to add a `DataSource`
variant for `brute_force_top_k` error propagation.

Go/no-go: `make lint` and `make test` must pass before proceeding.

### Stage B: benchmark integration

Modify `chutoro-benches/benches/hnsw.rs` to add the `hnsw_build_ef_sweep`
Criterion group and the `measure_recall_vs_ef` one-shot quality measurement,
gated behind the same `--list`/`--exact` guard pattern used by the memory
profiler. Register the new group in `criterion_group!`.

Go/no-go: `make check-fmt`, `make lint`, and `make test` must pass.

### Stage C: documentation and roadmap

Add §11.3 to `docs/chutoro-design.md` documenting parameter choices, benchmark
structure, recall methodology, and performance/quality trade-off guidance. Mark
2.1.4 done in `docs/roadmap.md`.

### Stage D: quality gates and verification

Run all quality gates and capture logs with `tee` and `set -o pipefail`.

## Concrete steps

1. Write ExecPlan to `docs/execplans/2-1-4-expand-parameter-coverage.md`.

2. Create `chutoro-benches/src/ef_sweep.rs`.

3. Create `chutoro-benches/src/recall.rs`.

4. Edit `chutoro-benches/src/error.rs` — add `DataSource` variant.

5. Edit `chutoro-benches/src/lib.rs` — add `pub mod ef_sweep;` and
   `pub mod recall;`.

6. Edit `chutoro-benches/benches/hnsw.rs` — add ef_sweep group, recall
   measurement, and register in `criterion_group!`.

7. Edit `docs/chutoro-design.md` — insert §11.3.

8. Edit `docs/roadmap.md` — mark 2.1.4 done.

9. Run formatting fixes and quality gates.

## Validation and acceptance

Done means all of the following are true:

- Benchmark behaviour:
  `cargo bench -p chutoro-benches --bench hnsw` includes the
  `hnsw_build_ef_sweep` group with 16 cases spanning (n, M, ef) = {500, 5000} ×
  {8, 24} × {M\*2, 100, 200, 400}.

- Recall reporting:
  After a benchmark run with
  `CHUTORO_BENCH_HNSW_RECALL_REPORT=1 cargo bench -p chutoro-benches`,
  `target/benchmarks/hnsw_recall_vs_ef.csv` exists with a header and 8 rows.

- Tests:
  `make test` passes including new parameterized rstest cases in `ef_sweep.rs`
  and `recall.rs` covering happy, unhappy, and edge-case paths.

- Documentation:
  `docs/chutoro-design.md` §11.3 documents parameter choices, benchmark
  structure, recall methodology, and performance/quality trade-off guidance.

- Roadmap:
  `docs/roadmap.md` item `2.1.4` is `[x]`.

- Quality gates:
  `make check-fmt`, `make lint`, and `make test` all succeed.

## Idempotence and recovery

- All file writes use atomic overwrite patterns. Re-running benchmark or
  report generation overwrites previous artifacts cleanly.
- If a benchmark run fails midway, re-running requires no manual cleanup.
- The ef_sweep group is additive; removing it only requires reverting changes
  to `hnsw.rs` and the `criterion_group!` macro.

## Interfaces and dependencies

New public interfaces in `chutoro-benches`:

In `chutoro-benches/src/ef_sweep.rs`:

    pub const EF_SWEEP_POINT_COUNTS: &[usize]
    pub const EF_SWEEP_MAX_CONNECTIONS: &[usize]
    pub const EF_CONSTRUCTION_VALUES: &[usize]
    pub fn resolve_ef_construction(m: usize, ef_raw: usize) -> usize
    pub fn make_hnsw_params_with_ef(
        m: usize, ef_construction: usize, seed: u64,
    ) -> Result<HnswParams, BenchSetupError>

In `chutoro-benches/src/recall.rs`:

    pub struct RecallScore { pub hits: usize, pub total: usize }
    pub fn brute_force_top_k<D: DataSource + Sync>(
        source: &D, query: usize, k: usize,
    ) -> Result<Vec<Neighbour>, DataSourceError>
    pub fn recall_at_k(
        oracle: &[Neighbour], observed: &[Neighbour], k: usize,
    ) -> RecallScore
    pub struct RecallMeasurement { … }
    pub fn write_recall_report(
        path: &Path, records: &[RecallMeasurement],
    ) -> Result<PathBuf, std::io::Error>

No new crate dependencies.
