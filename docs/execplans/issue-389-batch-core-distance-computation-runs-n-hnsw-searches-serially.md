# Parallelize batch core-distance computation (#389)

This ExecPlan (execution plan) is a living document. The sections `Constraints`,
`Tolerances`, `Risks`, `Progress`, `Surprises & discoveries`, `Decision log`,
`Outcomes & retrospective`, `Conformance basis`, and `Verification plan` must
be kept up to date as work proceeds.

Status: IN PROGRESS

Branch:
`issue-389-batch-core-distance-computation-runs-n-hnsw-searches-serially`

Roadmap item: none directly; follow-on to the 2.3.2 review tracked by GitHub
issue 388.

## Purpose / big picture

The one-shot CPU clustering pipeline builds an HNSW index, computes one core
distance per point, weights the harvested candidate edges by mutual
reachability, builds a minimum spanning forest with Kruskal, and extracts flat
cluster labels. The index build and the mutual-reachability transform already
run under Rayon. The core-distance stage between them does not: it performs one
full HNSW search per point on the calling thread.

A core distance is a point's distance to its `m`-th nearest non-self neighbour,
where `m` is the configured minimum cluster size. The stage is therefore `n`
independent read-only HNSW searches at search width
`ef = max(m + 1, ef_construction).min(n)`, followed by a pure selection over
each result list. On a multi-core host this is expected to be the largest
remaining CPU-pipeline win.

After this change, a caller using `Chutoro::run` on a 1,000-point or
10,000-point source observes the same cluster assignments, computed faster
because the searches run in parallel. A caller who monitors metrics sees a new
bounded `core_distances` label on `chutoro.cpu_pipeline.stage_seconds`. A
maintainer can measure the stage and the whole pipeline with a dedicated
Criterion target and an opt-in stage profile.

The change also consolidates the "`m - 1`-th neighbour, else last, else zero"
selection rule, which currently exists in three copies (batch pipeline,
session, benchmark helper), into one public function. That removes the risk of
the batch and incremental paths drifting apart.

Observable success:

1. `cargo nextest run --all-features -E 'test(core_distances)'` passes,
   including parity between explicit one-thread and eight-thread Rayon pools.
2. `make test` passes, so existing batch and session behaviour is unchanged.
3. `make bench`-style runs of the new `pipeline` target record before/after
   medians at `n = 1,000` and `n = 10,000`.
4. `make lint` and `make check-fmt` pass.

## Constraints

- Core-distance output must be identical, per index, at one thread and at
  several threads. No value may move between positions.
- When several points fail, the reported error must be the one with the lowest
  source index, at any thread count. This preserves the serial fail-fast
  behaviour's error identity, which is the defect class recorded in issue #206.
- Every index must still run its search. A failing point must not suppress
  searches at later indices in a way that changes observable side effects such
  as distance-cache population; document this difference from serial fail-fast.
- The HNSW build and trim-scoring locking must not change. Issue #378 was fixed
  by scoring trim jobs sequentially on the insert-mutex holder (PR #386), and
  `CpuHnsw::search` takes only the shared graph read guard plus the distance
  cache, never `insert_mutex`.
- Do not mutate `RAYON_NUM_THREADS` in tests. Use explicit
  `rayon::ThreadPoolBuilder` pools.
- Do not change the `ef` formula, the self-filtering rule, or the
  empty-input behaviour of the stage.
- The session recomputation methods `recompute_core_distances` and
  `recompute_core_distances_full` keep their current serial behaviour in this
  change. Parallelizing them is deferred to a filed follow-up issue because
  they hold `&mut self`, so searches must first be separated from writes.
- The new shared helper is public API. It must be documented, doctested, and
  exported only under the `cpu` feature.
- No new runtime dependency is added to `chutoro-core`. The benchmark crate may
  gain optional, profiled-only dependencies behind a new feature.
- Repository rules apply: modules start with `//!`, public items have Rustdoc,
  files stay under 400 lines, Clippy warnings are denied, and Markdown follows
  the documentation style guide.

## Tolerances (exception triggers)

Comparison base: `git rev-parse HEAD` at the start of the work, `e5006af`, and
the working tree relative to it. ExecPlan files are excluded from every size
count.

- Files: if the change touches more than 20 non-ExecPlan files, stop and
  escalate.
- Code: if production source, tests, and benchmark logic exceed 900 net lines,
  stop and escalate.
- Configuration: if manifests, features, and CI configuration exceed 120 net
  lines, stop and escalate.
- Documentation: if prose outside `docs/execplans/` exceeds 250 net lines, stop
  and escalate.
- Interface: if a public API change beyond `core_distance_from_neighbours` and
  its export is needed, stop and escalate.
- Dependencies: if a dependency beyond optional `metrics` and `metrics-util`
  under the `pipeline-profile` feature is needed, stop and escalate.
- Benchmarks: if the `pipeline` target cannot complete a measurement at
  `n = 10,000` within 30 minutes on the development host, stop and record the
  constraint instead of weakening the measurement silently.
- Iterations: if a focused test still fails after five targeted attempts, stop
  and escalate with the observed failure.
- Ambiguity: if the "lowest-index error" contract conflicts with an existing
  observable behaviour, stop and present the options.

No comparable past ExecPlan recorded budget increases for this area; recent
plans for the CPU pipeline are documentation-heavy but small in code. The code
budget above assumes roughly 250 lines of production code, 400 lines of tests,
and 150 lines of benchmark and profiling support.

## Risks

- Risk: concurrent searches contend on the shared `DistanceCache` (a `DashMap`
  with sharded LRU mutexes), so speed-up may be well below linear. Severity:
  medium. Likelihood: medium. Mitigation: measure the stage in isolation; if
  speed-up is poor, record cache observations using the existing feature-gated
  cache metrics rather than changing the cache.
- Risk: the parallel collector changes which searches run after a failure,
  which could surprise a caller who relies on fail-fast side effects. Severity:
  low. Likelihood: high (it is a real change). Mitigation: document it in the
  developer guide and the function's Rustdoc, and pin error identity with tests.
- Risk: moving the selection rule could change edge behaviour (empty list,
  sparse list, ties). Severity: high. Likelihood: low. Mitigation: preserve the
  exact branch structure, add unit and doctest cases for each branch, and keep
  existing session regression tests unchanged.
- Risk: the benchmark's `n = 10,000` case is slow enough to make the before and
  after runs onerous. Severity: low. Likelihood: medium. Mitigation: configure
  short measurement and a small sample size, follow the existing bench
  conventions, and record the Criterion settings in this plan.
- Risk: the distance cache persists between the one-thread and eight-thread
  trials, weakening the parallelism exercised by parity tests. Severity: low.
  Likelihood: high. Mitigation: where the test must force cold cache misses,
  wrap the source with a distinct metric descriptor so cache keys cannot
  collide; document why.

## Progress

- [x] (2026-11-17 14:20Z) Reconnaissance: confirmed #378 is fixed by #386
  (`28c27bb`), `search` does not take `insert_mutex`, and established the
  one-thread/eight-thread pool convention from existing tests.
- [x] (2026-11-17 14:30Z) Branch renamed and Lody session titled.
- [x] (2026-11-17 15:50Z) `EP-M1`: `core_distances` stage label, `pipeline`
  Criterion target for n = 1,000 and n = 10,000, opt-in `pipeline-profile` CSV,
  benchmark discovery coverage, and the serial `before` baseline.
- [x] (2026-11-17 16:20Z) `EP-M2`: public `core_distance_from_neighbours`,
  doctests, unit tests, export fixture, and the session, batch, and benchmark
  consumers migrated; the benchmark's unreachable invariant branch removed.
  Session regression and BDD suites pass unchanged.
- [x] (2026-11-17 17:05Z) `EP-M3`: parallel batch searches with ordered,
  lowest-index error selection, the generic collector seam, and the parity,
  tie, sparse, singleton, empty, property, synthetic-failure, and real-search
  self-pair tests at one and eight threads. Mutation evidence recorded in the
  verification plan.
- [x] (2026-11-17 18:10Z) `EP-M4` measurement: `pipeline` bench rerun against
  `--baseline before` at four threads and a fresh stage profile; medians, stage
  shares, and the distance-cache contention check recorded under
  `Artefacts and notes`.
- [x] (2026-11-17 19:05Z) `EP-M4` documentation and follow-up: developers'
  guide, design document, roadmap link, and issue
  [#430](https://github.com/leynos/chutoro/issues/430).
- [x] (2026-11-17 21:30Z) Full deterministic gates on the candidate tree:
  `make check-fmt`, `make lint` (interrogate, rustdoc, Clippy, Whitaker),
  `make test` (1,434 passed, 1 skipped, 0 failed), `make markdownlint`
  (including spelling), and `make nixie`.
- [x] (2026-11-17 22:40Z) CodeRabbit review addressed: lint expectations scoped
  to `pipeline_impl` (with `missing_docs` kept module-wide because an `expect`
  on a macro invocation is ignored), profile rows labelled with the generated
  source size, and the parity/tie/property trials given distinct metric
  descriptors so they start with a cold distance cache.
- [x] (2026-11-17 22:50Z) Rebased onto `origin/main` (`6ddcd0a`). The replay
  itself was clean; main had removed the process-argument wrappers
  `is_benchmark_discovery` and `is_exact_benchmark_probe` from
  `chutoro_benches::criterion_support` (#426), so the new bench was migrated to
  `is_benchmark_discovery_args(std::env::args())` and
  `is_exact_benchmark_probe_args(std::env::args())` in the same pass.
- [x] (2026-11-17 22:50Z) CodeRabbit round two addressed: the parity and tie
  fixtures are one `rstest` table over `(values, min cluster size)` crossed
  with the thread counts, and `criterion_group!` was replaced by a documented
  `benches` function, which removed the file-wide `missing_docs` expectation.
- [x] (2026-11-18 00:30Z) Post-rebase deterministic gates on `52965aa`:
  `make check-fmt`, `make test` (1,448 passed, 1 skipped, 0 failed; doctests
  ok), `make typecheck`, `make lint` (interrogate, rustdoc, Clippy, Whitaker),
  `make markdownlint`, `make spelling`, and `make nixie` all pass. The count
  exceeds the earlier 1,434 because the parallel-stage work adds 18 tests.
  Force-pushed with lease from `d7d4356` to `52965aa`.
- [x] (2026-11-18 00:50Z) CodeRabbit round three addressed (D9): the profile
  fallback resolves `CARGO_TARGET_DIR` through the injected environment, the
  fallback test asserts the complete path, and the benchmark commands in this
  plan pass `--features pipeline-profile`.
- [x] (2026-11-19 00:30Z) Review follow-up for the public documentation gap:
  the real-search failure test now runs the production call site at one and
  eight threads, each trial with its own metric descriptor, and the users'
  guide and migration notes document `core_distance_from_neighbours`,
  `chutoro.cpu_pipeline.stage_seconds`, and the batch parallelism contract.
- [x] (2026-11-19 04:55Z) Rebased onto `origin/main` (`bab414a`) after #427
  merged. All eleven commits replayed with one conflict in
  `docs/users-guide.md`: main's byte-histogram precision paragraph and this
  branch's stage-metrics paragraph sit in the same list, and both are kept.
  `git range-diff 6ddcd0a..d26817f bab414a..HEAD` shows ten identical commits
  and the eleventh differing only by the resolved context; `git diff --check`
  is clean. New head `616aa4a`, force-pushed with lease from `d26817f`.
- [x] (2026-11-19 05:40Z) Post-rebase deterministic gates on `616aa4a`:
  `make check-fmt`, `make test` (1,484 passed, 1 skipped, 0 failed; doctests
  ok), `make typecheck`, `make lint` (interrogate, rustdoc, Clippy, Whitaker),
  `make markdownlint` (0 issues), `make nixie`, and `make spelling` all pass.
  Hosted CI for the same head is green for build-test, Kani, benchmark smoke,
  property tests, Verus, CodeScene, and Gecko.
- [x] (2026-11-19 07:10Z) CodeRabbit round four addressed (D10, D11, D12): the
  peak-memory estimate counts one search state per Rayon worker, CSV stage
  names are quoted as a single field, the real-search regression records every
  query served and holds the early failure until the late one has failed, and
  the migration note now says which stage labels pre-existed.
- [x] (2026-11-19 09:40Z) Round-four lint compliance: `memory.rs` moves its
  test module to `memory_tests.rs` so the module stays under Whitaker's
  400-line limit; the peak-estimate inputs became a `Copy` struct so the
  estimator keeps four arguments; the test source's constructor is `const fn`
  and its lock recovery uses `PoisonError::into_inner` instead of `expect`; and
  the CSV predicate uses `str::contains([char; 4])`. Gates on the resulting
  tree: `make check-fmt`, `make typecheck`, `make test` (1,492 passed, 1
  skipped; doctests ok), and `make lint` (interrogate, rustdoc, Clippy,
  Whitaker) all pass.
- [ ] `EP-M4` delivery confirmation: re-push and final review.

## Surprises & discoveries

- Observation: the distance cache keys include the source's
  `MetricDescriptor`, so the same index can be reused across trials with
  distinct fault behaviour by giving each trial source a distinct descriptor.
  Evidence: `chutoro-core/src/hnsw/distance_cache/mod.rs` `DistanceKey`.
  Impact: parity and fault tests can share one built index without one trial
  poisoning the next trial's cache.
- Observation: `ensure_query_present` calls `distance(query, query)` whenever
  `ef > 1` and the query is absent from the result list, so a source that
  rejects self-pair distances fails exactly the searches whose query index it
  rejects. Evidence: `chutoro-core/src/hnsw/helpers.rs`. Impact: a real-search
  failure fixture can target specific point indices without needing to know
  HNSW traversal order.
- Observation: the four-thread stage speedup (3.81x at n = 1,000, 3.66x at
  n = 10,000) is close to the theoretical maximum, so the shared distance cache
  is not the bottleneck at this width and needs no change. Evidence:
  `Artefacts and notes`, after-measurement profile. Impact: the follow-up issue
  should not propose cache surgery without a wider-contention datum.
- Observation: `validate_distance` consults the distance cache and then calls
  `DataSource::distance` with no cache guard live: `begin_lookup` returns a
  `PendingMiss` after dropping the `DashMap` entry guard, and `complete_miss`
  inserts afterwards. Evidence: `chutoro-core/src/hnsw/validate.rs` and
  `distance_cache/mod.rs`. Impact: a test fixture may block inside `distance`
  without risking a lock held by another worker.
- Observation: `make lint-clippy` reports `chutoro-benches` failures only after
  the workspace's earlier crates compile, so an earlier run that stopped in
  `chutoro-core` hid four benchmark-target lints in `pipeline.rs` and
  `pipeline_profile.rs`. Evidence:
  `error: could not compile chutoro-benches (bench "pipeline")` after the
  `chutoro-core` fixes landed. Impact: always read the gate log to its final
  summary rather than the first failure block.

## Decision log

- D1. Decision: parallelize the batch stage only, and defer session
  parallelization to follow-up issue
  [#430](https://github.com/leynos/chutoro/issues/430), linked from roadmap
  item 11.2.4. Rationale: the batch stage has a clean read-only seam
  (`&CpuHnsw`, `&DataSource`). The session methods hold `&mut self`, so
  parallel searches must first be separated from the writes that follow;
  bundling that larger refactor would enlarge the risk surface of a performance
  change. The issue explicitly allows either decision provided it is explicit.
  Date/Author: 2026-11-17, implementation agent.
- D2. Decision: collect indexed results into `Vec<Result<T, E>>`, then scan
  serially for the first error. Rationale: `collect::<Result<Vec<_>>>()`
  returns whichever error Rayon observes first, which is nondeterministic. A
  serial scan over an order-preserving indexed collect yields the lowest-index
  error, matching the serial loop. Date/Author: 2026-11-17, implementation
  agent.
- D3. Decision: keep stage timing on the orchestration thread and record it
  only when the stage succeeds, matching the existing `record_stage` calls.
  Rationale: a failed run has no meaningful stage duration, and recording before
  `?` would count abandoned work. The clock is read once per stage, so the
  measurement cannot bias the searches themselves. Date/Author: 2026-11-17,
  implementation agent.
- D4. Decision: write the profile to the existing benchmark report
  convention, `<CARGO_TARGET_DIR>/benchmarks/pipeline_profile.csv`, instead of
  creating a new directory, and honour `CHUTORO_BENCH_PIPELINE_PROFILE_PATH` as
  a full-path override. Rationale: `neighbour_scoring::report_path` already
  owns the target-directory convention and `report_file::write_report_file`
  already creates missing parents; reusing them avoids a second, divergent
  report location. The override keeps reruns able to preserve an earlier
  profile. Date/Author: 2026-11-17, implementation agent.
- D5. Decision: leave the sharded distance cache unchanged. Rationale: the
  stage reaches 3.66-3.81x on four threads, so cache shard contention does not
  measurably limit this workload; changing the cache would add risk without
  evidence. The follow-up issue records the measurement rather than a
  speculative fix. Date/Author: 2026-11-17, implementation agent.
- D6. Decision: reuse the existing benchmark-file lint conventions
  (`#![expect(clippy::shadow_reuse, clippy::excessive_nesting)]`) and the local
  `panic_in_result_fn` suppression used by `mst.rs`, rather than restructuring
  the Criterion registration closures. Rationale: the lints describe
  unavoidable Criterion API shapes, and every other bench target in the crate
  carries the same targeted expectations. Date/Author: 2026-11-17,
  implementation agent.
- D7. Decision: declare the Criterion group as a documented `benches` function
  and keep `criterion_main!(benches)`, instead of using `criterion_group!`.
  Rationale: the macro expands to an undocumented public function, which forced
  a file-wide `missing_docs` suppression over every item in the benchmark; the
  handwritten function matches the macro's expansion
  (`Criterion::default().configure_from_args()` followed by the target call)
  and lets the lint stay enabled. Date/Author: 2026-11-17, implementation agent.
- D8. Decision: adapt the benchmark to main's argument-passing helpers
  (`is_benchmark_discovery_args`, `is_exact_benchmark_probe_args`) rather than
  reintroducing the removed process-argument wrappers. Rationale: #426 removed
  those wrappers from `chutoro_benches::criterion_support` so each decision is
  testable from a supplied argument sequence; matching every other bench target
  keeps one API. Date/Author: 2026-11-17, implementation agent.
- D9. Decision: resolve the fallback profile destination from the injected
  `Env` itself, using `report_parent_dir_value` and `report_path_value`,
  instead of `neighbour_scoring::report_path`, which reads `CARGO_TARGET_DIR`
  through `DefaultEnv`. Rationale: the developers' guide confines `DefaultEnv`
  to the application boundary, and a mocked environment must control the whole
  path for the fallback test to mean anything. `CARGO_TARGET_DIR_ENV` is now
  shared so the variable name is not repeated. Date/Author: 2026-11-18,
  implementation agent.
- D10. Decision: make the batch memory guard count one HNSW search state per
  Rayon worker through a new public
  `estimate_peak_bytes_for_hnsw_params_with_concurrency`, keeping the existing
  `estimate_peak_bytes_for_hnsw_params` at its single-search meaning. The guard
  derives the worker count from `rayon::current_num_threads()`. Rationale: the
  parallel stage holds one search state per live search, so the former
  one-search estimate under-counted peak memory and could admit a run that
  exhausts the limit. A separate explicit function keeps the old estimate
  usable for callers that know they run one search, and the `with_max_bytes`
  rejection is conservative rather than silent over-commitment. Date/Author:
  2026-11-19, implementation agent.
- D11. Decision: quote a CSV stage name only when it contains a comma, quote,
  carriage return, or line feed, doubling embedded quotes. Rationale: RFC 4180
  minimal quoting keeps the built-in labels unchanged while preventing a
  composed label from splitting the row. The profile still declares four fields
  per row. Date/Author: 2026-11-19, implementation agent.
- D12. Decision: strengthen the real-search regression with a recorded
  query set and a bounded ordering handshake rather than lengthening the
  synthetic collector test. Rationale: the production call site, not the
  collector alone, must be shown to search every point after a failure; and the
  eight-thread case must demonstrate a higher-index failure completing before
  the reported lower-index one. The handshake waits only when the pool has a
  second worker, so the one-thread case never waits on itself, and the timeout
  falls through so a scheduling anomaly fails the ordering witness instead of
  hanging the suite. The distance cache releases its shard guards before calling
  `DataSource::distance`, so the wait cannot deadlock on a cache lock.
  Date/Author: 2026-11-19, implementation agent.

## Outcomes & retrospective

Delivered as draft pull request
[#431](https://github.com/leynos/chutoro/pull/431). A user can now run the
one-shot CPU pipeline and see the batch core-distance stage use the surrounding
Rayon pool, with per-index results and error identity unchanged at one and
eight threads. The stage profile shows `core_distances` falling from about 37%
of the pipeline to about 13%, and the whole-run medians fall by 13.6% at n =
1,000 and 10.6% at n = 10,000.

What worked well: keeping the ordered collect and the serial first-error scan
as a tiny private seam made the determinism contract easy to state and to
mutate against, and the independent sequential oracle kept the parity tests
from sharing a defect with production. Measuring before parallelizing
established that the stage was worth the change and gave the after run a fixed
reference.

What to carry forward: a short-circuiting `collect::<Result<Vec<_>>>()` looks
equivalent but changes observable error identity at thread counts above one, so
the developers' guide now warns against "simplifying" it. Distinct metric
descriptors are the cheapest way to isolate distance-cache state between
trials. Full-workspace gates are expensive on this host under compile
admission; the focused suites for changed crates gave fast signal, while the
full run needed to wait for a drained queue. A clean textual rebase does not
prove that a benchmark still compiles: main removed the shared
`criterion_support` process-argument wrappers while this branch was open, so
the bench targets need a focused `cargo check` after every rebase even when
`range-diff` reports an unchanged replay.

Remaining work is tracked in
[#430](https://github.com/leynos/chutoro/issues/430) for the session path.

## Context and orientation

The relevant code at base revision `e5006af`:

- `chutoro-core/src/cpu_pipeline/mod.rs` owns the batch CPU pipeline.
  `run_cpu_pipeline_with_len` calls `build_index_and_core_distances` and
  `build_mutual_reachability_forest`. `compute_core_distances` is the serial
  loop to replace. `PipelineStage` is the bounded metric-label vocabulary.
  Tests live in `chutoro-core/src/cpu_pipeline/tests/`, registered by
  `tests/mod.rs`.
- `chutoro-core/src/hnsw/cpu/mod.rs` defines `CpuHnsw`. `search` takes the
  graph read guard and the shared `DistanceCache`, and never `insert_mutex`.
  `insert_with_collector` takes `insert_mutex`, and `score_trim_jobs`
  (`hnsw/cpu/trim.rs`) deliberately scores on the mutex-holding thread after
  the #378 deadlock.
- `chutoro-core/src/session/core_distance.rs` holds the session's private
  `core_distance_from_neighbours` and `effective_ef` helpers plus
  `recompute_core_distances` and `recompute_core_distances_full`, which call
  `search_non_self_neighbours` serially.
- `chutoro-core/src/hnsw/distance_cache/mod.rs` implements a sharded
  `DashMap` cache keyed by metric, left node, and right node.
- `chutoro-core/src/batch_metrics.rs` exposes
  `record_pipeline_stage(stage, elapsed)` for the
  `chutoro.cpu_pipeline.stage_seconds` histogram.
- `chutoro-benches/benches/` holds the Criterion targets (`hnsw`, `mst`,
  `edge_harvest`, `extraction`, `neighbour_scoring`, `hnsw_ef_sweep`), each
  declared with `harness = false` in `chutoro-benches/Cargo.toml`.
  `chutoro-benches/benches/internal/quality_pass.rs` contains the benchmark
  copy of the selection rule.
- `chutoro-benches/tests/benchmark_smoke.rs` asserts Criterion discovery and
  exact-probe coverage for each target.
- Public batch entrypoint: `ChutoroBuilder` in `chutoro-core/src/builder.rs`
  and `Chutoro::run` in `chutoro-core/src/chutoro/mod.rs`.
- Documentation: `docs/developers-guide.md` documents stage metrics and the
  benchmark workflow; `docs/chutoro-design.md` documents the batch pipeline;
  `docs/roadmap.md` item 11.2.4 covers session full refresh.

Test and gate commands (repository root):

```plaintext
cargo nextest run --all-features -p chutoro-core -E 'test(core_distances)'   # focused
make check-fmt
make lint
make test
make markdownlint
make nixie
```

`make test` runs `cargo nextest run --all-targets --all-features` and then
`cargo test --doc`. Benchmarks are excluded by filter from nextest and run
through `cargo bench`.

## Conformance basis

Upstream artefacts inspected at `e5006af`:

- GitHub issue #389 (this work): parallelize the batch core-distance stage,
  preserve output order, report the lowest-index error, measure before and
  after, deduplicate the selection rule, and decide explicitly about the
  session path.
- GitHub issue #388: prioritization of the 2.3.2 review follow-ups.
- GitHub issue #378 and PR #386: the trim-scoring deadlock fix that unblocked
  this work.
- `docs/chutoro-design.md` sections describing the batch pipeline and session
  core distances.
- `docs/developers-guide.md` sections on pipeline stage metrics and the
  benchmark regression workflow.
- `AGENTS.md` quality gates and `docs/documentation-style-guide.md`.

Requirements derived from the issue:

- `REQ-parity`: per-index core-distance output is identical at one and N
  threads.
- `REQ-lowest-error`: the reported error is the lowest-index failure at any
  thread count.
- `REQ-measure`: record before/after stage and whole-pipeline measurements at
  `n = 1,000` and `n = 10,000`.
- `REQ-dedup`: one shared selection helper serves the batch, session, and
  benchmark paths.
- `REQ-session-decision`: an explicit, recorded decision about session
  parallelization.

Trace links:

```plaintext
REQ-parity        -> EP-M3 -> cpu_pipeline::tests::core_distances::*
REQ-lowest-error  -> EP-M3 -> cpu_pipeline::tests::core_distances::reports_lowest_index_error_*
REQ-measure       -> EP-M1, EP-M4 -> bench target `pipeline`, execplan Artefacts
REQ-dedup         -> EP-M2 -> crate::core_distance::core_distance_from_neighbours
REQ-session-decision -> EP-M4 -> Decision log D1 and follow-up issue
```

## Verification plan

### Abstract model

The stage maps an index `i` in `0..n` to a result `R(i)` that is either a
finite core distance `d(i)` or an error `E(i)`. The parallel implementation
must produce `Ok([d(0), ..., d(n-1)])` when every `R(i)` is `Ok`, and
`Err(E(k))` where `k = min { i | R(i) is Err }` otherwise. Contract: the serial
implementation produces `Err(E(j))` for the first failing `j` and stops.

The selection rule `C(L, m)` over a non-self, ascending-distance list `L` is:
`L[m-1].distance` when `|L| >= m`; otherwise `L.last().distance`; otherwise
`0.0`.

### Invariants and lemmas

- `INV-order`: the output vector's position `i` equals the core distance of
  point `i`. Evidence must not rely on Rayon's scheduling order.
- `INV-error`: the returned error, when any point fails, is the lowest-index
  failure.
- `INV-every-index-runs`: the parallel map visits every index, so cache
  population and source-call side effects are independent of which index failed.
- `L-select`: `core_distance_from_neighbours` implements `C(L, m)` exactly,
  including empty and sparse lists.
- `L-parity`: if every `R(i)` is deterministic, then the parallel output equals
  the serial output elementwise.

### Axioms (trusted, not verified here)

- `AXIOM-rayon-indexed`: `rayon`'s `Vec` collection from an indexed parallel
  iterator preserves input order. Source: rayon's documented
  `IndexedParallelIterator` contract.
- `AXIOM-search-pure`: `CpuHnsw::search` is read-only with respect to the
  graph, and concurrent searches are safe. Source: the read-guard and
  `DashMap`-based cache implementation plus the existing parallel build.
- `AXIOM-datasource`: `DataSource::distance` is deterministic for a fixed pair
  and source instance. Source: the `DataSource` trait contract.

### Methods per obligation

- Obligation `L-select`.
  Method: parameterized unit tests plus doctests. Rationale: the rule is a
  finite partition (dense, sparse, empty, `m = 1`, ties). Artefact:
  `chutoro-core/src/core_distance.rs` tests and doctest. Evidence:
  `cargo nextest run --all-features -p chutoro-core -E 'test(core_distance)'`.
  Non-vacuity: each branch has at least one witness; a deliberately wrong
  branch (for example `last()` for dense lists) fails a named case.
- Obligation `INV-order` and `L-parity`.
  Method: property tests and named cases using explicit one-thread and
  eight-thread pools, compared with a sequential oracle. Rationale: order
  preservation over generated datasets is exactly the risk. Artefact:
  `chutoro-core/src/cpu_pipeline/tests/core_distances.rs`. Evidence:
  `successful_results_keep_source_order` reverses completion order with sleeps
  and still requires index order, so an unordered collection fails it.
  Non-vacuity: fixtures include ties, sparse lists, and a singleton; the oracle
  is a separate sequential loop over `index.search`.
- Obligation `INV-error`.
  Method: synthetic failure test with errors at indices 3, 7, and 11 and
  artificial delays at lower indices, plus a real-search self-pair failure
  fixture, each at one and eight threads. Rationale: the failure mode is
  scheduling-dependent, so it must be driven through real Rayon pools.
  Artefact: same test module. Evidence: mutating the seam to
  `results.collect::<Result<Vec<T>, E>>()` makes the eight-thread case return
  `Err(11)` instead of `Err(3)` (observed 2026-11-17), so the test rejects the
  completion-ordered collector for the intended reason. Non-vacuity: the
  delayed lower index actually fails, and the real fixture is shown to fail at
  the intended index.
- Obligation `INV-every-index-runs`.
  Method: a counter that records which index tasks ran in the presence of a
  failure. Rationale: side-effect behaviour is observable and cheap to pin.
  Artefact: same test module. Evidence: the ordered collector records all 16
  indices; the mutated collector recorded 4 at one thread and 12 at eight
  threads before returning.
- Obligation `REQ-measure`.
  Method: Criterion baseline comparison plus a local-recorder stage profile.
  Rationale: performance claims need before/after evidence on the same host.
  Artefact: `chutoro-benches/benches/pipeline.rs` and the execplan Artefacts
  section. Evidence: recorded medians and shares.

### Formal verification decision

No new algorithmic invariant beyond ordering and selection is introduced, and
the concurrency is data-parallel with no shared mutable domain state. Rayon's
indexed-order contract and the existing distance-cache concurrency are trusted
axioms exercised by the real implementation in the parity tests. Kani or Verus
would not add proportionate confidence for an order-preserving collection
wrapper over a third-party parallel iterator, so no proof is planned. The
selection rule is a pure function already covered by exhaustive finite cases.

### Suggested BDD features

The repository uses `rstest-bdd` for behavioural suites. This change has no
user-facing interaction flow beyond `Chutoro::run` producing the same labels,
which the existing behavioural suites already cover. A new BDD feature is
therefore not proposed; parity and error identity are pinned by `rstest` and
property tests at the unit seam where the behaviour is defined.

## Plan of work

### `EP-M1` Instrumentation, benchmark target, and serial baseline

Add `PipelineStage::CoreDistances` with label `core_distances`, wrap the
`compute_core_distances` call in `build_index_and_core_distances` with
`Instant`, and record the duration only after `?` succeeds. Update
`cpu_pipeline/tests/metrics.rs` to require exactly one sample per label for
`core_distances`, `mutual_reachability`, and `kruskal`.

Add `chutoro-benches/benches/pipeline.rs`, declared in
`chutoro-benches/Cargo.toml` with `harness = false`. The benchmark generates a
seeded 16-dimensional `SyntheticSource` outside the timed closure for
`n = 1,000` and `n = 10,000`, builds `ChutoroBuilder` with the same HNSW
parameters used by the other targets, and times `Chutoro::run`. Discovery and
nextest exact-probe paths register bounded no-op cases, following
`chutoro-benches/benches/mst.rs`.

Add the `pipeline-profile` feature to `chutoro-benches` enabling
`chutoro-core/metrics`, optional `metrics`, and optional
`metrics-util/debugging`. Under that feature, the `pipeline` target installs a
`DebuggingRecorder` for one untimed pipeline run before Criterion registration
and writes one CSV row per stage plus a whole-run row to the shared benchmark
report directory, `<CARGO_TARGET_DIR>/benchmarks/pipeline_profile.csv`, whose
full path is overridable with `CHUTORO_BENCH_PIPELINE_PROFILE_PATH`. The row
set includes each stage's seconds and share of the whole run, so the
`core_distances` share is explicit. Without the feature, the profile step is a
no-op.

Extend `chutoro-benches/tests/benchmark_smoke.rs` to assert `--list` coverage
and one `--exact` probe for the new target.

Run the serial baseline:

```plaintext
cargo bench -p chutoro-benches --features pipeline-profile --bench pipeline \
  -- --save-baseline before --noplot
```

Record medians for both sizes and the stage profile in `Artefacts and notes`.

### `EP-M2` Shared selection helper

Create `chutoro-core/src/core_distance.rs` with the public
`core_distance_from_neighbours(&[Neighbour], NonZeroUsize) -> f32`, gated on the
`cpu` feature and re-exported from `chutoro-core/src/lib.rs`. Document the
ordered, non-self precondition, the fallback behaviour, and the no-panic
contract. Add doctests and unit tests for the dense, sparse, empty, `m = 1`,
and tie cases, plus an integration-test fixture that proves the export exists
under the feature.

Replace the private session helper with the shared one, keeping the session's
filtering and error behaviour unchanged. Replace the batch copy by filtering
self by id and calling the helper. Replace the benchmark copy while retaining
its `f64` conversion, empty-list diagnostic, and `BenchSetupError` handling,
and delete the unreachable invariant branch.

Update `docs/repository-layout.md` for the new module.

### `EP-M3` Parallel batch searches

Add a private generic seam in `chutoro-core/src/cpu_pipeline/mod.rs` that
collects an indexed parallel iterator of `Result<T, E>` into a `Vec` and scans
it serially to return the first error in index order. Document that every index
runs. Use it in `compute_core_distances` with
`(0..inputs.items).into_par_iter()`; create the source-name `Arc` once and
clone it per failure; filter self; call the shared helper. Preserve the `ef`
formula and the empty-input behaviour.

Add `chutoro-core/src/cpu_pipeline/tests/core_distances.rs` and register it in
`tests/mod.rs`. Cover parity against a sequential oracle at one and eight
threads, ties, sparse and singleton cases, seeded property tests over the
collector, synthetic failures at indices 3, 7, and 11 with delayed lower
indices, the every-index-runs counter, and the real-search self-pair fixture
with distinct metric descriptors and a fresh query-11-only trial.

### `EP-M4` Measurement, documentation, and delivery

Rerun the `pipeline` benchmark with `--baseline before` and the stage profile
under the same conditions. Record medians, percentage changes, and
`core_distances` shares.

Update `docs/developers-guide.md` with the `core_distances` stage label, the
parallel execution and lowest-index error rule, the later-query side-effect
behaviour, and the `pipeline` benchmark instructions. Add a batch-stage
paragraph to `docs/chutoro-design.md`.

File the follow-up issue for session core-distance parallelization, link it
near roadmap item 11.2.4, and record the decision in the PR description.

## Milestones and plateaus

- `EP-M1` outcome: the serial stage is instrumented and measured.
  Requirements: `REQ-measure`. Acceptance evidence: three stage labels and a
  saved `before` Criterion baseline. Conformance check: no production behaviour
  change; metric cardinality still bounded; no new runtime dependency; no
  persisted-format change. Recovery: revert the benchmark and metrics edits.
  Remaining gaps: parallel execution. Compatibility decision: none.
- `EP-M2` outcome: one shared selection rule; batch output unchanged.
  Requirements: `REQ-dedup`. Acceptance evidence: helper tests plus the
  unchanged session regression tests and pipeline tests. Conformance check: the
  only public addition is the helper; no behaviour drift. Recovery: revert the
  helper and its three consumers together. Remaining gaps: parallel execution.
  Compatibility decision: none (pre-1.0 public API).
- `EP-M3` outcome: the batch stage runs in parallel with deterministic
  ordering and error identity. Requirements: `REQ-parity`, `REQ-lowest-error`.
  Acceptance evidence: the new test module at one and eight threads.
  Conformance check: the `ef` formula and self-filter are unchanged; the build
  and locking are untouched. Recovery: revert `compute_core_distances` to the
  serial loop; the helper remains. Remaining gaps: session path. Compatibility
  decision: none.
- `EP-M4` outcome: measured improvement, documented contract, filed
  follow-up. Requirements: `REQ-measure`, `REQ-session-decision`. Acceptance
  evidence: baseline comparison transcript, guide and design text, follow-up
  issue link. Conformance check: documentation matches implementation; no stale
  claim of a serial stage remains. Recovery: documentation-only changes are
  independent commits. Remaining gaps: session parallelization. Compatibility
  decision: none.

## Concrete steps

Working directory: repository root.

```plaintext
Red:     cargo nextest run --all-features -p chutoro-core -E 'test(core_distances)'
         -> the new module's tests compile but fail (missing parallel collector)
Green:   same command -> all tests pass
Refactor: make check-fmt && make lint -> pass
```

Gate sequence, run sequentially, in this order, with `pipefail` and logs:

```plaintext
set -o pipefail
make check-fmt 2>&1 | tee /tmp/check-fmt-$(basename "$PWD")-$(git branch --show-current).out
set -o pipefail
make lint 2>&1 | tee /tmp/lint-$(basename "$PWD")-$(git branch --show-current).out
set -o pipefail
make test 2>&1 | tee /tmp/test-$(basename "$PWD")-$(git branch --show-current).out
set -o pipefail
make markdownlint 2>&1 | tee /tmp/markdownlint-$(basename "$PWD")-$(git branch --show-current).out
set -o pipefail
make nixie 2>&1 | tee /tmp/nixie-$(basename "$PWD")-$(git branch --show-current).out
```

Commit gates are delegated to the `scrutineer` subagent; read its cited logs
rather than re-running gates.

## Validation and acceptance

- Run
  `cargo nextest run --all-features -p chutoro-core -E 'test(core_distances)'`
  and expect the parity, ordering, and error-identity tests to pass. Before
  `EP-M3`, the same selection already passes through the oracle path; the new
  parallel seam's tests fail until the seam exists.
- Run the metrics test
  `cargo nextest run --all-features -p chutoro-core -E 'test(every_stage_records_a_latency_sample)'`
  and expect three labels exactly.
- Run the after measurement and expect both benchmark IDs to report a change
  measurement:

  ```plaintext
  cargo bench -p chutoro-benches --features pipeline-profile --bench pipeline \
    -- --baseline before --noplot
  ```

- Run `make check-fmt`, `make lint`, `make test`, `make markdownlint`, and
  `make nixie`; expect all to pass.

Quality criteria:

- Tests: unit, property, and behavioural suites pass at `--all-features`.
- Verification: `INV-order`, `INV-error`, `INV-every-index-runs`,
  `L-select`, and `L-parity` are discharged.
- Lint/typecheck: Rustdoc, Clippy, and Whitaker pass with warnings denied.
- Performance: recorded before/after medians and stage shares.
- Security: no new trust boundary or external input surface.

## Idempotence and recovery

Edits are ordinary file changes and can be reverted per milestone with Git.
Benchmark baselines live under `target/criterion/`, are reproducible from the
recorded command, and can be deleted and rebuilt. The profile CSV is
regenerated on each profiled run. If the profile run fails after opening the
output, rerun after checking the path override; the file is truncated per run.

## Artefacts and notes

Reconnaissance evidence (base `e5006af`):

```plaintext
git log --oneline --all --grep='378\|386'
28c27bb Score trim jobs on the insert mutex holder (#378) (#386)
```

`CpuHnsw::search` (`chutoro-core/src/hnsw/cpu/mod.rs:254`) acquires
`read_graph_guard()` and uses `self.distance_cache`; it contains no
`insert_mutex` reference.

Baseline measurement (`EP-M1`, 2026-11-17, serial `compute_core_distances`):

```plaintext
RAYON_NUM_THREADS=4 cargo bench --offline --locked -p chutoro-benches \
  --features pipeline-profile --bench pipeline -- --save-baseline before
```

Host: 6-core Linux. Criterion group: sample size 10, default 3 s warm-up and 5
s measurement target; the logged "unable to complete 10 samples in 5.0 s"
warning is expected for these sizes. Results:

```plaintext
pipeline/n=1000         time:   [1.4361 s 1.4615 s 1.4941 s]
pipeline/n=10000        time:   [35.825 s 36.768 s 37.783 s]
```

Stage profile from the same process (one untimed run per size,
`target/benchmarks/pipeline_profile.csv`):

```plaintext
point_count,stage,seconds,share
1000,whole_run,1.747013,1.0000
1000,core_distances,0.656701,0.3759
1000,kruskal,0.002283,0.0013
1000,mutual_reachability,0.014921,0.0085
10000,whole_run,35.734580,1.0000
10000,core_distances,13.179279,0.3688
10000,kruskal,0.030802,0.0009
10000,mutual_reachability,0.058460,0.0016
```

The remaining whole-run time is the HNSW build and label extraction, which are
not stage-labelled. The `core_distances` stage is about 37% of the whole
pipeline at both sizes before parallelization.

After measurement (`EP-M4`, 2026-11-17, parallel `compute_core_distances`):

```plaintext
RAYON_NUM_THREADS=4 cargo bench --offline --locked -p chutoro-benches \
  --features pipeline-profile --bench pipeline -- --baseline before
```

Criterion compares against the saved `before` baseline under the same host and
settings:

```plaintext
pipeline/n=1000         time:   [1.1818 s 1.2259 s 1.2683 s]
                        change: [-19.478% -16.117% -12.940%] (p = 0.00 < 0.05)
pipeline/n=10000        time:   [31.590 s 33.926 s 37.188 s]
                        change: [-14.560% -7.7311% +1.9803%] (p = 0.10 > 0.05)
```

Saved medians: n = 1,000 falls from 1.4538 s to 1.2557 s (−13.6%); n = 10,000
falls from 36.6385 s to 32.7410 s (−10.6%). The n = 10,000 interval is wide
because ten samples of a ~35 s pipeline span minutes of host activity, so its
Criterion p-value does not reach the 0.05 threshold; the stage profile below
attributes the improvement to the intended stage either way.

Stage profile after the change (same process, one untimed run per size):

```plaintext
point_count,stage,seconds,share
1000,whole_run,1.298107,1.0000
1000,core_distances,0.172432,0.1328
1000,kruskal,0.002713,0.0021
1000,mutual_reachability,0.013450,0.0104
10000,whole_run,29.621945,1.0000
10000,core_distances,3.602843,0.1216
10000,kruskal,0.021009,0.0007
10000,mutual_reachability,0.027135,0.0009
```

The `core_distances` stage falls from 0.657 s to 0.172 s at n = 1,000 (3.81x)
and from 13.179 s to 3.603 s at n = 10,000 (3.66x) on a four-thread pool, so
the stage scales near-linearly and shows no pathological distance-cache
contention at this width. Its share of the pipeline falls from 37.6% to 13.3%
and from 36.9% to 12.2% respectively. Whole-pipeline gains are smaller than the
stage gain because the unlabelled HNSW build still dominates: Amdahl's law over
a 37% stage at four threads predicts a 1.28x whole-run ceiling, and the
observed medians sit below that ceiling rather than contradicting the stage
result.

## Interfaces and dependencies

New public API:

```rust
// chutoro-core/src/core_distance.rs, gated on feature "cpu"
pub fn core_distance_from_neighbours(
    neighbours: &[Neighbour],
    min_cluster_size: NonZeroUsize,
) -> f32;
```

Export: `chutoro_core::core_distance_from_neighbours`, re-exported under the
`cpu` feature. Ownership: the core crate owns the rule; the batch pipeline, the
session, and the benchmark crate consume it. Reuse policy: no call site may
re-implement the selection rule.

Private seam:

```rust
// chutoro-core/src/cpu_pipeline/mod.rs
fn collect_ordered_results<T, E>(
    results: impl ParallelIterator<Item = Result<T, E>>,
) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send;
```

Benchmark feature and dependencies (all optional):

```toml
[features]
pipeline-profile = ["chutoro-core/metrics", "dep:metrics", "dep:metrics-util"]
```

```toml
metrics = { version = "0.24.0", optional = true }
metrics-util = { version = "0.20", features = ["debugging"], optional = true }
```

No runtime dependency is added to `chutoro-core`.

## Revision note

- 2026-11-17: Initial ExecPlan written from the issue text and repository
  reconnaissance at `e5006af`; implementation proceeds under the user's
  explicit instruction to deliver, measure, and open a draft pull request.
- 2026-11-17: Rebased onto `origin/main` at `6ddcd0a` and folded in both
  CodeRabbit review rounds; the benchmark now uses main's `_args`
  `criterion_support` helpers and a documented Criterion group function (D7,
  D8).
- 2026-11-18: Folded in the third CodeRabbit review round: the profile
  fallback reads the injected environment (D9), and the plan's benchmark
  commands enable `pipeline-profile`.
- 2026-11-19: Added a two-thread-count call-site regression for the real-search
  failure path, and documented the public helper, the stage metric label, and
  the batch parallelism contract in the users' guide and migration notes.
- 2026-11-19: Rebased a second time onto `origin/main` at `bab414a`, resolving
  one `docs/users-guide.md` conflict by keeping both branches' paragraphs, and
  re-ran every deterministic gate on the new head.
- 2026-11-19: Folded in the fourth CodeRabbit review round: the peak-memory
  estimate is concurrency-aware (D10), profile stage names are CSV-safe (D11),
  the real-search regression proves full coverage and failure ordering (D12),
  and the migration note distinguishes the new `core_distances` label from the
  stage labels that pre-existed.
