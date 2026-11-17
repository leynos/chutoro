# Parallelize batch core-distance computation (#389)

This ExecPlan (execution plan) is a living document. The sections
`Constraints`, `Tolerances`, `Risks`, `Progress`, `Surprises & discoveries`,
`Decision log`, `Outcomes & retrospective`, `Conformance basis`, and
`Verification plan` must be kept up to date as work proceeds.

Status: IN PROGRESS

Branch: `issue-389-batch-core-distance-computation-runs-n-hnsw-searches-serially`

Roadmap item: none directly; follow-on to the 2.3.2 review tracked by GitHub
issue 388.

## Purpose / big picture

The one-shot CPU clustering pipeline builds an HNSW index, computes one core
distance per point, weights the harvested candidate edges by mutual
reachability, builds a minimum spanning forest with Kruskal, and extracts flat
cluster labels. The index build and the mutual-reachability transform already
run under Rayon. The core-distance stage between them does not: it performs one
full HNSW search per point on the calling thread.

A core distance is a point's distance to its `m`-th nearest non-self
neighbour, where `m` is the configured minimum cluster size. The stage is
therefore `n` independent read-only HNSW searches at search width
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
selection rule, which currently exists in three copies (batch pipeline, session,
benchmark helper), into one public function. That removes the risk of the batch
and incremental paths drifting apart.

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
  as distance-cache population; document this difference from serial
  fail-fast.
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
  with sharded LRU mutexes), so speed-up may be well below linear.
  Severity: medium. Likelihood: medium. Mitigation: measure the stage in
  isolation; if speed-up is poor, record cache observations using the existing
  feature-gated cache metrics rather than changing the cache.
- Risk: the parallel collector changes which searches run after a failure,
  which could surprise a caller who relies on fail-fast side effects.
  Severity: low. Likelihood: high (it is a real change). Mitigation: document
  it in the developer guide and the function's Rustdoc, and pin error identity
  with tests.
- Risk: moving the selection rule could change edge behaviour (empty list,
  sparse list, ties).
  Severity: high. Likelihood: low. Mitigation: preserve the exact branch
  structure, add unit and doctest cases for each branch, and keep existing
  session regression tests unchanged.
- Risk: the benchmark's `n = 10,000` case is slow enough to make the before and
  after runs onerous.
  Severity: low. Likelihood: medium. Mitigation: configure short measurement
  and a small sample size, follow the existing bench conventions, and record
  the Criterion settings in this plan.
- Risk: the distance cache persists between the one-thread and eight-thread
  trials, weakening the parallelism exercised by parity tests.
  Severity: low. Likelihood: high. Mitigation: where the test must force cold
  cache misses, wrap the source with a distinct metric descriptor so cache keys
  cannot collide; document why.

## Progress

- [x] (2026-11-17 14:20Z) Reconnaissance: confirmed #378 is fixed by #386
  (`28c27bb`), `search` does not take `insert_mutex`, and established the
  one-thread/eight-thread pool convention from existing tests.
- [x] (2026-11-17 14:30Z) Branch renamed and Lody session titled.
- [ ] `EP-M1`: stage timing, `pipeline` Criterion target, opt-in profile, and
  the recorded serial baseline.
- [ ] `EP-M2`: shared `core_distance_from_neighbours` helper and three
  consumers migrated.
- [ ] `EP-M3`: parallel batch searches with ordered, lowest-index error
  selection and evidence tests.
- [ ] `EP-M4`: after-measurement, documentation, follow-up issue, and delivery.

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
  rejects.
  Evidence: `chutoro-core/src/hnsw/helpers.rs`.
  Impact: a real-search failure fixture can target specific point indices
  without needing to know HNSW traversal order.

## Decision log

- D1. Decision: parallelize the batch stage only, and defer session
  parallelization to a follow-up issue.
  Rationale: the batch stage has a clean read-only seam (`&CpuHnsw`,
  `&DataSource`). The session methods hold `&mut self`, so parallel searches
  must first be separated from the writes that follow; bundling that larger
  refactor would enlarge the risk surface of a performance change. The issue
  explicitly allows either decision provided it is explicit.
  Date/Author: 2026-11-17, implementation agent.
- D2. Decision: collect indexed results into `Vec<Result<T, E>>`, then scan
  serially for the first error.
  Rationale: `collect::<Result<Vec<_>>>()` returns whichever error Rayon
  observes first, which is nondeterministic. A serial scan over an
  order-preserving indexed collect yields the lowest-index error, matching the
  serial loop.
  Date/Author: 2026-11-17, implementation agent.
- D3. Decision: keep stage timing on the orchestration thread and record it
  only when the stage succeeds, matching the existing `record_stage` calls.
  Rationale: per-worker timing would change the histogram's meaning and
  cardinality, and a failed stage never produced a latency sample before.
  Date/Author: 2026-11-17, implementation agent.

## Outcomes & retrospective

Pending implementation.

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
  population and source-call side effects are independent of which index
  failed.
- `L-select`: `core_distance_from_neighbours` implements `C(L, m)` exactly,
  including empty and sparse lists.
- `L-parity`: if every `R(i)` is deterministic, then the parallel output equals
  the serial output elementwise.

### Axioms (trusted, not verified here)

- `AXIOM-rayon-indexed`: `rayon`'s `Vec` collection from an indexed parallel
  iterator preserves input order. Source: rayon's documented `IndexedParallelIterator`
  contract.
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
  `chutoro-core/src/cpu_pipeline/tests/core_distances.rs`. Evidence: the test
  fails if the collector reorders values (for example by using `flat_map` or
  unordered collect). Non-vacuity: fixtures include ties, sparse lists, and a
  singleton; the oracle is a separate sequential loop over `index.search`.
- Obligation `INV-error`.
  Method: synthetic failure test with errors at indices 3, 7, and 11 and
  artificial delays at lower indices, plus a real-search self-pair failure
  fixture, each at one and eight threads. Rationale: the failure mode is
  scheduling-dependent, so it must be driven through real Rayon pools.
  Artefact: same test module. Evidence: the test fails if errors are collected
  through `collect::<Result<Vec<_>>>()`. Non-vacuity: the delayed lower index
  actually fails, and the real fixture is shown to fail at the intended index.
- Obligation `INV-every-index-runs`.
  Method: a counting source that records which query indices were searched in
  the presence of a failure. Rationale: side-effect behaviour is observable and
  cheap to pin. Artefact: same test module. Evidence: all `n` queries are
  recorded even when point 3 fails.
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
and writes one CSV row per stage plus a whole-run row to
`target/chutoro-benchmarks/pipeline_profile.csv` (overridable with
`CHUTORO_BENCH_PIPELINE_PROFILE_PATH`). The row set includes each stage's
seconds and share of the whole run, so the `core_distances` share is explicit.
Without the feature, the profile step is a no-op.

Extend `chutoro-benches/tests/benchmark_smoke.rs` to assert `--list` coverage
and one `--exact` probe for the new target.

Run the serial baseline:

```plaintext
cargo bench -p chutoro-benches --bench pipeline -- --save-baseline before --noplot
```

Record medians for both sizes and the stage profile in `Artefacts and notes`.

### `EP-M2` Shared selection helper

Create `chutoro-core/src/core_distance.rs` with the public
`core_distance_from_neighbours(&[Neighbour], NonZeroUsize) -> f32`, gated on
the `cpu` feature and re-exported from `chutoro-core/src/lib.rs`. Document the
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
it serially to return the first error in index order. Document that every
index runs. Use it in `compute_core_distances` with
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
  saved `before` Criterion baseline. Conformance check: no production
  behaviour change; metric cardinality still bounded; no new runtime
  dependency; no persisted-format change. Recovery: revert the benchmark and
  metrics edits. Remaining gaps: parallel execution.
  Compatibility decision: none.
- `EP-M2` outcome: one shared selection rule; batch output unchanged.
  Requirements: `REQ-dedup`. Acceptance evidence: helper tests plus the
  unchanged session regression tests and pipeline tests. Conformance check:
  the only public addition is the helper; no behaviour drift. Recovery: revert
  the helper and its three consumers together. Remaining gaps: parallel
  execution. Compatibility decision: none (pre-1.0 public API).
- `EP-M3` outcome: the batch stage runs in parallel with deterministic
  ordering and error identity.
  Requirements: `REQ-parity`, `REQ-lowest-error`. Acceptance evidence: the new
  test module at one and eight threads. Conformance check: the `ef` formula and
  self-filter are unchanged; the build and locking are untouched. Recovery:
  revert `compute_core_distances` to the serial loop; the helper remains.
  Remaining gaps: session path. Compatibility decision: none.
- `EP-M4` outcome: measured improvement, documented contract, filed
  follow-up.
  Requirements: `REQ-measure`, `REQ-session-decision`. Acceptance evidence:
  baseline comparison transcript, guide and design text, follow-up issue link.
  Conformance check: documentation matches implementation; no stale claim of a
  serial stage remains. Recovery: documentation-only changes are independent
  commits. Remaining gaps: session parallelization. Compatibility decision:
  none.

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

- Run `cargo nextest run --all-features -p chutoro-core -E 'test(core_distances)'`
  and expect the parity, ordering, and error-identity tests to pass. Before
  `EP-M3`, the same selection already passes through the oracle path; the new
  parallel seam's tests fail until the seam exists.
- Run the metrics test
  `cargo nextest run --all-features -p chutoro-core -E 'test(every_stage_records_a_latency_sample)'`
  and expect three labels exactly.
- Run `cargo bench -p chutoro-benches --bench pipeline -- --baseline before
  --noplot` and expect both benchmark IDs to report a change measurement.
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

Baseline and after measurements: to be recorded during `EP-M1` and `EP-M4`.

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
