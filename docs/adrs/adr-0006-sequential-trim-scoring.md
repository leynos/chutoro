# Architecture Decision Record (ADR) 006: sequential trim scoring under the insertion mutex

## Status

Accepted - 2026-11-14: trim scoring stays inside the insertion critical section
and runs sequentially on the thread that holds `insert_mutex`.

## Date

2026-11-14

## Context and problem statement

`CpuHnsw` parallelizes construction across Rayon workers, and each insertion
serializes its plan-apply-score-commit transaction through `insert_mutex`
(`chutoro-core/src/hnsw/cpu/mod.rs`). After applying a staged insertion, the
worker holding that mutex scored the resulting trim jobs with
`trim_jobs.into_par_iter().map(...).collect::<Result<Vec<_>, _>>()`.

Rayon's parallel `collect` bridges through a latch. The calling thread may park
on that latch instead of draining the work itself, which is safe only while
another worker can steal. Here every other worker was already blocked acquiring
`insert_mutex`, which the parked thread held, so none could steal. The pool
then made no progress at all: `CpuHnsw::build` and `CpuHnsw::build_with_edges`
never returned, and nothing was logged or reported as an error.

The stall was observed on 2026-11-08 while listing benchmark targets.
`cargo nextest list` builds bench binaries eagerly, and the extraction bench
builds a 1,000-point HNSW index at registration, so listing it really does
build an index. One child hung for 25 minutes with 9 CPU ticks across 19
threads: eighteen Rayon workers were parked in `Mutex::lock` inside
`insert_with_collector`, and the thread that should have drained the nested
work was parked in Rayon's latch. `make test-workflow-contracts` hung twice on
the same path. A standalone probe reproduced the stall with a four-worker pool.

The question this decision answers is how to keep the apply-score-commit
transaction correct while removing the nested dispatch that turns ordinary lock
contention into a permanent stall.

## Decision drivers

- The insertion transaction must stay atomic with respect to the graph. The
  `prepared` state produced by `apply` is derived from a graph snapshot and is
  committed later; another insertion must not interleave between the two.
- Correctness must not depend on scheduler timing. A fix that holds only when
  an extra worker happens to be free would leave consumers with a hang that
  reproduces intermittently.
- The parallel build is the point of the CPU backend, so a fix must not remove
  the outer parallelism that motivates it.
- Trim batches are small. Scoring produces at most one job per staged
  neighbour update, so per-insertion scoring is not a throughput bottleneck.

## Options considered

- **Score trim jobs outside the insertion critical section.** Rejected. The
  `prepared` state from `apply` is only valid while no other insertion mutates
  the graph before `commit`. Releasing `insert_mutex` for scoring would let
  another insertion apply and commit against the same graph, so the snapshot
  the commit relies on could no longer be shown valid without revalidation and
  a new ordering protocol. The transaction therefore keeps apply, score, and
  commit under one guard.
- **Dispatch sequentially only when the caller is already inside a Rayon
  worker or the pool is committed to an outer insert.** Rejected. Detecting
  that condition is fragile, and it is also the wrong test: in a parallel build
  the mutex holder usually is a Rayon worker, and the stall depends on whether
  any other worker can steal, not on how the caller entered. The conditional
  form also leaves two scoring paths to keep in step.
- **Restrict the outer parallel insert so it never calls code that spawns
  nested Rayon work, for example by scoring with the executor that drives the
  outer loop.** Rejected. It either removes the outer parallelism the build
  exists for or introduces a second executor boundary for one small nested
  batch, adding complexity without improving correctness.
- **Score the jobs sequentially on the calling thread, unconditionally.**
  Accepted. It removes core-owned nested dispatch from the critical section
  entirely, so no scheduling assumption remains, and the batch is small enough
  that the lost intra-insertion parallelism is negligible.

## Decision outcome

`CpuHnsw::score_trim_jobs` scores its jobs in input order on the calling thread:

```rust
trim_jobs
    .into_iter()
    .map(|job| self.run_trim_job(job, source))
    .collect()
```

The empty-batch early return is retained. Job validation, ranking, and the
`max_connections` cap are unchanged, and results stay in input order. The
critical section still spans apply, score, and commit, so the snapshot the
commit relies on remains valid. The mutex, the staging, the outer build
dispatch, and the RNG selection are otherwise untouched.

Because scoring is now sequential, first-error reporting becomes deterministic:
scoring stops at the first invalid job in input order, whereas the parallel
`collect` returned whichever error it reached first, which is not guaranteed to
be the leftmost in input order. Distance-cache reads and source
`batch_distances` calls (which may delegate to `distance_batch` and `distance`)
still happen at the same point in the transaction, after apply and before
commit under the mutex, but the order between jobs is now input order.

The change also adds two label-free instruments behind the existing optional
`metrics` feature, `chutoro.hnsw.trim.jobs_submitted_total` and
`chutoro.hnsw.trim.duration_seconds`, so the cost of serial scoring inside the
critical section stays observable.

## Architectural rationale

The lock protects a transaction, not just a data structure. Once scoring is
recognized as a step inside that transaction, the question stops being "can
this step be parallel?" and becomes "may anything inside the transaction wait
for work that needs the same lock?". It may not. Sequential scoring states that
rule directly in the code, rather than relying on a dispatcher to notice that
its workers are unavailable.

This extends the existing HNSW scoring invariant: distance evaluation runs
before graph write-lock mutation, and no code inside the insertion critical
section may block on, or dispatch work to, the Rayon pool.

## Known risks and limitations

- **Lost intra-insertion parallelism.** Trim batches no longer fan out. This is
  accepted because batches are small, at most one job per staged neighbour
  update, and the new metrics let a future regression in batch size surface.
- **Arbitrary `DataSource` callbacks are not made safe.** The fix removes
  nested dispatch owned by the core, not dispatch owned by a source
  implementation. A `DataSource` whose `distance`, `distance_batch`,
  `batch_distances`, or `metric_descriptor` blocks on, or dispatches to, the
  same Rayon pool while called under `insert_mutex` can still deadlock; trim
  batch construction reads the metric descriptor inside the critical section.
  The trait documentation now states this, together with the prohibition on
  re-entering the same index.
- **The deterministic guard reads source text.**
  `tests/workflow_contracts/trim_scoring_dispatch_test.py` pins the five
  functions on the scoring path — `insert_with_collector` in `cpu/mod.rs`;
  `score_trim_jobs`, `score_trim_jobs_with_metrics`, and `run_trim_job` in
  `cpu/trim.rs`; and `batch_distances_for_trim` in `hnsw/helpers.rs` — and
  rejects every recognized Rayon entry point in any of their bodies, including
  the `into_par_iter()`, `par_iter_mut()`, and `par_iter()` adapters, the
  parallel slice and string adapters (`.par_bridge()`, `.par_chunks()`,
  `.par_rchunks()`, `.par_windows()`, `.par_split()`, `.par_lines()`,
  `.par_chars()`, and `.par_drain()`, each standing for its mutable and
  `_exact` forms as well) and the `spawn_fifo`, `scope_fifo`, `join_context`,
  `broadcast`, and `spawn_broadcast` calls with their `ThreadPool` method forms
  such as `pool.scope(...)` and `scope.spawn(...)`. The generic
  `pool.install::<_, _>(...)` is recognized through its turbofish, and
  `pool.join(...)` because it takes two arguments where a slice or path join
  takes one. The join is read from its argument list rather than from what the
  first argument looks like: a top-level comma with a value after it means two
  arguments, so an inline closure, a bound callable such as
  `pool.join(left, right)`, and a call that builds one such as
  `pool.join(make_job(), || ())` are all caught, while a trailing comma leaves
  `names.join(separator,)` a slice join. Each remaining entry is a
  boundary-aware pattern, so a slice `names.join(", ")` is not read as a Rayon
  join. The synthetic sources these readings are checked against live in
  `trim_scoring_samples.py`, beside the reader rather than inside it. Dispatch
  in a listed helper is therefore caught; dispatch it cannot see still evades
  it: an unlisted helper called by one of the five, an aliased import such as
  `use rayon::spawn as run; run(...)`, or a `DataSource` that dispatches on its
  own account. The scan fails closed when a pinned function is missing, so a
  rename or a move cannot read as clean. The runtime scheduling tests in
  `hnsw::cpu::trim_tests` remain as broader coverage, and they may catch
  unlisted helper dispatch, but they are observations rather than proofs: the
  caller's first distance callback waits up to two seconds for another worker,
  and if no worker steals in that interval a dispatching implementation can
  still finish the batch on the caller and pass.
- **A watchdog stress test was considered and rejected.** An earlier
  watchdog-based stress test passed three times out of three on the unfixed
  code, because the stall is a scheduling race rather than a deterministic
  failure.
- **Broader scheduling changes need stronger evidence than this ADR's.**
  Neither the lexical scan nor the bounded runtime observations establish
  deadlock freedom, and none of them should be extended into a structural
  harness that claims to. A future change to scheduling beyond this critical
  section — a narrower lock, a second mutex, work stealing across insertions,
  or any path that may wait while holding a graph or cache lock — needs a
  bounded scheduler or lock model that covers the new interaction *and*
  production-path regressions. The concurrency limitations in
  [ADR-0002](adr-0002-adoption-of-kani-formal-verification.md) explain why the
  existing Kani harnesses do not model lock waits or deadlock freedom and
  cannot be stretched to cover this.

## Consequences

- Parallel builds of graphs large enough to produce trim jobs return instead of
  hanging. `cargo nextest list --all-targets`, and any gate that lists or runs
  benchmark binaries, no longer inherit the stall.
- Trim scoring is deterministic in input order, including first-error
  selection.
- Two new operational metrics describe the serial scoring cost.
- A future change that reintroduces one of the recognized Rayon entry-point
  markers into any of the five pinned functions fails the source contract
  deterministically, regardless of scheduler timing. Dispatch reached through
  an unlisted helper, an alias, or a source implementation is caught, if at
  all, only by the runtime observation.

## Validation evidence

The fix merged as PR #386 on 2026-11-14, in merge commit
`28c27bb9fa22599b68189655f42b393b129720ff`.

The checked-in evidence now has three parts.

- **Historical fixture.** The workflow contract carries a verbatim copy of the
  former `score_trim_jobs` from commit
  `56d6909c7019ac7caa6d485d11ebf5708b47528e`, `cpu/trim.rs` lines 35-48, and
  asserts that it reads as `into_par_iter` while the current scorer reads with
  no marker. The fixture is checked in rather than read from Git history, so a
  shallow checkout does not retire the evidence, and it cannot drift into
  agreement with the fixed code.
- **Direct runtime regression.**
  `direct_trim_scoring_stays_on_the_calling_worker` scores a synthetic batch in
  a dedicated four-worker pool and records the thread, the Rayon worker, and an
  `insert_mutex` probe for every distance callback. Every callback must come
  from the caller.
- **Insertion-path runtime regression.**
  `insertion_trim_scoring_runs_on_the_insert_mutex_holder` seeds an index with
  twelve nodes at `M = 4`, `ef_construction = 8`, and a one-entry distance
  cache, then inserts the thirteenth through `insert_with_collector` in a
  dedicated four-worker pool. A `WANTS_EDGES = true` collector marks the end of
  planning; every later distance callback must report the same thread, the same
  Rayon worker, and `WouldBlock` from a non-blocking `try_lock`. The one-entry
  cache forces at least two trim jobs to reach the source, so the regression
  observes real scoring rather than an empty batch. It shares the two-second
  steal window with the direct test.

`make test-scheduling-contract` runs one nextest invocation per regression,
each with a fully qualified exact-match filter, `--no-tests=fail`, and default
features, and the `build-test` CI job runs that target unconditionally beside
`make test-workflow-contracts`.
`tests/workflow_contracts/scheduling_contract_test.py` and
`scheduling_ci_test.py`, reading through `scheduling_reading.py`,
`rust_declaration_reading.py`, and `workflow_support.py`, pin the target
recipe, one invocation per fully qualified test name, each invocation's whole
filter expression as exactly one selector, the zero-test failure flag on each,
the absence of every feature-selection flag (`--all-features`,
`--no-default-features`, `--features` and its `=` form, and `-F`), the absence
of `--no-run` (which compiles the selected tests without executing them), the
unconditional and blocking execution of the step and its nextest installer
within an unconditional job, the order that installs `cargo-nextest` before the
target runs, and that the installer appears as a command in its own right
rather than as an argument to `echo` or inside a comment, with no operator that
would replace its status — neither directly after it (`|| true`, `; true`) nor
further along the same command line (`installer && echo installed || true`) —
and no interpreter mode that only parses it (`bash -n`). It also requires the
recipe's commands to run nextest in executable position without Make's
error-ignoring `-` prefix or a failure-masking shell suffix such as `|| true`.
Commands are tokenized with shell separators rather than whitespace alone, so
an unspaced `||true` reads as the operator it is. Finally, it checks that each
named regression is still declared under `cpu/trim_tests` with a test
attribute, since an undiscovered function cannot trip `--no-tests=fail`, and
that a declaration quoted inside a string literal does not stand in for one.

Positive evidence, on commit `9df7179` on 2026-11-20: both runtime regressions
pass, one nextest invocation each, 1 passed with 674 skipped apiece, and
`make check-fmt`, `make markdownlint` including spelling, `make lint`
(interrogate at 100% documented, rustdoc, Clippy and Whitaker),
`make typecheck`, `make nixie`, and `make test-workflow-contracts` (932 passed,
1 skipped) all pass. `make test` passed 1481 nextest tests with 1 skipped and
141 doctests with 24 ignored on head `447a824`; that is the last head to change
a Rust source, and the commits after it touch only these contracts and the
documentation this record describes.

Negative control (2026-11-19, scratch worktree detached at the branch head):
restoring `trim_jobs.into_par_iter()` in `score_trim_jobs` produced both
expected outcomes.

- The Python contract failed deterministically. Run from
  `tests/workflow_contracts` as `pytest -q trim_scoring_dispatch_test.py`, it
  collected 34 cases: two failed and thirty-two passed. The scan's failure named
  `('chutoro-core/src/hnsw/cpu/trim.rs', 'score_trim_jobs'): 'into_par_iter'`,
  and the historical-fixture test reported
  `the current scorer must score in place, found 'into_par_iter'`.
- An earlier run of the same mutation failed both runtime regressions: the
  direct test observed callbacks on four worker indices instead of the caller
  alone, and the insertion-path test observed a post-planning callback on a
  different thread from the `insert_mutex` holder. They failed in 0.024 s and
  0.009 s respectively because the dedicated pool had an idle worker to accept
  the steal offer.

The runtime outcome is scheduler-dependent by construction — a pool with no
idle worker can let the mutated caller finish the batch itself — which is why
the Python contract, not the runtime regressions, is the guarantee. The scratch
mutation was discarded and the candidate worktree was never changed.

## References

- [Deadlock report (#378)](https://github.com/leynos/chutoro/issues/378)
- [Fix (#386)](https://github.com/leynos/chutoro/pull/386)
- [Scheduling-contract assurance follow-up (#402)](https://github.com/leynos/chutoro/issues/402)
