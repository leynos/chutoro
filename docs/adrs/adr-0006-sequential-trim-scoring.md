# Architecture Decision Record (ADR) 006: sequential trim scoring under the insertion mutex

## Status

Accepted - 2026-11-14: trim scoring stays inside the insertion critical section
and runs sequentially on the thread that holds `insert_mutex`.

## Date

2026-11-14

## Context and problem statement

`CpuHnsw` parallelises construction across Rayon workers, and each insertion
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
  implementation. A `DataSource` whose `distance`, `distance_batch`, or
  `batch_distances` blocks on, or dispatches to, the same Rayon pool while
  called under `insert_mutex` can still deadlock. The trait documentation now
  states this, together with the prohibition on re-entering the same index.
- **The deterministic guard reads source text.**
  `tests/workflow_contracts/trim_scoring_dispatch_test.py` rejects every Rayon
  entry point in `score_trim_jobs`, but it cannot see dispatch reached through
  a helper. The runtime scheduling test in `hnsw::cpu::trim_tests` remains as
  broader coverage, and it is an observation rather than a proof: if no worker
  steals before the observation interval expires, a dispatching implementation
  could pass a single round by coincidence, which is why the test scores eight
  rounds with a fresh index per round.
- **A watchdog stress test was considered and rejected.** An earlier
  watchdog-based stress test passed three times out of three on the unfixed
  code, because the stall is a scheduling race rather than a deterministic
  failure.

## Consequences

- Parallel builds of graphs large enough to produce trim jobs return instead of
  hanging. `cargo nextest list --all-targets`, and any gate that lists or runs
  benchmark binaries, no longer inherit the stall.
- Trim scoring is deterministic in input order, including first-error
  selection.
- Two new operational metrics describe the serial scoring cost.
- A future change that reintroduces nested dispatch into the scorer fails the
  deterministic contract, whatever the scheduler does.

## Validation evidence

The fix merged as PR #386 on 2026-11-14.

Negative control: against a temporary `into_par_iter()` implementation, the
runtime test failed in 0.03 s having observed all four workers, and the source
contract rejects the dispatch deterministically.

Positive evidence: on the fixed tree the runtime test passes in about 0.3 s
over eight rounds, and the full gate set passed on commit `28a9687`
(`make check-fmt`, `make markdownlint` including spelling, `make lint`,
`make typecheck`, `make test` with 1374 passed and 1 skipped, `make nixie`, and
`make test-workflow-contracts` with 843 passed and 1 skipped). Hosted CI on the
same head was green.

## References

- [Deadlock report (#378)](https://github.com/leynos/chutoro/issues/378)
- [Fix (#386)](https://github.com/leynos/chutoro/pull/386)
- [Scheduling-contract assurance follow-up (#402)](https://github.com/leynos/chutoro/issues/402)
