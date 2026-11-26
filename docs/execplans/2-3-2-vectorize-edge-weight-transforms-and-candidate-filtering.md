# Vectorize edge-weight transforms and candidate filtering (2.3.2)

This ExecPlan (execution plan) is a living document. The sections `Constraints`,
`Tolerances`, `Risks`, `Progress`, `Surprises & discoveries`, `Decision log`,
and `Outcomes & retrospective` must be kept up to date as work proceeds.

Status: IN PROGRESS (revision 10, 2026-11-26); implementation and record are
complete, with one planned gate pending. The maintainer approved the plan and
authorized production work for Milestones 0 to 2. Gate A was measured and
fails: the decision-size ratio is 0.16% to 0.69% at head `7368d8d` and 0.0512%
to 0.3024% at head `4e96416`, far below the pre-registered 5%, so Milestones 3
to 6 are declined and the roadmap item closes as a measured null result.
Milestones 0 to 2 and 7 are delivered. **Pending:** the literal local
`make test` has no passing run on this host, for the infrastructure reason
recorded under "Gate disposition for the literal `make test`". That gate is
pending, not waived, and no equivalence is claimed to have been approved.

The literal local `make test` was not produced, and this record does not claim
it. Its setup script expires under this host's saturated compile queue, which
the developers' guide classes as a budget rather than a defect. Every test it
would run passed in partitions, and the hosted `build-test` suite that runs the
same nextest suite plus the doctest step is green on the published head. "Gate
disposition for the literal `make test`" under Validation and acceptance
records the detail and the one step a reader needs if they require the literal
command.

Roadmap item: 2.3.2 (Phase 2, "Hot-path optimizations"), `docs/roadmap.md`
lines 400-402. Design: `docs/chutoro-design.md` §6.3 "SIMD utilization".
Tracking issue: [#388](https://github.com/leynos/chutoro/issues/388). Draft
pull request: [#198](https://github.com/leynos/chutoro/pull/198).

## Source baseline for this revision

| Item                                | Value                                                                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Working branch                      | `2-3-2-vectorize-edge-weight-transforms-and-candidate-filtering`                                                             |
| Branch head before this revision    | `6f2f0c9` (three plan commits rebased onto `main`; not yet pushed)                                                           |
| Uncommitted work found at start     | Revision 3 of this file only; superseded by this revision                                                                    |
| Reviewed `main`                     | `051d395c87366628b03c451a0157933ed870f1ba` (2026-11-22)                                                                      |
| Reference `main` named by the brief | `5f13f07e10688ddb0b33e3fb2ef5afbc0e285013`, the parent of `051d395`                                                          |
| Proposed design inspected           | PR #453 head `16042ce0bbc4e9e0bd88ebdea9c2a9d7286a2238`, branch `docs/fearless-simd-adoption-2026-11-22`, based on `5f13f07` |
| Library contract inspected          | Fearless SIMD 1.2.0 changelog and `SimdBase` documentation                                                                   |

_Table 1: Commits and documents this revision was reconciled against._

The only commit between `5f13f07` and `051d395` is PR #432, which changes the
HNSW critical section and touches no file this plan edits. All source
references below are to `051d395` unless stated otherwise.

## Purpose / big picture

Chutoro builds an approximate nearest-neighbour graph, converts the harvested
candidate edges into mutual-reachability weights, builds a minimum spanning
forest (MSF) with a parallel Kruskal implementation, and extracts a cluster
hierarchy. Roadmap item 2.3.2 targets the work between harvesting and the
union-find: the weighting transform and the validation, filtering and
compaction that prepare edges for Kruskal.

This plan decides that work by measurement. It has three possible outcomes, all
of which are successful completions:

1. **Null result.** The current pipeline gives the targeted stages less than 5%
   of pipeline time. The expensive milestones are declined with the number that
   declined them, the remaining correctness guards land, and the roadmap item
   closes with a recorded null result. The historical evidence suggests this is
   the likeliest outcome, but it has not been measured on the current tree.
2. **Scalar improvement.** The gate passes and a scalar change, such as a packed
   sort record, achieves the measured win. No SIMD dependency is added.
3. **Fearless SIMD kernel.** Both gates pass, assembly inspection shows the
   scalar loop is not already effectively vectorized, and a private Fearless
   SIMD kernel produces a resolvable improvement with exact semantic parity.

After this plan completes, a developer can observe:

- A recorded, reproducible Gate A measurement on the current tree, with its
  noise band, in `Outcomes & retrospective` and `docs/chutoro-design.md` §6.3.
- `cargo nextest run -p chutoro-core -E 'test(/mst::/)'` passes, including a
  property comparing the parallel forest's **exact edge list** with the
  sequential oracle's.
- `chutoro-core/tests/features/mst_edge_preparation.feature` passes under
  `rstest-bdd`.
- The mutual-reachability transform reports its lowest-index lookup error at
  any thread count.
- Either roadmap item 2.3.2 is ticked with a recorded null result, or the gated
  milestone that shipped is documented with its evidence.

### Relationship to roadmap task 2.2.8

PR #453 proposes roadmap task 2.2.8: a bounded Fearless SIMD port of the dense
provider's existing query-to-points kernel. The two tasks share a dependency
and an architectural boundary and nothing else:

|                     | 2.2.8 dense port                                                 | 2.3.2 MST preparation                                          |
| ------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------- |
| Justification       | Less duplication and unsafe code, without an end-to-end speed-up | A measured performance opportunity                             |
| Entry gate          | RFC 0001 regression tolerance and net maintenance reduction      | Gate A (5% of pipeline) and, for SIMD, Gate B (10% of Kruskal) |
| Crate               | `chutoro-providers-dense`                                        | `chutoro-core`, behind `cpu`                                   |
| Completes the other | No                                                               | No                                                             |

_Table 2: The two Fearless SIMD tasks are independent._

Neither task is a prerequisite for the other. This plan does not create a
shared application-level SIMD framework, wrapper crate or public port, and it
does not mark 2.2.8 complete or in progress.

## Inventory

| Area                                                              | State     | Evidence or next action                                                          |
| ----------------------------------------------------------------- | --------- | -------------------------------------------------------------------------------- |
| Parallel mutual-reachability transform                            | Done      | `cpu_pipeline/mod.rs:125-135`, PR #276 (#204)                                    |
| Redundant `EdgeHarvest::new` sort removed                         | Done      | `cpu_pipeline/mod.rs:103-111` passes the transformed `Vec` directly; PR #276     |
| Slice entry point `parallel_kruskal_from_edges(&[CandidateEdge])` | Done      | `mst/mod.rs:324-327`; `EdgeHarvest::as_slice` at `hnsw/types.rs:340`             |
| Ordered validation buffer, lowest-index MST error                 | Done      | `mst/mod.rs:296-321`, PR #428 (#206)                                             |
| Allocation-free weight-group extension                            | Done      | `extend_with_accepted_edges`, `mst/mod.rs:255-267`, PR #279 (#205)               |
| Parallel core-distance searches                                   | Done      | `compute_core_distances`, `cpu_pipeline/mod.rs:221-239`, PR #431 (#389)          |
| Two-pool determinism and concurrent union-find tests              | Done      | `mst/property/concurrency.rs:101`, `mst/union_find.rs:225-260`, PR #225 (#201)   |
| Non-exhaustive error-code enums                                   | Done      | `define_error_codes!` in `mst/error.rs:52-66`, PR #427 (#203)                    |
| Users' guide MST error codes                                      | Done      | `docs/users-guide.md:531-564`, PR #434 (#207)                                    |
| Pipeline benchmark and stage profile                              | Done      | `chutoro-benches/benches/pipeline.rs`, `pipeline-profile` feature, PR #431       |
| Gate A measurements                                               | Done      | Milestone 1, first measurement: 0.1640%-0.6895% at n = 10 000, so the gate fails |
| Assembly inspection of scalar loops                               | Done      | Milestone 1: scalar `maxss`/`vmaxss`, no packed float max at either codegen      |
| Exact-edge-list oracle property                                   | Done      | Milestone 2; mutation-checked against `MstEdge::Ord`                             |
| BDD scenarios for edge preparation                                | Done      | Milestone 2: six scenarios, 6/6 passing                                          |
| Deterministic transform lookup errors                             | Done      | Milestone 2: observed red at eight threads, then green in both pools             |
| Dedup-before-sort                                                 | Declined  | Milestone 2: bit-keyed dedup would change the `±0.0` survivor; see decision log  |
| Gate B sub-stage instrument                                       | Declined  | Gate A failed: the ratio is below 5%, so the milestone is not entered            |
| Packed sort record                                                | Declined  | Gate A failed: the sort-only lever cannot justify itself                         |
| Candidate pre-filter (Filter-Kruskal)                             | Declined  | Gate A failed: the union-find walk is inside a sub-1% stage                      |
| Fearless SIMD kernel                                              | Declined  | Gate A failed: no SIMD dependency is added                                       |
| `fearless_simd = "1.2"`, `fearless_simd_macros = "0.1"`           | Not added | The Milestone 6 gates were not reached, so the exemption is unused               |
| Null-result record and roadmap closure                            | Done      | Milestone 7: design §6.3 update and roadmap 2.3.2 ticked                         |

_Table 3: Completed, outstanding, gated and remaining work. Paths are relative
to `chutoro-core/src/` unless shown otherwise._

## Governing references

Read these before editing anything:

- `AGENTS.md`, `docs/contents.md`, `docs/documentation-style-guide.md`.
- `docs/chutoro-design.md` §6.2 (parallel Kruskal sketch) and §6.3 (SIMD
  utilization, with its dated implementation updates).
- `docs/adrs/adr-0003-soa-prefetch-adapter-boundary.md`, the precedent for
  gating structural change on measurement and recording null results in place.
- `docs/adrs/adr-0006-sequential-trim-scoring.md`, the HNSW callback contract
  behind Constraint 10.
- `docs/developers-guide.md` "Whole-pipeline and stage-profile benchmarks"
  (line 2725) and "Benchmark regression workflow" (line 2684).
- `docs/users-guide.md` line 150 (the fixed `stage` label set) and "MST
  errors" (line 531).

Proposed, under review in PR #453 (read from `16042ce` until merged):

- ADR 0007, "Fearless SIMD kernel boundary"
  (`docs/adrs/adr-0007-fearless-simd-kernel-boundary.md`). Status: **Proposed**.
- RFC 0001, "Fearless SIMD query-to-points"
  (`docs/rfcs/rfc-0001-fearless-simd-query-to-points.md`), which governs 2.2.8
  and is cited here only for its token, dispatch and MSRV findings.
- The PR's §6.3 changes, including the new §6.3.1 "Fearless SIMD kernel
  boundary", and its roadmap entries for 2.2.8 and 2.3.2.

The architecture in ADR 0007 is a proposal under review. The **dependency
authorization is already effective**: the maintainer gave it directly, and it
does not depend on PR #453 merging. See the decision log.

Library contract: Fearless SIMD 1.2.0 (released 2026-10-25, MSRV 1.89). The
facts this plan relies on, from its changelog and API documentation:

- 1.1.0 added the classification masks `is_nan`, `is_infinite`, `is_finite`,
  `is_sign_positive` and `is_sign_negative`; order-preserving `compress`; and
  `MaskWiden`/`MaskNarrow`.
- 1.2.0 added `SimdFloat::to_bits` and `from_bits`.
- `max` is implementation-defined when either operand is NaN, and for mixed
  signed zeros.
- `max_precise` returns the non-NaN operand when exactly one operand is a
  quiet NaN. Its mixed signed-zero result is implementation-defined. With a
  signalling NaN operand it is **fully non-deterministic**.
- `from_slice` and `store_slice` require a slice of exactly the vector length.
- The `Avx2` token denotes the x86-64-v3 feature bundle, and `Avx512` an Ice
  Lake bundle. `Level::baseline()` returns `Sse2` on x86-64, so a baseline
  token is already SIMD.
- `assume_supported()` is the renamed unchecked constructor. This plan forbids
  it.

Skills: `leta` first, then `rust-router`, `rust-unit-testing`, `proptest`,
`nextest`, `execplans`, `commit-message`, `pr-creation`, `en-gb-oxendict`. Load
`rust-performance-and-layout` for Milestones 1, 4 and 6. Load `kani` for
Milestone 6. Load `verus` and `hexagonal-architecture` only if Milestone 5
proceeds.

## Constraints

These are hard invariants. Violating one requires escalation, not a workaround.

1. **Output equivalence.** For every input on which `parallel_kruskal` returns
   `Ok` today, it must return an identical `MinimumSpanningForest`: the same
   edge sequence, each edge's `sequence`, and the same `component_count`.
   Weight identity means `f32::to_bits` equality, not `==`.
2. **No behavioural divergence**, with one sanctioned exception. Error
   selection in the mutual-reachability transform may narrow from scheduling
   dependent to lowest input index (Milestone 2). That is a strengthening of an
   unspecified behaviour. Any other divergence is an escalation.
3. **Determinism.** Identical input produces identical output at any Rayon
   thread count. This is enforced on `main` by
   `run_thread_pool_determinism_property` (`mst/property/concurrency.rs:101`),
   the error-selection tests in `mst/tests/error_selection.rs`, and
   `concurrent_unions_match_sequential_partition` (`mst/union_find.rs:225`).
   Nothing in this plan may weaken them.
4. **No `unsafe`.** A Fearless SIMD kernel must use safe tokens and safe vector
   operations only: no `assume_supported`, no `kernel!` intrinsic wrappers, and
   no `core::arch` intrinsics.
5. **Public API: no changes.** `parallel_kruskal`, `MstEdge`,
   `MinimumSpanningForest`, `CandidateEdge`, `EdgeHarvest` and the `cpu`,
   `metrics`, `skeleton` and `gpu` features keep their signatures and
   semantics. No Fearless token, vector or mask appears in any public signature.
   `MstEdge` must not gain a field.
6. **MSRV 1.89.0** (`Cargo.toml:19`, `chutoro-core/Cargo.toml:5`). The
   resolved dependency graph of any change must build on Rust 1.89. The
   development toolchain is 1.99.1 (`rust-toolchain.toml`).
7. **No new dependency, with one narrow exemption.** Milestone 6 may add
   `fearless_simd = "1.2"` and, if used, `fearless_simd_macros = "0.1"`, as
   caret requirements declared in `[workspace.dependencies]` and consumed only
   by `chutoro-core` as optional dependencies activated by `cpu`. No other
   dependency, in any crate, is authorized.
8. **File-size cap:** no source file may exceed 400 lines.
   `chutoro-core/src/mst/mod.rs` has 379 lines and
   `chutoro-core/src/cpu_pipeline/mod.rs` has 307.
   `chutoro-core/src/mst/union_find.rs` has exactly 400; any edit there must
   first move its test module (from line 206) out.
9. **Dependency direction:** `chutoro-core` must not depend on
   `chutoro-providers-dense`, and must not adopt `DensePointView` or any other
   dense packing type.
10. **Do not modify** `chutoro-core/src/hnsw/**`,
    `chutoro-core/src/hierarchy/**` or `chutoro-providers/dense/**`. The HNSW
    critical section carries ADR-0006's scheduling contract (PRs #386, #429 and
    #432), and the dense provider is 2.2.8's territory.
11. **Union-find stays scalar.** Its atomic access, lock ordering and dependent
    traversals (`mst/union_find.rs`) are outside every kernel. Portable vectors
    do not justify concurrent non-atomic reads.

## Tolerances (exception triggers)

1. **Scope.** Milestone 2 exceeding 8 changed files or 400 net lines. Any gated
   milestone exceeding 12 files or 600 net lines.
2. **Interface.** Any change to an existing public signature, or any new public
   item.
3. **Dependencies.** Any manifest entry other than the two authorized in
   Constraint 7, or either authorized entry added outside Milestone 6, or added
   before its gates pass.
4. **Iterations.** A gate failing after four consecutive fix attempts.
5. **Evidence.** A gated milestone failing its pre-registered threshold: stop,
   record the null result beside the roadmap item, and continue with the
   applicable remaining work. An inconclusive measurement counts as not passing.
6. **Ambiguity.** Any place where this plan and the code disagree about current
   behaviour. Revision 1 tripped this five times, and revisions 2 and 3 were
   stale in more than ten places. See Surprises.
7. **Verification cost.** A Kani harness over fifteen minutes, or a Verus
   proof over five minutes.
8. **Time.** More than four hours on a single milestone.

## Measurement gates

Both gates are pre-registered and must not be weakened. Their definitions here
resolve the ambiguities that revisions 2 and 3 left open.

### Gate A: current-pipeline opportunity (≥ 5%)

```plaintext
Gate A ratio = (T_mutual_reachability + T_kruskal) / T_whole_run
```

- `T_mutual_reachability` is the `mutual_reachability` stage
  (`cpu_pipeline/mod.rs:103-108`): the parallel transform, including the
  indexed core-distance reads, output allocation and error collection.
- `T_kruskal` is the `kruskal` stage (`cpu_pipeline/mod.rs:110-113`): one whole
  `parallel_kruskal_from_edges` call. That includes `prepare_edge_list` (its
  validation, compaction, sort and dedup), the union-find walk and the final
  forest sort. Preparation is nested inside it and must not be added again.
- `T_whole_run` is the `whole_run` row the profile writes: one `Chutoro::run`
  call on the decision workload.
- `harvest_resort` is **removed** from the numerator. The redundant sort it
  timed no longer exists: `build_mutual_reachability_forest` passes the
  transformed `Vec` straight to `parallel_kruskal_from_edges`
  (`cpu_pipeline/mod.rs:103-111`), and no `EdgeHarvest::new` call remains in
  `cpu_pipeline/`. Do not reintroduce the sort to benchmark it.
- **Passes** only if the lowest ratio across the repeated runs is at least 5%.
  **Fails** if the highest ratio is below 5%. Anything else is inconclusive and
  does not pass.

Measured twice, because one production change landed between the runs.

- **First measurement, 2026-11-23 at head `7368d8d`** (the approval commit),
  with only test-only files dirty. No production library commit existed on the
  branch yet, so the measured library was `051d395`'s. Five four-thread runs
  gave 0.1640%, 0.1812%, 0.2345%, 0.5458% and 0.6895% at n = 10 000.
- **Re-measurement, 2026-11-25 at head `4e96416`**, after Milestone 2 changed
  `mutual_reachability_edges` to collect through `collect_ordered_results`.
  That change _is_ a production change to a timed stage: it stages the per-edge
  results in an intermediate buffer and folds it serially, where the previous
  code collected straight into a `Result`. It cannot alter accepted output, but
  it can alter `T_mutual_reachability`, so the first result is not transferred
  to the later head. Five four-thread runs on the same workload give 0.0512%,
  0.0769%, 0.1063%, 0.1417% and 0.3024% at n = 10 000 — minimum 0.0512%, median
  0.1063%, maximum 0.3024% — and 0.1518% to 0.3480% at n = 1000. The gate fails
  on the later head by a wider margin than on the earlier one, so the decline
  does not depend on which measurement is used.

Both measurement sets are recorded in `Outcomes & retrospective` and in
`docs/chutoro-design.md` §6.3.

### Gate B: classification and compaction share of Kruskal (≥ 10%)

```plaintext
Gate B ratio = T_classify_compact / T_kruskal
```

- `T_classify_compact` runs from entry to `prepare_edge_list` up to, but not
  including, `edge_list.par_sort_unstable()` (`mst/mod.rs:296-311`; the sort is
  at line 312). It covers the parallel `validate_and_canonicalize_edge` map
  into the ordered `validated` buffer, both `Vec` allocations, and the
  sequential scan that emits accepted `MstEdge` records and stops at the first
  error.
- Gate B excludes the sort, the dedup, `ConcurrentUnionFind::new`, the
  weight-group walk, the final `forest_edges.sort_unstable()` and the
  `EmptyGraph` check. It excludes the mutual-reachability transform, which sits
  outside Kruskal.
- It is measured on the tree as it stands after Milestones 2 and 4 (if 4
  shipped), on the Gate A decision workload, using the same pass/fail rule.

### Decision workload and method

| Parameter            | Value                                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Decision size        | `n = 10_000`; `n = 1_000` reported separately and never substituted                                                         |
| Source               | `SyntheticSource::generate`, uniform `[0, 1)` coordinates                                                                   |
| Dimensions, seed     | 16, 42 (`pipeline.rs` `DIMENSIONS`, `SEED`)                                                                                 |
| HNSW                 | `M = 16`, `ef_construction = 32`, RNG seed 42                                                                               |
| Clustering           | `min_cluster_size = 5`, `ExecutionStrategy::CpuOnly`                                                                        |
| Harvested edge count | Record it: it is unmeasured. Revision 2's "about `16n`" is an estimate                                                      |
| Build                | Cargo `bench` profile; `RUSTFLAGS` from `.cargo/config.toml` only, so no `target-cpu` and an x86-64 baseline (SSE2) codegen |
| Toolchain            | 1.99.1 for measurement; 1.89 for MSRV checks                                                                                |
| Hardware             | AMD Ryzen 9 3900 (Zen 2), 12 cores, 24 threads, AVX2 and FMA, no AVX-512                                                    |
| Threads              | Decision: `RAYON_NUM_THREADS=4` under `taskset -c 0-3`. Supplementary: 1 thread (matches CI) and 8                          |
| Backend              | Scalar (no SIMD kernel exists yet); every later run records the selected backend                                            |
| Baseline commit      | `051d395` plus this branch's documentation commits, recorded by SHA at run time                                             |

_Table 4: The pre-registered decision workload._

The #389 ExecPlan records its profile host as "6-core Linux". That does not
match the host described above. Treat its numbers as host-unverified history.

Method:

- Gate A uses the existing instrument unchanged: the `pipeline` Criterion
  target with `--features pipeline-profile`. The profile performs one untimed
  run per size. Repeat the profiled invocation five times, writing each run to
  its own `CHUTORO_BENCH_PIPELINE_PROFILE_PATH`, and report the minimum, median
  and maximum Gate A ratio at each size.
- The noise band is the spread across those runs, together with the Criterion
  interval for `pipeline/n=10000`. Report both. Revision 2 found that the `mst`
  micro-benchmark's `local-reference` comparison reports ±12-17% "regressions"
  on unchanged code, so stored-baseline comparisons are not evidence.
- Candidate comparisons in Milestones 4 and 6 are **interleaved**: build both
  candidate and baseline, alternate them in one session with
  `--save-baseline`/`--baseline` pairs, run at least three alternations, and
  use at least 100 samples and a 10-second measurement window for any group
  where a win is claimed.
- Criterion is the primary signal. `hyperfine` over `chutoro-cli` is
  corroboration only, and only where the expected effect is larger than the
  whole-binary noise. At the historical shares it is not, and the report must
  say so rather than run it for show.
- Union-find state is reinitialized for every timed trial. The pipeline and
  `parallel_kruskal` paths already construct a fresh `ConcurrentUnionFind` per
  call, inside the measured region. Any new micro-benchmark of the walk must use
  `iter_batched` with fresh state and state whether its setup is timed.

## Risks

- Risk: the targeted stages are too small to matter. Severity: high.
  Likelihood: high on historical evidence, but unmeasured on the current tree.
  Mitigation: Gate A decides it before any expensive work. Use the history
  below to set expectations, not to make the decision.

- Risk: the scalar loops are already effectively autovectorized, or would be
  with a target-specialized build. Severity: medium. Likelihood: unknown.
  Mitigation: Milestone 1 inspects the assembly before any kernel is proposed.

- Risk: a SIMD `max` silently changes accepted output. Severity: high.
  Likelihood: high without explicit handling. `max_precise` is fully
  non-deterministic on signalling NaNs and implementation-defined on mixed
  signed zeros, while the scalar reference treats every NaN alike. Mitigation:
  the exceptional-lane rule in "Semantic oracle".

- Risk: SIMD masks let a later error win, or let a padding lane create an error
  or an edge. Severity: high. Likelihood: medium. Mitigation: scalar ordered
  emission, no padding, and the tail and lane-boundary tests in Milestone 6.

- Risk: endpoint narrowing silently accepts out-of-range edges.
  `CandidateEdge::new(1 << 33, 1, 0.5, 0)` truncates to `0` under a `u32` cast.
  Severity: high. Likelihood: medium if narrowing is attempted. Mitigation:
  compare full-width 64-bit identifiers, and fall back to scalar on targets
  where `usize` is not 64-bit.

- Risk: a later compatible 1.x release of Fearless SIMD raises its MSRV.
  Severity: medium. Likelihood: medium over time. Mitigation: Milestone 6
  checks the _resolved_ graph with Rust 1.89. The caret requirement alone
  guarantees nothing.

- Risk: multiversioning increases compile time and binary size. Severity: low
  to medium. Mitigation: Milestone 6 records both, and the macro crate is added
  only if needed.

- Risk: no hardware here executes AVX-512 or AArch64 NEON. Severity: medium.
  Likelihood: certain. Mitigation: those backends stay disabled until executed
  differential evidence exists; see "Backend enablement".

- Risk: new private items fail lint. `chutoro-core` inherits the workspace
  table, including `missing_docs_in_private_items = "deny"` and
  `cast_possible_truncation = "deny"` (`Cargo.toml:75`, `:91`). Mitigation:
  document every item and lint workspace-wide. Running `cargo clippy -p` on a
  subset can report false `missing_const_for_fn` failures.

## Progress

Timestamps are mandatory (Tolerance 8).

- [x] (2026-08-16) Revision 1 drafted; six-lens design review completed.
- [x] (2026-08-16) Measured `parallel_kruskal` at n = 100, 500 and 1000;
      verified `weight_key` against `f32::total_cmp` over 14 400 pairs.
- [x] (2026-08-16) Revision 2 rewritten against the review.
- [x] (2026-08-16 to 2026-11-22) Revision 2 findings raised as issues #200 to
      #207, tracked in #388 with #378 and #389.
- [x] (2026-11-22) Rebased onto `main` at `051d395` with no conflicts.
- [x] (2026-11-22) Revision 3 drafted (uncommitted); superseded by revision 4.
- [x] (2026-11-22) Revision 4: reconciled against `051d395`, PR #453 at
      `16042ce` and Fearless SIMD 1.2.0; recorded the dependency authorization.
- [x] Delivered on `main` by other pull requests (see Table 3).
- [x] (2026-11-23) Milestone 0: baseline gates on `6e95659`. `check-fmt`,
      `typecheck` and `lint` pass. Nextest: 1579 tests. A whole `make test` run
      hit the 80-minute global timeout under compile-slot contention (838
      passed, 0 failed, 1 cancelled), so the suite was completed in two
      partitions with the same flags: 1558 passed and 1 skipped outside
      `chutoro-compile-contracts`, then 21 of 21 inside it. `make test-doc`
      passes.
- [x] (2026-11-23) Milestone 1: Gate A measurement and assembly inspection.
      Five profiled runs at 4 threads give 0.1640%, 0.1812%, 0.2345%, 0.5458%
      and 0.6895% at n = 10 000 (minimum, median and maximum 0.1640%, 0.2345%
      and 0.6895%), and 0.7130% to 2.4731% at n = 1000. Supplementary runs give
      0.2706% at one thread and 0.1499% at eight. The highest decision-size
      ratio is below 5%, so Gate A fails. Assembly inspection finds no packed
      float max at the default or `x86-64-v3` codegen; the transform loop is
      scalar `maxss`/`vmaxss`.
- [x] (2026-11-23) Milestone 2: remaining guards.
  - [x] Exact-edge-list oracle property. `run_oracle_equivalence_property`
        now compares the exact edge list, with `weight.to_bits()` identity,
        after the aggregate checks. Deliberate mutation evidence: reversing
        the source tie-break in `MstEdge::Ord` made the property fail with
        `edge list mismatch` on a 34-node `ManyIdentical` fixture while the
        aggregate checks still agreed; reverting the mutation restored the
        pass.
  - [x] BDD feature and step glue, with pinned fixtures.
        `chutoro-core/tests/features/mst_edge_preparation.feature` holds six
        scenarios and `chutoro-core/tests/mst_edge_preparation_bdd.rs` drives
        them through the public `parallel_kruskal` API. All six pass.
  - [x] Lowest-index lookup errors in the transform.
        `mutual_reachability_edges` now folds through `collect_ordered_results`.
        The new test was observed red at eight threads before the fix, naming
        the higher-index target instead of the lower-index source, and passes
        in both pinned pools afterwards. Recorded in
        `docs/chutoro-design.md` §6.3.
  - [x] Dedup-before-sort declined; the reason is in the decision log.
  - [x] (2026-11-24) Duplicate-selection coverage, added after the milestone
        review found the forest-level tests could not discriminate. A forest
        is identical whether preparation deduplicates or the union-find
        cycle-rejects the second copy, so `mst/tests/dedup.rs` calls
        `prepare_edge_list` directly. It pins that equal-weight copies collapse
        to the lowest `sequence`, that reversed orientations canonicalize
        first, that same-pair different-weight edges both survive, and that
        self-loops are dropped. Mutation evidence: disabling the `dedup_by`
        predicate fails `equal_weight_duplicates_keep_the_lowest_sequence`
        and `reversed_orientation_canonicalizes_before_deduplication` while
        the other two keep passing, and restoring it turns them green. The
        oracle property also gained a duplicate-heavy fixture at 2, 3, 8 and
        33 nodes.
  - [x] (2026-11-24) Deep property evidence. The five MST properties pass at
        `PROPTEST_CASES=25000`: 67/67 tests, 178.94 s, via
        `cargo test -p chutoro-core --lib mst::property::tests::`. nextest's
        default 60 s per-test budget is too small for that case count, so the
        run used the test harness directly rather than changing a repository
        budget.
  - [x] (2026-11-26) Lowest-index rule as a property. The hosted review's
        pre-merge check asked for the transform's rule to be covered by a
        property, not only by fixed examples.
        `missing_core_distances_report_the_lowest_index_property` generates a
        harvest with a non-empty suffix of nodes lacking a core distance,
        tiles it to 4096 edges so several failing indices sit on different
        workers, and asserts in one-thread and eight-thread pools that the
        reported endpoint is the one an independent oracle names, in the same
        role. Mutation evidence: checking the target before the source makes
        it, and the existing example case, fail with the target named where
        the source is expected; reverting restores both. The plain-slice API
        cannot express an arbitrary missing subset, so the property documents
        that the failing set is necessarily a suffix.
  - [x] (2026-11-26) Compile contract for the feature gate. The same review
        asked for the new `required-features` declaration to have a test.
        `chutoro-compile-contracts/tests/mst_feature_gate.rs` runs
        `cargo test -p chutoro-core --no-default-features --test
        mst_edge_preparation_bdd --no-run` and asserts Cargo refuses the
        target _by name_, naming the `cpu` feature, rather than compiling it.
        `mst_feature_gate` joins the `nested-cargo` serial group, which the
        nextest configuration asks every Cargo-spawning binary to do.
- [x] (2026-11-23) Milestone 3 (gated on A): declined, Gate A failed.
- [x] (2026-11-23) Milestone 4 (gated on A): declined, Gate A failed.
- [x] (2026-11-23) Milestone 5 (gated on A and its own case): declined, Gate A
      failed.
- [x] (2026-11-23) Milestone 6 (gated on A and B): declined, Gate A failed.
- [x] (2026-11-23) Milestone 7: design-document update, roadmap and final
      gates. `docs/chutoro-design.md` §6.3 carries the 2026-11-23
      implementation update with the measurement table; `docs/roadmap.md`
      ticks 2.3.2 with a one-line null result; `docs/users-guide.md` is
      untouched because no `stage` label shipped and no error wording changed.
- [x] (2026-11-24) Milestone review and final gates. The local CodeRabbit CLI
      review completed over the committed range and raised three minor
      findings, all repaired and recorded in the decision log. Gates on the
      final tree: `check-fmt`, `typecheck`, `lint` (interrogate, rustdoc,
      Clippy, Whitaker), `markdownlint`, `nixie` and `make test-doc` pass.
      Nextest passes 1567/1567 outside `chutoro-compile-contracts` and
      `benchmark_smoke`, and 6/6 in the MST scenario target; `benchmark_smoke`
      passes on its own in 1629 s of its 1800 s budget; the 21
      compile-contract tests pass across all nine binaries in per-binary
      invocations. A single whole-`make test` run still expires a per-test
      budget on the host's saturated compile admission; that is issue #454's
      gate-reliability defect, not a failure of any test in this range.
- [x] (2026-11-25) Gate A re-measured at head `4e96416` after the transform's
      collection changed. Five four-thread runs give 0.0512%, 0.0769%,
      0.1063%, 0.1417% and 0.3024% at n = 10 000 (minimum, median and maximum
      0.0512%, 0.1063% and 0.3024%), and 0.1518% to 0.3480% at n = 1000. The
      gate fails by a wider margin than at `7368d8d`, so the decline does not
      depend on the earlier measurement.
- [x] (2026-11-24) `make test` equivalence, as the milestone review asked for
      it to be stated. Locally the literal command stops in its own nextest
      _setup script_ (`warm-compile-contracts`), which is excluded from the
      measured test time but not from that script's 1800 s allowance; on this
      host the same suite takes about 5800 s, so the script expires before any
      test starts. Nothing in the range is implicated. The full set that
      `make test` would execute has been run: 1630/1630 tests outside
      `chutoro-compile-contracts` and `benchmark_smoke`, 1/1
      `benchmark_smoke`, 21/21 compile-contract tests with the gate's exact
      `RUSTFLAGS`, and every doctest binary. Hosted `build-test` runs the
      workspace nextest suite under the repository's own setup scripts and
      then `make test-doc`, alongside `make lint`, `make spelling`,
      `make test-workflow-contracts` and `make test-scheduling-contract`. Its
      differences from the literal command are coverage instrumentation and
      the coverage action's target selection. Two heads have their own
      results, and neither should be read for the other: head `42f9e8c`,
      run [45646271217 job
      139317999331](https://github.com/leynos/chutoro/actions/runs/45646271217/job/139317999331),
      `1634 tests run: 1634 passed, 1 skipped`; and head `4e96416`, run
      [45673781589 job
      139415750440](https://github.com/leynos/chutoro/actions/runs/45673781589/job/139415750440),
      `1642 tests run: 1642 passed, 1 skipped` in 123.5 s.

## Surprises & discoveries

### Found in revision 7 (2026-11-23, measured)

- Observation: **Gate A is far below its threshold on the current tree.**
  Evidence: five profiled runs on the Table 4 workload at four threads give
  ratios of 0.1640%, 0.1812%, 0.2345%, 0.5458% and 0.6895% at n = 10 000, and
  0.7130% to 2.4731% at n = 1000. The highest decision-size ratio is under
  0.7%. Impact: the gate fails under its own pass/fail rule, so Milestones 3 to
  6 are declined and the plan takes its null-result path.

- Observation: **the harvested edge count is about 34n, not about 16n.**
  Evidence: a one-off probe on the decision workload reports 340,465 harvested
  edges at n = 10 000 and 32,966 at n = 1000. Impact: revision 2's estimate
  understated the list by more than half, but the measured stage shares are
  smaller still, so the conclusion does not depend on the estimate.

- Observation: **the measurement window was noisy but the ratios were
  decisive.** Evidence: the host ran under heavy unrelated compile load from
  other agents for the whole window, and the `pipeline/n=10000` Criterion
  intervals span 43 s to 80 s. Even the highest ratio, at the noisiest moment,
  is 0.69%. Impact: the noise band is wide in absolute time but cannot close a
  gap of more than four percentage points.

- Observation: **the scalar loops are not already effectively vectorized, and
  a target-specialized build does not change that.** Evidence: scanning the
  emitted assembly of `chutoro-core`, the default build contains no `maxps`,
  `cmpps` or `vpcmpgtq` anywhere, and the transform's per-edge loop is scalar
  `maxss` twice per edge. The `x86-64-v3` build uses `vmaxss` twice per edge
  and still contains no packed float max. Impact: Gate A, not the assembly, is
  what declines the kernel; the assembly finding is recorded so a later reader
  does not have to repeat it.

- Observation: **the measurement method needed two bounded substitutions.**
  Evidence: assigning `RUSTFLAGS="--emit=asm"` would suppress the dependency
  rlibs, so the assembly was emitted with
  `cargo rustc -p chutoro-core --profile bench --lib -- --emit=asm`, which
  confines the flag to the crate under inspection and reuses the bench-profile
  cache. The supplementary one-thread and eight-thread runs bound Criterion with
  `--profile-time 1`, because the decision-size iteration costs minutes at one
  thread. Impact: neither substitution touches the five-run, four-thread Gate A
  measurement; both are recorded in the decision log.

### Found in revision 4 (2026-11-22)

- Observation: **the transform's lookup-error selection is nondeterministic.**
  Evidence: `mutual_reachability_edges` (`cpu_pipeline/mod.rs:125-135`) collects
  `par_iter().map(...)` straight into `Result<Vec<_>>`. The neighbouring
  `collect_ordered_results` (`cpu_pipeline/mod.rs:249-260`) documents that this
  "is not deterministic across thread counts". Impact: when several harvested
  endpoints lack a core distance, which `GraphInvariantViolation` is reported
  depends on scheduling. That is the same defect class as #206. Milestone 2
  fixes it by reusing the existing helper.

- Observation: **the transform does not validate finiteness.** Evidence:
  `mutual_reachability_edge` (`cpu_pipeline/mod.rs:139-160`) computes
  `distance.max(left_core).max(right_core)` unconditionally. Finiteness is
  checked later, by `validate_and_canonicalize_edge` (`mst/mod.rs:196-201`).
  Impact: a NaN or `-inf` raw distance with a finite core distance yields a
  finite, accepted weight today. A kernel that rejects non-finite operands
  early would change which inputs are accepted.

- Observation: **the scalar reference is not bit-deterministic for signed
  zeros.** Evidence: the Rust 1.99.1 and 1.89 documentation for `f32::max` says
  that when the inputs compare equal, "such as for the case of `+0.0` and
  `-0.0`, either input may be returned non-deterministically". With exactly one
  NaN operand, quiet or signalling, it returns the other. Impact: the semantic
  oracle can require bit equality only against the scalar reference _as
  compiled in the same binary_. It must treat mixed-zero ties as exceptional
  lanes; see "Semantic oracle".

- Observation: **signed zeros reach ordering.** `MstEdge::Ord` uses
  `total_cmp` (`mst/mod.rs:64-78`), which orders `-0.0` before `+0.0`. The
  dedup and the weight-group `chunk_by` use `partial_cmp == Equal`
  (`mst/mod.rs:313-319`, `:351-356`), which treats them as equal. Impact: a
  backend that returns a different zero sign can change the order within a
  group, and therefore which duplicate survives and the order of acceptance.

- Observation: **the `stage` label set is documented as fixed.** Evidence:
  `docs/users-guide.md:150-157`. Impact: a Gate B sub-stage instrument cannot
  quietly add labels. Milestone 3 records the choice explicitly.

- Observation: **`chutoro-core` has no SIMD code, backend selector or SIMD
  feature.** Its features are `cpu` (default), `metrics`, `skeleton` and `gpu`
  (`chutoro-core/Cargo.toml`), and nothing in `chutoro-core/src` uses
  `core::arch` or `target_feature`. Impact: a core kernel needs its own private
  selector. Reusing the dense selector would violate Constraint 9.

- Observation: **default builds target the x86-64 baseline.**
  `.cargo/config.toml` sets no `target-cpu`, so autovectorization uses SSE2
  only. Impact: Milestone 1 must inspect both the default build and an
  `x86-64-v3` build before drawing conclusions.

- Observation: **revision 3 overstated the evidence.** It treated the #431
  profile as deciding Gate A and "declined" Milestones 3 to 5. That profile
  predates `051d395`, comes from a host described differently, and is a single
  untimed run per size. Revision 4 restores Gate A as an outstanding
  measurement.

### Found while rebasing (revision 3)

- Most of revision 2's committed structural work landed piecemeal through
  issue-driven pull requests (Table 3).
- A third discarded `(sequence, natural Ord)` sort survives in
  `EdgeHarvest::from_parallel_inserts` (`hnsw/cpu/collectors.rs:72-85`,
  `hnsw/types.rs:302-303`). It is recorded as future work in
  `docs/chutoro-design.md` lines 1318-1323 and is unrecoverable here under
  Constraint 10.
- The oracle-equivalence property still compares only aggregates
  (`SequentialMstResult`, `mst/property/oracle.rs:70-77`). A
  `sequential_kruskal_edges` helper (`oracle.rs:112`) now exists.
- There are no BDD scenarios for the MST.
- Verus runs through `VERUS_PROOF_FILES` (`Makefile:129-131`) and
  `rust-prover-tools`, not `scripts/run-verus.sh`. ADRs live in
  `docs/adrs/adr-000N-*.md`.
- The first session-path MST caller will be roadmap item 11.2.1
  (`ClusteringSession::refresh`). The only production caller today is
  `cpu_pipeline/mod.rs:111`.

### Revision 2 findings, now resolved on `main`

- The workspace lint table was inert in most crates: fixed by #200 (PR #228).
- The concurrency property did not exercise concurrency: fixed by #201 (PR
  #225). `ci.yml:47` and `coverage-main.yml:53` still pin
  `RAYON_NUM_THREADS: '1'`, which is why tests use explicit pools.
- No Kani harness gated a merge: fixed by #202 (PR #227).
  `.github/workflows/kani-pr.yml` runs `make kani` for pull requests touching
  `**/kani_*.rs` and `mst/kani_harness.rs`. `make kani` (`Makefile:292-297`)
  includes two MST harnesses, and `kani-full` is unblocked per
  `docs/kani-full-hnsw-hypothesis-testing.md` §"Current conclusion".
- `MstErrorCode` was exhaustive: fixed by #203 (PR #427).
- The double sort, the per-group allocation and the nondeterministic MST error
  selection were fixed by #204, #205 and #206.
- The users' guide lacked MST errors: fixed by #207 (PR #434).
- The serial core-distance loop: fixed by #389 (PR #431). The incremental
  session path remains serial by design (`docs/chutoro-design.md` lines
  929-933).
- The parallel-build deadlock: fixed by #378 (PRs #386 and #429, ADR-0006),
  with the liveness fix completed in PR #432.

### Revision 2 findings, still true

- `try_union` takes no lock on the cycle-rejection path
  (`mst/union_find.rs:57-62`). That is why a post-sort connectivity sweep earns
  nothing.
- The development machine has no AVX-512, and no workflow runs on ARM. Every
  `runs-on` is `ubuntu-latest` or an x86-64 `ubicloud-standard-*` runner.
- `googletest`, `pretty_assertions` and `insta` are absent from `Cargo.lock`.

## Historical measurements (dated; not current facts)

**2026-08-16, revision 2, branch tree before PR #276.** `parallel_kruskal` took
1.8781 ms [1.8448, 1.9128] at n = 1000, against an 853.60 ms
`CpuHnsw::build_with_edges`, which is about 0.22% of the build. That comparison
**excluded the mutual-reachability transform and did not measure the whole
pipeline**, so it cannot set a present-day ceiling. Revision 2's figures of
13-17 ms at n = 10 000, about 0.07%, and 150-200 ms at n = 100 000 are
**extrapolations**, not measurements.

**2026-11-17 to 2026-11-19, #389 ExecPlan, a `main` predating `051d395`.**
These figures come from one untimed profiled run per size at
`RAYON_NUM_THREADS=4`, after core distances were parallelized:

| Stage                 | n = 1000   | Share  | n = 10 000  | Share  |
| --------------------- | ---------- | ------ | ----------- | ------ |
| `whole_run`           | 1.298107 s | 100%   | 29.621945 s | 100%   |
| `core_distances`      | 0.172432 s | 13.28% | 3.602843 s  | 12.16% |
| `mutual_reachability` | 0.013450 s | 1.04%  | 0.027135 s  | 0.09%  |
| `kruskal`             | 0.002713 s | 0.21%  | 0.021009 s  | 0.07%  |

_Table 5: Historical stage profile from the #389 ExecPlan (lines 860-935). Not
a Gate A result._

On those numbers the Gate A ratio would be about 1.25% at n = 1000 and 0.16% at
n = 10 000. That is an expectation for Milestone 1, not its outcome.

## Decision log

- Decision (2026-11-24): the milestone CodeRabbit CLI review raised three
  minor findings and all three were repaired.

  1. The reproduction block still prescribed `RUSTFLAGS="--emit=asm"`, the
     method the same plan records as broken. Valid: rewritten to the
     `cargo rustc --profile bench --lib` form that produced the evidence, along
     with the `target/asm-*` references in "Idempotence and recovery".
  2. The new BDD test target lacked `required-features = ["cpu"]`. Valid, and
     confirmed by measurement: `cargo check -p chutoro-core
     --no-default-features --tests` failed to resolve its imports and added a
     third broken target to the two that `origin/main` already has
     (`functional_ari_nmi`, `session_append_bdd`). A `[[test]]` block with
     `required-features = ["cpu"]` now skips it instead; the same command
     afterwards fails on exactly the two pre-existing targets and no other,
     and the target still builds under default features. The two pre-existing
     breakages are out of scope here and are not claimed as fixed.
  3. The ticked roadmap item read as an instruction that was implemented. Valid:
     reworded as an evaluation with a pre-registered gate, so the checkbox and
     the null-result note agree.

- Decision (2026-11-24): the hosted whole-plan completeness assessment
  returned three follow-ups and all three were actioned. Unlike the milestone
  review's cosmetic items these were substantive, so each carries its own
  evidence.

  1. **Deep property evidence was missing.** Valid: the plan's acceptance
     requires the five MST properties at `PROPTEST_CASES=25000`, while the
     pull-request lane runs 250 and the weekly lane had not run for this head.
     The run is now recorded: 67/67 tests, 178.94 s, all five properties green.
     It used `cargo test --lib` rather than nextest because nextest's 60 s
     per-test default is smaller than the case count needs; no repository
     budget was changed to obtain it.
  2. **Duplicate-selection coverage could not discriminate.** Valid, and the
     sharpest finding of the review: a forest looks the same whether
     preparation removes a duplicate or the union-find cycle-rejects it, so
     neither the BDD duplicate scenario nor the exact-edge-list property could
     fail if preparation stopped deduplicating. `mst/tests/dedup.rs` now calls
     `prepare_edge_list` directly, and the oracle property gained a
     duplicate-heavy fixture at four sizes. Mutation-checked: disabling the
     `dedup_by` predicate fails two of the four new tests, and restoring it
     turns them green.
  3. **The plan record was stale in three places.** Valid: the inventory still
     said "Remaining" for delivered work, no measurement SHA was recorded, and
     the `n = 1000` median was absent. The inventory is reconciled, Gate A now
     records `7368d8d` as the head of the first measurement and `4e96416` as the
     head of the separate re-measurement, and the design table carries both size
     medians.

  The same reply asked for the literal `make test` command's evidence gap to be
  stated or closed. It is stated in the Progress entry: the local command stops
  in its own nextest setup script under this host's queue, every test it would
  run has been executed in partitions, and the hosted `build-test` suite for
  head `42f9e8c` passed 1634 tests plus the doctest step.

- Decision (2026-11-26): the hosted formal review raised two inline findings
  and three failed pre-merge checks.

  1. **Plan header versus the open `make test` gate.** Valid as stated: the
     header said COMPLETE while the literal command had no passing run, and no
     maintainer agreement to an equivalence existed to record. Rather than
     invent one, the plan now states the gate's disposition explicitly under
     "Gate disposition for the literal `make test`", quotes the developers'
     guide's "a budget, not a detected hang" ruling, lists the partitioned and
     hosted evidence, and names the one remaining step for a reader who needs
     the literal command. The status line says the plan is complete for
     implementation and record, and says it does not claim that run.
  2. **Roadmap attribution.** Valid: the ticked item presented the `7368d8d`
     range as current. It now names the head for each range and records the
     re-measured 0.0512%-0.3024% alongside.

  The three pre-merge checks were actioned rather than rebutted. Developer
  documentation gained a "Test targets that need a crate feature" section and a
  paragraph on the transform's lowest-index rule in "Edge harvest and the MST
  boundary". The property check produced the generated lowest-index property
  above. The compile-time check produced `mst_feature_gate`.

- Decision (2026-11-26): record the hosted red rather than hide it. Adding the
  nested-build test `mst_scenario_target_requires_the_cpu_feature` made
  `make test-workflow-contracts` fail on head `9cc610e`, because
  `tests/workflow_contracts/trybuild_override_test.py` pins the exact set of
  `chutoro-compile-contracts` tests that spawn nested builds. That pin exists
  so a test cannot gain a nested build without also gaining its allowance. The
  new test is now named in the set, and the file passes 50/50. The failing head
  is kept in the record instead of being replaced by the fixing head.

- Decision (2026-11-23): **dedup-before-sort is declined.** The three
  Milestone 2 conditions do not all hold. The dedup key the milestone names is
  `(weight bits, source, target)`, but production dedup uses
  `partial_cmp == Equal` on the weight, which treats `-0.0` and `+0.0` as equal
  while a bit-keyed dedup would keep both; adopting the bit key would therefore
  change which duplicate survives and could change accepted output, violating
  Constraint 1. The code would also be more complex: today the dedup is one
  linear pass over the sorted list with no allocation, and the sort cannot be
  removed because Kruskal consumes the list in total order. Pre-sort dedup
  would need an auxiliary index to choose the lowest `sequence` per key, then
  still pay for the sort. No performance win is claimed.

- Decision (2026-11-23): **Gate A fails, so Milestones 3 to 6 are declined.**
  The measured decision-size ratios are 0.1640%, 0.1812%, 0.2345%, 0.5458% and
  0.6895% across five four-thread runs; the gate's rule fails the gate when the
  highest ratio is below 5%. Tolerance 5 requires recording the null result
  beside the roadmap item and continuing with the applicable remaining work,
  which is Milestone 2 and Milestone 7. Open questions Q1 and Q2 close as moot,
  as the 2026-11-23 authorization decision provided. Confirmed on 2026-11-25 by
  a re-measurement at the measured head, `4e96416`, whose ratios are 0.0512% to
  0.3024% at the decision size: Milestone 2's change to the transform's
  collection was a production change to a timed stage, so the first measurement
  was not transferred to later heads without being repeated.

- Decision (2026-11-23): record the assembly-inspection method substitution.
  `cargo rustc -p chutoro-core --profile bench --lib -- --emit=asm` was used
  instead of the plan's `RUSTFLAGS="--emit=asm" cargo build`, because an
  assigned `RUSTFLAGS` applies to every crate in the graph and suppresses the
  dependency rlibs the build needs. The flag therefore reaches only the crate
  under inspection, from the same bench profile and the same baseline and
  `x86-64-v3` codegen. The supplementary one-thread and eight-thread profile
  runs use Criterion `--profile-time 1` to bound their measurement phase.
  Neither substitution changes the five-run four-thread Gate A measurement or
  its pass/fail rule.

- Decision (2026-11-23): the maintainer approved this plan and instructed that
  implementation begin. The approval covers Milestones 0 to 2 as production
  work and closes the plan's approval gate. Milestones 3 to 6 remain gated on
  Gates A and B as pre-registered; the approval does not weaken either gate.

- Decision (2026-11-22): record the maintainer's dependency authorization as a
  narrow exemption to Constraint 7 and Tolerance 3. The authorized entries are
  `fearless_simd = "1.2"` and, where `#[simd]` is used,
  `fearless_simd_macros = "0.1"`, for a justified, gated kernel. The exemption
  is effective now and needs no further permission. It is independent of ADR
  0007's review status: ADR 0007 is the proposed architecture, and this plan
  follows it as guidance. If it is amended before merging, re-check Milestone
  6. The exemption does not authorize unrelated dependencies, an MSRV increase,
  unconditional vectorization, or changes to public feature semantics. This
  revision changes no manifest or lockfile.

- Decision (2026-11-22): 2.2.8 and 2.3.2 stay independent (Table 2). Sharing a
  dependency does not justify a shared framework. Core kernels live in
  `chutoro-core` and never depend on the dense provider.

- Decision (2026-11-22): withdraw revision 3's "declined on recorded evidence".
  Gate A must be measured on the current tree under Table 4. Revision 3's
  replacement of revision 2's bespoke benchmark with the `pipeline` target and
  profile stands for Gate A, because that instrument reports exactly the Gate A
  terms. Gate B needs a sub-stage instrument, which Milestone 3 builds only if
  Gate A passes.

- Decision (2026-11-22): remove `harvest_resort` from the Gate A numerator,
  after verifying that the sort is gone (see Gate A).

- Decision (2026-11-22): keep the pre-registered Milestone 6 entry rule exactly
  as revision 2 wrote it. The whole SIMD milestone, transform kernel included,
  requires Gate A **and** Gate B. Transform timing stays out of Gate B's
  numerator, because the transform sits outside Kruskal. This means a
  transform-only SIMD win cannot proceed while classification and compaction
  are below 10% of Kruskal. Changing that would weaken a pre-registered gate,
  so it is **flagged for the maintainer** as open question Q1 rather than
  decided here.

- Decision (2026-11-22): resolve the signed-zero and signalling-NaN conflict
  between `max_precise` and `f32::max` with an exceptional-lane rule (see
  "Semantic oracle"). Lanes whose operands include any NaN, or whose result is
  a zero produced from operands of mixed zero sign, are recomputed with the
  chutoro scalar expression. Nothing is normalized. The kernel defers to the
  same compiled scalar code the build uses today. Compatibility evidence comes
  from the differential tests in Milestone 6.

- Decision (2026-11-22): start any kernel with SIMD classification and scalar
  ordered emission. Use `compress` only if the measured emission cost after the
  first kernel justifies it, and then only under the joint-field rules in
  Milestone 6.

- Decision (2026-11-22): try `Simd::vectorize` before adding
  `fearless_simd_macros`. Add the macro only if assembly shows the generic body
  compiled without the token's target features, and record its compile-time
  cost.

- Decision (2026-11-23): the maintainer authorized production code for
  Milestones 0 to 2, and confirmed that Kruskal remains a candidate only
  through the gates. The expected path is: measure Gate A, record what is
  expected to be a null result, land the Milestone 2 guards, and close the item
  in Milestone 7. Open questions Q1 and Q2 apply only if Gate A passes, and are
  not answered until then. Rationale: at the historical share of about 0.16%,
  even eliminating both stages entirely would save at most that fraction of a
  run.

- Decision (2026-08-16, still in force): no `key` field on `MstEdge`; do not
  narrow the union-find arrays, which are already separate
  (`union_find.rs:20-28`); use plain assertions rather than adding `googletest`,
  `pretty_assertions` or `insta`; register this plan in `docs/contents.md`,
  which is done.

- Decision (2026-08-16, still in force): any pre-filter must be real
  Filter-Kruskal (partition, then filter). Its earlier justification rested on
  a false claim about `try_union` locking.

- Superseded: revision 2's `mst_prepare.rs` five-group benchmark (its
  `harvest_resort` and `mutual_reachability` groups are obsolete or covered by
  the profile), its union-find loop extraction (needed only for Milestone 5),
  `scripts/bench-mst-pipeline.sh`, and `PROOF_FILES` in `scripts/run-verus.sh`.

## Context and orientation

### Terms

- **HNSW** (Hierarchical Navigable Small World) is the approximate
  nearest-neighbour index (`chutoro-core/src/hnsw/`).
- **Candidate edge** is `CandidateEdge` (`hnsw/types.rs:159`): `source: usize`,
  `target: usize`, `distance: f32`, `sequence: u64`. It is an
  array-of-structures record, 32 bytes on 64-bit targets.
- **Core distance** is the distance to a point's `min_cluster_size`-th nearest
  non-self neighbour (`compute_core_distances`, `cpu_pipeline/mod.rs:221`).
- **Mutual reachability** is the weight
  `distance.max(core[source]).max(core[target])`, evaluated in that order
  (`cpu_pipeline/mod.rs:154-157`).
- **Fearless SIMD token** is a zero-sized value proving that a CPU feature
  bundle is available, obtained safely from `Level::new()`.

### Current control flow

`run_cpu_pipeline_with_len` (`cpu_pipeline/mod.rs:30`) runs these steps in
order:

1. `build_index_and_core_distances` (line 61) builds HNSW, harvests edges, and
   computes core distances in parallel. This is the `core_distances` stage.
2. `build_mutual_reachability_forest` (line 97) computes
   `mutual_reachability_edges` (the `mutual_reachability` stage), then calls
   `parallel_kruskal_from_edges` (the `kruskal` stage).
3. Labels are extracted.

Inside `parallel_kruskal_from_edges` (`mst/mod.rs:324-371`):

1. It checks `EmptyGraph`.
2. `prepare_edge_list` (lines 296-321) runs parallel validation into an
   ordered buffer, then a sequential scan to the first error or the accepted
   edges. It then calls `par_sort_unstable` by `MstEdge::Ord`, and finally
   dedups on equal weight and endpoints.
3. It allocates the union-find.
4. It walks the `chunk_by` equal-weight groups through
   `extend_with_accepted_edges`, with an early exit on `is_mst_complete`.
5. It sorts the forest.

## Semantic oracle

Every candidate implementation is compared with these rules. Each is exact.

**Transform.** For each edge, in input order:

1. Look up `core[source]`, then `core[target]`, with checked `get`. A missing
   entry is `GraphInvariantViolation` naming the endpoint role and node.
2. Compute `weight = distance.max(core[source]).max(core[target])` with Rust's
   `f32::max`, in that order.
3. Keep endpoints and `sequence` unchanged.
4. Accept every input value, including NaN, ±infinity and ±0. The transform
   validates nothing else.
5. With several failing lookups, report the lowest input index. Within one
   edge, the source lookup comes before the target lookup.

**Exceptional lanes.** A lane is exceptional if either of these holds:

- any of its three operands is NaN, quiet or signalling; or
- its result is `+0.0` or `-0.0` and its operands include both `+0.0` and
  `-0.0`.

These are exactly the cases where `max_precise` is non-deterministic or
implementation-defined, or where scalar `f32::max` is itself free to choose.
Exceptional lanes are recomputed with the step 2 expression in chutoro scalar
code. Every other lane's vector result must equal the scalar result bit for
bit. The kernel detects both conditions with `is_nan`, `is_sign_negative` and
equality to zero, so the check stays vectorized.

**Validation and classification.** For each transformed edge, in input order:

1. If `source >= node_count`, report `InvalidNodeId` with the original source.
2. Else if `target >= node_count`, report `InvalidNodeId` with the original
   target.
3. Else if the weight is not finite, report `NonFiniteWeight` with the
   original `(source, target)` orientation.
4. Else if `source == target`, drop the edge silently.
5. Otherwise emit `MstEdge` with canonical `(min, max)` endpoints, the same
   weight bits and the same `sequence`.

The first error by input index wins, and checks run in the order above.
`EmptyGraph` is decided before any edge is inspected.

**Ordering.** Sort, dedup, group, union-find walk and final sort are unchanged,
so the forest is identical under Constraint 1.

## Plan of work

### Milestone 0: baseline gates

No code changes.

```sh
set -o pipefail
git branch --show-current
make check-fmt 2>&1 | tee /tmp/check-fmt-chutoro-2-3-2-baseline.out
make typecheck 2>&1 | tee /tmp/typecheck-chutoro-2-3-2-baseline.out
make lint      2>&1 | tee /tmp/lint-chutoro-2-3-2-baseline.out
make test      2>&1 | tee /tmp/test-chutoro-2-3-2-baseline.out
```

Record the exact `make test` summary in `Progress`. Revision 2's figure of 1058
tests is stale. Under compile-slot contention, a nextest global-timeout
`SIGTERM` is not a test failure: check `FAIL=0` and the timeout's provenance.

### Milestone 1: Gate A and assembly inspection

1. Measure Gate A five times on the Table 4 workload:

   ```sh
   set -o pipefail
   for run in 1 2 3 4 5; do
     CHUTORO_BENCH_PIPELINE_PROFILE_PATH="/tmp/2-3-2-profile-$run.csv" \
     RAYON_NUM_THREADS=4 taskset -c 0-3 \
       cargo bench -p chutoro-benches --features pipeline-profile --bench pipeline \
       2>&1 | tee "/tmp/bench-pipeline-chutoro-2-3-2-$run.out"
   done
   ```

   Record the harvested edge count, the per-run ratios at both sizes, their
   minimum, median and maximum, and the Criterion interval. Repeat once at one
   thread and once at eight threads, and report those runs separately.

2. Inspect the optimized assembly of the two scalar loops: the transform
   closure in `mutual_reachability_edges` and the validation map in
   `prepare_edge_list`. Do this for the default build and for
   `-C target-cpu=x86-64-v3`:

   ```sh
   set -o pipefail
   cargo rustc -p chutoro-core --profile bench --lib -- \
     --emit=asm -C debuginfo=0 \
     2>&1 | tee /tmp/asm-default-2-3-2.out
   cargo rustc -p chutoro-core --profile bench --lib -- \
     --emit=asm -C debuginfo=0 -C target-cpu=x86-64-v3 \
     2>&1 | tee /tmp/asm-v3-2-3-2.out
   ```

   `cargo rustc` passes the flags to the crate under inspection only, whereas
   an assigned `RUSTFLAGS` would apply to every crate in the graph and suppress
   the dependency rlibs the build needs. Both builds reuse the bench-profile
   cache; the emitted `.s` files land in `target/release/deps/`, the newer file
   being the `x86-64-v3` one.

   Search the emitted `.s` files for the mangled closure symbols. Record
   whether the loop bodies contain packed instructions (`maxps`, `vmaxps`,
   `cmpps`, `vpcmpgtq` and so on) or only scalar ones (`maxss`, `ucomiss`). The
   array-of-structures layout is expected to defeat autovectorization, but that
   is an expectation, not a finding. Both builds use their own target
   directories, because overriding `RUSTFLAGS` would otherwise invalidate the
   shared `target/` cache. Both still queue for compile admission as normal.
   `cargo asm` is not installed.

3. Decide and record the outcome. If Gate A fails, Milestones 3 to 6 are
   declined: record the ratio, then complete Milestones 2 and 7. If it is
   inconclusive, tighten the method once (more runs, a quieter host window). If
   it is still inconclusive, treat it as a failure, as Tolerance 5 requires.

### Milestone 2: remaining guards (always executed)

None of these changes accepted output.

1. **Exact-edge-list oracle property.** Strengthen
   `run_oracle_equivalence_property` (`mst/property/equivalence.rs`) so that it
   asserts exact edge-list equality with `sequential_kruskal_edges`
   (`oracle.rs:112`): the same endpoints, `sequence` and `weight.to_bits()`, in
   the same order, while keeping the aggregate checks for diagnostics. Validate
   the property by deliberate mutation: reverse the tie-break in
   `MstEdge::Ord`, confirm the property fails, then revert. Record both
   outcomes.

2. **BDD scenarios.** Add
   `chutoro-core/tests/features/mst_edge_preparation.feature` and
   `chutoro-core/tests/mst_edge_preparation_bdd.rs`, following
   `tests/session_append_bdd.rs`: a `World` struct, a `#[fixture] fn world()`,
   steps returning `rstest_bdd::StepResult<(), BddStepError>`, and one
   `#[scenario]` function per scenario. Use only the public API.

   ```gherkin
   Feature: Minimum spanning forest edge preparation

     Scenario: Mutual-reachability weights drive forest selection
       Given a graph with 6 nodes and 9 candidate edges
       And core distances that raise the weight of edge 3
       When I build the minimum spanning forest
       Then the forest has 5 edges
       And the forest excludes edge 3

     Scenario: Duplicate candidate edges are collapsed
       Given a graph with 4 nodes and 8 candidate edges
       And 4 of the candidate edges are exact duplicates
       When I build the minimum spanning forest
       Then the forest has 3 edges
       And every forest edge has source less than target

     Scenario: Self-loops are discarded without failing
       Given a graph with 5 nodes and 7 candidate edges
       And 2 of the candidate edges are self-loops
       When I build the minimum spanning forest
       Then the forest has 4 edges

     Scenario: The lowest-index invalid edge is reported
       Given a graph with 4 nodes and 6 candidate edges
       And candidate edge 1 references node 9
       And candidate edge 5 has a non-finite weight
       When I build the minimum spanning forest
       Then forest construction fails with error code "INVALID_NODE_ID"

     Scenario: An edge that is both out of range and non-finite reports its node
       Given a graph with 4 nodes and 3 candidate edges
       And candidate edge 2 references node 9 with a non-finite weight
       When I build the minimum spanning forest
       Then forest construction fails with error code "INVALID_NODE_ID"

     Scenario: A disconnected graph yields a forest, not a tree
       Given a graph with 6 nodes and 4 candidate edges
       And the candidate edges span only nodes 0 to 3
       When I build the minimum spanning forest
       Then the forest reports 3 components
       And the forest is not a tree
   ```

   Pin every fixture before writing the step glue: the exact
   `(source, target, distance, sequence)` tuples and the core-distance vector,
   as a Gherkin data table or as named constants. Scenario 1 applies weights in
   the step with `distance.max(core[s]).max(core[t])`, and never reaches into
   the crate-private transform.

3. **Lowest-index lookup errors in the transform.** Route
   `mutual_reachability_edges` through the existing `collect_ordered_results`
   (`cpu_pipeline/mod.rs:249`), the abstraction-policy sweep result. Write the
   test first: in explicit one-thread and eight-thread pools, give a harvest
   two endpoints with no core distance, at a low and a high index, and assert
   that the low-index error is reported. The test may pass by luck before the
   fix, so record whether it was observed red. Update the helper's doc comment
   to name its second call site, and record the strengthening in
   `docs/chutoro-design.md` §6.3.

4. **Dedup-before-sort (exactness-gated).** Keep this only if three things
   hold: the exact-edge-list and two-pool properties pass unchanged; the dedup
   key is `(weight bits, source, target)` with `sequence` excluded, so that the
   lowest `sequence` survives; and the code is no more complex. Update
   `mst/property/oracle.rs:186-193` in lockstep, or decline the step and record
   why. Claim no performance win.

### Milestone 3 (gated on A): Gate B instrument and measurement

Enter only if Gate A passed. Build the Gate B instrument with the operations
defined under Gate B, then measure it on the Table 4 workload.

Recommended instrument: a crate-internal `KruskalTimings` value returned by a
`pub(crate)` profiled variant of `parallel_kruskal_from_edges`, called only by
`build_mutual_reachability_forest` under `cfg(feature = "metrics")`. Record two
additional bounded `stage` labels, `kruskal_classify_compact` and
`kruskal_rest`, documented as **nested inside** `kruskal` so that consumers do
not double-count. This adds labels to a set that `docs/users-guide.md:150-157`
calls fixed. That is a user-visible metrics change. It is permitted here only
with an update to the users' guide and the developers' guide in the same
commit; otherwise escalate under Tolerance 6. Record the abstraction-policy
sweep: no equivalent sub-stage hook exists today.

If Gate B fails, decline Milestone 6 and record the ratio.

### Milestone 4 (gated on A): packed sort record

Enter only if Gate A passed and the sort is measurably the dominant part of
`kruskal`. The design is a crate-internal 24-byte record with an
order-preserving key:

```rust
const fn weight_key(weight: f32) -> u32 {
    let bits = weight.to_bits();
    let sign_mask = 0u32.wrapping_sub(bits >> 31);
    bits ^ (sign_mask | 0x8000_0000)
}
```

It was verified against `total_cmp` over 14 400 pairs on 2026-08-16. The lever
is the record's width, not the comparator. Keep the key a sign-bit test, never
`weight < 0.0`. Compare interleaved, per "Decision workload and method".

### Milestone 5 (gated on A and its own case): Filter-Kruskal

Enter only if Gate A passed, the union-find walk is measurably dominant, and a
written algorithmic case shows partitioning removes sort work. A fast mask or
`compress` primitive is not evidence that an extra filtering pass helps, and
`try_union` already rejects cycles without locking.

If it is entered, implement real Filter-Kruskal and amend §6.3's "Keep the
global sort in Rayon" through an ADR. Several obligations follow:

- Keep the equal-weight tie-breaking of §6.2.
- Expose thresholds as values on an options struct, so that a property case
  can force the filter active on small graphs. MST fixtures cap at 64 nodes
  (`mst/property/strategies.rs:20`).
- Add a `debug_assert` that cleared endpoints share a root.
- Prove root-identity monotonicity (`parents[n] == n` is one-way) in Verus by
  induction over single unions, and register the proof in `VERUS_PROOF_FILES`.
- Name any Kani harness `kani_*.rs`, so that `kani-pr.yml` runs it.

### Milestone 6 (gated on A and B): private Fearless SIMD kernel

Enter only if Gate A passed, Gate B passed on the post-Milestone-4 tree, and
Milestone 1's assembly inspection shows that the scalar loops are not already
effectively vectorized. Otherwise record the null result, with the deciding
number or assembly excerpt, and skip to Milestone 7.

**Dependencies.**

- Add `fearless_simd = "1.2"` (and `fearless_simd_macros = "0.1"` only per the
  decision log) to `[workspace.dependencies]`.
- In `chutoro-core/Cargo.toml`, add
  `fearless_simd = { workspace = true, optional = true }` and append
  `"dep:fearless_simd"` to `cpu`. Add no new public feature.
- Show that the backend is absent without `cpu`:
  `cargo tree -p chutoro-core --no-default-features -e normal` lists no
  `fearless_simd`.
- Show the MSRV holds:
  `cargo +1.89 check -p chutoro-core --all-features --locked` and
  `cargo +1.89 test -p chutoro-core --no-run --locked` on the resolved
  lockfile. Record the resolved versions. Record development-dependency
  incompatibilities that already existed separately from new ones.
- Record clean-build time and `chutoro-cli` release binary size, before and
  after.

**Input representation.** Edges stay array-of-structures (`CandidateEdge`). The
kernel never sees `CandidateEdge`, `MstEdge` or errors. For each Rayon chunk,
scalar code stages plain buffers:

- For the transform: `distance: Vec<f32>`, plus `left_core: Vec<f32>` and
  `right_core: Vec<f32>` gathered with the checked lookups of the semantic
  oracle. A missing entry ends staging for that chunk with the ordered error.
- For classification: `source: Vec<u64>`, `target: Vec<u64>` and
  `weight: Vec<f32>`.

Staging and gather costs belong inside every measured region. Fearless SIMD
does not make indexed `core[...]` reads contiguous.

**Identifier width.** Compile the kernels only under
`cfg(target_pointer_width = "64")`, where `usize as u64` is lossless.
Elsewhere, use the scalar path. Never narrow identifiers. Compare against
`node_count as u64` in full 64-bit lanes, and build diagnostics from the
original `usize` values.

**Logical batch.** `L = S::f32s::LEN`, with `U = S::u64s::LEN` 64-bit lanes per
vector. That gives 8 and 4 on `Avx2`, 16 and 8 on `Avx512`, and 4 and 2 on
`Neon`. Each logical batch of `L` edges uses one `f32` vector and `L / U` `u64`
vectors per endpoint. Convert masks to integer bitmasks with
`SimdMask::to_bitmask`. Combine the `u64` sub-masks by shifting each by
`k * U`, in chutoro-owned integer code. Assert `L % U == 0` when selecting a
backend; if it fails, use scalar.

**Kernels (private, generic over `S: Simd`).**

- `cpu_pipeline/weight_kernel.rs`: given equal-length `&[f32]` slices `d`,
  `l` and `r`, write `w[i] = max_precise(max_precise(d, l), r)`. Return one
  exceptional-lane bitmask per batch, using the rule in "Semantic oracle". The
  caller recomputes flagged lanes in scalar code.
- `mst/classify_kernel.rs`: given `source`, `target` and `weight` slices and
  `node_count`, return per batch four bitmasks: `source_out_of_range`,
  `target_out_of_range`, `non_finite` (`!is_finite`) and `self_loop`.

Loop over `chunks_exact(L)` with `from_slice` on exact-length slices, then
process the remainder with the scalar reference functions. There is **no
padding**, so no inactive lane exists and none can create an error, an edge, or
a count. Every load and store covers exactly `L` elements inside a range proven
by `chunks_exact`. Bitmask bits at or above `L` are masked off before use.

**Ordered emission.** Scalar code walks each batch's lanes in index order. The
first lane with any error bit builds the error, using the oracle's precedence
and the original identifiers. A `self_loop` lane is skipped. Every other lane
emits a canonical `MstEdge`. Chunks are produced with
`par_chunks(...).map(...)` and `collect_into_vec`, then scanned sequentially,
so the lowest index wins by construction. Later chunks still run, exactly as
they do today. Do not use `compress` in the first kernel. A later `compress`
step must compress every field of the logical edge (source, target, weight,
original index, sequence) under the same logical mask. That means widening the
mask to the `u64` lanes with `MaskWiden`, or splitting it, and the step needs
its own measurement.

**Dispatch.** A private `EdgeKernelBackend` enum has the variants `Scalar`,
`Avx2(Avx2)`, `Avx512(Avx512)` and `Neon(Neon)`. It is chosen once per process
in a `OnceLock`:

1. Call `Level::new()`, then the safe `as_avx2()`, `as_avx512()` or
   `as_neon()` for the enabled variants only. See "Backend enablement".
2. Any other level, including `Sse2` and `Sse4_2`, selects `Scalar`.
3. Never fabricate a token. Never call unrestricted `dispatch!`. Never treat
   `Level::baseline()` as scalar.

Match on the backend once per Rayon chunk, and call the generic kernel inside
`token.vectorize(...)`, so that the loop is compiled with the token's target
features. Do not detect or dispatch per edge or per vector. `Scalar` runs the
existing chutoro code unchanged, never a Fearless fallback token.

**Injected selection for tests.** Add `pub(crate)` variants that take an
explicit `EdgeKernelBackend`, used by tests and by the profiled path:
`parallel_kruskal_from_edges_with` and `mutual_reachability_edges_with`.
Production calls the cached selection. Tests never mutate the environment.

**Backend enablement.**

| Backend                    | Compiled        | Selected in tests    | Executed here     | Enabled in production                                                          |
| -------------------------- | --------------- | -------------------- | ----------------- | ------------------------------------------------------------------------------ |
| `Scalar`                   | Yes             | Yes                  | Yes               | Always available                                                               |
| `Avx2` (x86-64-v3)         | Yes             | Yes                  | Yes, Ryzen 9 3900 | Only after the differential suite passes on this host                          |
| `Avx512` (Ice Lake bundle) | Yes             | Selection logic only | **No** hardware   | **No**, until executed differential evidence exists on Ice Lake-class hardware |
| `Neon` (AArch64)           | Not by any gate | No                   | **No** runner     | **No**, until an AArch64 host runs the suite                                   |

_Table 6: Report compilation, selection and execution separately; do not
conflate them._

**Tests.**

- Differential tests against the scalar oracle, for each backend that executes
  on the host. Cases cover lengths 0, 1, `L - 1`, `L`, `L + 1`, `2L + 3` and
  several blocks; masks that reject everything, keep everything, or mix at
  densities of 1/8, 1/2 and 7/8; and `+0.0`, `-0.0`, quiet NaN, a signalling NaN
  (`f32::from_bits(0x7F80_0001)` and its negation), `±inf`, `f32::MAX`,
  `f32::MIN` and the smallest subnormal, in every operand position. Also cover
  invalid endpoints at the first, middle and last lanes and in the remainder;
  both endpoints invalid together with a non-finite weight (precedence);
  duplicates; disconnected graphs; and identifiers `1 << 32` and
  `(1 << 33) + 1` with `node_count = 10`, which would truncate to valid `u32`
  values. Compare weights with `to_bits()`, compare accepted records exactly,
  and compare errors by full `MstError` or `ChutoroError` value.
- A proptest property comparing whole `parallel_kruskal_from_edges_with`
  results, scalar against every available backend, in one-thread and
  eight-thread pools.
- Selection tests over injected `EdgeKernelBackend` values: scalar-only,
  restricted, and automatic selection on the host.
- `cargo check -p chutoro-core --no-default-features`.
- An end-to-end check: a fixed `chutoro-cli` clustering result identical
  before and after.
- A Kani harness, `mst/kani_batch_arithmetic.rs`, for the chutoro-owned
  arithmetic only. It proves that `chunks_exact(L)` plus the remainder covers
  `[0, len)` exactly once, and that bitmask combination maps sub-mask bit `j`
  of part `k` to logical lane `k * U + j`, with no bit at or above `L`. It does
  not verify the library or the hardware. Its name matches `kani-pr.yml`'s
  `**/kani_*.rs` filter. Add it to `make kani` only if it runs within Tolerance
  7.

**Benchmarks and stop criteria.**

- Run interleaved comparisons of the `pipeline` target and the profiled
  `mutual_reachability`, `kruskal` and Gate B sub-stages on the Table 4
  workload, for scalar against each production-enabled backend.
- Keep the kernel only if the stage improvement's confidence interval excludes
  zero and exceeds the unchanged-code noise band, at n = 10 000.
- Report n = 1000 separately.
- Claim no pipeline speed-up unless the `pipeline` interval resolves one.
- A failed or inconclusive result removes the kernel, the dependency entries
  and the lockfile changes in the same branch, and records the null result. No
  unused backend may remain.

### Milestone 7: documentation, roadmap and final gates

Append an `_Implementation update (<date>)._` paragraph to
`docs/chutoro-design.md` §6.3, in the register of the 2026-09-20 and 2026-11-12
entries, covering:

- the Gate A result, with its noise band;
- the Gate B result, if measured;
- what shipped, and under which issue;
- every declined milestone, with its deciding number;
- the transform error-selection strengthening;
- that the union-find arrays already satisfy §6.3's structure-of-arrays
  wording.

If PR #453 has merged by then, place the paragraph so that it reads alongside
§6.3.1. Write an ADR only if Milestone 5 or 6 shipped something structural; a
null result needs only the design-document update, per ADR-0003.

Tick roadmap item 2.3.2 with a one-line outcome note. Do not touch 2.2.8. Update
`docs/users-guide.md` only if Milestone 3's labels shipped, or if error
wording changed.

```sh
set -o pipefail
make check-fmt    2>&1 | tee /tmp/check-fmt-chutoro-2-3-2-final.out
make typecheck    2>&1 | tee /tmp/typecheck-chutoro-2-3-2-final.out
make lint         2>&1 | tee /tmp/lint-chutoro-2-3-2-final.out
make test         2>&1 | tee /tmp/test-chutoro-2-3-2-final.out
make markdownlint 2>&1 | tee /tmp/markdownlint-chutoro-2-3-2-final.out
make nixie        2>&1 | tee /tmp/nixie-chutoro-2-3-2-final.out
make spelling     2>&1 | tee /tmp/spelling-chutoro-2-3-2-final.out
```

Then run a CodeRabbit round to convergence through `comenq-coderabbit`. The
GitHub app skips draft pull requests, but the local CLI does not.

## Concrete steps

Commit after each item with the `commit-message` skill, and land any
post-commit refactor separately.

1. `Record Gate A for MST preparation on the current tree` (plan update)
2. `Compare exact MST edge lists against the oracle`
3. `Specify MST edge preparation behaviour with BDD scenarios`
4. `Report the lowest-index missing core distance deterministically`
5. `Deduplicate MST candidates before sorting` (conditional), or a plan update
   recording why it was declined
6. Gated steps from Milestones 3 to 6, each with its own red/green evidence
7. `Record the 2.3.2 outcome and close the roadmap item`

The per-commit gate is `make check-fmt`, `make typecheck`, `make lint` and
`make test`, each piped through `tee` into a `/tmp` log named after the short
SHA. Documentation-only commits run `make markdownlint`, `make nixie` and
`make spelling`. Stage files first, because `spelling-phrase-check` walks
`git ls-files`.

## Validation and acceptance

- **Equivalence:** the strengthened oracle property passes and fails under the
  deliberate mutation. The oracle-equivalence, structural, concurrency-safety,
  thread-pool-determinism and lowest-index-error properties pass at
  `PROPTEST_CASES=25000`.
- **Determinism:** the existing two-pool tests pass unchanged; the new
  transform test passes in one-thread and eight-thread pools.
- **Behaviour:** all six BDD scenarios pass.
- **Gates recorded:** Gate A (and Gate B where reached), with workload, SHA,
  host, threads, backend, runs and noise band.
- **Kernel (only if shipped):** every Milestone 6 test and stop criterion is
  satisfied; Table 6 is filled in from evidence; MSRV, `cpu`-disabled,
  build-time and binary-size results are recorded.
- **Gates:** `make check-fmt`, `make typecheck`, `make lint`, `make test`,
  `make markdownlint`, `make nixie` and `make spelling` pass.

Claim no speed-up, execution coverage or passing gate without its recorded
result.

### Gate disposition for the literal `make test`

`make check-fmt`, `make typecheck`, `make lint`, `make markdownlint`,
`make nixie` and `make spelling` were each run to completion on this work and
pass. `make test` was **not** produced as a literal local run, and this plan
does not claim one.

The reason is the repository's own accounting, not a test failure. `make test`
selects the `default` nextest profile, which runs setup scripts before any test:
`warm-benches`, `warm-compile-contracts` and `warm-compile-contracts-default`,
with allowances of 1,800 s, 1,800 s and 600 s. On the development host the
shared compile pool has been saturated for days (up to several hundred queued
compile requests from other agents), and the all-features compile-contract
script needs roughly 5,800 s of wall clock there, so the script expires at
1,800 s. The repository's developers' guide, under "Test timeouts: four tiers,
outermost last", states that such an expiry is "a budget, not a detected hang",
and explains those allowances exist so that a saturated pool surfaces in the
setup script rather than as a red test.

What stands in its place, all on recorded heads:

- Every test the command would run was executed locally in partitions: 1,630 of
  1,630 outside `chutoro-compile-contracts` and `benchmark_smoke`, 1 of 1
  `benchmark_smoke`, all 21 compile-contract tests with the gate's exact
  `RUSTFLAGS`, the MST scenario target 6 of 6, and every doctest binary.
- The hosted `build-test` job runs the same nextest suite under the same
  setup scripts and then `make test-doc`, alongside `make lint`,
  `make spelling`, `make test-workflow-contracts` and
  `make test-scheduling-contract`. Its nextest summary was 1,642 passed and 1
  skipped on head `4e96416`, and the job was green there and on head `42f9e8c`.
  Its differences from the literal command are coverage instrumentation and the
  coverage action's target selection.

Head `9cc610e` is recorded as a **red** hosted result rather than a green one:
adding `mst_scenario_target_requires_the_cpu_feature` put a nested-build test
into `chutoro-compile-contracts` without adding it to the pinned set in
`tests/workflow_contracts/trybuild_override_test.py`, so
`make test-workflow-contracts` failed with "the discovery must find exactly…".
The pinned set now names the new test, and that contract file passes 50/50
locally. Any later head's result belongs to that head.

A reader who requires the literal command has one remaining step: run
`make test` on a host whose compile pool is not saturated. Nothing in this
branch is expected to fail there, because no test in the suite is red.

**Gate state: pending.** This plan does not record a passing literal run, does
not record a maintainer-approved equivalence, and does not treat the
substitutes above as the gate. They are the evidence available; the gate closes
when the command passes.

Claim no speed-up, execution coverage or passing gate without its recorded
result.

## Idempotence and recovery

Every step can be re-run. `target/criterion/`, `target/benchmarks/` and the
emitted `target/release/deps/*.s` assembly files may be deleted freely. Note
that a stale `target/criterion/` baseline makes Criterion plot a change against
an unrelated measurement, which can panic in plotters on a degenerate near-zero
timing; deleting the directory is the fix. A failed gated milestone is reverted
in its own commits, dependency entries included. On a gate failure, read the
`tee`d log rather than re-running the gate.

## Artefacts and notes

Revision 2's unchanged-code `mst` runs, which show why stored baselines are not
evidence:

```plaintext
parallel_kruskal/n=500   time: [1.0052 ms 1.0941 ms 1.2182 ms]
                         change: [+10.059% +16.669% +26.638%] (p = 0.00 < 0.05)
                         Performance has regressed.
parallel_kruskal/n=1000  time: [1.8448 ms 1.8781 ms 1.9128 ms]
                         change: [+9.4250% +11.801% +14.223%] (p = 0.00 < 0.05)
                         Performance has regressed.
```

Prior art:

- Osipov, Sanders and Singler, "The Filter-Kruskal Minimum Spanning Tree
  Algorithm", ALENEX 2009: `partition` and `filter` together, not `filter`
  alone.
- Fearless SIMD 1.2.0 changelog and `SimdBase` documentation (see "Governing
  references").
- Giesen, "Order-preserving bijections", and the IEEE 754 radix-sort key.

## Open questions for the maintainer

Both questions are moot unless Gate A passes (decision of 2026-11-23). If it
fails, they close with the plan.

- **Q1.** Should a transform-only SIMD kernel be allowed when Gate A passes but
  Gate B fails? Revision 4 says no, to keep the pre-registered rule. Changing
  it needs a documented decision that defines its own threshold.
- **Q2.** Is the Milestone 3 instrument's addition of nested `stage` labels
  acceptable, given that the users' guide calls the set fixed? This only arises
  if Gate A passes.

## Out of scope

- The 2.2.8 dense query-to-points port, and any change under
  `chutoro-providers/dense/**`.
- The discarded sort in `EdgeHarvest::from_parallel_inserts` (Constraint 10).
- The serial incremental session core-distance path.
- HNSW build performance, which is most of the pipeline, under its own roadmap
  item.
- An ARM CI runner.

## Outcomes & retrospective

Complete this at each milestone and at close: the Gate A and Gate B numbers,
what shipped and under which issue, what was declined and on what number, and
whether the scalar loops were already vectorized.

Interim note (2026-11-22): the cheap structural work landed through four
separate issues rather than through this plan. Whether anything expensive
remains is now a single pending measurement.

Interim note (2026-11-23): **Gate A fails.** On the Table 4 workload,
`(T_mutual_reachability + T_kruskal) / T_whole_run` is 0.1640% to 0.6895% at n
= 10 000 across five four-thread runs, with a median of 0.2345%; at n = 1000
the same ratio is 0.7130% to 2.4731%. One-thread and eight-thread runs give
0.2706% and 0.1499% at the decision size. Gate B, the packed sort record,
Filter-Kruskal and the Fearless SIMD kernel are all declined on that number, so
no dependency or manifest change was made. The scalar loops are scalar at both
codegen levels: the transform's per-edge work compiles to two `maxss` (default)
or `vmaxss` (`x86-64-v3`) instructions, and no packed float max appears
anywhere in the crate. What remains is the Milestone 2 guards and the Milestone
7 record.

Close (2026-11-23). **Roadmap item 2.3.2 closes as a measured null result.**

- **Gate A:** failed, on two measurements at different heads. At `7368d8d`,
  decision-size ratios across five four-thread runs were 0.2345%, 0.5458%,
  0.1640%, 0.1812% and 0.6895% — minimum 0.1640%, median 0.2345%, maximum
  0.6895% — with 0.7130% to 2.4731% at `n = 1000` and supplementary runs at
  0.2706% (one thread) and 0.1499% (eight). At `4e96416`, after the transform's
  collection changed, the five runs gave 0.0512%, 0.0769%, 0.1063%, 0.1417% and
  0.3024% — minimum 0.0512%, median 0.1063%, maximum 0.3024% — with 0.1518% to
  0.3480% at `n = 1000`. The Criterion `pipeline/n=10000` intervals spanning 43
  s to 96 s set the noise band; it cannot close a four-point gap on either set.
- **Gate B:** not measured. It is gated on Gate A, which failed.
- **Assembly:** the transform's per-edge loop is scalar. Two `maxss` per edge
  at the SSE2 baseline, two `vmaxss` under `-C target-cpu=x86-64-v3`, and no
  `maxps`, `vmaxps`, `cmpps` or `vpcmpgtq` float form anywhere in the crate.
- **Shipped** (`chutoro-core`, no issue of its own; the plan's Milestone 2):
  the exact-edge-list oracle property, six BDD edge-preparation scenarios, and
  deterministic lowest-index lookup errors in the mutual-reachability transform.
- **Declined on the Gate A number:** the Gate B instrument, the packed sort
  record, Filter-Kruskal and the Fearless SIMD kernel. No dependency was added,
  so `fearless_simd` and `fearless_simd_macros` remain absent from the
  manifests and the lockfile.
- **Dedup-before-sort:** declined separately, because its bit-keyed dedup
  would have changed which duplicate survives for `±0.0` and the code would
  have been more complex, not less.
- **Consequences for the open questions:** Q1 and Q2 close as moot, as the
  2026-11-23 authorization decision provided.
- **Where to look next:** the pipeline spends its time in the HNSW build and
  the core-distance searches, not between harvesting and the union-find. The
  §6.3 entry records the same conclusion for future readers.

## Revision note

**Revision 10 (2026-11-26).** Recorded the gate as pending rather than
complete, the red hosted result on `9cc610e` and its workflow-contract repair,
and the head attribution for every hosted figure.

**Revision 9 (2026-11-26).** Recorded the hosted whole-plan assessment's
rework: the Gate A re-measurement at `4e96416`, the direct `prepare_edge_list`
dedup tests and duplicate-heavy fixtures, the transform's lowest-index
property, the compile-contract check that the scenario target requires `cpu`,
the developers' guide entries, and an explicit gate disposition for the literal
`make test` in place of a claim that it ran.

**Revision 8 (2026-11-23).** Recorded the Milestone 2 delivery with its
mutation and red/green evidence, the Milestone 7 documentation and roadmap
updates, and the closure of the plan as a measured null result.

**Revision 7 (2026-11-23).** Recorded the Gate A measurement and its failure,
the assembly inspection, and the decision to decline Milestones 3 to 6.

**Revision 6 (2026-11-23).** Recorded the maintainer's approval of the plan and
the instruction to begin implementation, and set the status to APPROVED before
the first implementation commit.

**Revision 5 (2026-11-23).** Recorded the maintainer's authorization of
production work for Milestones 0 to 2, the expected null-result path, and that
Q1 and Q2 apply only if Gate A passes.

**Revision 4 (2026-11-22).** Revised for the Fearless SIMD direction proposed
in PR #453. The changes:

- Recorded the source baseline, the maintainer's dependency authorization as a
  narrow exemption, and 2.3.2's independence from 2.2.8.
- Withdrew revision 3's evidence-based decline, because Gate A is unmeasured on
  the current tree.
- Defined both gates exactly, including nesting, exclusions, the removed
  `harvest_resort` term, the decision workload, and the pass, fail and
  inconclusive rule.
- Added the semantic oracle, with an exceptional-lane rule for NaN and signed
  zero.
- Recorded two new findings: nondeterministic transform lookup errors, and
  scalar `f32::max`'s own signed-zero non-determinism.
- Replaced the intrinsic-based Milestone 5 with a private Fearless SIMD
  Milestone 6 that specifies representation, batch geometry, dispatch, backend
  enablement, tests and stop criteria.
- Renumbered the milestones: Gate B instrument (3), packed record (4),
  Filter-Kruskal (5), kernel (6), close (7).

**Revision 3 (2026-11-22, uncommitted).** Rebased onto `051d395` and marked
delivered work done.

**Revision 2 (2026-08-16).** Rewritten after a six-lens design review that
measured the target stage and corrected five factual errors in revision 1.
