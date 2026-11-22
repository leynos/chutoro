# Vectorize edge-weight transforms and candidate filtering (2.3.2)

This ExecPlan (execution plan) is a living document. The sections `Constraints`,
`Tolerances`, `Risks`, `Progress`, `Surprises & discoveries`, `Decision log`,
and `Outcomes & retrospective` must be kept up to date as work proceeds.

Status: DRAFT (revision 4, 2026-11-22). Documentation only; no production
change is authorized by this revision.

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

| Area                                                              | State                                  | Evidence or next action                                                        |
| ----------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------ |
| Parallel mutual-reachability transform                            | Done                                   | `cpu_pipeline/mod.rs:125-135`, PR #276 (#204)                                  |
| Redundant `EdgeHarvest::new` sort removed                         | Done                                   | `cpu_pipeline/mod.rs:103-111` passes the transformed `Vec` directly; PR #276   |
| Slice entry point `parallel_kruskal_from_edges(&[CandidateEdge])` | Done                                   | `mst/mod.rs:324-327`; `EdgeHarvest::as_slice` at `hnsw/types.rs:340`           |
| Ordered validation buffer, lowest-index MST error                 | Done                                   | `mst/mod.rs:296-321`, PR #428 (#206)                                           |
| Allocation-free weight-group extension                            | Done                                   | `extend_with_accepted_edges`, `mst/mod.rs:255-267`, PR #279 (#205)             |
| Parallel core-distance searches                                   | Done                                   | `compute_core_distances`, `cpu_pipeline/mod.rs:221-239`, PR #431 (#389)        |
| Two-pool determinism and concurrent union-find tests              | Done                                   | `mst/property/concurrency.rs:101`, `mst/union_find.rs:225-260`, PR #225 (#201) |
| Non-exhaustive error-code enums                                   | Done                                   | `define_error_codes!` in `mst/error.rs:52-66`, PR #427 (#203)                  |
| Users' guide MST error codes                                      | Done                                   | `docs/users-guide.md:531-564`, PR #434 (#207)                                  |
| Pipeline benchmark and stage profile                              | Done                                   | `chutoro-benches/benches/pipeline.rs`, `pipeline-profile` feature, PR #431     |
| Gate A on the current tree                                        | **Outstanding measurement**            | Milestone 1                                                                    |
| Assembly inspection of scalar loops                               | **Outstanding measurement**            | Milestone 1                                                                    |
| Exact-edge-list oracle property                                   | **Remaining**                          | Milestone 2                                                                    |
| BDD scenarios for edge preparation                                | **Remaining**                          | Milestone 2                                                                    |
| Deterministic transform lookup errors                             | **Remaining** (found in this revision) | Milestone 2                                                                    |
| Dedup-before-sort                                                 | **Remaining, exactness-gated**         | Milestone 2                                                                    |
| Gate B sub-stage instrument                                       | **Gated** on Gate A                    | Milestone 3                                                                    |
| Packed sort record                                                | **Gated** on Gate A                    | Milestone 4                                                                    |
| Candidate pre-filter (Filter-Kruskal)                             | **Gated** on Gate A and its own case   | Milestone 5                                                                    |
| Fearless SIMD kernel                                              | **Gated** on Gates A and B             | Milestone 6                                                                    |
| `fearless_simd = "1.2"`, `fearless_simd_macros = "0.1"`           | Authorized, **not added**              | Added only inside Milestone 6, after its gates pass                            |
| Null-result record and roadmap closure                            | **Remaining**                          | Milestone 7                                                                    |

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
- [ ] Milestone 0: baseline gates; record the `make test` count.
- [ ] Milestone 1: Gate A measurement and assembly inspection.
- [ ] Milestone 2: remaining guards.
  - [ ] Exact-edge-list oracle property.
  - [ ] BDD feature and step glue, with pinned fixtures.
  - [ ] Lowest-index lookup errors in the transform.
  - [ ] Dedup-before-sort, or a recorded reason for declining it.
- [ ] Milestone 3 (gated on A): Gate B instrument and measurement.
- [ ] Milestone 4 (gated on A): packed sort record.
- [ ] Milestone 5 (gated on A and its own case): Filter-Kruskal.
- [ ] Milestone 6 (gated on A and B): Fearless SIMD kernel.
- [ ] Milestone 7: design-document update, roadmap and final gates.

## Surprises & discoveries

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
   RUSTFLAGS="--emit=asm -C debuginfo=0" \
     cargo build --release -p chutoro-core --target-dir target/asm-default \
     2>&1 | tee /tmp/asm-default-2-3-2.out
   RUSTFLAGS="--emit=asm -C debuginfo=0 -C target-cpu=x86-64-v3" \
     cargo build --release -p chutoro-core --target-dir target/asm-v3 \
     2>&1 | tee /tmp/asm-v3-2-3-2.out
   ```

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

## Idempotence and recovery

Every step can be re-run. `target/criterion/`, `target/benchmarks/`,
`target/asm-default/` and `target/asm-v3/` may be deleted freely. A failed
gated milestone is reverted in its own commits, dependency entries included. On
a gate failure, read the `tee`d log rather than re-running the gate.

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

## Revision note

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
