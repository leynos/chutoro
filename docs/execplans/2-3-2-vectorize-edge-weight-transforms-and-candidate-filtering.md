# Vectorize edge-weight transforms and candidate filtering (2.3.2)

This ExecPlan (execution plan) is a living document. The sections `Constraints`,
`Tolerances`, `Risks`, `Progress`, `Surprises & discoveries`, `Decision log`,
and `Outcomes & retrospective` must be kept up to date as work proceeds.

Status: DRAFT (revision 2, after community-of-experts design review)

Roadmap item: 2.3.2 (Phase 2, "Hot-path optimizations"). See `docs/roadmap.md`
lines 400-402 and `docs/chutoro-design.md` §6.3 "SIMD utilization", lines
904-908.

## Purpose / big picture

Chutoro clusters data by building an approximate nearest-neighbour graph, then
converting the harvested candidate edges into a mutual-reachability minimum
spanning forest (MSF) with a parallel Kruskal implementation, then extracting a
cluster hierarchy from that forest. Roadmap item 2.3.2 targets the work between
harvesting candidate edges and feeding them to the union-find.

**Read the measured baseline below before doing anything else.** A design
review of this plan's first revision measured the stage this item targets and
found it very small relative to the pipeline. That measurement changes what
this plan should deliver, and it is stated first so nobody spends effort on the
strength of the roadmap wording alone.

### Measured baseline

On the development machine (AMD Ryzen 9 3900, twelve physical cores,
twenty-four threads, AVX2 but **no AVX-512**), with seed 42, sixteen dimensions,
`M = 16`:

| Stage                       | n = 1000                   | Source                |
| --------------------------- | -------------------------- | --------------------- |
| `CpuHnsw::build_with_edges` | 853.60 ms                  | measured, review      |
| `parallel_kruskal`          | 1.8781 ms [1.8448, 1.9128] | measured, this branch |
| MST share of HNSW build     | ≈ 0.22%                    | derived               |

_Table 1: Measured stage costs at one thousand points. The MST stage is roughly
one five-hundredth of the HNSW build that precedes it._

The share falls as `n` grows, because HNSW build is superlinear and Kruskal is
near-linear. Extrapolating the measured `parallel_kruskal` points (n = 100:
424.82 µs; n = 500: 1.0941 ms; n = 1000: 1.8781 ms) gives roughly 13-17 ms at n
= 10 000, against the 20.208-second ten-thousand-point HNSW build recorded in
`docs/chutoro-design.md` §6.3 for roadmap item 2.3.1 — about 0.07%. At one
hundred thousand points the same source records a 781.631-second build, against
an extrapolated 150-200 ms of MST: about 0.025%.

**Therefore: deleting the entire MST stage would buy under a quarter of one
percent end to end, and less as datasets grow.** Any plan for 2.3.2 that
proposes a structure-of-arrays staging layer, a backend dispatch table, three
sets of handwritten intrinsics, a parity suite, a Verus proof and an
architecture decision record is spending thousands of lines per hundredth of a
percent. The first revision of this plan proposed exactly that. It was wrong to.

### What this plan therefore delivers

1. **An instrument that settles the question**, including an end-to-end
   pipeline group so the denominator is measured rather than assumed, and a
   benchmarking configuration that can actually resolve the effects being
   claimed. See Milestone 1.
2. **The structural work that is worth doing on its own merits** — deleting a
   full redundant sort of the whole edge list, deleting a whole-list
   intermediate allocation, parallelizing a serial transform, hoisting a
   per-weight-group allocation, and making error reporting deterministic. This
   is roughly 150 net lines across two production files, needs no new
   abstraction, and is where the achievable win actually lives. See Milestone 2.
3. **Pre-registered, individually gated milestones** for everything expensive —
   the packed sort record, the SIMD kernels, and the candidate pre-filter —
   each with its own threshold and each with a documented null-result path,
   following the precedent ADR-003 set for roadmap items 2.3.3 to 2.3.5.
   Several of these are expected to end as null results, and that is a
   successful outcome, not a failure.

After Milestone 2 a developer can observe:

- `cargo bench -p chutoro-benches --bench mst` reports lower times for the
  `parallel_kruskal` group at n = 500 and n = 1000, outside the measured noise
  band, and a new `cpu_pipeline` group reports the end-to-end cost so the MST
  share is a number rather than a guess.
- `scripts/bench-mst-pipeline.sh` corroborates end to end through `hyperfine`
  on the command-line interface, where the effect size is large enough for
  whole-binary wall-clock timing to resolve it.
- Clustering output is unchanged, now guarded by a property that compares the
  **exact edge list** against the sequential oracle. The present suite compares
  only total weight, edge count and component count, so a reordering regression
  would pass today.
- Error reporting is deterministic: with several invalid edges, the reported
  `MstError` is always the one at the lowest input index, at any thread count.

## Relevant documentation and skills

- `AGENTS.md` — commit discipline, quality gates, the 400-line file cap, the
  abstraction/port/helper sweep policy, en-GB Oxford spelling for comments, the
  ban on environment mutation inside tests, and the `tee`-to-`/tmp` logging
  convention.
- `docs/roadmap.md` §2.3 — the roadmap item being delivered.
- `docs/chutoro-design.md` §3.2, §6.2 (parallel Kruskal sketch, lines 804-886),
  §6.3 (SIMD utilization, lines 887-938 and the implementation-update log).
- `docs/adr-003-soa-prefetch-adapter-boundary.md` — the precedent for gating
  structural change on measurement and recording null results in place.
- `docs/developers-guide.md` — "Benchmarks", especially "Benchmark regression
  workflow" and its standing policy that Criterion is the primary signal and
  `hyperfine` is corroboration.
- `docs/property-testing-design.md`,
  `docs/rust-testing-with-rstest-fixtures.md`,
  `docs/rust-doctest-dry-guide.md`,
  `docs/complexity-antipatterns-and-refactoring-strategies.md`,
  `docs/documentation-style-guide.md`, `docs/contents.md`.

Skills: `leta` first (add the worktree as a workspace), then `rust-router` and
`rust-performance-and-layout`, `rust-unit-testing`, `proptest`, `nextest`,
`execplans`, `commit-message`, `pr-creation`, `en-gb-oxendict`. Load `kani`,
`verus` and `hexagonal-architecture` only if the gated milestones proceed.

## Constraints

Hard invariants. Violation requires escalation, not a workaround.

1. **Output equivalence.** For every input on which `parallel_kruskal` returns
   `Ok` today, it must return an identical `MinimumSpanningForest`: the same
   edge sequence, including each edge's `sequence` value, and the same
   `component_count`. "Accepts" means "returns `Ok`".
2. **Two deliberate, documented divergences, and no others.** Error selection
   narrows from unspecified to lowest-input-index. The dedup and weight-group
   predicates may move from IEEE `==` to an order-consistent rule, which
   differs only when `-0.0` and `+0.0` carry identical endpoints; Milestone 2
   must demonstrate the final forest is unchanged in that case, or keep IEEE
   `==`. Any third divergence is an escalation.
3. **Determinism.** Identical input must produce byte-identical output at any
   Rayon thread count. This is currently untested — see Surprises — and
   Milestone 2 must add the test that makes it real.
4. **No `unsafe`** anywhere in committed scope. Only a gated SIMD milestone may
   introduce it, confined to its adapter module with a documented invariant
   list and parity coverage.
5. **Public API: additions only.** `parallel_kruskal`, `MstEdge`,
   `MinimumSpanningForest`, `CandidateEdge` and `EdgeHarvest` keep their
   signatures. `MstEdge` must not gain a field: it derives `PartialEq` and
   `Debug`, both public and both observable. `MstErrorCode` is **not**
   `#[non_exhaustive]`, so adding a variant to it is breaking and requires the
   preparatory step in Milestone 2.
6. **Minimum supported Rust version stays 1.89.0.** No nightly requirement in
   the default build.
7. **No new dependency** in any crate.
8. **File-size cap:** no source file over 400 lines.
   `chutoro-core/src/mst/mod.rs`
   is at 344 and is being edited; split it if the work crosses the cap.
9. **Dependency direction:** `chutoro-core` must not depend on
   `chutoro-providers-dense`.
10. **Do not modify** `chutoro-core/src/hnsw/**`,
    `chutoro-core/src/hierarchy/**`,
    or `chutoro-providers/dense/**`.

## Tolerances (exception triggers)

1. **Scope.** Milestone 2 exceeding 8 changed files or 250 net lines. Any gated
   milestone exceeding 12 files or 600 net lines.
2. **Interface.** Any change to an existing public signature; any new public
   item beyond `EdgeHarvest::as_slice`.
3. **Dependencies.** Any new manifest entry.
4. **Iterations.** A gate failing after four consecutive fix attempts.
5. **Evidence.** Any gated milestone failing its pre-registered threshold. Stop,
   record the null result beside the roadmap item, and move on. This is the
   expected outcome for at least one milestone.
6. **Ambiguity.** Any place where this plan and the code disagree about current
   behaviour. Revision 1 tripped this five times; see Surprises.
7. **Verification cost.** A Kani harness over fifteen minutes, or a Verus proof
   over five minutes.
8. **Time.** More than four hours on a single milestone.

## Risks

- Risk: the whole roadmap item is not worth its cost. Measured evidence puts
  `parallel_kruskal` at ≈ 0.22% of the HNSW build at n = 1000 and falling.
  Severity: high. Likelihood: high. Mitigation: this is the plan's central
  assumption, not a footnote. Milestone 1 measures the pipeline-relative share
  directly. Milestone 2 is scoped to work that is cheap and correct regardless.
  Everything else is individually gated and expected, in part, to be declined.

- Risk: the benchmark cannot resolve the effects being claimed.
  Severity: high. Likelihood: **confirmed, not hypothetical**. Evidence: running
  `cargo bench -p chutoro-benches --bench mst` twice on identical code produced
  `change: [+9.4250% +11.801% +14.223%] (p = 0.00)` at n = 1000 and
  `[+10.059% +16.669% +26.638%] (p = 0.00)` at n = 500, both reported as
  "Performance has regressed", with three outliers among twenty samples.
  Revision 1's acceptance criterion of "no group regressing by more than 3%"
  would fire spuriously on every run. Mitigation: Milestone 1 fixes the
  methodology before any production edit — larger sample sizes, longer
  measurement windows, pinned cores and thread counts, interleaved A/B rather
  than stored baselines, and n = 100 demoted to a tripwire.

- Risk: a SIMD `max` does not match `f32::max`. `_mm256_max_ps(a, b)` returns
  `b` when either operand is NaN, so `max(5.0, NaN)` is `NaN` under the
  intrinsic and `5.0` under `f32::max`. Through the mutual-reachability
  transform that turns an accepted edge into a `NonFiniteWeight` error.
  Severity: high. Likelihood: high if a SIMD milestone proceeds. Mitigation:
  the transform's NaN and signed-zero semantics must be stated normatively and
  differentially tested before any intrinsic is written.

- Risk: narrowing node ids to `u32` silently accepts out-of-range edges.
  `CandidateEdge::source()` is public `usize`;
  `CandidateEdge::new(1 << 33, 1, 0.5, 0)` with `node_count = 10` correctly
  errors today, but truncates to a valid id `0` if narrowed before validation.
  Severity: high. Likelihood: medium. Mitigation: validate endpoints against
  `node_count` in `usize` space, before any narrowing. Note that the workspace
  Clippy table is **not** active in `chutoro-core` (see Surprises), so
  `cast_possible_truncation` provides no automated guard here.

- Risk: padding lanes poison classification. Sentinels of `u32::MAX` and
  `f32::INFINITY` are exactly the two values the validator rejects, so a
  branch-free kernel classifying `padded_len()` lanes reports a padding index
  as the first rejection on every input. Severity: high. Likelihood: high if a
  SIMD milestone proceeds. Mitigation: pad index arrays with `0` and weights
  with `0.0` (an inert self-loop, which the policy drops), mirroring the
  zero-padding precedent in `chutoro-providers/dense/src/simd/point_view.rs`;
  and restrict any min-rejection reduction to `[0, len)`.

- Risk: pre-merge formal-verification signal is nil. `make kani` appears in no
  workflow; `make kani-full` runs post-merge on `main` only, and
  `docs/kani-full-hnsw-hypothesis-testing.md` records it as currently blocked.
  Severity: medium. Likelihood: high. Mitigation: do not rest any acceptance
  criterion on Kani. `make verus` **is** a pull-request gate (`ci.yml`), so a
  Verus obligation — if one is introduced at all — is the only formal signal
  that blocks a merge.

- Risk: the coverage ratchet (`ci.yml`, `with-ratchet: 'true'`) fails on added
  code that the runner cannot execute, consuming the iteration tolerance for
  reasons unrelated to correctness. Severity: medium. Likelihood: medium if a
  SIMD milestone proceeds. Mitigation: keep committed scope small; treat a
  ratchet failure on hardware-gated code as a Tolerance 5 evidence event, not a
  bug to fight.

- Risk: `googletest`, `pretty_assertions` and `insta` are named in `AGENTS.md`
  but absent from this workspace entirely. Severity: low. Likelihood: high.
  Mitigation: see the decision log. House style is plain assertions plus named
  helpers, and this plan follows it.

## Progress

Timestamps are mandatory: Tolerance 8 is a time tolerance and cannot be
enforced without them. Split any partially completed item into "done" and
"remaining" rather than leaving one checkbox ambiguous.

- [x] (2026-08-16) Revision 1 drafted.
- [x] (2026-08-16) Six-lens community-of-experts design review completed;
      findings recorded in `Surprises & discoveries` and `Decision log`.
- [x] (2026-08-16) Measured `parallel_kruskal` on this branch at n = 100, 500
      and 1000; recorded the noise band in `Artefacts and notes`.
- [x] (2026-08-16) Verified `weight_key` against `f32::total_cmp` over 14 400
      ordered pairs; zero mismatches.
- [x] (2026-08-16) Revision 2 rewritten against the review.
- [ ] Milestone 0: orientation and baseline gates.
  - [ ] Record the baseline `make test` count (expected: 1058 passed,
        1 skipped, per the 2.3.1 plan's recorded baseline — confirm and
        correct if it has moved).
  - [ ] `make check-fmt`, `make lint`, `make typecheck`, `make test` all green.
- [ ] Milestone 1: measurement instrument and pipeline-relative go/no-go.
  - [ ] Extract the union-find loop as a behaviour-preserving refactor.
  - [ ] Add `chutoro-benches/benches/mst_prepare.rs` with five groups.
  - [ ] Add and `shellcheck` `scripts/bench-mst-pipeline.sh`.
  - [ ] Record the pipeline-relative ratio and the go/no-go outcome.
- [ ] Milestone 2: committed structural work, with red tests first.
  - [ ] Red: exact-edge-list oracle property.
  - [ ] Red: two-thread-pool determinism test.
  - [ ] Red: lowest-index error-selection test (record the failing output).
  - [ ] Red: BDD feature and step glue, with fixtures fully specified.
  - [ ] Green: the six production edits, one commit each.
- [ ] Milestone 3 (gated): packed sort record and integer key.
- [ ] Milestone 4 (gated): candidate pre-filter, only as real Filter-Kruskal.
- [ ] Milestone 5 (gated): SIMD kernels.
- [ ] Milestone 6: documentation, roadmap update, and final gates.
  - [ ] CodeRabbit review round, run to convergence.

## Surprises & discoveries

Revision 1 of this plan asserted several things about the code that are false.
They are recorded here because each one changed a design decision, and because
Tolerance 6 makes plan-versus-code disagreement an escalation trigger.

- Observation: **`try_union` takes no lock on the cycle-rejection path.**
  Evidence: `chutoro-core/src/mst/union_find.rs:49-56` calls `find` twice and
  returns `Ok(false)` at the `left_root == right_root` check, before
  `lock_order` or `lock_root` are reached. Impact: revision 1 justified the
  entire candidate pre-filter on rejected edges paying "two path-compressing
  `find` walks … inside a mutex-protected `try_union`". They pay two
  path-halved walks and no mutex at all. The filter's justification collapses;
  it is demoted to a gated milestone that must earn its place by measurement.

- Observation: **the workspace Clippy table is not active in `chutoro-core`.**
  Evidence: `Cargo.toml` lines 25-89 define `[workspace.lints.clippy]`, but only
  `chutoro-bench-datasets/Cargo.toml` carries `[lints] workspace = true`.
  `chutoro-core/Cargo.toml:43-44` declares only `[lints.rust] unexpected_cfgs`.
  Impact: `cast_possible_truncation`, `indexing_slicing`, `float_arithmetic` and
  `unwrap_used` are inert in this crate. Revision 1's stated defence against
  silent `usize`-to-`u32` truncation did not exist. The same gap applies to
  `chutoro-cli`, `chutoro-providers/dense`, `chutoro-providers/text` and
  `chutoro-test-support`, which carry no `[lints]` section at all. Tracked as
  issue #200; opting the crates in is out of scope here and would not be quiet.

- Observation: **the concurrency property suite does not exercise concurrency.**
  Evidence: `run_concurrency_safety_property`
  (`chutoro-core/src/mst/property/concurrency.rs:22-93`) calls
  `parallel_kruskal` five times sequentially on one thread. No test under
  `chutoro-core/src/mst/` spawns threads. `ci.yml:24` and
  `coverage-main.yml:30` both pin `RAYON_NUM_THREADS: '1'`. Impact:
  `ConcurrentUnionFind`'s lock protocol, retry loop and memory orderings have
  never been exercised concurrently. Constraint 3 is unenforced today. Revision
  1 claimed this suite guarded the striped-lock design; it does not. Milestone
  2 adds the test that makes Constraint 3 real.

- Observation: **`make kani` runs in no workflow, and `make kani-full` is
  currently blocked.** Evidence: the only workflow reference is
  `nightly-kani.yml:39`, running `make kani-full` against `ref: main`
  post-merge. `docs/kani-full-hnsw-hypothesis-testing.md` §"Current conclusion"
  records it failing on CBMC budget exhaustion in string-panic unwinding.
  Impact: no Kani harness can be an acceptance criterion for a merge.

- Observation: **`MstErrorCode` is public, re-exported, and not
  `#[non_exhaustive]`**, unlike `MstError`. Evidence:
  `chutoro-core/src/mst/mod.rs:19` versus `:73-74`; both re-exported at
  `chutoro-core/src/lib.rs:51`. Note `chutoro-core/src/error.rs:11-59` has a
  `define_error_codes!` macro that emits `#[non_exhaustive]`; `MstErrorCode`
  bypasses it. Impact: adding a variant is breaking for downstream exhaustive
  matches.

- Observation: **the mutual-reachability transform is not in `mst/`**, and the
  harvest is sorted twice. Evidence: `chutoro-core/src/cpu_pipeline.rs:79-88`
  holds the transform; line 89 wraps the result in `EdgeHarvest::new`, which
  sorts by `(sequence, natural Ord)` at `chutoro-core/src/hnsw/types.rs:270`;
  and `chutoro-core/src/mst/mod.rs:283` immediately re-sorts by weight. Impact:
  one full sort of the entire edge list is pure waste and is the single largest
  item in this plan. A third sort exists inside
  `EdgeHarvest::from_parallel_inserts`, but Constraint 10 forbids touching
  `hnsw/**`, so it is not recoverable here.

- Observation: **`process_weight_group` allocates once per distinct weight.**
  Evidence: `chutoro-core/src/mst/mod.rs:248` —
  `let mut accepted = Vec::new();` inside a function called once per
  equal-weight group, consumed by `forest_edges.extend(accepted)` at line 321.
  Mutual-reachability weights are `f32`, so most groups have one member.
  Impact: tens of thousands of allocation and free pairs at n = 10 000. Not
  mentioned in revision 1; now part of committed scope.

- Observation: **rayon's `try_reduce` error selection is nondeterministic by
  construction**, not merely undocumented. It is left-biased only when both
  sides are `Break`; the first `Break` sets a shared abort flag, other folders
  return partial `Continue` values, and a later-index error can therefore win.
  Impact: revision 1 proposed discovering this empirically. A 64-run probe on a
  small input can pass by luck and record a false conclusion. It is decidable
  from the source and is recorded here instead.

- Observation: **there are three MST properties, not four**
  (`chutoro-core/src/mst/property/mod.rs`), and
  `test_cases_count_matches_macro_expectations` guards the eleven-entry case
  list, not the property count.

- Observation: **the development machine has no AVX-512.**
  Evidence: `/proc/cpuinfo` reports no `avx512f` on the Ryzen 9 3900 (Zen 2).
  Impact: an AVX-512 backend would sit at the top of the dispatch priority,
  ship to users, and never be executed or parity-tested locally.

- Observation: **no workflow compiles NEON, and the parity suite passes when
  no SIMD backend runs.** Evidence: every job in `.github/workflows/` uses
  `ubuntu-latest` or `ubicloud-standard-8`, both x86_64, while the dense
  provider's NEON kernels are gated on `target_arch = "arm"`/`"aarch64"` — so
  they have never been compiled, linted or executed by any gate, despite
  `simd_neon` being default-on. Separately, `dispatch::enabled_backends()`
  intersects compiled features with runtime CPUID and the parity suite only
  asserts the result is non-empty, which `Scalar` alone satisfies. Impact: a
  SIMD milestone here would add NEON kernels to that void inside the crate
  every consumer depends on, and its parity suite could pass having exercised
  nothing. If Milestone 5 proceeds, it must exclude NEON absent an ARM runner,
  and must assert that the intended backend actually ran.

- Observation: **the users' guide has no MST error section at all.** "Error
  handling" (`docs/users-guide.md`) documents only `ChutoroError` and
  `DataSourceError`. The surface most users actually see is
  `ChutoroError::CpuMstFailure { code }`.

- Observation: **`googletest`, `pretty_assertions` and `insta` are absent** from
  `Cargo.lock` and from every source file.

## Decision log

- Decision: lead the plan with the measured MST share and scope the work to
  match, rather than to the roadmap wording. Rationale: `parallel_kruskal` is ≈
  0.22% of the HNSW build at n = 1000 and falling. ADR-003 exists precisely to
  stop plausibility-driven structural change; applying it to this item means
  the expensive parts must be gated and may well be declined. Date/Author:
  2026-08-16, planning agent.

- Decision: drop the structure-of-arrays staging layer, the policy value
  object, the kernel function-pointer table and the backend dispatch enum from
  committed scope. Rationale: five parallel `Vec`s cannot be sorted —
  `par_sort_unstable` is a slice method — so the design silently required
  either a permutation sort with five gathers or a round trip back to an
  array-of-structures record, and the latter is what a packed record gives
  directly and more cheaply. The policy object could not carry an `MstError`
  through a lane-oriented kernel, so it would have been decorative at exactly
  the boundary it existed to protect. This is the "pattern transplant" failure
  the hexagonal-architecture guidance warns against. Date/Author: 2026-08-16,
  planning agent, after design review.

- Decision: demote the candidate pre-filter to a gated milestone, and require
  it to be real Filter-Kruskal (partition, then filter) if it proceeds at all.
  Rationale: its stated justification was factually wrong about `try_union`.
  Filter-Kruskal's saving is asymptotic in the _sort_ — filtered edges are
  never sorted. Taking `filter` without `partition` keeps the sweep cost and
  discards the win. The estimated net effect of the post-sort variant at n = 10
  000 is a loss of two to three milliseconds. It is also strictly worse on
  disconnected graphs, the common HDBSCAN case, because `is_mst_complete`
  (`mst/mod.rs:257-263`) requires `components() == 1` to break early, so the
  loop must scan every edge while the filter's halving trigger has already
  stopped firing. Date/Author: 2026-08-16, planning agent, after design review.

- Decision: do not add a `key` field to `MstEdge`.
  Rationale: `MstEdge` derives `PartialEq` and `Debug` and both are public and
  observable. A key field would make `weight: -0.0` and `weight: 0.0` compare
  unequal where they compare equal today, and would change `{:?}` output. Any
  packed record stays internal to the preparation stage and is converted to
  `MstEdge` at the end. Date/Author: 2026-08-16, planning agent, after design
  review.

- Decision: do not narrow the union-find's parent and rank arrays.
  Rationale: at eight bytes per parent the level-two cache crossover is around
  n = 65 536, more than six times the largest benchmark point, so the claimed
  cache benefit is unfalsifiable by this plan's own instrument. It would also
  have cost a breaking `MstErrorCode` variant and a users'-guide entry to guard
  a `node_count > u32::MAX` condition that needs roughly 1.5 TB of memory to
  reach. §6.3 asks for "cache-friendly structure-of-arrays parent and rank
  arrays"; `union_find.rs:20-22` already stores parents and ranks as separate
  arrays, so the constraint is satisfied. Milestone 6 documents that rather
  than churning it. Date/Author: 2026-08-16, planning agent, after design
  review.

- Decision: keep `hyperfine` in the plan, but point it at the command-line
  interface end to end rather than at a Criterion binary. Rationale: the task
  requires a `hyperfine` validation. At ten runs of whole-binary wall time it
  has no power to resolve a sub-millisecond effect inside a Criterion harness
  dominated by setup builds and warm-ups. End to end over the CLI, the effect
  is measured against something `hyperfine` can see. `docs/developers-guide.md`
  already states that Criterion is the primary signal and `hyperfine` is
  corroboration. Date/Author: 2026-08-16, planning agent, after design review.

- Decision: follow the repository's actual assertion style rather than adopting
  `googletest` and `pretty_assertions`; no `insta`. Rationale: none of the
  three appears anywhere in this workspace. Adding two test-framework
  dependencies as a side effect of a hot-path optimization is a cross-cutting
  change with its own review surface and breaches Tolerance 3. `AGENTS.md`
  scopes `insta` to "multivariant output format consistency"; this work
  produces a `Vec<MstEdge>`, and exact structural equality is the stronger
  assertion. If the user wants those crates adopted, that is a separate,
  workspace-wide change. Date/Author: 2026-08-16, planning agent.

- Decision: add a doctest only to `EdgeHarvest::as_slice`, and to no other new
  item. Rationale: `AGENTS.md` requires examples in function documentation, but
  every other item this plan adds is `pub(crate)` or private, and rustdoc does
  not run doctests on non-public items, so they would be unexecuted prose.
  `as_slice` is the one genuinely public addition. Recorded rather than
  silently omitted. Date/Author: 2026-08-16, planning agent, after design
  review.

- Decision: register this plan in `docs/contents.md` now, not at Milestone 6.
  Rationale: the plan explicitly contemplates halting at the Milestone 1
  go/no-go. Deferring registration means that in the most likely outcome the
  document is never indexed at all. Date/Author: 2026-08-16, planning agent,
  after design review.

- Decision: introduce no Verus obligation unless Milestone 4 proceeds, and if
  it does, target **root-identity monotonicity** rather than partition
  coarsening. Rationale: revision 1's proposed lemma had `ensures` as a direct
  instantiation of its own `requires`, so Verus would discharge it in one step
  — the restatement that `AGENTS.md` explicitly forbids. It also modelled an
  abstract partition with no root identifiers, whereas the invariant the filter
  actually depends on is that `parents[n] == n` is a one-way transition, so a
  root identifier is never reused for a different component. That is
  non-trivial and is the load-bearing property. Date/Author: 2026-08-16,
  planning agent, after design review.

## Context and orientation

This section assumes no prior knowledge of the repository.

### Terms

- **HNSW** — Hierarchical Navigable Small World, the approximate
  nearest-neighbour index built first (`chutoro-core/src/hnsw/`).
- **Candidate edge** — a `(source, target, distance, sequence)` record emitted
  during HNSW insertion (`chutoro-core/src/hnsw/types.rs:146-152`). `sequence`
  is a monotonic counter used only for deterministic tie-breaking.
- **Edge harvest** — `EdgeHarvest`, a newtype over `Vec<CandidateEdge>` whose
  constructors always sort by `(sequence, natural Ord)`.
- **Core distance** — the distance from a point to its `k`-th nearest
  neighbour, computed at `chutoro-core/src/cpu_pipeline.rs:65-77`.
- **Mutual reachability** — the FISHDBC edge weight
  `max(d(u, v), core(u), core(v))`, computed at `cpu_pipeline.rs:85`.
- **MSF / MST** — minimum spanning forest, and its connected special case.
- **Union-find** — the disjoint-set structure answering "already connected?"
  (`chutoro-core/src/mst/union_find.rs`).

### Current control flow

`run_cpu_pipeline_with_len` (`cpu_pipeline.rs:47-106`) builds the HNSW index
and harvests edges; computes core distances in a **serial** loop; applies the
mutual-reachability transform in a **serial** `.iter().map()`; builds the MSF;
extracts labels.

`parallel_kruskal` (`mst/mod.rs:188-193`) delegates to
`parallel_kruskal_from_edges` (lines 290-335), which calls `prepare_edge_list`
(lines 265-288) and then walks the sorted list in equal-weight groups, handing
each to `process_weight_group` (lines 241-255), which calls `try_union`
sequentially. `cpu_pipeline.rs:91` is the only production call site;
`chutoro-core/src/session/` does not yet call the MST at all.

## Interfaces and dependencies

Committed scope adds exactly one public item:

```rust
impl EdgeHarvest {
    /// Returns the harvested edges as a contiguous slice.
    #[must_use]
    pub fn as_slice(&self) -> &[CandidateEdge] { &self.0 }
}
```

and changes one crate-internal signature, whose only callers are
`mst/mod.rs:192` and `mst/kani_harness.rs:102,155`:

```rust
pub(crate) fn parallel_kruskal_from_edges(
    node_count: usize,
    edges: &[CandidateEdge],
) -> Result<MinimumSpanningForest, MstError>;
```

`chutoro-core/src/cpu_pipeline.rs` gains one private helper, keeping the
density model in the pipeline where it belongs rather than pushing it into the
MST module:

```rust
/// Applies the mutual-reachability transform to every harvested edge.
///
/// Weight is `max(distance, core[source], core[target])`, using `f32::max`
/// semantics: a NaN operand is ignored unless both are NaN. Signed-zero sign
/// is unspecified, matching the current implementation.
fn mutual_reachability_edges(harvest: &EdgeHarvest, core_distances: &[f32]) -> Vec<CandidateEdge>;
```

Milestone 3, only if gated in, adds a crate-internal packed record. It is never
exposed and never stored in `MstEdge`:

```rust
/// Sort-ready edge record: 24 bytes against `MstEdge`'s 32.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PreparedEdge {
    key: u32,      // order-preserving image of the weight
    source: u32,
    target: u32,
    sequence: u64,
}

/// Maps an `f32` to a `u32` whose unsigned order matches `f32::total_cmp` for
/// every bit pattern, including both signed zeros and both signs of quiet and
/// signalling NaN.
///
/// The predicate is on the **sign bit**, not on numeric negativity: `-0.0` is
/// not less than zero and NaN compares false against everything, so a
/// `weight < 0.0` test produces a silently wrong key.
const fn weight_key(weight: f32) -> u32 {
    let bits = weight.to_bits();
    let sign_mask = 0u32.wrapping_sub(bits >> 31); // all ones iff sign bit set
    bits ^ (sign_mask | 0x8000_0000)
}
```

This formula was verified on this branch against `f32::total_cmp` over 14 400
ordered pairs spanning ±NaN (quiet and signalling), ±infinity, ±0.0, denormals
and a strided sweep of the exponent and mantissa space, with zero mismatches.
It uses no signed casts, so it is safe under the workspace lint table should
`chutoro-core` ever opt in.

## Plan of work

### Milestone 0: orientation and baseline gates

No code changes. Confirm the branch, a clean tree, and passing gates, so later
failures are attributable.

```sh
set -o pipefail
git branch --show-current
make check-fmt 2>&1 | tee /tmp/check-fmt-chutoro-2-3-2-baseline.out
make lint      2>&1 | tee /tmp/lint-chutoro-2-3-2-baseline.out
make typecheck 2>&1 | tee /tmp/typecheck-chutoro-2-3-2-baseline.out
make test      2>&1 | tee /tmp/test-chutoro-2-3-2-baseline.out
```

Expect the last command to end with a summary of the form:

```plaintext
     Summary [  ##.###s] 1058 tests run: 1058 passed, 1 skipped
```

Record the exact count in `Progress`. If it differs from 1058, that is fine —
it is the number every later run is compared against, and a silent drop is what
this step exists to detect.

### Milestone 1: measurement instrument and go/no-go

This milestone must precede every production edit, and revision 1's claim that
it needs none is wrong: timing the union-find loop separately requires
extracting it from `parallel_kruskal_from_edges`. Do that extraction first, as
a pure, behaviour-preserving refactor with its own commit and its own passing
gates, so the instrument is not confounded with the work it measures.

Add `chutoro-benches/benches/mst_prepare.rs` (`harness = false`), following the
fallible-`_impl`-plus-thin-wrapper pattern in `docs/developers-guide.md`
"Adding a new benchmark". Reuse `mst.rs`'s constants and extend point counts to
`[100, 500, 1_000, 10_000]`. Time five groups:

1. `cpu_pipeline_end_to_end` — `run_cpu_pipeline`, so the **denominator is
   measured**. This is the group revision 1 omitted, and its absence is why its
   gate could not fail.
2. `mutual_reachability` — the transform alone.
3. `harvest_resort` — the `EdgeHarvest::new` sort this plan deletes.
4. `prepare_edge_list` — validation, canonicalization, sort, dedup.
5. `union_find_loop` — the weight-group walk.

Group 5 mutates the union-find, so a plain `b.iter()` would re-run it over an
already-merged structure, reject every edge at `union_find.rs:54`, and report
it as near-free — which would inflate groups 2 to 4 and push a naive gate
through. Use `iter_batched` with a fresh `ConcurrentUnionFind` per iteration,
and state explicitly whether its allocation is inside or outside the measured
region.

Fix the methodology, which the evidence in Risks shows is currently incapable
of resolving the claimed effects:

- `sample_size` at least 100 and `measurement_time` at least 10 s for any group
  where a win is claimed.
- Pin cores and threads: `taskset -c 0-11` and an explicit `RAYON_NUM_THREADS`,
  since Rayon otherwise starts twenty-four workers on twelve physical cores for
  a two-millisecond workload.
- Compare interleaved A/B in one session rather than against a `local-reference`
  baseline saved at an arbitrary earlier time.
- Demote n = 100 to a loose tripwire; at ~280 ns per edge it is almost entirely
  Rayon spawn overhead and its confidence interval is ±5% or worse.

Add `scripts/bench-mst-pipeline.sh`, modelled on
`scripts/bench-neighbour-scoring.sh` (same `require_command` guards, same
`tee`-to-`${TMPDIR:-/tmp}` logging), but running
`hyperfine --warmup 1 --runs 10` over the `chutoro-cli` binary on a fixed
synthetic input, where the effect is end-to-end and large enough to resolve.

**Go/no-go, pipeline-relative.** Record this ratio at n = 10 000:

```plaintext
(parallel_kruskal + mutual_reachability + harvest_resort) / cpu_pipeline_end_to_end
```

Proceed to Milestone 2 in all cases — its work is cheap and
correct regardless — but **Milestones 3, 4 and 5 require this ratio to be at
least 5%.** Below that, record the null result in `docs/chutoro-design.md` §6.3
and beside the roadmap item, complete Milestones 2 and 6, and stop. On the
present evidence this ratio is expected to be well under 1%, and closing the
item that way is a successful outcome.

### Milestone 2: red tests, then the committed structural work

**Red first.** None of these touch production code.

1. Strengthen `run_oracle_equivalence_property`
   (`chutoro-core/src/mst/property/equivalence.rs`) to compare the **exact edge
   list** against `sequential_kruskal`. This requires `SequentialMstResult`
   (`property/oracle.rs:16-23`) to gain an edge list, which it does not have
   today — revision 1 called this a guard that would "pass immediately" without
   noticing the code it must first write. Expect it to pass once written; it is
   what makes everything after it safe.
2. Add a determinism test that builds two explicit
   `rayon::ThreadPoolBuilder` pools, one and eight threads, runs the same
   fixture in each via `install`, and asserts exact `MinimumSpanningForest`
   equality. Do **not** set `RAYON_NUM_THREADS` from the test: `AGENTS.md`
   forbids environment mutation in tests, and `ci.yml:24` pins it to `1`
   anyway, so an environment-based test would silently guard nothing. This is
   the highest-value test in the plan and revision 1 did not contain it.
3. Add an error-selection test: an edge list with invalid edges at index 0 and
   at the final index, asserting the index-0 error is reported, run in the
   eight-thread pool. This is genuinely red today — see the rayon finding in
   Surprises.
4. Add `chutoro-core/tests/features/mst_edge_preparation.feature` with step glue
   in `chutoro-core/tests/mst_edge_preparation_bdd.rs`, following
   `chutoro-core/tests/session_append_bdd.rs` exactly: a `World` struct, a
   `#[fixture] fn world()`, steps returning
   `rstest_bdd::StepResult<(), BddStepError>`, one `#[scenario]` function per
   scenario.

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

These scenarios are not yet executable specifications, and must not be
committed until they are. "Core distances that raise the weight of edge 3" and
"the forest excludes edge 3" do not say which nine candidate edges exist or
which one is edge 3, and a six-node graph with nine unspecified edges is not
guaranteed to yield five forest edges. Before writing the step glue, pin every
fixture — the exact `(source, target, distance, sequence)` tuples and the exact
core-distance vector — either as a Gherkin data table in the feature file or as
a named constant in the `World` struct that the `Given` steps index. Two
implementers must not be able to invent different fixtures and both pass.

The fifth scenario pins intra-edge precedence. `validate_and_canonicalize_edge`
(`mst/mod.rs:195-239`) checks source bound, then target bound, then finiteness,
and reports the **original**, pre-canonicalization node id. A branch-free
rewrite that tests finiteness first would flip this silently.

**Then the production work**, in this order, each its own commit:

1. Add `EdgeHarvest::as_slice`, with a doctest.
2. Change `parallel_kruskal_from_edges` to take `&[CandidateEdge]`, deleting the
   `Vec<&CandidateEdge>` at `mst/mod.rs:269`.
3. Replace the `try_fold`/`try_reduce` chain (`mst/mod.rs:270-281`) with a
   pre-sized parallel map into one buffer, then a sequential first-`Err` scan
   in index order. This deletes the reduction tree's per-task allocations and
   copy volume, and makes error selection deterministic **by construction**
   rather than by a min-reduction bolted on.
4. Parallelize the mutual-reachability transform (`cpu_pipeline.rs:79-88`) via
   the new `mutual_reachability_edges` helper, and **delete the
   `EdgeHarvest::new` round trip at line 89**. This is the largest single item
   in the plan: a serial `sort_unstable_by` over roughly `16n` thirty-two-byte
   records with a four-key comparator, whose result is discarded eight lines
   later.
5. Hoist `process_weight_group`'s per-group `Vec::new()` (`mst/mod.rs:248`) by
   passing `&mut forest_edges` and pushing directly. Output order is unchanged
   because the current code already appends in the same order.
6. Move `dedup_by` before `par_sort_unstable`, if and only if a
   dedup-before-sort formulation preserves exact output. HNSW harvest emits both
   `(u, v)` and `(v, u)`, which collapse exactly after canonicalization, so
   this plausibly removes a large fraction of the sort input. If exactness
   cannot be preserved, leave the order alone and record why.

Keep the dedup and weight-group predicates IEEE-`==`-equivalent unless
Milestone 3 proceeds, in which case see Constraint 2. Note that
`property/oracle.rs:119-123` mirrors the current dedup rule and its doc comment
says so; update it in lockstep with any change.

### Milestone 3 (gated): packed sort record

Enter only if Milestone 1's ratio cleared 5% **and** `prepare_edge_list`'s sort
is measurably the dominant sub-stage.

Introduce `PreparedEdge` and `weight_key` as specified above. The lever is
**record width, not comparator**: 32 bytes to 24 in a memory-bound sort.
Revision 1 targeted the comparator, which is three integer operations plus
well-predicted branches and is worth a fraction of a millisecond at best —
below the measured noise floor.

If the `±0.0` divergence in Constraint 2 arises, demonstrate by test that the
final forest is unchanged: dedup only ever removes edges sharing endpoints, and
a surviving duplicate is rejected by `try_union` as a cycle, so dedup is work
reduction rather than a semantic filter. State the dedup key set explicitly as
`(key, source, target)` — excluding `sequence`, or dedup becomes a no-op and
the "lowest sequence survives" contract dies silently.

Add `rstest` unit tests for `weight_key` over `-f32::NAN`, `f32::NAN`, a
signalling pattern (`f32::from_bits(0x7F80_0001)`) and its negation, ±infinity,
±0.0, and ±1.0. Round-trip assertions must compare `.to_bits()`, since
`assert_eq!(f32::NAN, f32::NAN)` fails.

A Kani harness over two symbolic `f32` values is the right instrument for the
ordering property, but per the Risks section it cannot be an acceptance
criterion, because `make kani` runs in no workflow. Add it, validate it by
deliberate mutation, and record that its signal is post-merge only.

### Milestone 4 (gated): candidate pre-filter

Enter only if Milestone 1 cleared 5% **and** `union_find_loop` is measurably
dominant. On present analysis it is not, and this milestone is expected to end
as a null result.

If entered, implement **real Filter-Kruskal** — recursive partition around a
pivot weight, filtering between partitions so the heavy tail is never sorted —
not the post-sort sweep revision 1 proposed. That requires amending
`docs/chutoro-design.md` §6.3's "Keep the global sort in Rayon" instruction via
an ADR, and it puts the equal-weight-group tie-breaking mechanism documented in
§6.2 at risk, so it must land behind Milestone 2's exact-edge-list guard.

Any filter must be testable at the sizes the property suite actually generates.
MST fixtures cap at 64 nodes (`property/strategies.rs:17-22`), so any
edge-count threshold expressed as a benchmark-tuned constant would mean the
filter never fires in any test, and a soundness property would compare
"disabled" against "disabled" and pass vacuously. Expose the thresholds as
**values** on an options struct, and add a property case forcing them to their
most aggressive setting so the filter fires on eight-node graphs. Add a
`debug_assert` at mask-clear time that the two endpoints really do share a root.

Only here does a Verus obligation arise, and it must target root-identity
monotonicity as recorded in the decision log, deriving stability by induction
over single `union` steps rather than assuming a `coarsens` relation. Register
the new file in `PROOF_FILES` in `scripts/run-verus.sh`; note that
`edge_harvest_ordering.rs` and `edge_harvest_extract.rs` are checked only
transitively, as `mod` declarations from a registered file, so a new top-level
file that nobody registers is checked by nothing.

### Milestone 5 (gated): SIMD kernels

Enter only if Milestones 1 to 3 leave classification and compaction at a
measurable share, pre-registered at 10% of `parallel_kruskal`.

Before writing any intrinsic, check whether LLVM has already vectorized the
branch-free scalar code, using `--emit=asm` plus `objdump` (`cargo asm` is not
installed, not in the `Makefile` and not in CI). If the loop body already
contains `vpcmpgt`/`vpmovmskb`-class instructions, stop and record it.

If it proceeds: the mutual-reachability transform's NaN semantics must be fixed
normatively first, because `_mm256_max_ps(a, b)` returns `b` on a NaN operand
while `f32::max` ignores NaN — a Constraint 1 violation waiting to happen.
Padding must use inert zeros, not `u32::MAX`/`INFINITY`. Endpoint validation
must happen in `usize` space before any narrowing. Compaction must be
order-preserving: AVX-512 `_mm512_mask_compressstoreu_epi32`, or the AVX2
`_mm256_movemask_ps` plus 256-entry permutation-table approach — noting the
table is 8 KB competing for 32 KB of L1 data cache against a streaming edge
list, and that Zen 2 splits 256-bit AVX2 operations into two 128-bit
micro-operations.

An AVX-512 backend would sit at the top of the dispatch priority, ship to
users, and never execute on the development machine. Either exclude it, or
state explicitly that it is unverified locally and rests entirely on the parity
suite running elsewhere.

### Milestone 6: documentation, roadmap, and final gates

Append an `_Implementation update (<date>)._` paragraph to
`docs/chutoro-design.md` §6.3 in the register of the existing 2.2.x and 2.3.1
entries, recording the measured pipeline share, what shipped, and every gated
milestone that ended as a null result with its number.

Record in the same place that §6.3's "cache-friendly structure-of-arrays parent
and rank arrays" requirement is already satisfied by
`chutoro-core/src/mst/union_find.rs:20-22`, and that further narrowing was
declined on the measurement recorded in the decision log.

Update `docs/developers-guide.md` with the corrected benchmark methodology for
sub-millisecond stages — sample sizes, core pinning, interleaved A/B, and the
resolution limits of `hyperfine` at this scale. This is reusable beyond this
item.

Add an MST error section to `docs/users-guide.md`. It does not exist today, and
the surface most users see is `ChutoroError::CpuMstFailure { code }` rather than
`MstError` directly. Document the now-deterministic error selection.

Write an ADR only if Milestone 4 or 5 shipped something structural. A plan that
ends in measured null results needs a design-document update, not an ADR.

Register any new document in `docs/contents.md`. Mark roadmap item 2.3.2 `[x]`.

If any workflow's `exclude-globs` changed, update
`tests/workflow_contracts/mutation_testing_test.py::EXPECTED_WITH` in the same
commit — it asserts an exact string and `make test-workflow-contracts` is a
pull-request gate.

```sh
set -o pipefail
make check-fmt    2>&1 | tee /tmp/check-fmt-chutoro-2-3-2-final.out
make lint         2>&1 | tee /tmp/lint-chutoro-2-3-2-final.out
make typecheck    2>&1 | tee /tmp/typecheck-chutoro-2-3-2-final.out
make test         2>&1 | tee /tmp/test-chutoro-2-3-2-final.out
make markdownlint 2>&1 | tee /tmp/markdownlint-chutoro-2-3-2-final.out
make nixie        2>&1 | tee /tmp/nixie-chutoro-2-3-2-final.out
make verus        2>&1 | tee /tmp/verus-chutoro-2-3-2-final.out
make test-workflow-contracts 2>&1 | tee /tmp/contracts-chutoro-2-3-2-final.out
```

Finally, request a CodeRabbit review through the `comenq-coderabbit` skill and
run the response loop to convergence, replying to every thread with an
`@coderabbitai` mention. The 2.3.1 plan records two rounds producing eight
valid findings, one of them a `shellcheck` gap in the very script this plan
copies — so budget for it rather than treating it as optional. Open the pull
request with the `pr-creation` skill.

## Concrete steps

Run everything from the repository root. Commit after each numbered item using
the `commit-message` skill, and follow each functional commit with the
`AGENTS.md` post-commit review pass, landing any resulting refactor as its own
atomic commit.

1. `Extract the Kruskal union-find loop for measurement`
2. `Add MST pipeline benchmark and hyperfine script`
3. `Compare exact MST edge lists against the oracle`
4. `Assert MST determinism across two Rayon pool sizes`
5. `Specify MST edge preparation behaviour with BDD scenarios`
6. `Expose harvested candidate edges as a slice`
7. `Report the lowest-index invalid MST edge deterministically`
8. `Drop the redundant edge harvest sort before Kruskal`
9. `Reuse the forest buffer across Kruskal weight groups`
10. `Record MST edge preparation measurements` (docs, roadmap)

Per-commit gate. Docs-only commits may run the Markdown subset alone, but note
that `spelling-phrase-check` walks `git ls-files`, so it cannot see an
untracked file — stage before trusting a green run:

```sh
set -o pipefail
make check-fmt 2>&1 | tee "/tmp/check-fmt-chutoro-2-3-2-$(git rev-parse --short HEAD).out"
make lint      2>&1 | tee "/tmp/lint-chutoro-2-3-2-$(git rev-parse --short HEAD).out"
make typecheck 2>&1 | tee "/tmp/typecheck-chutoro-2-3-2-$(git rev-parse --short HEAD).out"
make test      2>&1 | tee "/tmp/test-chutoro-2-3-2-$(git rev-parse --short HEAD).out"
```

Additional gates for the specific artefacts this plan touches, all named in
`AGENTS.md` and all omitted by revision 1:

```sh
set -o pipefail
shellcheck scripts/bench-mst-pipeline.sh         # step 2 adds this script
mbake validate Makefile                          # only if the Makefile changed
action-validator .github/workflows/property-tests.yml  # only if a workflow changed
make test-workflow-contracts                     # if any exclude-globs changed
```

Expected transcript for the red stage of step 7, before the fix. The exact
wording of the reported error is what proves the test is red for the intended
reason rather than for an unrelated failure:

```plaintext
--- STDERR: chutoro-core mst::tests::reports_lowest_index_invalid_edge ---
thread 'main' panicked at chutoro-core/src/mst/tests/forests.rs:NNN:
assertion `left == right` failed: expected the index-0 edge to be reported
  left: InvalidNodeId { node: 97, node_count: 8 }
 right: InvalidNodeId { node: 9, node_count: 8 }
```

After step 7 the same command must pass in both the one-thread and the
eight-thread pool. Record both transcripts in `Progress`.

## Validation and acceptance

**Equivalence.** `cargo nextest run -p chutoro-core -E 'test(/mst::/)'` passes,
including the strengthened exact-edge-list property. Under
`PROPTEST_CASES=25000` the oracle-equivalence, structural-invariant and
concurrency-safety properties pass.

**Determinism.** The two-pool test asserts exact `MinimumSpanningForest`
equality between one-thread and eight-thread pools. The error-selection test
reports the lowest-index invalid edge in both pools.

**Behaviour.** All six scenarios in
`chutoro-core/tests/features/mst_edge_preparation.feature` pass.

**End to end.** Capture `chutoro-cli` cluster assignments for one fixed
synthetic input before and after; diff them; expect no differences. This is the
externally observable contract and the reason the end-to-end test exists.

**Performance.** Criterion, run with the corrected methodology from Milestone
1, shows `parallel_kruskal` improving at n = 500, n = 1000 and n = 10 000 by
more than the measured noise band for that configuration — **not** by a fixed
percentage, since the present instrument produces ±12-17% false regressions on
unmodified code. `scripts/bench-mst-pipeline.sh` corroborates end to end.
Criterion remains the primary signal per `docs/developers-guide.md`.

**Gates.** `make check-fmt`, `make lint`, `make test`, `make markdownlint`,
`make nixie` and `make verus` all pass.

Red-Green-Refactor evidence to record in `Progress`: the error-selection test's
observed pre-change output verbatim; the same command passing after step 7 of
Milestone 2; and `make lint` plus `make test` after each extraction.

## Idempotence and recovery

Every step is re-runnable. Criterion baselines under `target/criterion/` may be
deleted freely. Benchmarks and `make verus` do not modify the source tree.

On a gate failure, read the `tee`d log named in the command rather than
re-running the gate; re-run only after a fix. Each commit in the sequence above
is independently revertible. Log names are branch-scoped so they do not collide
with other agents' runs.

## Artefacts and notes

Measured on this branch, unmodified code, two consecutive runs — the evidence
behind the benchmarking-methodology risk:

```plaintext
parallel_kruskal/n=500   time: [1.0052 ms 1.0941 ms 1.2182 ms]
                         change: [+10.059% +16.669% +26.638%] (p = 0.00 < 0.05)
                         Performance has regressed.
                         Found 3 outliers among 20 measurements (15.00%)
parallel_kruskal/n=1000  time: [1.8448 ms 1.8781 ms 1.9128 ms]
                         change: [+9.4250% +11.801% +14.223%] (p = 0.00 < 0.05)
                         Performance has regressed.
```

Prior art consulted:

- Osipov, Sanders and Singler, "The Filter-Kruskal Minimum Spanning Tree
  Algorithm", ALENEX 2009. Its saving comes from `partition` plus `filter`
  together; `filter` alone is not the algorithm.
- Quickwit, "Filtering a Vector with SIMD Instructions (AVX-2 and AVX-512)" — a
  worked Rust treatment of order-preserving stream compaction.
- Giesen, "Order-preserving bijections", and the IEEE 754 radix-sort key
  transform, verified independently on this branch.
- Published AVX2 gather measurements showing `vgatherdps` matching or losing to
  scalar loads on memory-bound patterns, which is why gather is excluded.

## Out of scope, recorded for follow-up

Two findings from this plan's design review are larger than the item it covers
and must not be absorbed into it silently.

1. **The core-distance loop is serial.**
   `chutoro-core/src/cpu_pipeline.rs:65-77`
   runs `n` full HNSW searches at `ef = 32` on one thread, with two `Vec`
   allocations per point, on a twenty-four-thread machine. Parallelizing it is
   roughly six lines and is plausibly worth two orders of magnitude more than
   everything in this plan combined. It belongs in its own roadmap item.
2. **Almost no crate inherits the workspace lint table.** Tracked as
   [issue #200](https://github.com/leynos/chutoro/issues/200). Only
   `chutoro-bench-datasets` carries `[lints] workspace = true`;
   `chutoro-benches` hand-duplicates the table with a manual-sync caveat; and
   `chutoro-core`, `chutoro-cli`, `chutoro-providers/dense`,
   `chutoro-providers/text` and `chutoro-test-support` inherit nothing. Opting
   every crate in costs a measured 576 findings, and adding
   `missing_docs_in_private_items = "deny"` on top costs a further 914. Note
   that `missing_docs` and the rustdoc lints _are_ enforced, but through
   `.cargo/config.toml` rustflags and the `RUSTDOCFLAGS` override in
   `make lint-clippy` rather than through the manifest — so editor feedback
   diverges from CI. This is why revision 1's truncation guard did not exist.

## Outcomes & retrospective

To be completed at each milestone and at close. Compare against the purpose:
the measured pipeline share, what shipped, what was declined and on what
number. A close consisting largely of null results, with the cheap structural
work landed and the measurement documented, satisfies this plan.

## Revision note

**Revision 2 (2026-08-16).** Rewritten after a six-lens community-of-experts
design review. The review measured the target stage at ≈ 0.22% of the HNSW
build and found five factual errors in revision 1's description of the code,
the most consequential being that `try_union` takes no lock on the
cycle-rejection path — which was the sole justification for the candidate
pre-filter. Revision 1's structure-of-arrays layer, policy object, kernel port,
dispatch table, SIMD adapters, union-find narrowing and Verus proof are removed
from committed scope; several were unimplementable as specified (five parallel
`Vec`s cannot be sorted; `EdgeVerdict` could not derive `Copy`/`Eq` over
`MstError`; padding sentinels were the values the validator rejects). Committed
scope is now the cheap structural work plus the tests that make the invariants
real, with everything expensive individually gated behind a pipeline-relative
threshold and an explicit null-result path. The determinism and exact-edge-list
guards, absent or unenforceable in revision 1, are now the plan's core.

Also applied from the review: dated `Progress` entries with per-milestone
sub-checkboxes, so Tolerance 8 is enforceable; real commands and a red-stage
transcript in `Concrete steps`; the omitted `make typecheck`, `shellcheck`,
`mbake`, `action-validator` and workflow-contract gates; a CodeRabbit round and
pull-request step; a requirement that the BDD fixtures be pinned before the
feature file is committed; recorded decisions on doctests and on index
registration; and the finding that no workflow compiles NEON.

One deliberate deviation from the house layout remains:
`Outcomes & retrospective` sits near the end rather than eighth. The plan reads
better with the measured evidence and the milestone structure ahead of an
as-yet-empty retrospective, and the section is listed in the living-document
preamble either way.
