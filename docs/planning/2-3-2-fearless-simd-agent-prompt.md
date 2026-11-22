# Revise the 2.3.2 plan for Fearless SIMD

Revise the existing execution plan for roadmap item `2.3.2` on branch
`2-3-2-vectorize-edge-weight-transforms-and-candidate-filtering`, associated
with [pull request 198](https://github.com/leynos/chutoro/pull/198). The target
document is
`docs/execplans/2-3-2-vectorize-edge-weight-transforms-and-candidate-filtering.md`.
Deliver a reviewable plan revision with evidence, reconciled progress, and
implementable acceptance criteria. This task authorizes documentation work;
production implementation requires its own existing authorization.

## 1. Apply the design direction and dependency authorization

Read `AGENTS.md`, `docs/contents.md`, the documentation style guide, and these
governing references before editing:

- [ADR 0007: Fearless SIMD kernel boundary](../adrs/adr-0007-fearless-simd-kernel-boundary.md).
- [RFC 0001: Fearless SIMD query-to-points](../rfcs/rfc-0001-fearless-simd-query-to-points.md).
- [Chutoro design, section 6.3](../chutoro-design.md#63-simd-utilization),
  including the dated implementation updates.
- [Roadmap](../roadmap.md), especially the independent dense-provider task
  `2.2.8`, and ADR 0003's precedent for recording measured null results.

Until the companion documents reach main, read them from branch
`docs/fearless-simd-adoption-2026-11-22` and record the commit inspected.
Read-only access is sufficient; preserve the planning branch's existing work.
Distinguish proposed architecture under review from the maintainer's already
effective dependency authorization below.

The maintainer explicitly authorizes `fearless_simd = "1.2"` and, where used,
`fearless_simd_macros = "0.1"` for a justified gated kernel. Record this as a
narrow exemption to the plan's no-new-dependencies constraint and the
associated manifest-change tolerance. No further dependency permission is
needed for these entries. Retain the restriction for unrelated dependencies.
Use Cargo's compatible caret requirements, and preserve the minimum supported
Rust version (MSRV) of `1.89.0`. This plan revision makes no manifest or
lockfile changes; future implementation adds dependencies only when its gate
succeeds. Require the resolved dependency graph to build on Rust 1.89;
compatibility of release 1.2.0 does not guarantee the MSRV of every later 1.x
release permitted by the caret requirement.

Distinguish two decisions. Task `2.2.8` evaluates a bounded port of the dense
provider's existing query-to-points kernel, where reduced duplication and
unsafe code can justify adoption without an end-to-end speedup. Task `2.3.2`
introduces an optimization only after its performance gates pass. Neither task
completes the other. Sharing the dependency and the architectural boundary does
not justify a shared application-level SIMD framework.

## 2. Reconcile the plan with current code

Record the working branch, its head, the reviewed main commit, and existing
uncommitted changes. The reference main commit for this guidance is
`5f13f07e10688ddb0b33e3fb2ef5afbc0e285013`; inspect newer main before treating
it as the baseline. Preserve other agents' work. Do not reset, overwrite, or
replay already merged changes merely to match the old plan's structure.

Inspect the current CPU pipeline, MST preparation and union-find modules, their
tests, manifests, benchmark harnesses, and relevant workflows. Reconcile each
milestone into completed, remaining, or superseded work with source references.
In particular, verify the parallel mutual-reachability transform, removal of
the redundant harvest sort, direct slice entry point, ordered validation
buffer, allocation-free weight-group extension, and parallel core-distance
searches. Refresh obsolete claims about lint inheritance, concurrency coverage,
error enums, and proof gates from current evidence.

Preserve the August measurements as dated historical evidence. The recorded
0.22% compared `parallel_kruskal` with HNSW build time on an earlier tree; it
excluded the mutual-reachability transform and did not measure the whole
current pipeline. It cannot establish a present-day optimization ceiling. Label
larger-dataset extrapolations as estimates, not measurements. Do not restate
obsolete line numbers, test counts, or infrastructure assumptions as current
facts.

## 3. Preserve the measurement gates

Keep the existing entry gate requiring at least 5% of pipeline time for
expensive milestones. Measure current edge-preparation and Kruskal work against
a current complete pipeline run on the same workload and configuration. State
the numerator explicitly: mutual-reachability transformation plus
`parallel_kruskal`, with preparation already included inside Kruskal. Remove
`harvest_resort` from the current numerator after verifying that the redundant
sort has gone. Do not double-count nested stages or resurrect removed work for
the benchmark. Retain the plan's `n = 10_000` decision workload and state its
parameters; report additional sizes separately rather than replacing that
decision point.

Also retain Milestone 5's requirement that classification and compaction
account for at least 10% of `parallel_kruskal` after the preceding work. Define
exactly which operations that measurement includes. Keep transformation timings
separate where they sit outside Kruskal. Any ambiguity in the two gates must be
resolved in the plan before implementation; do not silently weaken either
threshold because the dependency is now authorized.

Specify reproducible workloads, dimensions, edge counts, seeds, compiler flags,
hardware, selected backend, Rayon thread counts, and baseline commits. Use
interleaved comparisons, fixed thread pools or documented core affinity, and
sample sizes and measurement windows that resolve the claimed effect. Report
uncertainty and the observed noise band. Include realistic indexed
core-distance reads, buffer preparation, dispatch, validation, and ordered
output emission; an isolated maximum instruction is insufficient evidence.
Reinitialize union-find state for each timed trial, and state whether setup
belongs inside the measured region.

Retain Criterion as the primary signal and command-line end-to-end timing as
corroboration where the expected effect is resolvable. Before proposing
explicit vectors, inspect optimized assembly for the current scalar loop and a
target-specialized scalar variant where useful. Record compiler commands and
findings. Existing effective autovectorization, an unmet gate, or a negligible
improvement should produce a documented null result and completion of the
remaining applicable documentation work.

## 4. Specify the kernel boundary and representation

If Milestone 5 proceeds, prefer a private Fearless SIMD kernel over a new
family of handwritten intrinsics. Keep graph rules, validation decisions, error
construction, ordering, and backend policy in chutoro. Give the kernel plain
numeric batches and return numeric results or masks. Keep library tokens,
vectors, masks, and dispatch machinery out of public interfaces.

Place core pipeline kernels in `chutoro-core` behind its optional `cpu`
feature. Plan optional dependency activation so a build without `cpu` does not
acquire the new backend. The core must not depend on the dense provider. Reuse
local abstractions only after checking their ownership and semantics; do not
transplant the dense provider's packing type into generic graph code.

Dispatch once per substantial batch or Rayon chunk, with the specialized loop
inside that boundary. Avoid feature detection and dispatch per edge or vector.
Respect feature enablement and explicit backend controls, and preserve a true
chutoro scalar fallback. A Fearless SIMD baseline token can represent SIMD, so
it does not substitute for the scalar contract. Obtain supported tokens safely;
do not assume that a single AVX2 or AVX-512 feature probe establishes every
feature required by a library token. Never fabricate an unchecked token to
retain a backend label.

Account for array-of-structures records and indexed core-distance access.
Fearless SIMD does not automatically make those reads contiguous or remove
packing costs. Start with SIMD classification and scalar ordered emission
unless measured compaction costs justify more. A later `compress` operation
must preserve the association of endpoints, weight, original index, and
sequence. Native `f32` and `u64` vectors have different lane counts; define a
common logical batch size and mask conversion or splitting explicitly. Validate
`usize` identifiers before any narrowing, and retain their original values for
diagnostics. Keep full-width operation or a scalar fallback where the validated
values cannot fit the chosen representation.

Handle tails with explicit active-lane masks or a scalar remainder. If padding
is used, keep it inert and exclude inactive lanes from rejection selection and
output counts. Explain bounds for every load, store, mask, and compaction
result. Do not let an inactive lane create a spurious error or accepted edge.

## 5. Preserve numerical and graph semantics

Document the scalar reference expression as the ordered evaluation of
`distance.max(core[source]).max(core[target])`. Specify the accepted input
domain before selecting a vector operation. `max_precise` handles quiet NaNs
more suitably than a raw platform maximum, but does not guarantee identical
signed-zero selection or signalling-NaN behaviour across backends.[^1] Separate
guaranteed behaviour from unspecified representation details.

Do not reject a raw NaN or negative infinity merely to simplify the kernel: the
current maximum can produce a finite transformed weight when another operand
supplies it. Validation before this transform can therefore change accepted
inputs. Preserve checked source and target core-distance lookups and their
errors. If existing contracts conflict over signed zero, propose an explicit
resolution with compatibility evidence; do not normalize values silently under
the description of a performance optimization.

Preserve MST validation precedence: source bound, target bound, weight
finiteness, then self-loop elimination. Preserve original endpoint orientation
in errors, input-index error selection, canonical endpoints in accepted
records, sequence values, and the final forest's edge order and component
count. Parallel masks must select the lowest failing original input index, then
apply the same within-edge precedence. Compacting valid lanes must not hide an
earlier error or permit a later error to win a scheduling race.

Keep deduplication and equal-weight grouping consistent with the accepted
plan's compatibility rules. `total_cmp` distinguishes signed zeros, whereas
ordinary floating-point equality does not. Tests for promised bitwise
preservation must compare `to_bits()`; derived `PartialEq` alone cannot prove
that guarantee. Test quiet and signalling NaNs, both infinities, both zeros,
finite extremes, and every relevant operand position. Compare accepted output
and exact error identity, not merely total forest weight.

Keep union-find's atomic access, synchronization, and dependent traversals
outside this kernel. Portable vectors do not justify concurrent non-atomic
reads or replace the proof required for a cycle filter. Introduce candidate
pre-filtering only if its own existing gate succeeds. If Filter-Kruskal is
still proposed, retain partitioning, equal-weight tie-breaking obligations,
forced-active small-graph tests, and substantive root-identity reasoning.
Availability of a fast mask or `compress` primitive is not evidence that an
additional filtering pass improves the algorithm.

## 6. Plan verification and acceptance evidence

Require scalar-reference differential tests for each enabled backend on
hardware that actually executes it. Separate successful compilation, backend
selection tests, and executed instruction-set coverage in the report. Record
unavailable hardware honestly and name the evidence required before enabling
that path. Use injected backend selection rather than environment mutation to
exercise fallbacks and restrictions. Test `cpu` disabled, scalar-only
execution, available restricted backends, and ordinary automatic selection.

Cover empty inputs, lengths below and around lane boundaries, multiple blocks,
all-rejected and all-kept masks, mixed mask densities, invalid endpoints,
duplicate edges, disconnected graphs, and large identifiers that would truncate
to apparently valid `u32` values. Preserve deterministic behaviour across
one-thread and eight-thread Rayon pools. Check complete forests, sequence
values, error precedence, and a fixed end-to-end clustering result. Keep proof
scope tied to introduced invariants; verify that any Kani or Verus obligation
actually runs through the current documented tooling and gates.

Use the 1.2 API as the planning baseline: version 1.1 added classification and
ordered compression, and version 1.2 retains MSRV 1.89.[^2] Record whether the
optional macro crate is needed, its compilation cost, and the selected dispatch
strategy. Specify benchmark evidence for the complete kernel path and pipeline,
plus binary-size or build-time changes if multiversioning makes them material.
Do not claim speedups, execution coverage, or passing gates without recorded
results.

## 7. Deliver the revised plan

Update constraints, tolerances, risks, progress, discoveries, decisions,
interfaces, milestones, validation, and retrospective consistently. Preserve
useful historical reasoning while identifying superseded statements. Include a
compact inventory of completed structural work, outstanding measurements, gated
implementation, dependencies, and explicit null-result outcomes. Keep unrelated
production refactors and the separate dense port outside this plan.

Record the dependency exemption and governing ADR in the decision log. Revise
the SIMD milestone so an implementer can identify the input representation,
semantic oracle, dispatch boundary, expected edits, tests, and stop criteria
without inventing missing policy. Retain the plan's remaining scope and time
tolerances unless a separately documented decision changes them. Link the
updated plan from the documentation index if needed.

Format the documentation according to repository rules and run the applicable
documentation gates, including `make markdownlint` and `make nixie`. Record
unavailable tools and exact validation limitations instead of substituting a
claim that checks passed. Follow repository commit and review procedures for
the authorized documentation update. Report the plan diff, reconciled source
baseline, dependency authorization, preserved gates, and unresolved evidence
requirements. Do not mark an unmeasured optimization or the separate task
`2.2.8` complete.

[^1]: [Fearless SIMD 1.2.0 `SimdBase` numerical contracts](https://docs.rs/fearless_simd/1.2.0/fearless_simd/trait.SimdBase.html).
[^2]: [Fearless SIMD 1.2.0 changelog](https://github.com/linebender/fearless_simd/blob/fearless_simd-v1.2.0/CHANGELOG.md).
