# RFC 0001: Fearless SIMD query-to-points

## Preamble

- **RFC number:** 0001
- **Status:** Proposed; pending pull-request review. Implementation pending.
- **Created:** 2026-11-22
- **Audience:** Dense-provider implementers and reviewers.
- **Roadmap:** [Task 2.2.8](../roadmap.md).
- **Decision:** [ADR 0007](../adrs/adr-0007-fearless-simd-kernel-boundary.md).

## Summary

Evaluate one private Fearless SIMD kernel for dense query-to-points Euclidean
distance. Assign independent candidate points to vector lanes, retain the
existing packed storage, and compare the candidate against the scalar oracle
and current architecture-specific kernels. Adoption requires preserved
behaviour, verified dispatch, execution on each substituted architecture, and
no material performance regression. Reduced duplication and project-owned
unsafe code justify the experiment; a speed-up remains an empirical question.

This proposal authorizes a bounded implementation task after review. It makes
no production-code, manifest, lockfile, or public-interface change itself.
Later minimum spanning tree (MST) work under roadmap item 2.3.2 retains its
separate measurement and semantic gates in ADR 0007.

## Problem

The dense provider maintains several implementations of the same packed-point
arithmetic and its safe-entry wrappers. The implementations differ in vector
width and intrinsic syntax. Consolidating that loop can reduce maintenance,
provided the replacement preserves feature controls, arithmetic, and the
existing provider boundary. A new abstraction must not introduce another
permanent set of equivalent kernels.

## Current state

The reviewed source baseline is Chutoro commit
`5f13f07e10688ddb0b33e3fb2ef5afbc0e285013`. Refresh this baseline in the
implementation plan before measuring or editing production code.

The batch entrypoint validates pairs in input order, detects a shared query,
and packs candidate rows when dimensions are nonzero, at least two candidates
exist, and the selected backend supports vectorization. It collects results
before writing the caller's output, preserving output atomicity on errors.[^1]

The scalar packed kernel accumulates dimensions in order for each candidate,
then takes the square root and converts non-finite results to `f32::NAN`.
Pairwise kernels instead reduce dimensions across lanes; they have different
rounding considerations and remain outside this migration.[^2]

`DensePointView` owns dimension-major coordinate blocks, aligned to 64 bytes
and padded with zeroes to a multiple of 16 points. Its storage conversions
contain existing unsafe code. Reusing it does not eliminate that code.[^3]

The selector caches its choice and prefers AVX-512, AVX2, NEON, optional
nightly portable SIMD, then scalar. Current x86 eligibility checks require
`avx512f` or `avx2`, respectively; ARM includes a 32-bit NEON path.[^4]
Existing parity uses an absolute `1.0e-5` epsilon for finite distances, accepts
corresponding NaNs, and requires zero distances for identical zero vectors.
This contract does not promise bit-identical output across architectures.[^5]

## Goals and non-goals

The goal is one generic implementation of the packed query-to-points loop, with
native-width vectors, safe loads and stores, and a measured reduction in
project-owned kernel duplication. The existing scalar implementation remains
the independent numerical oracle.

The task excludes pairwise-kernel migration, new metrics, public interfaces,
persistent dimension-major storage, packing-policy changes, buffer-reuse APIs,
additional thread pools, and changes to core Hierarchical Navigable Small World
(HNSW), MST, or hierarchy algorithms. Structure of Arrays (SoA) packing remains
private to the dense adapter under
[ADR 003](../adrs/adr-0003-soa-prefetch-adapter-boundary.md).

## Proposed design

### One synchronous internal kernel

Add one private function generic over Fearless SIMD's `Simd` capability token.
Its inputs remain the query row, an existing `DensePointView`, and the output
slice. It has no ownership of provider data or graph state. Resolve existing
helpers before introducing any new abstraction; do not add a public kernel
trait or a cross-crate adapter framework.

For each native-width chunk of candidates, initialize one accumulator.
Broadcast each query coordinate, load the corresponding contiguous candidate
coordinates, subtract, multiply each delta by itself, and add the squares. Each
lane retains one candidate's dimension order. Keep multiplication and addition
separate; do not introduce fused multiply-add, horizontal reductions,
approximate square roots, or relaxed floating-point operations. Finish each
valid lane using the existing scalar square root and normalization initially.

The kernel executes synchronously on its caller's thread. SIMD dispatch selects
an instruction set; it does not submit parallel work. The adapter must not
dispatch to Rayon, block on another worker, or re-enter the index. These
restrictions preserve the callback contract under the insertion mutex in
[ADR 006](../adrs/adr-0006-sequential-trim-scoring.md).

### Preserve feature selection before obtaining a token

Keep the existing selector as the authority for backend eligibility and
priority. For the query kernel only, select the existing backend first, then
obtain its corresponding safe Fearless token. Replace the query-only cache with
a private invocation enum that stores either that exact token or the legacy
function pointer. Match once per batch to invoke the generic kernel or legacy
entry. A bare function pointer cannot capture the token. Keep the pairwise
function-pointer cache unchanged.

Fearless SIMD 1.2 provides safe capability tokens, native-width vectors, and
checked slice operations. Its `Avx2` token denotes x86-64-v3, and `Avx512`
denotes an Ice Lake feature bundle. Both require more than the current local
probes. Native NEON support covers AArch64; 32-bit ARM needs its existing
implementation.[^6]

Use `Level::new()` and the matching safe `as_avx2`, `as_avx512`, or `as_neon`
conversion. A missing token selects the same legacy query implementation that
the existing selector chose. Never construct an unchecked token from a local
Boolean capability probe. Never call unrestricted `dispatch!` after local
selection: that could enable an instruction set disabled by Cargo features.
`Level::baseline()` also cannot represent the required scalar-only control,
because a target baseline may already include SIMD.[^6]

Preserve the existing scalar path directly when the selector returns scalar,
including `--no-default-features`. Keep the existing nightly portable-SIMD path
when it wins selection. No new production feature flag, environment override,
or second runtime selector belongs in the completed migration.

### Bound loads and output writes

Reuse `DensePointView` without changing its padding or allocation policy.
Derive the loop step from the selected token's floating-point lane count. Pass
exactly that many elements to each checked vector load and store, and authorize
a full-width load only when its range fits the coordinate block. A native width
alone does not prove compatibility with 16-point padding. Use complete chunks
and a scalar remainder independent of the padding width.

Write exactly the logical candidate count. Padding lanes cannot become output
distances, errors, or indices. Empty inputs perform no load or write, and
zero-dimensional inputs retain the established zero-distance behaviour.
Validate query and output lengths at the existing safe boundary before any
mutation. Leave the current pair-validation order and atomic output commit
outside the kernel.

### Dependencies and support floor

The implementation task may declare caret requirements `fearless_simd = "1.2"`
and, when using `#[simd]`, `fearless_simd_macros = "0.1"` in
`[workspace.dependencies]`. Only the dense provider consumes them in this task,
with `optional = true` in its manifest. Activate their `dep:` entries through
supported existing native SIMD features; do not expose new implicit dependency
features. Scalar-only and nightly-only portable-SIMD configurations retain
their dependency controls without pulling in an unused native backend. The
optional macro addresses target-feature propagation. Neither dependency enters
`chutoro-core` in this task. Record the resolved versions and feature graph in
the implementation report.[^7]

Keep Rust 1.89 as the minimum supported Rust version (MSRV), stable builds as
the default, and existing stable `--all-features` behaviour. The workspace
declares 1.89, but the dense manifest currently does not inherit that
field;[^8] the implementation must declare or inherit effective MSRV metadata.
Test the changed production dependency graph on 1.89 as well as the development
compiler. Record pre-existing test or development-dependency incompatibilities
separately from failures introduced by this change; neither resolver defaults
nor a successful newer-compiler build establish MSRV compatibility. Record
compile-time and binary-size changes; a smaller source file does not imply a
smaller build. This documentation change adds no dependency or metadata.

## Requirements

### Functional requirements

- **F1:** Preserve candidate order, duplicate candidates, pair orientation,
  validation precedence, error values, and output atomicity.
- **F2:** Preserve the existing scalar-oracle tolerance and non-finite policy.
  Compare finite results against the legacy query backend on the same host
  using their bit representations; investigate and reject unreviewed changes. A
  wider epsilon cannot hide a migration regression.
- **F3:** Preserve the positive-zero result for identical zero vectors,
  logical output length, and all existing zero-length fallback decisions.
- **F4:** Preserve current Cargo-feature selection and fallback behaviour,
  including explicit scalar-only operation and nightly portable SIMD.
- **F5:** Keep pairwise kernels, provider interfaces, and calling-thread
  execution unchanged.

### Technical and safety requirements

- **T1:** Use one generic packed-point arithmetic body; any retained legacy
  query body must serve a documented capability or verification exception.
  Demonstrate a net maintenance reduction including those retained bodies and
  all added dispatch, adapter, test, and proof code.
- **T2:** Obtain real capability tokens safely. Synthetic support masks may
  test the pure selector, but cannot authorize execution of unsupported code.
- **T3:** Prove load and store ranges from actual slice lengths. Padding and
  lane-width changes must not permit out-of-bounds access or visible padding.
- **T4:** Introduce no project-owned unsafe code, allocator policy, public
  abstraction, nested Rayon dispatch, or process-global testing override.
- **T5:** Execute each substituted instruction-set path on suitable hardware
  before enabling it. Compile-only checks are insufficient evidence.
- **T6:** Preserve the MSRV and stable feature matrix, and meet the
  pre-registered performance criteria below.

## Verification

Reuse existing dense fixtures, parity helpers, and backend enumeration. Keep
the scalar oracle independent of the new generic kernel. During evaluation,
retain legacy query entrypoints for direct comparisons; remove redundant copies
after adoption, except where the compatibility policy requires them.

| Obligation                         | Cases and method                                                                                                                                                           | Acceptance evidence                                                                                                             |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| F1, F3: batch behaviour            | Empty and singleton batches; shared query on either side; mixed pairs; duplicates and reordered candidates; output-length mismatch; invalid first and later endpoints.     | Existing errors and ordering; sentinel output remains unchanged on failure; exact logical output length.                        |
| F2: arithmetic                     | Scalar, legacy, and candidate comparisons with zeroes, signed zeroes, duplicates, random finite values, subnormals, large magnitudes, overflow, NaNs, and both infinities. | Existing tolerance passes; no unexplained same-host legacy finite-bit differences; canonical non-finite outputs.                |
| F3, T3: boundaries                 | Counts and dimensions 0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, and 33; larger non-multiples; every executable native width.                                             | No extra output; full loads fit their blocks; remainder values match the oracle.                                                |
| F4, T2: dispatch                   | Pure selector tests over compiled, runtime, and safe-token-availability masks; defaults, no defaults, individual native features, and combinations.                        | Disabled features never select candidate code; unavailable tokens retain the selected legacy path; scalar control stays scalar. |
| F4, T6: compiler matrix            | Rust 1.89 and stable pin; stable all-features; nightly with and without `nightly_portable_simd`; existing compile-contract fixtures.                                       | All supported configurations compile and preserve their documented backend selection.                                           |
| F5: integration                    | Dense batch tests, same-thread callback regression, and fixed seeded clustering fixtures before and after on one host with single-worker HNSW construction.                | Same caller thread; unchanged batch results and cluster assignments for the fixed fixtures; no new deadlock dependency.         |
| T3: bounded proofs                 | Extend existing Kani harnesses for owned dispatch eligibility, lane counts, tails, and logical output bounds.                                                              | Counterexamples fail the harness; record bounds and successful runs. No claim to prove dependency intrinsics.                   |
| T5: execution                      | AVX2, AArch64 NEON, and AVX-512 for every backend actually substituted; scalar and retained legacy paths.                                                                  | Report CPU, OS, compiler, Cargo features, resolved dependency versions, selected token, and executed cases.                     |
| T1, T4, T6: implementation quality | Review source, emitted assembly, unsafe-code inventory, code size, compile time, and benchmarks.                                                                           | No extra arithmetic framework, no newly owned unsafe blocks, preserved operations, and a documented maintenance reduction.      |

_Table 1: Verification matrix for the bounded query-to-points migration._

Tests must distinguish a skipped backend from an executed backend. For an
unavailable architecture or stronger capability bundle, keep the old production
route and record the gap. A successful x86 job cannot establish NEON parity. A
safe token proves instruction availability, not numerical equivalence or
performance.

The seeded integration fixtures must include duplicate points and near ties.
Compare neighbour selection and ordering as well as cluster assignments; kernel
tolerance alone cannot establish those downstream properties.

The Kani scope covers Chutoro-owned index arithmetic and selector logic. It
does not attempt to model Fearless intrinsics. Retain the repository's existing
proof gates and review any changed harness against a deliberate incorrect
bounds or eligibility mutation.

## Performance experiment and decision rule

Before implementing the candidate, register benchmark inputs, hardware,
compiler flags, thread counts, sample durations, acceptable slow-down for each
size class, and the comparison method. Use repeated unchanged-versus-unchanged
runs to establish the instrument's noise. Tighten the instrument until that
noise is smaller than the declared regression tolerance; otherwise record an
inconclusive result rather than accepting a noisy comparison.

Measure two levels independently:

1. Packed-kernel microbenchmarks reuse an already constructed `DensePointView`
   and preallocated output. They isolate arithmetic and dispatch costs.
2. Actual batch benchmarks include shared-query detection, validation,
   allocation, packing, and result transfer. They represent the path callers
   pay for and control the adoption decision.

Cover candidate counts 0, 1, 2, 4, 8, 16, 32, 64, and 128, including adjacent
lane-boundary counts. Cover dimensions 0, 1, 8, 16, 32, 128, and 768, plus
non-multiples and representative HNSW batches from the existing harness.
Include mixed-pair and scalar-only controls to detect unintended scope drift.

Run paired, interleaved baseline/candidate measurements in release builds with
fixed inputs, warmed caches, and pinned cores where available. Separate
initialization cost from steady-state calls. Retain raw samples and report
confidence intervals for relative change, rather than comparing one run with an
old stored baseline. Record packing allocations, CPU frequency conditions, and
the resolved backend so comparisons describe the same workload.

Adopt only when the upper confidence bound for regression stays within the
pre-registered tolerance for every required actual-batch size class, numerical
and compatibility gates pass, and the final patch reduces owned maintenance.
Report packed-kernel regressions even when packing hides them. A wide interval
is inconclusive, not evidence of equivalence. The MST requirement that targeted
work account for at least 5% of pipeline time does not apply to this
maintenance-oriented port. No end-to-end speed-up is required; adoption still
requires the declared regression tolerances and a useful maintenance reduction.
Do not claim a pipeline benefit from a microbenchmark.

The maintenance report must inventory old and new arithmetic bodies, retained
weak-capability and 32-bit ARM fallbacks, unsafe wrappers, selector branches,
cached invocation state, and supporting tests and proofs. Compare the complete
maintained implementation, including exceptions, against the original. Lines
removed from one module cannot establish a reduction when equivalent work moves
elsewhere. Preserving every legacy body while adding the generic body can
increase both implementation count and maintenance cost. Record a null result
if the complete patch cannot demonstrate a useful net reduction.

Enable a replacement only for backend and capability combinations that satisfy
every applicable gate. Retain the established route elsewhere. If the complete
proposed patch, including retained legacy routes, cannot meet the performance
and maintenance criteria, end the experiment, remove trial dependencies and
dispatch wiring, and record the null result beside task 2.2.8. Do not retain an
unused backend or invent a permanent runtime choice to avoid the decision.

## Compatibility and migration

1. Refresh source and dependency baselines; document the benchmark protocol,
   execution hosts, and exact feature-to-token mapping before production edits.
2. Add the private candidate and focused comparisons without changing public
   APIs, pairwise kernels, storage, packing policy, or core scheduling.
3. Run the verification matrix and performance experiment. Enable replacement
   only for backend/capability combinations with complete evidence; retain the
   corresponding legacy implementation elsewhere, including 32-bit ARM.
4. On adoption, remove temporary comparison plumbing, list every retained
   legacy exception, and update the design and roadmap with measured results.
   Otherwise close the experiment with the documented null result.

The implementation pull request must run applicable repository formatting,
lint, unit, behavioural, property, compile-contract, and proof gates. Publish
the executed-versus-skipped backend report with its evidence. This RFC records
requirements; it does not record successful execution of those gates.

## Alternatives considered

Keeping the current kernels avoids a dependency and preserves established
behaviour; it remains the fallback when the experiment fails. Replacing every
distance kernel at once combines the independent-point migration with changed
pairwise reduction trees and exceeds this task's scope. Unrestricted Fearless
dispatch would simplify selection but cannot preserve the existing feature
contract. A new public kernel framework would cross the adapter boundary
without a demonstrated caller requirement.

## Open questions

- Which execution hosts can provide AVX2, AArch64, and the required AVX-512
  capability bundle? Unverified replacements remain disabled.
- What per-size-class slow-down tolerance reflects current batch costs, and
  can the benchmark resolve it? Register the answer before candidate timing.
- Does `#[simd]` reduce boilerplate enough to justify its compile-time cost?
  Compare the supported macro and manual annotation routes within this kernel.

## Recommendation

Proceed with the bounded task after review. Adopt the generic query kernel only
where compatibility, execution evidence, and measurements support it. Keep
later pairwise and MST adoption as separately justified work.

## References

[^1]: [Dense batch boundary at the reviewed Chutoro commit](https://github.com/leynos/chutoro/blob/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/chutoro-providers/dense/src/simd/mod.rs).
[^2]: [Scalar kernels and normalization at the reviewed commit](https://github.com/leynos/chutoro/blob/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/chutoro-providers/dense/src/simd/kernels/mod.rs).
[^3]: [Packed point storage at the reviewed commit](https://github.com/leynos/chutoro/blob/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/chutoro-providers/dense/src/simd/point_view.rs).
[^4]: [Backend eligibility and priority at the reviewed commit](https://github.com/leynos/chutoro/blob/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/chutoro-providers/dense/src/simd/dispatch.rs).
[^5]: [Existing numerical parity contract at the reviewed commit](https://github.com/leynos/chutoro/blob/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/chutoro-providers/dense/src/simd/semantics.rs).
[^6]: [Fearless SIMD 1.2.0 API documentation](https://docs.rs/fearless_simd/1.2.0/fearless_simd/)
      and
      [capability implementation at its release commit](https://github.com/linebender/fearless_simd/blob/5fbc09b20058eee19ae2233e7ab95c5ea999ae81/fearless_simd/src/lib.rs).
[^7]: [Core release manifest](https://github.com/linebender/fearless_simd/blob/5fbc09b20058eee19ae2233e7ab95c5ea999ae81/fearless_simd/Cargo.toml)
      and
      [macro 0.1.0 documentation](https://docs.rs/fearless_simd_macros/0.1.0/fearless_simd_macros/).
[^8]: [Workspace package metadata](https://github.com/leynos/chutoro/blob/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/Cargo.toml)
      and
      [dense package manifest](https://github.com/leynos/chutoro/blob/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/chutoro-providers/dense/Cargo.toml).
