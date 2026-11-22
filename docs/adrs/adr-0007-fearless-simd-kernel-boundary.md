# Architectural decision record (ADR) 0007: Fearless SIMD kernel boundary

## Status

Proposed. The maintainer authorized the dependency intention and this design
record on 2026-11-22. Production adoption remains subject to verification and
measurement; this record does not claim a shipped port.

## Date

2026-11-22.

## Context and problem statement

The dense provider maintains architecture-specific Euclidean kernels, an
optional nightly portable implementation, a scalar reference, and cached
backend selection. Its packed query-to-points kernels express the same
coordinate loop with different intrinsics. Consolidating that loop can reduce
maintenance and local unsafe code without changing the provider contract.[^1]

Roadmap task 2.3.2 contemplates vectorizing mutual-reachability weighting and
candidate preparation in the minimum spanning tree (MST) pipeline. That work
has a separate performance justification. Its historical measurements cannot
justify either a current speed-up claim or rejection of a maintenance
improvement in the dense provider.

Fearless SIMD 1.2.0 supports stable Rust 1.89 and safe portable vector
operations. Its optional macro crate 0.1.0 supplies `#[simd]`. Its x86 tokens
require broader feature bundles than chutoro's current `avx2` and `avx512f`
checks, so replacing the dispatcher blindly changes which implementation
executes on some machines.[^2]

## Decision drivers

- Preserve public interfaces, feature controls, and the scalar path.
- Keep numerical policy and algorithmic ordering under chutoro's ownership.
- Reduce repeated kernels without a shared framework, new public trait, or
  dependency from core to the dense provider.
- Preserve ADR-0003's private dense storage boundary and ADR-0006's
  prohibition on Rayon scheduling inside insertion callbacks.
- Require behaviour and throughput evidence for a consolidation, and a
  measured opportunity for a new optimization.

## Proposed direction

Prefer Fearless SIMD for explicit SIMD leaf kernels that pass their local
adoption gate. Start with the bounded query-to-points experiment in
[RFC 0001](../rfcs/rfc-0001-fearless-simd-query-to-points.md) and roadmap task
2.2.8. Keep pairwise kernels outside that experiment. Task 2.3.2 may use the
same library independently after its measurement gates pass.

### Dependency authorization

The maintainer authorizes a narrow exception to 2.3.2's no-new-dependency
constraint and manifest-change tolerance:

- `fearless_simd = "1.2"` for the bounded dense kernel and any justified,
  gated core CPU kernel.
- `fearless_simd_macros = "0.1"` if the implementation uses `#[simd]`.

These are SemVer-compatible caret requirements. Authorization covers the
necessary manifest, feature wiring, and lockfile changes in a later
implementation. It does not authorize unrelated dependencies, a minimum
supported Rust version (MSRV) increase, unconditional MST vectorization, or
changes to public feature semantics. This documentation change adds neither
dependency.

Declare requirements centrally in `[workspace.dependencies]` when an
implementation introduces them, and opt in only consuming crates. Make dense
dependencies optional and activate them through its supported existing native
SIMD features; keep core dependencies optional and attached to `cpu` if a core
kernel ships. Scalar-only dense and non-CPU core builds must not activate an
unused backend dependency or gain a requirement to execute SIMD.

Retain Rust 1.89 compatibility. Check each consumer's effective `rust-version`,
selected dependency graph, and build under that compiler. Workspace package
metadata requires explicit inheritance. Future compatible Fearless releases may
raise their MSRV, so a caret requirement or a successful development toolchain
build alone does not prove compatibility.[^2]

### Ownership and execution boundary

Public provider APIs continue to accept ordinary Rust values. No Fearless
token, vector, or mask appears in a public signature. Dense packing, alignment,
allocation, and candidate order remain private to `DensePointView` and the
existing adapter.

The private query dispatcher respects the existing selector and Cargo feature
permissions first. It caches a query invocation containing either a safely
obtained token for that approved backend or its existing kernel function
pointer. One match per batch invokes the generic kernel or established
implementation. A bare function pointer cannot capture a token, so
token-bearing state belongs in this private invocation value.

Use the exact safe token only when its full capability requirements hold and
the replacement has execution evidence on that backend. Preserve the
established query implementation otherwise, including supported 32-bit ARM NEON
paths. Keep the pairwise cache and selector unchanged. An unrestricted
`dispatch!` call must not select a backend excluded by Cargo features.
Fabricated capability masks may test pure selection logic; they must never
manufacture tokens used to execute instructions.

The packed kernel assigns independent points to lanes and accumulates
dimensions in their existing order. It uses separate multiplication and
addition, retains final-distance normalization, and allocates nothing new.
Exact-width chunks plus a bounded remainder avoid assuming every future native
width divides the current 16-point padding. Safe loads and stores still require
slices of the documented length.[^3]

All dense kernel work runs synchronously on the calling thread. Instruction
selection must not introduce Rayon tasks, blocking work on the same pool, or
index re-entry. This preserves the callback contract under `insert_mutex` in
[ADR-0006](adr-0006-sequential-trim-scoring.md).

Core owns future edge weighting, validation, error selection, sorting, and
compaction. A private leaf kernel consumes checked inputs and returns ordinary
buffers or classifications. Core must not depend on the dense provider, adopt
`DensePointView`, or move domain policy into the SIMD library. Sharing
dependency versions does not require a shared wrapper crate.

### Numerical and validation contracts

Preserve dense `DistanceSemantics`: absolute epsilon `1e-5` and normalization
of non-finite final distances to NaN. Tolerance does not prove identical
neighbour ordering or clustering on near ties, so RFC 0001 requires downstream
evidence as well as kernel parity.[^4]

For 2.3.2, preserve the chained scalar mutual-reachability maxima and state the
zero-sign and NaN policy before choosing vector operations. `max_precise`
handles quiet-NaN selection but does not establish deterministic signed-zero or
signalling-NaN behaviour. Rejecting raw non-finite operands before weighting
can change accepted results. Numerical equality does not prove bit equality;
use explicit bit comparisons where required.[^5]

MST validation retains source-bound, target-bound, finite-weight, then
self-loop precedence; it reports the earliest invalid input and original
endpoint orientation. Validate `usize` endpoints before narrowing or indexed
loads. Preserve sequence numbers, complete records, and output order.
Vectorized classification with ordered scalar record emission is the first
candidate; full compaction requires additional evidence. Mixed-width fields
must describe the same logical edges when native lane counts differ.[^6]

## Verification and adoption

RFC 0001 defines the dense requirements and evidence matrix. Adoption needs
numerical, bounds, feature, native execution, downstream, and benchmark
evidence, with a recorded net reduction in maintained query-kernel code. Count
retained weak-capability and 32-bit ARM fallbacks, dispatch, and proof code;
adding a generic body while retaining every existing body may fail this gate.
Keep the established path for any target whose replacement lacks execution
evidence. Model-check chutoro-owned selection and tail arithmetic; those
bounded proofs do not verify the dependency or hardware.

Task 2.3.2 retains its current-pipeline opportunity threshold and
classification/compaction threshold. Reconcile completed work, measure the
remaining stages, inspect optimized code, and compare complete operations
including gathers, staging, and output construction. Do not revive a post-sort
connectivity sweep merely because masks are available. A Filter-Kruskal change
still needs algorithmic and ordering justification.

Dense consolidation can succeed with equivalent performance and a useful
maintenance reduction. It does not inherit the MST stage-share threshold. An
authorized dependency does not waive the MST gates. A failed or inconclusive
experiment retains the established path and records its result; it must not
leave an enabled experimental implementation.

## Options considered

| Option                                     | Consequence                                                        | Disposition                |
| ------------------------------------------ | ------------------------------------------------------------------ | -------------------------- |
| Keep existing intrinsic implementations    | Preserves behaviour but retains repeated query loops               | Baseline and rollback path |
| Bounded private Fearless kernel            | Tests consolidation with existing layout, controls, and references | Preferred experiment       |
| Replace all kernels and dispatch together  | Mixes reduction, capability, and numerical changes                 | Deferred                   |
| Require nightly `std::simd`                | Changes the default compiler contract                              | Rejected                   |
| Add a shared SIMD framework or public port | Adds scope without demonstrated reuse                              | Rejected for these tasks   |

_Table 1: Alternatives for the initial SIMD migration._

## Known risks and limitations

Stricter capability bundles may leave legacy implementations necessary.
Multiversioning can increase compile time and binary size. Small batches may
spend more time packing or dispatching than calculating distances. Numerical
tolerance can hide changes in neighbour ordering. The RFC requires evidence for
these risks; this record claims no measured improvement.

## References

[^1]: [Dense kernels at the reviewed chutoro baseline](https://github.com/leynos/chutoro/tree/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/chutoro-providers/dense/src/simd/kernels)
    and [backend selection](https://github.com/leynos/chutoro/blob/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/chutoro-providers/dense/src/simd/dispatch.rs).
[^2]: [Fearless SIMD 1.2.0 README](https://github.com/linebender/fearless_simd/blob/5fbc09b20058eee19ae2233e7ab95c5ea999ae81/fearless_simd/README.md)
    and [0.1.0 macro manifest](https://github.com/linebender/fearless_simd/blob/dcbd824a897a098ec0124c088f971ac6d890d0cd/fearless_simd_macros/Cargo.toml).
[^3]: [Fearless SIMD 1.2.0 load/store implementations](https://github.com/linebender/fearless_simd/blob/5fbc09b20058eee19ae2233e7ab95c5ea999ae81/fearless_simd/src/generated/simd_types.rs).
[^4]: [Current dense numerical semantics](https://github.com/leynos/chutoro/blob/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/chutoro-providers/dense/src/simd/semantics.rs).
[^5]: [Fearless SIMD vector operation semantics](https://docs.rs/fearless_simd/1.2.0/fearless_simd/trait.SimdBase.html)
    and [current weighting transform](https://github.com/leynos/chutoro/blob/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/chutoro-core/src/cpu_pipeline/mod.rs).
[^6]: [Current MST validation and ordering](https://github.com/leynos/chutoro/blob/5f13f07e10688ddb0b33e3fb2ef5afbc0e285013/chutoro-core/src/mst/mod.rs).
