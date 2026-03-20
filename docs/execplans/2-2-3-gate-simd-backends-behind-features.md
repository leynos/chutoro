# Execution plan (ExecPlan): roadmap 2.2.3 gate single instruction, multiple data (SIMD) backends behind Cargo features with runtime dispatch

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & discoveries`, `Decision log`, and
`Outcomes & retrospective` must be kept up to date as work proceeds.

Status: COMPLETE

## Purpose / big picture

Implement roadmap item `2.2.3` by making the dense-provider SIMD backends
compile only when their Cargo features are enabled, while still selecting the
best enabled backend exactly once at runtime and keeping the hot path free of
per-call feature branches.

Success is observable when:

- `chutoro-providers-dense` exposes Cargo features `simd_avx2`,
  `simd_avx512`, and `simd_neon`;
- the dense provider keeps current scalar correctness and all-or-nothing output
  semantics when every SIMD feature is disabled;
- x86 builds with enabled SIMD features continue to patch a function pointer
  once and then execute branch-free hot loops;
- the implementation defines one non-finite policy for scalar and SIMD
  reductions so CPU behaviour is stable now and GPU parity work has a clear
  contract later;
- `DensePointView<'a>` continues to guarantee 64-byte alignment, 16-lane
  padding, and `0.0_f32` tail fill, with regression coverage that proves those
  invariants still hold after feature gating lands;
- unit tests, using `rstest` parameterization where repetition exists, cover
  happy paths, unhappy paths, feature-disabled fallback, dispatch selection,
  non-finite handling, and edge cases;
- `docs/chutoro-design.md` records the shipped dispatch and non-finite policy;
- `docs/roadmap.md` marks item `2.2.3` done only after implementation and all
  validation commands succeed.

Implementation is complete. This document now serves as the execution record.

## Constraints

- Keep Rust source files under 400 lines. Split modules or tests rather than
  extending existing SIMD files past that limit.
- Preserve the public `DenseMatrixProvider` and `DataSource` interfaces.
- Do not add new crate dependencies. Use the standard library and existing
  workspace crates only.
- Preserve the existing dense-provider output contract: on failure, caller
  output buffers remain unchanged.
- Preserve the existing `DensePointView<'a>` storage contract introduced by
  roadmap item `2.2.2`: 64-byte aligned base storage, point counts padded to a
  16-lane multiple, and `0.0_f32` tail padding.
- Keep one-time function-pointer patching for hot paths. Do not add a runtime
  feature branch inside the per-distance loop.
- Preserve current scalar fallback behaviour for unsupported targets, disabled
  backend features, empty batches, one-point batches, and arbitrary-pair
  batches that do not use the Structure of Arrays (SoA) query-points path.
- Use `rstest` for repeated dispatch, feature-matrix, and non-finite test
  cases.
- Follow guidance from:
  - `docs/chutoro-design.md` (especially §6.3)
  - `docs/property-testing-design.md`
  - `docs/complexity-antipatterns-and-refactoring-strategies.md`
  - `docs/rust-testing-with-rstest-fixtures.md`
  - `docs/rust-doctest-dry-guide.md`

## Tolerances (exception triggers)

- Scope: if implementation requires edits in more than 10 files or more than
  800 net lines, stop and escalate before continuing.
- Interface: if feature gating appears to require a public API change outside
  `chutoro-providers/dense`, stop and escalate with options.
- Defaults: if preserving current default performance requires feature
  forwarding through additional crates, stop and decide explicitly whether the
  dense crate should default-enable SIMD features or the workspace should grow
  explicit forwarding.
- Neon scope: if satisfying `simd_neon` requires building a first production
  Neon intrinsic kernel rather than feature plumbing plus dispatch machinery,
  stop and escalate with the evidence gathered from the prototype checkpoint.
- Behaviour: if scalar and SIMD backends cannot be made to agree on a single
  non-finite result policy without widening the error surface, stop and
  escalate.
- Iterations: if `make lint` or `make test` still fails after 3 repair
  attempts, stop and escalate with the logs.
- Ambiguity: if roadmap wording and the existing §6.3 implementation notes
  disagree materially on whether `simd_neon` must be fully implemented in this
  item, stop and present options with trade-offs.

## Risks

- Risk: compile-time feature gating may silently drop the currently available
  AVX2 or AVX-512 path in default builds, causing an accidental performance
  regression. Severity: high. Likelihood: medium. Mitigation: make default
  feature behaviour an explicit decision, test `--no-default-features` and
  default builds separately, and record the choice in the design document.

- Risk: dispatch logic may become harder to reason about once compile-time
  feature masks and runtime CPU capability checks are mixed together. Severity:
  high. Likelihood: medium. Mitigation: factor selection into a pure helper
  that accepts "compiled support" and "runtime support" inputs, then cover it
  with parameterized unit tests.

- Risk: non-finite values may behave differently between scalar and SIMD
  reductions, especially for infinities and NaNs. Severity: high. Likelihood:
  medium. Mitigation: choose one canonical policy and test it at both the
  kernel layer and provider layer.

- Risk: introducing extra `cfg` branches into `simd/kernels.rs` may turn the
  file into a bumpy-road module. Severity: medium. Likelihood: high.
  Mitigation: isolate feature-mask helpers, backend selection, and any Neon
  experimentation into small single-purpose helpers or sibling modules.

- Risk: `simd_neon` may look complete in Cargo metadata while still lacking a
  real Neon intrinsic implementation. Severity: medium. Likelihood: medium.
  Mitigation: add an explicit checkpoint that determines whether this roadmap
  item is meant to ship Neon intrinsics now or just the feature/disptach slot,
  and document the result in both this plan and `docs/chutoro-design.md`.

## Progress

- [x] (2026-03-10 00:00Z) Drafted ExecPlan for roadmap item `2.2.3`.
- [x] (2026-03-11 00:20Z) Implementation approved and started.
- [x] (2026-03-11 00:55Z) Stage A complete: added selector and non-finite
  contract coverage in `chutoro-providers/dense/src/simd/tests.rs`.
- [x] (2026-03-11 01:20Z) Stage B complete: added dense-crate features plus a
  dedicated dispatch module and split typed wrappers out of the overlong
  `simd/mod.rs`.
- [x] (2026-03-11 01:45Z) Stage C complete: wired feature-gated AVX2,
  AVX-512, and Neon backends through one-time runtime selection and
  canonicalized non-finite reductions to `f32::NAN`.
- [x] (2026-03-11 02:10Z) Stage D complete: updated design documentation,
  marked roadmap item `2.2.3` done, and validated the dense feature matrix.
- [x] (2026-03-11 02:40Z) Re-ran repository quality gates after the final
  documentation fix.
- [x] (2026-03-18 00:00Z) Follow-up review fixes revalidated the dense
  `simd_neon`-only feature build with
  `cargo test -p chutoro-providers-dense --no-default-features --features simd_neon`.

## Surprises & discoveries

- Observation: runtime x86 dispatch already exists today, but it is not yet
  guarded by Cargo features. Evidence:
  `chutoro-providers/dense/src/simd/kernels.rs` currently uses `OnceLock` plus
  `is_x86_feature_detected!` to choose `Scalar`, `Avx2`, or `Avx512` without
  consulting feature flags. Impact: `2.2.3` is primarily feature-gating and
  dispatch-policy work, not a first introduction of runtime dispatch.

- Observation: `DensePointView<'a>` already provides the alignment and padding
  guarantees repeated in roadmap item `2.2.3`. Evidence:
  `chutoro-providers/dense/src/simd/point_view.rs` uses
  `#[repr(C, align(64))]`, pads to `MAX_SIMD_LANES`, and zero-fills unused
  lanes. Impact: this item should preserve and re-test those invariants rather
  than redesign them.

- Observation: the dense crate now declares explicit SIMD Cargo features in
  `chutoro-providers/dense/Cargo.toml`. Impact: this item made an explicit
  decision to keep them default-enabled rather than opt-in.

- Observation: Hierarchical Navigable Small World (HNSW) already rejects
  non-finite batch outputs after the data source returns them. Evidence:
  `chutoro-core/src/hnsw/validate.rs::validate_batch_without_cache`. Impact:
  the dense-provider SIMD contract can canonicalize non-finite results without
  changing the core error surface.

- Observation: there is no existing Neon kernel implementation in the dense
  provider. Evidence: the only current backend-specific intrinsics live under
  x86/x86_64 `std::arch` imports in
  `chutoro-providers/dense/src/simd/kernels.rs`. Impact: `simd_neon` is the
  only part of this roadmap item with genuine scope ambiguity.

- Observation: `simd/mod.rs` was already above the repository's 400-line target
  before `2.2.3` work started. Evidence: `wc -l` reported 423 lines before the
  implementation sweep. Impact: the feature-gating work split typed wrappers
  into `simd/types.rs` and backend selection into `simd/dispatch.rs` instead of
  adding more logic to the existing file.

- Observation: single-backend validation builds such as
  `cargo test -p chutoro-providers-dense --no-default-features --features simd_avx512`
   triggered dead-code warnings on private backend helpers that are only
  reached through the runtime dispatch table. Impact: feature-specific internal
  helpers should prefer exact conditional compilation so disabled backends are
  not compiled at all; if one helper still needs suppression, use the smallest
  possible `#[expect(dead_code, reason = "...")]` instead of `#[allow]`.

## Decision log

- Decision: plan around feature-gating the dense crate itself first, and avoid
  cross-crate feature forwarding unless testing proves it is necessary to
  preserve default behaviour. Rationale: only the dense provider owns the SIMD
  code today, so that is the narrowest coherent place to start. Date/Author:
  2026-03-10 / Codex.

- Decision: prefer a pure dispatch selector shaped like
  `choose_backend(compiled, runtime)` over scattering `cfg` and CPUID checks
  through the hot-path call sites. Rationale: this keeps the behaviour testable
  and preserves branch-free steady-state kernel calls. Date/Author: 2026-03-10
  / Codex.

- Decision: treat non-finite handling as an explicit contract of this roadmap
  item, not as an incidental property of IEEE arithmetic. Rationale: the
  roadmap and §6.3 explicitly call out CPU/GPU parity, and parity requires a
  written rule. Date/Author: 2026-03-10 / Codex.

- Decision: keep `DensePointView<'a>` unchanged unless the feature-gating work
  proves that a stronger invariant or helper is genuinely required. Rationale:
  `2.2.2` already delivered the layout contract; `2.2.3` should consume it, not
  reopen it. Date/Author: 2026-03-10 / Codex.

- Decision: keep `simd_avx2`, `simd_avx512`, and `simd_neon` default-enabled in
  `chutoro-providers-dense`. Rationale: the roadmap item is about gating
  backend compilation, not silently degrading the current default performance
  profile for downstream users. Date/Author: 2026-03-11 / Codex.

- Decision: implement a real Neon backend rather than leaving `simd_neon` as a
  placeholder flag. Rationale: roadmap wording names Neon as a backend, and the
  stable `std::arch` intrinsics needed for the existing Euclidean kernels were
  sufficient to stay within scope. Date/Author: 2026-03-11 / Codex.

- Decision: canonicalize any non-finite Euclidean reduction result to
  `f32::NAN` across scalar and SIMD paths. Rationale: this yields one stable
  invalid-output class and aligns with the HNSW validator, which already
  rejects non-finite distances. Date/Author: 2026-03-11 / Codex.

## Outcomes & retrospective

Implemented outcomes:

- dense SIMD backends are enabled or disabled by Cargo features;
- runtime selection still happens once and patches function pointers before
  hot-loop execution;
- scalar and SIMD paths now share one documented non-finite policy:
  canonicalize invalid reductions to `f32::NAN`;
- `simd_neon` now maps to a real ARM/AArch64 backend rather than an empty
  feature slot;
- `simd/mod.rs` was reduced by moving typed wrappers into `simd/types.rs` and
  backend selection into `simd/dispatch.rs`;
- roadmap item `2.2.3` is marked done.

Validation summary:

- targeted dense validation passed:
  - workspace-style dense test run
  - dense all-features Clippy run with warnings denied
  - dense no-default-features test run
  - dense `simd_avx2` feature-only test run
  - dense `simd_avx512` feature-only test run
  - dense `simd_neon` feature-only test run
  - dense all-features test run
- repository gates passed:
  - `make fmt`
  - `make markdownlint`
  - `make nixie`
  - `make check-fmt`
  - `make lint`
  - `make test`

Retrospective:

- The pure selector approach kept dispatch behaviour easy to test and let the
  feature-matrix validation exercise cases that the host CPU could not execute
  directly.
- The most pragmatic Neon implementation reused the existing Euclidean kernel
  shape rather than widening scope into a new packing strategy.
- Canonical `NaN` outputs provide a clearer contract than leaving scalar and
  SIMD backends to differ on `NaN` versus `infinity`.

## Context and orientation

Relevant current files and behaviour:

- `chutoro-providers/dense/Cargo.toml`
  - now declares default-enabled `simd_avx2`, `simd_avx512`, and `simd_neon`
    features;
  - therefore the dense crate can be built in scalar-only or selectively
    enabled SIMD modes.

- `chutoro-providers/dense/src/simd/kernels.rs`
  - defines `EuclideanBackend::{Scalar, Avx2, Avx512}`;
  - selects backends with `is_x86_feature_detected!`;
  - stores selected function pointers in `OnceLock`, which already provides the
    desired one-time patching shape for this roadmap item.

- `chutoro-providers/dense/src/simd/mod.rs`
  - exposes the row-major and Structure of Arrays (SoA) batch entrypoints used
    by
    `DenseMatrixProvider`;
  - currently asks `dispatch::euclidean_backend()` whether query-point packing
    should be used, but does not distinguish "compiled out" from
    "runtime unavailable".

- `chutoro-providers/dense/src/simd/point_view.rs`
  - packs selected dense rows into a dimension-major SoA view;
  - guarantees 64-byte alignment, lane-multiple padding, and deterministic
    zero-filled tails.

- `chutoro-providers/dense/src/provider.rs`
  - routes `distance_batch(...)` into
    `simd::euclidean_distance_batch_raw_pairs(...)`;
  - preserves all-or-nothing writes by letting the SIMD layer compute into a
    temporary buffer first.

- `chutoro-providers/dense/src/simd/tests.rs`
  - already covers SoA packing, output-length mismatches, parity with scalar
    references, output preservation on error, feature-gated backend selection,
    and explicit non-finite reduction rules.

- `chutoro-core/src/hnsw/validate.rs`
  - rejects non-finite results after `batch_distances(...)` returns;
  - therefore the dense-provider contract should remain "return a value;
    non-finite means invalid and will be rejected upstream" rather than
    changing return types.

Terms used in this plan:

- Cargo feature: a compile-time switch chosen by the build command, for example
  `--features simd_avx2`.
- CPUID runtime dispatch: checking CPU capabilities once at runtime and storing
  the chosen function pointer so future calls jump directly to the selected
  implementation.
- Non-finite: a floating-point value that is `NaN`, `+infinity`, or
  `-infinity`.
- Canonical non-finite result: a single agreed result, such as `f32::NAN`,
  returned whenever invalid input or reduction state is encountered.

## Plan of work

### Stage A: establish the contract with failing tests (red)

Add tests that describe the intended behaviour before production code changes.

Planned edits:

- `chutoro-providers/dense/src/simd/tests.rs`
  - add parameterized `rstest` coverage for a pure backend selector helper,
    with cases that vary:
    - compile-time enabled backends;
    - runtime x86 capability sets;
    - expected selected backend;
    - expected query-point packing eligibility;
  - add parameterized non-finite tests that exercise:
    - `NaN` in the query row;
    - `NaN` in a packed point row;
    - `+infinity` and `-infinity` in either input;
    - scalar and available SIMD entrypoints returning the same canonical
      non-finite result;
  - keep existing alignment and zero-padding tests in place as regression
    coverage.

- `chutoro-providers/dense/src/tests/provider.rs`
  - add provider-level tests proving that disabling SIMD features falls back to
    scalar behaviour without changing results or error semantics;
  - add unhappy-path coverage showing non-finite outputs stay non-finite and
    are not silently clamped.

Go/no-go:

- Do not proceed unless at least one new dispatch or non-finite test fails
  against the current implementation.

### Stage B: add feature flags and a testable dispatch model (green part 1)

Introduce compile-time backend masks and make backend selection explicit.

Planned edits:

- `chutoro-providers/dense/Cargo.toml`
  - add:

  ```toml
  [features]
  default = ["simd_avx2", "simd_avx512", "simd_neon"]
  simd_avx2 = []
  simd_avx512 = []
  simd_neon = []
  ```

  Keep the default set explicit so the current dense-provider performance
  profile remains unchanged unless a build opts out.

- `chutoro-providers/dense/src/simd/kernels.rs`
  - add a small compile-time description of enabled backends, for example
    `CompiledSimdSupport`;
  - add a small runtime capability description, for example
    `RuntimeSimdSupport`;
  - add a pure selector helper, for example
    `choose_euclidean_backend(compiled, runtime) -> EuclideanBackend`;
  - gate x86 entrypoints with both target-architecture `cfg` and the relevant
    Cargo feature `cfg`;
  - leave the function-pointer `OnceLock` shape in place, but make the
    initializer depend on the pure selector helper instead of ad hoc checks.

- `chutoro-providers/dense/src/simd/mod.rs`
  - replace any direct assumption that `Avx2` or `Avx512` exists with a helper
    that asks the selector which backend was actually compiled and chosen;
  - keep the hot path free of repeated dispatch logic.

Implementation notes:

- Treat `simd_avx512` and `simd_avx2` as independent compile-time toggles.
  When AVX-512 is enabled but AVX2 is disabled on an AVX2-only machine, the
  selector should return `Scalar`, not assume AVX2 is present.
- Keep the selector deterministic and order-preserving:
  `Avx512` wins over `Avx2`, which wins over `Neon`, which wins over `Scalar`,
  but only when both compile-time and runtime support exist.

Go/no-go:

- Do not proceed unless selector unit tests pass and existing parity tests do
  not regress.

### Stage C: resolve the non-finite policy and the `simd_neon` checkpoint (green part 2)

Complete the behaviour contract that §6.3 requires.

Planned edits:

- `chutoro-providers/dense/src/simd/kernels.rs`
  - add a tiny helper that canonicalizes reduction outputs:
    - if the final scalar or SIMD accumulation is finite, return the distance
      as today;
    - if any input or the final accumulation is non-finite, return
      `f32::NAN` so callers observe one stable invalid result class;
  - apply the helper consistently to scalar and query-points kernels.

- `docs/chutoro-design.md`
  - record the chosen non-finite policy in §6.3 and note that HNSW will reject
    the result as `NonFiniteDistance`.

- `chutoro-providers/dense/src/simd/kernels.rs` or a sibling ARM-specific
  module
  - perform a narrow `simd_neon` checkpoint:
    - determine whether stable `std::arch::{arm,aarch64}` intrinsics are
      sufficient for a bounded Neon implementation in this roadmap item;
    - if yes, add a Neon backend variant and runtime selection helper behind
      `simd_neon`;
    - if no, stop, document the blocker, and escalate before widening scope.

Implementation notes:

- The non-finite policy must be chosen once and reused everywhere. Do not let
  scalar return `infinity` while SIMD returns `NaN`, or vice versa.
- If the Neon checkpoint reveals that a real Neon intrinsic kernel is required
  but exceeds the tolerances, this ExecPlan must move to `BLOCKED` until the
  user resolves the scope question.

Go/no-go:

- Do not proceed to roadmap closure unless non-finite tests pass and the Neon
  checkpoint outcome is explicitly documented.

### Stage D: update docs, close the roadmap item, and run full validation

Record the final design and prove the repository is healthy.

Planned edits:

- `docs/chutoro-design.md`
  - update §6.3 with:
    - the new Cargo features;
    - the one-time runtime dispatch model;
    - the canonical non-finite policy;
    - the final decision on `simd_neon`;
    - a note that `DensePointView<'a>` alignment and zero-padding invariants are
      preserved from `2.2.2`.

- `docs/roadmap.md`
  - mark item `2.2.3` done (`[x]`) only after all validation commands pass.

Validation commands:

```sh
set -o pipefail; make fmt 2>&1 | tee /tmp/2-2-3-make-fmt.log
set -o pipefail; make markdownlint 2>&1 | tee /tmp/2-2-3-make-markdownlint.log
set -o pipefail; make nixie 2>&1 | tee /tmp/2-2-3-make-nixie.log
set -o pipefail; cargo test -p chutoro-providers-dense --no-default-features 2>&1 | tee /tmp/2-2-3-dense-no-default-features.log
set -o pipefail; cargo test -p chutoro-providers-dense --no-default-features --features simd_avx2 2>&1 | tee /tmp/2-2-3-dense-avx2.log
set -o pipefail; cargo test -p chutoro-providers-dense --no-default-features --features simd_avx512 2>&1 | tee /tmp/2-2-3-dense-avx512.log
set -o pipefail; cargo test -p chutoro-providers-dense --no-default-features --features simd_neon 2>&1 | tee /tmp/2-2-3-dense-neon.log
set -o pipefail; cargo test -p chutoro-providers-dense --all-features 2>&1 | tee /tmp/2-2-3-dense-all-features.log
set -o pipefail; make check-fmt 2>&1 | tee /tmp/2-2-3-make-check-fmt.log
set -o pipefail; make lint 2>&1 | tee /tmp/2-2-3-make-lint.log
set -o pipefail; make test 2>&1 | tee /tmp/2-2-3-make-test.log
```

Expected success signals:

- scalar-only dense tests pass with `--no-default-features`;
- enabled-feature dense tests pass for the explicit feature combinations above;
- `make check-fmt` exits `0`;
- `make lint` exits `0` with no Clippy warnings;
- `make test` exits `0` and includes the new dispatch and non-finite tests in
  the summary.

If the plain local `make test` run hits the existing repository-specific
nextest stall, capture the log, confirm whether the stall is unrelated, and do
not mark the work complete until a plain `make test` run succeeds.

## Acceptance evidence to capture during implementation

Add concise evidence to this document as the work progresses:

```plaintext
- Red stage:
  - cargo test -p chutoro-providers-dense simd::tests::<new selector test>
    ... FAILED ... expected Scalar when AVX2 feature disabled ...
  - cargo test -p chutoro-providers-dense simd::tests::<new non_finite test>
    ... FAILED ... scalar returned inf but SIMD returned NaN ...

- Green stage:
  - cargo test -p chutoro-providers-dense simd::tests::
    ... ok
  - cargo test -p chutoro-providers-dense --no-default-features
    ... ok
  - cargo test -p chutoro-providers-dense --all-features
    ... ok

- Final gates:
  - make check-fmt ... ok
  - make lint ... ok
  - make test ... ok
```

## Proposed implementation order

1. Add red tests for backend selection and non-finite policy.
2. Add dense-crate Cargo features and a pure dispatch selector.
3. Wire `OnceLock` initializers through the new selector.
4. Resolve and document the canonical non-finite policy.
5. Complete the `simd_neon` checkpoint and either implement it within
   tolerances or escalate.
6. Update `docs/chutoro-design.md`.
7. Mark `docs/roadmap.md` item `2.2.3` done only after all gates pass.

## Historical note

This ExecPlan begins as the required draft for roadmap task `2.2.3`. It should
remain in `DRAFT` status until the user approves implementation.
