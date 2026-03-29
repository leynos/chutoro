# Execution plan (ExecPlan): roadmap 2.2.4 optional nightly-only `std::simd` backend

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & discoveries`, `Decision log`, and
`Outcomes & retrospective` must be kept up to date as work proceeds.

Status: COMPLETED

## Purpose / big picture

Implement roadmap item `2.2.4`: add an optional, nightly-only `std::simd`
(portable SIMD) Euclidean distance backend to the dense provider, behind a
non-default Cargo feature. This gives the project a forward-looking SIMD
implementation using the `portable_simd` API while keeping the stable
`core::arch` intrinsic backends (AVX2, AVX-512, NEON) as the default path on
every stable toolchain.

A nightly CI job validates that the nightly feature compiles and produces
correct results, catching `portable_simd` API breakage before it reaches
contributors.

Success is observable when:

- `chutoro-providers-dense` exposes a new non-default Cargo feature
  `nightly_portable_simd`;
- enabling the feature and compiling with a nightly toolchain activates a
  `std::simd`-based Euclidean distance backend that participates in the
  existing one-time runtime dispatch;
- disabling the feature (the default) compiles and runs on stable Rust exactly
  as before, with no code paths or warnings affected by the nightly backend's
  existence;
- the nightly backend produces pairwise and query-to-points distances that
  agree with the scalar oracle within `1.0e-6` tolerance;
- the nightly backend respects the canonical non-finite policy (canonicalize
  non-finite results to `f32::NAN`);
- the nightly backend integrates into the existing `choose_euclidean_backend`
  dispatch at a defined priority slot;
- unit tests parameterized with `rstest` cover: dispatch selection with the
  portable SIMD backend compiled and available, dispatch selection when the
  feature is disabled, pairwise correctness at lane boundary vector lengths,
  query-to-points batch correctness, non-finite canonicalization, and
  `DensePointView` alignment and padding preservation;
- a new GitHub Actions workflow runs nightly and exercises the feature on a
  nightly Rust toolchain, reporting results independently from the stable CI;
- `docs/chutoro-design.md` §6.3 records the shipped feature name, dispatch
  priority, and tracking issue references;
- `docs/roadmap.md` marks item `2.2.4` done only after all validation commands
  succeed.

This item partially overlaps with roadmap item `2.2.5` (portable-SIMD gating
mechanics). The two items are distinguished by scope: `2.2.4` delivers the
backend implementation, feature gate, dispatch integration, unit tests, and
nightly CI job; `2.2.5` focuses on verifying that stable and nightly builds
coexist safely and adds explicit CI matrix checks for each combination. This
plan delivers the foundation that `2.2.5` will verify.

## Constraints

- Keep Rust source files under 400 lines. Split modules or tests rather than
  extending existing SIMD files past that limit.
- Preserve the public `DenseMatrixProvider` and `DataSource` interfaces. The
  new feature must not alter any public API signatures or error semantics.
- Do not add new crate dependencies. The `std::simd` backend uses only the
  standard library's `core::simd` module, which is available on nightly behind
  `#![feature(portable_simd)]`.
- The new Cargo feature must be non-default. Stable toolchain builds must see
  zero effect from its existence: no conditional compilation noise, no unused
  import warnings, no clippy warnings.
- Preserve the existing dense-provider output contract: on failure, caller
  output buffers remain unchanged.
- Preserve the existing `DensePointView<'a>` storage contract: 64-byte aligned
  base storage, point counts padded to a 16-lane multiple, `0.0_f32` tail
  padding. The new backend must consume these invariants, not alter them.
- Keep one-time function-pointer patching for hot paths. Do not add a runtime
  feature branch inside the per-distance loop.
- The nightly CI job must not block stable CI or pull request workflows. It
  runs on `main` on a schedule, not on pull requests.
- The crate-level `#![feature(portable_simd)]` attribute must only appear
  when the `nightly_portable_simd` feature is enabled. On stable, the attribute
  must be absent.
- Use `rstest` for parameterized tests. Follow guidance from:
  - `docs/chutoro-design.md` (especially §6.3)
  - `docs/property-testing-design.md`
  - `docs/complexity-antipatterns-and-refactoring-strategies.md`
  - `docs/rust-testing-with-rstest-fixtures.md`
  - `docs/rust-doctest-dry-guide.md`
- Use en-GB-oxendict spelling in all comments and documentation.

## Tolerances (exception triggers)

- Scope: if implementation requires edits in more than 15 files or more than
  1000 net lines, stop and escalate before continuing.
- Interface: if feature gating appears to require a public API change outside
  `chutoro-providers/dense`, stop and escalate with options.
- Nightly API breakage: if the `portable_simd` feature API has changed in a
  way that prevents compilation on the latest nightly toolchain, document the
  breakage, add a note to the tracking issue references, and downgrade the
  backend to a `compile_error!` stub behind the feature flag rather than
  blocking the entire plan.
- Non-finite policy: if the `std::simd` reduction produces different
  non-finite behaviour (for example, returning `infinity` where the scalar path
  returns `NaN`), apply `finalize_distance` canonicalization. If it is
  impossible to canonicalize reliably, stop and escalate.
- Iterations: if `make lint` or `make test` still fails after 3 repair
  attempts, stop and escalate with the logs.
- CI: if the nightly workflow cannot be validated locally due to toolchain
  availability, mark the CI stage as untested and document the limitation.

## Risks

- Risk: the `portable_simd` API is unstable and changes frequently on nightly,
  which may cause the nightly CI job to break unexpectedly. Severity: medium.
  Likelihood: high. Mitigation: isolate the nightly code behind tight feature
  gates so breakage is contained; document the tracking issues; accept that
  nightly breakage is expected and does not block stable work.

- Risk: dispatch priority for the portable SIMD backend may inadvertently
  downgrade performance on machines where a platform-specific backend (AVX-512,
  AVX2) would be faster. Severity: high. Likelihood: low. Mitigation: place the
  portable SIMD backend below all platform-specific backends in the dispatch
  priority order (AVX-512 > AVX2 > NEON > PortableSimd > Scalar), so it only
  activates when no platform-specific backend is both compiled and available.

- Risk: `cfg_attr` for crate-level `#![feature(portable_simd)]` may interact
  poorly with workspace-wide clippy or doc builds that use `--all-features`.
  Severity: medium. Likelihood: medium. Mitigation: the `make lint` and
  `make test` targets use `--all-features`, which will enable the nightly
  feature on stable and trigger a compile error. To avoid this, either make the
  feature conditional on nightly detection or ensure the `--all-features` path
  handles it. The recommended solution is to NOT include
  `nightly_portable_simd` in the `--all-features` gate by using a `cfg`
  attribute that depends on both the feature flag AND a nightly compiler
  detection (for example, `cfg(nightly)` via build script or
  `#[cfg(feature = "nightly_portable_simd")]` guarding `#![feature(...)]` with
  `cfg_attr`). Alternative: exclude the feature from `--all-features` by
  keeping it out of any default or forwarded feature set and accepting that
  `--all-features` on stable will fail; adjust `Makefile` to exclude the
  feature. The plan should prototype both approaches in Stage A and select one.

- Risk: the `std::simd` portable vector width may not align with the 16-lane
  packed storage in `DensePointView`, requiring multiple iterations or padding
  management. Severity: low. Likelihood: medium. Mitigation: use
  `Simd<f32, 16>` which matches `MAX_SIMD_LANES` (16), and handle tail elements
  with a scalar loop like the existing backends.

- Risk: `simd/kernels.rs` is already 182 lines; adding a new nightly kernel
  submodule reference plus dispatch plumbing may push the parent module close
  to the 400-line limit. Severity: low. Likelihood: low. Mitigation: the kernel
  implementations should live in a new `kernels/portable_simd.rs` submodule,
  keeping `kernels.rs` as a dispatch hub.

## Progress

- [x] Drafted ExecPlan for roadmap item `2.2.4`.
- [x] Plan approved and implementation started.
- [x] Added a non-default `nightly_portable_simd` feature plus build-script
  nightly detection that keeps stable `--all-features` builds compiling.
- [x] Implemented a nightly-only `Simd<f32, 16>` Euclidean backend for
  pairwise and query-to-points dense kernels.
- [x] Extended dispatch tests and entrypoint parity tests for the portable
  SIMD backend.
- [x] Added `.github/workflows/nightly-portable-simd.yml`.
- [x] Updated `docs/chutoro-design.md` §6.3 and marked roadmap item `2.2.4`
  complete.
- [x] Run all validation commands and record outcomes.

## Surprises & discoveries

- `cfg(nightly)` needed an explicit `cargo:rustc-check-cfg=cfg(nightly)` line
  in `build.rs` so stable warning-deny builds would not fail under
  `unexpected_cfgs`.
- The existing dense SIMD structure already separated dispatch from kernel
  implementations, so the nightly backend fit cleanly as a new kernel module
  and enum variant without widening public APIs.

## Decision log

- Chose the build-script `cfg(nightly)` approach instead of changing
  `Makefile` feature lists so stable `make lint` and `make test` can continue
  to use `--all-features`.
- Kept `PortableSimd` below AVX-512, AVX2, and NEON in dispatch order to avoid
  displacing hand-tuned intrinsic kernels on supported machines.
- Matched the existing structure-of-arrays (SoA) packing contract by using
  `Simd<f32, 16>` rather than a variable lane width.

## Outcomes & retrospective

- The dense provider now ships an optional nightly-only portable SIMD backend
  that coexists with stable builds and existing intrinsic backends.
- The implementation stayed local to the dense crate and preserved all public
  provider interfaces and error contracts.
- Final validation results are recorded after the code, lint, test, and
  documentation gates complete.

## Context and orientation

This section describes the current state of the dense-provider SIMD
infrastructure as it exists before `2.2.4` work begins.

### Repository layout

The SIMD module lives in `chutoro-providers/dense/src/simd/` and is structured
as:

```plaintext
chutoro-providers/dense/src/simd/
├── mod.rs             — Public API: euclidean_distance,
│                        euclidean_distance_batch_raw_pairs,
│                        query-point detection and packing.
├── dispatch.rs        — EuclideanBackend enum, CompiledSimdSupport,
│                        RuntimeSimdSupport, choose_euclidean_backend
│                        pure selector, OnceLock-cached euclidean_backend().
├── kernels.rs         — Kernel function type aliases, OnceLock dispatch
│                        statics, select_backend_fn! macro, scalar kernels,
│                        feature-gated entrypoints for AVX2/AVX-512/NEON.
├── kernels/
│   ├── x86_simd.rs    — AVX2 and AVX-512 squared-L2 + query-points kernels
│   │                    using std::arch intrinsics.
│   └── neon_simd.rs   — ARM NEON squared-L2 + query-points kernels using
│                        std::arch intrinsics.
├── point_view.rs      — DensePointView: 64-byte aligned structure-of-arrays
│                        (SoA) packing with 16-lane padding and 0.0 tail fill.
├── types.rs           — Domain wrappers (Distance, RowSlice, RowIndex, etc.)
│                        for boundary API.
├── tests.rs           — Unit tests: SoA packing, dispatch selection, scalar
│                        correctness, non-finite canonicalization.
└── tests/
    └── entrypoints.rs — Feature-gated entrypoint parity tests for x86 and
                         NEON backends.
```

### Existing Cargo features

`chutoro-providers/dense/Cargo.toml` declares:

```toml
[features]
default = ["simd_avx2", "simd_avx512", "simd_neon"]
simd_avx2 = []
simd_avx512 = []
simd_neon = []
```

All three SIMD features are default-enabled.

### Dispatch model

`dispatch.rs` defines a `choose_euclidean_backend(compiled, runtime)` pure
selector that returns the best backend from the priority order: AVX-512 > AVX2
> NEON > Scalar. The selected backend is cached in an `OnceLock` static.
`kernels.rs` uses a `select_backend_fn!` macro to map the cached backend enum
to a concrete function pointer at init time.

### Non-finite policy

All kernels route results through `finalize_distance(value: f32) -> f32`, which
returns `f32::NAN` for any non-finite result. This policy is documented in
`docs/chutoro-design.md` §6.3.

### Existing CI workflows

The repository has four CI workflows:

- `ci.yml`: PR-gated stable builds, lint, test, coverage.
- `nightly-kani.yml`: scheduled daily at 02:00 UTC on `main`; runs formal
  verification with a gate binary that checks for recent commits.
- `benchmark-regressions.yml`: weekly benchmark regression detection.
- `property-tests.yml`: weekly property-based testing.

The stable `rust-toolchain.toml` pins channel `1.93.1`. The nightly portable
SIMD job will use a separate `nightly` toolchain override.

### Makefile targets

Key targets:

- `make test`: runs `cargo nextest run --all-targets --all-features` with
  `RUSTFLAGS="-D warnings"`.
- `make lint`: runs `cargo clippy --all-targets --all-features -- -D warnings`
  plus `cargo doc --workspace --no-deps`.
- `make check-fmt`: runs `cargo fmt --all -- --check`.

Because `--all-features` is used, the `nightly_portable_simd` feature will be
enabled during `make lint` and `make test`. This means the feature gate must
compile silently on stable (the `#![feature(portable_simd)]` must not appear)
or the Makefile must be adjusted. This is a key technical decision addressed in
Stage A.

### Terms used in this plan

- `portable_simd`: Rust's nightly-only `std::simd` / `core::simd` API for
  cross-platform SIMD operations (`rust-lang/rust#86656`).
- `core::arch`: Rust's stable platform-specific intrinsics module (for
  example, `std::arch::x86_64::_mm256_loadu_ps`).
- non-finite: a floating-point value that is `NaN`, `+infinity`, or
  `-infinity`.
- dispatch priority: the ordered preference list used by
  `choose_euclidean_backend` to select the best available backend.
- lane: a single `f32` slot within a SIMD register.

## Plan of work

### Stage A: resolve the `--all-features` coexistence strategy (no code changes beyond prototyping)

The central technical challenge is that `make lint` and `make test` pass
`--all-features`, which will enable `nightly_portable_simd`. If
`#![feature(portable_simd)]` is emitted on stable, the build fails.

This stage researches and decides the coexistence strategy. Two options are
evaluated:

**Option 1: `cfg_attr` with nightly detection.** Use a build script or
`rustc_attrs` feature detection to emit `cargo:rustc-cfg=nightly` on nightly
compilers. Gate
`#![cfg_attr(feature = "nightly_portable_simd", feature(portable_simd))]`
behind the `nightly` cfg. On stable, the `cfg_attr` expands to nothing even
when `--all-features` enables the Cargo feature.

**Option 2: exclude from `--all-features` via Makefile.** Adjust the `Makefile`
targets to exclude `nightly_portable_simd` by using explicit feature lists
instead of `--all-features`, or accept that `--all-features` on stable will
fail and add a separate nightly lint/test target.

The recommended approach is **Option 1** because it preserves the existing
`--all-features` workflow and requires no Makefile changes. A `build.rs` that
emits `cargo:rustc-cfg=nightly` when the compiler reports a nightly version
string is a well-established pattern.

Planned research:

- Confirm that `rustc --version` output on nightly includes the string
  `"nightly"` and on stable does not.
- Confirm that `cargo:rustc-cfg=nightly` emitted by `build.rs` makes
  `#[cfg(nightly)]` available to the crate.
- Prototype the `cfg_attr` gating in a scratch file and verify it compiles
  cleanly on stable with the feature enabled.

Go/no-go:

- If Option 1 is confirmed feasible, proceed with it.
- If neither option works within tolerances, stop and escalate.

### Stage B: add feature flag, build script, and crate-level gating

Add the Cargo feature, build script, and crate-level `#![feature(...)]` gating.

Planned edits:

- `chutoro-providers/dense/Cargo.toml`
  - Add `nightly_portable_simd = []` to the `[features]` table. Do not add it
    to `default`.

- `chutoro-providers/dense/build.rs` (new file)
  - Create a minimal build script that:
    1. Reads the `RUSTC` environment variable (set by Cargo) or falls back to
       `"rustc"`.
    2. Runs `rustc --version` and checks if the output contains `"nightly"`.
    3. If nightly, emits `println!("cargo:rustc-cfg=nightly")`.
    4. Always emits `println!("cargo:rerun-if-changed=build.rs")`.

- `chutoro-providers/dense/src/lib.rs`
  - Add at the crate root:

  ```rust
  #![cfg_attr(
      all(feature = "nightly_portable_simd", nightly),
      feature(portable_simd)
  )]
  ```

  This activates `portable_simd` only when both the Cargo feature is enabled
  AND the compiler is nightly.

Go/no-go:

- `cargo check -p chutoro-providers-dense --features nightly_portable_simd` on
  stable must succeed without errors or warnings.
- `cargo check -p chutoro-providers-dense --all-features` on stable must
  succeed without errors or warnings.
- Do not proceed if either check fails.

### Stage C: implement portable SIMD kernels

Add the `std::simd` kernel implementations in a new submodule.

Planned edits:

- `chutoro-providers/dense/src/simd/kernels/portable_simd.rs` (new file)
  - Gate the entire module with
    `#[cfg(all(feature = "nightly_portable_simd", nightly))]`.
  - Import `std::simd::prelude::*` and `std::simd::Simd`.
  - Implement:
    1. `euclidean_distance_portable_simd(left: &[f32], right: &[f32]) -> f32`
       — Safe pairwise Euclidean distance using `Simd<f32, 16>` for the main
       loop and `squared_l2_tail` for remainder elements, followed by
       `finalize_distance`.
    2. `euclidean_distance_query_points_portable_simd_entry`
       `(query: &[f32], points: &DensePointView<'_>, out: &mut [f32])`
       — Safe query-to-points batch kernel using `Simd<f32, 16>` across
       padded coordinate blocks, with `finalize_distance` per lane.

  The lane width of 16 matches `MAX_SIMD_LANES` and `DensePointView`'s padding.
  The compiler will lower `Simd<f32, 16>` to the best available hardware
  instruction set (AVX-512 if available, AVX2 with two half-width operations
  otherwise, etc.).

- `chutoro-providers/dense/src/simd/kernels.rs`
  - Add a conditional module declaration:

  ```rust
  #[cfg(all(feature = "nightly_portable_simd", nightly))]
  mod portable_simd;
  ```

  - Extend `select_backend_fn!` macro to include a `portable_simd` arm,
    gated by `#[cfg(all(feature = "nightly_portable_simd", nightly))]`,
    mapping `EuclideanBackend::PortableSimd` to the new kernel functions.
  - Add `portable_simd` entrypoint references in `select_euclidean_kernel`
    and `select_euclidean_query_points_kernel`.

### Stage D: extend dispatch model

Add the `PortableSimd` backend variant and wire it into dispatch selection.

Planned edits:

- `chutoro-providers/dense/src/simd/dispatch.rs`
  - Add `PortableSimd` variant to `EuclideanBackend`.
  - Add a `portable_simd` field to `CompiledSimdSupport` and
    `RuntimeSimdSupport`.
  - Update `CompiledSimdSupport::new` to accept 4 boolean arguments (add
    `portable_simd: bool`).
  - Update `RuntimeSimdSupport::new` to accept 4 boolean arguments (add
    `portable_simd: bool`).
  - Update `compiled_simd_support()` to report `true` for portable SIMD when
    `cfg!(all(feature = "nightly_portable_simd", nightly))`.
  - Update `runtime_simd_support()` to report `true` for portable SIMD
    unconditionally when the feature is compiled in (portable SIMD has no
    runtime detection requirement; the compiler handles target feature
    selection).
  - Update `choose_euclidean_backend` priority order to:
    AVX-512 > AVX2 > NEON > PortableSimd > Scalar.
    The portable SIMD backend sits below all platform-specific intrinsic
    backends because the intrinsic backends are hand-tuned and proven faster
    for their target; portable SIMD serves as a better-than-scalar fallback
    for platforms without dedicated intrinsic backends.
  - Update `backend_supported` to handle `PortableSimd`.

- `chutoro-providers/dense/src/simd/mod.rs`
  - Update `should_pack_query_points_for_backend` to treat `PortableSimd` as
    SIMD-capable (the SoA query-points path should be used).

### Stage E: add unit tests

Add comprehensive parameterized tests for the new backend.

Planned edits:

- `chutoro-providers/dense/src/simd/tests.rs`
  - Extend the `choose_euclidean_backend_prefers_best_enabled_supported_backend`
    rstest cases to cover `PortableSimd` scenarios:
    - When `nightly_portable_simd` is compiled and no platform-specific
      backend is available, portable SIMD is selected.
    - When AVX2 is available, AVX2 wins over portable SIMD.
    - When only portable SIMD and scalar are available, portable SIMD wins.
  - Add `should_pack_query_points_for_backend` case for `PortableSimd`.

- `chutoro-providers/dense/src/simd/tests/entrypoints.rs`
  - Add a `#[cfg(all(feature = "nightly_portable_simd", nightly))]` test
    section:
    - `portable_simd_pairwise_matches_scalar`: generate vectors at multiple
      lengths (including below, at, and above 16-lane boundary: 1, 7, 16, 17,
      35, 67, 128), compute distances with both the portable SIMD kernel and
      the scalar kernel, and assert agreement within `1.0e-6`.
    - `portable_simd_query_points_matches_scalar`: generate a small matrix
      and candidate set, compute with both portable SIMD and scalar
      query-points kernels, and assert agreement.
    - `portable_simd_canonicalizes_non_finite_to_nan`: test that `INFINITY`
      input produces `NaN` output.
    - Use `rstest` `#[case]` parameterization for vector lengths.

- `chutoro-providers/dense/src/simd/tests.rs`
  - Add non-finite canonicalization test case for the portable SIMD backend
    if appropriate.

Go/no-go:

- All new tests pass on nightly with `--features nightly_portable_simd`.
- All existing tests continue to pass on stable without the feature.
- `make lint` passes (which uses `--all-features` on stable; the `cfg_attr`
  gating from Stage B ensures this compiles cleanly).
- `make test` passes.

### Stage F: add nightly CI workflow

Create a GitHub Actions workflow for nightly validation.

Planned edits:

- `.github/workflows/nightly-portable-simd.yml` (new file)
  - Follow the pattern from `.github/workflows/nightly-kani.yml`:
    - Schedule: `cron: '0 3 * * *'` (daily at 03:00 UTC, offset from the Kani
      job at 02:00 UTC).
    - Manual trigger: `workflow_dispatch` with optional `force_run` flag.
    - Concurrency group: `nightly-portable-simd`, `cancel-in-progress: false`.
    - Job `nightly-portable-simd`:
      - `runs-on: ubuntu-latest`
      - `timeout-minutes: 30`
      - Steps:
        1. Checkout `main`.
        2. Install nightly Rust toolchain (override the pinned stable):
           `rustup toolchain install nightly && rustup override set nightly`.
        3. Check for recent commits on `main` in the last 24 hours (reuse
           the same gating logic as the Kani workflow, or use a simpler
           inline check).
        4. Run
           `cargo test -p chutoro-providers-dense --features nightly_portable_simd`
           with `RUSTFLAGS="-D warnings"`.
        5. Run
           `cargo clippy -p chutoro-providers-dense --all-targets`
           `--features nightly_portable_simd -- -D warnings`.

### Stage G: update documentation and close roadmap item

Record the design decisions and mark the roadmap item done.

Planned edits:

- `docs/chutoro-design.md`
  - Add an implementation update paragraph to §6.3 recording:
    - The `nightly_portable_simd` feature name and that it is non-default.
    - The dispatch priority: AVX-512 > AVX2 > NEON > PortableSimd > Scalar.
    - The tracking issues: `rust-lang/rust#86656` (portable_simd),
      `rust-lang/rust#127356` (bf16 wrappers), `rust-lang/rust#127213`
      (AVX512_FP16 intrinsics).
    - The build script nightly detection strategy.
    - That unit test coverage validates pairwise, batch, and non-finite
      parity with the scalar oracle.

- `docs/roadmap.md`
  - Mark item `2.2.4` done (`[x]`) only after all validation commands pass.

Validation commands:

```sh
set -o pipefail; make check-fmt 2>&1 | tee /tmp/2-2-4-make-check-fmt.log
set -o pipefail; make lint 2>&1 | tee /tmp/2-2-4-make-lint.log
set -o pipefail; make test 2>&1 | tee /tmp/2-2-4-make-test.log
```

Expected success signals:

- `make check-fmt` exits `0`.
- `make lint` exits `0` with no Clippy warnings (the `nightly_portable_simd`
  feature is enabled by `--all-features` but the `cfg_attr` gating prevents
  `#![feature(portable_simd)]` from appearing on stable).
- `make test` exits `0` and includes all existing tests in the summary (the
  nightly-only tests are conditionally compiled out on stable).

If the plain local `make test` run hits the known nextest stall, run
`CI=1 make test` instead, capture the log, and confirm the stall is unrelated
before marking work complete.

## Concrete steps

All commands assume a working directory of `/home/user/project`.

### Verify stable baseline (before any changes)

```sh
set -o pipefail; make check-fmt 2>&1 | tee /tmp/2-2-4-baseline-check-fmt.log
set -o pipefail; make lint 2>&1 | tee /tmp/2-2-4-baseline-lint.log
set -o pipefail; CI=1 make test 2>&1 | tee /tmp/2-2-4-baseline-test.log
```

Expected: all pass. This confirms the repository is healthy before changes.

### After Stage B (feature flag and build script)

```sh
set -o pipefail; cargo check -p chutoro-providers-dense --all-features 2>&1 \
  | tee /tmp/2-2-4-all-features-stable-check.log
set -o pipefail; cargo check -p chutoro-providers-dense --features nightly_portable_simd 2>&1 \
  | tee /tmp/2-2-4-nightly-feature-stable-check.log
```

Expected: both succeed on stable without errors. The `cfg_attr` gating ensures
`#![feature(portable_simd)]` is not emitted.

### After Stage E (tests complete)

```sh
set -o pipefail; make check-fmt 2>&1 | tee /tmp/2-2-4-stage-e-check-fmt.log
set -o pipefail; make lint 2>&1 | tee /tmp/2-2-4-stage-e-lint.log
set -o pipefail; CI=1 make test 2>&1 | tee /tmp/2-2-4-stage-e-test.log
```

Expected: all pass. Nightly-only tests are compiled out on stable.

### After Stage G (documentation and roadmap closure)

```sh
set -o pipefail; make check-fmt 2>&1 | tee /tmp/2-2-4-final-check-fmt.log
set -o pipefail; make lint 2>&1 | tee /tmp/2-2-4-final-lint.log
set -o pipefail; CI=1 make test 2>&1 | tee /tmp/2-2-4-final-test.log
set -o pipefail; make fmt 2>&1 | tee /tmp/2-2-4-final-fmt.log
set -o pipefail; make markdownlint 2>&1 | tee /tmp/2-2-4-final-markdownlint.log
set -o pipefail; make nixie 2>&1 | tee /tmp/2-2-4-final-nixie.log
```

Expected: all pass with exit code 0.

## Validation and acceptance

Quality criteria:

- Tests: `make test` passes; all existing tests continue to pass; new
  dispatch and entrypoint tests are included in the test count.
- Lint: `make lint` exits `0`; no Clippy or rustdoc warnings.
- Format: `make check-fmt` exits `0`.
- Markdown: `make markdownlint` exits `0`.
- Mermaid: `make nixie` exits `0`.
- Nightly build:
  `cargo test -p chutoro-providers-dense --features nightly_portable_simd`
  succeeds on a nightly Rust toolchain (validated in CI or locally if nightly
  is available).

Quality method:

- Run the validation commands listed in the concrete steps section.
- Verify that the nightly CI workflow file is syntactically valid YAML.
- Confirm that `docs/roadmap.md` item `2.2.4` is marked `[x]`.
- Confirm that `docs/chutoro-design.md` §6.3 contains the new implementation
  update paragraph.

## Idempotence and recovery

All stages are re-runnable. If a stage fails partway:

- Revert uncommitted changes with `git checkout -- .` and retry.
- The build script (`build.rs`) is idempotent: it always re-emits the same
  `cargo:rustc-cfg` based on the compiler version.
- The `OnceLock` dispatch cache is process-scoped; restarting tests clears it.

## Artifacts and notes

(To be filled during implementation.)

## Interfaces and dependencies

### New Cargo feature

In `chutoro-providers/dense/Cargo.toml`:

```toml
[features]
default = ["simd_avx2", "simd_avx512", "simd_neon"]
simd_avx2 = []
simd_avx512 = []
simd_neon = []
nightly_portable_simd = []
```

### Crate-level feature gate

In `chutoro-providers/dense/src/lib.rs`:

```rust
#![cfg_attr(
    all(feature = "nightly_portable_simd", nightly),
    feature(portable_simd)
)]
```

### Build script

In `chutoro-providers/dense/build.rs`:

```rust
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rustc-check-cfg=cfg(nightly)");

    if is_nightly_compiler() {
        println!("cargo:rustc-cfg=nightly");
    }
}

fn is_nightly_compiler() -> bool {
    let rustc = std::env::var_os("RUSTC")
        .unwrap_or_else(|| std::ffi::OsString::from("rustc"));
    std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .is_some_and(|version| version.contains("nightly"))
}
```

### Dispatch enum extension

In `chutoro-providers/dense/src/simd/dispatch.rs`:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EuclideanBackend {
    Scalar,
    Avx2,
    Avx512,
    Neon,
    PortableSimd,
}
```

### Dispatch support structs extension

```rust
pub(super) struct CompiledSimdSupport {
    avx2: bool,
    avx512: bool,
    neon: bool,
    portable_simd: bool,
}

pub(super) struct RuntimeSimdSupport {
    avx2: bool,
    avx512: bool,
    neon: bool,
    portable_simd: bool,
}
```

### Priority order

```rust
for backend in [
    EuclideanBackend::Avx512,
    EuclideanBackend::Avx2,
    EuclideanBackend::Neon,
    EuclideanBackend::PortableSimd,
    EuclideanBackend::Scalar,
] { ... }
```

### Kernel interface

In `chutoro-providers/dense/src/simd/kernels/portable_simd.rs`:

```rust
//! Euclidean distance kernels using `std::simd` (portable SIMD).
//!
//! This module is only compiled when the `nightly_portable_simd` Cargo
//! feature is enabled and the compiler is nightly.

use std::simd::prelude::*;

use super::super::DensePointView;
use super::{finalize_distance, squared_l2_tail};

/// Pairwise Euclidean distance using portable SIMD with 16-lane vectors.
pub(super) fn euclidean_distance_portable_simd(
    left: &[f32],
    right: &[f32],
) -> f32 { ... }

/// Query-to-points batch Euclidean distance using portable SIMD.
pub(super) fn euclidean_distance_query_points_portable_simd_entry(
    query: &[f32],
    points: &DensePointView<'_>,
    out: &mut [f32],
) { ... }
```

### Nightly CI workflow

In `.github/workflows/nightly-portable-simd.yml`:

```yaml
name: Nightly portable SIMD

'on':
  schedule:
    - cron: '0 3 * * *'
  workflow_dispatch:
    inputs:
      force_run:
        description: >-
          Force run even without last-24-hour commits (UTC).
        required: false
        default: true
        type: boolean

concurrency:
  group: nightly-portable-simd
  cancel-in-progress: false

jobs:
  nightly-portable-simd:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    permissions:
      contents: read
    env:
      CARGO_TERM_COLOR: always
      RUSTFLAGS: '-D warnings'
    steps:
      - uses: actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8
        with:
          ref: main
      - name: Install nightly Rust
        run: |
          rustup toolchain install nightly --profile minimal \
            --component clippy rustfmt
          rustup override set nightly
      - name: Check for recent commits
        id: gate
        run: |
          LAST_COMMIT=$(git log -1 --format=%ct)
          NOW=$(date +%s)
          AGE=$(( NOW - LAST_COMMIT ))
          FORCE="${{ github.event_name == 'workflow_dispatch'
            && github.event.inputs.force_run || 'false' }}"
          if [ "$AGE" -lt 86400 ] || [ "$FORCE" = "true" ]; then
            echo "should_run=true" >> "$GITHUB_OUTPUT"
          else
            echo "should_run=false" >> "$GITHUB_OUTPUT"
          fi
      - name: Test portable SIMD backend
        if: steps.gate.outputs.should_run == 'true'
        run: >-
          cargo test -p chutoro-providers-dense
          --features nightly_portable_simd
      - name: Clippy portable SIMD backend
        if: steps.gate.outputs.should_run == 'true'
        run: >-
          cargo clippy -p chutoro-providers-dense
          --all-targets --features nightly_portable_simd
          -- -D warnings
```

## Proposed implementation order

1. Prototype and confirm the `build.rs` nightly detection strategy (Stage A).
2. Add the `nightly_portable_simd` feature, `build.rs`, and `cfg_attr`
   gating (Stage B).
3. Implement the portable SIMD pairwise and query-points kernels (Stage C).
4. Extend the dispatch model with `PortableSimd` (Stage D).
5. Add parameterized unit tests for dispatch selection, pairwise correctness,
   batch correctness, and non-finite handling (Stage E).
6. Add the nightly CI workflow (Stage F).
7. Update `docs/chutoro-design.md` and mark `docs/roadmap.md` item `2.2.4`
   done (Stage G).

## Historical note

This ExecPlan began as the required draft for roadmap task `2.2.4`. The user
approved implementation, the work shipped, and the status is now `COMPLETED`.
