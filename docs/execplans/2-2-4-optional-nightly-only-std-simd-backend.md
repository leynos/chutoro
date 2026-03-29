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

A nightly Continuous Integration (CI) job validates that the nightly feature
compiles and produces correct results, catching `portable_simd` API breakage
before it reaches contributors.

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
  nightly Rust toolchain, reporting results independently of the stable CI;
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

- [x] 1. Deliver roadmap item `2.2.4` end to end while preserving stable
  builds.

- [x] 1.1. Resolve the `--all-features` coexistence strategy (no code changes
  beyond prototyping). The central technical challenge is that `make lint` and
  `make test` pass `--all-features`, which will enable `nightly_portable_simd`.
  If `#![feature(portable_simd)]` is emitted on stable, the build fails.

- [x] 1.1.1. Evaluate Option 1: `cfg_attr` with nightly detection.
  Use a build script or `rustc_attrs` feature detection to emit
  `cargo:rustc-cfg=nightly` on nightly compilers. Gate
  `#![cfg_attr(feature = "nightly_portable_simd", feature(portable_simd))]`
  behind the `nightly` cfg so the `cfg_attr` expands to nothing on stable even
  when `--all-features` enables the Cargo feature.

- [x] 1.1.2. Evaluate Option 2: exclude from `--all-features` via `Makefile`.
  Adjust the `Makefile` targets to exclude `nightly_portable_simd` by using
  explicit feature lists instead of `--all-features`, or accept that
  `--all-features` on stable will fail and add a separate nightly lint/test
  target.

- [x] 1.1.3. Prefer Option 1 because it preserves the existing
  `--all-features` workflow and requires no Makefile changes. A `build.rs` that
  emits `cargo:rustc-cfg=nightly` when the compiler reports a nightly version
  string is a well-established pattern.

- [x] 1.1.4. Confirm that `rustc --version` output on nightly includes the
  string `"nightly"` and on stable does not.

- [x] 1.1.5. Confirm that `cargo:rustc-cfg=nightly` emitted by `build.rs` makes
  `#[cfg(nightly)]` available to the crate.

- [x] 1.1.6. Prototype the `cfg_attr` gating in a scratch file and verify it
  compiles cleanly on stable with the feature enabled.

- [x] 1.1.7. Proceed only if Option 1 is confirmed feasible; otherwise stop and
  escalate if neither option works within tolerances.

- [x] 1.2. Add the feature flag, build script, and crate-level gating.
  Add the Cargo feature, build script, and crate-level `#![feature(...)]`
  gating.

- [x] 1.2.1. Update
      [Cargo.toml](chutoro-providers/dense/Cargo.toml)
  to add `nightly_portable_simd = []` to the `[features]` table without adding
  it to `default`.

- [x] 1.2.2. Create
      [build.rs](chutoro-providers/dense/build.rs)
  so it reads `RUSTC` or falls back to `"rustc"`, runs `rustc --version`,
  checks for `"nightly"`, emits `println!("cargo:rustc-cfg=nightly")` when
  appropriate, and always emits `println!("cargo:rerun-if-changed=build.rs")`.

- [x] 1.2.3. Update
      [lib.rs](chutoro-providers/dense/src/lib.rs)
  to add the crate-root gating:

  ```rust
  #![cfg_attr(
      all(feature = "nightly_portable_simd", nightly),
      feature(portable_simd)
  )]
  ```

  This activates `portable_simd` only when both the Cargo feature is enabled
  and the compiler is nightly.

- [x] 1.2.4. Require `cargo check -p chutoro-providers-dense --features
  nightly_portable_simd` on stable and `cargo check -p chutoro-providers-dense
  --all-features
  ` on stable to succeed without errors or warnings before continuing.

- [x] 1.3. Implement the portable SIMD kernels.
  Add the `std::simd` kernel implementations in a new submodule.

- [x] 1.3.1. Create
  [portable_simd.rs](chutoro-providers/dense/src/simd/kernels/portable_simd.rs)
  behind `#[cfg(all(feature = "nightly_portable_simd", nightly))]`, import
  `std::simd::prelude::*` and `std::simd::Simd`, and implement the pairwise
  kernel `euclidean_distance_portable_simd` plus the query-to-points entrypoint
  `euclidean_distance_query_points_portable_simd_entry`.

- [x] 1.3.2. Keep the portable SIMD lane width at 16 so it matches
  `MAX_SIMD_LANES` and `DensePointView` padding. Rely on the compiler to lower
  `Simd<f32, 16>` to the best available hardware instruction set.

- [x] 1.3.3. Update
  [kernels.rs](chutoro-providers/dense/src/simd/kernels.rs) to add the
  conditional `portable_simd` module declaration, extend `select_backend_fn!`
  with a `portable_simd` arm, and add entrypoint references in
  `select_euclidean_kernel` and `select_euclidean_query_points_kernel`.

- [x] 1.4. Extend the dispatch model.
  Add the `PortableSimd` backend variant and wire it into dispatch selection.

- [x] 1.4.1. Update
  [dispatch.rs](chutoro-providers/dense/src/simd/dispatch.rs) to add
  `PortableSimd` to `EuclideanBackend`, add `portable_simd` fields to
  `CompiledSimdSupport` and `RuntimeSimdSupport`, update their constructors to
  accept four boolean arguments, report compiled support through
  `cfg!(all(feature = "nightly_portable_simd", nightly))`, report runtime
  support unconditionally when compiled in, update backend priority to AVX-512
  > AVX2 > NEON > PortableSimd > Scalar, and extend `backend_supported`.

- [x] 1.4.2. Update
  [mod.rs](chutoro-providers/dense/src/simd/mod.rs) so
  `should_pack_query_points_for_backend` treats `PortableSimd` as SIMD-capable
  and uses the structure-of-arrays query-points path.

- [x] 1.5. Add comprehensive parameterized unit tests for the new backend.

- [x] 1.5.1. Extend
  [tests.rs](chutoro-providers/dense/src/simd/tests.rs) so
  `choose_euclidean_backend_prefers_best_enabled_supported_backend` covers
  `PortableSimd`, including cases where portable SIMD is selected when no
  platform-specific backend is available, AVX2 wins when present, and portable
  SIMD wins over scalar when both are available.

- [x] 1.5.2. Add a `should_pack_query_points_for_backend` case for
  `PortableSimd` in [tests.rs](chutoro-providers/dense/src/simd/tests.rs).

- [x] 1.5.3. Extend
  [entrypoints.rs](chutoro-providers/dense/src/simd/tests/entrypoints.rs) with
  a `#[cfg(all(feature = "nightly_portable_simd", nightly))]` section covering
  `portable_simd_pairwise_matches_scalar`,
  `portable_simd_query_points_matches_scalar`,
  `portable_simd_canonicalizes_non_finite_to_nan`, and `rstest`
  parameterization for vector lengths 1, 7, 16, 17, 35, 67, and 128.

- [x] 1.5.4. Add a non-finite canonicalization test case for the portable SIMD
  backend in [tests.rs](chutoro-providers/dense/src/simd/tests.rs) if needed.

- [x] 1.5.5. Require all new tests to pass on nightly with
  `--features nightly_portable_simd`, all existing tests to pass on stable
  without the feature, `make lint` to pass under stable `--all-features`, and
  `make test` to pass before proceeding.

- [x] 1.6. Add the nightly CI workflow.
  Create a GitHub Actions workflow for nightly validation.

- [x] 1.6.1. Follow the pattern from `.github/workflows/nightly-kani.yml`
  while creating
  [nightly-portable-simd.yml](.github/workflows/nightly-portable-simd.yml) with
  schedule `cron: '0 3 * * *'`, a `workflow_dispatch` `force_run` flag, and
  concurrency group `nightly-portable-simd` with `cancel-in-progress: false`.

- [x] 1.6.2. Configure the `nightly-portable-simd` job to run on
  `ubuntu-latest` with `timeout-minutes: 30`.

- [x] 1.6.3. Sequence the workflow steps as checkout, recent-commit gate by
  fetching `origin/main` and comparing `git log -1 --format=%ct origin/main`
  against current UTC time with `force_run=true` bypass support, nightly
  toolchain install via
  `rustup toolchain install nightly --profile minimal --component clippy rustfmt`,
   and the following commands:

  ```sh
  RUSTFLAGS="-D warnings" cargo +nightly test -p chutoro-providers-dense --features nightly_portable_simd
  cargo +nightly clippy -p chutoro-providers-dense --all-targets --features nightly_portable_simd -- -D warnings
  ```

- [x] 1.7. Update documentation and close the roadmap item.
  Record the design decisions and mark the roadmap item done.

- [x] 1.7.1. Update
      [chutoro-design.md](docs/chutoro-design.md)
  §6.3 to record the non-default `nightly_portable_simd` feature name, dispatch
  priority `AVX-512 > AVX2 > NEON > PortableSimd > Scalar`, tracking issues:
  [^1], [^2], [^3], the build-script nightly detection strategy, and unit-test
  coverage for pairwise, batch, and non-finite parity with the scalar oracle.

- [x] 1.7.2. Mark [roadmap.md](docs/roadmap.md) item
  `2.2.4` done only after all validation commands pass.

- [x] 1.7.3. Run the validation commands:

  ```sh
  set -o pipefail; make check-fmt 2>&1 | tee /tmp/2-2-4-make-check-fmt.log
  set -o pipefail; make lint 2>&1 | tee /tmp/2-2-4-make-lint.log
  set -o pipefail; make test 2>&1 | tee /tmp/2-2-4-make-test.log
  ```

- [x] 1.7.4. Expect `make check-fmt` to exit `0`, `make lint` to exit `0`
  without Clippy warnings even though `nightly_portable_simd` is enabled by
  `--all-features`, and `make test` to exit `0` while keeping nightly-only
  tests conditionally compiled out on stable.

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
  `RUSTFLAGS="-D warnings" cargo +nightly test -p chutoro-providers-dense`
  `--features nightly_portable_simd` and `cargo +nightly clippy -p`
  `chutoro-providers-dense --all-targets --features nightly_portable_simd`
  `-- -D warnings` succeed on a nightly Rust toolchain (validated in CI or
  locally if nightly is available).

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
      - name: Install nightly Rust
        if: steps.gate.outputs.should_run == 'true'
        run: rustup toolchain install nightly --profile minimal --component clippy rustfmt
      - name: Check for recent commits
        id: gate
        env:
          FORCE_RUN: ${{ github.event_name == 'workflow_dispatch' && github.event.inputs.force_run || 'false' }}
        run: |
          if [ "${FORCE_RUN}" = "true" ]; then
            echo "should_run=true" >> "${GITHUB_OUTPUT}"
            exit 0
          fi

          git fetch --no-tags --depth=1 origin main:refs/remotes/origin/main

          now_epoch="$(date -u +%s)"
          commit_epoch="$(git log -1 --format=%ct origin/main)"
          if [ $((now_epoch - commit_epoch)) -le 86400 ]; then
            echo "should_run=true" >> "${GITHUB_OUTPUT}"
          else
            echo "should_run=false" >> "${GITHUB_OUTPUT}"
          fi
      - name: Test portable SIMD backend
        if: steps.gate.outputs.should_run == 'true'
        run: cargo +nightly test -p chutoro-providers-dense --features nightly_portable_simd
      - name: Clippy portable SIMD backend
        if: steps.gate.outputs.should_run == 'true'
        run: cargo +nightly clippy -p chutoro-providers-dense --all-targets --features nightly_portable_simd -- -D warnings
```

## Proposed implementation order

The canonical completed execution order is captured in the checked Phase/Step/
Task hierarchy under [Plan of work](#plan-of-work), so this duplicate summary
is no longer maintained separately.

## Historical note

This ExecPlan began as the required draft for roadmap task `2.2.4`. The user
approved implementation, the work shipped, and the status is now `COMPLETED`.

[^1]: `rust-lang/rust#86656` (portable_simd)
[^2]: `rust-lang/rust#127356` (bf16 wrappers)
[^3]: `rust-lang/rust#127213` (AVX512_FP16 intrinsics)
