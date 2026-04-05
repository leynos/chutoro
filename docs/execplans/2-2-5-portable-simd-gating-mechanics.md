# Execution plan (ExecPlan): roadmap 2.2.5 portable-SIMD gating mechanics

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & discoveries`, `Decision log`, and
`Outcomes & retrospective` must be kept up to date as work proceeds.

Status: COMPLETED

## Purpose / big picture

Complete roadmap item `2.2.5` by proving and hardening the coexistence boundary
between the stable dense-provider single-instruction multiple-data (SIMD)
backends and the optional nightly-only portable-SIMD backend. The goal is not
to re-implement the portable-SIMD kernel work from `2.2.4`; that work already
exists. The goal here is to make the gating contract explicit, test it
directly, add focused Continuous Integration (CI) coverage for the
stable-disabled and nightly-enabled paths, and then record the final design
contract in the design document before marking the roadmap item done.

Success is observable when:

- `chutoro-providers-dense` keeps the existing non-default Cargo feature
  `nightly_portable_simd`, and stable builds remain clean when that feature is
  not enabled;
- stable warning-deny builds also remain clean when workspace-wide
  `--all-features` gates are used, because nightly-only code is additionally
  guarded by `cfg(nightly)`;
- all crate-level and module-level portable-SIMD surfaces are consistently
  gated with the same predicate:
  `all(feature = "nightly_portable_simd", nightly)`;
- unit tests, using `rstest` where repetition exists, cover happy paths,
  unhappy paths, and edge cases for compile-time support masks, runtime support
  masks, and dispatch fallback when the nightly backend is unavailable;
- CI explicitly checks a stable build path with the nightly feature disabled
  and a nightly build path with the feature enabled;
- `docs/chutoro-design.md` records the final coexistence contract, including
  the existing underscore feature name, the `build.rs`-emitted `cfg(nightly)`
  mechanism, and the CI coverage;
- `docs/roadmap.md` marks item `2.2.5` done only after implementation and all
  validation commands succeed.

Because the repository already contains much of the raw plumbing from `2.2.4`,
this plan intentionally treats `2.2.5` as a verification and hardening item. If
the initial audit shows that only tests, CI, and documentation are missing,
keep the implementation that small.

## Constraints

- Keep Rust source files under 400 lines. Split tests or helpers rather than
  extending existing dense SIMD files past that limit.
- Preserve the public `DenseMatrixProvider` and `DataSource` interfaces. This
  item must not widen the error surface or change public signatures.
- Do not add new crate dependencies.
- Preserve the existing dense-provider feature names. Use the already-shipped
  `nightly_portable_simd` feature rather than renaming it to match roadmap
  example punctuation.
- Preserve the existing `build.rs` contract that registers `cfg(nightly)` via
  `cargo:rustc-check-cfg=cfg(nightly)` and only emits `cargo:rustc-cfg=nightly`
  when the active compiler reports itself as nightly.
- Preserve the current dispatch order:
  `Avx512 > Avx2 > Neon > PortableSimd > Scalar`.
- Preserve the existing non-finite policy: every backend canonicalizes
  non-finite Euclidean outputs to `f32::NAN`.
- Preserve the existing `DensePointView<'a>` layout contract introduced by
  `2.2.2`.
- Keep the nightly backend isolated behind
  `all(feature = "nightly_portable_simd", nightly)` at every crate, module,
  entrypoint, and test boundary. Avoid one-off predicates unless the audit
  proves a narrower guard is required.
- Use `rstest` for repeated coverage in unit tests.
- Follow guidance from:
  - `docs/chutoro-design.md` (especially §6.3)
  - `docs/property-testing-design.md`
  - `docs/complexity-antipatterns-and-refactoring-strategies.md`
  - `docs/rust-testing-with-rstest-fixtures.md`
  - `docs/rust-doctest-dry-guide.md`
- Use en-GB-oxendict spelling in comments and documentation.

## Tolerances (exception triggers)

- Scope: if finishing `2.2.5` requires edits in more than 12 files or more
  than 700 net lines, stop and escalate. This item should be smaller than
  `2.2.4`.
- Interface: if the coexistence fix appears to require feature forwarding or
  API changes outside `chutoro-providers/dense` and CI/docs files, stop and
  escalate with options.
- Build-system churn: if satisfying the CI requirement would require changing
  `Makefile` semantics for the whole workspace instead of adding focused CI
  commands or jobs, stop and escalate.
- Toolchain ambiguity: if the stable and nightly coexistence contract cannot be
  expressed using the existing `cfg(nightly)` mechanism, stop and document why
  the `build.rs` approach is insufficient before proposing alternatives.
- Validation: if `make lint` or `make test` still fails after 3 repair
  attempts, stop and escalate with the captured logs.
- Duplication: if the audit shows roadmap item `2.2.5` is already fully
  satisfied except for the unchecked roadmap box, stop and escalate before
  making no-op code churn.

## Risks

- Risk: `2.2.4` already implemented most of the gating mechanics, so `2.2.5`
  could accidentally duplicate code changes and create churn without improving
  safety. Severity: high. Likelihood: high. Mitigation: begin with a focused
  audit and only change code where the coexistence boundary is still implicit
  or untested.

- Risk: stable workspace gates currently use `--all-features`, which can mask
  the more precise roadmap requirement of checking the stable path with the
  nightly feature disabled. Severity: high. Likelihood: medium. Mitigation:
  keep the repository-wide gates, but add an explicit stable dense-provider CI
  command that does not enable `nightly_portable_simd`.

- Risk: tests may only cover pure selector logic with manually constructed
  masks, leaving the real compiled/runtime support helpers insufficiently
  exercised. Severity: medium. Likelihood: medium. Mitigation: add focused unit
  tests around `compiled_simd_support()` and `runtime_simd_support()`, with
  assertions that vary according to the active toolchain and feature set.

- Risk: adding CI coverage in the wrong place could make nightly instability
  block normal pull requests. Severity: medium. Likelihood: medium. Mitigation:
  keep stable verification in the main CI workflow and keep the nightly-enabled
  verification in the dedicated nightly workflow unless a lighter, non-blocking
  arrangement is explicitly preferred.

- Risk: scattered `cfg` expressions can turn the dense SIMD files into a
  bumpy-road maintenance problem. Severity: medium. Likelihood: medium.
  Mitigation: centralize on one predicate and add tests that prove the chosen
  behaviour instead of layering ad hoc guards.

## Progress

- [x] (2026-03-29 00:00Z) Audited `docs/roadmap.md`, `docs/chutoro-design.md`,
  the existing `2.2.3` and `2.2.4` ExecPlans, the dense SIMD crate, and the
  current CI workflows.
- [x] (2026-03-29 00:10Z) Confirmed that the repository already has:
  `nightly_portable_simd` in `Cargo.toml`, `cfg_attr` on the crate-level
  `#![feature(portable_simd)]`, a `build.rs` that emits `cfg(nightly)`,
  nightly-only module guards in `simd/kernels.rs`, and a dedicated nightly
  workflow.
- [x] (2026-03-30 00:20Z) Added focused support-mask tests for the actual
  `compiled_simd_support()`, `runtime_simd_support()`, and cached backend
  selection behaviour in `chutoro-providers/dense/src/simd/tests/`.
- [x] (2026-03-30 00:24Z) Tightened CI with an explicit stable dense-provider
  gating step in `.github/workflows/ci.yml` while retaining the dedicated
  nightly workflow for `nightly_portable_simd`.
- [x] (2026-03-30 00:27Z) Updated `docs/chutoro-design.md` with the final
  coexistence contract and marked roadmap item `2.2.5` done in
  `docs/roadmap.md`.
- [x] (2026-03-30 00:41Z) Ran formatting, lint, test, and documentation gates
  successfully and captured logs under `/tmp/2-2-5-impl-*`.

## Surprises & discoveries

- Discovery: roadmap item `2.2.5` is not greenfield work. The dense crate
  already contains the core mechanisms that the roadmap bullet list asks for:
  the feature exists, the crate-level `cfg_attr` exists, nightly modules are
  feature-guarded, and a nightly workflow already exists.

- Discovery: the main missing piece appears to be explicit proof, not raw
  plumbing. The stable CI workflow currently relies on broad workspace gates
  (`make lint`, `make test`), while the nightly-enabled path lives in
  `.github/workflows/nightly-portable-simd.yml`. The roadmap item still wants a
  crisp stable-disabled versus nightly-enabled verification story.

- Discovery: the build-script `cfg(nightly)` contract is already important to
  stable `--all-features` behaviour, so `2.2.5` should document and test that
  decision rather than revisiting it.

- Discovery: the selector-order tests already existed, so the missing coverage
  was not synthetic mask permutations but direct assertions that the real
  compile-time and runtime helpers report the expected support on the active
  host and toolchain.

## Decision log

- Decision: keep the existing feature name `nightly_portable_simd`. Rationale:
  the repository already ships this name in Cargo metadata, code, and
  documentation; renaming it now would create churn without improving the
  coexistence guarantee. Date/Author: 2026-03-29 / Codex.

- Decision: treat `2.2.5` as a verification and hardening follow-up to
  `2.2.4`, not as a second backend implementation task. Rationale: the audit
  shows the backend and most gating mechanics already exist. Date/Author:
  2026-03-29 / Codex.

- Decision: satisfy the roadmap CI requirement by combining a focused
  stable-disabled dense-provider check in the main CI workflow, the existing
  repository-wide stable gates, and the dedicated nightly-enabled workflow.
  Rationale: this proves both coexistence paths without making nightly
  instability block normal pull requests. Date/Author: 2026-03-29 / Codex.

## Outcomes & retrospective

- Shipped changes:
  - Added `chutoro-providers/dense/src/simd/tests/support_masks.rs` to assert
    the real compile-time support mask, runtime host detection mask, and
    cached backend choice.
  - Added an explicit stable dense-provider gating step to
    `.github/workflows/ci.yml` using `--no-default-features` plus the stable
    SIMD feature set.
  - Updated `docs/chutoro-design.md` and `docs/roadmap.md` to record the final
    coexistence contract and mark roadmap item `2.2.5` complete.
- Validation transcript summary:
  - `set -o pipefail; make fmt 2>&1 | tee /tmp/2-2-5-impl-make-fmt.log`
  - `set -o pipefail; make markdownlint 2>&1 | tee /tmp/2-2-5-impl-make-markdownlint.log`
  - `set -o pipefail; make nixie 2>&1 | tee /tmp/2-2-5-impl-make-nixie.log`
  - `set -o pipefail; make check-fmt 2>&1 | tee /tmp/2-2-5-impl-make-check-fmt.log`
  - `set -o pipefail; make lint 2>&1 | tee /tmp/2-2-5-impl-make-lint.log`
  - `set -o pipefail; CI=1 make test 2>&1 | tee /tmp/2-2-5-impl-make-test.log`
  - Final outcome: all gates passed; `CI=1 make test` finished with
    `Summary [462.408s] 915 tests run: 915 passed, 1 skipped`.
- Follow-up work deferred:
  - `2.2.6` remains the parity-suite item for cross-backend numeric coverage.
  - `2.2.7` remains the bounded Kani work for selector and tail-padding
    invariants.
- Implementation scope:
  - The audit was correct: no additional dense-crate gating logic was needed.
    `2.2.5` closed with tests, CI hardening, and documentation only.

## Context and orientation

The implementation surface for this roadmap item is intentionally narrow. Begin
in these files:

- `chutoro-providers/dense/Cargo.toml`
- `chutoro-providers/dense/build.rs`
- `chutoro-providers/dense/src/lib.rs`
- `chutoro-providers/dense/src/simd/dispatch.rs`
- `chutoro-providers/dense/src/simd/kernels.rs`
- `chutoro-providers/dense/src/simd/kernels/portable_simd.rs`
- `chutoro-providers/dense/src/simd/tests.rs`
- `chutoro-providers/dense/src/simd/tests/entrypoints.rs`
- `.github/workflows/ci.yml`
- `.github/workflows/nightly-portable-simd.yml`
- `docs/chutoro-design.md`
- `docs/roadmap.md`

The current audited state is:

1. `chutoro-providers/dense/Cargo.toml` already declares a non-default
   `nightly_portable_simd` feature.
2. `chutoro-providers/dense/src/lib.rs` already uses
   `#![cfg_attr(all(feature = "nightly_portable_simd", nightly), feature(portable_simd))]`.
3. `chutoro-providers/dense/build.rs` already emits both
   `cargo:rustc-check-cfg=cfg(nightly)` and, when appropriate,
   `cargo:rustc-cfg=nightly`.
4. `chutoro-providers/dense/src/simd/kernels.rs` already gates the
   `portable_simd` module and entrypoints behind
   `all(feature = "nightly_portable_simd", nightly)`.
5. `.github/workflows/nightly-portable-simd.yml` already runs nightly test and
   Clippy commands with `nightly_portable_simd` enabled.
6. `docs/roadmap.md` still shows `2.2.5` as incomplete.

That means the implementer should start by proving what is already true, then
only fill the real gaps.

## Implementation stages

## Stage 1: Audit and add the missing tests first

Before changing CI or code structure, tighten the unit-test evidence in
`chutoro-providers/dense/src/simd/tests.rs` and, if needed,
`chutoro-providers/dense/src/simd/tests/entrypoints.rs`.

The tests should answer these concrete questions:

1. When the nightly feature is unavailable at compile time, does dispatch fall
   back to the correct non-portable backend or scalar fallback?
2. When the nightly feature is compiled and runtime support is treated as
   available, does `choose_euclidean_backend(...)` choose `PortableSimd` only
   when no higher-priority backend is both compiled and available?
3. On stable builds, do the real `compiled_simd_support()` and
   `runtime_simd_support()` helpers report the portable backend as unavailable,
   even if workspace-wide `--all-features` is used?
4. On nightly builds with `nightly_portable_simd` enabled, do those helpers
   report the portable backend as available?

Prefer parameterized `rstest` coverage over hand-written repetition. The happy
paths are the cases where the portable backend is correctly selected or made
available. The unhappy paths are the cases where the feature is absent, the
toolchain is stable, or runtime availability is false, and the code must select
a non-portable fallback instead. Include at least one edge case where a
higher-priority backend suppresses `PortableSimd`.

If helper visibility makes real-helper assertions awkward, add the smallest
possible test-only accessor rather than broadening production visibility.

## Stage 2: Normalize any remaining gating inconsistencies

After the tests are in place, audit the dense crate for any portable-SIMD
surface that is not protected by the canonical predicate
`all(feature = "nightly_portable_simd", nightly)`.

Expected places to confirm:

1. The crate root in `chutoro-providers/dense/src/lib.rs`.
2. The portable-SIMD module declaration and entrypoints in
   `chutoro-providers/dense/src/simd/kernels.rs`.
3. Any entrypoint parity tests in
   `chutoro-providers/dense/src/simd/tests/entrypoints.rs`.
4. Any helper logic in `dispatch.rs` that computes compiled/runtime support
   masks.

If the audit finds no code gap, do not invent one. Record that the code was
already correct and move on to CI and documentation. If a gap is found, fix it
with the narrowest possible edit and keep all nightly-only references behind
the same predicate.

## Stage 3: Add explicit coexistence CI coverage

Add explicit CI checks that prove the two roadmap states:

1. Stable build with the nightly feature disabled.
2. Nightly build with the nightly feature enabled.

For the stable-disabled path, add a focused dense-provider job or step in
`.github/workflows/ci.yml`. Keep it limited to the dense crate so the intent is
obvious. The exact command may be adjusted if the audit shows a better
equivalent, but the preferred shape is:

```bash
set -o pipefail
cargo test -p chutoro-providers-dense --no-default-features \
  --features simd_avx2,simd_avx512,simd_neon 2>&1 \
  | tee /tmp/2-2-5-dense-stable-disabled-test.log
```

Pair it with a dense-only Clippy command of the same shape:

```bash
set -o pipefail
cargo clippy -p chutoro-providers-dense --all-targets --no-default-features \
  --features simd_avx2,simd_avx512,simd_neon -- -D warnings 2>&1 \
  | tee /tmp/2-2-5-dense-stable-disabled-clippy.log
```

This does not replace the workspace gates. It supplements them by proving that
the stable path works when `nightly_portable_simd` is truly disabled.

For the nightly-enabled path, keep using the dedicated
`.github/workflows/nightly-portable-simd.yml` workflow unless the audit shows a
clear reason to move or duplicate it. Confirm that it exercises the enabled
feature with both test and Clippy coverage. The current command shape is
already close to the target:

```bash
set -o pipefail
cargo +nightly test -p chutoro-providers-dense --features nightly_portable_simd 2>&1 \
  | tee /tmp/2-2-5-dense-nightly-enabled-test.log
```

```bash
set -o pipefail
cargo +nightly clippy -p chutoro-providers-dense --all-targets \
  --features nightly_portable_simd -- -D warnings 2>&1 \
  | tee /tmp/2-2-5-dense-nightly-enabled-clippy.log
```

If the nightly workflow needs wording or step-name changes so its purpose is
obviously about coexistence gating rather than just backend existence, make
that documentation improvement at the same time.

## Stage 4: Update the design record and roadmap

Update `docs/chutoro-design.md` §6.3 so it explicitly records the final
coexistence contract:

1. The dense crate keeps the non-default feature name
   `nightly_portable_simd`.
2. The crate root uses
   `cfg_attr(all(feature = "nightly_portable_simd", nightly), feature(portable_simd))`.
3. `build.rs` registers `cfg(nightly)` via `cargo:rustc-check-cfg=cfg(nightly)`
   and only emits `cargo:rustc-cfg=nightly` when the active compiler is nightly.
4. Portable-SIMD modules, entrypoints, and tests are isolated behind
   `all(feature = "nightly_portable_simd", nightly)`.
5. Stable CI verifies the dense crate with the nightly feature disabled, and
   the dedicated nightly workflow verifies the feature-enabled path.

Only after all validation commands pass should `docs/roadmap.md` mark `2.2.5`
done.

## Validation

Run the repository gates exactly as required by the project instructions, and
capture output with `tee`:

```bash
set -o pipefail; make fmt 2>&1 | tee /tmp/2-2-5-make-fmt.log
```

```bash
set -o pipefail; make markdownlint 2>&1 | tee /tmp/2-2-5-make-markdownlint.log
```

```bash
set -o pipefail; make nixie 2>&1 | tee /tmp/2-2-5-make-nixie.log
```

```bash
set -o pipefail; make check-fmt 2>&1 | tee /tmp/2-2-5-make-check-fmt.log
```

```bash
set -o pipefail; make lint 2>&1 | tee /tmp/2-2-5-make-lint.log
```

```bash
set -o pipefail; CI=1 make test 2>&1 | tee /tmp/2-2-5-make-test.log
```

Run the focused coexistence checks as well:

```bash
set -o pipefail; cargo test -p chutoro-providers-dense --no-default-features \
  --features simd_avx2,simd_avx512,simd_neon 2>&1 \
  | tee /tmp/2-2-5-dense-stable-disabled-test.log
```

```bash
set -o pipefail; cargo clippy -p chutoro-providers-dense --all-targets \
  --no-default-features --features simd_avx2,simd_avx512,simd_neon \
  -- -D warnings 2>&1 | tee /tmp/2-2-5-dense-stable-disabled-clippy.log
```

```bash
set -o pipefail; cargo +nightly test -p chutoro-providers-dense \
  --features nightly_portable_simd 2>&1 \
  | tee /tmp/2-2-5-dense-nightly-enabled-test.log
```

```bash
set -o pipefail; cargo +nightly clippy -p chutoro-providers-dense --all-targets \
  --features nightly_portable_simd -- -D warnings 2>&1 \
  | tee /tmp/2-2-5-dense-nightly-enabled-clippy.log
```

Success criteria:

1. All tests and lint commands exit with code `0`.
2. Stable dense-provider checks do not enable `nightly_portable_simd`.
3. Nightly dense-provider checks do enable `nightly_portable_simd`.
4. The design doc and roadmap reflect the shipped coexistence contract.

## Evidence to record in the completed plan

When this ExecPlan moves from `DRAFT` to `COMPLETED`, append concise evidence:

- the exact files changed;
- the names of any new or expanded `rstest` cases;
- the stable-disabled and nightly-enabled CI commands that passed;
- the final `make check-fmt`, `make lint`, and `make test` results;
- the design-doc paragraph or bullets added to §6.3;
- the roadmap checkbox update for `2.2.5`.
