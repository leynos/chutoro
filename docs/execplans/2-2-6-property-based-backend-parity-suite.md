# Execution plan (ExecPlan): roadmap 2.2.6 property-based backend parity suite

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & discoveries`, `Decision log`, and
`Outcomes & retrospective` must be kept up to date as work proceeds.

Status: COMPLETE

## Purpose / big picture

Complete roadmap item `2.2.6` by adding a property-based backend parity suite
for the dense-provider single-instruction multiple-data (SIMD) Euclidean
distance kernels. The suite exercises every backend that is both compiled into
the current binary and supported on the current host (scalar, advanced vector
extensions 2 (AVX2), AVX-512, ARM Advanced SIMD (Neon), and the optional
nightly portable-SIMD backend) and asserts that each one agrees with a single
scalar oracle reducer within a declared epsilon, under a single shared
`DistanceSemantics` contract.

Success is observable when:

- `chutoro-providers-dense` exposes a `DistanceSemantics` value object that
  fixes the epsilon, non-finite policy, zero-vector policy, and tie-breaking
  rule used across every backend, plus a scalar oracle reducer that any backend
  can be checked against.
- A new property-based parity suite generates pairwise and query-to-points
  inputs covering vector lengths around 16-lane boundaries, varying packed
  candidate counts and tail-padding patterns, duplicate vectors, all-zero
  vectors, and non-finite inputs, and asserts each compiled and runtime-
  supported backend matches the scalar oracle within the contract epsilon and
  canonicalizes non-finite results to `f32::NAN`.
- The new tests build under stable Rust and skip gracefully on hosts where
  AVX2, AVX-512, or Neon are not available, while also running the
  portable-SIMD backend under the existing nightly Continuous Integration (CI)
  job.
- `make check-fmt`, `make lint`, and `make test` all succeed locally on the
  default feature set.
- The Continuous Integration property-tests workflow runs the new
  dense-provider parity suite on every pull request to `main`, and the
  scheduled nightly portable-SIMD workflow includes the same suite with the
  `nightly_portable_simd` feature enabled.
- `docs/chutoro-design.md` §6.3 records the `DistanceSemantics` contract,
  the parity-suite coverage matrix, and the test-seam used to reach each
  backend entrypoint without weakening the existing visibility rules.
- `docs/roadmap.md` marks item `2.2.6` done only after implementation and all
  validation commands succeed.

The previous roadmap items in this section already supply the runtime support
masks (`compiled_simd_support`, `runtime_simd_support`), the dispatch order
(`Avx512 > Avx2 > Neon > PortableSimd > Scalar`), the `DensePointView<'a>`
packing contract, and the non-finite reduction rule (`finalize_distance(value)`
returns `f32::NAN` for any non-finite input). This plan does not change any of
those; it only adds verification on top.

This deliverable also unblocks two downstream roadmap items. Roadmap entry
`2.2.7` (`docs/roadmap.md` line 311) needs a stable `DistanceSemantics`
contract before it can author bounded Kani harnesses for SIMD tail padding and
runtime dispatch selection. Roadmap entries under `4.2` (GPU offload parity)
require the same contract to drive cross-architecture parity — the design
document explicitly states this at `docs/chutoro-design.md` line 1512: "The GPU
distance path should consume the same `DistanceSemantics` contract used by the
CPU backends (§6.3)." Keeping the contract crate-internal for `2.2.6` is a
deliberate scope-limit; promotion to a workspace-shared surface is left to the
later items that will actually consume it.

## Constraints

- Keep Rust source files under 400 lines. Split tests, strategies, or helpers
  rather than extending existing dense SIMD files past that limit.
- Preserve the public `DenseMatrixProvider` and `DataSource` interfaces. This
  item must not widen the public error surface or change public signatures.
- Preserve the existing dense-provider feature names: `simd_avx2`,
  `simd_avx512`, `simd_neon`, and the non-default `nightly_portable_simd`.
- Preserve the existing `build.rs` contract that registers `cfg(nightly)`
  via `cargo:rustc-check-cfg=cfg(nightly)` and only emits
  `cargo:rustc-cfg=nightly` when the active compiler reports itself as nightly.
- Preserve the current dispatch order:
  `Avx512 > Avx2 > Neon > PortableSimd > Scalar`.
- Preserve the existing non-finite policy: every backend canonicalizes
  non-finite Euclidean outputs to `f32::NAN`. The new contract must encode this
  rule; it must not soften it.
- Preserve the existing `DensePointView<'a>` layout contract introduced by
  `2.2.2`: 64-byte alignment, 16-lane padding, and deterministic `0.0_f32` tail
  fill.
- Keep nightly only test code isolated behind
  `all(feature = "nightly_portable_simd", nightly)` at every module,
  entrypoint, and test boundary. Avoid one-off predicates unless the audit
  proves a narrower guard is required.
- New parity tests must skip gracefully when a backend is not compiled in
  for the current target or not detected at runtime. They must not cause CI to
  fail merely because the runner lacks AVX-512 or Neon. The current
  backend-conditional pattern in `chutoro-providers/dense/src/simd/tests/` is
  the precedent to follow.
- Add `proptest` and `test-strategy` only as `dev-dependencies`. Do not add
  any new runtime dependency to `chutoro-providers-dense`.
- Use `rstest` only where parameterized example-based coverage adds value;
  reach for `proptest` whenever the input space is open. Do not duplicate a
  property as both an `rstest` case and a proptest generator.
- Property tests must read the `PROPTEST_CASES` and `CHUTORO_PBT_FORK`
  environment variables consistently with the existing chutoro-core
  property-suite conventions, so PR runs and weekly runs share one code path.
- Use British English with Oxford spelling in comments and documentation,
  and use the Oxford comma only when it improves clarity. American English is
  acceptable in identifiers where the public Rust ecosystem uses it
  (`canonicalize`, `optimize`, `vectorize`).
- Follow guidance from:
  - `docs/chutoro-design.md` (especially §6.3)
  - `docs/property-testing-design.md`
  - `docs/complexity-antipatterns-and-refactoring-strategies.md`
  - `docs/rust-testing-with-rstest-fixtures.md`
  - `docs/rust-doctest-dry-guide.md`

## Tolerances (exception triggers)

- Scope: if implementation requires changes to more than 12 source files or
  roughly 1,200 net lines of code (excluding generated proptest-regressions
  files), stop and escalate.
- Public interface: if any public API on `chutoro-providers-dense`,
  `chutoro-core`, or `chutoro-test-support` has to change to land the parity
  suite, stop and escalate.
- Dependencies: if any new runtime dependency is required, stop and
  escalate. New `dev-dependencies` are limited to `proptest` and
  `test-strategy` at the versions already used in `chutoro-core`.
- Iterations: if tests still fail after three rounds of investigation and
  fix, stop and escalate. Spurious flake (non-determinism between runs) must be
  treated as a bug in the strategy or oracle, not as a tolerance to widen.
- Time: if any single milestone in the plan of work below takes more than
  four hours of focused effort, stop and escalate.
- Ambiguity: if the design document or roadmap can be read more than one
  way and the choice materially changes the contract or the kernels under test,
  stop and present options with trade-offs.
- Epsilon: if matching all backends within a single fixed `f32` epsilon
  proves infeasible (for example, AVX-512 reductions diverge from scalar more
  than `1e-5` on inputs the suite must cover), stop and escalate rather than
  silently widening epsilon. The contract value matters more than passing the
  test.
- Real defect surfaced: if the property suite reproducibly fails for a
  backend that exists today (AVX2, AVX-512, Neon, or portable SIMD) and the
  failure is not a property-suite or oracle bug, stop. Record the minimal
  shrunk counterexample under `proptest-regressions/` and escalate so the
  reviewer can decide whether to fix the kernel, tighten the oracle, or restate
  the contract. Do not paper over a kernel divergence by relaxing the parity
  property.
- Runtime budget: if a single PR-tier invocation
  (`cargo nextest run -p chutoro-providers-dense simd::tests::parity::`) at
  `PROPTEST_CASES=250` and `CHUTORO_PBT_FORK=false` exceeds 60 seconds on the
  development host, stop and reduce strategy cardinality before continuing. The
  PR-tier suite should stay well under the existing chutoro-core property-suite
  runtime so it does not dominate `make test`.

## Risks

- Risk: Floating-point rounding differs between backends on inputs that mix
  large and small magnitudes, even when both are mathematically equivalent.
  Severity: medium. Likelihood: medium. Mitigation: choose epsilon as a
  function of vector length and magnitude (relative tolerance plus absolute
  floor); document the choice in `DistanceSemantics`; clamp the proptest value
  range so that intermediate squared sums stay within representable `f32`
  precision; reuse the existing scalar reducer as the reference, so parity is
  asymmetric (every backend matches scalar; backends are not compared to each
  other).
- Risk: CI runners may lack AVX-512 (and certainly lack Neon on x86 hosts),
  so the parity suite covers only a subset on any single runner. Severity:
  medium. Likelihood: high. Mitigation: gate each backend comparison on both
  compile-time feature and runtime detection, exactly as `tests/entrypoints.rs`
  already does; document expected CI coverage matrix; add a small invariant
  assertion that at least one SIMD backend ran (when any are compiled in and
  the runtime reports support), so that a silently empty suite is impossible.
- Risk: Non-finite inputs interact unpredictably with vector reductions,
  particularly under AVX-512 fused multiply-add (FMA). Severity: low.
  Likelihood: medium. Mitigation: require only the canonicalization invariant
  (any non-finite intermediate or final yields `f32::NAN`), not bit-equality
  with scalar; rely on the existing `finalize_distance` reducer in every
  entrypoint.
- Risk: `DensePointView<'a>` only supports query-centric batches, so the
  query-to-points parity strategy must produce shared-query pair lists.
  Severity: low. Likelihood: low. Mitigation: build the view directly from a
  matrix of generated rows plus a candidate index list, mirroring the existing
  `DensePointView::from_row_indices` API.
- Risk: Adding `proptest` and `test-strategy` to `chutoro-providers-dense`
  pulls them into the dev-dependency closure for downstream
  `chutoro-providers-dense` consumers' tests. Severity: low. Likelihood: low.
  Mitigation: add only as `[dev-dependencies]`; downstream crates pulling in
  this provider already build it with `cargo test --workspace` and accept the
  existing `bytes`, `rstest`, and `trybuild` cost.
- Risk: Property tests can dominate runtime under `--all-features` because
  the suite re-runs the same generators against multiple backends. Severity:
  medium. Likelihood: medium. Mitigation: factor each property to compute the
  scalar oracle once per generated input and then loop over enabled backends;
  honour `PROPTEST_CASES`, so the PR-tier budget stays small; add a dedicated
  nextest filter so the suite is easy to run in isolation.
- Risk: The test seam needed to drive each backend entrypoint independently
  could leak internals via overly broad visibility. Severity: low. Likelihood:
  medium. Mitigation: keep all entrypoints `pub(super)` (their current
  visibility) and place the parity tests inside the crate's existing
  `simd::tests` tree so they need no visibility widening; if an integration
  test tier is desired, expose a narrow `pub(crate)` parity facade instead of
  widening every kernel function's visibility.
- Risk: Choosing the wrong location or visibility for `DistanceSemantics` now
  forces a churny migration when GPU work in `4.2.x` and the bounded Kani
  harness in `2.2.7` start consuming it. Severity: medium. Likelihood: medium.
  Mitigation: keep the type and its policy enums confined to the dense
  provider's `#[cfg(test)]` tree for this milestone; capture the migration
  expectation in the `Decision log` so the later items can lift the contract
  into `chutoro-core` (or an internal shared crate) without redesigning it.
- Risk: Shrinking interacts poorly with `f32::NAN` and `±∞` because partial
  ordering and float equality are undefined for non-finite values. Severity:
  low. Likelihood: high. Mitigation: keep finite and non-finite generators in
  separate `prop_oneof!` arms so a finite-flavour failure shrinks against
  finite-only data; assert non-finite parity via `is_nan()` rather than numeric
  comparison; document this in `assert_close` so the helper itself encodes the
  rule.

## Progress

- [x] Draft this ExecPlan and present it for approval.
- [x] Stage A: orient and propose. Confirm the placement of the parity
  suite (in-crate `src/simd/tests/parity*.rs` modules), confirm the
  `DistanceSemantics` shape, and confirm the dev-dependency additions.
- [x] Stage B: scaffolding. Add `proptest` and `test-strategy` as
  `dev-dependencies`; introduce `DistanceSemantics`; add a backend-iterator
  test seam that returns the set of currently runnable backends.
- [x] Stage C: implement strategies and properties for pairwise distance.
- [x] Stage C: implement strategies and properties for query-to-points
  distance.
- [x] Stage C: implement non-finite parity property and zero-vector or
  duplicate-vector property.
- [x] Stage D: CI wiring (`property-tests.yml` matrix entry,
  `nightly-portable-simd.yml` invocation), design-doc capture, and roadmap flip
  to done.
- [x] Stage D: validation. Run `make check-fmt`, `make lint`, and
  `make test`, capturing transcripts via `tee`.

- 2026-05-17T11:12:35Z: Execution began on branch
  `2-2-6-property-based-backend-parity-suite`. Loaded `leta`, `rust-router`,
  and the ExecPlan workflow. Created the `leta` workspace for the worktree.
  Completed Stage A orientation: scalar and backend pairwise entrypoints
  already canonicalize through `finalize_distance`; support masks remain the
  right runtime gate; `DensePointView<'a>` packs 16-lane aligned, zero-padded
  query-point batches; and the existing query-to-points optimized entrypoints
  are reachable through `select_euclidean_query_points_kernel`, with AVX2,
  AVX-512, and Neon implementations kept inside their backend modules.
- 2026-05-17T11:34:24Z: Completed Stages B and C in the first implementation
  milestone. Added dense-provider dev dependencies for `proptest`,
  `test-strategy`, and the existing workspace `chutoro-test-support` helper;
  introduced a test-only `DistanceSemantics` contract; added support-mask
  driven backend enumeration and entrypoint mapping; and added pairwise,
  query-to-points, non-finite, zero-vector, and duplicate-vector property
  coverage under `chutoro-providers/dense/src/simd/tests/parity/`. Targeted
  `cargo nextest run -p chutoro-providers-dense simd::tests::parity::` passed
  with 3 tests. `make check-fmt`, `make lint`, and `make test` passed before
  CodeRabbit review; `make test` ran 947 tests, all passed, with 1 skipped.
  CodeRabbit found three issues; all were fixed and the targeted parity suite
  passed again.
- 2026-05-17T12:13:45Z: Cleared the remaining CodeRabbit concerns. Final
  CodeRabbit pass reported 0 findings. Final validation for the first milestone
  passed: `make check-fmt`, `make lint`, targeted
  `markdownlint-cli2 docs/execplans/2-2-6-property-based-backend-parity-suite.md`,
  and `make test`. The final `make test` run completed 947 tests with 947
  passed and 1 skipped.
- 2026-05-17T12:18:00Z: Started Stage D. Added the `dense_simd` property-test
  suite to both pull-request and weekly CI matrices, extended the path filter
  to cover `chutoro-providers/dense/**`, added the nightly portable-SIMD parity
  invocation, and drafted the design-document and developers-guide updates. The
  roadmap remains unchecked until the validation gates pass.
- 2026-05-17T12:29:03Z: Stage D pre-roadmap validation passed. `make check-fmt`,
  `make lint`, and `make test` all succeeded; the test run completed 947 tests
  with 947 passed and 1 skipped. Roadmap item `2.2.6` was then flipped to done
  as required by the plan.
- 2026-05-17T12:46:17Z: Final Stage D validation passed on the roadmap-done
  content. Targeted `markdownlint-cli2` passed for the touched Markdown files,
  `make check-fmt` passed, `make lint` passed, and `make test` completed 947
  tests with 947 passed and 1 skipped. `coderabbit review --agent` reported 0
  findings for the milestone.
- 2026-05-18T19:34:48Z: Follow-up review found that non-finite inputs only
  reached pairwise backend entries. Added a dedicated non-finite
  query-to-points fixture that injects `NaN` or infinities into either the
  query row or a candidate point row, plus a property that drives every
  query-to-points backend entry against the scalar oracle. Targeted
  `cargo nextest run -p chutoro-providers-dense simd::tests::parity::` now runs
  4 parity tests and passes.

## Surprises & discoveries

- 2026-05-17T11:12:35Z: The public shared surface for proptest configuration is
  `chutoro_test_support::ci::property_test_profile::ProptestRunProfile`.
- 2026-05-17T11:12:35Z: The historical property-test environment variable is
  misspelled as `PROGTEST_CASES` in `chutoro-test-support`, while
  `.github/workflows/property-tests.yml` already sets both `PROGTEST_CASES` and
  `PROPTEST_CASES`. The dense parity suite should use the shared profile loader
  so it remains compatible with the current workspace convention.
- 2026-05-17T11:12:35Z: The dense query-to-points SIMD entry surface is split
  by module. Portable-SIMD has a top-level
  `kernels::euclidean_distance_query_points_portable_simd_entry`, while AVX2,
  AVX-512, and Neon query-to-points entrypoints live inside their backend
  implementation modules and are currently selected only through
  `select_euclidean_query_points_kernel`.
- 2026-05-17T11:34:24Z: `make test` on this repository is intentionally broad
  because it runs `cargo nextest run --all-targets --all-features`; local
  validation therefore includes benchmark harness tests and took roughly 10.5
  minutes on this machine.
- 2026-05-17T12:18:00Z: The global Markdown formatting target is not suitable
  for this milestone because existing unrelated documentation violates the
  repository's 80-column Markdown line-length rule. Stage D uses targeted
  `markdownlint-cli2` validation for the touched Markdown files, while Rust
  formatting still uses the workspace Cargo formatter through `make check-fmt`.
- 2026-05-18T19:34:48Z: The original non-finite property name was accurate for
  pairwise entries but too broad for the completed roadmap policy. Batched
  query-to-points kernels need their own generated non-finite fixture because
  `query_points_fixture()` intentionally mixes only finite, duplicate, and zero
  rows.

## Decision log

- Decision: place the parity suite inside `chutoro-providers-dense` as
  in-crate `#[cfg(test)] mod` files under `src/simd/tests/parity/` rather than
  as integration tests under `tests/`. Rationale: keeps the existing kernel
  visibility (`pub(super)`) intact while letting the suite reach every backend
  entrypoint; matches the existing pattern in `src/simd/tests/entrypoints.rs`;
  lets the cfg gates for AVX2, AVX-512, Neon, and portable-SIMD be expressed
  once at module level rather than repeated across an integration test
  boundary. Date/Author: 2026-05-02, planning.
- Decision: define `DistanceSemantics` as a value object exposing the
  contract (`epsilon`, non-finite policy, zero-vector policy, tie-breaking
  rule) plus two oracle methods (`oracle_pairwise` and `oracle_query_points`),
  rather than a trait that backends implement. Rationale: the contract is fixed
  for the Euclidean kernel; existing backends are free functions, not types; a
  value object keeps the reusable comparison logic in one place without forcing
  a trait implementation onto every backend. Date/Author: 2026-05-02, planning.
- Decision: gate each backend comparison on the existing
  `compiled_simd_support`/`runtime_simd_support` masks rather than rolling a
  new feature-detection mechanism for the tests. Rationale: those masks are
  already the source of truth used by `select_euclidean_kernel` and
  `select_euclidean_query_points_kernel`, and reusing them keeps the tests
  honest about what dispatch would actually pick. Date/Author: 2026-05-02,
  planning.
- Decision: implement the new parity suite without proptest forking on PR
  runs (`CHUTORO_PBT_FORK=false`, `PROPTEST_CASES=250`) and with forking on the
  weekly run (`CHUTORO_PBT_FORK=true`, `PROPTEST_CASES` taking the same budget
  the existing chutoro-core suites use). Rationale: matches the established
  two-tier model in `docs/property-testing-design.md` §5 and keeps PR feedback
  fast. Date/Author: 2026-05-02, planning.
- Decision: do not author a dedicated ADR for `DistanceSemantics` as part of
  `2.2.6`. Rationale: `docs/chutoro-design.md` §6.3 already specifies the
  contract in prose, and the implementation update note added in Stage D
  records the realized shape, the calibrated epsilon, and the policy enums. An
  ADR is only required if Stage C calibration surfaces a substantive trade-off
  (for example, an epsilon policy the GPU side cannot honour); in that case,
  write `docs/adr-003-distance-semantics-contract.md` following the
  dotted-numbering convention of `adr-001-commit-post-processing.md` and
  `adr-002-adoption-of-kani-formal-verification.md`, and reference it from
  §6.3. Date/Author: 2026-05-04, planning.
- Decision: build the dense parity suite's proptest `Config` from
  `chutoro_test_support::ci::property_test_profile::ProptestRunProfile`.
  Rationale: the test-support profile loader is the workspace-shared policy
  surface. This preserves the public API constraint and avoids coupling the
  dense provider's tests to `chutoro-core` internals. Date/Author: 2026-05-17,
  implementation.
- Decision: keep backend availability enumeration in `dispatch.rs` and place
  test-only entrypoint mapping helpers in `kernels.rs`. Rationale:
  `dispatch.rs` owns support-mask logic, while `kernels.rs` can legally refer
  to private backend implementation modules such as `x86_simd` and `neon_simd`
  without widening their visibility. Date/Author: 2026-05-17, implementation.
- Decision: address CodeRabbit's first milestone findings before committing.
  Rationale: the review identified valid robustness issues in generated fixture
  failure handling and empty-backend assertions, plus a broad `let _ = self` in
  the semantics oracle methods. Replacing fixture failure branches with precise
  `expect(...)`, asserting non-empty entry lists in the non-finite property,
  and reading policy fields through a small debug contract assertion keeps the
  test intent explicit. Date/Author: 2026-05-17, implementation.
- Decision: keep the policy enums in `DistanceSemantics` despite having one
  variant each today. Rationale: the ExecPlan explicitly calls for named policy
  fields, so the CPU parity suite can document and later migrate the same
  contract into GPU parity and tie-breaking verification work. A short code
  comment now records that future extension point. Date/Author: 2026-05-17,
  implementation.
- Decision: add a separate `non_finite_query_points_fixture()` rather than
  mixing non-finite values into the existing finite query-to-points fixture.
  Rationale: the finite fixture still gives focused coverage for ordinary
  parity, duplicate rows, and zero rows, while the non-finite fixture keeps
  canonicalization failures shrinkable and makes the batched-kernel policy
  explicit. Date/Author: 2026-05-18, implementation follow-up.

Append further decisions during execution.

## Outcomes & retrospective

Roadmap item `2.2.6` is complete. The dense provider now has a crate-internal
`DistanceSemantics` contract, property-based pairwise and query-to-points SIMD
backend parity coverage, pairwise and query-to-points non-finite
canonicalization coverage, and CI wiring for stable property-test and nightly
portable-SIMD tiers. The design document, developers' guide, and roadmap now
describe the implemented contract and extension workflow.

The main retrospective item is CI cost visibility. The parity suite itself is
small, but local `make test` includes benchmark harness tests and repeatedly
takes around ten minutes. Future ExecPlans should distinguish targeted
behavioural validation from full workspace gate cost earlier, while still
retaining the full gate before commit.

## Context and orientation

A reader who has just cloned the repository should expect the following
landmarks. All paths are repository-relative.

`chutoro-providers/dense/Cargo.toml` defines the SIMD feature flags. The
defaults are `simd_avx2`, `simd_avx512`, `simd_neon`; the optional
`nightly_portable_simd` feature is non-default and additionally requires a
nightly compiler.

`chutoro-providers/dense/src/simd/` is the kernel tree:

- `mod.rs` is the public-to-the-crate facade. It re-exports
  `DensePointView<'a>` and the scalar wrappers (`euclidean_distance(...)` and
  friends), and it owns the high-level `euclidean_distance_batch_raw_pairs`
  entry. `tests` is declared at the bottom under `#[cfg(test)]`.
- `dispatch.rs` defines `EuclideanBackend`, `CompiledSimdSupport`, and
  `RuntimeSimdSupport`, plus the cached `OnceLock` selector. The selection
  helpers `compiled_simd_support()` and `runtime_simd_support()` are the single
  source of truth for backend availability.
- `kernels.rs` exposes the per-backend entry functions
  (`euclidean_distance_avx2_entry`, `..._avx512_entry`, `..._neon_entry`, and
  `..._portable_simd_entry`) plus the scalar reference
  (`euclidean_distance_scalar` and the
  `euclidean_distance_query_points_scalar`). Every entry funnels through
  `finalize_distance`, which returns `f32::NAN` for any non-finite final
  reduction.
- `point_view.rs` defines `DensePointView<'a>`, the 64-byte aligned,
  16-lane padded Structure of Arrays packing for query-centric batches.
- `kernels/x86_simd.rs`, `kernels/neon_simd.rs`, and
  `kernels/portable_simd.rs` are the per-backend implementation modules, each
  gated by the same `cfg` predicate the parity suite must reuse.
- `tests.rs` is the top of the in-crate test tree. It declares the
  `entrypoints` and `support_masks` submodules and contains the existing rstest
  cases for `DensePointView`, dispatch, and basic non-finite handling. It
  defines a `close(left, right)` helper that compares `Distance` values within
  `1.0e-6_f32`.
- `tests/entrypoints.rs` is the precedent for backend-conditional testing:
  it compiles each backend test only when both the feature and the target
  architecture are available, and skips at runtime when CPUID does not report
  the relevant feature.
- `tests/support_masks.rs` is the precedent for parameterized
  rstest coverage of the dispatch matrix.

The chutoro-core property-test pattern lives at
`chutoro-core/src/hnsw/tests/property/`. It is not a strict template for this
work because it tests a different kind of system, but it shows the project's
preferred decomposition (`strategies.rs`, dataset generators, property modules
per invariant). The parity suite should follow the same spirit: one strategies
module, one or more property modules, and small shared helpers.

The shared property-suite configuration source of truth lives at
`chutoro-test-support/src/ci/property_test_profile.rs`.
`chutoro_test_support::ci::property_test_profile::ProptestRunProfile` turns
`PROPTEST_CASES` and `CHUTORO_PBT_FORK` into a typed profile. The new parity
suite should use `ProptestRunProfile::load(default_cases, default_fork)` to
obtain the case count and fork flag for `proptest::test_runner::Config`, so
PR-tier and weekly tier budgets stay aligned across the workspace. For example:

```rust
let profile = ProptestRunProfile::load(default_cases, false);
ProptestConfig {
    cases: profile.cases(),
    fork: profile.fork(),
    ..ProptestConfig::default()
}
```

`docs/users-guide.md` does not document SIMD backend selection, distance
semantics, or non-finite policy, and `2.2.6` keeps the dense provider's public
API unchanged. No user-facing prose change is required for this milestone; the
implementation-update note in `docs/chutoro-design.md` §6.3 plus the
developers-guide subsection added in Stage D are sufficient.

CI lives in `.github/workflows/`:

- `ci.yml` runs the default test matrix and the dense-provider gating
  step that proves stable builds with
  `--no-default-features --features simd_avx2,simd_avx512,simd_neon` are clean.
- `property-tests.yml` runs the chutoro-core property suites under PR and
  weekly tiers. The new parity suite must extend this workflow.
- `nightly-portable-simd.yml` runs the dense-provider tests with
  `nightly_portable_simd` enabled on a scheduled cadence. The new parity suite
  must run there too, so the portable-SIMD backend is exercised.

The `Makefile` exposes `check-fmt`, `lint`, `test`, `fmt`, `markdownlint`,
`nixie`, `kani`, `kani-full`, `verus`, and `bench`. Use `make test` and
`make lint` for validation; `cargo nextest run` is the underlying runner.

Glossary (terms used below; defined here so the reader does not have to infer
them):

- *Backend.* The concrete implementation of a Euclidean distance kernel
  for a particular instruction set or fallback. Today the dense provider has
  five: scalar, AVX2, AVX-512, Neon, and portable-SIMD.
- *Oracle.* A trusted reference implementation against which other
  implementations are checked. Here the scalar kernel is the oracle.
- *Lane boundary.* A vector length at or near the SIMD register width
  (16 single-precision lanes for AVX-512 and the portable-SIMD implementation;
  8 for AVX2; 4 for Neon). The kernels handle remainder tails differently from
  the main loop, so lengths around `n * 16` exercise more code paths than
  uniformly random lengths.
- *Tail padding.* The zero-filled lanes appended by `DensePointView<'a>`
  when the candidate count is not a multiple of 16. Backends must read these
  lanes safely and produce zero contribution.

## Plan of work

The plan progresses through four stages with go/no-go validation at each stage
boundary.

### Stage A: orient and propose (no code changes)

Read the kernel tree top-to-bottom and confirm the four observations the plan
rests on: backend entrypoints already canonicalize non-finite results;
`DensePointView<'a>` already enforces alignment and padding; support masks are
the right oracle for runtime gating; and the existing `tests/entrypoints.rs`
shows the established compile-and-runtime gating idiom. If any of these turn
out to be false, stop and update this plan before scaffolding code.

Produce a one-page sketch of the `DistanceSemantics` value object and the
test-seam shape. Confirm with the reviewer that the approach matches the design
document's "shared distance semantics and verification seam" intent. The seam
itself is the `EnabledBackends` enumerator described in Stage B; do not invent
new visibility surfaces.

Stage A exit gate: the reviewer has acknowledged the contract sketch and the
tests-only scope; no source changes have been made yet.

### Stage B: scaffolding and tests (small, verifiable diffs)

1. Update `chutoro-providers/dense/Cargo.toml` to add
   `proptest = "1.8.0"` and `test-strategy = "0.4.3"` to `[dev-dependencies]`.
   Use the same versions as `chutoro-core`. Do not add anything to
   `[dependencies]`.
2. Introduce `chutoro-providers/dense/src/simd/semantics.rs`:
   - Define `DistanceSemantics` (struct, not trait) with fields for
     `epsilon: f32`, `non_finite_policy: NonFinitePolicy`,
     `zero_vector_policy: ZeroVectorPolicy`, and
     `tie_breaking: TieBreakingPolicy`.
   - Provide `DistanceSemantics::default_euclidean()` that returns the
     contract values used by the existing kernels (epsilon TBD by the
     audit in stage C; non-finite policy `CanonicaliseToNan`;
     zero-vector policy `ReturnZero`; tie-breaking `LowestRowIndexFirst`,
     even though Euclidean per-pair scoring does not currently break ties
     by index, the contract documents the choice for parity with future
     selection logic).
   - Provide `oracle_pairwise(left: &[f32], right: &[f32]) -> f32` that
     delegates to `kernels::euclidean_distance_scalar` and
     `oracle_query_points(query: &[f32], points: &DensePointView<'_>,
     out: &mut [f32])` that delegates to
     `kernels::euclidean_distance_query_points_scalar`. Wrapping these
     keeps the oracle change-point in one place if the contract
     evolves.
   - Provide `assert_close(actual: f32, expected: f32)` and
     `assert_query_close(actual: &[f32], expected: &[f32])` helpers that
     centralize the epsilon comparison, including the non-finite carve-
     out (both must be `NaN` when the contract demands canonicalization).
3. Introduce a small backend-iterator test seam in
   `chutoro-providers/dense/src/simd/dispatch.rs` (still `pub(super)`) that
   yields the `EuclideanBackend` variants currently both compiled and
   runtime-supported. Use the existing `compiled_simd_support`/
   `runtime_simd_support` accessors. Provide a matching helper that maps a
   backend variant to its pairwise entry function pointer and to its
   query-to-points entry function pointer, each returning `Option<...>` so the
   test can skip absent backends.
4. Wire `mod semantics;` into `simd/mod.rs` under `#[cfg(test)]` so the
   semantics surface is available only to the in-crate test tree. If the
   reviewer prefers semantics to be part of the released crate API, the gate
   can be relaxed in a later patch; this plan keeps the contract internal until
   a consumer needs it.
5. Create `chutoro-providers/dense/src/simd/tests/parity/` containing:
   - `mod.rs` declaring `strategies`, `pairwise`, `query_points`, and
     `non_finite` submodules.
   - `strategies.rs` defining proptest strategies for vector lengths
     near lane boundaries (`prop_oneof!` mixing exact lane multiples,
     `lane * n - 1`, `lane * n + 1`, plus a small uniform range), value
     ranges that keep squared sums representable in `f32`, and explicit
     "duplicate row" and "all-zero row" injectors via `prop_oneof!`.
     Strategies must return owned `Vec<f32>` data and the metadata
     needed to construct a `RowMajorMatrix`.
   - `pairwise.rs`, `query_points.rs`, and `non_finite.rs` each holding
     one or two `proptest!` blocks. Each property iterates over the
     enabled backend set, computes the oracle once, and asserts every
     enabled backend matches within the contract epsilon (or that all
     produce `NaN` for the non-finite property).
6. Re-run `make check-fmt` and `make lint` after each step in this
   stage; both must succeed before moving on. The first proptest pass does not
   need to find bugs; it needs to compile and run on the default feature set.

Stage B exit gate: `cargo nextest run -p chutoro-providers-dense` runs the new
parity suite under the default features without failure on the local host, and
`make lint` is clean.

### Stage C: implementation (minimal change to satisfy tests)

Most "implementation" work in this item is test code; the kernels already
satisfy the parity contract. Only address kernel changes if the property tests
find a real divergence between scalar and a SIMD backend; treat any such
finding as a bug and fix the kernel, not the oracle.

The property work in this stage:

1. Tune the contract epsilon. Run a brief offline calibration: for vector
   lengths `1`, `15`, `16`, `17`, `31`, `32`, `33`, `47`, `48`, `64`, `127`,
   `128`, `129`, `255`, `256`, generate 1,000 random pairs in `[-1.0, 1.0]` and
   record `max(|scalar - backend|)` for each enabled backend. Set the contract
   epsilon to the smallest value above the worst observed delta, rounded up.
   Document the calibration in the `Decision log` and in the
   `DistanceSemantics::default_euclidean` doc comment.
2. Pairwise property (`pairwise.rs`): generate a `(left, right)` pair
   under the strategy. Compute the oracle. For each enabled backend, compute
   the backend output. Assert
   `DistanceSemantics::assert_close(backend_output, oracle_output)`. Add
   shrinking-friendly debug formatting (vector length, lane class, and
   first-mismatch index) so a failure prints something diagnosable.
3. Query-to-points property (`query_points.rs`): generate a matrix of
   `rows x dimension` `f32` values, plus a candidate row-index list of length
   1..=`MAX_CANDIDATES` (cover 1, 16, 17, and a small open range). Build a
   `DensePointView<'a>` via `DensePointView::from_row_indices`. For each
   enabled backend that has a query-to-points entrypoint, compute the backend
   output and compare to the oracle vector via `assert_query_close`. Use a
   shared-query candidate list so the existing `should_pack_query_points` gate
   fires.
4. Non-finite property (`non_finite.rs`): generate base finite vectors,
   then sprinkle in `f32::NAN`, `f32::INFINITY`, and `f32::NEG_INFINITY` at
   random positions under `prop_oneof!`. Assert every enabled backend's output
   is `NaN`. Do not require equality between backends' non-finite outputs
   beyond the canonicalization invariant.
5. Coverage assertion: when any SIMD backend is compiled in and the
   runtime reports support for at least one of them, the property harness must
   record that at least one non-scalar backend ran during the proptest pass. A
   property that silently degenerates to "scalar versus scalar" because of a
   misconfigured cfg gate would not catch anything; the harness should `panic!`
   after the proptest pass with a clear message in that case (gated to
   `#[cfg(any(feature = ...))]` to remain quiet on hosts where no SIMD backend
   is compiled in).
6. Re-run `make check-fmt`, `make lint`, and `make test` after each of
   the three property additions. Investigate any failure to root cause before
   continuing.

Stage C exit gate: `make test` is clean on the default feature set;
`cargo +nightly test -p chutoro-providers-dense --features nightly_portable_simd`
runs the parity suite locally (or, if a nightly compiler is unavailable on the
development machine, the gating `tests/portable_simd_gating.rs` proof of
compilation passes) and shows the portable-SIMD backend participating.

### Stage D: hardening, documentation, cleanup

1. Extend `.github/workflows/property-tests.yml`:
   - Add `dense_simd` to the `matrix.suite` array on both
     `property-tests-pr` and `property-tests-weekly` jobs.
   - Add a corresponding case in the `Run property suite` step that
     invokes
     `cargo nextest run --profile ci -p chutoro-providers-dense
     simd::tests::parity::` with the same `tee` redirection pattern.
   - Extend the `paths` filter so this workflow also fires on changes
     under `chutoro-providers/dense/**`.
2. Extend `.github/workflows/nightly-portable-simd.yml` to invoke the
   parity suite after the existing dense-provider test step:

   ```sh
   cargo +nightly test -p chutoro-providers-dense \
     --features nightly_portable_simd \
     simd::tests::parity
   ```

   Keep the existing `--features nightly_portable_simd` flag so the
   portable-SIMD backend is in the enabled set.
3. Update `docs/chutoro-design.md` §6.3 to record:
   - The `DistanceSemantics` value object and its fields.
   - The scalar-oracle reducer as the canonical reference, including
     that every backend funnels through `finalize_distance` so non-
     finite outputs canonicalize to `f32::NAN`.
   - The parity-suite coverage matrix (lane boundaries, padding,
     duplicates, zeros, non-finite inputs) and where it lives in the
     tree.
   - The CI coverage map (PR-tier `property-tests.yml`, weekly tier,
     and nightly portable-SIMD).
   - A short implementation update note timestamped with the merge
     date, mirroring the format already used for items `2.2.1` through
     `2.2.5`.
4. Update `docs/developers-guide.md` with a short subsection describing
   how to extend the parity suite when a new backend or kernel is added: where
   to register a backend in `dispatch.rs::enabled_backends`, how to wire the
   entry function into `pairwise_entry` and `query_points_entry`, how to add a
   strategy variant, and how the proptest-regressions file acts as a regression
   guard. This satisfies the task brief's "internally facing interfaces or
   practices" requirement and gives a future contributor a concrete, in-tree
   on-ramp.
5. Update `docs/roadmap.md` only after `make test`, `make lint`,
   `make markdownlint`, `make fmt`, and `make nixie` are clean and the
   design-doc update has been written: change `[ ]` to `[x]` on item `2.2.6`
   (`docs/roadmap.md` line 304).
6. Final validation pass: run `make check-fmt`, `make lint`, `make test`,
   `make markdownlint`, `make fmt`, and `make nixie`, redirecting each to a
   `tee` log under `/tmp` (see "Concrete steps" below). Upload nothing; the
   logs are local only.

Stage D exit gate: all six Make targets succeed; the design document records
the contract; the roadmap shows `2.2.6` complete; the working tree contains
exactly the planned new files plus the targeted edits.

## Concrete steps

Run from the repository root.

```bash
git branch --show-current
```

Expect a non-`main` branch name, for example `session/e987bf41` or a
descriptively named feature branch such as
`feature/2-2-6-simd-backend-parity-suite`.

```bash
make check-fmt 2>&1 \
  | tee /tmp/check-fmt-chutoro-$(git branch --show-current | tr '/' '-').out
make lint 2>&1 \
  | tee /tmp/lint-chutoro-$(git branch --show-current | tr '/' '-').out
make test 2>&1 \
  | tee /tmp/test-chutoro-$(git branch --show-current | tr '/' '-').out
make markdownlint 2>&1 \
  | tee /tmp/markdownlint-chutoro-$(git branch --show-current | tr '/' '-').out
make fmt 2>&1 \
  | tee /tmp/fmt-chutoro-$(git branch --show-current | tr '/' '-').out
make nixie 2>&1 \
  | tee /tmp/nixie-chutoro-$(git branch --show-current | tr '/' '-').out
```

Each command must finish with the relevant success line in the tail of its log:

- `check-fmt`: no output is success.
- `lint`: ends with `Finished` from clippy and a clean `cargo doc`.
- `test`: ends with the cargo-nextest summary indicating zero failures.
- `markdownlint`: ends without Markdown lint errors.
- `fmt`: completes without leaving unexpected Markdown or Rust formatting
  changes.
- `nixie`: validates Mermaid diagrams without errors.

Targeted parity-suite invocation (handy during development):

```bash
cargo nextest run -p chutoro-providers-dense simd::tests::parity:: 2>&1 \
  | tee /tmp/parity-chutoro-$(git branch --show-current | tr '/' '-').out
```

Nightly portable-SIMD parity invocation (only on a host with a nightly
toolchain):

```bash
cargo +nightly test -p chutoro-providers-dense \
  --features nightly_portable_simd simd::tests::parity 2>&1 \
  | tee /tmp/parity-nightly-chutoro-$(git branch --show-current | tr '/' '-').out
```

Mark the roadmap done only after the six Make targets pass cleanly.

```bash
sed -i 's/^- \[ \] 2\.2\.6\./- [x] 2.2.6./' docs/roadmap.md
```

Verify with `git diff docs/roadmap.md` and inspect the surrounding lines to
confirm only the intended entry changed.

## Validation and acceptance

Quality criteria (what "done" means):

- Tests: every assertion in the new parity suite passes under default
  features and under `--features nightly_portable_simd` on a nightly compiler.
  The existing test suite still passes.
- Lint: `make lint` reports no warnings or errors; `cargo doc` builds
  without warnings.
- Format: `make check-fmt` reports no diffs.
- Coverage: at least one SIMD backend (other than scalar) participates
  in the parity comparison on x86 CI runners; on the nightly run, the
  portable-SIMD backend additionally participates.
- Determinism: reruns of the suite under the same `PROPTEST_CASES` and
  same seed reproduce the same outcome. Any non-determinism between runs is
  treated as a bug and reported.
- Contract: `DistanceSemantics::default_euclidean()` documents the
  exact epsilon and policy values used by the suite; that doc comment is the
  canonical source.

Quality method (how we check):

- Local: run the six Make targets above with `tee` redirection, inspect
  the tails for the success markers, and inspect `git status` to confirm only
  the planned files changed. Documentation or diagram changes require
  `make markdownlint`, `make fmt`, and `make nixie`; `markdownlint-cli2` is not
  the primary repository gate.
- CI: `property-tests.yml` PR job exercises `dense_simd`; the merge is
  not eligible until that job is green. The weekly run extends the case budget;
  failures there must be triaged on the following workday.
- CI nightly: `nightly-portable-simd.yml` exercises the parity suite
  with the portable-SIMD backend enabled on its scheduled cadence.
- Manual review: a reviewer reads
  `chutoro-providers/dense/src/simd/semantics.rs` and at least one property
  module to confirm that the contract is single-sourced and that no backend
  "passes" by being silently absent.

Acceptance is granted when all of the above are observed and the roadmap entry
is flipped from `[ ]` to `[x]`.

## Idempotence and recovery

- The plan creates new files and edits a small number of existing files.
  Re-running the plan from scratch (after a `git stash` or branch reset) is
  safe; nothing here mutates external state.
- Adding `proptest` and `test-strategy` to `[dev-dependencies]` is
  reversible by reverting the `Cargo.toml` change. No `Cargo.lock` movement
  should be required for the runtime closure.
- The proptest framework writes a `proptest-regressions/` directory
  alongside test sources when a property fails. Treat any committed regression
  file as load-bearing (it pins a known-bad case); do not delete one without
  first confirming the underlying bug is fixed.
- If a kernel divergence is found, fix the kernel rather than the
  oracle. Capture the case in `proptest-regressions/` so the suite guards
  against regression. If the divergence reflects a contract change rather than
  a bug, escalate per the tolerance rules.
- If `make test` flakes at the property tier on a tight CI runner,
  reduce `PROPTEST_CASES` for the PR tier rather than removing generators.
  Document the change in the `Decision log`.

## Artifacts and notes

Expected new and edited files:

- `chutoro-providers/dense/Cargo.toml` (edit; `[dev-dependencies]`).
- `chutoro-providers/dense/src/simd/mod.rs` (small edit; declare
  `mod semantics;` under `#[cfg(test)]`).
- `chutoro-providers/dense/src/simd/dispatch.rs` (small edit; add the
  test-only backend-iterator helpers, still `pub(super)`).
- `chutoro-providers/dense/src/simd/semantics.rs` (new).
- `chutoro-providers/dense/src/simd/tests.rs` (small edit; declare
  `mod parity;`).
- `chutoro-providers/dense/src/simd/tests/parity/mod.rs` (new).
- `chutoro-providers/dense/src/simd/tests/parity/strategies.rs` (new).
- `chutoro-providers/dense/src/simd/tests/parity/pairwise.rs` (new).
- `chutoro-providers/dense/src/simd/tests/parity/query_points.rs` (new).
- `chutoro-providers/dense/src/simd/tests/parity/non_finite.rs` (new).
- `.github/workflows/property-tests.yml` (edit; matrix and `paths`).
- `.github/workflows/nightly-portable-simd.yml` (edit; parity step).
- `docs/chutoro-design.md` (edit; §6.3 implementation update).
- `docs/roadmap.md` (edit; tick item `2.2.6`).
- This ExecPlan (edits to living sections as work proceeds).

Expected validation transcripts (only the success markers; do not paste full
logs into the plan):

```plaintext
$ make check-fmt
$ make lint
...
   Compiling chutoro-providers-dense v0.1.0 ...
    Finished `dev` profile [optimized + debuginfo] target(s) in ...
$ make test
...
     Summary [   N tests run: N passed, 0 failed, 0 skipped ]
```

## Interfaces and dependencies

Be prescriptive. The following interfaces must exist at the end of the
implementation milestone.

In `chutoro-providers/dense/src/simd/semantics.rs`:

```rust
use super::{DensePointView, kernels};

#[derive(Clone, Copy, Debug)]
pub(crate) struct DistanceSemantics {
    pub(crate) epsilon: f32,
    pub(crate) non_finite_policy: NonFinitePolicy,
    pub(crate) zero_vector_policy: ZeroVectorPolicy,
    pub(crate) tie_breaking: TieBreakingPolicy,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum NonFinitePolicy {
    CanonicaliseToNan,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ZeroVectorPolicy {
    ReturnZero,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum TieBreakingPolicy {
    LowestRowIndexFirst,
}

impl DistanceSemantics {
    pub(crate) const fn default_euclidean() -> Self;
    pub(crate) fn oracle_pairwise(&self, left: &[f32], right: &[f32]) -> f32;
    pub(crate) fn oracle_query_points(
        &self,
        query: &[f32],
        points: &DensePointView<'_>,
        out: &mut [f32],
    );
    pub(crate) fn assert_close(&self, actual: f32, expected: f32);
    pub(crate) fn assert_query_close(&self, actual: &[f32], expected: &[f32]);
}
```

In `chutoro-providers/dense/src/simd/dispatch.rs` (test-only additions):

```rust
#[cfg(test)]
pub(super) fn enabled_backends() -> Vec<EuclideanBackend>;

#[cfg(test)]
pub(super) fn pairwise_entry(
    backend: EuclideanBackend,
) -> Option<fn(&[f32], &[f32]) -> f32>;

#[cfg(test)]
pub(super) fn query_points_entry(
    backend: EuclideanBackend,
) -> Option<fn(&[f32], &super::DensePointView<'_>, &mut [f32])>;
```

The test seam returns `None` for any backend not present in both
`compiled_simd_support()` and `runtime_simd_support()`. The pairwise entry for
`Scalar` must return `Some(kernels::euclidean_distance_scalar)` so the parity
tests can compare scalar against itself as a sanity-control case.

In `chutoro-providers/dense/src/simd/tests/parity/strategies.rs`:

```rust
use proptest::prelude::*;

pub(super) fn lane_boundary_length() -> impl Strategy<Value = usize>;
pub(super) fn finite_vector(length: usize) -> impl Strategy<Value = Vec<f32>>;
pub(super) fn non_finite_vector(length: usize) -> impl Strategy<Value = Vec<f32>>;
pub(super) fn pairwise_inputs() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)>;
pub(super) fn query_points_inputs(
) -> impl Strategy<Value = (Vec<f32>, Vec<Vec<f32>>)>;
```

`lane_boundary_length()` mixes `prop_oneof!`:

- `Just(1)`, `Just(7)`, `Just(8)`, `Just(15)`, `Just(16)`, `Just(17)`,
  `Just(31)`, `Just(32)`, `Just(33)`, `Just(47)`, `Just(48)`,
- a small uniform range `1..=256_usize`,
- and `Just(usize::pow(2, n))` for representative powers of two up to 256.

`finite_vector(length)` produces values in a clamped range (proposal:
`-1024.0..=1024.0` with a small-magnitude bias) so squared sums stay in `f32`
precision. `non_finite_vector(length)` mixes finite values with sentinel `NaN`,
`INFINITY`, and `NEG_INFINITY` via `prop_oneof!`.

In
`chutoro-providers/dense/src/simd/tests/parity/{pairwise,query_points,non_finite}.rs`:

```rust
proptest! {
    #[test]
    fn pairwise_backends_match_scalar_oracle(
        (left, right) in strategies::pairwise_inputs(),
    ) {
        let semantics = DistanceSemantics::default_euclidean();
        let oracle = semantics.oracle_pairwise(&left, &right);
        for backend in dispatch::enabled_backends() {
            if let Some(entry) = dispatch::pairwise_entry(backend) {
                let actual = entry(&left, &right);
                semantics.assert_close(actual, oracle);
            }
        }
    }
}
```

Equivalent harnesses for the query-to-points and non-finite cases.

In `chutoro-providers/dense/Cargo.toml` (`[dev-dependencies]`):

```toml
proptest = "1.8.0"
test-strategy = "0.4.3"
```

In `.github/workflows/property-tests.yml` (`matrix.suite`):

```yaml
matrix:
  suite: [hnsw, edge_harvest, mst, dense_simd]
```

with a matching `dense_simd)` arm in the `Run property suite` step that invokes
`cargo nextest run --profile ci -p chutoro-providers-dense simd::tests::parity::`.

In `.github/workflows/nightly-portable-simd.yml`, after the existing
`Test dense portable SIMD backend` step, add:

```yaml
- name: Test dense parity suite under portable SIMD
  if: ${{ steps.gate.outputs.should_run == 'true' }}
  run: cargo +nightly test -p chutoro-providers-dense \
    --features nightly_portable_simd simd::tests::parity
```

These are the only interfaces and surfaces this plan introduces. The public API
of `chutoro-providers-dense`, `chutoro-core`, and `chutoro-test-support` does
not change.
