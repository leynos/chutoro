# Execution plan (ExecPlan): roadmap 2.2.1 CPU single instruction, multiple data (SIMD) distance kernels using `core::arch` and optional `std::simd`

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & discoveries`, `Decision log`, and
`Outcomes & retrospective` must be kept up to date as work proceeds.

Status: COMPLETE

## Purpose / big picture

Implement roadmap item `2.2.1` by adding CPU Single Instruction, Multiple Data
(SIMD) distance kernels with x86 Advanced Vector Extensions 2 (AVX2) and
AVX-512-aware dispatch, and make the Hierarchical Navigable Small World (HNSW)
scoring path use `distance_batch` by default.

Success is observable when:

- Dense numeric distance batches are computed through a SIMD-capable path with
  deterministic scalar fallback.
- HNSW candidate scoring calls a path that ultimately uses
  `DataSource::distance_batch` by default.
- Unit tests (with `rstest` parameterization where repetition exists) cover
  happy paths, unhappy paths, and edge cases for dispatch, correctness, and
  error propagation.
- `docs/chutoro-design.md` records the final design decisions for kernel
  dispatch and fallback semantics.
- `docs/roadmap.md` marks item `2.2.1` as done after implementation completes.
- Quality gates pass: `make check-fmt`, `make lint`, and `make test`.

## Constraints

- Keep all Rust source files under 400 lines. Split modules if needed.
- Preserve public `DataSource` method signatures and return types.
- Do not add external dependencies for SIMD work; use `std` and existing
  workspace crates.
- Keep HNSW determinism and tie-break behaviour unchanged.
- Preserve existing non-finite distance rejection behaviour in
  `hnsw/validate.rs`.
- Use `rstest` for parameterized test coverage where multiple scenarios share
  one assertion shape.
- Follow guidance from:
  - `docs/chutoro-design.md` (especially §6.3)
  - `docs/property-testing-design.md`
  - `docs/complexity-antipatterns-and-refactoring-strategies.md`
  - `docs/rust-testing-with-rstest-fixtures.md`
  - `docs/rust-doctest-dry-guide.md`

## Tolerances (exception triggers)

- Scope: if work requires edits in more than 14 files or more than 900 net
  lines, stop and escalate.
- Interface: if roadmap intent requires changing public `DataSource` method
  signatures, stop and escalate with options.
- Dependencies: if implementation appears to require a new crate dependency,
  stop and escalate.
- Compatibility: stable kernels must compile on the minimum supported Rust
  version (MSRV) `1.89.0` (AVX-512 stabilization baseline), and optional
  `std::simd` code must stay behind a nightly only gate.
- Iterations: if `make lint` or `make test` still fails after 3 repair
  attempts, stop and escalate with logs.
- Ambiguity: if roadmap wording conflicts materially with §6.3 design wording,
  stop and log decision options before proceeding.

## Risks

- Risk: SIMD and scalar kernels may diverge numerically on borderline
  floating-point inputs. Severity: high. Likelihood: medium. Mitigation: add
  equivalence tests with explicit epsilon, including odd dimensions and tail
  handling.

- Risk: Runtime dispatch and `#[target_feature]` boundaries can introduce
  unsafe call-site mistakes. Severity: high. Likelihood: medium. Mitigation:
  isolate unsafe calls in a narrow module and test fallback and specialization
  selection separately.

- Risk: changing batch call paths can break error contracts (length mismatch,
  out-of-bounds propagation, non-finite handling). Severity: high. Likelihood:
  medium. Mitigation: add regression tests for each error class in
  `chutoro-core/tests/datasource.rs` and
  `chutoro-core/src/hnsw/tests/errors.rs`.

- Risk: overloading existing files can reintroduce bumpy-road structure and
  high cognitive complexity. Severity: medium. Likelihood: medium. Mitigation:
  extract small helper functions and keep search/validate modules focused.

## Progress

- [x] (2026-03-02 00:00Z) Drafted ExecPlan for roadmap item `2.2.1`.
- [x] (2026-03-02 00:40Z) Stage A complete: added default-delegation contract
  tests in `chutoro-core/src/datasource.rs`,
  `chutoro-core/tests/datasource.rs`, and
  `chutoro-core/src/hnsw/tests/build.rs`.
- [x] (2026-03-02 00:55Z) Stage B complete: added
  `chutoro-providers/dense/src/simd/mod.rs` and
  `chutoro-providers/dense/src/simd/kernels.rs` with scalar and AVX2 kernels
  plus x86 runtime dispatch.
- [x] (2026-03-02 01:05Z) Stage C complete: changed
  `DataSource::batch_distances` default to delegate to `distance_batch`, and
  routed dense provider batch scoring through the SIMD module.
- [x] (2026-03-02 01:20Z) Stage D complete: updated
  `docs/chutoro-design.md`, marked roadmap item `2.2.1` done, and passed
  `make fmt`, `make markdownlint`, `make nixie`, `make check-fmt`, `make lint`,
  and `make test`.

## Surprises & discoveries

- Observation: HNSW scoring currently calls `batch_distances` during search
  validation, while `DenseMatrixProvider` currently specializes
  `distance_batch`, not `batch_distances`. Evidence:
  `chutoro-core/src/hnsw/validate.rs` calls `source.batch_distances(...)`;
  `chutoro-providers/dense/src/provider.rs` overrides `distance_batch(...)`
  only. Impact: current HNSW scoring bypasses dense-provider batch
  specialization.

- Observation: there is already a test asserting HNSW uses `batch_distances`
  (`uses_batch_distances_during_scoring`), but no test that default
  `batch_distances` delegates to `distance_batch`. Evidence:
  `chutoro-core/src/hnsw/tests/build.rs` and `chutoro-core/src/datasource.rs`
  tests. Impact: a contract test is needed before path changes.

- Observation: `std::simd` remains unavailable on stable toolchains, including
  `1.93.1` (latest stable release on 2026-02-12). Evidence: `rustc` emits
  `E0658: use of unstable library feature portable_simd`.[^1] Impact: the SIMD
  implementation uses stable `std::arch` AVX2 intrinsics and scalar fallback.

- Observation: AVX-512 intrinsics and
  `#[target_feature(enable = "avx512f")]` are stable as of Rust `1.89.0`.
  Evidence: Rust `1.89.0` release notes[^2] and closed tracking issue
  `rust-lang/rust#111137`.[^3] Impact: the stable branch can run a real AVX-512
  kernel instead of degrading to AVX2/scalar.

## Decision log

- Decision: interpret roadmap item `2.2.1` as requiring HNSW scoring to flow
  through the `distance_batch` contract by default while preserving existing
  `batch_distances(query, candidates)` call-sites for ergonomics. Rationale:
  this aligns roadmap wording with the current query-centric call graph and
  avoids broad API churn. Date/Author: 2026-03-02 / Codex.

- Decision: implement SIMD kernels first for dense Euclidean distance batches
  and keep non-dense providers on scalar fallback. Rationale: dense numeric
  data is the target for SIMD wins, and this keeps scope bounded to roadmap
  item `2.2.1`. Date/Author: 2026-03-02 / Codex.

- Decision: keep AVX2/AVX-512 dispatch internal to dense provider internals and
  avoid introducing new Cargo feature flags in this item. Rationale:
  feature-flag gating is separately tracked by roadmap item `2.2.3`.
  Date/Author: 2026-03-02 / Codex.

- Decision: satisfy roadmap `2.2.1` on stable toolchain by implementing SIMD
  via `std::arch` instead of `std::simd`. Rationale: `std::simd` remains
  unstable on stable, but AVX2/AVX-512 specialization and default
  `distance_batch` scoring path can be delivered safely with stable Rust.
  Date/Author: 2026-03-02 / Codex.

- Decision: promote AVX-512 from detect-only fallback to active stable kernel
  path and bump MSRV to `1.89.0`. Rationale: AVX-512 intrinsics are stabilized,
  so the stable implementation should take direct advantage of them while
  preserving scalar fallback. Date/Author: 2026-03-02 / Codex.

## Outcomes & retrospective

Implemented outcomes:

- Added dense-provider SIMD kernel module
  (`chutoro-providers/dense/src/simd/mod.rs` and
  `chutoro-providers/dense/src/simd/kernels.rs`) with
  - scalar Euclidean kernel,
  - AVX2 specialization via `std::arch`,
  - x86 runtime dispatch with AVX-512 detection and stable fallback semantics.
- Updated `DenseMatrixProvider` to route both single-distance and
  `distance_batch` calculations through SIMD-aware kernels.
- Changed default `DataSource::batch_distances` to delegate to
  `distance_batch`, making pair-batch specializations the default HNSW scoring
  path.
- Added regression coverage for
  - default delegation behaviour in trait unit and integration tests,
  - HNSW scoring path exercising `distance_batch` via default delegation,
  - dense-provider batch parity against scalar references across odd and
    lane-tail dimensions.
- Updated design documentation and marked `docs/roadmap.md` item `2.2.1` done.

Validation summary:

- Focused tests passed:
  - `cargo test -p chutoro-core --test datasource`
  - `cargo test -p chutoro-core hnsw::tests::build::`
  - `cargo test -p chutoro-providers-dense provider::`
- Quality gates passed:
  - `make fmt`
  - `make markdownlint`
  - `make nixie`
  - `make check-fmt`
  - `make lint`
  - `make test` (`791 passed, 1 skipped`).

Retrospective:

- The largest implementation constraint remains stable-toolchain support for
  `std::simd`, which is still nightly only.
- Preserving a narrow kernel-module boundary kept the change coherent and made
  the fallback policy explicit.

## Context and orientation

Relevant current files and behaviour:

- `chutoro-core/src/datasource.rs`:
  - `batch_distances(query, candidates)` now delegates to
    `distance_batch(pairs, out)` by constructing `(query, candidate)` pairs.
  - `distance_batch(pairs, out)` still defaults to repeated `distance(...)`,
    includes output-length checks, and leaves `out` unchanged on error.
- `chutoro-core/src/hnsw/validate.rs`:
  - HNSW batch scoring validates via `source.batch_distances(...)`.
  - The no-cache path validates non-finite outputs returned from
    `source.batch_distances(...)`.
- `chutoro-providers/dense/src/provider.rs`:
  - Dense provider overrides `distance_batch` and routes pair batches through
    `chutoro-providers/dense/src/simd/` kernels.
  - Dense provider does not override `batch_distances`, so it inherits the
    datasource default delegation into `distance_batch`.
- `chutoro-core/src/hnsw/tests/build.rs`:
  - existing regression test confirms HNSW exercises `batch_distances`.

Definitions used in this plan:

- SIMD: Single Instruction, Multiple Data; one instruction operates over
  multiple lanes at once.
- AVX2/AVX-512: x86 vector instruction sets with wider lane counts.
- Scalar fallback: a non-SIMD implementation used when specialization is not
  available.

## Plan of work

### Stage A: establish contract and failing tests (red)

Add tests that lock expected behaviour before implementation changes.

Planned edits:

- `chutoro-core/tests/datasource.rs`
  - add parameterized `rstest` coverage proving default
    `batch_distances(query, candidates)` delegates to `distance_batch`.
  - add unhappy-path tests for propagated `OutputLengthMismatch` and
    out-of-bounds errors from delegated `distance_batch` implementations.
- `chutoro-core/src/hnsw/tests/build.rs`
  - add an instrumentation test where only `distance_batch` is specialized,
    then assert HNSW scoring increments batch-call counters.
- `chutoro-providers/dense/src/tests/provider.rs`
  - add parameterized parity tests comparing scalar reference distances with the
    new dense-provider batch implementation on varied dimensions and pair
    layouts.

Go/no-go:

- Do not proceed unless at least one new test fails before implementation.

### Stage B: implement SIMD kernels (green part 1)

Introduce a small kernel module for dense Euclidean distance batches with clear
scalar fallback and runtime dispatch.

Planned edits:

- `chutoro-providers/dense/src/simd/mod.rs` (new)
  - add module-level docs (`//!`) describing dispatch and safety boundaries.
  - implement typed wrappers and public kernel boundary APIs.
- `chutoro-providers/dense/src/simd/kernels.rs` (new)
  - add module-level docs (`//!`) describing dispatch and safety boundaries.
  - implement:
    - scalar kernel
    - x86 `#[target_feature(enable = "avx2")]` specialization
    - AVX-512-aware dispatch entrypoint with stable fallback semantics
    - one-time runtime selection helper using CPU feature detection.
- `chutoro-providers/dense/src/provider.rs`
  - route `distance_batch` to the new kernel module while preserving existing
    error semantics.
- `chutoro-providers/dense/src/lib.rs`
  - wire the new internal module.

Go/no-go:

- Do not proceed unless dense provider tests show numerical parity within the
  agreed epsilon and all error paths still pass.

### Stage C: make `distance_batch` the default HNSW scoring path

Wire default query-candidate scoring so HNSW benefits from `distance_batch`
specialization without broad call-site rewrites.

Planned edits:

- `chutoro-core/src/datasource.rs`
  - update default `batch_distances(query, candidates)` to build pair tuples,
    call `distance_batch`, and return collected outputs.
  - keep existing public method signatures unchanged.
- `chutoro-core/src/hnsw/validate.rs`
  - keep existing `batch_distances` call graph but add explicit output-length
    contract checks in validation paths where needed.
- `chutoro-core/src/hnsw/helpers.rs`
  - keep cache + miss resolution logic intact; verify delegated batch calls
    preserve current cache semantics.

Go/no-go:

- Do not proceed unless HNSW build/search tests pass and instrumentation
  confirms delegated `distance_batch` execution in scoring.

### Stage D: documentation, roadmap, and hardening

Finalize docs and run mandatory quality gates.

Planned edits:

- `docs/chutoro-design.md`
  - record concrete design decisions implemented for 2.2.1:
    dispatch strategy, fallback semantics, and scoring-path contract.
- `docs/roadmap.md`
  - mark `2.2.1` as done (`[x]`) once implementation and tests are complete.
- If needed, add brief operator-facing notes to `docs/developers-guide.md`
  about expected SIMD behaviour and fallback.

Go/no-go:

- Do not mark roadmap complete until all tests and gates pass.

## Concrete steps

Run from repository root (`<repo-root>`). Use `set -o pipefail` and `tee` for
every long-running command.

1. Baseline and red tests.

   ```bash
   set -o pipefail; make test 2>&1 | tee /tmp/2-2-1-prechange-test.log
   ```

   Expected excerpt:

   ```plaintext
   ... FAILED ... <new SIMD contract test name> ...
   ```

2. Implement Stage B and Stage C changes, then run focused tests.

   ```bash
   set -o pipefail; cargo test -p chutoro-core datasource:: 2>&1 | tee /tmp/2-2-1-core-datasource.log
   set -o pipefail; cargo test -p chutoro-core hnsw::tests::build:: 2>&1 | tee /tmp/2-2-1-core-hnsw-build.log
   set -o pipefail; cargo test -p chutoro-providers-dense provider:: 2>&1 | tee /tmp/2-2-1-dense-provider.log
   ```

   Expected excerpt:

   ```plaintext
   ... test result: ok. ... passed ...
   ```

3. Run formatting and Markdown gates after documentation updates.

   ```bash
   set -o pipefail; make fmt 2>&1 | tee /tmp/2-2-1-make-fmt.log
   set -o pipefail; make markdownlint 2>&1 | tee /tmp/2-2-1-make-markdownlint.log
   set -o pipefail; make nixie 2>&1 | tee /tmp/2-2-1-make-nixie.log
   ```

   Expected excerpt:

   ```plaintext
   ... markdownlint ... 0 errors ...
   ... nixie ... OK ...
   ```

4. Run required repository quality gates.

   ```bash
   set -o pipefail; make check-fmt 2>&1 | tee /tmp/2-2-1-make-check-fmt.log
   set -o pipefail; make lint 2>&1 | tee /tmp/2-2-1-make-lint.log
   set -o pipefail; make test 2>&1 | tee /tmp/2-2-1-make-test.log
   ```

   Expected excerpt:

   ```plaintext
   ... Finished ...
   ... test result: ok. ...
   ```

## Validation and acceptance

Acceptance criteria:

- Behaviour:
  - HNSW scoring uses a code path that delegates to `distance_batch` by
    default.
  - Dense provider distance batches use SIMD-capable kernels with deterministic
    fallback.
- Tests:
  - New/updated tests pass for happy, unhappy, and edge cases.
  - New contract test fails before implementation and passes after.
- Documentation:
  - `docs/chutoro-design.md` describes the chosen dispatch and fallback design.
  - `docs/roadmap.md` item `2.2.1` is marked done.
- Gates:
  - `make check-fmt`
  - `make lint`
  - `make test`

Quality method:

- Use command transcripts in `/tmp/2-2-1-*.log` as implementation evidence.
- Confirm relevant new test names appear in the passing run output.

## Idempotence and recovery

- All test and lint commands are safe to rerun.
- If formatting modifies files unexpectedly, rerun `make check-fmt` after
  `make fmt`.
- If `make test` flakes due to known timeout-sensitive benches, rerun once and
  record both logs in `Decision log` and `Surprises & discoveries`.
- If dispatch changes cause architecture-specific failures, force scalar path
  locally, keep tests green, and escalate before merging specialization.

## Artifacts and notes

During implementation, retain concise evidence snippets from:

- `/tmp/2-2-1-prechange-test.log`
- `/tmp/2-2-1-core-datasource.log`
- `/tmp/2-2-1-core-hnsw-build.log`
- `/tmp/2-2-1-dense-provider.log`
- `/tmp/2-2-1-make-check-fmt.log`
- `/tmp/2-2-1-make-lint.log`
- `/tmp/2-2-1-make-test.log`

When this plan is revised, append key evidence and decision updates here.

## Interfaces and dependencies

Public interfaces that must remain stable:

- `chutoro_core::DataSource`
  - `fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError>`
  - `fn batch_distances(&self, query: usize, candidates: &[usize])`
    `-> Result<Vec<f32>, DataSourceError>`
  - `fn distance_batch(&self, pairs: &[(usize, usize)], out: &mut [f32])`
    `-> Result<(), DataSourceError>`

Planned internal interfaces:

- `chutoro-providers/dense/src/simd/mod.rs`
  - `pub(crate)` typed boundary entry points for Euclidean batch distance
    computation.
- `chutoro-providers/dense/src/simd/kernels.rs`
  - internal scalar and SIMD kernel implementations used by `simd/mod.rs`.
- `chutoro-providers/dense/src/provider.rs`
  - `distance_batch` delegates to kernel entry point and preserves existing
    error contracts.
- `chutoro-core/src/datasource.rs`
  - default `batch_distances` delegates to `distance_batch`.

No new crate dependencies are planned.

## Revision note

Initial draft created on 2026-03-02 to implement roadmap task `2.2.1` with
explicit staged validation and quality-gate requirements.

Implementation update on 2026-03-02:

- Marked status `COMPLETE` and updated all stage checkpoints.
- Documented stable-toolchain constraints (`portable_simd` still nightly
  only)[^1] and AVX-512 stabilization (`rust-lang/rust#111137`)[^3] with
  updated minimum supported Rust version (MSRV).
- Recorded outcomes, validation evidence, and roadmap/design doc updates.

[^1]: <https://github.com/rust-lang/rust/issues/86656>
[^2]:
    [Rust 1.89.0 release notes](https://blog.rust-lang.org/2025/08/07/Rust-1.89.0/)
[^3]: <https://github.com/rust-lang/rust/issues/111137>
