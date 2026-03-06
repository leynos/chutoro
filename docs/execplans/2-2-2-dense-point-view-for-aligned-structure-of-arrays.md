# Execution plan (ExecPlan): roadmap 2.2.2 introduce `DensePointView<'a>` for aligned Structure of Arrays (SoA) access with a scalar fallback

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & discoveries`, `Decision log`, and
`Outcomes & retrospective` must be kept up to date as work proceeds.

Status: DRAFT

## Purpose / big picture

Implement roadmap item `2.2.2` by introducing an internal `DensePointView<'a>`
abstraction for dense numeric providers. The new type must present aligned
Structure of Arrays (SoA) data to CPU distance kernels while retaining the
current scalar fallback semantics when SoA packing or SIMD use is not
applicable.

Success is observable when:

- dense Euclidean batch scoring can obtain aligned, zero-padded, stride-1 SoA
  views through `DensePointView<'a>` instead of operating only on raw row-major
  slices;
- the dense provider still returns the same distances and the same
  `DataSourceError` variants for happy paths, unhappy paths, and edge cases;
- unit tests, using `rstest` parameterization where repetition exists, cover
  SoA packing, alignment guarantees, zero-padding, scalar fallback, output
  preservation on error, and out-of-bounds handling;
- `docs/chutoro-design.md` records the final `DensePointView<'a>` design
  decisions in §6.3 once implementation is complete;
- `docs/roadmap.md` marks item `2.2.2` done only after implementation,
  documentation, and quality gates are complete;
- quality gates pass: `make check-fmt`, `make lint`, and `make test`.

## Constraints

- Keep Rust source files under 400 lines. Split modules or test files rather
  than extending an existing SIMD file past that limit.
- Preserve public `DataSource` signatures and the public surface of
  `DenseMatrixProvider`; `DensePointView<'a>` is an internal dense-provider
  detail.
- Do not add new crate dependencies. Use the existing standard library,
  workspace crates, and current SIMD implementation approach.
- Preserve the current scalar fallback contract for non-x86 targets, disabled
  CPU features, empty batches, and error cases.
- Preserve all-or-nothing writes for batch distance outputs: on failure, the
  caller-provided output buffer must remain unchanged.
- Keep current dense-provider storage ingestion semantics unchanged: null rows,
  null values, ragged inputs, and invalid column types must still be rejected
  exactly as they are now.
- Use `rstest` parameterized cases for repeated shape- and dimension-based
  coverage.
- Follow guidance from:
  - `docs/chutoro-design.md` (especially §6.3)
  - `docs/property-testing-design.md`
  - `docs/complexity-antipatterns-and-refactoring-strategies.md`
  - `docs/rust-testing-with-rstest-fixtures.md`
  - `docs/rust-doctest-dry-guide.md`

## Tolerances (exception triggers)

- Scope: if the work requires edits in more than 12 files or more than 900 net
  lines, stop and escalate before continuing.
- Interface: if making `DensePointView<'a>` useful appears to require a public
  API change outside `chutoro-providers/dense`, stop and escalate with options.
- Layout: if 64-byte alignment and lane-multiple padding cannot be delivered
  without allocator or ownership changes beyond dense-provider internals, stop
  and escalate.
- Backends: if implementation forces roadmap item `2.2.3` feature gating to be
  done early, stop and escalate rather than silently widening scope.
- Behaviour: if SoA packing changes floating-point results beyond a documented
  `1.0e-6_f32` tolerance for existing Euclidean cases, stop and investigate
  before proceeding.
- Iterations: if `make lint` or `make test` fails after 3 repair attempts,
  stop and escalate with captured logs.

## Risks

- Risk: SoA packing may accidentally reorder coordinates or candidate rows,
  producing numerically plausible but incorrect distances. Severity: high.
  Likelihood: medium. Mitigation: add explicit packing-layout tests that check
  column-major lane contents and zero-padded tails before running kernel parity
  checks.

- Risk: alignment guarantees may be asserted but not actually enforced by the
  backing buffer used for SoA packing. Severity: high. Likelihood: medium.
  Mitigation: test pointer alignment directly and keep alignment logic in one
  constructor instead of scattering it across provider and kernel code.

- Risk: introducing `DensePointView<'a>` into the hot path can make the SIMD
  boundary more complex and reintroduce a bumpy-road structure in
  `simd/mod.rs`. Severity: medium. Likelihood: high. Mitigation: isolate the
  view type, packing helpers, and tests into dedicated modules with small,
  single-purpose functions.

- Risk: error handling may partially fill an SoA scratch buffer or output
  vector before an out-of-bounds pair is detected. Severity: high. Likelihood:
  medium. Mitigation: validate indices before writing outputs and continue to
  compute batch results into a temporary buffer before copying into caller
  storage.

- Risk: the current row-major path may still be faster for very small batches,
  making unconditional SoA packing a regression. Severity: medium. Likelihood:
  medium. Mitigation: keep an explicit scalar fallback path and document the
  threshold or conditions that choose it.

## Progress

- [x] (2026-03-06 00:00Z) Drafted ExecPlan for roadmap item `2.2.2`.
- [ ] Stage A complete: confirm current dense SIMD and provider contracts with
  failing tests for SoA view requirements.
- [ ] Stage B complete: add `DensePointView<'a>` storage and packing helpers
  with alignment and zero-padding guarantees.
- [ ] Stage C complete: integrate `DensePointView<'a>` into dense-provider
  batch scoring while preserving scalar fallback and current error semantics.
- [ ] Stage D complete: update `docs/chutoro-design.md`, mark roadmap item
  `2.2.2` done, and pass `make check-fmt`, `make lint`, and `make test`.

## Surprises & discoveries

- Observation: roadmap item `2.2.1` is already complete, but the dense SIMD
  kernels still consume row-major `&[f32]` slices through `RowSlice<'a>` and
  `RowMajorMatrix<'a>`. Evidence: `chutoro-providers/dense/src/provider.rs`,
  `chutoro-providers/dense/src/simd/mod.rs`, and
  `chutoro-providers/dense/src/simd/kernels.rs`. Impact: `2.2.2` is primarily a
  layout-and-boundary refactor plus fallback policy work, not a first-time
  kernel introduction.

- Observation: §6.3 requires more than "some internal helper"; it calls for an
  internal `DensePointView<'a>` with SoA packing, stride-1 access, 64-byte
  alignment, and lane-multiple zero padding. Evidence: `docs/chutoro-design.md`
  §6.3. Impact: implementation must make these guarantees explicit and tested,
  not implied by comments.

- Observation: the current dense provider already preserves all-or-nothing
  output semantics by computing results into a temporary `Vec<f32>` before
  copying into the caller buffer. Evidence:
  `chutoro-providers/dense/src/simd/mod.rs::collect_euclidean_distance_batch`
  and `chutoro-core/src/datasource/tests/batch_first_source.rs`. Impact: this
  behaviour must remain intact after introducing SoA scratch storage.

- Observation: there is existing parameterized parity coverage for odd
  dimensions and AVX tail cases in
  `chutoro-providers/dense/src/tests/provider.rs`, but no tests yet for SoA
  layout contents, padding, or alignment. Impact: Stage A must add contract
  tests that fail before implementation.

## Decision log

- Decision: keep `DensePointView<'a>` internal to
  `chutoro-providers/dense/src/simd/` unless implementation proves the type is
  useful outside dense-provider internals. Rationale: the roadmap item is about
  internal SoA access preconditions, not expanding the public API surface.
  Date/Author: 2026-03-06 / Codex.

- Decision: treat scalar fallback as a first-class part of the feature, not
  merely an unsupported-platform escape hatch. Rationale: §6.3 explicitly
  requires scalar fallback, and small or irregular batches may still be better
  served by the current row-major scalar path. Date/Author: 2026-03-06 / Codex.

- Decision: record the final `DensePointView<'a>` shape, alignment guarantee,
  zero-padding rule, and fallback-selection policy in `docs/chutoro-design.md`
  during implementation rather than only in this plan. Rationale: the roadmap
  and design document must stay synchronized once the code lands. Date/Author:
  2026-03-06 / Codex.

## Outcomes & retrospective

No implementation has been executed yet. Completion for this plan means:

- `DensePointView<'a>` exists as a documented internal abstraction with tests
  proving its SoA packing, 64-byte alignment, and zero-padded tail behaviour.
- Dense-provider batch scoring uses `DensePointView<'a>` when the SoA path is
  applicable and falls back to the current scalar-compatible path otherwise.
- Existing distance and ingestion behaviour remains unchanged from the
  perspective of `DenseMatrixProvider` callers.
- `docs/chutoro-design.md` and `docs/roadmap.md` reflect the shipped design.
- `make check-fmt`, `make lint`, and `make test` all pass.

Retrospective notes must be added here during implementation after the final
approach and any trade-offs are known.

## Context and orientation

Relevant current files and behaviour:

- `chutoro-providers/dense/src/provider.rs`
  - owns the row-major `Vec<f32>` backing store;
  - exposes `distance(...)` via `row_slice(...)`;
  - exposes `distance_batch(...)` through
    `simd::euclidean_distance_batch_raw_pairs(...)`.
- `chutoro-providers/dense/src/simd/mod.rs`
  - currently defines row-major wrappers such as `RowSlice<'a>`,
    `RowMajorMatrix<'a>`, `MatrixValues<'a>`, and `DistanceBuffer<'a>`;
  - validates pair/output length matches and out-of-bounds row access;
  - computes batch results into a temporary vector before copying into the
    caller output slice.
- `chutoro-providers/dense/src/simd/kernels.rs`
  - currently contains scalar, AVX2, and AVX-512 Euclidean kernels operating
    on raw contiguous row slices;
  - uses one-time runtime kernel selection with `OnceLock`.
- `chutoro-providers/dense/src/simd/tests.rs`
  - covers basic kernel parity, length mismatches, and output preservation on
    error, but not SoA packing invariants.
- `chutoro-providers/dense/src/tests/provider.rs`
  - covers provider-level parity across odd dimensions and lane-tail cases.

Definitions used in this plan:

- Structure of Arrays (SoA): storing values by coordinate lane across many
  points, rather than keeping each point's full row contiguous.
- Alignment: placing the start of a buffer at a byte boundary such as 64 bytes
  so wide loads are safe and efficient.
- Zero padding: extending a packed SoA buffer to a lane multiple and filling
  unused tail entries with `0.0_f32` so kernels can read whole lanes without
  bounds checks.
- Scalar fallback: the non-SoA, non-SIMD-compatible distance path retained for
  correctness and portability.

## Plan of work

### Stage A: establish failing tests and the desired contract (red)

Add tests that describe the intended `DensePointView<'a>` behaviour before any
production code changes.

Planned edits:

- `chutoro-providers/dense/src/simd/tests.rs`
  - add parameterized `rstest` coverage that checks:
    - SoA packing order for small hand-verified matrices;
    - zero-padding for dimensions or candidate counts that are not lane
      multiples;
    - pointer alignment for packed storage;
    - scalar fallback selection for empty, one-point, or otherwise unsuitable
      packing cases.
- `chutoro-providers/dense/src/tests/provider.rs`
  - add provider-level cases proving SoA-backed distance batches match scalar
    references across:
    - odd dimensions;
    - batch sizes smaller than one SIMD lane;
    - repeated candidate rows;
    - empty batches;
    - out-of-bounds pairs that must leave caller output unchanged.

Go/no-go:

- Do not proceed unless at least one new test fails against the current
  row-major-only implementation.

### Stage B: add `DensePointView<'a>` and packing helpers (green part 1)

Introduce a dedicated SoA view type with explicit storage ownership and tested
invariants.

Planned edits:

- `chutoro-providers/dense/src/simd/point_view.rs` (new, preferred) or a
  similarly named dedicated module
  - add module-level docs (`//!`) explaining what `DensePointView<'a>` owns or
    borrows, when it is used, and what alignment/padding guarantees it makes;
  - implement:
    - `DensePointView<'a>`;
    - a small typed wrapper for aligned packed storage if needed;
    - constructors that validate row indices and pack row-major dense data into
      SoA order;
    - helpers exposing logical row count, padded lane count, dimension, and
      aligned coordinate blocks.
- `chutoro-providers/dense/src/simd/mod.rs`
  - re-export or wire the new module internally;
  - keep boundary types small and avoid turning `mod.rs` into a mixed packing +
    dispatch + kernel orchestration file.

Implementation notes:

- Prefer separating ownership from view logic if it keeps borrowing simple.
- Validate indices before writing packed output buffers.
- Keep padding deterministic: unused packed entries must always be
  `0.0_f32`.

Go/no-go:

- Do not proceed unless packing and alignment tests pass and no existing kernel
  parity tests regress.

### Stage C: integrate `DensePointView<'a>` into scoring with scalar fallback (green part 2)

Use the new view in dense-provider batch scoring while preserving current error
and output contracts.

Planned edits:

- `chutoro-providers/dense/src/provider.rs`
  - build `DensePointView<'a>` for batched candidate scoring;
  - continue to reject output-length mismatches up front;
  - preserve all-or-nothing writes to the caller output buffer.
- `chutoro-providers/dense/src/simd/mod.rs`
  - add a batch entrypoint that can choose between:
    - the SoA-backed hot path; and
    - the current scalar-compatible path for fallback cases.
- `chutoro-providers/dense/src/simd/kernels.rs`
  - adapt kernels only as needed to consume SoA-packed blocks or helper access
    methods from `DensePointView<'a>`;
  - keep raw-kernel complexity contained in this file rather than leaking
    primitive indexing back into the provider.

Go/no-go:

- Do not proceed unless provider-level parity, out-of-bounds, and output
  preservation tests all pass.

### Stage D: document the final design and close the roadmap item

Record what shipped and validate the whole repository.

Planned edits:

- `docs/chutoro-design.md`
  - update §6.3 with the final `DensePointView<'a>` design:
    - where the type lives;
    - what "aligned" means in the implementation;
    - how zero-padding works;
    - when scalar fallback is selected.
- `docs/roadmap.md`
  - mark `2.2.2` done (`[x]`) once the implementation and tests are complete.

Validation commands:

```sh
set -o pipefail; make fmt 2>&1 | tee /tmp/2-2-2-make-fmt.log
set -o pipefail; make markdownlint 2>&1 | tee /tmp/2-2-2-make-markdownlint.log
set -o pipefail; make nixie 2>&1 | tee /tmp/2-2-2-make-nixie.log
set -o pipefail; make check-fmt 2>&1 | tee /tmp/2-2-2-make-check-fmt.log
set -o pipefail; make lint 2>&1 | tee /tmp/2-2-2-make-lint.log
set -o pipefail; make test 2>&1 | tee /tmp/2-2-2-make-test.log
```

Expected success signals:

- `make check-fmt` exits `0` with no formatting diffs required.
- `make lint` exits `0` with no Clippy warnings.
- `make test` exits `0` and includes the new dense SoA tests in the summary.

If the local default `make test` profile stalls in a known nextest edge case,
use the captured log to identify whether the stall is unrelated. Do not mark
the work complete until a plain `make test` run succeeds.

## Acceptance evidence to capture during implementation

Add concise evidence to this document as the work progresses:

```plaintext
- Red stage:
  - cargo test -p chutoro-providers-dense simd::tests::<new failing test name>
    ... FAILED ... expected DensePointView alignment/padding contract ...

- Green stage:
  - cargo test -p chutoro-providers-dense simd::tests::
    ... ok
  - cargo test -p chutoro-providers-dense tests::provider::
    ... ok

- Final gates:
  - make check-fmt ... ok
  - make lint ... ok
  - make test ... ok
```

## Proposed implementation order

1. Add red tests for packing order, zero padding, alignment, and fallback.
2. Introduce `DensePointView<'a>` in its own module with a minimal typed API.
3. Wire provider batch scoring through the new view and keep temporary-buffer
   output semantics.
4. Update §6.3 in `docs/chutoro-design.md`.
5. Mark roadmap item `2.2.2` done only after all gates pass.

## Approval gate

This file is the draft phase required by the `execplans` workflow. Do not begin
implementation from this plan until the user explicitly approves it or requests
specific revisions.
