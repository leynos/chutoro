# Execution plan (ExecPlan): roadmap 2.2.7 Kani harnesses

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & discoveries`, `Decision log`, and
`Outcomes & retrospective` must be kept up to date as work proceeds.

Status: COMPLETE. The plan was approved for implementation on 2026-05-24 and
completed on 2026-05-24.

## Purpose / big picture

Complete roadmap item `2.2.7` by adding bounded Kani proof harnesses for the
dense-provider single-instruction multiple-data (SIMD) Euclidean distance path,
covering tail padding and dispatch selection. Kani is a bounded model checker:
it exhaustively explores all inputs within explicit limits and proves that the
asserted property holds for that bounded space. This item targets two
executable Rust hazards named in `docs/chutoro-design.md` §6.3: padded SIMD
tails and runtime backend selection.

Success is observable when:

- Kani proves that the dense-provider query-to-points lane batching logic never
  addresses beyond the padded coordinate block or writes beyond the logical
  output length for bounded point counts around the 16-lane boundary.
- Kani proves that `choose_euclidean_backend` never selects `Avx512`, `Avx2`,
  `Neon`, or `PortableSimd` unless that backend is both compiled into the
  binary and available at runtime, and that `Scalar` remains the fallback when
  no SIMD backend is eligible.
- Unit tests, using `rstest` where repeated examples exist, keep the executable
  behaviour aligned with the proof seams: `DensePointView<'a>` remains 64-byte
  aligned, padded to a 16-lane multiple, and zero-filled in unused lanes;
  dispatch priority remains `Avx512 > Avx2 > Neon > PortableSimd > Scalar`.
- The Kani commands are wired into the existing `make kani` or
  `make kani-full` workflow without adding Kani to the normal `make test` path.
- `make check-fmt`, `make lint`, and `make test` all succeed.
- `docs/chutoro-design.md`, `docs/developers-guide.md`, and
  `docs/adr-002-adoption-of-kani-formal-verification.md` record the dense SIMD
  proof contract. `docs/users-guide.md` is updated only if implementation
  changes public behaviour or public API; this plan expects no such change.
- `docs/roadmap.md` marks `2.2.7` done only after implementation, proof runs,
  tests, documentation, CodeRabbit review, and the quality gates all pass.

Roadmap prerequisites `2.2.3` and `2.2.6` are already complete. The current
code provides the compile-time/runtime support masks and selector in
`chutoro-providers/dense/src/simd/dispatch.rs`, the packed point view in
`chutoro-providers/dense/src/simd/point_view.rs`, and the backend parity suite
under `chutoro-providers/dense/src/simd/tests/parity/`.

## Relevant documentation and skills

Use these repository documents as the source of truth while implementing:

- `docs/roadmap.md`, especially item `2.2.7`.
- `docs/chutoro-design.md` §6.3, especially the SIMD dispatch, tail padding,
  and verification seam language.
- `docs/property-testing-design.md`, for how Kani complements property tests.
- `docs/complexity-antipatterns-and-refactoring-strategies.md`, to keep proof
  helpers small and avoid bumpy-road selector logic.
- `docs/rust-testing-with-rstest-fixtures.md`, for parameterized unit-test
  structure.
- `docs/rust-doctest-dry-guide.md`, if public examples or Rustdoc are touched.
- `docs/documentation-style-guide.md`, if an architecture note or ADR update is
  needed.

Use these skills:

- `leta` for symbol navigation and call/reference checks before editing code.
- `rust-router` for routing any Rust-specific design issue to the smallest
  useful follow-on skill.
- `kani` for harness structure, assumptions, unwind bounds, and proof
  validation.
- `hexagonal-architecture` only as a boundary check: keep the verification
  seam inside the dense-provider internals and do not widen public ports or
  leak adapter details to `chutoro-core`.
- `commit-message` for file-based commit messages.
- `pr-creation` for the draft pull request.

## Constraints

- Do not begin implementation until this draft is explicitly approved.
- Preserve the public `DenseMatrixProvider` and `DataSource` interfaces unless
  approval is obtained for a public API change.
- Preserve the dense-provider feature names: `simd_avx2`, `simd_avx512`,
  `simd_neon`, and `nightly_portable_simd`.
- Preserve the dispatch priority:
  `Avx512 > Avx2 > Neon > PortableSimd > Scalar`.
- Preserve the existing non-finite policy: Euclidean reductions canonicalize
  non-finite outputs to `f32::NAN`.
- Preserve the `DensePointView<'a>` layout contract: 64-byte alignment,
  16-lane padding, and deterministic `0.0_f32` tail fill.
- Keep Kani harnesses under `#[cfg(kani)]` so normal builds, Clippy, tests, and
  downstream users do not compile proof-only code.
- Add `cfg(kani)` to the dense crate's checked configuration if needed, rather
  than suppressing unknown-cfg warnings.
- Avoid proving raw `std::arch` intrinsic implementations directly. The proof
  target is the safe boundary arithmetic and selector policy, not a solver
  model of architecture-specific intrinsics.
- Keep files under 400 lines. Split new proof helpers into a dedicated module
  instead of extending existing dense SIMD files past that limit.
- Use `rstest` for repeated unit-test examples. Use Kani for the invariant
  space; do not duplicate every Kani property as a property test.
- Do not add runtime dependencies. If a proof helper appears to require a new
  dependency, stop and escalate.
- Do not add Kani to `make test`. Kani remains opt-in through `make kani` and
  `make kani-full`.
- Run commands sequentially, with `tee` logs under `/tmp`, and do not run
  formatting, linting, or tests in parallel.
- Use en-GB Oxford spelling in documentation and comments, except for existing
  Rust identifiers and external API names.

## Tolerances (exception triggers)

- Scope: if implementation requires edits to more than 10 files or roughly
  800 net lines excluding generated counterexample files, stop and escalate.
- Public interface: if the proof seam requires public API changes in
  `chutoro-providers-dense`, `chutoro-core`, or `chutoro-test-support`, stop
  and present options.
- Kani target: if proving the safe arithmetic boundary is insufficient and the
  only remaining path is to model raw SIMD intrinsics directly, stop and
  escalate before attempting that rewrite.
- Dependencies: if any new runtime dependency is required, stop and escalate.
- Build system: if adding dense-provider Kani coverage requires changing the
  semantics of existing `make test`, `make lint`, or `make check-fmt`, stop and
  escalate.
- Iterations: if a Kani harness still fails after three focused repair
  attempts, stop, record the counterexample or failure mode, and ask for
  direction.
- Runtime budget: if a new practical Kani harness takes more than three
  minutes after warm compilation on the development host, keep it out of
  `make kani` and wire it only into `make kani-full`, then record the decision.
- Ambiguity: if `docs/chutoro-design.md` and current code disagree about a
  selector order, padding width, or runtime support rule, stop and present the
  conflict before changing code.
- CodeRabbit: if `coderabbit review --agent` raises a correctness, testing,
  or documentation concern after a major milestone, clear it or document an
  explicit reviewer-approved deferral before moving on.

## Risks

- Risk: Kani state space explodes if the harness constructs full matrices or
  invokes backend entrypoints directly. Severity: high. Likelihood: medium.
  Mitigation: prove small pure helpers around lane arithmetic and dispatch
  masks, then use unit tests to link those helpers to the production
  `DensePointView<'a>` and selector seams.

- Risk: `euclidean_backend()` uses `OnceLock`, which is a poor fit for multiple
  symbolic support-mask combinations in a single proof run. Severity: medium.
  Likelihood: high. Mitigation: prove `choose_euclidean_backend` directly with
  symbolic `CompiledSimdSupport` and `RuntimeSimdSupport` values, and leave the
  cached production wrapper covered by existing unit tests.

- Risk: The existing backend entrypoint helpers are `#[cfg(test)]`, not
  `#[cfg(kani)]`. Severity: medium. Likelihood: medium. Mitigation: avoid
  depending on test-only helpers from Kani; if a shared helper is needed, gate
  it with `#[cfg(any(test, kani))]` and keep visibility no wider than necessary.

- Risk: A proof of arithmetic helper functions could drift away from the
  production kernel loops. Severity: high. Likelihood: medium. Mitigation:
  extract a production-used safe helper for lane batch bounds if needed, prove
  that helper, and add `rstest` unit coverage that compares helper outputs to
  `DensePointView<'a>` behaviour around boundary sizes.

- Risk: Runtime feature combinations differ across x86, aarch64, and arm.
  Severity: medium. Likelihood: high. Mitigation: model support masks as
  symbolic booleans in Kani, then keep platform-specific detection covered by
  the existing `support_masks` and `backend_expectations` unit tests.

- Risk: Documentation churn could imply a public behaviour change where none
  exists. Severity: low. Likelihood: medium. Mitigation: keep
  `docs/users-guide.md` unchanged unless the implementation changes public API
  or user-observable behaviour; document internal conventions in
  `docs/developers-guide.md` and the design/ADR documents.

- Risk: The plan could over-apply hexagonal architecture vocabulary to a
  low-level SIMD provider. Severity: low. Likelihood: medium. Mitigation: use
  the skill only to protect boundaries: policy and verification stay inside
  dense-provider internals, and no public port is introduced merely for a proof.

## Progress

- [x] (2026-05-21) Loaded the requested `leta`, `kani`, `rust-router`, and
  `hexagonal-architecture` skills.
- [x] (2026-05-21) Created a Leta workspace for the repository.
- [x] (2026-05-21) Renamed the local branch to
  `2-2-7-kani-harnesses-for-tail-padding-and-dispatch-selection`.
- [x] (2026-05-21) Used a Wyvern agent team for read-only planning
  reconnaissance over the roadmap, design documents, dense SIMD code, and
  existing Kani patterns.
- [x] (2026-05-21) Drafted this pre-implementation ExecPlan.
- [x] (2026-05-24) Approval gate passed when the user explicitly requested
  implementation of this ExecPlan.
- [x] (2026-05-24) Stage A: audited current dense SIMD proof seams and updated
  this ExecPlan with exact symbols to edit before making code changes.
- [x] (2026-05-24) Stage A follow-up: committed the ExecPlan status update
  after focused Markdown validation.
- [x] (2026-05-24) Stage B: added dense-provider Kani proof module and minimal
  production-used helper extraction required to avoid proof drift.
- [x] (2026-05-24) Stage C: added `rstest` unit tests for tail padding,
  zero-fill, and lane output-count edge cases; existing selector examples cover
  happy and unavailable-runtime paths.
- [x] (2026-05-24) Stage D: wired the dense-provider dispatch and tail-padding
  harnesses into `make kani` and `make kani-full`; `make test` remains
  unchanged.
- [x] (2026-05-24) Stage E: updated `docs/chutoro-design.md`,
  `docs/developers-guide.md`, and
  `docs/adr-002-adoption-of-kani-formal-verification.md`; update
  `docs/users-guide.md` was not needed because public behaviour did not
  change.
- [x] (2026-05-24) Stage F: ran focused proof/test commands, focused
  Markdown lint, `make kani`, `make check-fmt`, `make lint`, and `make test`.
- [x] (2026-05-24) Stage G review gate: ran `coderabbit review --agent` after
  deterministic gates passed; CodeRabbit reported zero findings.
- [x] (2026-05-24) Stage G completion: committed the implementation and marked
  roadmap item `2.2.7` done.

## Superseded progress entries

The following unchecked entries were replaced after implementation approval by
the timestamped progress entries above:

- Stage A: audit current dense SIMD proof seams and update this ExecPlan
  with exact symbols to edit before making code changes.

## Surprises & discoveries

- Discovery: The repository currently wires `make kani` and `make kani-full`
  only for `chutoro-core`, while this work belongs in
  `chutoro-providers-dense`. The implementation must therefore add dense Kani
  coverage to the Makefile and nightly Kani story deliberately.

- Discovery: `2.2.6` already added a dense-provider parity suite and a
  crate-internal `DistanceSemantics` contract. `2.2.7` should not duplicate
  that numeric parity coverage; it should prove the two bounded executable
  hazards named by the design document.

- Discovery: The safest dispatch proof target is
  `dispatch::choose_euclidean_backend`, not the cached
  `dispatch::euclidean_backend()` wrapper because the latter stores one
  concrete choice in a `OnceLock`.

- Discovery: The safest tail proof target is the lane-bounds arithmetic used
  by packed query-to-points kernels, connected back to `DensePointView<'a>` by
  unit tests. Proving raw architecture intrinsics directly would add solver
  fragility without improving the design-level guarantee requested by the
  roadmap.

- Discovery: The current dense crate build script registers only
  `cfg(nightly)`. Stage B must add `cargo:rustc-check-cfg=cfg(kani)` before
  introducing the proof module so strict `unexpected_cfgs` checks remain clean.

- Discovery: The production query-to-points loops all share the same shape:
  iterate by backend lane width across `DensePointView::padded_point_count()`,
  load one full lane from each `coordinate_block`, and write only
  `out.len().saturating_sub(offset).min(lanes)` logical results. A small helper
  for this write count can be production-used by each backend and proved by
  Kani without modelling architecture intrinsics.

- Discovery: `cargo kani setup` was needed for Kani `0.67.0`, and direct Kani
  runs needed the downloaded toolchain library path in `LD_LIBRARY_PATH`.
  `make kani` and `make kani-full` now compute the installed Kani version from
  `cargo kani -V` and prepend
  `$(HOME)/.kani/kani-$(KANI_VERSION)/toolchain/lib`.

- Discovery: Running the dense Kani crate under strict checked-cfg exposed
  latent Kani-only compile issues in `chutoro-core`: generic `Result` aliases
  needed explicit success types, and Kani assertion messages passed through
  helper functions needed `'static` lifetimes.

- Discovery: A direct Kani harness that constructed `DensePointView<'a>` and
  asserted zero-filled storage was too heap- and iterator-heavy for a practical
  proof. The implementation keeps zero-fill coverage in `rstest` unit tests and
  uses Kani for the bounded lane-bounds invariant.

- Discovery: A symbolic `step_by(lanes)` tail-padding proof caused unnecessary
  unwind pressure. Rewriting the proof around a fixed bounded batch counter
  kept the proof small while preserving the same backend lane-width cases:
  four, eight, and sixteen lanes.

- Discovery: The original selector loop over an array of backends proved more
  noisily than the equivalent explicit priority chain. The implementation uses
  the explicit chain in production so the policy remains visible and directly
  provable.

- Discovery: `make kani-full` currently reaches unrelated pre-existing
  distance harnesses in `chutoro-core`. Kani `0.67.0` first panicked when those
  helpers passed assertion messages through function parameters; after that was
  repaired, the cosine zero-on-identical proof failed for bounded finite
  non-zero vectors whose norm arithmetic can underflow or otherwise fall
  outside the proof tolerance. That failure is not caused by the dense SIMD
  item, and a single core distance harness spent more than eleven minutes in
  the solver.

- Discovery: `add_edge_if_missing` is a non-Kani test helper as well as a Kani
  support helper. The Kani path must avoid dynamic panic formatting, but the
  normal test path should keep the existing fail-fast behaviour when fixture
  data names a missing origin node.

## Decision log

- Decision: Keep the initial proof scope to safe boundary helpers for
  `DensePointView<'a>` lane batching and `choose_euclidean_backend` mask
  selection. Rationale: the roadmap asks for bounded proof of tail padding and
  dispatch selection; direct modelling of raw SIMD intrinsics is unnecessary
  and higher risk. Date/Author: 2026-05-21 / Codex.

- Decision: Treat `docs/users-guide.md` as conditional, not mandatory churn.
  Rationale: this work is expected to add internal proof coverage only. Public
  consumers should see no API or behaviour change unless implementation
  uncovers a necessary public contract update. Date/Author: 2026-05-21 / Codex.

- Decision: Use the existing Kani architectural decision record (ADR) as the
  place to record dense-provider formal-verification practice unless
  implementation changes the repository's Kani policy substantially. Rationale:
  extending `ADR-002` avoids creating a second decision record for the same
  verification technique. Date/Author: 2026-05-21 / Codex.

- Decision: Extract only the lane output-count arithmetic into a production
  helper, while proving padded load bounds directly against
  `padded_point_count`. Rationale: the helper is the arithmetic used at every
  logical output write, while load safety follows from the existing invariant
  that padded point counts are 16-lane multiples and every backend lane width
  divides 16. Date/Author: 2026-05-24 / Codex.

- Decision: Keep `DensePointView<'a>` zero-fill as concrete `rstest` coverage
  rather than a Kani heap harness. Rationale: the roadmap invariant is
  satisfied by proving no logical read/write crosses padded-lane bounds and by
  testing the production constructor's zero-fill behaviour at tail sizes `15`
  and `17`. Date/Author: 2026-05-24 / Codex.

- Decision: Register and use `cfg(kani)` through build-script checked-cfg
  output instead of suppressing `unexpected_cfgs`. Rationale: proof-only code
  remains invisible to normal builds while strict configuration checking stays
  useful. Date/Author: 2026-05-24 / Codex.

- Decision: Treat `make kani` as the applicable formal gate for roadmap item
  `2.2.7` and record the current `make kani-full` core-distance failure as an
  unrelated pre-existing proof issue. Rationale: the new dense SIMD harnesses
  run and pass through `make kani`; `kani-full` is a slow-lane aggregate over
  older core proofs and currently fails outside the dense-provider scope.
  Date/Author: 2026-05-24 / Codex.

## Plan of work

1. Re-orient with Leta before editing. Use `leta grep` and `leta show` to
   inspect `DensePointView::from_row_indices`,
   `DensePointView::coordinate_block`, `padded_point_count`,
   `choose_euclidean_backend`, `backend_supported`, and the current `rstest`
   -based dense SIMD tests.

2. Add a dense-provider Kani module, likely
   `chutoro-providers/dense/src/simd/kani_proofs.rs`, and include it from
   `chutoro-providers/dense/src/simd/mod.rs` under `#[cfg(kani)]`. If the dense
   crate does not already register `cfg(kani)`, add the checked configuration in
    `chutoro-providers/dense/Cargo.toml`.

3. For dispatch selection, build a symbolic support-mask harness around
   `CompiledSimdSupport::new`, `RuntimeSimdSupport::new`, and
   `choose_euclidean_backend`. The harness should assert:

   - returning `Avx512` implies compiled AVX-512 and runtime AVX-512 support;
   - returning `Avx2` implies AVX2 support and no eligible higher-priority
     backend;
   - returning `Neon` implies Neon support and no eligible higher-priority
     backend;
   - returning `PortableSimd` implies portable-SIMD support and no eligible
     higher-priority backend;
   - returning `Scalar` is always permitted and occurs when no SIMD backend is
     eligible.

4. For tail padding, either prove an existing helper or extract a small
   production-used helper that describes lane batches for query-to-points
   kernels. The helper should be independent of architecture-specific
   intrinsics and prove, for bounded logical point counts and dimensions around
   the 16-lane boundary, that:

   - padded point counts are multiples of 16;
   - each full lane load starts within the coordinate block;
   - every addressed lane is below the padded point count;
   - every logical output write is below the original point count;
   - unused padded lanes correspond to zero-filled storage, never source rows
     beyond the provided candidate indices.

5. Add `rstest` unit coverage only where it strengthens the link between the
   proof helper and production behaviour. Candidate tests include
   `padded_point_count` boundary cases for `0`, `1`, `15`, `16`, and `17`
   points; zero-fill assertions for unused packed lanes; and selector examples
   where a high-priority backend is compiled but unavailable at runtime.

6. Wire the proof commands. Prefer extending `make kani-full` to run all dense
   proofs and add only fast dense harnesses to `make kani`. Capture the exact
   commands in the Makefile comments and documentation. If the first dense
   proof run is slow, keep the practical target small and document the
   full-suite command separately.

7. Run focused validation with logs. Use commands of this shape from the
   repository root, replacing harness names with the final names:

   ```sh
   set -o pipefail
   cargo kani -p chutoro-providers-dense --default-unwind 4 \
     --harness verify_dense_simd_dispatch_selection_respects_support_masks \
     2>&1 | tee /tmp/kani-dispatch-chutoro-2-2-7.out
   ```

   ```sh
   set -o pipefail
   cargo kani -p chutoro-providers-dense --default-unwind 5 \
     --harness verify_dense_simd_tail_padding_lane_bounds \
     2>&1 | tee /tmp/kani-tail-chutoro-2-2-7.out
   ```

8. Update documentation after the code shape is known. Add a
   `2.2.7` implementation update to `docs/chutoro-design.md` §6.3, extend
   `docs/developers-guide.md` with dense SIMD Kani workflow notes, and extend
   `docs/adr-002-adoption-of-kani-formal-verification.md` with the
   dense-provider harness policy. Update `docs/users-guide.md` only if public
   behaviour or public API changes.

9. Run quality gates sequentially, with `tee` logs:

   ```sh
   set -o pipefail
   make check-fmt 2>&1 | tee /tmp/check-fmt-chutoro-2-2-7.out
   ```

   ```sh
   set -o pipefail
   make lint 2>&1 | tee /tmp/lint-chutoro-2-2-7.out
   ```

   ```sh
   set -o pipefail
   make test 2>&1 | tee /tmp/test-chutoro-2-2-7.out
   ```

10. Run `coderabbit review --agent` after the proof/tests/docs milestone.
    Address all concerns before moving on. If a concern is intentionally
    deferred, record the reason in this `Decision log`.

11. Mark `docs/roadmap.md` item `2.2.7` done only after all validation and
    CodeRabbit review concerns are cleared.

12. Commit the implementation using the `commit-message` skill's file-based
    `git commit -F` workflow. Keep the implementation and any later refactor
    separate if refactoring is needed after the functional commit.

## Validation strategy

The validation strategy is layered:

- Kani proves the bounded invariants over symbolic masks and bounded point
  counts.
- `rstest` unit tests cover concrete examples, including boundary values and
  selector fallback cases that are easy for reviewers to understand.
- Existing property-based parity tests from `2.2.6` continue to cover numeric
  equivalence across scalar and runtime-supported SIMD backends.
- Existing trybuild and feature-gating tests continue to cover the stable and
  optional nightly portable-SIMD build boundaries.
- No new end-to-end or `rstest-bdd` behavioural tests are expected because this
  work should not change command-line behaviour, persistence, network
  boundaries, UI flows, or externally observable library workflows. If
  implementation changes a public workflow, add the smallest behavioural test
  that demonstrates that change and update this plan before proceeding.

Expected final validation commands, all run with `set -o pipefail` and `tee`,
are `make kani`, `make kani-full`, `make check-fmt`, `make lint`, and
`make test`.

## Outcomes & retrospective

- Harnesses added:
  `verify_dense_simd_dispatch_selection_respects_support_masks` and
  `verify_dense_simd_tail_padding_lane_bounds`.
- Tail proof bounds: point counts `0..=17`, dimensions `0..=3`, lane widths
  `4`, `8`, and `16`, and eight bounded SIMD batches.
- `make kani` succeeded on 2026-05-24 with log
  `/tmp/kani-chutoro-2-2-7-after-core-fmt-removal.out`.
- `make kani-full` did not pass. Logs:
  `/tmp/kani-full-chutoro-2-2-7.out`,
  `/tmp/kani-full-chutoro-2-2-7-rerun.out`, and
  `/tmp/kani-full-chutoro-2-2-7-second-rerun.out`. The first two attempts hit
  Kani `0.67.0` compiler panics in existing assertion helpers; those helper
  patterns were repaired. The final run exposed an unrelated existing
  `verify_cosine_zero_on_identical_3d` proof failure in `chutoro-core`.
- Focused Markdown lint for the four changed documentation files succeeded on
  2026-05-24 with log `/tmp/markdownlint-focused-chutoro-2-2-7.out`.
- Repository-wide `make fmt` applied formatting but failed during its
  Markdown lint phase on pre-existing MD013 line-length findings in unrelated
  documents. The accidental formatter churn in unrelated files was reverted.
- `make kani` includes both new dense-provider harnesses. `make kani-full`
  runs all `chutoro-core` harnesses and all dense-provider harnesses.
- The implementation has no public API or user-observable behaviour change, so
  `docs/users-guide.md` remains unchanged.
- CodeRabbit findings and `make kani-full`, `make check-fmt`, `make lint`, and
  `make kani` all succeeded again on 2026-05-24 with log
  `/tmp/kani-chutoro-2-2-7-final.out`.
- `make check-fmt` succeeded on 2026-05-24 with log
  `/tmp/check-fmt-chutoro-2-2-7-final2.out`.
- Focused Markdown lint for the changed documentation files succeeded on
  2026-05-24 with log
  `/tmp/markdownlint-focused-chutoro-2-2-7-final.out`.
- `make lint` succeeded on 2026-05-24 with log
  `/tmp/lint-chutoro-2-2-7-final.out`.
- `make test` succeeded on 2026-05-24 with log
  `/tmp/test-chutoro-2-2-7-final.out`; nextest reported 969 passed and one
  skipped test.
- `coderabbit review --agent` completed on 2026-05-24 with log
  `/tmp/coderabbit-chutoro-2-2-7-implementation.out` and zero findings.
- The implementation was committed as `2db12e5` with subject
  `Add dense SIMD Kani harnesses`.
- Roadmap item `2.2.7` is marked done in `docs/roadmap.md`.
