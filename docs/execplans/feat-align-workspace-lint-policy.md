# Execution Plan (ExecPlan): bring workspace crates to lint parity

This ExecPlan is a living record for issue #200. It records completed stages,
current lint evidence, and the bounded next stage so the work can resume
without relying on an interactive session.

Status: IN PROGRESS

## Purpose / Big Picture

Make the root `Cargo.toml` the effective lint-policy source of truth. Every
workspace member will inherit the policy, except for documented and
continuously checked exceptions. The finished state has no whole-library-crate
Whitaker filesystem exemption, denies private-item documentation debt, and
keeps local editor feedback aligned with commit gates.

## Conformance Basis

- Issue #200 requires every member to inherit `[workspace.lints]` or have a
  documented, checked exception.
- The supplied implementation plan defines four phases: policy bootstrap,
  library opt-in, benches and private documentation, then Whitaker narrowing
  plus developers' documentation.
- `Cargo.toml` is the target source of truth for Rust, Clippy, and Rustdoc
  lint policy.

## Constraints

- Preserve runtime behaviour while remediating lints.
- Keep the root lint policy strict; do not solve onboarding by weakening it.
- Prefer code fixes. A suppression must be scoped to the item that needs it,
  use `#[expect(..., reason = "...")]`, and explain why a code change is not
  sound or practical.
- Do not use a crate-level blanket allowance.
- Keep every changed Rust file formatted and below the 400-line limit.
- Run `make check-fmt`, `make lint`, `make typecheck`, and `make test` at the
  end of every completed, green stage. Intermediate commits may retain known
  lint failures while a lint family is being resolved, as authorized by the
  issue owner on 2026-08-23.

## Tolerances (Exception Triggers)

- Stop for direction if satisfying a lint requires a public API or persisted
  format change.
- Stop for direction if a lint requires a cross-module reorganization that
  invalidates the staged on-boarding design.
- Stop for direction if a candidate suppression would cover an entire crate,
  module unrelated to the diagnosed site, or a lint family without a specific
  reason.
- Record every gate failure and its next remediation family before starting a
  different crate.

## Risks

- The root `self_named_module_files = "deny"` policy conflicts with several
  established `foo.rs` plus `foo/` module layouts. Moving these files is a
  structural refactor, not a mechanical lint repair.
- `indexing_slicing` and `float_arithmetic` protect algorithmic paths. Their
  remediation must preserve HNSW and clustering semantics rather than merely
  replace expressions.
- Full all-target linting includes test-only modules; the earlier measurement
  counted fewer diagnostics than the live all-target gate.
- The shared Cargo cache is contended by other worktrees. A test gate may need
  to wait for the active owner rather than bypassing Cargo's package-cache
  lock.

## Progress

- [x] 2026-08-23: Created the Leta workspace and loaded the requested Rust
      and architecture skills.
- [x] 2026-08-23: Added workspace `check-cfg` declarations for `kani`,
      `coverage`, `nightly`, and Dylint's `dylint_lib` configuration name.
- [x] 2026-08-23: Opted `chutoro-providers-text` into workspace lints and
      replaced exact float comparisons in its Levenshtein tests with bitwise
      equality for integral `f32` scores.
- [x] 2026-08-23: Passed `make check-fmt`, `make lint`, `make typecheck`, and
      `make test` for the policy bootstrap. CodeRabbit reported no findings.
- [x] 2026-08-23: Opted `chutoro-core` into workspace lints and applied the
      machine-applicable `missing_const_for_fn` and `use_self` fixes.
- [x] 2026-08-24: Rechecked the core mechanical checkpoint: formatting and
      type checking pass; the intentional Clippy baseline is 604 diagnostics.
- [x] 2026-08-24: Moved the seven existing self-named module files to their
      already-present `mod.rs` locations. Formatting and type checking pass;
      scoped core Clippy reports no remaining `self_named_module_files` finding
      and now reaches 187 core-library diagnostics.
- [x] 2026-08-24: Applied and reviewed the machine-applicable
      `doc_markdown` repairs in core source and property-test documentation.
      A scoped run with all policy lints disabled except `doc_markdown` passes.
- [x] 2026-08-24: Documented every reported core `Result` error and deliberate
      panic contract. Scoped core Clippy passes with `missing_errors_doc` and
      `missing_panics_doc` denied.
- [ ] 2026-08-24: Repeat the checkpoint test gate once unrelated shared Cargo
      work releases its package-cache lock; the first run reached 172 of 1,085
      tests before infrastructure contention suspended it.
- [ ] Resolve `chutoro-core` structural, bounds-safety, numerical, and test
      lint families in bounded commits.
- [ ] Opt in the remaining library crates and retire duplicated Rustdoc flags.
- [ ] Replace the benches mirror, deny private item documentation debt, and
      document private items.
- [ ] Narrow Whitaker filesystem exclusions and document the exception model.
- [ ] Run final gates, CodeRabbit review, publish the renamed branch, and
      create the requested draft pull request.

## Surprises & Discoveries

- The supposedly free `chutoro-providers-text` opt-in exposed four
  `float_cmp` diagnostics in `tests/textsource.rs`; these are fixed in the
  first stage.
- After the core opt-in and mechanical fixes, the full all-target Clippy gate
  reports 604 errors, rather than the 295 originally measured for primary
  spans. The difference includes test targets and policy lints that were not in
  the initial grouped estimate.
- The CodeGraph MCP workspace reindex call cannot run because this session's
  approval policy rejects it. Leta remains available for semantic navigation.
- `git stash apply --index` cannot create the linked-worktree `index.lock` due
  to a read-only filesystem. Applying the stash patch without `--index` works
  and the stash remains as the recovery copy.
- The former Git-metadata restriction has cleared. The core checkpoint can now
  be committed; retain `stash@{0}` until its committed diff is independently
  verified.
- The 2026-08-24 checkpoint gate confirms `make check-fmt` and `make
  typecheck` pass. `make lint` fails with the recorded 604 core Clippy
  diagnostics. `make test` is incomplete because unrelated suspended Cargo
  jobs hold the shared package-cache lock; it reached 172 of 1,085 tests.
- Moving the self-named modules is a content-preserving layout migration: all
  seven destination directories already existed for their child modules. The
  scoped core Clippy run now has no `self_named_module_files` diagnostics and
  reaches 187 remaining errors in the library target.
- Clippy supplied machine-applicable `doc_markdown` changes for eleven files.
  They add Rustdoc code formatting only; no identifiers or behaviour change.
- Command-line lint levels propagate to local dependencies. The semantic
  Rustdoc check therefore uses `--no-deps`; otherwise it diagnoses
  `chutoro-test-support` before that crate inherits the workspace policy.

## Decision Log

- 2026-08-23: Use staged onboarding instead of one all-or-nothing migration.
  Rationale: the issue owner explicitly selected the staged approach after live
  diagnostics substantially exceeded the planning count.
- 2026-08-23: Keep the core mechanical lint pass as an intermediate checkpoint
  even though the complete lint gate remains red. Rationale: it removes a
  coherent, machine-applicable lint family and makes subsequent diagnostics
  more specific. The issue owner explicitly authorized intermediate commits
  with remaining lint failures.
- 2026-08-23: Do not apply bare `#[allow(no_std_fs_operations)]`. Rationale:
  ordinary warning-denied builds reject it as an unknown lint; any single-site
  Whitaker exemption must use the guarded
  `cfg_attr(dylint_lib = "whitaker_suite", ...)` form and the workspace
  check-cfg declaration.

## Stage 2a: Core Mechanical Lint Onboarding

`chutoro-core/Cargo.toml` inherits workspace lints. The stage converts
const-eligible accessors and constructors to `const fn`, and replaces clear
type-name repetition with `Self`. It does not alter algorithmic behaviour.

Run `cargo clippy -p chutoro-core --all-features --all-targets -- -D warnings`
to establish the remaining baseline. The expected current result is failure
with structural, safety, numerical, and test diagnostics. Capture the command
output under `/tmp` and record the next lint family before editing it.

The stage is complete only when the mechanical diff is reviewed, formatted, and
committed. Its remaining gate failures are deliberately carried to the next
stage, where each family is resolved rather than globally allowed.

## Stage 2b: Core Module Layout

Rust resolves a module declared as `mod foo;` identically from `foo.rs` or
`foo/mod.rs`; moving a parent module to the latter form permits its existing
child directory without violating the workspace's
`self_named_module_files = "deny"` policy. Move only the seven reported parent
modules, retain all module declarations and contents, and do not alter public
paths.

The focused acceptance evidence is `make check-fmt` and `make typecheck`
passing, plus `cargo clippy -p chutoro-core --all-features --all-targets -- -D
warnings` containing no `self_named_module_files` diagnostic. This stage may
remain lint-red for the next explicit family; record the new baseline first.

## Stage 2c: Core Documentation Markup

Apply only Clippy's machine-applicable `doc_markdown` suggestions, then review
every changed Rustdoc line to confirm it quotes an identifier or symbolic
expression accurately. Run a scoped Clippy invocation that disables the
workspace's unrelated explicit lint denials and enables only `doc_markdown`;
it must complete without a documentation-markup warning. The next stage owns
missing error and panic sections, which require semantic documentation rather
than formatting.

## Stage 2d: Core Error and Panic Contracts

Add a precise `# Errors` section to each public core API reported by
`missing_errors_doc`, describing the actual condition represented by its error
type. Add `# Panics` only to the two APIs that deliberately panic, describing
their concrete precondition or test-only lock failure. Verify with
`cargo clippy -p chutoro-core --all-features --all-targets --no-deps` while
denying only `missing_errors_doc` and `missing_panics_doc`; `--no-deps` keeps
the scoped check from enforcing a future lint policy on dependencies that are
not yet opted in.

## Verification Plan

- Const conversion invariant: each changed function remains callable with the
  same inputs and returns the same value. Verification: `make typecheck` and
  `make test` after the checkpoint; existing call sites exercise both normal
  and test-only paths.
- Lint-policy invariant: `chutoro-core` uses `[lints] workspace = true` rather
  than a duplicate or partial table. Verification: inspect its manifest and run
  package Clippy with all features and targets.
- Numeric and indexing invariant: later remediation must not change the
  ordering, distance, or error behaviour of HNSW and clustering operations.
  Verification: targeted existing tests before each edit, then `make test` at
  the green-stage boundary. No new mathematical lemma is introduced by this
  mechanical stage.

## Outcomes & Retrospective

The final outcome is not yet achieved. The first green policy-bootstrap stage
is committed as `b8330a7`. The core mechanical checkpoint is committed as its
own intentionally lint-red stage after the exact residual diagnostics and
incomplete shared-cache test run have been recorded.

## Revision Note

Created on 2026-08-23 after the issue owner selected staged onboarding and
authorized intermediate commits with known lint failures. It records the live
diagnostic expansion and the Git-metadata access constraint so the next agent
can resume safely.
