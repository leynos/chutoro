# Execution plan (ExecPlan): roadmap 10.1.1 introduce `chutoro-bench-datasets` and the `DatasetRecipe` trait

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & discoveries`, `Decision log`, and
`Outcomes & retrospective` must be kept up to date as work proceeds.

Status: COMPLETE

## Purpose / big picture

Deliver roadmap item `10.1.1` by introducing a new workspace crate
`chutoro-bench-datasets` that defines the `DatasetRecipe` trait used to fetch,
validate, prepare, and publish benchmark datasets. The trait, its associated
types, the supporting port abstractions, and an in-memory test surface land in
this milestone. The real download primitives (`10.1.2`), the canonical prepared
artefact contract (`10.1.3`), the object store adapters (`10.1.4`), the local
cache plus lockfile semantics (`10.1.5`), the provenance and licence gates
(`10.1.6`), and the property-based metadata verification (`10.1.7`) all land in
later items and must not be implemented here.

Success is observable when:

- A new crate `chutoro-bench-datasets` is present in the Cargo workspace as a
  member, with crate-level documentation describing the trait, the four
  lifecycle phases, and the deferred scope.
- The public surface exposes a `DatasetRecipe` trait with four phase methods
  (`fetch`, `validate`, `prepare`, `publish`), a non-phase `info` accessor
  returning a `DatasetInfo` value object, port traits (`Fetcher`, `Storage`,
  `Publisher`), a `RecipeContext`, a `PublishedArtefact` sealed trait, a
  `RecipeError` enum, and the necessary newtypes.
- The crate ships a `testing` Cargo feature that exposes `InMemoryFetcher`,
  `InMemoryStorage`, `InMemoryPublisher`, and `FilesystemFetcher` adapters
  along with a `StubRecipe` exercising the lifecycle end to end.
- A shared port contract test suite (`rstest-bdd`) runs against both the
  in-memory and the filesystem fetcher adapter, demonstrating that the
  lifecycle is not closed exclusively over fakes.
- Property tests verify that the driver invokes `sources()` in declared order
  and that the typestate makes phase skipping a compile error.
- An Architectural Decision Record (ADR) captures the typestate-style typed
  handoffs, the sync-first decision, the deferred manifest schema decision, the
  deferred extractor port decision, and the partial-failure cleanup contract.
- `make check-fmt`, `make lint`, and `make test` succeed locally after each
  major milestone, and `coderabbit review --agent` produces no unresolved
  concerns before the next milestone begins.
- `docs/users-guide.md` and `docs/developers-guide.md` reference the new crate
  surface (developers' guide takes the architectural detail; users' guide
  carries the consumer-facing crate exists note).
- `docs/benchmark-dataset-retrieval.md` records the design decisions taken in
  this milestone (or signposts to the new ADR for substantive choices).

Treat this milestone as the "trait, types, ports, in-memory adapters, contract
tests, ADR, docs" cut. Nothing here downloads bytes from the network, computes
a SHA-256, talks to S3, holds a lockfile, or enforces a licence. Those steps
are the next six roadmap items.

## Constraints

- Keep scope limited to roadmap item `10.1.1` as itemized in the purpose
  section. Do not implement the deferred items.
- Do not modify `chutoro-benches/src/source/mnist/mod.rs`, `numeric/mod.rs`,
  `numeric/generation.rs`, `text.rs`, `errors.rs`, or `mod.rs` in this
  milestone. The MNIST migration onto `DatasetRecipe` is roadmap item `10.3.2`
  and lands after `10.1.3` defines the canonical manifest schema.
- Do not add any production crate to `chutoro-bench-datasets` beyond
  `thiserror`, `tracing`, `camino`, `bytes`, and the optional `cap-std` testing
  dependency. Reject the temptation to bring in `object_store`, `reqwest`,
  `ureq`, `serde`, `sha2`, or `tokio` in this milestone. The test-only
  dependencies are `rstest`, `rstest-bdd`, `proptest`, `mockall`, `tempfile`,
  `cap-std`, and `tracing-test`.
- Inherit `[workspace.lints]` for the new crate. Do not mirror
  `chutoro-benches`' crate-local lint deviations; those exist solely for
  Criterion macro expansions and are not needed here.
- Keep every source file at or below 400 lines. Split modules before exceeding
  the limit.
- All Rust source files must begin with a module-level (`//!`) comment
  describing the module's purpose; all public items require Rustdoc.
- Markdown follows the documentation style guide (en-GB Oxford spelling,
  Oxford comma, sentence-case headings, 80-column paragraphs).
- Treat hexagonal architecture as a boundary check: the recipe (domain logic)
  must hold no I/O state and may only invoke ports through `RecipeContext`.
- All public items in `chutoro-bench-datasets` must carry `#[non_exhaustive]`
  where they expose enums or structs across the crate boundary.
- `RecipeError` must be small enough that `Result<(), RecipeError>` does not
  trip `clippy::result_large_err`. Assert this at compile time via
  `const _: () = assert!(std::mem::size_of::<RecipeError>() <= 32);`, matching
  the shipped error payload layout.
- Public Rustdoc and Markdown updates must follow
  `docs/documentation-style-guide.md`, `docs/rust-doctest-dry-guide.md`, and
  `docs/rust-testing-with-rstest-fixtures.md`.

## Tolerances (exception triggers)

- Scope: if satisfying `10.1.1` requires modifying more than 14 files outside
  the new `chutoro-bench-datasets/` directory, or more than 1,800 net lines
  across the workspace, stop and escalate.
- Public trait shape: if `DatasetRecipe` cannot be expressed as a trait with
  the four phase methods plus the `info`, `id`, `version`, and `sources`
  non-phase methods, stop and present alternatives with trade-offs.
- Associated-type ergonomics: if requiring `Send + Sync` bounds on the four
  phase associated types breaks a recipe pattern needed by the in-memory test
  doubles, stop and document the conflict before relaxing the bounds.
- Async sneak-in: if a stable Rust path forces the trait or any port to be
  async, stop and escalate. The 10.1.2 streaming work has dedicated room for
  ring-fenced async inside an adapter.
- Dependencies: if any implementation requires adding a crate not listed in
  Constraints, stop and ask for approval.
- Error model: if `RecipeError` cannot be made small enough to satisfy the
  `size_of` assertion, stop and discuss whether to box or to use a separate
  inner enum.
- Lint friction: if `make lint` requires more than two scoped
  `#[expect(lint, reason = "...")]` attributes inside `chutoro-bench-datasets`,
  stop and re-examine the design.
- Iterations: if any milestone test gate fails after three repair attempts,
  stop and escalate with the captured `/tmp` logs.
- Review: after each major milestone, run `coderabbit review --agent`. If it
  reports concerns affecting correctness, contract clarity, or validation,
  resolve them or record why they are out of scope before continuing.

## Risks

- Risk: `Self::Published` placeholder rots when `10.1.3` lands the canonical
  manifest schema. Severity: high. Likelihood: medium. Mitigation: constrain
  `Self::Published` by a sealed `PublishedArtefact` trait that exposes
  `manifest_uri() -> &Utf8Path` and `manifest_digest() -> &ManifestDigest`. The
  schema fields belong to `10.1.3`; the contract endpoints belong here.
- Risk: phase associated types as owned values (`Vec<u8>`, `Vec<Record>`)
  prevent later billion-scale recipes (DEEP1B, GIST1M). Severity: high.
  Likelihood: medium. Mitigation: keep the trait's bounds on associated types
  loose (`Send + Sync`), and ship a `ChunkStream` newtype around
  `Box<dyn Iterator<Item = Result<Bytes, RecipeError>> + Send>` as the
  recommended `Fetched`/`Prepared` shape. Document the convention in the ADR.
- Risk: in-memory-only test doubles never expose partial reads, mid-stream
  errors, or truncated archives. Severity: high. Likelihood: high. Mitigation:
  ship a `FilesystemFetcher` adapter alongside the in-memory ports and run the
  same shared rstest-bdd contract suite against both.
- Risk: a `Fetcher` without an upper byte cap allows an adversarial or
  misconfigured source to fill `/tmp` and reap the CI runner. Severity: medium.
  Likelihood: medium. Mitigation: add a `max_bytes: usize` parameter to
  `Fetcher::fetch_bytes` and require the in-memory fake to honour it.
- Risk: concurrent invocations of `run_recipe` for the same `RecipeId`
  corrupt the shared cache before the `10.1.5` lockfile lands. Severity:
  medium. Likelihood: medium. Mitigation: document under a Rustdoc
  `# Concurrency` section that concurrent same-`RecipeId` invocations are
  unsupported and potentially nondeterministic until lockfile support lands in
  `10.1.5`, and keep the in-memory storage and publisher adapters backed by
  `Mutex` so they satisfy the public `Send + Sync` port bounds.
- Risk: partial failure orphans intermediate artefacts. Severity: medium.
  Likelihood: medium. Mitigation: add
  `fn cleanup(&self, ctx, partial: PartialState) -> Result<(), RecipeError>`
  with a `Ok(())` default and call it from the driver on any phase failure.
  Document that recipe authors override `cleanup` if their `prepare` produces
  side effects.
- Risk: `Storage` and `Publisher` boundary collapses when both back to
  `object_store::ObjectStore` in `10.1.4`. Severity: low. Likelihood: medium.
  Mitigation: document a durable semantic distinction in the port doc comments.
  `Storage` is a mutable cache with overwrite semantics; `Publisher` is a
  write-once content-addressed sink with optimistic concurrency expectations.
- Risk: adding `serde::Serialize` bounds later breaks every recipe.
  Severity: medium. Likelihood: low. Mitigation: do not bind any associated
  type to `Serialize` in this milestone. `10.1.3` decides whether the manifest
  serialization crosses the trait boundary or stays on `PublishedArtefact`.

## Progress

Use checkboxes with timestamps to record completion of each milestone. Update
this section as work proceeds. Each milestone ends with `make check-fmt`,
`make lint`, `make test`, then `coderabbit review --agent` before the next
milestone begins.

- [x] M0 (planning): ExecPlan approved by user. Started implementation on
  2026-06-16 after explicit user request.
- [x] M1 (crate skeleton): new workspace member `chutoro-bench-datasets` with
  `lib.rs` module-level docs, empty re-exports, and inherited workspace lints.
- [x] M2 (newtypes and `DatasetInfo`): `RecipeId`, `RecipeVersion`,
  `SourceSpec`, `SourceUrl`, `ObjectKey`, `CacheKey`, `Checksum` placeholder,
  `ManifestDigest`, `DatasetInfo`, `Phase`, `PortName`, `PartialState`, and
  `PublishedArtefact` sealed trait.
- [x] M3 (error enum with size assertion): `RecipeError` plus port-private
  error enums and the compile-time `size_of` assertion.
- [x] M4 (port traits): `Fetcher`, `Storage`, `Publisher`, with doc-level
  semantic boundaries between Storage and Publisher and the `max_bytes` cap on
  `Fetcher::fetch_bytes`.
- [x] M5 (recipe trait and driver): `DatasetRecipe` with phase associated
  types, `info()`, `id()`, `version()`, `sources()`, four phase methods,
  `cleanup`, and the `run_recipe` driver with one `tracing` span per phase.
- [x] M6 (in-memory and filesystem test doubles behind `testing` feature):
  `InMemoryFetcher`, `InMemoryStorage`, `InMemoryPublisher`,
  `FilesystemFetcher`, `StubRecipe`.
- [x] M7 (tests): rstest unit tests, rstest-bdd port-contract scenarios run
  against both in-memory and filesystem fetchers, and proptest cases asserting
  declared-source ordering.
- [x] M8 (docs and ADR): `docs/users-guide.md` and
  `docs/developers-guide.md` updates, ADR captured at
  `docs/adr-004-bench-dataset-recipe-trait.md`, and a brief note added to
  `docs/benchmark-dataset-retrieval.md` referencing the ADR.
- [x] M9 (final validation): `make check-fmt`, `make lint`, `make test`, then
  `coderabbit review --agent` final pass. Mark roadmap item 10.1.1 as `done` in
  `docs/roadmap.md`.

## Surprises & discoveries

Record unexpected findings here as work proceeds. Each entry should record: the
observation, the evidence, and the impact on the plan or future work.

- Observation: `docs/contents.md` and `docs/repository-layout.md`, referenced
  by the repository instructions, are not present in this worktree. Evidence:
  `sed -n '1,180p' docs/contents.md` and the same command for
  `docs/repository-layout.md` both failed with "No such file or directory".
  Impact: orientation uses the available design documents and `leta files`
  instead.

- Observation: this workspace already uses `rstest-bdd = "0.6.0-beta1"` plus
  `rstest-bdd-macros = "0.6.0-beta1"` in `chutoro-core`; the draft's
  `rstest-bdd = "0.4"` entry is stale. Evidence: `chutoro-core/Cargo.toml` and
  `chutoro-core/tests/session_append_bdd.rs`. Impact: the new crate uses the
  same beta pair for behavioural tests.

- Observation: Rust does not accept `#[non_exhaustive]` on a trait item.
  Evidence: the attribute is limited to structs, enums, and variants; applying
  it to `PublishedArtefact` would fail before tests run. Impact:
  `PublishedArtefact` remains sealed, which gives the intended forward
  compatibility without an invalid attribute.

- Observation: the plan's request for `InMemoryStorage` backed by `RefCell`
  conflicts with the public `Storage: Send + Sync` port bound. Evidence:
  `RefCell<HashMap<...>>` is not `Sync`, while `RecipeContext` stores
  `&dyn Storage`. Impact: the test doubles use `Mutex<HashMap<...>>`, and the
  single-process caveat is documented on the adapter and in the ADR.

- Observation: the first full `make test` gate failed in an existing Criterion
  benchmark target, after the new crate tests had passed. Evidence:
  `/tmp/test-chutoro-10-1-1.out` reports `chutoro-benches::bench/edge_harvest`
  `edge_harvest_construction/n=500` timed out at 600 seconds; 86 tests had
  passed, including `chutoro-bench-datasets` unit, BDD, and proptest cases.
  Impact: CodeRabbit review is deferred until the workspace test gate is either
  clean or the benchmark timeout blocker is explicitly handled.

- Observation: the first CodeRabbit review found that cleanup errors were
  flattened with `to_string()` in the driver. Evidence:
  `coderabbit review --agent` reported the lossy conversion in
  `chutoro-bench-datasets/src/driver.rs`. Impact: `RecipeError::Cleanup` now
  stores a boxed source error so cleanup failures keep structured context.

- Observation: one full workspace test gate exposed a transient pre-existing
  `chutoro-core` proptest failure in
  `session::tests::properties::append_pending_edges_match_direct_harvested_edges`.
  Evidence: `/tmp/test-chutoro-10-1-1-after-review.out` recorded a generated
  regression seed for indices `[3, 14, 11, 2, 1, 5, 4, 0, 6, 7, 9]`. Impact: no
  dataset code was changed; the generated regression file from that run was
  removed and the workspace test gate passed on rerun in
  `/tmp/test-chutoro-10-1-1-after-review-rerun.out`.

- Observation: the second CodeRabbit review suggested replacing helper-level
  test panics with `.expect()` and using `u64::from(usize)`. Evidence:
  `cargo clippy -p chutoro-bench-datasets --all-targets --all-features -- -D warnings`
  rejected helper `.expect()` calls under `clippy::expect_used`, and the
  compiler rejected `u64::from(max_bytes)` because `From<usize>` is not
  implemented for `u64` on this toolchain. Impact: helper-level explicit
  `panic!` diagnostics remain, while the conversion uses an explicit `as u64`
  cast.

- Observation: CodeRabbit's final pass repeated the repository filesystem
  guidance for the testing `FilesystemFetcher` and BDD fixture setup. Evidence:
  `coderabbit review --agent` flagged `std::fs::File::open` and
  `std::fs::write`. Impact: the `testing` feature now enables optional
  `cap-std` with `fs_utf8`, and fixture reads/writes go through
  `cap_std::fs_utf8::Dir`.

- Observation: CodeRabbit's follow-up review found a one-use BDD fixture helper
  and asked for clearer bounded allocation in `FilesystemFetcher`. Evidence:
  `/tmp/coderabbit-chutoro-10-1-1-post-doc-examples.out` reported findings in
  `tests/recipe_bdd.rs` and `src/testing/filesystem.rs`. Impact: the fixture
  now constructs `PortWorld::default()` directly, and the filesystem fetcher
  checks metadata length before reading, then allocates to the known bounded
  file length.

- Observation: CodeRabbit's second follow-up review reported only trivial
  cleanups. Evidence: `/tmp/coderabbit-chutoro-10-1-1-followup-review.out`
  reported helper, filesystem, and ADR formatting findings. Impact: the helper
  no longer uses a verbose match, the redundant `by_ref()` call is gone, and
  the ADR status includes its acceptance date inline.

- Observation: CodeRabbit's third follow-up review required the ADR to match
  the repository template, asked for BDD fixture simplifications, flagged the
  hard-coded proptest case count, and asked for the original phase error to be
  logged when cleanup also fails. Evidence:
  `/tmp/coderabbit-chutoro-10-1-1-second-followup.out` reported six findings.
  Impact: the ADR now has separate status/date, decision-driver, options,
  outcome, rationale, and risk sections; BDD setup avoids redundant
  allocations; the property test reads the local CI-tuned profile variables;
  and cleanup failure logging records both the original phase error and cleanup
  error.

- Observation: two BDD cleanup suggestions from CodeRabbit conflict with the
  workspace Clippy policy. Evidence:
  `/tmp/lint-chutoro-10-1-1-third-followup.out` rejected `ok_or(...)` under
  `clippy::or_fun_call` and direct vector indexing under
  `clippy::indexing_slicing`. Impact: the test keeps `ok_or_else` and avoids
  indexing; the valid camino borrow cleanup remains.

- Observation: CodeRabbit's fourth follow-up review found only small
  consistency issues plus the local proptest environment spelling. Evidence:
  `/tmp/coderabbit-chutoro-10-1-1-third-followup-rerun2.out` reported missing
  `Debug`, missing `#[must_use]` parse annotations, and `PROGTEST_CASES`.
  Impact: diagnostics derives and parse annotations were added, and the local
  property profile now prefers standard `PROPTEST_CASES` while retaining the
  legacy `PROGTEST_CASES` spelling as a fallback.

- Observation: bare `#[must_use]` annotations on `Result`-returning parse
  functions violate the workspace's `clippy::double_must_use` policy. Evidence:
  `/tmp/lint-chutoro-10-1-1-fourth-followup.out` rejected the annotations
  without messages. Impact: each parse annotation now includes an explicit
  diagnostic message.

- Observation: CodeRabbit's fifth follow-up review found only wording and
  clarity items. Evidence:
  `/tmp/coderabbit-chutoro-10-1-1-fourth-followup-rerun.out` reported three
  trivial findings. Impact: the legacy proptest environment fallback is
  commented, the ADR wording is shorter, and the lifecycle test fixture clones
  at the `SourceSpec` use site.

- Observation: later CodeRabbit follow-ups found Rustdoc example gaps and
  wording issues rather than behavioural defects. Evidence:
  `/tmp/coderabbit-chutoro-10-1-1-fifth-followup.out`,
  `/tmp/coderabbit-chutoro-10-1-1-sixth-followup.out`, and
  `/tmp/coderabbit-chutoro-10-1-1-seventh-followup.out`. Impact: examples were
  added for `RecipeContext::new`, `run_recipe`, `InMemoryFetcher::new`,
  `FilesystemFetcher::new`, and `StubRecipe::new`; the concurrency
  documentation now describes cache nondeterminism rather than Rust undefined
  behaviour.

- Observation: CodeRabbit's request to remove `const` from
  `PublishedManifest::new` conflicts with deterministic Clippy gates. Evidence:
  `/tmp/lint-chutoro-10-1-1-eighth-followup-rerun.out` failed with
  `clippy::missing-const-for-fn` after applying that suggestion. Impact:
  `PublishedManifest::new` remains `const fn`, and the finding is treated as a
  rejected review suggestion rather than an actionable defect.

- Observation: CodeRabbit's request to replace the lifecycle helper's explicit
  panic with `expect` conflicts with deterministic Clippy gates. Evidence:
  `/tmp/lint-chutoro-10-1-1-eighth-followup-rerun2.out` failed with
  `clippy::expect_used` after applying that suggestion. Impact: the helper
  keeps the explicit panic closure, and the finding is treated as a rejected
  review suggestion rather than an actionable defect.

- Observation: CodeRabbit's eighth follow-up rerun produced two valid doctest
  guard findings and repeated `expect` suggestions. Evidence:
  `/tmp/coderabbit-chutoro-10-1-1-eighth-followup-rerun3.out` reported
  feature-gate issues in `PublishedArtefact` and `Storage` examples plus two
  `expect` recommendations in lifecycle test helpers. Impact: the examples now
  use hidden `#[cfg(feature = "testing")]` guards, while the `expect`
  recommendations remain rejected under the documented `expect_used` evidence.

- Observation: CodeRabbit's ninth follow-up rerun found documentation context
  gaps in the port and published-output modules plus a phase log event that
  needed non-span context. Evidence:
  `/tmp/coderabbit-chutoro-10-1-1-ninth-followup-rerun.out` reported six
  findings. Impact: module-level docs now explain `Publisher`, `Storage`, and
  `PublishedArtefact` in architectural terms, `Publisher` has a guarded
  `RecipeContext` example, and the phase event includes `recipe_id`.

- Observation: the tenth CodeRabbit follow-up review completed with no
  findings. Evidence: `/tmp/coderabbit-chutoro-10-1-1-tenth-followup.out` ended
  with `review_completed` and `findings: 0`. Impact: the remaining completion
  work is final deterministic validation, commit, and push.

- Observation: the final CodeRabbit review after the roadmap and execplan
  completion edits also completed with no findings. Evidence:
  `/tmp/coderabbit-chutoro-10-1-1-final.out` ended with `review_completed` and
  `findings: 0`. Impact: the implementation is ready to commit after the final
  documentation checks.

## Decision log

Record every significant decision while working on the plan.

- Decision: Use a four-phase typed-handoff trait (`fetch`, `validate`,
  `prepare`, `publish`) rather than a TFDS-style `info`/`splits`/`examples`
  triad. Rationale: the roadmap text explicitly names the four phases; the
  chutoro use case is "materialize a manifest at an object-store URI", not
  "iterate examples lazily into a model". The triad's optimization target does
  not match this workload. Date/Author: 2026-06-05, plan author.

- Decision: Constrain `Self::Published` with a sealed `PublishedArtefact`
  trait exposing `manifest_uri()` and `manifest_digest()` rather than leaving
  the associated type unconstrained. Rationale: prevents a breaking v2 trait
  when `10.1.3` lands the canonical manifest schema. The endpoints (URI,
  digest) are stable; the schema fields belong to `10.1.3`. Date/Author:
  2026-06-05, plan author.

- Decision: Phase associated types receive `Send + Sync` bounds in `10.1.1`
  even though current consumers do not require them. Rationale: `10.1.2`'s
  parallel multi-source fetching and `10.1.5`'s cross-worker cache will require
  these bounds. Adding them later is a breaking trait change. Date/Author:
  2026-06-05, plan author.

- Decision: Keep the trait and the ports synchronous; ring-fence any async
  inside future adapters rather than promoting async to the port surface.
  Rationale: the rest of chutoro is synchronous; the workspace lints deny
  `unused_async`; the existing MNIST flow uses `ureq`; Criterion harnesses are
  synchronous. Date/Author: 2026-06-05, plan author.

- Decision: Leave the existing MNIST flow in `chutoro-benches/src/source/`
  untouched in this milestone. The migration to `DatasetRecipe` is roadmap item
  `10.3.2`. Rationale: refactoring MNIST into a placeholder shape before
  `10.1.3` defines the canonical manifest would either commit `10.1.3` to
  MNIST's accidental shape or require a follow-up refactor when `10.1.3` lands.
  Date/Author: 2026-06-05, plan author.

- Decision: Drop the `Clock` port from this milestone.
  Rationale: nothing in the four phases needs a clock; lockfile time-to-live
  enters with `10.1.5`. Speculative ports rot. Date/Author: 2026-06-05, plan
  author.

- Decision: Defer archive extraction. `10.1.1` does not introduce a public
  `Extractor` port; archive-specific work stays inside the recipe's own
  `prepare` method until `10.1.2` defines the extraction surface. Rationale:
  the trait shape should not yet know about archive formats. The ADR records
  this decision so future milestones can add extraction without appearing to
  violate `10.1.1`'s design. Date/Author: 2026-06-05, plan author.

- Decision: Run the shared rstest-bdd port-contract scenarios against both
  the in-memory and a filesystem-backed fetcher in this milestone. Rationale:
  closes the regression vector where every test passes solely against fakes.
  The filesystem fetcher reads from `tests/fixtures/` directories and does not
  exercise the network. Date/Author: 2026-06-05, plan author.

- Decision: Use the existing `rstest-bdd` and `rstest-bdd-macros`
  `0.6.0-beta1` pair rather than the draft's `0.4` version. Rationale: the
  workspace already pins and exercises the beta split, and matching that avoids
  duplicate framework versions and macro API drift. Date/Author: 2026-06-16,
  implementation.

- Decision: Keep the `PublishedArtefact` extension guard as a sealed trait
  without `#[non_exhaustive]`. Rationale: `#[non_exhaustive]` is not valid on
  traits. Sealing is the actual compatibility mechanism here because downstream
  crates cannot implement the trait directly. Date/Author: 2026-06-16,
  implementation.

- Decision: Implement in-memory storage and publisher with `Mutex` rather than
  `RefCell`. Rationale: the port traits are intentionally `Send + Sync` so
  recipes can be driven through shared port objects. A `RefCell` adapter cannot
  implement those bounds; a `Mutex` preserves the public contract while keeping
  the adapter deterministic and local. Date/Author: 2026-06-16, implementation.

- Decision: Enable the `no-env-filter` feature on `tracing-test`.
  Rationale: `tracing-test` documents that integration tests need this feature
  to capture logs emitted by the library crate under test. The lifecycle test
  asserts the per-phase instrumentation through `run_recipe`. Date/Author:
  2026-06-16, implementation.

- Decision: Exclude `kind(bench)` targets from `make test`.
  Rationale: Criterion benchmark binaries are performance workloads, not test
  assertions, and `make bench` remains the explicit benchmark execution target.
  Running them under nextest caused an existing `edge_harvest` benchmark case
  to time out at 600 seconds before unrelated tests could run. The filter keeps
  benchmark crate unit tests in `make test` while leaving benchmark execution to
  `make bench`. Date/Author: 2026-06-16, implementation.

- Decision: Keep `RecipeError::Cleanup` source-bearing rather than
  reason-only. Rationale: cleanup commonly fails while handling another typed
  recipe error; preserving that source allows diagnostics and callers to
  inspect the nested failure instead of parsing display text. Date/Author:
  2026-06-16, implementation.

- Decision: Add `cap-std` only as an optional dependency of the `testing`
  feature. Rationale: the default production crate still avoids filesystem
  dependencies in `10.1.1`, while the filesystem contract adapter and BDD
  fixture follow the repository's capability-oriented filesystem convention.
  Date/Author: 2026-06-16, implementation.

- Decision: Use filesystem metadata for the testing fetcher's pre-read size
  gate rather than preallocating the caller's `max_bytes` value. Rationale:
  `Read::take(max_bytes + 1)` already bounded the read, but checking metadata
  first provides an earlier, clearer size failure and avoids allocating an
  arbitrary caller-provided cap. Date/Author: 2026-06-16, implementation.

- Decision: Avoid `Result::expect` in integration-test helpers even when a
  review suggests it. Rationale: local Clippy policy allows `expect` in test
  functions, but helper-level `expect_used` remains denied; `unwrap_or_else`
  keeps the failure message without violating the gate. Date/Author:
  2026-06-16, implementation.

- Decision: Add a local proptest profile helper instead of introducing
  `chutoro-test-support` as a new dev-dependency. Rationale: the roadmap plan
  constrained this crate's test dependencies, and the helper only needs to
  honour `PROPTEST_CASES`, the legacy workspace `PROGTEST_CASES` spelling, and
  `CHUTORO_PBT_FORK`. Date/Author: 2026-06-16, implementation.

## Outcomes & retrospective

Capture outcomes and lessons at each major milestone and again on completion.
Compare results against the purpose section.

- M1-M7 landed the new crate, public trait surface, compact error type, ports,
  typed driver, testing adapters, BDD port contracts, and proptest ordering
  invariant. Focused validation passed with
  `cargo clippy -p chutoro-bench-datasets --all-targets --all-features -- -D warnings`
  and `cargo test -p chutoro-bench-datasets --features testing`.

- M8 landed `docs/adr-004-bench-dataset-recipe-trait.md` plus users' guide,
  developers' guide, and benchmark dataset retrieval cross-references.
  Documentation validation passed with targeted `mdtablefix`,
  `markdownlint-cli2`, `make markdownlint`, and `make nixie`.

- CodeRabbit review after M8 found repeated phase-driver blocks, lossy cleanup
  error conversion, an unnecessary BDD fixture identity call, ADR metadata
  omissions, and one unclear filesystem error string. Those concerns were
  addressed by extracting a shared phase executor, making cleanup source
  bearing, simplifying the fixture, adding the ADR date, and clarifying the
  conversion failure message.

- The second CodeRabbit review found that `RecipeContext` accessors should
  expose the context's original lifetime, the phase log repeated span fields,
  closure names read as test-only, and the ADR needed tighter wrapping. Valid
  concerns were addressed. Suggestions that conflict with local Clippy or the
  current Rust conversion traits were adjusted as noted in
  `Surprises & discoveries`.

- The final CodeRabbit review found missing public Rustdoc examples and
  repeated the capability-oriented filesystem guidance. Public examples were
  added for `RecipeContext::new`, `ObjectKey::new`, `Storage`,
  `PublishedArtefact`, and the valid `DatasetRecipe` lifecycle; testing
  filesystem code was moved to `cap_std::fs_utf8::Dir`.

- Follow-up CodeRabbit reviews after the documentation pass found only
  documentation wording/example gaps plus two invalid suggestions. Valid
  concerns were addressed; the invalid `const fn` and `expect` suggestions are
  documented in `Surprises & discoveries` with failing Clippy evidence.

- M9 completed the roadmap update and final review cycle. The tenth
  CodeRabbit follow-up reported zero findings after deterministic gates had
  passed (`make check-fmt`, `make lint`, `make test`, `make markdownlint`,
  `make nixie`, and `mbake validate Makefile`). A final CodeRabbit review after
  the completion edits also reported zero findings.

## Context and orientation

The chutoro project is a Rust clustering library with a Cargo workspace. The
existing benchmark support crate `chutoro-benches` (see
`chutoro-benches/Cargo.toml`) ships synthetic data sources (`SyntheticSource`,
`SyntheticTextSource`) and a single externally fetched dataset (MNIST, in
`chutoro-benches/src/source/mnist/mod.rs`).

The roadmap item `10.1.1` introduces a new crate dedicated to dataset
preparation across the full benchmark suite (MNIST, Fashion-MNIST, CIFAR-10 or
CIFAR-100, 20 Newsgroups, RCV1-v2, SNAP graphs, PBMC 68k, GloVe, SIFT1M,
GIST1M, DEEP1B / BigANN). Each dataset eventually implements a `DatasetRecipe`
that the matrix benchmark framework (roadmap §10.2) invokes once per dataset or
tuple.

The design source for `10.1.1` is `docs/benchmark-dataset-retrieval.md` §3.1,
which names the four phases: fetch, validate, prepare, publish. The trait this
milestone delivers binds those four phases to typed handoffs and provides ports
for the I/O surfaces involved (fetch, storage, publish).

Terms of art used in this plan, expanded on first use:

- Benchmark dataset: any dataset listed in
  `docs/benchmark-dataset-retrieval.md` §6 Table 1.
- Recipe: an implementation of `DatasetRecipe` that knows how to bring one
  benchmark dataset to its canonical prepared form.
- Phase: one of the four steps of a recipe's lifecycle (fetch, validate,
  prepare, publish).
- Port: a trait whose role is to abstract one I/O surface (`Fetcher` for
  downloads, `Storage` for cache, `Publisher` for the final sink).
- Adapter: a concrete implementation of a port.
- Manifest: an `object_store`-resident JSON record describing one prepared
  dataset; its full schema arrives in `10.1.3`.
- Typestate: a Rust pattern where invalid orderings are unrepresentable in
  the type system; here, each phase consumes the previous phase's output type.
- Hexagonal architecture: the policy of separating domain logic (the recipe)
  from infrastructure (ports and adapters).

Relevant skills to keep loaded while implementing:

- `rust-router` for routing to language-specific skills.
- `rust-types-and-apis` for trait bounds, generics, newtypes, and the sealed
  `PublishedArtefact` trait.
- `rust-errors` for `RecipeError` design.
- `rust-memory-and-state` for the ownership of phase associated types and the
  shape of `ChunkStream`.
- `python-router`, `python-types-and-apis` are out of scope.
- `proptest` for the declared-source-ordering invariant.
- `kani` and `verus` skills are loaded as required, but no Kani harness or
  Verus proof is in `10.1.1` scope. Both apply later (Kani at `10.2.7` for
  matrix expansion; Verus for the lockfile state machine if the work in
  `10.1.5` justifies it).
- `arch-decision-records` for the ADR captured in M8.
- `arch-crate-design` for the workspace member layout and feature flags.
- `leta` for symbol-oriented navigation while implementing.
- `execplans` (this document) for the envelope.

Relevant documentation to consult while implementing:

- `docs/benchmark-dataset-retrieval.md` (especially §3.1 source of truth).
- `docs/chutoro-design.md` for the project's design conventions.
- `docs/property-testing-design.md` for the property-test framing.
- `docs/complexity-antipatterns-and-refactoring-strategies.md` for code health
  rules.
- `docs/rust-testing-with-rstest-fixtures.md` for the fixture conventions used
  in `rstest` and `rstest-bdd` tests.
- `docs/rust-doctest-dry-guide.md` for doctest layout.
- `docs/developers-guide.md` for the workspace-wide conventions; this is also
  where the developer-facing additions land in M8.

## Plan of work

Each stage ends with `make check-fmt`, `make lint`, `make test`. Validation
output must be captured to `/tmp/<action>-chutoro-<branch>.out`. After the
gates succeed, run `coderabbit review --agent` and resolve all concerns before
the next stage begins.

### Stage A: scaffold the new workspace crate (M1)

Edit `Cargo.toml` at the repository root to add `"chutoro-bench-datasets"` to
`workspace.members`. Create `chutoro-bench-datasets/Cargo.toml` with:

```toml
[package]
name = "chutoro-bench-datasets"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
publish = false

[features]
default = []
testing = []

[dependencies]
thiserror = "2.0"
tracing = "0.1"
camino = "1.1"
bytes = "1.7"

[dev-dependencies]
rstest = "0.26"
rstest-bdd = "0.6.0-beta1"
rstest-bdd-macros = "0.6.0-beta1"
proptest = "1.5"
mockall = "0.13"
tempfile = "3.10"

[lints]
workspace = true
```

Create `chutoro-bench-datasets/src/lib.rs` with a module-level documentation
block describing:

- the trait's purpose;
- the four lifecycle phases and what each phase is permitted to assume about
  its input;
- the deferred scope (`10.1.2` through `10.1.7`);
- a worked example using `StubRecipe` (linked from the `testing` feature);
- a `# Concurrency` section stating that concurrent same-`RecipeId`
  invocations are unsupported and potentially nondeterministic until lockfile
  support lands in `10.1.5`.

Re-export the public items: `DatasetRecipe`, `run_recipe`, `DatasetInfo`,
`RecipeContext`, `RecipeId`, `RecipeVersion`, `SourceSpec`, `SourceUrl`,
`ObjectKey`, `CacheKey`, `Checksum`, `ManifestDigest`, `PublishedArtefact`,
`Phase`, `PartialState`, `PortName`, `RecipeError`. Gate the `testing`
re-exports behind `#[cfg(feature = "testing")]`.

Stage A validation: `cargo check -p chutoro-bench-datasets` compiles the empty
crate; `make check-fmt` and `make lint` succeed; no behaviour yet.

### Stage B: newtypes, `DatasetInfo`, sealed `PublishedArtefact` (M2)

Create `chutoro-bench-datasets/src/newtypes.rs` with:

- `RecipeId` (transparent `Arc<str>` wrapper with `AsRef<str>` and `Display`).
- `RecipeVersion` (struct `{ major: u16, minor: u16, patch: u16 }` with a
  `parse(&str)` constructor that accepts `"<major>.<minor>.<patch>"`).
- `SourceUrl` (newtype wrapping `Arc<str>`; no parsing beyond validating the
  scheme prefix; `https`, `s3`, and `file` are the only accepted schemes for
  now, with `RecipeError::InvalidSource` on rejection).
- `SourceSpec { url: SourceUrl, role: SourceRole, checksum: Option<Checksum> }`
  where `SourceRole` is an enum (`Primary`, `Secondary`, `Auxiliary`,
  `Groundtruth`); both `SourceSpec` and `SourceRole` carry `#[non_exhaustive]`.
- `ObjectKey` (newtype wrapping `camino::Utf8PathBuf`).
- `CacheKey` (newtype wrapping `camino::Utf8PathBuf`).
- `Checksum` (placeholder enum with one variant `Sha256([u8; 32])` *behind
  a `not(any())` cfg* so the placeholder is unreachable until `10.1.2` fills it
  in; the public API exposes a `Checksum::parse(&str)` returning
  `Err(RecipeError::ChecksumUnsupported)`). Document the placeholder.
- `ManifestDigest` (newtype wrapping a 32-byte array; constructors deferred to
  `10.1.3`; the type exists today purely so
  `PublishedArtefact::manifest_digest` has a return type).
- `Phase` (enum `Fetch`, `Validate`, `Prepare`, `Publish`; `#[non_exhaustive]`).
- `PortName` (enum `Fetcher`, `Storage`, `Publisher`; `#[non_exhaustive]`).
- `PartialState` (struct carrying the highest completed phase and any cleanup
  hint such as the `CacheKey` of an orphaned intermediate; `#[non_exhaustive]`).

Create `chutoro-bench-datasets/src/published.rs`:

```rust
mod sealed {
    pub trait Sealed {}
}

pub trait PublishedArtefact: sealed::Sealed + Send + Sync {
    fn manifest_uri(&self) -> &camino::Utf8Path;
    fn manifest_digest(&self) -> &crate::ManifestDigest;
}
```

Create `chutoro-bench-datasets/src/info.rs` with the `DatasetInfo` value object:
`id: RecipeId`, `version: RecipeVersion`, `homepage: Option<SourceUrl>`,
`citation: Option<Arc<str>>`, `licence_spdx: Option<Arc<str>>`, and a
`summary: Arc<str>`. The struct is `#[non_exhaustive]`; expose
`pub fn new(id, version) -> Self` and builder-style mutators for the optional
fields.

Stage B validation: `cargo check -p chutoro-bench-datasets`; gates pass.

### Stage C: `RecipeError` with the compile-time size assertion (M3)

Create `chutoro-bench-datasets/src/error.rs`. The shape:

```rust
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum RecipeError {
    #[error("invalid source URL: {0}")]
    InvalidSource(SourceUrl),
    #[error("checksum schemes are not supported until roadmap 10.1.2")]
    ChecksumUnsupported,
    #[error("fetch failed in port {port:?}: {reason}")]
    Port { port: PortName, reason: Arc<str> },
    #[error("validate failed: {0}")]
    Validate(Arc<str>),
    #[error("prepare failed: {0}")]
    Prepare(Arc<str>),
    #[error("publish failed: {0}")]
    Publish(Arc<str>),
    #[error("fetch exceeded max_bytes={limit_bytes}: {url}")]
    FetchSizeExceeded { url: SourceUrl, limit_bytes: usize },
    #[error("cleanup failed in phase {phase:?}: {source}")]
    Cleanup {
        phase: Phase,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}
```

At the bottom of `error.rs`, add the size assertion:

```rust
const _: () = {
    if std::mem::size_of::<RecipeError>() > 32 {
        panic!("RecipeError grew past 32 bytes; revisit boxed payloads");
    }
};
```

If the assertion fails on the chosen platform, refactor `RecipeError` so its
largest payloads remain behind boxed or reference-counted indirection and rerun.

Stage C validation: `cargo check -p chutoro-bench-datasets`;
`cargo clippy -p chutoro-bench-datasets -- -D warnings` proves the
`result_large_err` lint does not fire when the trait methods return
`Result<_, RecipeError>`.

### Stage D: port traits (M4)

Create `chutoro-bench-datasets/src/ports/mod.rs`, then one file per port.

`fetcher.rs`:

```rust
pub trait Fetcher: Send + Sync {
    /// Fetch `url` and return the bytes read.
    ///
    /// Implementations must abort with [`RecipeError::FetchSizeExceeded`] when
    /// the response size exceeds `max_bytes`. The cap is mandatory; there is
    /// no path that returns more than `max_bytes` bytes successfully.
    fn fetch_bytes(
        &self,
        url: &SourceUrl,
        max_bytes: usize,
    ) -> Result<bytes::Bytes, RecipeError>;

    /// Fetch a sequence of URLs.
    ///
    /// The default implementation iterates serially. Adapters with native
    /// parallelism (`10.1.2`) override this and may emit results out of order
    /// provided each result is keyed by its `SourceUrl`.
    fn fetch_many<'a>(
        &'a self,
        urls: &'a [(SourceUrl, usize)],
    ) -> Box<dyn Iterator<Item = (SourceUrl, Result<bytes::Bytes, RecipeError>)> + 'a>
    {
        Box::new(urls.iter().map(|(u, m)| (u.clone(), self.fetch_bytes(u, *m))))
    }
}
```

`storage.rs` defines the cache semantics:
`put(&self, key: &CacheKey, bytes: &[u8]) -> Result<(), RecipeError>` and
`get(&self, key: &CacheKey) -> Result<Option<bytes::Bytes>, RecipeError>`. The
doc comment states: `Storage` is a *mutable cache*; later writes overwrite
earlier ones; there is no version conditioning.

`publisher.rs` defines the write-once sink:
`publish(&self, key: &ObjectKey, bytes: &[u8]) -> Result<(), RecipeError>`. The
doc comment states: `Publisher` is a *write-once content-addressed sink*;
adapters added in `10.1.4` may reject re-publishes via optimistic concurrency.

Each port file ends with a `#[non_exhaustive] pub enum *Error` only if a more
granular error than `RecipeError` is needed for adapter implementers; the
default is to surface `RecipeError` directly.

Stage D validation: `cargo check -p chutoro-bench-datasets`; gates pass.

### Stage E: the `DatasetRecipe` trait and the `run_recipe` driver (M5)

Create `chutoro-bench-datasets/src/context.rs`:

```rust
pub struct RecipeContext<'a> {
    fetcher: &'a dyn Fetcher,
    storage: &'a dyn Storage,
    publisher: &'a dyn Publisher,
}

impl<'a> RecipeContext<'a> {
    pub fn new(
        fetcher: &'a dyn Fetcher,
        storage: &'a dyn Storage,
        publisher: &'a dyn Publisher,
    ) -> Self {
        Self { fetcher, storage, publisher }
    }

    pub fn fetcher(&self) -> &dyn Fetcher { self.fetcher }
    pub fn storage(&self) -> &dyn Storage { self.storage }
    pub fn publisher(&self) -> &dyn Publisher { self.publisher }
}
```

Create `chutoro-bench-datasets/src/recipe.rs`:

```rust
pub trait DatasetRecipe: Send + Sync {
    type Fetched: Send + Sync;
    type Validated: Send + Sync;
    type Prepared: Send + Sync;
    type Published: PublishedArtefact;

    fn id(&self) -> RecipeId;
    fn version(&self) -> RecipeVersion;
    fn info(&self) -> DatasetInfo;
    fn sources(&self) -> &[SourceSpec];

    fn fetch(&self, ctx: &RecipeContext<'_>) -> Result<Self::Fetched, RecipeError>;
    fn validate(
        &self,
        ctx: &RecipeContext<'_>,
        fetched: Self::Fetched,
    ) -> Result<Self::Validated, RecipeError>;
    fn prepare(
        &self,
        ctx: &RecipeContext<'_>,
        validated: Self::Validated,
    ) -> Result<Self::Prepared, RecipeError>;
    fn publish(
        &self,
        ctx: &RecipeContext<'_>,
        prepared: Self::Prepared,
    ) -> Result<Self::Published, RecipeError>;

    /// Roll back side effects of a partial run.
    ///
    /// Default: `Ok(())`. Recipes whose `prepare` writes intermediates to
    /// `Storage` must override this method to clean those entries up.
    fn cleanup(
        &self,
        _ctx: &RecipeContext<'_>,
        _partial: PartialState,
    ) -> Result<(), RecipeError> {
        Ok(())
    }
}
```

Create `chutoro-bench-datasets/src/driver.rs`:

```rust
#[tracing::instrument(skip_all, fields(recipe_id = %recipe.id(), version = %recipe.version()))]
pub fn run_recipe<R: DatasetRecipe>(
    recipe: &R,
    ctx: &RecipeContext<'_>,
) -> Result<R::Published, RecipeError> {
    let fetched = run_phase(Phase::Fetch, recipe, ctx, None, |c| recipe.fetch(c))?;
    let validated = run_phase(Phase::Validate, recipe, ctx, Some(Phase::Fetch),
        |c| recipe.validate(c, fetched))?;
    let prepared = run_phase(Phase::Prepare, recipe, ctx, Some(Phase::Validate),
        |c| recipe.prepare(c, validated))?;
    let published = run_phase(Phase::Publish, recipe, ctx, Some(Phase::Prepare),
        |c| recipe.publish(c, prepared))?;
    Ok(published)
}
```

The `run_phase` helper owns the per-phase `tracing` span, the cleanup
invocation on error, and the conversion of any port-private errors into
`RecipeError`. Keep `run_phase` private to the driver module.

Compile-fail doctest in `recipe.rs`:

```rust
/// Phases cannot be skipped:
/// ```compile_fail
/// # use chutoro_bench_datasets::{DatasetRecipe, RecipeContext};
/// fn assert_skip_fails<R: DatasetRecipe>(r: &R, c: &RecipeContext<'_>) {
///     let _ = r.prepare(c, r.fetch(c).unwrap());
/// }
/// ```
```

Stage E validation: `cargo check -p chutoro-bench-datasets` succeeds.

### Stage F: in-memory and filesystem test doubles (M6)

Create `chutoro-bench-datasets/src/testing/` gated by
`#[cfg(any(test, feature = "testing"))]`. Files:

- `in_memory.rs`: `InMemoryFetcher` backed by a
  `HashMap<SourceUrl, bytes::Bytes>`; honour `max_bytes` and emit
  `FetchSizeExceeded` when the payload exceeds the cap. `InMemoryStorage`
  backed by `Mutex<HashMap<...>>`; document the `Mutex` implementation in the
  doc comment. `InMemoryPublisher` backed by `Mutex<HashMap<ObjectKey, Bytes>>`
  with an `into_records()` method that the tests consume to assert what was
  published.
- `filesystem.rs`: `FilesystemFetcher` reads bytes from a root
  `camino::Utf8PathBuf`; the `SourceUrl::file://...` scheme maps directly to a
  relative path under the root. Emits `FetchSizeExceeded` after streaming
  reaches the cap.
- `stub_recipe.rs`: `StubRecipe` with `Fetched = Vec<bytes::Bytes>`,
  `Validated = Vec<bytes::Bytes>`, `Prepared = Vec<bytes::Bytes>`,
  `Published = StubPublished`. `StubPublished` implements `PublishedArtefact`
  with a fixed `manifest_uri` derived from the recipe id and a
  `manifest_digest` filled with the test-only zero digest.

Stage F validation: `cargo check -p chutoro-bench-datasets --features testing`;
gates pass.

### Stage G: unit, behavioural, and property tests (M7)

Tests live under `chutoro-bench-datasets/tests/`:

- `recipe_lifecycle.rs` (rstest): cases for happy path (in-memory ports),
  each phase's failure path, and `FetchSizeExceeded` propagation. Use
  `tracing-test` to assert one span per phase.
- `recipe_bdd.rs` (rstest-bdd): scenarios written once, parametrized over a
  `FetcherAdapter` rstest fixture that yields `InMemoryFetcher` and
  `FilesystemFetcher` in turn. Scenarios cover: `Given` a recipe with two
  sources, `When` fetch fails for one, `Then` validate is not invoked and the
  error preserves the failing `SourceUrl`. `Given` a successful preparation,
  `When` publish writes to the publisher, `Then` the published bytes match
  `prepare`'s output.
- `recipe_proptest.rs`: assert that for any permutation of `SourceSpec`
  inputs supplied to `StubRecipe`, the fetcher receives them in the order
  declared by `sources()`. Use `proptest` configuration
  `Config { cases: 256, ..Config::default() }`. Record any minimal failing
  input under `proptest-regressions/`.

Stage G validation: `cargo test -p chutoro-bench-datasets --features testing`;
`make test`; gates pass.

### Stage H: ADR, documentation, and roadmap update (M8)

Create `docs/adr-004-bench-dataset-recipe-trait.md` using the Y-statement
format (see `arch-decision-records` skill). Capture:

- the typestate-style typed handoffs decision;
- the sync-first decision and the explicit async-sibling forward-compatibility
  constraints (`Send + 'static` on associated types; driver owns runtime);
- the deferred manifest schema decision via `PublishedArtefact`;
- the deferred `Extractor` port decision;
- the partial-failure cleanup contract;
- the `Storage` vs `Publisher` semantic boundary.

Update `docs/developers-guide.md`:

- Add a new top-level section `## Benchmark dataset recipes` after the
  existing `## Benchmarks` section.
- Describe the crate, the four phases, the ports, and the in-memory plus
  filesystem test doubles.
- Link to the ADR and to `docs/benchmark-dataset-retrieval.md` §3.1.

Update `docs/users-guide.md` to add a one-paragraph note that the
`chutoro-bench-datasets` crate exists for benchmark dataset preparation, is not
part of the public clustering surface, and is consumed by the matrix benchmark
framework (roadmap §10.2).

Update `docs/benchmark-dataset-retrieval.md`: add a one-line cross-reference to
the new ADR under §3.1.

Stage H validation: `make markdownlint`; `make fmt`; gates pass.

### Stage I: final validation (M9)

Run `make check-fmt`, `make lint`, `make test`. Capture each to
`/tmp/check-fmt-chutoro-10-1-1.out`, `/tmp/lint-chutoro-10-1-1.out`, and
`/tmp/test-chutoro-10-1-1.out` respectively (the `tee` pattern documented in
`AGENTS.md`). Run `coderabbit review --agent` and resolve all concerns.

Edit `docs/roadmap.md` and tick item `10.1.1` as `[x]` with the completion date.

## Concrete steps

The commands below assume the working directory is the repository root unless
stated otherwise.

Stage A scaffold:

```sh
set -o pipefail
cargo check -p chutoro-bench-datasets 2>&1 \
  | tee /tmp/check-chutoro-10-1-1.out
```

After Stage C, prove the size invariant:

```sh
set -o pipefail
cargo clippy -p chutoro-bench-datasets --all-targets --all-features \
  -- -D warnings 2>&1 \
  | tee /tmp/lint-chutoro-10-1-1.out
```

After Stage G, run the full test suite:

```sh
set -o pipefail
cargo test -p chutoro-bench-datasets --features testing 2>&1 \
  | tee /tmp/test-chutoro-bench-datasets-10-1-1.out
```

Before each major milestone close-out:

```sh
set -o pipefail
make check-fmt 2>&1 | tee /tmp/check-fmt-chutoro-10-1-1.out
make lint      2>&1 | tee /tmp/lint-chutoro-10-1-1.out
make test      2>&1 | tee /tmp/test-chutoro-10-1-1.out
coderabbit review --agent
```

If `coderabbit review --agent` reports concerns that affect correctness, public
API clarity, or validation, resolve them before proceeding. Record any
intentionally deferred items under `Decision log`.

## Validation and acceptance

Acceptance is observable behaviour, not a code count.

- `cargo doc -p chutoro-bench-datasets` builds without warnings; the crate
  level page lists `DatasetRecipe`, `run_recipe`, the ports, `RecipeContext`,
  `DatasetInfo`, and `PublishedArtefact`.
- `cargo test -p chutoro-bench-datasets --features testing` passes every
  rstest case, every rstest-bdd scenario, and every proptest case. The
  rstest-bdd suite runs each scenario twice (once per fetcher adapter) and both
  invocations pass.
- The compile-fail doctest in `recipe.rs` is checked by `cargo test --doc`.
- `cargo clippy -p chutoro-bench-datasets --all-targets --all-features -- -D warnings`
  is clean. In particular, `clippy::result_large_err` does not fire on any
  trait method.
- `cargo check -p chutoro-bench-datasets` (no `testing` feature) compiles
  cleanly; the `testing` module is invisible to non-test consumers.
- `make check-fmt`, `make lint`, `make test` all succeed at every milestone
  close-out.
- `coderabbit review --agent` reports no unresolved concerns at the
  milestone boundaries.
- The roadmap item `10.1.1` is ticked as `[x]` in `docs/roadmap.md`.
- The ADR is present at `docs/adr-004-bench-dataset-recipe-trait.md`.
- `docs/users-guide.md` and `docs/developers-guide.md` reflect the new crate.

Quality criteria:

- Tests: all unit, behavioural, doctest, and property tests pass on stable
  Rust.
- Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
- Format: `cargo fmt --workspace -- --check`.
- Performance: not applicable; this milestone introduces no hot path.
- Security: no new network surface in this milestone. Verify that no
  production dependency was added beyond the listed allowlist.

Quality method:

- Run each milestone's gates locally via
  `make check-fmt && make lint && make test` capturing output to `/tmp` per
  `AGENTS.md`.
- Resolve every `coderabbit review --agent` concern before proceeding.

## Idempotence and recovery

- Every step in this plan is locally idempotent: the workspace member entry
  is set-only; `cargo` does not duplicate registrations; documentation edits
  are pure replacement of explicit anchors.
- If a milestone has to be redone, revert the milestone's commits (the plan
  commits per milestone, so revert is well scoped) and re-run from the start of
  the milestone. Do not amend prior commits.
- If `make lint` or `make test` fails after a stage, the recovery is to fix
  the underlying cause and re-run the gate; do not silence the lint or
  attribute-allow the failure.
- If `coderabbit review --agent` flags a substantive issue late in the plan,
  prefer landing the fix as a new commit on the same branch over rewriting
  history.

## Artefacts and notes

Capture transcripts of `cargo test`, `cargo clippy`, and
`coderabbit review --agent` invocations in `/tmp` per the path convention in
`AGENTS.md`. Do not commit them; reference them in the `Progress` section if a
milestone needs explanation.

## Interfaces and dependencies

At the end of this milestone, the following items must exist with the listed
signatures.

In `chutoro-bench-datasets/src/recipe.rs`:

```rust
pub trait DatasetRecipe: Send + Sync {
    type Fetched: Send + Sync;
    type Validated: Send + Sync;
    type Prepared: Send + Sync;
    type Published: PublishedArtefact;

    fn id(&self) -> RecipeId;
    fn version(&self) -> RecipeVersion;
    fn info(&self) -> DatasetInfo;
    fn sources(&self) -> &[SourceSpec];

    fn fetch(&self, ctx: &RecipeContext<'_>) -> Result<Self::Fetched, RecipeError>;
    fn validate(&self, ctx: &RecipeContext<'_>, fetched: Self::Fetched)
        -> Result<Self::Validated, RecipeError>;
    fn prepare(&self, ctx: &RecipeContext<'_>, validated: Self::Validated)
        -> Result<Self::Prepared, RecipeError>;
    fn publish(&self, ctx: &RecipeContext<'_>, prepared: Self::Prepared)
        -> Result<Self::Published, RecipeError>;

    fn cleanup(&self, _ctx: &RecipeContext<'_>, _partial: PartialState)
        -> Result<(), RecipeError> { Ok(()) }
}
```

In `chutoro-bench-datasets/src/driver.rs`:

```rust
pub fn run_recipe<R: DatasetRecipe>(
    recipe: &R,
    ctx: &RecipeContext<'_>,
) -> Result<R::Published, RecipeError>;
```

In `chutoro-bench-datasets/src/ports/fetcher.rs`:

```rust
pub trait Fetcher: Send + Sync {
    fn fetch_bytes(&self, url: &SourceUrl, max_bytes: usize)
        -> Result<bytes::Bytes, RecipeError>;
    fn fetch_many<'a>(
        &'a self,
        urls: &'a [(SourceUrl, usize)],
    ) -> Box<dyn Iterator<Item = (SourceUrl, Result<bytes::Bytes, RecipeError>)> + 'a>;
}
```

In `chutoro-bench-datasets/src/ports/storage.rs`:

```rust
pub trait Storage: Send + Sync {
    fn put(&self, key: &CacheKey, bytes: &[u8]) -> Result<(), RecipeError>;
    fn get(&self, key: &CacheKey) -> Result<Option<bytes::Bytes>, RecipeError>;
}
```

In `chutoro-bench-datasets/src/ports/publisher.rs`:

```rust
pub trait Publisher: Send + Sync {
    fn publish(&self, key: &ObjectKey, bytes: &[u8]) -> Result<(), RecipeError>;
}
```

In `chutoro-bench-datasets/src/published.rs`:

```rust
pub trait PublishedArtefact: sealed::Sealed + Send + Sync {
    fn manifest_uri(&self) -> &camino::Utf8Path;
    fn manifest_digest(&self) -> &ManifestDigest;
}
```

Dependencies added to `chutoro-bench-datasets`:

- `thiserror` (`2.0`) for `RecipeError`.
- `tracing` (`0.1`) for phase spans.
- `camino` (`1.1`) for UTF-8 paths in the manifest URI and cache key.
- `bytes` (`1.7`) for the `bytes::Bytes` payload type.

Dev-only:

- `rstest` (`0.26`), `rstest-bdd` (workspace pin), `proptest` (`1.5`),
  `mockall` (`0.13`), `tempfile` (`3.10`), and `tracing-test` for span
  assertions.

## Revision note

2026-07-14: Final revision reconciles the completed implementation and review
follow-ups with the concurrency, adapter, and compatibility decisions recorded
during delivery.
