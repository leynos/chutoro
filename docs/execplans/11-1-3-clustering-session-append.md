# Implement `ClusteringSession::append`

This ExecPlan (execution plan) is a living document. The sections
`Constraints`, `Tolerances`, `Risks`, `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work
proceeds.

Status: DRAFT

Implementation must not begin until this plan is approved.

## Purpose / big picture

Deliver roadmap item `11.1.3` by giving `ClusteringSession` its first mutable
operation: `append(&mut self, indices: &[usize])`. A caller who already owns a
data source with appendable or externally managed rows will be able to insert a
batch of point indices into the session's live CPU Hierarchical Navigable Small
World (HNSW) index. Each point insertion must call the public edge-harvesting
path, `CpuHnsw::insert_harvesting`, and store all returned `CandidateEdge`
values in the session's `pending_edges` buffer for later refresh work.

Success is visible when a session built through `ChutoroBuilder::build_session`
can append a slice such as `&[0, 1, 2]`, `session.point_count()` then reports
`3`, and unit tests prove that `pending_edges` contains the exact union of
edges returned by the per-index `insert_harvesting` calls. Appending an empty
slice is a no-op. Appending the first point may harvest zero edges, because an
empty HNSW index has no prior neighbours. Duplicate or out-of-bounds indices
must return an error through the existing public `ChutoroError` surface.

This plan only authorizes the append and edge-buffering step. It does not
authorize incremental core-distance calculation, minimum spanning tree (MST)
refresh, labels, seeded-session constructors, deletion, compaction, or stable
cluster identity.

## Constraints

- Keep scope limited to roadmap item `11.1.3`: implement
  `ClusteringSession::append`, accumulate harvested `CandidateEdge` values in
  session state, expose only the public API needed for append, and update
  documentation affected by append becoming available.
- Do not implement `ClusteringSession::refresh`, `refresh_full`, `labels`,
  `from_source`, `new_empty`, core-distance recomputation, automatic
  `refresh_every_n` triggering, MST merging, or historical-edge retention.
- Preserve existing `CpuHnsw::insert`, `CpuHnsw::insert_harvesting`,
  `ChutoroBuilder::build`, and `Chutoro::run` behaviour.
- Route every insertion through `CpuHnsw::insert_harvesting`; do not duplicate
  HNSW insertion or edge-harvesting logic inside the session module.
- Keep the session CPU-only behind the existing `cpu` feature gate. Do not add
  a GPU session path in this item.
- Do not add a new production dependency. The original task instructions
  authorize `rstest-bdd` where applicable, so adding it as a dev-dependency
  for a justified behavioural test does not require a separate approval gate.
- Keep files below 400 lines. If `chutoro-core/src/session/mod.rs` or
  `chutoro-core/src/session/tests.rs` would exceed that limit, split the module
  before adding more code.
- Public Rustdoc and Markdown updates must use en-GB Oxford spelling and
  follow `docs/documentation-style-guide.md` and
  `docs/rust-doctest-dry-guide.md`.
- Use `leta` for symbol-oriented code navigation while implementing. Use
  `rust-router` to select any additional Rust-specific skill if a borrow,
  error, API, or layout problem emerges.
- Treat hexagonal architecture as a boundary check, not a pattern transplant:
  session orchestration may call the HNSW port-like public API, but it must not
  copy adapter internals or make downstream refresh policy depend on HNSW
  private implementation details.

## Tolerances (exception triggers)

- Scope: if satisfying `11.1.3` requires modifying more than 8 files or more
  than 350 net lines outside this ExecPlan, stop and escalate.
- Interface: if `append` cannot be implemented as
  `pub fn append(&mut self, indices: &[usize]) -> Result<()>`, stop and present
  alternatives with trade-offs.
- Semantics: if all-or-nothing rollback is required to satisfy review feedback,
  stop and escalate. The current HNSW mutation API does not expose rollback.
- Dependencies: if implementation requires adding any production crate or any
  test-only crate other than the requested `rstest-bdd`, stop and ask for
  approval.
- Error model: if a new public error enum or new `ChutoroError` variant appears
  necessary, stop and justify the compatibility impact before proceeding.
- Testing: if `append` cannot be validated without a public
  `pending_edges` accessor, stop and decide whether to add an accessor or keep
  validation in module-level unit tests.
- Formal verification: if the implementation introduces a new pure helper with
  non-trivial ordering, set, or transition invariants, pause and decide whether
  `proptest`, Kani, or Verus is the right level of rigour.
- Validation: if `make check-fmt`, `make lint`, or `make test` still fails
  after two repair attempts caused by this change, stop and escalate with the
  captured `/tmp` logs.
- Review: after each major milestone, run `coderabbit review --agent`. If it
  reports concerns that affect correctness, public API clarity, or validation,
  address them or record why they are out of scope before moving on.

## Risks

- Risk: `append(indices)` inserts indices into an existing immutable
  `DataSource`; it does not push raw records into the source. Severity: medium.
  Likelihood: high. Mitigation: document that callers must manage source
  storage and pass indices that are already valid for that source.

- Risk: partial failure semantics are easy to misunderstand.
  Severity: high. Likelihood: medium. Mitigation: define `append` as fail-fast
  with partial progress: successful insertions before the failing index remain
  in the index, and their harvested edges remain in `pending_edges`. Document
  and test this explicitly.

- Risk: duplicate indices can fail after earlier indices in the same slice have
  already mutated the index. Severity: medium. Likelihood: medium. Mitigation:
  add tests for duplicate and mixed-success slices and make the public docs
  clear that callers wanting all-or-nothing behaviour must validate slices
  before calling `append`.

- Risk: out-of-bounds indices may surface from the HNSW insertion path as
  `HnswError::DataSource` rather than a session-specific error. Severity:
  medium. Likelihood: medium. Mitigation: map `HnswError` through a small
  helper that preserves data-source failures as `ChutoroError::DataSource` and
  maps structural HNSW failures to `ChutoroError::CpuHnswFailure`.

- Risk: tests that assert exact `CandidateEdge` ordering may be brittle if
  HNSW internals later change. Severity: low. Likelihood: medium. Mitigation:
  compare multisets or sequence of direct `insert_harvesting` results against
  session `pending_edges`, rather than inventing hard-coded expected edges.

- Risk: adding BDD infrastructure for a narrow library API could expand scope
  without improving confidence. Severity: low. Likelihood: medium. Mitigation:
  because the original instructions explicitly mention `rstest-bdd`, the
  implementer may add it as a dev-dependency when a behavioural scenario is
  useful, but should avoid empty Gherkin scaffolding that merely repeats unit
  tests.

## Progress

- [x] (2026-05-19 00:00Z) Loaded the requested `leta`, `rust-router`, `kani`,
  `verus`, `execplans`, `hexagonal-architecture`, `firecrawl-mcp`,
  `commit-message`, and `pr-creation` skills.
- [x] (2026-05-19 00:00Z) Created a Leta workspace for this repository.
- [x] (2026-05-19 00:00Z) Renamed the local branch to
  `11-1-3-clustering-session-append`. The remote branch did not exist yet, so
  upstream tracking will be established on first push.
- [x] (2026-05-19 00:00Z) Reviewed `AGENTS.md`, `Makefile`,
  `docs/roadmap.md`, `docs/chutoro-design.md` §12.3-§12.4,
  `docs/property-testing-design.md`,
  `docs/rust-testing-with-rstest-fixtures.md`,
  `docs/rust-doctest-dry-guide.md`,
  `docs/complexity-antipatterns-and-refactoring-strategies.md`,
  `docs/users-guide.md`, and `docs/developers-guide.md`.
- [x] (2026-05-19 00:00Z) Used a Wyvern agent team to inspect the implementation
  surface and validation surface for `11.1.3`.
- [x] (2026-05-19 00:00Z) Used Firecrawl to check prior art and tooling gaps:
  the original flexible-clustering HNSW implementation, the FISHDBC search
  results, and `rstest-bdd` crate behaviour.
- [x] (2026-05-19 00:00Z) Drafted this ExecPlan for approval before
  implementation begins.
- [x] (2026-05-19 00:00Z) Validated the planning-only branch with
  plan-specific Markdown linting, `make check-fmt`, `make lint`, `make test`,
  and `coderabbit review --agent`.
- [x] (2026-05-20 00:00Z) Revised the dependency and validation gates to
  reflect that `rstest-bdd` was explicitly authorised by the task instructions.
- [ ] Receive explicit approval to implement this plan.
- [ ] Implement `ClusteringSession::append` and supporting error mapping.
- [ ] Add unit and property tests for append behaviour.
- [ ] Update user, developer, design, and roadmap documentation.
- [ ] Run formatting, linting, tests, and CodeRabbit review.
- [ ] Commit the implementation and push the branch.

## Surprises & Discoveries

- Observation: `ClusteringSession` already owns a `CpuHnsw` and reports
  `point_count()` by delegating to `index.len()`, so append can produce an
  observable state change without adding a new public index accessor.

- Observation: `ClusteringSession` fields that were reserved for later mutable
  work currently carry leading underscores, including `_pending_edges` and
  `_source`. Implementing append should rename only the fields it actually uses
  to `pending_edges` and `source`; unrelated reserved fields can keep their
  underscore prefixes until their roadmap items use them.

- Observation: `docs/users-guide.md` and `docs/developers-guide.md` both state
  that append is not yet available. These statements must change when this
  roadmap item is implemented, while refresh and batch bootstrap remain future
  work.

- Observation: there are no `.feature` files and no existing `rstest-bdd`
  dependency or usage in this repository slice. Firecrawl confirmed
  `rstest-bdd` is designed to run under `cargo test` and reuse `rstest`
  fixtures. The original task instructions authorize `rstest-bdd`; the
  implementation should add it only if a behavioural append scenario gives
  reviewers stronger signal than the unit and property tests.

- Observation: the upstream flexible-clustering HNSW implementation mutates
  the graph as each item is added and does not suggest a transactional append
  rollback model. That supports documenting fail-fast partial progress rather
  than inventing a rollback layer for this roadmap item.

- Observation: `make fmt` currently reaches pre-existing Markdown line-length
  violations in unrelated documentation after running repository-wide Markdown
  formatting. The new ExecPlan itself passes direct `markdownlint-cli2`.

## Decision Log

- Decision: implement `append` as
  `pub fn append(&mut self, indices: &[usize]) -> Result<()>`. Rationale: the
  method mutates `pending_edges` and session state, and the roadmap asks for a
  slice of new point indices rather than ownership of source records.

- Decision: `append` is fail-fast with partial progress.
  Rationale: `CpuHnsw::insert_harvesting` mutates the index as it goes, and the
  current graph API has no transactional rollback. Pre-validating every index
  would still not make HNSW insertion failures rollback-safe, so the honest
  contract is to preserve successful earlier inserts and return the first
  failure.

- Decision: do not expose `pending_edges` as a public accessor in this item
  unless implementation proves it necessary. Rationale: `pending_edges` is
  internal refresh state for later `11.2.x` roadmap items. Unit tests in the
  `session` module can inspect private fields to prove the acceptance criteria
  without expanding the public API.

- Decision: map `HnswError::DataSource { .. }` to
  `ChutoroError::DataSource` and map other `HnswError` variants to
  `ChutoroError::CpuHnswFailure`. Rationale: this preserves the public error
  taxonomy already used by the batch path and keeps source failures inspectable
  through `data_source_code()`.

- Decision: use `rstest` and a narrow `proptest` sequence property as the
  required validation baseline. Add `rstest-bdd` if the implementation can
  express a useful behavioural append scenario without duplicating unit tests.
  Rationale: the user explicitly authorised `rstest-bdd`, but append is still a
  small state mutation best proven by comparing session behaviour to direct
  HNSW insertion. Kani and Verus remain reserved for bounded structural
  invariants or pure helper proofs; no such helper is planned.

- Decision: keep automatic `refresh_every_n` triggering out of this item.
  Rationale: roadmap item `11.2.3` explicitly owns automatic refresh behaviour.
  `11.1.3` must accumulate pending edges only.

## Implementation Plan

Begin by refreshing local context with these commands:

```sh
git branch --show-current
git status --short --branch
leta workspace add "$PWD"
leta show ClusteringSession -n 10
leta show CpuHnsw.insert_harvesting -n 10
```

The expected branch is `11-1-3-clustering-session-append`. If the branch is not
correct, stop and fix the branch before editing.

### Stage A: add failing tests for append

Work in `chutoro-core/src/session/tests.rs`. Add `rstest` cases covering these
scenarios before adding production code:

- `append_empty_slice_is_noop`: build a session over a non-empty
  `SessionTestSource`, call `append(&[])`, and assert `point_count() == 0`,
  `snapshot_version() == 0`, and `pending_edges` remains empty.
- `append_single_index_increases_point_count`: append `[0]` and assert
  `point_count() == 1`; accept zero pending edges for this first insertion.
- `append_batch_accumulates_direct_harvested_edges`: build one session and one
  direct `CpuHnsw` with the same `HnswParams`, append a batch such as
  `[0, 1, 2, 3]`, and compare the session's private `pending_edges` with the
  concatenated vectors returned by calling
  `direct.insert_harvesting(index, source.as_ref())` for the same indices.
- `append_rejects_duplicate_index`: append `[0]`, then append `[0]` again and
  assert a `ChutoroError::CpuHnswFailure`.
- `append_rejects_out_of_bounds_index`: append an index equal to
  `source.len()` and assert `ChutoroError::DataSource` with
  `DataSourceErrorCode::OutOfBounds`.
- `append_failure_preserves_prior_successes`: append `[0, source.len()]`,
  assert the second index fails, and assert `point_count() == 1`.
- `append_does_not_publish_label_snapshot`: append valid indices and assert
  `snapshot_version() == 0`, because refresh owns snapshot publication.

Add a small `proptest` case that generates unique append sequences within a
bounded source length, appends them once, and asserts `point_count()` equals
the number of unique appended indices. Keep the bound small enough to stay fast
in normal `make test` runs, for example source lengths up to 32.

If the tests need deterministic HNSW output, use `HnswParams::with_rng_seed` as
the existing tests do.

If a behavioural test can add useful reviewer signal, add `rstest-bdd` as a
dev-dependency and create a small feature file for the append lifecycle. Keep
the scenario focused on observable behaviour: build an empty session, append a
slice of valid indices, observe `point_count()`, and verify that append does
not publish a label snapshot. Do not add BDD steps that only duplicate the
private `pending_edges` unit assertions.

### Stage B: implement `append`

Work in `chutoro-core/src/session/mod.rs`.

Update the module comment so it no longer says the session is only an empty
shell. It should say that the session supports append-only HNSW insertion and
buffers harvested edges, while refresh and labels remain future work.

Rename the used reserved fields:

```rust
_pending_edges: Vec<CandidateEdge>,
_source: Arc<D>,
```

to:

```rust
pending_edges: Vec<CandidateEdge>,
source: Arc<D>,
```

Leave unrelated reserved fields with underscores until their roadmap items use
them.

Add a private helper if needed:

```rust
fn map_hnsw_error(&self, error: HnswError) -> ChutoroError
```

or a free function with the source name as an argument. It should preserve
`HnswError::DataSource` as `ChutoroError::DataSource` and map other HNSW
failures to `ChutoroError::CpuHnswFailure` using `error.code().as_str()` and
`error.to_string()`.

Add the public method:

```rust
pub fn append(&mut self, indices: &[usize]) -> Result<()>
```

For each `index` in `indices`, call:

```rust
let edges = self.index.insert_harvesting(index, self.source.as_ref())?;
self.pending_edges.extend(edges);
```

with the actual error mapping helper applied to the `?` path. The method should
not update `snapshot_version`, `_last_refresh_len`, `_core_distances`,
`_mst_edges`, `_historical_edges`, or `_labels`.

Update `ClusteringSession` Rustdoc to show a small append example. Use a
fallible doctest style from `docs/rust-doctest-dry-guide.md`; do not use
`.unwrap()` or `.expect()` in public examples.

### Stage C: update public surface tests

Work in:

- `chutoro-core/tests/trybuild/session_api_cpu_enabled.rs`
- `chutoro-core/tests/session_api_surface.rs`

Update the CPU-enabled trybuild fixture so it calls `session.append(&[0])?` or
equivalent. Keep the non-`Send + Sync` compile-fail fixture unchanged unless
the compiler output changes because of the new method. If the stderr fixture
changes, update it with the new expected compiler output and explain why in the
`Surprises & Discoveries` section.

### Stage D: update documentation

Update `docs/chutoro-design.md` §12.3 and §12.4 to record that roadmap item
`11.1.3` has implemented append-only insertion through
`CpuHnsw::insert_harvesting`, and that refresh, labels, and core-distance
updates remain future work.

Update `docs/users-guide.md`:

- Add `append(&[usize])` to the session lifecycle example.
- Explain that append accepts indices into the backing `DataSource`, not raw
  records.
- Explain fail-fast partial progress.
- Remove or narrow the statement that append is not available. Refresh and
  full batch bootstrap should still be documented as future work.

Update `docs/developers-guide.md`:

- Add `pub fn append(&mut self, indices: &[usize]) -> Result<()>` to the
  session API list.
- Document the internal convention that append stores harvested edges in
  `pending_edges` and does not trigger refresh or snapshot version updates.

Update `docs/roadmap.md` only after validation passes, changing `11.1.3` from
`[ ]` to `[x]`. Do not mark any later roadmap item done.

### Stage E: validate and review

Run validations sequentially and capture logs with `tee` under `/tmp`:

```sh
make fmt 2>&1 | tee "/tmp/fmt-chutoro-$(git branch --show-current).out"
make markdownlint 2>&1 | tee \
  "/tmp/markdownlint-chutoro-$(git branch --show-current).out"
make check-fmt 2>&1 | tee \
  "/tmp/check-fmt-chutoro-$(git branch --show-current).out"
make lint 2>&1 | tee "/tmp/lint-chutoro-$(git branch --show-current).out"
make test 2>&1 | tee "/tmp/test-chutoro-$(git branch --show-current).out"
coderabbit review --agent 2>&1 | tee \
  "/tmp/coderabbit-chutoro-$(git branch --show-current).out"
```

Run `make nixie` only if the implementation changes Mermaid diagrams. This plan
does not require new diagrams.

If `make fmt` changes Markdown generated by this plan or by documentation
updates, inspect those changes before committing.

### Stage F: commit and push

Commit only after the gates pass. Use the file-based commit-message workflow
from the `commit-message` skill:

```sh
git status --short
git diff -- docs chutoro-core
git add docs chutoro-core
COMMIT_MSG_DIR="$(mktemp -d)"
cat > "$COMMIT_MSG_DIR/COMMIT_MSG.md" << 'ENDOFMSG'
Implement session append edge harvesting

Add `ClusteringSession::append` so sessions can insert point indices through
the public HNSW edge-harvesting path and retain harvested candidate edges for
future refresh work.

Document the append-only contract, partial failure semantics, and the remaining
refresh limitations.
ENDOFMSG
git commit -F "$COMMIT_MSG_DIR/COMMIT_MSG.md"
rm -rf "$COMMIT_MSG_DIR"
```

Push the branch with upstream tracking:

```sh
git push -u origin 11-1-3-clustering-session-append
```

For this pre-implementation ExecPlan PR, commit only the plan file and open a
draft PR before any implementation begins.

## Validation Plan

The red phase must prove the new tests fail before `append` exists. The green
phase must pass targeted session tests before running full gates. Useful
targeted commands are:

```sh
cargo test -p chutoro-core session::tests::append --all-features
cargo test -p chutoro-core --test session_api_surface --all-features
```

The final required gates are:

```sh
make check-fmt
make lint
make test
```

Because this change edits Markdown, also run:

```sh
make fmt
make markdownlint
```

Expected successful evidence is that all commands exit with status `0`, the new
append tests pass, and CodeRabbit has no unresolved concerns that affect the
plan or implementation.

## References

- Roadmap item: `docs/roadmap.md` §11.1.3.
- Design source: `docs/chutoro-design.md` §12.3 and §12.4.
- Testing guidance: `docs/property-testing-design.md`,
  `docs/rust-testing-with-rstest-fixtures.md`, and
  `docs/rust-doctest-dry-guide.md`.
- Complexity guidance:
  `docs/complexity-antipatterns-and-refactoring-strategies.md`.
- Prior local plans:
  `docs/execplans/11-1-1-make-edge-harvesting-hnsw-insertion-path-public.md`
  and the `11.1.2` `SessionConfig` ExecPlan.
- Firecrawl prior art search: the upstream flexible-clustering HNSW source, the
  FISHDBC arXiv paper, and the `rstest-bdd` crates.io page.
- Skills signposted for implementation: `leta`, `rust-router`,
  `hexagonal-architecture`, `kani`, `verus`, `execplans`, `firecrawl-mcp`,
  `commit-message`, and `pr-creation`.

## Outcomes & Retrospective

No implementation has begun. This draft captures the intended behaviour, scope
boundaries, validation strategy, and approval gate for roadmap item `11.1.3`.
