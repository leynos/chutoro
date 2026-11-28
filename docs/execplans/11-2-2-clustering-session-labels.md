# 11.2.2 Publish label snapshots to readers without blocking the writer

This ExecPlan (execution plan) is a living document. The sections `Constraints`,
`Tolerances (exception triggers)`, `Risks`, `Progress`,
`Surprises & discoveries`, `Decision log`, `Outcomes & retrospective`,
`Conformance basis`, and `Verification plan` must be kept up to date as work
proceeds.

Status: DRAFT (awaiting design review and user approval)

Branch: `11-2-2-clustering-session-labels`

Roadmap item: `11.2.2` in `docs/roadmap.md`.

## Purpose / big picture

Chutoro is a Rust clustering library implementing FISHDBC (a hierarchical,
density-based clustering algorithm built on an approximate nearest-neighbour
graph). Its `ClusteringSession` type lets a caller add points to a live
clustering over time. The caller appends source indices with
`session.append(&[...])`, calls `session.refresh()` to recompute the
clustering, and reads the resulting flat labels (one cluster number per point)
with `session.labels()`, which returns an `Arc<Vec<usize>>` (a
reference-counted, immutable vector). Each successful refresh "publishes" a new
label snapshot and advances `session.snapshot_version()` by one.

Roadmap item 11.2.1 already shipped `labels(&self) -> Arc<Vec<usize>>`, because
its acceptance criteria were phrased in terms of that accessor (see
`docs/adrs/adr-0010-incremental-refresh-domain-boundary.md`). What 11.2.1 did
not deliver is the property this item names: reading the latest snapshot
**without blocking the writer**. Design §12.6 promises a single-writer,
multiple-reader model in which "readers never block the writer and vice versa".
Today that is false for any reader on another thread:

- `append` and `refresh` take `&mut self`, so the only way to share one session
  between a writer thread and reader threads is to wrap it in a lock, typically
  `Arc<RwLock<ClusteringSession<D>>>`. The repository's own concurrency tests
  (`chutoro-core/src/session/tests/concurrency.rs`) do exactly that.
- Under that lock, a reader calling `labels()` holds a read guard, so the
  writer's `refresh()` cannot start until every reader lets go, and every
  reader waits for the whole refresh (an O(E log E) rebuild) before it can read
  the previous snapshot it could perfectly well have used.
- A reader also cannot read `labels()` and `snapshot_version()` as one
  consistent pair without holding the lock across both calls, because they are
  stored in two separate fields.

After this change, a caller obtains a detached, cloneable reader handle and
gives it to any number of threads:

```rust,ignore
let mut session = ChutoroBuilder::new()
    .with_min_cluster_size(2)
    .build_session(source)?;
let reader = session.label_reader(); // LabelReader: Clone + Send + Sync + 'static

let dashboard = std::thread::spawn(move || {
    let snapshot = reader.snapshot(); // one consistent (version, labels) pair
    (snapshot.version(), snapshot.labels().len())
});

session.append(&[0, 1, 2, 3])?; // the writer keeps &mut access; no lock needed
session.refresh()?;             // publishes atomically; never waits for readers
```

Readers never hold a lock, never borrow the session, and never see a version
from one refresh paired with labels from another. Publishing a new snapshot is
a single atomic pointer swap that does not wait for readers, and a reader that
keeps an old snapshot keeps exactly that snapshot, unchanged, for as long as it
likes. The existing `session.labels()` keeps its signature and now reads the
same published cell.

Preserved behaviour: `labels()` still returns `Arc<Vec<usize>>`; its length
still equals `point_count()` after every successful refresh; the version still
advances by exactly one per successful refresh and saturates at `u64::MAX`;
publication remains all-or-nothing, so a failed refresh publishes nothing. The
one deliberate signature change is that `snapshot_version()` stops being a
`const fn`, because it now reads an atomically published cell.

### What this plan does and does not authorize

It authorizes two new public types (`LabelSnapshot`, `LabelReader`), two new
`ClusteringSession` methods (`label_reader`, `label_snapshot`), the removal of
`const` from `ClusteringSession::snapshot_version`, one new production
dependency (`arc-swap`, decision `D1`, which needs explicit user approval), and
one new ADR. It does not authorize count-triggered refresh (11.2.3),
`refresh_full` (11.2.4), historical-edge retention (11.2.5), the full snapshot
consistency suite (11.4.4), or the session state-machine suite (11.4.5). 11.4.4
and 11.4.5 build on the tests introduced here rather than being replaced by
them.

## Constraints

- Implementation begins only after the user explicitly approves this plan,
  including decision `D1` (the `arc-swap` dependency) or its std-only fallback.
- `ClusteringSession::labels(&self) -> Arc<Vec<usize>>` keeps its exact
  signature (roadmap wording and ADR-0010 commitment).
- Publication stays all-or-nothing: labels and version change together, only
  after every fallible step of `refresh` succeeds
  (`chutoro-core/src/session/refresh_impl.rs`, `publish_refresh`).
- Exactly one code path may publish: the session's own refresh. No public type
  may expose a way to publish or mutate a snapshot.
- No new `ChutoroError` variant. Nothing in this item can fail at run time.
- `arc-swap` must not appear in any public signature. It is an implementation
  detail behind private types, so it can be replaced without an API change.
- `arc-swap` is optional and enabled only by the `cpu` feature, matching the
  `cpu` gate on the whole session module (`chutoro-core/src/lib.rs`).
- Every dependency uses a caret requirement (`AGENTS.md`, dependency
  management).
- No code file may exceed 400 lines (`AGENTS.md`). `session/mod.rs` is 284
  lines today; new accessors go in a new file if `mod.rs` would cross 400.
- Every new module starts with a `//!` comment; every public item has Rustdoc
  with an example (`AGENTS.md`, `docs/rust-doctest-dry-guide.md`).
- No `unwrap`/`expect` outside tests, no lint suppressions without a scoped
  reason (`Cargo.toml` `[workspace.lints.clippy]`), and no `unsafe`.
- Tests must not mutate environment variables.
- Historical ExecPlans (`docs/execplans/11-2-1-*.md` and earlier) and accepted
  ADR-0010 must not be edited.
- Do not install, upgrade or downgrade Whitaker; `make lint` uses the installed
  wrapper.

## Tolerances (exception triggers)

Comparison base: `main` at `028c976`. Counting command, run from the repository
root:

```bash
git diff --numstat 028c976 -- . ':(exclude)docs/execplans/*'
```

Classification is whole-file, except where noted:

- Code: `*.rs` under `chutoro-core/` and `chutoro-compile-contracts/`,
  `.feature` files, and the fenced Rust program in
  `chutoro-core/docs/session-refresh-example.md` (hunk-based: the fence is
  executable test input).
- Configuration: `Cargo.toml` files and `Cargo.lock`.
- Documentation: other `*.md` under `docs/` and the prose of
  `chutoro-core/docs/session-refresh-example.md`.
- This ExecPlan is excluded from every count.

Limits, each measured as net lines (additions minus deletions) and enforced
independently:

- Files: if more than 24 non-ExecPlan files change, stop and escalate.
- Code: if code exceeds +1,200 net lines, stop and escalate.
- Configuration: if configuration exceeds +20 net lines, stop and escalate.
- Documentation: if documentation exceeds +500 net lines, stop and escalate.
- Interface: any public item beyond `LabelSnapshot`, `LabelReader`,
  `ClusteringSession::label_reader`, `ClusteringSession::label_snapshot`, and
  the `const` removal on `snapshot_version` requires escalation.
- Dependencies: any production dependency other than `arc-swap` (caret `1.9.2`,
  optional, `cpu`-gated), or any new dev-dependency, requires escalation.
- Architecture: changing the all-or-nothing publish rule, giving readers any
  mutation or publish capability, or adding a lock to the writer's publish path
  requires escalation.
- Iterations: if a gate still fails after three fix attempts on the same
  cause, stop and escalate.
- Verification: if the concurrency stress test is flaky (fails on re-run
  without a code change) even once, stop and escalate; do not add retries.
- Time: if a milestone takes more than six hours of active work, stop and
  escalate.
- Ambiguity: if the noise-label representation (issue #462) has to be settled
  to finish this item, stop and escalate; this plan treats label values as
  opaque.

Budget evidence (see `Context and orientation`): the closest predecessor,
11.2.1, set 14 files / +1,000 lines and finished at 43 files / +6,710
insertions after one approved increase. Its growth came from a new pure
pipeline, a Verus proof file, a build-script example loader and five milestones
of documentation — none of which recur here. 11.1.2, 11.1.3 and 11.1.4 stayed
inside 8–12 file limits with no recorded increase. This item adds one small
module, rewires two fields, and is test-heavy, so the estimate is about 14–18
files, +700–900 code lines (roughly two thirds tests), +6 configuration lines,
and +300–380 documentation lines. The limits carry about 30% contingency
because 11.2.1 showed that `-D warnings`, the 400-line cap and compile
contracts add files late.

## Risks

- Risk: the user declines a new production dependency.
  Severity: medium. Likelihood: medium. Mitigation: decision `D1` names a
  std-only fallback, `std::sync::RwLock<Arc<LabelSnapshot>>` with O(1) critical
  sections (one `Arc` clone or one pointer store). The public API, tests and
  documents are identical under either choice; only the private cell and the
  ADR's outcome change. The fallback weakens "never blocks" to "blocks for at
  most one pointer copy", which the documents must then say.
- Risk: `arc-swap` behaviour is misunderstood (for example, long-held guards
  slowing writers). Severity: medium. Likelihood: low. Mitigation: the private
  cell never returns a `Guard`; every read converts to an owned
  `Arc<LabelSnapshot>` immediately (`load_full`), which the `arc-swap`
  performance notes recommend for values held for a long time.
- Risk: the concurrency stress test is flaky or slow under compile-admission
  contention. Severity: medium. Likelihood: low. Mitigation: its assertions are
  schedule-independent (monotonicity and pairing hold for every interleaving);
  overlap counts are reported, never asserted; sizes are small (eight points,
  four readers, bounded loops); a `recv_timeout` turns a hang into a failure.
- Risk: the trybuild compile contract or the no-`cpu` fixture needs updating
  for the new types. Severity: low. Likelihood: high. Mitigation: planned
  explicitly in `EP-M3`.
- Risk: `snapshot_version()` losing `const` breaks a caller. Severity: low.
  Likelihood: low. Mitigation: a repository search found no `const` caller; the
  crate is pre-1.0 and unreleased, so no compatibility shim is warranted.
- Risk: the saturation test writes the private `snapshot_version` field, which
  disappears. Severity: low. Likelihood: certain. Mitigation: the test seeds
  the cell through the private publisher instead (`EP-M2`).
- Risk: `arc-swap` does not build on the workspace MSRV (Rust 1.89).
  Severity: low. Likelihood: low. Mitigation: arc-swap 1.9.2 already builds in
  this workspace's dev graph (via `rstest-bdd` → `i18n-embed`) and declares no
  `rust-version`; CI's toolchain matrix is the arbiter.

## Progress

- [x] (2026-11-28 14:10Z) Branch renamed to `11-2-2-clustering-session-labels`,
  upstream configured for `origin/11-2-2-clustering-session-labels`.
- [x] (2026-11-28 14:40Z) Reconnaissance by a Wyvern team: session code,
  test infrastructure, documentation surface, and historical budgets.
- [x] (2026-11-28 14:50Z) Prior-art check on `arc-swap` progress guarantees.
- [x] (2026-11-28 15:30Z) Initial draft written.
- [ ] Community-of-experts design review applied.
- [ ] Plan approved by the user, including decision `D1`.
- [ ] `EP-M1` Snapshot value types and private publication cell, with unit
  tests.
- [ ] `EP-M2` Session rewired onto the cell; `label_reader` and
  `label_snapshot` exposed; sequential and concurrent tests green.
- [ ] `EP-M3` Behavioural scenarios, compile contracts, and the published
  example.
- [ ] `EP-M4` Documentation, ADR-0012, and roadmap closure.

## Surprises & discoveries

- Observation: `ClusteringSession::labels` already exists.
  Evidence: `chutoro-core/src/session/mod.rs` defines
  `pub fn labels(&self) -> Arc<Vec<usize>>`; the 11.2.1 Decision log and
  ADR-0010 record that it landed early. Impact: this item's substance is the
  non-blocking, multi-reader contract, not the accessor.
- Observation: the "never blocks the writer" claim is untrue for cross-thread
  readers and is untested. Evidence: `session/tests/concurrency.rs` shares the
  session as `Arc<RwLock<ClusteringSession>>`, never calls `labels()`, and
  never refreshes while readers run. The same claim appears in design §12.3 and
  §12.6, ADR-0010, the `ClusteringSession` Rustdoc and the users' guide.
  Impact: the gap is real and observable, which motivates the reader handle.
- Observation: labels and version are separate fields, so no reader could read
  a consistent pair without a lock. Impact: the snapshot pairs them.
- Observation: stale references. `session/mod.rs` cites tests in
  `session/tests.rs` (actual: `session/tests/concurrency.rs`), and
  `session/tests/concurrency.rs` says the session is "defined in
  `session_impl.rs`" (it is defined in `session/mod.rs`). Impact: fixed in
  `EP-M2` because those comments describe the concurrency model being changed.
- Observation: a rebuild-stage refresh failure leaves core distances recomputed
  and dirty bits cleared, while labels, version, pending edges and backbone are
  unchanged. Evidence: `refresh_impl.rs`, `recompute_and_rebuild`. Impact: none
  for publication (the published cell is untouched); recorded so nobody reads
  "all-or-nothing" as covering core distances.

## Decision log

- D1. Decision (needs user approval): publish snapshots through
  `arc_swap::ArcSwap<LabelSnapshot>` (crate `arc-swap`, caret `1.9.2`,
  optional, enabled by the `cpu` feature). Rationale: §12.6 promises that
  readers never block the writer. `arc-swap` documents lock-free writers and
  lock-free, mostly wait-free readers; a reader's retained `Arc` never delays a
  store. It is the established Rust crate for this "read-mostly, atomically
  replaced `Arc`" pattern, is MIT OR Apache-2.0, and already sits in
  `Cargo.lock` at 1.9.2. Rejected alternative, kept as the fallback if `D1` is
  declined: `std::sync::RwLock<Arc<LabelSnapshot>>`, where each critical
  section is one `Arc` clone or one pointer store. That uses no dependency, but
  a writer can still wait for a reader for the duration of a clone and the lock
  can be poisoned (recoverable here because the protected value is always a
  valid `Arc`, but it adds a path to reason about). Also rejected: a
  handwritten `AtomicPtr` scheme (needs `unsafe` and its own reclamation
  proof); and "documentation only, keep `RwLock<ClusteringSession>`" (does not
  meet the roadmap item). ADR-0010's "no new production dependency" driver was
  scoped to 11.2.1 and does not bind this item; ADR-0012 records the new
  decision. Date/Author: 2026-11-28, planning.
- D2. Decision: pair version and labels in one immutable `LabelSnapshot` value
  and publish that value as a unit. Rationale: separate fields cannot be read
  consistently without a lock; one `Arc<LabelSnapshot>` makes the pair a single
  atomic unit (`INV-PAIR`). Date/Author: 2026-11-28, planning.
- D3. Decision: split capabilities. A private, non-`Clone` `SnapshotPublisher`
  owned by the session is the only type that can store; the public
  `LabelReader` can only load. Both wrap the same
  `Arc<ArcSwap<LabelSnapshot>>`. Rationale: makes "exactly one writer"
  (`L-SINGLE-WRITER`) a consequence of ownership and visibility rather than of
  discipline. Date/Author: 2026-11-28, planning.
- D4. Decision: `ClusteringSession::labels` and `snapshot_version` read the
  cell; the separate `labels` and `snapshot_version` fields are removed.
  Rationale: one source of truth, so the session's own reads cannot disagree
  with readers' (`INV-SAME-PUBLICATION`). Date/Author: 2026-11-28, planning.
- D5. Decision: no Verus proof, no Kani harness, no `loom` model for this item.
  Rationale: see `Verification plan`, `Formal verification decision`.
  Date/Author: 2026-11-28, planning.
- D6. Decision: a reader outliving its session keeps returning the final
  published snapshot; there is no "session closed" signal. Rationale: the
  snapshot is still a valid, immutable result; a liveness signal would add API
  with no current consumer. Date/Author: 2026-11-28, planning.
- D7. Decision: extend the existing `session-refresh` tested example rather
  than add a second marked fence. Rationale: `chutoro-core/build.rs` extracts
  exactly one example identifier (`EXAMPLE_ID = "session-refresh"`); a second
  fence would need loader changes outside this item's scope. Date/Author:
  2026-11-28, planning.
- D8. Decision: no `insta` snapshot tests.
  Rationale: this item adds no multivariant output format; the values under
  test are small integers and vectors that `assert_eq!` states exactly.
  Date/Author: 2026-11-28, planning.

## Outcomes & retrospective

Not started. To be completed at each milestone and at the end.

## Context and orientation

Inspected base revision: `028c976` (`main`, "Cancel superseded pull-request
runs in every PR workflow (#481)"). Relocate symbols with
`rg -n 'fn labels|fn snapshot_version|fn publish_refresh' chutoro-core/src/session`.

The session module lives in `chutoro-core/src/session/` and is compiled only
with the `cpu` feature (on by default; `chutoro-core/src/lib.rs` declares
`#[cfg(feature = "cpu")] mod session;` and re-exports `ClusteringSession`,
`SessionConfig` and `SessionRefreshPolicy`).

- `session/mod.rs` (284 lines) defines
  `ClusteringSession<D: DataSource + Send + Sync>`, its private fields, and
  read accessors: `config`, `point_count`, `snapshot_version` (`const fn`,
  returns the `snapshot_version: u64` field), `labels` (returns
  `Arc::clone(&self.labels)`), and `core_distance`. It also holds a compile-time
  `Send + Sync` assertion for the session.
- `session/session_impl.rs` (286 lines) constructs the session
  (`new_with_index_result` sets `labels: Arc::new(Vec::new())` and
  `snapshot_version: 0`) and implements `append`.
- `session/refresh_impl.rs` (208 lines) implements `refresh`. It calls
  `recompute_and_rebuild`, and only on `Ok` calls `publish_refresh`, which
  infallibly sets `self.labels = Arc::new(outcome.labels)`, replaces the
  backbone, clears `pending_edges`, records `last_refresh_len` and sets
  `self.snapshot_version = self.snapshot_version.saturating_add(1)`. On `Err`
  it logs a `warn!` and returns the error with nothing published.
- `session/refresh/mod.rs` is the pure rebuild (`rebuild_mst_labels`). Labels
  come from `extract_labels_from_mst` (`chutoro-core/src/hierarchy/mod.rs`):
  cluster ids `0..k`, noise `k` when present, all zeros when no cluster is
  selected. This plan treats label values as opaque.
- Tests are under `session/tests/`: `common.rs` (fixtures `session_builder`,
  `make_session`), `concurrency.rs` (two `RwLock`-based reader tests),
  `refresh.rs` (including
  `refresh_saturates_snapshot_version_at_the_representational_limit`, which
  writes `session.snapshot_version = u64::MAX` directly), `refresh_errors.rs` (a
  `SnapshotState` capture helper and `FailableSource` from
  `chutoro-core/tests/support/failable_source.rs`, which has `fail()` and
  `recover()`), and `refresh_properties.rs`.
- Behavioural tests use `rstest-bdd` 0.6.0 (`rstest-bdd` and
  `rstest-bdd-macros` dev-dependencies of `chutoro-core`). The pattern is in
  `chutoro-core/tests/session_append_bdd.rs` (298 lines) with
  `chutoro-core/tests/features/session_append.feature`: a `#[fixture] fn world`
  returning `StepResult<World, _>`, `#[given]/#[when]/#[then]` steps, and one
  `#[scenario(path = ..., name = ...)]` function per scenario.
- The published example `chutoro-core/docs/session-refresh-example.md` is
  extracted by `chutoro-core/build.rs` (single `EXAMPLE_ID`) and run by
  `chutoro-core/tests/users_guide_session_refresh.rs` and by the append BDD
  suite.
- `chutoro-compile-contracts/tests/session_api_surface.rs` runs `trybuild`
  `pass` cases (including `tests/trybuild/session_api_cpu_enabled.rs`) and
  builds `tests/fixtures/session_api_without_cpu`, asserting the session types
  are unavailable without `cpu`.
- Dev-dependencies already present in `chutoro-core`: `rstest` 0.28,
  `rstest-bdd` 0.6, `proptest` 1.8 (lock 1.12), `googletest` 0.14.4,
  `pretty_assertions` 1.4.1, `insta` 1.49. Not present anywhere: `loom`,
  `shuttle`. `arc-swap` 1.9.2 is in `Cargo.lock` only transitively.
- Proof tooling: Verus proofs live in top-level `verus/` and run with
  `make verus` (default list in the `Makefile`); Kani harnesses run with
  `make kani`. Neither covers session publication.
- Gates: `make check-fmt`, `make lint` (interrogate, rustdoc, Clippy with
  `-D warnings`, Whitaker), `make test` (nextest over all targets and features,
  then doctests), plus `make markdownlint` and `make nixie` for documentation.

Abstraction sweep (per the `AGENTS.md` abstraction policy): a repository-wide
search for `RwLock<Arc`, `Mutex<Arc`, `AtomicPtr`, `ArcSwap` and `OnceLock<Arc`
in `*.rs` found no existing publication cell, so the new publisher/reader pair
duplicates nothing.

Budget survey: 11.1.2 (10 files / 450 lines, no increase), 11.1.3 (8 files /
350 lines, no increase; one approved dev-dependency), 11.1.4 (12 files / 600
lines, no increase), 11.2.1 (14 files / 1,000 lines; finished at 43 files and
+6,710 insertions with one user-approved increase driven by approved scope
growth — a pure pipeline, a Verus file, an example loader — and by repository
constraints: `-D warnings` dead-code gating, a four-argument Clippy limit, and
the 400-line cap). Nothing in this item resembles 11.2.1's growth drivers, so
the limits in `Tolerances` sit between 11.1.4 and 11.2.1.

Relevant documentation and skills for the implementer:

- `docs/chutoro-design.md` §12.2, §12.3 (session lifecycle, "Reading"), §12.6
  (concurrency model), §12.7 (differential and state-machine testing).
- `docs/adrs/adr-0010-incremental-refresh-domain-boundary.md` (why `labels()`
  landed early; publish is one commit block).
- `docs/property-testing-design.md` (property-test conventions and CI routing).
- `docs/rust-testing-with-rstest-fixtures.md` (fixtures and parameterized
  cases).
- `docs/rust-doctest-dry-guide.md` (doctest style, hidden setup lines).
- `docs/complexity-antipatterns-and-refactoring-strategies.md` (keep helpers
  small; avoid "bumpy road" functions that Whitaker and CodeScene flag).
- `docs/developers-guide.md` "Session public APIs", "Assertion crates", and
  "Choosing a verification tool".
- `docs/documentation-style-guide.md` ADR section (required headings and
  template).
- Skills: `rust-router`, then `rust-async-and-concurrency` (shared state,
  `Send`/`Sync`), `rust-types-and-apis` (public type shape),
  `rust-unit-testing` (rstest, googletest, pretty_assertions), `proptest`,
  `rust-verification` (method selection), `arch-decision-records` (ADR-0012),
  `codegraph-mcp` for navigation.

## Conformance basis

Upstream artefacts at revision `028c976`:

- `docs/roadmap.md` item 11.2.2 (`RM-11.2.2`): "Implement
  `ClusteringSession::labels` returning the latest `Arc<Vec<usize>>` label
  snapshot without blocking the writer. Requires 11.2.1."
- `docs/chutoro-design.md` §12.2 (`DES-12.2-SWMR`: "Single writer, concurrent
  readers"; immutable label snapshot), §12.3 phase 4 (`DES-12.3-READ`: readers
  get `Arc<Vec<usize>>` that does not block the writer), §12.6
  (`DES-12.6-NONBLOCK`: readers never block the writer and vice versa;
  `DES-12.6-SWAP`: a new snapshot is published atomically by swapping the
  `Arc`).
- ADR-0010 (accepted 2026-11-23): `labels()` signature fixed; publish is
  all-or-nothing (`ADR10-ATOMIC-PUBLISH`).
- Roadmap item 11.2.1 acceptance (`RM-11.2.1-ACC`): `labels().len()` equals
  total point count after refresh; version increments by exactly one per
  refresh.
- Downstream consumers: 11.4.4 (snapshot consistency under concurrent readers)
  and 11.4.5 (session state machine) build on this item's tests.
- No Terms of Reference document exists for this work.

Requirements derived here:

- `REQ-LATEST`: any reader can obtain the most recently published snapshot.
- `REQ-NONBLOCK`: no reader action can delay publication, and publication never
  waits for readers; readers never wait for a refresh in progress.
- `REQ-PAIR`: a reader can obtain version and labels as one consistent pair.
- `REQ-SIGNATURE`: `labels(&self) -> Arc<Vec<usize>>` is unchanged.
- `REQ-PRESERVE`: length, per-refresh increment, saturation and all-or-nothing
  publication from 11.2.1 still hold.

Gaps at the base revision:

- `GAP-CROSS-THREAD`: cross-thread readers must lock the whole session.
- `GAP-TORN-PAIR`: labels and version are separate fields.
- `GAP-UNTESTED`: no test reads labels while a refresh runs.

Trace links:

```plaintext
RM-11.2.2 -> DES-12.6-NONBLOCK -> REQ-NONBLOCK -> EP-M2
  -> session::tests::label_reader::writer_publishes_while_a_reader_holds_a_snapshot
RM-11.2.2 -> DES-12.6-SWAP -> REQ-LATEST -> EP-M1, EP-M2
  -> session::tests::label_reader_model::reader_tracks_publications (proptest)
DES-12.2-SWMR -> REQ-PAIR -> D2 -> EP-M2
  -> session::tests::label_reader::concurrent_readers_see_paired_monotone_snapshots
ADR10-ATOMIC-PUBLISH -> REQ-PRESERVE -> EP-M2
  -> session::tests::refresh_errors::failed_refresh_leaves_reader_snapshot_unchanged
RM-11.2.1-ACC -> REQ-PRESERVE -> EP-M1 -> session::snapshot::tests::next_snapshot_version_*
  ; EP-M2 -> refresh_saturates_snapshot_version_at_the_representational_limit
REQ-SIGNATURE -> EP-M3 -> chutoro-compile-contracts trybuild session_label_reader.rs
DES-12.3-READ -> EP-M3 -> tests/session_labels_bdd.rs scenarios; tests/users_guide_session_refresh.rs
```

## Verification plan

The verification is designed before the implementation: the abstract model
below fixes what must be true, and the implementation structure (one immutable
snapshot value, one private publisher, one public loader) is chosen so each
obligation reduces either to a small local argument or to a documented
third-party axiom.

### Abstract model

Let a snapshot be a pair `s = (v, L)` of a version `v ∈ u64` and a label vector
`L`. The session has a point count `n` and a publication history
`H = [h₀, h₁, …, h_k]`, where `h₀ = (0, [])`. A publication cell `C` always
holds the last element of `H`.

Transitions, all performed by the single writer:

- `append(ids)`: may change `n`; `H` and `C` unchanged.
- `refresh()` succeeding with labels `L′` where `|L′| = n`: append
  `h_{k+1} = (sat_inc(h_k.v), L′)` to `H` and set `C := h_{k+1}`, where
  `sat_inc(v) = v + 1` if `v < u64::MAX`, else `u64::MAX`.
- `refresh()` failing: `H` and `C` unchanged.

Reader operations, any thread, any time:

- `load()`: returns the value of `C` at some instant during the call (the
  linearization point).

Preconditions: none at the API level. `label_reader()` and `load()` are total
and infallible. Malformed input is impossible because readers supply no input.

### Invariants and lemmas

- `INV-PAIR` (`REQ-PAIR`): every value returned by `LabelReader::snapshot`,
  `ClusteringSession::label_snapshot`, and the pair (`labels()`,
  `snapshot_version()`) read from one `snapshot()` is some `h_i ∈ H` — never a
  version from one publication with labels from another.
- `INV-LATEST` (`REQ-LATEST`): if the writer's publication of `h_k` happens
  before a `load()` begins, that `load()` returns `h_j` with `j ≥ k`. With no
  concurrent writer it returns exactly `h_k`.
- `INV-MONO` (`REQ-LATEST`): two `load()` calls ordered in one thread return
  `h_i` then `h_j` with `i ≤ j`, so the observed versions never decrease.
- `INV-IMMUTABLE`: once returned, a snapshot's version and labels never change.
- `INV-ATOMIC-PUBLISH` (`REQ-PRESERVE`): a failed refresh leaves `C` unchanged.
- `INV-LEN` (`REQ-PRESERVE`): `|h_i.L|` equals the point count at the moment
  `h_i` was published (11.2.1's obligation, preserved).
- `INV-SAME-PUBLICATION`: `session.labels()` and `reader.labels()` called with
  no publication in between return the same allocation (`Arc::ptr_eq`).
- `INV-NONBLOCK` (`REQ-NONBLOCK`): a `publish` completes in a bounded number of
  its own steps regardless of what readers do, including holding any number of
  snapshots indefinitely; a `load()` never waits for an in-progress refresh.
- `L-SAT` (version arithmetic): `sat_inc(v) ≥ v`, with equality only at
  `u64::MAX`; therefore `h_i.v = min(i, u64::MAX)` and versions in `H` are
  non-decreasing.
- `L-SINGLE-WRITER`: only the session's private `SnapshotPublisher` can store
  into `C`, and it is reachable only from `&mut self` methods, so publications
  are totally ordered by the writer's program order and `H` is well defined.
- `L-MONO-COMPOSE`: `INV-MONO` follows from `L-SINGLE-WRITER` (publications form
  a sequence), `AXIOM-LINEARIZABLE` (each load reads one element of that
  sequence, in an order consistent with real time) and `L-SAT` (the sequence's
  versions are non-decreasing).

### Axioms (trusted, not verified here)

- `AXIOM-LINEARIZABLE`: `ArcSwap::load_full` and `ArcSwap::store` are
  linearizable: each load returns the `Arc` stored by the most recent store in
  some total order consistent with real time. Source: `arc-swap` 1.9 crate
  documentation. Boundary: `session/snapshot.rs` private cell.
- `AXIOM-LOCK-FREE`: `ArcSwap` writers are lock-free and readers are lock-free
  and mostly wait-free; a held `Arc` obtained by `load_full` does not delay a
  store. Source: `arc-swap` `docs::performance`. Boundary: same. (Under the
  `D1` fallback this axiom is replaced by `std::sync::RwLock` mutual exclusion
  with O(1) critical sections, and `INV-NONBLOCK` weakens to "waits at most one
  pointer copy".)
- `AXIOM-ARC`: `std::sync::Arc<T>` gives shared immutable access; without
  interior mutability in `T`, the pointee cannot change while shared. Boundary:
  `LabelSnapshot` (no `Cell`, `RefCell`, atomics or `Mutex` fields), and no
  code calls `Arc::get_mut` or `Arc::make_mut` on published data.
- `AXIOM-OWNERSHIP`: Rust's `&mut self` exclusivity and module privacy; the
  compiler enforces `L-SINGLE-WRITER` once the publisher is private and not
  `Clone`.

### Methods per obligation

- Obligation: `L-SAT`.
  Method: parameterized `rstest` table on the pure helper
  `next_snapshot_version(u64) -> u64`, plus a `proptest` property. Rationale:
  the function is one saturating add; a table covers both regimes and the
  boundary, and the property checks the closed form over a range. Domain:
  `{0, 1, 41, u64::MAX - 1, u64::MAX}`; property over `v in any::<u64>()`
  asserting `next ≥ v` and `next == v + 1` when `v < u64::MAX`. Artefact:
  `chutoro-core/src/session/snapshot.rs` `#[cfg(test)] mod tests`. Evidence:
  `cargo nextest run -p chutoro-core -E 'test(next_snapshot_version)'` fails to
  compile before `EP-M1`, then passes. Non-vacuity: the table includes both the
  increment case and the saturated case; mutation `v.wrapping_add(1)` fails the
  `u64::MAX` row, and `v` (no increment) fails the `0` row.
- Obligation: `INV-LATEST`, `INV-MONO` (sequential form), `INV-PAIR`,
  `INV-SAME-PUBLICATION`, `INV-IMMUTABLE`, `INV-LEN`. Method: model-based
  `proptest` over operation sequences on a real session. Rationale: the
  properties are about histories of interleaved writer and reader operations;
  generated sequences reach orderings a handwritten test would not (reader
  created before vs after refreshes, refresh with no new points, many readers,
  held snapshots across several refreshes). Domain: sequences of 1–24
  operations over a 12-point numeric source with `min_cluster_size` in `1..=3`:
  `AppendNext(k ∈ 0..=3)` (next contiguous indices, capped at the source
  length), `Refresh`, `NewReader`, `Load(reader)`, `HoldSnapshot(reader)`,
  `SessionRead`. Oracle (independent of the implementation): a model holding
  `n` (points appended), `k` (successful refresh count), the expected version
  `min(k, u64::MAX)`, and the point count at each publication; plus copies
  (`Vec<usize>` clones) of every held snapshot taken when it was held.
  Assertions: each `Load` returns version `k` and length equal to the model's
  `n` at the last refresh; each held snapshot still equals its copy after every
  later operation; `SessionRead` returns an `Arc` that is `ptr_eq` to a
  reader's fresh load. Artefact:
  `chutoro-core/src/session/tests/label_reader_model.rs`. Evidence: fails to
  compile before `EP-M2`; passes after with the default case count. Record the
  `proptest` regression file policy: commit any regression file the run
  creates. Non-vacuity: classify generated cases with `prop_assert!` counters
  exposed through `proptest`'s `TestRunner` statistics or explicit counters
  asserted at the end of a dedicated `#[test]` that runs the same strategy for
  256 cases and requires at least one case with a reader created before two or
  more refreshes and one case with a snapshot held across a refresh. Mutations
  that must be rejected: `LabelReader` caching the snapshot at creation (stale
  `Load`); `publish_refresh` skipping the cell store (stale `Load` and broken
  `ptr_eq`); `session.labels()` reading a leftover field (broken `ptr_eq`).
- Obligation: `INV-ATOMIC-PUBLISH`.
  Method: named `rstest` test using `FailableSource`. Rationale: one specific
  failure path; the existing `refresh_errors.rs` harness already injects the
  failure. Domain: a session with one successful refresh, then a forced failing
  refresh, then recovery and a successful refresh. Artefact:
  `failed_refresh_leaves_reader_snapshot_unchanged` in
  `chutoro-core/src/session/tests/refresh_errors.rs`. Evidence: fails to
  compile before `EP-M2`; passes after; reader snapshot before and after the
  failure is equal (`pretty_assertions::assert_eq!`), and after recovery the
  version is 2. Non-vacuity: the test first asserts that the failing refresh
  actually returned `Err`; mutation "publish before the fallible rebuild" is
  rejected.
- Obligation: `INV-MONO` and `INV-PAIR` under real concurrency.
  Method: multithreaded stress test with `std::thread::scope`. Rationale: the
  sequential model cannot exercise overlap between a load and a store; this
  test runs real overlap and checks schedule-independent assertions. Domain: an
  8-point source; the writer performs four cycles of "append two points,
  refresh", so publication `i` has `2i` labels (a precomputed oracle table
  `expected_len[i]`); four reader threads load in a loop until the writer
  signals completion, then load once more. Assertions per reader: versions
  never decrease; every observed snapshot has
  `labels.len() == expected_len[version]`; the final load returns version 4.
  Overlap counts (distinct versions seen while the writer was running) are
  logged with `tracing::debug!`, never asserted. Artefact:
  `concurrent_readers_see_paired_monotone_snapshots` in
  `chutoro-core/src/session/tests/label_reader.rs`. Evidence: fails to compile
  before `EP-M2`; passes after; re-run ten times locally with `--test-threads`
  unchanged and record that none failed. Non-vacuity: the final-load assertion
  makes the test fail if readers never observe publication; the `expected_len`
  pairing makes a torn pair fail. Seeded fault for the record: temporarily
  publish version and labels through two separate cells; the test is expected
  to fail intermittently (it cannot catch a torn pair deterministically, which
  is why `INV-PAIR` is also argued structurally by `D2`: one
  `Arc<LabelSnapshot>` is stored and loaded as one pointer).
- Obligation: `INV-NONBLOCK`.
  Method: named progress test plus a contrast test, both with a timeout guard.
  Rationale: the property is "the writer finishes while a reader is parked
  holding a snapshot and a handle"; a deadlock is turned into a failure by
  `mpsc::Receiver::recv_timeout`. Domain: a reader thread takes a snapshot,
  signals "holding", and parks until told to release; meanwhile the writer
  appends and refreshes and reports completion on a channel the main thread
  awaits with a 30-second timeout. Assertions: the writer completes before the
  reader releases; the reader's held snapshot is unchanged; a fresh load after
  release returns the new version. Artefact:
  `writer_publishes_while_a_reader_holds_a_snapshot` in
  `session/tests/label_reader.rs`. Non-vacuity (contrast test):
  `shared_session_lock_blocks_the_writer_while_a_reader_holds_it` shows that
  the pattern this item replaces — a reader holding a read guard on
  `Arc<RwLock<ClusteringSession>>` — makes `try_write()` fail with
  `WouldBlock`, so the harness can observe blocking when it exists. This
  replaces and absorbs the two existing `RwLock` tests in
  `session/tests/concurrency.rs`, whose comments are corrected. Residual gap:
  lock-freedom of `publish` itself is `AXIOM-LOCK-FREE`, not something a test
  can establish.
- Obligation: `L-SINGLE-WRITER`.
  Method: compile-time contracts and code review. Rationale: it is a visibility
  and ownership property, which the compiler checks exactly. Artefact:
  `SnapshotPublisher` is `pub(super)` (private to the session module) and does
  not derive `Clone`; `chutoro-compile-contracts` gains a `pass` fixture proving
  `LabelReader: Clone + Send + Sync + 'static` and that a reader moved into a
  spawned thread coexists with `session.refresh()` (the reader does not borrow
  the session). Evidence: `cargo nextest run -p chutoro-compile-contracts`
  passes. Non-vacuity: the fixture fails to compile if `LabelReader` gains a
  lifetime parameter or borrows the session.

### Formal verification decision

No Verus proof, Kani harness, or `loom` model is planned (decision `D5`).

- The only repository-owned mathematical content is `L-SAT`, a single
  saturating increment. A Verus lemma would restate the specification of
  `u64::saturating_add`; the table and property tests exercise it completely at
  the boundary that matters.
- `L-SINGLE-WRITER` is discharged by the Rust compiler (`AXIOM-OWNERSHIP`),
  which is stronger evidence than a model of it.
- The substantive concurrency claims (`INV-LATEST`, `INV-MONO`,
  `INV-NONBLOCK`) reduce, through `L-MONO-COMPOSE`, to `arc-swap`'s documented
  guarantees. Verifying them would mean verifying `arc-swap`'s internals, which
  this repository treats as a third-party axiom. `arc-swap` has no `loom`
  integration, and this item's own code contains no atomics or locks that a
  `loom` model could explore. ADR-0002 already records that Kani is unsuitable
  for concurrency.
- `L-MONO-COMPOSE` is stated here and in the design document as an informal
  proof sketch (single totally ordered history + linearizable reads +
  non-decreasing versions ⇒ non-decreasing observations). It remains an
  unproved theorem statement in documentation; no `assume`, `admit` or trusted
  proof body is introduced.

What stronger methods would add: a Verus model of a linearizable register would
prove `L-MONO-COMPOSE` over an abstraction, but the abstraction would assume
exactly `AXIOM-LINEARIZABLE`, so the proof would restate its premise. If a
later item replaces `arc-swap` with repository-owned atomics, that item must
revisit this decision with `loom`.

### Suggested BDD features

Proposed feature file `chutoro-core/tests/features/session_labels.feature`, run
by `chutoro-core/tests/session_labels_bdd.rs` (new file; the append suite is
already 298 lines). Scenarios trace to `REQ-LATEST`, `REQ-PAIR`,
`REQ-PRESERVE`, and `EP-M3`.

```gherkin
Feature: Reading clustering session label snapshots
  Scenario: A reader created before any refresh sees the empty snapshot
    Given an empty clustering session over 4 points
    When I create a label reader
    Then the reader sees version 0 with 0 labels

  Scenario: A reader observes a refresh published after it was created
    Given an empty clustering session over 4 points
    And a label reader for the session
    When I append source indices "0,1,2,3"
    And I refresh the session
    Then the reader sees version 1 with 4 labels
    And the reader and the session return the same labels

  Scenario: A held snapshot is unchanged by a later refresh
    Given a refreshed clustering session over 4 points
    And a label reader for the session
    When the reader holds its current snapshot
    And I refresh the session
    Then the held snapshot still has version 1 with 4 labels
    And the reader sees version 2 with 4 labels

  Scenario: A failed refresh leaves the reader's snapshot unchanged
    Given a refreshed clustering session over a failable source of 4 points
    And a label reader for the session
    When the source starts failing
    And I refresh the session
    Then the refresh is rejected
    And the reader sees version 1 with 4 labels

  Scenario: A reader outlives its session
    Given a refreshed clustering session over 4 points
    And a label reader for the session
    When I drop the session
    Then the reader sees version 1 with 4 labels
```

Runner: `cargo nextest run -p chutoro-core --test session_labels_bdd`. The step
file reuses `FailableSource` via
`#[path = "support/failable_source.rs"] mod failable_source;`, as the session
unit tests already do.

## Plan of work

### `EP-M1` Snapshot value types and private publication cell

Assigned: `REQ-PAIR`, `REQ-PRESERVE` (`L-SAT`), decisions `D1`–`D3`.

Red: create `chutoro-core/src/session/snapshot.rs` containing only its `//!`
module comment and a `#[cfg(test)] mod tests` with the `L-SAT` table and
property, plus tests that a fresh publisher's reader loads `(0, [])`, that a
store is visible to an existing reader, and that a cloned reader sees the same
publication. Declare `mod snapshot;` in `session/mod.rs`. Run the focused
command; it fails to compile because `next_snapshot_version`,
`SnapshotPublisher`, `LabelSnapshot` and `LabelReader` do not exist.

Green: add `arc-swap = { version = "1.9.2", optional = true }` to
`chutoro-core/Cargo.toml` and `"dep:arc-swap"` to the `cpu` feature. In
`snapshot.rs` implement:

- `pub struct LabelSnapshot { version: u64, labels: Arc<Vec<usize>> }` deriving
  `Clone, Debug, PartialEq, Eq`, with `pub const fn version(&self) -> u64`,
  `pub fn labels(&self) -> Arc<Vec<usize>>` and
  `pub fn as_slice(&self) -> &[usize]`, plus
  `pub(super) fn new(version, labels: Vec<usize>)` and `pub(super) fn empty()`.
- `pub(super) const fn next_snapshot_version(current: u64) -> u64`
  (`current.saturating_add(1)`).
- `pub(super) struct SnapshotPublisher { cell: Arc<ArcSwap<LabelSnapshot>> }`
  (not `Clone`) with `new() -> Self` (holding `LabelSnapshot::empty()`),
  `current(&self) -> Arc<LabelSnapshot>` (`load_full`),
  `publish_next(&self, labels: Vec<usize>)` (computes the next version from the
  current snapshot and stores), `reader(&self) -> LabelReader`, and a
  `#[cfg(test)] fn publish(&self, snapshot: LabelSnapshot)` used only to seed
  boundary states. `publish_next` takes `&self` but is reachable only through
  the session's `&mut self` refresh path; that is what makes the
  load-then-store sequence race-free (`L-SINGLE-WRITER`). Document this in a
  comment.
- `pub struct LabelReader { cell: Arc<ArcSwap<LabelSnapshot>> }` deriving
  `Clone`, with a manual `Debug` showing the current version, and
  `pub fn snapshot(&self) -> LabelSnapshot`,
  `pub fn labels(&self) -> Arc<Vec<usize>>`, and
  `pub fn snapshot_version(&self) -> u64`. Every method loads once with
  `load_full` and never returns an `arc-swap` guard.

Re-export `LabelSnapshot` and `LabelReader` from `session/mod.rs` and from
`chutoro-core/src/lib.rs` under the existing `#[cfg(feature = "cpu")]` line.
Because nothing outside tests uses the publisher yet, `-D warnings` would
reject dead code; to keep the plateau lint-clean, `EP-M1` and `EP-M2` are
committed together if dead-code warnings appear (11.2.1 hit the same
constraint). Do not add `#[allow(dead_code)]`.

Refactor: keep each function short; confirm `snapshot.rs` stays well under 400
lines.

### `EP-M2` Session publishes through the cell; reader accessors

Assigned: `REQ-LATEST`, `REQ-NONBLOCK`, `REQ-PAIR`, `REQ-SIGNATURE`,
`REQ-PRESERVE`, decisions `D4`, `D6`.

Red: add `session/tests/label_reader.rs` (the concurrency and progress tests,
and the contrast test), `session/tests/label_reader_model.rs` (the model-based
property), and `failed_refresh_leaves_reader_snapshot_unchanged` in
`session/tests/refresh_errors.rs`; register the new modules in
`session/tests/mod.rs`. They fail to compile because
`ClusteringSession::label_reader` and `label_snapshot` do not exist.

Green:

1. In `session/mod.rs`, replace the fields `labels: Arc<Vec<usize>>` and
   `snapshot_version: u64` with `published: snapshot::SnapshotPublisher`.
2. In `session/session_impl.rs` `new_with_index_result`, initialize
   `published: SnapshotPublisher::new()`.
3. In `session/refresh_impl.rs` `publish_refresh`, replace the two assignments
   with `self.published.publish_next(outcome.labels)`, keeping the other
   commits in the same infallible block. Move the saturation comment to
   `next_snapshot_version`.
4. Rewrite accessors: `snapshot_version(&self) -> u64` (no longer `const`)
   returns `self.published.current().version()`; `labels(&self)` returns
   `self.published.current().labels()`; add
   `label_snapshot(&self) -> LabelSnapshot` and
   `label_reader(&self) -> LabelReader`, each with Rustdoc and a runnable
   example. If `session/mod.rs` would exceed 400 lines, put the four label
   accessors in a new `session/labels_impl.rs` `impl` block instead.
5. Update `refresh_saturates_snapshot_version_at_the_representational_limit`
   in `session/tests/refresh.rs` to seed the boundary with
   `session.published.publish(LabelSnapshot::new(u64::MAX, Vec::new()))`
   instead of writing the removed field.
6. Retire `session/tests/concurrency.rs`: its two `RwLock` tests move into
   `label_reader.rs` as the single contrast test, and the file and its `mod`
   line are removed. Correct the stale `session/tests.rs` reference in the
   `ClusteringSession` Rustdoc "Concurrency model" section, which is rewritten
   to describe `label_reader`.

Refactor: run the focused tests, then `make check-fmt`, `make lint`,
`make test`.

### `EP-M3` Behavioural scenarios, compile contracts, published example

Assigned: `DES-12.3-READ`, `REQ-SIGNATURE`, `L-SINGLE-WRITER`.

Red: add `chutoro-core/tests/features/session_labels.feature` (the scenarios
above) and `chutoro-core/tests/session_labels_bdd.rs`; add
`chutoro-compile-contracts/tests/trybuild/session_label_reader.rs` and its
`cases.pass(...)` line in `tests/session_api_surface.rs`; extend
`tests/fixtures/session_api_without_cpu` and its assertions so `LabelReader` and
`LabelSnapshot` are reported unavailable without `cpu`. Before `EP-M2` these
would fail; after `EP-M2` they should pass at once, so record the red evidence
for this milestone by running the new scenario file against a temporary
mutation (the reader caching its snapshot at creation) and noting the failing
scenario, then revert the mutation.

Green: extend the fenced program in
`chutoro-core/docs/session-refresh-example.md` to create a reader before
`append`, move it into a `std::thread::spawn` closure that returns
`reader.snapshot()`, join after `refresh`, and assert the thread saw a complete
snapshot (version 0 with 0 labels or version 1 with 4 labels) and that a fresh
`reader.labels()` equals `session.labels()`. Keep the marker unchanged (decision
`D7`).

Refactor: focused runs, then the three gates.

### `EP-M4` Documentation, ADR-0012, roadmap closure

Assigned: documentation requirements of the task; closes `RM-11.2.2`.

- Add `docs/adrs/adr-0012-label-snapshot-publication.md` (status `Accepted`
  with the approval date once the user approves `D1`; `Proposed` until then),
  following the style guide's required sections, with options (ArcSwap,
  `RwLock<Arc<_>>`, `AtomicPtr`, lock the session), a Y-statement, the axioms,
  and consequences (reader outliving session, memory held by old snapshots:
  `8n` bytes per retained snapshot on 64-bit targets).
- `docs/chutoro-design.md`: §12.3 "Reading" phase and Figure 3 sketch
  (`published: SnapshotPublisher` replaces `labels` and `snapshot_version`);
  §12.6 "Reader threads" bullet rewritten around `LabelReader`, the paired
  snapshot, the capability split, and the informal `L-MONO-COMPOSE` argument;
  reference ADR-0012.
- `docs/users-guide.md` "Refreshing labels": replace the claim that a reader
  can hold a snapshot "while the writer appends" with a new subsection "Reading
  labels from other threads" showing `label_reader`, `snapshot()`,
  version/labels pairing, held-snapshot immutability, behaviour after a failed
  refresh, and after the session is dropped.
- `docs/developers-guide.md` "Session public APIs": add the new signatures and
  the `LabelSnapshot`/`LabelReader` surface; add a short "Snapshot publication"
  convention stating the re-use policy: `SnapshotPublisher` is the only
  publication path for session outputs; it is owned by `ClusteringSession` and
  used only from `&mut self` writer paths; later published artefacts (for
  example, diagnostics in roadmap 14) extend `LabelSnapshot` or sit beside it
  in the same published value rather than adding a second cell, so readers keep
  one consistent pair; `arc-swap` must not leak into public signatures.
- `docs/repository-layout.md` "Core session module": add `snapshot.rs` (and
  `labels_impl.rs` if created).
- `docs/contents.md`: add ADR-0012 and this ExecPlan.
- `docs/roadmap.md`: mark 11.2.2 `[x]`.
- Run `make fmt`, `make markdownlint`, `make nixie`, then the three gates.
  Re-count footnote definitions in any edited file (`make fmt` can merge
  wrapped footnote lines).

Stale-claim search, excluding `docs/execplans/`:

```bash
rg -n "never block|without blocking|RwLock<ClusteringSession|session/tests.rs" docs chutoro-core/src
```

## Milestones and plateaus

- `EP-M1` — snapshot types, publisher and reader exist with unit tests.
  Requirements: `REQ-PAIR` (structure), `L-SAT`. Evidence: `snapshot.rs` tests
  pass. Conformance check: `REQ-PAIR` structure present; design §12.6 still
  followed; no assumption falsified; only `D1`'s dependency added; no trust
  boundary or persisted-format change; trace links current. Recovery: revert
  the milestone commit. Remaining gaps: the session still uses its old fields.
  Compatibility decision: none. If dead-code lints prevent a clean plateau,
  `EP-M1` and `EP-M2` share one commit (recorded in `Decision log`).
- `EP-M2` — the session publishes through the cell and exposes reader
  accessors. Requirements: `REQ-LATEST`, `REQ-NONBLOCK`, `REQ-PAIR`,
  `REQ-SIGNATURE`, `REQ-PRESERVE`; closes `GAP-CROSS-THREAD`, `GAP-TORN-PAIR`,
  `GAP-UNTESTED`. Evidence: `label_reader.rs`, `label_reader_model.rs`,
  `refresh_errors.rs`, `refresh.rs` pass; all three gates pass. Conformance
  check: all-or-nothing publish preserved; `labels()` signature unchanged; only
  the authorized interface added; no new trust boundary; trace links current.
  Recovery: revert to `EP-M1`. Remaining gaps: BDD, contracts, docs.
  Compatibility decision: none (`snapshot_version` loses `const`; pre-1.0,
  unreleased).
- `EP-M3` — behavioural scenarios, compile contracts and the published example
  pass. Evidence: `session_labels_bdd`, `session_api_surface`,
  `users_guide_session_refresh` pass. Conformance check as above. Recovery:
  revert the milestone commit. Remaining gaps: docs. Compatibility: none.
- `EP-M4` — documents, ADR-0012 and roadmap updated; Markdown gates and the
  three gates pass. Conformance: reconcile design §12.3/§12.6 and ADR-0012 with
  the shipped code before marking `COMPLETE`. Recovery: documentation-only
  revert. Remaining gaps: 11.4.4 and 11.4.5 extend the tests.

## Concrete steps

Working directory: the repository root. Use `set -o pipefail` so a failing gate
cannot appear to pass through `tee`.

```plaintext
EP-M1 red:   cargo nextest run -p chutoro-core -E 'test(/session::snapshot::/)'
             -> error[E0425]/E0412: cannot find `next_snapshot_version` / `LabelSnapshot`
EP-M1 green: same command -> all snapshot tests pass
EP-M2 red:   cargo nextest run -p chutoro-core \
               -E 'test(/session::tests::label_reader/) | test(failed_refresh_leaves_reader_snapshot_unchanged)'
             -> error[E0599]: no method named `label_reader` found for struct `ClusteringSession`
EP-M2 green: same command -> all pass; then run the concurrency test ten times:
             for i in $(seq 10); do
               cargo nextest run -p chutoro-core \
                 -E 'test(concurrent_readers_see_paired_monotone_snapshots)' || break
             done
EP-M3 red:   (temporary mutation: LabelReader caches its snapshot)
             cargo nextest run -p chutoro-core --test session_labels_bdd
             -> scenario "A reader observes a refresh published after it was created" fails; revert mutation
EP-M3 green: cargo nextest run -p chutoro-core --test session_labels_bdd --test users_guide_session_refresh
             cargo nextest run -p chutoro-compile-contracts
```

Gate sequence after each milestone, run sequentially (or delegated to the
`scrutineer` agent), with logs under `/tmp`:

```bash
set -o pipefail
make check-fmt 2>&1 | tee /tmp/check-fmt-chutoro-11-2-2-clustering-session-labels.out
make lint 2>&1 | tee /tmp/lint-chutoro-11-2-2-clustering-session-labels.out
make test 2>&1 | tee /tmp/test-chutoro-11-2-2-clustering-session-labels.out
```

After documentation changes also run `make markdownlint` and `make nixie`. A
`make test` global-timeout expiry under compile-admission contention is not a
test failure; check that the summary reports zero failed tests before
re-running.

## Validation and acceptance

Behavioural acceptance:

- A reader created with `session.label_reader()` before any refresh returns
  version 0 and no labels; after `append(&[0, 1, 2, 3])` and `refresh()` on the
  writer, the same reader, on another thread, returns version 1 and four
  labels, without any lock in the caller's code.
- While a reader thread holds a snapshot and does not release it, the writer's
  `refresh()` completes, and the held snapshot is unchanged afterwards.
- After a failed refresh, every reader still returns the previous version and
  labels.
- `session.labels()` keeps its signature and returns the same allocation a
  reader returns.

Quality criteria:

- Tests: `make test` passes; the new tests named in `Verification plan` fail
  before their milestone and pass after.
- Verification: `L-SAT`, `INV-LATEST`, `INV-MONO`, `INV-PAIR`,
  `INV-SAME-PUBLICATION`, `INV-IMMUTABLE`, `INV-LEN`, `INV-ATOMIC-PUBLISH`
  discharged by the named tests; `INV-NONBLOCK` by the progress test plus
  `AXIOM-LOCK-FREE`; `L-SINGLE-WRITER` by the compile contract. No Verus target
  changes, so `make verus` is not an acceptance criterion; it must still pass
  in CI unchanged.
- Lint and format: `make check-fmt` and `make lint` pass with no new
  suppressions.
- Documentation: `make markdownlint` and `make nixie` pass.
- Performance: no benchmark gate; `labels()` gains one atomic load, which is
  negligible next to a refresh.

## Idempotence and recovery

All steps are re-runnable. Each milestone is one commit (or `EP-M1`+`EP-M2`
together, see above), so `git revert` restores the previous plateau. Temporary
mutations used for red evidence are reverted before committing; confirm with
`git diff --stat` that only intended files changed. Delete any `proptest`
regression file created only by a seeded mutation.

## Artefacts and notes

Reconnaissance at `028c976` (observed, not yet implementation evidence):

```plaintext
$ cargo tree -i arc-swap -e normal,dev --workspace
arc-swap v1.9.2
└── i18n-embed v0.16.0
    └── rstest-bdd v0.6.0
        [dev-dependencies]
        └── chutoro-core v0.1.0
```

`arc-swap` performance documentation (fetched 2026-11-28): readers are
lock-free and mostly wait-free; writers are lock-free; holding guards for a
long time causes reference-count contention, so prefer `load_full` for values
kept long.

## Interfaces and dependencies

New dependency (subject to `D1` approval), in `chutoro-core/Cargo.toml`:

```toml
[features]
cpu = ["dep:rand", "dep:rayon", "dep:dashmap", "dep:lru", "dep:arc-swap"]

[dependencies]
arc-swap = { version = "1.9.2", optional = true }
```

New public items in `chutoro_core` (all behind `cpu`):

```rust
/// One immutable published label snapshot: a version and its labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelSnapshot { /* version: u64, labels: Arc<Vec<usize>> */ }

impl LabelSnapshot {
    pub const fn version(&self) -> u64;
    pub fn labels(&self) -> Arc<Vec<usize>>;
    pub fn as_slice(&self) -> &[usize];
}

/// Cloneable, `Send + Sync + 'static` handle that reads the latest snapshot.
#[derive(Clone)]
pub struct LabelReader { /* cell: Arc<ArcSwap<LabelSnapshot>> */ }

impl LabelReader {
    pub fn snapshot(&self) -> LabelSnapshot;
    pub fn labels(&self) -> Arc<Vec<usize>>;
    pub fn snapshot_version(&self) -> u64;
}

impl<D: DataSource + Send + Sync> ClusteringSession<D> {
    pub fn labels(&self) -> Arc<Vec<usize>>;          // unchanged signature
    pub fn snapshot_version(&self) -> u64;            // no longer const
    pub fn label_snapshot(&self) -> LabelSnapshot;    // new
    pub fn label_reader(&self) -> LabelReader;        // new
}
```

Private items in `chutoro-core/src/session/snapshot.rs`:

```rust
pub(super) const fn next_snapshot_version(current: u64) -> u64;
pub(super) struct SnapshotPublisher { /* cell: Arc<ArcSwap<LabelSnapshot>> */ }
impl SnapshotPublisher {
    pub(super) fn new() -> Self;
    pub(super) fn current(&self) -> Arc<LabelSnapshot>;
    pub(super) fn publish_next(&self, labels: Vec<usize>);
    pub(super) fn reader(&self) -> LabelReader;
}
```

Ownership and re-use: `ClusteringSession` owns the only `SnapshotPublisher`.
`LabelReader` values are created only by `ClusteringSession::label_reader` and
are freely cloned. No other module may construct a publisher. No error type is
involved; every new function is infallible.

## Revision note

- 2026-11-28: initial draft from Wyvern reconnaissance at `028c976` and an
  `arc-swap` prior-art check. Awaiting community-of-experts review and user
  approval of the plan and of decision `D1`.
