# 11.2.2 Publish label snapshots to readers without blocking the writer

This ExecPlan (execution plan) is a living document. The sections `Constraints`,
`Tolerances (exception triggers)`, `Risks`, `Progress`,
`Surprises & discoveries`, `Decision log`, `Outcomes & retrospective`,
`Conformance basis`, and `Verification plan` must be kept up to date as work
proceeds.

Status: DRAFT (community-of-experts design review applied; awaiting user
approval)

Branch: `11-2-2-clustering-session-labels`

Roadmap item: `11.2.2` in `docs/roadmap.md`.

## Purpose / big picture

Chutoro is a Rust clustering library implementing FISHDBC, a hierarchical,
density-based clustering algorithm built on an approximate nearest-neighbour
graph. Its `ClusteringSession` type maintains a clustering that grows over
time. The caller appends source indices with `session.append(&[...])`, calls
`session.refresh()` to recompute the clustering, and reads the resulting flat
labels (one cluster number per point) with `session.labels()`. That accessor
returns an `Arc<Vec<usize>>`, a reference-counted vector that nobody can
modify. Each successful refresh _publishes_ a new label snapshot, meaning it
replaces the session's current snapshot with a new one, and advances
`session.snapshot_version()` by one.

Roadmap item 11.2.1 already shipped `labels(&self) -> Arc<Vec<usize>>`, because
its acceptance criteria were phrased in terms of that accessor (see
`docs/adrs/adr-0010-incremental-refresh-domain-boundary.md`). It did not
deliver the property this item names: reading the latest snapshot **without
blocking the writer**. Design §12.6 promises a single-writer, multiple-reader
model in which "readers never block the writer and vice versa". Today that is
false for any reader on another thread:

- `append` and `refresh` take `&mut self`, so a program that shares one session
  between a writer thread and reader threads has to wrap it in a lock, typically
  `Arc<RwLock<ClusteringSession<D>>>`. The repository's own concurrency tests
  (`chutoro-core/src/session/tests/concurrency.rs`) do exactly that.
- Under that lock, a reader calling `labels()` holds a read guard, so the
  writer's `refresh()` cannot start until every reader has let go. Every reader
  then waits for the whole refresh, which rebuilds a minimum spanning tree in
  O(E log E) time for E candidate edges, before it can read even the previous
  snapshot.
- A reader cannot read `labels()` and `snapshot_version()` as one consistent
  pair without holding the lock across both calls, because they are stored in
  two separate fields. Reading a version from one refresh and labels from
  another is called a _torn read_ in this plan.

After this change, a caller asks the session for a detached reader handle and
gives clones of it to any number of threads:

```rust,ignore
let mut session = ChutoroBuilder::new()
    .with_min_cluster_size(2)
    .build_session(source)?;
let reader = session.snapshot_reader(); // SnapshotReader: Clone + Send + Sync + 'static

std::thread::scope(|scope| {
    scope.spawn(|| {
        let snapshot = reader.snapshot(); // Arc<ClusteringSnapshot>: one consistent pair
        report(snapshot.version(), snapshot.label_slice());
    });
    session.append(&[0, 1, 2, 3])?; // the writer keeps &mut access; no caller lock
    session.refresh()            // publication never waits for a refresh or a held snapshot
})?;
```

Readers never borrow the session, never hold a lock while doing their own work,
and never see a torn read. A refresh computes everything outside any lock.
Publication then replaces one pointer inside a critical section, and a reader's
critical section is a single reference-count increment. A _critical section_ is
the code that runs while a lock is held. A reader that keeps an old snapshot
keeps exactly that snapshot, unchanged, for as long as it likes, without
delaying the writer. The existing `session.labels()` keeps its signature.

Preserved behaviour: `labels()` still returns `Arc<Vec<usize>>`, and its length
still equals `point_count()` after every successful refresh. The version still
advances by exactly one per successful refresh and saturates at `u64::MAX`.
Publication remains all-or-nothing, so a failed refresh publishes nothing. One
signature change is deliberate: `snapshot_version()` stops being a `const fn`,
because it now reads through the published snapshot.

Clarified behaviour, documented for the first time: a snapshot covers points
`[0, len)` only. After `append` without `refresh`, `point_count()` exceeds the
snapshot's length, so a reader must index labels by snapshot length, not by
`point_count()`. Core distances are not part of the snapshot.

### What this plan does and does not authorize

It authorizes:

- two new public types, `ClusteringSnapshot` (the published value) and
  `SnapshotReader` (the handle);
- two new `ClusteringSession` methods, `snapshot` and `snapshot_reader`;
- removing `const` from `ClusteringSession::snapshot_version`;
- one new ADR, ADR-0012, and an amendment banner on ADR-0010;
- a `FailableSource::with_values` constructor in shared test support;
- rewording roadmap items 11.4.4 and 12.1.1 so they match what ships.

It adds no production dependency (decision `D1`).

It does not authorize:

- count-triggered refresh (11.2.3);
- `refresh_full` (11.2.4);
- historical-edge retention (11.2.5);
- the full snapshot-consistency suite (11.4.4) or the session state-machine
  suite (11.4.5), both of which build on the tests introduced here;
- the richer `ClusteringSnapshot` fields of design §13.1 (probabilities,
  cluster statistics, lineage, `ClusterLabel`, `SnapshotVersion`), which remain
  roadmap 12.1.x.

## Constraints

- Implementation begins only after the user explicitly approves this plan.
- `ClusteringSession::labels(&self) -> Arc<Vec<usize>>` keeps its exact
  signature (roadmap wording and ADR-0010 commitment).
- Publication stays all-or-nothing. The published snapshot changes only after
  every fallible step of `refresh` succeeds, inside the infallible
  `publish_refresh` block in `chutoro-core/src/session/refresh_impl.rs`.
- Exactly one code path may publish: `ClusteringSession::publish_refresh`,
  through a publisher method that takes `&mut self`. No public type may expose
  a way to publish or modify a snapshot.
- No lock is held while a refresh computes, while a reader runs its own code,
  or while an old snapshot is freed.
- No new production or development dependency. `arc-swap` is explicitly
  rejected for this item (`D1`).
- No new `ChutoroError` variant. Nothing in this item can fail at run time.
- No `unsafe`, no `unwrap`/`expect` outside tests, and no lint suppression
  without a tightly scoped reason (`Cargo.toml` `[workspace.lints.clippy]`
  denies `indexing_slicing`, `expect_used` and others in all code;
  `clippy.toml` only relaxes `expect` in tests).
- No code file may exceed 400 lines (`AGENTS.md`). `session/mod.rs` is 284
  lines today; new accessors go in a new file if `mod.rs` would cross 400.
- Every new module starts with a `//!` comment, and every public item has
  Rustdoc with an example (`AGENTS.md`, `docs/rust-doctest-dry-guide.md`).
- Tests must not mutate environment variables.
- Historical ExecPlans (`docs/execplans/11-2-1-*.md` and earlier) must not be
  edited. ADR-0010's decision text must not be rewritten; it receives only the
  style guide's `Amended by ADR-0012` banner and an `## Amendments` section.
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
  `*.feature` and `*.stderr` files, and the fenced Rust program in
  `chutoro-core/docs/session-refresh-example.md`. That file is classified hunk
  by hunk, because its fence is executable test input.
- Configuration: `Cargo.toml` files and `Cargo.lock`, including the
  compile-contract fixture's lockfile.
- Documentation: other `*.md` under `docs/` and the prose of
  `chutoro-core/docs/session-refresh-example.md`.
- This ExecPlan is excluded from every count.

Limits, each measured as net lines (additions minus deletions) and enforced
independently:

- Files: if more than 28 non-ExecPlan files change, stop and escalate.
- Code: if code exceeds +1,300 net lines, stop and escalate.
- Configuration: any configuration change at all is an exception, because no
  dependency is added. Stop and escalate.
- Documentation: if documentation exceeds +600 net lines, stop and escalate.
- Interface: any public item beyond `ClusteringSnapshot`, `SnapshotReader`,
  `ClusteringSession::snapshot`, `ClusteringSession::snapshot_reader` and the
  `const` removal on `snapshot_version` requires escalation.
- Architecture: holding a lock across refresh computation or reader code,
  giving readers any publish or modify capability, or publishing from any path
  other than `publish_refresh` requires escalation.
- Iterations: if a gate still fails after three fix attempts on the same
  cause, stop and escalate.
- Flakiness: if any new concurrency test fails even once on a re-run without a
  code change, stop and escalate. Do not add retries.
- Time: if a milestone takes more than six hours of active work, stop and
  escalate.
- Ambiguity: if the noise-label representation (issue #462) or the
  `SnapshotVersion` newtype has to be settled to finish this item, stop and
  escalate. This plan treats label values as opaque and keeps versions as `u64`.

Budget evidence (see `Context and orientation`): the closest predecessor,
11.2.1, set 14 files and +1,000 lines. It finished at 43 files and +6,710
insertions after one approved increase. Its growth came from a new pure
pipeline, a Verus proof file, a build-script example loader and five milestones
of documentation, and none of those recur here. 11.1.2, 11.1.3 and 11.1.4
stayed inside 8–12 file limits with no recorded increase. This item adds one
small module, rewires two fields, and is test-heavy. The estimate is about 20
files and +800–1,000 code lines, roughly 70% of them tests, with zero
configuration lines and +400–480 documentation lines. The limits carry about
30% contingency because 11.2.1 showed that `-D warnings`, the 400-line cap and
compile contracts add files late.

## Risks

- Risk: the std lock makes "never blocks" weaker than §12.6's wording.
  Severity: medium. Likelihood: certain. Mitigation: the plan states the
  precise guarantee and updates §12.6, ADR-0010 (by amendment banner), the
  roadmap wording and the users' guide to match. The guarantee is that
  publication never waits for a refresh, for reader code, or for a held
  snapshot. It can wait only for another thread's single pointer copy, and that
  wait is longer only if the scheduler preempts (interrupts) the thread holding
  the lock. ADR-0012 records the lock-free alternative and the evidence that
  would justify it later.
- Risk: a later roadmap item adds a second publication path, for example
  count-triggered refresh inside `append` (11.2.3), `refresh_full` (11.2.4),
  seeding (11.3.1) or restore (12.2.3), and breaks version monotonicity or
  silently strands existing readers. Severity: high. Likelihood: medium.
  Mitigation:
  - `publish_next` takes `&mut self`, so the compiler rejects concurrent
    publishers.
  - The developers' guide states that every publication goes through
    `publish_refresh`.
  - The publisher is created once, in the session constructor. Replacing it
    would freeze every existing reader, so the guide forbids it.
  - Seeded or restored sessions must use a named
    `SnapshotPublisher::with_initial` extension, which this item does not add.
- Risk: the concurrency stress test hangs or flakes under compile-admission
  contention on a shared 6-core host. Severity: medium. Likelihood: low.
  Mitigation:
  - Barriers make the observed version set deterministic.
  - Every wait has a deadline.
  - A drop guard on the writer releases waiting readers if it panics.
  - Results are collected and asserted on the main thread.
  - Thread counts are small: four readers and four cycles.
- Risk: the trybuild compile contract or the no-`cpu` fixture needs updating
  for the new types. Severity: low. Likelihood: certain. Mitigation: planned in
  `EP-M2`. The fixture lockfile
  `chutoro-compile-contracts/tests/fixtures/session_api_without_cpu/Cargo.lock`
  is unaffected because no dependency is added. Do not regenerate it.
- Risk: `snapshot_version()` losing `const` breaks a caller. Severity: low.
  Likelihood: low. Mitigation: a repository search found no `const` use (the
  `public_const_apis.rs` contract does not use it), and the crate is pre-1.0
  and unreleased, so no compatibility shim is warranted. Recorded in
  `docs/migration-notes.md`.
- Risk: the saturation test writes the private `snapshot_version` field, which
  disappears. Severity: low. Likelihood: certain. Mitigation: the test seeds
  the publisher through a test-only method (`EP-M1`).
- Risk: `ClusteringSnapshot` lands before 12.1.1 defines its full shape.
  Severity: medium. Likelihood: certain. Mitigation: all fields are private, so
  adding §13.1's fields later is not a breaking change. `Eq` is not derived,
  because a later `f32` field would force its removal. The 12.1.1 roadmap text
  is reworded to "extend" this type (`D2`).

## Progress

- [x] (2026-11-28 14:10Z) Branch renamed to `11-2-2-clustering-session-labels`,
  upstream configured for `origin/11-2-2-clustering-session-labels`.
- [x] (2026-11-28 14:40Z) Reconnaissance by a Wyvern team: session code,
  test infrastructure, documentation surface, and historical budgets.
- [x] (2026-11-28 14:50Z) Prior-art check on `arc-swap` progress guarantees.
- [x] (2026-11-28 15:30Z) Initial draft written and committed (`5242ba3`).
- [x] (2026-11-28 16:45Z) Community-of-experts design review (four reviewers:
  structure and contracts, failure modes and cost, verification and tests,
  alternatives and long-term fit) applied in this revision.
- [x] (2026-11-28 17:20Z) Branch pushed; draft pull request #482 opened.
- [ ] Plan approved by the user.
- [ ] `EP-M1` Internal rewire: the session publishes through a private
  `SnapshotPublisher`; all existing tests stay green.
- [ ] `EP-M2` Public reader surface: `ClusteringSnapshot`, `SnapshotReader`,
  accessors, unit, model, concurrency, behavioural and compile-contract tests.
- [ ] `EP-M3` Published example, documentation, ADR-0012, roadmap closure.

## Surprises & discoveries

- Observation: `ClusteringSession::labels` already exists.
  Evidence: `chutoro-core/src/session/mod.rs` defines
  `pub fn labels(&self) -> Arc<Vec<usize>>`; the 11.2.1 Decision log and
  ADR-0010 record that it landed early. Impact: this item's substance is the
  non-blocking, multi-reader contract, not the accessor.
- Observation: the "never blocks the writer" claim is untrue for cross-thread
  readers and is untested. Evidence: `session/tests/concurrency.rs` shares the
  session as `Arc<RwLock<ClusteringSession>>`, never calls `labels()`, and
  never refreshes while readers run. The same claim appears in design §12.3,
  §12.5 ("Early `labels()` accessor") and §12.6, in ADR-0010, in the
  `ClusteringSession` Rustdoc and in the users' guide. Impact: the gap is real,
  which motivates the reader handle. The wording is corrected everywhere it
  appears.
- Observation: labels and version are separate fields, so no reader could read
  a consistent pair without a lock. Impact: the snapshot pairs them.
- Observation: design §13.1 already specifies a `ClusteringSnapshot` that
  "replaces the raw `Arc<Vec<usize>>`", and roadmap 12.1.1 requires 11.2.2.
  Evidence: `docs/chutoro-design.md` §13.1; `docs/roadmap.md` 12.1.1. Impact:
  this item introduces that type's first two fields instead of a separately
  named type that 12.1.1 would then have to replace (`D2`).
- Observation: `arc-swap` 1.9.2, the version in `Cargo.lock`, has known
  memory-safety bugs. Evidence: the crate's own `CHANGELOG.md`. Version 1.9.4
  fixed a use-after-free involving strong and weak pointers stored in different
  instances. Version 1.9.5 fixed `load` returning a value stored into a
  _different_ instance after a freed address was reused. Versions 1.9.0 and
  1.9.1 strengthened orderings because the "original proofs [were] based on
  wrong reading of standard". Impact: decision `D1` changed from `arc-swap` to
  the standard library.
- Observation: stale references. `session/mod.rs` cites tests in
  `session/tests.rs`, but they are in `session/tests/concurrency.rs`.
  `session/tests/concurrency.rs` says the session is "defined in
  `session_impl.rs`", but it is defined in `session/mod.rs`. The
  `publish_refresh` comment in `refresh_impl.rs` cites the 11.2.1 plan's
  tolerance. Impact: fixed in `EP-M1` and `EP-M2`, because those comments
  describe code being changed.
- Observation: a rebuild-stage refresh failure leaves core distances
  recomputed and dirty bits cleared. Labels, version, pending edges and
  backbone stay unchanged. Evidence: `refresh_impl.rs`,
  `recompute_and_rebuild`. Impact: none for publication. The users' guide says
  that core distances are not part of the snapshot.
- Observation: `FailableSource` always serves three values, and a refresh with
  no dirty core distances never queries the source. Evidence:
  `chutoro-core/tests/support/failable_source.rs` (`new`);
  `session/core_distance.rs` returns early when nothing is dirty. Impact: a
  "failing refresh" test must append a point before enabling failure, as
  `refresh_errors.rs::failed_refresh_preserves_snapshot_and_pending` does.
- Observation: `last_refresh_len` is written by `publish_refresh` and never
  read, and it always equals the published length. Impact: out of scope here.
  11.2.3 should derive its count trigger from the snapshot length and may
  remove the field then (`D9`).

## Decision log

- D1. Decision: publish through
  `std::sync::RwLock<Arc<ClusteringSnapshot>>` shared between the session's
  publisher and its readers. Each critical section is one pointer replacement
  (writer) or one `Arc` clone (reader), and the old snapshot is dropped after
  the lock is released. Rationale:
  - It meets the roadmap's need. Readers never wait for a refresh, and the
    writer never waits for reader code or for a held snapshot.
  - It adds no dependency, no imported `unsafe` code and no MSRV risk, and it
    can later be modelled with `loom` or paired with a `Condvar` for change
    notification.
  - On Linux, `std`'s `RwLock` is futex-based and prefers writers, so readers
    cannot starve the writer.
  - Poisoning, which happens when a thread panics while holding the lock, is
    unreachable: neither critical section can panic (`Arc::clone` aborts on
    overflow and does not unwind). Lock results are still handled with
    `PoisonError::into_inner`, never `expect`.
  Rejected:
  - `arc-swap` (lock-free). 1.9.2 has the memory-safety bugs recorded in
    `Surprises & discoveries`, even 1.9.5 brings about 70 `unsafe` sites and
    the `rustversion` dependency into the production graph, and no contention
    measurement shows the lock is a bottleneck. ADR-0012 records `arc-swap`
    1.9.5 or later as a measured future option behind the same private type.
  - A home-grown `AtomicPtr` scheme: needs `unsafe` and its own memory
    reclamation proof.
  - Splitting the session at construction into writer and reader: a breaking
    change that gains nothing over `snapshot_reader(&self)`.
  - `std::sync::mpsc`: queue semantics with a single consumer, not "latest
    value".
  - `tokio::sync::watch`: brings tokio into the core crate.
  - `left-right`: makes the writer wait for readers.
  - A seqlock: needs `Copy` data or `unsafe`.
  - Documenting `Arc<RwLock<ClusteringSession>>`: does not meet the roadmap
    item.
  ADR-0010's "no new production dependency" driver was scoped to 11.2.1, so it
  was not the reason for this choice. Date/Author: 2026-11-28, revised after
  design review (original draft chose `arc-swap`).
- D2. Decision: the published value is the `ClusteringSnapshot` of design
  §13.1, introduced here with its first two fields, `version: u64` and
  `labels: Arc<Vec<usize>>`, all fields private. It derives `Clone` and
  `PartialEq`, not `Eq`, and has a handwritten `Debug` that prints the version
  and the label count rather than every label. Rationale: one name for the
  published read model avoids a guaranteed rename in 12.1.1. Private fields
  make 12.1.1's additions non-breaking. Avoiding `Eq` keeps a later `f32` field
  possible. A compact `Debug` keeps `tracing` fields small. Rejected: a separate
  `LabelSnapshot` that 12.1.1 would supersede, which means two public types
  and a churned name. Date/Author: 2026-11-28, design review.
- D3. Decision: versions stay `u64`. The `SnapshotVersion` newtype from §13.1
  is deferred to 12.1.1. Rationale: `snapshot_version() -> u64` is existing
  public API, 11.2.4 subtracts versions, and 12.1.1 owns the snapshot's type
  vocabulary, including `ClusterLabel`. Introducing half of that vocabulary
  here would pre-empt its design. Date/Author: 2026-11-28, design review.
- D4. Decision: split capabilities between two types.
  - `SnapshotPublisher` is private to the session module, does not implement
    `Clone`, and is owned by the session. It holds a writer-local
    `latest: Arc<ClusteringSnapshot>` and the shared cell, and its only
    mutating method is `publish_next(&mut self, labels)`.
  - `SnapshotReader` is public and can only read.
  Rationale: `&mut self` turns "exactly one writer" (`L-SINGLE-WRITER`) into a
  compiler-checked property rather than a comment. The writer-local copy means
  the session's own reads never touch the lock, and `publish_next` never has to
  load before it stores. Date/Author: 2026-11-28, design review (the draft used
  `&self`).
- D5. Decision: `SnapshotReader` exposes `snapshot() -> Arc<ClusteringSnapshot>`
  and `snapshot_version() -> u64`, and has no `labels()`. Rationale: separate
  `labels()` and `snapshot_version()` calls on a reader would bring back the
  torn read this item removes. `snapshot_version()` is kept as a documented
  cheap "has anything changed?" check. Returning `Arc<ClusteringSnapshot>`
  keeps the copy O(1) as §13.1's fields arrive. The session's own `labels()` and
  `snapshot_version()` stay, because `&self` already excludes a concurrent
  refresh. Date/Author: 2026-11-28, design review.
- D6. Decision: `ClusteringSession::labels`, `snapshot_version` and `snapshot`
  read the publisher's writer-local `latest`. The fields
  `labels: Arc<Vec<usize>>` and `snapshot_version: u64` are removed. Rationale:
  there is one source of truth, so `session.labels()` and
  `reader.snapshot().labels()` return the same allocation
  (`INV-SAME-PUBLICATION`). Date/Author: 2026-11-28, planning.
- D7. Decision: a reader outliving its session keeps returning the final
  published snapshot, and there is no "session closed" signal. Rationale: the
  snapshot is still a valid, immutable result, and a liveness signal would add
  API with no current consumer. ADR-0012 records that a leaked reader keeps the
  final snapshot's memory alive, about `8n` bytes for `n` labels on 64-bit
  targets. Date/Author: 2026-11-28, planning.
- D8. Decision: keep the published example single-threaded and put the
  threaded demonstration in the `snapshot_reader` Rustdoc using
  `std::thread::scope`. Rationale: `chutoro-core/build.rs` extracts exactly one
  example identifier (`EXAMPLE_ID = "session-refresh"`). Its generated `run()`
  returns `Result<(), ChutoroError>`, which cannot carry a thread-join error
  without `expect`. A racy "either version 0 or 1" assertion would also teach
  the wrong pattern. Date/Author: 2026-11-28, design review.
- D9. Decision: leave `last_refresh_len` untouched.
  Rationale: removing it is unrelated to reader publication. 11.2.3 owns the
  count trigger that would replace it. Date/Author: 2026-11-28, design review.
- D10. Decision: no Verus proof, Kani harness or `loom` model.
  Rationale: see `Formal verification decision`. Date/Author: 2026-11-28,
  planning, wording revised after design review.
- D11. Decision: `insta` is not used for label data. The two handwritten
  `Debug` outputs are new formatted output, so each is pinned with an exact
  `assert_eq!` on `format!("{:?}", …)` instead. Rationale: each output is one
  short line, and an exact assertion is clearer than a snapshot file.
  Date/Author: 2026-11-28, design review.

## Outcomes & retrospective

Not started. To be completed at each milestone and at the end.

## Context and orientation

Inspected base revision: `028c976` (`main`, "Cancel superseded pull-request
runs in every PR workflow (#481)"). To relocate symbols, run
`rg -n 'fn labels|fn snapshot_version|fn publish_refresh' chutoro-core/src/session`.

The session module lives in `chutoro-core/src/session/` and is compiled only
with the `cpu` feature, which is on by default. `chutoro-core/src/lib.rs`
declares `#[cfg(feature = "cpu")] mod session;` and re-exports
`ClusteringSession`, `SessionConfig` and `SessionRefreshPolicy`.

- `session/mod.rs` (284 lines) defines
  `ClusteringSession<D: DataSource + Send + Sync>`, which derives `Debug`,
  together with its private fields and its read accessors:
  - `config`;
  - `point_count`;
  - `snapshot_version`, a `const fn` returning the `snapshot_version: u64`
    field;
  - `labels`, which returns `Arc::clone(&self.labels)`;
  - `core_distance`.

  It also holds a compile-time `Send + Sync` assertion for the session.
- `session/session_impl.rs` (286 lines) constructs the session and implements
  `append`. Its `new_with_index_result` sets `labels: Arc::new(Vec::new())` and
  `snapshot_version: 0`.
- `session/refresh_impl.rs` (208 lines) implements `refresh`. It calls
  `recompute_and_rebuild`, and only on `Ok` calls `publish_refresh`. That
  method cannot fail. It sets `self.labels = Arc::new(outcome.labels)`,
  replaces the backbone, clears `pending_edges`, records `last_refresh_len`,
  and sets `self.snapshot_version = self.snapshot_version.saturating_add(1)`. On
  `Err`, `refresh` logs a `warn!` and returns the error with nothing published.
- `session/refresh/mod.rs` is the pure rebuild, `rebuild_mst_labels`. Labels
  come from `extract_labels_from_mst` (`chutoro-core/src/hierarchy/mod.rs`):
  cluster ids `0..k`, noise as `k` when present, and all zeros when no cluster
  is selected. This plan treats label values as opaque.
- Tests live under `session/tests/`:
  - `common.rs` provides the `session_builder` and `make_session` fixtures.
  - `concurrency.rs` holds two `RwLock`-based reader tests.
  - `refresh.rs` includes
    `refresh_saturates_snapshot_version_at_the_representational_limit`, which
    writes `session.snapshot_version = u64::MAX` directly.
  - `refresh_errors.rs` has a `SnapshotState` capture helper and uses
    `FailableSource` from `chutoro-core/tests/support/failable_source.rs`.
    That source serves the three values `0.0, 1.0, 2.0` and has `fail()` and
    `recover()`.
  - `refresh_properties.rs` holds the existing refresh properties.
- Proptest conventions: `crate::test_utils::suite_proptest_config(cases)`
  (`chutoro-core/src/test_utils.rs`) sets per-suite case counts. A suite that
  checks coverage across its cases uses an explicit `TestRunner` with
  `fork: false` and collects results in a `RefCell`, as in
  `chutoro-core/src/hnsw/tests/property/edge_harvest_output/suite.rs`. An
  explicit runner writes no regression file unless `source_file` is set.
  ADR-0001 records the resulting warning.
- Behavioural tests use `rstest-bdd` 0.6.0, through the `rstest-bdd` and
  `rstest-bdd-macros` dev-dependencies of `chutoro-core`. The pattern is in
  `chutoro-core/tests/session_append_bdd.rs` (298 lines) with
  `chutoro-core/tests/features/session_append.feature`:
  - a `#[fixture] fn world` returning `StepResult<World, _>`;
  - `#[given]`, `#[when]` and `#[then]` step functions;
  - one `#[scenario(path = ..., name = ...)]` function per scenario.

  Steps are registered per test binary, so a new binary cannot call the append
  suite's steps; it reuses only their wording.
- The published example `chutoro-core/docs/session-refresh-example.md` is
  extracted by `chutoro-core/build.rs`, which supports a single `EXAMPLE_ID`.
  It is run by `chutoro-core/tests/users_guide_session_refresh.rs` and by the
  append BDD suite.
- `chutoro-compile-contracts/tests/session_api_surface.rs` runs `trybuild`
  cases:
  - `pass` cases, including `tests/trybuild/session_api_cpu_enabled.rs`;
  - `compile_fail` cases with `.stderr` files, for example
    `run_cpu_pipeline_is_private.rs`.

  It also builds `tests/fixtures/session_api_without_cpu` and asserts that the
  session types are unavailable without `cpu`.
- Dev-dependencies already present in `chutoro-core`:
  - `rstest` 0.28 and `rstest-bdd` 0.6;
  - `proptest` 1.8 (1.12 in the lockfile);
  - `googletest` 0.14.4, `pretty_assertions` 1.4.1 and `insta` 1.49.

  `loom` and `shuttle` are not present anywhere.
- `.config/nextest.toml` sets `slow-timeout = 60s` with `terminate-after = 1`.
- Proof tooling:
  - Verus proofs live in the top-level `verus/` directory and run with
    `make verus`, which uses the default list in the `Makefile`.
  - Kani harnesses run with `make kani`.
  - Neither covers session publication.
- Gates:
  - `make check-fmt`;
  - `make lint`, which runs interrogate, rustdoc, Clippy with `-D warnings`,
    and Whitaker;
  - `make test`, which runs nextest over all targets and features, then the
    doctests;
  - `make markdownlint` (with spelling) and `make nixie` for documentation.

Abstraction sweep, as the `AGENTS.md` abstraction policy requires: a
repository-wide search of `*.rs` files for `RwLock<Arc`, `Mutex<Arc`,
`AtomicPtr`, `ArcSwap` and `OnceLock<Arc` found no existing publication cell,
so the new publisher/reader pair duplicates nothing.

Budget survey:

- 11.1.2: 10 files / 450 lines, no increase.
- 11.1.3: 8 files / 350 lines, no increase. It had one approved
  dev-dependency.
- 11.1.4: 12 files / 600 lines, no increase.
- 11.2.1: 14 files / 1,000 lines. It finished at 43 files and +6,710
  insertions after one user-approved increase. Approved scope growth drove the
  increase: a pure pipeline, a Verus file and an example loader. Repository
  constraints added to it: `-D warnings` dead-code gating, a four-argument
  Clippy limit, and the 400-line cap.

Nothing in this item resembles 11.2.1's growth drivers, so the limits in
`Tolerances` sit between 11.1.4 and 11.2.1.

Terms used in this plan:

- _Single writer, multiple readers_: one thread changes the session while any
  number of threads read published results.
- _Critical section_: code that runs while a lock is held.
- _Linearizable_: every operation appears to take effect at one instant
  between its start and its end. A lock guarantees this for the operations it
  protects.
- _Lock-free_: some thread always makes progress even if others are paused.
- _Wait-free_: every thread finishes in a bounded number of its own steps.
- _Red-Green-Refactor_: first write a test that fails for the expected reason
  (red), then make the smallest change that passes it (green), then tidy the
  code with the test still passing (refactor).
- _Plateau_: a milestone's end state, which must be coherent and pass every
  gate.
- _Non-vacuity_: evidence that a test or proof could fail if the code were
  wrong.
- _Negative control_: a deliberate fault that a test must reject.
- _Trybuild compile contract_: a test that asserts a snippet compiles (`pass`)
  or fails to compile with a given error (`compile_fail`).
- _Y-statement_: the one-sentence ADR summary "In the context of …, facing …,
  we decided for …, to achieve …, accepting …".

Relevant documentation and skills for the implementer:

- `docs/chutoro-design.md`:
  - §12.2 and §12.3 (session lifecycle and the "Reading" phase);
  - §12.5 (the "Early `labels()` accessor" paragraph);
  - §12.6 (concurrency model);
  - §12.7 (differential and state-machine testing);
  - §13.1 (`ClusteringSnapshot`).
- `docs/adrs/adr-0010-incremental-refresh-domain-boundary.md` explains why
  `labels()` landed early and why publication is one commit block.
- `docs/property-testing-design.md` covers property-test conventions and CI
  routing.
- `docs/rust-testing-with-rstest-fixtures.md` covers fixtures and
  parameterized cases.
- `docs/rust-doctest-dry-guide.md` covers doctest style and hidden setup lines.
- `docs/complexity-antipatterns-and-refactoring-strategies.md`: keep helpers
  small, and avoid "bumpy road" functions that Whitaker and CodeScene flag.
- `docs/developers-guide.md`: the "Session public APIs", "Assertion crates"
  and "Choosing a verification tool" sections.
- `docs/documentation-style-guide.md`: the ADR section (required headings,
  template and amendment banner).
- Skills:
  - `rust-router`, then `rust-async-and-concurrency` for shared state and
    `Send`/`Sync`;
  - `rust-types-and-apis` for the shape of the public types;
  - `rust-unit-testing` for rstest, googletest and pretty_assertions;
  - `proptest`;
  - `rust-verification` for method selection;
  - `arch-decision-records` for ADR-0012;
  - `codegraph-mcp` for navigation.

## Conformance basis

Upstream artefacts at revision `028c976`:

- `docs/roadmap.md` item 11.2.2 (`RM-11.2.2`): "Implement
  `ClusteringSession::labels` returning the latest `Arc<Vec<usize>>` label
  snapshot without blocking the writer. Requires 11.2.1."
- `docs/chutoro-design.md`:
  - §12.2, `DES-12.2-SWMR`: "Single writer, concurrent readers", with an
    immutable label snapshot.
  - §12.3 phase 4, `DES-12.3-READ`: readers get an `Arc<Vec<usize>>` that does
    not block the writer.
  - §12.6, `DES-12.6-NONBLOCK`: readers never block the writer and vice versa.
  - §12.6, `DES-12.6-SWAP`: a new snapshot is published atomically by swapping
    the `Arc`.
  - §13.1, `DES-13.1-SNAPSHOT`: `ClusteringSnapshot` is the immutable,
    versioned read model.
- ADR-0010 (accepted 2026-11-23) fixes the `labels()` signature and makes
  publication all-or-nothing (`ADR10-ATOMIC-PUBLISH`).
- Roadmap item 11.2.1 acceptance (`RM-11.2.1-ACC`): `labels().len()` equals the
  total point count after refresh, and the version increments by exactly one
  per refresh.
- Downstream consumers of this item:
  - 11.2.3, 11.2.4 and 11.3.1 add new publication triggers;
  - 11.4.4 and 11.4.5 test the snapshot;
  - 11.5.1 and 11.5.2 document it and expose it in the CLI;
  - 12.1.1 extends `ClusteringSnapshot`;
  - 12.2.3 restores a session at a non-zero version;
  - 14.3.1 compacts the session.
- No Terms of Reference document exists for this work.

`DES-12.6-NONBLOCK` is refined, not followed literally (decision `D1`). The
refined statement is that readers never wait for a refresh, and the writer
never waits for reader code or held snapshots. The two can contend only for a
single pointer copy. Design §12.6 and ADR-0010 (by amendment banner) are
updated to match. This refinement is part of the plan the user is asked to
approve.

Requirements derived here:

- `REQ-LATEST`: any reader can obtain the most recently published snapshot.
- `REQ-NONBLOCK`: no reader waits for a refresh, and the writer never waits for
  reader code or for held snapshots. Contention is limited to O(1) critical
  sections.
- `REQ-PAIR`: a reader obtains version and labels as one consistent value.
- `REQ-SIGNATURE`: `labels(&self) -> Arc<Vec<usize>>` is unchanged.
- `REQ-PRESERVE`: the length, per-refresh increment, saturation and
  all-or-nothing publication rules from 11.2.1 still hold.
- `REQ-EXTENSIBLE`: 12.1.1 can add §13.1's fields without breaking this
  item's API.

Gaps at the base revision:

- `GAP-CROSS-THREAD`: cross-thread readers must lock the whole session.
- `GAP-TORN-PAIR`: labels and version are separate fields.
- `GAP-UNTESTED`: no test reads labels while a refresh runs.

Trace links:

```plaintext
RM-11.2.2 -> DES-12.6-NONBLOCK -> REQ-NONBLOCK -> D1 -> EP-M2
  -> session::tests::snapshot_reader::writer_publishes_while_a_reader_holds_a_snapshot
RM-11.2.2 -> DES-12.6-SWAP -> REQ-LATEST -> EP-M2
  -> session::tests::snapshot_reader_model::reader_tracks_publications
DES-12.2-SWMR -> REQ-PAIR -> D2, D5 -> EP-M2
  -> session::tests::snapshot_reader::concurrent_readers_see_paired_monotone_snapshots
ADR10-ATOMIC-PUBLISH -> REQ-PRESERVE -> EP-M2
  -> session::tests::refresh_errors::failed_refresh_leaves_reader_snapshot_unchanged
RM-11.2.1-ACC -> REQ-PRESERVE -> EP-M1 -> session::snapshot::tests::next_snapshot_version_cases
  ; EP-M1 -> refresh_saturates_snapshot_version_at_the_representational_limit
REQ-SIGNATURE, D4 -> EP-M2 -> trybuild session_snapshot_reader.rs and snapshot_publisher_is_private.rs
DES-13.1-SNAPSHOT -> REQ-EXTENSIBLE -> D2, D3 -> EP-M2 (private fields, no Eq) -> EP-M3 roadmap 12.1.1 wording
DES-12.3-READ -> EP-M2 -> tests/session_snapshot_bdd.rs ; EP-M3 -> tests/users_guide_session_refresh.rs
```

## Verification plan

The verification is designed before the implementation. The abstract model
below fixes what must be true. The implementation structure was then chosen so
that each obligation reduces to a small local argument or to a documented
standard-library axiom. That structure is one immutable snapshot value, one
private publisher with an `&mut self` publish, one public reader, and one
`RwLock` whose critical sections touch only a pointer.

### Abstract model

A snapshot is a pair `s = (v, L)` of a version `v ∈ u64` and a label vector
`L`. The session has a point count `n` and a publication history
`H = [h₀, h₁, …, h_k]`. The initial snapshot `h₀` is `(0, [])` for every
session constructed today. Later constructors may choose another initial
snapshot (see `L-SAT`). A shared publication cell `C` always holds the last
element of `H`. The writer also keeps a local copy `W` that is always equal to
`C`.

Transitions, all performed by the single writer:

- `append(ids)`: may change `n`; `H`, `C` and `W` are unchanged.
- `refresh()` succeeding with labels `L′`, where `|L′| = n`: append
  `h_{k+1} = (sat_inc(h_k.v), L′)` to `H`, then set `C := h_{k+1}` and
  `W := h_{k+1}`. Here `sat_inc(v) = v + 1` if `v < u64::MAX`, and `u64::MAX`
  otherwise.
- `refresh()` failing: `H`, `C` and `W` are unchanged.

Reader operations, on any thread at any time:

- `load()`: returns the value of `C` at one instant during the call.

There are no preconditions at the API level. `snapshot_reader()`, `snapshot()`
and `load()` are total and cannot fail, and readers supply no input, so
malformed input is impossible.

### Invariants and lemmas

- `INV-PAIR` (`REQ-PAIR`): every `Arc<ClusteringSnapshot>` returned by
  `SnapshotReader::snapshot` or `ClusteringSession::snapshot` is some
  `h_i ∈ H`. Its version and labels always come from the same publication.
- `INV-LATEST` (`REQ-LATEST`): if the writer's publication of `h_k` happens
  before a `load()` begins, that `load()` returns `h_j` with `j ≥ k`. With no
  concurrent writer it returns exactly `h_k`.
- `INV-MONO` (`REQ-LATEST`): if one thread makes two `load()` calls in order,
  they return `h_i` and then `h_j` with `i ≤ j`, so the versions it observes
  never decrease.
- `INV-IMMUTABLE`: once returned, a snapshot's version and labels never change.
- `INV-ATOMIC-PUBLISH` (`REQ-PRESERVE`): a failed refresh leaves `C` and `W`
  unchanged.
- `INV-LEN` (`REQ-PRESERVE`): `|h_i.L|` equals the point count at the moment
  `h_i` was published. This is 11.2.1's obligation, preserved.
- `INV-SAME-PUBLICATION`: when no publication happens between them,
  `session.labels()` and `reader.snapshot().labels()` return the same allocation
  (`Arc::ptr_eq`).
- `INV-NONBLOCK` (`REQ-NONBLOCK`):
  - The writer holds the lock only to replace one pointer, and readers hold it
    only to clone one `Arc`.
  - No lock is held during refresh computation, during reader code, or while
    an old snapshot is freed.
  - Holding any number of snapshots never delays publication.
- `L-SAT` (version arithmetic): `sat_inc(v) ≥ v`, with equality only at
  `u64::MAX`. Versions in `H` are therefore non-decreasing from any initial
  snapshot. From `h₀ = (0, [])` this gives `h_i.v = min(i, u64::MAX)`.
- `L-SINGLE-WRITER`: only `SnapshotPublisher::publish_next(&mut self, …)` can
  write `C`. The publisher is private to `session`, does not implement `Clone`,
  and is owned by one session. Every call to `publish_next` therefore needs
  `&mut ClusteringSession`, so publications are totally ordered by the writer's
  program order, and `H` is well defined.
- `L-MONO-COMPOSE`: `INV-MONO` follows from three facts:
  - `L-SINGLE-WRITER`: publications form one sequence;
  - `AXIOM-RWLOCK`: each load reads one element of that sequence, in an order
    consistent with real time;
  - `L-SAT`: the versions in that sequence never decrease.

### Axioms (trusted, not verified here)

- `AXIOM-RWLOCK`: `std::sync::RwLock` gives mutual exclusion between a writer
  and readers. Releasing a guard happens-before any later acquisition of the
  lock, so operations under the lock are linearizable. On Linux the
  implementation prefers writers. Source: the standard library documentation for
  `RwLock` and its futex implementation. Boundary: the shared cell in
  `session/snapshot.rs`.
- `AXIOM-ARC`: `std::sync::Arc<T>` gives shared, unchangeable access. Without
  interior mutability in `T`, the pointee cannot change while it is shared.
  `Arc::clone` cannot unwind. Boundary: `ClusteringSnapshot` has no `Cell`,
  `RefCell`, atomic or `Mutex` field, and no code calls `Arc::get_mut` or
  `Arc::make_mut` on published data.
- `AXIOM-OWNERSHIP`: Rust's `&mut self` exclusivity and module privacy. The
  compiler enforces `L-SINGLE-WRITER` once `publish_next` takes `&mut self` and
  the publisher type is private and not `Clone`.

### Methods per obligation

- Obligation: `L-SAT`.
  Method: a parameterized `rstest` table over the pure helper
  `next_snapshot_version(u64) -> u64`. Rationale: the function is one
  saturating addition. A table covers both regimes and the boundary. A
  `proptest` over `any::<u64>()` would almost never reach `u64::MAX` and would
  only restate `saturating_add`, so none is used. Domain: the rows `0 → 1`,
  `1 → 2`, `41 → 42`, `u64::MAX - 1 → u64::MAX`, and `u64::MAX → u64::MAX`.
  Artefact: `next_snapshot_version_cases` in the `#[cfg(test)]` module of
  `chutoro-core/src/session/snapshot.rs`. Evidence: fails to compile before
  `EP-M1` and passes afterwards. Non-vacuity: the rows include both the
  increment and the saturated case. The mutant `v.wrapping_add(1)` fails the
  `u64::MAX` row, and returning `v` unchanged fails the `0` row.
- Obligation: `REQ-PRESERVE` across the rewire (saturation).
  Method: the existing named test
  `refresh_saturates_snapshot_version_at_the_representational_limit`, adapted.
  It seeds the publisher at `u64::MAX` through the test-only `seed_for_test`
  and refreshes. In `EP-M2` it also asserts that a reader created _before_
  seeding observes `u64::MAX` and four labels. Saturation stays out of the
  behavioural suite, because reaching it needs private seeding.
- Obligation: `INV-LATEST`, `INV-MONO` (sequential form), `INV-PAIR`,
  `INV-SAME-PUBLICATION`, `INV-IMMUTABLE`, `INV-LEN`, `INV-ATOMIC-PUBLISH`.
  Method: a model-based `proptest` over sequences of operations on a real
  session. Rationale: these are properties of histories in which writer and
  reader operations interleave. Generated sequences reach orderings that named
  tests would not, for example:
  - a reader cloned between refreshes;
  - a refresh with no new points;
  - a failed refresh between successful ones;
  - many readers;
  - snapshots held across several refreshes.
  Domain: sequences of 1–24 operations over a ten-point `FailableSource` with
  `min_cluster_size` in `1..=3`. The operations are:
  - `AppendNext(k ∈ 0..=3)`: appends the next contiguous indices, capped at
    the source length.
  - `Refresh`.
  - `FailingRefresh`: calls `fail()`, then `refresh()`, then `recover()`, so no
    append ever runs while the source is failing.
  - `NewReader`.
  - `CloneReader(r)`.
  - `Load(r)`.
  - `HoldSnapshot(r)`.
  - `SessionRead`.

  Reader indices resolve as `r % readers.len()`, and an operation is a no-op
  when no reader exists. That keeps the generator total (no `prop_assume`) and
  keeps sequences valid when shrinking removes a `NewReader`. Oracle, which is
  independent of the implementation:
  - The model holds `n` (points appended), `k` (successful refreshes) and the
    published length.
  - It also holds a `Vec<usize>` copy of every snapshot taken by
    `HoldSnapshot`.
  - `FailingRefresh` accepts either outcome. On `Ok` the model advances `k`
    and the published length. On `Err` the next load must be the same
    allocation as before (`Arc::ptr_eq`). This keeps the oracle free of
    dirty-tracking details.
  Assertions:
  - Each `Load` returns version `k` and the model's published length.
  - Every held snapshot still equals its copy after every later operation.
  - `SessionRead` checks that `session.snapshot()` is `ptr_eq` to a fresh
    reader load, and that `session.labels()` is `ptr_eq` to that load's
    `labels()`.
  - `snapshot_version()` agrees on both sides.
  Artefact: `reader_tracks_publications` in
  `chutoro-core/src/session/tests/snapshot_reader_model.rs`. It needs a
  `FailableSource::with_values(Vec<f32>, FailureMode)` constructor in
  `chutoro-core/tests/support/failable_source.rs`. Evidence: fails to compile
  before `EP-M2`, and passes afterwards at `suite_proptest_config(64)`.
  Non-vacuity:
  - The suite runs an explicit `TestRunner` built from
    `suite_proptest_config(64)` with `fork: false` and
    `source_file: Some(file!())`.
  - Each case returns the coverage classes that a pure `classify(&[Op])`
    assigns to it, pushed into a `RefCell<Vec<_>>`.
  - After `run`, the suite asserts that the collected count equals the case
    count.
  - It also asserts that each class appeared at least once:
    - a reader created before two or more successful refreshes;
    - a snapshot held across a refresh;
    - a cloned reader observing a later refresh;
    - a `FailingRefresh` that returned `Err`.
  - Any regression file goes under `chutoro-core/proptest-regressions/` and is
    committed.

  Negative controls that must be rejected, each run once by hand and then
  reverted:
  - `SnapshotReader` caching its snapshot at creation (a stale `Load`);
  - a `Clone` that copies the snapshot into a new cell (the cloned reader goes
    stale);
  - `publish_refresh` updating `latest` but not the shared cell (a stale
    `Load`, and `ptr_eq` breaks);
  - a publication on the failure path (a held-copy mismatch, or the version
    advancing on `Err`).
- Obligation: `INV-ATOMIC-PUBLISH` as a named regression.
  Method: a named `rstest` test using `FailableSource`. Domain: append
  `[0, 1]`, refresh, create a reader, append `[2]`, call `fail()`, then
  refresh. Then call `recover()` and refresh again. Artefact:
  `failed_refresh_leaves_reader_snapshot_unchanged` in
  `chutoro-core/src/session/tests/refresh_errors.rs`. Evidence: fails to
  compile before `EP-M2`. After it, the failing refresh returns `Err`, which is
  asserted with
  `assert_that!(err, matches_pattern!(ChutoroError::DataSource { .. }))`. The
  reader still sees version 1 with two labels, compared with
  `pretty_assertions::assert_eq!`. After recovery it sees version 2 with three
  labels. Non-vacuity: the test asserts that the failure really happened. The
  "publish before the fallible rebuild" mutant is rejected.
- Obligation: `INV-MONO` and `INV-PAIR` under real concurrency.
  Method: a multithreaded test with per-cycle barriers. Rationale: the
  sequential model cannot exercise a load that overlaps a store. This test
  overlaps them, and its observed-version set is deterministic. Domain: an
  eight-point source and four reader threads, all in `std::thread::scope`. The
  writer runs four cycles. Each cycle appends two points, refreshes, and waits
  at a `Barrier` shared with the readers. Readers behave as follows:
  - Before cycle 1, each reader loads once (observing version 0), and all
    threads, writer included, meet at the barrier. Without that meeting, a fast
    writer could publish before a reader's first load, and the reader would
    never see version 0.
  - Each reader then loads in a loop, recording every distinct version it
    sees.
  - In cycle `i` it loads until it observes version `≥ i`, then waits at the
    barrier.
  - Every spin has a 10-second deadline, after which the reader stops and
    records a timeout.
  - Readers call `std::thread::yield_now()` between loads.

  Writer safeguards:
  - A drop guard on the writer sets a `done` flag (`Release`/`Acquire`) if it
    panics.
  - It also breaks the barrier wait by recording an abort, so readers exit
    instead of hanging.

  Results are returned from the threads and asserted on the main thread:
  - every observed snapshot satisfies `labels.len() == 2 * version`, a closed
    form that needs no indexing;
  - each reader's versions never decrease;
  - each reader observed exactly `{0, 1, 2, 3, 4}`;
  - no reader timed out.

  Artefact: `concurrent_readers_see_paired_monotone_snapshots` in
  `chutoro-core/src/session/tests/snapshot_reader.rs`. Evidence: fails to
  compile before `EP-M2`. Afterwards it passes, and
  `cargo nextest run … --stress-count 50` passes without a failure.
  Non-vacuity: the exact observed-set assertion fails if readers miss a
  publication. The closed-form length check fails on a torn pair. A seeded
  fault that publishes version and labels through two separate cells is
  recorded as best-effort evidence only, because a torn pair cannot be forced
  deterministically. `INV-PAIR` also rests structurally on `D2` and `D5`: one
  `Arc<ClusteringSnapshot>` is stored and loaded as one pointer, and the reader
  exposes no separate field reads.
- Obligation: `INV-NONBLOCK`.
  Method: a named progress test, plus a contrast test that runs the same
  harness. Harness:
  - A reader thread takes a snapshot, signals "holding", and parks until it is
    told to release, or until its release channel disconnects.
  - The writer runs on a detached `std::thread::spawn`, appends and refreshes,
    then reports completion on a channel.
  - The main thread waits with `recv_timeout`.

  Progress test: the timeout is 30 seconds, inside nextest's 60-second
  slow-timeout. The writer must complete while the reader is still parked. The
  reader's held snapshot must be unchanged, and a fresh load after release must
  return the new version. Contrast test: the parked reader instead holds a read
  guard on `Arc<RwLock<ClusteringSession>>`, which is the pattern this item
  replaces. The harness must observe a timeout after 200 ms. Then the guard is
  released and the writer is joined. The contrast test is deterministic,
  because a held read guard always blocks a writer. It proves the harness can
  detect blocking. Artefacts:
  `writer_publishes_while_a_reader_holds_a_snapshot` and
  `shared_session_lock_blocks_the_writer_while_a_reader_holds_it` in
  `session/tests/snapshot_reader.rs`. These replace the two `RwLock` tests in
  `session/tests/concurrency.rs`, which is removed. Honest scope: the progress
  test is a regression guard. It cannot fail while the public API never hands
  out a guard. The substantive argument for `INV-NONBLOCK` is structural:
  - the writer's critical section is `std::mem::replace` of one `Arc`;
  - each reader's critical section is one `Arc::clone`;
  - the old snapshot is dropped after the guard is released;
  - no guard escapes `snapshot.rs`.

  Code review checks each of these points against `AXIOM-RWLOCK`.
- Obligation: `L-SINGLE-WRITER`, and the reader's independence from the
  session borrow. Method: trybuild compile contracts and code review. Artefacts:
  - A `pass` fixture,
    `chutoro-compile-contracts/tests/trybuild/session_snapshot_reader.rs`. It
    asserts `SnapshotReader: Clone + Send + Sync + 'static` and
    `ClusteringSnapshot: Send + Sync`. It also moves a reader into a spawned
    thread and then calls `session.refresh()` while that thread is alive,
    which shows the reader does not borrow the session.
  - A `compile_fail` fixture, `snapshot_publisher_is_private.rs` with its
    `.stderr` file. It tries to name `chutoro_core::SnapshotPublisher` and to
    construct a `ClusteringSnapshot` directly, and both must fail to compile.
  - The no-`cpu` fixture asserts that `ClusteringSnapshot` and
    `SnapshotReader` are unavailable.

  Evidence: `cargo nextest run -p chutoro-compile-contracts` passes.
  Non-vacuity: the `pass` fixture stops compiling if `SnapshotReader` gains a
  lifetime parameter. The `compile_fail` fixture's `.stderr` file pins the
  exact privacy error.
- Obligation: compact `Debug` output (`D2`, `D11`).
  Method: exact `assert_eq!` on `format!("{snapshot:?}")` and
  `format!("{reader:?}")`, for example
  `ClusteringSnapshot { version: 1, label_count: 4 }`.

### Formal verification decision

No Verus proof, Kani harness or `loom` model is planned (decision `D10`).

- Verus is not warranted. The only repository-owned proof content is the
  monotonicity of `min(·, u64::MAX)` behind `L-SAT`, which is shallow. A Verus
  model of `L-MONO-COMPOSE` would combine that fact with `AXIOM-RWLOCK`, which
  is assumed, so it would add no substantive assurance. The publication model
  is also scheduled to change in 11.2.3, 11.2.4 and 11.3.1. The developers'
  guide ("Choosing a verification tool") warns that a proof transcribed from
  production logic drifts when that logic changes underneath it.
- `L-SINGLE-WRITER` is discharged by the Rust compiler (`AXIOM-OWNERSHIP`) and
  the `compile_fail` contract. That is stronger evidence than a model of it.
- Kani is unsuitable for concurrency (ADR-0002).
- `loom` would explore the standard library's lock, and this item's own code
  has no atomics or lock-free logic of its own. It would also be a new
  dependency. If a later item replaces the lock with repository-owned atomics or
  `arc-swap`, that item must revisit this decision with `loom`.
- `L-MONO-COMPOSE` is stated in this plan and in design §12.6 as an informal
  proof sketch. It remains an unproved theorem statement in documentation. No
  `assume`, `admit` or trusted proof body is introduced.

`make verus` is not an acceptance criterion, because no proof file changes. It
must still pass unchanged in CI.

### Suggested BDD features

The feature file is `chutoro-core/tests/features/session_snapshot.feature`, run
by `chutoro-core/tests/session_snapshot_bdd.rs`. That is a new step binary,
because the append suite is already 298 lines and steps do not cross binaries.
Step wording mirrors `session_append.feature` where the meaning is the same.
The world holds `session: Option<ClusteringSession<…>>` so that a scenario can
drop the session. Scenarios trace to `REQ-LATEST`, `REQ-PAIR` and
`REQ-PRESERVE`, and to `EP-M2`.

```gherkin
Feature: Reading clustering session snapshots
  Scenario: A reader created before any refresh sees the empty snapshot
    Given a clustering session over 4 points with min cluster size 2
    When I create a snapshot reader
    Then the reader sees version 0 with 0 labels

  Scenario: A reader observes a refresh published after it was created
    Given a clustering session over 4 points with min cluster size 2
    And a snapshot reader for the session
    When I append source indices "0,1,2,3"
    And I refresh the session
    Then the reader sees version 1 with 4 labels
    And the reader and the session return the same labels

  Scenario: A reader created after a refresh sees the latest snapshot
    Given a clustering session over 4 points with min cluster size 2
    When I append source indices "0,1,2,3"
    And I refresh the session
    And I create a snapshot reader
    Then the reader sees version 1 with 4 labels

  Scenario: Refreshing an empty session publishes an empty snapshot
    Given a clustering session over 4 points with min cluster size 2
    And a snapshot reader for the session
    When I refresh the session
    Then the reader sees version 1 with 0 labels

  Scenario: Appending without refreshing does not change the snapshot
    Given a clustering session over 4 points with min cluster size 2
    And a snapshot reader for the session
    When I append source indices "0,1"
    And I refresh the session
    And I append source indices "2,3"
    Then the session contains 4 points
    And the reader sees version 1 with 2 labels

  Scenario: A held snapshot is unchanged by a later refresh
    Given a clustering session over 4 points with min cluster size 2
    And a snapshot reader for the session
    When I append source indices "0,1,2,3"
    And I refresh the session
    And the reader holds its current snapshot
    And I refresh the session
    Then the held snapshot has version 1 with 4 labels
    And the reader sees version 2 with 4 labels

  Scenario: Cloned and independent readers see the same publication
    Given a clustering session over 4 points with min cluster size 2
    And a snapshot reader for the session
    When I clone the snapshot reader
    And I create a second snapshot reader
    And I append source indices "0,1,2,3"
    And I refresh the session
    Then all readers see version 1 with 4 labels

  Scenario: A reader on another thread sees the published snapshot
    Given a clustering session over 4 points with min cluster size 2
    And a snapshot reader for the session
    When I append source indices "0,1,2,3"
    And I refresh the session
    And another thread reads the snapshot
    Then the thread saw version 1 with 4 labels

  Scenario: A failed refresh leaves the reader's snapshot unchanged
    Given a clustering session over a failable source
    And a snapshot reader for the session
    When I append source indices "0,1"
    And I refresh the session
    And I append source indices "2"
    And the source starts failing
    And I refresh the session
    Then the refresh is rejected
    And the reader sees version 1 with 2 labels

  Scenario: A reader outlives its session
    Given a clustering session over 4 points with min cluster size 2
    And a snapshot reader for the session
    When I append source indices "0,1,2,3"
    And I refresh the session
    And I drop the session
    Then the reader sees version 1 with 4 labels
```

Run the scenarios with
`cargo nextest run -p chutoro-core --test session_snapshot_bdd`. The step file
reuses `FailableSource` through
`#[path = "support/failable_source.rs"] mod failable_source;`, the same way the
session unit tests include it.

## Plan of work

### `EP-M1` Internal rewire through a private publisher

Assigned: `REQ-PRESERVE`, `L-SAT`, `L-SINGLE-WRITER` (structure), decisions
`D1`, `D4` and `D6`. The milestone changes no behaviour: every existing session
test is the safety net, and the new `L-SAT` table is the red test.

Red: create `chutoro-core/src/session/snapshot.rs`. It contains a `//!` module
comment and a `#[cfg(test)] mod tests` holding the
`next_snapshot_version_cases` table and a test that a fresh publisher's
`latest()` is `(0, [])`. Declare `mod snapshot;` in `session/mod.rs`. Run the
focused command. It fails to compile because `next_snapshot_version` and
`SnapshotPublisher` do not exist.

Green: in `snapshot.rs`, implement only what the session uses now, so that no
dead code exists:

- `pub(crate) struct ClusteringSnapshot`, holding `version: u64` and
  `labels: Arc<Vec<usize>>`, with `pub(crate) const fn version(&self) -> u64`
  and `pub(crate) fn labels(&self) -> Arc<Vec<usize>>`. It derives `Clone` and
  `PartialEq` and has a handwritten `Debug` (`D2`). It also has a private
  constructor, `fn new(version, labels: Vec<usize>)`. Visibility becomes `pub`
  in `EP-M2`.
- `const fn next_snapshot_version(current: u64) -> u64`, which returns
  `current.saturating_add(1)`. Its Rustdoc carries the saturation explanation
  moved from `publish_refresh`.
- `pub(super) struct SnapshotPublisher`, deriving `Debug` and not `Clone`, with
  these fields:

  ```rust
  latest: Arc<ClusteringSnapshot>,
  shared: Arc<RwLock<Arc<ClusteringSnapshot>>>,
  ```

  and these methods:
  - `new() -> Self`, which holds the empty snapshot at version 0;
  - `latest(&self) -> &Arc<ClusteringSnapshot>`;
  - `publish_next(&mut self, labels: Vec<usize>)`;
  - `#[cfg(test)] pub(super) fn seed_for_test(&mut self, snapshot)`, which
    `debug_assert!`s that the version does not decrease.

  `publish_next` builds the next snapshot outside the lock, stores it in the
  shared cell with `std::mem::replace` under the write guard, drops the guard,
  then drops the previous `Arc`, and finally sets `latest`. The write guard is
  taken with `.write().unwrap_or_else(PoisonError::into_inner)`.

Then, in the session:

1. In `session/mod.rs`, replace the fields `labels: Arc<Vec<usize>>` and
   `snapshot_version: u64` with `published: snapshot::SnapshotPublisher`.
   Update the field documentation.
2. In `session/session_impl.rs`, have `new_with_index_result` initialize
   `published: SnapshotPublisher::new()`. This is the only place a publisher is
   created.
3. In `session/refresh_impl.rs`, have `publish_refresh` replace the two
   assignments with `self.published.publish_next(outcome.labels)`. The other
   updates stay in the same block, which cannot fail. Rewrite the method's
   comment to drop the stale reference to the 11.2.1 plan's tolerance, and
   correct the module documentation that names the removed fields.
4. Rewrite the accessors. `snapshot_version(&self) -> u64` loses `const` and
   returns `self.published.latest().version()`. `labels(&self)` returns
   `self.published.latest().labels()`.
5. In `refresh_saturates_snapshot_version_at_the_representational_limit`
   (`session/tests/refresh.rs`), replace the field write with
   `session.published.seed_for_test(...)` at `u64::MAX`.

Refactor: run the focused tests, then `make check-fmt`, `make lint` and
`make test`. Commit.

### `EP-M2` Public snapshot reader surface

Assigned:

- requirements `REQ-LATEST`, `REQ-NONBLOCK`, `REQ-PAIR`, `REQ-SIGNATURE` and
  `REQ-EXTENSIBLE`;
- decisions `D2`, `D3`, `D5`, `D7` and `D11`;
- gaps `GAP-CROSS-THREAD`, `GAP-TORN-PAIR` and `GAP-UNTESTED`.

Red (outside-in): write every new test before any production code. Each fails
to compile with error E0599 (no method `snapshot_reader`) or error E0412/E0433
(unresolved `SnapshotReader`).

- `session/tests/snapshot_reader.rs` holds the concurrency, progress and
  contrast tests, plus the `Debug` pins.
- `session/tests/snapshot_reader_model.rs` holds the model property.
- `failed_refresh_leaves_reader_snapshot_unchanged` goes in
  `session/tests/refresh_errors.rs`.
- Register these modules in `session/tests/mod.rs`, and add
  `FailableSource::with_values`.
- Add the behavioural suite:
  `chutoro-core/tests/features/session_snapshot.feature` and
  `chutoro-core/tests/session_snapshot_bdd.rs`.
- Add the compile contracts:
  - the `pass` fixture `tests/trybuild/session_snapshot_reader.rs`;
  - the `compile_fail` fixture `tests/trybuild/snapshot_publisher_is_private.rs`
    with its `.stderr` file;
  - both registered in `tests/session_api_surface.rs`;
  - the no-`cpu` fixture's source and its assertions extended for the new
    types.

Green:

- In `snapshot.rs`, make `ClusteringSnapshot` public with
  `pub const fn version`, `pub fn labels` and
  `pub fn label_slice(&self) -> &[usize]`. Each has Rustdoc and an example, and
  the type documentation says that labels cover points `[0, len)`.
- Add
  `pub struct SnapshotReader { shared: Arc<RwLock<Arc<ClusteringSnapshot>>> }`.
  It derives `Clone` and has a handwritten `Debug` that shows the current
  version. It has two methods,
  `pub fn snapshot(&self) -> Arc<ClusteringSnapshot>` and
  `pub fn snapshot_version(&self) -> u64`. Each takes the read guard
  (recovering from poison with `into_inner`), clones one `Arc` or reads one
  `u64`, and releases the guard before returning.
- Add `SnapshotPublisher::reader(&self) -> SnapshotReader`.
- In `session/mod.rs`, or in a new `session/snapshot_impl.rs` if `mod.rs`
  would exceed 400 lines, add
  `pub fn snapshot(&self) -> Arc<ClusteringSnapshot>`, which returns
  `Arc::clone(self.published.latest())`, and
  `pub fn snapshot_reader(&self) -> SnapshotReader`. Each has Rustdoc. The
  `snapshot_reader` example uses `std::thread::scope` to show a reader thread
  running beside the writer.
- Rewrite the "Concurrency model" section of the `ClusteringSession` Rustdoc
  around `snapshot_reader`, and remove the stale `session/tests.rs` reference.
- Re-export `ClusteringSnapshot` and `SnapshotReader` from `session/mod.rs`
  and from `chutoro-core/src/lib.rs`, under the existing
  `#[cfg(feature = "cpu")]` line.
- Delete `session/tests/concurrency.rs` and its `mod` line. Its intent now
  lives in the contrast test.
- Extend the saturation test so that a reader created before seeding observes
  `u64::MAX`.

Negative controls: apply each mutant named in `Verification plan` in turn. Run
the model property and the behavioural suite, record which test rejected the
mutant, then revert it. Confirm with `git diff --stat` that no mutant remains,
and delete any regression file a mutant created.

Refactor: run the focused tests, run the concurrency test under
`--stress-count 50`, then the three gates. Commit.

### `EP-M3` Published example, documentation, ADR-0012, roadmap closure

Assigned: the task's documentation requirements. This milestone closes
`RM-11.2.2`.

- `chutoro-core/docs/session-refresh-example.md`: keep the example
  single-threaded (`D8`). Create a reader before `append`. After `refresh`,
  assert that `reader.snapshot().version() == 1`, that its `label_slice()`
  equals `&*session.labels()`, and that `session.snapshot()` and the reader's
  snapshot are the same allocation. Keep the marker unchanged. Run
  `cargo nextest run -p chutoro-core --test users_guide_session_refresh --test session_append_bdd`.
- New ADR, `docs/adrs/adr-0012-session-snapshot-publication.md`. It follows
  the style guide's required sections, and its status is `Accepted` with the
  approval date. Contents:
  - the options of `D1`, in a comparison table with a caption;
  - the Y-statement;
  - the refined meaning of "without blocking";
  - the axioms;
  - consequences: a reader outlives its session, a held or leaked snapshot
    retains about `8n` bytes, and versions are monotone from any initial
    snapshot;
  - a "Future options" subsection naming `arc-swap` 1.9.5 or later behind the
    same private type, once a contention measurement shows the need, and
    noting that a lock-free cell would need `loom` evidence.
- ADR-0010: add the `Amended by ADR-0012 (2026-11-28)` banner to its Status
  section, and an `## Amendments` section. The amendment records that 11.2.2
  refines "never blocks a writer" to the guarantee in ADR-0012. The original
  text is preserved.
- `docs/chutoro-design.md`:
  - §12.2: refine the "single writer, concurrent readers" bullet.
  - §12.3: update the "Reading" phase, and change the Figure 3 sketch so that
    `published: SnapshotPublisher` replaces `labels` and `snapshot_version`.
  - §12.5: update the "Early `labels()` accessor" paragraph to say that 11.2.2
    delivered the reader contract.
  - §12.6: rewrite the "Reader threads" bullet around `SnapshotReader`, the
    paired snapshot, the capability split, the exact critical sections, and the
    informal `L-MONO-COMPOSE` sketch, and reference ADR-0012.
  - §13.1: add a sentence saying 11.2.2 introduced `ClusteringSnapshot` with
    `version` and `labels`, and that 12.1.1 extends it.
- `docs/users-guide.md` "Refreshing labels": replace the claim that a reader
  can hold a snapshot "while the writer appends". Add a subsection, "Reading
  snapshots from other threads", covering:
  - `snapshot_reader` and `snapshot()`;
  - why to prefer one `snapshot()` over separate calls;
  - held-snapshot immutability;
  - indexing by snapshot length, not `point_count()`;
  - core distances not being part of the snapshot;
  - behaviour after a failed refresh and after the session is dropped;
  - memory retention.
- `docs/developers-guide.md` "Session public APIs": add the new signatures.
  Add a "Snapshot publication" convention stating the re-use policy:
  - `SnapshotPublisher` is the only publication path for session outputs.
  - It is created once, in the session constructor, and never replaced.
  - Every publication goes through `ClusteringSession::publish_refresh`. That
    includes the future count-triggered refresh (11.2.3) and `refresh_full`
    (11.2.4).
  - Sessions that start from a non-empty state (11.3.1 `from_source`, 12.2.3
    `restore`, 14.3.1 compaction) add a
    `SnapshotPublisher::with_initial(ClusteringSnapshot)` constructor rather
    than seeding by publishing.
  - Later published data (roadmap 12.1.x fields, and diagnostics in roadmap
    13.3) is added to `ClusteringSnapshot` rather than to a second cell, so
    readers keep one consistent value.
  - No lock guard may escape `snapshot.rs`, and no lock may be held across
    computation.
- `docs/repository-layout.md` "Core session module": add `snapshot.rs` (and
  `snapshot_impl.rs` if created).
- `docs/migration-notes.md`: add an "Additive API: session snapshot readers"
  entry. It notes that `snapshot_version()` is no longer `const` and that no
  dependency or error variant is added.
- `docs/contents.md`: add ADR-0012 and this ExecPlan.
- `docs/roadmap.md`:
  - mark 11.2.2 `[x]`;
  - reword 11.4.4 so that concurrent readers use `snapshot_reader()`;
  - reword 12.1.1 from "Define" to "Extend the `ClusteringSnapshot`
    introduced by 11.2.2".
- Run `make fmt`, `make markdownlint` and `make nixie`, then the three gates.
  Afterwards, recount the footnote definitions in each edited file, because
  `make fmt` can merge wrapped `[^n]:` lines.

Stale-claim search, excluding historical plans:

```bash
rg -n "never block|without blocking|RwLock<ClusteringSession|session/tests.rs|labels\(\) accessor" \
  docs chutoro-core/src --glob '!docs/execplans/**'
```

Every remaining hit must either describe the refined guarantee or be ADR
history preserved under its amendment banner.

## Milestones and plateaus

- `EP-M1`: the session publishes through a private `SnapshotPublisher`, and
  nothing observable changes.
  - Requirements: `REQ-PRESERVE`, `L-SAT`, and the structure for
    `L-SINGLE-WRITER`.
  - Evidence: `next_snapshot_version_cases`, the adapted saturation test and
    the whole existing session suite pass, and so do all three gates.
  - Conformance check:
    - the preservation requirements are satisfied;
    - design §12.6 is followed;
    - no assumption is falsified;
    - there is no public interface change except the `const` removal, which is
      authorized;
    - no trust boundary or persisted format changes;
    - trace links are current.
  - Recovery: revert the milestone commit.
  - Remaining gaps: no public reader yet.
  - Compatibility decision: none. `snapshot_version` loses `const`; the crate
    is pre-1.0 and unreleased.
- `EP-M2`: the public reader surface works, and every new test passes.
  - Requirements: `REQ-LATEST`, `REQ-NONBLOCK`, `REQ-PAIR`, `REQ-SIGNATURE`
    and `REQ-EXTENSIBLE`. It closes `GAP-CROSS-THREAD`, `GAP-TORN-PAIR` and
    `GAP-UNTESTED`.
  - Evidence: these pass, together with all three gates:
    - `snapshot_reader.rs`, with stress-count 50;
    - `snapshot_reader_model.rs`, including its coverage assertion;
    - `refresh_errors.rs`;
    - `session_snapshot_bdd`;
    - `session_api_surface`.
  - Conformance check:
    - all-or-nothing publication is preserved;
    - the `labels()` signature is unchanged;
    - only the authorized interface was added;
    - no trust boundary changes;
    - trace links are current.
  - Recovery: revert to `EP-M1`.
  - Remaining gaps: the published example, documentation and roadmap.
  - Compatibility decision: none.
- `EP-M3`: the published example, design documents, ADRs, guides, migration
  notes and roadmap all match the shipped code.
  - Evidence: `users_guide_session_refresh` passes, and the Markdown gates and
    the three gates pass.
  - Conformance: reconcile design §12.2, §12.3, §12.5, §12.6 and §13.1,
    ADR-0010 and ADR-0012 with the code before marking the plan `COMPLETE`.
  - Recovery: a documentation-only revert.
  - Remaining gaps: 11.4.4 and 11.4.5 extend the tests.

## Concrete steps

Working directory: the repository root. Use `set -o pipefail` so a failing gate
cannot appear to pass through `tee`.

```plaintext
EP-M1 red:   cargo nextest run -p chutoro-core -E 'test(/session::snapshot::/)'
             -> error[E0425]: cannot find function `next_snapshot_version`
EP-M1 green: same command -> passes; then
             cargo nextest run -p chutoro-core -E 'test(/session::/)' -> all existing session tests pass
EP-M2 red:   cargo nextest run -p chutoro-core \
               -E 'test(/session::tests::snapshot_reader/) | test(failed_refresh_leaves_reader_snapshot_unchanged)'
             -> error[E0599]: no method named `snapshot_reader` found for struct `ClusteringSession`
             cargo nextest run -p chutoro-core --test session_snapshot_bdd -> same E0599
EP-M2 green: the commands above pass; then
             cargo nextest run -p chutoro-core --stress-count 50 \
               -E 'test(concurrent_readers_see_paired_monotone_snapshots)'
             cargo nextest run -p chutoro-compile-contracts
EP-M3:       cargo nextest run -p chutoro-core --test users_guide_session_refresh --test session_append_bdd
```

Gate sequence after each milestone. Run the gates one after another, or
delegate them to the `scrutineer` gate-runner agent, with logs under `/tmp`:

```bash
set -o pipefail
make check-fmt 2>&1 | tee /tmp/check-fmt-chutoro-11-2-2-clustering-session-labels.out
make lint 2>&1 | tee /tmp/lint-chutoro-11-2-2-clustering-session-labels.out
make test 2>&1 | tee /tmp/test-chutoro-11-2-2-clustering-session-labels.out
```

After documentation changes, also run `make markdownlint` and `make nixie`. A
`make test` global-timeout expiry under compile-admission contention is not a
test failure. Check that the summary reports zero failed tests before
re-running.

## Validation and acceptance

Behavioural acceptance:

- A reader created with `session.snapshot_reader()` before any refresh returns
  version 0 and no labels. After the writer runs `append(&[0, 1, 2, 3])` and
  `refresh()`, the same reader returns version 1 and four labels, on another
  thread, with no lock in the caller's code.
- While a reader thread holds a snapshot and does not release it, the writer's
  `refresh()` completes, and the held snapshot is unchanged afterwards.
- After a failed refresh, every reader still returns the previous snapshot.
- After `append` without `refresh`, readers still return the previous
  snapshot, whose length is below `point_count()`.
- `session.labels()` keeps its signature and returns the same allocation as
  the reader's snapshot.

Quality criteria:

- Tests: `make test` passes. Each new test named in `Verification plan` fails
  before its milestone and passes afterwards. The concurrency test passes
  `--stress-count 50`.
- Verification:
  - the named tests discharge `L-SAT`, `INV-LATEST`, `INV-MONO`, `INV-PAIR`,
    `INV-SAME-PUBLICATION`, `INV-IMMUTABLE`, `INV-LEN` and
    `INV-ATOMIC-PUBLISH`;
  - the structural argument and its regression guard discharge
    `INV-NONBLOCK`;
  - the `compile_fail` contract discharges `L-SINGLE-WRITER`;
  - each negative control is recorded as rejected.
- Lint and format: `make check-fmt` and `make lint` pass with no new
  suppressions.
- Documentation: `make markdownlint` and `make nixie` pass, and the
  stale-claim search is clean.
- Performance: there is no benchmark gate. `labels()` and `snapshot_version()`
  read the writer-local `Arc` and take no lock. A reader's `snapshot()` takes
  one uncontended read lock, which is negligible next to a refresh.

## Idempotence and recovery

All steps are re-runnable. Each milestone is one commit, so `git revert`
restores the previous plateau. Mutants used as negative controls are reverted
before committing; confirm with `git diff --stat` that only intended files
changed. Delete any `proptest` regression file created only by a mutant.

## Artefacts and notes

Reconnaissance at `028c976`. These are observations, not implementation
evidence:

```plaintext
$ cargo tree -i arc-swap -e normal,dev --workspace
arc-swap v1.9.2
└── i18n-embed v0.16.0
    └── rstest-bdd v0.6.0
        [dev-dependencies]
        └── chutoro-core v0.1.0
```

`arc-swap` changelog, read from the local registry copy of 1.9.5 on 2026-11-28:

```plaintext
# 1.9.5
* Fix `load` returning a value stored into a different instance (type confusion)
  after the address of a freed value is reused by another instance (#210).
# 1.9.4
* Fix use-after-free when strong and weak pointers to the same allocation are
  stored in different instances (#214).
# 1.9.0
* Promote certain orderings to SeqCst. Original proofs based on wrong reading of
  standard :-(.
```

Design review, 2026-11-28. Four independent reviewers covered:

- structure and contracts;
- failure modes and cost;
- verification and tests;
- alternatives and long-term fit.

The reviewers converged on these findings:

- `publish_next` must take `&mut self`.
- The reader's separate `labels()` and `snapshot_version()` calls re-created
  the torn read.
- The plan ignored §13.1's `ClusteringSnapshot`.
- The behavioural "failed refresh" scenario could not fail as written.
- The proptest non-vacuity mechanism did not exist.
- The ten-run shell loop could not fail.
- `arc-swap` 1.9.2 is unsafe to depend on.

All of these are resolved in this revision. The strongest alternative they
offered, std `RwLock<Arc<_>>`, was adopted.

## Interfaces and dependencies

No dependency is added or changed.

New public items in `chutoro_core`, all behind `cpu`:

```rust
/// An immutable, versioned record of one published clustering state.
/// Extended by roadmap 12.1.1 (design §13.1); all fields are private.
#[derive(Clone, PartialEq)]
pub struct ClusteringSnapshot { /* version: u64, labels: Arc<Vec<usize>> */ }

impl ClusteringSnapshot {
    pub const fn version(&self) -> u64;
    pub fn labels(&self) -> Arc<Vec<usize>>;
    pub fn label_slice(&self) -> &[usize];
}
impl fmt::Debug for ClusteringSnapshot { /* version and label count only */ }

/// Cloneable, `Send + Sync + 'static` handle that reads the latest snapshot.
#[derive(Clone)]
pub struct SnapshotReader { /* shared: Arc<RwLock<Arc<ClusteringSnapshot>>> */ }

impl SnapshotReader {
    pub fn snapshot(&self) -> Arc<ClusteringSnapshot>;
    pub fn snapshot_version(&self) -> u64; // a change check; use snapshot() for data
}
impl fmt::Debug for SnapshotReader { /* current version only */ }

impl<D: DataSource + Send + Sync> ClusteringSession<D> {
    pub fn labels(&self) -> Arc<Vec<usize>>;            // unchanged signature
    pub fn snapshot_version(&self) -> u64;              // no longer const
    pub fn snapshot(&self) -> Arc<ClusteringSnapshot>;  // new
    pub fn snapshot_reader(&self) -> SnapshotReader;    // new
}
```

Private items in `chutoro-core/src/session/snapshot.rs`:

```rust
const fn next_snapshot_version(current: u64) -> u64;

#[derive(Debug)]
pub(super) struct SnapshotPublisher {
    latest: Arc<ClusteringSnapshot>,
    shared: Arc<RwLock<Arc<ClusteringSnapshot>>>,
}

impl SnapshotPublisher {
    pub(super) fn new() -> Self;
    pub(super) fn latest(&self) -> &Arc<ClusteringSnapshot>;
    pub(super) fn publish_next(&mut self, labels: Vec<usize>);
    pub(super) fn reader(&self) -> SnapshotReader;
    #[cfg(test)]
    pub(super) fn seed_for_test(&mut self, snapshot: ClusteringSnapshot);
}
```

Test support:
`FailableSource::with_values(values: Vec<f32>, mode: FailureMode)` in
`chutoro-core/tests/support/failable_source.rs`.

Ownership and re-use: `ClusteringSession` owns the only `SnapshotPublisher`,
created in its constructor. `SnapshotReader` values are created only by
`ClusteringSession::snapshot_reader`, and they can be cloned freely. No error
type is involved, because every new function is infallible.

## Revision note

- 2026-11-28 (initial draft): written from Wyvern reconnaissance at `028c976`
  and an `arc-swap` prior-art check.
- 2026-11-28 (design-review revision): applied the four-reviewer
  community-of-experts review.
  - Changed:
    - `D1` now uses the std `RwLock<Arc<_>>` cell; `arc-swap` is rejected
      because of its 1.9.x memory-safety history.
    - The published type became §13.1's `ClusteringSnapshot`, and the handle
      became `SnapshotReader`.
    - The publisher takes `&mut self` and keeps a writer-local copy.
    - The reader no longer has a separate `labels()`.
    - The milestones were re-cut, so `EP-M1` is a behaviour-preserving rewire
      with no dead code.
    - The BDD scenarios were rebuilt on a working failure pattern and gained
      seven cases.
    - The proptest coverage mechanism now uses an explicit `TestRunner`, and
      the concurrency test uses barriers and deadlines.
    - The progress test gained a matching contrast harness, and a
      `compile_fail` privacy contract was added.
    - Documentation scope now includes migration notes, the ADR-0010
      amendment, §12.5, §13.1 and the 11.4.4 and 12.1.1 wording.
  - Effect on remaining work: the public API names changed, configuration is
    now zero lines, and the documentation budget rose by 100 lines. The plan
    needs user approval before `EP-M1`.
