# Implement incremental core-distance computation

This ExecPlan (execution plan) is a living document. The sections `Constraints`,
`Tolerances`, `Risks`, `Progress`, `Surprises & Discoveries`, `Decision Log`,
and `Outcomes & Retrospective` must be kept up to date as work proceeds.

Status: COMPLETED

This plan was approved for implementation by the user on 2026-06-16.

## Purpose / big picture

Deliver roadmap item `11.1.4` by giving `ClusteringSession` its first numerical
output: per-point **core distances**. The core distance of a point `p` at
parameter `m` is the distance from `p` to its `m`-th nearest neighbour in the
backing data source, excluding `p` itself. Core distances are the input to
mutual-reachability weighting, which in turn drives the minimum spanning tree
(MST) refresh in roadmap item `11.2.1`.

After this work lands, a session that has been built and appended to will
report a finite `core_distance(i)` for every successfully inserted point, and
the values will agree with a full batch `Chutoro::run` recomputation when the
caller forces a full recompute. The cheap incremental path will recompute core
distances only for the newly appended points and for the existing points that
appear as neighbours of those new points; a separate explicit
`recompute_core_distances_full` method recomputes every point's core distance
by mirroring the batch path.

Success is observable in three ways:

1. After `session.append(&[0, 1, 2, 3])` and (separately) calling the new
   `session.recompute_core_distances()`, `session.core_distance(0)` returns the
   distance from point `0` to its `m`-th nearest neighbour, where
   `m = min_cluster_size`.
2. A `proptest` property shows that under an arbitrary append sequence,
   `core_distance(i)` is monotonically non-increasing only after `i` has at
   least `min_cluster_size` non-self neighbours. Before that saturation point,
   the batch-compatible fallback rule can increase.
3. A `proptest` parity property shows that `recompute_core_distances_full`
   produces the same vector as the batch `cpu_pipeline.rs` path on the same
   dataset.

This plan only authorizes incremental core-distance computation and a
full-recompute escape hatch. It does **not** authorize MST refresh, label
extraction, snapshot publication, automatic `refresh_every_n` triggering,
seeded-session constructors, deletion, compaction, ARI/NMI baseline checks,
historical-edge retention, or stable cluster identity.

## Constraints

- Keep scope to roadmap item `11.1.4`: compute core distances for newly
  appended points and for existing points that appeared as their neighbours,
  expose a `recompute_core_distances_full` escape hatch, expose a read-only
  `core_distance(point)` accessor, and update documentation that the public
  session surface gained a new method and a new accessor.
- Do not implement `ClusteringSession::refresh`, `refresh_full`,
  `update_mst`, `labels`, label snapshot publication, `from_source`,
  `new_empty`, baseline caching, ARI/NMI triggers, automatic `refresh_every_n`
  behaviour, historical-edge retention, or `set_baseline`.
- Preserve existing `CpuHnsw::insert`, `CpuHnsw::insert_harvesting`,
  `ChutoroBuilder::build`, `ChutoroBuilder::build_session`,
  `ClusteringSession::append`, and `Chutoro::run` behaviour.
- Route every HNSW query through `CpuHnsw::search`. Do not bypass the public
  CPU HNSW surface or reach into private adapter internals.
- Keep the session CPU-only behind the existing `cpu` feature gate. Do not
  add a GPU code path.
- Do not add any new production dependency. Test-only crates already in the
  workspace (`rstest`, `rstest-bdd`, `proptest`, `kani`) may be used; adding
  any other test crate requires a fresh approval gate.
- Keep files below 400 lines. If `chutoro-core/src/session/mod.rs` or
  `chutoro-core/src/session/session_impl.rs` would exceed that limit, split the
  module before adding more code. The session module already splits
  configuration into `config.rs`; the new core-distance work should live in a
  sibling `core_distance.rs`.
- Public Rustdoc and Markdown updates must use en-GB Oxford spelling and
  follow `docs/documentation-style-guide.md`, `docs/rust-doctest-dry-guide.md`,
  and `docs/rust-testing-with-rstest-fixtures.md`.
- Use `leta` for symbol-oriented code navigation while implementing. Use
  `rust-router` to select any additional Rust-specific skill if a borrow,
  error, API, or layout issue emerges. Load `proptest`, `kani`, and `verus` on
  demand.
- Treat hexagonal architecture as a boundary check, not a pattern transplant.
  The pure helper `core_distance_from_neighbours` is the domain primitive; the
  HNSW lookup is an adapter call. Pure helpers must not import HNSW types beyond
  `Neighbour`; adapter glue must not embed the fallback rule.

## Tolerances (exception triggers)

- Scope: if satisfying `11.1.4` requires modifying more than 12 files or more
  than 600 net lines outside this ExecPlan, stop and escalate.
- Interface: if `recompute_core_distances` cannot be implemented as
  `pub fn recompute_core_distances(&mut self) -> Result<()>` or
  `recompute_core_distances_full` as
  `pub fn recompute_core_distances_full(&mut self) -> Result<()>`, stop and
  present alternatives with trade-offs.
- Public type: if a new public error variant or new public type other than
  the read-only accessor and the two methods above appears necessary, stop and
  justify the compatibility impact before proceeding.
- Semantics: if the dirty-set fail-fast contract has to weaken to
  silently-skip stale points, stop and present alternatives with trade-offs.
- Dependencies: if implementation requires adding any production crate, stop
  and ask for approval.
- Formal verification: if the Verus proof for
  `core_distance_from_neighbours` cannot be discharged within four iterations,
  drop to a `proptest` invariant and record the decision.
- Validation: if `make check-fmt`, `make lint`, or `make test` still fails
  after two repair attempts caused by this change, stop and escalate with the
  captured `/tmp` logs.
- Review: after each major milestone, run `coderabbit review --agent`. If it
  reports concerns that affect correctness, public API clarity, or validation,
  address them or record why they are out of scope before moving on.
- Time: if a milestone takes more than 8 working hours of wall time, stop
  and escalate.

## Risks

- Risk: the incremental recompute set ("union of new points' HNSW
  neighbours") misses existing points whose true `m`-th nearest neighbour is
  now the new point but who themselves were not visited by HNSW for the new
  point. Severity: medium. Likelihood: high. Mitigation: ship
  `recompute_core_distances_full` in this milestone and document the
  approximation explicitly in `docs/chutoro-design.md` §12.4. Roadmap item
  `11.2.4` will use this method as the trigger-(c) `refresh_full` body. The
  property test in Stage A.3 asserts the parity *of the full-recompute path*
  against batch, so the canary against silent drift is in place from day one.

- Risk: using `f32::INFINITY` as the "not yet computed" sentinel can leak
  into mutual-reachability weights computed by later roadmap items. Severity:
  high. Likelihood: medium. Mitigation: route every read through the
  `core_distance(i)` accessor, which returns `None` when the cell is `INFINITY`
  and `None` when `i >= point_count()`. Document the convention in
  `chutoro-core/src/session/core_distance.rs`. Also maintain a
  `dirty_core_distances: Vec<bool>` whose cells are clear iff the corresponding
  cell holds a real value; the accessor consults the dirty vector, not the
  sentinel, so future refactors cannot accidentally treat `INFINITY` as a real
  distance.

- Risk: an append fails partway through and leaves earlier successful
  insertions without core distances. Severity: high. Likelihood: medium.
  Mitigation: track those points as dirty in `dirty_core_distances`. Define
  `recompute_core_distances` to clear dirty bits for the points it processes.
  Document explicitly that callers reading `core_distance(i)` immediately after
  a failed append may see `None` for the successfully inserted points until a
  `recompute_core_distances` call has been made. This is enforceable as an
  invariant later: roadmap item `11.2.1` (`refresh`) will refuse to proceed
  while the dirty set is non-empty.

- Risk: append-time cost grows with `|new| * ef + |touched_existing| * ef`,
  which can balloon in dense regions. Severity: medium. Likelihood: medium.
  Mitigation: keep recompute outside of `append`. The caller invokes
  `recompute_core_distances` explicitly between `append` and any future
  `refresh`. Expose the fan-out as
  `chutoro.session.core_distance.touched_existing_per_recompute`.

- Risk: monotonic non-increasing only holds once a point has at least
  `min_cluster_size` non-self neighbours, not against the under-populated
  fallback value. Severity: low. Likelihood: medium. Mitigation: the proptest
  invariant compares values only after saturation and only across paths that
  call `recompute_core_distances_full` between observations, removing both the
  fallback and staleness confounds. A separate proptest checks exact
  incremental-vs-full parity for the first append.

- Risk: HNSW `search` returns a vector of `Neighbour` that includes the
  query if the index already contains it; the batch path filters this out.
  Severity: medium. Likelihood: high. Mitigation: do the same filter in the new
  adapter helper and assert in a test that filtering matches batch behaviour.
  The filtering rule lives in the adapter, not in the pure helper.

- Risk: divergence from the FISHDBC reference implementation, which
  piggy-backs on the HNSW distance cache rather than running a fresh k-NN
  search. Severity: low. Likelihood: high. Mitigation: this divergence can
  produce *more accurate* core distances than the reference after saturation.
  Before a point has at least `min_cluster_size` non-self neighbours, the
  batch-compatible fallback rule can still increase. Document it in
  `docs/chutoro-design.md` §12.4 and the roadmap item `11.1.4` monotonicity
  property so core-distance reviewers expect benign drift only in the saturated
  regime.

- Risk: the new `core_distance(point) -> Option<f32>` API conflates "out of
  range" with "not yet computed". Severity: low. Likelihood: medium.
  Mitigation: document both conditions explicitly in the Rustdoc and expose
  `point_count()` so callers can disambiguate. A typed `Result`-shaped API is
  deferred until a real consumer outside the session module is built (roadmap
  item `11.5.x`).

## Progress

- [x] (2026-06-05 00:00Z) Loaded the `leta`, `rust-router`, `verus`, `kani`,
  and `execplans` skills.
- [x] (2026-06-05 00:00Z) Created a `leta` workspace for the worktree.
- [x] (2026-06-05 00:00Z) Renamed the local branch to
  `11-1-4-incremental-core-distance-computation`. The remote branch does not
  yet exist; upstream tracking will be established on first push.
- [x] (2026-06-05 00:00Z) Reviewed `docs/roadmap.md` §11.1.4 and
  `docs/chutoro-design.md` §12.3 and §12.4; surveyed existing
  `ClusteringSession`, `SessionConfig`, `CpuHnsw::insert_harvesting`,
  `CpuHnsw::search`, and `cpu_pipeline::run_cpu_pipeline_with_len`.
- [x] (2026-06-05 00:00Z) Ran a Firecrawl prior-art sweep on FISHDBC core
  distance semantics, the HDBSCAN core distance definition, mutual
  reachability, hnswlib's self-hit behaviour, and Rust HNSW crate options.
- [x] (2026-06-05 00:00Z) Ran a Logisphere community-of-experts design review
  on the initial sketch and incorporated the recommendations (separate
  recompute method, dirty vector, full recompute escape hatch, pure
  recompute-set decision helper, fan-out histogram, mandatory Verus proof,
  single accessor that treats `INFINITY` as unset).
- [x] (2026-06-05 00:00Z) Drafted this ExecPlan for approval before
  implementation begins.
- [x] (2026-06-16 00:00Z) Received explicit implementation approval in the
  task request and moved this ExecPlan into implementation.
- [x] (2026-06-16 00:00Z) Loaded the implementation skills named by the plan:
  `leta`, `rust-router`, `verus`, `proptest`, `rust-verification`, `kani`,
  `rust-unit-testing`, `rust-errors`, `arch-crate-design`,
  `rust-types-and-apis`, `hexagonal-architecture`, `execplans`,
  `commit-message`, and `pr-creation`.
- [x] (2026-06-16 00:00Z) Started Stage A and rechecked the planned
  monotonicity property against the batch fallback rule before adding tests.
- [x] (2026-06-16 00:00Z) Stage A red test run captured the intended missing
  API/module failures in
  `/tmp/red-core-distance-chutoro-11-1-4-incremental-core-distance-computation.out`.
- [x] (2026-06-16 00:00Z) Stage A added unit, BDD, and proptest coverage for
  dirty core-distance reads, incremental recompute, full recompute, fallback
  semantics, pure recompute targets, saturated monotonicity, and batch parity.
- [x] (2026-06-16 00:00Z) Stage B implemented
  `chutoro-core/src/session/core_distance.rs` with the pure helper functions
  and added `verus/session_core_distance.rs`.
- [x] (2026-06-16 00:00Z) Stage B validation passed with `make verus`; the
  existing edge-harvest proof reported `21 verified, 0 errors` and the new
  session core-distance proof reported `4 verified, 0 errors`.
- [x] (2026-06-16 00:00Z) Stage C implemented source-indexed core-distance
  storage, dirty tracking, the `core_distance` accessor,
  `recompute_core_distances`, `recompute_core_distances_full`, and metrics
  instrumentation.
- [x] (2026-06-16 00:00Z) Stage C focused validation passed:
  `cargo test -p chutoro-core session::tests::core_distance --all-features` ran
  15 core-distance tests successfully.
- [x] (2026-06-16 00:00Z) Stage D updated the BDD lifecycle scenario, the
  CPU-enabled trybuild API fixture, `docs/users-guide.md`,
  `docs/developers-guide.md`, and `docs/chutoro-design.md`. The roadmap
  checkbox remains unchecked until the full validation gates pass.
- [x] (2026-06-16 00:00Z) Stage E full validation passed with `make fmt`,
  `make markdownlint`, `make check-fmt`, `make lint`, `make test`, and
  `make verus`. `make test` reported `987 passed, 1 skipped`, and Verus reported
  `21 verified, 0 errors` for edge harvest plus `4 verified, 0 errors` for
  session core-distance proofs.
- [x] (2026-06-16 00:00Z) Stage D marked `docs/roadmap.md` §11.1.4 complete
  after the full validation gates passed.
- [x] (2026-06-16 00:00Z) Stage E CodeRabbit review passed with
  `coderabbit review --agent`, reporting `findings: 0`. Earlier CodeRabbit
  comments were addressed by removing production `expect` paths and clarifying
  the core-distance test helpers without exceeding the 400-line file limit.
- [x] (2026-06-16 00:00Z) Stage F committed the implementation as
  `fa1427b`, pushed the branch, and updated draft PR #133 with the
  implementation walkthrough and validation log.
- [x] Stage A: add failing tests that pin the read-only accessor, the
  incremental recompute method, the dirty-bit semantics, the full-recompute
  escape hatch, the monotonicity property, and the batch parity property.
- [x] Stage B: implement the pure `core_distance_from_neighbours` helper
  in `chutoro-core/src/session/core_distance.rs` and prove it in Verus.
- [x] Stage C: implement the recompute engine and the public methods in
  `chutoro-core/src/session/session_impl.rs`, route the accessor through the
  dirty vector, and wire telemetry.
- [x] Stage D: update the trybuild fixture, the public session-API surface
  test, and `docs/{users-guide,developers-guide,chutoro-design,roadmap}.md`.
- [x] Stage E: run validation (`make fmt`, `make markdownlint`,
  `make check-fmt`, `make lint`, `make test`, optional `make verus`,
  `coderabbit review --agent`).
- [x] Stage F: commit, push the branch, and update the draft PR with the
  validation log.

## Surprises & Discoveries

- Observation: the FISHDBC reference does *not* implement the cleaner
  spec in roadmap `11.1.4`. Reference behaviour piggy-backs on the HNSW
  distance cache populated during insertion. Implementing the roadmap-specified
  k-NN search produces more accurate core distances than the reference,
  monotonically. Evidence: `matteodellamico/flexible-clustering/fishdbc.py`
  lines 116–156 and the HDBSCAN core distance definition in
  `hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html`. Impact: the
  incremental core-distance workstream in roadmap item `11.1.4` and
  `docs/chutoro-design.md` section 12.4 record the drift caveat relative to a
  reference port.

- Observation: HNSW `search(source, query, ef)` returns the query itself
  among its neighbours when the query is already in the index. The batch
  pipeline filters it out before computing core distance. Evidence:
  `chutoro-core/src/cpu_pipeline.rs` lines 64–76; the `ensure_query_present`
  step inside `CpuHnsw::search`. Impact: the incremental adapter must apply the
  same filter; the pure helper assumes the filter has already happened.

- Observation: `_core_distances` already exists as a leading-underscore
  field of `ClusteringSession`. The append milestone retained the underscore
  until this roadmap item used it. Evidence:
  `chutoro-core/src/session/mod.rs:89`. Impact: this milestone renames the
  field to `core_distances` and adds the sibling `dirty_core_distances` field.

- Observation: the planned "core distances are monotonically non-increasing
  across all appends" property conflicts with the batch fallback rule for
  under-populated indices. Example: with `min_cluster_size = 2`, point `0`
  inserted alongside only point `1` has fallback core distance `1.0`; after
  point `2` is appended, the true second-neighbour core distance becomes `2.0`.
  Batch parity and the under-populated fallback tests require this increase.
  Evidence: `chutoro-core/src/cpu_pipeline.rs` uses the last available non-self
  neighbour when fewer than `min_cluster_size` neighbours exist. Impact: Stage
  A keeps the explicit fallback tests and narrows the monotonicity property to
  observations where the point already had at least `min_cluster_size` non-self
  neighbours before the append.

- Observation: the repository guidance points new work at `docs/contents.md`,
  but that file does not exist in this branch. `docs/repository-layout.md` has
  since been added. Evidence: a direct read of `docs/contents.md` returned "No
  such file or directory". Impact: orientation used `AGENTS.md`, this ExecPlan,
  semantic `leta` queries, and the local session, CPU pipeline, HNSW,
  users-guide, developers-guide, design, and roadmap files instead.

- Observation: the workspace does not currently depend on `fixedbitset`.
  Evidence:
  `rg "fixedbitset|FixedBitSet" Cargo.toml chutoro-core/Cargo.toml chutoro-core/src`
  found no dependency or use. Impact: dirty core-distance state is implemented
  with `Vec<bool>` to satisfy the no-new-production-dependency constraint.

## Decision Log

- Decision: keep core-distance recompute out of `append` and require the
  caller to invoke `recompute_core_distances` explicitly between `append` and
  any future `refresh`. Rationale: per-point recompute fan-out can grow
  significantly in dense regions; coupling it to `append` would make append
  latency hard to predict and would hide a metric (fan-out) that future
  incident response needs. Treating recompute as a discrete pipeline stage
  matches the "micro-batched snapshot model" described in
  `docs/chutoro-design.md` §12.2. The community review recommended this split.
  Date/Author: 2026-06-05, planning.

- Decision: ship `recompute_core_distances_full` in this milestone, even
  though roadmap item `11.2.4` is the eventual owner of the `refresh_full`
  trigger path. Rationale: without a full-recompute path, the proptest battery
  cannot include a batch-parity property, and there is no canary against silent
  drift across many append cycles. The full-recompute body is a small loop that
  mirrors `cpu_pipeline.rs`; it is cheap to implement now and unblocks the
  property test that the design document already requires. This decision is
  consistent with `docs/chutoro-design.md` §12.4's description of a full
  core-distance recomputation path. Roadmap item `11.2.4` will reuse the same
  method body. Date/Author: 2026-06-05, planning.

- Decision: maintain a `dirty_core_distances: Vec<bool>` indexed by
  source index. A clear bit means the cell holds a real value; a set bit means
  the cell is stale or never computed. Rationale: enforces the invariant "every
  successfully inserted point has a real core distance" without requiring the
  recompute call to succeed before `append` returns. `append` sets dirty bits
  for newly inserted points; `recompute_core_distances` clears them as it
  processes points; future `refresh` (roadmap `11.2.1`) will refuse to proceed
  while any dirty bit is set. The `Vec<bool>` storage avoids a new production
  dependency while keeping `core_distances` densely indexable for later
  mutual-reachability work. Date/Author: 2026-06-05, planning.

- Decision: use `f32::INFINITY` as the storage sentinel but route every
  read through the `core_distance(i)` accessor, which consults
  `dirty_core_distances` rather than the sentinel. Rationale: dense `Vec<f32>`
  storage is simpler than `Vec<Option<f32>>`; the dirty vector is the
  authoritative "is this real?" source; the sentinel is a debugging aid:
  observing `INFINITY` in mutual-reachability output means the dirty vector was
  not consulted. The community review recommended a single guarded accessor;
  this design enforces that with the dirty vector. Date/Author: 2026-06-05,
  planning.

- Decision: factor the "which existing points need recompute?" step into
  a pure helper
  `recompute_targets(new_indices, neighbour_lists) -> RecomputeSet`, living in
  `core_distance.rs`. The HNSW adapter is responsible only for performing the
  searches and applying the self-hit filter. Rationale: matches the domain/port
  split. The recompute-target decision is data-shaped and easy to
  property-test; the HNSW search is adapter-shaped and easy to
  integration-test. The community review flagged this as the main boundary leak
  in the original sketch. Date/Author: 2026-06-05, planning.

- Decision: discharge the Verus proof for
  `core_distance_from_neighbours` in this milestone rather than treating it as
  optional. Rationale: the function is short (around six lines), its two
  obligations (selection equation and fallback rule) are both within reach of
  Verus's solver, and the proof unlocks confidence in the monotonicity
  proptest. Cost is low; benefit is permanent. The community review
  reclassified it from optional to required. Date/Author: 2026-06-05, planning.

- Decision: do not add a Kani harness for the pure helper in this
  milestone. Rationale: Verus over an unbounded sequence is a stronger
  statement than Kani over a bounded one; once Verus has discharged the same
  obligation, an additional Kani harness adds maintenance cost without changing
  the trust story. Kani may be revisited if a future `core_distance.rs` change
  introduces structural state that benefits from bounded model checking.
  Date/Author: 2026-06-05, planning.

- Decision: keep the public accessor as
  `pub fn core_distance(&self, point: usize) -> Option<f32>` with `None`
  returned both for `point >= point_count()` and for cells whose dirty bit is
  set. Rationale: the design document does not yet specify a richer error type
  for session reads. `point_count()` is already public, so the caller can
  disambiguate. The community review marked a typed error as the better
  long-term option; that is deferred to the public-surface hardening pass in
  roadmap item `11.5.1`. Date/Author: 2026-06-05, planning.

- Decision: treat the 2026-06-16 user request to proceed with implementation
  as the explicit approval gate for this ExecPlan. Rationale: the request
  directly instructed implementation of the planned functionality and repeated
  the requirement to keep the ExecPlan current. This satisfies the
  draft-to-execution approval gate without changing the plan's scope or
  tolerances. Date/Author: 2026-06-16, implementation.

- Decision: constrain the monotonicity proptest to the saturated regime, where
  each observed point already has at least `min_cluster_size` non-self
  neighbours before the append. Rationale: monotonic non-increase is true for
  the m-th nearest-neighbour statistic once the m-th neighbour exists. It is
  false while the batch fallback is using "last available neighbour" for
  under-populated points. Keeping both batch parity and fallback compatibility
  is more important than proving an over-broad property. Date/Author:
  2026-06-16, implementation.

- Decision: implement dirty core-distance tracking with `Vec<bool>` rather
  than `FixedBitSet`. Rationale: `fixedbitset` is not already in the workspace
  dependency graph, and this milestone explicitly forbids new production
  dependencies. The `Vec<bool>` representation preserves the same
  source-indexed invariant: `true` means stale or never computed; `false` plus
  a finite stored distance means `core_distance(i)` may return `Some(_)`.
  Date/Author: 2026-06-16, implementation.

- Decision: store core distances by source index rather than dense insertion
  ordinal. Rationale: `append` accepts source indices and does not require
  contiguous insertion order. Source-indexed storage lets `core_distance(i)`
  use the same identifier that callers passed to `append`, including sessions
  that append `[5]` before lower source indices. Date/Author: 2026-06-16,
  implementation.

## Outcomes & Retrospective

The milestone landed as planned with one scoped deviation: dirty core-distance
state uses `Vec<bool>` rather than `FixedBitSet` because the workspace did not
already depend on `fixedbitset`, and the task forbade new production
dependencies. The more important semantic deviation came from the proof and
property review: monotonicity only holds once a point has at least
`min_cluster_size` non-self neighbours. Before saturation, the batch-compatible
fallback rule can increase as a later neighbour becomes available.

The focused proptests did not reveal additional runtime regressions after that
scope correction. The Verus proof effort stayed small because the proof covers
the pure selection and fallback rules rather than the HNSW adapter. CodeRabbit
review caught clarity and panic-path issues in the final shape, all of which
were addressed before the implementation commit.

## Context and orientation

This worktree is a Rust workspace rooted at the repository root. The relevant
crate is `chutoro-core`. Code under `chutoro-core/src/session/` already
implements the public session surface up to roadmap item `11.1.3`. Specifically:

- `chutoro-core/src/session/mod.rs` defines the public
  `ClusteringSession` struct and its read-only accessors `config()`,
  `point_count()`, and `snapshot_version()`.
- `chutoro-core/src/session/session_impl.rs` defines session construction
  (`new_with_capacity`) and the mutating `append` method.
- `chutoro-core/src/session/config.rs` defines `SessionConfig` and
  `SessionRefreshPolicy`.
- `chutoro-core/src/session/tests/common.rs` defines the shared
  `SessionTestSource`, the `session_builder` fixture, and
  `harvest_expected_edges`.
- `chutoro-core/src/session/tests/append.rs` exercises `append`.

The batch CPU pipeline lives at `chutoro-core/src/cpu_pipeline.rs`. Lines 64–76
contain the canonical core-distance computation: for each point `p`,
`index.search(source, p, ef)` is called, the result is filtered to remove the
self-hit, and the core distance is either
`others[min_cluster_size - 1].distance` (when there are enough others) or the
distance of the last available neighbour (fallback for under-populated
indices), or `0.0` if the filtered list is empty. The effective `ef` used by
the batch path is `max(min_cluster_size + 1, ef_construction).min(items)`. The
incremental work in this plan mirrors that rule so that
`recompute_core_distances_full` produces the same vector as `cpu_pipeline.rs`
on the same dataset.

The HNSW public surface in `chutoro-core/src/hnsw/cpu/mod.rs`:

```rust
pub fn search<D: DataSource + Sync>(
    &self,
    source: &D,
    query: usize,
    ef: NonZeroUsize,
) -> Result<Vec<Neighbour>, HnswError>;
```

returns the query node itself among its neighbours when the query is already in
the index; the caller must filter it out, matching the batch code at
`cpu_pipeline.rs:69`.

`Neighbour` is defined in `chutoro-core/src/hnsw/types.rs` as a public struct
with `id: usize` and `distance: f32`.

This plan adds three pieces of code:

1. A new file `chutoro-core/src/session/core_distance.rs` that contains
   the pure helpers (`core_distance_from_neighbours`, `recompute_targets`, and
   `effective_ef`).
2. New private fields and three new public methods on
   `ClusteringSession` in `chutoro-core/src/session/mod.rs` and
   `chutoro-core/src/session/session_impl.rs`.
3. A new test file `chutoro-core/src/session/tests/core_distance.rs`
   that exercises the new behaviour with `rstest`, with a small `proptest`
   battery for monotonicity and batch parity.

The plan also adds a single Verus proof file at
`verus/session_core_distance.rs` (project layout per the `verus` skill) and
registers it with the existing `make verus` target.

## Plan of work

### Stage A: failing tests

Work in `chutoro-core/src/session/tests/core_distance.rs` (new file). Register
the module from `chutoro-core/src/session/tests.rs`.

Add the following `rstest` cases, all failing before any production change:

1. `core_distance_returns_none_before_append`: build an empty session
   over a four-point source; assert `core_distance(0) == None`.
2. `core_distance_returns_none_after_append_before_recompute`: build a
   session, `append(&[0, 1, 2, 3])`, then assert each
   `core_distance(i) == None`. This pins the dirty-bit semantics.
3. `recompute_core_distances_clears_dirty_bits`: same as above, then
   call `session.recompute_core_distances()?`; assert each `core_distance(i)` is
   `Some(_)` and finite.
4. `recompute_core_distances_matches_batch_per_point`: build a small
   four-point source, append all indices, call
   `recompute_core_distances_full()`, and compare each `core_distance(i)`
   against the value computed by directly replicating the `cpu_pipeline.rs`
   lines 64–76 logic. Tolerance: exact equality with `f32::total_cmp`; the two
   paths perform the same arithmetic in the same order.
5. `core_distance_uses_min_cluster_size_minus_one_offset`: build a
   session with `min_cluster_size = 3` and a six-point source; verify that
   `core_distance(0)` after recompute equals the third element of the
   sorted-non-self distances from point 0, mirroring `cpu_pipeline.rs`.
6. `core_distance_fallback_when_index_smaller_than_min_cluster_size`:
   append two points with `min_cluster_size = 3`, run recompute, and verify the
   fallback rule returns the distance to the single non-self neighbour rather
   than zero.
7. `core_distance_empty_neighbour_list_yields_zero`: append a single
   point with `min_cluster_size = 3` and verify `core_distance(0) == Some(0.0)`
   after recompute, matching the `unwrap_or(0.0)` branch in `cpu_pipeline.rs`.
8. `recompute_core_distances_recomputes_touched_existing_points`:
   append `&[0, 1]`, recompute, capture `core_distance(0)`; then append `&[2]`
   and recompute; assert that `core_distance(0)` is now strictly less than or
   equal to the captured value (the new point may have become a closer
   neighbour for point 0).
9. `core_distance_out_of_range_returns_none`: build a session, append a
   single point, assert `core_distance(point_count()) == None`.
10. `recompute_core_distances_full_recomputes_all_points`: build a
    session, append eight points, call `recompute_core_distances()`
    once (incremental), append two more, call
    `recompute_core_distances_full()`, and confirm every cell is
    `Some(_)` and finite.
11. `recompute_targets_excludes_query_itself`: a unit test against the
    pure `recompute_targets` helper using hand-crafted
    `(new_indices, neighbour_lists)` to confirm that no `new` point is
    placed in the recompute-existing set.
12. `recompute_targets_unions_existing_neighbours`: pure unit test
    verifying the set semantics with two new points sharing one
    existing neighbour.

Add a `proptest` battery with these properties:

- `prop_core_distance_monotonically_non_increasing_after_saturation`:
  generate a `min_cluster_size` between 1 and 4 inclusive and an append
  sequence; after each append invoke `recompute_core_distances_full()`; assert
  monotonic non-increase only for points that have at least `min_cluster_size`
  non-self neighbours before the comparison. Source length is bounded to 24.
  The guard is required because the fallback value can increase before
  saturation.
- `prop_recompute_full_matches_batch`: generate a four-to-twelve
  point source and a `min_cluster_size` between 1 and 3; build an empty
  session, append every index, call `recompute_core_distances_full()`, and
  assert the resulting vector equals the output of a freshly run
  `cpu_pipeline::run_cpu_pipeline_with_len` core-distance loop on the same
  source (extracted as a small test-only helper if not already exposed).
- `prop_incremental_recompute_matches_full_for_first_append`:
  build a session, append all indices, call `recompute_core_distances()`
  (incremental), and assert it equals `recompute_core_distances_full()`
  (because on the first append every point is new and the incremental set
  equals the full set).

Persist `proptest` regression files under
`chutoro-core/proptest-regressions/session/` as the test framework already does
for the append milestone.

Add an `rstest-bdd` behavioural scenario that demonstrates the public
lifecycle: build session → append → recompute → read `core_distance(i)`. Reuse
the existing `rstest-bdd` setup added in roadmap item `11.1.3` rather than
introducing new harness.

Confirm the new tests fail before any production change is made:

```sh
cargo test -p chutoro-core session::tests::core_distance --all-features
```

### Stage B: pure helpers and Verus proof

Create `chutoro-core/src/session/core_distance.rs` and register it from
`chutoro-core/src/session/mod.rs`. Implement the following pure helpers:

```rust
use std::num::NonZeroUsize;
use crate::Neighbour;

/// Computes the core distance from a sorted-ascending list of
/// neighbours that has already had the query point removed.
///
/// Returns `neighbours[m - 1].distance` when there are at least
/// `m` neighbours, the last available neighbour's distance when
/// the list is non-empty but shorter than `m`, or `0.0` when the
/// list is empty.
pub(super) fn core_distance_from_neighbours(
    neighbours: &[Neighbour],
    min_cluster_size: NonZeroUsize,
) -> f32;

/// Computes the effective `ef` used when querying neighbours
/// during core-distance recompute. Mirrors `cpu_pipeline.rs`.
pub(super) fn effective_ef(
    min_cluster_size: NonZeroUsize,
    ef_construction: NonZeroUsize,
    point_count: NonZeroUsize,
) -> NonZeroUsize;

/// Pure decision function: given the set of newly-inserted point
/// indices and the per-new-point neighbour lists returned by HNSW
/// (already filtered to remove self-hits), returns the set of
/// existing points whose core distances must be recomputed.
///
/// The returned vector is deduplicated and in ascending order.
pub(super) fn recompute_targets(
    new_indices: &[usize],
    neighbour_lists: &[&[Neighbour]],
) -> Vec<usize>;
```

`core_distance_from_neighbours` mirrors `cpu_pipeline.rs` exactly.
`effective_ef` mirrors the existing batch formula
`max(min_cluster_size + 1, ef_construction).min(point_count)`.
`recompute_targets` collects the union of all neighbour IDs across
`neighbour_lists`, removes any IDs that are themselves in `new_indices`, sorts
ascending, and deduplicates.

Add the Verus proof at `verus/session_core_distance.rs` with these obligations:

1. `lemma_core_distance_selection`: if
   `neighbours.len() >= m`, then
   `core_distance_from_neighbours(neighbours, m) == neighbours[m - 1].distance`.
2. `lemma_core_distance_fallback`: if `0 < neighbours.len() < m`, then:

   ```rust
   core_distance_from_neighbours(neighbours, m) == neighbours[neighbours.len() - 1].distance
   ```

3. `lemma_core_distance_empty`:
   `core_distance_from_neighbours(&[], m) == 0.0`.
4. `lemma_core_distance_monotone_under_saturated_prefix`: if `prefix`
   is a prefix of `extended` and `prefix` already has at least `m` neighbours,
   then the extended core distance is less than or equal to the prefix core
   distance. The saturation precondition is required because the fallback rule
   can increase before the m-th non-self neighbour exists.

Use the `vstd::seq` primitives and follow the project layout in the `verus`
skill. Register the proof file with `make verus` by adding it to
`scripts/run-verus.sh` (or the existing proof manifest).

### Stage C: engine and public methods

Work in `chutoro-core/src/session/mod.rs` and
`chutoro-core/src/session/session_impl.rs`.

In `mod.rs`, rename `_core_distances: Vec<f32>` to `core_distances: Vec<f32>`
and add a sibling field `dirty_core_distances: Vec<bool>`. Update the
`assert_send_sync` compile-time check if needed.

Add three public methods on `ClusteringSession`:

```rust
/// Returns the core distance for `point`, or `None` if `point`
/// is out of range or its core distance has not been computed
/// yet.
pub fn core_distance(&self, point: usize) -> Option<f32>;

/// Recomputes core distances for every point whose dirty bit is
/// set: that is, every newly-inserted point since the last
/// recompute, plus every existing point that appears as a
/// neighbour of one of those new points.
pub fn recompute_core_distances(&mut self) -> Result<()>;

/// Recomputes core distances for every point in the index by
/// mirroring the batch `cpu_pipeline.rs` path. Clears every
/// dirty bit. This is the body that roadmap item 11.2.4 will
/// invoke as the trigger-(c) `refresh_full` path.
pub fn recompute_core_distances_full(&mut self) -> Result<()>;
```

`core_distance` consults `dirty_core_distances` first; if the bit is set or
missing, return `None`. It also returns `None` for unset, out-of-range, or
non-finite cells; otherwise return `Some(core_distances[point])`.

Modify `ClusteringSession::append` in `session_impl.rs` to:

1. Before inserting, resize `core_distances` and `dirty_core_distances`
   together to cover the highest requested source index, filling new
   `core_distances` cells with `f32::INFINITY`.
2. For each successful insertion at source index `i`, set bit `i`
   in `dirty_core_distances`. This must happen even on the path that ultimately
   returns an error for a later index (the community-review pre-mortem fix).

Implement `recompute_core_distances` as follows:

1. Collect the list of dirty source indices that are new (set bit
   AND no previous value, identified by `core_distances[i] == f32::INFINITY`).
2. For each new index, call `CpuHnsw::search(self.source.as_ref(), index, ef)`
   with
   `ef = effective_ef(min_cluster_size, hnsw_params.ef_construction(), point_count())`.
3. Filter the self-hit from each neighbour list.
4. Use `recompute_targets` to derive the set of existing points
   that need recompute.
5. For each new index, write its core distance using
   `core_distance_from_neighbours` and clear its dirty bit.
6. For each existing index in the recompute-targets set, run an
   HNSW search, filter the self-hit, compute the core distance, and clear its
   dirty bit if currently set. If it is not dirty, still update the value (its
   true core distance may have shrunk).

Implement `recompute_core_distances_full` by running the same loop over
`0..point_count()`.

Add the error mapping for HNSW search failures via the existing
`map_hnsw_error` helper from `session_impl.rs`.

Wire the telemetry (gated behind `feature = "metrics"`):

- Counter `chutoro.session.core_distance.queries_total` incremented
  once per HNSW search.
- Counter `chutoro.session.core_distance.recomputed_existing` for
  existing-point recomputes.
- Counter `chutoro.session.core_distance.appends_left_dirty_total`
  incremented once per call to `recompute_core_distances` whose pre-state had a
  non-empty dirty set (this is the community-review-recommended canary for
  "appends that left dirty points behind").
- Histogram `chutoro.session.core_distance.touched_existing_per_recompute`
  recording the size of the recompute-targets set per call.
- Histogram `chutoro.session.core_distance.recompute_seconds`
  recording wall time per recompute call, using the same `MonotonicClock`
  injection point as the existing append histogram.

### Stage D: documentation and public surface tests

Update:

- `chutoro-core/tests/trybuild/session_api_cpu_enabled.rs`: add a
  call to `session.recompute_core_distances()?` and a read of
  `session.core_distance(0)`.
- `chutoro-core/tests/session_api_surface.rs`: extend the public
  surface assertion to include the three new methods. Update the CPU-disabled
  compile-fail fixture only if its stderr changes; if it does, record the new
  expected output and explain in `Surprises & Discoveries`.
- `docs/users-guide.md` "Incremental clustering sessions"
  section: add `recompute_core_distances`, `recompute_core_distances_full`, and
  `core_distance` to the lifecycle example. Document that:
  - `recompute_core_distances` must be called after `append` and
    before any future `refresh` work to refresh stale core
    distances; reading `core_distance(i)` for a newly-appended
    `i` before recompute returns `None`.
  - `recompute_core_distances_full` is more expensive but
    re-establishes parity with a from-scratch batch run; this is
    the path future drift-correction work will use.
- `docs/developers-guide.md` "Session public APIs" section: add
  the three new methods, the dirty-vector invariant, and the pure-helper layout
  in `session/core_distance.rs`. Note that `core_distance(i)` returns `None`
  for both out-of-range and dirty cells, and direct callers should disambiguate
  with `point_count()`.
- `docs/chutoro-design.md` §12.4: add a "v1 implemented behaviour"
  note describing the incremental recompute set, the full recompute escape
  hatch, the dirty-bit invariant, and the benign downward divergence from the
  FISHDBC reference. No new diagrams are required.
- `docs/roadmap.md` §11.1.4: mark `[x]` only after all gates
  pass.

No ADR is required; the relevant decisions live in this ExecPlan's Decision Log
and reference `docs/chutoro-design.md` §12.4.

### Stage E: validate and review

Run validations sequentially, capturing logs with `tee` under `/tmp` per the
AGENTS.md guidance. Per-action and per-branch log filenames are constructed
with the documented template
`/tmp/$ACTION-$(get-project)-$(git branch --show-current).out`.

```sh
make fmt 2>&1 | tee \
  "/tmp/fmt-chutoro-$(git branch --show-current).out"
make markdownlint 2>&1 | tee \
  "/tmp/markdownlint-chutoro-$(git branch --show-current).out"
make check-fmt 2>&1 | tee \
  "/tmp/check-fmt-chutoro-$(git branch --show-current).out"
make lint 2>&1 | tee \
  "/tmp/lint-chutoro-$(git branch --show-current).out"
make test 2>&1 | tee \
  "/tmp/test-chutoro-$(git branch --show-current).out"
make verus 2>&1 | tee \
  "/tmp/verus-chutoro-$(git branch --show-current).out"
coderabbit review --agent 2>&1 | tee \
  "/tmp/coderabbit-chutoro-$(git branch --show-current).out"
```

`make nixie` is only required if Mermaid diagrams change; this plan does not
change any.

If CodeRabbit reports correctness concerns or public-API clarity concerns,
address them or record an explicit out-of-scope decision in `Decision Log`
before moving on.

### Stage F: commit and push

Commit only after the gates pass. Use the file-based commit-message workflow
per the `commit-message` skill:

```sh
git status --short
git diff -- chutoro-core docs
git add chutoro-core docs verus
COMMIT_MSG_DIR="$(mktemp -d)"
cat > "$COMMIT_MSG_DIR/COMMIT_MSG.md" << 'ENDOFMSG'
Implement incremental core-distance computation

Add `ClusteringSession::recompute_core_distances` and
`recompute_core_distances_full` so a session can derive
HDBSCAN core distances for newly appended points and for the
existing points that appeared as neighbours of those new
points. Add the read-only `core_distance(point)` accessor.

Track stale and freshly-inserted cells in a dirty vector so
that callers reading core distances before recompute see
`None` rather than a sentinel, and so that a future refresh
can refuse to proceed while dirty cells remain.

Carry the Verus proof for the pure
`core_distance_from_neighbours` helper, mirroring the
selection equation, the fallback rule, and the prefix
monotonicity property.
ENDOFMSG
git commit -F "$COMMIT_MSG_DIR/COMMIT_MSG.md"
rm -rf "$COMMIT_MSG_DIR"
```

For this pre-implementation ExecPlan PR, commit only the plan file and open a
draft PR before any implementation begins.

Push the branch with upstream tracking:

```sh
git push -u origin 11-1-4-incremental-core-distance-computation
```

## Concrete steps

Run from the worktree root:

```sh
git branch --show-current
git status --short --branch
leta workspace add "$PWD"
leta show ClusteringSession -n 10
leta show CpuHnsw.search -n 10
```

Expected branch: `11-1-4-incremental-core-distance-computation`.

The targeted in-loop commands while developing are:

```sh
cargo test -p chutoro-core session::tests::core_distance \
  --all-features
cargo test -p chutoro-core --test session_api_surface \
  --all-features
```

Expected successful evidence: every command in Stage E exits with status `0`,
the new core-distance tests pass, the Verus proof file verifies, and CodeRabbit
has no unresolved concerns.

## Validation and acceptance

Acceptance phrased as behaviour:

- Running
  `cargo test -p chutoro-core session::tests::core_distance --all-features`
  exits 0; the eleven new `rstest` cases and the three `proptest` properties
  all pass.
- `make verus` exits 0 with the new `session_core_distance.rs`
  proof discharged.
- `make check-fmt`, `make lint`, and `make test` exit 0.
- After
  `let mut s = builder.build_session(source)?; s.append(&[0,1,2,3])?; s.recompute_core_distances()?;`,
  every `s.core_distance(i)` for `i in 0..4` returns `Some(_)` with a finite
  value, and the four values equal a hand-derived computation against the
  source's pairwise distances at `min_cluster_size`.
- After `s.recompute_core_distances_full()`, every cell holds
  the same value as a freshly run `cpu_pipeline.rs` core-distance loop on the
  same dataset.

Quality criteria:

- Tests: the new rstest, rstest-bdd, and proptest suites pass;
  the existing append tests continue to pass.
- Lint and type check: `make lint` (clippy `-D warnings` plus
  rustdoc with `--cfg docsrs -D warnings`) succeeds.
- Format: `make check-fmt` succeeds.
- Verification: `make verus` succeeds for
  `verus/session_core_distance.rs`.
- Performance: no benchmark threshold is part of this milestone;
  fan-out histogram telemetry is documented but not asserted.
- Security: no new attack surface; the public API expansion is
  purely read-only plus two mutating session methods that cannot exceed the
  existing `append`-time data-source access pattern.

Quality method: rerun the full validation block from Stage E locally; rely on
the existing CI nextest profile for the default validation run.

## Idempotence and recovery

All steps are re-runnable safely. Stage A creates a new test file; re-running
the failing tests is harmless. Stage B adds a new pure module and a new Verus
file; both can be deleted and re-added. Stage C is a series of edits to
existing files; mid-edit interruption can be recovered with `git restore`.
Stage D edits Markdown and trybuild fixtures; re-running them is harmless.
Stage E commands are idempotent. Stage F is the only mildly destructive step
(push); use `git push -u origin <branch>` (no force) and re-run safely if it
fails.

If a step fails partway, capture the failure in `Surprises & Discoveries`,
decide whether to roll back the partial change with `git restore -SW <paths>`,
and resume. Never use `git push --force` to recover; create a new commit
instead.

## Artifacts and notes

The community design review identified two changes that materially improved the
plan:

1. Shipping `recompute_core_distances_full` in this milestone so
   the proptest battery includes a batch-parity property.
2. Tracking a `dirty_core_distances` vector so the fail-fast
   append contract does not leave half-initialized sessions masquerading as
   healthy.

The Firecrawl prior-art sweep clarified that the FISHDBC reference
implementation does not do a fresh k-NN search at all during `add()`. The
roadmap-specified k-NN-based path produces *more accurate* core distances than
the reference, monotonically after saturation. The roadmap item `11.1.4`
monotonicity property records that benign downward drift is limited to the
saturated regime relative to a literal reference port.

## Interfaces and dependencies

In `chutoro-core/src/session/core_distance.rs`, define:

```rust
use std::num::NonZeroUsize;
use crate::Neighbour;

pub(super) fn core_distance_from_neighbours(
    neighbours: &[Neighbour],
    min_cluster_size: NonZeroUsize,
) -> f32;

pub(super) fn effective_ef(
    min_cluster_size: NonZeroUsize,
    ef_construction: NonZeroUsize,
    point_count: NonZeroUsize,
) -> NonZeroUsize;

pub(super) fn recompute_targets(
    new_indices: &[usize],
    neighbour_lists: &[&[Neighbour]],
) -> Vec<usize>;
```

In `chutoro-core/src/session/mod.rs`, extend `ClusteringSession`:

```rust
pub struct ClusteringSession<D: DataSource + Send + Sync> {
    config: SessionConfig,
    index: CpuHnsw,
    core_distances: Vec<f32>,
    dirty_core_distances: Vec<bool>,
    _mst_edges: Vec<MstEdge>,
    _historical_edges: Vec<CandidateEdge>,
    pending_edges: Vec<CandidateEdge>,
    _labels: Arc<Vec<usize>>,
    snapshot_version: u64,
    source: Arc<D>,
    _last_refresh_len: usize,
    #[cfg(feature = "metrics")]
    clock: std::sync::Arc<dyn clock::MonotonicClock>,
}
```

and expose:

```rust
impl<D: DataSource + Send + Sync> ClusteringSession<D> {
    pub fn core_distance(&self, point: usize) -> Option<f32>;
    pub fn recompute_core_distances(&mut self) -> Result<()>;
    pub fn recompute_core_distances_full(&mut self) -> Result<()>;
}
```

No external dependency changes were made. The dirty set uses `Vec<bool>` because
`fixedbitset` was not already a workspace dependency and this milestone
forbids new production dependencies.

## References

- Roadmap item: `docs/roadmap.md` §11.1.4.
- Design source: `docs/chutoro-design.md` §12.3 and §12.4.
- Prior local plans:
  `docs/execplans/11-1-1-make-edge-harvesting-hnsw-insertion-path-public.md`,
  `docs/execplans/11-1-2-define-session-config-carrying-clustering-parameters.md`,
  and `docs/execplans/11-1-3-clustering-session-append.md`.
- Batch reference path: `chutoro-core/src/cpu_pipeline.rs` lines
  47–95 for the canonical core-distance loop, the mutual reachability formula,
  and the effective `ef` rule.
- Testing guidance: `docs/property-testing-design.md`,
  `docs/rust-testing-with-rstest-fixtures.md`, and
  `docs/rust-doctest-dry-guide.md`.
- Complexity guidance:
  `docs/complexity-antipatterns-and-refactoring-strategies.md`.
- Documentation style: `docs/documentation-style-guide.md`.
- Verus pinning and runner:
  [`rust-prover-tools`](https://github.com/leynos/rust-prover-tools) per the
  `verus` skill, plus `docs/verus-toolchain.md`.
- Skills signposted for implementation: `leta`, `rust-router`,
  `rust-types-and-apis`, `rust-errors`, `hexagonal-architecture`, `proptest`,
  `verus`, `execplans`, `commit-message`, and `pr-creation`.
- Firecrawl prior art: the FISHDBC reference at
  `github.com/matteodellamico/flexible-clustering/blob/master/flexible_clustering/fishdbc.py`,
  the HDBSCAN core distance definition at
  `hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html`, the FISHDBC arXiv
  paper at `arxiv.org/abs/1910.07283`, and the hnswlib README at
  `github.com/nmslib/hnswlib`.
- Community design review: a Logisphere community-of-experts
  pre-implementation review was run on this design sketch; the full report is
  summarised in `Decision Log` and folded into the plan body.
