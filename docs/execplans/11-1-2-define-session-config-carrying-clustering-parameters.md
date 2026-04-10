# Define `SessionConfig` and add `ChutoroBuilder::build_session`

This ExecPlan (execution plan) is a living document. The sections
`Constraints`, `Tolerances`, `Risks`, `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work
proceeds.

Status: PROPOSED

Implementation must not begin until this plan is approved.

`PLANS.md` was not found in the repository root at the time of writing, so no
additional plan-governance file applies.

## Purpose / big picture

Deliver roadmap item `11.1.2` by introducing a public session-oriented
configuration surface for incremental clustering. After this change,
`ChutoroBuilder` will be able to derive a validated `SessionConfig` containing
the session-relevant clustering knobs, and `ChutoroBuilder::build_session(...)`
will return an empty `ClusteringSession` that is ready for later roadmap work
on `append`, `refresh`, and batch bootstrap.

Success is visible when:

- `SessionConfig` exists as a public CPU-session configuration type.
- `ChutoroBuilder` can carry and expose HNSW and session refresh-policy
  settings in addition to `min_cluster_size`.
- `build_session` constructs a `ClusteringSession` from builder-derived
  configuration without prematurely running the batch pipeline.
- Unit tests cover happy paths, unhappy paths, and edge cases with
  `rstest` parameterization where it improves coverage.
- `docs/chutoro-design.md` records the final API and initialization
  decisions, `docs/roadmap.md` marks `11.1.2` as done, and `make check-fmt`,
  `make lint`, and `make test` succeed.

## Constraints

- Keep scope limited to roadmap item `11.1.2`: define the session
  configuration model, add the builder/session construction surface, add the
  minimum read-only session accessors needed for tests and Rustdoc, and update
  the design and roadmap documents.
- Do not implement `ClusteringSession::append`, `refresh`,
  `refresh_full`, `labels`, `from_source`, or `new_empty` in this change. Those
  belong to later roadmap items.
- Preserve the existing `ChutoroBuilder::build()` and `Chutoro::run()`
  behaviour for the stateless batch path.
- Reuse existing validated types where possible. `SessionConfig` should
  store strong types such as `NonZeroUsize` and `HnswParams` rather than
  re-validating raw integers throughout the session code.
- Keep new files below 400 lines. Prefer a small `session/` module split
  over one large file if the public session surface and tests start to sprawl.
- Prefer whole-object builder setters for new configuration clusters
  (for example, `with_hnsw_params(...)`) over a large family of per-field
  setters unless implementation review proves that object setters are
  materially less ergonomic.
- The incremental session surface is CPU-specific today. New public
  session types and builder methods should therefore be gated behind the `cpu`
  feature unless implementation review discovers a cleaner compatibility story.
- Public Rustdoc must follow `docs/rust-doctest-dry-guide.md`, and
  documentation updates must use en-GB spelling and pass the Markdown
  validators.

## Tolerances (exception triggers)

- Scope: if satisfying `11.1.2` requires implementing any later roadmap
  item or modifying more than 10 files or 450 net lines, stop and escalate.
- Interface: if the session factory cannot be expressed as one builder
  method plus a narrow public session/config surface, stop and present the
  alternatives before proceeding.
- Dependencies: if a new crate, feature flag, or test-only dependency is
  required, stop and escalate.
- Error model: if supporting `build_session` requires a new public error
  enum instead of reusing `ChutoroError`, stop and justify that change before
  proceeding.
- Behaviour: if `build_session` cannot initialize an empty session
  without also seeding HNSW, MST, and labels, stop and revisit the task split
  against roadmap items `11.3.1` and `11.3.2`.
- Validation: if `make lint` or `make test` still fails after two repair
  iterations caused by this change, stop and escalate with the failing logs.
- Ambiguity: if the intended ownership model for the session data source
  or the acceptable behaviour for empty/undersized sources is not accepted
  during review, pause implementation and confirm those API semantics first.

## Risks

- Risk: `build_session` ownership is ambiguous between taking `D`,
  `&D`, or `Arc<D>`. Severity: medium Likelihood: high Mitigation: prefer
  `Arc<D>` in the plan because `ClusteringSession` owns long-lived state and
  later concurrent readers need shared access to the source; record the final
  rationale in the design document.

- Risk: `build_session` may accidentally inherit the batch path’s
  empty-source and undersized-source rejection semantics. Severity: high
  Likelihood: medium Mitigation: treat session construction as configuration
  validation plus empty-state initialization only; add explicit tests proving
  that empty and undersized sources are accepted for session creation.

- Risk: adding session-specific fields directly to `ChutoroBuilder`
  could turn builder validation into a “Bumpy Road”. Severity: medium
  Likelihood: medium Mitigation: factor shared validation into a small helper
  that both `build()` and `build_session()` use, and keep session-specific
  configuration grouped into dedicated value types.

- Risk: a public `ClusteringSession` with no useful read-only surface is
  awkward to test and document. Severity: medium Likelihood: medium Mitigation:
  add only the minimal inspection methods needed to show that the session was
  initialized correctly, such as `config()`, `point_count()`, and
  `snapshot_version()`.

- Risk: the refresh-policy type may be over-designed too early if it
  tries to include all future `11.2.x` fields now. Severity: medium Likelihood:
  medium Mitigation: define a minimal v1 session refresh-policy value that
  captures only the behaviour needed for `11.1.2` and the next immediate
  dependency (`refresh_every_n`), then extend it in `11.2.3` and `11.2.4`.

## Progress

- [x] (2026-04-10 00:00Z) Reviewed `docs/roadmap.md`,
  `docs/chutoro-design.md` §10.1 and §12.3-§12.4, the testing guidance
  documents, the current `ChutoroBuilder` implementation, and the existing
  `11.1.1` ExecPlan format.
- [x] (2026-04-10 00:00Z) Confirmed the current codebase has no session
  module, no `SessionConfig`, no `build_session`, and no builder-carried
  HNSW/session refresh knobs; the CPU pipeline still hard-codes
  `HnswParams::default()`.
- [x] (2026-04-10 00:00Z) Drafted this ExecPlan for approval before any
  implementation work begins.

## Surprises & Discoveries

- Observation: `CpuHnsw::with_capacity(params, capacity)` is already
  public and supports an empty index. Evidence:
  `chutoro-core/src/hnsw/cpu/mod.rs` exposes
  `with_capacity(...) -> Result<Self, HnswError>`. Impact: `build_session` can
  initialize an empty session now without pulling `11.3.1` batch bootstrap work
  into this task.

- Observation: the batch CPU path still constructs
  `HnswParams::default()` internally. Evidence:
  `chutoro-core/src/cpu_pipeline.rs` sets
  `let params = HnswParams::default();`. Impact: builder-carried HNSW params
  introduced for sessions can land without changing the existing batch pipeline
  yet, but the plan should keep that asymmetry explicit.

- Observation: `build()` currently rejects invalid `min_cluster_size`
  and GPU preference in `ChutoroBuilder`, not in `Chutoro`. Evidence:
  `chutoro-core/src/builder.rs`. Impact: `build_session` should share the same
  validation helper rather than duplicating the same checks and risking drift.

## Decision Log

- Decision: prefer a CPU-gated public surface consisting of
  `SessionConfig`, `SessionRefreshPolicy`, `ClusteringSession<D>`, and
  `ChutoroBuilder::build_session(...)`. Rationale: the incremental session
  architecture in `docs/chutoro-design.md` is explicitly built around the CPU
  HNSW engine, so feature-gating the surface keeps the public API honest until
  a non-CPU session backend exists. Date/Author: 2026-04-10 / assistant

- Decision: plan around `build_session(self, source: Arc<D>)` producing
  an empty session rather than seeding HNSW or labels. Rationale: this matches
  the roadmap split, leaves `11.3.1` and `11.3.2` meaningful, and avoids
  smuggling batch bootstrap behaviour into the configuration task. Date/Author:
  2026-04-10 / assistant

- Decision: prefer validated session config fields over raw integers in
  the stored config. Rationale: `NonZeroUsize`, `HnswParams`, and a dedicated
  refresh policy value prevent repeated validation branches and keep later
  session logic simpler and easier to test. Date/Author: 2026-04-10 / assistant

- Decision: prefer whole-object builder setters/getters for HNSW params
  and session refresh policy. Rationale: this is the smallest API addition that
  still lets the builder derive `SessionConfig` cleanly, while avoiding a
  premature explosion of narrow builder methods. Date/Author: 2026-04-10 /
  assistant

## Outcomes & Retrospective

Pending implementation. This section should capture whether the shipped session
surface stayed within the planned scope, whether the chosen factory semantics
proved adequate for `11.1.3` and `11.3.x`, and which follow-up refactors were
deferred.

## Context and orientation

Today the public orchestration surface is batch-only:

1. `ChutoroBuilder::build()` validates `min_cluster_size` and
   `execution_strategy`, then returns a `Chutoro`.
2. `Chutoro::run(&source)` validates the source and runs the batch CPU
   pipeline.
3. The CPU pipeline hard-codes `HnswParams::default()` and computes
   labels eagerly.

There is no public session abstraction yet. The design document’s incremental
architecture introduces one:

- `SessionConfig` carries the validated clustering/session parameters.
- `ClusteringSession<D>` owns the live HNSW index and incremental state.
- Later roadmap items layer `append`, `refresh`, label snapshots, and
  batch bootstrap on top of that session skeleton.

The key implementation split for `11.1.2` is therefore:

- Define the public configuration/state scaffolding now.
- Keep the initial session empty and inert beyond read-only inspection.
- Leave all mutation and refresh workflows for later tasks.

## Plan of work

### Stage A: settle the public API shape and validation boundaries

Add the minimum new builder-carried configuration needed to derive a session
config:

- `HnswParams` stored on `ChutoroBuilder`, defaulting to
  `HnswParams::default()`.
- A dedicated session refresh-policy value stored on
  `ChutoroBuilder`, defaulting to manual refresh with no `refresh_every_n`
  threshold enabled.

The preferred public additions are:

```rust
#[cfg(feature = "cpu")]
pub struct SessionConfig {
    min_cluster_size: NonZeroUsize,
    hnsw_params: HnswParams,
    refresh_policy: SessionRefreshPolicy,
}

#[cfg(feature = "cpu")]
pub struct SessionRefreshPolicy {
    refresh_every_n: Option<NonZeroUsize>,
}

#[cfg(feature = "cpu")]
impl ChutoroBuilder {
    pub fn with_hnsw_params(mut self, params: HnswParams) -> Self;
    pub fn hnsw_params(&self) -> &HnswParams;
    pub fn with_session_refresh_policy(
        mut self,
        policy: SessionRefreshPolicy,
    ) -> Self;
    pub fn session_refresh_policy(&self) -> &SessionRefreshPolicy;
    pub fn build_session<D: DataSource + Sync>(
        self,
        source: Arc<D>,
    ) -> Result<ClusteringSession<D>>;
}
```

Keep `build()` and `build_session()` on a shared validation path so
`min_cluster_size == 0` and unsupported execution strategies remain consistent.
The important semantic difference is that `build_session` must not reject
sources whose current length is `0` or below `min_cluster_size`, because
session creation does not itself cluster.

### Stage B: add a narrow `ClusteringSession` skeleton

Introduce a CPU-gated session module with a small public type:

```rust
#[cfg(feature = "cpu")]
pub struct ClusteringSession<D: DataSource + Sync> {
    config: SessionConfig,
    index: CpuHnsw,
    core_distances: Vec<f32>,
    mst_edges: Vec<MstEdge>,
    historical_edges: Vec<CandidateEdge>,
    pending_edges: Vec<CandidateEdge>,
    labels: Arc<Vec<usize>>,
    snapshot_version: u64,
    source: Arc<D>,
    last_refresh_len: usize,
}
```

For `11.1.2`, initialize it in an empty state:

- `index` via `CpuHnsw::with_capacity(config.hnsw_params().clone(),
  source.len().max(1))`;
- all edge/core-distance vectors empty;
- `labels` empty;
- `snapshot_version == 0`;
- `last_refresh_len == 0`.

Add only the narrow read-only methods needed to make the returned type
inspectable and testable, for example:

- `config(&self) -> &SessionConfig`
- `point_count(&self) -> usize`
- `snapshot_version(&self) -> u64`

Do not add append, refresh, or source-mutation methods yet.

### Stage C: document and test the API thoroughly

Add integration tests in a dedicated session-focused file instead of
overloading `chutoro-core/tests/chutoro.rs`. The expected coverage is:

1. builder defaults now include default session/HNSW values;
2. `build_session` derives `SessionConfig` correctly from parameterized
   builder inputs;
3. `build_session` accepts empty and undersized sources and initializes
   an empty session state;
4. `build_session` rejects zero `min_cluster_size`;
5. `build_session` honours the CPU-only session restriction for
   unsupported execution strategies;
6. the new public Rustdoc examples compile.

Use `rstest` for the configuration-mapping cases and source-size edge cases.
Keep fixtures fallible only if setup genuinely becomes fallible.

Update `docs/chutoro-design.md` §12.3 to record:

- the final `build_session` signature;
- the choice to create an empty session instead of seeding;
- the choice to store validated builder-derived config in
  `SessionConfig`;
- the minimal initial `SessionRefreshPolicy` shape and why it does not
  yet include the later `11.2.4` drift-trigger fields.

Then mark roadmap item `11.1.2` as done in `docs/roadmap.md`.

### Stage D: run the full validation chain and close the task

Because the change introduces Rust code and Markdown updates, run the full
validator chain after targeted tests are green:

- `make fmt`
- `make check-fmt`
- `make markdownlint`
- `make nixie`
- `make lint`
- `make test`

Keep all logs under `/tmp/11-1-2-*.log` using `tee` and `set -o pipefail`.

## Concrete steps

All commands below are run from the repository root: `/home/user/project`.

1. Review the current builder, batch pipeline, and HNSW capacity API.

   ```bash
   rg -n "pub struct ChutoroBuilder|pub fn build\\(|run_cpu_pipeline_with_len" chutoro-core/src
   rg -n "pub fn with_capacity|pub fn insert_harvesting" chutoro-core/src/hnsw/cpu/mod.rs
   ```

2. Introduce the session module and export the new public CPU-gated
   types from `chutoro-core/src/lib.rs`.

3. Extend `ChutoroBuilder` with session/HNSW config storage, shared
   validation helpers, and `build_session(...)`.

4. Add focused integration tests, likely in a new file such as:

   - `chutoro-core/tests/session_builder.rs`

   Expected test names can be close to:

   - `builder_defaults_include_session_defaults`
   - `build_session_derives_config_from_builder`
   - `build_session_accepts_empty_and_undersized_sources`
   - `build_session_rejects_zero_min_cluster_size`
   - `build_session_rejects_unsupported_execution_strategy`
   - `build_session_initializes_empty_state`

5. Run quick targeted checks during iteration.

   ```bash
   cargo test -p chutoro-core session_builder -- --nocapture
   cargo test -p chutoro-core build_session -- --nocapture
   cargo test -p chutoro-core builder_defaults -- --nocapture
   ```

6. Update `docs/chutoro-design.md` and `docs/roadmap.md` only after the
   API names and session-construction semantics are settled.

7. Run the required validators sequentially and keep logs.

   ```bash
   set -o pipefail && make fmt 2>&1 | tee /tmp/11-1-2-fmt.log
   set -o pipefail && make check-fmt 2>&1 | tee /tmp/11-1-2-check-fmt.log
   set -o pipefail && make markdownlint 2>&1 | tee /tmp/11-1-2-markdownlint.log
   set -o pipefail && make nixie 2>&1 | tee /tmp/11-1-2-nixie.log
   set -o pipefail && make lint 2>&1 | tee /tmp/11-1-2-lint.log
   set -o pipefail && make test 2>&1 | tee /tmp/11-1-2-test.log
   ```

## Validation and acceptance

The change is done only when all of the following are true:

- `SessionConfig`, `SessionRefreshPolicy`, and `ClusteringSession` are
  public CPU-session types exported from `chutoro-core`.
- `ChutoroBuilder` can carry session-relevant HNSW and refresh settings
  and derive a validated `SessionConfig`.
- `ChutoroBuilder::build_session(...)` returns an empty session without
  running the batch clustering pipeline.
- Empty and undersized sources are accepted for `build_session(...)`
  even though they remain invalid for `Chutoro::run(...)`.
- Unit tests cover builder/config mapping, empty-session initialization,
  invalid configuration, and the selected execution-strategy semantics.
- `docs/chutoro-design.md` records the chosen API and
  `docs/roadmap.md` marks `11.1.2` as done.
- `make check-fmt`, `make lint`, and `make test` pass.
- Because this task also changes Markdown, `make fmt`,
  `make markdownlint`, and `make nixie` pass as well.

Observable acceptance criteria:

1. A parameterized test proves that `SessionConfig` contains the exact
   builder-selected `min_cluster_size`, `HnswParams`, and refresh-policy values.
2. A session-construction test proves that `build_session(...)` starts
   with `point_count() == 0`, `snapshot_version() == 0`, and empty label and
   edge buffers.
3. An edge-case test proves that `build_session(...)` accepts a source
   whose current `len()` is below `min_cluster_size`.

## Idempotence and recovery

All implementation and validation steps are safe to re-run. If formatting
changes files, rerun the downstream validators in the same order. If a targeted
test fails, fix the issue locally first, rerun that targeted check, and only
then rerun the full validator chain. Keep the logs in `/tmp/11-1-2-*.log` for
any escalation.

## Artifacts and notes

- Primary builder file: `chutoro-core/src/builder.rs`
- Likely new session files:
  `chutoro-core/src/session/mod.rs` and `chutoro-core/src/session/config.rs`
- Likely integration test file:
  `chutoro-core/tests/session_builder.rs`
- Required design records:
  `docs/chutoro-design.md` and `docs/roadmap.md`
- Execution log files:
  `/tmp/11-1-2-fmt.log`, `/tmp/11-1-2-check-fmt.log`,
  `/tmp/11-1-2-markdownlint.log`, `/tmp/11-1-2-nixie.log`,
  `/tmp/11-1-2-lint.log`, `/tmp/11-1-2-test.log`

## Interfaces and dependencies

Preferred public interfaces:

```rust
#[cfg(feature = "cpu")]
pub struct SessionConfig { /* validated session parameters */ }

#[cfg(feature = "cpu")]
pub struct SessionRefreshPolicy { /* manual or refresh_every_n */ }

#[cfg(feature = "cpu")]
pub struct ClusteringSession<D: DataSource + Sync> { /* empty session state */ }

#[cfg(feature = "cpu")]
impl ChutoroBuilder {
    pub fn with_hnsw_params(self, params: HnswParams) -> Self;
    pub fn with_session_refresh_policy(
        self,
        policy: SessionRefreshPolicy,
    ) -> Self;
    pub fn build_session<D: DataSource + Sync>(
        self,
        source: Arc<D>,
    ) -> Result<ClusteringSession<D>>;
}
```

The implementation should continue to rely on existing types and should not add
new dependencies:

- `ChutoroBuilder`
- `ChutoroError`
- `CpuHnsw`
- `HnswParams`
- `CandidateEdge`
- `MstEdge`
- `DataSource`

## Approval checkpoints

Approval of this plan should explicitly confirm these two semantic choices
before implementation starts:

1. `build_session(...)` takes ownership of the source via `Arc<D>`.
2. `build_session(...)` creates an empty session and therefore accepts
   empty and undersized sources.

If either choice is rejected, revise this ExecPlan before touching Rust code.

## Revision note (2026-04-10)

Initial draft created from roadmap item `11.1.2`, the current builder and CPU
pipeline implementation, the incremental session design in
`docs/chutoro-design.md`, and the repository’s testing and documentation
guidance. It proposes a narrow session/config surface, records the main API
decisions that need approval, and preserves the later incremental
append/refresh tasks as separate roadmap work.
