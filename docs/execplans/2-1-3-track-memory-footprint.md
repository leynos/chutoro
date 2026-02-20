# Execution Plan (ExecPlan): track HNSW memory footprint in CPU benchmarks

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds.

Status: DRAFT

PLANS.md is not present in this repository, so no additional plan constraints
apply.

## Purpose / Big Picture

Implement roadmap item 2.1.3 by extending benchmark infrastructure so HNSW
build runs report memory footprint alongside elapsed time. The implementation
will record peak resident memory during HNSW build, compute `memory/point` and
`memory/edge`, and report results for `M in {8, 12, 16, 24}`.

Success is observable when benchmark output includes per-configuration timing
plus memory metrics, scaling checks confirm edge growth is approximately
`n * M`, unit tests cover happy/unhappy/edge cases with parameterized `rstest`
cases, design decisions are documented in `docs/chutoro-design.md`, roadmap
entry `2.1.3` is marked done, and quality gates pass: `make check-fmt`,
`make lint`, and `make test`.

## Constraints

- Keep files under 400 lines by splitting modules where needed.
- Preserve existing `chutoro-core` public behaviour; this is a benchmark and
  observability change, not an algorithm rewrite.
- Do not weaken lint policy; any lint exceptions must stay tightly scoped and
  justified.
- Keep benchmarks deterministic where possible by reusing explicit seeds.
- Use en-GB-oxendict spelling in docs/comments.
- Avoid adding dependencies unless necessary for measurement correctness.
- New behaviour must include unit tests using `rstest` parameterization where
  repetition would otherwise occur.
- Benchmark memory reporting must include both successful runs and clear error
  reporting for unsupported or malformed measurement inputs.

## Tolerances (Exception Triggers)

- Scope: if implementation needs changes in more than 12 files or more than
  700 net lines, stop and escalate.
- Interface: if satisfying this item requires a breaking public API change in
  `chutoro-core`, stop and escalate.
- Dependencies: if more than one new crate is required for memory measurement,
  stop and escalate.
- Iterations: if `make lint` or `make test` fails after 3 repair attempts,
  stop and escalate with captured logs.
- Ambiguity: if memory-per-edge semantics cannot be defined without conflicting
  interpretations, stop and present options with trade-offs.

## Risks

- Risk: process-level peak memory is hard to observe without allocator hooks.
  Severity: high. Likelihood: medium. Mitigation: use a separate in-process
  Linux `/proc` sampler dedicated to benchmark profiling and document limits.
- Risk: measured memory can include temporary allocations unrelated to graph
  structure. Severity: medium. Likelihood: medium. Mitigation: use a fixed
  benchmark path and report both elapsed time and edge count context.
- Risk: edge-count scaling may vary due parallel insertion order.
  Severity: medium. Likelihood: medium. Mitigation: validate against tolerance
  bands and monotonic trends instead of exact equality.
- Risk: benchmark-only code may regress under strict Clippy lints.
  Severity: medium. Likelihood: medium. Mitigation: keep helpers small,
  fallible, and covered by focused unit tests.

## Progress

- [x] (2026-02-20 20:08Z) Drafted ExecPlan for roadmap item 2.1.3.
- [ ] Implement memory profiling module and tests in `chutoro-benches`.
- [ ] Integrate memory metrics into HNSW benchmark reporting for
      `M in {8, 12, 16, 24}`.
- [ ] Update design documentation and roadmap state.
- [ ] Run formatting, linting, and test quality gates.

## Surprises & Discoveries

- Observation: project memory MCP resources are not exposed in this
  environment (`list_mcp_resources` and `list_mcp_resource_templates` returned
  empty results). Evidence: tool output contained zero resources and zero
  templates. Impact: this plan relies on repository sources only.
- Observation: current HNSW benchmark sweep still uses `M in {8, 16}` in
  `chutoro-benches/benches/hnsw.rs`. Evidence: `MAX_CONNECTIONS` constant
  currently has two values. Impact: roadmap item 2.1.4 overlap must be managed
  while implementing 2.1.3.

## Decision Log

- Decision: implement memory tracking via a separate profiler module in
  `chutoro-benches` instead of Criterion custom measurement first. Rationale:
  this keeps timing benchmarks intact, reduces Criterion macro complexity, and
  still satisfies roadmap wording ("custom measurement or a separate
  profiler"). Date/Author: 2026-02-20 (Codex)
- Decision: define `memory/edge` against harvested HNSW build edges from
  `CpuHnsw::build_with_edges`. Rationale: edge counts are already available
  from the benchmarked path and avoid broad `chutoro-core` API expansion for
  this roadmap item. Date/Author: 2026-02-20 (Codex)
- Decision: enforce scaling validation via tolerance bands plus monotonic trend
  checks, not strict equality. Rationale: parallel insertion order and dataset
  geometry can shift absolute counts while preserving expected growth shape.
  Date/Author: 2026-02-20 (Codex)

## Outcomes & Retrospective

Pending implementation. At completion this section will summarise:

- achieved memory and timing reporting behaviour,
- final scaling observations across `M` values,
- test/lint gate outcomes,
- follow-up work, if any.

## Context and Orientation

Current benchmark infrastructure lives in `chutoro-benches`:

- `chutoro-benches/benches/hnsw.rs` runs timing benchmarks for HNSW build
  paths and currently sweeps only `M in {8, 16}`.
- `chutoro-benches/src/params.rs` defines benchmark parameter display structs.
- `chutoro-benches/src/error.rs` aggregates setup errors for benchmark code.
- `chutoro-benches/src/lib.rs` exports benchmark support modules.

Roadmap item `2.1.3` requires memory tracking during HNSW build and reporting:

- peak memory during build,
- `memory/point`,
- `memory/edge`,
- coverage for `M in {8, 12, 16, 24}`,
- validation that memory scales with approximately `n * M` edges.

Related docs that must stay aligned:

- `docs/roadmap.md` (item completion state),
- `docs/chutoro-design.md` (§11 benchmarking guidance),
- `docs/property-testing-design.md` (invariant mindset for validation),
- `docs/rust-testing-with-rstest-fixtures.md` (parameterized fixture style),
- `docs/rust-doctest-dry-guide.md` (if rustdoc examples are added),
- `docs/complexity-antipatterns-and-refactoring-strategies.md` (keep helper
  logic decomposed and readable).

## Plan of Work

### Stage A: profiler scaffolding and pure metric helpers

Add a benchmark-support profiling module, likely under
`chutoro-benches/src/profiling/`, that can:

- sample process resident memory while a closure executes,
- capture elapsed wall time for the same run,
- compute derived metrics (`bytes_per_point`, `bytes_per_edge`),
- evaluate scaling assertions against expected edge growth.

Keep the core arithmetic and parsing helpers pure so they are easy to test.
Error handling should use semantic enum variants wired through
`BenchSetupError`.

Go/no-go: stop if Linux process memory sampling proves too unstable for
reproducible results and present alternatives (Criterion custom measurement,
child-process `/usr/bin/time` sampling, allocator-specific stats).

### Stage B: benchmark integration for HNSW memory reporting

Update `chutoro-benches/benches/hnsw.rs` to:

- extend `MAX_CONNECTIONS` to `{8, 12, 16, 24}`,
- execute profiled build runs (using `CpuHnsw::build_with_edges`) for each
  `(n, M)` case,
- emit timing and memory metrics in benchmark output and in a machine-readable
  report file under `target/criterion/` or `target/benchmarks/`.

The report should include at least:

- `point_count`,
- `max_connections`,
- `elapsed_ms`,
- `peak_memory_bytes`,
- `memory_per_point_bytes`,
- `edge_count`,
- `memory_per_edge_bytes`,
- `expected_edges = point_count * max_connections`,
- scaling verdict and deviation.

Go/no-go: stop if integrating memory profiling into existing Criterion flow
causes unmanageable benchmark harness complexity; split memory profiling into a
dedicated bench binary while preserving existing timing benches.

### Stage C: tests (happy, unhappy, edge cases) with rstest coverage

Add unit tests for profiling and scaling helpers using `rstest` with named
cases that cover:

- happy paths:
  valid memory sample parsing, derived metric computation, monotonic scaling.
- unhappy paths:
  malformed `/proc` status content, missing fields, zero-edge division guard,
  unsupported platform handling.
- edge cases:
  tiny datasets (`n=1`), high `M` values, tolerance boundary conditions.

Where helper fixtures are reused, implement fallible fixtures returning
`Result` and consume them with `?` from tests returning `Result`.

Go/no-go: stop if deterministic tests cannot be achieved without introducing
global state coupling; refactor helpers to accept injected sample readers.

### Stage D: documentation, roadmap, and quality gates

Update docs to capture design decisions and acceptance criteria:

- add a subsection in `docs/chutoro-design.md` (for example `11.2`) describing
  memory measurement method, assumptions, and scaling interpretation.
- mark `2.1.3` as done in `docs/roadmap.md` only after all gates pass.

Run required quality gates and keep logs with `tee` and `set -o pipefail`.

## Concrete Steps

1. Add profiling support module(s) and exports.

   - Edit `chutoro-benches/src/lib.rs`.
   - Add files under `chutoro-benches/src/profiling/` (or equivalent):
     parser/sampler logic, derived metric helpers, and tests.
   - Extend `chutoro-benches/src/error.rs` with profiling-specific error
     variants.

2. Integrate benchmark reporting.

   - Edit `chutoro-benches/benches/hnsw.rs`:
     expand `MAX_CONNECTIONS`, collect memory+timing samples, and report
     `memory/point` + `memory/edge`.
   - Edit `chutoro-benches/src/params.rs` if a dedicated memory-report
     parameter display type improves readability.

3. Add tests with parameterized `rstest` cases.

   - Add/extend unit tests in profiling module files.
   - Cover parsing, arithmetic, tolerance checks, and error paths.

4. Update docs and roadmap.

   - Edit `docs/chutoro-design.md` with methodology and rationale.
   - Edit `docs/roadmap.md` and mark item `2.1.3` done after validation.

5. Run formatting and documentation checks.

   - Command (repo root):

         set -o pipefail
         make fmt 2>&1 | tee /tmp/execplan-2-1-3-make-fmt.log

   - Command (repo root):

         set -o pipefail
         make markdownlint 2>&1 | tee /tmp/execplan-2-1-3-markdownlint.log

   - Command (repo root):

         set -o pipefail
         make nixie 2>&1 | tee /tmp/execplan-2-1-3-nixie.log

6. Run required code quality gates.

   - Command (repo root):

         set -o pipefail
         make check-fmt 2>&1 | tee /tmp/execplan-2-1-3-check-fmt.log

   - Command (repo root):

         set -o pipefail
         make lint 2>&1 | tee /tmp/execplan-2-1-3-lint.log

   - Command (repo root):

         set -o pipefail
         make test 2>&1 | tee /tmp/execplan-2-1-3-test.log

7. Run benchmark verification command.

   - Command (repo root):

         set -o pipefail
         cargo bench -p chutoro-benches --bench hnsw 2>&1 | tee /tmp/execplan-2-1-3-hnsw-bench.log

   - Expected observable output:
     HNSW benchmark cases include `M=8`, `M=12`, `M=16`, `M=24` and memory
     metrics are printed or persisted in the configured report artifact.

## Validation and Acceptance

Done means all of the following are true:

- Benchmark behaviour:
  HNSW benchmark reporting includes elapsed time plus peak memory, memory per
  point, and memory per edge for each `M` value in `{8, 12, 16, 24}`.
- Scaling behaviour:
  reported edge-growth checks pass and indicate approximately linear growth
  with `n * M`, with explicit tolerance documented in code and design docs.
- Tests:
  new profiling tests pass, including parameterized `rstest` happy/unhappy/edge
  cases.
- Docs:
  design decision record exists in `docs/chutoro-design.md`.
- Roadmap:
  `docs/roadmap.md` item `2.1.3` is `[x]`.
- Quality gates:
  `make check-fmt`, `make lint`, and `make test` succeed.

## Idempotence and Recovery

- Profiling and reporting steps must be safe to rerun; report files should be
  overwritten atomically or versioned by timestamp.
- If a benchmark run fails midway, rerunning the same command should not
  require manual cleanup beyond optional temporary log removal in `/tmp`.
- Do not delete existing benchmark artifacts outside the task scope.

## Interfaces and Dependencies

Planned interfaces (names may vary, intent is fixed):

- A profiling entrypoint in `chutoro-benches` that executes a closure and
  returns: `elapsed_duration`, `peak_memory_bytes`, and derived metrics.
- A scaling-validation helper that accepts `(point_count, max_connections,
  edge_count)` and returns a pass/fail verdict with deviation detail.
- Benchmark reporting that prints and/or serializes memory metrics per run in a
  stable shape for later comparison.

Dependency intent:

- Prefer standard library + Linux `/proc` parsing first.
- Add at most one profiling dependency only if standard library sampling cannot
  satisfy correctness and testability needs.

## Revision Note

Initial draft created for roadmap item `2.1.3`; no implementation has started
yet.
