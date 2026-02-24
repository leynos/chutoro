# Execution Plan (ExecPlan): implement CLI memory guards (`--max-bytes`)

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds.

Status: COMPLETE

PLANS.md is not present in this repository, so no additional plan constraints
apply.

## Purpose / Big Picture

Implement roadmap item 2.1.5 by adding a `--max-bytes` memory guard to the CLI,
backed by a core-library memory estimation module. The guard computes a
conservative pre-flight estimate of peak memory consumption for the CPU
pipeline and rejects execution when the estimate exceeds the configured limit.
Benchmarks using the core library directly also benefit because the guard lives
in `ChutoroBuilder` / `Chutoro`.

Success is observable when:

- The CLI accepts `--max-bytes <value>` with human-readable suffixes (e.g.,
  `512M`, `2G`, plain bytes).
- When estimated memory exceeds `max_bytes`, a clear `MemoryLimitExceeded`
  error is returned before any pipeline allocation.
- Benchmarks using the core library can set `with_max_bytes()` to fail fast
  on oversized datasets.
- A new `chutoro-core/src/memory.rs` module provides a documented, testable
  estimation function.
- Expected memory requirements per dataset size are documented in
  `docs/chutoro-design.md`.
- Quality gates pass: `make check-fmt`, `make lint`, and `make test`.

## Constraints

- Keep files under 400 lines by splitting modules where needed.
- Preserve existing `chutoro-core` public behaviour; the memory guard is
  additive and opt-in (`None` means no limit).
- Do not weaken lint policy; any lint exceptions must stay tightly scoped and
  justified.
- Use en-GB-oxendict spelling in docs/comments.
- Avoid adding dependencies. The byte-size parser is handwritten (the
  grammar is trivial: integer + optional suffix).
- New behaviour must include unit tests using `rstest` parameterization where
  repetition would otherwise occur.
- Error messages must be clear, human-readable, and include both the
  estimated and limit values.
- The memory estimation must be conservative (overestimate rather than
  underestimate) to avoid false negatives.

## Tolerances (Exception Triggers)

- Scope: if implementation needs changes in more than 12 files or more than
  700 net lines, stop and escalate.
- Interface: if satisfying this item requires a breaking public API change in
  `chutoro-core`, stop and escalate.
- Dependencies: if more than one new crate is required, stop and escalate.
- Iterations: if `make lint` or `make test` fails after 3 repair attempts,
  stop and escalate with captured logs.
- Ambiguity: if memory estimation accuracy cannot be validated against
  observed profiling data (from 2.1.3) without conflicting interpretations,
  stop and present options with trade-offs.

## Risks

- Risk: memory estimation formula may be inaccurate for extreme parameter
  combinations. Severity: medium. Likelihood: medium. Mitigation: use a
  conservative safety multiplier (1.5×) and document assumptions; calibrate
  against profiling data from roadmap 2.1.3.
- Risk: distance cache memory is bounded separately and may not be
  well-captured by the formula. Severity: low. Likelihood: medium. Mitigation:
  include `DEFAULT_MAX_ENTRIES × 80` bytes in the estimate.
- Risk: the `--max-bytes` CLI parser for suffixes may have edge cases (case
  sensitivity, whitespace, decimal values). Severity: low. Likelihood: medium.
  Mitigation: test with rstest parameterized cases covering all suffix
  combinations.
- Risk: adding `max_bytes` to `ChutoroBuilder` and `Chutoro` increases
  struct size and constructor complexity. Severity: low. Likelihood: low.
  Mitigation: the field is `Option<u64>` (9 bytes with padding), negligible
  overhead.

## Progress

- [x] (2026-02-23) Drafted ExecPlan for roadmap item 2.1.5.
- [x] (2026-02-23) Stage A: Memory estimation module
  (`chutoro-core/src/memory.rs`) with `estimate_peak_bytes()`,
  `format_bytes()`, constants, and rstest-parameterized unit tests.
- [x] (2026-02-23) Stage B: `MemoryLimitExceeded` error variant, builder
  `with_max_bytes()` setter, and pre-flight guard in `Chutoro::run_with_len()`.
- [x] (2026-02-23) Stage C: CLI `--max-bytes` flag with `parse_byte_size()`
  parser; memory guard tests extracted to `test_memory_guard.rs` to keep
  `tests.rs` under 400 lines.
- [x] (2026-02-23) Stage D: Added §11.4 to `docs/chutoro-design.md` with
  estimation formula and memory table; marked 2.1.5 done in `docs/roadmap.md`.
- [x] (2026-02-23) Passed quality gates: `make check-fmt`, `make lint`, and
  `make test` (775 tests, 0 failures).

## Surprises & Discoveries

- Observation: `DistanceCacheConfig::DEFAULT_MAX_ENTRIES` is gated behind
  `#[cfg(feature = "cpu")]` and therefore inaccessible from the unconditional
  `memory` module. Evidence: compilation failed when importing the constant.
  Impact: duplicated the value as `DEFAULT_CACHE_MAX_ENTRIES: u64 = 1_048_576`
  in `memory.rs` with a comment documenting the coupling.
- Observation: adding memory guard tests to `chutoro-cli/src/cli/tests.rs`
  pushed the file to 526 lines, exceeding the 400-line constraint. Evidence:
  `wc -l` after adding tests. Impact: extracted memory guard tests into a
  separate `test_memory_guard.rs` file included via
  `#[path = "test_memory_guard.rs"] mod test_memory_guard;`.

## Decision Log

- Decision: **Core library guard with CLI pass-through.** The `--max-bytes`
  limit is stored in `ChutoroBuilder` via `with_max_bytes()` and checked in
  `Chutoro::run_with_len()` before pipeline dispatch. Rationale: the roadmap
  requires benchmarks to fail fast on oversized datasets, and benchmarks use
  the core library directly; keeping the guard in core also benefits library
  consumers. The CLI simply passes the parsed value through. Date/Author:
  2026-02-23 (DevBoxer)

- Decision: **Static estimation function, not runtime measurement.** The
  memory check runs pre-flight using a formula based on `(n, M)` rather than
  sampling actual allocations. Rationale: pre-flight estimation is
  deterministic, has zero allocation overhead, and fails before any work
  begins. Runtime measurement would only detect the problem mid-execution.
  Date/Author: 2026-02-23 (DevBoxer)

- Decision: **No new dependencies for byte-size parsing.** A simple
  handwritten parser supporting `K`, `M`, `G`, `T` suffixes (case-insensitive)
  is sufficient. Rationale: the grammar is trivial (integer + optional suffix);
  adding a crate would violate the principle of minimal dependencies.
  Date/Author: 2026-02-23 (DevBoxer)

- Decision: **Use `u64` for byte limits and estimates** rather than `usize`.
  Rationale: on 32-bit platforms, `usize` is only 4 GiB. `u64` is future-proof
  and consistent with byte-oriented values. Date/Author: 2026-02-23 (DevBoxer)

## Outcomes & Retrospective

Implemented roadmap item 2.1.5 end-to-end.

- Added `chutoro-core/src/memory.rs` (265 lines): `estimate_peak_bytes()`
  with conservative 1.5× safety multiplier, `format_bytes()` for human-readable
  display (binary units), and rstest-parameterized unit tests covering happy
  paths, edge cases, overflow protection, and formatting.
- Added `MemoryLimitExceeded` error variant to `ChutoroError` in
  `chutoro-core/src/error.rs` with stable code `CHUTORO_MEMORY_LIMIT_EXCEEDED`
  and structured fields (data source, point count, estimated vs limit bytes
  with human-readable display strings).
- Extended `ChutoroBuilder` with `with_max_bytes()` setter and `max_bytes()`
  accessor; extended `Chutoro` with `max_bytes` field and pre-flight guard in
  `run_with_len()` that checks `estimate_peak_bytes()` against the configured
  limit before pipeline dispatch.
- Added `--max-bytes` CLI flag to `RunCommand` with handwritten
  `parse_byte_size()` parser supporting K/KB/KiB, M/MB/MiB, G/GB/GiB, T/TB/TiB
  suffixes (case-insensitive) with overflow-safe arithmetic.
- Extracted CLI memory guard tests into
  `chutoro-cli/src/cli/test_memory_guard.rs` (167 lines) to keep `tests.rs`
  under the 400-line constraint. Tests cover: 17 valid suffix cases, 5
  rejection cases, overflow, integration with `run_command()` for both exceeded
  and sufficient limits, zero-byte limit, and clap flag parsing.
- Added §11.4 to `docs/chutoro-design.md` with the estimation formula,
  memory table for 10K–10M points across M = {8, 16, 24}, usage guidance, and
  documented limitations.
- Marked roadmap item 2.1.5 done in `docs/roadmap.md`.
- All quality gates passed: `make check-fmt`, `make lint`, `make test`
  (775 tests, 0 failures).
- No new crate dependencies were introduced.
- All files remain under 400 lines.

## Context and Orientation

### Memory consumers in the CPU pipeline

Tracing through `run_cpu_pipeline_with_len()`:

1. **Hierarchical Navigable Small World (HNSW) graph** — `n` nodes, each
   with up to `2*M` level-0 neighbours stored as `Vec<usize>`. Dominant cost:
   `n * 2 * M * sizeof(usize)`.
2. **Distance Cache** — bounded by `DEFAULT_MAX_ENTRIES` (1,048,576).
   Estimated ~80 bytes per entry.
3. **Candidate Edges** — `~n * M` edges, each `CandidateEdge` (32 bytes:
   `usize + usize + f32 + u64` with padding).
4. **Core Distances** — `Vec<f32>` with `n` entries (4 bytes each).
5. **Mutual Edges** — same count and size as candidate edges.
6. **MST Forest** — `n − 1` edges, each `MstEdge` (32 bytes).

### Estimation formula

```text
hnsw_adjacency     = n × (2 × M) × 8
hnsw_node_overhead = n × 80
distance_cache     = 1_048_576 × 80
candidate_edges    = n × M × 32
core_distances     = n × 4
mutual_edges       = n × M × 32
mst_forest         = n × 32

estimated_bytes = (sum of above) × 1.5  (safety multiplier)
```

Default `HnswParams`: `M = 16`, `ef_construction = 64`.
`DistanceCacheConfig::DEFAULT_MAX_ENTRIES` = 1,048,576.

### Key files and current line counts

| File                              | Lines | Role                     |
| --------------------------------- | ----- | ------------------------ |
| `chutoro-core/src/builder.rs`     | 160   | Builder: add `max_bytes` |
| `chutoro-core/src/chutoro.rs`     | 290   | Runtime: add guard       |
| `chutoro-core/src/error.rs`       | 215   | Errors: add variant      |
| `chutoro-cli/src/cli/commands.rs` | 330   | CLI: add `--max-bytes`   |
| `chutoro-cli/src/cli/tests.rs`    | 363   | CLI tests: add coverage  |
| New: `chutoro-core/src/memory.rs` | ~0    | Estimation module        |

_Table 1: Key files and pre-change line counts._

All files remain under 400 lines after changes.

## Plan of Work

### Stage A: memory estimation module

Create `chutoro-core/src/memory.rs` with:

- `estimate_peak_bytes(point_count: usize, max_connections: usize) -> u64`
  computing the conservative pre-flight estimate.
- `format_bytes(bytes: u64) -> String` for human-readable display (binary
  units: KiB, MiB, GiB, TiB).
- Internal constants for the safety multiplier and struct sizes.
- Unit tests using `rstest` parameterized cases:
  - Happy paths: estimation for various `(n, M)` combos.
  - Edge cases: `n = 0` returns 0, `n = 1` returns small positive.
  - Overflow protection: very large `n` does not panic.
  - Format: `format_bytes(0)` = `"0 B"`, `format_bytes(1024)` =
    `"1.0 KiB"`, etc.

Wire into `chutoro-core/src/lib.rs` with `mod memory;` and
`pub use crate::memory::{estimate_peak_bytes, format_bytes};`.

Go/no-go: stop if the estimation function cannot be expressed without accessing
internal HNSW types that are not publicly exported.

### Stage B: error variant, builder, and runtime guard

1. `chutoro-core/src/error.rs`: add `MemoryLimitExceeded` variant with
   fields `data_source`, `point_count`, `estimated_bytes`, `max_bytes`,
   `estimated_display`, `limit_display`. Add `CHUTORO_MEMORY_LIMIT_EXCEEDED`
   error code.

2. `chutoro-core/src/builder.rs`: add `max_bytes: Option<u64>` field,
   `with_max_bytes()` setter, `max_bytes()` accessor, pass-through in `build()`.

3. `chutoro-core/src/chutoro.rs`: add `max_bytes` field and accessor. In
   `run_with_len()`, after existing validations and before `choose_backend()`,
   check `estimate_peak_bytes()` against the limit and return
   `MemoryLimitExceeded` if exceeded.

4. Tests in `chutoro-core/src/chutoro.rs`:
   - `max_bytes` high enough → pipeline runs normally.
   - `max_bytes` too low → `MemoryLimitExceeded`.
   - `max_bytes = Some(0)` → rejects any non-empty dataset.
   - `max_bytes = None` → no limit.

Go/no-go: stop if adding the variant to `define_error_codes!` causes
exhaustiveness issues that require invasive refactoring.

### Stage C: CLI `--max-bytes` flag

1. `chutoro-cli/src/cli/commands.rs`: add `max_bytes: Option<u64>` field to
   `RunCommand` with
   `#[arg(long = "max-bytes", value_parser = parse_byte_size)]`.

2. Add `parse_byte_size(s: &str) -> Result<u64, String>` supporting: plain
   integers, `K`/`KB`/`KiB`, `M`/`MB`/`MiB`, `G`/`GB`/`GiB`, `T`/`TB`/`TiB`
   (case-insensitive). Uses `checked_mul` for overflow safety.

3. In `run_command()`, pass `max_bytes` through to the builder.

4. Tests in `chutoro-cli/src/cli/tests.rs`:
   - `parse_byte_size`: plain numbers, all suffix variants, case
     insensitivity.
   - `parse_byte_size` rejects: empty, invalid numbers, unknown suffixes,
     overflow.
   - CLI integration: `--max-bytes 100` with a real dataset →
     `CliError::Core(ChutoroError::MemoryLimitExceeded { .. })`.
   - CLI integration: absent `--max-bytes` → no limit.

Go/no-go: stop if `commands.rs` exceeds 400 lines after adding the parser;
extract into a `byte_size.rs` module.

### Stage D: documentation, roadmap, and quality gates

1. `docs/chutoro-design.md`: add section 11.4 with estimation formula,
   memory table per dataset size, and guidance.

2. `docs/roadmap.md`: mark `2.1.5` as done.

3. Run quality gates:

   ```sh
   set -o pipefail
   make check-fmt 2>&1 | tee /tmp/execplan-2-1-5-check-fmt.log
   make lint      2>&1 | tee /tmp/execplan-2-1-5-lint.log
   make test      2>&1 | tee /tmp/execplan-2-1-5-test.log
   ```

## Validation and Acceptance

Done means all of the following are true:

- CLI behaviour: `--max-bytes` flag is accepted with human-readable suffixes.
  Oversized datasets produce a clear `MemoryLimitExceeded` error before any
  pipeline allocation. Absent flag imposes no limit.
- Core library behaviour: `ChutoroBuilder::with_max_bytes()` propagates
  through to `Chutoro::run()`. Pre-flight estimation rejects datasets that
  exceed the limit with a structured error.
- Estimation: `estimate_peak_bytes()` is conservative and does not
  underestimate. Values are reasonable when compared against profiling data
  from 2.1.3.
- Tests: new parameterized `rstest` tests cover happy/unhappy/edge cases for
  estimation, byte-size parsing, builder pass-through, and CLI integration.
- Docs: design decision record exists in `docs/chutoro-design.md` §11.4 with
  memory table.
- Roadmap: `docs/roadmap.md` item `2.1.5` is `[x]`.
- Quality gates: `make check-fmt`, `make lint`, and `make test` succeed.

## Idempotence and Recovery

- All estimation functions are pure and deterministic.
- No temporary files or state are created during estimation.
- Quality gate runs are safe to rerun at any point.
- If a stage fails midway, rerunning from the beginning of that stage is
  safe.

## Interfaces and Dependencies

Planned interfaces (names may vary, intent is fixed):

- `chutoro_core::estimate_peak_bytes(point_count, max_connections) -> u64`
- `chutoro_core::format_bytes(bytes: u64) -> String`
- `ChutoroBuilder::with_max_bytes(self, bytes: u64) -> Self`
- `Chutoro::max_bytes(&self) -> Option<u64>`
- `ChutoroError::MemoryLimitExceeded { .. }` with stable code
  `CHUTORO_MEMORY_LIMIT_EXCEEDED`
- `RunCommand::max_bytes: Option<u64>` — CLI field parsed from
  `--max-bytes`
- `parse_byte_size(s: &str) -> Result<u64, String>` — CLI-internal parser

Dependency intent: no new crates.
