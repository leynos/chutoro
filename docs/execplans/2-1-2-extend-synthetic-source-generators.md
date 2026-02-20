# Execution Plan (ExecPlan): extend SyntheticSource generators and add Modified National Institute of Standards and Technology (MNIST) baseline

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & Discoveries`, `Decision Log`, and
`Outcomes & Retrospective` must be kept up to date as work proceeds.

Status: COMPLETE

PLANS.md is not present in this repository, so no additional plan constraints
apply.

## Purpose / Big Picture

Implement roadmap item 2.1.2 by extending `chutoro-benches` so benchmarks can
run against harder synthetic geometries and one real-world baseline dataset.
After this change, benchmark setup code can build:

- Gaussian blob datasets with configurable cluster count, separation, and
  anisotropy.
- Ring/manifold datasets that are non-linearly separable.
- Text datasets that exercise Levenshtein distance.
- MNIST 70,000 x 784 vectors via a deterministic download-and-cache helper.

Success is observable when benchmark code can construct each generator via
public APIs, all new unit tests (including parameterized `rstest` cases) pass,
design decisions are recorded in `docs/chutoro-design.md`, roadmap entry 2.1.2
is marked done, and quality gates pass: `make check-fmt`, `make lint`,
`make test`.

## Constraints

- Keep every source file under 400 lines (`AGENTS.md`), including tests.
- Preserve strict lint policy: no silencing unless tightly scoped with reason.
- Keep existing benchmark behaviours working; current benchmarks must still run
  with a default numeric generator.
- Preserve `chutoro-core` public API; all work is in benchmark support and
  docs.
- Use caret Semantic Versioning (SemVer) dependencies only; do not use wildcard
  or open-ended ranges.
- Use en-GB-oxendict spelling in comments and documentation updates.
- Follow doctest guidance for any new rustdoc examples, and avoid `unwrap` in
  non-test paths.
- Use `rstest` parameterized tests where coverage would otherwise duplicate
  assertions.

## Tolerances (Exception Triggers)

- Scope: if implementation requires changes to more than 18 files or 900 net
  lines, stop and escalate.
- Interface: if existing benchmark entry points require breaking signature
  changes in bench files, stop and present options.
- Dependencies: if MNIST support requires more than two new crates (expected:
  HTTP client + gzip decoder), stop and escalate.
- Iterations: if `make lint` or `make test` fails after 3 repair cycles, stop
  and escalate with failure logs.
- Data source ambiguity: if MNIST provenance/licensing choice is unclear,
  pause and confirm source URL/checksum policy before merging.

## Risks

- Risk: `chutoro-benches/src/source.rs` is already near the 400-line limit.
  Severity: high. Likelihood: high. Mitigation: split into `src/source/`
  submodules (`numeric`, `text`, `mnist`, `config`, `tests`) and keep each
  module focused.
- Risk: network-dependent MNIST download logic can make tests flaky.
  Severity: high. Likelihood: medium. Mitigation: isolate downloader behind a
  trait/function parameter, test parsing/cache logic with local fixtures only,
  and keep unit tests offline.
- Risk: introducing text generators may blur numeric and string source
  responsibilities. Severity: medium. Likelihood: medium. Mitigation: use
  explicit source types (for example `SyntheticVectorSource` and
  `SyntheticTextSource`) with clear metric descriptors.
- Risk: benchmark regressions from large MNIST allocations.
  Severity: medium. Likelihood: medium. Mitigation: cache parsed MNIST files
  and avoid repeated decode work inside benchmark iteration closures.

## Progress

- [x] (2026-02-16 00:00Z) Draft ExecPlan created for roadmap item 2.1.2.
- [x] (2026-02-16 01:10Z) Finalized source API: numeric `SyntheticSource`,
  `SyntheticTextSource`, Gaussian/manifold/text configs, and `MnistConfig`.
- [x] (2026-02-16 02:00Z) Implemented source module split and MNIST
  download-and-cache helper with offline-testable download client abstraction.
- [x] (2026-02-16 02:25Z) Added benchmark integration for Gaussian, manifold,
  and text sources in `benches/hnsw.rs`; kept MNIST benchmark opt-in via
  `CHUTORO_BENCH_ENABLE_MNIST=1`.
- [x] (2026-02-16 02:45Z) Added parameterized `rstest` coverage for happy,
  unhappy, and edge cases including MNIST parse/cache validation.
- [x] (2026-02-16 02:55Z) Updated `docs/chutoro-design.md` stress mapping and
  marked roadmap entry 2.1.2 done in `docs/roadmap.md`.
- [x] (2026-02-16 03:20Z) Passed quality/documentation gates:
  `make fmt`, `make markdownlint`, `make nixie`, `make check-fmt`, `make lint`,
  `make test`.

## Surprises & Discoveries

- Observation: project memory Model Context Protocol (MCP) resources are not
  currently exposed in this execution environment (`list_mcp_resources`
  returned empty), so no historical Qdrant notes could be retrieved before
  drafting. Evidence: tool output contained zero resources and zero templates.
  Impact: this plan relies on repository docs and existing execplans only.
- Observation: strict Clippy settings in `chutoro-benches` required
  `numeric/mod.rs` layout and removal of slice/indexing patterns in generator
  code and tests. Evidence: `self_named_module_files`, `indexing_slicing`,
  `integer_division_remainder_used`, and related lint failures during
  `make lint`. Impact: helpers were refactored to iterator/get-based access and
  explicit cycling logic (without `%`) for cluster assignment.

## Decision Log

- Decision: treat this as a focused benchmark-support change in
  `chutoro-benches`, not a `chutoro-core` feature. Rationale: roadmap 2.1.2 is
  explicitly under benchmarking infrastructure and should not widen the core
  runtime API. Date/Author: 2026-02-16 (Codex)
- Decision: include MNIST loader as download + on-disk cache helper rather than
  checking dataset files into git. Rationale: repository size remains small
  while preserving reproducible benchmark setup. Date/Author: 2026-02-16 (Codex)
- Decision: require offline unit tests for MNIST parsing/cache behaviour.
  Rationale: keeps CI deterministic and avoids external service dependency.
  Date/Author: 2026-02-16 (Codex)
- Decision: gate MNIST benchmark execution behind
  `CHUTORO_BENCH_ENABLE_MNIST=1` instead of running it in the default bench
  loop. Rationale: keeps `make test` deterministic and avoids network/download
  cost in standard CI while preserving a real-world baseline path for dedicated
  profiling runs. Date/Author: 2026-02-16 (Codex)

## Outcomes & Retrospective

Completed roadmap item 2.1.2 end-to-end.

- Implemented source module split under `chutoro-benches/src/source/` with
  numeric (`SyntheticSource`), text (`SyntheticTextSource`), and MNIST support.
- Added Gaussian blob, ring/swiss-roll manifold, and Levenshtein-oriented text
  generators with typed configs and validation.
- Added MNIST 70,000 x 784 download-and-cache helper with offline unit tests
  for malformed/truncated input and cache reuse.
- Extended HNSW benchmark coverage with diverse synthetic sources and optional
  MNIST baseline execution.
- Updated design documentation stress mapping and marked roadmap item 2.1.2 as
  done.

Trade-off: to keep CI/offline loops stable, MNIST benchmarking is opt-in while
the loader and cache logic remain fully implemented and tested.

## Context and Orientation

Current benchmark support lives in `chutoro-benches`:

- `chutoro-benches/src/source/mod.rs` now re-exports numeric, text, and MNIST
  helpers from focused submodules.
- Benchmarks in `chutoro-benches/benches/*.rs` construct this source via
  typed configs (uniform, Gaussian/manifold, and text) with deterministic seeds.
- `chutoro-benches/src/error.rs` aggregates benchmark setup errors.

Roadmap item 2.1.2 in `docs/roadmap.md` requires four additions:

- richer synthetic numeric distributions (Gaussian blobs and ring/manifold),
- string data for Levenshtein distance,
- MNIST baseline with download-and-cache support,
- documentation mapping each pattern to stressed pipeline stages.

Design and test guidance to follow while implementing:

- `docs/chutoro-design.md` (architecture and rationale updates),
- `docs/property-testing-design.md` (generator quality mindset),
- `docs/complexity-antipatterns-and-refactoring-strategies.md` (avoid bumpy
  road functions via small composable helpers),
- `docs/rust-testing-with-rstest-fixtures.md` (fixtures + parameterized tests),
- `docs/rust-doctest-dry-guide.md` (clean doctests for public APIs).

## Plan of Work

Stage A: reshape benchmark source modules for extensibility.

Refactor `chutoro-benches/src/source.rs` into a `chutoro-benches/src/source/`
module tree so new generator families do not push files over 400 lines.
`src/source/mod.rs` should define shared public types and re-export submodule
constructors. Keep compatibility helpers so existing benchmark files need only
minimal edits.

Go/no-go: stop if this refactor alone causes lint regressions that cannot be
resolved without relaxing current lint policy.

Stage B: implement generator families and explicit configurations.

Add configuration structs and constructors for:

- Gaussian blobs: `point_count`, `dimensions`, `cluster_count`, `separation`,
  `anisotropy` (scalar or per-dimension form), `seed`.
- Ring/manifold: parameters for ring radius/thickness (2D/3D embeddings) and
  manifold noise/embedding dimensions.
- Text strings: corpus size, token length bounds, alphabet controls, optional
  template words for near-neighbour edit-distance stress.

Keep validations explicit and typed; each invalid configuration should produce
semantic error variants in `SyntheticError`.

Go/no-go: stop if generator API design requires breaking all existing benchmark
constructor call sites without an obvious compatibility layer.

Stage C: add MNIST download-and-cache helper.

Add a dedicated module (for example `chutoro-benches/src/source/mnist.rs`) that:

- downloads MNIST files from a pinned base URL,
- caches raw/downloaded artifacts under a deterministic cache directory,
- validates headers/counts and decodes to a 70,000 x 784 float matrix,
- exposes a constructor returning a `DataSource` implementation suitable for
  Euclidean benchmarks.

Implementation should use retry-safe writes (`*.part` then atomic rename) and
clear error messages for corrupt cache files or partial downloads.

Go/no-go: stop if parsing shows inconsistent record counts across files and
cannot be resolved by cache invalidation.

Stage D: integrate with benchmarks and document stress mapping.

Update benchmark setup code so selected benchmark groups can opt into the new
source families without duplicating setup logic. Add or update benchmark
parameter structs if needed. Document pattern-to-pipeline stress mapping in
`docs/chutoro-design.md`, including at minimum:

- Gaussian blobs: Hierarchical Navigable Small World (HNSW) neighbourhood
  quality and edge harvest quality under separable vs overlapping clusters.
- Ring/manifold: non-linear neighbourhoods stressing ANN recall and minimum
  spanning tree (MST) candidate sufficiency.
- Text/Levenshtein: expensive non-vector distance path, branch-heavy scoring,
  and non-metric behaviour.
- MNIST: realistic high-dimensional baseline for end-to-end CPU timings.

Mark roadmap item `2.1.2` as done in `docs/roadmap.md` only after all gates
pass.

## Concrete Steps

1. Inspect and split source module.

   - Create:
     `chutoro-benches/src/source/mod.rs`,
     `chutoro-benches/src/source/numeric.rs`,
     `chutoro-benches/src/source/text.rs`,
     `chutoro-benches/src/source/mnist.rs`, and
     `chutoro-benches/src/source/tests.rs`.
   - Adjust `chutoro-benches/src/lib.rs` exports.
   - Keep a compatibility constructor for existing uniform vector usage.

2. Implement generator configs and constructors.

   - Add typed config structs plus validation helpers.
   - Expand `SyntheticError` with generator-specific failure variants.
   - Keep constructors deterministic by explicit random number generator (RNG)
     seed threading.

3. Implement MNIST helper with cache.

   - Add download, decode, and cache validation functions.
   - Ensure helper is idempotent when cache is already valid.
   - Add failure-path tests for corrupt headers, truncated payloads,
     and cache-miss download failures.

4. Update benchmark call sites.

   - Keep existing benches green with default uniform generator.
   - Add at least one benchmark setup path using each new generator family
     (can be additional groups or parameterized dataset factory functions).

5. Add test coverage using `rstest`.

   - Happy-path generator output shape tests with matrix of cases.
   - Unhappy-path validation tests for each config family.
   - Edge-case tests: zero/one clusters, extreme anisotropy values,
     ring thickness bounds, text length boundaries, deterministic seeds,
     and MNIST cache reuse.

6. Update design and roadmap docs.

   - Add design note in `docs/chutoro-design.md` documenting generator API
     rationale and stress-stage mapping.
   - Mark `docs/roadmap.md` item 2.1.2 as `[x]` only after validation passes.

7. Run quality gates with logging.

       set -o pipefail
       make check-fmt 2>&1 | tee /tmp/make-check-fmt-2-1-2.log

       set -o pipefail
       make lint 2>&1 | tee /tmp/make-lint-2-1-2.log

       set -o pipefail
       make test 2>&1 | tee /tmp/make-test-2-1-2.log

   Documentation changes expected in this task should also run:

       set -o pipefail
       make markdownlint 2>&1 | tee /tmp/make-markdownlint-2-1-2.log

       set -o pipefail
       make nixie 2>&1 | tee /tmp/make-nixie-2-1-2.log

## Validation and Acceptance

The implementation is complete when all of the following are true:

- `SyntheticSource` supports Gaussian blob, ring/manifold, and text generators
  with validated configuration APIs.
- MNIST loader can populate cache and reuse it without re-downloading.
- Unit tests cover happy/unhappy/edge paths with broad `rstest`
  parameterization where appropriate.
- `make check-fmt`, `make lint`, and `make test` pass.
- Documentation records generator-to-pipeline stress mapping and design
  decisions.
- `docs/roadmap.md` marks item 2.1.2 as done.

Expected command outcomes:

- `make check-fmt`: exits 0 with no formatting diffs.
- `make lint`: exits 0 with no Clippy warnings.
- `make test`: exits 0 and includes new source generator test modules.

## Idempotence and Recovery

- Generator constructors and cache helpers must be safe to re-run.
- If cache content is corrupt, delete only the affected cached file(s) and
  rerun; do not delete unrelated benchmark artifacts.
- If download is interrupted, rerun should detect and replace partial files
  (`*.part`) safely.
- If any quality gate fails, fix the reported issue and rerun only failed
  command(s) first, then rerun the full gate sequence.

## Artifacts and Notes

Keep the following evidence artifacts until review is complete:

- `/tmp/make-check-fmt-2-1-2.log`
- `/tmp/make-lint-2-1-2.log`
- `/tmp/make-test-2-1-2.log`
- `/tmp/make-markdownlint-2-1-2.log`
- `/tmp/make-nixie-2-1-2.log`

For MNIST cache behaviour, capture one short transcript showing:

- first run downloads and caches,
- second run reuses cache without network access.

## Interfaces and Dependencies

Planned public benchmark-support interfaces (final names may vary, but intent
is fixed):

- `chutoro_benches::source::SyntheticSource` remains available for numeric
  vectors (backward compatibility).
- New typed configuration models for generator families (Gaussian,
  ring/manifold, text, MNIST).
- A dedicated MNIST helper function that returns a benchmark-ready data source
  and uses deterministic local caching.

Expected dependency additions (subject to implementation confirmation):

- one HTTP client crate for dataset download,
- one gzip decompression crate for MNIST file decoding.

No new dependency should be added unless directly required by the above
features.

## Revision note

2026-02-16: Initial draft created for roadmap item 2.1.2 with explicit module
split strategy, generator API plan, MNIST cache helper plan, validation gates,
and documentation update requirements.

2026-02-16: Completed implementation. Updated progress, discoveries, decisions,
and outcomes to reflect delivered code, documentation changes, and passing
quality gates.
