# Documentation contents

This index lists the maintained documentation for Chutoro and the reason to
open each document.

## Start here

- [Documentation contents](contents.md): this index for the documentation set.
- [Repository layout](repository-layout.md): a contributor-oriented map of the
  repository tree and the responsibilities of major paths.
- [User's guide](users-guide.md): user-facing guidance for applying Chutoro.
- [Developer's guide](developers-guide.md): maintainer-facing build, test, and
  contribution guidance.
- [Documentation style guide](documentation-style-guide.md): the writing,
  naming, formatting, and document-type rules for repository documentation.

## Product and system design

- [Chutoro design](chutoro-design.md): the primary architecture and system
  design reference for Chutoro.
- [Property testing design](property-testing-design.md): the property-testing
  strategy and invariant coverage model.
- [Benchmark dataset retrieval](benchmark-dataset-retrieval.md): reference
  guidance for benchmark dataset discovery and retrieval.
- [Verus toolchain](verus-toolchain.md): the Verus proof toolchain setup and
  operating notes.

## Maintainer references

- [Roadmap](roadmap.md): the active development roadmap and sequenced work
  items.
- [Complexity antipatterns and refactoring strategies](complexity-antipatterns-and-refactoring-strategies.md):
  refactoring guidance for recognizing and correcting maintainability risks.
- [Reliable testing in Rust via dependency injection](reliable-testing-in-rust-via-dependency-injection.md):
  testing guidance for isolating Rust behaviour through dependency injection.
- [Rust doctest DRY guide](rust-doctest-dry-guide.md): guidance for keeping
  Rust documentation tests concise and maintainable.
- [Rust testing with rstest fixtures](rust-testing-with-rstest-fixtures.md):
  guidance for shared test setup using `rstest`.

## Decision records

- [ADR 001: commit post-processing](adr-001-commit-post-processing.md):
  accepted guidance for commit post-processing behaviour.
- [ADR 002: adoption of Kani formal verification](adr-002-adoption-of-kani-formal-verification.md):
  accepted guidance for using Kani bounded model checking in this repository.

## Execution plans

- [Execution plans](execplans/): implementation plans for roadmap tasks,
  verification work, and other substantial changes.
  - [11-1-1 Make edge harvesting hnsw insertion path public](
    execplans/11-1-1-make-edge-harvesting-hnsw-insertion-path-public.md)
  - [11-1-2 Define session config carrying
    clustering parameters](
    execplans/11-1-2-define-session-config-carrying-clustering-parameters.md)
  - [11-1-3 Clustering session append](
    execplans/11-1-3-clustering-session-append.md)
  - [2-1-1 Benchmark suite](execplans/2-1-1-benchmark-suite.md)
  - [2-1-2 Extend synthetic source generators](
    execplans/2-1-2-extend-synthetic-source-generators.md)
  - [2-1-3 Track memory footprint](execplans/2-1-3-track-memory-footprint.md)
  - [2-1-4 Expand parameter coverage](
    execplans/2-1-4-expand-parameter-coverage.md)
  - [2-1-5 CLI memory guard](execplans/2-1-5-cli-memory-guard.md)
  - [2-1-6 Clustering quality tracking](
    execplans/2-1-6-clustering-quality-tracking.md)
  - [2-1-7 Establish a CI regression detection
    strategy](execplans/2-1-7-establish-a-ci-regression-detection-strategy.md)
  - [2-2-1 CPU distance kernels using std simd](
    execplans/2-2-1-cpu-distance-kernels-using-std-simd.md)
  - [2-2-2 Dense point view for aligned structure of
    arrays](execplans/2-2-2-dense-point-view-for-aligned-structure-of-arrays.md)
  - [2-2-3 Gate SIMD backends behind features](
    execplans/2-2-3-gate-simd-backends-behind-features.md)
  - [2-2-4 Optional nightly-only std simd backend](
    execplans/2-2-4-optional-nightly-only-std-simd-backend.md)
  - [2-2-5 Portable simd gating mechanics](
    execplans/2-2-5-portable-simd-gating-mechanics.md)
  - [2-2-6 Property-based backend parity suite](
    execplans/2-2-6-property-based-backend-parity-suite.md)
  - [2-2-7 Kani harnesses for tail padding and
    dispatch selection](
    execplans/2-2-7-kani-harnesses-for-tail-padding-and-dispatch-selection.md)
  - [Phase 1 bounded Kani harness](execplans/phase-1-bounded-kani-harness.md)
  - [Phase 1 candidate edge harvest property suite](
    execplans/phase-1-candidate-edge-harvest-property-suite.md)
  - [Phase 1 HNSW graph invariant Kani harness](
    execplans/phase-1-hnsw-graph-invariant-kani-harness.md)
  - [Phase 1 HNSW Kani eviction deferred scrub
    scenario](execplans/phase-1-hnsw-kani-eviction-deferred-scrub-scenario.md)
  - [Phase 1 nightly slow CI job](execplans/phase-1-nightly-slow-ci-job.md)
  - [Phase 1 parallel Kruskal property suite](
    execplans/phase-1-parallel-kruskal-property-suite.md)
  - [Phase 1 property test CI integration](
    execplans/phase-1-property-test-ci-integration.md)
  - [Phase 1 validate the harvested output](
    execplans/phase-1-validate-the-harvested-output.md)
  - [Phase 1 verus proofs for edge harvest
    primitives](execplans/phase-1-verus-proofs-for-edge-harvest-primitives.md)
  - [Replace 3 node bidirectionality harness with commit
    path harness](
    execplans/replace-3-node-bidirectionality-harness-with-commit-path-harness.md)
