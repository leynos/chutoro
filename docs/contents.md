# Documentation contents

This index lists the maintained documentation for Chutoro and the reason to
open each document.

## Start here

- [Documentation contents](contents.md): this index for the documentation set.
- [Repository layout](repository-layout.md): a contributor-oriented map of the
  repository tree and the responsibilities of major paths.
- [Users' guide](users-guide.md): user-facing guidance for applying Chutoro.
- [Developers' guide](developers-guide.md): maintainer-facing build, test, and
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
