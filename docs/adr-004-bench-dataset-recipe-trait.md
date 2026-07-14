# Architecture Decision Record (ADR) 004: Benchmark dataset recipe trait

## Status

Accepted - 2026-06-16: Synchronous `DatasetRecipe` trait with typed phase
handoffs and sealed publication contract.

## Date

2026-06-16

## Context and Problem Statement

Roadmap item 10.1.1 introduces `chutoro-bench-datasets`, a workspace crate for
benchmark dataset preparation. The crate must define the stable recipe and port
contracts now, while later roadmap items add real network download, archive
extraction, manifest schemas, object-store adapters, local lockfiles, and
provenance checks.

## Decision Drivers

- Compile-time phase ordering must prevent publication of unvalidated or
  unprepared bytes.
- Recipe implementations need typed handoffs without coupling 10.1.1 to the
  future manifest schema.
- The benchmark stack is currently synchronous and Criterion-oriented.
- Later object-store and downloader work must be able to hide infrastructure
  details behind stable ports.
- Partial failures need an explicit recipe-owned cleanup contract.

## Options Considered

- Single untyped callback: rejected because phase ordering would be enforced
  only by convention.
- TFDS-style example iterator: rejected because chutoro publishes prepared
  benchmark artefacts rather than lazily serving training examples.
- Async trait methods: rejected for 10.1.1 because the current benchmark stack
  is synchronous; future adapters can hide runtime details internally.
- Placeholder manifest schema: rejected because 10.1.3 owns the canonical
  prepared artefact schema.
- Sealed publication contract: accepted because URI and digest endpoints are
  stable before the manifest schema lands.
- Separate storage and publisher ports: accepted because cache mutation and
  final write-once publication carry different semantics.

## Decision Outcome

The project adopts a `DatasetRecipe` trait with typed handoffs between `fetch`,
`validate`, `prepare`, and `publish`. This achieves compile-time phase ordering
and accepts the extra associated types each recipe must define.

The recipe and port traits remain synchronous with `Send + Sync` handoff types.
This keeps the current benchmark stack simple and Criterion-friendly, while
future adapters remain free to hide asynchronous internals behind synchronous
methods.

A sealed `PublishedArtefact` trait exposes only manifest URI and digest until
10.1.3 defines the canonical manifest schema. External crates use the concrete
`PublishedManifest` type until then.

Archive extraction is deferred. 10.1.1 does not introduce a public `Extractor`
port, so early recipes express archive-specific preparation logic inside their
own `prepare` method.

Partial failures are handled through a `cleanup` hook that receives
`PartialState`. Automatic cache deletion is not performed by the driver because
recipe-owned side effects must remain explicit.

`Storage` and `Publisher` remain separate ports. The split preserves the
semantic distinction between mutable cache and final write-once publication,
accepting some duplicated adapter boilerplate.

## Architectural Rationale

The design follows the project's hexagonal architecture guidance by keeping
recipe policy separate from infrastructure adapters. Recipes invoke I/O through
`RecipeContext` ports rather than owning download, cache, or publication state.
The typed phase outputs encode lifecycle rules in Rust's type system, while the
sealed publication contract keeps future manifest evolution local to the
dataset crate.

## Known Risks and Limitations

- The associated types add ceremony for small recipes.
- The sealed publication contract is intentionally narrow until 10.1.3.
- The synchronous port surface requires future async adapters to manage their
  own runtime boundaries.
- Recipes with side effects must implement cleanup carefully.

## Consequences

- Recipe authors cannot skip phases through the public trait because each
  phase consumes the previous phase's associated type.
- The crate has no network, object-store, checksum, extraction, lockfile, or
  licence enforcement dependency in 10.1.1.
- In-memory test adapters use `Mutex` to satisfy the public `Send + Sync` port
  bounds while remaining local deterministic test doubles.
