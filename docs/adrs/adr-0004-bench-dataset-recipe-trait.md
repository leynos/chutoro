# Architecture Decision Record (ADR) 004: Benchmark dataset recipe trait

## Status

Accepted - 2026-06-16: Synchronous `DatasetRecipe` trait with typed phase
handoffs and sealed publication contract.

## Date

2026-06-16

## Context and problem statement

Roadmap item 10.1.1 introduces `chutoro-bench-datasets`, a workspace crate for
benchmark dataset preparation. The crate must define the stable recipe and port
contracts now, while later roadmap items add real network download, archive
extraction, manifest schemas, object-store adapters, local lockfiles, and
provenance checks.

## Decision drivers

- Compile-time phase ordering must prevent publication of unvalidated or
  unprepared bytes.
- Recipe implementations need typed handoffs without coupling 10.1.1 to the
  future manifest schema.
- The benchmark stack is currently synchronous and Criterion-oriented.
- Later object-store and downloader work must be able to hide infrastructure
  details behind stable ports.
- Partial failures need an explicit recipe-owned cleanup contract.

## Options considered

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

## Decision outcome

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

## Architectural rationale

The design follows the project's hexagonal architecture guidance by keeping
recipe policy separate from infrastructure adapters. Recipes invoke I/O through
`RecipeContext` ports rather than owning download, cache, or publication state.
The typed phase outputs encode lifecycle rules in Rust's type system, while the
sealed publication contract keeps future manifest evolution local to the
dataset crate.

## Known risks and limitations

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

## Addendum — 2026-11-22

Checksum parsing landed after 10.1.1. `Checksum` is now an inhabited
`{ algorithm, bytes: [u8; 32] }` value that parses `sha256:<hex>`. Parsing
reports failures through `IntegrityFailure`, boxed in `RecipeError::Integrity`.
`RecipeError::ChecksumUnsupported` is removed. Fetch-time verification remains
deferred to roadmap item 10.1.2, so the `Consequences` statement above still
describes the 10.1.1 milestone.

## Addendum — 2026-11-24

Issue #216 removed the `orphaned_cache_key` field from `PartialState`. The
field was always `None` under `run_recipe`, and the driver has no record of the
cache keys a recipe writes through `Storage`. The decision above is unchanged:
cleanup remains a recipe-owned hook, and the driver still performs no automatic
cache deletion. Recipes now record or derive their own `CacheKey` values and
remove partial output through their own mechanism. See the
[migration notes](../migration-notes.md#020).

## Related

- [ADR 008: GloVe source and ZIP scope](adr-0008-glove-source-and-zip-scope.md)
  records the archive-format boundary that 10.1.2 subsequently implements. It
  does not change the decision above.
