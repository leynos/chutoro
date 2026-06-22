# Repository layout

This document maps important repository paths to their current
responsibilities. Use it alongside the user's guide, developers guide, and
design document when deciding where a change belongs.

## Core session module

Session code is split by responsibility under `chutoro-core/src/session/`:

- `mod.rs` owns the `ClusteringSession` struct, lightweight read-only
  accessors, public re-exports, and the high-level Rustdoc contract.
- `config.rs` owns `SessionRefreshPolicy` and `SessionConfig`, the small value
  types carried by each session.
- `session_impl.rs` owns construction, `append`, HNSW error mapping, and the
  edge-harvesting write path.
- `core_distance.rs` owns the pure core-distance helpers and the recompute
  workflow. The pure helpers must not depend on HNSW adapter internals beyond
  the public `Neighbour` value.
- `clock.rs` is compiled only with the `metrics` feature and owns the
  monotonic-clock support used for deterministic latency tests.
