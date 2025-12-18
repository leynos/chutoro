# Chutoro

This is a generated project using [Copier](https://copier.readthedocs.io/).

The default CPU backend executes the FISHDBC pipeline (HNSW construction with
candidate edge harvest, mutual-reachability MST construction, and stability
based hierarchy extraction) as described in
[`docs/chutoro-design.md`](docs/chutoro-design.md).

## Distance cache determinism and metrics

The shared `DistanceCache` resolves neighbour ties deterministically: when
distances are equal it chooses the lower item id; if ids match it falls back to
the insertion sequence number. This rule is used across builds and tests to
guarantee stable outputs under fixed seeds.

With the `metrics` feature flag enabled the cache emits the following telemetry
via the `metrics` crate:

- `distance_cache_hits` (counter)
- `distance_cache_misses` (counter)
- `distance_cache_evictions` (counter)
- `distance_cache_lookup_latency_histogram` (histogram, seconds)

These metric names are stable for downstream crates.

## Debugging property tests

Mutation properties (`hnsw_mutations_preserve_invariants_proptest`) exercise
add, delete, and reconfigure paths with invariant sweeps.

- Run: `cargo test -p chutoro-core hnsw_mutations_preserve_invariants_proptest`
- Reproduce: `cargo test --seed <SEED> -p chutoro-core \
  hnsw_mutations_preserve_invariants_proptest --exact --nocapture`
- Verbose logs: `RUST_LOG=debug cargo test --seed <SEED> -p chutoro-core \
  hnsw_mutations_preserve_invariants_proptest --exact --nocapture`
- Stress runs: raise `PROPTEST_CASES` in `chutoro-core/src/hnsw/tests/mod.rs`
  or set the env var when invoking `cargo test`.
- See `docs/property-testing-design.md#debugging-proptest-failures` for
  interpreting violations and log output.
