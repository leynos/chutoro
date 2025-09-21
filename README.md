# Chutoro

This is a generated project using [Copier](https://copier.readthedocs.io/).

> **Walking skeleton:** The CPU-only build currently partitions inputs into
> placeholder buckets sized by `min_cluster_size`. This stub proves the
> orchestration wiring and will be replaced by the HNSW + MST +
> hierarchy extraction pipeline described in
> [`docs/chutoro-design.md`](docs/chutoro-design.md).

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
