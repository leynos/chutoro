# Chutoro

*High-performance FISHDBC clustering in Rust.*

Chutoro implements the
[FISHDBC](https://doi.org/10.1016/j.ins.2019.06.065) algorithm — a
scalable, density-based clustering method that discovers clusters of
arbitrary shape without requiring the user to specify the number of
clusters or a distance scale. It builds on the lineage of DBSCAN and
HDBSCAN\* while replacing the expensive all-pairs distance matrix with
an approximate HNSW graph, making it practical for large and
high-dimensional datasets.

______________________________________________________________________

## Why chutoro?

- **No parameter guessing**: FISHDBC extracts an optimal flat
  clustering from a density hierarchy — no need to choose `eps` or
  `k` up front.
- **Bring your own metric**: implement the `DataSource` trait with any
  distance function and chutoro handles the rest.
- **Rust-safe parallelism**: the CPU backend uses Rayon for parallel
  HNSW insertion and MST construction with zero `unsafe` in user-facing
  code.
- **Extensible by design**: feature-gated backends (CPU today, GPU
  planned) and a builder API that stays stable as the engine evolves.

______________________________________________________________________

## Quick start

### Installation

```toml
[dependencies]
chutoro-core = "0.1.0"
```

### Basic usage

```rust
use chutoro_core::{
    ChutoroBuilder, DataSource, DataSourceError, ExecutionStrategy,
};

struct Points(Vec<f32>);

impl DataSource for Points {
    fn len(&self) -> usize { self.0.len() }
    fn name(&self) -> &str { "points" }
    fn distance(&self, i: usize, j: usize)
        -> Result<f32, DataSourceError>
    {
        let a = self.0.get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self.0.get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
    }
}

let chutoro = ChutoroBuilder::new()
    .with_min_cluster_size(3)
    .with_execution_strategy(ExecutionStrategy::CpuOnly)
    .build()?;

let result = chutoro.run(&Points(vec![
    1.0, 1.1, 1.2, 5.0, 5.1, 5.2,
]))?;

println!("Found {} clusters", result.cluster_count());
# Ok::<(), chutoro_core::ChutoroError>(())
```

______________________________________________________________________

## Features

- Four-stage CPU pipeline: HNSW construction, candidate edge harvest,
  Kruskal MST, and stability-based hierarchy extraction.
- Built-in Euclidean and cosine distance helpers with input validation
  and pre-computed norm caching.
- Property-tested and formally verified (Kani, Verus) core
  primitives.
- Criterion benchmarks with synthetic data generators (Gaussian
  blobs, rings, text, MNIST).
- Optional `metrics` crate integration for cache-hit telemetry on hot
  paths.
- CLI tool (`chutoro-cli`) and bundled data-source providers for
  dense vectors (Parquet/Arrow) and text (string similarity).

______________________________________________________________________

## Learn more

- [Users' guide](docs/users-guide.md) — full API documentation and
  usage patterns
- [Developers' guide](docs/developers-guide.md) — contributing,
  benchmarks, and formal verification
- [Design document](docs/chutoro-design.md) — architecture and
  literature survey
- [Roadmap](docs/roadmap.md) — planned features and progress

______________________________________________________________________

## Licence

ISC — see [LICENSE](LICENSE) for details.

______________________________________________________________________

## Contributing

Contributions welcome! Please see [AGENTS.md](AGENTS.md) for
guidelines.
