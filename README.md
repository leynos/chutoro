# Chutoro

[![Ask DeepWiki][dw]][deepwiki] [![Crates.io Version][crates]][package]

[dw]: https://deepwiki.com/badge.svg
[deepwiki]: https://deepwiki.com/leynos/chutoro
[crates]: https://img.shields.io/crates/v/chutoro-core "crates.io package"
[package]: https://crates.io/crates/chutoro-core

*High-performance FISHDBC clustering in Rust.*

Chutoro implements [FISHDBC](https://doi.org/10.1016/j.ins.2019.06.065), a
density-based clustering method that finds clusters of arbitrary shape without
being told how many to look for. It follows DBSCAN and HDBSCAN\*, but replaces
their expensive all-pairs distance matrix with an approximate Hierarchical
Navigable Small World (HNSW) graph, so it stays practical for large and
high-dimensional datasets.

______________________________________________________________________

## Why chutoro?

- **One intuitive knob**: choose the smallest group worth calling a cluster.
  There is no `eps` radius to tune and no cluster count to guess.
- **Bring your own metric**: implement the `DataSource` trait with any distance
  function, from Euclidean vectors to edit distance between strings.
- **Parallel and safe**: the CPU backend builds its HNSW index and minimum
  spanning tree in parallel with Rayon, and `chutoro-core` is written entirely
  in safe Rust.
- **Checked, not just tested**: property tests, Kani model checking, and Verus
  proofs cover the graph and tree invariants the clustering relies on.

______________________________________________________________________

## Quick start

Chutoro v0.1.0-beta1 is the first public preview. Interfaces may still change
before v0.1.0; the [migration guide](docs/v0-1-0-migration-guide.md) records
every change.

### Try it from the command line

Install the command-line tool:

<!-- tested-example: readme-cli-install -->

```sh
cargo install chutoro-cli@0.1.0-beta1
```

Save a few words, one per line, as `words.txt`:

<!-- tested-example: readme-cli-corpus -->

```text
apple
apply
apples
appel
banana
bananas
bandana
banane
```

Then ask Chutoro to group them by edit distance:

<!-- tested-example: readme-cli-run -->

```sh
chutoro-cli run --min-cluster-size 2 text words.txt
```

The summary lists each line's index and the cluster it joined:

<!-- tested-example: readme-cli-output -->

```text
data source: words
clusters: 2
0	0
1	0
2	0
3	0
4	1
5	1
6	1
7	1
```

### Use it as a library

Add `chutoro-core` to your `Cargo.toml`:

<!-- tested-example: readme-library-dependency -->

```toml
[dependencies]
chutoro-core = "0.1.0-beta1"
```

Tell Chutoro how far apart your items are, then run it:

<!-- tested-example: readme-library-usage -->

```rust
use chutoro_core::{ChutoroBuilder, ChutoroError, DataSource, DataSourceError};

/// Each row is one point in 2-D space.
struct Points(Vec<[f32; 2]>);

impl DataSource for Points {
    fn len(&self) -> usize { self.0.len() }
    fn name(&self) -> &str { "points-2d" }
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok(a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt())
    }
}

fn main() -> Result<(), ChutoroError> {
    let points = Points(vec![
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.1, 0.2], // near the origin
        [5.0, 5.0], [5.1, 4.9], [5.0, 5.2], [5.2, 5.1], // near (5, 5)
    ]);

    let chutoro = ChutoroBuilder::new().with_min_cluster_size(3).build()?;
    let result = chutoro.run(&points)?;

    println!("Found {} clusters", result.cluster_count());
    assert_eq!(result.cluster_count(), 2);
    Ok(())
}
```

The [users' guide](docs/users-guide.md) explains what `min_cluster_size` means,
how to read the result, and how to tune larger runs.

______________________________________________________________________

## Features

- A four-stage CPU pipeline: HNSW construction, candidate-edge harvest, a
  parallel Kruskal minimum spanning tree, and stability-based cluster extraction
  ([design document](docs/chutoro-design.md)).
- Ready-made data sources: dense `f32` vectors from Parquet or Arrow
  (`chutoro-providers-dense`) and UTF-8 text compared by Levenshtein distance
  (`chutoro-providers-text`).
- A command-line tool, `chutoro-cli`, for clustering Parquet vectors and text
  files without writing Rust.
- Validated Euclidean and cosine distance helpers, with cached norms for
  repeated cosine queries.
- An optional memory guard that refuses runs whose estimated peak memory
  exceeds a limit you set.
- Optional `metrics` and `tracing` instrumentation that never records source
  names or payload data.
- A preview incremental `ClusteringSession` API that appends points to a live
  index; refresh and label snapshots are on the [roadmap](docs/roadmap.md).
- Property tests, Kani harnesses (HNSW, MST, distance, and SIMD invariants),
  and Verus proofs (edge harvest and core distances)
  ([developers' guide](docs/developers-guide.md#verus-proofs)).

______________________________________________________________________

## Learn more

- [Users' guide](docs/users-guide.md) — from a first run to tuning,
  observability, and the lower-level building blocks
- [Migration guide](docs/v0-1-0-migration-guide.md) — changes for anyone
  upgrading from a pre-release snapshot
- [Developers' guide](docs/developers-guide.md) — contributing, benchmarks, and
  formal verification
- [Design document](docs/chutoro-design.md) — architecture and literature
  survey
- [Roadmap](docs/roadmap.md) — planned features and progress

______________________________________________________________________

## Licence

ISC — see [LICENSE](LICENSE) for details.

______________________________________________________________________

## Contributing

Contributions are welcome. Start with the
[developers' guide](docs/developers-guide.md); automated contributors should
also follow [AGENTS.md](AGENTS.md).
