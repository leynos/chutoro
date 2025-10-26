use proptest::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};

use super::{
    datasets::{
        GeneratedDataset, generate_clustered_dataset, generate_duplicate_dataset,
        generate_manifold_dataset, generate_uniform_dataset,
    },
    types::{HnswFixture, HnswParamsSeed, VectorDistribution},
};

pub(super) fn hnsw_fixture_strategy() -> impl Strategy<Value = HnswFixture> {
    (
        any::<VectorDistribution>(),
        any::<u64>(),
        hnsw_params_strategy(),
    )
        .prop_map(|(distribution, seed, params)| {
            let mut rng = SmallRng::seed_from_u64(seed);
            let dataset = match distribution {
                VectorDistribution::Uniform => generate_uniform_dataset(&mut rng),
                VectorDistribution::Clustered => generate_clustered_dataset(&mut rng),
                VectorDistribution::Manifold => generate_manifold_dataset(&mut rng),
                VectorDistribution::Duplicates => generate_duplicate_dataset(&mut rng),
            };
            map_dataset(distribution, params, dataset)
        })
        .prop_filter("datasets must contain at least two vectors", |fixture| {
            fixture.vectors.len() > 1
        })
}

pub(super) fn hnsw_params_strategy() -> impl Strategy<Value = HnswParamsSeed> {
    (
        2_usize..=32,
        1_usize..=4,
        2_usize..=12,
        0.2_f64..=2.0,
        any::<u64>(),
    )
        .prop_map(
            |(max_connections, ef_multiplier, max_level, level_multiplier, rng_seed)| {
                let ef_construction = max_connections * ef_multiplier.max(1);
                HnswParamsSeed {
                    max_connections,
                    ef_construction,
                    level_multiplier,
                    max_level,
                    rng_seed,
                }
            },
        )
}

fn map_dataset(
    distribution: VectorDistribution,
    params: HnswParamsSeed,
    dataset: GeneratedDataset,
) -> HnswFixture {
    HnswFixture {
        distribution,
        vectors: dataset.vectors,
        metadata: dataset.metadata,
        params,
    }
}
