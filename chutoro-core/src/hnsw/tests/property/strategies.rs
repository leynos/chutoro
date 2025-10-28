//! Strategy builders for property-based HNSW tests.
//!
//! Provides combinators that synthesise dataset fixtures and parameter seeds
//! for use in proptest suites.

use proptest::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};

use super::{
    datasets::{
        GeneratedDataset, generate_clustered_dataset, generate_duplicate_dataset,
        generate_manifold_dataset, generate_uniform_dataset,
    },
    types::{HnswFixture, HnswParamsSeed, VectorDistribution},
};

/// Generates HNSW fixtures covering multiple vector distributions.
///
/// # Examples
///
/// ```ignore
/// use crate::hnsw::tests::property::strategies::hnsw_fixture_strategy;
/// use proptest::prelude::*;
///
/// let strategy = hnsw_fixture_strategy();
/// proptest!(|(fixture in strategy)| {
///     prop_assert!(fixture.vectors.len() > 1);
/// });
/// ```
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

/// Samples plausible HNSW parameter configurations.
///
/// # Examples
///
/// ```ignore
/// use crate::hnsw::tests::property::strategies::hnsw_params_strategy;
/// use proptest::prelude::*;
///
/// let strategy = hnsw_params_strategy();
/// proptest!(|(params in strategy)| {
///     prop_assert!(params.max_connections >= 2);
/// });
/// ```
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

/// Converts a generated dataset into an [`HnswFixture`].
///
/// # Examples
///
/// ```ignore
/// use crate::hnsw::tests::property::datasets::generate_uniform_dataset;
/// use crate::hnsw::tests::property::strategies::map_dataset;
/// use crate::hnsw::tests::property::types::{
///     HnswParamsSeed,
///     VectorDistribution,
/// };
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let params = HnswParamsSeed {
///     max_connections: 4,
///     ef_construction: 4,
///     level_multiplier: 1.0,
///     max_level: 4,
///     rng_seed: 0,
/// };
/// let mut rng = SmallRng::seed_from_u64(1);
/// let dataset = generate_uniform_dataset(&mut rng);
/// let fixture = map_dataset(VectorDistribution::Uniform, params, dataset);
/// assert_eq!(fixture.distribution, VectorDistribution::Uniform);
/// ```
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
