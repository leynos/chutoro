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
    graph_topologies::generate_graph_for_topology,
    types::{
        EdgeHarvestPlan, GraphFixture, GraphTopology, HnswFixture, HnswParamsSeed, IdempotencyPlan,
        MAX_REBUILD_ATTEMPTS, MIN_REBUILD_ATTEMPTS, MutationOperationSeed, MutationPlan,
        VectorDistribution,
    },
};

const MAX_MUTATION_STEPS: usize = 12;

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

/// Samples mutation plans covering add/delete/reconfigure sequences.
pub(super) fn mutation_plan_strategy() -> impl Strategy<Value = MutationPlan> {
    (
        any::<u16>(),
        prop::collection::vec(mutation_operation_seed_strategy(), 1..=MAX_MUTATION_STEPS),
    )
        .prop_map(|(initial_population_hint, operations)| MutationPlan {
            initial_population_hint,
            operations,
        })
}

fn mutation_operation_seed_strategy() -> impl Strategy<Value = MutationOperationSeed> {
    prop_oneof![
        any::<u16>().prop_map(|slot_hint| MutationOperationSeed::Add { slot_hint }),
        any::<u16>().prop_map(|slot_hint| MutationOperationSeed::Delete { slot_hint }),
        hnsw_params_strategy().prop_map(|params| MutationOperationSeed::Reconfigure { params }),
    ]
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

/// Samples idempotency test plans for verifying duplicate insertion rejection.
///
/// Generates plans specifying which nodes to attempt re-insertion on and how
/// many times to retry each. The actual node indices are resolved during test
/// execution by mapping hints modulo the fixture length.
pub(super) fn idempotency_plan_strategy() -> impl Strategy<Value = IdempotencyPlan> {
    (prop::collection::vec(any::<u16>(), 1..=8), 1_usize..=5).prop_map(
        |(duplicate_hints, attempts_per_index)| IdempotencyPlan {
            duplicate_hints,
            attempts_per_index,
        },
    )
}

/// Samples edge harvest plans for verifying candidate edge harvesting consistency.
///
/// Generates plans with `rebuild_attempts` in the range
/// [`MIN_REBUILD_ATTEMPTS`]..=[`MAX_REBUILD_ATTEMPTS`] (currently 2..=5).
/// The minimum ensures meaningful determinism checks (comparing at least two builds).
/// The maximum keeps property runs within time budgets.
#[expect(
    dead_code,
    reason = "prepared for future proptest-based edge harvest tests"
)]
pub(super) fn edge_harvest_plan_strategy() -> impl Strategy<Value = EdgeHarvestPlan> {
    (MIN_REBUILD_ATTEMPTS..=MAX_REBUILD_ATTEMPTS).prop_map(EdgeHarvestPlan::new)
}

/// Generates graph fixtures covering multiple topologies for edge harvest testing.
///
/// Produces graphs with random, scale-free, lattice, and disconnected structures
/// for testing candidate edge harvest algorithms. All generators guarantee at
/// least one edge, eliminating the need for filtering.
///
/// # Examples
///
/// ```ignore
/// use crate::hnsw::tests::property::strategies::graph_fixture_strategy;
/// use proptest::prelude::*;
///
/// let strategy = graph_fixture_strategy();
/// proptest!(|(fixture in strategy)| {
///     prop_assert!(!fixture.graph.edges.is_empty());
/// });
/// ```
pub(super) fn graph_fixture_strategy() -> impl Strategy<Value = GraphFixture> {
    (any::<GraphTopology>(), any::<u64>()).prop_map(|(topology, seed)| {
        let mut rng = SmallRng::seed_from_u64(seed);
        let graph = generate_graph_for_topology(topology, &mut rng);
        GraphFixture { topology, graph }
    })
}
