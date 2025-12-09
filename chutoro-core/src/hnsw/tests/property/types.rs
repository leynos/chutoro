//! Type definitions and metadata structures for property-based HNSW tests.
//!
//! Provides enums and fixtures consumed by the HNSW property strategies and
//! generators.

use test_strategy::Arbitrary;

use crate::{
    DataSourceError,
    hnsw::{HnswError, HnswParams},
};

use super::support::DenseVectorSource;

/// Kind of dataset produced by [`HnswFixture`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Arbitrary)]
pub(super) enum VectorDistribution {
    /// Points sampled uniformly in a bounding hypercube.
    #[weight(3)]
    Uniform,
    /// Dense clusters generated around random centroids.
    #[weight(3)]
    Clustered,
    /// Points lying near a low-dimensional manifold embedded in the ambient
    /// space.
    #[weight(2)]
    Manifold,
    /// Uniform data with explicit duplicate vectors to stress idempotency.
    #[weight(1)]
    Duplicates,
}

/// Metadata describing how the dataset was synthesised.
#[derive(Clone, Debug)]
pub(super) enum DistributionMetadata {
    /// No additional structure beyond the sampled bounds.
    Uniform {
        /// Half-width of the hypercube used during sampling.
        bound: f32,
    },
    /// Records the clusters used for generation.
    Clustered {
        /// Information about the generated clusters in insertion order.
        clusters: Vec<ClusterInfo>,
    },
    /// Records the manifold basis and jitter applied to samples.
    Manifold {
        /// Ambient dimensionality of the dataset.
        ambient_dim: usize,
        /// Intrinsic manifold dimensionality.
        intrinsic_dim: usize,
        /// Orthonormal basis spanning the manifold.
        basis: Vec<Vec<f32>>,
        /// Maximum per-axis noise added after projection.
        noise_bound: f32,
        /// Origin offset applied before adding coefficients.
        origin: Vec<f32>,
    },
    /// Explicit duplicate indices generated for the dataset.
    Duplicates {
        /// Groups of indices that contain identical vectors.
        groups: Vec<Vec<usize>>,
    },
}

/// Cluster specification captured when synthesising clustered datasets.
#[derive(Clone, Debug)]
pub(super) struct ClusterInfo {
    /// Index of the first vector belonging to the cluster.
    pub start: usize,
    /// Number of vectors contained in the cluster.
    pub len: usize,
    /// Radius used when sampling offsets around the centroid.
    pub radius: f32,
    /// Centroid vector the cluster was sampled around.
    pub centroid: Vec<f32>,
}

/// Strategy output bundling vector data with sampled configuration.
#[derive(Clone, Debug)]
pub(super) struct HnswFixture {
    /// Dataset distribution metadata.
    pub distribution: VectorDistribution,
    /// Vector data produced by the generator.
    pub vectors: Vec<Vec<f32>>,
    /// Additional metadata describing the dataset structure.
    pub metadata: DistributionMetadata,
    /// Sampled parameter configuration for building an index.
    pub params: HnswParamsSeed,
}

impl HnswFixture {
    /// Returns the vector dimensionality for the generated dataset.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.vectors.first().map_or(0, |vector| vector.len())
    }

    /// Converts the fixture into a [`DenseVectorSource`] suitable for building
    /// HNSW indices in property tests.
    pub fn into_source(self) -> Result<DenseVectorSource, DataSourceError> {
        DenseVectorSource::new("hnsw-fixture", self.vectors)
    }
}

/// Test parameter configuration derived from property generators.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct HnswParamsSeed {
    pub max_connections: usize,
    pub ef_construction: usize,
    pub level_multiplier: f64,
    pub max_level: usize,
    pub rng_seed: u64,
}

impl HnswParamsSeed {
    /// Builds concrete [`HnswParams`] from the sampled seed.
    pub fn build(&self) -> Result<HnswParams, HnswError> {
        HnswParams::new(self.max_connections, self.ef_construction).map(|params| {
            params
                .with_level_multiplier(self.level_multiplier)
                .with_max_level(self.max_level)
                .with_rng_seed(self.rng_seed)
        })
    }
}

/// Sequence of HNSW mutation operations applied during the property test.
#[derive(Clone, Debug)]
pub(super) struct MutationPlan {
    /// Hint controlling the number of nodes inserted before applying
    /// operations.
    pub initial_population_hint: u16,
    /// Operations to attempt after seeding the index.
    pub operations: Vec<MutationOperationSeed>,
}

/// Mutation operation sampled by the strategy.
#[derive(Clone, Debug)]
pub(super) enum MutationOperationSeed {
    /// Inserts a node selected from the not-yet-inserted pool.
    Add { slot_hint: u16 },
    /// Deletes a node selected from the inserted pool.
    Delete { slot_hint: u16 },
    /// Reconfigures the index by deriving parameters from the provided seed.
    Reconfigure { params: HnswParamsSeed },
}

/// Plan describing which nodes to attempt duplicate insertion on.
///
/// Used by the idempotency property to verify that repeated insertions of the
/// same node leave the graph state unchanged.
#[derive(Clone, Debug)]
pub(super) struct IdempotencyPlan {
    /// Hints for selecting indices to duplicate (mapped to actual indices via
    /// modulo during test execution).
    pub duplicate_hints: Vec<u16>,
    /// Number of duplicate attempts per selected index (1-5).
    pub attempts_per_index: usize,
}
