//! Type definitions and metadata structures for property-based HNSW tests.
//!
//! Provides enums and fixtures consumed by the HNSW property strategies and
//! generators.

use test_strategy::Arbitrary;

use crate::{
    CandidateEdge, DataSourceError,
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

/// Plan for edge harvest property testing.
///
/// Used to verify that candidate edge harvesting produces consistent,
/// structurally valid results.
#[derive(Clone, Debug)]
pub(super) struct EdgeHarvestPlan {
    /// Number of times to rebuild with the same seed to check determinism.
    ///
    /// # Valid Range
    ///
    /// Must be at least 2 for meaningful determinism checks (comparing multiple
    /// rebuilds). Upper bounds are chosen to keep property runs within time budgets
    /// (typically 2-5).
    rebuild_attempts: usize,
}

/// Minimum rebuild attempts required for meaningful determinism checks.
pub(super) const MIN_REBUILD_ATTEMPTS: usize = 2;

/// Maximum rebuild attempts to keep property runs within time budgets.
pub(super) const MAX_REBUILD_ATTEMPTS: usize = 5;

impl EdgeHarvestPlan {
    /// Creates a new edge harvest plan with the specified rebuild attempts.
    ///
    /// # Panics
    ///
    /// Panics if `rebuild_attempts < MIN_REBUILD_ATTEMPTS` (currently 2).
    #[must_use]
    pub fn new(rebuild_attempts: usize) -> Self {
        assert!(
            rebuild_attempts >= MIN_REBUILD_ATTEMPTS,
            "rebuild_attempts must be at least {MIN_REBUILD_ATTEMPTS} for meaningful \
             determinism checks, got {rebuild_attempts}"
        );
        Self { rebuild_attempts }
    }

    /// Returns the number of rebuild attempts.
    #[must_use]
    pub fn rebuild_attempts(&self) -> usize {
        self.rebuild_attempts
    }
}

// ============================================================================
// Graph Topology Types for Edge Harvest Testing
// ============================================================================

/// Kind of graph topology produced by graph generators for edge harvest testing.
///
/// These are graph structures (edge sets) as opposed to vector distributions.
/// Used for testing the candidate edge harvest algorithm with various graph
/// topologies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Arbitrary)]
pub(super) enum GraphTopology {
    /// Erdos-Renyi random graph with uniform edge probability.
    #[weight(3)]
    Random,
    /// Scale-free graph with power-law degree distribution (Barabasi-Albert model).
    #[weight(2)]
    ScaleFree,
    /// Grid/lattice graph with uniform local connectivity.
    #[weight(2)]
    Lattice,
    /// Graph with multiple disconnected components.
    #[weight(2)]
    Disconnected,
}

/// Metadata describing how a graph topology was synthesised.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum GraphMetadata {
    /// Erdos-Renyi random graph.
    Random {
        /// Number of nodes in the graph.
        node_count: usize,
        /// Edge probability used during generation.
        edge_probability: f64,
    },
    /// Scale-free graph using Barabasi-Albert preferential attachment.
    ScaleFree {
        /// Number of nodes in the graph.
        node_count: usize,
        /// Number of edges to attach from each new node.
        edges_per_new_node: usize,
        /// Exponent of the power-law distribution.
        exponent: f64,
    },
    /// Grid/lattice structure.
    Lattice {
        /// Grid dimensions (rows, columns).
        dimensions: (usize, usize),
        /// Whether diagonal edges are included.
        with_diagonals: bool,
    },
    /// Disconnected components.
    Disconnected {
        /// Number of components.
        component_count: usize,
        /// Sizes of each component.
        component_sizes: Vec<usize>,
    },
}

/// Generated graph with edges and metadata for property testing.
#[derive(Clone, Debug)]
pub(super) struct GeneratedGraph {
    /// Number of nodes in the graph.
    pub node_count: usize,
    /// Generated edges as `CandidateEdge` instances.
    pub edges: Vec<CandidateEdge>,
    /// Metadata describing the generation parameters.
    pub metadata: GraphMetadata,
}

/// Test fixture bundling graph topology with generated graph structure.
#[derive(Clone, Debug)]
pub(super) struct GraphFixture {
    /// Graph topology type.
    pub topology: GraphTopology,
    /// Generated graph structure.
    pub graph: GeneratedGraph,
}
