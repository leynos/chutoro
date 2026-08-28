//! Type definitions for MST property-based tests.
//!
//! Provides the fixture, configuration, and weight distribution types used
//! by the graph generation strategies and property functions.

use crate::CandidateEdge;
use mockable::{DefaultEnv, Env};

/// Weight distribution strategy for generated graphs.
///
/// Controls how edge weights are assigned during graph generation, producing
/// inputs that stress different aspects of the parallel Kruskal implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WeightDistribution {
    /// Each edge has a unique weight drawn from a continuous range.
    Unique,
    /// Large groups of edges share identical weights, stressing tie-breaking.
    ManyIdentical,
    /// Sparse graph with approximately `1.5n` to `2n` edges.
    Sparse,
    /// Dense graph approaching a complete graph (edge probability 0.7-0.95).
    Dense,
    /// Multiple disconnected components with no cross-component edges.
    Disconnected,
}

/// Fixture for MST property tests.
///
/// Captures the node count, generated candidate edges, and the weight
/// distribution used during generation, providing full context for failure
/// diagnosis.
#[derive(Clone, Debug)]
pub(super) struct MstFixture {
    /// Number of nodes in the graph.
    pub node_count: usize,
    /// Generated candidate edges with weights and sequence numbers.
    pub edges: Vec<CandidateEdge>,
    /// Weight distribution used during generation.
    pub distribution: WeightDistribution,
}

/// Configuration for the concurrency safety property.
///
/// Controls how many times the parallel Kruskal algorithm is re-executed on
/// the same input to detect race-induced non-determinism.
pub(super) struct ConcurrencyConfig {
    /// Number of times to repeat the MST computation per input.
    pub repetitions: usize,
}

/// Minimum number of repetitions required for a meaningful determinism
/// check (baseline + at least one comparison run).
const MIN_CONCURRENCY_REPS: usize = 2;

impl ConcurrencyConfig {
    /// Loads the configuration from environment variables, falling back to
    /// sensible defaults.
    ///
    /// The environment variable `CHUTORO_MST_PBT_CONCURRENCY_REPS` controls
    /// the repetition count (default: 5).  Values below
    /// [`MIN_CONCURRENCY_REPS`] are clamped upward so the property always
    /// performs at least one comparison run against the baseline.
    pub(super) fn load() -> Self {
        Self::load_with_env(&DefaultEnv)
    }

    /// Load property-test configuration through an injected environment reader.
    fn load_with_env(env: &dyn Env) -> Self {
        let repetitions = env
            .string("CHUTORO_MST_PBT_CONCURRENCY_REPS")
            .and_then(|s| s.parse().ok())
            .unwrap_or(5)
            .max(MIN_CONCURRENCY_REPS);
        Self { repetitions }
    }
}
