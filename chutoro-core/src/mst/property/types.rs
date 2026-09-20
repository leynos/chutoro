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

#[cfg(test)]
mod tests {
    //! Unit tests for MST property configuration.

    use super::*;
    use mockable::MockEnv;
    use rstest::rstest;

    #[rstest]
    #[case(None, 5)]
    #[case(Some("8"), 8)]
    #[case(Some("invalid"), 5)]
    #[case(Some("1"), MIN_CONCURRENCY_REPS)]
    fn load_with_env_uses_valid_overrides_and_safe_defaults(
        #[case] value: Option<&str>,
        #[case] expected: usize,
    ) {
        let configured_value = value.map(str::to_owned);
        let mut env = MockEnv::new();
        env.expect_string().returning(move |key| {
            assert_eq!(key, "CHUTORO_MST_PBT_CONCURRENCY_REPS");
            configured_value.clone()
        });

        assert_eq!(ConcurrencyConfig::load_with_env(&env).repetitions, expected);
    }
}
