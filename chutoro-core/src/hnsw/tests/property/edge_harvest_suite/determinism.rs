//! Determinism property checks for graph generation.

use proptest::test_runner::{TestCaseError, TestCaseResult};
use rand::{SeedableRng, rngs::SmallRng};

use super::{GraphTopology, generate_graph_for_topology};

/// Property 1: Determinism — same seed produces identical output.
///
/// Verifies that for a given random seed and topology, the graph generator
/// produces identical results across multiple invocations.
pub(super) fn run_graph_determinism_property(seed: u64, topology: GraphTopology) -> TestCaseResult {
    let mut rng1 = SmallRng::seed_from_u64(seed);
    let mut rng2 = SmallRng::seed_from_u64(seed);

    let graph1 = generate_graph_for_topology(topology, &mut rng1);
    let graph2 = generate_graph_for_topology(topology, &mut rng2);

    if graph1.node_count != graph2.node_count {
        return Err(TestCaseError::fail(format!(
            "{topology:?}: node_count mismatch: {} vs {}",
            graph1.node_count, graph2.node_count
        )));
    }

    if graph1.metadata != graph2.metadata {
        return Err(TestCaseError::fail(format!(
            "{topology:?}: metadata mismatch: left={:?} right={:?}",
            graph1.metadata, graph2.metadata
        )));
    }

    if graph1.edges.len() != graph2.edges.len() {
        return Err(TestCaseError::fail(format!(
            "{topology:?}: edge count mismatch: {} vs {}",
            graph1.edges.len(),
            graph2.edges.len()
        )));
    }

    for (i, (e1, e2)) in graph1.edges.iter().zip(graph2.edges.iter()).enumerate() {
        if e1 != e2 {
            return Err(TestCaseError::fail(format!(
                "{topology:?}: edge {i} differs: {e1:?} vs {e2:?}"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    /// Helper to run determinism property for a single (seed, topology) pair.
    fn assert_determinism(seed: u64, topology: GraphTopology) {
        run_graph_determinism_property(seed, topology).unwrap_or_else(|e| {
            panic!("determinism failed for seed={seed}, topology={topology:?}: {e}")
        });
    }

    // ========================================================================
    // Determinism Property Tests (rstest)
    // ========================================================================

    #[rstest]
    #[case::random_seed_42(42, GraphTopology::Random)]
    #[case::scale_free_seed_42(42, GraphTopology::ScaleFree)]
    #[case::lattice_seed_42(42, GraphTopology::Lattice)]
    #[case::disconnected_seed_42(42, GraphTopology::Disconnected)]
    #[case::random_seed_12345(12345, GraphTopology::Random)]
    #[case::scale_free_seed_12345(12345, GraphTopology::ScaleFree)]
    #[case::lattice_seed_12345(12345, GraphTopology::Lattice)]
    #[case::disconnected_seed_12345(12345, GraphTopology::Disconnected)]
    fn graph_determinism_rstest(#[case] seed: u64, #[case] topology: GraphTopology) {
        run_graph_determinism_property(seed, topology).expect("determinism property must hold");
    }

    #[test]
    fn determinism_across_multiple_seeds() {
        // Test determinism with additional seeds beyond rstest cases.
        let seeds = [0, 1, 999, 65535, u64::MAX];
        let topologies = [
            GraphTopology::Random,
            GraphTopology::ScaleFree,
            GraphTopology::Lattice,
            GraphTopology::Disconnected,
        ];
        for seed in seeds {
            for topology in topologies {
                assert_determinism(seed, topology);
            }
        }
    }
}
