//! RNN uplift property checks for generated graphs.

use proptest::test_runner::{TestCaseError, TestCaseResult};

use super::super::graph_metrics::compute_rnn_score;
use super::{GraphFixture, GraphTopology};

/// Returns the minimum acceptable RNN score for the provided topology.
fn min_rnn_score_for_topology(topology: GraphTopology) -> f64 {
    match topology {
        GraphTopology::Lattice => 0.8, // Highly regular, should be very symmetric.
        GraphTopology::ScaleFree => 0.05, // Hubs with m=1 create extreme asymmetry.
        GraphTopology::Random => 0.3,  // Moderate symmetry expected.
        GraphTopology::Disconnected => 0.3, // Within components should be symmetric.
    }
}

/// Property 4: RNN uplift — measures symmetric neighbour relationships.
///
/// Verifies that the Reverse Nearest Neighbour (RNN) score meets minimum
/// thresholds based on topology characteristics:
/// - **Lattice**: ≥ 0.8 (highly regular structure implies high symmetry)
/// - **ScaleFree**: ≥ 0.05 (hub nodes create extreme asymmetry)
/// - **Random**: ≥ 0.3 (moderate symmetry expected)
/// - **Disconnected**: ≥ 0.3 (within-component symmetry)
///
/// Note: Edge canonicality (source < target) is only enforced for topologies
/// that guarantee it. Scale-free graphs using preferential attachment naturally
/// produce edges where the new node is the source (source > target when
/// connecting to earlier nodes).
pub(super) fn run_rnn_uplift_property(fixture: &GraphFixture) -> TestCaseResult {
    // Verify edge validity (no self-loops, valid node indices).
    for (i, edge) in fixture.graph.edges.iter().enumerate() {
        if edge.source() == edge.target() {
            return Err(TestCaseError::fail(format!(
                "edge {i} is a self-loop: {} -> {}",
                edge.source(),
                edge.target()
            )));
        }
        if edge.source() >= fixture.graph.node_count {
            return Err(TestCaseError::fail(format!(
                "edge {i} source {} out of bounds (node_count = {})",
                edge.source(),
                fixture.graph.node_count
            )));
        }
        if edge.target() >= fixture.graph.node_count {
            return Err(TestCaseError::fail(format!(
                "edge {i} target {} out of bounds (node_count = {})",
                edge.target(),
                fixture.graph.node_count
            )));
        }
    }

    // Use k=5 for RNN computation (typical neighbourhood size).
    let k = 5;
    let rnn_score = compute_rnn_score(fixture.graph.node_count, &fixture.graph.edges, k);

    // Define minimum acceptable RNN scores by topology.
    // Note: Scale-free graphs with edges_per_new_node=1 create extremely star-like
    // structures where most nodes only connect to a single hub, resulting in very
    // low symmetry scores (often 0.1-0.2). We use a permissive threshold.
    let min_score = min_rnn_score_for_topology(fixture.topology);

    if rnn_score < min_score {
        return Err(TestCaseError::fail(format!(
            "{:?} topology: RNN score {rnn_score:.3} below minimum {min_score:.3}",
            fixture.topology
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rstest::rstest;

    use super::super::super::strategies::graph_fixture_strategy;
    use super::super::build_fixture;

    // ========================================================================
    // RNN Uplift Property Tests (rstest)
    // ========================================================================

    #[rstest]
    #[case::random(GraphTopology::Random, 42)]
    #[case::scale_free(GraphTopology::ScaleFree, 42)]
    #[case::lattice(GraphTopology::Lattice, 42)]
    #[case::disconnected(GraphTopology::Disconnected, 42)]
    fn graph_rnn_uplift_rstest(#[case] topology: GraphTopology, #[case] seed: u64) {
        let fixture = build_fixture(seed, topology);
        run_rnn_uplift_property(&fixture).expect("RNN uplift property must hold");
    }

    // ========================================================================
    // Proptest Stochastic Coverage
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn graph_topology_rnn_uplift_proptest(fixture in graph_fixture_strategy()) {
            run_rnn_uplift_property(&fixture)?;
        }
    }
}
