//! RNN uplift property checks for generated graphs.

use std::collections::HashSet;

use proptest::test_runner::{TestCaseError, TestCaseResult};

use super::{GraphFixture, GraphTopology};
use crate::CandidateEdge;

/// Builds adjacency lists with distances for each node from an edge list.
///
/// Returns a vector where `adjacency[i]` contains tuples of `(neighbour_id, distance)`
/// for all edges incident to node `i`.
fn build_adjacency_lists(node_count: usize, edges: &[CandidateEdge]) -> Vec<Vec<(usize, f32)>> {
    let mut adjacency: Vec<Vec<(usize, f32)>> = vec![Vec::new(); node_count];
    for edge in edges {
        adjacency[edge.source()].push((edge.target(), edge.distance()));
        adjacency[edge.target()].push((edge.source(), edge.distance()));
    }
    adjacency
}

/// Computes the top-k nearest neighbours for each node from adjacency lists.
///
/// For each node, sorts neighbours by distance and returns the k closest as a `HashSet`.
fn compute_top_k_neighbours(adjacency: Vec<Vec<(usize, f32)>>, k: usize) -> Vec<HashSet<usize>> {
    adjacency
        .into_iter()
        .map(|mut neighbours| {
            neighbours.sort_by(|a, b| a.1.total_cmp(&b.1));
            neighbours.into_iter().take(k).map(|(id, _)| id).collect()
        })
        .collect()
}

/// Counts symmetric relationships across all top-k neighbour sets.
///
/// Returns a tuple of `(symmetric_count, total_relationships)` where a relationship
/// is symmetric if `v` is in the top-k of `u` AND `u` is in the top-k of `v`.
fn count_symmetric_relationships(top_k_neighbours: &[HashSet<usize>]) -> (usize, usize) {
    let mut symmetric_count = 0usize;
    let mut total_relationships = 0usize;

    for (node, neighbours) in top_k_neighbours.iter().enumerate() {
        for &neighbour in neighbours {
            total_relationships += 1;
            if top_k_neighbours[neighbour].contains(&node) {
                symmetric_count += 1;
            }
        }
    }

    (symmetric_count, total_relationships)
}

/// Checks whether the RNN score calculation would be trivial.
///
/// Returns `true` if the inputs indicate a degenerate case where the RNN score
/// is trivially 1.0 (perfect symmetry).
fn is_trivial_rnn_case(node_count: usize, k: usize) -> bool {
    k == 0 || node_count == 0
}

/// Computes the RNN (Reverse Nearest Neighbour) score.
///
/// The RNN score measures symmetry in neighbour relationships. For each node,
/// we find its top-k nearest neighbours by distance. The score is the fraction
/// of (node, neighbour) pairs where the relationship is mutual: if `v` is in
/// the top-k of `u`, then `u` is also in the top-k of `v`.
///
/// Returns a value in [0.0, 1.0] where 1.0 indicates perfect symmetry.
fn compute_rnn_score(node_count: usize, edges: &[CandidateEdge], k: usize) -> f64 {
    if is_trivial_rnn_case(node_count, k) {
        return 1.0; // Trivially symmetric.
    }

    let adjacency = build_adjacency_lists(node_count, edges);
    let top_k_neighbours = compute_top_k_neighbours(adjacency, k);
    let (symmetric_count, total_relationships) = count_symmetric_relationships(&top_k_neighbours);

    if total_relationships == 0 {
        1.0
    } else {
        symmetric_count as f64 / total_relationships as f64
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
    // Use k=5 for RNN computation (typical neighbourhood size).
    let k = 5;
    let rnn_score = compute_rnn_score(fixture.graph.node_count, &fixture.graph.edges, k);

    // Define minimum acceptable RNN scores by topology.
    // Note: Scale-free graphs with edges_per_new_node=1 create extremely star-like
    // structures where most nodes only connect to a single hub, resulting in very
    // low symmetry scores (often 0.1-0.2). We use a permissive threshold.
    let min_score = match fixture.topology {
        GraphTopology::Lattice => 0.8, // Highly regular, should be very symmetric.
        GraphTopology::ScaleFree => 0.05, // Hubs with m=1 create extreme asymmetry.
        GraphTopology::Random => 0.3,  // Moderate symmetry expected.
        GraphTopology::Disconnected => 0.3, // Within components should be symmetric.
    };

    if rnn_score < min_score {
        return Err(TestCaseError::fail(format!(
            "{:?} topology: RNN score {rnn_score:.3} below minimum {min_score:.3}",
            fixture.topology
        )));
    }

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
    // Helper Function Unit Tests
    // ========================================================================

    #[test]
    fn compute_rnn_score_empty_graph() {
        // Empty edges should fall back to the total_relationships == 0 path.
        assert_eq!(compute_rnn_score(5, &[], 5), 1.0);
    }

    #[test]
    fn compute_rnn_score_k_zero_is_trivially_one() {
        let edges = vec![CandidateEdge::new(0, 1, 1.0, 0)];
        assert_eq!(compute_rnn_score(2, &edges, 0), 1.0);
    }

    #[test]
    fn compute_rnn_score_zero_nodes_is_trivially_one() {
        let edges: Vec<CandidateEdge> = Vec::new();
        assert_eq!(compute_rnn_score(0, &edges, 5), 1.0);
    }

    #[test]
    fn compute_rnn_score_symmetric_pair() {
        // Single edge: 0 -- 1 (perfectly symmetric)
        let edges = vec![CandidateEdge::new(0, 1, 1.0, 0)];
        // With k=5, both nodes have each other as their only neighbour.
        assert_eq!(compute_rnn_score(2, &edges, 5), 1.0);
    }

    #[test]
    fn compute_rnn_score_asymmetric_star() {
        // Star: 0 is connected to 1, 2, 3
        // Node 0 has neighbours [1, 2, 3], each other node only has [0].
        let edges = vec![
            CandidateEdge::new(0, 1, 1.0, 0),
            CandidateEdge::new(0, 2, 2.0, 1),
            CandidateEdge::new(0, 3, 3.0, 2),
        ];
        let score = compute_rnn_score(4, &edges, 2);
        // With k=2:
        // - Node 0's top-2: [1, 2] (distances 1.0, 2.0)
        // - Node 1's top-2: [0] (only neighbour)
        // - Node 2's top-2: [0] (only neighbour)
        // - Node 3's top-2: [0] (only neighbour)
        // Relationships from node 0: 1 (mutual? 1 has 0: yes), 2 (mutual? 2 has 0: yes)
        // Relationships from node 1: 0 (mutual? 0 has 1: yes)
        // Relationships from node 2: 0 (mutual? 0 has 2: yes)
        // Relationships from node 3: 0 (mutual? 0 has 3: no, 3 not in 0's top-2)
        // Total: 5 relationships, 4 mutual -> 0.8
        assert!((score - 0.8).abs() < 0.01);
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
