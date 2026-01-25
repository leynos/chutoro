//! Connectivity preservation checks for generated graphs.

use proptest::test_runner::{TestCaseError, TestCaseResult};

use super::{GraphFixture, GraphMetadata, GraphTopology};
use crate::CandidateEdge;

/// Counts connected components using union-find with path compression.
///
/// Returns the number of distinct connected components in the graph.
fn count_connected_components(node_count: usize, edges: &[CandidateEdge]) -> usize {
    if node_count == 0 {
        return 0;
    }

    let mut parent: Vec<usize> = (0..node_count).collect();

    // Find with path compression (iterative to avoid stack overflow).
    fn find(parent: &mut [usize], mut node: usize) -> usize {
        let mut root = node;
        while parent[root] != root {
            root = parent[root];
        }
        // Path compression: point all visited nodes directly to root.
        while parent[node] != root {
            let next = parent[node];
            parent[node] = root;
            node = next;
        }
        root
    }

    // Union all edges.
    for edge in edges {
        let root_s = find(&mut parent, edge.source());
        let root_t = find(&mut parent, edge.target());
        if root_s != root_t {
            parent[root_t] = root_s;
        }
    }

    // Count unique roots.
    (0..node_count)
        .filter(|&i| find(&mut parent, i) == i)
        .count()
}

/// Validates connectivity expectations based on graph metadata.
///
/// Returns `Ok(())` if the actual component count matches expectations for the
/// topology, or `Err(message)` describing the validation failure.
fn validate_connectivity_for_metadata(
    metadata: &GraphMetadata,
    actual_components: usize,
) -> Result<(), String> {
    match metadata {
        GraphMetadata::Disconnected {
            component_count, ..
        } => {
            // Disconnected graphs should have at least the specified components.
            // May have more if internal edges fail to connect all nodes within a component.
            if actual_components < *component_count {
                return Err(format!(
                    "disconnected graph has fewer components than expected: {actual_components} < {component_count}"
                ));
            }
        }
        GraphMetadata::Lattice { .. } => {
            // Lattice grids are always connected by construction.
            if actual_components != 1 {
                return Err(format!(
                    "lattice should be connected, found {actual_components} components"
                ));
            }
        }
        GraphMetadata::ScaleFree { node_count, .. } => {
            // Scale-free graphs built with Barabasi-Albert model are connected
            // by construction (each new node attaches to existing nodes).
            if actual_components > 1 && *node_count > 3 {
                return Err(format!(
                    "scale-free graph with {node_count} nodes has {actual_components} components (expected 1)"
                ));
            }
        }
        GraphMetadata::Random { .. } => {
            // Random graphs may or may not be connected depending on edge probability.
            // We don't enforce connectivity for random graphs; this is informational.
        }
    }
    Ok(())
}

/// Property 3: Connectivity preservation — connected topologies remain connected.
///
/// Verifies expected connectivity based on topology:
/// - **Lattice**: Must have exactly 1 connected component
/// - **ScaleFree**: Must have exactly 1 component (for n > 3, due to initial clique)
/// - **Random**: Informational only (connectivity is probabilistic)
/// - **Disconnected**: Must have at least `component_count` components
pub(super) fn run_connectivity_preservation_property(fixture: &GraphFixture) -> TestCaseResult {
    let actual_components =
        count_connected_components(fixture.graph.node_count, &fixture.graph.edges);

    validate_connectivity_for_metadata(&fixture.graph.metadata, actual_components)
        .map_err(TestCaseError::fail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rstest::rstest;

    use super::super::super::strategies::graph_fixture_strategy;
    use super::super::build_fixture;

    // ========================================================================
    // Connectivity Preservation Property Tests (rstest)
    // ========================================================================

    #[rstest]
    #[case::random(GraphTopology::Random, 42)]
    #[case::scale_free(GraphTopology::ScaleFree, 42)]
    #[case::lattice(GraphTopology::Lattice, 42)]
    #[case::disconnected(GraphTopology::Disconnected, 42)]
    fn graph_connectivity_rstest(#[case] topology: GraphTopology, #[case] seed: u64) {
        let fixture = build_fixture(seed, topology);
        run_connectivity_preservation_property(&fixture)
            .expect("connectivity preservation property must hold");
    }

    // ========================================================================
    // Helper Function Unit Tests
    // ========================================================================

    #[test]
    fn count_connected_components_empty_graph() {
        assert_eq!(count_connected_components(0, &[]), 0);
    }

    #[test]
    fn count_connected_components_isolated_nodes() {
        assert_eq!(count_connected_components(5, &[]), 5);
    }

    #[test]
    fn count_connected_components_fully_connected() {
        // Triangle: 0 -- 1 -- 2 -- 0
        let edges = vec![
            CandidateEdge::new(0, 1, 1.0, 0),
            CandidateEdge::new(1, 2, 1.0, 1),
            CandidateEdge::new(0, 2, 1.0, 2),
        ];
        assert_eq!(count_connected_components(3, &edges), 1);
    }

    #[test]
    fn count_connected_components_two_components() {
        // Component 1: 0 -- 1, Component 2: 2 -- 3
        let edges = vec![
            CandidateEdge::new(0, 1, 1.0, 0),
            CandidateEdge::new(2, 3, 1.0, 1),
        ];
        assert_eq!(count_connected_components(4, &edges), 2);
    }

    // ========================================================================
    // Proptest Stochastic Coverage
    // ========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn graph_topology_connectivity_proptest(fixture in graph_fixture_strategy()) {
            run_connectivity_preservation_property(&fixture)?;
        }
    }
}
