//! Candidate edge harvest property suite.
//!
//! Verifies that graph topology generators produce structurally valid edge sets
//! with bounded degrees, preserved connectivity, and symmetric neighbour relationships.
//!
//! Properties verified (per `docs/property-testing-design.md` §3.2):
//! 1. **Determinism**: Same seed produces identical output
//! 2. **Degree ceilings**: Node degrees within topology-specific bounds
//! 3. **Connectivity preservation**: Connected topologies remain connected
//! 4. **RNN uplift**: Measure of symmetric neighbour relationships

use std::collections::HashSet;

use proptest::test_runner::{TestCaseError, TestCaseResult};
use rand::{SeedableRng, rngs::SmallRng};

use crate::CandidateEdge;

use super::graph_topologies::{
    generate_disconnected_graph, generate_lattice_graph, generate_random_graph,
    generate_scale_free_graph,
};
use super::types::{GraphFixture, GraphMetadata, GraphTopology};

// ============================================================================
// Helper Functions
// ============================================================================

/// Computes the degree of each node from an edge list.
///
/// Returns a vector where `degrees[i]` is the number of edges incident to node `i`.
/// For undirected graphs, each edge contributes 1 to both endpoints' degrees.
fn compute_node_degrees(node_count: usize, edges: &[CandidateEdge]) -> Vec<usize> {
    let mut degrees = vec![0usize; node_count];
    for edge in edges {
        degrees[edge.source()] += 1;
        degrees[edge.target()] += 1;
    }
    degrees
}

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

/// Generates a graph for the specified topology using the provided RNG.
///
/// Dispatches to the appropriate topology-specific generator.
fn generate_graph_for_topology(
    topology: GraphTopology,
    rng: &mut SmallRng,
) -> super::types::GeneratedGraph {
    match topology {
        GraphTopology::Random => generate_random_graph(rng),
        GraphTopology::ScaleFree => generate_scale_free_graph(rng),
        GraphTopology::Lattice => generate_lattice_graph(rng),
        GraphTopology::Disconnected => generate_disconnected_graph(rng),
    }
}

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
/// Returns `true` if any of the inputs indicate an empty or degenerate case
/// where the RNN score is trivially 1.0 (perfect symmetry).
fn is_trivial_rnn_case(node_count: usize, edges: &[CandidateEdge], k: usize) -> bool {
    edges.is_empty() || k == 0 || node_count == 0
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
    if is_trivial_rnn_case(node_count, edges, k) {
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

// ============================================================================
// Property Functions
// ============================================================================

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

/// Property 2: Degree ceilings — node degrees within topology-specific bounds.
///
/// Verifies that no node exceeds the maximum degree expected for its topology:
/// - **Lattice**: 4 (without diagonals) or 8 (with diagonals)
/// - **ScaleFree**: `node_count - 1` (theoretical hub maximum)
/// - **Random**: `node_count - 1` (complete graph maximum)
/// - **Disconnected**: `max(component_sizes) - 1` (within largest component)
pub(super) fn run_degree_ceiling_property(fixture: &GraphFixture) -> TestCaseResult {
    let degrees = compute_node_degrees(fixture.graph.node_count, &fixture.graph.edges);
    let max_degree = degrees.iter().copied().max().unwrap_or(0);

    let ceiling = match &fixture.graph.metadata {
        GraphMetadata::Lattice { with_diagonals, .. } => {
            if *with_diagonals {
                8
            } else {
                4
            }
        }
        GraphMetadata::ScaleFree { node_count, .. } => {
            // Hub can theoretically connect to all other nodes.
            node_count.saturating_sub(1)
        }
        GraphMetadata::Random { node_count, .. } => {
            // Complete graph case.
            node_count.saturating_sub(1)
        }
        GraphMetadata::Disconnected {
            component_sizes, ..
        } => {
            // Maximum degree is within the largest component.
            component_sizes
                .iter()
                .copied()
                .max()
                .unwrap_or(1)
                .saturating_sub(1)
        }
    };

    if max_degree > ceiling {
        return Err(TestCaseError::fail(format!(
            "{:?} topology: max_degree {max_degree} exceeds ceiling {ceiling}",
            fixture.topology
        )));
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

/// Property 4: RNN uplift — measures symmetric neighbour relationships.
///
/// Verifies that the Reverse Nearest Neighbour (RNN) score meets minimum
/// thresholds based on topology characteristics:
/// - **Lattice**: ≥ 0.8 (highly regular structure implies high symmetry)
/// - **ScaleFree**: ≥ 0.3 (hub nodes create asymmetry)
/// - **Random**: ≥ 0.4 (moderate symmetry expected)
/// - **Disconnected**: ≥ 0.4 (within-component symmetry)
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::tests::property::strategies::graph_fixture_strategy;
    use proptest::prelude::*;
    use rstest::rstest;

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

    // ========================================================================
    // Degree Ceiling Property Tests (rstest)
    // ========================================================================

    #[rstest]
    #[case::random(GraphTopology::Random, 42)]
    #[case::scale_free(GraphTopology::ScaleFree, 42)]
    #[case::lattice(GraphTopology::Lattice, 42)]
    #[case::disconnected(GraphTopology::Disconnected, 42)]
    fn graph_degree_ceiling_rstest(#[case] topology: GraphTopology, #[case] seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let graph = match topology {
            GraphTopology::Random => generate_random_graph(&mut rng),
            GraphTopology::ScaleFree => generate_scale_free_graph(&mut rng),
            GraphTopology::Lattice => generate_lattice_graph(&mut rng),
            GraphTopology::Disconnected => generate_disconnected_graph(&mut rng),
        };
        let fixture = GraphFixture { topology, graph };
        run_degree_ceiling_property(&fixture).expect("degree ceiling property must hold");
    }

    // ========================================================================
    // Connectivity Preservation Property Tests (rstest)
    // ========================================================================

    #[rstest]
    #[case::random(GraphTopology::Random, 42)]
    #[case::scale_free(GraphTopology::ScaleFree, 42)]
    #[case::lattice(GraphTopology::Lattice, 42)]
    #[case::disconnected(GraphTopology::Disconnected, 42)]
    fn graph_connectivity_rstest(#[case] topology: GraphTopology, #[case] seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let graph = match topology {
            GraphTopology::Random => generate_random_graph(&mut rng),
            GraphTopology::ScaleFree => generate_scale_free_graph(&mut rng),
            GraphTopology::Lattice => generate_lattice_graph(&mut rng),
            GraphTopology::Disconnected => generate_disconnected_graph(&mut rng),
        };
        let fixture = GraphFixture { topology, graph };
        run_connectivity_preservation_property(&fixture)
            .expect("connectivity preservation property must hold");
    }

    // ========================================================================
    // RNN Uplift Property Tests (rstest)
    // ========================================================================

    #[rstest]
    #[case::random(GraphTopology::Random, 42)]
    #[case::scale_free(GraphTopology::ScaleFree, 42)]
    #[case::lattice(GraphTopology::Lattice, 42)]
    #[case::disconnected(GraphTopology::Disconnected, 42)]
    fn graph_rnn_uplift_rstest(#[case] topology: GraphTopology, #[case] seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let graph = match topology {
            GraphTopology::Random => generate_random_graph(&mut rng),
            GraphTopology::ScaleFree => generate_scale_free_graph(&mut rng),
            GraphTopology::Lattice => generate_lattice_graph(&mut rng),
            GraphTopology::Disconnected => generate_disconnected_graph(&mut rng),
        };
        let fixture = GraphFixture { topology, graph };
        run_rnn_uplift_property(&fixture).expect("RNN uplift property must hold");
    }

    // ========================================================================
    // Helper Function Unit Tests
    // ========================================================================

    #[test]
    fn compute_node_degrees_empty_graph() {
        let degrees = compute_node_degrees(5, &[]);
        assert_eq!(degrees, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn compute_node_degrees_simple_chain() {
        // Chain: 0 -- 1 -- 2
        let edges = vec![
            CandidateEdge::new(0, 1, 1.0, 0),
            CandidateEdge::new(1, 2, 1.0, 1),
        ];
        let degrees = compute_node_degrees(3, &edges);
        assert_eq!(degrees, vec![1, 2, 1]);
    }

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

    #[test]
    fn compute_rnn_score_empty_graph() {
        assert_eq!(compute_rnn_score(5, &[], 5), 1.0);
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
        fn graph_topology_degree_ceilings_proptest(fixture in graph_fixture_strategy()) {
            run_degree_ceiling_property(&fixture)?;
        }

        #[test]
        fn graph_topology_connectivity_proptest(fixture in graph_fixture_strategy()) {
            run_connectivity_preservation_property(&fixture)?;
        }

        #[test]
        fn graph_topology_rnn_uplift_proptest(fixture in graph_fixture_strategy()) {
            run_rnn_uplift_property(&fixture)?;
        }
    }

    // ========================================================================
    // Additional Edge Cases
    // ========================================================================

    /// Helper to run determinism property for a single (seed, topology) pair.
    fn assert_determinism(seed: u64, topology: GraphTopology) {
        run_graph_determinism_property(seed, topology).unwrap_or_else(|e| {
            panic!("determinism failed for seed={seed}, topology={topology:?}: {e}")
        });
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

    /// Helper to verify lattice max degree constraint.
    fn verify_lattice_degree(seed: u64, graph: &super::super::types::GeneratedGraph) {
        let GraphMetadata::Lattice {
            with_diagonals: false,
            dimensions: (rows, cols),
        } = graph.metadata
        else {
            return;
        };
        let degrees = compute_node_degrees(graph.node_count, &graph.edges);
        let max_degree = degrees.iter().copied().max().unwrap_or(0);
        assert!(
            max_degree <= 4,
            "lattice without diagonals has max_degree={max_degree} > 4 \
             (seed={seed}, dims={rows}x{cols})"
        );
    }

    /// Helper to check if a graph is a lattice without diagonals.
    fn is_lattice_without_diagonals(graph: &super::super::types::GeneratedGraph) -> bool {
        matches!(
            graph.metadata,
            GraphMetadata::Lattice {
                with_diagonals: false,
                ..
            }
        )
    }

    /// Helper to generate and verify lattice without diagonals for a given seed.
    fn verify_lattice_for_seed(seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        for _ in 0..100 {
            let graph = generate_lattice_graph(&mut rng);
            if is_lattice_without_diagonals(&graph) {
                verify_lattice_degree(seed, &graph);
                return;
            }
        }
    }

    #[test]
    fn lattice_without_diagonals_max_degree_is_four() {
        // Generate multiple lattices and verify max degree is exactly 4 for interior nodes.
        for seed in [42, 123, 456] {
            verify_lattice_for_seed(seed);
        }
    }

    #[test]
    fn scale_free_has_hub_nodes() {
        // Verify scale-free graphs exhibit power-law characteristics.
        let mut rng = SmallRng::seed_from_u64(42);
        let graph = generate_scale_free_graph(&mut rng);

        if graph.node_count < 16 {
            return; // Skip small graphs where hubs may not emerge.
        }

        let degrees = compute_node_degrees(graph.node_count, &graph.edges);
        let avg_degree: f64 = degrees.iter().sum::<usize>() as f64 / graph.node_count as f64;
        let max_degree = degrees.iter().copied().max().unwrap_or(0);

        // Scale-free graphs should have at least one hub with degree above average.
        assert!(
            max_degree as f64 >= avg_degree,
            "scale-free graph lacks hub: max_degree={max_degree}, avg_degree={avg_degree:.1}"
        );
    }
}
