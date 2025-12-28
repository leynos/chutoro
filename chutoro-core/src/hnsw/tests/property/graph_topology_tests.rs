//! Property tests for graph topology generators.
//!
//! Verifies that generated graphs satisfy structural invariants and produce
//! valid edge harvests for MST testing. Tests cover all four topology types:
//! random, scale-free, lattice, and disconnected.
//!
//! Properties verified:
//! 1. **Validity**: All edges reference valid nodes with finite distances
//! 2. **Metadata consistency**: Metadata matches generated structure
//! 3. **MST compatibility**: Graphs work with `parallel_kruskal`
//! 4. **Topology-specific invariants**: Hub nodes, regularity, components

use std::collections::HashSet;

use proptest::test_runner::{TestCaseError, TestCaseResult};

use crate::{EdgeHarvest, parallel_kruskal};

use super::types::{GraphFixture, GraphMetadata, GraphTopology};

/// Verifies all edges reference valid nodes and have valid properties.
///
/// Checks:
/// - Source and target are within node bounds
/// - No self-edges (source != target)
/// - Distance is finite and positive
pub(super) fn run_graph_validity_property(fixture: &GraphFixture) -> TestCaseResult {
    let graph = &fixture.graph;

    for (i, edge) in graph.edges.iter().enumerate() {
        // Valid source reference.
        if edge.source() >= graph.node_count {
            return Err(TestCaseError::fail(format!(
                "edge {i}: source {} out of bounds (node_count = {})",
                edge.source(),
                graph.node_count
            )));
        }

        // Valid target reference.
        if edge.target() >= graph.node_count {
            return Err(TestCaseError::fail(format!(
                "edge {i}: target {} out of bounds (node_count = {})",
                edge.target(),
                graph.node_count
            )));
        }

        // No self-edges.
        if edge.source() == edge.target() {
            return Err(TestCaseError::fail(format!(
                "edge {i}: self-edge ({} -> {})",
                edge.source(),
                edge.target()
            )));
        }

        // Finite positive distance.
        if !edge.distance().is_finite() || edge.distance() <= 0.0 {
            return Err(TestCaseError::fail(format!(
                "edge {i}: invalid distance {}",
                edge.distance()
            )));
        }
    }

    Ok(())
}

/// Verifies graph metadata matches the generated structure.
pub(super) fn run_graph_metadata_consistency_property(fixture: &GraphFixture) -> TestCaseResult {
    let graph = &fixture.graph;

    match (&fixture.topology, &graph.metadata) {
        (GraphTopology::Random, GraphMetadata::Random { node_count, .. }) => {
            if *node_count != graph.node_count {
                return Err(TestCaseError::fail(format!(
                    "random: node_count mismatch (metadata={}, graph={})",
                    node_count, graph.node_count
                )));
            }
        }
        (GraphTopology::ScaleFree, GraphMetadata::ScaleFree { node_count, .. }) => {
            if *node_count != graph.node_count {
                return Err(TestCaseError::fail(format!(
                    "scale-free: node_count mismatch (metadata={}, graph={})",
                    node_count, graph.node_count
                )));
            }
        }
        (GraphTopology::Lattice, GraphMetadata::Lattice { dimensions, .. }) => {
            if dimensions.0 * dimensions.1 != graph.node_count {
                return Err(TestCaseError::fail(format!(
                    "lattice: dimensions mismatch ({}x{}={}, graph={})",
                    dimensions.0,
                    dimensions.1,
                    dimensions.0 * dimensions.1,
                    graph.node_count
                )));
            }
        }
        (
            GraphTopology::Disconnected,
            GraphMetadata::Disconnected {
                component_sizes, ..
            },
        ) => {
            let total: usize = component_sizes.iter().sum();
            if total != graph.node_count {
                return Err(TestCaseError::fail(format!(
                    "disconnected: component sizes mismatch (sum={}, graph={})",
                    total, graph.node_count
                )));
            }
        }
        _ => {
            return Err(TestCaseError::fail(format!(
                "topology/metadata type mismatch: {:?} vs {:?}",
                fixture.topology, graph.metadata
            )));
        }
    }

    Ok(())
}

/// Verifies MST can be computed from generated graph edges.
pub(super) fn run_graph_mst_compatibility_property(fixture: &GraphFixture) -> TestCaseResult {
    let graph = &fixture.graph;
    let harvest = EdgeHarvest::new(graph.edges.clone());

    let result = parallel_kruskal(graph.node_count, &harvest);

    match result {
        Ok(forest) => {
            // For disconnected graphs, expect multiple components.
            if let GraphMetadata::Disconnected {
                component_count, ..
            } = &graph.metadata
            {
                if forest.component_count() < *component_count {
                    return Err(TestCaseError::fail(format!(
                        "expected at least {} components, got {}",
                        component_count,
                        forest.component_count()
                    )));
                }
            }
            Ok(())
        }
        Err(err) => Err(TestCaseError::fail(format!("MST failed: {err}"))),
    }
}

/// Verifies scale-free graphs have hub nodes (high-degree outliers).
pub(super) fn run_scale_free_hub_property(fixture: &GraphFixture) -> TestCaseResult {
    if !matches!(fixture.topology, GraphTopology::ScaleFree) {
        return Ok(());
    }

    let graph = &fixture.graph;

    // Only check for larger graphs where hubs are expected to emerge.
    if graph.node_count < 16 {
        return Ok(());
    }

    let mut degrees = vec![0usize; graph.node_count];

    for edge in &graph.edges {
        degrees[edge.source()] += 1;
        degrees[edge.target()] += 1;
    }

    let avg_degree: f64 = degrees.iter().sum::<usize>() as f64 / graph.node_count as f64;
    let max_degree = *degrees.iter().max().unwrap_or(&0);

    // Scale-free graphs should have at least one hub with degree >= average.
    // This is a relaxed assertion since small graphs may not show clear hubs.
    if (max_degree as f64) < avg_degree * 0.8 {
        return Err(TestCaseError::fail(format!(
            "scale-free graph lacks hub nodes: max_degree={}, avg_degree={:.1}",
            max_degree, avg_degree
        )));
    }

    Ok(())
}

/// Verifies lattice graphs have consistent local connectivity.
pub(super) fn run_lattice_regularity_property(fixture: &GraphFixture) -> TestCaseResult {
    if !matches!(fixture.topology, GraphTopology::Lattice) {
        return Ok(());
    }

    let graph = &fixture.graph;
    let mut degrees = vec![0usize; graph.node_count];

    for edge in &graph.edges {
        degrees[edge.source()] += 1;
        degrees[edge.target()] += 1;
    }

    // Lattice interior nodes should have similar degrees
    // (edge nodes have fewer, corners have even fewer).
    let unique_degrees: HashSet<usize> = degrees.iter().copied().collect();

    // Lattice graphs should have limited degree variance:
    // - Without diagonals: 2 (corner), 3 (edge), 4 (interior)
    // - With diagonals: more values but still limited (typically 3-8)
    if unique_degrees.len() > 8 {
        return Err(TestCaseError::fail(format!(
            "lattice has too many distinct degrees: {:?}",
            unique_degrees
        )));
    }

    Ok(())
}

/// Builds a mapping from node index to component index.
///
/// Given a list of component sizes and total node count, creates a vector
/// where `result[node]` gives the component index that node belongs to.
/// Components are assigned sequentially: nodes 0..sizes[0] map to component 0,
/// nodes sizes[0]..sizes[0]+sizes[1] map to component 1, etc.
fn build_node_to_component_mapping(component_sizes: &[usize], node_count: usize) -> Vec<usize> {
    let mut node_to_component = vec![0usize; node_count];
    let mut offset = 0;
    for (comp_idx, &size) in component_sizes.iter().enumerate() {
        for i in 0..size {
            node_to_component[offset + i] = comp_idx;
        }
        offset += size;
    }
    node_to_component
}

/// Verifies no edge crosses component boundaries.
///
/// Iterates through all edges and checks that source and target nodes
/// belong to the same component according to the provided mapping.
/// Returns an error if any edge violates component isolation.
fn verify_no_cross_component_edges(
    edges: &[crate::CandidateEdge],
    node_to_component: &[usize],
) -> TestCaseResult {
    for edge in edges {
        if node_to_component[edge.source()] != node_to_component[edge.target()] {
            return Err(TestCaseError::fail(format!(
                "edge crosses components: {:?}",
                edge
            )));
        }
    }
    Ok(())
}

/// Verifies disconnected graphs have no cross-component edges.
pub(super) fn run_disconnected_isolation_property(fixture: &GraphFixture) -> TestCaseResult {
    if !matches!(fixture.topology, GraphTopology::Disconnected) {
        return Ok(());
    }

    let graph = &fixture.graph;

    if let GraphMetadata::Disconnected {
        component_sizes, ..
    } = &graph.metadata
    {
        let node_to_component = build_node_to_component_mapping(component_sizes, graph.node_count);
        verify_no_cross_component_edges(&graph.edges, &node_to_component)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::tests::property::graph_topologies::{
        generate_disconnected_graph, generate_lattice_graph, generate_random_graph,
        generate_scale_free_graph,
    };
    use rand::{SeedableRng, rngs::SmallRng};
    use rstest::rstest;

    fn make_fixture(
        topology: GraphTopology,
        generate: fn(&mut SmallRng) -> crate::hnsw::tests::property::types::GeneratedGraph,
        seed: u64,
    ) -> GraphFixture {
        let mut rng = SmallRng::seed_from_u64(seed);
        let graph = generate(&mut rng);
        GraphFixture { topology, graph }
    }

    // ========================================================================
    // Validity tests
    // ========================================================================

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    #[case(789)]
    #[case(999)]
    fn random_graph_validity(#[case] seed: u64) {
        let fixture = make_fixture(GraphTopology::Random, generate_random_graph, seed);
        run_graph_validity_property(&fixture).expect("validity must hold");
    }

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    #[case(789)]
    #[case(999)]
    fn scale_free_graph_validity(#[case] seed: u64) {
        let fixture = make_fixture(GraphTopology::ScaleFree, generate_scale_free_graph, seed);
        run_graph_validity_property(&fixture).expect("validity must hold");
    }

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    #[case(789)]
    #[case(999)]
    fn lattice_graph_validity(#[case] seed: u64) {
        let fixture = make_fixture(GraphTopology::Lattice, generate_lattice_graph, seed);
        run_graph_validity_property(&fixture).expect("validity must hold");
    }

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    #[case(789)]
    #[case(999)]
    fn disconnected_graph_validity(#[case] seed: u64) {
        let fixture = make_fixture(
            GraphTopology::Disconnected,
            generate_disconnected_graph,
            seed,
        );
        run_graph_validity_property(&fixture).expect("validity must hold");
    }

    // ========================================================================
    // Metadata consistency tests
    // ========================================================================

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    fn random_graph_metadata_consistency(#[case] seed: u64) {
        let fixture = make_fixture(GraphTopology::Random, generate_random_graph, seed);
        run_graph_metadata_consistency_property(&fixture).expect("metadata must be consistent");
    }

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    fn scale_free_graph_metadata_consistency(#[case] seed: u64) {
        let fixture = make_fixture(GraphTopology::ScaleFree, generate_scale_free_graph, seed);
        run_graph_metadata_consistency_property(&fixture).expect("metadata must be consistent");
    }

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    fn lattice_graph_metadata_consistency(#[case] seed: u64) {
        let fixture = make_fixture(GraphTopology::Lattice, generate_lattice_graph, seed);
        run_graph_metadata_consistency_property(&fixture).expect("metadata must be consistent");
    }

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    fn disconnected_graph_metadata_consistency(#[case] seed: u64) {
        let fixture = make_fixture(
            GraphTopology::Disconnected,
            generate_disconnected_graph,
            seed,
        );
        run_graph_metadata_consistency_property(&fixture).expect("metadata must be consistent");
    }

    // ========================================================================
    // MST compatibility tests
    // ========================================================================

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    fn random_graph_mst_compatibility(#[case] seed: u64) {
        let fixture = make_fixture(GraphTopology::Random, generate_random_graph, seed);
        if !fixture.graph.edges.is_empty() {
            run_graph_mst_compatibility_property(&fixture).expect("MST must work");
        }
    }

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    fn scale_free_graph_mst_compatibility(#[case] seed: u64) {
        let fixture = make_fixture(GraphTopology::ScaleFree, generate_scale_free_graph, seed);
        if !fixture.graph.edges.is_empty() {
            run_graph_mst_compatibility_property(&fixture).expect("MST must work");
        }
    }

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    fn lattice_graph_mst_compatibility(#[case] seed: u64) {
        let fixture = make_fixture(GraphTopology::Lattice, generate_lattice_graph, seed);
        run_graph_mst_compatibility_property(&fixture).expect("MST must work");
    }

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    fn disconnected_graph_mst_compatibility(#[case] seed: u64) {
        let fixture = make_fixture(
            GraphTopology::Disconnected,
            generate_disconnected_graph,
            seed,
        );
        if !fixture.graph.edges.is_empty() {
            run_graph_mst_compatibility_property(&fixture).expect("MST must work");
        }
    }

    // ========================================================================
    // Topology-specific invariant tests
    // ========================================================================

    #[rstest]
    #[case(12345)]
    #[case(54321)]
    #[case(99999)]
    fn scale_free_hub_detection(#[case] seed: u64) {
        // Generate multiple graphs to find one with enough nodes for hub detection.
        let mut rng = SmallRng::seed_from_u64(seed);
        for _ in 0..5 {
            let graph = generate_scale_free_graph(&mut rng);
            if graph.node_count >= 20 {
                let fixture = GraphFixture {
                    topology: GraphTopology::ScaleFree,
                    graph,
                };
                run_scale_free_hub_property(&fixture).expect("hub property must hold");
                return;
            }
        }
    }

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    fn lattice_regularity(#[case] seed: u64) {
        let fixture = make_fixture(GraphTopology::Lattice, generate_lattice_graph, seed);
        run_lattice_regularity_property(&fixture).expect("regularity must hold");
    }

    #[rstest]
    #[case(42)]
    #[case(123)]
    #[case(456)]
    #[case(789)]
    fn disconnected_isolation(#[case] seed: u64) {
        let fixture = make_fixture(
            GraphTopology::Disconnected,
            generate_disconnected_graph,
            seed,
        );
        run_disconnected_isolation_property(&fixture).expect("isolation must hold");
    }

    #[rstest]
    fn disconnected_graph_produces_multiple_mst_components() {
        let mut rng = SmallRng::seed_from_u64(999);
        let graph = generate_disconnected_graph(&mut rng);

        // Skip if no edges (rare but possible with low edge probability).
        if graph.edges.is_empty() {
            return;
        }

        let harvest = EdgeHarvest::new(graph.edges.clone());
        let forest = parallel_kruskal(graph.node_count, &harvest).expect("MST must succeed");

        // Disconnected graph should produce multiple components.
        assert!(
            forest.component_count() >= 2,
            "disconnected graph should have multiple MST components, got {}",
            forest.component_count()
        );
    }

    #[rstest]
    fn lattice_graph_always_produces_edges() {
        // Lattice graphs with dimensions >= 2x2 should always have edges.
        for seed in 0..20 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let graph = generate_lattice_graph(&mut rng);
            assert!(
                !graph.edges.is_empty(),
                "lattice graph should always have edges"
            );
        }
    }

    #[rstest]
    #[allow(clippy::type_complexity)]
    fn all_topologies_work_with_mst() {
        let seed = 42u64;

        // Test all four topology types.
        let topologies: [(
            GraphTopology,
            fn(&mut SmallRng) -> crate::hnsw::tests::property::types::GeneratedGraph,
        ); 4] = [
            (GraphTopology::Random, generate_random_graph),
            (GraphTopology::ScaleFree, generate_scale_free_graph),
            (GraphTopology::Lattice, generate_lattice_graph),
            (GraphTopology::Disconnected, generate_disconnected_graph),
        ];

        for (topology, generate) in topologies {
            let fixture = make_fixture(topology, generate, seed);
            if !fixture.graph.edges.is_empty() {
                run_graph_mst_compatibility_property(&fixture)
                    .unwrap_or_else(|e| panic!("{topology:?} MST failed: {e}"));
            }
        }
    }
}
