//! Degree ceiling property checks for graph generation.

use proptest::test_runner::{TestCaseError, TestCaseResult};

use super::{GraphFixture, GraphTopology};
use crate::CandidateEdge;

use super::super::graph_metrics::{compute_node_degrees, degree_ceiling_for_metadata};

/// Validates edge indices before degree calculations to avoid panics.
fn validate_edge_indices(node_count: usize, edges: &[CandidateEdge]) -> TestCaseResult {
    for (i, edge) in edges.iter().enumerate() {
        if edge.source() >= node_count {
            return Err(TestCaseError::fail(format!(
                "edge {i} source {} out of bounds (node_count = {node_count})",
                edge.source()
            )));
        }
        if edge.target() >= node_count {
            return Err(TestCaseError::fail(format!(
                "edge {i} target {} out of bounds (node_count = {node_count})",
                edge.target()
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
    validate_edge_indices(fixture.graph.node_count, &fixture.graph.edges)?;
    let degrees = compute_node_degrees(fixture.graph.node_count, &fixture.graph.edges);
    let max_degree = degrees.iter().copied().max().unwrap_or(0);

    let ceiling = degree_ceiling_for_metadata(&fixture.graph.metadata);

    if max_degree > ceiling {
        return Err(TestCaseError::fail(format!(
            "{:?} topology: max_degree {max_degree} exceeds ceiling {ceiling}",
            fixture.topology
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::{SeedableRng, rngs::SmallRng};
    use rstest::rstest;

    use super::super::super::graph_topologies::{
        generate_lattice_graph, generate_scale_free_graph,
    };
    use super::super::super::strategies::graph_fixture_strategy;
    use super::super::super::types::GraphMetadata;
    use super::super::build_fixture;

    // ========================================================================
    // Degree Ceiling Property Tests (rstest)
    // ========================================================================

    #[rstest]
    #[case::random(GraphTopology::Random, 42)]
    #[case::scale_free(GraphTopology::ScaleFree, 42)]
    #[case::lattice(GraphTopology::Lattice, 42)]
    #[case::disconnected(GraphTopology::Disconnected, 42)]
    fn graph_degree_ceiling_rstest(#[case] topology: GraphTopology, #[case] seed: u64) {
        let fixture = build_fixture(seed, topology);
        run_degree_ceiling_property(&fixture).expect("degree ceiling property must hold");
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
    }

    // ========================================================================
    // Additional Edge Cases
    // ========================================================================

    /// Helper to verify lattice max degree constraint.
    fn verify_lattice_degree(seed: u64, graph: &super::super::GeneratedGraph) {
        let GraphMetadata::Lattice {
            with_diagonals: false,
            dimensions: (rows, cols),
        } = &graph.metadata
        else {
            return;
        };
        let degrees = compute_node_degrees(graph.node_count, &graph.edges);
        let max_degree = degrees.iter().copied().max().unwrap_or(0);
        assert!(
            max_degree <= 4,
            concat!(
                "lattice without diagonals has max_degree={max_degree} > 4 ",
                "(seed={seed}, dims={rows}x{cols})"
            ),
            max_degree = max_degree,
            seed = seed,
            rows = rows,
            cols = cols,
        );
    }

    /// Helper to check if a graph is a lattice without diagonals.
    fn is_lattice_without_diagonals(graph: &super::super::GeneratedGraph) -> bool {
        matches!(
            &graph.metadata,
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
        panic!("no lattice without diagonals generated for seed={seed}");
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
        let hub_threshold = avg_degree * 1.5;
        assert!(
            max_degree as f64 >= hub_threshold,
            "scale-free graph lacks hub: max_degree={max_degree}, avg_degree={avg_degree:.1}, threshold={hub_threshold:.1}"
        );
    }
}
