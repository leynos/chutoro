//! Unit tests for graph topology property validation.

use rand::{SeedableRng, rngs::SmallRng};
use rstest::rstest;

use crate::hnsw::tests::property::graph_topologies::{
    generate_disconnected_graph, generate_lattice_graph, generate_random_graph,
    generate_scale_free_graph,
};
use crate::hnsw::tests::property::types::{GeneratedGraph, GraphTopology};
use crate::{EdgeHarvest, parallel_kruskal};

use super::{
    GraphFixture, run_disconnected_isolation_property, run_graph_metadata_consistency_property,
    run_graph_mst_compatibility_property, run_graph_validity_property,
    run_lattice_regularity_property, run_scale_free_hub_property,
};

fn make_fixture(
    topology: GraphTopology,
    generate: fn(&mut SmallRng) -> GeneratedGraph,
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
    panic!(
        "failed to generate a scale-free graph with node_count >= 20 after 5 attempts (seed={seed})"
    );
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
#[expect(
    clippy::type_complexity,
    reason = "explicit array type needed for topology/generator pairs"
)]
fn all_topologies_work_with_mst() {
    let seed = 42u64;

    // Test all four topology types.
    let topologies: [(GraphTopology, fn(&mut SmallRng) -> GeneratedGraph); 4] = [
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
