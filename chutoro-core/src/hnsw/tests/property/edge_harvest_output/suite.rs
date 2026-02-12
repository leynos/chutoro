//! Harvested-output property checks for candidate edge harvesting.

use proptest::test_runner::{TestCaseError, TestCaseResult, TestRunner};

use super::super::graph_metrics::{
    compute_node_degrees, compute_rnn_score, count_connected_components,
    degree_ceiling_for_metadata, median,
};
use super::super::graph_topology_tests::validate_edge;
use super::super::strategies::graph_fixture_strategy_for_topology;
use super::super::types::{GraphFixture, GraphMetadata, GraphTopology};
use super::harvest::{harvest_candidate_edges, harvest_k_for_metadata};
use super::{CONNECTIVITY_PRESERVATION_THRESHOLD, HARVEST_CASES_PER_TOPOLOGY};
use crate::test_utils::suite_proptest_config;

/// Captures per-fixture metrics for harvested-output property checks.
#[derive(Clone, Copy, Debug)]
pub(super) struct HarvestedMetrics {
    /// Number of connected components in the input graph.
    pub(super) input_components: usize,
    /// Number of connected components in the harvested output graph.
    pub(super) output_components: usize,
    /// Output RNN score minus input RNN score.
    pub(super) rnn_delta: f64,
}

fn validate_harvested_edges(node_count: usize, edges: &[crate::CandidateEdge]) -> TestCaseResult {
    for (i, edge) in edges.iter().enumerate() {
        validate_edge(edge, node_count, i)?;
    }
    Ok(())
}

fn validate_degree_constraints(
    node_count: usize,
    edges: &[crate::CandidateEdge],
    metadata: &GraphMetadata,
) -> TestCaseResult {
    let degrees = compute_node_degrees(node_count, edges);
    let max_degree = degrees.iter().copied().max().unwrap_or(0);
    let ceiling = degree_ceiling_for_metadata(metadata);

    if max_degree > ceiling {
        return Err(TestCaseError::fail(format!(
            "max_degree {max_degree} exceeds ceiling {ceiling}"
        )));
    }

    Ok(())
}

fn validate_component_delta(input_components: usize, output_components: usize) -> TestCaseResult {
    if output_components > input_components + 1 {
        return Err(TestCaseError::fail(format!(
            "output components {output_components} exceed input {input_components} by more than 1"
        )));
    }
    Ok(())
}

/// Evaluates harvested output metrics for a single fixture.
pub(super) fn evaluate_harvested_output(
    fixture: &GraphFixture,
) -> Result<HarvestedMetrics, TestCaseError> {
    let harvested = harvest_candidate_edges(fixture).map_err(TestCaseError::fail)?;

    validate_harvested_edges(fixture.graph.node_count, &harvested)?;
    validate_degree_constraints(
        fixture.graph.node_count,
        &harvested,
        &fixture.graph.metadata,
    )?;

    let input_components =
        count_connected_components(fixture.graph.node_count, &fixture.graph.edges);
    let output_components = count_connected_components(fixture.graph.node_count, &harvested);
    validate_component_delta(input_components, output_components)?;

    let k = harvest_k_for_metadata(&fixture.graph.metadata)
        .min(fixture.graph.node_count.saturating_sub(1));
    let input_rnn = compute_rnn_score(fixture.graph.node_count, &fixture.graph.edges, k);
    let output_rnn = compute_rnn_score(fixture.graph.node_count, &harvested, k);
    let rnn_delta = output_rnn - input_rnn;

    Ok(HarvestedMetrics {
        input_components,
        output_components,
        rnn_delta,
    })
}

fn min_rnn_delta_for_topology(topology: GraphTopology) -> f64 {
    match topology {
        GraphTopology::ScaleFree => 0.0,
        GraphTopology::Lattice | GraphTopology::Random | GraphTopology::Disconnected => 0.05,
    }
}

/// Runs the harvested-output suite for a specific topology.
pub(super) fn run_harvested_output_suite_for_topology(topology: GraphTopology) -> TestCaseResult {
    let config = suite_proptest_config(HARVEST_CASES_PER_TOPOLOGY);
    let cases = config.cases as usize;
    let mut runner = TestRunner::new(config);
    let strategy = graph_fixture_strategy_for_topology(topology);

    let metrics = std::cell::RefCell::new(Vec::with_capacity(cases));
    runner.run(&strategy, |fixture| {
        let case_metrics = evaluate_harvested_output(&fixture)?;
        metrics.borrow_mut().push(case_metrics);
        Ok(())
    })?;

    let metrics = metrics.into_inner();
    if metrics.len() != cases {
        return Err(TestCaseError::fail(format!(
            "{topology:?} expected {cases} cases, got {}",
            metrics.len()
        )));
    }

    let mut deltas: Vec<f64> = metrics.iter().map(|m| m.rnn_delta).collect();
    let median_delta = median(&mut deltas);
    let min_delta = min_rnn_delta_for_topology(topology);
    if median_delta < min_delta {
        return Err(TestCaseError::fail(format!(
            "{topology:?} median RNN delta {median_delta:.3} below minimum {min_delta:.3}"
        )));
    }

    let connected_cases: Vec<&HarvestedMetrics> =
        metrics.iter().filter(|m| m.input_components == 1).collect();
    if !connected_cases.is_empty() {
        let preserved = connected_cases
            .iter()
            .filter(|m| m.output_components == 1)
            .count();
        let ratio = preserved as f64 / connected_cases.len() as f64;
        if ratio < CONNECTIVITY_PRESERVATION_THRESHOLD {
            return Err(TestCaseError::fail(format!(
                "{topology:?} connectivity preserved in {:.1}% ({} / {}), below {:.1}%",
                ratio * 100.0,
                preserved,
                connected_cases.len(),
                CONNECTIVITY_PRESERVATION_THRESHOLD * 100.0
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::graph_topologies;
    use super::super::super::types::GraphMetadata;
    use super::*;
    use crate::CandidateEdge;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rstest::rstest;

    // ========================================================================
    // Harvested Output Property Tests (rstest)
    // ========================================================================

    #[rstest]
    #[case::random(GraphTopology::Random, 42)]
    #[case::scale_free(GraphTopology::ScaleFree, 42)]
    #[case::lattice(GraphTopology::Lattice, 42)]
    #[case::disconnected(GraphTopology::Disconnected, 42)]
    fn harvested_output_rstest_cases(#[case] topology: GraphTopology, #[case] seed: u64) {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let graph = graph_topologies::generate_graph_for_topology(topology, &mut rng);
        let fixture = GraphFixture { topology, graph };
        evaluate_harvested_output(&fixture).expect("harvested output must satisfy invariants");
    }

    // ========================================================================
    // Helper Function Unit Tests (happy and unhappy paths)
    // ========================================================================

    #[test]
    fn harvested_edge_validation_rejects_self_loop() {
        let edges = vec![CandidateEdge::new(0, 0, 1.0, 0)];
        let result = validate_harvested_edges(1, &edges);
        assert!(result.is_err());
    }

    #[test]
    fn harvested_degree_constraints_reject_overflow() {
        let edges = vec![
            CandidateEdge::new(0, 1, 1.0, 0),
            CandidateEdge::new(0, 2, 1.0, 1),
            CandidateEdge::new(0, 3, 1.0, 2),
            CandidateEdge::new(0, 4, 1.0, 3),
            CandidateEdge::new(0, 5, 1.0, 4),
        ];
        let metadata = GraphMetadata::Lattice {
            dimensions: (2, 3),
            with_diagonals: false,
        };
        let result = validate_degree_constraints(6, &edges, &metadata);
        assert!(result.is_err());
    }

    #[test]
    fn harvested_component_delta_rejects_large_increase() {
        let result = validate_component_delta(1, 3);
        assert!(result.is_err());
    }

    // ========================================================================
    // Proptest Coverage (256 cases per topology)
    // ========================================================================

    #[test]
    fn harvested_output_random_proptest() -> TestCaseResult {
        run_harvested_output_suite_for_topology(GraphTopology::Random)
    }

    #[test]
    fn harvested_output_scale_free_proptest() -> TestCaseResult {
        run_harvested_output_suite_for_topology(GraphTopology::ScaleFree)
    }

    #[test]
    fn harvested_output_lattice_proptest() -> TestCaseResult {
        run_harvested_output_suite_for_topology(GraphTopology::Lattice)
    }

    #[test]
    fn harvested_output_disconnected_proptest() -> TestCaseResult {
        run_harvested_output_suite_for_topology(GraphTopology::Disconnected)
    }

    // ========================================================================
    // Additional edge cases
    // ========================================================================

    proptest! {
        #[test]
        fn harvested_output_handles_small_random_graph(seed in any::<u64>()) {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let graph = graph_topologies::generate_random_graph(&mut rng);
            let fixture = GraphFixture { topology: GraphTopology::Random, graph };
            evaluate_harvested_output(&fixture)?;
        }
    }
}
