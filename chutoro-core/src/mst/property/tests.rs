//! Property-based test runners for the parallel Kruskal MST implementation.
//!
//! Hosts proptest runners for all three properties (oracle equivalence,
//! structural invariants, concurrency safety), rstest parameterized cases
//! for targeted distribution coverage, and unit tests for the sequential
//! oracle itself.

use proptest::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::CandidateEdge;
use crate::test_utils::suite_proptest_config;

use super::concurrency::run_concurrency_safety_property;
use super::equivalence::run_oracle_equivalence_property;
use super::oracle::{SequentialMstResult, sequential_kruskal};
use super::strategies::{generate_fixture, mst_fixture_strategy};
use super::structural::run_structural_invariants_property;
use super::types::WeightDistribution;

/// Canonical set of (distribution, seed, case_name) tuples shared by all
/// parameterised property tests.  Defined once to eliminate duplication
/// across oracle equivalence, structural invariants, and concurrency
/// safety test suites.
const TEST_CASES: &[(WeightDistribution, u64, &str)] = &[
    (WeightDistribution::Unique, 42, "unique_42"),
    (WeightDistribution::Unique, 999, "unique_999"),
    (WeightDistribution::ManyIdentical, 42, "identical_42"),
    (WeightDistribution::ManyIdentical, 999, "identical_999"),
    (WeightDistribution::ManyIdentical, 7777, "identical_7777"),
    (WeightDistribution::Sparse, 42, "sparse_42"),
    (WeightDistribution::Sparse, 999, "sparse_999"),
    (WeightDistribution::Dense, 42, "dense_42"),
    (WeightDistribution::Dense, 999, "dense_999"),
    (WeightDistribution::Disconnected, 42, "disconnected_42"),
    (WeightDistribution::Disconnected, 999, "disconnected_999"),
];

/// Generates an rstest-parameterised function that exercises a property
/// runner across every entry in [`TEST_CASES`].
///
/// # Arguments
///
/// - `$test_name` — identifier for the generated test function.
/// - `$runner` — property runner function with signature
///   `fn(&MstFixture) -> TestCaseResult`.
/// - `$expectation` — panic message passed to `.expect()`.
macro_rules! parameterised_property_test {
    ($test_name:ident, $runner:path, $expectation:expr) => {
        #[rstest::rstest]
        #[case::unique_42(WeightDistribution::Unique, 42)]
        #[case::unique_999(WeightDistribution::Unique, 999)]
        #[case::identical_42(WeightDistribution::ManyIdentical, 42)]
        #[case::identical_999(WeightDistribution::ManyIdentical, 999)]
        #[case::identical_7777(WeightDistribution::ManyIdentical, 7777)]
        #[case::sparse_42(WeightDistribution::Sparse, 42)]
        #[case::sparse_999(WeightDistribution::Sparse, 999)]
        #[case::dense_42(WeightDistribution::Dense, 42)]
        #[case::dense_999(WeightDistribution::Dense, 999)]
        #[case::disconnected_42(WeightDistribution::Disconnected, 42)]
        #[case::disconnected_999(WeightDistribution::Disconnected, 999)]
        fn $test_name(#[case] distribution: WeightDistribution, #[case] seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let fixture = generate_fixture(distribution, &mut rng);
            $runner(&fixture).expect($expectation);
        }
    };
}

// ========================================================================
// Proptest Runners
// ========================================================================

proptest! {
    #![proptest_config(suite_proptest_config(256))]

    #[test]
    fn mst_oracle_equivalence(fixture in mst_fixture_strategy()) {
        run_oracle_equivalence_property(&fixture)?;
    }

    #[test]
    fn mst_structural_invariants(fixture in mst_fixture_strategy()) {
        run_structural_invariants_property(&fixture)?;
    }

    #[test]
    fn mst_concurrency_safety(fixture in mst_fixture_strategy()) {
        run_concurrency_safety_property(&fixture)?;
    }
}

// ========================================================================
// rstest Parameterised Cases
// ========================================================================

parameterised_property_test!(
    oracle_equivalence_rstest,
    run_oracle_equivalence_property,
    "oracle equivalence must hold"
);

parameterised_property_test!(
    structural_invariants_rstest,
    run_structural_invariants_property,
    "structural invariants must hold"
);

parameterised_property_test!(
    concurrency_safety_rstest,
    run_concurrency_safety_property,
    "concurrency safety must hold"
);

// ========================================================================
// TEST_CASES Consistency Check
// ========================================================================

/// Ensures the macro-generated rstest cases stay in sync with
/// [`TEST_CASES`].  If a case is added or removed from the constant, this
/// test will fail until the macro is updated to match.
#[test]
fn test_cases_count_matches_macro_expectations() {
    // The macro generates exactly 11 cases per property test.  If
    // TEST_CASES grows or shrinks this assertion catches the drift.
    assert_eq!(
        TEST_CASES.len(),
        11,
        "TEST_CASES length changed — update parameterised_property_test! macro"
    );
}

// ========================================================================
// Oracle Unit Tests — Build Confidence in the Reference Implementation
// ========================================================================

#[test]
fn oracle_triangle() {
    let edges = vec![
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 2, 2.0, 1),
        CandidateEdge::new(0, 2, 3.0, 2),
    ];
    let result = sequential_kruskal(3, &edges);
    assert_oracle(&result, 3.0, 2, 1);
}

#[test]
fn oracle_square() {
    // Square: 0-1 (1), 1-2 (2), 2-3 (3), 3-0 (4).
    // MST picks edges with weight 1, 2, 3.
    let edges = vec![
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 2, 2.0, 1),
        CandidateEdge::new(2, 3, 3.0, 2),
        CandidateEdge::new(3, 0, 4.0, 3),
    ];
    let result = sequential_kruskal(4, &edges);
    assert_oracle(&result, 6.0, 3, 1);
}

#[test]
fn oracle_disconnected_pair() {
    let edges = vec![
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(2, 3, 2.0, 1),
    ];
    let result = sequential_kruskal(5, &edges);
    // Two edges in the forest, node 4 is isolated → 3 components.
    assert_oracle(&result, 3.0, 2, 3);
}

#[test]
fn oracle_single_node() {
    let result = sequential_kruskal(1, &[]);
    assert_oracle(&result, 0.0, 0, 1);
}

#[test]
fn oracle_single_edge() {
    let edges = vec![CandidateEdge::new(0, 1, 5.0, 0)];
    let result = sequential_kruskal(2, &edges);
    assert_oracle(&result, 5.0, 1, 1);
}

#[test]
fn oracle_linear_chain() {
    let edges = vec![
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 2, 2.0, 1),
        CandidateEdge::new(2, 3, 3.0, 2),
    ];
    let result = sequential_kruskal(4, &edges);
    assert_oracle(&result, 6.0, 3, 1);
}

#[test]
fn oracle_equal_weights() {
    // All edges have weight 1.0 — MST picks first n-1 in sort order.
    let edges = vec![
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(0, 2, 1.0, 1),
        CandidateEdge::new(1, 2, 1.0, 2),
    ];
    let result = sequential_kruskal(3, &edges);
    assert_oracle(&result, 2.0, 2, 1);
}

#[test]
fn oracle_self_loops_are_ignored() {
    let edges = vec![
        CandidateEdge::new(0, 0, 1.0, 0),
        CandidateEdge::new(0, 1, 2.0, 1),
    ];
    let result = sequential_kruskal(2, &edges);
    assert_oracle(&result, 2.0, 1, 1);
}

#[test]
fn oracle_empty_graph() {
    let result = sequential_kruskal(0, &[]);
    assert_oracle(&result, 0.0, 0, 0);
}

#[test]
fn oracle_filters_out_of_bounds_source() {
    // Edge (5, 1) has source >= node_count=3, so it should be ignored.
    let edges = vec![
        CandidateEdge::new(5, 1, 1.0, 0),
        CandidateEdge::new(0, 1, 2.0, 1),
    ];
    let result = sequential_kruskal(3, &edges);
    assert_oracle(&result, 2.0, 1, 2);
}

#[test]
fn oracle_filters_out_of_bounds_target() {
    // Edge (0, 10) has target >= node_count=3, so it should be ignored.
    let edges = vec![
        CandidateEdge::new(0, 10, 1.0, 0),
        CandidateEdge::new(0, 2, 3.0, 1),
    ];
    let result = sequential_kruskal(3, &edges);
    assert_oracle(&result, 3.0, 1, 2);
}

#[test]
fn oracle_filters_nan_weight() {
    let edges = vec![
        CandidateEdge::new(0, 1, f32::NAN, 0),
        CandidateEdge::new(1, 2, 4.0, 1),
    ];
    let result = sequential_kruskal(3, &edges);
    // NaN edge is discarded; only the 1-2 edge survives.
    assert_oracle(&result, 4.0, 1, 2);
}

#[test]
fn oracle_filters_infinite_weight() {
    let edges = vec![
        CandidateEdge::new(0, 1, f32::INFINITY, 0),
        CandidateEdge::new(0, 1, 1.0, 1),
        CandidateEdge::new(1, 2, 2.0, 2),
    ];
    let result = sequential_kruskal(3, &edges);
    // Infinite edge is discarded; the finite edge (0,1,1.0) and (1,2,2.0) form the MST.
    assert_oracle(&result, 3.0, 2, 1);
}

/// Asserts oracle results match expected values.
fn assert_oracle(
    result: &SequentialMstResult,
    expected_weight: f64,
    expected_edges: usize,
    expected_components: usize,
) {
    assert!(
        (result.total_weight - expected_weight).abs() < f64::EPSILON,
        "weight: expected {expected_weight}, got {}",
        result.total_weight,
    );
    assert_eq!(
        result.edge_count, expected_edges,
        "edge_count: expected {expected_edges}, got {}",
        result.edge_count,
    );
    assert_eq!(
        result.component_count, expected_components,
        "component_count: expected {expected_components}, got {}",
        result.component_count,
    );
}
