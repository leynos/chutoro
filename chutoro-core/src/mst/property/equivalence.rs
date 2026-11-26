//! Property 1: Equivalence with sequential oracle.
//!
//! For any generated input graph, verifies that the parallel Kruskal
//! implementation produces exactly the forest a trusted sequential Kruskal
//! oracle produces: the same edges, in the same order, with bit-identical
//! weights and the same sequence numbers. The total weight, edge count and
//! component count are checked first because their failure messages are
//! easier to read than an edge-list diff.
//!
//! See `docs/property-testing-design.md` §4.3.1.

use proptest::test_runner::{TestCaseError, TestCaseResult};

use crate::{EdgeHarvest, MstEdge, parallel_kruskal};

use super::helpers::total_weight_f64;
use super::oracle::{sequential_kruskal, sequential_kruskal_edges};
use super::types::MstFixture;

/// Runs the oracle equivalence property for the given fixture.
///
/// Compares the parallel Kruskal output against the sequential oracle.
/// Total weights are accumulated as `f64` to avoid order-dependent `f32`
/// rounding differences. The final check compares the exact edge lists, with
/// weights compared by their bit patterns rather than with `==`.
pub(super) fn run_oracle_equivalence_property(fixture: &MstFixture) -> TestCaseResult {
    let harvest = EdgeHarvest::new(fixture.edges.clone());

    let parallel_result = parallel_kruskal(fixture.node_count, &harvest).map_err(|e| {
        TestCaseError::fail(format!(
            "parallel_kruskal failed: {e} (distribution={:?}, nodes={}, edges={})",
            fixture.distribution,
            fixture.node_count,
            fixture.edges.len(),
        ))
    })?;

    let oracle = sequential_kruskal(fixture.node_count, &fixture.edges);

    let parallel_weight = total_weight_f64(parallel_result.edges());

    if !parallel_weight.total_cmp(&oracle.total_weight).is_eq() {
        return Err(TestCaseError::fail(format!(
            "total weight mismatch: parallel={parallel_weight}, oracle={} \
             (distribution={:?}, nodes={}, edges={})",
            oracle.total_weight,
            fixture.distribution,
            fixture.node_count,
            fixture.edges.len(),
        )));
    }

    if parallel_result.edges().len() != oracle.edge_count {
        return Err(TestCaseError::fail(format!(
            "edge count mismatch: parallel={}, oracle={} \
             (distribution={:?}, nodes={}, edges={})",
            parallel_result.edges().len(),
            oracle.edge_count,
            fixture.distribution,
            fixture.node_count,
            fixture.edges.len(),
        )));
    }

    if parallel_result.component_count() != oracle.component_count {
        return Err(TestCaseError::fail(format!(
            "component count mismatch: parallel={}, oracle={} \
             (distribution={:?}, nodes={}, edges={})",
            parallel_result.component_count(),
            oracle.component_count,
            fixture.distribution,
            fixture.node_count,
            fixture.edges.len(),
        )));
    }

    let parallel_edges: Vec<_> = parallel_result.edges().iter().map(edge_identity).collect();
    let oracle_edges: Vec<_> = sequential_kruskal_edges(fixture.node_count, &fixture.edges)
        .into_iter()
        .map(|(source, target, weight, sequence)| (source, target, weight.to_bits(), sequence))
        .collect();
    if parallel_edges != oracle_edges {
        return Err(TestCaseError::fail(format!(
            "edge list mismatch: parallel={parallel_edges:?}, oracle={oracle_edges:?} \
             (distribution={:?}, nodes={})",
            fixture.distribution, fixture.node_count,
        )));
    }

    Ok(())
}

/// Returns an edge's `(source, target, weight bits, sequence)` identity.
const fn edge_identity(edge: &MstEdge) -> (usize, usize, u32, u64) {
    (
        edge.source(),
        edge.target(),
        edge.weight().to_bits(),
        edge.sequence(),
    )
}
