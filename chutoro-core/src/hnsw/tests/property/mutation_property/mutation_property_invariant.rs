//! Regression checks for mutation property invariant handling.

use super::*;
use crate::hnsw::graph::NodeContext;
use crate::hnsw::tests::property::support::DenseVectorSource;

#[test]
fn graph_invariant_violation_on_delete_is_reported_as_skipped() {
    let params = HnswParams::new(1, 1).expect("params must be valid");
    let mut index = CpuHnsw::with_capacity(params.clone(), 3).expect("index must allocate");

    // Build a graph where deleting the entry node strands an isolated vertex,
    // forcing delete_node_for_test to surface a GraphInvariantViolation.
    index
        .write_graph(|graph| {
            graph.insert_first(NodeContext {
                node: 0,
                level: 0,
                sequence: 0,
            })?;
            graph.attach_node(NodeContext {
                node: 1,
                level: 0,
                sequence: 1,
            })?;
            graph.attach_node(NodeContext {
                node: 2,
                level: 0,
                sequence: 2,
            })?;
            graph.node_mut(0).expect("node 0").neighbours_mut(0).push(1);
            graph.node_mut(1).expect("node 1").neighbours_mut(0).push(0);
            Ok(())
        })
        .expect("graph construction must succeed");

    let source = DenseVectorSource::new("stub", vec![vec![0.0_f32]]).expect("source");
    let mut pools = MutationPools::new(3);
    pools.mark_inserted(0);
    let mut active_params = params;

    let mut runner = MutationRunner {
        index: &mut index,
        pools: &mut pools,
        source: &source,
        active_params: &mut active_params,
    };

    let outcome = runner
        .apply(&MutationOperationSeed::Delete { slot_hint: 0 })
        .expect("delete should not hard-fail");

    assert!(
        !outcome.applied,
        "GraphInvariantViolation should yield a skipped mutation, not applied"
    );
    assert!(
        outcome.summary.starts_with("delete aborted:"),
        "expected skip summary to start with 'delete aborted:', got {}",
        outcome.summary
    );
}
