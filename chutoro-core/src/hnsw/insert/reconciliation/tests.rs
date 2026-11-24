//! Tests reverse-edge removal and self-target rejection during reconciliation.

use super::{DeferredScrub, EdgeReconciler, UpdateContext};
use crate::hnsw::{
    error::HnswError,
    graph::{Graph, NodeContext},
    insert::test_helpers::{add_edge_if_missing, assert_bidirectional_edge},
    params::HnswParams,
};
use rstest::rstest;

/// Complete reconciler state used to prove rejected calls leave no trace.
type ReconciliationSnapshot = (Vec<(usize, Vec<Vec<usize>>)>, Vec<(usize, usize, usize)>);

/// Failure to construct a reconciliation fixture.
#[derive(Debug, thiserror::Error)]
enum FixtureError {
    /// The graph constructor rejected a fixture node.
    #[error("graph construction failed: {0}")]
    Graph(#[from] HnswError),
    /// Node 0 was absent from the constructed graph.
    #[error("origin node 0 is missing")]
    MissingOrigin,
    /// Node 0 did not expose the requested level.
    #[error("origin node 0 does not expose level {0}")]
    MissingOriginLevel(usize),
}

fn graph_with_nodes(level: usize, node_count: usize) -> Result<Graph, HnswError> {
    let params = HnswParams::new(2, 4)?;
    let mut graph = Graph::with_capacity(params, node_count);
    graph.insert_first(NodeContext {
        node: 0,
        level,
        sequence: 0,
    })?;
    for (node, sequence) in (1..node_count).zip(1_u64..) {
        graph.attach_node(NodeContext {
            node,
            level,
            sequence,
        })?;
    }
    Ok(graph)
}

fn update_context(level: usize) -> UpdateContext {
    UpdateContext {
        origin: 0,
        level,
        max_connections: 2,
    }
}

/// Builds the three-node self-target fixture and its seeded scrub queue.
fn self_target_fixture(
    neighbours: &[usize],
    seeded_scrubs: &[(usize, usize, usize)],
) -> Result<(Graph, Vec<DeferredScrub>), FixtureError> {
    let level = 1;
    let mut graph = graph_with_nodes(level, 3)?;
    let origin = graph.node_mut(0).ok_or(FixtureError::MissingOrigin)?;
    let origin_neighbours = origin
        .neighbours_mut(level)
        .ok_or(FixtureError::MissingOriginLevel(level))?;
    origin_neighbours.extend_from_slice(neighbours);

    let scrubs = seeded_scrubs
        .iter()
        .map(|&(scrub_origin, scrub_target, scrub_level)| DeferredScrub {
            origin: scrub_origin,
            target: scrub_target,
            level: scrub_level,
        })
        .collect();
    Ok((graph, scrubs))
}

/// Captures every node's neighbour lists and the deferred scrub queue.
fn snapshot_state(graph: &Graph, scrubs: &[DeferredScrub]) -> ReconciliationSnapshot {
    let node_lists = graph
        .nodes_iter()
        .map(|(node_id, node)| {
            let levels = (0..node.level_count())
                .map(|level| node.neighbours(level).to_vec())
                .collect();
            (node_id, levels)
        })
        .collect();
    let queue = scrubs
        .iter()
        .map(|scrub| (scrub.origin, scrub.target, scrub.level))
        .collect();
    (node_lists, queue)
}

#[rstest]
#[case::base_layer(0, true)]
#[case::upper_layer(1, false)]
fn remove_reverse_edge_reports_isolation_only_at_the_base_layer(
    #[case] level: usize,
    #[case] is_isolated: bool,
) {
    let mut graph = graph_with_nodes(level, 2).expect("test graph must be valid");
    graph
        .node_mut(1)
        .expect("target node must be present")
        .neighbours_mut(level)
        .expect("target must expose the level")
        .push(0);

    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert_eq!(
        reconciler.remove_reverse_edge(&update_context(level), 1),
        is_isolated,
    );
    assert_eq!(
        reconciler
            .graph
            .node(1)
            .expect("target node must be present")
            .neighbours(level),
        [] as [usize; 0]
    );
}

#[test]
fn remove_reverse_edge_returns_false_when_the_reverse_edge_is_missing() {
    let mut graph = graph_with_nodes(0, 2).expect("test graph must be valid");
    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert!(!reconciler.remove_reverse_edge(&update_context(0), 1));
}

#[test]
fn remove_reverse_edge_returns_false_when_target_or_level_is_missing() {
    let mut graph = graph_with_nodes(0, 2).expect("test graph must be valid");
    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert!(!reconciler.remove_reverse_edge(&update_context(0), 2));
    assert!(!reconciler.remove_reverse_edge(&update_context(1), 1));
}

#[rstest]
#[case(0)]
#[case(1)]
fn ensure_reverse_edge_links_a_valid_target_back_to_the_origin(#[case] level: usize) {
    let mut graph = graph_with_nodes(level, 2).expect("test graph must be valid");
    add_edge_if_missing(&mut graph, 0, 1, level);

    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert!(reconciler.ensure_reverse_edge(&update_context(level), 1));
    assert_bidirectional_edge!(reconciler.graph, 0, 1, level);
    assert_eq!(
        reconciler
            .graph
            .node(1)
            .expect("target node must be present")
            .neighbours(level),
        [0].as_slice(),
    );
}

#[cfg(debug_assertions)]
#[rstest]
#[case::spare_capacity(&[1], &[])]
#[case::at_capacity(&[1, 2], &[(2, 1, 1)])]
#[case::existing_self_loop(&[0, 1], &[])]
fn ensure_reverse_edge_asserts_on_a_self_target(
    #[case] neighbours: &[usize],
    #[case] seeded_scrubs: &[(usize, usize, usize)],
) {
    let (mut graph, scrubs) =
        self_target_fixture(neighbours, seeded_scrubs).expect("self-target fixture must be valid");
    let before = snapshot_state(&graph, &scrubs);
    let mut reconciler = EdgeReconciler::new(&mut graph);
    reconciler.deferred_scrubs = scrubs;
    let ctx = update_context(1);

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        reconciler.ensure_reverse_edge(&ctx, 0)
    }))
    .expect_err("a self-target must trigger the debug assertion");

    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied());
    assert!(
        message.is_some_and(|text| text.contains(
            "ensure_reverse_edge requires target != origin; target 0, origin 0, level 1"
        )),
        "unexpected panic message: {message:?}",
    );
    assert_eq!(
        snapshot_state(reconciler.graph, &reconciler.deferred_scrubs),
        before,
        "a rejected self-target must not mutate graph state",
    );
}

#[cfg(not(debug_assertions))]
#[rstest]
#[case::spare_capacity(&[1], &[])]
#[case::at_capacity(&[1, 2], &[(2, 1, 1)])]
#[case::existing_self_loop(&[0, 1], &[])]
fn ensure_reverse_edge_rejects_a_self_target_without_mutation(
    #[case] neighbours: &[usize],
    #[case] seeded_scrubs: &[(usize, usize, usize)],
) {
    let (mut graph, scrubs) =
        self_target_fixture(neighbours, seeded_scrubs).expect("self-target fixture must be valid");
    let before = snapshot_state(&graph, &scrubs);
    let mut reconciler = EdgeReconciler::new(&mut graph);
    reconciler.deferred_scrubs = scrubs;

    assert!(
        !reconciler.ensure_reverse_edge(&update_context(1), 0),
        "a self-target must be rejected",
    );
    assert_eq!(
        snapshot_state(reconciler.graph, &reconciler.deferred_scrubs),
        before,
        "a rejected self-target must not mutate graph state",
    );
}
