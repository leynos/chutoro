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

fn graph_with_two_nodes(level: usize) -> Result<Graph, HnswError> {
    let params = HnswParams::new(2, 4)?;
    let mut graph = Graph::with_capacity(params, 2);
    graph.insert_first(NodeContext {
        node: 0,
        level,
        sequence: 0,
    })?;
    graph.attach_node(NodeContext {
        node: 1,
        level,
        sequence: 1,
    })?;
    Ok(graph)
}

fn graph_with_three_nodes(level: usize) -> Result<Graph, HnswError> {
    let params = HnswParams::new(2, 4)?;
    let mut graph = Graph::with_capacity(params, 3);
    graph.insert_first(NodeContext {
        node: 0,
        level,
        sequence: 0,
    })?;
    graph.attach_node(NodeContext {
        node: 1,
        level,
        sequence: 1,
    })?;
    graph.attach_node(NodeContext {
        node: 2,
        level,
        sequence: 2,
    })?;
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
    let mut graph = graph_with_three_nodes(level)?;
    let origin = graph.node_mut(0).ok_or(FixtureError::MissingOrigin)?;
    let origin_neighbours = origin
        .neighbours_mut(level)
        .ok_or(FixtureError::MissingOriginLevel(level))?;
    origin_neighbours.extend_from_slice(neighbours);

    let scrubs = seeded_scrubs
        .iter()
        .map(|&(origin, target, level)| DeferredScrub {
            origin,
            target,
            level,
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

#[test]
fn remove_reverse_edge_reports_a_base_layer_target_it_isolates() {
    let mut graph = graph_with_two_nodes(0).expect("test graph must be valid");
    graph
        .node_mut(1)
        .expect("target node must be present")
        .neighbours_mut(0)
        .expect("target must have a base layer")
        .push(0);

    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert!(reconciler.remove_reverse_edge(&update_context(0), 1));
    assert_eq!(
        reconciler
            .graph
            .node(1)
            .expect("target node must be present")
            .neighbours(0),
        [] as [usize; 0]
    );
}

#[test]
fn remove_reverse_edge_does_not_report_an_upper_layer_target_as_isolated() {
    let mut graph = graph_with_two_nodes(1).expect("test graph must be valid");
    graph
        .node_mut(1)
        .expect("target node must be present")
        .neighbours_mut(1)
        .expect("target must have an upper layer")
        .push(0);

    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert!(!reconciler.remove_reverse_edge(&update_context(1), 1));
    assert_eq!(
        reconciler
            .graph
            .node(1)
            .expect("target node must be present")
            .neighbours(1),
        [] as [usize; 0]
    );
}

#[test]
fn remove_reverse_edge_returns_false_when_the_reverse_edge_is_missing() {
    let mut graph = graph_with_two_nodes(0).expect("test graph must be valid");
    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert!(!reconciler.remove_reverse_edge(&update_context(0), 1));
}

#[test]
fn remove_reverse_edge_returns_false_when_target_or_level_is_missing() {
    let mut graph = graph_with_two_nodes(0).expect("test graph must be valid");
    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert!(!reconciler.remove_reverse_edge(&update_context(0), 2));
    assert!(!reconciler.remove_reverse_edge(&update_context(1), 1));
}

#[rstest]
#[case(0)]
#[case(1)]
fn ensure_reverse_edge_links_a_valid_target_back_to_the_origin(#[case] level: usize) {
    let mut graph = graph_with_two_nodes(level).expect("test graph must be valid");
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
        message.is_some_and(|message| message.contains(
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
