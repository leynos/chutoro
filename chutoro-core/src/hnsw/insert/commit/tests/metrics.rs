//! Metrics tests for commit-path reconciliation.

use super::*;
use metrics_util::debugging::{DebugValue, DebuggingRecorder};

/// Records a counter when removed-edge reconciliation heals an isolated node.
#[test]
fn base_layer_healing_records_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    ::metrics::with_local_recorder(&recorder, || {
        let params = HnswParams::new(2, 4).expect("test parameters must be valid");
        let max_connections = params.max_connections();
        let mut graph = Graph::with_capacity(params, 3);

        insert_node(&mut graph, 0, 0, 0).expect("insert node 0");
        insert_node(&mut graph, 1, 0, 1).expect("insert node 1");
        insert_node(&mut graph, 2, 0, 2).expect("insert node 2");
        add_edge_if_missing(&mut graph, 0, 1, 0);
        add_edge_if_missing(&mut graph, 1, 0, 0);

        let update = build_update(0, 0, vec![2], max_connections);
        let new_node = NewNodeContext { id: 2, level: 0 };
        let mut applicator = CommitApplicator::new(&mut graph);
        let (reciprocated, _) = applicator
            .apply_neighbour_updates(vec![update], max_connections, new_node)
            .expect("apply neighbour updates");
        applicator
            .apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)
            .expect("apply new-node neighbours");
    });

    let healed_nodes =
        snapshotter
            .snapshot()
            .into_hashmap()
            .into_iter()
            .find_map(|(key, (_, _, value))| {
                (key.key().name() == "chutoro.hnsw.reconciliation.healed_nodes_total")
                    .then_some(value)
            });
    assert_eq!(
        healed_nodes,
        Some(DebugValue::Counter(1)),
        "base-layer isolation healing must increment its counter once"
    );
}

/// Records an upper-layer counter when a deferred scrub removes an orphaned edge.
#[test]
fn orphaned_deferred_scrub_records_counter() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    ::metrics::with_local_recorder(&recorder, || {
        let params = HnswParams::new(1, 4).expect("test parameters must be valid");
        let ctx = EvictionTestContext::new(params).expect("eviction fixture must initialize");
        let update = build_update(0, 1, vec![1], ctx.max_connections);
        ctx.apply_updates(vec![update])
            .expect("apply eviction update");
    });

    let orphan_scrubs =
        snapshotter
            .snapshot()
            .into_hashmap()
            .into_iter()
            .find_map(|(key, (_, _, value))| {
                let has_upper_layer = key
                    .key()
                    .labels()
                    .any(|label| label.key() == "layer" && label.value() == "upper");
                (key.key().name() == "chutoro.hnsw.reconciliation.orphan_scrubs_total"
                    && has_upper_layer)
                    .then_some(value)
            });
    assert_eq!(
        orphan_scrubs,
        Some(DebugValue::Counter(1)),
        "an orphaned upper-layer forward edge must increment its counter once"
    );
}
