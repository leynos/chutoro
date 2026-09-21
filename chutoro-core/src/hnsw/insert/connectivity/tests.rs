//! Regression coverage for connectivity healing and touched-node tracking.

use std::collections::HashSet;

use rstest::rstest;

use super::super::test_helpers::TestHelpers;
use super::{ConnectivityHealer, LinearVisitedSet};
use crate::hnsw::{
    graph::{Graph, NodeContext},
    params::HnswParams,
};

/// The linear-scan set must report insertions exactly as `HashSet` does,
/// because Kani builds substitute it for the production `HashSet` inside
/// the healing work queues.
/// Verifies the Kani substitute retains `HashSet` insertion semantics.
#[rstest]
#[case::all_unique(&[1, 2, 3, 4])]
#[case::immediate_duplicate(&[7, 7])]
#[case::interleaved_duplicates(&[3, 1, 3, 2, 1, 3])]
#[case::single(&[0])]
#[case::empty(&[])]
fn linear_set_matches_hash_set_semantics(#[case] sequence: &[usize]) {
    let mut linear = LinearVisitedSet::default();
    let mut hashed = HashSet::new();
    for &id in sequence {
        assert_eq!(
            linear.insert(id),
            hashed.insert(id),
            "insert({id}) diverged from HashSet semantics",
        );
    }
}

/// Verifies cleanup mutations remain visible to localized reciprocity healing.
#[test]
fn reachability_link_tracks_cleaned_owner_for_localized_reciprocity() {
    let params = HnswParams::new(1, 4).expect("parameters");
    let mut graph = Graph::with_capacity(params, 4);
    for node in 0..4 {
        let context = NodeContext {
            node,
            level: 0,
            sequence: node as u64,
        };
        if node == 0 {
            graph.insert_first(context).expect("insert entry");
        } else {
            graph.attach_node(context).expect("attach node");
        }
    }

    graph
        .node_mut(0)
        .expect("entry")
        .neighbours_mut(0)
        .expect("entry base layer")
        .extend([3, 1]);
    graph
        .node_mut(1)
        .expect("evicted node")
        .neighbours_mut(0)
        .expect("evicted base layer")
        .extend([0, 3]);
    graph
        .node_mut(3)
        .expect("neighbour")
        .neighbours_mut(0)
        .expect("neighbour base layer")
        .push(0);

    let visited = vec![true, false, false, false];
    assert!(
        TestHelpers::new(&mut graph).try_connect_unreachable_node(2, &visited, 1),
        "the saturated entry should accept the fallback link"
    );

    let touched = graph.take_touched_nodes();
    assert_eq!(
        touched,
        vec![(0, 0), (1, 0), (2, 0)],
        "linking must track origin, new node, and evicted cleanup owner"
    );

    TestHelpers::new(&mut graph).enforce_bidirectional_for_touched(&touched, 1);
    let evicted = graph.node(1).expect("evicted node remains");
    let neighbour = graph.node(3).expect("neighbour remains");
    assert!(
        evicted.neighbours(0).contains(&3) && neighbour.neighbours(0).contains(&1),
        "localized healing must repair the retained edge of the cleaned owner"
    );
    assert!(
        graph.take_touched_nodes().is_empty(),
        "localized healing must consume its mutation queue"
    );
}

/// Verifies iterative eviction records each adjacency owner it mutates.
#[test]
fn iterative_eviction_healing_tracks_every_mutated_owner() {
    let params = HnswParams::new(1, 4).expect("parameters");
    let mut graph = Graph::with_capacity(params, 4);
    for node in 0..4 {
        let context = NodeContext {
            node,
            level: 0,
            sequence: node as u64,
        };
        if node == 0 {
            graph.insert_first(context).expect("insert entry");
        } else {
            graph.attach_node(context).expect("attach node");
        }
    }

    graph
        .node_mut(0)
        .expect("entry")
        .neighbours_mut(0)
        .expect("entry base layer")
        .extend([3, 1]);
    graph
        .node_mut(1)
        .expect("first eviction")
        .neighbours_mut(0)
        .expect("first eviction base layer")
        .push(0);

    let context = super::super::types::UpdateContext {
        origin: 0,
        level: 0,
        max_connections: 1,
    };
    assert!(
        ConnectivityHealer::new(&mut graph).link_new_node(&context, 2),
        "the base-layer link should start iterative healing"
    );

    let touched = graph.take_touched_nodes();
    assert_eq!(
        touched,
        vec![(0, 0), (1, 0), (2, 0)],
        "each iterative eviction mutation must remain visible to local healing"
    );
}

/// Verifies pre-existing reciprocal links do not enqueue duplicate mutations.
#[test]
fn duplicate_directed_links_do_not_record_touched_nodes() {
    let params = HnswParams::new(1, 2).expect("parameters");
    let mut graph = Graph::with_capacity(params, 2);
    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert entry");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 0,
            sequence: 1,
        })
        .expect("attach neighbour");
    graph
        .node_mut(0)
        .expect("entry")
        .neighbours_mut(0)
        .expect("entry base layer")
        .push(1);
    graph
        .node_mut(1)
        .expect("neighbour")
        .neighbours_mut(0)
        .expect("neighbour base layer")
        .push(0);

    let context = super::super::types::UpdateContext {
        origin: 0,
        level: 0,
        max_connections: 1,
    };
    assert!(
        ConnectivityHealer::new(&mut graph).link_new_node(&context, 1),
        "existing reciprocal links should succeed"
    );
    assert!(
        graph.take_touched_nodes().is_empty(),
        "duplicate links must not create touched-node entries"
    );
}
