//! Executor regression for fallback retention during reciprocal reconciliation.

use super::*;

/// Builds the symmetric five-node cycle with a vacant insertion slot.
///
/// Clearing fixture touches isolates the queue assertions to the actual commit.
fn fallback_cycle() -> Result<Graph, HnswError> {
    let mut graph = setup_basic_graph(1, 4, 6)?;
    insert_entry_node(&mut graph, 0)?;
    for node in 1..5 {
        attach_test_node(&mut graph, node, 0, node as u64)?;
    }
    for (origin, targets) in [
        (0, [1, 4]),
        (1, [0, 2]),
        (2, [1, 3]),
        (3, [2, 4]),
        (4, [3, 0]),
    ] {
        for target in targets {
            add_edge_if_missing(&mut graph, origin, target, 0);
        }
    }
    let _ = graph.take_touched_nodes();
    Ok(graph)
}

/// Preserves the workspace fallback's tail edge through another owner's update.
///
/// At base-layer capacity two, only one other planned owner can insert a reverse
/// edge into the selected fallback owner, so its newly appended edge survives.
#[test]
fn commit_retains_workspace_fallback_after_second_owner_reconciliation() {
    let params = HnswParams::new(1, 4).expect("params must be valid");
    let mut graph = fallback_cycle().expect("cycle graph must build");

    let mut executor = InsertionExecutor::new(&mut graph);
    let (mut prepared, trim_jobs) = executor
        .apply(
            NodeContext {
                node: 5,
                level: 0,
                sequence: 5,
            },
            ApplyContext {
                params: &params,
                plan: InsertionPlan {
                    layers: vec![LayerPlan {
                        level: 0,
                        neighbours: vec![
                            Neighbour {
                                id: 2,
                                distance: 0.0,
                            },
                            Neighbour {
                                id: 3,
                                distance: 1.0,
                            },
                        ],
                    }],
                },
            },
        )
        .expect("apply the bounded two-owner plan");
    // Staging drains a HashMap; choose the valid order that writes the fallback
    // before the second owner exercises front eviction on that fallback list.
    prepared.updates.sort_by_key(|update| update.node);
    assert_eq!(trim_jobs.len(), 2, "both full owners must require trimming");
    let trims = trim_jobs
        .into_iter()
        .map(|job| TrimResult {
            node: job.node,
            ctx: job.ctx,
            neighbours: match job.node {
                2 => vec![1, 3],
                3 => vec![2, 4],
                other => panic!("unexpected trim owner {other}"),
            },
        })
        .collect();
    executor
        .commit(prepared, trims)
        .expect("commit workspace fallback");

    assert_bidirectional_edge!(&graph, 2, 5, 0);
    for (node, expected) in [
        (2, vec![5, 3]),
        (5, vec![2]),
        (0, vec![1, 4]),
        (4, vec![3, 0]),
    ] {
        assert_eq!(
            graph.node(node).expect("cycle owner exists").neighbours(0),
            expected,
            "owner {node} must preserve fallback or unrelated adjacency"
        );
    }
    let touched = graph.take_touched_nodes();
    for pair in [(2, 0), (3, 0), (5, 0)] {
        assert!(
            touched.contains(&pair),
            "commit must track mutated owner {pair:?}"
        );
    }
}
