//! Invariant-model Kani harnesses for bounded HNSW graph state.

use std::collections::HashSet;

use crate::hnsw::{
    graph::{Graph, NodeContext},
    insert::{KaniUpdateContext, apply_reconciled_update_for_kani},
    invariants::has_no_self_loops,
    params::HnswParams,
    types::EntryPoint,
};

fn setup_four_node_graph(params: HnswParams) -> Option<Graph> {
    let mut graph = Graph::with_capacity(params, 4);
    if graph
        .insert_first(NodeContext {
            node: 0,
            level: 1,
            sequence: 0,
        })
        .is_err()
    {
        kani::assert(false, "failed to insert node 0");
        return None;
    }
    if graph
        .attach_node(NodeContext {
            node: 1,
            level: 1,
            sequence: 1,
        })
        .is_err()
    {
        kani::assert(false, "failed to attach node 1");
        return None;
    }
    if graph
        .attach_node(NodeContext {
            node: 2,
            level: 0,
            sequence: 2,
        })
        .is_err()
    {
        kani::assert(false, "failed to attach node 2");
        return None;
    }
    if graph
        .attach_node(NodeContext {
            node: 3,
            level: 0,
            sequence: 3,
        })
        .is_err()
    {
        kani::assert(false, "failed to attach node 3");
        return None;
    }
    Some(graph)
}

fn graph_neighbours_are_unique(graph: &Graph) -> bool {
    for (_node_id, node) in graph.nodes_iter() {
        for level in 0..node.level_count() {
            let mut seen = HashSet::new();
            for &neighbour in node.neighbours(level) {
                if !seen.insert(neighbour) {
                    return false;
                }
            }
        }
    }
    true
}

fn symbolic_update_level() -> usize {
    let level = kani::any::<usize>();
    kani::assume(level <= 1);
    level
}

fn bounded_level_one_node_for_kani() -> usize {
    let node = kani::any::<usize>();
    kani::assume(node < 2);
    node
}

fn update_origin_for_level(level: usize) -> usize {
    if level == 1 {
        bounded_level_one_node_for_kani()
    } else {
        bounded_node_id_for_kani()
    }
}

fn upper_layer_peer(origin: usize) -> usize {
    if origin == 0 { 1 } else { 0 }
}

fn deduped_targets(first: usize, second: usize) -> Vec<usize> {
    let mut targets = vec![first, second];
    if first == second {
        targets.pop();
    }
    targets
}

/// Verifies that no node has itself as a neighbour (no self-loops).
///
/// This harness creates a bounded 4-node graph and nondeterministically adds
/// edges between distinct nodes. Since the edge addition helper never creates
/// self-loops, this verifies that the invariant holds for all possible edge
/// configurations.
///
/// # Verification Bounds
///
/// - **Nodes**: 4 (IDs 0, 1, 2, 3)
/// - **Levels**: 2 (levels 0 and 1)
/// - **Edges**: Nondeterministic selection between distinct nodes
#[kani::proof]
#[kani::unwind(10)]
fn verify_no_self_loops_4_nodes() {
    let Ok(params) = HnswParams::new(2, 2) else {
        kani::assert(false, "failed to construct bounded HNSW params");
        return;
    };
    let max_connections = params.max_connections();
    let Some(mut graph) = setup_four_node_graph(params) else {
        return;
    };

    let level = symbolic_update_level();
    let origin = update_origin_for_level(level);
    let target = bounded_node_id_for_kani();
    let ctx = KaniUpdateContext::new(origin, level, max_connections);
    let mut next = deduped_targets(target, upper_layer_peer(origin));
    apply_reconciled_update_for_kani(&mut graph, ctx, &mut next);

    if kani::any::<bool>() {
        let second_level = symbolic_update_level();
        let second_origin = update_origin_for_level(second_level);
        let second_target = bounded_node_id_for_kani();
        let second_ctx = KaniUpdateContext::new(second_origin, second_level, max_connections);
        let mut second_next = deduped_targets(second_target, upper_layer_peer(second_origin));
        apply_reconciled_update_for_kani(&mut graph, second_ctx, &mut second_next);
    }

    kani::assert(
        has_no_self_loops(&graph),
        "no self-loops invariant violated",
    );
}

/// Verifies that neighbour lists contain no duplicates.
///
/// This harness drives the production reconciliation/write-back helper and
/// inspects the resulting graph adjacency rather than a separate model.
///
/// # Verification Bounds
///
/// - **Nodes**: 4 (IDs 0, 1, 2, 3)
/// - **Levels**: 2 (levels 0 and 1)
/// - **Updates**: Nondeterministic edge addition via reconciliation
#[kani::proof]
#[kani::unwind(10)]
fn verify_neighbour_uniqueness_4_nodes() {
    let Ok(params) = HnswParams::new(2, 2) else {
        kani::assert(false, "failed to construct bounded HNSW params");
        return;
    };
    let max_connections = params.max_connections();
    let Some(mut graph) = setup_four_node_graph(params) else {
        return;
    };

    let level = symbolic_update_level();
    let origin = update_origin_for_level(level);
    let first_target = bounded_node_id_for_kani();
    let second_target = bounded_node_id_for_kani();
    let ctx = KaniUpdateContext::new(origin, level, max_connections);
    let mut next = deduped_targets(first_target, upper_layer_peer(origin));
    apply_reconciled_update_for_kani(&mut graph, ctx, &mut next);

    let mut replacement = deduped_targets(second_target, upper_layer_peer(origin));
    apply_reconciled_update_for_kani(&mut graph, ctx, &mut replacement);

    kani::assert(
        graph_neighbours_are_unique(&graph),
        "neighbour uniqueness invariant violated",
    );
}

fn bounded_node_id_for_kani() -> usize {
    let id: usize = kani::any();
    kani::assume(id < 4);
    id
}

/// Verifies entry-point validity and maximality after insertions.
///
/// This harness inserts nodes with nondeterministically chosen levels and
/// verifies that the entry point is always valid and has the maximum level
/// across all nodes.
///
/// # Verification Bounds
///
/// - **Nodes**: 4 (IDs 0, 1, 2, 3)
/// - **Levels**: Up to 3 (max_level = 2, so levels 0, 1, 2)
/// - **Updates**: Nondeterministic level assignment and entry promotion
///
/// # Invariant Under Test
///
/// The entry-point validity invariant states that:
/// - If the graph is empty, there is no entry point.
/// - If the graph is non-empty, the entry point exists, references a valid
///   node, and has a level at least as high as any other node in the graph.
#[kani::proof]
#[kani::unwind(12)]
fn verify_entry_point_validity_4_nodes() {
    let levels = [
        bounded_entry_level_for_kani(),
        bounded_entry_level_for_kani(),
        bounded_entry_level_for_kani(),
        bounded_entry_level_for_kani(),
    ];
    let mut entry = EntryPoint {
        node: 0,
        level: levels[0],
    };
    promote_entry_model_for_kani(&mut entry, 1, levels[1]);
    promote_entry_model_for_kani(&mut entry, 2, levels[2]);
    promote_entry_model_for_kani(&mut entry, 3, levels[3]);
    kani::assert(
        entry_is_valid_for_kani(entry, &levels),
        "entry-point validity invariant violated",
    );
}

fn bounded_entry_level_for_kani() -> usize {
    let level: usize = kani::any();
    kani::assume(level <= 2);
    level
}

fn promote_entry_model_for_kani(entry: &mut EntryPoint, node: usize, level: usize) {
    if Graph::should_promote_entry_for_kani(Some(*entry), level) {
        *entry = EntryPoint { node, level };
    }
}

fn entry_is_valid_for_kani(entry: EntryPoint, levels: &[usize; 4]) -> bool {
    entry.node < levels.len()
        && entry.level == levels[entry.node]
        && entry.level >= levels[0]
        && entry.level >= levels[1]
        && entry.level >= levels[2]
        && entry.level >= levels[3]
}
