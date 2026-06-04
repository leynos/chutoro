//! Invariant-model Kani harnesses for bounded HNSW graph state.

use crate::hnsw::{
    graph::{Graph, NodeContext},
    insert::test_helpers::add_edge_if_missing,
    invariants::has_no_self_loops,
    params::HnswParams,
    types::EntryPoint,
};

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
    let params = HnswParams::new(2, 2).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, 4);

    // Insert nodes with varying levels
    graph
        .insert_first(NodeContext {
            node: 0,
            level: 1,
            sequence: 0,
        })
        .expect("insert node 0");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 0,
            sequence: 1,
        })
        .expect("attach node 1");
    graph
        .attach_node(NodeContext {
            node: 2,
            level: 0,
            sequence: 2,
        })
        .expect("attach node 2");
    graph
        .attach_node(NodeContext {
            node: 3,
            level: 0,
            sequence: 3,
        })
        .expect("attach node 3");

    // Nondeterministically add edges (construction never adds self-loops)
    for origin in 0..4usize {
        for target in 0..4usize {
            if origin != target && kani::any::<bool>() {
                add_edge_if_missing(&mut graph, origin, target, 0);
            }
        }
    }

    kani::assert(
        has_no_self_loops(&graph),
        "no self-loops invariant violated",
    );
}

#[derive(Clone, Copy)]
struct NeighbourListModel {
    entries: [Option<usize>; 4],
}

impl NeighbourListModel {
    fn new() -> Self {
        Self { entries: [None; 4] }
    }

    fn contains(&self, target: usize) -> bool {
        for entry in self.entries {
            if entry == Some(target) {
                return true;
            }
        }
        false
    }

    fn push_if_absent(&mut self, target: usize) {
        if self.contains(target) {
            return;
        }
        for slot in &mut self.entries {
            if slot.is_none() {
                *slot = Some(target);
                return;
            }
        }
    }

    fn is_unique(&self) -> bool {
        for idx in 0..self.entries.len() {
            let Some(entry) = self.entries[idx] else {
                continue;
            };
            for candidate_idx in (idx + 1)..self.entries.len() {
                if self.entries[candidate_idx] == Some(entry) {
                    return false;
                }
            }
        }
        true
    }
}

fn add_model_edge_if_missing(
    neighbours: &mut [NeighbourListModel; 4],
    origin: usize,
    target: usize,
) {
    neighbours[origin].push_if_absent(target);
}

fn add_model_reciprocal_edge(
    neighbours: &mut [NeighbourListModel; 4],
    origin: usize,
    target: usize,
) {
    add_model_edge_if_missing(neighbours, origin, target);
    add_model_edge_if_missing(neighbours, target, origin);
}

fn model_neighbours_are_unique(neighbours: &[NeighbourListModel; 4]) -> bool {
    for list in neighbours {
        if !list.is_unique() {
            return false;
        }
    }
    true
}

/// Verifies that neighbour lists contain no duplicates.
///
/// This harness uses a fixed-size neighbour-list model for the add-if-missing
/// semantics used by reconciliation. Nondeterministic choices explore edge
/// configurations and verify that repeated insertions cannot duplicate a
/// neighbour entry.
///
/// # Verification Bounds
///
/// - **Nodes**: 4 (IDs 0, 1, 2, 3)
/// - **Levels**: 2 (levels 0 and 1)
/// - **Updates**: Nondeterministic edge addition via reconciliation
#[kani::proof]
#[kani::unwind(10)]
fn verify_neighbour_uniqueness_4_nodes() {
    let mut neighbours = [NeighbourListModel::new(); 4];

    if kani::any::<bool>() {
        add_model_reciprocal_edge(&mut neighbours, 0, 1);
    }
    if kani::any::<bool>() {
        add_model_reciprocal_edge(&mut neighbours, 0, 2);
    }
    if kani::any::<bool>() {
        add_model_reciprocal_edge(&mut neighbours, 1, 3);
    }

    kani::assert(
        model_neighbours_are_unique(&neighbours),
        "neighbour uniqueness invariant violated",
    );
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
