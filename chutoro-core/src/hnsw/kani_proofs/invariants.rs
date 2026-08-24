//! Invariant-model Kani harnesses for bounded HNSW graph state.

use crate::hnsw::{
    graph::{Graph, NodeContext},
    insert::{KaniUpdateContext, ensure_reverse_edge_for_kani, test_helpers::add_edge_if_missing},
    invariants::has_no_self_loops,
    params::HnswParams,
    types::EntryPoint,
};

fn setup_two_node_graph(params: HnswParams) -> Option<Graph> {
    let mut graph = Graph::with_capacity(params, 2);
    if graph
        .insert_first_for_kani(NodeContext {
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
        .attach_node_for_kani(NodeContext {
            node: 1,
            level: 1,
            sequence: 1,
        })
        .is_err()
    {
        kani::assert(false, "failed to attach node 1");
        return None;
    }

    // Guard against vacuous proofs: a no-op constructor would leave the
    // graph empty and every downstream invariant trivially satisfied.
    kani::assert(
        graph.node(0).is_some_and(|node| node.level_count() == 2),
        "node 0 must exist with two levels after construction",
    );
    kani::assert(
        graph.node(1).is_some_and(|node| node.level_count() == 2),
        "node 1 must exist with two levels after construction",
    );
    kani::assert(
        graph.entry().is_some_and(|entry| entry.node == 0),
        "entry point must reference node 0 after construction",
    );
    Some(graph)
}

fn slice_has_no_duplicates(neighbours: &[usize]) -> bool {
    // A linear scan keeps the assertion path free of `HashSet`'s symbolic
    // SipHash state, which is intractable under Kani.
    for idx in 0..neighbours.len() {
        for candidate in (idx + 1)..neighbours.len() {
            if neighbours[idx] == neighbours[candidate] {
                return false;
            }
        }
    }
    true
}

fn graph_neighbours_are_unique(graph: &Graph) -> bool {
    for (_node_id, node) in graph.nodes_iter() {
        for level in 0..node.level_count() {
            if !slice_has_no_duplicates(node.neighbours(level)) {
                return false;
            }
        }
    }
    true
}

/// Prepares the bounded two-node graph and update context shared by the
/// per-level reverse-edge proofs.
///
/// Constructs bounded parameters, builds the two-node graph, seeds the
/// forward edge from node 0 to node 1 nondeterministically, and returns the
/// graph together with the origin's update context. Returns `None` when
/// construction fails, after asserting the failure.
///
/// This helper is private to this module and serves the bounded two-node
/// Kani reconciliation proofs only. It must not become a general
/// graph-construction abstraction: widening it beyond two nodes, a concrete
/// level, or this seeding pattern would reintroduce the state-space growth
/// recorded in the developers' guide, "Kani CI policy".
fn setup_reverse_edge_proof(level: usize) -> Option<(Graph, KaniUpdateContext)> {
    let Ok(params) = HnswParams::new_for_kani(2, 2) else {
        kani::assert(false, "failed to construct bounded HNSW params");
        return None;
    };
    let max_connections = params.max_connections();
    let mut graph = setup_two_node_graph(params)?;

    if kani::any::<bool>() {
        add_edge_if_missing(&mut graph, 0, 1, level);
    }
    let ctx = KaniUpdateContext::new(0, level, max_connections);
    Some((graph, ctx))
}

/// Verifies that no node has itself as a neighbour (no self-loops).
///
/// This harness drives the production `EdgeReconciler::ensure_reverse_edge`
/// path on a bounded 2-node graph, with a nondeterministic choice of whether
/// the forward edge is seeded first, and asserts that no self-loop appears.
///
/// The graph is bounded at two nodes and the level is a concrete argument:
/// the full reconciled-update helper and symbolic level indices push the
/// solver past the tractable CBMC state space (see the developers' guide,
/// "Kani CI policy"). Broader configurations are covered by the
/// graph-topology property suites.
///
/// # Verification Bounds
///
/// - **Nodes**: 2 (IDs 0, 1), both exposing levels 0 and 1
/// - **Levels**: One concrete level per proof entry point
/// - **Edges**: Nondeterministic forward-edge seeding
#[kani::proof]
#[kani::solver(kissat)]
#[kani::unwind(4)]
fn verify_no_self_loops_2_nodes_base_layer() {
    check_no_self_loops_at_level(0);
}

/// Level-1 sibling of [`verify_no_self_loops_2_nodes_base_layer`].
#[kani::proof]
#[kani::solver(kissat)]
#[kani::unwind(4)]
fn verify_no_self_loops_2_nodes_upper_layer() {
    check_no_self_loops_at_level(1);
}

/// Shared body for the per-level no-self-loop proofs.
fn check_no_self_loops_at_level(level: usize) {
    let Some((mut graph, ctx)) = setup_reverse_edge_proof(level) else {
        return;
    };
    let added = ensure_reverse_edge_for_kani(&mut graph, ctx, 1);
    kani::assert(added, "reverse edge must be ensured");

    kani::assert(
        has_no_self_loops(&graph),
        "no self-loops invariant violated",
    );
}

/// Verifies that neighbour lists contain no duplicates.
///
/// This harness drives `EdgeReconciler::ensure_reverse_edge` twice for the
/// same `(origin, target, level)` tuple, with nondeterministic forward-edge
/// seeding, and asserts that the repeated reconciliation never duplicates a
/// neighbour entry.
///
/// The bounds are chosen for the same tractability reason as
/// [`verify_no_self_loops_2_nodes_base_layer`].
///
/// # Verification Bounds
///
/// - **Nodes**: 2 (IDs 0, 1), both exposing levels 0 and 1
/// - **Levels**: One concrete level per proof entry point
/// - **Updates**: Two reconciliations of the same edge
#[kani::proof]
#[kani::solver(kissat)]
#[kani::unwind(4)]
fn verify_neighbour_uniqueness_2_nodes_base_layer() {
    check_neighbour_uniqueness_at_level(0);
}

/// Level-1 sibling of [`verify_neighbour_uniqueness_2_nodes_base_layer`].
#[kani::proof]
#[kani::solver(kissat)]
#[kani::unwind(4)]
fn verify_neighbour_uniqueness_2_nodes_upper_layer() {
    check_neighbour_uniqueness_at_level(1);
}

/// Shared body for the per-level neighbour-uniqueness proofs.
fn check_neighbour_uniqueness_at_level(level: usize) {
    let Some((mut graph, ctx)) = setup_reverse_edge_proof(level) else {
        return;
    };
    let first = ensure_reverse_edge_for_kani(&mut graph, ctx, 1);
    kani::assert(first, "first reconciliation must succeed");
    let second = ensure_reverse_edge_for_kani(&mut graph, ctx, 1);
    kani::assert(second, "repeated reconciliation must succeed");

    kani::assert(
        graph_neighbours_are_unique(&graph),
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
#[kani::solver(kissat)]
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
