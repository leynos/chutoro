//! Base-layer healing tests for eviction-driven connectivity repair.

use super::*;
use rstest::rstest;

/// Context for base layer healing tests with a 4-node graph at level 0.
struct HealingTestContext {
    graph: Graph,
    max_connections: usize,
}

impl HealingTestContext {
    /// Creates a test graph with 4 nodes at level 0, where node 1 is at capacity
    /// with bidirectional edges to nodes 0 and 2, and node 2 is only connected
    /// to node 1.
    fn new(params: HnswParams) -> Result<Self, HnswError> {
        let max_connections = params.max_connections();
        let mut graph = Graph::with_capacity(params, 4);

        insert_node(&mut graph, 0, 0, 0)?;
        insert_node(&mut graph, 1, 0, 1)?;
        insert_node(&mut graph, 2, 0, 2)?;
        insert_node(&mut graph, 3, 0, 3)?;

        assert!(
            add_edge_if_missing(&mut graph, 1, 2, 0),
            "node 1 should exist at level 0",
        );
        assert!(
            add_edge_if_missing(&mut graph, 1, 0, 0),
            "node 1 should exist at level 0",
        );
        assert!(
            add_edge_if_missing(&mut graph, 2, 1, 0),
            "node 2 should exist at level 0",
        );
        assert!(
            add_edge_if_missing(&mut graph, 0, 1, 0),
            "node 0 should exist at level 0",
        );

        Ok(Self {
            graph,
            max_connections,
        })
    }

    fn apply_updates(
        mut self,
        updates: Vec<(StagedUpdate, Vec<usize>)>,
        new_node: NewNodeContext,
    ) -> Result<Graph, HnswError> {
        let mut applicator = CommitApplicator::new(&mut self.graph);
        let (reciprocated, _) =
            applicator.apply_neighbour_updates(updates, self.max_connections, new_node)?;
        applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;
        Ok(self.graph)
    }
}

#[rstest]
fn eviction_at_base_layer_triggers_healing() -> Result<(), HnswError> {
    let params = HnswParams::new(1, 4)?;
    let ctx = HealingTestContext::new(params)?;
    let update = build_update(3, 0, vec![1], ctx.max_connections);
    let new_node = NewNodeContext { id: 3, level: 0 };
    let graph = ctx.apply_updates(vec![update], new_node)?;

    let node2 = graph.node(2).expect("node 2 should exist");
    assert!(
        node2.neighbours(0).contains(&0),
        "node 2 should be healed to connect to entry node 0",
    );

    Ok(())
}
