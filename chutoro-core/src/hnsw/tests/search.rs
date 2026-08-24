//! Search-specific tests covering greedy descent and layer traversal.

use std::num::NonZeroUsize;

use rstest::rstest;

use crate::{
    DataSource,
    hnsw::{
        CpuHnsw, HnswParams,
        graph::{Graph, NodeContext, SearchContext},
    },
};

use super::fixtures::DummySource;

#[rstest]
fn greedy_descent_selects_closest_neighbour() {
    let source = DummySource::new(vec![1.0, 0.8, 0.6, 0.0]);
    let params = HnswParams::new(2, 4).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, source.len());
    graph
        .attach_node(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("attach entry");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 0,
            sequence: 1,
        })
        .expect("attach neighbour one");
    graph
        .attach_node(NodeContext {
            node: 2,
            level: 0,
            sequence: 2,
        })
        .expect("attach neighbour two");
    graph
        .attach_node(NodeContext {
            node: 3,
            level: 0,
            sequence: 3,
        })
        .expect("attach query node");
    graph
        .node_mut(0)
        .expect("entry must exist")
        .neighbours_mut(0)
        .extend([1, 2]);

    let ctx = SearchContext {
        query: 3,
        entry: 0,
        level: 0,
    };
    let result = graph
        .searcher()
        .greedy_search_layer(None, &source, ctx)
        .expect("search must succeed");
    assert_eq!(result, 2);
}

fn equal_distance_stopping_fixture() -> Result<(DummySource, Graph), String> {
    let source = DummySource::new(vec![0.0, 1.0, 1.0, 0.2]);
    let params = HnswParams::new(2, 4).map_err(|error| format!("params must be valid: {error}"))?;
    let mut graph = Graph::with_capacity(params, source.len());

    graph
        .insert_first(NodeContext {
            node: 1,
            level: 0,
            sequence: 1,
        })
        .map_err(|error| format!("seed entry point: {error}"))?;
    graph
        .attach_node(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .map_err(|error| format!("attach query node: {error}"))?;
    graph
        .attach_node(NodeContext {
            node: 2,
            level: 0,
            sequence: 2,
        })
        .map_err(|error| format!("attach tie candidate: {error}"))?;
    graph
        .attach_node(NodeContext {
            node: 3,
            level: 0,
            sequence: 3,
        })
        .map_err(|error| format!("attach hidden closer node: {error}"))?;

    graph
        .node_mut(1)
        .ok_or_else(|| "entry must exist".to_owned())?
        .neighbours_mut(0)
        .extend([2]);
    graph
        .node_mut(2)
        .ok_or_else(|| "tie candidate must exist".to_owned())?
        .neighbours_mut(0)
        .extend([1, 3]);
    graph
        .node_mut(3)
        .ok_or_else(|| "closer node must exist".to_owned())?
        .neighbours_mut(0)
        .extend([2]);

    Ok((source, graph))
}

#[rstest]
fn layer_search_halts_on_equal_distance_candidates() {
    let (source, graph) =
        equal_distance_stopping_fixture().expect("equal-distance stopping fixture must be valid");

    let ctx = SearchContext {
        query: 0,
        entry: 1,
        level: 0,
    }
    .with_ef(1);

    let neighbours = graph
        .searcher()
        .search_layer(None, &source, ctx)
        .expect("layer search must succeed");

    assert_eq!(neighbours.len(), 1, "ef=1 should cap the result set");
    let neighbour = neighbours
        .first()
        .expect("a successful ef=1 search returns one neighbour");
    assert_eq!(
        neighbour.id, 1,
        "layer search must remain at the entry when ties meet the bound",
    );
    assert!(
        neighbour.distance.total_cmp(&1.0).is_eq(),
        "entry distance defines the stopping bound for equal candidates",
    );
}

#[rstest]
fn layer_search_orders_equal_distance_deterministically() {
    let source = DummySource::new(vec![0.0, 1.0, 1.0, 1.0]);
    let params = HnswParams::new(3, 4).expect("parameters must be valid");
    let mut graph = Graph::with_capacity(params, source.len());

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("seed entry point");

    for (node, sequence) in [(1, 1_u64), (2, 2_u64), (3, 3_u64)] {
        graph
            .attach_node(NodeContext {
                node,
                level: 0,
                sequence,
            })
            .expect("attach equidistant node");
        graph
            .node_mut(node)
            .expect("node must exist")
            .neighbours_mut(0)
            .extend([0]);
    }

    graph
        .node_mut(0)
        .expect("entry must exist")
        .neighbours_mut(0)
        .extend([1, 2, 3]);

    let ctx = SearchContext {
        query: 0,
        entry: 0,
        level: 0,
    }
    .with_ef(3);

    let neighbours = graph
        .searcher()
        .search_layer(None, &source, ctx)
        .expect("layer search must succeed");
    let ids: Vec<_> = neighbours
        .into_iter()
        .map(|neighbour| neighbour.id)
        .collect();

    assert_eq!(ids, vec![0, 1, 2], "ordering must remain stable under ties");
}

#[rstest]
fn search_respects_minimum_ef() {
    let source = DummySource::new(vec![0.0, 1.5, 3.0]);
    let params = HnswParams::new(2, 4)
        .expect("params must be valid")
        .with_rng_seed(29);
    let index = CpuHnsw::build(&source, params).expect("build must succeed");

    let neighbours = index
        .search(
            &source,
            0,
            NonZeroUsize::new(1).expect("ef must be non-zero"),
        )
        .expect("search must succeed");
    assert_eq!(neighbours.len(), 1);
    let entry = index.inspect_graph(|graph| graph.entry().expect("entry exists").node);
    let entry_distance = source
        .distance(0, entry)
        .expect("entry distance must be valid");
    assert!(
        neighbours
            .first()
            .expect("a successful ef=1 search returns one neighbour")
            .distance
            <= entry_distance,
        "with ef=1 the search should keep a candidate no worse than the entry point",
    );
}
