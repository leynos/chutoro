//! Unit tests for the CPU HNSW index.

use super::*;
use crate::{
    MetricDescriptor,
    datasource::DataSource,
    error::DataSourceError,
    hnsw::{HnswParams, graph::NodeContext, insert::test_helpers::add_edge_if_missing},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering as AtomicOrdering},
        mpsc,
    },
    thread,
    time::Duration,
};

#[test]
fn insert_waits_for_mutex() {
    let params = HnswParams::new(2, 4).expect("params").with_rng_seed(31);
    let index = Arc::new(CpuHnsw::with_capacity(params, 2).expect("index"));
    let source = Arc::new(TestSource::new(vec![0.0, 1.0]));

    let guard = index.insert_mutex.lock().expect("mutex");
    let (started_tx, started_rx) = mpsc::channel();
    let finished = Arc::new(AtomicBool::new(false));

    let handle = {
        let shared_index = Arc::clone(&index);
        let shared_source = Arc::clone(&source);
        let completion_flag = Arc::clone(&finished);
        thread::spawn(move || {
            started_tx.send(()).expect("report thread start");
            shared_index
                .insert(0, &*shared_source)
                .expect("insert must succeed");
            completion_flag.store(true, AtomicOrdering::SeqCst);
        })
    };

    started_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("spawned thread should start");
    // The insert cannot complete while this thread holds the mutex, so the
    // flag must still be unset regardless of scheduling.
    assert!(
        !finished.load(AtomicOrdering::SeqCst),
        "insert should block while the mutex is held"
    );

    drop(guard);
    handle.join().expect("thread joins");
    assert!(finished.load(AtomicOrdering::SeqCst));
}

#[test]
fn heal_for_test_repairs_inserted_edges_without_sweeping_unrelated_edges() {
    let params = HnswParams::new(2, 4).expect("params").with_rng_seed(41);
    let index = CpuHnsw::with_capacity(params, 4).expect("index");
    let source = TestSource::new(vec![0.0, 1.0, 2.0, 3.0]);

    index.insert(0, &source).expect("insert entry");
    index.heal_for_test();
    index.insert(1, &source).expect("insert neighbour");

    index
        .write_graph(|graph| {
            let touched = graph.take_touched_nodes();
            assert!(
                touched.contains(&(1, 0)),
                "insertion must record the new node for localized healing"
            );
            graph.record_touched_nodes(touched);

            graph
                .attach_node(NodeContext {
                    node: 2,
                    level: 0,
                    sequence: 2,
                })
                .expect("attach unrelated node");
            graph
                .attach_node(NodeContext {
                    node: 3,
                    level: 0,
                    sequence: 3,
                })
                .expect("attach unrelated target");
            graph
                .node_mut(0)
                .expect("entry must exist")
                .neighbours_mut(0)
                .expect("entry must expose the base layer")
                .retain(|neighbour| *neighbour != 1);
            add_edge_if_missing(graph, 0, 2, 0);
            add_edge_if_missing(graph, 2, 0, 0);
            add_edge_if_missing(graph, 0, 3, 0);
            add_edge_if_missing(graph, 3, 0, 0);
            add_edge_if_missing(graph, 2, 3, 0);
            Ok(())
        })
        .expect("prepare asymmetric edges");

    index.heal_for_test();

    index.inspect_graph(|graph| {
        let node0 = graph.node(0).expect("entry must remain");
        let node1 = graph.node(1).expect("inserted node must remain");
        let node2 = graph.node(2).expect("unrelated node must remain");
        let node3 = graph.node(3).expect("unrelated target must remain");
        assert!(
            node0.neighbours(0).contains(&1),
            "healing must restore the reciprocal edge for the touched insertion"
        );
        assert!(node1.neighbours(0).contains(&0));
        assert!(
            node2.neighbours(0).contains(&3),
            "the unrelated asymmetric edge is preserved for the locality check"
        );
        assert!(
            !node3.neighbours(0).contains(&2),
            "healing must not sweep an untouched asymmetric edge"
        );
    });
    let touched = index
        .write_graph(|graph| Ok(graph.take_touched_nodes()))
        .expect("read healing queue");
    assert!(touched.is_empty(), "healing must drain the insertion queue");
}

#[test]
fn heal_for_test_drains_tracking_created_by_deletion() {
    let params = HnswParams::new(2, 4).expect("params").with_rng_seed(43);
    let mut index = CpuHnsw::with_capacity(params, 3).expect("index");
    let source = TestSource::new(vec![0.0, 1.0, 2.0]);

    for node in 0..source.len() {
        index.insert(node, &source).expect("insert node");
    }
    index
        .write_graph(|graph| Ok(graph.take_touched_nodes()))
        .expect("clear insertion tracking");

    assert!(
        index.delete_node_for_test(1).expect("delete node"),
        "existing node should be deleted"
    );
    let touched = index
        .write_graph(|graph| Ok(graph.take_touched_nodes()))
        .expect("read deletion tracking");
    assert!(
        !touched.is_empty(),
        "deletion must record changed adjacency lists for healing"
    );
    index
        .write_graph(|graph| {
            graph.record_touched_nodes(touched);
            Ok(())
        })
        .expect("restore deletion tracking");

    index.heal_for_test();

    assert!(index.inspect_graph(|graph| graph.node(1).is_none()));
    let healed_touched = index
        .write_graph(|graph| Ok(graph.take_touched_nodes()))
        .expect("read healing queue");
    assert!(
        healed_touched.is_empty(),
        "healing must drain the deletion queue"
    );
}

#[test]
fn heal_for_test_repairs_reachability_and_drains_its_tracking() {
    let params = HnswParams::new(2, 4).expect("params").with_rng_seed(47);
    let index = CpuHnsw::with_capacity(params, 3).expect("index");

    index
        .write_graph(|graph| {
            graph.insert_first(NodeContext {
                node: 0,
                level: 0,
                sequence: 0,
            })?;
            graph.attach_node(NodeContext {
                node: 1,
                level: 0,
                sequence: 1,
            })?;
            graph.attach_node(NodeContext {
                node: 2,
                level: 0,
                sequence: 2,
            })?;
            add_edge_if_missing(graph, 0, 1, 0);
            add_edge_if_missing(graph, 1, 0, 0);
            let _ = graph.take_touched_nodes();
            Ok(())
        })
        .expect("prepare disconnected graph");

    index.heal_for_test();

    index.inspect_graph(|graph| {
        let entry = graph.node(0).expect("entry must remain");
        let repaired = graph.node(2).expect("isolated node must remain");
        assert!(
            entry.neighbours(0).contains(&2),
            "reachability repair must link the isolated node from the entry"
        );
        assert!(
            repaired.neighbours(0).contains(&0),
            "reachability repair must create a reciprocal link"
        );
    });
    let touched = index
        .write_graph(|graph| Ok(graph.take_touched_nodes()))
        .expect("read healing queue");
    assert!(
        touched.is_empty(),
        "healing must consume tracking created by reachability repair"
    );
}
#[derive(Clone)]
struct TestSource {
    data: Vec<f32>,
}

impl TestSource {
    fn new(data: Vec<f32>) -> Self {
        Self { data }
    }
}

impl DataSource for TestSource {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &'static str {
        "test"
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let left_value = self
            .data
            .get(left)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let right_value = self
            .data
            .get(right)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        Ok(left_value
            .mul_add(1.0, std::ops::Neg::neg(*right_value))
            .abs())
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("test")
    }
}
