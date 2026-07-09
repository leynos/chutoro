//! Unit tests for the CPU HNSW index.

use super::*;
use crate::{MetricDescriptor, datasource::DataSource, error::DataSourceError, hnsw::HnswParams};
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
        let index = Arc::clone(&index);
        let source = Arc::clone(&source);
        let finished = Arc::clone(&finished);
        thread::spawn(move || {
            started_tx.send(()).expect("report thread start");
            index.insert(0, &*source).expect("insert must succeed");
            finished.store(true, AtomicOrdering::SeqCst);
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

    fn name(&self) -> &str {
        "test"
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        Ok((self.data[left] - self.data[right]).abs())
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("test")
    }
}
