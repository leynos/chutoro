//! Tests for CPU HNSW build and insertion orchestration helpers.

use super::*;
use crate::{MetricDescriptor, datasource::DataSource, error::DataSourceError, hnsw::HnswParams};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering as AtomicOrdering},
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
    let started = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(AtomicBool::new(false));

    let handle = {
        let index = Arc::clone(&index);
        let source = Arc::clone(&source);
        let started = Arc::clone(&started);
        let finished = Arc::clone(&finished);
        thread::spawn(move || {
            started.store(true, AtomicOrdering::SeqCst);
            index.insert(0, &*source).expect("insert must succeed");
            finished.store(true, AtomicOrdering::SeqCst);
        })
    };

    thread::sleep(Duration::from_millis(50));
    assert!(started.load(AtomicOrdering::SeqCst));
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
