//! Tests for the CPU HNSW edge-collector boundary.
//!
//! `CpuHnsw::build` inserts through `NoopCollector`, so these tests cover the
//! non-harvesting insertion path that the per-insert candidate-edge allocation
//! was removed from. They live apart from [`super::unit_tests`] to keep both
//! modules within the module length policy.

use super::{CpuHnsw, collectors::EdgeCollector, unit_tests::TestSource};
use crate::hnsw::{HnswParams, insert::test_helpers::edge_extraction_count, types::CandidateEdge};

/// Collector that records any edges the insertion path hands it.
///
/// Declaring `WANTS_EDGES = false` mimics `NoopCollector`: counting the calls
/// and the edges handed over lets a test prove the collector boundary stays
/// closed for an insertion whose plan does contain neighbours.
struct RecordingCollector {
    calls: usize,
    edges: usize,
}

impl EdgeCollector for RecordingCollector {
    const WANTS_EDGES: bool = false;

    fn collect(&mut self, edges: Vec<CandidateEdge>) {
        self.calls += 1;
        self.edges += edges.len();
    }
}

/// Verifies a non-harvesting insertion skips candidate-edge extraction.
///
/// Two mutants would each reinstate the per-insert edge allocation this path
/// exists to avoid: removing the `if C::WANTS_EDGES` guard, or hoisting
/// `extract_candidate_edges` above it while leaving `collect` gated. The
/// extraction probe catches both because it counts calls to
/// `extract_candidate_edges`, not edges handed to a collector. The harvesting
/// insertion at the end is a positive control: it shows the probe observes a
/// real extraction, so a dead probe cannot make the guard assertion vacuous.
#[test]
fn non_harvesting_insert_skips_edge_extraction() {
    let params = HnswParams::new(2, 4).expect("params").with_rng_seed(53);
    let index = CpuHnsw::with_capacity(params, 3).expect("index");
    let source = TestSource::new(vec![0.0, 1.0, 2.0]);

    index.insert(0, &source).expect("seed entry point");

    // A second insertion into a populated graph plans real neighbours, so both
    // a dropped guard and a hoisted extraction would show up below.
    let before = edge_extraction_count();
    let mut collector = RecordingCollector { calls: 0, edges: 0 };
    index
        .insert_with_collector(1, &source, &mut collector)
        .expect("insert must succeed");

    assert_eq!(
        collector.calls, 0,
        "non-harvesting collector must not be called"
    );
    assert_eq!(collector.edges, 0, "no candidate edges may be handed to it");
    assert_eq!(
        edge_extraction_count(),
        before,
        "non-harvesting insertion must not extract candidate edges"
    );

    let edges = index
        .insert_harvesting(2, &source)
        .expect("harvesting insert must succeed");
    assert!(!edges.is_empty(), "harvesting insert must find neighbours");
    assert!(
        edge_extraction_count() > before,
        "the extraction probe must observe harvesting insertions"
    );
}
