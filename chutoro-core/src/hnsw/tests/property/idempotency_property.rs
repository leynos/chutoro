//! Insertion idempotency property for CPU HNSW.
//!
//! Verifies that repeated duplicate insertions leave the graph state unchanged.
//! The property builds an index, snapshots its structure, attempts duplicate
//! insertions (which are rejected), and then verifies the structure remains
//! identical to the snapshot.
//!
//! The HNSW implementation rejects duplicate insertions with
//! [`HnswError::DuplicateNode`](crate::hnsw::HnswError::DuplicateNode) before
//! any state mutation occurs, so the property validates that this rejection
//! mechanism preserves graph integrity.

use proptest::{
    prop_assume,
    test_runner::{TestCaseError, TestCaseResult},
};

use super::types::{HnswFixture, IdempotencyPlan};
use crate::hnsw::tests::property::support::DenseVectorSource;
use crate::hnsw::tests::support::is_coverage_job;
use crate::{CpuHnsw, DataSource, HnswError, hnsw::types::EntryPoint};

const MIN_IDEMPOTENCY_FIXTURE_LEN: usize = 2;
const MAX_IDEMPOTENCY_FIXTURE_LEN: usize = 64;
const COVERAGE_MAX_IDEMPOTENCY_FIXTURE_LEN: usize = 16;
const COVERAGE_MAX_DUPLICATE_INDICES: usize = 3;
const COVERAGE_MAX_ATTEMPTS_PER_INDEX: usize = 2;

/// Runs the idempotency property: builds an index, snapshots its state, attempts
/// duplicate insertions (which are rejected), and verifies the graph state is
/// unchanged.
pub(super) fn run_idempotency_property(
    fixture: HnswFixture,
    plan: IdempotencyPlan,
) -> TestCaseResult {
    let is_coverage_job = is_coverage_job();
    let params = fixture
        .params
        .build()
        .map_err(|err| TestCaseError::fail(format!("invalid params: {err}")))?;
    let source = build_idempotency_source(fixture, is_coverage_job)
        .map_err(|err| TestCaseError::fail(format!("fixture -> source failed: {err}")))?;

    let len = source.len();
    prop_assume!(len >= MIN_IDEMPOTENCY_FIXTURE_LEN);
    prop_assume!(len <= MAX_IDEMPOTENCY_FIXTURE_LEN);

    // Build the index
    let index = CpuHnsw::build(&source, params)
        .map_err(|err| TestCaseError::fail(format!("index build failed: {err}")))?;

    // Snapshot the graph state before duplicate attempts
    let snapshot = snapshot_graph(&index);

    // Attempt duplicate insertions
    let duplicate_indices = duplicate_indices_for_job(&plan, len, is_coverage_job);
    let attempts_per_index = attempts_per_index_for_job(&plan, is_coverage_job);
    for &node in &duplicate_indices {
        for attempt in 0..attempts_per_index {
            let result = index.insert(node, &source);
            match result {
                Err(HnswError::DuplicateNode { node: rejected }) if rejected == node => {
                    // Expected: duplicate was rejected
                }
                Err(other) => {
                    return Err(TestCaseError::fail(format!(
                        "unexpected error on duplicate insert of node {node} \
                         (attempt {attempt}): {other}"
                    )));
                }
                Ok(()) => {
                    return Err(TestCaseError::fail(format!(
                        "duplicate insert for node {node} should have been rejected \
                         (attempt {attempt})"
                    )));
                }
            }
        }
    }

    // Verify graph state is unchanged
    if !graph_matches_snapshot(&index, &snapshot) {
        return Err(TestCaseError::fail(
            "graph state changed after duplicate insertion attempts",
        ));
    }

    Ok(())
}

/// Resolves plan hints to actual node indices within the fixture length.
fn resolve_duplicate_indices(plan: &IdempotencyPlan, len: usize) -> Vec<usize> {
    plan.duplicate_hints
        .iter()
        .map(|&hint| usize::from(hint) % len)
        .collect()
}

fn build_idempotency_source(
    fixture: HnswFixture,
    is_coverage_job: bool,
) -> Result<DenseVectorSource, crate::error::DataSourceError> {
    let max_fixture_len = if is_coverage_job {
        COVERAGE_MAX_IDEMPOTENCY_FIXTURE_LEN
    } else {
        MAX_IDEMPOTENCY_FIXTURE_LEN
    };
    let vectors = fixture.vectors.into_iter().take(max_fixture_len).collect();
    DenseVectorSource::new("hnsw-fixture", vectors)
}

fn duplicate_indices_for_job(
    plan: &IdempotencyPlan,
    len: usize,
    is_coverage_job: bool,
) -> Vec<usize> {
    let duplicate_indices = resolve_duplicate_indices(plan, len);
    if is_coverage_job {
        duplicate_indices
            .into_iter()
            .take(COVERAGE_MAX_DUPLICATE_INDICES)
            .collect()
    } else {
        duplicate_indices
    }
}

fn attempts_per_index_for_job(plan: &IdempotencyPlan, is_coverage_job: bool) -> usize {
    if is_coverage_job {
        plan.attempts_per_index.min(COVERAGE_MAX_ATTEMPTS_PER_INDEX)
    } else {
        plan.attempts_per_index
    }
}

/// Snapshot of a node's structural state.
#[derive(Clone, Debug, PartialEq)]
struct NodeSnapshot {
    /// Sorted neighbour lists at each level.
    neighbours: Vec<Vec<usize>>,
}

/// Snapshot of an HNSW graph's structural state.
#[derive(Clone, Debug, PartialEq)]
struct GraphSnapshot {
    /// Entry point, if present.
    entry: Option<EntryPoint>,
    /// Node snapshots indexed by node ID; None for unoccupied slots.
    nodes: Vec<Option<NodeSnapshot>>,
}

/// Captures a snapshot of the graph's structural state.
fn snapshot_graph(index: &CpuHnsw) -> GraphSnapshot {
    index
        .inspect_graph(|graph| {
            let entry = graph.entry();
            let nodes = (0..graph.capacity())
                .map(|id| graph.node(id).map(snapshot_node))
                .collect();
            GraphSnapshot { entry, nodes }
        })
        .expect("graph snapshot must acquire the test graph lock")
}

/// Captures a snapshot of a single node's structural state.
fn snapshot_node(node: &crate::hnsw::node::Node) -> NodeSnapshot {
    let neighbours = (0..node.level_count())
        .map(|level| {
            let mut nbrs: Vec<_> = node.neighbours(level).to_vec();
            nbrs.sort_unstable();
            nbrs
        })
        .collect();
    NodeSnapshot { neighbours }
}

/// Checks whether the current graph state matches a snapshot.
fn graph_matches_snapshot(index: &CpuHnsw, snapshot: &GraphSnapshot) -> bool {
    let current = snapshot_graph(index);
    current == *snapshot
}

#[cfg(test)]
mod tests {
    //! Regression coverage for duplicate-insertion idempotency helpers.

    use super::*;
    use crate::hnsw::tests::property::types::{
        DistributionMetadata, HnswParamsSeed, VectorDistribution,
    };
    use rstest::rstest;

    fn make_fixture(vector_count: usize, seed: u64) -> HnswFixture {
        let vectors: Vec<Vec<f32>> = (0..vector_count)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32])
            .collect();
        HnswFixture {
            distribution: VectorDistribution::Uniform,
            vectors,
            metadata: DistributionMetadata::Uniform { bound: 100.0 },
            params: HnswParamsSeed {
                max_connections: 4,
                ef_construction: 8,
                level_multiplier: 1.0,
                max_level: 4,
                rng_seed: seed,
            },
        }
    }

    #[rstest]
    #[case(2, 1, 42)]
    #[case(4, 3, 123)]
    #[case(8, 5, 456)]
    #[case(16, 2, 789)]
    fn idempotency_rstest_cases(
        #[case] vector_count: usize,
        #[case] attempts: usize,
        #[case] seed: u64,
    ) {
        let fixture = make_fixture(vector_count, seed);
        let plan = IdempotencyPlan {
            duplicate_hints: (0..vector_count as u16).collect(),
            attempts_per_index: attempts,
        };
        run_idempotency_property(fixture, plan).expect("idempotency property must hold");
    }

    #[rstest]
    fn idempotency_single_node_duplicate() {
        let fixture = make_fixture(3, 99);
        let plan = IdempotencyPlan {
            duplicate_hints: vec![0],
            attempts_per_index: 5,
        };
        run_idempotency_property(fixture, plan).expect("single duplicate must preserve state");
    }

    #[rstest]
    fn idempotency_all_nodes_duplicated() {
        let fixture = make_fixture(5, 77);
        let plan = IdempotencyPlan {
            duplicate_hints: vec![0, 1, 2, 3, 4],
            attempts_per_index: 3,
        };
        run_idempotency_property(fixture, plan).expect("all nodes duplicated must preserve state");
    }

    #[rstest]
    fn resolve_duplicate_indices_uses_modulo() {
        let plan = IdempotencyPlan {
            duplicate_hints: vec![0, 5, 10, 100],
            attempts_per_index: 1,
        };
        let resolved = resolve_duplicate_indices(&plan, 4);
        assert_eq!(resolved, vec![0, 1, 2, 0]);
    }

    #[rstest]
    fn duplicate_indices_for_job_limits_coverage_workload() {
        let plan = IdempotencyPlan {
            duplicate_hints: vec![0, 1, 2, 3, 4],
            attempts_per_index: 5,
        };
        let resolved = duplicate_indices_for_job(&plan, 8, true);

        assert_eq!(resolved, vec![0, 1, 2]);
    }

    #[rstest]
    fn attempts_per_index_for_job_limits_coverage_workload() {
        let plan = IdempotencyPlan {
            duplicate_hints: vec![0],
            attempts_per_index: 5,
        };

        assert_eq!(attempts_per_index_for_job(&plan, true), 2);
        assert_eq!(attempts_per_index_for_job(&plan, false), 5);
    }

    #[rstest]
    #[case(true, COVERAGE_MAX_IDEMPOTENCY_FIXTURE_LEN, 101)]
    #[case(false, MAX_IDEMPOTENCY_FIXTURE_LEN, 202)]
    fn build_idempotency_source_truncates_fixture(
        #[case] is_coverage: bool,
        #[case] expected_len: usize,
        #[case] seed: u64,
    ) {
        let fixture = make_fixture(expected_len + 5, seed);

        let source = build_idempotency_source(fixture, is_coverage).expect("source");

        assert_eq!(source.len(), expected_len);
    }

    #[rstest]
    fn snapshot_detects_graph_structure() {
        let fixture = make_fixture(4, 55);
        let params = fixture.params.build().expect("params");
        let source = fixture.into_source().expect("source");

        let index = CpuHnsw::build(&source, params).expect("index");
        let snapshot = snapshot_graph(&index);

        // Verify snapshot captures entry point
        assert!(
            snapshot.entry.is_some(),
            "snapshot must capture entry point"
        );

        // Verify all nodes are present
        let occupied_count = snapshot.nodes.iter().filter(|n| n.is_some()).count();
        assert_eq!(occupied_count, 4, "all 4 nodes should be in snapshot");

        // Verify snapshot matches itself
        assert!(
            graph_matches_snapshot(&index, &snapshot),
            "graph must match its own snapshot"
        );
    }

    #[rstest]
    fn duplicate_insertion_returns_expected_error() {
        let fixture = make_fixture(3, 11);
        let params = fixture.params.build().expect("params");
        let source = fixture.into_source().expect("source");

        let index = CpuHnsw::build(&source, params).expect("index");
        let result = index.insert(0, &source);

        assert!(
            matches!(result, Err(HnswError::DuplicateNode { node: 0 })),
            "duplicate insert must return DuplicateNode error, got {result:?}",
        );
    }
}
