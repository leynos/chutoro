//! Error-boundary tests for the CPU pipeline MST stage.
//!
//! Pins the translation from `MstError` to `ChutoroError::CpuMstFailure` and
//! exercises the real pipeline boundary in explicit one-thread and eight-thread
//! pools, so a nondeterministic error selection would surface here rather than
//! only in a direct MST test.

use rayon::ThreadPoolBuilder;

use crate::{CandidateEdge, ChutoroError, ChutoroErrorCode, EdgeHarvest, MstError};

use super::super::{build_mutual_reachability_forest, map_cpu_mst_error};

/// Thread counts the pipeline boundary is pinned in.
const THREAD_COUNTS: [usize; 2] = [1, 8];

/// Extracts a `CpuMstFailure`'s inner code and message.
///
/// Returns an error rather than panicking so each test keeps its own
/// assertions at the `#[test]` boundary.
fn cpu_mst_failure_fields(error: &ChutoroError) -> Result<(&str, &str), String> {
    if error.code() != ChutoroErrorCode::CpuMstFailure {
        return Err(format!("expected CpuMstFailure, got {error:?}"));
    }
    match error {
        ChutoroError::CpuMstFailure { code, message } => Ok((code, message)),
        other => Err(format!("expected CpuMstFailure, got {other:?}")),
    }
}

#[test]
fn map_cpu_mst_error_preserves_the_inner_code_and_message() {
    let error = MstError::InvalidNodeId {
        node: 5,
        node_count: 3,
    };

    let mapped = map_cpu_mst_error(&error);
    let (code, message) =
        cpu_mst_failure_fields(&mapped).expect("mapper must produce a CpuMstFailure");

    assert_eq!(code, "INVALID_NODE_ID");
    assert_eq!(message, "edge references node 5, but node_count is 3");
}

#[test]
fn build_mutual_reachability_forest_reports_the_first_weighted_edge() {
    // `items` is the Kruskal node count, while the core-distance vector is
    // longer so the weighting transform accepts endpoints that the MST stage
    // must then reject. The first harvested edge is the one at sequence 0.
    let items = 2;
    let core_distances = [0.0_f32; 5];
    let harvest = EdgeHarvest::new(vec![
        CandidateEdge::new(0, 3, 1.0, 0),
        CandidateEdge::new(4, 1, 1.0, 1),
    ]);

    for threads in THREAD_COUNTS {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("explicit Rayon pool must build");
        let error = pool
            .install(|| {
                build_mutual_reachability_forest("fixture", items, &harvest, &core_distances)
            })
            .expect_err("out-of-range harvested endpoints must fail");

        let (code, message) =
            cpu_mst_failure_fields(&error).expect("pipeline must produce a CpuMstFailure");

        assert_eq!(code, "INVALID_NODE_ID", "pipeline with {threads} thread(s)");
        assert_eq!(
            message, "edge references node 3, but node_count is 2",
            "pipeline with {threads} thread(s)"
        );
    }
}
