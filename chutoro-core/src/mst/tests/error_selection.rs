//! Regression tests for deterministic MST error selection.
//!
//! When several candidate edges are invalid, the reported error must be the one
//! at the lowest input index. Sampling repeated runs could pass by luck, so
//! these cases pin the contract directly: the guarantee comes from the ordered
//! buffer and sequential scan in `prepare_edge_list`, not from the absence of a
//! race in one particular schedule. Each case runs in explicit one-thread and
//! eight-thread pools rather than through `RAYON_NUM_THREADS`, which the
//! repository forbids tests from touching and which CI pins to one.

use rayon::ThreadPoolBuilder;

use crate::{CandidateEdge, EdgeHarvest};

use super::super::{MstError, parallel_kruskal, parallel_kruskal_from_edges};

/// Thread counts the error-selection contract is pinned in.
const THREAD_COUNTS: [usize; 2] = [1, 8];

/// Builds one explicit Rayon pool per pinned thread count.
fn pools() -> Result<Vec<(usize, rayon::ThreadPool)>, String> {
    THREAD_COUNTS
        .into_iter()
        .map(|threads| {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| format!("failed to build a {threads}-thread pool: {error}"))?;
            Ok((threads, pool))
        })
        .collect()
}

/// Collects the direct slice path's error from every pinned pool.
fn slice_errors(
    node_count: usize,
    edges: &[CandidateEdge],
) -> Result<Vec<(usize, MstError)>, String> {
    let mut observed = Vec::new();
    for (threads, pool) in pools()? {
        match pool.install(|| parallel_kruskal_from_edges(node_count, edges)) {
            Ok(_) => {
                return Err(format!(
                    "the input must be rejected with {threads} thread(s)"
                ));
            }
            Err(error) => observed.push((threads, error)),
        }
    }
    Ok(observed)
}

/// Collects the harvest path's error from every pinned pool.
fn harvest_errors(
    node_count: usize,
    harvest: &EdgeHarvest,
) -> Result<Vec<(usize, MstError)>, String> {
    let mut observed = Vec::new();
    for (threads, pool) in pools()? {
        match pool.install(|| parallel_kruskal(node_count, harvest)) {
            Ok(_) => {
                return Err(format!(
                    "the input must be rejected with {threads} thread(s)"
                ));
            }
            Err(error) => observed.push((threads, error)),
        }
    }
    Ok(observed)
}

#[test]
fn small_multi_invalid_input_reports_the_lowest_index() {
    let edges = [
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(5, 0, 1.0, 1),
        CandidateEdge::new(0, 6, 1.0, 2),
        CandidateEdge::new(0, 1, f32::NAN, 3),
    ];
    let expected = MstError::InvalidNodeId {
        node: 5,
        node_count: 3,
    };

    for (threads, error) in
        slice_errors(3, &edges).expect("pools must build and the input must be rejected")
    {
        assert_eq!(
            &error, &expected,
            "direct slice path with {threads} thread(s)"
        );
    }
}

#[test]
fn large_multi_invalid_input_reports_the_lowest_index() {
    // The non-finite edge sits at the lower index, so it must win even though
    // the out-of-range edge is also present and the list is large enough for
    // Rayon to split it across many tasks.
    let edges: Vec<CandidateEdge> = (0..4096)
        .map(|index| match index {
            1024 => CandidateEdge::new(0, 1, f32::NAN, 1024),
            3072 => CandidateEdge::new(9, 0, 1.0, 3072),
            _ => CandidateEdge::new(index & 3, (index + 1) & 3, 1.0, index as u64),
        })
        .collect();
    let expected = MstError::NonFiniteWeight { left: 0, right: 1 };

    for (threads, error) in
        slice_errors(4, &edges).expect("pools must build and the input must be rejected")
    {
        assert_eq!(
            &error, &expected,
            "direct slice path with {threads} thread(s)"
        );
    }
}

#[test]
fn invalid_edge_after_a_complete_tree_is_still_reported() {
    let edges = [
        CandidateEdge::new(0, 1, 1.0, 0),
        CandidateEdge::new(1, 2, 2.0, 1),
        CandidateEdge::new(7, 8, 3.0, 2),
    ];
    let expected = MstError::InvalidNodeId {
        node: 7,
        node_count: 3,
    };

    for (threads, error) in
        slice_errors(3, &edges).expect("pools must build and the input must be rejected")
    {
        assert_eq!(
            &error, &expected,
            "direct slice path with {threads} thread(s)"
        );
    }
}

#[test]
fn harvest_order_defines_the_input_index() {
    // The raw vector's first invalid edge is `(8, 1)`, but `EdgeHarvest` sorts
    // by sequence, so `(0, 7)` is validated first. The contract follows the
    // harvest order because that is the slice the public entry point observes.
    let harvest = EdgeHarvest::new(vec![
        CandidateEdge::new(8, 1, 1.0, 9),
        CandidateEdge::new(0, 7, 1.0, 1),
        CandidateEdge::new(0, 1, 1.0, 5),
    ]);
    let expected = MstError::InvalidNodeId {
        node: 7,
        node_count: 2,
    };

    for (threads, error) in
        harvest_errors(2, &harvest).expect("pools must build and the input must be rejected")
    {
        assert_eq!(&error, &expected, "harvest path with {threads} thread(s)");
    }
}
