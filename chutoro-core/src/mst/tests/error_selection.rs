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

/// Runs `attempt` in every pinned pool and asserts each reports `expected`.
///
/// The attempt is expected to fail validation; returning `Ok` means the
/// contract was not exercised. Mismatches are reported as errors rather than
/// panics so the helper stays usable in a `Result`-returning test body.
fn assert_every_pool_reports<F>(expected: &MstError, attempt: F) -> Result<(), String>
where
    F: Fn() -> Result<(), MstError> + Sync,
{
    for (threads, pool) in pools()? {
        match pool.install(&attempt) {
            Ok(()) => {
                return Err(format!(
                    "the input must be rejected with {threads} thread(s)"
                ));
            }
            Err(error) if &error == expected => {}
            Err(error) => {
                return Err(format!(
                    "{threads} thread(s) reported {error:?}, expected {expected:?}"
                ));
            }
        }
    }
    Ok(())
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

    assert_every_pool_reports(&expected, || {
        parallel_kruskal_from_edges(3, &edges).map(|_| ())
    })
    .expect("pools must build and the input must be rejected");
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

    assert_every_pool_reports(&expected, || {
        parallel_kruskal_from_edges(4, &edges).map(|_| ())
    })
    .expect("pools must build and the input must be rejected");
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

    assert_every_pool_reports(&expected, || {
        parallel_kruskal_from_edges(3, &edges).map(|_| ())
    })
    .expect("pools must build and the input must be rejected");
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

    assert_every_pool_reports(&expected, || parallel_kruskal(2, &harvest).map(|_| ()))
        .expect("pools must build and the input must be rejected");
}
