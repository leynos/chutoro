//! Concurrent stress coverage for the striped-lock union-find protocol.

use std::sync::{Arc, Barrier};

use rand::{Rng, SeedableRng, rngs::SmallRng};
use rstest::rstest;

use super::super::MstError;
use super::ConcurrentUnionFind;

const MAX_THREAD_COUNT: usize = 8;
const ROUND_COUNT: usize = 4;
const NODE_COUNT: usize = ROUND_COUNT * (MAX_THREAD_COUNT + 1);

/// Verify concurrent unions preserve the sequential oracle's partition.
#[rstest]
#[case(42, 2)]
#[case(999, 4)]
#[case(7_777, 8)]
fn concurrent_unions_match_sequential_partition(#[case] seed: u64, #[case] thread_count: usize) {
    let rounds = Arc::new(coordinated_edges(seed, thread_count));
    let union_find = Arc::new(ConcurrentUnionFind::new(NODE_COUNT));
    let round_start = Arc::new(Barrier::new(thread_count));
    let round_complete = Arc::new(Barrier::new(thread_count));

    let handles: Vec<_> = (0..thread_count)
        .map(|worker_index| {
            let worker_rounds = Arc::clone(&rounds);
            let worker_union_find = Arc::clone(&union_find);
            let worker_round_start = Arc::clone(&round_start);
            let worker_round_complete = Arc::clone(&round_complete);

            std::thread::spawn(move || -> Result<(), MstError> {
                for round in 0..ROUND_COUNT {
                    // Fresh pivots retain conflicting root-lock attempts
                    // without allowing an early worker to saturate the partition.
                    worker_round_start.wait();
                    let edges = worker_rounds.get(round).ok_or_else(|| {
                        test_invariant("scheduled round must exist", round, worker_rounds.len())
                    })?;
                    let (left, right) = edges.get(worker_index).copied().ok_or_else(|| {
                        test_invariant(
                            "scheduled worker edge must exist",
                            worker_index,
                            edges.len(),
                        )
                    })?;
                    worker_union_find.try_union(left, right)?;
                    worker_round_complete.wait();
                }
                Ok(())
            })
        })
        .collect();

    for handle in handles {
        handle
            .join()
            .expect("union worker must not panic")
            .expect("generated worker range and node identifiers must be valid");
    }

    let concurrent_labels = normalised_labels(|node| union_find.root_of(node))
        .expect("concurrent union-find nodes must remain valid");
    let oracle_edges: Vec<_> = rounds.iter().flatten().copied().collect();
    let (oracle_labels, oracle_components) =
        sequential_oracle(&oracle_edges).expect("sequential oracle nodes must remain valid");

    assert_eq!(concurrent_labels, oracle_labels);
    assert_eq!(union_find.components(), oracle_components);
}
/// Build sparse, seed-stable rounds with a shared pivot per worker group.
fn coordinated_edges(seed: u64, thread_count: usize) -> Vec<Vec<(usize, usize)>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..ROUND_COUNT)
        .map(|round| round_edges(&mut rng, round, thread_count))
        .collect()
}
/// Generate a conflicting worker edge for each thread in a schedule round.
fn round_edges(rng: &mut SmallRng, round: usize, thread_count: usize) -> Vec<(usize, usize)> {
    let group_start = round * (thread_count + 1);
    let pivot = group_start + thread_count;
    (0..thread_count)
        .map(|worker_index| orient_edge(rng, pivot, group_start + worker_index))
        .collect()
}
/// Randomly choose the direction of an otherwise fixed conflicting edge.
fn orient_edge(rng: &mut SmallRng, pivot: usize, leaf: usize) -> (usize, usize) {
    if rng.gen_bool(0.5) {
        (pivot, leaf)
    } else {
        (leaf, pivot)
    }
}
/// Map every node to its component's smallest node identifier.
fn normalised_labels(
    root_of: impl Fn(usize) -> Result<usize, MstError>,
) -> Result<Vec<usize>, MstError> {
    let mut component_minimums = [NODE_COUNT; NODE_COUNT];
    let component_count = component_minimums.len();
    for node in 0..NODE_COUNT {
        let root = root_of(node)?;
        let component_minimum = component_minimums.get_mut(root).ok_or_else(|| {
            test_invariant(
                "component root must be within the normalisation table",
                root,
                component_count,
            )
        })?;
        *component_minimum = (*component_minimum).min(node);
    }

    (0..NODE_COUNT)
        .map(|node| {
            let root = root_of(node)?;
            component_minimums.get(root).copied().ok_or_else(|| {
                test_invariant(
                    "component root must be within the normalisation table",
                    root,
                    component_minimums.len(),
                )
            })
        })
        .collect()
}
/// Apply the schedule to a scalar union-find oracle.
fn sequential_oracle(edges: &[(usize, usize)]) -> Result<(Vec<usize>, usize), MstError> {
    let mut parents: Vec<usize> = (0..NODE_COUNT).collect();
    let parent_count = parents.len();
    let mut components = NODE_COUNT;

    for &(left, right) in edges {
        let left_root = scalar_find(&parents, left)?;
        let right_root = scalar_find(&parents, right)?;
        if left_root != right_root {
            set_parent(&mut parents, right_root, left_root, parent_count)?;
            components -= 1;
        }
    }

    Ok((
        normalised_labels(|node| scalar_find(&parents, node))?,
        components,
    ))
}
/// Link a scalar child root to its parent root.
fn set_parent(
    parents: &mut [usize],
    child_root: usize,
    parent_root: usize,
    parent_count: usize,
) -> Result<(), MstError> {
    let parent = parents.get_mut(child_root).ok_or_else(|| {
        test_invariant(
            "sequential root must be within the parent table",
            child_root,
            parent_count,
        )
    })?;
    *parent = parent_root;
    Ok(())
}

/// Find a scalar oracle root without path compression.
fn scalar_find(parents: &[usize], node: usize) -> Result<usize, MstError> {
    let mut current = node;
    loop {
        let parent = *parents.get(current).ok_or_else(|| {
            test_invariant(
                "sequential node must be within the parent table",
                current,
                parents.len(),
            )
        })?;
        if parent == current {
            return Ok(current);
        }
        current = parent;
    }
}

/// Construct a descriptive invariant error for test-only checked access.
fn test_invariant(invariant: &'static str, index: usize, lock_count: usize) -> MstError {
    MstError::InvariantViolation {
        invariant,
        index,
        lock_count,
    }
}
