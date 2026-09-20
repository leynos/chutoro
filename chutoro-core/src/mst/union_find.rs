//! Concurrent union-find for the parallel Kruskal implementation.
//!
//! This union-find prioritizes correctness and deadlock avoidance over maximum
//! throughput. Disjoint unions can proceed in parallel while maintaining a
//! consistent lock ordering.
//!
//! The implementation uses one lock per node id (acquired by root id), locking
//! `(min_root, max_root)` to remain deadlock-free. Each union re-validates that
//! the roots used to derive the lock order are still current after acquiring
//! locks; if they change, the attempt is retried.

use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

use super::MstError;

/// Lock-striped disjoint-set state used by parallel Kruskal processing.
pub(super) struct ConcurrentUnionFind {
    /// Parent identifier for each disjoint-set node.
    parents: Vec<AtomicUsize>,
    /// Union-by-rank value for each root candidate.
    ranks: Vec<AtomicUsize>,
    /// Number of connected components remaining.
    components: AtomicUsize,
    /// Per-root locks acquired in deterministic order.
    locks: Vec<Mutex<()>>,
}

impl ConcurrentUnionFind {
    /// Allocate singleton sets for every graph node.
    pub(super) fn new(node_count: usize) -> Self {
        let mut parents = Vec::with_capacity(node_count);
        let mut ranks = Vec::with_capacity(node_count);
        for id in 0..node_count {
            parents.push(AtomicUsize::new(id));
            ranks.push(AtomicUsize::new(0));
        }

        let locks = (0..node_count).map(|_| Mutex::new(())).collect();

        Self {
            parents,
            ranks,
            components: AtomicUsize::new(node_count),
            locks,
        }
    }

    /// Return the current number of disjoint components.
    pub(super) fn components(&self) -> usize {
        self.components.load(Ordering::Acquire)
    }

    /// Join two nodes' sets, retrying when concurrent unions change roots.
    pub(super) fn try_union(&self, left: usize, right: usize) -> Result<bool, MstError> {
        loop {
            let left_root = self.find(left)?;
            let right_root = self.find(right)?;

            if left_root == right_root {
                return Ok(false);
            }

            let lock_pair = lock_order(left_root, right_root);
            let (first_lock, second_lock) = lock_pair;
            let _first_guard = self.lock_root(first_lock)?;
            let _second_guard = (second_lock != first_lock)
                .then(|| self.lock_root(second_lock))
                .transpose()?;

            let current_left_root = self.find(left)?;
            let current_right_root = self.find(right)?;

            if current_left_root == current_right_root {
                return Ok(false);
            }

            if lock_order(current_left_root, current_right_root) != lock_pair {
                continue;
            }

            if !self.is_root(current_left_root)? || !self.is_root(current_right_root)? {
                continue;
            }

            return self.union_roots(current_left_root, current_right_root);
        }
    }

    /// Acquire the mutex associated with a root identifier.
    fn lock_root(&self, index: usize) -> Result<std::sync::MutexGuard<'_, ()>, MstError> {
        let lock = self.locks.get(index).ok_or(MstError::InvariantViolation {
            invariant: "root lock index must be within the lock table",
            index,
            lock_count: self.locks.len(),
        })?;

        lock.lock().map_err(|_| MstError::LockPoisoned {
            resource: "union-find root lock",
        })
    }

    /// Link two currently locked roots using rank and deterministic ties.
    fn union_roots(&self, left_root: usize, right_root: usize) -> Result<bool, MstError> {
        let left_rank = self.rank_at(left_root)?.load(Ordering::Relaxed);
        let right_rank = self.rank_at(right_root)?.load(Ordering::Relaxed);

        let (parent, child) = choose_parent_child(left_root, right_root, left_rank, right_rank);

        self.parent_at(child)?.store(parent, Ordering::Release);

        if left_rank == right_rank {
            self.rank_at(parent)?.fetch_add(1, Ordering::Relaxed);
        }

        self.components.fetch_sub(1, Ordering::AcqRel);
        Ok(true)
    }

    /// Report whether a node currently points to itself.
    fn is_root(&self, node: usize) -> Result<bool, MstError> {
        Ok(self.parent_at(node)?.load(Ordering::Acquire) == node)
    }

    /// Find a root while applying lock-free path halving.
    fn find(&self, node: usize) -> Result<usize, MstError> {
        let mut current = node;
        loop {
            let parent = self.parent_at(current)?.load(Ordering::Acquire);

            if parent == current {
                return Ok(current);
            }

            let grandparent = self.parent_at(parent)?.load(Ordering::Acquire);

            if grandparent != parent {
                self.parent_at(current)?
                    .store(grandparent, Ordering::Release);
            }

            current = parent;
        }
    }

    /// Return the atomic parent slot for a validated node identifier.
    fn parent_at(&self, index: usize) -> Result<&AtomicUsize, MstError> {
        self.parents.get(index).ok_or(MstError::InvariantViolation {
            invariant: "union-find parent index must be within the parent table",
            index,
            lock_count: self.locks.len(),
        })
    }

    /// Return the atomic rank slot for a validated node identifier.
    fn rank_at(&self, index: usize) -> Result<&AtomicUsize, MstError> {
        self.ranks.get(index).ok_or(MstError::InvariantViolation {
            invariant: "union-find rank index must be within the rank table",
            index,
            lock_count: self.locks.len(),
        })
    }

    /// Return a node's current root for partition assertions.
    #[cfg(test)]
    pub(super) fn root_of(&self, node: usize) -> Result<usize, MstError> {
        let mut current = node;
        loop {
            let parent = self.parent_at(current)?.load(Ordering::Acquire);
            if parent == current {
                return Ok(current);
            }
            current = parent;
        }
    }
}

/// Order two roots consistently to avoid lock-order inversions.
const fn lock_order(first: usize, second: usize) -> (usize, usize) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

/// Choose parent and child roots using rank then deterministic identifier ties.
const fn choose_parent_child(
    left_root: usize,
    right_root: usize,
    left_rank: usize,
    right_rank: usize,
) -> (usize, usize) {
    if left_rank > right_rank {
        return (left_root, right_root);
    }
    if right_rank > left_rank {
        return (right_root, left_root);
    }

    lock_order(left_root, right_root)
}

#[cfg(test)]
mod tests {
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
    fn concurrent_unions_match_sequential_partition(
        #[case] seed: u64,
        #[case] thread_count: usize,
    ) {
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
        let pivot = round * (MAX_THREAD_COUNT + 1);
        (0..thread_count)
            .map(|worker_index| orient_edge(rng, pivot, pivot + worker_index + 1))
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
}
