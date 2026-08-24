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
