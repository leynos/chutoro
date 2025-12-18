//! Concurrent union-find for the parallel Kruskal implementation.
//!
//! This union-find prioritises correctness and deadlock avoidance over maximum
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

pub(super) struct ConcurrentUnionFind {
    parents: Vec<AtomicUsize>,
    ranks: Vec<AtomicUsize>,
    components: AtomicUsize,
    locks: Vec<Mutex<()>>,
}

impl ConcurrentUnionFind {
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

    pub(super) fn components(&self) -> usize {
        self.components.load(Ordering::Acquire)
    }

    pub(super) fn try_union(&self, left: usize, right: usize) -> Result<bool, MstError> {
        loop {
            let left_root = self.find(left);
            let right_root = self.find(right);

            if left_root == right_root {
                return Ok(false);
            }

            let lock_pair = lock_order(left_root, right_root);
            let (first_lock, second_lock) = lock_pair;
            let _first_guard = self.lock_root(first_lock)?;
            let _second_guard = (second_lock != first_lock)
                .then(|| self.lock_root(second_lock))
                .transpose()?;

            let left_root = self.find(left);
            let right_root = self.find(right);

            if left_root == right_root {
                return Ok(false);
            }

            if lock_order(left_root, right_root) != lock_pair {
                continue;
            }

            if !self.is_root(left_root) || !self.is_root(right_root) {
                continue;
            }

            return self.union_roots(left_root, right_root);
        }
    }

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

    fn union_roots(&self, left_root: usize, right_root: usize) -> Result<bool, MstError> {
        let left_rank = self.ranks[left_root].load(Ordering::Relaxed);
        let right_rank = self.ranks[right_root].load(Ordering::Relaxed);

        let (parent, child) = choose_parent_child(left_root, right_root, left_rank, right_rank);

        self.parents[child].store(parent, Ordering::Release);

        if left_rank == right_rank {
            self.ranks[parent].fetch_add(1, Ordering::Relaxed);
        }

        self.components.fetch_sub(1, Ordering::AcqRel);
        Ok(true)
    }

    fn is_root(&self, node: usize) -> bool {
        self.parents[node].load(Ordering::Acquire) == node
    }

    fn find(&self, node: usize) -> usize {
        let mut current = node;
        loop {
            let parent = self.parents[current].load(Ordering::Acquire);

            if parent == current {
                return current;
            }

            let grandparent = self.parents[parent].load(Ordering::Acquire);

            if grandparent != parent {
                self.parents[current].store(grandparent, Ordering::Release);
            }

            current = parent;
        }
    }
}

fn lock_order(first: usize, second: usize) -> (usize, usize) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

fn choose_parent_child(
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
