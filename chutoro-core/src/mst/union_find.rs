//! Concurrent union-find for the parallel Kruskal implementation.
//!
//! This union-find prioritises correctness and deadlock avoidance over maximum
//! throughput. It uses a striped lock table so disjoint unions can proceed in
//! parallel while maintaining a consistent lock ordering.

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
    lock_mask: usize,
}

impl ConcurrentUnionFind {
    pub(super) fn new(node_count: usize) -> Self {
        let mut parents = Vec::with_capacity(node_count);
        let mut ranks = Vec::with_capacity(node_count);
        for id in 0..node_count {
            parents.push(AtomicUsize::new(id));
            ranks.push(AtomicUsize::new(0));
        }

        let stripes = lock_stripes(node_count);
        let lock_mask = stripes.saturating_sub(1);

        let locks = (0..stripes).map(|_| Mutex::new(())).collect();

        Self {
            parents,
            ranks,
            components: AtomicUsize::new(node_count),
            locks,
            lock_mask,
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

            let (first_lock, second_lock) =
                lock_order(self.lock_index(left_root), self.lock_index(right_root));

            if first_lock == second_lock {
                let _guard = self.lock_stripe(first_lock)?;
                match self.try_union_after_lock(left, right)? {
                    UnionAttempt::Done(result) => return Ok(result),
                    UnionAttempt::Retry => continue,
                }
            }

            let _first_guard = self.lock_stripe(first_lock)?;
            let _second_guard = self.lock_stripe(second_lock)?;
            match self.try_union_after_lock(left, right)? {
                UnionAttempt::Done(result) => return Ok(result),
                UnionAttempt::Retry => continue,
            }
        }
    }

    fn lock_stripe(&self, index: usize) -> Result<std::sync::MutexGuard<'_, ()>, MstError> {
        self.locks
            .get(index)
            .ok_or(MstError::LockPoisoned {
                resource: "union-find striped lock table",
            })?
            .lock()
            .map_err(|_| MstError::LockPoisoned {
                resource: "union-find striped lock",
            })
    }

    fn try_union_after_lock(&self, left: usize, right: usize) -> Result<UnionAttempt, MstError> {
        let left_root = self.find(left);
        let right_root = self.find(right);

        if left_root == right_root {
            return Ok(UnionAttempt::Done(false));
        }

        if !self.is_root(left_root) || !self.is_root(right_root) {
            return Ok(UnionAttempt::Retry);
        }

        Ok(UnionAttempt::Done(self.union_roots(left_root, right_root)?))
    }

    fn union_roots(&self, left_root: usize, right_root: usize) -> Result<bool, MstError> {
        let left_rank = self.ranks[left_root].load(Ordering::Relaxed);
        let right_rank = self.ranks[right_root].load(Ordering::Relaxed);

        let (parent, child) = if left_rank > right_rank {
            (left_root, right_root)
        } else if right_rank > left_rank {
            (right_root, left_root)
        } else if left_root <= right_root {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };

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

    fn lock_index(&self, node: usize) -> usize {
        node & self.lock_mask
    }
}

enum UnionAttempt {
    Done(bool),
    Retry,
}

fn lock_order(first: usize, second: usize) -> (usize, usize) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

fn lock_stripes(node_count: usize) -> usize {
    const MAX_STRIPES: usize = 4096;
    if node_count <= 1 {
        return 1;
    }

    let stripes = node_count.next_power_of_two();
    stripes.min(MAX_STRIPES)
}
