//! Tracks node lifecycle for stateful HNSW property tests.
//!
//! The pool maintains two mutually exclusive collections of node identifiers:
//! `inserted` for nodes currently present in the HNSW index, and `available` for
//! nodes eligible for insertion (either never used or previously deleted).
//!
//! The `available` pool is kept sorted so property tests can select nodes
//! deterministically. The mutation operations (`mark_inserted`, `mark_deleted`)
//! preserve the invariant that every node belongs to exactly one pool.

/// Tracks node membership for stateful property tests with two mutually
/// exclusive pools.
///
/// Inserted nodes are currently present in the HNSW index, while available nodes
/// are ready to be inserted during mutation steps.
pub(super) struct MutationPools {
    /// Nodes currently present in the HNSW index.
    inserted: Vec<usize>,
    /// Nodes available for insertion, kept sorted for deterministic selection.
    available: Vec<usize>,
}

impl MutationPools {
    /// Creates a new pool with the given capacity of node identifiers.
    ///
    /// # Parameters
    /// - `capacity`: The total number of nodes available to distribute between
    ///   the inserted and available pools.
    ///
    /// # Returns
    /// A pool seeded with all node identifiers in the available set.
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            inserted: Vec::new(),
            available: (0..capacity).collect(),
        }
    }

    /// Moves up to `count` nodes from available into inserted and returns them.
    ///
    /// # Parameters
    /// - `count`: The maximum number of nodes to move into the inserted pool.
    ///
    /// # Returns
    /// The list of node identifiers promoted into the inserted pool.
    ///
    /// # Invariants
    /// The available pool remains sorted after draining.
    pub(super) fn bootstrap(&mut self, count: usize) -> Vec<usize> {
        let take = count.min(self.available.len());
        let seeded: Vec<usize> = self.available.drain(0..take).collect();
        self.inserted.extend(&seeded);
        seeded
    }

    /// Selects a node from the available pool using the provided hint.
    ///
    /// # Parameters
    /// - `hint`: A value used to deterministically index into the pool.
    ///
    /// # Returns
    /// The chosen node identifier, or `None` if the pool is empty.
    pub(super) fn select_available(&self, hint: u16) -> Option<usize> {
        Self::select_from(&self.available, hint)
    }

    /// Selects a node from the inserted pool using the provided hint.
    ///
    /// # Parameters
    /// - `hint`: A value used to deterministically index into the pool.
    ///
    /// # Returns
    /// The chosen node identifier, or `None` if the pool is empty.
    pub(super) fn select_inserted(&self, hint: u16) -> Option<usize> {
        Self::select_from(&self.inserted, hint)
    }

    /// Marks a node as inserted if it exists in the available pool.
    ///
    /// # Parameters
    /// - `node`: The node identifier to move into the inserted pool.
    ///
    /// # Invariants
    /// The available pool remains sorted for deterministic selection.
    pub(super) fn mark_inserted(&mut self, node: usize) {
        if Self::remove_value(&mut self.available, node) {
            self.inserted.push(node);
        }
    }

    /// Marks a node as deleted by returning it to the available pool.
    ///
    /// # Parameters
    /// - `node`: The node identifier to move into the available pool.
    ///
    /// # Invariants
    /// The available pool is kept sorted for deterministic selection.
    pub(super) fn mark_deleted(&mut self, node: usize) {
        if Self::remove_value(&mut self.inserted, node) {
            self.available.push(node);
            // Keep available sorted so selection remains deterministic for tests.
            self.available.sort_unstable();
        }
    }

    fn select_from(list: &[usize], hint: u16) -> Option<usize> {
        if list.is_empty() {
            None
        } else {
            let idx = usize::from(hint) % list.len();
            list.get(idx).copied()
        }
    }

    fn remove_value(list: &mut Vec<usize>, value: usize) -> bool {
        if let Some(position) = list.iter().position(|&candidate| candidate == value) {
            list.remove(position);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MutationPools;
    use rstest::rstest;

    #[rstest]
    fn mutation_pools_track_membership() {
        let mut pools = MutationPools::new(4);
        let seeded = pools.bootstrap(2);
        assert_eq!(seeded, vec![0, 1]);
        assert_eq!(pools.select_available(0), Some(2));
        assert_eq!(pools.select_inserted(0), Some(0));
        pools.mark_deleted(0);
        assert_eq!(pools.select_available(0), Some(0));
        pools.mark_inserted(2);
        assert_eq!(pools.select_available(0), Some(0));
    }

    #[rstest]
    fn mutation_pools_zero_capacity_is_empty() {
        let pools = MutationPools::new(0);
        assert_eq!(pools.select_available(0), None);
        assert_eq!(pools.select_inserted(0), None);
    }

    #[rstest]
    fn mutation_pools_bootstrap_clamps_to_capacity() {
        let mut pools = MutationPools::new(2);
        let seeded = pools.bootstrap(5);
        assert_eq!(seeded, vec![0, 1]);
        assert_eq!(pools.select_available(0), None);
        assert_eq!(pools.select_inserted(0), Some(0));
    }

    #[rstest]
    fn mutation_pools_select_from_empty_pools_returns_none() {
        let mut pools = MutationPools::new(1);
        pools.bootstrap(1);
        assert_eq!(pools.select_available(0), None);
        pools.mark_deleted(0);
        assert_eq!(pools.select_inserted(0), None);
    }

    #[rstest]
    fn mutation_pools_deleted_nodes_are_sorted() {
        let mut pools = MutationPools::new(5);
        pools.bootstrap(5);
        pools.mark_deleted(4);
        pools.mark_deleted(1);
        pools.mark_deleted(3);
        assert_eq!(pools.select_available(0), Some(1));
        assert_eq!(pools.select_available(1), Some(3));
        assert_eq!(pools.select_available(2), Some(4));
    }

    #[rstest]
    fn mutation_pools_marking_missing_nodes_is_noop() {
        let mut pools = MutationPools::new(3);
        let seeded = pools.bootstrap(1);
        assert_eq!(seeded, vec![0]);
        assert_eq!(pools.select_available(0), Some(1));
        assert_eq!(pools.select_available(1), Some(2));
        assert_eq!(pools.select_inserted(0), Some(0));

        pools.mark_inserted(0);
        pools.mark_inserted(9);
        pools.mark_deleted(2);
        pools.mark_deleted(9);

        assert_eq!(pools.select_available(0), Some(1));
        assert_eq!(pools.select_available(1), Some(2));
        assert_eq!(pools.select_inserted(0), Some(0));
    }
}
