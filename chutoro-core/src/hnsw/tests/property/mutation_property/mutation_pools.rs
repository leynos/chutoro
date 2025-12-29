//! Mutation pools used by the stateful mutation property.

pub(super) struct MutationPools {
    inserted: Vec<usize>,
    available: Vec<usize>,
}

impl MutationPools {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            inserted: Vec::new(),
            available: (0..capacity).collect(),
        }
    }

    pub(super) fn bootstrap(&mut self, count: usize) -> Vec<usize> {
        let take = count.min(self.available.len());
        let seeded: Vec<usize> = self.available.drain(0..take).collect();
        self.inserted.extend(&seeded);
        seeded
    }

    pub(super) fn select_available(&self, hint: u16) -> Option<usize> {
        Self::select_from(&self.available, hint)
    }

    pub(super) fn select_inserted(&self, hint: u16) -> Option<usize> {
        Self::select_from(&self.inserted, hint)
    }

    pub(super) fn mark_inserted(&mut self, node: usize) {
        if Self::remove_value(&mut self.available, node) {
            self.inserted.push(node);
        }
    }

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
}
