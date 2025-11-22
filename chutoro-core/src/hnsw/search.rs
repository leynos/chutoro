//! Layer search routines for the CPU HNSW graph.
//!
//! Implements greedy descent and best-first per-layer search whilst enforcing
//! finite distance invariants. Non-finite values are rejected before they can
//! pollute the traversal state.

use std::collections::{BinaryHeap, HashSet};

use crate::DataSource;

use super::{
    distance_cache::DistanceCache,
    error::HnswError,
    graph::{ExtendedSearchContext, NeighbourSearchContext, SearchContext},
    node::Node,
    types::Neighbour,
    validate::{validate_batch_distances, validate_distance},
};

use super::graph::Graph;

#[derive(Debug)]
struct SearchState {
    visited: HashSet<usize>,
    candidates: BinaryHeap<CandidateNeighbour>,
    best: BinaryHeap<BestNeighbour>,
    discovered: HashSet<usize>,
}

impl SearchState {
    fn new(entry: SearchNeighbour) -> Self {
        // Fallback when `ef` is not available at the call-site.
        Self::with_capacity(entry, 64)
    }

    fn with_capacity(entry: SearchNeighbour, ef: usize) -> Self {
        let queue_capacity = ef.max(1);
        let set_capacity = queue_capacity.saturating_mul(4);

        let visited = HashSet::with_capacity(set_capacity);

        let mut candidates = BinaryHeap::with_capacity(queue_capacity);
        candidates.push(CandidateNeighbour(entry));

        let mut best = BinaryHeap::with_capacity(queue_capacity);
        best.push(BestNeighbour(entry));

        let mut discovered = HashSet::with_capacity(set_capacity);
        discovered.insert(entry.id);

        Self {
            visited,
            candidates,
            best,
            discovered,
        }
    }

    fn pop_candidate(&mut self) -> Option<SearchNeighbour> {
        self.candidates
            .pop()
            .map(|CandidateNeighbour(neighbour)| neighbour)
    }

    fn should_terminate(&self, ef: usize, candidate_distance: f32) -> bool {
        if self.best.len() < ef {
            return false;
        }

        self.best
            .peek()
            .is_some_and(|BestNeighbour(furthest)| candidate_distance >= furthest.distance)
    }

    fn mark_processed(&mut self, candidate: usize) -> bool {
        self.visited.insert(candidate)
    }

    fn discover(&mut self, candidate: usize) -> bool {
        self.discovered.insert(candidate)
    }

    fn try_enqueue(&mut self, candidate: SearchNeighbour, ef: usize) {
        let id = candidate.id;
        if self.visited.contains(&id) {
            return;
        }
        if self.best.len() >= ef
            && self
                .best
                .peek()
                .is_some_and(|BestNeighbour(furthest)| candidate.distance >= furthest.distance)
        {
            return;
        }

        self.candidates.push(CandidateNeighbour(candidate));
        self.best.push(BestNeighbour(candidate));
        self.enforce_capacity(ef);
    }

    fn enforce_capacity(&mut self, ef: usize) {
        while self.best.len() > ef {
            self.best.pop();
        }
    }

    fn finalise(self) -> Vec<Neighbour> {
        let mut neighbours: Vec<_> = self.best.into_vec();
        neighbours.sort_unstable();
        neighbours
            .into_iter()
            .map(|BestNeighbour(neighbour)| neighbour.into_public())
            .collect()
    }
}

/// Internal representation of a neighbour encountered during search enriched
/// with an insertion sequence for deterministic tie-breaking.
#[derive(Clone, Copy, Debug)]
struct SearchNeighbour {
    id: usize,
    distance: f32,
    sequence: u64,
}

impl SearchNeighbour {
    /// Builds a neighbour snapshot used by the search queues.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use crate::hnsw::search::SearchNeighbour;
    ///
    /// let neighbour = SearchNeighbour::new(5, 0.42, 7);
    /// assert_eq!(neighbour.id, 5);
    /// ```
    fn new(id: usize, distance: f32, sequence: u64) -> Self {
        Self {
            id,
            distance,
            sequence,
        }
    }

    /// Converts the search neighbour into the public [`Neighbour`] type.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use crate::hnsw::search::SearchNeighbour;
    ///
    /// let neighbour = SearchNeighbour::new(1, 0.1, 2);
    /// let public = neighbour.into_public();
    /// assert_eq!(public.id, 1);
    /// ```
    fn into_public(self) -> Neighbour {
        Neighbour {
            id: self.id,
            distance: self.distance,
        }
    }
}

fn compare_neighbours(left: &SearchNeighbour, right: &SearchNeighbour) -> std::cmp::Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.id.cmp(&right.id))
        .then_with(|| left.sequence.cmp(&right.sequence))
}

macro_rules! impl_neighbour_wrapper {
    ($name:ident, $cmp:expr) => {
        impl Eq for $name {}

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                $cmp(&self.0, &other.0) == std::cmp::Ordering::Equal
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                $cmp(&self.0, &other.0)
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
    };
}

#[derive(Clone, Copy, Debug)]
struct CandidateNeighbour(SearchNeighbour);

impl_neighbour_wrapper!(
    CandidateNeighbour,
    |left: &SearchNeighbour, right: &SearchNeighbour| { compare_neighbours(right, left) }
);

#[derive(Clone, Copy, Debug)]
struct BestNeighbour(SearchNeighbour);

impl_neighbour_wrapper!(BestNeighbour, compare_neighbours);

/// Bundles the optional distance cache and data source used to validate
/// distances during search.
#[derive(Clone, Copy, Debug)]
struct SearchInputs<'a, D: DataSource + Sync> {
    cache: Option<&'a DistanceCache>,
    source: &'a D,
}

impl<'a, D: DataSource + Sync> SearchInputs<'a, D> {
    /// Creates a new wrapper around the cache and data source used by search.
    fn new(cache: Option<&'a DistanceCache>, source: &'a D) -> Self {
        Self { cache, source }
    }

    /// Validates and returns the distance between two nodes.
    fn validate_distance(&self, left: usize, right: usize) -> Result<f32, HnswError> {
        validate_distance(self.cache, self.source, left, right)
    }

    /// Validates and returns the distances from the query node to candidates.
    fn validate_batch(&self, query: usize, candidates: &[usize]) -> Result<Vec<f32>, HnswError> {
        validate_batch_distances(self.cache, self.source, query, candidates)
    }
}

#[derive(Debug)]
pub(crate) struct LayerSearcher<'graph> {
    graph: &'graph Graph,
}

impl<'graph> LayerSearcher<'graph> {
    pub(super) fn new(graph: &'graph Graph) -> Self {
        Self { graph }
    }

    pub(super) fn greedy_search_layer<D: DataSource + Sync>(
        &self,
        cache: Option<&DistanceCache>,
        source: &D,
        ctx: SearchContext,
    ) -> Result<usize, HnswError> {
        let inputs = SearchInputs::new(cache, source);
        let mut current = ctx.entry();
        let mut current_dist = inputs.validate_distance(ctx.query(), current)?;
        let mut improved = true;
        while improved {
            improved = false;
            let Some(node) = self.graph.node(current) else {
                return Err(HnswError::GraphInvariantViolation {
                    message: format!(
                        "node {current} missing during greedy search at level {}",
                        ctx.level()
                    ),
                });
            };

            let search_ctx = ctx.with_distance(current_dist);
            let next = self.find_better_neighbour(&inputs, search_ctx, node)?;

            if let Some(neighbour) = next {
                current = neighbour.id;
                current_dist = neighbour.distance;
                improved = true;
            }
        }
        Ok(current)
    }

    fn find_better_neighbour<D: DataSource + Sync>(
        &self,
        inputs: &SearchInputs<'_, D>,
        ctx: NeighbourSearchContext,
        node: &Node,
    ) -> Result<Option<SearchNeighbour>, HnswError> {
        let neighbours = node.neighbours(ctx.level());
        if neighbours.is_empty() {
            return Ok(None);
        }

        let distances = inputs.validate_batch(ctx.query(), neighbours)?;
        if let Some((best_id, best_dist)) = neighbours
            .iter()
            .copied()
            .zip(distances)
            .min_by(|a, b| a.1.total_cmp(&b.1))
        {
            if best_dist < ctx.current_dist {
                let sequence = self.sequence_for_node(best_id, "greedy search")?;
                return Ok(Some(SearchNeighbour::new(best_id, best_dist, sequence)));
            }
        }
        Ok(None)
    }

    fn sequence_or_invariant(&self, node: usize, message: String) -> Result<u64, HnswError> {
        self.graph
            .node_sequence(node)
            .ok_or(HnswError::GraphInvariantViolation { message })
    }

    fn sequence_for_node(&self, node: usize, context: &str) -> Result<u64, HnswError> {
        self.sequence_or_invariant(
            node,
            format!("sequence missing for node {node} during {context}"),
        )
    }

    pub(super) fn search_layer<D: DataSource + Sync>(
        &self,
        cache: Option<&DistanceCache>,
        source: &D,
        ctx: ExtendedSearchContext,
    ) -> Result<Vec<Neighbour>, HnswError> {
        let inputs = SearchInputs::new(cache, source);
        let entry = ctx.entry();
        let entry_dist = inputs.validate_distance(ctx.query(), entry)?;
        let entry_sequence = self.sequence_for_node(entry, "layer search")?;

        let entry_neighbour = SearchNeighbour::new(entry, entry_dist, entry_sequence);

        let mut state = if ctx.ef == 0 {
            SearchState::new(entry_neighbour)
        } else {
            SearchState::with_capacity(entry_neighbour, ctx.ef)
        };

        while let Some(candidate) = state.pop_candidate() {
            if state.should_terminate(ctx.ef, candidate.distance) {
                break;
            }

            let Some(node) = self.graph.node(candidate.id) else {
                return Err(HnswError::GraphInvariantViolation {
                    message: format!(
                        "node {} missing during layer search at level {}",
                        candidate.id,
                        ctx.level()
                    ),
                });
            };

            if !state.mark_processed(candidate.id) {
                continue;
            }

            let fresh: Vec<_> = node
                .neighbours(ctx.level())
                .iter()
                .copied()
                .filter(|candidate| state.discover(*candidate))
                .collect();
            if fresh.is_empty() {
                continue;
            }

            let distances = inputs.validate_batch(ctx.query(), &fresh)?;
            for (candidate, distance) in fresh.into_iter().zip(distances.into_iter()) {
                let sequence = self.sequence_for_node(candidate, "layer expansion")?;
                state.try_enqueue(SearchNeighbour::new(candidate, distance, sequence), ctx.ef);
            }
        }
        Ok(state.finalise())
    }
}
