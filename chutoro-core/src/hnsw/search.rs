//! Layer search routines for the CPU HNSW graph.
//!
//! Implements greedy descent and best-first per-layer search whilst enforcing
//! finite distance invariants. Non-finite values are rejected before they can
//! pollute the traversal state.

use std::collections::{BinaryHeap, HashSet};

use crate::DataSource;

use super::{
    error::HnswError,
    graph::{ExtendedSearchContext, NeighbourSearchContext, SearchContext},
    node::Node,
    types::{Neighbour, ReverseNeighbour},
    validate::{validate_batch_distances, validate_distance},
};

use super::graph::Graph;

#[derive(Debug)]
struct SearchState {
    visited: HashSet<usize>,
    candidates: BinaryHeap<ReverseNeighbour>,
    best: BinaryHeap<Neighbour>,
    discovered: HashSet<usize>,
}

impl SearchState {
    fn new(entry: usize, distance: f32) -> Self {
        // Fallback when `ef` is not available at the call-site.
        Self::with_capacity(entry, distance, 64)
    }

    fn with_capacity(entry: usize, distance: f32, ef: usize) -> Self {
        let queue_capacity = ef.max(1);
        let set_capacity = queue_capacity.saturating_mul(4);

        let visited = HashSet::with_capacity(set_capacity);

        let mut candidates = BinaryHeap::with_capacity(queue_capacity);
        candidates.push(ReverseNeighbour::new(entry, distance));

        let mut best = BinaryHeap::with_capacity(queue_capacity);
        best.push(Neighbour {
            id: entry,
            distance,
        });

        let mut discovered = HashSet::with_capacity(set_capacity);
        discovered.insert(entry);

        Self {
            visited,
            candidates,
            best,
            discovered,
        }
    }

    fn pop_candidate(&mut self) -> Option<ReverseNeighbour> {
        self.candidates.pop()
    }

    fn should_terminate(&self, ef: usize, candidate_distance: f32) -> bool {
        self.best.len() >= ef
            && self
                .best
                .peek()
                .is_some_and(|furthest| candidate_distance > furthest.distance)
    }

    fn mark_processed(&mut self, candidate: usize) -> bool {
        self.visited.insert(candidate)
    }

    fn discover(&mut self, candidate: usize) -> bool {
        self.discovered.insert(candidate)
    }

    fn try_enqueue(&mut self, candidate: usize, distance: f32, ef: usize) {
        if self.visited.contains(&candidate) {
            return;
        }
        if self.best.len() >= ef
            && self
                .best
                .peek()
                .is_some_and(|furthest| distance > furthest.distance)
        {
            return;
        }

        self.candidates
            .push(ReverseNeighbour::new(candidate, distance));
        self.best.push(Neighbour {
            id: candidate,
            distance,
        });
        self.enforce_capacity(ef);
    }

    fn enforce_capacity(&mut self, ef: usize) {
        while self.best.len() > ef {
            self.best.pop();
        }
    }

    fn finalise(self) -> Vec<Neighbour> {
        let mut neighbours = self.best.into_vec();
        neighbours.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        neighbours
    }
}

#[derive(Debug)]
pub(super) struct LayerSearcher<'graph> {
    graph: &'graph Graph,
}

impl<'graph> LayerSearcher<'graph> {
    pub(super) fn new(graph: &'graph Graph) -> Self {
        Self { graph }
    }

    pub(super) fn greedy_search_layer<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: SearchContext,
    ) -> Result<usize, HnswError> {
        let mut current = ctx.entry();
        let mut current_dist = validate_distance(source, ctx.query(), current)?;
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
            let next = self.find_better_neighbour(source, search_ctx, node)?;

            if let Some((next, next_dist)) = next {
                current = next;
                current_dist = next_dist;
                improved = true;
            }
        }
        Ok(current)
    }

    fn find_better_neighbour<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: NeighbourSearchContext,
        node: &Node,
    ) -> Result<Option<(usize, f32)>, HnswError> {
        let neighbours = node.neighbours(ctx.level());
        if neighbours.is_empty() {
            return Ok(None);
        }

        let distances = validate_batch_distances(source, ctx.query(), neighbours)?;
        if let Some((best_id, best_dist)) = neighbours
            .iter()
            .copied()
            .zip(distances)
            .min_by(|a, b| a.1.total_cmp(&b.1))
        {
            return Ok((best_dist < ctx.current_dist).then_some((best_id, best_dist)));
        }
        Ok(None)
    }

    pub(super) fn search_layer<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: ExtendedSearchContext,
    ) -> Result<Vec<Neighbour>, HnswError> {
        let entry = ctx.entry();
        let entry_dist = validate_distance(source, ctx.query(), entry)?;

        let mut state = if ctx.ef == 0 {
            SearchState::new(entry, entry_dist)
        } else {
            SearchState::with_capacity(entry, entry_dist, ctx.ef)
        };

        while let Some(ReverseNeighbour { inner }) = state.pop_candidate() {
            if state.should_terminate(ctx.ef, inner.distance) {
                break;
            }

            let Some(node) = self.graph.node(inner.id) else {
                return Err(HnswError::GraphInvariantViolation {
                    message: format!(
                        "node {} missing during layer search at level {}",
                        inner.id,
                        ctx.level()
                    ),
                });
            };

            if !state.mark_processed(inner.id) {
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

            let distances = validate_batch_distances(source, ctx.query(), &fresh)?;
            for (candidate, distance) in fresh.into_iter().zip(distances.into_iter()) {
                state.try_enqueue(candidate, distance, ctx.ef);
            }
        }
        Ok(state.finalise())
    }
}
