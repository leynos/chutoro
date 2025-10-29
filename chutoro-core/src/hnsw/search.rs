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
    types::{Neighbour, RankedNeighbour, ReverseNeighbour},
    validate::{validate_batch_distances, validate_distance},
};

use super::graph::Graph;

#[derive(Debug)]
struct SearchState {
    visited: HashSet<usize>,
    candidates: BinaryHeap<ReverseNeighbour>,
    best: BinaryHeap<RankedNeighbour>,
    discovered: HashSet<usize>,
}

impl SearchState {
    fn new(entry: usize, distance: f32, sequence: u64) -> Self {
        // Fallback when `ef` is not available at the call-site.
        Self::with_capacity(entry, distance, sequence, 64)
    }

    fn with_capacity(entry: usize, distance: f32, sequence: u64, ef: usize) -> Self {
        let queue_capacity = ef.max(1);
        let set_capacity = queue_capacity.saturating_mul(4);

        let visited = HashSet::with_capacity(set_capacity);

        let mut candidates = BinaryHeap::with_capacity(queue_capacity);
        candidates.push(ReverseNeighbour::new(entry, distance, sequence));

        let mut best = BinaryHeap::with_capacity(queue_capacity);
        best.push(RankedNeighbour::new(entry, distance, sequence));

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
        if self.best.len() < ef {
            return false;
        }

        self.best
            .peek()
            .is_some_and(|furthest| candidate_distance >= furthest.neighbour.distance)
    }

    fn mark_processed(&mut self, candidate: usize) -> bool {
        self.visited.insert(candidate)
    }

    fn discover(&mut self, candidate: usize) -> bool {
        self.discovered.insert(candidate)
    }

    fn try_enqueue(&mut self, candidate: RankedNeighbour, ef: usize) {
        let id = candidate.neighbour.id;
        if self.visited.contains(&id) {
            return;
        }
        if self.best.len() >= ef
            && self
                .best
                .peek()
                .is_some_and(|furthest| candidate.neighbour.distance >= furthest.neighbour.distance)
        {
            return;
        }

        self.candidates
            .push(ReverseNeighbour::from_ranked(candidate));
        self.best.push(candidate);
        self.enforce_capacity(ef);
    }

    fn enforce_capacity(&mut self, ef: usize) {
        while self.best.len() > ef {
            self.best.pop();
        }
    }

    fn finalise(self) -> Vec<Neighbour> {
        let mut neighbours = self.best.into_vec();
        neighbours.sort_unstable();
        neighbours
            .into_iter()
            .map(RankedNeighbour::into_neighbour)
            .collect()
    }
}

#[derive(Debug)]
pub(super) struct LayerSearcher<'graph> {
    graph: &'graph Graph,
}

struct NeighbourLookup<'a, D: DataSource + Sync> {
    cache: Option<&'a DistanceCache>,
    source: &'a D,
    ctx: NeighbourSearchContext,
    node: &'a Node,
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
        let mut current = ctx.entry();
        let mut current_dist = validate_distance(cache, source, ctx.query(), current)?;
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
            let next = self.find_better_neighbour(NeighbourLookup {
                cache,
                source,
                ctx: search_ctx,
                node,
            })?;

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
        lookup: NeighbourLookup<'_, D>,
    ) -> Result<Option<(usize, f32)>, HnswError> {
        let NeighbourLookup {
            cache,
            source,
            ctx,
            node,
        } = lookup;

        let neighbours = node.neighbours(ctx.level());
        if neighbours.is_empty() {
            return Ok(None);
        }

        let distances = validate_batch_distances(cache, source, ctx.query(), neighbours)?;
        if let Some((best_id, best_dist)) = neighbours
            .iter()
            .copied()
            .zip(distances)
            .min_by(|a, b| a.1.total_cmp(&b.1))
        {
            if best_dist < ctx.current_dist {
                self.sequence_or_invariant(
                    best_id,
                    format!("sequence missing for node {best_id} during greedy search"),
                )?;
                return Ok(Some((best_id, best_dist)));
            }
        }
        Ok(None)
    }

    fn sequence_or_invariant(&self, node: usize, message: String) -> Result<u64, HnswError> {
        self.graph
            .node_sequence(node)
            .ok_or(HnswError::GraphInvariantViolation { message })
    }

    pub(super) fn search_layer<D: DataSource + Sync>(
        &self,
        cache: Option<&DistanceCache>,
        source: &D,
        ctx: ExtendedSearchContext,
    ) -> Result<Vec<Neighbour>, HnswError> {
        let entry = ctx.entry();
        let entry_dist = validate_distance(cache, source, ctx.query(), entry)?;
        let entry_sequence =
            self.graph
                .node_sequence(entry)
                .ok_or_else(|| HnswError::GraphInvariantViolation {
                    message: format!("sequence missing for node {entry} during layer search"),
                })?;

        let mut state = if ctx.ef == 0 {
            SearchState::new(entry, entry_dist, entry_sequence)
        } else {
            SearchState::with_capacity(entry, entry_dist, entry_sequence, ctx.ef)
        };

        while let Some(ReverseNeighbour { inner }) = state.pop_candidate() {
            if state.should_terminate(ctx.ef, inner.neighbour.distance) {
                break;
            }

            let Some(node) = self.graph.node(inner.neighbour.id) else {
                return Err(HnswError::GraphInvariantViolation {
                    message: format!(
                        "node {} missing during layer search at level {}",
                        inner.neighbour.id,
                        ctx.level()
                    ),
                });
            };

            if !state.mark_processed(inner.neighbour.id) {
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

            let distances = validate_batch_distances(cache, source, ctx.query(), &fresh)?;
            for (candidate, distance) in fresh.into_iter().zip(distances.into_iter()) {
                let sequence = self.sequence_or_invariant(
                    candidate,
                    format!("sequence missing for node {candidate} during layer expansion"),
                )?;
                state.try_enqueue(RankedNeighbour::new(candidate, distance, sequence), ctx.ef);
            }
        }
        Ok(state.finalise())
    }
}
