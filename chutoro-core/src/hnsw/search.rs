use std::collections::{BinaryHeap, HashSet};

use crate::DataSource;

use super::{
    error::HnswError,
    graph::{
        ExtendedSearchContext, NeighbourSearchContext, ProcessNodeContext, ScoredCandidates,
        SearchContext,
    },
    types::{Neighbour, ReverseNeighbour},
    validate::{validate_batch_distances, validate_distance},
};

use super::graph::Graph;

impl Graph {
    pub(crate) fn greedy_search_layer<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: SearchContext,
    ) -> Result<usize, HnswError> {
        let mut current = ctx.entry;
        let mut current_dist = validate_distance(source, ctx.query, current)?;
        let mut improved = true;
        while improved {
            improved = false;
            let Some(node) = self.node(current) else {
                return Err(HnswError::GraphInvariantViolation {
                    message: format!(
                        "node {current} missing during greedy search at level {}",
                        ctx.level
                    ),
                });
            };

            let search_ctx = NeighbourSearchContext {
                query: ctx.query,
                level: ctx.level,
                current_dist,
            };
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
        node: &super::node::Node,
    ) -> Result<Option<(usize, f32)>, HnswError> {
        let neighbours = node.neighbours(ctx.level);
        if neighbours.is_empty() {
            return Ok(None);
        }

        let distances = validate_batch_distances(source, ctx.query, neighbours)?;
        for (candidate, candidate_dist) in neighbours.iter().copied().zip(distances) {
            if candidate_dist < ctx.current_dist {
                return Ok(Some((candidate, candidate_dist)));
            }
        }
        Ok(None)
    }

    pub(crate) fn search_layer<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: ExtendedSearchContext,
    ) -> Result<Vec<Neighbour>, HnswError> {
        let entry_dist = validate_distance(source, ctx.query, ctx.entry)?;
        let mut state = LayerSearchState::new(ctx.entry, entry_dist);

        while let Some(ReverseNeighbour { inner }) = state.candidates.pop() {
            if self.should_terminate_search(&state.best, ctx.ef, inner.distance) {
                break;
            }

            let process_ctx = ProcessNodeContext {
                query: ctx.query,
                level: ctx.level,
                ef: ctx.ef,
                node_id: inner.id,
            };
            self.process_neighbours(source, process_ctx, &mut state)?;
        }

        Ok(self.finalize_results(state.into_best()))
    }

    fn should_terminate_search(
        &self,
        best: &BinaryHeap<Neighbour>,
        ef: usize,
        candidate_distance: f32,
    ) -> bool {
        best.len() >= ef
            && best
                .peek()
                .map(|furthest| candidate_distance > furthest.distance)
                .unwrap_or(false)
    }

    fn process_neighbours<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: ProcessNodeContext,
        state: &mut LayerSearchState,
    ) -> Result<(), HnswError> {
        let Some(node) = self.node(ctx.node_id) else {
            return Ok(());
        };

        let fresh = self.collect_unvisited_neighbours(node, ctx.level, state);
        if fresh.is_empty() {
            return Ok(());
        }

        let distances = validate_batch_distances(source, ctx.query, &fresh)?;
        let scored = ScoredCandidates::new(fresh, distances);
        self.update_search_state_with_candidates(scored, ctx.ef, state);

        Ok(())
    }

    fn collect_unvisited_neighbours(
        &self,
        node: &super::node::Node,
        level: usize,
        state: &mut LayerSearchState,
    ) -> Vec<usize> {
        node.neighbours(level)
            .iter()
            .copied()
            .filter(|candidate| state.visited.insert(*candidate))
            .collect()
    }

    fn update_search_state_with_candidates(
        &self,
        scored: ScoredCandidates,
        ef: usize,
        state: &mut LayerSearchState,
    ) {
        for (candidate, candidate_distance) in scored.into_iter() {
            if !self.should_add_candidate(&state.best, ef, candidate_distance) {
                continue;
            }

            state
                .candidates
                .push(ReverseNeighbour::new(candidate, candidate_distance));
            state.best.push(Neighbour {
                id: candidate,
                distance: candidate_distance,
            });
            if state.best.len() > ef {
                state.best.pop();
            }
        }
    }

    fn should_add_candidate(
        &self,
        best: &BinaryHeap<Neighbour>,
        ef: usize,
        candidate_distance: f32,
    ) -> bool {
        best.len() < ef
            || best
                .peek()
                .map(|furthest| candidate_distance < furthest.distance)
                .unwrap_or(true)
    }

    fn finalize_results(&self, best: BinaryHeap<Neighbour>) -> Vec<Neighbour> {
        let mut neighbours: Vec<_> = best.into_vec();
        neighbours.sort_unstable();
        neighbours
    }
}

struct LayerSearchState {
    visited: HashSet<usize>,
    candidates: BinaryHeap<ReverseNeighbour>,
    best: BinaryHeap<Neighbour>,
}

impl LayerSearchState {
    fn new(entry: usize, entry_distance: f32) -> Self {
        let mut visited = HashSet::new();
        visited.insert(entry);

        let mut candidates = BinaryHeap::new();
        candidates.push(ReverseNeighbour::new(entry, entry_distance));

        let mut best = BinaryHeap::new();
        best.push(Neighbour {
            id: entry,
            distance: entry_distance,
        });

        Self {
            visited,
            candidates,
            best,
        }
    }

    fn into_best(self) -> BinaryHeap<Neighbour> {
        self.best
    }
}
