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

        let mut visited = HashSet::new();
        visited.insert(entry);

        let mut candidates = BinaryHeap::new();
        candidates.push(ReverseNeighbour::new(entry, entry_dist));

        let mut best = BinaryHeap::new();
        best.push(Neighbour {
            id: entry,
            distance: entry_dist,
        });

        while let Some(ReverseNeighbour { inner }) = candidates.pop() {
            if Self::should_terminate_layer(&best, ctx.ef, inner.distance) {
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

            let fresh: Vec<_> = node
                .neighbours(ctx.level())
                .iter()
                .copied()
                .filter(|candidate| visited.insert(*candidate))
                .collect();
            if fresh.is_empty() {
                continue;
            }

            let distances = validate_batch_distances(source, ctx.query(), &fresh)?;
            let scored = fresh.into_iter().zip(distances.into_iter());
            Self::enqueue_candidates(&mut best, &mut candidates, ctx, scored);
        }
        let mut neighbours: Vec<_> = best.into_vec();
        neighbours.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        Ok(neighbours)
    }

    fn should_terminate_layer(
        best: &BinaryHeap<Neighbour>,
        ef: usize,
        candidate_distance: f32,
    ) -> bool {
        best.len() >= ef
            && best
                .peek()
                .is_some_and(|furthest| candidate_distance > furthest.distance)
    }

    fn should_enqueue_candidate(best: &BinaryHeap<Neighbour>, ef: usize, distance: f32) -> bool {
        best.len() < ef
            || best
                .peek()
                .is_some_and(|furthest| distance < furthest.distance)
    }

    fn enforce_layer_capacity(best: &mut BinaryHeap<Neighbour>, ef: usize) {
        if best.len() > ef {
            best.pop();
        }
    }

    fn enqueue_candidates<I>(
        best: &mut BinaryHeap<Neighbour>,
        candidates: &mut BinaryHeap<ReverseNeighbour>,
        ctx: ExtendedSearchContext,
        neighbours: I,
    ) where
        I: IntoIterator<Item = (usize, f32)>,
    {
        for (candidate, distance) in neighbours {
            if !Self::should_enqueue_candidate(best, ctx.ef, distance) {
                continue;
            }

            candidates.push(ReverseNeighbour::new(candidate, distance));
            best.push(Neighbour {
                id: candidate,
                distance,
            });
            Self::enforce_layer_capacity(best, ctx.ef);
        }
    }
}
