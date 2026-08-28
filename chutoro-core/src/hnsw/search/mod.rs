//! Layer search routines for the CPU HNSW graph.
//!
//! Implements greedy descent and best-first per-layer search whilst enforcing
//! finite distance invariants. Non-finite values are rejected before they can
//! pollute the traversal state.

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

mod state;

use state::{SearchNeighbour, SearchState};

/// Bundles the optional distance cache and data source used to validate
/// distances during search.
#[derive(Clone, Copy, Debug)]
struct SearchInputs<'a, D: DataSource + Sync> {
    /// Optional cache used before querying the data source.
    cache: Option<&'a DistanceCache>,
    /// Data source that computes uncached distances.
    source: &'a D,
}

impl<'a, D: DataSource + Sync> SearchInputs<'a, D> {
    /// Creates a new wrapper around the cache and data source used by search.
    const fn new(cache: Option<&'a DistanceCache>, source: &'a D) -> Self {
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

/// Executes greedy and best-first traversals against one HNSW graph.
#[derive(Debug)]
pub(crate) struct LayerSearcher<'graph> {
    /// Graph whose nodes and insertion sequences are searched.
    graph: &'graph Graph,
}

impl<'graph> LayerSearcher<'graph> {
    /// Bind a layer searcher to one immutable graph.
    pub(super) const fn new(graph: &'graph Graph) -> Self {
        Self { graph }
    }

    /// Descend one layer by repeatedly choosing a strictly closer neighbour.
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

    /// Find the strictly closest neighbour available from a graph node.
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
            && best_dist < ctx.current_dist
        {
            let sequence = self.sequence_for_node(best_id, "greedy search")?;
            return Ok(Some(SearchNeighbour::new(best_id, best_dist, sequence)));
        }
        Ok(None)
    }

    /// Look up a node sequence or surface a graph-invariant violation.
    fn sequence_or_invariant(&self, node: usize, message: String) -> Result<u64, HnswError> {
        self.graph
            .node_sequence(node)
            .ok_or(HnswError::GraphInvariantViolation { message })
    }

    /// Build an invariant-aware insertion-sequence lookup message.
    fn sequence_for_node(&self, node: usize, context: &str) -> Result<u64, HnswError> {
        self.sequence_or_invariant(
            node,
            format!("sequence missing for node {node} during {context}"),
        )
    }

    /// Search one layer with bounded best-first exploration.
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
                .filter(|neighbour_id| state.discover(*neighbour_id))
                .collect();
            if fresh.is_empty() {
                continue;
            }

            let distances = inputs.validate_batch(ctx.query(), &fresh)?;
            for (neighbour_id, distance) in fresh.into_iter().zip(distances.into_iter()) {
                let sequence = self.sequence_for_node(neighbour_id, "layer expansion")?;
                state.try_enqueue(
                    SearchNeighbour::new(neighbour_id, distance, sequence),
                    ctx.ef,
                );
            }
        }
        Ok(state.finalise())
    }
}
