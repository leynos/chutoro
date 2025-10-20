//! Internal graph representation for the CPU HNSW implementation.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
};

use crate::DataSource;

use super::{error::HnswError, params::HnswParams};

#[derive(Clone, Copy, Debug)]
pub(crate) struct NodeContext {
    pub(crate) node: usize,
    pub(crate) level: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SearchContext {
    pub(crate) query: usize,
    pub(crate) entry: usize,
    pub(crate) level: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExtendedSearchContext {
    pub(crate) query: usize,
    pub(crate) entry: usize,
    pub(crate) level: usize,
    pub(crate) ef: usize,
}

#[derive(Clone, Copy, Debug)]
struct EdgeContext {
    level: usize,
    max_connections: usize,
}

#[derive(Clone, Copy, Debug)]
struct DescentContext {
    query: usize,
    entry: EntryPoint,
    target_level: usize,
}

#[derive(Clone, Copy, Debug)]
struct LayerPlanContext {
    query: usize,
    current: usize,
    target_level: usize,
    ef: usize,
}

#[derive(Clone, Copy, Debug)]
struct NeighbourSearchContext {
    query: usize,
    level: usize,
    current_dist: f32,
}

#[derive(Clone, Copy, Debug)]
struct ProcessNodeContext {
    query: usize,
    level: usize,
    ef: usize,
    node_id: usize,
}

#[derive(Clone, Copy, Debug)]
struct NodePair {
    from: usize,
    to: usize,
}

#[derive(Debug)]
pub(crate) struct ApplyContext<'a, D> {
    pub(crate) node: NodeContext,
    pub(crate) params: &'a HnswParams,
    pub(crate) plan: InsertionPlan,
    pub(crate) source: &'a D,
}

struct CandidateBatch {
    candidates: Vec<usize>,
    distances: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct Graph {
    nodes: Vec<Option<Node>>,
    entry: Option<EntryPoint>,
}

impl Graph {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: vec![None; capacity],
            entry: None,
        }
    }

    pub(crate) fn entry(&self) -> Option<EntryPoint> {
        self.entry
    }

    pub(crate) fn insert_first(&mut self, node: usize, level: usize) -> Result<(), HnswError> {
        let slot = self
            .nodes
            .get_mut(node)
            .ok_or_else(|| HnswError::InvalidParameters {
                reason: format!("node {node} is outside pre-allocated capacity"),
            })?;
        if slot.is_some() {
            return Err(HnswError::DuplicateNode { node });
        }
        *slot = Some(Node::new(level));
        self.entry = Some(EntryPoint { node, level });
        Ok(())
    }

    pub(crate) fn plan_insertion<D: DataSource + Sync>(
        &self,
        ctx: NodeContext,
        params: &HnswParams,
        source: &D,
    ) -> Result<InsertionPlan, HnswError> {
        let entry = self.entry.ok_or(HnswError::GraphEmpty)?;
        let target_level = ctx.level.min(entry.level);
        let descent_ctx = DescentContext {
            query: ctx.node,
            entry,
            target_level,
        };
        let current = self.greedy_descend_to_target_level(source, descent_ctx)?;
        let layer_ctx = LayerPlanContext {
            query: ctx.node,
            current,
            target_level,
            ef: params.ef_construction(),
        };
        let layers = self.build_layer_plans_from_target(source, layer_ctx)?;
        Ok(InsertionPlan { layers })
    }

    fn greedy_descend_to_target_level<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: DescentContext,
    ) -> Result<usize, HnswError> {
        let mut current = ctx.entry.node;
        if ctx.entry.level > ctx.target_level {
            for level in ((ctx.target_level + 1)..=ctx.entry.level).rev() {
                current = self.greedy_search_layer(
                    source,
                    SearchContext {
                        query: ctx.query,
                        entry: current,
                        level,
                    },
                )?;
            }
        }
        Ok(current)
    }

    fn build_layer_plans_from_target<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: LayerPlanContext,
    ) -> Result<Vec<LayerPlan>, HnswError> {
        let mut layers = Vec::with_capacity(ctx.target_level + 1);
        let mut current = ctx.current;
        for level in (0..=ctx.target_level).rev() {
            let candidates = self.search_layer(
                source,
                ExtendedSearchContext {
                    query: ctx.query,
                    entry: current,
                    level,
                    ef: ctx.ef,
                },
            )?;
            if let Some(best) = candidates.first() {
                current = best.id;
            }
            layers.push(LayerPlan {
                level,
                neighbours: candidates,
            });
        }
        layers.reverse();
        Ok(layers)
    }

    pub(crate) fn apply_insertion<D: DataSource + Sync>(
        &mut self,
        ctx: ApplyContext<'_, D>,
    ) -> Result<(), HnswError> {
        let ApplyContext {
            node,
            params,
            plan,
            source,
        } = ctx;
        let NodeContext { node, level } = node;
        let slot = self
            .nodes
            .get_mut(node)
            .ok_or_else(|| HnswError::InvalidParameters {
                reason: format!("node {} is outside pre-allocated capacity", node),
            })?;
        if slot.is_some() {
            return Err(HnswError::DuplicateNode { node });
        }
        *slot = Some(Node::new(level));
        if level > self.entry.map(|entry| entry.level).unwrap_or(0) {
            self.entry = Some(EntryPoint { node, level });
        }

        for layer in plan.layers.into_iter().filter(|layer| layer.level <= level) {
            let mut to_link = layer.neighbours;
            to_link.truncate(params.max_connections());
            let edge_ctx = EdgeContext {
                level: layer.level,
                max_connections: params.max_connections(),
            };
            let mut trim_targets = HashSet::new();
            for neighbour in to_link {
                let updated = self.link_bidirectional(
                    NodePair {
                        from: node,
                        to: neighbour.id,
                    },
                    edge_ctx,
                )?;
                for changed in updated {
                    trim_targets.insert(changed);
                }
            }
            for changed in trim_targets {
                self.trim_neighbours(changed, edge_ctx, source)?;
            }
        }
        Ok(())
    }

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
                continue;
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
        node: &Node,
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
        self.update_search_state_with_candidates(
            CandidateBatch {
                candidates: fresh,
                distances,
            },
            ctx.ef,
            state,
        );

        Ok(())
    }

    fn collect_unvisited_neighbours(
        &self,
        node: &Node,
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
        batch: CandidateBatch,
        ef: usize,
        state: &mut LayerSearchState,
    ) {
        let CandidateBatch {
            candidates,
            distances,
        } = batch;
        for (candidate, candidate_distance) in candidates.into_iter().zip(distances) {
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
        neighbours.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        neighbours
    }

    fn node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id).and_then(Option::as_ref)
    }

    fn node_mut(&mut self, id: usize) -> Option<&mut Node> {
        self.nodes.get_mut(id).and_then(Option::as_mut)
    }

    fn link_bidirectional(
        &mut self,
        pair: NodePair,
        ctx: EdgeContext,
    ) -> Result<Vec<usize>, HnswError> {
        let mut updated = Vec::with_capacity(2);
        if self.link_one_way(pair, ctx)? {
            updated.push(pair.from);
        }
        if self.link_one_way(
            NodePair {
                from: pair.to,
                to: pair.from,
            },
            ctx,
        )? {
            updated.push(pair.to);
        }
        Ok(updated)
    }

    fn link_one_way(&mut self, pair: NodePair, ctx: EdgeContext) -> Result<bool, HnswError> {
        let node = self
            .node_mut(pair.from)
            .ok_or_else(|| HnswError::GraphInvariantViolation {
                message: format!("node {} missing during link", pair.from),
            })?;
        let list = node.neighbours_mut(ctx.level);
        if !list.contains(&pair.to) {
            list.push(pair.to);
            return Ok(true);
        }
        Ok(false)
    }

    fn trim_neighbours<D: DataSource + Sync>(
        &mut self,
        node: usize,
        ctx: EdgeContext,
        source: &D,
    ) -> Result<(), HnswError> {
        let candidates = {
            let node_ref =
                self.node_mut(node)
                    .ok_or_else(|| HnswError::GraphInvariantViolation {
                        message: format!("node {node} missing during trim"),
                    })?;
            let list = node_ref.neighbours_mut(ctx.level);
            if list.len() <= ctx.max_connections {
                return Ok(());
            }
            list.clone()
        };

        let distances = validate_batch_distances(source, node, &candidates)?;
        let mut scored: Vec<_> = candidates.into_iter().zip(distances).collect();
        scored.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        scored.truncate(ctx.max_connections);

        let node_ref = self
            .node_mut(node)
            .ok_or_else(|| HnswError::GraphInvariantViolation {
                message: format!("node {node} missing during trim"),
            })?;
        let list = node_ref.neighbours_mut(ctx.level);
        list.clear();
        list.extend(scored.into_iter().map(|(candidate, _)| candidate));
        Ok(())
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

#[derive(Clone, Debug)]
struct Node {
    neighbours: Vec<Vec<usize>>,
}

impl Node {
    fn new(level: usize) -> Self {
        let mut neighbours = Vec::with_capacity(level + 1);
        neighbours.resize_with(level + 1, Vec::new);
        Self { neighbours }
    }

    fn neighbours(&self, level: usize) -> &[usize] {
        self.neighbours.get(level).map_or(&[], Vec::as_slice)
    }

    fn neighbours_mut(&mut self, level: usize) -> &mut Vec<usize> {
        self.neighbours
            .get_mut(level)
            .expect("levels are initialised during construction")
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EntryPoint {
    pub(crate) node: usize,
    pub(crate) level: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct InsertionPlan {
    pub(crate) layers: Vec<LayerPlan>,
}

#[derive(Clone, Debug)]
pub(crate) struct LayerPlan {
    pub(crate) level: usize,
    pub(crate) neighbours: Vec<Neighbour>,
}

/// Neighbour discovered during a search, including its distance from the query.
///
/// # Examples
/// ```
/// use chutoro_core::Neighbour;
///
/// let neighbour = Neighbour { id: 3, distance: 0.42 };
/// assert_eq!(neighbour.id, 3);
/// assert!(neighbour.distance < 1.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbour {
    /// Index of the neighbour within the [`DataSource`].
    pub id: usize,
    /// Distance between the query item and [`Neighbour::id`].
    pub distance: f32,
}

impl Eq for Neighbour {}

impl Ord for Neighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Neighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct ReverseNeighbour {
    inner: Neighbour,
}

impl ReverseNeighbour {
    fn new(id: usize, distance: f32) -> Self {
        Self {
            inner: Neighbour { id, distance },
        }
    }
}

impl Eq for ReverseNeighbour {}

impl PartialEq for ReverseNeighbour {
    fn eq(&self, other: &Self) -> bool {
        self.inner.distance == other.inner.distance && self.inner.id == other.inner.id
    }
}

impl Ord for ReverseNeighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .inner
            .distance
            .partial_cmp(&self.inner.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.inner.id.cmp(&self.inner.id))
    }
}

impl PartialOrd for ReverseNeighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) fn validate_distance<D: DataSource + Sync>(
    source: &D,
    left: usize,
    right: usize,
) -> Result<f32, HnswError> {
    let value = source.distance(left, right)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(HnswError::NonFiniteDistance { left, right })
    }
}

pub(crate) fn validate_batch_distances<D: DataSource + Sync>(
    source: &D,
    query: usize,
    candidates: &[usize],
) -> Result<Vec<f32>, HnswError> {
    let distances = source.batch_distances(query, candidates)?;
    for (candidate, distance) in candidates.iter().copied().zip(distances.iter().copied()) {
        if !distance.is_finite() {
            return Err(HnswError::NonFiniteDistance {
                left: query,
                right: candidate,
            });
        }
    }
    Ok(distances)
}
