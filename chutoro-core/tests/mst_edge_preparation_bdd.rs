//! Behavioural acceptance tests for minimum spanning forest edge preparation.
//!
//! These tests use the [`rstest-bdd`] framework driven by the Gherkin feature
//! file at `chutoro-core/tests/features/mst_edge_preparation.feature`. Each
//! scenario builds a pinned candidate-edge list, changes the edges it names,
//! and calls the public [`parallel_kruskal`] entry point. Mutual-reachability
//! weights are applied in the step itself with
//! `distance.max(core[source]).max(core[target])`, so the scenarios never
//! reach into the crate-private pipeline transform.

use rstest::fixture;
use rstest_bdd::StepResult;
use rstest_bdd_macros::{given, scenario, then, when};

use chutoro_core::{CandidateEdge, EdgeHarvest, MinimumSpanningForest, MstError, parallel_kruskal};

/// A scenario step referred to an edge or node the fixture does not have.
#[derive(Debug)]
struct BddStepError(String);

impl std::fmt::Display for BddStepError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "invalid MST edge-preparation fixture: {}",
            self.0
        )
    }
}

impl std::error::Error for BddStepError {}

/// Scenario state: the candidate graph and the forest built from it.
#[derive(Debug, Default)]
struct EdgePreparationWorld {
    /// Number of nodes the forest is built over.
    node_count: usize,
    /// Candidate edges in input order; edge `i` has sequence `i`.
    edges: Vec<CandidateEdge>,
    /// Per-node core distances, when the scenario applies them.
    core_distances: Option<Vec<f32>>,
    /// Outcome of the most recent forest construction.
    outcome: Option<Result<MinimumSpanningForest, MstError>>,
}

impl EdgePreparationWorld {
    /// Returns the candidate edge at `index`, or a fixture error.
    fn edge(&self, index: usize) -> StepResult<CandidateEdge, BddStepError> {
        self.edges
            .get(index)
            .copied()
            .ok_or_else(|| BddStepError(format!("no candidate edge {index}")))
    }

    /// Replaces the candidate edge at `index`, or reports a fixture error.
    fn replace_edge(&mut self, index: usize, edge: CandidateEdge) -> StepResult<(), BddStepError> {
        let slot = self
            .edges
            .get_mut(index)
            .ok_or_else(|| BddStepError(format!("no candidate edge {index}")))?;
        *slot = edge;
        Ok(())
    }

    /// Returns the built forest, panicking with the error if construction failed.
    fn forest(&self) -> &MinimumSpanningForest {
        match &self.outcome {
            Some(Ok(forest)) => forest,
            Some(Err(error)) => panic!("forest construction failed: {error}"),
            None => panic!("the forest has not been built"),
        }
    }
}

/// Builds the pinned base edge: nodes `(i mod n, (i + 1) mod n)`, distance
/// `i + 1` and sequence `i`.
fn base_edge(index: usize, node_count: usize) -> StepResult<CandidateEdge, BddStepError> {
    let sequence = u64::try_from(index).map_err(|error| BddStepError(error.to_string()))?;
    let rank = u16::try_from(index + 1).map_err(|error| BddStepError(error.to_string()))?;
    let wrap = |node: usize| {
        node.checked_rem(node_count)
            .ok_or_else(|| BddStepError("a graph needs at least one node".to_owned()))
    };
    Ok(CandidateEdge::new(
        wrap(index)?,
        wrap(index + 1)?,
        f32::from(rank),
        sequence,
    ))
}

/// Applies mutual-reachability weighting with the pipeline's expression.
fn mutual_reachability(
    edge: &CandidateEdge,
    core_distances: &[f32],
) -> StepResult<CandidateEdge, BddStepError> {
    let core = |node: usize| {
        core_distances
            .get(node)
            .copied()
            .ok_or_else(|| BddStepError(format!("node {node} has no core distance")))
    };
    let weight = edge
        .distance()
        .max(core(edge.source())?)
        .max(core(edge.target())?);
    Ok(CandidateEdge::new(
        edge.source(),
        edge.target(),
        weight,
        edge.sequence(),
    ))
}

#[fixture]
fn world() -> EdgePreparationWorld {
    EdgePreparationWorld::default()
}

#[given("a graph with {nodes:usize} nodes and {count:usize} candidate edges")]
fn graph_with_edges(
    world: &mut EdgePreparationWorld,
    nodes: usize,
    count: usize,
) -> StepResult<(), BddStepError> {
    world.node_count = nodes;
    world.edges = (0..count)
        .map(|index| base_edge(index, nodes))
        .collect::<Result<_, _>>()?;
    Ok(())
}

#[given(
    "node {node:usize} has core distance {raised:u16} and every other node has core distance {base:u16}"
)]
fn raised_core_distance(
    world: &mut EdgePreparationWorld,
    node: usize,
    raised: u16,
    base: u16,
) -> StepResult<(), BddStepError> {
    let mut core_distances = vec![f32::from(base); world.node_count];
    let slot = core_distances
        .get_mut(node)
        .ok_or_else(|| BddStepError(format!("no node {node}")))?;
    *slot = f32::from(raised);
    world.core_distances = Some(core_distances);
    Ok(())
}

#[given("the last {count:usize} candidate edges are exact copies of the first {first:usize}")]
fn duplicated_edges(
    world: &mut EdgePreparationWorld,
    count: usize,
    first: usize,
) -> StepResult<(), BddStepError> {
    if count != first || 2 * count > world.edges.len() {
        return Err(BddStepError(format!(
            "cannot copy the first {first} edges over the last {count}"
        )));
    }
    let offset = world.edges.len() - count;
    for index in 0..count {
        let copy = world.edge(index)?;
        world.replace_edge(offset + index, copy)?;
    }
    Ok(())
}

#[given("the last {count:usize} candidate edges are self-loops")]
fn self_loop_edges(world: &mut EdgePreparationWorld, count: usize) -> StepResult<(), BddStepError> {
    let start = world
        .edges
        .len()
        .checked_sub(count)
        .ok_or_else(|| BddStepError(format!("fewer than {count} candidate edges")))?;
    for index in start..world.edges.len() {
        let edge = world.edge(index)?;
        let looped = CandidateEdge::new(
            edge.source(),
            edge.source(),
            edge.distance(),
            edge.sequence(),
        );
        world.replace_edge(index, looped)?;
    }
    Ok(())
}

#[given("candidate edge {index:usize} references node {node:usize}")]
fn edge_references_node(
    world: &mut EdgePreparationWorld,
    index: usize,
    node: usize,
) -> StepResult<(), BddStepError> {
    let edge = world.edge(index)?;
    let moved = CandidateEdge::new(edge.source(), node, edge.distance(), edge.sequence());
    world.replace_edge(index, moved)
}

#[given("candidate edge {index:usize} has a non-finite weight")]
fn edge_has_non_finite_weight(
    world: &mut EdgePreparationWorld,
    index: usize,
) -> StepResult<(), BddStepError> {
    let edge = world.edge(index)?;
    let infinite = CandidateEdge::new(edge.source(), edge.target(), f32::INFINITY, edge.sequence());
    world.replace_edge(index, infinite)
}

#[given("the candidate edges span only nodes 0 to {last:usize}")]
fn edges_span_prefix(
    world: &mut EdgePreparationWorld,
    last: usize,
) -> StepResult<(), BddStepError> {
    let span = last + 1;
    world.edges = (0..world.edges.len())
        .map(|index| base_edge(index, span))
        .collect::<Result<_, _>>()?;
    Ok(())
}

#[when("I build the minimum spanning forest")]
fn build_forest(world: &mut EdgePreparationWorld) -> StepResult<(), BddStepError> {
    let edges = match &world.core_distances {
        Some(core_distances) => world
            .edges
            .iter()
            .map(|edge| mutual_reachability(edge, core_distances))
            .collect::<Result<_, _>>()?,
        None => world.edges.clone(),
    };
    world.outcome = Some(parallel_kruskal(world.node_count, &EdgeHarvest::new(edges)));
    Ok(())
}

#[then("the forest has {count:usize} edges")]
fn forest_has_edges(world: &EdgePreparationWorld, count: usize) {
    assert_eq!(world.forest().edges().len(), count);
}

#[then("the forest excludes the edge between nodes {left:usize} and {right:usize}")]
fn forest_excludes_edge(world: &EdgePreparationWorld, left: usize, right: usize) {
    let (low, high) = (left.min(right), left.max(right));
    assert!(
        world
            .forest()
            .edges()
            .iter()
            .all(|edge| (edge.source(), edge.target()) != (low, high)),
        "forest {:?} contains the edge ({low}, {high})",
        world.forest().edges()
    );
}

#[then("every forest edge has source less than target")]
fn forest_edges_are_canonical(world: &EdgePreparationWorld) {
    assert!(
        world
            .forest()
            .edges()
            .iter()
            .all(|edge| edge.source() < edge.target())
    );
}

#[then("forest construction fails with error code {code:string}")]
fn forest_construction_fails(world: &EdgePreparationWorld, code: &str) {
    match &world.outcome {
        Some(Err(error)) => assert_eq!(error.code().as_str(), code),
        other => panic!("expected error {code}, got {other:?}"),
    }
}

#[then("the forest reports {count:usize} components")]
fn forest_reports_components(world: &EdgePreparationWorld, count: usize) {
    assert_eq!(world.forest().component_count(), count);
}

#[then("the forest is not a tree")]
fn forest_is_not_a_tree(world: &EdgePreparationWorld) {
    assert!(!world.forest().is_tree());
}

#[scenario(
    path = "tests/features/mst_edge_preparation.feature",
    name = "Mutual-reachability weights drive forest selection"
)]
fn mutual_reachability_weights_drive_forest_selection(
    _world: EdgePreparationWorld,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/mst_edge_preparation.feature",
    name = "Duplicate candidate edges are collapsed"
)]
fn duplicate_candidate_edges_are_collapsed(
    _world: EdgePreparationWorld,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/mst_edge_preparation.feature",
    name = "Self-loops are discarded without failing"
)]
fn self_loops_are_discarded_without_failing(
    _world: EdgePreparationWorld,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/mst_edge_preparation.feature",
    name = "The lowest-index invalid edge is reported"
)]
fn the_lowest_index_invalid_edge_is_reported(
    _world: EdgePreparationWorld,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/mst_edge_preparation.feature",
    name = "An edge that is both out of range and non-finite reports its node"
)]
fn an_out_of_range_non_finite_edge_reports_its_node(
    _world: EdgePreparationWorld,
) -> StepResult<(), BddStepError> {
    Ok(())
}

#[scenario(
    path = "tests/features/mst_edge_preparation.feature",
    name = "A disconnected graph yields a forest, not a tree"
)]
fn a_disconnected_graph_yields_a_forest(
    _world: EdgePreparationWorld,
) -> StepResult<(), BddStepError> {
    Ok(())
}
