//! Verus proofs for edge harvest primitives.

use vstd::prelude::*;
use vstd::relations::sorted_by;
use vstd::seq_lib::*;

mod edge_harvest_extract;
mod edge_harvest_ordering;

fn main() {}

verus! {

/// Identifier for a node in the HNSW graph.
pub type NodeId = nat;
/// Monotonic insertion sequence number for candidate edges.
pub type Sequence = nat;
/// Distance metric value used for ordering edges.
pub type Distance = int;

/// A neighbour entry in a layer plan.
pub struct NeighbourSpec {
    pub id: NodeId,
    pub distance: Distance,
}

/// The neighbour list for a single layer.
pub struct LayerPlanSpec {
    pub neighbours: Seq<NeighbourSpec>,
}

/// An insertion plan containing all layer plans for a node.
pub struct InsertionPlanSpec {
    pub layers: Seq<LayerPlanSpec>,
}

/// Candidate edge specification for harvesting.
pub struct CandidateEdgeSpec {
    pub source: NodeId,
    pub target: NodeId,
    pub distance: Distance,
    pub sequence: Sequence,
}

impl CandidateEdgeSpec {
    /// Returns a canonical edge with ordered endpoints.
    ///
    /// ## Example
    /// ```text
    /// (source=3, target=1) yields (source=1, target=3).
    /// ```
    pub open spec fn canonicalise(self) -> Self {
        if self.source <= self.target {
            self
        } else {
            CandidateEdgeSpec {
                source: self.target,
                target: self.source,
                distance: self.distance,
                sequence: self.sequence,
            }
        }
    }
}

/// Orders edges by distance, then source, target, and sequence.
///
/// ## Example
/// ```text
/// edge_ord_leq((d=1, s=0, t=2, q=7), (d=2, s=0, t=1, q=1)) == true.
/// ```
pub open spec fn edge_ord_leq(a: CandidateEdgeSpec, b: CandidateEdgeSpec) -> bool {
    if a.distance < b.distance {
        true
    } else if a.distance > b.distance {
        false
    } else if a.source < b.source {
        true
    } else if a.source > b.source {
        false
    } else if a.target < b.target {
        true
    } else if a.target > b.target {
        false
    } else {
        a.sequence <= b.sequence
    }
}

/// Orders edges by sequence, then by the natural edge ordering.
///
/// ## Example
/// ```text
/// edge_leq((q=1, d=2, s=0, t=1), (q=2, d=0, s=9, t=9)) == true.
/// ```
pub open spec fn edge_leq(a: CandidateEdgeSpec, b: CandidateEdgeSpec) -> bool {
    if a.sequence < b.sequence {
        true
    } else if a.sequence > b.sequence {
        false
    } else {
        edge_ord_leq(a, b)
    }
}

/// Counts neighbours that are not the source node.
///
/// ## Example
/// ```text
/// count_non_self([self, other], source_node) == 1.
/// ```
pub open spec fn count_non_self(
    neighbours: Seq<NeighbourSpec>,
    source_node: NodeId,
) -> nat
    decreases neighbours.len(),
{
    if neighbours.len() == 0 {
        0
    } else {
        let head = neighbours.first();
        let rest = neighbours.drop_first();
        if head.id == source_node {
            count_non_self(rest, source_node)
        } else {
            1 + count_non_self(rest, source_node)
        }
    }
}

/// Counts non-self neighbours across all layers.
///
/// ## Example
/// ```text
/// count_layers([layer0: [self], layer1: [other]], source_node) == 1.
/// ```
pub open spec fn count_layers(layers: Seq<LayerPlanSpec>, source_node: NodeId) -> nat
    decreases layers.len(),
{
    if layers.len() == 0 {
        0
    } else {
        let head = layers.first();
        let rest = layers.drop_first();
        count_non_self(head.neighbours, source_node) + count_layers(rest, source_node)
    }
}

/// Extracts candidate edges from a single layer, filtering self-neighbours.
///
/// ## Example
/// ```text
/// extract_from_layer(source=0, seq=1, neighbours=[(0), (2)]) yields 1 edge to 2.
/// ```
pub open spec fn extract_from_layer(
    source_node: NodeId,
    source_sequence: Sequence,
    neighbours: Seq<NeighbourSpec>,
) -> Seq<CandidateEdgeSpec>
    decreases neighbours.len(),
{
    if neighbours.len() == 0 {
        Seq::<CandidateEdgeSpec>::empty()
    } else {
        let head = neighbours.first();
        let rest = neighbours.drop_first();
        let rest_edges = extract_from_layer(source_node, source_sequence, rest);
        if head.id == source_node {
            rest_edges
        } else {
            let edge = CandidateEdgeSpec {
                source: source_node,
                target: head.id,
                distance: head.distance,
                sequence: source_sequence,
            };
            Seq::<CandidateEdgeSpec>::empty().push(edge).add(rest_edges)
        }
    }
}

/// Extracts candidate edges from all layers in order.
///
/// ## Example
/// ```text
/// extract_from_layers(source=0, seq=1, [layer0, layer1]) concatenates each layer's edges.
/// ```
pub open spec fn extract_from_layers(
    source_node: NodeId,
    source_sequence: Sequence,
    layers: Seq<LayerPlanSpec>,
) -> Seq<CandidateEdgeSpec>
    decreases layers.len(),
{
    if layers.len() == 0 {
        Seq::<CandidateEdgeSpec>::empty()
    } else {
        let head = layers.first();
        let rest = layers.drop_first();
        let head_edges = extract_from_layer(source_node, source_sequence, head.neighbours);
        let rest_edges = extract_from_layers(source_node, source_sequence, rest);
        head_edges.add(rest_edges)
    }
}

/// Extracts candidate edges from an insertion plan.
///
/// ## Example
/// ```text
/// extract_candidate_edges(source, seq, plan) matches extract_from_layers(plan.layers).
/// ```
pub open spec fn extract_candidate_edges(
    source_node: NodeId,
    source_sequence: Sequence,
    plan: InsertionPlanSpec,
) -> Seq<CandidateEdgeSpec> {
    extract_from_layers(source_node, source_sequence, plan.layers)
}

/// Shared invariants for extracted edges.
///
/// ## Example
/// ```text
/// edges_common_invariants(edges, len, source, seq) holds when every edge
/// has the expected length, source, and sequence.
/// ```
pub open spec fn edges_common_invariants(
    edges: Seq<CandidateEdgeSpec>,
    expected_len: nat,
    source_node: NodeId,
    source_sequence: Sequence,
) -> bool {
    &&& edges.len() == expected_len
    &&& forall|i: int| #![auto] 0 <= i < edges.len() ==> edges[i].source == source_node
    &&& forall|i: int| #![auto] 0 <= i < edges.len() ==> edges[i].target != source_node
    &&& forall|i: int| #![auto] 0 <= i < edges.len() ==> edges[i].sequence == source_sequence
}

/// Invariants for a single layer extraction.
///
/// ## Example
/// ```text
/// extract_layer_invariants(neighbours, source, seq) holds when all non-self
/// neighbours become edges with the same source and sequence.
/// ```
pub open spec fn extract_layer_invariants(
    neighbours: Seq<NeighbourSpec>,
    source_node: NodeId,
    source_sequence: Sequence,
) -> bool {
    let edges = extract_from_layer(source_node, source_sequence, neighbours);
    edges_common_invariants(
        edges,
        count_non_self(neighbours, source_node),
        source_node,
        source_sequence,
    )
}

/// Invariants for extracting edges from all layers.
///
/// ## Example
/// ```text
/// extract_layers_invariants(layers, source, seq) holds when counts and fields
/// match the sum of each layer.
/// ```
pub open spec fn extract_layers_invariants(
    layers: Seq<LayerPlanSpec>,
    source_node: NodeId,
    source_sequence: Sequence,
) -> bool {
    let edges = extract_from_layers(source_node, source_sequence, layers);
    edges_common_invariants(
        edges,
        count_layers(layers, source_node),
        source_node,
        source_sequence,
    )
}

/// Invariants for extracting edges from an insertion plan.
///
/// ## Example
/// ```text
/// extract_plan_invariants(plan, source, seq) holds when the plan's layers
/// satisfy the shared edge invariants.
/// ```
pub open spec fn extract_plan_invariants(
    plan: InsertionPlanSpec,
    source_node: NodeId,
    source_sequence: Sequence,
) -> bool {
    let edges = extract_candidate_edges(source_node, source_sequence, plan);
    edges_common_invariants(
        edges,
        count_layers(plan.layers, source_node),
        source_node,
        source_sequence,
    )
}

/// Sorted edge harvest results.
pub struct EdgeHarvestSpec {
    pub edges: Seq<CandidateEdgeSpec>,
}

impl EdgeHarvestSpec {
    /// Sorts edges by the edge ordering.
    ///
    /// ## Example
    /// ```text
    /// from_unsorted([e2, e1]).edges is sorted by edge_leq.
    /// ```
    pub open spec fn from_unsorted(edges: Seq<CandidateEdgeSpec>) -> EdgeHarvestSpec {
        EdgeHarvestSpec {
            edges: edges.sort_by(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_leq(a, b)),
        }
    }
}

/// Invariants for edge harvesting after sorting.
///
/// ## Example
/// ```text
/// edge_harvest_invariants(edges) holds when sorting preserves the multiset.
/// ```
pub open spec fn edge_harvest_invariants(edges: Seq<CandidateEdgeSpec>) -> bool {
    let harvest = EdgeHarvestSpec::from_unsorted(edges);
    &&& harvest.edges.to_multiset() =~= edges.to_multiset()
    &&& sorted_by(harvest.edges, |a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_leq(a, b))
}

proof fn lemma_edge_harvest_from_unsorted_invariants(edges: Seq<CandidateEdgeSpec>)
    ensures
        edge_harvest_invariants(edges),
{
    edge_harvest_ordering::lemma_edge_leq_total_ordering();
    edges.lemma_sort_by_ensures(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_leq(a, b));
}

} // verus!
