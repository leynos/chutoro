//! Verus proofs for edge harvest primitives.

use vstd::prelude::*;
use vstd::relations::sorted_by;
use vstd::seq_lib::*;

mod edge_harvest_extract;
mod edge_harvest_ordering;

fn main() {}

verus! {

pub type NodeId = nat;
pub type Sequence = nat;
pub type Distance = int;

pub struct NeighbourSpec {
    pub id: NodeId,
    pub distance: Distance,
}

pub struct LayerPlanSpec {
    pub neighbours: Seq<NeighbourSpec>,
}

pub struct InsertionPlanSpec {
    pub layers: Seq<LayerPlanSpec>,
}

pub struct CandidateEdgeSpec {
    pub source: NodeId,
    pub target: NodeId,
    pub distance: Distance,
    pub sequence: Sequence,
}

impl CandidateEdgeSpec {
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

pub open spec fn edge_leq(a: CandidateEdgeSpec, b: CandidateEdgeSpec) -> bool {
    if a.sequence < b.sequence {
        true
    } else if a.sequence > b.sequence {
        false
    } else {
        edge_ord_leq(a, b)
    }
}

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

pub open spec fn extract_candidate_edges(
    source_node: NodeId,
    source_sequence: Sequence,
    plan: InsertionPlanSpec,
) -> Seq<CandidateEdgeSpec> {
    extract_from_layers(source_node, source_sequence, plan.layers)
}

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

pub struct EdgeHarvestSpec {
    pub edges: Seq<CandidateEdgeSpec>,
}

impl EdgeHarvestSpec {
    pub open spec fn from_unsorted(edges: Seq<CandidateEdgeSpec>) -> EdgeHarvestSpec {
        EdgeHarvestSpec {
            edges: edges.sort_by(|a: CandidateEdgeSpec, b: CandidateEdgeSpec| edge_leq(a, b)),
        }
    }
}

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
