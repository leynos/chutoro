//! Extraction invariants for edge harvest primitives.

use vstd::prelude::*;
use vstd::seq::group_seq_axioms;

verus! {

use super::*;

proof fn lemma_prepend_first_edge_preserves_common_invariants(
    first_edge: CandidateEdgeSpec,
    rest_edges: Seq<CandidateEdgeSpec>,
    rest_expected: nat,
    source_node: NodeId,
    source_sequence: Sequence,
)
    requires
        edges_common_invariants(rest_edges, rest_expected, source_node, source_sequence),
        first_edge.source == source_node,
        first_edge.target != source_node,
        first_edge.sequence == source_sequence,
    ensures
        edges_common_invariants(
            Seq::<CandidateEdgeSpec>::empty().push(first_edge).add(rest_edges),
            1 + rest_expected,
            source_node,
            source_sequence,
        ),
{
    broadcast use group_seq_axioms;

    let prefix = Seq::<CandidateEdgeSpec>::empty().push(first_edge);
    let edges = prefix.add(rest_edges);
    assert(prefix.len() == 1);
    assert(edges.len() == 1 + rest_expected);

    assert forall|i: int| #![auto] 0 <= i < edges.len() implies edges[i].source == source_node by {
        if i == 0 {
            assert(edges[i] == prefix[i]);
            assert(prefix[i] == first_edge);
        } else {
            let j = i - 1;
            assert(0 <= j < rest_edges.len());
            assert(edges[i] == rest_edges[j]);
        }
    }

    assert forall|i: int| #![auto] 0 <= i < edges.len() implies edges[i].target != source_node by {
        if i == 0 {
            assert(edges[i] == prefix[i]);
            assert(prefix[i] == first_edge);
        } else {
            let j = i - 1;
            assert(0 <= j < rest_edges.len());
            assert(edges[i] == rest_edges[j]);
        }
    }

    assert forall|i: int| #![auto] 0 <= i < edges.len() implies edges[i].sequence == source_sequence by {
        if i == 0 {
            assert(edges[i] == prefix[i]);
            assert(prefix[i] == first_edge);
        } else {
            let j = i - 1;
            assert(0 <= j < rest_edges.len());
            assert(edges[i] == rest_edges[j]);
        }
    }
}

proof fn lemma_concat_preserves_common_invariants(
    head_edges: Seq<CandidateEdgeSpec>,
    rest_edges: Seq<CandidateEdgeSpec>,
    head_expected: nat,
    rest_expected: nat,
    source_node: NodeId,
    source_sequence: Sequence,
)
    requires
        edges_common_invariants(head_edges, head_expected, source_node, source_sequence),
        edges_common_invariants(rest_edges, rest_expected, source_node, source_sequence),
    ensures
        edges_common_invariants(
            head_edges.add(rest_edges),
            head_expected + rest_expected,
            source_node,
            source_sequence,
        ),
{
    broadcast use group_seq_axioms;

    let edges = head_edges.add(rest_edges);
    assert(head_edges.len() == head_expected);
    assert(rest_edges.len() == rest_expected);
    assert(edges.len() == head_expected + rest_expected);

    assert forall|i: int| #![auto] 0 <= i < edges.len() implies edges[i].source == source_node by {
        if i < head_edges.len() {
            assert(edges[i] == head_edges[i]);
        } else {
            let j = i - head_edges.len();
            assert(0 <= j < rest_edges.len());
            assert(edges[i] == rest_edges[j]);
        }
    }

    assert forall|i: int| #![auto] 0 <= i < edges.len() implies edges[i].target != source_node by {
        if i < head_edges.len() {
            assert(edges[i] == head_edges[i]);
        } else {
            let j = i - head_edges.len();
            assert(0 <= j < rest_edges.len());
            assert(edges[i] == rest_edges[j]);
        }
    }

    assert forall|i: int| #![auto] 0 <= i < edges.len() implies edges[i].sequence == source_sequence by {
        if i < head_edges.len() {
            assert(edges[i] == head_edges[i]);
        } else {
            let j = i - head_edges.len();
            assert(0 <= j < rest_edges.len());
            assert(edges[i] == rest_edges[j]);
        }
    }
}

proof fn lemma_extract_from_layer_invariants(
    neighbours: Seq<NeighbourSpec>,
    source_node: NodeId,
    source_sequence: Sequence,
)
    ensures
        extract_layer_invariants(neighbours, source_node, source_sequence),
    decreases neighbours.len(),
{
    if neighbours.len() == 0 {
        let edges = extract_from_layer(source_node, source_sequence, neighbours);
        assert(edges.len() == 0);
    } else {
        let head = neighbours.first();
        let rest = neighbours.drop_first();
        lemma_extract_from_layer_invariants(rest, source_node, source_sequence);

        let edges = extract_from_layer(source_node, source_sequence, neighbours);
        let rest_edges = extract_from_layer(source_node, source_sequence, rest);

        if head.id == source_node {
            assert(edges =~= rest_edges);
            assert(edges.len() == rest_edges.len());
            assert(count_non_self(neighbours, source_node) == count_non_self(rest, source_node));

            assert forall|i: int| #![auto] 0 <= i < edges.len() implies edges[i].source == source_node by {
                assert(edges[i] == rest_edges[i]);
            }
            assert forall|i: int| #![auto] 0 <= i < edges.len() implies edges[i].target != source_node by {
                assert(edges[i] == rest_edges[i]);
            }
            assert forall|i: int| #![auto] 0 <= i < edges.len() implies edges[i].sequence == source_sequence by {
                assert(edges[i] == rest_edges[i]);
            }
        } else {
            let first_edge = CandidateEdgeSpec {
                source: source_node,
                target: head.id,
                distance: head.distance,
                sequence: source_sequence,
            };
            let prefix = Seq::<CandidateEdgeSpec>::empty().push(first_edge);
            let rest_expected = count_non_self(rest, source_node);
            assert(prefix.len() == 1);
            assert(edges == prefix.add(rest_edges));
            assert(count_non_self(neighbours, source_node) == 1 + rest_expected);

            lemma_prepend_first_edge_preserves_common_invariants(
                first_edge,
                rest_edges,
                rest_expected,
                source_node,
                source_sequence,
            );
            assert(edges_common_invariants(
                edges,
                1 + rest_expected,
                source_node,
                source_sequence,
            ));
        }
    }
}

proof fn lemma_extract_from_layers_invariants(
    layers: Seq<LayerPlanSpec>,
    source_node: NodeId,
    source_sequence: Sequence,
)
    ensures
        extract_layers_invariants(layers, source_node, source_sequence),
    decreases layers.len(),
{
    if layers.len() == 0 {
        let edges = extract_from_layers(source_node, source_sequence, layers);
        assert(edges.len() == 0);
    } else {
        let head = layers.first();
        let rest = layers.drop_first();
        let head_edges = extract_from_layer(source_node, source_sequence, head.neighbours);
        let rest_edges = extract_from_layers(source_node, source_sequence, rest);

        lemma_extract_from_layer_invariants(head.neighbours, source_node, source_sequence);
        lemma_extract_from_layers_invariants(rest, source_node, source_sequence);
        let head_expected = count_non_self(head.neighbours, source_node);
        let rest_expected = count_layers(rest, source_node);
        assert(head_edges.len() == head_expected);
        assert(rest_edges.len() == rest_expected);

        let edges = extract_from_layers(source_node, source_sequence, layers);
        assert(count_layers(layers, source_node) == head_expected + rest_expected);
        assert(edges == head_edges.add(rest_edges));
        lemma_concat_preserves_common_invariants(
            head_edges,
            rest_edges,
            head_expected,
            rest_expected,
            source_node,
            source_sequence,
        );
        assert(edges_common_invariants(
            edges,
            head_expected + rest_expected,
            source_node,
            source_sequence,
        ));
    }
}

/// Proves invariants for extracting candidate edges from an insertion plan.
///
/// ## Example
/// ```text
/// If a plan has only self-neighbours, the extracted edges are empty.
/// ```
pub(super) proof fn lemma_extract_candidate_edges_invariants(
    plan: InsertionPlanSpec,
    source_node: NodeId,
    source_sequence: Sequence,
)
    ensures
        extract_plan_invariants(plan, source_node, source_sequence),
{
    lemma_extract_from_layers_invariants(plan.layers, source_node, source_sequence);
}

} // verus!
