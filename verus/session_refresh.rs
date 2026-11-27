//! Verus proofs for the session refresh domain boundary.
//!
//! Two obligations from the 11.2.1 ExecPlan are discharged here:
//!
//! - **OBL-1 reweight faithfulness.** `mutual_reachability_weight` is the
//!   pointwise maximum of the raw distance and both endpoint core distances,
//!   so it is commutative in the endpoints, at least every argument, and
//!   depends only on its three arguments — never on a previously stored
//!   weight. A session that retains raw backbone edges therefore reweights
//!   identically no matter how many refreshes have already run.
//! - **OBL-2 merge multiset preservation.** The combined candidate sequence is
//!   the concatenation `mst_backbone ++ historical_edges ++ pending_edges`, and
//!   reweighting preserves every element's endpoints and sequence at the same
//!   index, changing only the distance field.
//!
//! ## What is proved, and what is trusted
//!
//! The proofs verify properties of an *independent model*: `Distance` is
//! `int` and `reweight_edges` is a `Seq` map. They do **not** verify the Rust
//! functions. The correspondence between model and production is a stated
//! trust boundary, justified as follows and covered by sampled tests rather
//! than by a refinement proof:
//!
//! - `mutual_reachability_weight` takes three `f32` values and returns
//!   `distance.max(core_source).max(core_target)`. Over **finite** `f32` that
//!   is the same total order the integer model uses, so every order-based
//!   lemma transfers as a claim about *numeric* order and nothing more.
//! - `f32::max` does **not** propagate NaN. Per the pinned toolchain's
//!   standard library, it returns the other operand when exactly one is NaN,
//!   and NaN only when both are. So the maximum can mask a non-finite input
//!   rather than surfacing it, and finiteness is a precondition of every
//!   claim here rather than a consequence of the formula. The model says
//!   nothing about NaN or infinity.
//! - The model's equality is numeric equality only. `+0.0` and `-0.0` are
//!   numerically equal. The equal-weight **grouping** in
//!   `parallel_kruskal_from_edges` uses numeric `partial_cmp`, so it groups
//!   the two zeros together. **Deduplication** is different:
//!   `prepare_edge_list` sorts with `MstEdge`'s `total_cmp` ordering (which
//!   separates the zeros) and then removes only *adjacent* duplicates, so two
//!   numerically equal edges with the same endpoints can both survive when
//!   the sort places other edges between them. Grouping therefore follows the
//!   model; deduplication and intra-tie ordering do not. None of the lemmas
//!   below assert anything about tie ordering or deduplication, so the
//!   discrepancy does not weaken them; it does mean this file must not be read
//!   as justifying signed-zero deduplication or ordering.
//! - `combine_raw_edges` performs three `extend_from_slice` calls, which is
//!   sequence concatenation. `mutual_reachability_edges` maps the reweight
//!   over a Rayon `par_iter` and `.collect()`; order preservation there is
//!   pinned by the adjacent `parallel_transform_matches_the_sequential_form`
//!   test, which compares transformed vectors rather than forests, at one and
//!   eight threads.
//! - `recover_raw_backbone` maps accepted forest edges back to raw inputs by
//!   canonical `(source, target, sequence)`, copying from the combined raw
//!   inputs. Nothing in the type system forces that copy to stay raw; it is
//!   the recovery code that does, and
//!   `backbone_retains_raw_distances_not_mutual_reachability_weights` checks
//!   it on a fixture. The history-independence lemmas below say only that the
//!   *weight* depends on its three arguments.
//!
//! Finiteness is a precondition, and it is **not** proved here. HNSW rejects
//! non-finite distances on insert and search, but `Kruskal` validates the
//! weight *after* reweighting, which cannot establish that its inputs were
//! finite: `f32::max` can return the numeric operand when the other is NaN.
//! The pure rebuild also accepts directly constructed inputs. So these lemmas
//! are not offered as a proof that every reachable session state is finite,
//! nor as a refinement proof of the Rust functions — only that, *given* finite
//! inputs, the reweighted value depends on nothing but its three arguments.

use vstd::prelude::*;

fn main() {}

verus! {

/// Identifier for a point in the session.
pub type NodeId = nat;
/// Monotonic insertion sequence number for candidate edges.
pub type Sequence = nat;
/// Distance or weight value, modelled as an integer.
pub type Distance = int;

/// Specification mirror of `chutoro_core::CandidateEdge`.
pub struct CandidateEdgeSpec {
    /// First endpoint of the edge.
    pub source: NodeId,
    /// Second endpoint of the edge.
    pub target: NodeId,
    /// Raw distance or mutual-reachability weight.
    pub distance: Distance,
    /// Insertion sequence used for deterministic tie-breaking.
    pub sequence: Sequence,
}

/// Specification mirror of `mutual_reachability_weight`.
///
/// The production function is `distance.max(core_source).max(core_target)` on
/// `f32`; over the finite, ordered `int` model used here that is the maximum of
/// the three values.
pub open spec fn mutual_reachability_weight(
    distance: Distance,
    core_source: Distance,
    core_target: Distance,
) -> Distance {
    if distance >= core_source && distance >= core_target {
        distance
    } else if core_source >= core_target {
        core_source
    } else {
        core_target
    }
}

/// Applies the weight formula to one edge, keeping its identity fields.
pub open spec fn reweight_edge(
    edge: CandidateEdgeSpec,
    core_distances: Seq<Distance>,
) -> CandidateEdgeSpec
    recommends
        (edge.source as int) < core_distances.len(),
        (edge.target as int) < core_distances.len(),
{
    CandidateEdgeSpec {
        source: edge.source,
        target: edge.target,
        distance: mutual_reachability_weight(
            edge.distance,
            core_distances[edge.source as int],
            core_distances[edge.target as int],
        ),
        sequence: edge.sequence,
    }
}

/// Reweights a whole candidate sequence, preserving order and length.
pub open spec fn reweight_edges(
    edges: Seq<CandidateEdgeSpec>,
    core_distances: Seq<Distance>,
) -> Seq<CandidateEdgeSpec> {
    Seq::new(
        edges.len(),
        |index: int| reweight_edge(edges[index], core_distances),
    )
}

// ---------------------------------------------------------------------------
// OBL-1: reweight faithfulness
// ---------------------------------------------------------------------------

/// The weight is at least the raw distance.
pub proof fn lemma_weight_at_least_distance(
    distance: Distance,
    core_source: Distance,
    core_target: Distance,
)
    ensures
        mutual_reachability_weight(distance, core_source, core_target) >= distance,
{
}

/// The weight is at least each endpoint's core distance.
pub proof fn lemma_weight_at_least_cores(
    distance: Distance,
    core_source: Distance,
    core_target: Distance,
)
    ensures
        mutual_reachability_weight(distance, core_source, core_target) >= core_source,
        mutual_reachability_weight(distance, core_source, core_target) >= core_target,
{
}

/// The weight is commutative in the two endpoint core distances.
pub proof fn lemma_weight_is_symmetric(
    distance: Distance,
    core_source: Distance,
    core_target: Distance,
)
    ensures
        mutual_reachability_weight(distance, core_source, core_target)
            == mutual_reachability_weight(distance, core_target, core_source),
{
}

/// **The substantive OBL-1 claim.** The weight depends only on the raw
/// distance and the two current core distances. Two reweights of the same raw
/// edge under the same core distances agree, whatever history produced them.
pub proof fn lemma_weight_is_history_independent(
    raw_distance: Distance,
    first_core_source: Distance,
    first_core_target: Distance,
    second_core_source: Distance,
    second_core_target: Distance,
)
    requires
        first_core_source == second_core_source,
        first_core_target == second_core_target,
    ensures
        mutual_reachability_weight(raw_distance, first_core_source, first_core_target)
            == mutual_reachability_weight(raw_distance, second_core_source, second_core_target),
{
}

/// A previously weighted edge cannot be lowered by reweighting its weight.
///
/// This is the concrete defect the raw-backbone retention removes: with raw
/// distance `1`, current cores `0` and `0`, the correct weight is `1`, whereas
/// feeding back a stale weight of `9` yields `9`.
pub proof fn lemma_stale_weight_ratchets(
    stale_weight: Distance,
    core_source: Distance,
    core_target: Distance,
)
    requires
        stale_weight >= core_source,
        stale_weight >= core_target,
    ensures
        mutual_reachability_weight(stale_weight, core_source, core_target) == stale_weight,
{
}

/// Non-vacuity witness for OBL-1: every argument can be the strict maximum in
/// turn, ties are covered, and the stale-weight counterexample really differs
/// from the correct result.
///
/// Each `max` arm of the piecewise definition is therefore reachable, so the
/// lemma statements are not vacuous in any arm.
pub proof fn lemma_weight_witnesses()
    ensures
        mutual_reachability_weight(1, 2, 0) == 2,  // source core is the maximum
        mutual_reachability_weight(1, 0, 3) == 3,  // target core is the maximum
        mutual_reachability_weight(5, 2, 3) == 5,  // raw distance is the maximum
        mutual_reachability_weight(0, 0, 0) == 0,  // three-way tie
        mutual_reachability_weight(4, 4, 1) == 4,  // two-way tie
        mutual_reachability_weight(1, 0, 0) != mutual_reachability_weight(9, 0, 0),
{
}

// ---------------------------------------------------------------------------
// OBL-2: merge multiset preservation
// ---------------------------------------------------------------------------

/// Reweighting preserves sequence length.
pub proof fn lemma_reweight_preserves_len(
    edges: Seq<CandidateEdgeSpec>,
    core_distances: Seq<Distance>,
)
    ensures
        reweight_edges(edges, core_distances).len() == edges.len(),
{
}

/// Reweighting preserves every element's endpoints and sequence at its index.
pub proof fn lemma_reweight_preserves_identity(
    edges: Seq<CandidateEdgeSpec>,
    core_distances: Seq<Distance>,
    index: int,
)
    requires
        0 <= index < edges.len(),
    ensures
        reweight_edges(edges, core_distances)[index].source == edges[index].source,
        reweight_edges(edges, core_distances)[index].target == edges[index].target,
        reweight_edges(edges, core_distances)[index].sequence == edges[index].sequence,
{
}

/// The concatenation of the three merge buffers has the summed length.
///
/// Scope, stated so it is not over-read: this proves the length law for
/// sequence concatenation and, via [`lemma_merge_membership`] below, each
/// buffer's membership and multiplicity in the result. It is a property of the
/// model's merge; the production merge is three `extend_from_slice` calls, and
/// the file header records that correspondence as a trust boundary rather
/// than a refinement proof.
pub proof fn lemma_merge_length(
    backbone: Seq<CandidateEdgeSpec>,
    historical: Seq<CandidateEdgeSpec>,
    pending: Seq<CandidateEdgeSpec>,
)
    ensures
        (backbone + historical + pending).len()
            == backbone.len() + historical.len() + pending.len(),
{
}

/// Each merge buffer appears in the combined sequence, at its own index and
/// with its own multiset multiplicity.
///
/// Together with the length law this is the substantive OBL-2 claim: the merge
/// neither drops, duplicates, nor reorders a candidate. The index formulas are
/// the concatenation offsets, so they also pin the merge order that
/// `recover_raw_backbone` depends on.
pub proof fn lemma_merge_membership(
    backbone: Seq<CandidateEdgeSpec>,
    historical: Seq<CandidateEdgeSpec>,
    pending: Seq<CandidateEdgeSpec>,
    index: int,
)
    requires
        0 <= index < backbone.len() + historical.len() + pending.len(),
    ensures
        ({
            let combined = backbone + historical + pending;
            (index < backbone.len()
                ==> combined[index] == backbone[index])
            && (backbone.len() <= index < backbone.len() + historical.len()
                ==> combined[index] == historical[index - backbone.len()])
            && (index >= backbone.len() + historical.len()
                ==> combined[index]
                    == pending[index - backbone.len() - historical.len()])
        }),
{
}

/// Non-vacuity witness for OBL-2: non-empty two- and three-buffer instances
/// whose elements are checked, not merely counted.
pub proof fn lemma_merge_witnesses()
    ensures
        two_buffer_witness().len() == 2,
        two_buffer_witness()[0].source == 0,
        two_buffer_witness()[1].sequence == 1,
        three_buffer_witness().len() == 3,
        three_buffer_witness()[1].target == 2,
        three_buffer_witness()[2].distance == 3,
{
}

/// A two-element concatenation witness.
pub open spec fn two_buffer_witness() -> Seq<CandidateEdgeSpec> {
    Seq::empty().push(
        CandidateEdgeSpec { source: 0, target: 1, distance: 1, sequence: 0 },
    ) + Seq::empty().push(
        CandidateEdgeSpec { source: 1, target: 2, distance: 2, sequence: 1 },
    )
}

/// A three-element concatenation witness.
pub open spec fn three_buffer_witness() -> Seq<CandidateEdgeSpec> {
    two_buffer_witness() + Seq::empty().push(
        CandidateEdgeSpec { source: 0, target: 2, distance: 3, sequence: 2 },
    )
}

}