//! Verus proofs for session core-distance primitives.

use vstd::prelude::*;

fn main() {}

verus! {

/// Ordered distance values for a non-self neighbour list.
pub type Distance = int;

/// Mirrors `session::core_distance::core_distance_from_neighbours`.
///
/// ## Example
/// ```text
/// core_distance([1, 3, 5], 2) == 3.
/// ```
pub open spec fn core_distance(neighbours: Seq<Distance>, min_cluster_size: nat) -> Distance
    recommends
        min_cluster_size > 0,
{
    if neighbours.len() >= min_cluster_size {
        neighbours[(min_cluster_size - 1) as int]
    } else if neighbours.len() > 0 {
        neighbours[neighbours.len() - 1]
    } else {
        0
    }
}

/// Defines a prefix relation between neighbour sequences.
///
/// ## Example
/// ```text
/// [1, 2] is a prefix of [1, 2, 3].
/// ```
pub open spec fn is_prefix(prefix: Seq<Distance>, extended: Seq<Distance>) -> bool {
    prefix.len() <= extended.len()
        && forall|i: int| 0 <= i < prefix.len() ==> #[trigger] prefix[i] == extended[i]
}

/// Proves selection of the `m - 1` neighbour when enough neighbours exist.
pub proof fn lemma_core_distance_selection(neighbours: Seq<Distance>, min_cluster_size: nat)
    requires
        min_cluster_size > 0,
        neighbours.len() >= min_cluster_size,
    ensures
        core_distance(neighbours, min_cluster_size)
            == neighbours[(min_cluster_size - 1) as int],
{
}

/// Proves fallback to the last available neighbour for under-populated input.
pub proof fn lemma_core_distance_fallback(neighbours: Seq<Distance>, min_cluster_size: nat)
    requires
        min_cluster_size > 0,
        0 < neighbours.len() < min_cluster_size,
    ensures
        core_distance(neighbours, min_cluster_size) == neighbours[neighbours.len() - 1],
{
}

/// Proves the empty-neighbour fallback.
pub proof fn lemma_core_distance_empty(min_cluster_size: nat)
    requires
        min_cluster_size > 0,
    ensures
        core_distance(Seq::<Distance>::empty(), min_cluster_size) == 0,
{
}

/// Proves saturated core distances cannot increase when appending a suffix.
///
/// The production fallback rule can increase while fewer than
/// `min_cluster_size` neighbours exist. Once the prefix is saturated, both
/// sequences select the same `m - 1` element.
pub proof fn lemma_core_distance_monotone_under_saturated_prefix(
    prefix: Seq<Distance>,
    extended: Seq<Distance>,
    min_cluster_size: nat,
)
    requires
        min_cluster_size > 0,
        prefix.len() >= min_cluster_size,
        is_prefix(prefix, extended),
    ensures
        core_distance(extended, min_cluster_size) <= core_distance(prefix, min_cluster_size),
{
    let index = (min_cluster_size - 1) as int;
    assert(0 <= index < prefix.len());
    assert(index < extended.len());
    assert(prefix[index] == extended[index]);
    assert(core_distance(prefix, min_cluster_size) == prefix[index]);
    assert(core_distance(extended, min_cluster_size) == extended[index]);
}

}
