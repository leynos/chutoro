//! Shared core-distance selection rule.
//!
//! FISHDBC and HDBSCAN define a point's core distance as the distance to its
//! `m`-th nearest neighbour, where `m` is the minimum cluster size. The batch
//! CPU pipeline, the incremental session path, and the benchmark harness all
//! consume the same rule, so it lives here once instead of being re-implemented
//! per call site. Adapter code is still responsible for searching HNSW and
//! removing the query point from the neighbour list.

use std::num::NonZeroUsize;

use crate::Neighbour;

/// Computes a point's core distance from its sorted non-self neighbours.
///
/// `neighbours` must be ordered by ascending distance and must not contain the
/// query point. The function itself never panics: it reads the `m - 1` element
/// when at least `m` neighbours are available, falls back to the farthest
/// supplied neighbour when the list is shorter, and returns `0.0` for an empty
/// list. Because selection is positional, equal-distance neighbours resolve in
/// list order, which keeps results deterministic for a given search result.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use chutoro_core::{Neighbour, core_distance_from_neighbours};
///
/// let neighbours = [
///     Neighbour { id: 1, distance: 0.5 },
///     Neighbour { id: 2, distance: 1.5 },
///     Neighbour { id: 3, distance: 2.5 },
/// ];
/// let min_cluster_size = NonZeroUsize::new(2).expect("two is non-zero");
///
/// // At least `m` neighbours: take the `m - 1` element.
/// assert_eq!(core_distance_from_neighbours(&neighbours, min_cluster_size), 1.5);
///
/// // Too few neighbours: take the farthest available one.
/// let large = NonZeroUsize::new(5).expect("five is non-zero");
/// assert_eq!(core_distance_from_neighbours(&neighbours, large), 2.5);
/// assert_eq!(core_distance_from_neighbours(&[], large), 0.0);
/// ```
#[must_use]
pub fn core_distance_from_neighbours(
    neighbours: &[Neighbour],
    min_cluster_size: NonZeroUsize,
) -> f32 {
    neighbours
        .get(min_cluster_size.get().saturating_sub(1))
        .or_else(|| neighbours.last())
        .map_or(0.0, |neighbour| neighbour.distance)
}

#[cfg(test)]
mod tests {
    //! Selection-rule tests covering dense, sparse, empty, and tie cases.

    use std::num::NonZeroUsize;

    use rstest::rstest;

    use super::core_distance_from_neighbours;
    use crate::Neighbour;

    /// Builds neighbours with increasing distances starting at `first`.
    #[expect(
        clippy::float_arithmetic,
        clippy::cast_precision_loss,
        reason = "fixture distances are small whole numbers built from indices"
    )]
    fn neighbours(count: usize, first: f32) -> Vec<Neighbour> {
        (0..count)
            .map(|index| Neighbour {
                id: index,
                distance: first + index as f32,
            })
            .collect()
    }

    /// Returns `count` as a non-zero cluster size.
    fn min_cluster_size(count: usize) -> NonZeroUsize {
        NonZeroUsize::new(count).expect("test cluster sizes are non-zero")
    }

    #[rstest]
    #[case::exactly_m(3, 3, 2.0)]
    #[case::more_than_m(5, 3, 2.0)]
    #[case::fewer_than_m(2, 3, 1.0)]
    #[case::singleton_list(1, 4, 0.0)]
    fn selects_the_ranked_neighbour_or_the_farthest(
        #[case] available: usize,
        #[case] cluster_size: usize,
        #[case] expected: f32,
    ) {
        let neighbours = neighbours(available, 0.0);

        assert_eq!(
            core_distance_from_neighbours(&neighbours, min_cluster_size(cluster_size)),
            expected
        );
    }

    #[test]
    fn empty_list_has_zero_core_distance() {
        assert_eq!(core_distance_from_neighbours(&[], min_cluster_size(2)), 0.0);
    }

    #[test]
    fn minimum_cluster_size_one_selects_the_nearest_neighbour() {
        let neighbours = neighbours(3, 0.5);

        assert_eq!(
            core_distance_from_neighbours(&neighbours, min_cluster_size(1)),
            0.5
        );
    }

    #[test]
    fn equal_distances_resolve_in_list_order() {
        let tied = [
            Neighbour {
                id: 7,
                distance: 1.0,
            },
            Neighbour {
                id: 9,
                distance: 1.0,
            },
            Neighbour {
                id: 4,
                distance: 1.0,
            },
        ];

        assert_eq!(
            core_distance_from_neighbours(&tied, min_cluster_size(2)),
            1.0
        );
        assert_eq!(
            core_distance_from_neighbours(&tied, min_cluster_size(9)),
            1.0
        );
    }
}
