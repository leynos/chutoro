//! Distance-semantics contract shared by dense SIMD parity tests.

use super::{DensePointView, kernels};

// These enums intentionally start with one variant each. They keep the
// verification contract explicit now while leaving a narrow migration point for
// future GPU parity and nearest-neighbour tie-breaking policy work.

/// Policy for non-finite Euclidean distance outputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NonFinitePolicy {
    /// Convert every non-finite final distance to `f32::NAN`.
    CanonicaliseToNan,
}

/// Policy for vectors whose Euclidean norm difference is zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ZeroVectorPolicy {
    /// Return exactly `0.0` for identical zero-valued vectors.
    ReturnZero,
}

/// Policy for ordering equal distances in future selection logic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TieBreakingPolicy {
    /// Prefer the lowest row index when two candidate distances tie.
    LowestRowIndexFirst,
}

/// Dense Euclidean distance semantics used by backend parity checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DistanceSemantics {
    epsilon: f32,
    non_finite_policy: NonFinitePolicy,
    zero_vector_policy: ZeroVectorPolicy,
    tie_breaking: TieBreakingPolicy,
}

impl DistanceSemantics {
    /// Builds the default Euclidean contract for dense SIMD kernels.
    ///
    /// The `1.0e-5` epsilon is intentionally wider than the older scalar
    /// smoke-test tolerance because SIMD reductions accumulate lanes in a
    /// different order from the scalar oracle.
    #[must_use]
    pub(crate) const fn default_euclidean() -> Self {
        Self {
            epsilon: 1.0e-5_f32,
            non_finite_policy: NonFinitePolicy::CanonicaliseToNan,
            zero_vector_policy: ZeroVectorPolicy::ReturnZero,
            tie_breaking: TieBreakingPolicy::LowestRowIndexFirst,
        }
    }

    /// Computes the scalar oracle for a pairwise Euclidean distance.
    #[must_use]
    pub(crate) fn oracle_pairwise(self, left: &[f32], right: &[f32]) -> f32 {
        self.debug_assert_contract();
        kernels::euclidean_distance_scalar(left, right)
    }

    /// Computes the scalar oracle for query-to-points Euclidean distances.
    pub(crate) fn oracle_query_points(
        self,
        query: &[f32],
        points: &DensePointView<'_>,
        out: &mut [f32],
    ) {
        debug_assert_eq!(
            out.len(),
            points.point_count(),
            "output slice length must match point count",
        );
        self.debug_assert_contract();
        kernels::euclidean_distance_query_points_scalar(query, points, out);
    }

    /// Asserts that an actual backend output matches the scalar oracle.
    pub(crate) fn assert_close(self, actual: f32, expected: f32) {
        if self.should_accept_non_finite(actual, expected) {
            return;
        }

        let delta = (actual - expected).abs();
        assert!(
            delta <= self.epsilon,
            "actual={actual}, expected={expected}, delta={delta}, epsilon={}",
            self.epsilon,
        );
    }

    /// Asserts that all query-to-points outputs match the scalar oracle.
    pub(crate) fn assert_query_close(self, actual: &[f32], expected: &[f32]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "query output lengths must match",
        );
        for (index, (actual_distance, expected_distance)) in actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .enumerate()
        {
            if self.should_accept_non_finite(actual_distance, expected_distance) {
                continue;
            }
            let delta = (actual_distance - expected_distance).abs();
            assert!(
                delta <= self.epsilon,
                concat!(
                    "index={index}, actual={actual_distance}, expected={expected_distance}, ",
                    "delta={delta}, epsilon={epsilon}",
                ),
                index = index,
                actual_distance = actual_distance,
                expected_distance = expected_distance,
                delta = delta,
                epsilon = self.epsilon,
            );
        }
    }

    fn should_accept_non_finite(self, actual: f32, expected: f32) -> bool {
        matches!(self.non_finite_policy, NonFinitePolicy::CanonicaliseToNan)
            && actual.is_nan()
            && expected.is_nan()
    }

    fn debug_assert_contract(self) {
        debug_assert!(matches!(
            self.zero_vector_policy,
            ZeroVectorPolicy::ReturnZero,
        ));
        debug_assert!(matches!(
            self.tie_breaking,
            TieBreakingPolicy::LowestRowIndexFirst,
        ));
    }
}
