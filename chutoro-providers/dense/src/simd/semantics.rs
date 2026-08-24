//! Euclidean distance semantics shared by dense SIMD parity tests.
//!
//! The semantics contract is a small value object that records the tolerance
//! and policy choices every backend must follow: `1.0e-5` epsilon,
//! `CanonicaliseToNan` for non-finite distances, `ReturnZero` for identical
//! zero vectors, and `LowestRowIndexFirst` for future equal-distance ordering.
//! Keeping those decisions in one object makes the parity assertions describe
//! behaviour rather than re-encoding scattered constants in each property.
//!
//! The epsilon is deliberately `1.0e-5` because SIMD implementations may
//! reduce lanes in a different order from the scalar oracle. That lane-ordering
//! divergence can produce small rounding differences even when the backend is
//! correct, so parity checks accept that bounded numerical variation while
//! still rejecting meaningful drift.
//!
//! The scalar oracle delegates directly to
//! [`kernels::euclidean_distance_scalar`] and
//! [`kernels::euclidean_distance_query_points_scalar`]. Those functions are the
//! unvectorised dense Euclidean implementation, so they define the ground truth
//! used to compare each SIMD entry point.
//!
//! This module sits between [`super::dispatch`] and [`super::kernels`]:
//! dispatch decides which backends are available, kernels resolve concrete
//! entry points, and the semantics contract defines how their outputs are
//! compared.

use proptest::prelude::{prop_assert, prop_assert_eq};
use proptest::test_runner::TestCaseResult;

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

    /// Checks that an actual backend output matches the scalar oracle.
    ///
    /// Reported as a [`TestCaseResult`] rather than a bare assertion so the
    /// proptest call sites keep their shrink reporting.
    pub(crate) fn check_close(self, actual: f32, expected: f32) -> TestCaseResult {
        if self.should_accept_non_finite(actual, expected) {
            return Ok(());
        }

        let delta = (actual - expected).abs();
        prop_assert!(
            delta <= self.epsilon,
            "actual={actual}, expected={expected}, delta={delta}, epsilon={}",
            self.epsilon,
        );
        Ok(())
    }

    /// Checks that all query-to-points outputs match the scalar oracle.
    ///
    /// Reported as a [`TestCaseResult`] for the same shrink-reporting reason as
    /// [`Self::check_close`].
    pub(crate) fn check_query_close(self, actual: &[f32], expected: &[f32]) -> TestCaseResult {
        prop_assert_eq!(
            actual.len(),
            expected.len(),
            "query output lengths must match"
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
            prop_assert!(
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
        Ok(())
    }

    /// Returns whether a non-finite result satisfies the configured policy.
    ///
    /// When the policy is `CanonicaliseToNan`, matching `NaN` values are
    /// accepted directly so callers can skip the finite-distance epsilon check.
    fn should_accept_non_finite(self, actual: f32, expected: f32) -> bool {
        matches!(self.non_finite_policy, NonFinitePolicy::CanonicaliseToNan)
            && actual.is_nan()
            && expected.is_nan()
    }

    /// Asserts in debug mode that single-variant policy fields are unchanged.
    ///
    /// This guards the current `ReturnZero` zero-vector policy and
    /// `LowestRowIndexFirst` tie-breaking policy while keeping release builds
    /// free of extra checks.
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
