//! Diagnostic helpers for neighbour-scoring benchmarks.

use std::time::Duration;

mod build_profile;
mod profiling;
mod report;

#[doc(hidden)]
pub use build_profile::{
    BUILD_PROFILE_ENV, BUILD_PROFILE_REPORT, REPORT_DIR_NAME, build_profile_report_target,
    build_profile_report_target_value, report_parent_dir, report_parent_dir_value, report_path,
    report_path_value, should_collect_build_profile, should_collect_build_profile_value,
    truthy_env_value,
};
#[doc(hidden)]
pub use profiling::{duration_nanos, saturating_add_u64, saturating_add_usize};
pub use report::{
    BuildProfileReportRow, LaneUtilisationReportRow, write_build_profile_report_csv,
    write_lane_utilisation_report_csv,
};

/// Number of candidate lanes represented by one dense-provider `SoA` block.
const SIMD_LANES: usize = 16;

/// Calculate active-lane utilisation in basis points for one candidate bucket.
///
/// Returns `0` for an empty bucket. Non-empty buckets are rounded down to the
/// integer basis-point value implied by padding the bucket to a full
/// `SIMD_LANES` block.
///
/// # Examples
///
/// ```
/// use chutoro_benches::neighbour_scoring::lane_utilisation_basis_points;
///
/// assert_eq!(lane_utilisation_basis_points(8), 5_000);
/// assert_eq!(lane_utilisation_basis_points(16), 10_000);
/// ```
#[expect(
    clippy::integer_division,
    reason = "basis-point utilisation is an integer diagnostic report"
)]
#[expect(
    clippy::integer_division_remainder_used,
    reason = "basis-point utilisation is an integer diagnostic report"
)]
#[must_use]
pub fn lane_utilisation_basis_points(candidate_count: usize) -> usize {
    if candidate_count == 0 {
        return 0;
    }
    let padded = candidate_count
        .checked_next_multiple_of(SIMD_LANES)
        .map_or_else(
            || padded_lane_count(candidate_count),
            |padded| padded as u128,
        );
    let basis_points = (candidate_count as u128).saturating_mul(10_000) / padded;
    usize::try_from(basis_points).unwrap_or(10_000)
}

const fn padded_lane_count(candidate_count: usize) -> u128 {
    let candidate_count_u128 = candidate_count as u128;
    let simd_lanes = SIMD_LANES as u128;
    candidate_count_u128.div_ceil(simd_lanes) * simd_lanes
}

/// Calculate a duration ratio in basis points, clamped to `0..=10_000`.
///
/// A zero whole duration returns `0`; a part longer than the whole returns
/// `10_000`.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use chutoro_benches::neighbour_scoring::duration_basis_points;
///
/// assert_eq!(
///     duration_basis_points(Duration::from_millis(1), Duration::from_millis(4)),
///     2_500,
/// );
/// ```
#[expect(
    clippy::integer_division,
    reason = "basis-point duration ratio is an integer diagnostic report"
)]
#[expect(
    clippy::integer_division_remainder_used,
    reason = "basis-point duration ratio is an integer diagnostic report"
)]
#[must_use]
pub fn duration_basis_points(part: Duration, whole: Duration) -> usize {
    if whole.is_zero() {
        return 0;
    }
    let whole_nanos = whole.as_nanos();
    let part_nanos = part.as_nanos().min(whole_nanos);
    let rounded_basis_points = ((part_nanos * 10_000) + (whole_nanos / 2)) / whole_nanos;

    usize::try_from(rounded_basis_points.min(10_000)).unwrap_or(10_000)
}

/// Return the integer median of diagnostic values sorted in ascending order.
///
/// Empty input returns `0`. Even-length input returns the midpoint between the
/// two central values, rounded down.
///
/// # Examples
///
/// ```
/// use chutoro_benches::neighbour_scoring::sorted_median;
///
/// assert_eq!(sorted_median(&[8, 16, 24]), 16);
/// assert_eq!(sorted_median(&[8, 16, 24, 32]), 20);
/// ```
#[expect(
    clippy::integer_division,
    reason = "integer median is an integer diagnostic report"
)]
#[expect(
    clippy::integer_division_remainder_used,
    reason = "integer median is an integer diagnostic report"
)]
#[must_use]
pub fn sorted_median(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }

    let upper_index = values.len() / 2;
    let Some(&upper) = values.get(upper_index) else {
        return 0;
    };
    if values.len().is_multiple_of(2) {
        let Some(&lower) = values.get(upper_index.saturating_sub(1)) else {
            return upper;
        };
        lower + (upper - lower) / 2
    } else {
        upper
    }
}

#[cfg(test)]
mod tests {
    //! Example and property checks for neighbour-scoring diagnostics.

    use std::time::Duration;

    use proptest::prelude::*;
    use rstest::rstest;

    use super::{SIMD_LANES, duration_basis_points, lane_utilisation_basis_points, sorted_median};

    #[rstest]
    #[case(0, 0)]
    #[case(8, 5_000)]
    #[case(16, 10_000)]
    #[case(24, 7_500)]
    #[case(usize::MAX, 9_999)]
    fn lane_utilisation_handles_candidate_counts(
        #[case] candidate_count: usize,
        #[case] expected: usize,
    ) {
        assert_eq!(lane_utilisation_basis_points(candidate_count), expected);
    }

    #[rstest]
    #[case(&[], 0)]
    #[case(&[8, 16, 24], 16)]
    #[case(&[8, 16, 24, 32], 20)]
    #[case(&[usize::MIN, usize::MIN], usize::MIN)]
    #[case(&[8, 8, 8, 8], 8)]
    #[case(&[usize::MAX - 2, usize::MAX], usize::MAX - 1)]
    fn sorted_median_returns_expected_value(#[case] values: &[usize], #[case] expected: usize) {
        assert_eq!(sorted_median(values), expected);
    }

    proptest! {
        #[test]
        fn sorted_median_respects_sorted_slice_contract(
            mut values in prop::collection::vec(any::<usize>(), 0..64),
        ) {
            values.sort_unstable();
            let result = sorted_median(&values);

            if values.is_empty() {
                prop_assert_eq!(result, 0);
            } else {
                let upper_index = values.len() >> 1;
                let upper = *values.get(upper_index).expect("upper median value exists");
                if values.len().is_multiple_of(2) {
                    let lower = *values
                        .get(upper_index.saturating_sub(1))
                        .expect("lower median value exists");
                    prop_assert!(lower <= result);
                    prop_assert!(result <= upper);
                    prop_assert_eq!(result, lower + ((upper - lower) >> 1));
                } else {
                    prop_assert_eq!(result, upper);
                }
            }
        }
    }

    #[rstest]
    #[case(Duration::from_nanos(1), Duration::ZERO, 0)]
    #[case(Duration::from_millis(1), Duration::from_millis(4), 2_500)]
    #[case(Duration::from_millis(1), Duration::from_millis(6), 1_667)]
    #[case(Duration::from_millis(8), Duration::from_millis(4), 10_000)]
    fn duration_basis_points_reports_expected_values(
        #[case] part: Duration,
        #[case] whole: Duration,
        #[case] expected: usize,
    ) {
        assert_eq!(duration_basis_points(part, whole), expected);
    }

    proptest! {
        #[test]
        fn lane_utilisation_is_bounded(candidate_count in 1_usize..=1_000_000) {
            let utilisation = lane_utilisation_basis_points(candidate_count);

            prop_assert!((1..=10_000).contains(&utilisation));
            if candidate_count.next_multiple_of(SIMD_LANES) == candidate_count {
                prop_assert_eq!(utilisation, 10_000);
            }
        }

        #[test]
        fn lane_utilisation_increases_within_one_padding_block(
            bucket in 0_usize..=1_024,
            lower_offset in 1_usize..=SIMD_LANES,
            offset_delta in 0_usize..SIMD_LANES,
        ) {
            let upper_offset = lower_offset.saturating_add(offset_delta).min(SIMD_LANES);
            let block_start = bucket.saturating_mul(SIMD_LANES);
            let lower_candidate_count = block_start.saturating_add(lower_offset);
            let upper_candidate_count = block_start.saturating_add(upper_offset);

            prop_assert!(
                lane_utilisation_basis_points(lower_candidate_count)
                    <= lane_utilisation_basis_points(upper_candidate_count)
            );
        }

        #[test]
        fn duration_basis_points_is_bounded_and_clamped(
            part_nanos in 0_u64..=1_000_000_000,
            whole_nanos in 1_u64..=1_000_000_000,
        ) {
            let part = Duration::from_nanos(part_nanos);
            let whole = Duration::from_nanos(whole_nanos);
            let basis_points = duration_basis_points(part, whole);

            prop_assert!((0..=10_000).contains(&basis_points));
            if part_nanos == 0 {
                prop_assert_eq!(basis_points, 0);
            }
            if part_nanos >= whole_nanos {
                prop_assert_eq!(basis_points, 10_000);
            }
        }

        #[test]
        fn duration_basis_points_increases_with_part(
            whole_nanos in 1_u64..=1_000_000_000,
            first_part_nanos in 0_u64..=1_000_000_000,
            second_part_nanos in 0_u64..=1_000_000_000,
        ) {
            let lower = first_part_nanos.min(second_part_nanos);
            let upper = first_part_nanos.max(second_part_nanos);
            let whole = Duration::from_nanos(whole_nanos);

            prop_assert!(
                duration_basis_points(Duration::from_nanos(lower), whole)
                    <= duration_basis_points(Duration::from_nanos(upper), whole)
            );
        }
    }
}
