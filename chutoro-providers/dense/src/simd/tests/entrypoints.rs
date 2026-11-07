//! Entry-point-specific SIMD backend tests.

#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "simd_avx2", feature = "simd_avx512")
    ),
    all(
        feature = "simd_neon",
        any(target_arch = "arm", target_arch = "aarch64")
    ),
    all(feature = "nightly_portable_simd", nightly)
))]
use super::kernels;
#[cfg(all(feature = "nightly_portable_simd", nightly))]
use super::{DensePointView, Dimension, MatrixValues, RowCount, RowIndex, RowMajorMatrix};
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "simd_avx2", feature = "simd_avx512")
    ),
    all(feature = "nightly_portable_simd", nightly)
))]
use super::{Distance, assert_close};
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "simd_avx2", feature = "simd_avx512")
    ),
    all(feature = "nightly_portable_simd", nightly)
))]
use rstest::rstest;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "simd_avx2", feature = "simd_avx512")
))]
#[rstest]
#[cfg_attr(
    all(
        feature = "simd_avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    case::avx2(
        "avx2",
        kernels::euclidean_distance_avx2_entry,
        (35_u32, 0.5_f32, 0.25_f32)
    )
)]
#[cfg_attr(
    all(
        feature = "simd_avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    case::avx512(
        "avx512f",
        kernels::euclidean_distance_avx512_entry,
        (67_u32, 0.125_f32, 0.375_f32)
    )
)]
fn x86_entrypoint_matches_scalar_when_available(
    #[case] runtime_feature: &str,
    #[case] entry: fn(&[f32], &[f32]) -> f32,
    #[case] input: (u32, f32, f32),
) {
    if !x86_feature_detected(runtime_feature) {
        return;
    }

    let (len, left_scale, right_scale) = input;
    let (left, right) = scaled_pair_fixture(len, left_scale, right_scale);

    let expected = kernels::euclidean_distance_scalar(&left, &right);
    let actual = entry(&left, &right);
    assert_close!(Distance::new(actual), Distance::new(expected));
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "simd_avx2", feature = "simd_avx512")
))]
#[rstest]
#[cfg_attr(
    all(
        feature = "simd_avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    case::avx2("avx2", kernels::euclidean_distance_avx2_entry)
)]
#[cfg_attr(
    all(
        feature = "simd_avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ),
    case::avx512("avx512f", kernels::euclidean_distance_avx512_entry)
)]
fn x86_entrypoint_canonicalizes_non_finite_to_nan_when_available(
    #[case] runtime_feature: &str,
    #[case] entry: fn(&[f32], &[f32]) -> f32,
) {
    if !x86_feature_detected(runtime_feature) {
        return;
    }

    let left = [f32::INFINITY, 1.0_f32];
    let right = [0.0_f32, 0.0_f32];
    let actual = entry(&left, &right);

    assert!(actual.is_nan());
}

#[cfg(all(feature = "nightly_portable_simd", nightly))]
#[rstest]
#[case(1_u32)]
#[case(7_u32)]
#[case(16_u32)]
#[case(17_u32)]
#[case(35_u32)]
#[case(67_u32)]
#[case(128_u32)]
fn portable_simd_pairwise_matches_scalar(#[case] len: u32) {
    let (left, right) = scaled_pair_fixture(len, 0.5_f32, 0.25_f32);

    let expected = kernels::euclidean_distance_scalar(&left, &right);
    let actual = kernels::euclidean_distance_portable_simd_entry(&left, &right);

    assert_close!(Distance::new(actual), Distance::new(expected));
}

#[cfg(all(feature = "nightly_portable_simd", nightly))]
#[rstest]
#[case(3_usize)]
#[case(16_usize)]
#[case(17_usize)]
#[case(33_usize)]
fn portable_simd_query_points_matches_scalar(#[case] point_count: usize) {
    /// Width chosen so every point row contributes more than one lane.
    const DIMENSION: usize = 4;

    let (values, point_indices) = distinct_query_points_fixture(point_count, DIMENSION);
    let matrix = RowMajorMatrix::new(
        MatrixValues::new(&values),
        RowCount::new(point_count + 1),
        Dimension::new(DIMENSION),
    );
    let query = matrix.row(RowIndex::new(0)).expect("query row");
    let points = DensePointView::from_row_indices(matrix, &point_indices).expect("point view");
    let mut expected = vec![0.0_f32; point_count];
    let mut actual = vec![0.0_f32; point_count];

    kernels::euclidean_distance_query_points_scalar(query.as_slice(), &points, &mut expected);
    kernels::euclidean_distance_query_points_portable_simd_entry(
        query.as_slice(),
        &points,
        &mut actual,
    );

    assert_query_outputs_match(&actual, &expected);
}

/// Builds a query-to-points fixture whose rows are pairwise distinct.
///
/// Row `0` holds the query and rows `1..=point_count` hold the packed points.
/// Consecutive point rows differ by half a unit in every column, so an
/// incorrect chunk offset, a skipped later block, or a lost scalar tail changes
/// at least one output distance. The chosen values are exactly representable in
/// `f32`, keeping the scalar comparison exact.
#[cfg(all(feature = "nightly_portable_simd", nightly))]
#[expect(
    clippy::cast_precision_loss,
    reason = "the bounded fixture row and column indices fit exactly in f32"
)]
#[expect(
    clippy::float_arithmetic,
    reason = "the fixture deliberately generates scaled floating-point values"
)]
fn distinct_query_points_fixture(
    point_count: usize,
    dimension: usize,
) -> (Vec<f32>, Vec<RowIndex>) {
    let row_count = point_count + 1;
    let values = (0..row_count)
        .flat_map(|row| {
            (0..dimension).map(move |column| row as f32 * 0.5 + column as f32 * 0.25 + 1.0)
        })
        .collect();
    let point_indices = (1..=point_count).map(RowIndex::new).collect();
    (values, point_indices)
}

/// Asserts every packed-query output matches its scalar reference.
///
/// Reports the failing output index so a wrong block offset or a missing
/// trailing block is easy to locate.
#[cfg(all(feature = "nightly_portable_simd", nightly))]
fn assert_query_outputs_match(actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "query output count must match",
    );
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        let actual_distance = Distance::new(actual_value);
        let expected_distance = Distance::new(expected_value);
        let delta = super::distance_delta(actual_distance, expected_distance);
        assert!(
            delta <= super::CLOSE_TOLERANCE,
            "output {index}: expected {expected_value}, got {actual_value}, delta {delta}",
        );
    }
}

#[cfg(all(feature = "nightly_portable_simd", nightly))]
#[test]
fn portable_simd_canonicalizes_non_finite_to_nan() {
    let left = [f32::INFINITY, 1.0_f32];
    let right = [0.0_f32, 0.0_f32];

    assert!(kernels::euclidean_distance_portable_simd_entry(&left, &right).is_nan());
}

#[cfg(all(
    feature = "simd_neon",
    any(target_arch = "arm", target_arch = "aarch64")
))]
#[test]
fn neon_entrypoint_canonicalizes_non_finite_to_nan() {
    #[cfg(target_arch = "arm")]
    if !std::arch::is_arm_feature_detected!("neon") {
        return;
    }

    let left = [f32::INFINITY, 1.0_f32];
    let right = [0.0_f32, 0.0_f32];
    let actual = kernels::euclidean_distance_neon_entry(&left, &right);

    assert!(actual.is_nan());
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "simd_avx2", feature = "simd_avx512")
))]
fn x86_feature_detected(feature: &str) -> bool {
    match feature {
        "avx2" => std::arch::is_x86_feature_detected!("avx2"),
        "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
        unexpected => panic!("unexpected x86 runtime feature: {unexpected}"),
    }
}

/// Builds a bounded, scaled vector pair for pairwise backend comparisons.
#[expect(
    clippy::cast_precision_loss,
    reason = "the bounded u32 fixture lengths fit exactly in f32"
)]
#[expect(
    clippy::float_arithmetic,
    reason = "the fixture deliberately generates scaled floating-point vectors"
)]
fn scaled_pair_fixture(len: u32, left_scale: f32, right_scale: f32) -> (Vec<f32>, Vec<f32>) {
    let left = (0_u32..len)
        .map(|index| index as f32 * left_scale)
        .collect();
    let right = (0_u32..len)
        .map(|index| (len - index) as f32 * right_scale)
        .collect();
    (left, right)
}
