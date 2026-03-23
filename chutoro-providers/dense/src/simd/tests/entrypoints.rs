//! Entry-point-specific SIMD backend tests.

#[cfg(all(
    feature = "simd_neon",
    any(target_arch = "arm", target_arch = "aarch64")
))]
use super::kernels;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "simd_avx2", feature = "simd_avx512")
))]
use super::{Distance, close, kernels};
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "simd_avx2", feature = "simd_avx512")
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
    let left: Vec<f32> = (0_u32..len)
        .map(|index| index as f32 * left_scale)
        .collect();
    let right: Vec<f32> = (0_u32..len)
        .map(|index| (len - index) as f32 * right_scale)
        .collect();

    let expected = kernels::euclidean_distance_scalar(&left, &right);
    let actual = entry(&left, &right);
    close(Distance::new(actual), Distance::new(expected));
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
        _ => unreachable!("unexpected x86 runtime feature"),
    }
}
