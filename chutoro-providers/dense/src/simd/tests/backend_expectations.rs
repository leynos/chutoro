//! Backend expectation snapshots for SIMD tests.
//!
//! Keeps the semantics-contract and enabled-backend discovery snapshots out of
//! the root SIMD test module while preserving the same compile-time feature,
//! target, and runtime CPU checks.

use super::super::{dispatch, semantics};

#[test]
fn distance_semantics_contract_snapshot() {
    assert_eq!(
        format!("{:?}", semantics::DistanceSemantics::default_euclidean()),
        concat!(
            "DistanceSemantics { epsilon: 1e-5, ",
            "non_finite_policy: CanonicaliseToNan, ",
            "zero_vector_policy: ReturnZero, ",
            "tie_breaking: LowestRowIndexFirst }",
        ),
    );
}

/// Returns `true` when the AVX-512 backend is expected to appear in
/// [`dispatch::enabled_backends`] on the current host.
///
/// True iff the `simd_avx512` feature is compiled in, the target is
/// x86/x86_64, and AVX-512F is detected at runtime.
fn avx512_backend_expected() -> bool {
    cfg!(feature = "simd_avx512")
        && cfg!(any(target_arch = "x86", target_arch = "x86_64"))
        && runtime_avx512_expectation()
}

/// Returns `true` when the AVX2 backend is expected to appear in
/// [`dispatch::enabled_backends`] on the current host.
///
/// True iff the `simd_avx2` feature is compiled in, the target is
/// x86/x86_64, and AVX2 is detected at runtime.
fn avx2_backend_expected() -> bool {
    cfg!(feature = "simd_avx2")
        && cfg!(any(target_arch = "x86", target_arch = "x86_64"))
        && runtime_avx2_expectation()
}

/// Returns `true` when the Neon backend is expected to appear in
/// [`dispatch::enabled_backends`] on the current host.
///
/// True iff the `simd_neon` feature is compiled in, the target is
/// arm/aarch64, and Neon is detected at runtime.
fn neon_backend_expected() -> bool {
    cfg!(feature = "simd_neon")
        && cfg!(any(target_arch = "arm", target_arch = "aarch64"))
        && runtime_neon_expectation()
}

#[test]
fn enabled_backends_output_matches_support_snapshot() {
    let mut expected = Vec::new();
    if avx512_backend_expected() {
        expected.push(dispatch::EuclideanBackend::Avx512);
    }
    if avx2_backend_expected() {
        expected.push(dispatch::EuclideanBackend::Avx2);
    }
    if neon_backend_expected() {
        expected.push(dispatch::EuclideanBackend::Neon);
    }
    if cfg!(all(feature = "nightly_portable_simd", nightly)) {
        expected.push(dispatch::EuclideanBackend::PortableSimd);
    }
    expected.push(dispatch::EuclideanBackend::Scalar);

    assert_eq!(dispatch::enabled_backends(), expected);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn runtime_avx2_expectation() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn runtime_avx2_expectation() -> bool {
    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn runtime_avx512_expectation() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn runtime_avx512_expectation() -> bool {
    false
}

#[cfg(target_arch = "arm")]
fn runtime_neon_expectation() -> bool {
    std::arch::is_arm_feature_detected!("neon")
}

#[cfg(target_arch = "aarch64")]
fn runtime_neon_expectation() -> bool {
    true
}

#[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
fn runtime_neon_expectation() -> bool {
    false
}
