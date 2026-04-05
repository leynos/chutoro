//! Tests for actual compile-time and runtime SIMD support masks.
//!
//! These runtime unit tests verify that the support detection functions
//! (`compiled_simd_support`, `runtime_simd_support`) behave correctly given
//! the active feature gates and target architecture. Complementary compile-time
//! tests enforce the stable/nightly portable-SIMD gating contract at build time
//! via trybuild in `tests/portable_simd_gating.rs`.

use super::super::dispatch::{
    self, CompiledSimdSupport, RuntimeSimdSupport, compiled_simd_support, runtime_simd_support,
};

/// Verifies that `compiled_simd_support()` returns a mask matching the active
/// feature gates and target architecture.
///
/// Complementary compile-time checks for the nightly portable-SIMD matrix are
/// provided by the trybuild tests in `tests/trybuild/`:
/// - `portable_simd_without_feature.rs` (compile-fail when feature absent)
/// - `portable_simd_with_feature.rs` (compile-pass when feature present)
#[test]
fn compiled_support_matches_active_target_and_feature_gates() {
    let expected = CompiledSimdSupport::new(
        cfg!(feature = "simd_avx2") && cfg!(any(target_arch = "x86", target_arch = "x86_64")),
        cfg!(feature = "simd_avx512") && cfg!(any(target_arch = "x86", target_arch = "x86_64")),
        cfg!(feature = "simd_neon") && cfg!(any(target_arch = "arm", target_arch = "aarch64")),
        cfg!(all(feature = "nightly_portable_simd", nightly)),
    );

    assert_eq!(compiled_simd_support(), expected);
}

#[test]
fn runtime_support_matches_host_detection_rules() {
    let expected = RuntimeSimdSupport::new(
        runtime_avx2_expectation(),
        runtime_avx512_expectation(),
        runtime_neon_expectation(),
        cfg!(all(feature = "nightly_portable_simd", nightly)),
    );

    assert_eq!(runtime_simd_support(), expected);
}

#[test]
fn cached_backend_selection_matches_support_masks() {
    let compiled = compiled_simd_support();
    let runtime = runtime_simd_support();

    assert_eq!(
        dispatch::euclidean_backend(),
        dispatch::choose_euclidean_backend(compiled, runtime)
    );
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
