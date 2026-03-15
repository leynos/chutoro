//! Compile-time and runtime backend selection for dense SIMD kernels.

use std::sync::OnceLock;

/// Euclidean distance backend chosen for the current build and machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EuclideanBackend {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

/// Backends compiled into the current binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompiledSimdSupport {
    avx2: bool,
    avx512: bool,
    neon: bool,
}

impl CompiledSimdSupport {
    /// Builds a support mask for parameterized tests.
    #[must_use]
    pub(super) const fn new(avx2: bool, avx512: bool, neon: bool) -> Self {
        Self { avx2, avx512, neon }
    }
}

/// Backends available on the current machine at runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RuntimeSimdSupport {
    avx2: bool,
    avx512: bool,
    neon: bool,
}

impl RuntimeSimdSupport {
    /// Builds a runtime support mask for parameterized tests.
    #[must_use]
    pub(super) const fn new(avx2: bool, avx512: bool, neon: bool) -> Self {
        Self { avx2, avx512, neon }
    }
}

static EUCLIDEAN_BACKEND: OnceLock<EuclideanBackend> = OnceLock::new();

pub(super) fn euclidean_backend() -> EuclideanBackend {
    *EUCLIDEAN_BACKEND.get_or_init(select_euclidean_backend)
}

pub(super) fn compiled_simd_support() -> CompiledSimdSupport {
    CompiledSimdSupport::new(
        cfg!(feature = "simd_avx2") && cfg!(any(target_arch = "x86", target_arch = "x86_64")),
        cfg!(feature = "simd_avx512") && cfg!(any(target_arch = "x86", target_arch = "x86_64")),
        cfg!(feature = "simd_neon") && cfg!(any(target_arch = "arm", target_arch = "aarch64")),
    )
}

pub(super) fn runtime_simd_support() -> RuntimeSimdSupport {
    RuntimeSimdSupport::new(
        runtime_avx2_support(),
        runtime_avx512_support(),
        runtime_neon_support(),
    )
}

#[must_use]
pub(super) fn choose_euclidean_backend(
    compiled: CompiledSimdSupport,
    runtime: RuntimeSimdSupport,
) -> EuclideanBackend {
    if compiled.avx512 && runtime.avx512 {
        EuclideanBackend::Avx512
    } else if compiled.avx2 && runtime.avx2 {
        EuclideanBackend::Avx2
    } else if compiled.neon && runtime.neon {
        EuclideanBackend::Neon
    } else {
        EuclideanBackend::Scalar
    }
}

fn select_euclidean_backend() -> EuclideanBackend {
    choose_euclidean_backend(compiled_simd_support(), runtime_simd_support())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn runtime_avx2_support() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn runtime_avx2_support() -> bool {
    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn runtime_avx512_support() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn runtime_avx512_support() -> bool {
    false
}

#[cfg(target_arch = "arm")]
fn runtime_neon_support() -> bool {
    std::arch::is_arm_feature_detected!("neon")
}

#[cfg(target_arch = "aarch64")]
fn runtime_neon_support() -> bool {
    true
}

#[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
fn runtime_neon_support() -> bool {
    false
}
