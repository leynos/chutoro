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

/// Returns the initialized Euclidean backend for the current build and host.
///
/// This is the public accessor for the backend selected once via
/// [`choose_euclidean_backend`]. Subsequent calls reuse the cached choice.
pub(super) fn euclidean_backend() -> EuclideanBackend {
    *EUCLIDEAN_BACKEND.get_or_init(select_euclidean_backend)
}

/// Returns compile-time SIMD support flags for the current target and features.
///
/// The returned mask reports which backend implementations were compiled into
/// the binary by Cargo features and target architecture:
/// `simd_avx2`/`simd_avx512` for x86 or x86_64, and `simd_neon` for arm or
/// aarch64.
pub(super) fn compiled_simd_support() -> CompiledSimdSupport {
    CompiledSimdSupport::new(
        cfg!(feature = "simd_avx2") && cfg!(any(target_arch = "x86", target_arch = "x86_64")),
        cfg!(feature = "simd_avx512") && cfg!(any(target_arch = "x86", target_arch = "x86_64")),
        cfg!(feature = "simd_neon") && cfg!(any(target_arch = "arm", target_arch = "aarch64")),
    )
}

/// Returns runtime SIMD support flags detected on the current machine.
///
/// This checks AVX2 and AVX-512F with x86 CPUID helpers, checks NEON at
/// runtime on 32-bit ARM, and treats AArch64 as NEON-capable because Advanced
/// SIMD is part of the base architecture.
pub(super) fn runtime_simd_support() -> RuntimeSimdSupport {
    RuntimeSimdSupport::new(
        runtime_avx2_support(),
        runtime_avx512_support(),
        runtime_neon_support(),
    )
}

/// Chooses the best Euclidean backend available to both compile-time and
/// runtime support masks.
///
/// The selection order is deterministic: prefer AVX-512, then AVX2, then
/// NEON, and fall back to `Scalar` when no SIMD backend is both compiled and
/// available at runtime.
#[must_use]
pub(super) fn choose_euclidean_backend(
    compiled: CompiledSimdSupport,
    runtime: RuntimeSimdSupport,
) -> EuclideanBackend {
    for backend in [
        EuclideanBackend::Avx512,
        EuclideanBackend::Avx2,
        EuclideanBackend::Neon,
        EuclideanBackend::Scalar,
    ] {
        if backend_supported(&compiled, &runtime, backend) {
            return backend;
        }
    }

    EuclideanBackend::Scalar
}

fn select_euclidean_backend() -> EuclideanBackend {
    choose_euclidean_backend(compiled_simd_support(), runtime_simd_support())
}

fn backend_supported(
    compiled: &CompiledSimdSupport,
    runtime: &RuntimeSimdSupport,
    variant: EuclideanBackend,
) -> bool {
    match variant {
        EuclideanBackend::Avx512 => compiled.avx512 && runtime.avx512,
        EuclideanBackend::Avx2 => compiled.avx2 && runtime.avx2,
        EuclideanBackend::Neon => compiled.neon && runtime.neon,
        EuclideanBackend::Scalar => true,
    }
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
    // AArch64 mandates Advanced SIMD, so there is no separate runtime probe.
    true
}

#[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
fn runtime_neon_support() -> bool {
    false
}
