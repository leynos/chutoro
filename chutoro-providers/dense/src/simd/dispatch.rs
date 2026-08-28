//! Compile-time and runtime backend selection for dense SIMD kernels.

use std::sync::OnceLock;

/// Ordered SIMD backends preferred over the scalar implementation.
const EUCLIDEAN_SIMD_BACKEND_PRIORITY: [EuclideanBackend; 4] = [
    EuclideanBackend::Avx512,
    EuclideanBackend::Avx2,
    EuclideanBackend::Neon,
    EuclideanBackend::PortableSimd,
];

/// Euclidean distance backend chosen for the current build and machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EuclideanBackend {
    /// Portable scalar implementation.
    Scalar,
    /// x86 AVX2 implementation.
    Avx2,
    /// x86 AVX-512 implementation.
    Avx512,
    /// ARM NEON implementation.
    Neon,
    /// Nightly portable-SIMD implementation.
    PortableSimd,
}

/// Backends compiled into the current binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompiledSimdSupport {
    /// CPU-specific SIMD support compiled for this target.
    cpu: CpuSimdSupport,
    /// Whether the portable-SIMD implementation was compiled.
    portable_simd: bool,
}

impl CompiledSimdSupport {
    /// Builds a support mask for parameterized tests.
    #[must_use]
    pub(super) const fn new(cpu: CpuSimdSupport, portable_simd: bool) -> Self {
        Self { cpu, portable_simd }
    }
}

/// CPU SIMD backends represented by the target's native feature probes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CpuSimdSupport {
    /// Whether the AVX2 implementation is supported.
    avx2: bool,
    /// Whether the AVX-512 implementation is supported.
    avx512: bool,
    /// Whether the NEON implementation is supported.
    neon: bool,
}

impl CpuSimdSupport {
    /// Builds the CPU SIMD portion of a backend support mask.
    #[must_use]
    pub(super) const fn new(avx2: bool, avx512: bool, neon: bool) -> Self {
        Self { avx2, avx512, neon }
    }
}

/// Backends available on the current machine at runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RuntimeSimdSupport {
    /// CPU-specific SIMD support detected at runtime.
    cpu: CpuSimdSupport,
    /// Whether portable SIMD is enabled for this runtime.
    portable_simd: bool,
}

impl RuntimeSimdSupport {
    /// Builds a runtime support mask for parameterized tests.
    #[must_use]
    pub(super) const fn new(cpu: CpuSimdSupport, portable_simd: bool) -> Self {
        Self { cpu, portable_simd }
    }
}

/// Lazily initialized Euclidean backend selected for this process.
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
/// `simd_avx2`/`simd_avx512` for x86 or `x86_64`, and `simd_neon` for arm or
/// aarch64. The optional `nightly_portable_simd` backend additionally requires
/// a nightly compiler so stable `--all-features` builds remain valid.
pub(super) const fn compiled_simd_support() -> CompiledSimdSupport {
    CompiledSimdSupport::new(
        CpuSimdSupport::new(
            cfg!(feature = "simd_avx2") && cfg!(any(target_arch = "x86", target_arch = "x86_64")),
            cfg!(feature = "simd_avx512") && cfg!(any(target_arch = "x86", target_arch = "x86_64")),
            cfg!(feature = "simd_neon") && cfg!(any(target_arch = "arm", target_arch = "aarch64")),
        ),
        cfg!(all(feature = "nightly_portable_simd", nightly)),
    )
}

/// Returns runtime SIMD support flags detected on the current machine.
///
/// This checks AVX2 and AVX-512F with x86 CPUID helpers, checks NEON at
/// runtime on 32-bit ARM, and treats `AArch64` as NEON-capable because Advanced
/// SIMD is part of the base architecture. Portable SIMD has no separate
/// runtime probe beyond the compile-time nightly feature gate.
pub(super) fn runtime_simd_support() -> RuntimeSimdSupport {
    RuntimeSimdSupport::new(
        CpuSimdSupport::new(
            runtime_avx2_support(),
            runtime_avx512_support(),
            runtime_neon_support(),
        ),
        cfg!(all(feature = "nightly_portable_simd", nightly)),
    )
}

/// Returns every [`EuclideanBackend`] that is both compiled into this build
/// and available on the current host at runtime.
///
/// Intersects [`compiled_simd_support`] with [`runtime_simd_support`] and
/// filters each variant through [`backend_supported`]. Used by parity tests
/// to enumerate only the backends that can actually execute in the current
/// test process, so the property suite adapts automatically to the build
/// configuration and host CPU.
#[cfg(test)]
pub(super) fn enabled_backends() -> Vec<EuclideanBackend> {
    let compiled = compiled_simd_support();
    let runtime = runtime_simd_support();
    EUCLIDEAN_SIMD_BACKEND_PRIORITY
        .into_iter()
        .chain([EuclideanBackend::Scalar])
        .filter(|backend| backend_supported(compiled, runtime, *backend))
        .collect()
}

/// Chooses the best Euclidean backend available to both compile-time and
/// runtime support masks.
///
/// The selection order is deterministic: prefer AVX-512, then AVX2, then
/// NEON, then portable SIMD, and fall back to `Scalar` when no SIMD backend is
/// both compiled and available at runtime.
#[must_use]
pub(super) fn choose_euclidean_backend(
    compiled: CompiledSimdSupport,
    runtime: RuntimeSimdSupport,
) -> EuclideanBackend {
    select_backend(compiled, runtime)
}

/// Select the process-wide backend from compiled and runtime support.
fn select_euclidean_backend() -> EuclideanBackend {
    choose_euclidean_backend(compiled_simd_support(), runtime_simd_support())
}

/// Select the highest-priority backend supported by both masks.
fn select_backend(compiled: CompiledSimdSupport, runtime: RuntimeSimdSupport) -> EuclideanBackend {
    for backend in EUCLIDEAN_SIMD_BACKEND_PRIORITY {
        if backend_supported(compiled, runtime, backend) {
            return backend;
        }
    }

    EuclideanBackend::Scalar
}

/// Determine whether a backend is enabled in both support masks.
const fn backend_supported(
    compiled: CompiledSimdSupport,
    runtime: RuntimeSimdSupport,
    variant: EuclideanBackend,
) -> bool {
    match variant {
        EuclideanBackend::Avx512 => compiled.cpu.avx512 && runtime.cpu.avx512,
        EuclideanBackend::Avx2 => compiled.cpu.avx2 && runtime.cpu.avx2,
        EuclideanBackend::Neon => compiled.cpu.neon && runtime.cpu.neon,
        EuclideanBackend::PortableSimd => compiled.portable_simd && runtime.portable_simd,
        EuclideanBackend::Scalar => true,
    }
}

/// Detect runtime AVX2 support on x86 targets.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn runtime_avx2_support() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

/// Report that AVX2 is unavailable on non-x86 targets.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn runtime_avx2_support() -> bool {
    false
}

/// Detect runtime AVX-512 support on x86 targets.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn runtime_avx512_support() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
}

/// Report that AVX-512 is unavailable on non-x86 targets.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn runtime_avx512_support() -> bool {
    false
}

/// Detect runtime NEON support on 32-bit ARM targets.
#[cfg(target_arch = "arm")]
fn runtime_neon_support() -> bool {
    std::arch::is_arm_feature_detected!("neon")
}

/// Report baseline NEON support on AArch64 targets.
#[cfg(target_arch = "aarch64")]
fn runtime_neon_support() -> bool {
    // AArch64 mandates Advanced SIMD, so there is no separate runtime probe.
    true
}

/// Report that NEON is unavailable on non-ARM targets.
#[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
const fn runtime_neon_support() -> bool {
    false
}
