//! Private backend-selection policy for one-shot clustering.
//!
//! This module is limited to [`crate::Chutoro`] orchestration. It keeps
//! feature availability, strategy resolution, and metric labels consistent;
//! it is not a general backend registry or public extension point.

use crate::ExecutionStrategy;

/// Whether this build includes the CPU execution pipeline.
const CPU_PATH_AVAILABLE: bool = cfg!(feature = "cpu");
// The `gpu` feature currently exposes the orchestration surface only;
// no accelerated implementation ships yet.
/// Whether this build includes a usable GPU execution pipeline.
const GPU_PATH_AVAILABLE: bool = false;

/// Concrete backend selected after resolving an execution strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendChoice {
    /// Execute through the CPU pipeline.
    Cpu,
    /// Execute through the GPU pipeline.
    Gpu,
}

/// Resolves the configured strategy to the backend that would run.
pub(crate) const fn choose_backend(strategy: ExecutionStrategy) -> BackendChoice {
    match strategy {
        ExecutionStrategy::Auto => {
            if CPU_PATH_AVAILABLE {
                BackendChoice::Cpu
            } else {
                BackendChoice::Gpu
            }
        }
        ExecutionStrategy::CpuOnly => BackendChoice::Cpu,
        ExecutionStrategy::GpuPreferred => BackendChoice::Gpu,
    }
}

/// Reports whether the configured strategy has no compiled implementation.
pub(crate) const fn is_backend_unavailable(strategy: ExecutionStrategy) -> bool {
    match strategy {
        ExecutionStrategy::Auto => !(CPU_PATH_AVAILABLE || GPU_PATH_AVAILABLE),
        ExecutionStrategy::CpuOnly => !CPU_PATH_AVAILABLE,
        ExecutionStrategy::GpuPreferred => !GPU_PATH_AVAILABLE,
    }
}

/// Returns the bounded label for the selected or unavailable backend.
pub(crate) const fn backend_label(strategy: ExecutionStrategy) -> &'static str {
    if is_backend_unavailable(strategy) {
        "unavailable"
    } else {
        match choose_backend(strategy) {
            BackendChoice::Cpu => "cpu",
            BackendChoice::Gpu => "gpu",
        }
    }
}
