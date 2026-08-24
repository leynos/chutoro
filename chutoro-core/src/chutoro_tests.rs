//! Unit tests for the Chutoro runtime facade.

use std::num::NonZeroUsize;

use super::*;
use crate::ChutoroBuilder;

fn execution_config(min_cluster_size: NonZeroUsize) -> ExecutionConfig {
    #[cfg(feature = "cpu")]
    {
        ExecutionConfig::new(min_cluster_size, crate::HnswParams::default())
    }
    #[cfg(not(feature = "cpu"))]
    {
        ExecutionConfig::new(min_cluster_size)
    }
}

#[test]
fn gpu_preferred_requires_gpu_feature() {
    let chutoro = Chutoro::new(
        execution_config(NonZeroUsize::new(1).expect("literal 1 is non-zero")),
        ExecutionStrategy::GpuPreferred,
        None,
    );
    let err = chutoro.backend_unavailable_error();
    assert!(matches!(
        err,
        Some(ChutoroError::BackendUnavailable {
            requested: ExecutionStrategy::GpuPreferred
        })
    ));
}

#[test]
fn backend_available_when_features_enabled() {
    if cfg!(feature = "cpu") {
        for strategy in [ExecutionStrategy::Auto, ExecutionStrategy::CpuOnly] {
            let chutoro = Chutoro::new(
                execution_config(NonZeroUsize::new(1).expect("literal 1 is non-zero")),
                strategy,
                None,
            );
            assert!(chutoro.backend_unavailable_error().is_none());
        }
    }

    let chutoro = Chutoro::new(
        execution_config(NonZeroUsize::new(1).expect("literal 1 is non-zero")),
        ExecutionStrategy::GpuPreferred,
        None,
    );
    assert!(matches!(
        chutoro.backend_unavailable_error(),
        Some(ChutoroError::BackendUnavailable {
            requested: ExecutionStrategy::GpuPreferred
        })
    ));
}

#[test]
fn max_bytes_none_imposes_no_limit() {
    let chutoro = ChutoroBuilder::new().build().expect("build must succeed");
    assert_eq!(chutoro.max_bytes(), None);
}

#[test]
fn max_bytes_propagates_through_builder() {
    let chutoro = ChutoroBuilder::new()
        .with_max_bytes(1_000_000)
        .build()
        .expect("build must succeed");
    assert_eq!(chutoro.max_bytes(), Some(1_000_000));
}

#[cfg(feature = "cpu")]
#[test]
fn builder_hnsw_params_reach_chutoro_execution_config() {
    let params = crate::HnswParams::new(4, 16).expect("parameters must be valid");
    let chutoro = ChutoroBuilder::new()
        .with_hnsw_params(params.clone())
        .build()
        .expect("build must succeed");

    assert_eq!(chutoro.hnsw_params(), &params);
}
