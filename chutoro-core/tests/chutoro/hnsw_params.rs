//! HNSW-parameter and memory-limit cases for the `Chutoro` orchestration API.
//!
//! These cases size the memory guard from configured HNSW parameters and check
//! that the CPU pipeline receives the builder's values. They are a private child
//! of the `chutoro` integration-test root, which declares them with `#[path]` so
//! they still run in one integration-test binary while both files stay within
//! the module-size budget.

use chutoro_core::{
    ChutoroBuilder, ChutoroError, DataSource, DistanceCacheConfig, ExecutionStrategy, HnswParams,
    estimate_peak_bytes, estimate_peak_bytes_for_hnsw_params_with_concurrency,
};
use chutoro_test_support::tracing::{RecordingLayer, pin_callsite_interest};
use rstest::rstest;
use std::num::NonZeroUsize;
use tracing_subscriber::layer::SubscriberExt;

use crate::common::Dummy;
use crate::dummy;

/// Mirrors the concurrency the memory guard derives for this test thread.
///
/// The guard sizes concurrent search state from the ambient Rayon pool, and the
/// tests call it from the same context, so the wide pool is the honest width.
fn ambient_search_concurrency() -> NonZeroUsize {
    NonZeroUsize::new(rayon::current_num_threads()).unwrap_or(NonZeroUsize::MIN)
}

/// Returns the cluster size the memory-limit fixtures configure.
const fn cluster_size() -> NonZeroUsize {
    match NonZeroUsize::new(2) {
        Some(size) => size,
        None => NonZeroUsize::MIN,
    }
}

#[rstest]
fn run_memory_limit_uses_builder_hnsw_params(dummy: Dummy) {
    let hnsw_params = HnswParams::new(4, 16).expect("parameters must be valid");
    let estimated_bytes = estimate_peak_bytes_for_hnsw_params_with_concurrency(
        dummy.len(),
        cluster_size(),
        &hnsw_params,
        ambient_search_concurrency(),
    );
    assert_ne!(estimated_bytes, estimate_peak_bytes(dummy.len(), 16));

    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .with_execution_strategy(ExecutionStrategy::CpuOnly)
        .with_hnsw_params(hnsw_params)
        .with_max_bytes(estimated_bytes - 1)
        .build()
        .expect("configuration must be valid");

    let err = chutoro
        .run(&dummy)
        .expect_err("configured memory limit must be enforced");
    assert!(matches!(
        err,
        ChutoroError::MemoryLimitExceeded {
            point_count,
            estimated_bytes: actual_estimated_bytes,
            ..
        } if point_count == dummy.len() && actual_estimated_bytes == estimated_bytes
    ));
}

#[rstest]
fn run_memory_limit_uses_configured_distance_cache_capacity(dummy: Dummy) {
    let default_params = HnswParams::new(16, 64).expect("parameters must be valid");
    let cache_capacity =
        NonZeroUsize::new(DistanceCacheConfig::DEFAULT_MAX_ENTRIES.saturating_mul(2))
            .expect("doubled default cache capacity must be non-zero");
    let hnsw_params = default_params
        .clone()
        .with_distance_cache_max_entries(cache_capacity);
    let searches = ambient_search_concurrency();
    let default_estimate = estimate_peak_bytes_for_hnsw_params_with_concurrency(
        dummy.len(),
        cluster_size(),
        &default_params,
        searches,
    );
    let parameter_aware_estimate = estimate_peak_bytes_for_hnsw_params_with_concurrency(
        dummy.len(),
        cluster_size(),
        &hnsw_params,
        searches,
    );
    let max_bytes = parameter_aware_estimate - 1;
    assert!(
        max_bytes > default_estimate,
        "the configured cache must increase the estimate beyond the default cache"
    );

    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .with_execution_strategy(ExecutionStrategy::CpuOnly)
        .with_hnsw_params(hnsw_params)
        .with_max_bytes(max_bytes)
        .build()
        .expect("configuration must be valid");

    let error = chutoro
        .run(&dummy)
        .expect_err("parameter-aware memory limit must be enforced");
    assert!(matches!(
        error,
        ChutoroError::MemoryLimitExceeded {
            estimated_bytes,
            ..
        } if estimated_bytes == parameter_aware_estimate
    ));
}

#[rstest]
fn run_passes_builder_hnsw_params_to_cpu_pipeline(dummy: Dummy) {
    let _pin = pin_callsite_interest();
    let hnsw_params = HnswParams::new(1, 2).expect("parameters must be valid");
    let layer = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .with_execution_strategy(ExecutionStrategy::CpuOnly)
        .with_hnsw_params(hnsw_params)
        .build()
        .expect("configuration must be valid");

    let result = tracing::subscriber::with_default(subscriber, || chutoro.run(&dummy))
        .expect("custom HNSW configuration must complete the CPU pipeline");
    assert_eq!(result.assignments().len(), dummy.len());

    let event = layer
        .events()
        .into_iter()
        .find(|event| {
            event
                .fields
                .get("message")
                .is_some_and(|message| message == "building CPU HNSW index")
        })
        .expect("CPU pipeline must record its HNSW construction parameters");
    assert_eq!(event.fields.get("max_connections"), Some(&"1".to_owned()));
    assert_eq!(
        event.fields.get("configured_ef_construction"),
        Some(&"2".to_owned())
    );
    assert_eq!(
        event.fields.get("effective_ef_construction"),
        Some(&"2".to_owned())
    );
}
