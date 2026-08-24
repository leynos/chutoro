//! Property tests for shared one-shot and session execution configuration.

use std::sync::{Arc, atomic::AtomicUsize};

use proptest::prelude::*;
use tracing_subscriber::layer::SubscriberExt;

use super::*;
use crate::{
    ChutoroBuilder, ExecutionStrategy, HnswParams, estimate_peak_bytes_for_hnsw_params,
    test_utils::{CountingSource, suite_proptest_config},
};
use chutoro_test_support::tracing::RecordingLayer;

fn shared_execution_config_strategy() -> impl Strategy<Value = (usize, usize, usize, usize)> {
    (4_usize..=8, 1_usize..=4, 0_usize..=4).prop_flat_map(
        |(point_count, max_connections, extra_construction_width)| {
            (
                Just(point_count),
                1_usize..=point_count,
                Just(max_connections),
                Just(max_connections + extra_construction_width),
            )
        },
    )
}

proptest! {
    #![proptest_config(suite_proptest_config(4))]

    #[test]
    fn shared_execution_config_reaches_batch_and_session_paths(
        (point_count, min_cluster_size, max_connections, ef_construction)
            in shared_execution_config_strategy(),
    ) {
        let params = HnswParams::new(max_connections, ef_construction)
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let source = CountingSource::new(
            (0..point_count).map(|point| point as f32).collect(),
            Arc::new(AtomicUsize::new(0)),
        );
        let estimate = estimate_peak_bytes_for_hnsw_params(point_count, &params);

        let memory_limited = ChutoroBuilder::new()
            .with_min_cluster_size(min_cluster_size)
            .with_execution_strategy(ExecutionStrategy::CpuOnly)
            .with_hnsw_params(params.clone())
            .with_max_bytes(estimate - 1)
            .build()
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let memory_error = memory_limited
            .run(&source)
            .expect_err("one byte below the generated estimate must reject the batch");
        let ChutoroError::MemoryLimitExceeded { estimated_bytes, .. } = memory_error else {
            return Err(TestCaseError::fail("memory guard must return MemoryLimitExceeded"));
        };
        prop_assert_eq!(estimated_bytes, estimate);

        let session = ChutoroBuilder::new()
            .with_min_cluster_size(min_cluster_size)
            .with_execution_strategy(ExecutionStrategy::CpuOnly)
            .with_hnsw_params(params.clone())
            .build_session(Arc::new(source.clone()))
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        prop_assert_eq!(session.config().min_cluster_size().get(), min_cluster_size);
        prop_assert_eq!(session.config().hnsw_params(), &params);

        let layer = RecordingLayer::default();
        let subscriber = tracing_subscriber::registry().with(layer.clone());
        let runnable = ChutoroBuilder::new()
            .with_min_cluster_size(min_cluster_size)
            .with_execution_strategy(ExecutionStrategy::CpuOnly)
            .with_hnsw_params(params.clone())
            .with_max_bytes(estimate)
            .build()
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let result = tracing::subscriber::with_default(subscriber, || runnable.run(&source));
        prop_assert!(result.is_ok());

        let event = layer
            .events()
            .into_iter()
            .find(|event| {
                event
                    .fields
                    .get("message")
                    .is_some_and(|message| message == "building CPU HNSW index")
            })
            .ok_or_else(|| TestCaseError::fail("CPU HNSW construction event must be recorded"))?;
        prop_assert_eq!(event.fields.get("max_connections"), Some(&max_connections.to_string()));
        prop_assert_eq!(
            event.fields.get("configured_ef_construction"),
            Some(&ef_construction.to_string()),
        );
        prop_assert_eq!(
            event.fields.get("effective_ef_construction"),
            Some(&params.effective_ef_construction(point_count).to_string()),
        );
    }
}
