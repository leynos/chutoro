//! Optional memory profiling for HNSW Criterion benchmarks.

use std::{path::PathBuf, time::Duration};

use mockable::{DefaultEnv, Env};

use chutoro_benches::{
    ef_sweep::make_bench_source,
    error::BenchSetupError,
    profiling::{
        EdgeScalingBounds, HnswMemoryInput, HnswMemoryRecord, ProfilingError,
        measure_peak_resident_set_size, write_hnsw_memory_report,
    },
};
use chutoro_core::CpuHnsw;

use super::environment::{memory_report_path_with_env, should_collect_memory_profile_with_env};
use super::{MAX_CONNECTIONS, POINT_COUNTS, make_hnsw_params};

/// Sampling cadence for peak resident-set-size profiling.
const MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_millis(2);

/// Multiplicative edge-scaling tolerance around `expected = n * M`.
const EDGE_SCALING_BOUNDS: EdgeScalingBounds = EdgeScalingBounds::new(8, 8);

/// Collect and write optional HNSW memory measurements before benchmark setup.
pub(super) fn profile_hnsw_memory_impl() -> Result<Option<PathBuf>, BenchSetupError> {
    profile_hnsw_memory_impl_with_env(&DefaultEnv)
}

/// Collect memory measurements through an injected environment reader.
fn profile_hnsw_memory_impl_with_env(env: &dyn Env) -> Result<Option<PathBuf>, BenchSetupError> {
    if !should_collect_memory_profile_with_env(env) {
        return Ok(None);
    }

    let report_path = memory_report_path_with_env(env);
    let mut records = Vec::new();

    for &point_count in POINT_COUNTS {
        let source = make_bench_source(point_count)?;

        for &m in MAX_CONNECTIONS {
            let params = make_hnsw_params(m)?;
            let ef_construction = params.ef_construction();
            let (build_result, measurement) =
                match measure_peak_resident_set_size(MEMORY_SAMPLE_INTERVAL, || {
                    CpuHnsw::build_with_edges(&source, params.clone())
                }) {
                    Ok(measurement) => measurement,
                    Err(ProfilingError::UnsupportedPlatform { .. }) => return Ok(None),
                    Err(err) => return Err(err.into()),
                };
            let (_index, harvest) = build_result?;
            records.push(HnswMemoryRecord::new(
                HnswMemoryInput {
                    point_count,
                    max_connections: m,
                    ef_construction,
                    measurement,
                    edge_count: harvest.len(),
                },
                EDGE_SCALING_BOUNDS,
            )?);
        }
    }

    write_hnsw_memory_report(&report_path, &records)
        .map(Some)
        .map_err(BenchSetupError::from)
}
