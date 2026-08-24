//! Bounded metrics emitted by the one-shot clustering execution path.
//!
//! This module owns the stable batch metric vocabulary. It accepts only
//! resource values and bounded labels, keeping source names and payload data
//! outside the metrics surface.

use crate::Result;

const RUNS_TOTAL: &str = "chutoro.batch.runs_total";
#[cfg(feature = "cpu")]
const MAX_CONNECTIONS: &str = "chutoro.batch.max_connections";
#[cfg(feature = "cpu")]
const EFFECTIVE_EF_CONSTRUCTION: &str = "chutoro.batch.effective_ef_construction";
#[cfg(feature = "cpu")]
const ESTIMATED_BYTES: &str = "chutoro.batch.estimated_bytes";
#[cfg(feature = "cpu")]
const MEMORY_LIMIT_BYTES: &str = "chutoro.batch.memory_limit_bytes";

/// Records the final outcome of a one-shot batch run.
pub(crate) fn record_outcome<T>(backend: &'static str, result: &Result<T>) {
    let (outcome, error_code) = match result {
        Ok(_) => ("success", "none"),
        Err(error) => ("error", error.code().as_str()),
    };

    metrics::describe_counter!(
        RUNS_TOTAL,
        metrics::Unit::Count,
        "Total one-shot batch runs by backend, outcome, and stable error code."
    );
    metrics::counter!(
        RUNS_TOTAL,
        "backend" => backend,
        "outcome" => outcome,
        "error_code" => error_code
    )
    .increment(1);
}

/// Records CPU HNSW and memory observations for a one-shot batch run.
#[cfg(feature = "cpu")]
pub(crate) fn record_cpu_resources(
    max_connections: usize,
    effective_ef_construction: usize,
    estimated_bytes: u64,
    memory_limit_bytes: Option<u64>,
) {
    metrics::describe_histogram!(
        MAX_CONNECTIONS,
        metrics::Unit::Count,
        "Configured CPU HNSW maximum connections for one-shot batch runs."
    );
    metrics::describe_histogram!(
        EFFECTIVE_EF_CONSTRUCTION,
        metrics::Unit::Count,
        "Dataset-bounded CPU HNSW construction search width for one-shot batch runs."
    );
    metrics::describe_histogram!(
        ESTIMATED_BYTES,
        metrics::Unit::Bytes,
        "Estimated peak bytes for one-shot CPU batch runs."
    );
    metrics::histogram!(MAX_CONNECTIONS, "backend" => "cpu").record(max_connections as f64);
    metrics::histogram!(EFFECTIVE_EF_CONSTRUCTION, "backend" => "cpu")
        .record(effective_ef_construction as f64);
    metrics::histogram!(ESTIMATED_BYTES, "backend" => "cpu").record(estimated_bytes as f64);

    if let Some(limit) = memory_limit_bytes {
        metrics::describe_histogram!(
            MEMORY_LIMIT_BYTES,
            metrics::Unit::Bytes,
            "Configured memory-limit bytes for one-shot CPU batch runs."
        );
        metrics::histogram!(MEMORY_LIMIT_BYTES, "backend" => "cpu").record(limit as f64);
    }
}
