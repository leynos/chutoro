//! Bounded metrics emitted by the one-shot clustering execution path.
//!
//! This module owns the stable batch metric vocabulary. It accepts only
//! resource values and bounded labels, keeping source names and payload data
//! outside the metrics surface.

use crate::Result;

#[cfg(feature = "cpu")]
use std::time::Duration;

/// Counter name for final one-shot batch outcomes.
const RUNS_TOTAL: &str = "chutoro.batch.runs_total";
#[cfg(feature = "cpu")]
/// Histogram name for configured CPU HNSW connection counts.
const MAX_CONNECTIONS: &str = "chutoro.batch.max_connections";
#[cfg(feature = "cpu")]
/// Histogram name for dataset-bounded CPU construction search width.
const EFFECTIVE_EF_CONSTRUCTION: &str = "chutoro.batch.effective_ef_construction";
#[cfg(feature = "cpu")]
/// Histogram name for the estimated CPU batch peak-memory requirement.
const ESTIMATED_BYTES: &str = "chutoro.batch.estimated_bytes";
#[cfg(feature = "cpu")]
/// Histogram name for configured CPU batch memory limits.
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
    metrics::histogram!(MAX_CONNECTIONS, "backend" => "cpu")
        .record(bounded_count_as_f64(max_connections));
    metrics::histogram!(EFFECTIVE_EF_CONSTRUCTION, "backend" => "cpu")
        .record(bounded_count_as_f64(effective_ef_construction));
    metrics::histogram!(ESTIMATED_BYTES, "backend" => "cpu")
        .record(bounded_bytes_as_f64(estimated_bytes));

    if let Some(limit) = memory_limit_bytes {
        metrics::describe_histogram!(
            MEMORY_LIMIT_BYTES,
            metrics::Unit::Bytes,
            "Configured memory-limit bytes for one-shot CPU batch runs."
        );
        metrics::histogram!(MEMORY_LIMIT_BYTES, "backend" => "cpu")
            .record(bounded_bytes_as_f64(limit));
    }
}

/// Histogram name for CPU pipeline stage wall-clock time.
#[cfg(feature = "cpu")]
const PIPELINE_STAGE_SECONDS: &str = "chutoro.cpu_pipeline.stage_seconds";

/// Records the wall-clock duration of one CPU pipeline stage.
///
/// The `stage` label is supplied by the pipeline's own stage vocabulary, whose
/// variants are the complete label set. Passing the label rather than text
/// derived from runtime data keeps the histogram's cardinality bounded.
#[cfg(feature = "cpu")]
pub(crate) fn record_pipeline_stage(stage: &'static str, elapsed: Duration) {
    metrics::describe_histogram!(
        PIPELINE_STAGE_SECONDS,
        metrics::Unit::Seconds,
        "Wall-clock seconds spent in each CPU pipeline stage, by bounded stage name."
    );
    metrics::histogram!(PIPELINE_STAGE_SECONDS, "stage" => stage).record(elapsed.as_secs_f64());
}

/// Converts a count to the exact range representable by the metrics recorder.
#[cfg(feature = "cpu")]
fn bounded_count_as_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

/// Largest integer an `f64` represents exactly: 2^53 bytes, or 8 PiB.
#[cfg(feature = "cpu")]
const MAX_EXACT_F64_BYTES: u64 = 1 << 53;

/// Converts a byte value to the exact range representable by the metrics recorder.
///
/// Memory limits routinely exceed `u32::MAX`, so the value is rebuilt from its
/// two 32-bit halves, each of which converts losslessly. Both the product and
/// the sum stay below 2^53, so the result is exact; only values above 8 PiB
/// saturate.
#[cfg(feature = "cpu")]
fn bounded_bytes_as_f64(value: u64) -> f64 {
    const HALF_WORD: f64 = 4_294_967_296.0;
    let bounded = value.min(MAX_EXACT_F64_BYTES);
    let high = u32::try_from(bounded >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(bounded & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(high).mul_add(HALF_WORD, f64::from(low))
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    //! Unit tests for exact byte conversion.

    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::zero(0, 0.0)]
    #[case::u32_max(u64::from(u32::MAX), 4_294_967_295.0)]
    #[case::eight_gibibytes(8 << 30, 8_589_934_592.0)]
    #[case::odd_above_u32(u64::from(u32::MAX) + 2, 4_294_967_297.0)]
    #[case::exact_limit(MAX_EXACT_F64_BYTES, 9_007_199_254_740_992.0)]
    #[case::saturates(u64::MAX, 9_007_199_254_740_992.0)]
    fn bytes_convert_exactly_up_to_two_to_the_fifty_third(
        #[case] bytes: u64,
        #[case] expected: f64,
    ) {
        assert_eq!(bounded_bytes_as_f64(bytes).to_bits(), expected.to_bits());
    }
}
