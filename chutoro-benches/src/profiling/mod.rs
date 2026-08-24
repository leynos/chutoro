//! Memory profiling support for HNSW benchmarks.
//!
//! Provides a Linux resident-set sampler plus report helpers that compute
//! memory-per-point and memory-per-edge metrics for benchmark runs.

mod memory_sampler;

use std::{
    fs,
    path::{Path, PathBuf},
};

pub use memory_sampler::{PeakRssMeasurement, measure_peak_resident_set_size};
use thiserror::Error;

/// Validates whether harvested edge count is within expected scaling bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EdgeScalingBounds {
    /// Divisor used for the lower edge-count bound.
    lower_multiplier: usize,
    /// Multiplier used for the upper edge-count bound.
    upper_multiplier: usize,
}

impl EdgeScalingBounds {
    /// Creates scaling bounds expressed as multiplicative tolerances.
    ///
    /// `lower_multiplier = 8` means the measured edge count is accepted when it
    /// is at least `expected / 8`.
    #[must_use]
    pub const fn new(lower_multiplier: usize, upper_multiplier: usize) -> Self {
        Self {
            lower_multiplier,
            upper_multiplier,
        }
    }

    /// Return the divisor used for the lower edge-count bound.
    #[must_use]
    const fn lower_multiplier(self) -> usize {
        self.lower_multiplier
    }

    /// Return the multiplier used for the upper edge-count bound.
    #[must_use]
    const fn upper_multiplier(self) -> usize {
        self.upper_multiplier
    }
}

impl Default for EdgeScalingBounds {
    fn default() -> Self {
        Self::new(8, 8)
    }
}

/// Errors raised while sampling or reporting benchmark memory metrics.
#[derive(Debug, Error)]
pub enum ProfilingError {
    /// Any input/output failure while reading process information or writing reports.
    #[error("profiling I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The `/proc/self/status` parser could not locate a required key.
    #[error("missing field `{field}` in /proc/self/status")]
    MissingProcField {
        /// The required field name.
        field: &'static str,
    },
    /// A numeric field in `/proc/self/status` could not be parsed.
    #[error("invalid numeric value `{value}` for /proc field `{field}`")]
    InvalidProcField {
        /// The required field name.
        field: &'static str,
        /// The unparsable value.
        value: String,
    },
    /// A process status field had an unexpected unit.
    #[error("unsupported unit `{unit}` for /proc field `{field}`")]
    UnsupportedProcUnit {
        /// The required field name.
        field: &'static str,
        /// The unit found in `/proc`.
        unit: String,
    },
    /// Sampling is unavailable on the current operating system.
    #[error("peak resident-set sampling is not supported on `{os}`")]
    UnsupportedPlatform {
        /// Name of the unsupported operating system.
        os: &'static str,
    },
    /// A denominator was zero while deriving memory metrics.
    #[error("cannot compute metric because `{context}` is zero")]
    ZeroDenominator {
        /// Name of the zero denominator.
        context: &'static str,
    },
    /// Arithmetic overflow occurred while deriving or validating metrics.
    #[error("arithmetic overflow while computing `{context}`")]
    Overflow {
        /// Name of the overflowed calculation.
        context: &'static str,
    },
    /// The peak-memory sampler thread failed to join successfully.
    #[error("failed to join peak-memory sampler thread")]
    SamplerThreadPanicked,
    /// The peak-memory sampler encountered poisoned shared state.
    #[error("peak-memory sampler lock was poisoned")]
    SamplerLockPoisoned,
    /// Sampling interval must be greater than zero.
    #[error("sampling interval must be greater than zero")]
    ZeroSamplingInterval,
}

/// Input payload used to construct a [`HnswMemoryRecord`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HnswMemoryInput {
    /// Number of points inserted into HNSW.
    pub point_count: usize,
    /// Configured HNSW max connections (`M`).
    pub max_connections: usize,
    /// Configured HNSW `ef_construction`.
    pub ef_construction: usize,
    /// Measured elapsed time and peak memory delta from baseline.
    pub measurement: PeakRssMeasurement,
    /// Number of harvested candidate edges.
    pub edge_count: usize,
}

/// Single row in the HNSW memory profile report.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HnswMemoryRecord {
    /// Number of points in the profiled index.
    point_count: usize,
    /// Configured maximum connections per HNSW node.
    max_connections: usize,
    /// Configured construction search width.
    ef_construction: usize,
    /// Elapsed build time in milliseconds.
    elapsed_millis: u128,
    /// Peak resident-set increase in bytes.
    peak_rss_bytes: u64,
    /// Number of harvested candidate edges.
    edge_count: usize,
    /// Expected edge count from points and maximum connections.
    expected_edges: usize,
    /// Whether the measured edge count falls within configured bounds.
    edge_scaling_ok: bool,
    /// Absolute difference between measured and expected edge counts.
    edge_deviation: usize,
    /// Peak memory increase per point in bytes.
    memory_per_point_bytes: u64,
    /// Peak memory increase per edge in bytes.
    memory_per_edge_bytes: u64,
}

impl HnswMemoryRecord {
    /// Builds a report row from measured HNSW build stats.
    ///
    /// # Errors
    ///
    /// Returns [`ProfilingError`] when edge expectations overflow, metric
    /// denominators are zero, or scaling bounds are invalid.
    pub fn new(input: HnswMemoryInput, bounds: EdgeScalingBounds) -> Result<Self, ProfilingError> {
        let expected_edges = input.point_count.checked_mul(input.max_connections).ok_or(
            ProfilingError::Overflow {
                context: "expected_edges",
            },
        )?;
        let memory_per_point_bytes = divide_metric(
            input.measurement.peak_rss_bytes,
            input.point_count,
            "point_count",
        )?;
        let memory_per_edge_bytes = divide_metric(
            input.measurement.peak_rss_bytes,
            input.edge_count,
            "edge_count",
        )?;
        let (edge_scaling_ok, edge_deviation) =
            validate_edge_scaling(input.edge_count, expected_edges, bounds)?;

        Ok(Self {
            point_count: input.point_count,
            max_connections: input.max_connections,
            ef_construction: input.ef_construction,
            elapsed_millis: input.measurement.elapsed.as_millis(),
            peak_rss_bytes: input.measurement.peak_rss_bytes,
            edge_count: input.edge_count,
            expected_edges,
            edge_scaling_ok,
            edge_deviation,
            memory_per_point_bytes,
            memory_per_edge_bytes,
        })
    }

    /// Return the CSV header for memory profile records.
    const fn csv_header() -> &'static str {
        concat!(
            "point_count,max_connections,ef_construction,elapsed_ms,peak_rss_bytes,",
            "memory_per_point_bytes,edge_count,memory_per_edge_bytes,expected_edges,",
            "edge_scaling_ok,edge_deviation\n",
        )
    }

    /// Render this memory profile record as one CSV row.
    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            self.point_count,
            self.max_connections,
            self.ef_construction,
            self.elapsed_millis,
            self.peak_rss_bytes,
            self.memory_per_point_bytes,
            self.edge_count,
            self.memory_per_edge_bytes,
            self.expected_edges,
            self.edge_scaling_ok,
            self.edge_deviation,
        )
    }
}

/// Divide a metric after validating its non-zero denominator.
#[expect(
    clippy::integer_division,
    clippy::integer_division_remainder_used,
    reason = "Metrics are intentionally truncated to whole bytes after a non-zero denominator check."
)]
fn divide_metric(
    numerator: u64,
    denominator: usize,
    context: &'static str,
) -> Result<u64, ProfilingError> {
    if denominator == 0 {
        return Err(ProfilingError::ZeroDenominator { context });
    }
    let denominator_u64 = u64::try_from(denominator).map_err(|_| ProfilingError::Overflow {
        context: "usize_to_u64_denominator",
    })?;
    Ok(numerator / denominator_u64)
}

/// Check edge-count scaling and report its deviation from expectation.
fn validate_edge_scaling(
    edge_count: usize,
    expected_edges: usize,
    bounds: EdgeScalingBounds,
) -> Result<(bool, usize), ProfilingError> {
    if bounds.lower_multiplier() == 0 {
        return Err(ProfilingError::ZeroDenominator {
            context: "lower_multiplier",
        });
    }
    if bounds.upper_multiplier() == 0 {
        return Err(ProfilingError::ZeroDenominator {
            context: "upper_multiplier",
        });
    }
    let lower_scale =
        edge_count
            .checked_mul(bounds.lower_multiplier())
            .ok_or(ProfilingError::Overflow {
                context: "lower_scaled_edge_count",
            })?;
    let upper_bound = expected_edges
        .checked_mul(bounds.upper_multiplier())
        .ok_or(ProfilingError::Overflow {
            context: "upper_bound_edges",
        })?;
    let scaling_ok = lower_scale >= expected_edges && edge_count <= upper_bound;
    Ok((scaling_ok, edge_count.abs_diff(expected_edges)))
}

/// Writes HNSW memory profiling records to a comma-separated report file.
///
/// # Errors
///
/// Returns [`ProfilingError`] when creating the parent directory or writing
/// the report file fails.
pub fn write_hnsw_memory_report(
    report_path: impl AsRef<Path>,
    records: &[HnswMemoryRecord],
) -> Result<PathBuf, ProfilingError> {
    let report_file_path = report_path.as_ref().to_path_buf();
    if let Some(parent) = report_file_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut output = String::from(HnswMemoryRecord::csv_header());
    for record in records {
        output.push_str(&record.to_csv_row());
    }
    fs::write(&report_file_path, output)?;
    Ok(report_file_path)
}

#[cfg(test)]
mod tests;
