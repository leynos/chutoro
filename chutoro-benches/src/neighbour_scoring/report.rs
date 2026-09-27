//! CSV report rendering for neighbour-scoring diagnostics.

use std::{
    borrow::Cow,
    io::{self, Write},
    time::Duration,
};

use super::{duration_basis_points, lane_utilization_basis_points, padded_lane_count};

/// One row in the lane-utilization diagnostic CSV report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaneUtilizationReportRow<'a> {
    /// Bucket label, for example `realistic` or `diagnostic`.
    pub bucket_kind: &'a str,
    /// Number of active candidates scored in the bucket.
    pub candidate_count: usize,
}

/// One row in the HNSW build-profile CSV report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuildProfileReportRow {
    /// Number of synthetic points in the profiled build.
    pub point_count: usize,
    /// Number of dimensions in each synthetic point.
    pub dimension: usize,
    /// Elapsed wall-clock build duration.
    pub build_elapsed: Duration,
    /// Accumulated time spent in batched `DataSource` scoring calls.
    pub batch_scoring_time: Duration,
    /// Number of batched scoring calls.
    pub batch_calls: usize,
    /// Number of scalar scoring calls.
    pub scalar_calls: usize,
    /// Total candidates scored through batched calls.
    pub total_batch_candidates: usize,
    /// Smallest observed batch size.
    pub min_batch: usize,
    /// Largest observed batch size.
    pub max_batch: usize,
    /// Median observed batch size.
    pub median_batch: usize,
}

/// Determine whether a CSV field needs quoting.
fn csv_requires_quotes(field: &str) -> bool {
    field
        .chars()
        .any(|ch| matches!(ch, ',' | '"' | '\n' | '\r'))
}

/// Return a CSV-safe representation of one field.
fn csv_escape(field: &str) -> Cow<'_, str> {
    if csv_requires_quotes(field) {
        let escaped = field.replace('"', "\"\"");
        Cow::Owned(format!("\"{escaped}\""))
    } else {
        Cow::Borrowed(field)
    }
}

/// Write the lane-utilization report schema and rows.
///
/// # Examples
///
/// ```
/// use chutoro_benches::neighbour_scoring::{
///     LaneUtilizationReportRow, write_lane_utilization_report_csv,
/// };
///
/// let mut output = Vec::new();
/// let result = write_lane_utilization_report_csv(
///     &mut output,
///     [LaneUtilizationReportRow {
///         bucket_kind: "realistic",
///         candidate_count: 8,
///     }],
/// );
/// assert!(result.is_ok());
/// assert_eq!(
///     output,
///     concat!(
///         "bucket_kind,candidate_count,padded_lanes,wasted_lanes,",
///         "lane_utilisation_basis_points\n",
///         "realistic,8,16,8,5000\n",
///     )
///     .as_bytes(),
/// );
/// ```
///
/// # Errors
///
/// Returns any error reported by the supplied writer.
pub fn write_lane_utilization_report_csv<'a, W, I>(mut writer: W, rows: I) -> io::Result<()>
where
    W: Write,
    I: IntoIterator<Item = LaneUtilizationReportRow<'a>>,
{
    writeln!(
        writer,
        "bucket_kind,candidate_count,padded_lanes,wasted_lanes,lane_utilisation_basis_points"
    )?;
    for row in rows {
        let padded = padded_lane_count(row.candidate_count);
        let wasted = padded.saturating_sub(row.candidate_count as u128);
        let utilization = lane_utilization_basis_points(row.candidate_count);
        let bucket_kind = csv_escape(row.bucket_kind);
        writeln!(
            writer,
            "{bucket_kind},{},{padded},{wasted},{utilization}",
            row.candidate_count
        )?;
    }
    Ok(())
}

/// Write the HNSW build-profile report schema and rows.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use chutoro_benches::neighbour_scoring::{
///     BuildProfileReportRow, write_build_profile_report_csv,
/// };
///
/// let mut output = Vec::new();
/// let result = write_build_profile_report_csv(
///     &mut output,
///     [BuildProfileReportRow {
///         point_count: 10,
///         dimension: 2,
///         build_elapsed: Duration::from_millis(4),
///         batch_scoring_time: Duration::from_millis(1),
///         batch_calls: 2,
///         scalar_calls: 1,
///         total_batch_candidates: 16,
///         min_batch: 8,
///         max_batch: 8,
///         median_batch: 8,
///     }],
/// );
/// assert!(result.is_ok());
/// assert_eq!(
///     output,
///     concat!(
///         "point_count,dimension,build_seconds,accumulated_batch_scoring_seconds,",
///         "accumulated_batch_scoring_vs_wall_basis_points,batch_calls,scalar_calls,",
///         "total_batch_candidates,min_batch,max_batch,median_batch\n",
///         "10,2,0.004000000,0.001000000,2500,2,1,16,8,8,8\n",
///     )
///     .as_bytes(),
/// );
/// ```
///
/// # Errors
///
/// Returns any error reported by the supplied writer.
pub fn write_build_profile_report_csv<W, I>(mut writer: W, rows: I) -> io::Result<()>
where
    W: Write,
    I: IntoIterator<Item = BuildProfileReportRow>,
{
    writeln!(
        writer,
        concat!(
            "point_count,dimension,build_seconds,accumulated_batch_scoring_seconds,",
            "accumulated_batch_scoring_vs_wall_basis_points,batch_calls,scalar_calls,",
            "total_batch_candidates,min_batch,max_batch,median_batch"
        )
    )?;
    for row in rows {
        let accumulated_scoring_vs_wall_basis_points =
            duration_basis_points(row.batch_scoring_time, row.build_elapsed);
        writeln!(
            writer,
            "{},{},{:.9},{:.9},{},{},{},{},{},{},{}",
            row.point_count,
            row.dimension,
            row.build_elapsed.as_secs_f64(),
            row.batch_scoring_time.as_secs_f64(),
            accumulated_scoring_vs_wall_basis_points,
            row.batch_calls,
            row.scalar_calls,
            row.total_batch_candidates,
            row.min_batch,
            row.max_batch,
            row.median_batch,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
