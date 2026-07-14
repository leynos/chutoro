//! CSV report rendering for neighbour-scoring diagnostics.

use std::{
    borrow::Cow,
    io::{self, Write},
    time::Duration,
};

use super::{duration_basis_points, lane_utilisation_basis_points, padded_lane_count};

/// One row in the lane-utilisation diagnostic CSV report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaneUtilisationReportRow<'a> {
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

fn csv_requires_quotes(field: &str) -> bool {
    field
        .chars()
        .any(|ch| matches!(ch, ',' | '"' | '\n' | '\r'))
}

fn csv_escape(field: &str) -> Cow<'_, str> {
    if csv_requires_quotes(field) {
        let escaped = field.replace('"', "\"\"");
        Cow::Owned(format!("\"{escaped}\""))
    } else {
        Cow::Borrowed(field)
    }
}

/// Write the lane-utilisation report schema and rows.
///
/// # Examples
///
/// ```
/// use chutoro_benches::neighbour_scoring::{
///     LaneUtilisationReportRow, write_lane_utilisation_report_csv,
/// };
///
/// let mut output = Vec::new();
/// let result = write_lane_utilisation_report_csv(
///     &mut output,
///     [LaneUtilisationReportRow {
///         bucket_kind: "realistic",
///         candidate_count: 8,
///     }],
/// );
/// assert!(result.is_ok());
/// ```
///
/// # Errors
///
/// Returns any error reported by the supplied writer.
pub fn write_lane_utilisation_report_csv<'a, W, I>(mut writer: W, rows: I) -> io::Result<()>
where
    W: Write,
    I: IntoIterator<Item = LaneUtilisationReportRow<'a>>,
{
    writeln!(
        writer,
        "bucket_kind,candidate_count,padded_lanes,wasted_lanes,lane_utilisation_basis_points"
    )?;
    for row in rows {
        let padded = padded_lane_count(row.candidate_count);
        let wasted = padded.saturating_sub(row.candidate_count as u128);
        let utilisation = lane_utilisation_basis_points(row.candidate_count);
        let bucket_kind = csv_escape(row.bucket_kind);
        writeln!(
            writer,
            "{bucket_kind},{},{padded},{wasted},{utilisation}",
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
mod tests {
    //! Tests for neighbour-scoring CSV report rendering.
    use super::{
        BuildProfileReportRow, LaneUtilisationReportRow, duration_basis_points,
        lane_utilisation_basis_points, padded_lane_count, write_build_profile_report_csv,
        write_lane_utilisation_report_csv,
    };
    use proptest::prelude::*;
    use rstest::rstest;
    use std::time::Duration;

    #[rstest]
    #[case::multiple_rows(
        vec![
            LaneUtilisationReportRow {
                bucket_kind: "realistic",
                candidate_count: 8,
            },
            LaneUtilisationReportRow {
                bucket_kind: "diagnostic",
                candidate_count: 24,
            },
        ],
        concat!(
            "bucket_kind,candidate_count,padded_lanes,wasted_lanes,",
            "lane_utilisation_basis_points\n",
            "realistic,8,16,8,5000\n",
            "diagnostic,24,32,8,7500\n",
        )
    )]
    #[case::escaped_bucket_kind(
        vec![
            LaneUtilisationReportRow {
                bucket_kind: "diagnostic,\n\"quoted\"\r",
                candidate_count: 16,
            },
        ],
        concat!(
            "bucket_kind,candidate_count,padded_lanes,wasted_lanes,",
            "lane_utilisation_basis_points\n",
            "\"diagnostic,\n\"\"quoted\"\"\r\",16,16,0,10000\n",
        )
    )]
    fn write_lane_utilisation_report_csv_renders_schema_and_rows(
        #[case] rows: Vec<LaneUtilisationReportRow<'static>>,
        #[case] expected: &str,
    ) {
        let mut csv_buffer = Vec::new();
        let result = write_lane_utilisation_report_csv(&mut csv_buffer, rows);

        result.expect("render lane-utilisation CSV report");
        assert_eq!(String::from_utf8_lossy(&csv_buffer), expected);
    }

    #[rstest]
    #[case::single_row(
        vec![
            BuildProfileReportRow {
                point_count: 10_000,
                dimension: 128,
                build_elapsed: Duration::from_millis(4),
                batch_scoring_time: Duration::from_millis(1),
                batch_calls: 2,
                scalar_calls: 3,
                total_batch_candidates: 48,
                min_batch: 8,
                max_batch: 32,
                median_batch: 16,
            },
        ],
        concat!(
            "point_count,dimension,build_seconds,",
            "accumulated_batch_scoring_seconds,",
            "accumulated_batch_scoring_vs_wall_basis_points,batch_calls,",
            "scalar_calls,total_batch_candidates,min_batch,max_batch,median_batch\n",
            "10000,128,0.004000000,0.001000000,2500,2,3,48,8,32,16\n",
        )
    )]
    #[case::header_only(
        Vec::new(),
        concat!(
            "point_count,dimension,build_seconds,",
            "accumulated_batch_scoring_seconds,",
            "accumulated_batch_scoring_vs_wall_basis_points,batch_calls,",
            "scalar_calls,total_batch_candidates,min_batch,max_batch,median_batch\n",
        )
    )]
    fn write_build_profile_report_csv_renders_schema_and_rows(
        #[case] rows: Vec<BuildProfileReportRow>,
        #[case] expected: &str,
    ) {
        let mut csv_buffer = Vec::new();
        let result = write_build_profile_report_csv(&mut csv_buffer, rows);

        result.expect("render build-profile CSV report");
        assert_eq!(String::from_utf8_lossy(&csv_buffer), expected);
    }
    fn csv_label() -> impl Strategy<Value = String> {
        prop::collection::vec(
            prop_oneof![
                Just(','),
                Just('"'),
                Just('\n'),
                Just('\r'),
                Just(' '),
                (b'a'..=b'z').prop_map(char::from),
                (b'0'..=b'9').prop_map(char::from),
            ],
            0..32,
        )
        .prop_map(|chars| chars.into_iter().collect())
    }

    // Test-only parser for the newline-terminated CSV emitted by these writers.
    // It relies on '\n', treats bare '\r' as data, and drops unterminated rows.
    fn parse_csv_records(csv: &str) -> Vec<Vec<String>> {
        let mut records = Vec::new();
        let mut record = Vec::new();
        let mut field = String::new();
        let mut chars = csv.chars().peekable();
        let mut in_quotes = false;

        while let Some(ch) = chars.next() {
            match ch {
                '"' if in_quotes && chars.peek() == Some(&'"') => {
                    field.push('"');
                    let _ = chars.next();
                }
                '"' => in_quotes = !in_quotes,
                ',' if !in_quotes => {
                    record.push(std::mem::take(&mut field));
                }
                '\n' if !in_quotes => {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                }
                _ => field.push(ch),
            }
        }

        records
    }
    proptest! {
        #[test]
        fn lane_utilisation_report_round_trips_generated_rows(
            bucket_kind in csv_label(),
            candidate_count in any::<usize>(),
        ) {
            let row = LaneUtilisationReportRow {
                bucket_kind: &bucket_kind,
                candidate_count,
            };
            let mut csv_buffer = Vec::new();
            write_lane_utilisation_report_csv(&mut csv_buffer, [row])
                .expect("render generated lane-utilisation CSV row");
            let csv = String::from_utf8(csv_buffer)
                .expect("lane-utilisation CSV is UTF-8");
            let records = parse_csv_records(&csv);
            prop_assert_eq!(records.len(), 2);
            let data_record = records.get(1).expect("CSV data row exists");
            prop_assert_eq!(
                data_record.first().expect("CSV bucket-kind field exists"),
                &bucket_kind
            );
            prop_assert_eq!(
                data_record
                    .get(1)
                    .expect("CSV candidate-count field exists"),
                &candidate_count.to_string()
            );

            let padded_lanes: u128 = data_record
                .get(2)
                .expect("CSV padded-lanes field exists")
                .parse()
                .expect("parse padded lane count");
            let wasted_lanes: u128 = data_record
                .get(3)
                .expect("CSV wasted-lanes field exists")
                .parse()
                .expect("parse wasted lane count");
            let utilisation: usize = data_record
                .get(4)
                .expect("CSV lane-utilisation field exists")
                .parse()
                .expect("parse lane-utilisation basis points");
            let expected_padded = padded_lane_count(candidate_count);
            prop_assert!(padded_lanes >= candidate_count as u128);
            prop_assert_eq!(padded_lanes, expected_padded);
            prop_assert_eq!(wasted_lanes, padded_lanes - candidate_count as u128);
            prop_assert_eq!(utilisation, lane_utilisation_basis_points(candidate_count));
            prop_assert!(utilisation <= 10_000);
        }

        #[test]
        fn build_profile_report_renders_generated_duration_ratio(
            build_elapsed_nanos in prop_oneof![Just(0_u64), 1_u64..=1_000, 1_001_u64..=u64::MAX],
            batch_scoring_nanos in prop_oneof![Just(0_u64), 1_u64..=1_000, 1_001_u64..=u64::MAX],
        ) {
            let row = BuildProfileReportRow {
                point_count: 10_000,
                dimension: 128,
                build_elapsed: Duration::from_nanos(build_elapsed_nanos),
                batch_scoring_time: Duration::from_nanos(batch_scoring_nanos),
                batch_calls: 2,
                scalar_calls: 3,
                total_batch_candidates: 48,
                min_batch: 8,
                max_batch: 32,
                median_batch: 16,
            };
            let mut csv_buffer = Vec::new();
            write_build_profile_report_csv(&mut csv_buffer, [row])
                .expect("render generated build-profile CSV row");
            let csv = String::from_utf8(csv_buffer).expect("build-profile CSV is UTF-8");
            let records = parse_csv_records(&csv);
            let data_record = records.get(1).expect("CSV data row exists");
            let ratio: usize = data_record
                .get(4)
                .expect("CSV duration-ratio field exists")
                .parse()
                .expect("parse duration ratio");
            prop_assert_eq!(ratio, duration_basis_points(row.batch_scoring_time, row.build_elapsed));
        }
    }
}
