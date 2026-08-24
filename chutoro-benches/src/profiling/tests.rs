//! Unit tests for profiling output.

use super::*;
use rstest::rstest;
use std::{fs, time::Duration};

#[derive(Debug)]
struct ScalingCase {
    peak_rss_bytes: u64,
    point_count: usize,
    max_connections: usize,
    edge_count: usize,
    expected_scaling: bool,
}

fn measurement(bytes: u64, millis: u64) -> PeakRssMeasurement {
    PeakRssMeasurement {
        elapsed: Duration::from_millis(millis),
        peak_rss_bytes: bytes,
    }
}

#[rstest]
#[case::balanced(ScalingCase {
    peak_rss_bytes: 3_200,
    point_count: 100,
    max_connections: 8,
    edge_count: 1_600,
    expected_scaling: true,
})]
#[case::too_sparse(ScalingCase {
    peak_rss_bytes: 3_200,
    point_count: 100,
    max_connections: 8,
    edge_count: 90,
    expected_scaling: false,
})]
#[case::too_dense(ScalingCase {
    peak_rss_bytes: 3_200,
    point_count: 100,
    max_connections: 8,
    edge_count: 7_000,
    expected_scaling: false,
})]
fn memory_record_reports_edge_scaling(#[case] case: ScalingCase) {
    let record = HnswMemoryRecord::new(
        HnswMemoryInput {
            point_count: case.point_count,
            max_connections: case.max_connections,
            ef_construction: case.max_connections.saturating_mul(2),
            measurement: measurement(case.peak_rss_bytes, 17),
            edge_count: case.edge_count,
        },
        EdgeScalingBounds::default(),
    )
    .expect("valid scaling case must build");
    assert_eq!(record.edge_scaling_ok, case.expected_scaling);
}

#[rstest]
#[case::zero_points(0, 10, "point_count")]
#[case::zero_edges(10, 0, "edge_count")]
fn memory_record_rejects_zero_denominators(
    #[case] point_count: usize,
    #[case] edge_count: usize,
    #[case] expected_context: &'static str,
) {
    let err = HnswMemoryRecord::new(
        HnswMemoryInput {
            point_count,
            max_connections: 8,
            ef_construction: 16,
            measurement: measurement(8_000, 10),
            edge_count,
        },
        EdgeScalingBounds::default(),
    )
    .expect_err("zero denominator must fail");
    assert!(matches!(
        err,
        ProfilingError::ZeroDenominator { context } if context == expected_context
    ));
}

#[rstest]
fn write_hnsw_memory_report_persists_header_and_rows() {
    let temp_path = std::env::temp_dir().join("hnsw_memory_profile_report_test.csv");
    let records = vec![
        HnswMemoryRecord::new(
            HnswMemoryInput {
                point_count: 100,
                max_connections: 8,
                ef_construction: 16,
                measurement: measurement(10_000, 9),
                edge_count: 800,
            },
            EdgeScalingBounds::default(),
        )
        .expect("record must build"),
    ];
    let written_path =
        write_hnsw_memory_report(&temp_path, &records).expect("report write must succeed");
    let contents = fs::read_to_string(&written_path).expect("report must be readable");
    assert!(contents.starts_with("point_count,max_connections"));
    assert!(contents.contains('\n'));
    fs::remove_file(written_path).expect("temp report cleanup must succeed");
}
