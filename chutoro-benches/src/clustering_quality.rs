//! Clustering-quality metric helpers for benchmark reporting.
//!
//! This module provides Adjusted Rand Index (ARI) and Normalised Mutual
//! Information (NMI) scoring against ground-truth labels, together with a CSV
//! report writer used by benchmark harnesses.

use std::{
    fs,
    path::{Path, PathBuf},
};

pub use chutoro_core::{ClusteringQualityError, ClusteringQualityScore};

/// A single row in the ARI/NMI quality report.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusteringQualityMeasurement {
    /// Number of points in the synthetic dataset.
    pub point_count: usize,
    /// HNSW maximum connections per node (M).
    pub max_connections: usize,
    /// HNSW construction beam width (`ef_construction`).
    pub ef_construction: usize,
    /// Minimum cluster size used for hierarchy extraction.
    pub min_cluster_size: usize,
    /// ARI score against Gaussian ground truth.
    pub ari: f64,
    /// NMI score against Gaussian ground truth.
    pub nmi: f64,
    /// HNSW build time in milliseconds for this configuration.
    pub build_time_millis: u128,
}

impl ClusteringQualityMeasurement {
    const fn csv_header() -> &'static str {
        concat!(
            "point_count,max_connections,ef_construction,min_cluster_size,",
            "ari,nmi,build_time_ms\n"
        )
    }

    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{:.6},{:.6},{}\n",
            self.point_count,
            self.max_connections,
            self.ef_construction,
            self.min_cluster_size,
            self.ari,
            self.nmi,
            self.build_time_millis,
        )
    }
}

/// Computes Adjusted Rand Index (ARI) for two cluster labellings.
///
/// # Errors
///
/// Returns [`ClusteringQualityError::LabelLengthMismatch`] when label vectors
/// have different lengths.
pub fn adjusted_rand_index(
    ground_truth: &[usize],
    predicted: &[usize],
) -> Result<f64, ClusteringQualityError> {
    chutoro_core::adjusted_rand_index(ground_truth, predicted)
}

/// Computes Normalised Mutual Information (NMI) for two cluster labellings.
///
/// # Errors
///
/// Returns [`ClusteringQualityError::LabelLengthMismatch`] when label vectors
/// have different lengths.
pub fn normalized_mutual_information(
    ground_truth: &[usize],
    predicted: &[usize],
) -> Result<f64, ClusteringQualityError> {
    chutoro_core::normalized_mutual_information(ground_truth, predicted)
}

/// Computes ARI and NMI in one call.
///
/// # Errors
///
/// Returns [`ClusteringQualityError::LabelLengthMismatch`] when label vectors
/// have different lengths.
pub fn clustering_quality_score(
    ground_truth: &[usize],
    predicted: &[usize],
) -> Result<ClusteringQualityScore, ClusteringQualityError> {
    chutoro_core::clustering_quality_score(ground_truth, predicted)
}

/// Writes clustering-quality measurements to a CSV report file.
///
/// # Errors
///
/// Returns [`std::io::Error`] when directory creation or file writing fails.
pub fn write_clustering_quality_report(
    report_path: impl AsRef<Path>,
    records: &[ClusteringQualityMeasurement],
) -> Result<PathBuf, std::io::Error> {
    let report_file_path = report_path.as_ref().to_path_buf();
    if let Some(parent) = report_file_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut output = String::from(ClusteringQualityMeasurement::csv_header());
    for record in records {
        output.push_str(&record.to_csv_row());
    }
    fs::write(&report_file_path, output)?;
    Ok(report_file_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::identity(vec![0, 0, 1, 1], vec![0, 0, 1, 1], 1.0, 1.0)]
    #[case::permutation(vec![0, 0, 1, 1], vec![3, 3, 2, 2], 1.0, 1.0)]
    #[case::single_cluster(vec![0, 0, 0, 0], vec![7, 7, 7, 7], 1.0, 1.0)]
    #[case::all_singletons(vec![0, 1, 2, 3], vec![4, 5, 6, 7], 1.0, 1.0)]
    fn clustering_quality_score_matches_perfect_assignments(
        #[case] ground_truth: Vec<usize>,
        #[case] predicted: Vec<usize>,
        #[case] expected_ari: f64,
        #[case] expected_nmi: f64,
    ) {
        let score =
            clustering_quality_score(&ground_truth, &predicted).expect("scores should compute");

        assert_eq!(score.ari.to_bits(), expected_ari.to_bits());
        assert_eq!(score.nmi.to_bits(), expected_nmi.to_bits());
    }

    #[rstest]
    #[case::empty(vec![], vec![])]
    #[case::singleton(vec![0], vec![1])]
    fn clustering_quality_score_handles_degenerate_inputs(
        #[case] ground_truth: Vec<usize>,
        #[case] predicted: Vec<usize>,
    ) {
        let score =
            clustering_quality_score(&ground_truth, &predicted).expect("scores should compute");
        assert!(score.ari.is_finite());
        assert!(score.nmi.is_finite());
    }

    #[test]
    fn clustering_quality_score_rejects_length_mismatch() {
        let err = clustering_quality_score(&[0, 1], &[0])
            .expect_err("mismatched label vectors must fail");
        assert_eq!(
            err,
            ClusteringQualityError::LabelLengthMismatch {
                ground_truth_len: 2,
                predicted_len: 1,
            }
        );
    }

    #[test]
    fn metric_ranges_are_bounded() {
        let score = clustering_quality_score(&[0, 0, 1, 1, 2, 2], &[0, 1, 0, 1, 2, 2])
            .expect("scores should compute");
        assert!(score.ari.is_finite());
        assert!(score.ari >= -1.0);
        assert!(score.ari <= 1.0);
        assert!(score.nmi.is_finite());
        assert!(score.nmi >= 0.0);
        assert!(score.nmi <= 1.0);
    }

    #[test]
    fn write_clustering_quality_report_writes_header_and_rows() {
        let temp_dir = tempfile::tempdir().expect("tempdir should be created");
        let report_path = temp_dir.path().join("quality.csv");
        let rows = vec![ClusteringQualityMeasurement {
            point_count: 1000,
            max_connections: 16,
            ef_construction: 100,
            min_cluster_size: 5,
            ari: 0.812_345,
            nmi: 0.723_456,
            build_time_millis: 42,
        }];

        let written_path = write_clustering_quality_report(&report_path, &rows)
            .expect("quality report should be written");
        let written =
            std::fs::read_to_string(&written_path).expect("quality report should be readable");

        assert!(written.starts_with(
            "point_count,max_connections,ef_construction,min_cluster_size,ari,nmi,build_time_ms"
        ));
        assert!(written.contains("1000,16,100,5,0.812345,0.723456,42"));
    }
}
