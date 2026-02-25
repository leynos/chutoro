//! Clustering-quality metric helpers for benchmark reporting.
//!
//! This module provides Adjusted Rand Index (ARI) and Normalized Mutual
//! Information (NMI) scoring against ground-truth labels, together with a CSV
//! report writer used by benchmark harnesses.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

/// Clustering-quality metrics computed for a single benchmark configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusteringQualityScore {
    /// Adjusted Rand Index score in `[-1.0, 1.0]`.
    pub ari: f64,
    /// Normalized Mutual Information score in `[0.0, 1.0]`.
    pub nmi: f64,
}

/// Errors raised while computing clustering-quality metrics.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum ClusteringQualityError {
    /// Ground-truth and predicted labels had different lengths.
    #[error("label length mismatch: ground_truth={ground_truth_len}, predicted={predicted_len}")]
    LabelLengthMismatch {
        /// Number of ground-truth labels.
        ground_truth_len: usize,
        /// Number of predicted labels.
        predicted_len: usize,
    },
}

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

const fn validate_label_lengths(
    ground_truth: &[usize],
    predicted: &[usize],
) -> Result<usize, ClusteringQualityError> {
    if ground_truth.len() != predicted.len() {
        return Err(ClusteringQualityError::LabelLengthMismatch {
            ground_truth_len: ground_truth.len(),
            predicted_len: predicted.len(),
        });
    }
    Ok(ground_truth.len())
}

#[expect(
    clippy::cast_precision_loss,
    clippy::float_arithmetic,
    reason = "ARI pair-count formula is defined with floating-point combinatorics."
)]
fn comb2(value: usize) -> f64 {
    let as_float = value as f64;
    as_float * (as_float - 1.0) / 2.0
}

type AriPairCounts = HashMap<(usize, usize), usize>;
type AriContingencyBuild = (Vec<usize>, Vec<usize>, AriPairCounts);

fn build_ari_contingency(left: &[usize], right: &[usize]) -> AriContingencyBuild {
    let mut left_ids = HashMap::<usize, usize>::new();
    let mut right_ids = HashMap::<usize, usize>::new();
    let mut left_counts = Vec::<usize>::new();
    let mut right_counts = Vec::<usize>::new();
    let mut contingency = HashMap::<(usize, usize), usize>::new();

    for (&left_label, &right_label) in left.iter().zip(right) {
        let left_index = *left_ids.entry(left_label).or_insert_with(|| {
            let id = left_counts.len();
            left_counts.push(0);
            id
        });
        let right_index = *right_ids.entry(right_label).or_insert_with(|| {
            let id = right_counts.len();
            right_counts.push(0);
            id
        });

        let Some(left_count) = left_counts.get_mut(left_index) else {
            continue;
        };
        let Some(right_count) = right_counts.get_mut(right_index) else {
            continue;
        };

        *left_count += 1;
        *right_count += 1;
        *contingency.entry((left_index, right_index)).or_insert(0) += 1;
    }

    (left_counts, right_counts, contingency)
}

type ClusterCounts = HashMap<usize, usize>;
type PairCounts = HashMap<(usize, usize), usize>;
type ContingencyTableBuild = (ClusterCounts, ClusterCounts, PairCounts);

fn build_contingency_table(left: &[usize], right: &[usize]) -> ContingencyTableBuild {
    let mut left_counts = HashMap::<usize, usize>::new();
    let mut right_counts = HashMap::<usize, usize>::new();
    let mut contingency = HashMap::<(usize, usize), usize>::new();

    for (&left_label, &right_label) in left.iter().zip(right) {
        *left_counts.entry(left_label).or_insert(0) += 1;
        *right_counts.entry(right_label).or_insert(0) += 1;
        *contingency.entry((left_label, right_label)).or_insert(0) += 1;
    }

    (left_counts, right_counts, contingency)
}

#[expect(
    clippy::float_arithmetic,
    reason = "ARI definition requires floating-point arithmetic."
)]
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
    let item_count = validate_label_lengths(ground_truth, predicted)?;
    if item_count < 2 {
        return Ok(1.0);
    }

    let (left_counts, right_counts, contingency) = build_ari_contingency(ground_truth, predicted);
    let sum_pair_counts: f64 = contingency.values().copied().map(comb2).sum();
    let sum_left_clusters: f64 = left_counts.iter().copied().map(comb2).sum();
    let sum_right_clusters: f64 = right_counts.iter().copied().map(comb2).sum();
    let total = comb2(item_count);
    if total == 0.0 {
        return Ok(1.0);
    }

    let expected = (sum_left_clusters * sum_right_clusters) / total;
    let max_index = 0.5 * (sum_left_clusters + sum_right_clusters);
    let denominator = max_index - expected;
    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok((sum_pair_counts - expected) / denominator)
    }
}

#[expect(
    clippy::float_arithmetic,
    clippy::cast_precision_loss,
    reason = "NMI definition requires floating-point arithmetic."
)]
/// Computes Normalized Mutual Information (NMI) for two cluster labellings.
///
/// # Errors
///
/// Returns [`ClusteringQualityError::LabelLengthMismatch`] when label vectors
/// have different lengths.
pub fn normalised_mutual_information(
    ground_truth: &[usize],
    predicted: &[usize],
) -> Result<f64, ClusteringQualityError> {
    let item_count = validate_label_lengths(ground_truth, predicted)?;
    if item_count == 0 {
        return Ok(1.0);
    }

    let (left_counts, right_counts, contingency) = build_contingency_table(ground_truth, predicted);

    let item_count_f64 = item_count as f64;
    let mut mutual_information = 0.0_f64;
    for (&(left_label, right_label), &count) in &contingency {
        let Some(&left_count_usize) = left_counts.get(&left_label) else {
            continue;
        };
        let Some(&right_count_usize) = right_counts.get(&right_label) else {
            continue;
        };
        let count_f64 = count as f64;
        let left_count = left_count_usize as f64;
        let right_count = right_count_usize as f64;
        mutual_information += (count_f64 / item_count_f64)
            * ((count_f64 * item_count_f64) / (left_count * right_count)).ln();
    }

    let left_entropy = entropy(&left_counts, item_count_f64);
    let right_entropy = entropy(&right_counts, item_count_f64);
    if left_entropy == 0.0 && right_entropy == 0.0 {
        Ok(1.0)
    } else if left_entropy == 0.0 || right_entropy == 0.0 {
        Ok(0.0)
    } else {
        Ok(mutual_information / (left_entropy * right_entropy).sqrt())
    }
}

#[expect(
    clippy::float_arithmetic,
    clippy::cast_precision_loss,
    reason = "entropy computation requires floating-point arithmetic."
)]
fn entropy(counts: &ClusterCounts, item_count_f64: f64) -> f64 {
    let mut entropy = 0.0_f64;
    for &count in counts.values() {
        let probability = (count as f64) / item_count_f64;
        entropy -= probability * probability.ln();
    }
    entropy
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
    let ari = adjusted_rand_index(ground_truth, predicted)?;
    let nmi = normalised_mutual_information(ground_truth, predicted)?;
    Ok(ClusteringQualityScore { ari, nmi })
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
