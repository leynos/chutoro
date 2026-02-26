//! Clustering-quality metric helpers shared across crates and tests.
//!
//! This module provides Adjusted Rand Index (ARI) and Normalized Mutual
//! Information (NMI) scoring for partition comparisons.

use std::collections::HashMap;

/// ARI and NMI values computed from two labellings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusteringQualityScore {
    /// Adjusted Rand Index in `[-1.0, 1.0]`.
    pub ari: f64,
    /// Normalized Mutual Information in `[0.0, 1.0]`.
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
    /// Internal contingency state violated expected invariants.
    #[error("internal clustering-quality invariant violated while {context}")]
    InvariantViolation {
        /// Human-readable context describing which lookup failed.
        context: &'static str,
    },
}

type ClusterCounts = HashMap<usize, usize>;
type PairCounts = HashMap<(usize, usize), usize>;
type ContingencyTableBuild = (ClusterCounts, ClusterCounts, PairCounts);

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
    clippy::cast_precision_loss,
    clippy::float_arithmetic,
    reason = "ARI/NMI combinatorics require floating-point arithmetic."
)]
fn comb2(value: usize) -> f64 {
    let as_float = value as f64;
    as_float * (as_float - 1.0) / 2.0
}

#[expect(
    clippy::float_arithmetic,
    reason = "ARI definition requires floating-point arithmetic."
)]
fn adjusted_rand_index_from_contingency(
    item_count: usize,
    left_counts: &ClusterCounts,
    right_counts: &ClusterCounts,
    contingency: &PairCounts,
) -> f64 {
    if item_count < 2 {
        return 1.0;
    }

    let sum_pair_counts: f64 = contingency.values().copied().map(comb2).sum();
    let sum_left_clusters: f64 = left_counts.values().copied().map(comb2).sum();
    let sum_right_clusters: f64 = right_counts.values().copied().map(comb2).sum();
    let total = comb2(item_count);
    if total == 0.0 {
        return 1.0;
    }

    let expected = (sum_left_clusters * sum_right_clusters) / total;
    let max_index = 0.5 * (sum_left_clusters + sum_right_clusters);
    let denominator = max_index - expected;
    if denominator == 0.0 {
        1.0
    } else {
        (sum_pair_counts - expected) / denominator
    }
}

#[expect(
    clippy::cast_precision_loss,
    clippy::float_arithmetic,
    reason = "NMI definition requires floating-point arithmetic."
)]
fn mutual_information_from_contingency(
    item_count: usize,
    left_counts: &ClusterCounts,
    right_counts: &ClusterCounts,
    contingency: &PairCounts,
) -> Result<f64, ClusteringQualityError> {
    let item_count_f64 = item_count as f64;
    let mut mutual_information = 0.0_f64;
    for (&(left_label, right_label), &count) in contingency {
        let left_count = left_counts.get(&left_label).copied().ok_or(
            ClusteringQualityError::InvariantViolation {
                context: "reading left marginal count",
            },
        )? as f64;
        let right_count = right_counts.get(&right_label).copied().ok_or(
            ClusteringQualityError::InvariantViolation {
                context: "reading right marginal count",
            },
        )? as f64;
        let count_f64 = count as f64;
        mutual_information += (count_f64 / item_count_f64)
            * ((count_f64 * item_count_f64) / (left_count * right_count)).ln();
    }
    Ok(mutual_information)
}

#[expect(
    clippy::cast_precision_loss,
    clippy::float_arithmetic,
    reason = "entropy computation requires floating-point arithmetic."
)]
fn entropy(counts: &ClusterCounts, item_count: usize) -> f64 {
    let item_count_f64 = item_count as f64;
    let mut entropy = 0.0_f64;
    for &count in counts.values() {
        let probability = (count as f64) / item_count_f64;
        entropy -= probability * probability.ln();
    }
    entropy
}

#[derive(Clone, Copy)]
enum NmiEntropyRegime {
    Both,
    One,
    Neither,
}

fn classify_nmi_entropy_regime(left_entropy: f64, right_entropy: f64) -> NmiEntropyRegime {
    if left_entropy == 0.0 && right_entropy == 0.0 {
        NmiEntropyRegime::Both
    } else if left_entropy == 0.0 || right_entropy == 0.0 {
        NmiEntropyRegime::One
    } else {
        NmiEntropyRegime::Neither
    }
}

#[expect(
    clippy::float_arithmetic,
    reason = "NMI definition requires floating-point arithmetic."
)]
fn normalized_mutual_information_from_contingency(
    item_count: usize,
    left_counts: &ClusterCounts,
    right_counts: &ClusterCounts,
    contingency: &PairCounts,
) -> Result<f64, ClusteringQualityError> {
    if item_count == 0 {
        return Ok(1.0);
    }

    let mutual_information =
        mutual_information_from_contingency(item_count, left_counts, right_counts, contingency)?;
    let left_entropy = entropy(left_counts, item_count);
    let right_entropy = entropy(right_counts, item_count);

    Ok(
        match classify_nmi_entropy_regime(left_entropy, right_entropy) {
            NmiEntropyRegime::Both => 1.0,
            NmiEntropyRegime::One => 0.0,
            NmiEntropyRegime::Neither => mutual_information / (left_entropy * right_entropy).sqrt(),
        },
    )
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
    let item_count = validate_label_lengths(ground_truth, predicted)?;
    let (left_counts, right_counts, contingency) = build_contingency_table(ground_truth, predicted);
    Ok(adjusted_rand_index_from_contingency(
        item_count,
        &left_counts,
        &right_counts,
        &contingency,
    ))
}

/// Computes Normalized Mutual Information (NMI) for two cluster labellings.
///
/// # Errors
///
/// Returns [`ClusteringQualityError::LabelLengthMismatch`] when label vectors
/// have different lengths.
pub fn normalized_mutual_information(
    ground_truth: &[usize],
    predicted: &[usize],
) -> Result<f64, ClusteringQualityError> {
    let item_count = validate_label_lengths(ground_truth, predicted)?;
    let (left_counts, right_counts, contingency) = build_contingency_table(ground_truth, predicted);
    normalized_mutual_information_from_contingency(
        item_count,
        &left_counts,
        &right_counts,
        &contingency,
    )
}

/// Computes ARI and NMI in one pass over contingency statistics.
///
/// # Errors
///
/// Returns [`ClusteringQualityError::LabelLengthMismatch`] when label vectors
/// have different lengths.
pub fn clustering_quality_score(
    ground_truth: &[usize],
    predicted: &[usize],
) -> Result<ClusteringQualityScore, ClusteringQualityError> {
    let item_count = validate_label_lengths(ground_truth, predicted)?;
    let (left_counts, right_counts, contingency) = build_contingency_table(ground_truth, predicted);
    let ari =
        adjusted_rand_index_from_contingency(item_count, &left_counts, &right_counts, &contingency);
    let nmi = normalized_mutual_information_from_contingency(
        item_count,
        &left_counts,
        &right_counts,
        &contingency,
    )?;
    Ok(ClusteringQualityScore { ari, nmi })
}
