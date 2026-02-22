//! Recall measurement helpers for HNSW benchmark quality reporting.
//!
//! Provides a brute-force nearest-neighbour oracle, a set-intersection
//! recall scorer, and a CSV report writer so benchmarks can quantify
//! build-time versus recall trade-offs across `ef_construction` values.

use std::{
    collections::{BinaryHeap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use chutoro_core::{DataSource, DataSourceError, Neighbour};

/// Integer-only recall score, avoiding `float_arithmetic` lint concerns.
///
/// Convert to a fraction only at display/report boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecallScore {
    /// Number of true neighbours found by the approximate search.
    pub hits: usize,
    /// Target count (min of k and oracle length).
    pub total: usize,
}

/// Computes the exact top-k nearest neighbours by exhaustive scan.
///
/// Returns up to `k` neighbours sorted by ascending distance. When
/// `k == 0` or the source is empty, returns an empty vector.
///
/// **Note:** the query point itself is included in the result (distance 0).
/// Callers that need to exclude self-neighbours for recall scoring should
/// filter the returned list (e.g. `.retain(|n| n.id != query)`).
///
/// # Errors
///
/// Returns [`DataSourceError`] if any distance computation fails.
pub fn brute_force_top_k<D: DataSource>(
    source: &D,
    query: usize,
    k: usize,
) -> Result<Vec<Neighbour>, DataSourceError> {
    if k == 0 {
        return Ok(Vec::new());
    }

    let mut heap: BinaryHeap<Neighbour> = BinaryHeap::with_capacity(k);
    for candidate in 0..source.len() {
        let distance = source.distance(query, candidate)?;
        heap.push(Neighbour {
            id: candidate,
            distance,
        });
        if heap.len() > k {
            heap.pop();
        }
    }

    let mut neighbours = heap.into_vec();
    neighbours.sort_unstable();
    Ok(neighbours)
}

/// Computes recall@k as integer hits over a target count.
///
/// Measures the overlap between oracle (ground-truth) and observed
/// (approximate) neighbour lists, each truncated to
/// `min(k, oracle.len(), observed.len())`.
///
/// # Examples
///
/// ```
/// use chutoro_benches::recall::{recall_at_k, RecallScore};
/// use chutoro_core::Neighbour;
///
/// let oracle = vec![
///     Neighbour { id: 0, distance: 0.1 },
///     Neighbour { id: 1, distance: 0.2 },
/// ];
/// let observed = vec![
///     Neighbour { id: 0, distance: 0.1 },
///     Neighbour { id: 2, distance: 0.3 },
/// ];
/// let score = recall_at_k(&oracle, &observed, 2);
/// assert_eq!(score, RecallScore { hits: 1, total: 2 });
/// ```
#[must_use]
pub fn recall_at_k(oracle: &[Neighbour], observed: &[Neighbour], k: usize) -> RecallScore {
    if k == 0 || oracle.is_empty() {
        return RecallScore { hits: 0, total: 0 };
    }
    let target = k.min(oracle.len()).min(observed.len());
    if target == 0 {
        return RecallScore { hits: 0, total: 0 };
    }
    let oracle_ids: HashSet<usize> = oracle.iter().take(target).map(|n| n.id).collect();
    let hits = observed
        .iter()
        .take(target)
        .filter(|neighbour| oracle_ids.contains(&neighbour.id))
        .count();
    RecallScore {
        hits,
        total: target,
    }
}

/// A single row in the recall-versus-ef_construction report.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecallMeasurement {
    /// Number of points in the dataset.
    pub point_count: usize,
    /// HNSW maximum connections per node (M).
    pub max_connections: usize,
    /// HNSW search width during construction.
    pub ef_construction: usize,
    /// Aggregated recall score across all queries.
    pub recall: RecallScore,
    /// Wall-clock build time in milliseconds.
    pub build_time_millis: u128,
}

impl RecallMeasurement {
    const fn csv_header() -> &'static str {
        "point_count,max_connections,ef_construction,recall_hits,recall_total,recall_fraction,build_time_ms\n"
    }

    fn to_csv_row(&self) -> String {
        let fraction = recall_fraction(self.recall);
        format!(
            "{},{},{},{},{},{},{}\n",
            self.point_count,
            self.max_connections,
            self.ef_construction,
            self.recall.hits,
            self.recall.total,
            fraction,
            self.build_time_millis,
        )
    }
}

/// Formats a recall score as a decimal fraction string.
///
/// Returns `"0.000000"` when the total is zero to avoid division by zero.
#[expect(
    clippy::float_arithmetic,
    clippy::cast_precision_loss,
    reason = "Recall fraction is inherently a float ratio; only used for human-readable CSV output."
)]
fn recall_fraction(score: RecallScore) -> String {
    if score.total == 0 {
        return "0.000000".to_owned();
    }
    format!("{:.6}", score.hits as f64 / score.total as f64)
}

/// Writes recall measurement records to a CSV report file.
///
/// Creates parent directories if they do not exist. Returns the
/// canonical path of the written report file.
///
/// # Errors
///
/// Returns [`std::io::Error`] if directory creation or file writing fails.
pub fn write_recall_report(
    report_path: impl AsRef<Path>,
    records: &[RecallMeasurement],
) -> Result<PathBuf, std::io::Error> {
    let report_file_path = report_path.as_ref().to_path_buf();
    if let Some(parent) = report_file_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut output = String::from(RecallMeasurement::csv_header());
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

    fn neighbour(id: usize, distance: f32) -> Neighbour {
        Neighbour { id, distance }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "Test-only helper; indices are small enough that precision loss is irrelevant."
    )]
    fn neighbours_from_ids(ids: &[usize]) -> Vec<Neighbour> {
        ids.iter()
            .enumerate()
            .map(|(idx, &id)| neighbour(id, idx as f32))
            .collect()
    }

    // -- recall_at_k: happy paths --------------------------------------

    #[rstest]
    #[case::perfect_recall(
        vec![0, 1, 2], vec![0, 1, 2], 3,
        RecallScore { hits: 3, total: 3 }
    )]
    #[case::partial_recall(
        vec![0, 1, 2], vec![0, 2, 5], 3,
        RecallScore { hits: 2, total: 3 }
    )]
    #[case::zero_recall(
        vec![0, 1, 2], vec![3, 4, 5], 3,
        RecallScore { hits: 0, total: 3 }
    )]
    fn recall_at_k_computes_expected_score(
        #[case] oracle_ids: Vec<usize>,
        #[case] observed_ids: Vec<usize>,
        #[case] k: usize,
        #[case] expected: RecallScore,
    ) {
        let oracle = neighbours_from_ids(&oracle_ids);
        let observed = neighbours_from_ids(&observed_ids);
        assert_eq!(recall_at_k(&oracle, &observed, k), expected);
    }

    // -- recall_at_k: edge cases ---------------------------------------

    #[rstest]
    #[case::k_zero(vec![0, 1], vec![0, 1], 0, RecallScore { hits: 0, total: 0 })]
    #[case::empty_oracle(vec![], vec![0, 1], 2, RecallScore { hits: 0, total: 0 })]
    #[case::k_exceeds_oracle(vec![0], vec![0, 1], 5, RecallScore { hits: 1, total: 1 })]
    #[case::observed_shorter(vec![0, 1, 2], vec![0], 3, RecallScore { hits: 1, total: 1 })]
    #[case::empty_observed(vec![0, 1], vec![], 2, RecallScore { hits: 0, total: 0 })]
    fn recall_at_k_handles_edge_cases(
        #[case] oracle_ids: Vec<usize>,
        #[case] observed_ids: Vec<usize>,
        #[case] k: usize,
        #[case] expected: RecallScore,
    ) {
        let oracle = neighbours_from_ids(&oracle_ids);
        let observed = neighbours_from_ids(&observed_ids);
        assert_eq!(recall_at_k(&oracle, &observed, k), expected);
    }

    // -- brute_force_top_k ---------------------------------------------

    /// Minimal distance-matrix data source for testing.
    #[derive(Clone)]
    struct MatrixSource {
        distances: Vec<Vec<f32>>,
    }

    impl DataSource for MatrixSource {
        fn len(&self) -> usize {
            self.distances.len()
        }
        fn name(&self) -> &'static str {
            "matrix"
        }

        fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
            let row = self
                .distances
                .get(i)
                .ok_or(DataSourceError::OutOfBounds { index: i })?;
            row.get(j)
                .copied()
                .ok_or(DataSourceError::OutOfBounds { index: j })
        }
    }

    #[rstest]
    fn brute_force_top_k_returns_empty_when_k_is_zero() {
        let source = MatrixSource {
            distances: vec![vec![0.0, 0.5], vec![0.5, 0.0]],
        };
        let result = brute_force_top_k(&source, 0, 0).expect("k=0 must succeed");
        assert!(result.is_empty());
    }

    #[rstest]
    fn brute_force_top_k_returns_sorted_nearest() {
        let source = MatrixSource {
            distances: vec![
                vec![0.0, 0.3, 0.9, 0.1],
                vec![0.3, 0.0, 0.6, 0.4],
                vec![0.9, 0.6, 0.0, 0.8],
                vec![0.1, 0.4, 0.8, 0.0],
            ],
        };
        let result = brute_force_top_k(&source, 0, 2).expect("top-2 must succeed");
        assert_eq!(result.len(), 2);
        // Nearest to node 0: node 0 (0.0) and node 3 (0.1)
        assert_eq!(result.first().map(|n| n.id), Some(0));
        assert_eq!(result.get(1).map(|n| n.id), Some(3));
    }

    #[rstest]
    fn brute_force_top_k_returns_all_when_k_exceeds_len() {
        let source = MatrixSource {
            distances: vec![vec![0.0, 0.4], vec![0.4, 0.0]],
        };
        let result = brute_force_top_k(&source, 0, 10).expect("k>len must return all");
        assert_eq!(result.len(), 2);
    }

    #[rstest]
    fn brute_force_top_k_handles_empty_source() {
        let source = MatrixSource {
            distances: Vec::new(),
        };
        let result = brute_force_top_k(&source, 0, 1).expect("empty source must succeed");
        assert!(result.is_empty());
    }

    // -- write_recall_report -------------------------------------------

    #[rstest]
    fn write_recall_report_persists_header_and_rows() {
        let temp_path = std::env::temp_dir().join("recall_vs_ef_test.csv");
        let records = vec![RecallMeasurement {
            point_count: 1_000,
            max_connections: 8,
            ef_construction: 100,
            recall: RecallScore { hits: 9, total: 10 },
            build_time_millis: 42,
        }];
        let written_path =
            write_recall_report(&temp_path, &records).expect("report write must succeed");
        let contents = fs::read_to_string(&written_path).expect("report must be readable");
        assert!(contents.starts_with("point_count,max_connections"));
        // Header + 1 data row
        assert_eq!(contents.lines().count(), 2);
        assert!(contents.contains("1000,8,100,9,10,0.900000,42"));
        fs::remove_file(written_path).expect("temp report cleanup must succeed");
    }

    // -- recall_fraction -----------------------------------------------

    #[rstest]
    #[case::zero_total(RecallScore { hits: 0, total: 0 }, "0.000000")]
    #[case::full(RecallScore { hits: 10, total: 10 }, "1.000000")]
    #[case::half(RecallScore { hits: 5, total: 10 }, "0.500000")]
    fn recall_fraction_formats_correctly(#[case] score: RecallScore, #[case] expected: &str) {
        assert_eq!(recall_fraction(score), expected);
    }
}
