//! Functional clustering tests comparing exact and HNSW-based pipelines.
//!
//! These tests verify that the approximate CPU pipeline (HNSW candidate edge
//! harvest + Kruskal MST + hierarchy extraction) produces a clustering that is
//! close to an exact baseline computed from the full mutual-reachability graph
//! on small public datasets.

use std::num::NonZeroUsize;

use rstest::rstest;

use chutoro_core::{
    CandidateEdge, DataSource, DataSourceError, EdgeHarvest, HierarchyConfig, MetricDescriptor,
    adjusted_rand_index, extract_labels_from_mst, normalized_mutual_information, parallel_kruskal,
    run_cpu_pipeline,
};

fn parse_csv_rows(input: &str, dims: usize) -> Vec<Vec<f32>> {
    input
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut parts = line.split(',');
            let mut row = Vec::with_capacity(dims);
            for _ in 0..dims {
                let value = parts
                    .next()
                    .unwrap_or_else(|| panic!("missing column in line: {line}"))
                    .parse::<f32>()
                    .unwrap_or_else(|err| panic!("failed to parse float in line '{line}': {err}"));
                row.push(value);
            }
            row
        })
        .collect()
}

#[derive(Clone, Debug)]
struct DenseVectors {
    metric: MetricDescriptor,
    rows: Vec<Vec<f32>>,
}

impl DenseVectors {
    fn new(metric: &'static str, rows: Vec<Vec<f32>>) -> Self {
        Self {
            metric: MetricDescriptor::new(metric),
            rows,
        }
    }

    fn dim(&self) -> usize {
        self.rows.first().map(|row| row.len()).unwrap_or_default()
    }
}

impl DataSource for DenseVectors {
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn name(&self) -> &str {
        "dense-vectors"
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        self.metric.clone()
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let left = self
            .rows
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let right = self
            .rows
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        if left.len() != right.len() {
            return Err(DataSourceError::DimensionMismatch {
                left: left.len(),
                right: right.len(),
            });
        }
        if left.is_empty() {
            return Err(DataSourceError::ZeroDimension);
        }
        let mut sum = 0.0_f32;
        for (&a, &b) in left.iter().zip(right.iter()) {
            let diff = a - b;
            sum += diff * diff;
        }
        Ok(sum.sqrt())
    }
}

fn core_distances_exact<D: DataSource>(source: &D, min_cluster_size: usize) -> Vec<f32> {
    let n = source.len();
    let mut core = vec![0.0_f32; n];
    for (i, core_value) in core.iter_mut().enumerate() {
        let mut distances = Vec::with_capacity(n.saturating_sub(1));
        for j in 0..n {
            if i == j {
                continue;
            }
            distances.push(source.distance(i, j).expect("distance must succeed"));
        }
        distances.sort_by(|a, b| a.total_cmp(b));
        *core_value = distances
            // Select the k-th nearest neighbour distance as the core distance
            // (0-indexed, so `k-1`), matching HDBSCAN's definition.
            .get(min_cluster_size.saturating_sub(1))
            .copied()
            .or_else(|| distances.last().copied())
            .unwrap_or(0.0);
    }
    core
}

fn complete_mutual_reachability_edges<D: DataSource>(source: &D, core: &[f32]) -> EdgeHarvest {
    let n = source.len();
    let mut edges = Vec::new();
    let mut seq = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = source.distance(i, j).expect("distance must succeed");
            let weight = dist.max(core[i]).max(core[j]);
            edges.push(CandidateEdge::new(i, j, weight, seq));
            seq += 1;
        }
    }
    EdgeHarvest::new(edges)
}

fn exact_pipeline<D: DataSource>(source: &D, min_cluster_size: NonZeroUsize) -> Vec<usize> {
    let core = core_distances_exact(source, min_cluster_size.get());
    let edges = complete_mutual_reachability_edges(source, &core);
    let forest = parallel_kruskal(source.len(), &edges).expect("exact MST should succeed");
    extract_labels_from_mst(
        source.len(),
        forest.edges(),
        HierarchyConfig::new(min_cluster_size),
    )
    .expect("exact hierarchy extraction should succeed")
}

fn approx_pipeline<D: DataSource + Sync>(source: &D, min_cluster_size: NonZeroUsize) -> Vec<usize> {
    let result =
        run_cpu_pipeline(source, min_cluster_size).expect("approx pipeline should succeed");
    result
        .assignments()
        .iter()
        .map(|id| usize::try_from(id.get()).expect("cluster identifiers fit usize"))
        .collect()
}

#[test]
fn nmi_is_one_when_both_partitions_have_single_cluster() {
    let labels = vec![0, 0, 0, 0];
    assert_eq!(
        normalized_mutual_information(&labels, &labels).expect("NMI should compute"),
        1.0
    );
}

#[test]
fn metrics_identity_and_permutation_are_one() {
    let labels = vec![0, 0, 1, 1, 2, 2];
    assert_eq!(
        adjusted_rand_index(&labels, &labels).expect("ARI should compute"),
        1.0
    );
    assert!(
        (normalized_mutual_information(&labels, &labels).expect("NMI should compute") - 1.0).abs()
            < 1e-12
    );

    let permuted = vec![1, 1, 2, 2, 0, 0];
    assert_eq!(
        adjusted_rand_index(&labels, &permuted).expect("ARI should compute"),
        1.0
    );
    assert!(
        (normalized_mutual_information(&labels, &permuted).expect("NMI should compute") - 1.0)
            .abs()
            < 1e-12
    );
}

#[test]
fn metrics_are_finite_for_non_trivial_partitions() {
    let left = vec![0, 0, 0, 1, 1, 2];
    let right = vec![0, 1, 0, 1, 2, 2];
    let ari = adjusted_rand_index(&left, &right).expect("ARI should compute");
    let nmi = normalized_mutual_information(&left, &right).expect("NMI should compute");

    assert!(ari.is_finite());
    assert!(ari <= 1.0);
    assert!(ari >= -1.0);

    assert!(nmi.is_finite());
    assert!(nmi <= 1.0);
    assert!(nmi >= 0.0);
}

#[derive(Clone, Copy, Debug)]
struct Dataset {
    name: &'static str,
    dims: usize,
    data: &'static str,
}

fn iris_dataset() -> Dataset {
    Dataset {
        name: "iris",
        dims: 4,
        data: include_str!("data/iris.csv"),
    }
}

fn ruspini_dataset() -> Dataset {
    Dataset {
        name: "ruspini",
        dims: 2,
        data: include_str!("data/ruspini.csv"),
    }
}

/// Verifies approximate HNSW pipeline clustering quality against exact baseline.
///
/// The iris dataset uses relaxed thresholds (0.65) compared to ruspini (0.95)
/// because iris has overlapping class boundaries and higher inherent variance.
/// The HNSW approximation introduces additional variance through:
/// - Non-deterministic graph construction (level assignment, edge selection)
/// - Approximate nearest-neighbour search affecting core distance estimates
///
/// Ruspini's well-separated clusters tolerate little approximation error, while
/// iris's fuzzy boundaries mean even small edge-set differences can shift cluster
/// assignments, leading to lower but acceptable ARI/NMI scores.
#[rstest]
#[case(iris_dataset(), 5, 0.65, 0.65)]
#[case(ruspini_dataset(), 4, 0.95, 0.95)]
fn hnsw_pipeline_matches_exact_baseline(
    #[case] dataset: Dataset,
    #[case] min_cluster_size: usize,
    #[case] min_ari: f64,
    #[case] min_nmi: f64,
) {
    let rows = parse_csv_rows(dataset.data, dataset.dims);
    let source = DenseVectors::new("euclidean", rows);
    assert_eq!(source.dim(), dataset.dims);

    let min_cluster_size =
        NonZeroUsize::new(min_cluster_size).expect("min_cluster_size must be non-zero");

    let exact = exact_pipeline(&source, min_cluster_size);
    let approx = approx_pipeline(&source, min_cluster_size);

    let ari = adjusted_rand_index(&exact, &approx).expect("ARI should compute");
    let nmi = normalized_mutual_information(&exact, &approx).expect("NMI should compute");

    assert!(
        ari >= min_ari,
        "dataset={} ARI {} < {} (clusters exact={}, approx={})",
        dataset.name,
        ari,
        min_ari,
        exact.iter().copied().max().unwrap_or(0) + 1,
        approx.iter().copied().max().unwrap_or(0) + 1
    );
    assert!(
        nmi >= min_nmi,
        "dataset={} NMI {} < {} (clusters exact={}, approx={})",
        dataset.name,
        nmi,
        min_nmi,
        exact.iter().copied().max().unwrap_or(0) + 1,
        approx.iter().copied().max().unwrap_or(0) + 1
    );
}
