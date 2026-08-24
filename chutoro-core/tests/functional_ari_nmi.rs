//! Functional clustering tests comparing exact and HNSW-based pipelines.
//!
//! These tests verify that the approximate CPU pipeline (HNSW candidate edge
//! harvest + Kruskal MST + hierarchy extraction) produces a clustering that is
//! close to an exact baseline computed from the full mutual-reachability graph
//! on small public datasets.

use std::cmp::Ordering;
use std::error::Error;
use std::io;
use std::num::NonZeroUsize;

use rstest::rstest;

use chutoro_core::{
    CandidateEdge, CpuHnsw, DataSource, DataSourceError, EdgeHarvest, HierarchyConfig, HnswParams,
    MetricDescriptor, adjusted_rand_index, extract_labels_from_mst, normalized_mutual_information,
    parallel_kruskal,
};

/// Parses `dims` comma-separated floats from each non-blank line of `input`.
///
/// # Errors
///
/// Returns [`io::Error`] when a line has too few columns or a column does not
/// parse as `f32`, so callers surface malformed fixture data as a test failure
/// rather than an opaque panic inside shared setup.
fn parse_csv_rows(input: &str, dims: usize) -> Result<Vec<Vec<f32>>, io::Error> {
    input
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| parse_csv_row(line, dims))
        .collect()
}

fn parse_csv_row(line: &str, dims: usize) -> Result<Vec<f32>, io::Error> {
    let mut parts = line.split(',');
    let mut row = Vec::with_capacity(dims);
    for _ in 0..dims {
        let part = parts.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing column in line: {line}"),
            )
        })?;
        let value = part.parse::<f32>().map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse float in line '{line}': {err}"),
            )
        })?;
        row.push(value);
    }
    Ok(row)
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
        self.rows
            .first()
            .map(std::vec::Vec::len)
            .unwrap_or_default()
    }
}

impl DataSource for DenseVectors {
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn name(&self) -> &'static str {
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
            let difference = a.mul_add(1.0, std::ops::Neg::neg(b));
            sum = difference.mul_add(difference, sum);
        }
        Ok(sum.sqrt())
    }
}

fn core_distances_exact<D: DataSource>(
    source: &D,
    min_cluster_size: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let n = source.len();
    let mut core = vec![0.0_f32; n];
    for (i, core_value) in core.iter_mut().enumerate() {
        let mut distances = Vec::with_capacity(n.saturating_sub(1));
        for j in 0..n {
            if i == j {
                continue;
            }
            distances.push(source.distance(i, j)?);
        }
        distances.sort_by(f32::total_cmp);
        *core_value = distances
            // Select the k-th nearest neighbour distance as the core distance
            // (0-indexed, so `k-1`), matching HDBSCAN's definition.
            .get(min_cluster_size.saturating_sub(1))
            .copied()
            .or_else(|| distances.last().copied())
            .unwrap_or(0.0);
    }
    Ok(core)
}

fn complete_mutual_reachability_edges<D: DataSource>(
    source: &D,
    core: &[f32],
) -> Result<EdgeHarvest, Box<dyn Error>> {
    let n = source.len();
    let mut edges = Vec::new();
    let mut seq = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = source.distance(i, j)?;
            let left_core_distance = *core.get(i).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "missing left core distance")
            })?;
            let right_core_distance = *core.get(j).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "missing right core distance")
            })?;
            let weight = dist.max(left_core_distance).max(right_core_distance);
            edges.push(CandidateEdge::new(i, j, weight, seq));
            seq += 1;
        }
    }
    Ok(EdgeHarvest::new(edges))
}

fn exact_pipeline<D: DataSource>(
    source: &D,
    min_cluster_size: NonZeroUsize,
) -> Result<Vec<usize>, Box<dyn Error>> {
    let core = core_distances_exact(source, min_cluster_size.get())?;
    let edges = complete_mutual_reachability_edges(source, &core)?;
    let forest = parallel_kruskal(source.len(), &edges)?;
    Ok(extract_labels_from_mst(
        source.len(),
        forest.edges(),
        HierarchyConfig::new(min_cluster_size),
    )?)
}

fn high_recall_hnsw_params() -> Result<HnswParams, Box<dyn Error>> {
    Ok(HnswParams::new(32, 128)?.with_rng_seed(0x1A11_CE5E))
}

fn approx_pipeline<D: DataSource + Sync>(
    source: &D,
    min_cluster_size: NonZeroUsize,
) -> Result<Vec<usize>, Box<dyn Error>> {
    let params = high_recall_hnsw_params()?;
    let (index, harvested) = CpuHnsw::build_with_edges(source, params.clone())?;
    let items = source.len();
    let desired = min_cluster_size
        .get()
        .saturating_add(1)
        .max(params.ef_construction())
        .min(items);
    let ef = NonZeroUsize::new(desired)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "ef must be non-zero"))?;

    let mut core_distances = Vec::with_capacity(items);
    let core_neighbour_index = min_cluster_size.get().saturating_sub(1);
    for point in 0..items {
        let neighbours = index.search(source, point, ef)?;
        let others: Vec<_> = neighbours.into_iter().filter(|n| n.id != point).collect();
        let core = others.get(core_neighbour_index).map_or_else(
            || others.last().map_or(0.0, |neighbour| neighbour.distance),
            |neighbour| neighbour.distance,
        );
        core_distances.push(core);
    }

    let mutual_edges: Vec<CandidateEdge> = harvested
        .iter()
        .map(|edge| -> Result<CandidateEdge, io::Error> {
            let left = edge.source();
            let right = edge.target();
            let dist = edge.distance();
            let left_core_distance = *core_distances.get(left).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "missing left core distance")
            })?;
            let right_core_distance = *core_distances.get(right).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "missing right core distance")
            })?;
            let weight = dist.max(left_core_distance).max(right_core_distance);
            Ok(CandidateEdge::new(left, right, weight, edge.sequence()))
        })
        .collect::<Result<_, _>>()?;
    let mutual_harvest = EdgeHarvest::new(mutual_edges);

    let forest = parallel_kruskal(items, &mutual_harvest)?;
    Ok(extract_labels_from_mst(
        items,
        forest.edges(),
        HierarchyConfig::new(min_cluster_size),
    )?)
}

#[test]
fn nmi_is_one_when_both_partitions_have_single_cluster() {
    let labels = vec![0, 0, 0, 0];
    assert_eq!(
        normalized_mutual_information(&labels, &labels)
            .expect("NMI should compute")
            .total_cmp(&1.0),
        Ordering::Equal
    );
}

#[test]
fn metrics_identity_and_permutation_are_one() {
    let labels = vec![0, 0, 1, 1, 2, 2];
    assert_eq!(
        adjusted_rand_index(&labels, &labels)
            .expect("ARI should compute")
            .total_cmp(&1.0),
        Ordering::Equal
    );
    assert_eq!(
        normalized_mutual_information(&labels, &labels)
            .expect("NMI should compute")
            .total_cmp(&1.0),
        Ordering::Equal
    );

    let permuted = vec![1, 1, 2, 2, 0, 0];
    assert_eq!(
        adjusted_rand_index(&labels, &permuted)
            .expect("ARI should compute")
            .total_cmp(&1.0),
        Ordering::Equal
    );
    assert_eq!(
        normalized_mutual_information(&labels, &permuted)
            .expect("NMI should compute")
            .total_cmp(&1.0),
        Ordering::Equal
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

const fn iris_dataset() -> Dataset {
    Dataset {
        name: "iris",
        dims: 4,
        data: include_str!("data/iris.csv"),
    }
}

const fn ruspini_dataset() -> Dataset {
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
    #[case] minimum_cluster_size: usize,
    #[case] min_ari: f64,
    #[case] min_nmi: f64,
) -> Result<(), Box<dyn Error>> {
    let rows = parse_csv_rows(dataset.data, dataset.dims)?;
    let source = DenseVectors::new("euclidean", rows);
    if source.dim() != dataset.dims {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "parsed dataset does not match declared dimensionality",
        )
        .into());
    }

    let min_cluster_size = NonZeroUsize::new(minimum_cluster_size).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "min_cluster_size must be non-zero",
        )
    })?;

    let exact = exact_pipeline(&source, min_cluster_size)?;
    let approx = approx_pipeline(&source, min_cluster_size)?;

    let ari = adjusted_rand_index(&exact, &approx).expect("ARI should compute");
    let nmi = normalized_mutual_information(&exact, &approx).expect("NMI should compute");

    let exact_cluster_count = exact.iter().copied().max().unwrap_or(0).saturating_add(1);
    let approximate_cluster_count = approx.iter().copied().max().unwrap_or(0).saturating_add(1);
    if ari < min_ari {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                concat!(
                    "dataset={} ARI {ari} < {min_ari} (clusters exact={exact_cluster_count}, ",
                    "approx={approximate_cluster_count})"
                ),
                dataset.name,
                ari = ari,
                min_ari = min_ari,
                exact_cluster_count = exact_cluster_count,
                approximate_cluster_count = approximate_cluster_count,
            ),
        )
        .into());
    }
    if nmi < min_nmi {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                concat!(
                    "dataset={} NMI {nmi} < {min_nmi} (clusters exact={exact_cluster_count}, ",
                    "approx={approximate_cluster_count})"
                ),
                dataset.name,
                nmi = nmi,
                min_nmi = min_nmi,
                exact_cluster_count = exact_cluster_count,
                approximate_cluster_count = approximate_cluster_count,
            ),
        )
        .into());
    }
    Ok(())
}
