//! Functional clustering tests comparing exact and HNSW-based pipelines.
//!
//! These tests verify that the approximate CPU pipeline (HNSW candidate edge
//! harvest + Kruskal MST + hierarchy extraction) produces a clustering that is
//! close to an exact baseline computed from the full mutual-reachability graph
//! on small public datasets.

use std::num::NonZeroUsize;

use rstest::rstest;

use chutoro_core::{
    CandidateEdge, CpuHnsw, DataSource, DataSourceError, EdgeHarvest, HierarchyConfig, HnswParams,
    MetricDescriptor, extract_labels_from_mst, parallel_kruskal,
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

fn hnsw_pipeline(
    source: &DenseVectors,
    min_cluster_size: NonZeroUsize,
    params: HnswParams,
) -> Vec<usize> {
    let (index, harvested) =
        CpuHnsw::build_with_edges(source, params.clone()).expect("HNSW build should succeed");

    let n = source.len();
    let ef = {
        let desired = min_cluster_size
            .get()
            .saturating_add(1)
            .max(params.ef_construction())
            .min(n);
        NonZeroUsize::new(desired).unwrap_or_else(|| NonZeroUsize::new(1).expect("non-zero"))
    };

    let mut core = Vec::with_capacity(n);
    for point in 0..n {
        let neighbours = index
            .search(source, point, ef)
            .expect("HNSW search should succeed");
        let others: Vec<_> = neighbours.into_iter().filter(|n| n.id != point).collect();
        let k = min_cluster_size.get();
        let value = if others.len() >= k {
            others[k - 1].distance
        } else {
            others.last().map(|n| n.distance).unwrap_or(0.0)
        };
        core.push(value);
    }

    let mutual: Vec<CandidateEdge> = harvested
        .iter()
        .map(|edge| {
            let left = edge.source();
            let right = edge.target();
            let weight = edge.distance().max(core[left]).max(core[right]);
            CandidateEdge::new(left, right, weight, edge.sequence())
        })
        .collect();
    let mutual = EdgeHarvest::new(mutual);
    let forest = parallel_kruskal(source.len(), &mutual).expect("approx MST should succeed");
    extract_labels_from_mst(
        source.len(),
        forest.edges(),
        HierarchyConfig::new(min_cluster_size),
    )
    .expect("approx hierarchy extraction should succeed")
}

fn comb2(value: usize) -> f64 {
    let v = value as f64;
    v * (v - 1.0) / 2.0
}

fn adjusted_rand_index(left: &[usize], right: &[usize]) -> f64 {
    assert_eq!(left.len(), right.len());
    let n = left.len();
    if n < 2 {
        return 1.0;
    }

    use std::collections::HashMap;
    let mut left_ids = HashMap::<usize, usize>::new();
    let mut right_ids = HashMap::<usize, usize>::new();
    let mut left_counts = Vec::<usize>::new();
    let mut right_counts = Vec::<usize>::new();
    let mut contingency = HashMap::<(usize, usize), usize>::new();

    for (&l, &r) in left.iter().zip(right.iter()) {
        let li = *left_ids.entry(l).or_insert_with(|| {
            let id = left_counts.len();
            left_counts.push(0);
            id
        });
        let ri = *right_ids.entry(r).or_insert_with(|| {
            let id = right_counts.len();
            right_counts.push(0);
            id
        });
        left_counts[li] += 1;
        right_counts[ri] += 1;
        *contingency.entry((li, ri)).or_insert(0) += 1;
    }

    let sum_ij: f64 = contingency.values().copied().map(comb2).sum();
    let sum_i: f64 = left_counts.iter().copied().map(comb2).sum();
    let sum_j: f64 = right_counts.iter().copied().map(comb2).sum();
    let total = comb2(n);
    if total == 0.0 {
        return 1.0;
    }

    let expected = (sum_i * sum_j) / total;
    let max_index = 0.5 * (sum_i + sum_j);
    let denom = max_index - expected;
    if denom == 0.0 {
        0.0
    } else {
        (sum_ij - expected) / denom
    }
}

fn normalised_mutual_information(left: &[usize], right: &[usize]) -> f64 {
    assert_eq!(left.len(), right.len());
    let n = left.len();
    if n == 0 {
        return 1.0;
    }

    use std::collections::HashMap;
    let mut left_counts = HashMap::<usize, usize>::new();
    let mut right_counts = HashMap::<usize, usize>::new();
    let mut contingency = HashMap::<(usize, usize), usize>::new();

    for (&l, &r) in left.iter().zip(right.iter()) {
        *left_counts.entry(l).or_insert(0) += 1;
        *right_counts.entry(r).or_insert(0) += 1;
        *contingency.entry((l, r)).or_insert(0) += 1;
    }

    let n_f64 = n as f64;
    let mut mi = 0.0_f64;
    for ((l, r), count) in contingency {
        let count = count as f64;
        if count == 0.0 {
            continue;
        }
        let pl = *left_counts.get(&l).expect("exists") as f64;
        let pr = *right_counts.get(&r).expect("exists") as f64;
        mi += (count / n_f64) * ((count * n_f64) / (pl * pr)).ln();
    }

    let entropy = |counts: &HashMap<usize, usize>| {
        let mut h = 0.0_f64;
        for &count in counts.values() {
            let p = (count as f64) / n_f64;
            h -= p * p.ln();
        }
        h
    };

    let h_left = entropy(&left_counts);
    let h_right = entropy(&right_counts);
    if h_left == 0.0 || h_right == 0.0 {
        0.0
    } else {
        mi / (h_left * h_right).sqrt()
    }
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

#[rstest]
#[case(iris_dataset(), 5, 0.80, 0.80)]
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
    let params = HnswParams::new(32, 128)
        .expect("params must be valid")
        .with_rng_seed(0xDEC0_DED1);
    let approx = hnsw_pipeline(&source, min_cluster_size, params);

    let ari = adjusted_rand_index(&exact, &approx);
    let nmi = normalised_mutual_information(&exact, &approx);

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
