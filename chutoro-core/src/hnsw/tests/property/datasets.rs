use rand::{Rng, distributions::Uniform, rngs::SmallRng};

use super::{
    support::{dot, l2_norm, unit_vector},
    types::{ClusterInfo, DistributionMetadata},
};

pub(super) struct GeneratedDataset {
    pub vectors: Vec<Vec<f32>>,
    pub metadata: DistributionMetadata,
}

pub(super) fn generate_uniform_dataset(rng: &mut SmallRng) -> GeneratedDataset {
    let dimension = rng.gen_range(2..=16);
    let len = rng.gen_range(8..=64);
    let bound = rng.gen_range(1.0..=10.0);
    let dist = Uniform::new_inclusive(-bound, bound);
    let vectors = (0..len)
        .map(|_| sample_vector(dimension, rng, dist))
        .collect();
    GeneratedDataset {
        vectors,
        metadata: DistributionMetadata::Uniform { bound },
    }
}

pub(super) fn generate_clustered_dataset(rng: &mut SmallRng) -> GeneratedDataset {
    let dimension = rng.gen_range(2..=24);
    let cluster_count = rng.gen_range(2..=5);
    let points_per_cluster = rng.gen_range(4..=12);
    let radius = rng.gen_range(0.05..=0.75);
    let dist_centroid = Uniform::new_inclusive(-12.0, 12.0);
    let offset = Uniform::new_inclusive(-radius, radius);
    let mut vectors = Vec::with_capacity(cluster_count * points_per_cluster);
    let mut clusters = Vec::with_capacity(cluster_count);
    for _ in 0..cluster_count {
        let centroid = sample_vector(dimension, rng, dist_centroid);
        let start = vectors.len();
        for _ in 0..points_per_cluster {
            let mut point = centroid.clone();
            for coord in &mut point {
                *coord += rng.sample(offset);
            }
            vectors.push(point);
        }
        clusters.push(ClusterInfo {
            start,
            len: points_per_cluster,
            radius,
            centroid,
        });
    }
    GeneratedDataset {
        vectors,
        metadata: DistributionMetadata::Clustered { clusters },
    }
}

pub(super) fn generate_manifold_dataset(rng: &mut SmallRng) -> GeneratedDataset {
    let ambient_dim: usize = rng.gen_range(3..=24);
    let mut intrinsic_dim: usize = rng.gen_range(1..=3);
    intrinsic_dim = intrinsic_dim.min(ambient_dim.saturating_sub(1)).max(1);
    let noise_bound = rng.gen_range(0.0..=0.05);
    let len = rng.gen_range(10..=48);
    let origin = sample_vector(ambient_dim, rng, Uniform::new_inclusive(-6.0, 6.0));
    let mut basis = orthonormal_basis(ambient_dim, intrinsic_dim, rng);
    if basis.is_empty() {
        basis = vec![unit_vector(ambient_dim, 0)];
    }
    let vectors = (0..len)
        .map(|_| generate_manifold_point(rng, &origin, &basis, noise_bound))
        .collect();
    GeneratedDataset {
        vectors,
        metadata: DistributionMetadata::Manifold {
            ambient_dim,
            intrinsic_dim: basis.len(),
            basis,
            noise_bound,
            origin,
        },
    }
}

pub(super) fn generate_duplicate_dataset(rng: &mut SmallRng) -> GeneratedDataset {
    let dimension = rng.gen_range(2..=16);
    let base_len = rng.gen_range(6..=24);
    let bound = rng.gen_range(1.0..=8.0);
    let dist = Uniform::new_inclusive(-bound, bound);
    let mut vectors: Vec<Vec<f32>> = (0..base_len)
        .map(|_| sample_vector(dimension, rng, dist))
        .collect();
    let mut groups = Vec::new();
    let duplicate_groups = rng.gen_range(1..=4);
    for _ in 0..duplicate_groups {
        let source_index = rng.gen_range(0..vectors.len());
        let copies = rng.gen_range(2..=4);
        let mut indices = vec![source_index];
        let template = vectors[source_index].clone();
        for _ in 1..copies {
            vectors.push(template.clone());
            indices.push(vectors.len() - 1);
        }
        groups.push(indices);
    }
    GeneratedDataset {
        vectors,
        metadata: DistributionMetadata::Duplicates { groups },
    }
}

fn sample_vector(dimension: usize, rng: &mut SmallRng, dist: Uniform<f32>) -> Vec<f32> {
    (0..dimension).map(|_| rng.sample(dist)).collect()
}

fn orthonormal_basis(
    ambient_dim: usize,
    intrinsic_dim: usize,
    rng: &mut SmallRng,
) -> Vec<Vec<f32>> {
    let mut basis = Vec::new();
    for axis in 0..intrinsic_dim {
        let mut candidate = sample_vector(ambient_dim, rng, Uniform::new_inclusive(-1.0, 1.0));
        gram_schmidt_step(&mut candidate, &basis);
        let norm = l2_norm(&candidate);
        if norm > f32::EPSILON {
            for value in &mut candidate {
                *value /= norm;
            }
            basis.push(candidate);
        } else {
            basis.push(unit_vector(ambient_dim, axis));
        }
    }
    basis
}

fn gram_schmidt_step(vector: &mut [f32], basis: &[Vec<f32>]) {
    for base in basis {
        let projection = dot(vector, base);
        for (value, base_component) in vector.iter_mut().zip(base) {
            *value -= projection * base_component;
        }
    }
}

fn generate_manifold_point(
    rng: &mut SmallRng,
    origin: &[f32],
    basis: &[Vec<f32>],
    noise_bound: f32,
) -> Vec<f32> {
    let coeffs: Vec<f32> = (0..basis.len())
        .map(|_| rng.gen_range(-4.0..=4.0))
        .collect();
    let mut point = project_onto_manifold(origin, basis, &coeffs);
    apply_noise(rng, &mut point, noise_bound);
    point
}

fn project_onto_manifold(origin: &[f32], basis: &[Vec<f32>], coeffs: &[f32]) -> Vec<f32> {
    let mut point = origin.to_vec();
    for (basis_vec, coeff) in basis.iter().zip(coeffs) {
        for (value, basis_value) in point.iter_mut().zip(basis_vec) {
            *value += coeff * basis_value;
        }
    }
    point
}

fn apply_noise(rng: &mut SmallRng, point: &mut [f32], noise_bound: f32) {
    if noise_bound <= 0.0 {
        return;
    }
    for value in point {
        *value += rng.gen_range(-noise_bound..=noise_bound);
    }
}
