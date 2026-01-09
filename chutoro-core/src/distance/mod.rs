//! Distance primitives for built-in numeric metrics.
//!
//! The walking skeleton exposes scalar implementations for Euclidean and
//! cosine distances. These routines validate their inputs and surface detailed
//! errors so callers can react appropriately during ingestion or algorithmic
//! execution.

mod cosine;
mod euclidean;
mod helpers;
mod types;

pub use self::cosine::cosine_distance;
pub use self::euclidean::euclidean_distance;
pub use self::types::{CosineNorms, Distance, DistanceError, Norm, Result, VectorKind};

// ============================================================================
// Kani Formal Verification
// ============================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::{cosine_distance, euclidean_distance};

    /// Tolerance for floating-point comparisons in distance functions.
    const EPSILON: f32 = 1e-5;

    /// Verifies Euclidean distance symmetry: d(a, b) = d(b, a).
    ///
    /// This harness uses 3-dimensional vectors with nondeterministic finite
    /// values and verifies that the distance function is symmetric.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_euclidean_symmetry_3d() {
        let a: [f32; 3] = [kani::any(), kani::any(), kani::any()];
        let b: [f32; 3] = [kani::any(), kani::any(), kani::any()];

        // Constrain to finite values
        for &v in &a {
            kani::assume(v.is_finite());
        }
        for &v in &b {
            kani::assume(v.is_finite());
        }

        let ab = euclidean_distance(&a, &b);
        let ba = euclidean_distance(&b, &a);

        match (ab, ba) {
            (Ok(d1), Ok(d2)) => {
                kani::assert(
                    (d1.value() - d2.value()).abs() < EPSILON,
                    "euclidean distance symmetry violated",
                );
            }
            (Err(_), Err(_)) => {
                // Both error is acceptable (same validation failure)
            }
            _ => {
                kani::assert(false, "asymmetric error behavior in euclidean distance");
            }
        }
    }

    /// Verifies Euclidean distance is zero on identical inputs: d(v, v) = 0.
    ///
    /// This harness verifies that the distance from a vector to itself is zero
    /// (within floating-point tolerance).
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_euclidean_zero_on_identical_3d() {
        let v: [f32; 3] = [kani::any(), kani::any(), kani::any()];

        for &x in &v {
            kani::assume(x.is_finite());
        }

        if let Ok(d) = euclidean_distance(&v, &v) {
            kani::assert(
                d.value().abs() < EPSILON,
                "euclidean distance not zero on identical inputs",
            );
        }
    }

    /// Verifies Euclidean distance is non-negative: d(a, b) >= 0.
    ///
    /// This harness verifies the non-negativity property of the Euclidean
    /// distance metric.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_euclidean_non_negative_3d() {
        let a: [f32; 3] = [kani::any(), kani::any(), kani::any()];
        let b: [f32; 3] = [kani::any(), kani::any(), kani::any()];

        for &v in &a {
            kani::assume(v.is_finite());
        }
        for &v in &b {
            kani::assume(v.is_finite());
        }

        if let Ok(d) = euclidean_distance(&a, &b) {
            kani::assert(d.value() >= 0.0, "euclidean distance must be non-negative");
        }
    }

    /// Verifies cosine distance symmetry: d(a, b) = d(b, a).
    ///
    /// This harness uses 3-dimensional vectors with nondeterministic finite
    /// values and verifies that the cosine distance function is symmetric.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_cosine_symmetry_3d() {
        let a: [f32; 3] = [kani::any(), kani::any(), kani::any()];
        let b: [f32; 3] = [kani::any(), kani::any(), kani::any()];

        for &v in &a {
            kani::assume(v.is_finite());
        }
        for &v in &b {
            kani::assume(v.is_finite());
        }

        let ab = cosine_distance(&a, &b, None);
        let ba = cosine_distance(&b, &a, None);

        match (ab, ba) {
            (Ok(d1), Ok(d2)) => {
                kani::assert(
                    (d1.value() - d2.value()).abs() < EPSILON,
                    "cosine distance symmetry violated",
                );
            }
            (Err(_), Err(_)) => {
                // Both error is acceptable (e.g., zero-magnitude vectors)
            }
            _ => {
                kani::assert(false, "asymmetric error behavior in cosine distance");
            }
        }
    }

    /// Verifies cosine distance is zero on identical non-zero inputs: d(v, v) = 0.
    ///
    /// This harness verifies that the cosine distance from a non-zero vector
    /// to itself is zero (within floating-point tolerance).
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_cosine_zero_on_identical_3d() {
        let v: [f32; 3] = [kani::any(), kani::any(), kani::any()];

        for &x in &v {
            kani::assume(x.is_finite());
        }

        if let Ok(d) = cosine_distance(&v, &v, None) {
            kani::assert(
                d.value().abs() < EPSILON,
                "cosine distance not zero on identical inputs",
            );
        }
    }

    /// Verifies cosine distance is bounded: 0 <= d(a, b) <= 2.
    ///
    /// Cosine similarity ranges from -1 to 1, so cosine distance (1 - similarity)
    /// ranges from 0 to 2.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_cosine_bounded_3d() {
        let a: [f32; 3] = [kani::any(), kani::any(), kani::any()];
        let b: [f32; 3] = [kani::any(), kani::any(), kani::any()];

        for &v in &a {
            kani::assume(v.is_finite());
        }
        for &v in &b {
            kani::assume(v.is_finite());
        }

        if let Ok(d) = cosine_distance(&a, &b, None) {
            kani::assert(
                d.value() >= 0.0 && d.value() <= 2.0,
                "cosine distance must be in [0, 2]",
            );
        }
    }
}
