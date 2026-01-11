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
    use super::{Result, cosine_distance, euclidean_distance};

    /// Tolerance for floating-point comparisons in distance functions.
    const EPSILON: f32 = 1e-5;

    // ========================================================================
    // Helper Functions
    // ========================================================================

    /// Creates a 3D f32 array with nondeterministic finite values.
    ///
    /// Each component is constrained to be finite via `kani::assume`.
    fn make_finite_3d_vector() -> [f32; 3] {
        let v: [f32; 3] = [kani::any(), kani::any(), kani::any()];
        for &x in &v {
            kani::assume(x.is_finite());
        }
        v
    }

    /// Asserts that two distance results are symmetric.
    ///
    /// Verifies d(a, b) = d(b, a) within EPSILON tolerance, or that both
    /// computations fail with errors (acceptable for invalid inputs).
    fn assert_symmetric_distance(d1: Result, d2: Result, metric_name: &str) {
        match (d1, d2) {
            (Ok(dist1), Ok(dist2)) => {
                kani::assert(
                    (dist1.value() - dist2.value()).abs() < EPSILON,
                    &format!("{} distance symmetry violated", metric_name),
                );
            }
            (Err(_), Err(_)) => {
                // Both error is acceptable (same validation failure)
            }
            _ => {
                kani::assert(
                    false,
                    &format!("asymmetric error behavior in {} distance", metric_name),
                );
            }
        }
    }

    /// Asserts that a distance result is zero within tolerance.
    ///
    /// Used to verify d(v, v) = 0 for identity property.
    fn assert_zero_distance(d: Result, metric_name: &str) {
        if let Ok(dist) = d {
            kani::assert(
                dist.value().abs() < EPSILON,
                &format!("{} distance not zero on identical inputs", metric_name),
            );
        }
    }

    // ========================================================================
    // Proof Functions
    // ========================================================================

    /// Verifies Euclidean distance symmetry: d(a, b) = d(b, a).
    ///
    /// This harness uses 3-dimensional vectors with nondeterministic finite
    /// values and verifies that the distance function is symmetric.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_euclidean_symmetry_3d() {
        let a = make_finite_3d_vector();
        let b = make_finite_3d_vector();

        let ab = euclidean_distance(&a, &b);
        let ba = euclidean_distance(&b, &a);

        assert_symmetric_distance(ab, ba, "euclidean");
    }

    /// Verifies Euclidean distance is zero on identical inputs: d(v, v) = 0.
    ///
    /// This harness verifies that the distance from a vector to itself is zero
    /// (within floating-point tolerance).
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_euclidean_zero_on_identical_3d() {
        let v = make_finite_3d_vector();

        assert_zero_distance(euclidean_distance(&v, &v), "euclidean");
    }

    /// Verifies Euclidean distance is non-negative: d(a, b) >= 0.
    ///
    /// This harness verifies the non-negativity property of the Euclidean
    /// distance metric.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_euclidean_non_negative_3d() {
        let a = make_finite_3d_vector();
        let b = make_finite_3d_vector();

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
        let a = make_finite_3d_vector();
        let b = make_finite_3d_vector();

        let ab = cosine_distance(&a, &b, None);
        let ba = cosine_distance(&b, &a, None);

        assert_symmetric_distance(ab, ba, "cosine");
    }

    /// Verifies cosine distance is zero on identical non-zero inputs: d(v, v) = 0.
    ///
    /// This harness verifies that the cosine distance from a non-zero vector
    /// to itself is zero (within floating-point tolerance).
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_cosine_zero_on_identical_3d() {
        let v = make_finite_3d_vector();

        assert_zero_distance(cosine_distance(&v, &v, None), "cosine");
    }

    /// Verifies cosine distance is bounded: 0 <= d(a, b) <= 2.
    ///
    /// Cosine similarity ranges from -1 to 1, so cosine distance (1 - similarity)
    /// ranges from 0 to 2.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_cosine_bounded_3d() {
        let a = make_finite_3d_vector();
        let b = make_finite_3d_vector();

        if let Ok(d) = cosine_distance(&a, &b, None) {
            kani::assert(
                d.value() >= 0.0 && d.value() <= 2.0,
                "cosine distance must be in [0, 2]",
            );
        }
    }
}
