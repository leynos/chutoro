//! Test fixture loading utilities.
//!
//! Provides functions for loading test fixtures from embedded JSON files.

use serde::Deserialize;

#[derive(Deserialize)]
struct BootstrapVectors(Vec<Vec<f32>>);

/// Loads bootstrap uniform vectors from the embedded fixture file.
pub(super) fn load_bootstrap_uniform_vectors_from_fixture() -> Vec<Vec<f32>> {
    const RAW: &str = include_str!("bootstrap_uniform_vectors.json");
    serde_json::from_str::<BootstrapVectors>(RAW)
        .expect("bootstrap uniform vectors fixture should parse")
        .0
}
