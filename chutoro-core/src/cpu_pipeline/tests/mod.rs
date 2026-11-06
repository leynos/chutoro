//! Tests for the CPU pipeline orchestration helpers.
//!
//! The production pipeline is compared against a sequential reference so the
//! parallel mutual-reachability transform and the slice-based Kruskal entry
//! point are checked end to end, not only through their own unit tests.

mod fixture;
#[cfg(feature = "metrics")]
mod metrics;
mod mutual_reachability;
mod pipeline_reference;
mod reference;
