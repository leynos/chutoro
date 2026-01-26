//! Candidate edge harvest output property suite.
//!
//! Validates harvested output graphs derived from generated topologies.

pub(super) const HARVEST_CASES_PER_TOPOLOGY: u32 = 256;
pub(super) const CONNECTIVITY_PRESERVATION_THRESHOLD: f64 = 0.95;

mod harvest;
mod suite;
