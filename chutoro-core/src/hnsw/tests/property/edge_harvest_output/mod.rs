//! Candidate edge harvest output property suite.
//!
//! Validates harvested output graphs derived from generated topologies.

pub(super) const HARVEST_CASES_PER_TOPOLOGY: u32 = 256;
/// Minimum percentage of connected inputs that must remain connected.
pub(super) const CONNECTIVITY_PRESERVATION_PERCENT: usize = 95;

mod harvest;
mod suite;
