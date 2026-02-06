//! Property-based tests for the parallel Kruskal MST implementation.
//!
//! Verifies the parallel Kruskal algorithm against a sequential oracle,
//! validates structural invariants (acyclicity, connectivity, edge count),
//! and checks for concurrency-induced non-determinism across graph
//! topologies with varied weight distributions.
//!
//! See `docs/property-testing-design.md` Section 4.

mod concurrency;
mod equivalence;
mod oracle;
mod strategies;
mod structural;
#[cfg(test)]
mod tests;
mod types;
