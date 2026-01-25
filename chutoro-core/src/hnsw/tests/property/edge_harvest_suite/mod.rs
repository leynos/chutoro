//! Candidate edge harvest property suite.
//!
//! Verifies that graph topology generators produce structurally valid edge sets
//! with bounded degrees, preserved connectivity, and symmetric neighbour relationships.
//!
//! Properties verified (per `docs/property-testing-design.md` §3.2):
//! 1. **Determinism**: Same seed produces identical output
//! 2. **Degree ceilings**: Node degrees within topology-specific bounds
//! 3. **Connectivity preservation**: Connected topologies remain connected
//! 4. **RNN uplift**: Measure of symmetric neighbour relationships

use rand::{SeedableRng, rngs::SmallRng};

use super::graph_topologies::generate_graph_for_topology;
pub(super) use super::types::{GeneratedGraph, GraphFixture, GraphMetadata, GraphTopology};

mod connectivity;
mod degree_ceiling;
mod determinism;
mod rnn_uplift;

#[cfg(test)]
fn build_fixture(seed: u64, topology: GraphTopology) -> GraphFixture {
    let mut rng = SmallRng::seed_from_u64(seed);
    let graph = generate_graph_for_topology(topology, &mut rng);
    GraphFixture { topology, graph }
}
