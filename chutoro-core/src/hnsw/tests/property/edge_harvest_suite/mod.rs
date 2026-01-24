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

use super::graph_topologies::{
    generate_disconnected_graph, generate_lattice_graph, generate_random_graph,
    generate_scale_free_graph,
};
pub(super) use super::types::{GeneratedGraph, GraphFixture, GraphMetadata, GraphTopology};

mod connectivity;
mod degree_ceiling;
mod determinism;
mod rnn_uplift;

/// Generates a graph for the specified topology using the provided RNG.
///
/// Dispatches to the appropriate topology-specific generator.
fn generate_graph_for_topology(topology: GraphTopology, rng: &mut SmallRng) -> GeneratedGraph {
    match topology {
        GraphTopology::Random => generate_random_graph(rng),
        GraphTopology::ScaleFree => generate_scale_free_graph(rng),
        GraphTopology::Lattice => generate_lattice_graph(rng),
        GraphTopology::Disconnected => generate_disconnected_graph(rng),
    }
}

#[cfg(test)]
fn build_fixture(seed: u64, topology: GraphTopology) -> GraphFixture {
    let mut rng = SmallRng::seed_from_u64(seed);
    let graph = generate_graph_for_topology(topology, &mut rng);
    GraphFixture { topology, graph }
}
