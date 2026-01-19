# ExecPlan: Phase 1 — Candidate Edge Harvest Property Suite

**Status**: Complete
**Issue**: See `docs/roadmap.md` Phase 1
**Branch**: `terragon/add-candidate-edge-harvest-suite-bug81f`

______________________________________________________________________

## Big Picture

Add a candidate edge harvest property suite covering the four properties
specified in `docs/property-testing-design.md` §3.2:

1. **Determinism**: Identical output across runs for fixed seed
2. **Degree Ceilings**: Node degrees bounded per topology constraints
3. **Connectivity Preservation**: Connected inputs yield connected outputs
4. **Reverse Nearest Neighbour (RNN) Uplift**: Symmetric edge ratio metrics

The suite extends the existing graph topology test infrastructure and
integrates with `graph_fixture_strategy()` for comprehensive coverage.

______________________________________________________________________

## Constraints

- Must follow existing test organisation in
  `/root/repo/chutoro-core/src/hnsw/tests/property/`
- Use `rstest` for parameterised test cases with `#[case(...)]`
- Use proptest strategies where appropriate
- All code must pass `make check-fmt`, `make lint`, `make test`
- Document design decisions in this plan
- Mark roadmap entry as done on completion

______________________________________________________________________

## Design Decisions

### D1: Scope Clarification

The existing `edge_harvest_property.rs` tests HNSW edge harvest during
index construction (via `build_with_edges()`). The new suite tests the
**graph topology generators** themselves (`generate_random_graph`, etc.)
as specified in §3.2, which focuses on graph-level properties rather than
HNSW-specific behaviour.

**Rationale**: The property-testing-design document §3.2 specifies testing
candidate edge harvest algorithms across different graph structures. The
graph generators produce these structures for downstream MST testing.

### D2: Determinism Implementation

Determinism is verified by:

1. Seeding `SmallRng` with identical seeds
2. Calling the same generator twice
3. Asserting exact edge vector equality (edges are already sorted)

This complements the existing HNSW rebuild tolerance test which allows
variance due to Rayon's non-deterministic thread scheduling.

### D3: Degree Ceiling Bounds

Topology-specific ceilings:

| Topology     | Ceiling                           | Rationale                       |
| ------------ | --------------------------------- | ------------------------------- |
| Lattice      | 4 (no diagonals) or 8 (diagonals) | Grid connectivity bound         |
| ScaleFree    | `node_count - 1`                  | Hub worst case                  |
| Random       | `node_count - 1`                  | Complete graph worst case       |
| Disconnected | `max(component_sizes) - 1`        | Within largest component        |

### D4: Connectivity via Union-Find

A simple sequential union-find implementation with path compression counts
connected components. Assertions by topology:

| Topology     | Assertion                                     |
| ------------ | --------------------------------------------- |
| Lattice      | Exactly 1 component                           |
| ScaleFree    | Exactly 1 component (for n > 3)               |
| Random       | Informational only (probabilistic)            |
| Disconnected | At least `component_count` components         |

**Rationale**: Lattice grids and Barabasi-Albert scale-free graphs are
connected by construction. Random graphs may be disconnected depending on
edge probability. Disconnected graphs must have at least the specified
number of components.

### D5: RNN Uplift Metric

RNN score = fraction of top-k neighbours that are mutual. For node `u` with
neighbour `v` in its top-k, the relationship is symmetric if `u` is also in
`v`'s top-k.

- Use k=5 as typical neighbourhood size
- Build adjacency lists with distances
- Sort by distance, take top-k per node
- Count symmetric relationships

Minimum thresholds by topology:

| Topology     | Threshold | Rationale                    |
| ------------ | --------- | ---------------------------- |
| Lattice      | 0.8       | Highly regular structure     |
| ScaleFree    | 0.3       | Hubs create asymmetry        |
| Random       | 0.4       | Moderate symmetry expected   |
| Disconnected | 0.4       | Within-component symmetry    |

______________________________________________________________________

## Implementation Summary

### Files Created

| File                                                               | Purpose                            |
| ------------------------------------------------------------------ | ---------------------------------- |
| `chutoro-core/src/hnsw/tests/property/edge_harvest_suite.rs`       | New property suite module          |

### Files Modified

| File                                                        | Change                          |
| ----------------------------------------------------------- | ------------------------------- |
| `chutoro-core/src/hnsw/tests/property/mod.rs`               | Register new module             |
| `docs/roadmap.md`                                           | Mark task complete              |

### Helper Functions Implemented

```rust
fn compute_node_degrees(node_count: usize, edges: &[CandidateEdge]) -> Vec<usize>
fn count_connected_components(node_count: usize, edges: &[CandidateEdge]) -> usize
fn compute_rnn_score(node_count: usize, edges: &[CandidateEdge], k: usize) -> f64
```

### Property Functions Implemented

```rust
pub(super) fn run_graph_determinism_property(seed: u64, topology: GraphTopology) -> TestCaseResult
pub(super) fn run_degree_ceiling_property(fixture: &GraphFixture) -> TestCaseResult
pub(super) fn run_connectivity_preservation_property(fixture: &GraphFixture) -> TestCaseResult
pub(super) fn run_rnn_uplift_property(fixture: &GraphFixture) -> TestCaseResult
```

### Test Coverage

**Rstest parameterised cases** (20 total):

- 8 determinism cases (2 seeds × 4 topologies)
- 4 degree ceiling cases (1 per topology)
- 4 connectivity cases (1 per topology)
- 4 RNN uplift cases (1 per topology)

**Proptest stochastic coverage** (3 tests, 64 cases each):

- `graph_topology_degree_ceilings_proptest`
- `graph_topology_connectivity_proptest`
- `graph_topology_rnn_uplift_proptest`

**Helper function unit tests** (9 tests):

- `compute_node_degrees_empty_graph`
- `compute_node_degrees_simple_chain`
- `count_connected_components_empty_graph`
- `count_connected_components_isolated_nodes`
- `count_connected_components_fully_connected`
- `count_connected_components_two_components`
- `compute_rnn_score_empty_graph`
- `compute_rnn_score_symmetric_pair`
- `compute_rnn_score_asymmetric_star`

**Additional edge case tests** (3 tests):

- `determinism_across_multiple_seeds`
- `lattice_without_diagonals_max_degree_is_four`
- `scale_free_has_hub_nodes`

______________________________________________________________________

## Verification Steps

1. **Format check**:

   ```bash
   set -o pipefail && make check-fmt 2>&1 | tee /tmp/check-fmt.log
   ```

2. **Lint check**:

   ```bash
   set -o pipefail && make lint 2>&1 | tee /tmp/lint.log
   ```

3. **Run all tests**:

   ```bash
   set -o pipefail && make test 2>&1 | tee /tmp/test.log
   ```

4. **Run new property tests specifically**:

   ```bash
   cargo test -p chutoro-core edge_harvest_suite
   ```

______________________________________________________________________

## Progress Log

| Date       | Status   | Notes                                      |
| ---------- | -------- | ------------------------------------------ |
| 2026-01-19 | Complete | Implementation finished, tests passing     |

______________________________________________________________________

## References

- `docs/property-testing-design.md` §3.2 — Property specifications
- `docs/roadmap.md` Phase 1 — Task entry
- `chutoro-core/src/hnsw/tests/property/graph_topology_tests/mod.rs` — Existing patterns
- `chutoro-core/src/hnsw/tests/property/edge_harvest_property.rs` — Related tests
- `chutoro-core/src/hnsw/tests/property/strategies.rs` — `graph_fixture_strategy()`
