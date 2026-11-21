# Architecture Decision Record (ADR) 002: Adoption of Kani Formal Verification

## Status

Accepted

Amended 2026-11-22: a dated entry in the amendment log below records the
concurrency boundary. The decision is unchanged. See [Amendments](#amendments).

## Date

2026-11-20

## Context

The chutoro library implements an HNSW (Hierarchical Navigable Small World)
graph that relies on several structural invariants for correctness:

1. **Bidirectional links**: Every edge `(source, target)` has a reverse edge
   `(target, source)` at the same layer
2. **Layer consistency**: Nodes at layer L exist at all layers below L
3. **Degree bounds**: Node degrees respect per-layer connection limits
4. **Reachability**: All nodes are reachable from the entry point

Currently, these invariants are verified via property-based testing using
proptest (v1.8.0). While proptest provides strong probabilistic coverage
through random input generation and intelligent shrinking, it cannot
exhaustively verify all possible states. Recent property test failures
documented in ADR-001 demonstrate the challenge of achieving complete coverage
for edge cases, particularly around reciprocity enforcement during trimming
operations.

Kani is a bit-precise model checker for Rust that uses symbolic execution to
exhaustively verify all possible inputs within bounded constraints. Rather than
sampling random inputs, Kani explores *every* possible execution path,
providing formal proofs of correctness for small configurations.

## Decision

Kani is adopted for formal verification of HNSW structural invariants and dense
SIMD boundary contracts, starting with the bidirectional links invariant on
bounded graph configurations and the selector and tail-padding invariants.

### Implementation Approach

1. **Conditional Compilation**: Harnesses use `#[cfg(kani)]` to prevent any
   interference with normal builds or test execution

2. **Module Location**: Harnesses reside beside the code they verify, such as
   `chutoro-core/src/hnsw/kani_proofs/` and
   `chutoro-providers/dense/src/simd/kani_proofs.rs`, enabling access to
   internal types via `pub(crate)` visibility without exposing them publicly

3. **Bounded Verification**: Two tiers are maintained: a practical 2-node
   smoke/reconciliation harness and small dense SIMD boundary harnesses for
   quick feedback, and broader harnesses for slower coverage (run via
   `make kani-full`)

4. **Makefile Integration**: `make kani` runs the practical harnesses, while
   `make kani-full` runs the full suite

### Complementary Testing Strategy

The verification strategy is intentionally split across three tools:

- `Proptest` and differential tests provide probabilistic sampling over
  realistic integrations, backend parity checks, and external-format
  behaviours. They scale to dozens or thousands of items, produce shrunk
  counterexamples, and belong in every pull request (PR) path.
- `Kani` provides exhaustive bounded exploration for executable Rust with tight
  state limits. It is the best fit for unsafe wrappers, dispatch selectors,
  parser-like logic, and bounded state machines, and it belongs in the nightly
  or other slow-lane verification path.
- `Verus` provides deductive proofs once logic has been extracted into pure
  helpers or explicit state transitions. It is the best fit for functional
  invariants, identity matching, refresh-policy gates, and remap proofs where
  bounded symbolic execution becomes unwieldy.

Property-based and differential testing remain the primary tools for realistic
configurations, backend parity, and behaviour dominated by external systems or
device code. Kani is reserved for bounded executable Rust, especially unsafe
wrapper layers, dispatch selectors, parser-like checkpoint envelopes, and
host-side state machines. Verus is the right tool once a helper has been
factored into a pure, specification-friendly function or transition system,
such as refresh-policy gating, stable-identity matching, or compaction remap
logic.

Kani should not be used as the first tool for concurrency-heavy code or raw GPU
kernels. Its own limitations around concurrency support mean those areas are
better covered by mocked property suites and CPU-versus-device differential
tests, with any proof effort focused on the pure host planner around them.

## Consequences

### Positive

- **Formal proofs (bounded)**: The 3-node harness is intended to verify all 64
  possible edge configurations, but it still times out in this environment (see
  Findings)
- **Bug discovery**: Exhaustive exploration may find edge cases missed by
  random sampling, particularly subtle interactions between edge operations
- **Executable specification**: Kani harnesses serve as machine-verified
  documentation of what invariants must hold
- **Confidence multiplier**: Proven invariants for small cases reduce the
  debugging burden when proptest finds issues in larger configurations

### Negative

- **Tool overhead**: Developers need Kani installed to run `make kani`;
  however, this is optional and not required for normal development
- **Verification time**: Bounded checking takes minutes rather than seconds,
  making it unsuitable for tight feedback loops
- **Scalability ceiling**: Verification is tractable only for small bounds
  due to combinatorial explosion; this limits what can be formally proven
- **Learning curve**: Kani's symbolic model and harness patterns differ from
  runtime testing, requiring developer education

### Neutral

- **CI integration deferred**: Initially manual invocation only; CI integration
  planned for a future phase once harness stability is validated
- **Scope limited**: Focus on the bidirectional links invariant on bounded graph
  configurations and dense SIMD boundary contracts (including selector and
  tail-padding invariants); other invariants follow validating the approach

## Findings to Date

### Initial Harness Development

The first harness (`verify_bidirectional_links_commit_path_3_nodes`)
demonstrates:

1. **Graph construction works**: `Graph::with_capacity` and `attach_node`
   operate correctly under Kani's symbolic execution without modification

2. **Commit-path reconciliation**: `CommitApplicator::apply_neighbour_updates`
   exercises removed-edge reconciliation, reverse-edge insertion, and deferred
   scrubs without requiring a bespoke harness path

3. **Eviction coverage**: Running the harness at level 1 with
   `max_connections = 1` ensures the eviction path is exercised with a bounded,
   3-node setup

4. **Precondition assumptions**: Guarding against invalid node ids, invalid
   levels, and duplicate neighbours keeps the solver focused on production
   inputs while remaining exhaustive within the bounded space

### Design Observations

- The existing `check_bidirectional` function uses `EvaluationMode` and
  `tracing`, adding complexity unsuitable for verification. A simplified,
  self-contained assertion is more appropriate for Kani harnesses.

- The harness structure (setup, nondeterministic population, constraint
  enforcement, invariant assertion) provides a clear, reusable pattern for
  future invariant harnesses.

- Separating edge population from constraint enforcement mirrors the actual
  HNSW insertion flow and makes the harness easier to understand.

### 2025-12-27: Kani Setup and Reconciliation Harness

- `cargo install --locked kani-verifier` now succeeds on Rust 1.88.0 and
  installs Kani v0.66.0 (which downloads `nightly-2025-11-05` for verification
  runs).

- Kani emits warnings about unsupported constructs (`caller_location`,
  `foreign function`) and about treating concurrency primitives (atomics,
  thread-locals) as sequential. These are only problematic if the relevant code
  is reachable by the harness.

- `make kani` now runs the smoke harness and the 2-node reconciliation harness.
  In this environment, the smoke harness completes in ~96 seconds and the
  2-node reconciliation harness completes in ~14 seconds (total ~2m 14s,
  excluding compilation).

- The previous exhaustive 3-node harness did not complete within 10 minutes.
  It has since been replaced by the commit-path harness:

    cargo kani -p chutoro-core --default-unwind 10 \
      --harness verify_bidirectional_links_commit_path_3_nodes

  This harness uses tighter assumptions and deterministic setup to keep Kani
  practical while still exercising eviction and deferred scrubs.

- The reconciliation coverage is split into a practical 2-node harness that
  calls `ensure_reverse_edge_for_kani` (wrapping
  `EdgeReconciler::ensure_reverse_edge`) and a heavier 3-node harness that calls
  `apply_reconciled_update_for_kani` (which exercises removed-edge
  reconciliation, added-edge reconciliation, and deferred scrubs).
- A targeted mutation test that skips inserting the reverse edge causes the
  2-node reconciliation harness to fail with "bidirectional invariant violated
  after reconciliation", confirming that the harness is sensitive to
  reciprocity defects.

### Audit Recommendations (Bidirectionality Verification)

- Replace the 3-node bidirectionality harness that calls
  `enforce_bidirectional_constraint` with a harness that drives production
  commit paths (for example, `CommitApplicator::apply_neighbour_updates` or
  `EdgeReconciler::reconcile_*`). The current 3-node harness inserts reverse
  edges itself, so it cannot detect missing reciprocity in production code.
- Add a focused eviction/deferred-scrub scenario: pre-fill a target neighbour
  list to its `max_connections`, force `ensure_reverse_edge` to evict, and then
  assert reciprocity after `apply_deferred_scrubs`. This exercises the
  correctness of eviction plus scrub logic directly.
- Add explicit `kani::assume` guards that mirror production preconditions
  (in-range node ids, deduped neighbour lists) to reduce state explosion while
  still covering legitimate inputs. Document these assumptions in the harness
  docs so they are not mistaken for proof obligations.
- Keep `make kani` as the practical local gate and reserve `make kani-full` for
  a nightly "slow" CI run once it is stable; do not add Kani to normal test
  runs.

### 2026-01-03: Commit-Path Harness Replacement

- Replaced the 3-node bidirectionality harness with a commit-path harness that
  calls `CommitApplicator::apply_neighbour_updates` and
  `CommitApplicator::apply_new_node_neighbours`, ensuring deferred scrubs are
  exercised in the formal model.
- Moved the harness to level 1 with `max_connections = 1` so eviction occurs
  in a 3-node graph without increasing the node bound.
- Added explicit `kani::assume` preconditions in the commit-path helper to
  align with production invariants and keep Kani runtimes manageable.

### 2026-08-24: Tractability Restructure and CI Gating

- Hand-tracing the three-node reconciliation harness exposed a production
  ordering defect (healing clobbered by the origin's write-back); the fix and
  the retirement of the intractable deterministic harnesses are recorded in
  [kani-full-hnsw-hypothesis-testing.md](../kani-full-hnsw-hypothesis-testing.md)
  and the developers' guide "Kani CI policy" section.
- The commit-path, three-node reconciliation, and eviction-scrub harnesses
  exceeded practical CBMC budgets and are replaced by exact unit-test twins;
  the no-self-loop and neighbour-uniqueness invariants are now proved on the
  `ensure_reverse_edge` surface with per-level two-node harnesses.
- `make kani` gates pull requests via the path-filtered `kani-pr.yml`
  workflow, and `make kani-full` (17 harnesses, roughly 18 minutes) runs
  nightly post-merge.
- MST proofs verify a bounded `cfg(kani)` sequential model of parallel
  Kruskal (at most four nodes, six canonical edges, fixed-size forest
  representation) rather than the Rayon production path; the modelling boundary
  is closed by exhaustive equivalence tests in
  `chutoro-core/src/mst/tests/kani_model_equivalence.rs`, and out-of-domain
  inputs fail loudly with invariant violations.

### 2026-11-21: Base-Layer Connectivity Healing Proof (Issue 255)

- A direct Kani harness now reaches the production
  `ConnectivityHealer::ensure_base_connectivity` through the `#[cfg(kani)]`
  bridge `ensure_base_connectivity_for_kani` in
  `chutoro-core/src/hnsw/insert/mod.rs`. The harness
  `verify_base_layer_healing_links_isolated_node_2_nodes` builds two level-0
  nodes with fresh lists, heals node 1 with a base-layer limit of two, and
  asserts the exact post-state: node 0 lists `[1]`, node 1 lists `[0]`, and the
  entry point is unchanged.
- The harness verifies in 9.57 s on Kani 0.67.0 with rustc 1.99.1 (24 CPUs,
  125 GiB RAM): 0 of 1240 checks failed, both `kani::cover!` properties were
  satisfied, and the trace contains zero aborted paths. The only unwind failure
  at `#[kani::unwind(4)]` was `memcmp.unwind.0`; `unwind(9)` covers the
  eight-byte slice comparison.
- The earlier direct harness's 2,590-path, twenty-minute abort is not inherent
  to the healer. A three-node probe that also added one symbolic boolean
  (`kani::any::<bool>()`) for an optional spare neighbour and raised the unwind
  bound exceeded 600 s with more than 6,400 aborted paths before being stopped.
  Because it changed three factors together, it is consistent with the recorded
  two-to-three-node cliff but does not isolate which factor dominated. The
  concrete two-node rung is the retained proof.
- The proof is non-vacuous. Temporarily skipping the reverse directed link in
  `link_new_node_inner` makes the harness fail with
  `healing must link the isolated node back to the entry node`; temporarily
  pre-seeding the reciprocal edge makes it fail with
  `healing precondition: both base lists must start empty`. Both mutations were
  reverted.
- The normal healing tests and the `LinearVisitedSet` equivalence tests remain.
  The proof covers link creation on fresh lists only; eviction cascades,
  re-healing of evicted nodes, and global connectivity remain test-covered.
- Final gate results on the branch: `RUSTFLAGS="-D warnings" make kani`
  verified 7 of 7 harnesses (2262.5 s wall, exit 0), and
  `RUSTFLAGS="-D warnings" make kani-full` verified 18 of 18 `chutoro-core`
  harnesses and both `chutoro-providers-dense` harnesses (2523.6 s wall, exit
  0). The healing harness verified in 8.42 s and 7.71 s respectively, with both
  covers satisfied in each tier.

### Verification Targets (Next Invariants)

The following invariants are explicitly defined to avoid ambiguity, and are
intended as future formal verification targets:

1. **No self-loops**: For every node `u` and layer `l`, `u` is not in
   `N_l(u)`.
2. **Neighbour list uniqueness**: For every node `u` and layer `l`, the list
   `N_l(u)` contains no duplicates (set semantics).
3. **Entry-point validity and maximality**: If the graph contains at least one
   node, the entry point is set to an existing node, and its level equals the
   maximum level present in the graph (for all nodes `v`,
   `level(entry) >= level(v)`).
4. **MST structural correctness (CPU/GPU)**: For any connected candidate graph
   on `n` nodes, the MST output has exactly `n-1` edges, is acyclic, and
   connects all nodes. For disconnected graphs with `c` components, the output
   is a forest with `n-c` edges.
5. **Distance kernel consistency (CPU/GPU)**: For any metric distance
   implementation, distances are symmetric and zero on identical inputs, and
   CPU and GPU implementations agree within a defined tolerance `epsilon` for
   the same inputs.

## Next Steps

1. **Validate harness**: Run `cargo kani` and verify successful completion
   without counterexamples.

2. **Commit-path harnesses**: Replace the 3-node bidirectionality harness with
   a commit-path harness that drives reconciliation and deferred scrubs, and
   add an eviction-focused scenario.

3. **Expand invariants**: Add harnesses for the explicit targets listed above,
   including no-self-loops, neighbour uniqueness, entry-point maximality, MST
   structural correctness, and distance kernel consistency.

4. **Target bounded state machines**: Add harnesses for executable Rust logic
   that fits Kani's strengths, including SIMD dispatch selectors, checkpoint
   envelope parsers, and plugin-wrapper lifecycle transitions.

5. **Increase bounds**: Test 4-node and 2-layer configurations to explore more
   complex interaction patterns once the commit-path harnesses are stable.

6. **CI integration**: Add a nightly "slow" Kani verification job that runs
   `make kani-full` only when main has new commits that day.

7. **Performance tuning**: Investigate C Bounded Model Checker (CBMC)
   timeouts, adjust bounds, and consider smaller harnesses or reduced
   nondeterminism to keep Kani runs practical.

8. **Coordinate with Verus**: When a bounded harness keeps growing due to
   control-flow complexity, extract the corresponding pure helper and move the
   proof obligation to Verus instead of forcing it through symbolic execution.

9. **Documentation**: Record verification results, any discovered issues, and
   update this ADR with findings.

## Change Control

- 2025-12-27: Installed kani-verifier v0.66.0 (Rust 1.88), added practical
  smoke + 2-node reconciliation harnesses, introduced `kani-full` for the
  heavier 3-node runs, and updated findings with current timings/timeouts.
- 2025-12-27: Recorded audit recommendations, added explicit verification
  targets (five additional invariants), and clarified the plan for a nightly
  "slow" Kani CI job while keeping normal test runs unchanged.
- 2026-08-24: Retired harnesses past the CBMC tractability cliff in favour of
  unit-test twins and narrow-surface proofs, fixed the reconciliation
  write-back ordering defect the investigation exposed, and wired `make kani`
  into pull-request CI with `make kani-full` remaining the nightly tier.
- 2026-08-24: Documented the MST Kani model boundary and its equivalence-test
  closure in the design document and this ADR.
- 2026-11-19: Recorded the concurrency boundary explicitly. The HNSW harnesses
  and the graph-structure obligations FV04 and FV08 model graph state —
  adjacency, reciprocity, levels, and the commit invariants over them — not
  scheduling, lock acquisition, or lock waits, so they establish nothing about
  deadlock freedom or about which thread runs a callback. Liveness obligations
  stay separate from graph-structure obligations. Kani treats concurrency
  primitives as sequential, and the insertion critical-section contract
  therefore stays with the workflow contract and the runtime regressions
  described in [ADR-006](adr-0006-sequential-trim-scoring.md). A future change
  to scheduling beyond that critical section must be accompanied by a bounded
  scheduler or lock model plus production-path regressions, not by extending
  these structural harnesses.
- 2026-11-21: Proved base-layer healing link creation on a concrete two-node
  graph (issue 255), added the `ensure_base_connectivity_for_kani` bridge and
  the fast-tier harness, and recorded the three-node probe measured while
  widening it.

## Amendments

- 2026-11-19: recorded that the HNSW harnesses and obligations FV04 and FV08
  model graph state rather than scheduling, lock acquisition, or lock waits,
  and that scheduling beyond the insertion critical section needs a bounded
  scheduler or lock model plus production-path regressions.

## References

- [Kani Rust Verifier](https://github.com/model-checking/kani)
- [Kani Documentation](https://model-checking.github.io/kani/)
- [Property Testing Design](../property-testing-design.md)
- [ADR-001: Commit Post-processing](./adr-0001-commit-post-processing.md)
- [Execution Plan: Phase 1](../execplans/phase-1-hnsw-graph-invariant-kani-harness.md)
