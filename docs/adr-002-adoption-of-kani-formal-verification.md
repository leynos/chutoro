# Architecture Decision Record (ADR) 002: Adoption of Kani Formal Verification

## Status

Accepted

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

Kani is adopted for formal verification of HNSW structural invariants, starting
with the bidirectional links invariant on bounded graph configurations.

### Implementation Approach

1. **Conditional Compilation**: Harnesses use `#[cfg(kani)]` to prevent any
   interference with normal builds or test execution

2. **Module Location**: Harnesses reside in
   `chutoro-core/src/hnsw/kani_proofs.rs`, enabling access to internal types
   via `pub(crate)` visibility without exposing them publicly

3. **Bounded Verification**: Two tiers are maintained: a practical 2-node
   smoke/reconciliation harness for quick feedback, and a 3-node exhaustive
   harness for broader coverage (run via `make kani-full`)

4. **Makefile Integration**: `make kani` runs the practical harnesses, while
   `make kani-full` runs the full suite

### Complementary Testing Strategy

The two approaches serve different purposes and complement each other:

| Aspect                                  | Proptest                      | Kani                 |
| --------------------------------------- | ----------------------------- | -------------------- |
| Coverage model                          | Probabilistic sampling        | Exhaustive (bounded) |
| Typical scale                           | Hundreds of nodes             | 3-10 nodes           |
| Execution time                          | Seconds                       | Minutes              |
| Failure output                          | Minimal shrunk counterexample | Full execution trace |
| Primary use case                        | Regression catching           | Invariant proofs     |
| Continuous Integration (CI) integration | Every pull request (PR)       | Nightly (planned)    |

Proptest remains the primary testing tool for catching regressions on realistic
configurations. Kani provides formal guarantees for small configurations,
increasing confidence that core invariant logic is sound before proptest
explores larger state spaces.

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
- **Scope limited**: Only bidirectional invariant initially; other invariants
  follow after validating the approach

## Findings to Date

### Initial Harness Development

The first harness (`verify_bidirectional_links_3_nodes_1_layer`) demonstrates:

1. **Graph construction works**: `Graph::with_capacity` and `attach_node`
   operate correctly under Kani's symbolic execution without modification

2. **Nondeterministic edges**: `kani::any::<bool>()` effectively generates all
   edge combinations; Kani explores all 2^6 = 64 configurations

3. **Unwind bounds**: `#[kani::unwind(10)]` provides sufficient headroom for
   3-node iteration with safety margin

4. **Vec operations supported**: `Vec::push`, `Vec::contains`, and iteration
   work under Kani; more complex operations may require bounded alternatives

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
  In this environment the smoke harness completes in ~96 seconds and the 2-node
  reconciliation harness completes in ~14 seconds (total ~2m 14s, excluding
  compilation).

- The 3-node exhaustive harness still does not complete within 10 minutes
  (`cargo kani -p chutoro-core --default-unwind 10 --harness \
  verify_bidirectional_links_3_nodes_1_layer
  `), so the 64-configuration proof remains aspirational for now.

- The reconciliation coverage is split into a practical 2-node harness that
  calls `ensure_reverse_edge_for_kani` (wrapping
  `EdgeReconciler::ensure_reverse_edge`) and a heavier 3-node harness that
  calls `apply_reconciled_update_for_kani` (which exercises removed-edge
  reconciliation, added-edge reconciliation, and deferred scrubs).

## Next Steps

1. **Validate harness**: Run `cargo kani` and verify successful completion
   without counterexamples

2. **Expand coverage**: Add harnesses for degree bounds and layer consistency
   invariants following the established pattern

3. **Increase bounds**: Test 4-node and 2-layer configurations to explore
   more complex interaction patterns

4. **CI integration**: Add nightly Kani verification job to GitHub Actions
   once harness suite stabilizes

5. **Performance tuning**: Investigate CBMC timeouts, adjust bounds, and
   consider smaller harnesses or reduced nondeterminism to keep Kani runs
   practical.

6. **Documentation**: Record verification results, any discovered issues, and
   update this ADR with findings

## Change Control

- 2025-12-27: Installed kani-verifier v0.66.0 (Rust 1.88), added practical
  smoke + 2-node reconciliation harnesses, introduced `kani-full` for the
  heavier 3-node runs, and updated findings with current timings/timeouts.

## References

- [Kani Rust Verifier](https://github.com/model-checking/kani)
- [Kani Documentation](https://model-checking.github.io/kani/)
- [Property Testing Design](./property-testing-design.md)
- [ADR-001: Commit Post-processing](./adr-001-commit-post-processing.md)
- [Execution Plan: Phase 1](./execplans/phase-1-hnsw-graph-invariant-kani-harness.md)
