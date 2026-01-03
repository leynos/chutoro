# Execution Plan: Phase 1 - HNSW Graph Invariant Kani Harness

## Objective

Add Kani formal verification harnesses to the chutoro library to exhaustively
verify the HNSW graph bidirectional (symmetry) invariant for bounded graph
configurations.

## Background

The HNSW implementation relies on bidirectional links: for every edge
`(source, target, level)`, there must exist a corresponding edge
`(target, source, level)`. This invariant is currently validated via
property-based testing with proptest. Kani provides bounded model checking to
exhaustively verify this invariant for small graph configurations, providing
formal proofs rather than probabilistic coverage.

### Why Kani?

| Aspect         | Proptest                        | Kani                 |
| -------------- | ------------------------------- | -------------------- |
| Coverage       | Probabilistic (random sampling) | Exhaustive (bounded) |
| Scale          | Hundreds of nodes               | 3-10 nodes           |
| Speed          | Seconds                         | Minutes              |
| Failure output | Shrunk counterexample           | Execution trace      |
| Guarantee      | High confidence                 | Formal proof         |

## Scope

### In Scope

- Kani harness for bidirectional link invariant
- 3-node, 2-level commit-path configuration
- Makefile integration (`make kani`)
- Architecture Decision Record (ADR) documenting Kani adoption rationale

### Out of Scope (Future Phases)

- Multi-layer verification
- Larger graph bounds (4+ nodes)
- Other invariants (degree bounds, reachability, layer consistency)
- Continuous Integration (CI) integration (deferred to Phase 2)

## Implementation Checklist

### 1. Configure Cargo.toml

- [x] Add `cfg(kani)` to the allowed config list in `chutoro-core/Cargo.toml`

```toml
[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(kani)'] }
```

### 2. Create Kani Harness Module

- [x] Create `chutoro-core/src/hnsw/kani_proofs.rs`

Contents:

- Module-level `//!` documentation explaining purpose
- `verify_bidirectional_links_smoke_2_nodes_1_layer` harness with:
  - Deterministic 2-node setup to validate the Kani toolchain
- `verify_bidirectional_links_commit_path_3_nodes` harness with:
  - `#[kani::proof]` attribute
  - `#[kani::unwind(10)]` for loop bounds
  - Graph setup with 3 nodes at level 1 to allow eviction at capacity 1
  - Commit-path reconciliation via `CommitApplicator::apply_neighbour_updates`
  - Deferred scrub scenario to validate eviction cleanup
  - Invariant assertion using `kani::assert`
- `verify_bidirectional_links_reconciliation_2_nodes_1_layer` harness with:
  - Production reconciliation path via `ensure_reverse_edge_for_kani`
- `verify_bidirectional_links_reconciliation_3_nodes_1_layer` harness with:
  - Heavier reconciliation path via `apply_reconciled_update_for_kani`
- Helper functions: `add_bidirectional_edge`, `add_edge_if_missing`, and
  `push_if_absent`

### 3. Update HNSW Module

- [x] Add conditional inclusion in `chutoro-core/src/hnsw/mod.rs`:

```rust
#[cfg(kani)]
mod kani_proofs;
```

### 4. Add Makefile Target

- [x] Add `kani` and `kani-full` targets to `/root/repo/Makefile`:

```makefile
kani: ## Run Kani practical harnesses (smoke + 2-node reconciliation)
    cargo kani -p chutoro-core --default-unwind 4 \
        --harness verify_bidirectional_links_smoke_2_nodes_1_layer
    cargo kani -p chutoro-core --default-unwind 4 \
        --harness verify_bidirectional_links_reconciliation_2_nodes_1_layer

kani-full: ## Run all Kani formal verification harnesses
    cargo kani -p chutoro-core --default-unwind 10
```

### 5. Create Documentation

- [x] Create this execution plan (`docs/execplans/phase-1-...`)
- [x] Create ADR-002 (`docs/adr-002-adoption-of-kani-formal-verification.md`)

## Verification

After implementation, verify with:

```bash
# Verify normal build unaffected
make build
make test
make lint

# Run Kani verification (requires Kani installation)
make kani

# Or run specific harness:
cargo kani -p chutoro-core --harness verify_bidirectional_links_commit_path_3_nodes

# Or run the full suite:
make kani-full
```

## Success Criteria

- [x] Harness exists and is conditionally compiled under `cfg(kani)`
- [x] `make kani` target invokes Kani verification
- [x] Normal `make build` and `make test` pass (harness does not affect them)
- [x] ADR documents adoption rationale and findings

## Risks and Mitigations

| Risk                         | Likelihood | Impact | Mitigation                   |
| ---------------------------- | ---------- | ------ | ---------------------------- |
| Kani version incompatibility | Low        | Medium | Document tested Kani version |
| Long verification times      | Medium     | Low    | Use small bounds for `kani`  |
| Vec operations unsupported   | Low        | Medium | Use bounded arrays if needed |
| Developers without Kani      | Medium     | Low    | Keep `make kani` optional    |

## Future Work (Phase 2+)

1. **Expand graph bounds**: Verify 4-node and multi-layer configurations
2. **Additional invariants**: Add harnesses for degree bounds, layer
   consistency, and reachability
3. **CI integration**: Add nightly Kani verification job to GitHub Actions
4. **Performance**: Optimize harness for faster verification cycles
5. **Documentation**: Record verification results and discovered issues

## References

- [Kani Rust Verifier](https://github.com/model-checking/kani)
- [Kani Documentation](https://model-checking.github.io/kani/)
- [Property Testing Design](../property-testing-design.md)
- [ADR-002: Adoption of Kani](../adr-002-adoption-of-kani-formal-verification.md)
