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
- 3-node, 1-layer graph configuration
- Makefile integration (`make kani`)
- ADR documenting Kani adoption rationale

### Out of Scope (Future Phases)

- Multi-layer verification
- Larger graph bounds (4+ nodes)
- Other invariants (degree bounds, reachability, layer consistency)
- CI integration (deferred to Phase 2)

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
- `verify_bidirectional_links_3_nodes_1_layer` harness with:
  - `#[kani::proof]` attribute
  - `#[kani::unwind(10)]` for loop bounds
  - Graph setup with 3 nodes at level 0
  - Nondeterministic edge population via `kani::any::<bool>()`
  - Bidirectional enforcement simulating insertion behaviour
  - Invariant assertion using `kani::assert`
- Helper functions: `populate_edges_nondeterministically`,
  `enforce_bidirectional_constraint`, `assert_bidirectional_invariant`

### 3. Update HNSW Module

- [x] Add conditional inclusion in `chutoro-core/src/hnsw/mod.rs`:

```rust
#[cfg(kani)]
mod kani_proofs;
```

### 4. Add Makefile Target

- [x] Add `kani` target to `/root/repo/Makefile`:

```makefile
kani: ## Run Kani formal verification harnesses
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
cargo kani -p chutoro-core --harness verify_bidirectional_links_3_nodes_1_layer
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
| Long verification times      | Medium     | Low    | Use small bounds (3 nodes)   |
| Vec operations unsupported   | Low        | Medium | Use bounded arrays if needed |
| Developers without Kani      | Medium     | Low    | Keep `make kani` optional    |

## Future Work (Phase 2+)

1. **Expand graph bounds**: Verify 4-node and multi-layer configurations
2. **Additional invariants**: Add harnesses for degree bounds, layer
   consistency, and reachability
3. **CI integration**: Add nightly Kani verification job to GitHub Actions
4. **Performance**: Optimise harness for faster verification cycles
5. **Documentation**: Record verification results and discovered issues

## References

- [Kani Rust Verifier](https://github.com/model-checking/kani)
- [Kani Documentation](https://model-checking.github.io/kani/)
- [Property Testing Design](../property-testing-design.md)
- [ADR-002: Adoption of Kani](../adr-002-adoption-of-kani-formal-verification.md)
