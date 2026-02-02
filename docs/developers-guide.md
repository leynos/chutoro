# Developers Guide

This guide collects day-to-day practices for contributors working on the
Chutoro codebase. It complements the more specialised documents in `docs/` and
keeps operational guidance in one place.

## Verus Proofs

We use Verus for formal verification of edge harvest primitives. Run proofs via
`make verus`, which is idempotent and will install the pinned Verus release and
required Rust toolchain as needed.

### Quantifier Trigger Annotations

Verus prints warnings when it selects quantifier triggers automatically. Do not
ignore these warnings. Prefer explicit annotations so the prover behaviour is
stable and predictable:

- Use `#[trigger]` when a specific term should control instantiation.
- Use `#![auto]` only when the automatically chosen trigger is acceptable and
  the quantifier is straightforward.
- Avoid `--triggers-mode silent` in CI because it hides trigger-selection
  changes.

Example:

```rust
assert forall|i: int| #![auto] 0 <= i < edges.len() implies edges[i].source == source_node;
```
