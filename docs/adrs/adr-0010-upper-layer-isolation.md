# Architectural decision record (ADR) 010: upper-layer isolation policy for HNSW insertion reconciliation

## Status

Accepted - 2026-11-24: base-layer reachability is the normative connectivity
requirement; upper-layer isolation after an eviction is permitted and is not
healed by insertion reconciliation.

## Date

2026-11-24

## Context and problem statement

HNSW connectivity healing is intentionally limited to the base layer.
`ConnectivityHealer::ensure_base_connectivity` repairs a node only when
insertion reconciliation reports that its level-0 neighbour list became empty,
either because a removed reverse edge isolated it or because a deferred scrub
removed an orphaned forward edge. The same situations at level 1 and above are
accepted without repair. An evicted node can therefore hold an empty neighbour
list at level 1 or above for the rest of the index's life, until a later
insertion happens to select it as a neighbour again.

That behaviour reduces recall on some graphs, but it does not remove base-layer
reachability, because search descends from the entry point through the upper
layers and then performs best-first search at level 0. The repository had no
explicit statement that this asymmetry was intended, so the code, the design
documents and the property checks could drift apart without anything
identifying the change as a policy decision. Issue #233 records the question
and the source review comment on PR #227 records the original observation.

The decision is not about whether upper-layer isolation is possible. It is
about whether the repository treats it as a defect to repair or as an accepted
exclusion from the connectivity invariant.

## Decision drivers

- Level-0 best-first search produces the result set; upper layers only guide
  descent, so base-layer reachability is the correctness requirement.
- Upper-layer degree bounds are deliberately small, so eviction is routine and
  any repair policy would run frequently.
- Healing upper layers would add cross-layer repair logic, more expensive
  reconciliation, and new failure modes for a link that is an optimization.
- The documented invariant must match what the implementation actually
  enforces; overstating the guarantee is worse than recording the exclusion.
- Property tests and focused tests need a stated policy so that an empty
  upper-layer list is accepted deliberately rather than tolerated by accident.

## Options considered

### Option A: document upper-layer isolation as a deliberate invariant exclusion

Accepted. The policy states that base-layer reachability and reciprocity are
required, that upper-layer reciprocity and degree bounds remain required, and
that an empty upper-layer list after eviction is permitted. The recall
trade-off and its mitigation through `ef` are documented, and the remaining
enforcement gaps are recorded rather than hidden.

### Option B: implement upper-layer healing

Rejected. It would require new invariants for upper-layer connectivity, changes
to removed-edge reconciliation and deferred scrubs at every level, and focused
tests for each repair path. Those changes would run on every trim eviction,
increase churn in short upper-layer lists, and risk repeated eviction cycles
between nodes competing for the same small degree budget. No demonstrated
defect requires the stronger guarantee, and the base-layer search can still
reach an upper-isolated node when `ef` is large enough.

| Topic                   | Option A                                  | Option B                                     |
| ----------------------- | ----------------------------------------- | -------------------------------------------- |
| Base-layer reachability | Required and locally repaired             | Required and locally repaired                |
| Upper-layer isolation   | Permitted after eviction                  | Repaired after every eviction                |
| Reconciliation cost     | Local level-0 repair only                 | Repair at every populated level              |
| Recall risk             | Possible, mitigated by `ef`               | Lower on some graphs                         |
| Implementation risk     | None beyond documenting current behaviour | New churn and eviction cycles in small lists |

_Table 1: Comparison of the documented exclusion and upper-layer healing._

## Decision outcome

Adopt Option A. The normative requirements are:

- Every populated node is reachable from the entry point through level-0 edges.
- Every retained level-0 edge is reciprocated, subject to the level-0 degree
  bound.
- Once the graph holds more than one populated node, insertion reconciliation
  does not leave a populated node with an empty level-0 list. The first
  insertion has no other node to connect to, so the requirement starts at the
  second node.
- Every retained upper-layer edge is reciprocated and stays within the
  level-specific degree bound.
- An empty neighbour list at level 1 or above is permitted after an eviction.

Four causes can empty an upper-layer list, and all are accepted:

- Trimming evicts the last neighbour from an existing node's upper-layer list.
- Removed-edge reconciliation drops a reverse edge and leaves the target empty
  at that layer.
- A deferred scrub removes an orphaned forward edge and leaves its origin empty
  at that layer.
- The commit-time reciprocal fallback finds no candidate that can accept a
  back-link at that layer, so the new node commits without an upper-layer
  neighbour.

The reciprocal-link fallback remains an attempt made at commit time for the new
node. It is not a persistent guarantee: a later insertion can evict the link,
and no later pass restores it above level 0.

The policy is recorded in
[`docs/chutoro-design.md` §6.5.1](../chutoro-design.md#651-layer-specific-connectivity-policy)
and reflected in `docs/property-testing-design.md` §2.2.

## Goals and non-goals

- Goals:
  - State the layer-specific connectivity contract explicitly.
  - Explain why the base layer and upper layers have different requirements.
  - Record the expected recall trade-off and the `ef` mitigation.
  - Align the invariant documentation, focused tests and source comments with
    the policy.
- Non-goals:
  - Changing production reconciliation behaviour.
  - Adding a public API for upper-layer connectivity.
  - Claiming full base-layer enforcement. The global base-layer component gap
    remains open and is tracked by
    [issue #393](https://github.com/leynos/chutoro/issues/393).

## Known risks and limitations

- **Recall may be lower than a fully healed graph.** An upper-isolated node
  cannot act as a descent stepping stone. Increasing `ef` explores more level-0
  candidates and can recover the node, at additional distance evaluations.
- **Isolation is not always permanent.** A later insertion can select the node
  as a neighbour or trimming can rewrite its list, so the graph can recover
  upper-layer connectivity without any repair pass.
- **Base-layer enforcement is local, not global.** `ensure_base_connectivity`
  reacts to an empty level-0 list reported during insertion reconciliation.
  Neither it nor the cross-layer `reachability()` checker detects a base-layer
  split in which every node still has a neighbour inside its own component.
- **The cross-layer checker is not the normative check.** A base-layer split
  bridged by an upper-layer edge passes `check_all()`. The test-only base-layer
  oracle exists to make the normative requirement testable.
- **A future recall regression could reopen the decision.** If benchmarks show
  that upper-layer isolation costs material recall that `ef` cannot recover,
  Option B should be revisited and this ADR superseded rather than silently
  amended.

## Architectural rationale

The base layer is the domain's completeness boundary: it holds every point and
performs the search that answers queries. Upper layers are an index-side
acceleration structure, and their connectivity is a quality-of-service property
rather than a correctness property. Recording that distinction keeps the
invariant honest and lets reconciliation stay local, bounded and cheap. The
test-only oracle keeps the stronger base-layer requirement visible without
turning it into an unverified production claim.

## Consequences

- An empty level-1-or-above list is an accepted state, not an invariant
  violation, and tests may assert it directly.
- The base-layer requirement is stated normatively and checked by a dedicated
  test-only oracle.
- Insertion reconciliation keeps its current behaviour and cost.
- The global base-layer component gap is documented as a divergence and
  remains owned by issue #393.
- ADR 0001's per-layer guarantee wording is amended to a commit-time attempt.

## Validation evidence

- Focused tests build an entry node at level 1 with an empty upper-layer list
  over a reciprocal level-0 chain. All public invariants pass, the upper list
  stays empty, and the base-layer oracle reports no unreachable node.
- A direct search test shows greedy descent remaining at the entry node when
  its upper list is empty, then level-0 best-first search returning the full
  chain in distance order.
- A reverse-edge removal test and a deferred-scrub test assert that an emptied
  upper-layer list does not trigger base-layer healing and that level-0 edges
  remain intact.
- The bootstrap reachability property runs the base-layer oracle after every
  sequential insertion and retains the public `check_all()` assertion.

## References

- [Issue #233: decide and document the HNSW upper-layer isolation policy](https://github.com/leynos/chutoro/issues/233)
- [PR #227 review comment that raised the question](https://github.com/leynos/chutoro/pull/227#issuecomment-5400755118)
- [Issue #393: complete the HNSW invariant oracle](https://github.com/leynos/chutoro/issues/393)
- [Chutoro design §6.5.1](../chutoro-design.md#651-layer-specific-connectivity-policy)
- [Property testing design §2.2](../property-testing-design.md#22-core-structural-invariants-of-a-valid-hnsw-graph)
