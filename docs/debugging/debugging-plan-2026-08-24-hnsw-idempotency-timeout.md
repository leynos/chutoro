# Debugging Plan: HNSW idempotency property timeout after stack rebase

**Generated**: 2026-08-24
**Issue ID**: Rebase of #223 onto #228
**Severity**: High — blocks the required rebase validation gate
**Falsification sub-agent**: alchemist
**Planning agent boundary**: This document was prepared by the planning agent.
Falsification must be executed by the named sub-agent, not by the planning
agent.

## Problem Statement

After rebasing the environment-reader branch onto the lint-policy stack,
`make test` passed 1,081 tests but timed out
`hnsw_idempotency_preserved_proptest` at its 600-second nextest limit. The same
branch passed the complete test gate before the stack rebase. The expected
behaviour is that the idempotency property completes within the configured
timeout without weakening its workload or timeout.

## Context Summary

| Aspect              | Details                                                           |
| ------------------- | ----------------------------------------------------------------- |
| First observed      | Rebased commit `2342187` on 2026-08-24                            |
| Reproduction rate   | One complete workspace run; isolated run not yet attempted        |
| Affected components | HNSW idempotency property test and nextest scheduling             |
| Recent changes      | Rebase onto #228; `is_coverage_job` now reads via `mockable::Env` |

### Error Artefacts

```plaintext
TIMEOUT [600.004s] chutoro-core
hnsw::tests::property::tests::hnsw_idempotency_preserved_proptest
Summary 1082 tests run: 1081 passed, 1 timed out, 1 skipped
```

### Information Gaps

- The isolated runtime of the affected property test is not yet known.
- A one-run timeout cannot distinguish shared-machine contention from a
  regression in the test's effective workload.

______________________________________________________________________

## Hypotheses

### H1: Workspace concurrency caused the timeout

**Claim**: The property test remains within its budget when executed alone; the
full workspace run's concurrent workload caused the 600-second timeout.

**Plausibility**: Medium — the rebased HNSW idempotency runner still caps its
case count at 16 and disables per-case forking, while the failure occurred near
the end of a concurrent workspace run.

**Prediction**: An isolated nextest invocation of the exact property test
completes successfully within 600 seconds.

#### H1 Falsification Plan

| Step | Action                                                                                                                    | Expected Negative Result                                               |
| ---- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1    | Run `cargo nextest run -p chutoro-core hnsw_idempotency_preserved_proptest` once, without changing environment variables. | A timeout or failure disproves contention as a sufficient explanation. |

**Tooling**: Cargo Nextest, existing shared Cargo cache, and a `/tmp` log.

**Confidence on falsification**: High — the command removes other workspace
tests while retaining the same compiled test and nextest timeout policy.

______________________________________________________________________

### H2: Shared full-suite load delays edge-harvest sorting coverage

**Claim**: The post-rebase timeout in
`build_with_edges_edges_sorted_by_sequence` is caused by shared machine load,
not by a change to edge-harvest behaviour.

**Plausibility**: Medium — it was the final test after 1,095 successful tests,
and the configuration explicitly permits 180 seconds for this heavy test under
parallel load.

**Prediction**: The exact selector completes within 180 seconds when it is the
only Nextest workload.

#### H2 Falsification Plan

| Step | Action                                                                 | Expected negative result                                               |
| ---- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1    | Run the exact Nextest selector without changing environment variables. | A timeout or failure disproves contention as a sufficient explanation. |

**Tooling**: Cargo Nextest and the existing shared Cargo cache.

**Confidence on falsification**: High — the experiment retains the same test
and timeout policy while removing other workspace tests.

______________________________________________________________________

## Recommended Execution Order

1. **H1** — It is the smallest decisive experiment and does not alter test
   configuration, code, or process-global environment state.
2. **H2** — It tests the only remaining non-lint full-gate failure without
   weakening the edge-harvest coverage workload.

## Termination Criteria

- **Root cause identified**: The isolated test either completes, isolating
  shared-run contention, or times out, falsifying H1 and requiring a revised
  plan for the property workload.
- **Escalation trigger**: If the isolated command times out or fails, stop and
  revise the hypothesis plan before altering implementation or test budgets.

## Notes for Executing Agent

Run only the exact command in H1 and return a verdict of falsified,
not-falsified, or inconclusive with the log path and elapsed time. Do not edit
tracked files, modify environment variables, run the full workspace gate, or
change nextest timeouts.

## Recorded Result

H1 was not falsified on 2026-08-24. The isolated command passed in 0.213
seconds (13.886 seconds including compilation), so the prior 600-second timeout
is consistent with shared full-suite contention rather than a regression in the
rebased HNSW idempotency property.

H2 is pending delegated falsification on 2026-08-28. The exact command is
`cargo nextest run -p chutoro-core build_with_edges_edges_sorted_by_sequence`.
H2 was not falsified: the isolated test passed in 0.041 seconds (17 seconds
including package-cache overhead), well below the configured 180-second limit.
Evidence:
`/tmp/h2-hnsw-idempotency-timeout-issue-177-221-remove-in-process-env-mutation.out`.
