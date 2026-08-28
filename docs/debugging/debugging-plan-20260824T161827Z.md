# Debugging plan: benchmark smoke HNSW stall

**Generated**: 2026-08-24T16:18:27Z **Issue ID**: validation blocker during
issue #200 work **Severity**: medium **Falsification sub-agent**: alchemist
**Planning agent boundary**: This document was prepared by the planning agent.
Falsification must be executed by the named sub-agent, not by the planning
agent.

## Problem statement

`cargo test -p chutoro-benches --all-features` passes 139 unit tests, but its
`benchmark_smoke` integration test stalls in the exact HNSW Criterion probe.
The probe should complete a bounded 100-point measurement. After more than
seven minutes, all Rayon workers waited in `CpuHnsw::insert_with_collector`.
The smoke test was terminated after capturing a backtrace. This blocks a green
gate for a manifest-only workspace-lint and filesystem-boundary change.

## Context summary

The observed benchmark-stall context was:

| Aspect              | Details                                                            |
| ------------------- | ------------------------------------------------------------------ |
| First observed      | 2026-08-24 while validating benches lint enrolment                 |
| Reproduction rate   | One standalone `cargo test` run                                    |
| Affected components | `benchmark_smoke`, `benches/hnsw.rs`, `CpuHnsw::build`             |
| Recent changes      | Only bench lint inheritance and Dylint exclusions; no HNSW changes |

### Error artefacts

```plaintext
Benchmarking hnsw_build/n=100,M=8,ef=16: Collecting 10 samples in estimated
527.89 ms (10 iterations)

All 19 HNSW threads waited in futexes. Worker backtraces ended at
CpuHnsw::insert_with_collector; the owner waited in Rayon while
CpuHnsw::build was called from Criterion's iter_batched.
```

### Information gaps

- The prior full workspace gate reported green, but its exact process
  environment and the smoke probe's Rayon worker count were not recorded.
- It is not yet known whether the stall is deterministic or specific to the
  default Rayon pool size in this host environment.

______________________________________________________________________

## Hypotheses

### H1: the default Rayon worker count creates an insertion lock convoy

**Claim**: The exact benchmark stalls only when `CpuHnsw::build` uses the
host-default Rayon pool; a single-worker pool completes the same fixed-seed
probe promptly.

**Plausibility**: High — every captured worker was contending for the insertion
mutex, and the probe's 100-point input does not justify multi-minute work.

**Prediction**: With `RAYON_NUM_THREADS=1`, the exact `hnsw` probe completes
within 30 seconds and prints Criterion timing output.

#### H1 falsification plan

The H1 falsification steps were:

| Step | Action                                                                  | Expected Negative Result                                                        |
| ---- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1    | Run the exact probe with `RAYON_NUM_THREADS=1` and a 30-second timeout. | A timeout or the same worker stall disproves the worker-count claim.            |
| 2    | If step 1 completes, repeat with `RAYON_NUM_THREADS=2`.                 | A prompt two-worker run weakens the claim that only serial execution avoids it. |

**Tooling**: `timeout`, Cargo, and the existing exact benchmark command.

**Confidence on falsification**: High for distinguishing pool-size sensitivity
from a deterministic fixed-seed algorithmic stall.

______________________________________________________________________

### H2: the fixed-seed benchmark input deadlocks regardless of pool size

**Claim**: The generated 100-point source and HNSW parameters hit a
deterministic lock-order deadlock independent of Rayon worker count.

**Plausibility**: Medium — all workers waited in the same insertion path, but
the owner also waited through the full Criterion sample loop.

**Prediction**: The exact probe stalls with both one and two Rayon workers.

#### H2 falsification plan

The H2 falsification steps were:

| Step | Action                                                  | Expected Negative Result                                                          |
| ---- | ------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 1    | Compare the bounded exact probe at one and two workers. | A prompt completion at either worker count disproves a pool-independent deadlock. |

**Tooling**: The same bounded Cargo commands as H1.

**Confidence on falsification**: High once H1's two runs are complete.

______________________________________________________________________

## Recommended execution order

1. **H1** — it is the cheapest decisive experiment and may identify a bounded
   smoke-fixture configuration fix.
2. **H2** — it follows directly only if the one-worker run stalls.

## Termination criteria

- **Root cause identified**: One worker count completes while another stalls,
  or every tested worker count stalls.
- **Escalation trigger**: Both bounded runs time out or show a different stack
  signature; revise the hypotheses before modifying production code.

## Notes for executing agent

Run only the two stated experiments. Do not edit tracked files, run full
repository gates, or terminate unrelated processes. Return `falsified`,
`not-falsified`, or `inconclusive` for each hypothesis, with exact command
results and timing evidence.

## Falsification record

- 2026-08-24: The initial one- and two-worker 30-second probes were
  inconclusive. Both timed out during Cargo compilation before Criterion
  emitted the benchmark label, so neither result exercised `CpuHnsw::build`.
  Logs: `/tmp/benchmark-smoke-h1-rayon1.out` and
  `/tmp/benchmark-smoke-h2-rayon2.out`.
- 2026-08-24: After the release artefact was warm, the one-worker probe reached
  Criterion collection but timed out at 30 seconds. The two-worker probe
  completed in 12 seconds with an observed interval of 110.93–195.74 ms. This
  falsifies both the one-worker-completes and all-worker-counts-stall claims.
  The smoke fixture now gives only its spawned benchmark commands two workers;
  it still exercises the same fixed-seed benchmark label and Criterion path.
