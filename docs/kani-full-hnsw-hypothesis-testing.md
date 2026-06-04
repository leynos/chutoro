# Debugging Plan: `make kani-full` HNSW Non-Completion

**Generated**: 2026-06-03  
**Issue ID**: 2.2.7 follow-up verification  
**Severity**: Medium

## Problem Statement

`make kani-full` does not currently complete within the available command
budget on this branch. The expected behaviour is that the full Kani tier either
verifies all harnesses or fails with a concrete counter-example. The observed
run was stopped while the Bounded Model Checker for C (CBMC) was checking the HNSW
`verify_entry_point_validity_4_nodes` harness and repeatedly unwinding Rust
`core::str` panic/slice-error formatting paths.

## Context Summary

- **First observed**: 2026-06-03 on branch
  `2-2-7-kani-harnesses-for-tail-padding-and-dispatch-selection`.
- **Reproduction rate**: one full-run observation, followed by targeted
  falsification runs.
- **Affected components**: `chutoro-core` HNSW Kani full tier.
- **Recent changes**: dense SIMD Kani harnesses and lane-output property
  coverage.

## Error Artefacts

The stopped `make kani-full` run wrote
`/tmp/kani-full-chutoro-2-2-7-status.out`. Relevant log patterns include:

```text
verify_entry_point_validity_4_nodes
core::str::slice_error_fail
core::str::slice_error_fail_rt
core::str::<impl str>::floor_char_boundary
```

The run had already completed several `chutoro-core` distance harnesses with
`VERIFICATION:- SUCCESSFUL` before reaching the HNSW entry-point harness.

## Information Gaps

- Whether a single-harness run of `verify_entry_point_validity_4_nodes`
  reproduces the same string-unwinding pattern.
- Whether the path is driven by panic-capable construction calls such as
  `.expect(...)`, or by the `is_entry_point_valid` predicate itself.
- Whether the dense SIMD 2.2.7 harnesses still pass independently during the
  same diagnostic window.
- Whether the observed non-completion is deterministic proof-state growth or
  transient environmental contention.

## Hypotheses

### H1: The HNSW Entry-Point Harness Is the Reproducer

**Claim**: `make kani-full` is blocked specifically by
`verify_entry_point_validity_4_nodes`, not merely by aggregate suite scheduling.

**Plausibility**: High. The full-run process inspection identified CBMC running
that harness when the command was stopped.

**Prediction**: If this hypothesis holds, running only the harness will either
time out or emit the same `core::str` string/panic path patterns.

#### H1 Falsification Plan

1. Run the single HNSW entry-point harness with a bounded timeout:

   ```bash
   timeout 300s cargo kani -p chutoro-core --default-unwind 10 \
     --harness verify_entry_point_validity_4_nodes \
     2>&1 | tee /tmp/kani-h1-entry-point-validity.out
   ```

   Expected negative result: the harness completes successfully without
   `slice_error_fail` or `floor_char_boundary`.

2. Search the log for the harness and string-unwinding markers:

   ```bash
   rg -n -e "Checking harness" \
     -e "verify_entry_point_validity_4_nodes" \
     -e "slice_error_fail" \
     -e "floor_char_boundary" \
     -e "VERIFICATION:-|FAILED|FAILURE|error:" \
     /tmp/kani-h1-entry-point-validity.out | tail -80
   ```

   Expected negative result: the log lacks the target harness name and
   string-unwinding markers.

**Tooling**: `cargo kani`, `timeout`, `tee`, `rg`.

**Confidence on falsification**: High. A successful single-harness run would
directly disprove this as the local reproducer.

**Execution status**: Not falsified. Subagent Laplace ran the single harness
with a 300-second timeout. The command exited `124` and wrote
`/tmp/kani-h1-entry-point-validity.out`. The log again contains
`hnsw::kani_proofs::verify_entry_point_validity_4_nodes`,
`core::str::slice_error_fail`, `core::str::slice_error_fail_rt`, and
`core::str::<impl str>::floor_char_boundary`.

### H2: Panic-Capable Harness Construction Drives the Solver Growth

**Claim**: Kani is exploring panic/error formatting paths reached from
panic-capable construction in the HNSW harness, such as `.expect(...)`, rather
than from `is_entry_point_valid` itself.

**Plausibility**: Medium. The observed log repeatedly unwinds `core::str`
slice-error and panic formatting paths, while `is_entry_point_valid` is a small
boolean predicate.

**Prediction**: If this hypothesis holds, the H1 or full-run logs will contain
panic/string formatting paths, and the harness or construction path will contain
panic-capable calls before the invariant assertion.

#### H2 Falsification Plan

1. Search logs and code for panic/string paths and harness construction:

   ```bash
   rg -n -e "slice_error_fail|floor_char_boundary" \
     -e "panic|expect" \
     -e "insert_first|attach_node|promote_entry" \
     -e "is_entry_point_valid" \
     /tmp/kani-full-chutoro-2-2-7-status.out \
     /tmp/kani-h1-entry-point-validity.out \
     chutoro-core/src/hnsw/kani_proofs/ \
     chutoro-core/src/hnsw/{graph,node,params}.rs \
     chutoro-core/src/hnsw/invariants/helpers.rs
   ```

   Expected negative result: logs contain no panic/string paths, or the harness
   and predicate contain no panic-capable construction paths.

**Tooling**: `rg`.

**Confidence on falsification**: Medium. The grep can rule out the immediate
panic-path theory, but a surviving result still needs code-level causality.

**Execution status**: Not falsified. The initial inspection command included a
non-existent `chutoro-core/src/hnsw/graph.rs` path, so Laplace reran the
inspection against the actual HNSW graph paths. The harness still contained
panic-capable `.expect(...)` construction in the former monolithic
`chutoro-core/src/hnsw/kani_proofs.rs`; the predicate at
`chutoro-core/src/hnsw/invariants/helpers.rs:169` remained a small boolean
check. This supported the panic-formatting-path hypothesis but did not by
itself prove causality.

### H3: Dense SIMD 2.2.7 Changes Cause the Failure

**Claim**: The non-completion is caused by the new dense SIMD Kani harnesses or
nearby dense-provider changes rather than by the HNSW full-tier harness.

**Plausibility**: Low. The stopped full run was in `chutoro-core`, not the dense
provider, and the dense harnesses had passed in earlier targeted runs.

**Prediction**: If this hypothesis holds, at least one targeted dense SIMD
harness will fail or reproduce the same non-completion pattern.

#### H3 Falsification Plan

1. Run the dense dispatch-selection harness:

   ```bash
   cargo kani -p chutoro-providers-dense --default-unwind 4 \
     --harness verify_dense_simd_dispatch_selection_respects_support_masks \
     2>&1 | tee /tmp/kani-h3-dense-dispatch.out
   ```

   Expected negative result: the dispatch harness succeeds.

2. Run the dense tail-padding harness:

   ```bash
   cargo kani -p chutoro-providers-dense --default-unwind 5 \
     --harness verify_dense_simd_tail_padding_lane_bounds \
     2>&1 | tee /tmp/kani-h3-dense-tail.out
   ```

   Expected negative result: the tail-padding harness succeeds.

**Tooling**: `cargo kani`, `tee`.

**Confidence on falsification**: High for the direct 2.2.7 harnesses. It would
not rule out every possible branch interaction, but it would rule out the
targeted dense harnesses as the immediate failing checks.

**Execution status**: Falsified for the targeted dense SIMD harnesses. The bare
Kani command first failed before verification because `kani-compiler` could
not load `libLLVM.so.21.1-rust-1.93.0-nightly`; Laplace reran with Kani's
toolchain library path on `LD_LIBRARY_PATH`. Both reruns succeeded:

- `/tmp/kani-h3-dense-dispatch.out`: `VERIFICATION:- SUCCESSFUL`.
- `/tmp/kani-h3-dense-tail.out`: `VERIFICATION:- SUCCESSFUL`.

### H4: The Non-Completion Is Environmental Contention

**Claim**: The non-completion is due to local resource contention, not
deterministic Kani proof-state growth.

**Plausibility**: Medium. This machine is shared, and other agents may be using
Cargo or solver resources, but the repeated string-unwinding trace suggests a
deterministic proof-shape issue.

**Prediction**: If this hypothesis holds, resource checks will show material
CPU, memory, or disk pressure, and a bounded single-harness run will not
reproduce the same pattern under adequate resources.

#### H4 Falsification Plan

1. Inspect local resources and active solver/Cargo work:

   ```bash
   date
   uptime
   free -h
   df -h . /tmp
   pgrep -af "cargo kani|cbmc|kani-compiler|rustc|cargo" | head -40
   ```

   Expected negative result: memory and disk are adequate, and no unrelated
   solver storm is visible.

2. Compare the resource snapshot with H1's single-harness result.

   Expected negative result: H1 reproduces the same string-unwinding pattern
   under adequate resources.

**Tooling**: `date`, `uptime`, `free`, `df`, `pgrep`.

**Confidence on falsification**: Medium. Resource checks can rule out obvious
contention but not subtle solver scheduling variability.

**Execution status**: Falsified for obvious resource contention. Laplace
reported load averages of `1.01`, `1.36`, and `1.73`, roughly `92 GiB` of
available memory, and roughly `847 GB` of available disk. The single HNSW
harness still reproduced the timeout and string-unwinding pattern under those
conditions.

## Recommended Execution Order

1. **H1**: cheapest decisive reproduction of the suspected active harness.
2. **H2**: code and log inspection explains the observed string-unwinding path.
3. **H3**: verifies whether the recent dense SIMD work is directly implicated.
4. **H4**: separates deterministic proof growth from local resource pressure.

## Termination Criteria

- **Root cause narrowed**: One hypothesis survives its falsification test while
  the others are eliminated or downgraded.
- **Escalation trigger**: All hypotheses are falsified, or the single-harness
  run reports a concrete Kani counter-example instead of non-completion.

## Current Conclusion

The likely immediate blocker is the HNSW
`verify_entry_point_validity_4_nodes` full-tier harness. The dense SIMD 2.2.7
harnesses are not the direct cause, and the run does not appear to be blocked by
obvious resource contention. The strongest surviving theory is that the HNSW
harness construction leaves panic-capable paths available to Kani, causing CBMC
to spend the budget in Rust `core::str` panic and slice-error formatting.

## Notes for Executing Agent

- Run diagnostic commands sequentially; do not run Kani jobs in parallel.
- Keep logs in `/tmp`, but do not use `/tmp` as a build target.
- Do not edit source files during falsification.
- Preserve exact command output paths so later work can audit the results.
