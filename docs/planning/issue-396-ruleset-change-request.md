# FV07 delivery notes: #215 comment and ruleset change request

This note records the two outward-facing actions that follow the code change.
Neither has been posted or applied: both are drafts for review, because they
change a public conversation and a live repository setting.

## 1. Draft comment for issue #215

> **Update (FV07, #396): Kani runs on every pull request**
>
> Two of the claims below are now stale and one is still correct. Read the
> "Still correct" paragraph carefully: it is narrower than it first appears.
>
> **Stale: "No Kani harness gates anything."** `make kani` runs on every pull
> request to `main` through `.github/workflows/kani-pr.yml`, which no longer
> carries a path filter, so the `kani` context is reported on every change.
> `kani` is being added to the `main-required-checks` ruleset alongside
> `build-test` and `verus-proofs`; until an administrator applies that change
> the job runs without blocking a merge, which is a narrower claim than "it
> gates".
>
> **Stale: "`make kani-full` … does not currently complete."** The full
> 17-harness suite completes on the audited baseline `56d6909`. The retired
> harnesses described in `docs/kani-full-hnsw-hypothesis-testing.md` were
> replaced by narrower proofs and unit-test twins, and the tractability policy
> is in the developer's guide.
>
> **Still correct: the drift hazard.** A Verus proof under `verus/` is a
> standalone transcription with no build-level tie to `src/`. The asymmetry
> this issue was opened for therefore still exists in a different form:
>
> - `make verus` is enforced, expensive per pull request, and not
>   drift-sensitive at build time.
> - `make kani` is enforced, bounded, and compiles the production code the
>   harness reaches, so the harnesses that drive production paths cannot
>   disagree with `src/` without failing to compile.
>
> That last point is narrower than "Kani cannot drift", and the difference
> matters. Several harnesses do not drive production paths: the MST proofs
> verify a `#[cfg(kani)]` sequential model in place of the Rayon
> implementation, and the dense SIMD harnesses model the selector and lane
> arithmetic rather than executing intrinsics. Those harnesses can stay green
> while production drifts, exactly as a transcribed Verus proof can. Their
> refinement is carried by exhaustive equivalence tests over a bounded input
> domain, not by the harness itself, and a domain boundary or a changed
> production ordering is outside what those tests enumerate. FV05 is the
> ledger that would make that obligation explicit; this work does not
> implement it.
>
> The cost asymmetry the issue asked to document is therefore a difference in
> *guarantee* rather than in enforcement: Kani proves properties of the code a
> harness reaches, within bounds, and of a model where it does not; Verus
> proves properties of a model throughout. The guidance in the developer's
> guide, "Choosing a verification tool", reflects that.
>
> **Not implemented here: FV05.** The proof-to-production refinement ledger is
> separate work.

## 2. Draft ruleset change request

Post-merge, from an administrator account, against ruleset `18427957`
(`main-required-checks`).

Current state, read from the API on 2026-11-23:

```plaintext
required_status_checks:
  - context: build-test      integration_id: 15368
  - context: verus-proofs    integration_id: 15368
```

Required state:

```plaintext
required_status_checks:
  - context: build-test      integration_id: 15368
  - context: verus-proofs    integration_id: 15368
  - context: kani            integration_id: 15368
```

### Preconditions

1. The workflow change has merged to `main`, so `kani-pr.yml` declares no
   `paths` filter.
2. A pull request has produced a `kani` check run reported by app `15368`
   (GitHub Actions).
3. `gh api repos/leynos/chutoro/rulesets/18427957` shows the current state
   above.

### Change

Add one entry to the existing `required_status_checks` rule, preserving
`strict_required_status_checks_policy: false`,
`do_not_enforce_on_create: false`, the `deletion` rule, `conditions`, and
`bypass_actors` exactly as they are.

### Verification

```sh
gh api repos/leynos/chutoro/rulesets/18427957 \
  --jq '.rules[] | select(.type == "required_status_checks")
        | .parameters.required_status_checks'
gh api repos/leynos/chutoro/rules/branches/main
```

The first must list three contexts, each with `integration_id` 15368. The
second is the effective-rules view and must show the same three.

Then confirm enforcement rather than configuration:

- have an administrator review the `bypass_actors` list, which currently gives
  `RepositoryRole` id 5 `always` bypass;
- from a non-bypass account, open a pull request that breaks one **fast-tier**
  proof and confirm the `kani` check fails and the merge button is blocked. The
  fast tier is what the pull-request job runs, so a full-tier proof would be
  broken without that job noticing. Any `fast` entry in
  `tools/kani/harnesses.toml` will do;
  `hnsw::kani_proofs::bidirectional::
  verify_bidirectional_links_smoke_2_nodes_1_layer`
  is the quickest, because it verifies in seconds. Confirm the `kani` check
  itself is red, not merely that the merge is blocked: another failing check
  would block the merge too;
- restore the proof and confirm the check passes and the merge is unblocked.

### Rollback

Remove the `kani` entry and nothing else. The ruleset is a single
`required_status_checks` rule with a `deletion` rule; no other setting changes
in this request.
