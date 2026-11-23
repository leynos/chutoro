# FV07 delivery notes: #215 comment and ruleset change request

This note records the two outward-facing actions that follow the code change.
Neither has been posted or applied: both are drafts for review, because they
change a public conversation and a live repository setting.

## 1. Draft comment for issue #215

> **Update (FV07, #396): Kani now gates merges**
>
> The two claims below are now stale, and the model-drift warning is still
> correct.
>
> **Stale: "No Kani harness gates anything."** `make kani` runs on every pull
> request to `main` through `.github/workflows/kani-pr.yml`, which no longer
> carries a path filter, and `kani` is a required status check on the
> `main-required-checks` ruleset alongside `build-test` and `verus-proofs`. A
> failing or missing proof run blocks a merge.
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
> - `make kani` is enforced, bounded, and *does* compile the production code
>   under proof, so it cannot silently disagree with `src/`.
>
> The cost asymmetry the issue asked to document is now a difference in
> *guarantee* rather than in enforcement: Kani proves properties of the code
> that ships, within bounds; Verus proves properties of a model of it. The
> guidance in the developer's guide, "Choosing a verification tool", reflects
> that.
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
- from a non-bypass account, open a pull request that breaks one Kani proof and
  confirm the `kani` check fails and the merge button is blocked;
- restore the proof and confirm the check passes and the merge is unblocked.

### Rollback

Remove the `kani` entry and nothing else. The ruleset is a single
`required_status_checks` rule with a `deletion` rule; no other setting changes
in this request.
