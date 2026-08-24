"""Contract tests for the pull-request Kani gate workflow.

The workflow is declarative configuration: it decides when the Kani gate
runs, what the checkout may do with the workflow token, and which Kani
version verifies the proofs. These tests parse the workflow with PyYAML
and pin that contract, so drift (losing a path filter, unpinning an
action or the verifier, widening permissions, or dropping the gating
step) fails CI on the pull request that introduces it.

These tests assert shapes and relationships, never specific pinned
values. Restating a pin in a test only duplicates the thing that changes,
so a routine bump fails the build for no defect. Accordingly: action pins
are checked for shape (a 40-hex commit SHA on the correct action path) and
for cross-workflow consistency in ``action_pins_test``; the path filter is
checked by deriving the Kani surface from the tree and asserting the
filter covers it; and the verifier version is checked only for the
workflow deriving it from ``tools/kani/VERSION`` -- the single source of
truth also read by the Makefile and by ``prover-tools kani install``.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pathspec
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "kani-pr.yml"
NIGHTLY_PATH = REPO_ROOT / ".github" / "workflows" / "nightly-kani.yml"
#: Single source of truth for the pinned Kani verifier version.
KANI_VERSION_PATH = REPO_ROOT / "tools" / "kani" / "VERSION"
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
#: A literal version argument, e.g. ``--version 1.2.3``. The workflow must
#: interpolate the pin file instead, so this pattern must never match.
VERSION_LITERAL_RE = re.compile(r"--version\s+[\"']?\d")

CHECKOUT_RE = re.compile(r"^actions/checkout@[0-9a-f]{40}$")
SETUP_RUST_RE = re.compile(
    r"^leynos/shared-actions/\.github/actions/setup-rust@[0-9a-f]{40}$"
)

#: Non-source inputs that change what the gate runs, so a change to any of
#: them must trigger it. Everything else is derived from the tree below.
GATE_INPUTS = ("Makefile", "Cargo.lock", "tools/kani/VERSION")


def _kani_surface() -> list[str]:
    """Return every Kani harness path in the tree, repository-relative.

    Derived rather than enumerated so a harness added in a new location
    fails this contract until the path filter covers it.
    """
    surface = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.rglob("kani_*.rs")
        if "target" not in path.parts
    }
    surface |= {
        path.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.rglob("kani_proofs/*.rs")
        if "target" not in path.parts
    }
    return sorted(surface)


def _uncovered(paths: list[str], patterns: list[str]) -> list[str]:
    """Return the paths no filter pattern matches."""
    spec = pathspec.PathSpec.from_lines("gitignore", patterns)
    return [path for path in paths if not spec.match_file(path)]



@pytest.fixture(scope="module")
def workflow() -> dict[str, object]:
    """Parse the workflow file once for every contract test."""
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _triggers(workflow: dict[str, object]) -> dict[str, object]:
    """Return the ``on:`` mapping (PyYAML parses the bare key as True)."""
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict), "the workflow must declare an on: mapping"
    return triggers


def _kani_job(workflow: dict[str, object]) -> dict[str, object]:
    """Return the single gating job."""
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict), "the workflow must declare a jobs mapping"
    assert list(jobs) == ["kani"], (
        f"expected a single job named 'kani', found {sorted(jobs)}"
    )
    return jobs["kani"]


def _steps(workflow: dict[str, object]) -> list[dict[str, object]]:
    """Return the gating job's step list."""
    steps = _kani_job(workflow).get("steps")
    assert isinstance(steps, list) and steps, "jobs.kani.steps is missing"
    return steps


def test_pull_request_trigger_covers_the_kani_surface(
    workflow: dict[str, object],
) -> None:
    """The gate fires on main-targeted PRs touching Kani-relevant paths."""
    triggers = _triggers(workflow)
    pull_request = triggers.get("pull_request")
    assert isinstance(pull_request, dict), "on.pull_request is missing"
    assert pull_request.get("branches") == ["main"], (
        f"on.pull_request.branches must be ['main'], got "
        f"{pull_request.get('branches')!r}"
    )
    assert pull_request.get("types") == ["opened", "synchronize", "reopened"], (
        f"on.pull_request.types must cover opened/synchronize/reopened, got "
        f"{pull_request.get('types')!r}"
    )
    patterns = pull_request.get("paths")
    assert isinstance(patterns, list) and patterns, (
        f"on.pull_request.paths must list filter patterns, got {patterns!r}"
    )

    surface = _kani_surface()
    assert surface, "no Kani harnesses were discovered; the derivation is broken"
    missed = _uncovered(surface, patterns)
    assert not missed, (
        "on.pull_request.paths must cover every Kani harness in the tree; "
        f"these are not matched by any pattern: {missed!r}"
    )

    missed_inputs = _uncovered(list(GATE_INPUTS), patterns)
    assert not missed_inputs, (
        "on.pull_request.paths must cover the inputs that change what the "
        f"gate runs; these are not matched by any pattern: {missed_inputs!r}"
    )

    assert "workflow_dispatch" in triggers, "on.workflow_dispatch is missing"


def test_workflow_permissions_are_read_only(workflow: dict[str, object]) -> None:
    """The workflow token grants contents: read and nothing broader."""
    permissions = workflow.get("permissions")
    assert permissions == {"contents": "read"}, (
        f"permissions must be exactly {{'contents': 'read'}}, got {permissions!r}"
    )


def test_concurrency_cancels_superseded_runs(workflow: dict[str, object]) -> None:
    """A newer push cancels the previous run for the same ref."""
    concurrency = workflow.get("concurrency")
    assert isinstance(concurrency, dict), "the workflow must declare concurrency"
    assert concurrency.get("group") == "kani-pr-${{ github.ref }}", (
        f"concurrency.group must key on the triggering ref, got "
        f"{concurrency.get('group')!r}"
    )
    assert concurrency.get("cancel-in-progress") is True, (
        f"concurrency.cancel-in-progress must be true, got "
        f"{concurrency.get('cancel-in-progress')!r}"
    )


def test_job_timeout_is_tighter_than_the_nightly_budget(
    workflow: dict[str, object],
) -> None:
    """The PR gate is bounded, and strictly tighter than the nightly tier."""
    timeout = _kani_job(workflow).get("timeout-minutes")
    assert isinstance(timeout, int) and timeout > 0, (
        f"jobs.kani.timeout-minutes must be a positive integer, got {timeout!r}"
    )

    nightly = yaml.safe_load(NIGHTLY_PATH.read_text(encoding="utf-8"))
    nightly_jobs = nightly.get("jobs")
    assert isinstance(nightly_jobs, dict) and nightly_jobs, (
        "the nightly Kani workflow must declare a job to compare against"
    )
    nightly_timeout = next(iter(nightly_jobs.values())).get("timeout-minutes")
    assert isinstance(nightly_timeout, int), (
        f"the nightly job must declare timeout-minutes, got {nightly_timeout!r}"
    )
    assert timeout < nightly_timeout, (
        "the pull-request gate must be tighter than the nightly tier: "
        f"PR is {timeout} minutes, nightly is {nightly_timeout}"
    )


def test_checkout_is_pinned_and_does_not_persist_credentials(
    workflow: dict[str, object],
) -> None:
    """Checkout is SHA-pinned and drops the token before running PR code."""
    steps = _steps(workflow)
    checkout = steps[0]
    uses = checkout.get("uses")
    assert isinstance(uses, str) and CHECKOUT_RE.match(uses), (
        f"the first step must be actions/checkout pinned to a 40-hex commit "
        f"SHA, got {uses!r}"
    )
    with_block = checkout.get("with")
    assert isinstance(with_block, dict) and (
        with_block.get("persist-credentials") is False
    ), (
        "the checkout step must set persist-credentials: false so the "
        f"workflow token is not retained in Git configuration, got "
        f"{with_block!r}"
    )


def test_setup_rust_is_pinned_to_a_commit_sha(workflow: dict[str, object]) -> None:
    """The shared setup-rust action is referenced at a full commit SHA."""
    steps = _steps(workflow)
    uses_values = [step.get("uses") for step in steps if step.get("uses")]
    assert any(
        isinstance(uses, str) and SETUP_RUST_RE.match(uses) for uses in uses_values
    ), (
        "a step must use leynos/shared-actions setup-rust pinned to a "
        f"40-hex commit SHA, got {uses_values!r}"
    )


def test_kani_install_is_locked_and_version_pinned(
    workflow: dict[str, object],
) -> None:
    """The verifier installs with --locked at an exact approved version."""
    steps = _steps(workflow)
    install_runs = [
        step.get("run")
        for step in steps
        if isinstance(step.get("run"), str) and "kani-verifier" in step["run"]
    ]
    assert len(install_runs) == 1, (
        f"expected exactly one Kani install step, found {len(install_runs)}"
    )
    run = install_runs[0]
    assert "--locked" in run, (
        f"the install step must pass --locked, got {run!r}"
    )
    assert "tools/kani/VERSION" in run, (
        "the install step must read the pinned version from "
        f"tools/kani/VERSION rather than restating it, got {run!r}"
    )
    assert VERSION_LITERAL_RE.search(run) is None, (
        "the install step must not hardcode a version literal; the pin "
        f"belongs solely in tools/kani/VERSION, got {run!r}"
    )
    assert "cargo kani setup" in run, (
        f"the install step must run cargo kani setup, got {run!r}"
    )


def test_pinned_kani_version_file_is_a_bare_semver() -> None:
    """The pin file holds one machine-readable version and nothing else."""
    assert KANI_VERSION_PATH.is_file(), (
        f"the pinned Kani version file is missing: {KANI_VERSION_PATH}"
    )
    raw = KANI_VERSION_PATH.read_text(encoding="utf-8")
    version = raw.strip()
    assert SEMVER_RE.match(version), (
        f"tools/kani/VERSION must contain a bare MAJOR.MINOR.PATCH version, "
        f"got {raw!r}"
    )


def test_gating_step_runs_the_practical_suite(workflow: dict[str, object]) -> None:
    """The gate runs make kani, the fast practical tier."""
    steps = _steps(workflow)
    run_steps = [step.get("run") for step in steps if isinstance(step.get("run"), str)]
    assert any(run.strip() == "make kani" for run in run_steps), (
        f"a step must run 'make kani' as the gating command, got {run_steps!r}"
    )
