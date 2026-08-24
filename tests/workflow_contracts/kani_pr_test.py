"""Contract tests for the pull-request Kani gate workflow.

The workflow is declarative configuration: it decides when the Kani gate
runs, what the checkout may do with the workflow token, and which Kani
version verifies the proofs. These tests parse the workflow with PyYAML
and pin that contract, so drift (losing a path filter, unpinning an
action or the verifier, widening permissions, or dropping the gating
step) fails CI on the pull request that introduces it.

Action pins are owned by Dependabot, so these tests assert the shape of
each pin (a 40-hex commit SHA on the correct action path) rather than a
specific value. The Kani verifier version is asserted exactly because a
silent verifier upgrade can change what the proofs mean.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "kani-pr.yml"
)

CHECKOUT_RE = re.compile(r"^actions/checkout@[0-9a-f]{40}$")
SETUP_RUST_RE = re.compile(
    r"^leynos/shared-actions/\.github/actions/setup-rust@[0-9a-f]{40}$"
)

#: Path filters that must trigger the gate: the workflow itself, every
#: Kani harness, the modules under proof, the Makefile that defines the
#: tier, and the Cargo manifests that shape the dependency graph of the
#: crates being verified.
EXPECTED_PATHS = [
    ".github/workflows/kani-pr.yml",
    "**/kani_*.rs",
    "chutoro-core/src/hnsw/kani_proofs/**",
    "chutoro-core/src/mst/kani_harness.rs",
    "chutoro-providers/dense/src/simd/kani_proofs.rs",
    "chutoro-core/src/hnsw/**",
    "chutoro-core/src/mst/**",
    "chutoro-providers/dense/src/simd/**",
    "Makefile",
    "Cargo.toml",
    "Cargo.lock",
    "chutoro-core/Cargo.toml",
    "chutoro-providers/dense/Cargo.toml",
]


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
    assert pull_request.get("paths") == EXPECTED_PATHS, (
        "on.pull_request.paths must be exactly the documented Kani surface: "
        f"expected {EXPECTED_PATHS!r}, got {pull_request.get('paths')!r}"
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
    """The PR gate stays well under the nightly 120-minute tier."""
    timeout = _kani_job(workflow).get("timeout-minutes")
    assert timeout == 30, (
        f"jobs.kani.timeout-minutes must be 30, got {timeout!r}"
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
    assert "cargo install --locked kani-verifier --version 0.67.0" in run, (
        f"the install step must pin kani-verifier with --locked and an exact "
        f"--version, got {run!r}"
    )
    assert "cargo kani setup" in run, (
        f"the install step must run cargo kani setup, got {run!r}"
    )


def test_gating_step_runs_the_practical_suite(workflow: dict[str, object]) -> None:
    """The gate runs make kani, the fast practical tier."""
    steps = _steps(workflow)
    run_steps = [step.get("run") for step in steps if isinstance(step.get("run"), str)]
    assert any(run.strip() == "make kani" for run in run_steps), (
        f"a step must run 'make kani' as the gating command, got {run_steps!r}"
    )
