"""Contract-test Chutoro's initial Namespace runner assignment.

Parse the workflow YAML and assert the runner labels that protect the
Namespace migration and retained property-test capacity from drift. Run these
checks with ``make test-workflow-contracts``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _job(workflow_name: str, job_name: str) -> dict[str, object]:
    """Load one named job from a repository workflow.

    For example, ``_job("property-tests.yml", "property-tests-pr")`` returns
    that job's mapping, including its ``runs-on`` entry. This helper remains a
    read-only, test-local seam for this module's workflow contract checks.
    """
    workflow_path = ROOT / ".github" / "workflows" / workflow_name
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict), f"{workflow_name} must parse to a mapping"
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict), f"{workflow_name} must declare jobs"
    job = jobs.get(job_name)
    assert isinstance(job, dict), f"{workflow_name} must declare {job_name}"
    return job


def test_bounded_simd_verification_uses_the_shared_namespace_profile() -> None:
    """Keep the compatible nightly verification runner assignment stable."""
    job = _job("nightly-portable-simd.yml", "nightly-portable-simd")
    assert job.get("runs-on") == "namespace-profile-default", (
        "nightly-portable-simd.yml:nightly-portable-simd must use namespace-profile-default"
    )


def test_eight_core_property_jobs_remain_on_their_current_runner() -> None:
    """Preserve the nextest CI profile's required eight-core capacity."""
    pr_job = _job("property-tests.yml", "property-tests-pr")
    weekly_job = _job("property-tests.yml", "property-tests-weekly")
    assert pr_job.get("runs-on") == "ubicloud-standard-8", (
        "property-tests.yml:property-tests-pr must use ubicloud-standard-8"
    )
    assert weekly_job.get("runs-on") == "ubicloud-standard-8", (
        "property-tests.yml:property-tests-weekly must use ubicloud-standard-8"
    )
