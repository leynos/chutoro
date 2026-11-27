"""Contract-test the scheduled benchmark baseline comparison lane.

The lane failed on every scheduled run from the day it was added until #261,
because it looked for a saved baseline at ``target/criterion/ci-reference``.
Criterion writes each benchmark's data beneath ``<output-root>/<benchmark-id>/
<baseline-name>/``, so that path never exists. The lane also copied a directory
and re-ran the benchmark, which measured twice and never compared anything.

These tests pin the replacement sequence to the helper that resolves the real
layout, and they pin the job-status policy that decides when the lane is red.
A future edit that reintroduces the literal path, that lets a successful save
count as a pass, or that uploads the evidence outside the job's temp directory
fails here rather than in a scheduled run nobody reads.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import typing as typ

import pytest
from workflow_support import job, read_workflow_text, steps

WORKFLOW = "benchmark-regressions.yml"
COMPARE_JOB = "benchmark-baseline-compare"

#: The literal path that made every earlier run fail before comparing.
OBSOLETE_PATH = "target/criterion/ci-reference"

#: Helper invocations the lane must reach, in the order they must appear.
HELPER_SEQUENCE = ("inventory", "inventory", "stage", "classify")


def _steps(job_name: str = COMPARE_JOB) -> list[dict[str, typ.Any]]:
    """Return the comparison job's steps."""
    return steps(job(WORKFLOW, job_name))


def _scripts(job_name: str = COMPARE_JOB) -> list[str]:
    """Return every ``run:`` script in the job, in order."""
    return [
        step["run"]
        for step in _steps(job_name)
        if isinstance(step.get("run"), str)
    ]


def _script_text(job_name: str = COMPARE_JOB) -> str:
    """Return the job's scripts joined into one document."""
    return "\n".join(_scripts(job_name))


def _helper_invocations(text: str) -> list[str]:
    """Return the helper subcommands the text invokes, in order."""
    pattern = re.compile(
        r"benchmark_baseline\s+--\s+(inventory|stage|classify)"
        r"|--bin\s+benchmark_baseline\s+--\s+(inventory|stage|classify)"
    )
    found: list[str] = []
    for match in pattern.finditer(text):
        subcommand = match.group(1) or match.group(2)
        found.append(subcommand)
    return found


def test_the_obsolete_baseline_path_is_gone() -> None:
    """The single-directory assumption must not come back."""
    text = read_workflow_text(_workflow_path())

    assert OBSOLETE_PATH not in text, (
        f"{WORKFLOW} still looks for {OBSOLETE_PATH}; Criterion stores one "
        "directory per benchmark id beneath the output root"
    )


def test_the_literal_transfer_is_gone() -> None:
    """Copying a baseline directory was the mechanism that hid the real error."""
    text = _script_text()
    joined = re.sub(r"\s+", " ", text)

    assert "cp -R" not in joined, "the lane must stage through the helper"
    assert "rm -rf" not in joined, "the lane must not delete baseline trees inline"


def _reference_path_scripts() -> str:
    """Return the scripts that run only when a reference commit exists."""
    condition = "${{ steps.reference.outputs.has_reference == 'true' }}"
    return "\n".join(
        step["run"]
        for step in _steps()
        if step.get("if") == condition and isinstance(step.get("run"), str)
    )


def test_helper_subcommands_run_in_order() -> None:
    """Both measurements are inventoried, then staged, then classified."""
    invocations = _helper_invocations(_reference_path_scripts())

    assert invocations == list(HELPER_SEQUENCE), (
        "expected inventory, inventory, stage, classify; found "
        f"{invocations}"
    )


def test_evidence_lives_outside_the_reference_worktree() -> None:
    """Evidence must survive the worktree removal that follows the comparison."""
    text = _script_text()

    assert "RUNNER_TEMP}/bench-evidence/" in text, (
        "evidence must be written below RUNNER_TEMP"
    )
    assert "/tmp/chutoro-reference-" not in text, (
        "the reference worktree must not share the evidence directory"
    )


def test_no_reference_warns_without_a_substitute_measurement() -> None:
    """A shallow or root checkout must record an outcome, not fake a pass."""
    guard = [
        step
        for step in _steps()
        if "has_reference != 'true'" in str(step.get("if", ""))
    ]
    assert len(guard) == 1, "exactly one step owns the no-reference path"
    text = guard[0].get("run", "")

    assert "--no-reference" in text, "the warning path must record a reason"
    assert "exit 0" in text, "a missing reference is a warning, not a failure"
    assert "::warning::" in text, "the outcome must be visible in the run summary"
    assert "--save-baseline" not in text, (
        "a missing reference must not save a current-only baseline as a substitute"
    )


def test_a_pass_requires_classification() -> None:
    """A successful save or comparison is not a pass."""
    classify_steps = [
        step
        for step in _steps()
        if "Classify benchmark comparison" in step.get("name", "")
    ]
    assert len(classify_steps) == 1, "the lane must classify exactly once"
    script = classify_steps[0].get("run", "")
    assert "outcome=pass" in script, "only a classify result of pass is a success"

    for step in _steps():
        name = step.get("name", "")
        if "Save current baseline" in name or "Save reference baseline" in name:
            assert "outcome=pass" not in step.get("run", ""), name


def _classify_case_arms(script: str) -> dict[str, str]:
    """Split a shell ``case`` block into its labelled arms.

    The labels the workflow uses are exit statuses, so the mapping from a
    helper result to a job status is readable without running the script.
    """
    arms: dict[str, str] = {}
    pattern = re.compile(r"^\s*(\d+|\*)\)\n((?:.*\n)*?)\s*;;", re.M)
    for match in pattern.finditer(script):
        arms[match.group(1)] = match.group(2)
    return arms


def test_a_regression_fails_with_a_performance_message() -> None:
    """A regression is the only red the policy attributes to performance."""
    classify = [
        step.get("run", "")
        for step in _steps()
        if "Classify benchmark comparison" in step.get("name", "")
    ]
    assert len(classify) == 1
    arms = _classify_case_arms(classify[0])

    assert "::error::performance regression" in arms.get("1", ""), arms
    assert "exit 1" in arms.get("1", ""), "a regression must fail the step"


def test_the_job_status_policy_maps_every_helper_outcome() -> None:
    """Each documented outcome maps to the job status the policy states."""
    classify = [
        step.get("run", "")
        for step in _steps()
        if "Classify benchmark comparison" in step.get("name", "")
    ]
    assert len(classify) == 1
    script = classify[0]
    arms = _classify_case_arms(script)
    for label in ("0", "1", "3", "4", "*"):
        assert label in arms, f"the classifier has no arm for exit {label}"

    assert "outcome=pass" in arms["0"], arms["0"]
    assert "exit" not in arms["0"], "a pass must not exit non-zero"
    assert "exit 1" in arms["1"], "a regression must fail the step"
    assert "::warning::" in arms["3"], "incompatible must warn"
    assert "exit" not in arms["3"] or "exit 0" in arms["3"], arms["3"]
    assert "::warning::" in arms["4"], "no_reference must warn"
    assert "exit" not in arms["4"] or "exit 0" in arms["4"], arms["4"]
    assert "exit 2" in arms["*"], "a lane defect must fail the step"


def test_a_lane_defect_says_it_is_not_a_regression() -> None:
    """A broken lane must not read as a measured slowdown."""
    text = _script_text()

    assert "lane defect, not a performance regression" in text, (
        "broken evidence must be distinguishable from a regression"
    )


def test_evidence_upload_covers_every_stage_log() -> None:
    """The uploaded evidence must include both logs and every manifest."""
    uploads = [
        step
        for step in _steps()
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]

    assert len(uploads) == 1, "the lane uploads exactly one evidence artifact"
    upload = uploads[0]
    assert upload.get("if") == "always()", "evidence must be uploaded on failure"
    options = upload.get("with", {})
    assert options.get("if-no-files-found") == "error"
    assert int(options.get("retention-days", 0)) > 0, "retention must be explicit"
    path = options.get("path", "")
    assert "bench-evidence" in path, path


def test_failed_stage_is_recorded_in_the_verdict() -> None:
    """The failure handler must name the stage in the durable evidence."""
    text = _script_text()

    assert "--failed-stage" in text, "the verdict must record the failed stage"
    assert "current-stage" in text, "each step must publish the stage it owns"


def test_pull_request_path_coverage_is_preserved() -> None:
    """A helper change must still run the lane on a pull request."""
    text = read_workflow_text(_workflow_path())

    assert "'chutoro-test-support/**'" in text, (
        "the pull-request path filter must keep covering the helper"
    )


def _workflow_path() -> typ.Any:
    """Return the workflow file path."""
    from workflow_support import WORKFLOW_DIR

    return WORKFLOW_DIR / WORKFLOW


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("--load-baseline", "the comparison must load the measured current data"),
        ("--baseline ci-reference", "the comparison must name the reference baseline"),
        ("--save-baseline ci-current", "the current measurement is saved once"),
        ("--save-baseline ci-reference", "the reference measurement is saved once"),
    ],
)
def test_criterion_flags_are_present(feature: str, expected: str) -> None:
    """The comparison must not measure again."""
    text = re.sub(r"\s+", " ", _script_text())

    assert feature in text, expected


def test_the_comparison_selects_both_baselines_and_the_filter() -> None:
    """The comparison loads the measured data and names the reference once."""
    compare_steps = [
        step
        for step in _steps()
        if "Compare current against reference baseline" in step.get("name", "")
    ]
    assert len(compare_steps) == 1, "the lane compares in exactly one step"
    script = compare_steps[0].get("run", "")

    for flag in (
        "--load-baseline ci-current",
        "--baseline ci-reference",
        "--noplot",
        "--color never",
    ):
        assert flag in script, f"the comparison must pass {flag}"
    assert "filter.txt" in script, "the comparison must use the emitted filter"
    assert "--exact" not in script, "an exact filter would select nothing"