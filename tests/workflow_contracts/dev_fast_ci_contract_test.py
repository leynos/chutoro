"""Contract tests for dev-fast CI workflow provisioning and lint sequencing."""

from __future__ import annotations

import os
import re
import subprocess
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from workflow_support import ROOT

MAKEFILE_PATH = ROOT / "Makefile"
DEV_FAST_CONFIG = "tools/dev-fast/config.toml"
WORKFLOW_PATHS = {
    "ci.yml": ROOT / ".github" / "workflows" / "ci.yml",
    "coverage-main.yml": ROOT / ".github" / "workflows" / "coverage-main.yml",
}


def _workflow_job(workflow_name: str, job_name: str) -> dict[str, Any]:
    workflow = yaml.safe_load(WORKFLOW_PATHS[workflow_name].read_text(encoding="utf-8"))
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict), f"{workflow_name} must define jobs"
    job = jobs.get(job_name)
    assert isinstance(job, dict), f"{workflow_name} must define job {job_name!r}"
    return job


def _step_named(job: dict[str, Any], name: str) -> tuple[int, dict[str, Any]]:
    steps = job.get("steps")
    assert isinstance(steps, list), "workflow job must define a step list"
    matches = [
        (index, step) for index, step in enumerate(steps) if step.get("name") == name
    ]
    assert len(matches) == 1, f"expected one {name!r} step, found {len(matches)}"
    index, step = matches[0]
    assert isinstance(step, dict), f"step {name!r} must be a mapping"
    return index, step


@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    (
        pytest.param("ci.yml", "build-test", id="pull-request-ci"),
        pytest.param("coverage-main.yml", "coverage-upload", id="main-coverage"),
    ),
)
def test_ci_installs_development_prerequisites_before_lint(
    workflow_name: str, job_name: str
) -> None:
    job = _workflow_job(workflow_name, job_name)
    install_index, install_step = _step_named(
        job, "Install development build prerequisites"
    )
    lint_index, lint_step = _step_named(job, "Lint")
    assert install_index < lint_index, (
        f"{workflow_name} must install the development toolchain before lint"
    )
    assert install_step.get("run", "").strip() == "make install-dev-fast"
    assert lint_step.get("run", "").strip() == "make lint"


def _string_values(value: Any) -> Iterator[str]:
    """Yield workflow scalar strings recursively for focused config checks."""
    if isinstance(value, str):
        yield value
        return

    if isinstance(value, dict):
        nested_values = value.values()
    elif isinstance(value, list):
        nested_values = value
    else:
        return

    for nested in nested_values:
        yield from _string_values(nested)


def test_string_values_preserves_container_order_and_omits_non_strings() -> None:
    workflow_value = {
        "first": "one",
        "nested": [
            "two",
            {"third": "three", "ignored": 7, "fourth": ["four", None]},
            False,
        ],
        "also_ignored": None,
    }

    assert list(_string_values(workflow_value)) == ["one", "two", "three", "four"]


def _dense_simd_commands(step: dict[str, Any]) -> list[str]:
    script = step.get("run")
    assert isinstance(script, str), "Dense stable SIMD gating must use a run script"
    joined = re.sub(r"\\\n[ \t]*", " ", script)
    return [
        line.strip()
        for line in joined.splitlines()
        if line.lstrip().startswith("cargo ")
    ]


@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    (
        pytest.param("ci.yml", "build-test", id="pull-request-ci"),
        pytest.param("coverage-main.yml", "coverage-upload", id="main-coverage"),
    ),
)
def test_ci_direct_simd_and_coverage_routes_stay_unfragmented(
    workflow_name: str, job_name: str
) -> None:
    job = _workflow_job(workflow_name, job_name)
    _, dense_step = _step_named(job, "Dense stable SIMD gating")
    commands = _dense_simd_commands(dense_step)
    assert len(commands) == 2, (
        f"{workflow_name} must keep both direct dense SIMD Cargo commands: {commands!r}"
    )
    for command in commands:
        assert "--config" not in command and DEV_FAST_CONFIG not in command, (
            f"direct SIMD command must remain unfragmented: {command!r}"
        )

    _, coverage_step = _step_named(job, "Test and Measure Coverage")
    coverage_values = "\n".join(_string_values(coverage_step))
    assert (
        DEV_FAST_CONFIG not in coverage_values and "--config" not in coverage_values
    ), (
        f"coverage action inputs must not select the dev-fast fragment: "
        f"{coverage_values!r}"
    )


def _make_probe(path: Path, role: str) -> None:
    """Write a controlled fake gate that records overlap with its peer."""
    source = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import os
        from pathlib import Path
        import time

        state = Path(os.environ["DEV_FAST_PROBE_STATE"])
        role = {role!r}
        peer = "whitaker" if role == "cargo" else "cargo"
        lock = state / "lock"

        def acquire_lock():
            deadline = time.monotonic() + 5
            while True:
                try:
                    lock.mkdir()
                    return
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        raise SystemExit("timed out acquiring probe lock")
                    time.sleep(0.005)

        acquire_lock()
        if (state / f"{{peer}}.running").exists():
            (state / "overlap").write_text(f"{{role}} overlapped {{peer}}\\n")
        (state / f"{{role}}.running").write_text("running\\n")
        lock.rmdir()

        time.sleep(0.15)

        acquire_lock()
        (state / f"{{role}}.running").unlink()
        lock.rmdir()
        """
    )
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_parallel_lint_serializes_clippy_and_whitaker(tmp_path: Path) -> None:
    cargo = tmp_path / "probe-cargo"
    whitaker = tmp_path / "probe-whitaker"
    state = tmp_path / "state"
    state.mkdir()
    _make_probe(cargo, "cargo")
    _make_probe(whitaker, "whitaker")

    environment = os.environ.copy()
    environment["DEV_FAST_PROBE_STATE"] = str(state)
    completed = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-j2",
            "--file",
            str(MAKEFILE_PATH),
            "lint",
            f"CARGO={cargo}",
            f"WHITAKER={whitaker}",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, (
        f"controlled lint probe failed: {completed.stdout}\n{completed.stderr}"
    )
    assert not (state / "overlap").exists(), (
        "lint-clippy and lint-whitaker ran concurrently under make -j2"
    )
