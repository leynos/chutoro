"""Contract tests for the Python docstring-coverage gate in the Makefile.

`make lint` must reach an `interrogate` run that covers every Python file the
repository owns, at a threshold of 100, with a guard that fails on any `MISSED`
row (interrogate rounds its percentage to one decimal, so the threshold alone
passes a lone gap in a large repository), and CI must have uv before it runs
`make lint`. These tests read the recipe and the workflow, and also run the
target against a stub tool, so a deleted flag, a dropped prerequisite or a
guard that stops failing breaks a test rather than silently restoring the gap.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "Makefile"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _makefile_text() -> str:
    """Return the Makefile source."""
    return MAKEFILE.read_text(encoding="utf-8")


def _variable(name: str) -> str:
    """Return the value a `NAME ?=` or `NAME =` assignment gives, joined."""
    pattern = re.compile(rf"^{name}\s*\??=\s*((?:.*\\\n)*.*)$", re.MULTILINE)
    match = pattern.search(_makefile_text())
    assert match, f"the Makefile must assign {name}"
    return re.sub(r"\\\n\s*", " ", match.group(1)).strip()


def _prerequisites(target: str) -> list[str]:
    """Return the prerequisites declared for *target*."""
    pattern = re.compile(rf"^{re.escape(target)}:([^=\n]*)$", re.MULTILINE)
    match = pattern.search(_makefile_text())
    assert match, f"the Makefile must declare a {target} target"
    return match.group(1).split("##")[0].split()


def _tracked(*patterns: str) -> list[str]:
    """Return the Git-tracked files matching *patterns*.

    Git is the authority on what the repository owns: a walk would also see
    the virtual environments CI creates beside the sources. Names are read from
    NUL-separated output, so a path containing whitespace stays whole.
    """
    output = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(ROOT), "ls-files", "-z", *patterns],  # noqa: S607 - git
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [name for name in output.split("\0") if name]


def _python_directories() -> set[Path]:
    """Return the directories that hold Python source or stubs Git tracks."""
    return {Path(name).parent for name in _tracked("*.py", "*.pyi")}


def test_lint_depends_on_interrogate() -> None:
    """Dropping the prerequisite would remove the gate from `make lint`."""
    assert "interrogate" in _prerequisites("lint")


def test_interrogate_is_declared_phony() -> None:
    """A file named `interrogate` must not make the target look up to date."""
    declared = re.findall(r"^\.PHONY:((?:.*\\\n)*.*)$", _makefile_text(), re.MULTILINE)
    names = " ".join(declared).replace("\\\n", " ").split()
    assert "interrogate" in names, "interrogate must be a .PHONY target"


def test_the_tool_is_pinned_and_run_through_uv() -> None:
    """An unpinned tool could change the verdict with no repository change."""
    assert _variable("INTERROGATE_VERSION") == "1.7.0"
    command = _variable("INTERROGATE")
    if re.search(r"^UV_GATE\s*[:?]?=", _makefile_text(), re.MULTILINE):
        assert "$(UV_GATE) tool" in command, "reach uv through the vendored helper"
        return
    assert re.search(r"(\$\(UV\)|\buv) tool run\b", command), "run via uv tool run"
    if re.search(r"^UV\s*[:?]?=", _makefile_text(), re.MULTILINE):
        assert "$(UV) tool run" in command, "honour the configured $(UV) executable"
    assert "--from 'interrogate==$(INTERROGATE_VERSION)'" in command


def test_the_flags_demand_a_full_exact_report() -> None:
    """Exactly: every definition printed, threshold 100, stubs when present.

    Equality, not a deny-list: any other flag (an ignore or exclude option, or
    `-q` and `-o`, which suppress or redirect the report the guard reads) could
    relax the gate or blind the MISSED guard.
    """
    flags = _variable("INTERROGATE_FLAGS").split()
    expected = ["-vv", "--fail-under", "100"]
    if _tracked("*.pyi"):
        expected = ["-vv", "--ext", "pyi", "--fail-under", "100"]
    assert flags == expected, f"INTERROGATE_FLAGS must be exactly {expected}, not {flags}"


def test_every_python_directory_is_a_target() -> None:
    """A Python directory outside `INTERROGATE_TARGETS` is not checked."""
    targets = [Path(t) for t in _variable("INTERROGATE_TARGETS").split()]
    uncovered = [
        str(directory)
        for directory in sorted(_python_directories())
        if not any(directory.is_relative_to(target) for target in targets)
    ]
    assert not uncovered, f"add these to INTERROGATE_TARGETS: {uncovered}"


def test_ci_installs_uv_before_it_runs_make_lint() -> None:
    """Each ci.yml job with an exact `make lint` step has uv first; one exists."""
    workflow = yaml.safe_load(CI.read_text(encoding="utf-8"))
    checked = 0
    for name, job in workflow["jobs"].items():
        steps = job.get("steps", [])
        lint = [i for i, s in enumerate(steps) if str(s.get("run", "")).strip() == "make lint"]
        if not lint:
            continue
        uv = [i for i, s in enumerate(steps) if "setup-uv" in str(s.get("uses", ""))]
        assert uv, f"job {name} runs make lint without setting up uv"
        assert min(uv) < lint[0], f"job {name} sets up uv after make lint"
        checked += 1
    assert checked, "no CI job runs `make lint`"


def _run_target(tmp_path: Path, script: str) -> subprocess.CompletedProcess[str]:
    """Run `make interrogate` with the tool replaced by a stub script."""
    stub = tmp_path / "stub.sh"
    stub.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$(dirname "$0")/args.txt"\n'
        + textwrap.dedent(script),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    make = shutil.which("make")
    assert make
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [make, "-C", str(ROOT), "interrogate", f"INTERROGATE={stub}"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_target_passes_when_the_tool_reports_no_gap(tmp_path: Path) -> None:
    """A clean report and a zero status pass."""
    result = _run_target(tmp_path, "echo '| f.py | COVERED |'\nexit 0\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "| f.py | COVERED |" in result.stdout


def test_the_target_passes_the_configured_flags_and_targets(tmp_path: Path) -> None:
    """The tool receives exactly the configured flags, then the targets."""
    _run_target(tmp_path, "exit 0\n")
    received = (tmp_path / "args.txt").read_text(encoding="utf-8").split()
    expected = _variable("INTERROGATE_FLAGS").split() + _variable("INTERROGATE_TARGETS").split()
    assert received == expected


def test_the_target_fails_when_the_tool_fails(tmp_path: Path) -> None:
    """A failing tool fails the target."""
    result = _run_target(tmp_path, "echo 'FAILED'\nexit 1\n")
    assert result.returncode != 0


@pytest.mark.parametrize(
    "cell",
    ["|  MISSED |", "|   MISSED    |"],
    ids=["interrogate-default-padding", "wider-padding"],
)
def test_the_target_fails_on_a_missed_row_the_percentage_hid(
    tmp_path: Path, cell: str
) -> None:
    """A MISSED row in a report from a successful run still fails the target.

    The guard must not depend on how many spaces pad the status cell.
    """
    result = _run_target(
        tmp_path,
        f"echo '| f.main (L3) {cell}'\necho 'RESULT: PASSED'\nexit 0\n",
    )
    assert result.returncode != 0, "a MISSED row must fail the target"
    assert "undocumented definitions remain" in result.stderr, result.stderr
