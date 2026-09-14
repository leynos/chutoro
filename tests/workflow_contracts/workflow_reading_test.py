"""How the workflow reading fails, and what it reads when it does not.

The lane derivations take their documents as a parameter, so the only
place that touches the filesystem is ``workflow_support``. These drive
that boundary with directories written for a case: this repository has
no unreadable workflow, no workflow that is not YAML, and no missing
workflow directory, so none of them can be described against the real
tree.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import typing as typ

import pytest
from coverage_lanes import _lanes
from timeout_budgets import WATCHDOG_VARIABLE
from workflow_support import (
    WorkflowReadError,
    all_workflow_documents,
    read_workflow_document,
)

if typ.TYPE_CHECKING:
    from pathlib import Path

COVERAGE_STEP: typ.Final[str] = (
    "leynos/shared-actions/.github/actions/generate-coverage@abc123"
)


def test_a_missing_workflow_directory_is_refused(tmp_path: Path) -> None:
    """Reading no workflow must fail, not read as no lane to check.

    `Path.glob` yields nothing for a missing path, so the reading
    returned an empty mapping and every lane assertion above it passed
    having read no workflow at all.
    """
    with pytest.raises(WorkflowReadError, match=r"not a directory"):
        all_workflow_documents(tmp_path / "workflows")


def test_a_workflow_path_that_is_a_file_is_refused(tmp_path: Path) -> None:
    """A file where a directory belongs reaches the same empty result.

    Stated apart from the missing case because `Path.glob` gets there by
    a different route, and a guard testing only for existence would pass
    this one.
    """
    not_a_directory = tmp_path / "workflows"
    not_a_directory.write_text("", encoding="utf-8")

    with pytest.raises(WorkflowReadError, match=r"not a directory"):
        all_workflow_documents(not_a_directory)


def test_a_missing_workflow_file_is_refused(tmp_path: Path) -> None:
    """A path that is not there names itself rather than raising OSError."""
    with pytest.raises(WorkflowReadError, match=r"could not be read"):
        read_workflow_document(tmp_path / "ci.yml")


def test_a_workflow_that_is_not_yaml_is_refused(tmp_path: Path) -> None:
    """A parse failure arrives as the boundary's own error, naming the file.

    A contract cannot say anything about a workflow it never saw, and a
    bare `yaml.YAMLError` from inside a query reports the fault without
    saying which file carried it.
    """
    path = tmp_path / "ci.yml"
    path.write_text("jobs: [unclosed\n", encoding="utf-8")

    with pytest.raises(WorkflowReadError, match=r"is not YAML"):
        read_workflow_document(path)


def test_the_lane_reading_takes_its_documents_as_a_parameter() -> None:
    """The derivation reads no file, so a lane can be described at all.

    Every workflow here sets its watchdog on the coverage step, so a
    reading driven only by the real tree agrees with several wrong ones.
    """
    documents = {
        "ci.yml": {
            "jobs": {
                "build-test": {
                    "timeout-minutes": 60,
                    "steps": [
                        {
                            "name": "Test and Measure Coverage",
                            "uses": COVERAGE_STEP,
                            "env": {WATCHDOG_VARIABLE: "2400"},
                        }
                    ],
                }
            }
        }
    }

    lanes = tuple(_lanes(documents))

    assert [(lane.workflow, lane.step, lane.watchdog) for lane in lanes] == [
        ("ci.yml", "Test and Measure Coverage", 2400.0)
    ], f"the injected document must be the only thing read; got {lanes}"
