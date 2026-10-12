"""Contract tests for Dependabot's repository update sources.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEPENDABOT_PATH = REPOSITORY_ROOT / ".github" / "dependabot.yml"
RUST_TOOLCHAIN_MANIFEST = REPOSITORY_ROOT / "rust-toolchain.toml"


@pytest.fixture(scope="module")
def dependabot_config() -> dict[str, Any]:
    """Parse the repository's Dependabot configuration once per module."""
    document = yaml.safe_load(DEPENDABOT_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict), "Dependabot configuration must be a mapping"
    return document


def _assert_rust_toolchain_entry(config: dict[str, Any]) -> dict[str, Any]:
    """Assert the root toolchain update contract and return its entry.

    Examples
    --------
    >>> entry = {
    ...     "package-ecosystem": "rust-toolchain",
    ...     "directory": "/",
    ...     "open-pull-requests-limit": 5,
    ...     "labels": ["dependencies", "rust-toolchain"],
    ...     "schedule": {"interval": "weekly"},
    ... }
    >>> _assert_rust_toolchain_entry({"updates": [entry]})[
    ...     "package-ecosystem"
    ... ]
    'rust-toolchain'
    """
    updates = config.get("updates")
    assert isinstance(updates, list), "Dependabot updates must be a list"
    entries = [
        update
        for update in updates
        if isinstance(update, dict)
        and update.get("package-ecosystem") == "rust-toolchain"
    ]
    assert len(entries) == 1, "expected exactly one rust-toolchain update entry"

    entry = entries[0]
    assert entry.get("directory") == "/", (
        "rust-toolchain updates must target the repository root"
    )
    assert entry.get("open-pull-requests-limit") == 5, (
        "rust-toolchain updates must limit open pull requests to five"
    )
    labels = entry.get("labels")
    assert isinstance(labels, list) and set(labels) == {
        "dependencies",
        "rust-toolchain",
    }, "rust-toolchain updates must carry the dependency and toolchain labels"
    schedule = entry.get("schedule")
    assert isinstance(schedule, dict) and schedule.get("interval") == "weekly", (
        "rust-toolchain updates must run weekly"
    )
    return entry


def test_dependabot_ecosystems_are_unique_and_expected(
    dependabot_config: dict[str, Any],
) -> None:
    """Require one entry each for Cargo, Actions and the Rust toolchain."""
    updates = dependabot_config["updates"]
    ecosystems = [update["package-ecosystem"] for update in updates]

    assert len(ecosystems) == len(set(ecosystems)), (
        "each Dependabot ecosystem must have exactly one update entry"
    )
    assert set(ecosystems) == {"cargo", "github-actions", "rust-toolchain"}


def test_rust_toolchain_entry_tracks_the_root_manifest_weekly(
    dependabot_config: dict[str, Any],
) -> None:
    """Keep the pinned root toolchain manifest under weekly Dependabot updates."""
    _assert_rust_toolchain_entry(dependabot_config)

    assert RUST_TOOLCHAIN_MANIFEST.is_file(), (
        f"Dependabot's root toolchain manifest is missing: {RUST_TOOLCHAIN_MANIFEST}"
    )


@pytest.mark.parametrize(
    ("defect", "expected_message"),
    [
        pytest.param("missing", "exactly one rust-toolchain", id="missing-entry"),
        pytest.param("duplicate", "exactly one rust-toolchain", id="duplicate-entry"),
        pytest.param("directory", "repository root", id="wrong-directory"),
        pytest.param("schedule", "run weekly", id="wrong-schedule"),
        pytest.param("limit", "limit open pull requests to five", id="wrong-limit"),
        pytest.param(
            "missing-label", "dependency and toolchain labels", id="missing-label"
        ),
        pytest.param(
            "wrong-label", "dependency and toolchain labels", id="wrong-label"
        ),
    ],
)
def test_rust_toolchain_contract_rejects_invalid_entries(
    dependabot_config: dict[str, Any], defect: str, expected_message: str
) -> None:
    """Reject missing or malformed toolchain update entries."""
    config = copy.deepcopy(dependabot_config)
    updates = config["updates"]
    entry = next(
        update
        for update in updates
        if update["package-ecosystem"] == "rust-toolchain"
    )

    match defect:
        case "missing":
            config["updates"] = [
                update
                for update in updates
                if update["package-ecosystem"] != "rust-toolchain"
            ]
        case "duplicate":
            updates.append(copy.deepcopy(entry))
        case "directory":
            entry["directory"] = "/crates"
        case "schedule":
            entry["schedule"]["interval"] = "daily"
        case "limit":
            entry["open-pull-requests-limit"] = 4
        case "missing-label":
            entry["labels"].remove("rust-toolchain")
        case "wrong-label":
            entry["labels"][1] = "cargo"
        case _:
            pytest.fail(f"unknown test defect: {defect}")

    with pytest.raises(AssertionError, match=expected_message):
        _assert_rust_toolchain_entry(config)


def test_existing_cargo_and_github_actions_groups_are_preserved(
    dependabot_config: dict[str, Any],
) -> None:
    """Keep the existing Arrow, security and Actions grouping policies."""
    updates = dependabot_config["updates"]
    cargo = next(update for update in updates if update["package-ecosystem"] == "cargo")
    actions = next(
        update for update in updates if update["package-ecosystem"] == "github-actions"
    )

    assert cargo["groups"] == {
        "arrow": {"patterns": ["arrow*", "parquet"]},
        "arrow-security": {
            "applies-to": "security-updates",
            "patterns": ["arrow*", "parquet"],
        },
        "minor-and-patch": {
            "patterns": ["*"],
            "update-types": ["minor", "patch"],
        },
    }
    assert actions["groups"] == {
        "shared-actions": {"patterns": ["leynos/shared-actions*"]},
        "github-actions": {
            "patterns": ["*"],
            "update-types": ["minor", "patch"],
        },
    }
