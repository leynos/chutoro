"""Protect the existing Cargo and GitHub Actions update group contracts."""

from dependabot_config_test import DEPENDABOT_PATH, read_dependabot_document


def test_existing_cargo_and_github_actions_groups_are_preserved() -> None:
    """Keep the existing Arrow, security and Actions grouping policies."""
    updates = read_dependabot_document(DEPENDABOT_PATH)["updates"]
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
