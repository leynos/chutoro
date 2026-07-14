"""Test exact phrase-policy enforcement."""

import importlib
from pathlib import Path
import subprocess
import types

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
PROHIBITED = "hand" + "-written"
TITLE_PROHIBITED = "Hand" + "-written"


@pytest.fixture
def checker(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import the standalone phrase checker from the scripts directory."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    importlib.invalidate_caches()
    return importlib.import_module("typos_rollout_check")


def initialize(path: Path, files: dict[str, str]) -> None:
    """Create and stage a small tracked-file fixture."""
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)


def policy_files(*, local_phrase: str = "") -> dict[str, str]:
    """Return minimal generated, shared, and local policy documents."""
    return {
        "typos.toml": (
            f"# Policy for {PROHIBITED} corrections.\n"
            '[files]\nextend-exclude = ["*.md", "!README.md"]\n\n'
            '[default]\nextend-ignore-re = ["`[^`\\\\n]+`"]\n'
        ),
        ".typos-oxendict-base.toml": (
            f'[phrases.corrections]\n"{PROHIBITED}" = "handwritten"\n'
        ),
        "typos.local.toml": local_phrase,
    }


def test_load_policy_combines_phrase_and_generated_scan_policy(
    checker: types.ModuleType, tmp_path: Path
) -> None:
    """Load phrases from shared policy and scan settings from generated TOML."""
    files = policy_files(
        local_phrase='[phrases.corrections]\n"fit-for-purpose" = "suitable"\n'
    )
    initialize(tmp_path, files)

    policy = checker.load_policy(tmp_path)

    assert policy.phrase_corrections == (
        ("fit-for-purpose", "suitable"),
        (PROHIBITED, "handwritten"),
    )
    assert policy.ignore_patterns == (r"`[^`\n]+`",)
    assert policy.excluded_files == ("*.md", "!README.md")

    (tmp_path / ".typos-oxendict-base.toml").unlink()
    with pytest.raises(FileNotFoundError, match="docs/developers-guide.md"):
        checker.load_policy(tmp_path)


def test_checker_preserves_boundaries_masking_and_exclusions(
    checker: types.ModuleType, tmp_path: Path
) -> None:
    """Report phrases only when boundaries and effective policy allow them."""
    initialize(
        tmp_path,
        {
            "README.md": f"{PROHIBITED}\n{TITLE_PROHIBITED} prose\n`{PROHIBITED}`\n",
            "skip.md": f"{PROHIBITED}\n",
            "joined.md": "pre-hand" + "-written\n",
            **policy_files(),
        },
    )

    findings = checker.check_phrase_corrections(tmp_path, checker.load_policy(tmp_path))

    assert [(item.line, item.phrase) for item in findings] == [
        (1, PROHIBITED),
        (2, TITLE_PROHIBITED),
    ]


def test_main_reports_location_and_exit_two(
    checker: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return two and preserve the established path diagnostic."""
    initialize(
        tmp_path,
        {"README.md": f"Prefer {PROHIBITED}.\n", **policy_files()},
    )

    assert checker.main(["--repository", str(tmp_path)]) == 2
    assert capsys.readouterr().out == (f"README.md:1:8: {PROHIBITED} -> handwritten\n")
