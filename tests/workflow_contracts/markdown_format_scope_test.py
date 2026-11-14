"""Exercise the real Markdown formatter and linter scope.

`make fmt` combines two independent file selectors. `mdtablefix` takes its
Markdown file list from Git, so a `.gitignore` entry keeps a generated tree
out of formatting. `markdownlint-cli2` expands its own `**/*.md` glob and
filters it through the `ignores` list in `.markdownlint-cli2.jsonc`, where
`.gitignore` has no effect. The two lists can therefore drift apart, and a
generated tree reached by only one of them is either rewritten by the
formatter or reported by the linter.

These tests build a throwaway Git repository that contains a generated
`.verus/` tree, both at the root and nested, plus an ordinary document. They
then run the real tools against it to prove the generated Markdown is left
untouched and unlinted while ordinary Markdown stays in scope.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / ".markdownlint-cli2.jsonc"
MDTABLEFIX_RULES = ("--wrap", "--renumber", "--breaks", "--ellipsis", "--fences")

#: Generated trees used by the fixture. The nested entry proves the ignore
#: patterns are not anchored to the repository root.
GENERATED_DIRS = (".verus", "nested/.verus")

#: Markdown that both tools must change: the paragraph exceeds the 80-column
#: wrap limit, and the ellipsis is not the typographic character.
UNFORMATTED_MARKDOWN = (
    "# Scope fixture\n"
    "\n"
    "This paragraph is deliberately much longer than eighty columns so the wrap"
    " rule rewrites it during formatting, and an ellipsis follows ...\n"
)


def _require(tool: str) -> None:
    """Skip when an optional Markdown tool is not installed."""
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not installed")


def _run(command: list[str], repo: Path) -> subprocess.CompletedProcess[str]:
    """Run one command inside the fixture repository."""
    return subprocess.run(
        command,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def _string_end(text: str, start: int) -> int:
    """Return the index just past the JSON string that opens at ``start``."""
    index = start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == '"':
            return index + 1
        index += 1
    return len(text)


def _line_comment_end(text: str, start: int) -> int:
    """Return the index where the `//` comment starting at ``start`` ends."""
    newline = text.find("\n", start)
    return len(text) if newline == -1 else newline


def _block_comment_end(text: str, start: int) -> int:
    """Return the index where the `/* */` comment starting at ``start`` ends."""
    comment_end = text.find("*/", start + 2)
    return len(text) if comment_end == -1 else comment_end + 2


def _strip_jsonc_comments(text: str) -> str:
    """Return JSON text with JavaScript comments removed.

    `.markdownlint-cli2.jsonc` carries `//` comments, including the one that
    explains the `.verus` exclusion, so the file is not plain JSON.
    """
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            string_end = _string_end(text, index)
            result.append(text[index:string_end])
            index = string_end
        elif text.startswith("//", index):
            index = _line_comment_end(text, index)
        elif text.startswith("/*", index):
            index = _block_comment_end(text, index)
        else:
            result.append(char)
            index += 1
    return "".join(result)


@pytest.fixture
def markdown_scope_repo(tmp_path: Path) -> Path:
    """Build a throwaway Git repository with generated and ordinary Markdown."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    for generated_dir in GENERATED_DIRS:
        (repo / generated_dir).mkdir(parents=True)
    (repo / ".gitignore").write_text(".verus/\n", encoding="utf-8")
    ordinary = repo / "docs" / "ordinary.md"
    ordinary.write_text(UNFORMATTED_MARKDOWN, encoding="utf-8")
    for generated_dir in GENERATED_DIRS:
        (repo / generated_dir / "generated.md").write_text(
            UNFORMATTED_MARKDOWN,
            encoding="utf-8",
        )
    shutil.copyfile(CONFIG_PATH, repo / CONFIG_PATH.name)
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "add", ".gitignore", str(ordinary)],
        check=True,
    )
    return repo


def _generated_files(repo: Path) -> dict[Path, bytes]:
    """Return the bytes of every generated Markdown file before formatting."""
    paths = (repo / generated_dir / "generated.md" for generated_dir in GENERATED_DIRS)
    return {path: path.read_bytes() for path in paths}


def test_mdtablefix_formats_ordinary_markdown_and_spares_generated_verus(
    markdown_scope_repo: Path,
) -> None:
    """The formatter changes the ordinary document and no generated one."""
    _require("mdtablefix")
    before = _generated_files(markdown_scope_repo)
    ordinary = markdown_scope_repo / "docs" / "ordinary.md"
    original_ordinary = ordinary.read_text(encoding="utf-8")

    completed = _run(
        ["mdtablefix", "--in-place", "--git", "--include-untracked", *MDTABLEFIX_RULES],
        markdown_scope_repo,
    )

    assert completed.returncode == 0, completed.stderr
    assert ordinary.read_text(encoding="utf-8") != original_ordinary
    for path, content in before.items():
        assert path.read_bytes() == content, f"{path} was reformatted"


def test_mdtablefix_check_reports_only_ordinary_markdown(
    markdown_scope_repo: Path,
) -> None:
    """The check gate inspects only Markdown that Git does not ignore."""
    _require("mdtablefix")

    completed = _run(
        ["mdtablefix", "--check", "--git", "--include-untracked", *MDTABLEFIX_RULES],
        markdown_scope_repo,
    )

    assert completed.returncode == 1
    output = completed.stdout + completed.stderr
    assert "docs/ordinary.md" in output
    assert "generated.md" not in output


def test_markdownlint_cli2_ignores_generated_verus_markdown(
    markdown_scope_repo: Path,
) -> None:
    """The configured ignores keep generated Markdown out of the lint run."""
    _require("markdownlint-cli2")

    completed = _run(
        [
            "markdownlint-cli2",
            "--config",
            str(markdown_scope_repo / CONFIG_PATH.name),
            "**/*.md",
        ],
        markdown_scope_repo,
    )

    assert completed.returncode == 1
    assert "Linting: 1 file" in completed.stdout
    output = completed.stdout + completed.stderr
    assert "docs/ordinary.md" in output
    assert "generated.md" not in output


def test_markdownlint_config_excludes_generated_verus_at_any_depth() -> None:
    """The committed ignore pattern is the one both tools rely on."""
    config = json.loads(_strip_jsonc_comments(CONFIG_PATH.read_text(encoding="utf-8")))

    assert "**/.verus/**" in config["ignores"]
    assert "**/*.md" not in config["ignores"]