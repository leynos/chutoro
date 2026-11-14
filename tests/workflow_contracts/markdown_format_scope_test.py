"""Exercise the real Markdown formatter and linter scope.

`make fmt` combines two independent file selectors. `mdtablefix` takes its
Markdown file list from Git, so a `.gitignore` entry keeps a generated tree out
of formatting. `markdownlint-cli2` expands its own `**/*.md` glob and filters it
through the `ignores` list in `.markdownlint-cli2.jsonc`, where `.gitignore` has
no effect. `make markdownlint` builds a third list with `find` and its
`-not -path` rules. The three lists can therefore drift apart, and a generated
tree reached by only one of them is either rewritten by the formatter or
reported by a linter.

These tests copy the committed `.gitignore`, `.markdownlint-cli2.jsonc` and
`Makefile` into a throwaway repository that also holds a generated `.verus/`
tree and an ordinary document, then run the real targets against it. Removing
each exclusion in turn shows the generated file would otherwise be rewritten or
reported, so a test fails when an exclusion stops being load-bearing rather
than only when the file lists drift.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import typing as typ
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = ".markdownlint-cli2.jsonc"
GITIGNORE_NAME = ".gitignore"
MAKEFILE_NAME = "Makefile"
GENERATED_DIR = ".verus"
NESTED_GENERATED_DIR = "nested/.verus"
ORDINARY_MARKDOWN = Path("docs", "ordinary.md")
GENERATED_MARKDOWN = Path(GENERATED_DIR, "generated.md")
NESTED_GENERATED_MARKDOWN = Path(NESTED_GENERATED_DIR, "generated.md")
GENERATED_MARKDOWN_PATHS = (GENERATED_MARKDOWN, NESTED_GENERATED_MARKDOWN)
GENERATED_IGNORE_PATTERN = "**/.verus/**"

#: Files the Makefile reads while parsing. Without them every target stops on
#: the parse-time checks before a recipe can run.
MAKEFILE_SUPPORT_FILES = ("tools/dev-fast/TOOLCHAIN", "tools/kani/VERSION")

#: The linter version the pinned `markdownlint-cli2-action` bundles. The
#: fallback below must run the linter the hosted lint step runs, or the
#: contract would be checked against a different tool.
MDLINT_VERSION = "0.23.2"

#: The Rust and spelling steps are stubbed because the fixture carries only
#: Markdown; the Markdown commands in those same targets run as committed.
CARGO_STUB = "CARGO=true"
SPELLING_STUB = "TYPOS_CONFIG_BUILDER=true"

#: Markdown that both tools must change: the paragraph exceeds the 80-column
#: wrap limit, and the ellipsis is not the typographic character.
UNFORMATTED_MARKDOWN = (
    "# Scope fixture\n"
    "\n"
    "This paragraph is deliberately much longer than eighty columns so the wrap"
    " rule rewrites it during formatting, and an ellipsis follows ...\n"
)


#: A markdownlint diagnostic names the file, line and column before the rule.
#: The linter also echoes the paths it was handed, and an ignored path appears
#: in that echo, so only a positioned line counts as a finding.
_FINDING_RE = re.compile(r"\.md:\d+:\d+")


def _output(completed: subprocess.CompletedProcess[str]) -> str:
    """Return both streams; the linter reports its findings on stderr."""
    return completed.stdout + completed.stderr


def _finding_lines(output: str) -> list[str]:
    """Return only the diagnostic lines from a linter run."""
    return [line for line in output.splitlines() if _FINDING_RE.search(line)]


def _run(command: list[str], repo: Path) -> subprocess.CompletedProcess[str]:
    """Run one command inside the fixture repository."""
    return subprocess.run(
        command,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def _make(
    repo: Path, target: str, *overrides: str
) -> subprocess.CompletedProcess[str]:
    """Run one Makefile target against the fixture repository."""
    return _run(["make", target, CARGO_STUB, *overrides], repo)


def _require(tool: str) -> None:
    """Fail in CI when a tool is missing, and skip on a developer machine.

    Continuous integration installs `mdtablefix` before this suite runs, so a
    missing binary there means the instalment regressed. Skipping would keep
    the gate green while proving nothing.
    """
    if shutil.which(tool) is not None:
        return
    if os.environ.get("CI"):
        pytest.fail(f"{tool} is required in CI but is not on PATH")
    pytest.skip(f"{tool} is not installed")


def _markdownlint_command() -> list[str]:
    """Return the linter invocation, falling back to the pinned version.

    The hosted lint step runs the pinned action, which keeps the command-line
    tool inside the action's own module tree and leaves nothing on `PATH` for a
    later step. Running the same version through `npx` keeps the contract
    exercised in that job instead of quietly skipping it.
    """
    if shutil.which("markdownlint-cli2") is not None:
        return ["markdownlint-cli2"]
    if shutil.which("npx") is not None:
        return ["npx", "--yes", f"markdownlint-cli2@{MDLINT_VERSION}"]
    return []


def _lint_override() -> str:
    """Return the `MDLINT` override naming the linter to run."""
    return f"MDLINT={' '.join(_markdownlint_command())}"


def _markdownlint(repo: Path, *overrides: str) -> subprocess.CompletedProcess[str]:
    """Run `make markdownlint`, which needs the linter on `PATH` or via `npx`."""
    if not _markdownlint_command():
        _require("markdownlint-cli2")
    return _make(repo, "markdownlint", _lint_override(), *overrides)


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


def _load_lint_config(path: Path) -> dict[str, typ.Any]:
    """Return the lint configuration as data, with its comments stripped."""
    return json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))


def _remove_generated_ignore(repo: Path) -> None:
    """Rewrite the fixture's lint configuration without the `.verus` pattern.

    The result is plain JSON: the comments explain the committed file to its
    readers, and this copy only has to be understood by the linter.
    """
    path = repo / CONFIG_NAME
    config = _load_lint_config(path)
    config["ignores"] = [
        pattern for pattern in config["ignores"] if pattern != GENERATED_IGNORE_PATTERN
    ]
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def markdown_scope_repo(tmp_path: Path) -> Path:
    """Build a throwaway Git repository from the committed configuration.

    The generated tree exists both at the root and nested. The nested copy is
    what proves the lint pattern is not anchored to the root; the `find`
    recipes only exclude `./.verus/*`, so a nested tree is covered by the
    pattern alone.
    """
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    for generated_dir in (GENERATED_DIR, NESTED_GENERATED_DIR):
        (repo / generated_dir).mkdir(parents=True)
    for name in (GITIGNORE_NAME, CONFIG_NAME, MAKEFILE_NAME):
        shutil.copyfile(ROOT / name, repo / name)
    for name in MAKEFILE_SUPPORT_FILES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    for path in (ORDINARY_MARKDOWN, *GENERATED_MARKDOWN_PATHS):
        (repo / path).write_text(UNFORMATTED_MARKDOWN, encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "add", GITIGNORE_NAME, str(ORDINARY_MARKDOWN)],
        check=True,
    )
    return repo


def test_make_fmt_reformats_ordinary_markdown_and_spares_generated_verus(
    markdown_scope_repo: Path,
) -> None:
    """`make fmt` rewrites the ordinary document and no generated one."""
    _require("mdtablefix")
    if not _markdownlint_command():
        _require("markdownlint-cli2")
    generated_before = {
        path: (markdown_scope_repo / path).read_bytes()
        for path in GENERATED_MARKDOWN_PATHS
    }
    ordinary = markdown_scope_repo / ORDINARY_MARKDOWN
    original_ordinary = ordinary.read_text(encoding="utf-8")

    completed = _make(markdown_scope_repo, "fmt", _lint_override())

    assert completed.returncode == 0, _output(completed)
    assert ordinary.read_text(encoding="utf-8") != original_ordinary
    for path, content in generated_before.items():
        assert (markdown_scope_repo / path).read_bytes() == content, (
            f"{path} was reformatted"
        )


def test_make_check_fmt_reports_only_ordinary_markdown(
    markdown_scope_repo: Path,
) -> None:
    """The check gate inspects only Markdown that Git does not ignore."""
    _require("mdtablefix")

    completed = _make(markdown_scope_repo, "check-fmt")
    output = _output(completed)

    assert completed.returncode != 0, output
    assert "docs/ordinary.md" in output
    assert "generated.md" not in output


def test_make_markdownlint_reports_only_ordinary_markdown(
    markdown_scope_repo: Path,
) -> None:
    """The lint target reports the ordinary document and no generated one.

    Both generated trees are present. The `find` recipe skips the root one and
    the configuration pattern skips the nested one, so the report mentions
    neither.
    """
    completed = _markdownlint(markdown_scope_repo, SPELLING_STUB)
    findings = _finding_lines(_output(completed))

    assert completed.returncode != 0, findings
    assert any("docs/ordinary.md" in line for line in findings), findings
    assert not any("generated.md" in line for line in findings), findings


def test_check_fmt_reports_generated_markdown_without_the_gitignore_entry(
    markdown_scope_repo: Path,
) -> None:
    """Removing the `.gitignore` entry puts the generated file back in scope."""
    _require("mdtablefix")
    (markdown_scope_repo / GITIGNORE_NAME).write_text("", encoding="utf-8")

    completed = _make(markdown_scope_repo, "check-fmt")

    assert "generated.md" in _output(completed)


def test_markdownlint_reports_generated_markdown_without_the_ignore_pattern(
    markdown_scope_repo: Path,
) -> None:
    """Removing the lint ignore makes the nested generated file a finding.

    The nested copy is the one the target can see: the `find` recipe excludes
    only `./.verus/*`, so the nested tree reaches the linter and only the
    configuration pattern keeps it out of the report.
    """
    _remove_generated_ignore(markdown_scope_repo)

    completed = _markdownlint(markdown_scope_repo, SPELLING_STUB)
    findings = _finding_lines(_output(completed))

    assert any(str(NESTED_GENERATED_MARKDOWN) in line for line in findings), findings


def test_markdownlint_config_excludes_generated_markdown_at_any_depth() -> None:
    """The committed ignore pattern is the one both tools rely on."""
    config = _load_lint_config(ROOT / CONFIG_NAME)

    assert GENERATED_IGNORE_PATTERN in config["ignores"]
    assert "**/*.md" not in config["ignores"]