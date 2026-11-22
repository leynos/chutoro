"""Read one Make target's recipe as the commands Make runs.

A recipe line begins with one tab; the rest is the command, and further
indentation is shell text. A backslash-newline pair joins the next line before
the shell reads either, so a comment is stripped from the joined text and a
``#`` that continues a word stays in the command. A target declared inside a
definition or an inactive conditional has no recipe Make will run, so its
header is refused.

This module holds that reading; ``scheduling_reading`` applies it to the
scheduling target, and the split keeps both files inside the 400-line limit
``AGENTS.md`` sets.
"""

from __future__ import annotations

import re

from makefile_structure_reading import declaring_lines
from scheduling_comment_reading import strip_shell_comment


def recipe_commands(text: str, target: str) -> list[str]:
    """Return one target's recipe as logical shell commands.

    Backslash continuations are joined into a single command, shell comments
    are removed, and blank lines are dropped. Make ignores a blank line inside
    a recipe, so the recipe does not end at one: two invocations separated by
    a blank line are still both commands of the target. A continuation deletes
    the backslash and the newline with nothing inserted, so a word split across
    the pair is rejoined rather than broken. Commands are returned rather than
    the joined recipe so a caller can check an invocation on its own; counting
    matches across the whole recipe would accept one command that supplies a
    match twice beside another that supplies none.

    Examples
    --------
    >>> recipe_commands("build:\\n\\tcargo build --release\\n", "build")
    ['cargo build --release']
    >>> recipe_commands("build:\\n\\t# cargo build\\n", "build")
    []
    >>> recipe_commands("build:\\n\\tcargo build\\n\\n\\tcargo test\\n", "build")
    ['cargo build', 'cargo test']
    >>> split = "build:" + chr(10) + chr(9) + "run --no-" + chr(92) + chr(10)
    >>> recipe_commands(split + chr(9) + "run" + chr(10), "build")
    ['run --no-run']
    >>> joined = "build:" + chr(10) + chr(9) + "run " + chr(92) + chr(10)
    >>> recipe_commands(joined + chr(9) + "--release" + chr(10), "build")
    ['run --release']
    >>> spaced = "build:" + chr(10) + chr(9) + "run --no-test=" + chr(92)
    >>> recipe_commands(spaced + chr(10) + chr(9) + "    fail" + chr(10), "build")
    ['run --no-test= fail']
    >>> masked = "build:" + chr(10) + chr(9) + "run >log" + chr(92)
    >>> recipe_commands(masked + chr(10) + chr(9) + "#log || true" + chr(10), "build")
    ['run >log#log || true']
    >>> unused = "define UNUSED" + chr(10) + "build:" + chr(10) + "endef" + chr(10)
    >>> recipe_commands(unused, "build")
    Traceback (most recent call last):
        ...
    ValueError: build is not declared with a recipe
    """
    pattern = re.compile(
        rf"^{re.escape(target)}:[^\n]*\n"
        r"(?P<recipe>(?:[ \t]*#[^\n]*\n|[ \t]+[^\n]*\n|[ \t]*\n)*)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None or not _declares(text, target):
        message = f"{target} is not declared with a recipe"
        raise ValueError(message)
    return _recipe_commands(match.group("recipe"))


def _declares(text: str, target: str) -> bool:
    """Return whether `target` is declared outside a definition or conditional.

    Examples
    --------
    >>> _declares("build:" + chr(10) + chr(9) + "run" + chr(10), "build")
    True
    >>> _declares("define UNUSED" + chr(10) + "build:" + chr(10) + "endef", "b")
    False
    """
    return any(
        line.startswith(f"{target}:") for line in declaring_lines(text)
    )


def _recipe_commands(recipe: str) -> list[str]:
    """Return a recipe's commands, resolving continuations.

    A continuation joins the next physical line before the shell is read, so
    the comment is stripped from the joined text: a ``#`` that continues a
    redirection's word stays in the command, and a mask after it stays
    visible.

    Examples
    --------
    >>> _recipe_commands(chr(9) + "run" + chr(10))
    ['run']
    >>> masked = chr(9) + "run >log" + chr(92) + chr(10) + chr(9) + "#log"
    >>> _recipe_commands(masked)
    ['run >log#log']
    """
    commands: list[str] = []
    pending = ""
    for raw in recipe.splitlines():
        recipe_line = raw[1:] if raw.startswith("\t") else raw
        merged = f"{pending}{recipe_line}"
        continues = merged.endswith("\\")
        command = strip_shell_comment(merged[:-1] if continues else merged)
        if continues:
            pending = command
            continue
        if command.strip():
            commands.append(" ".join(command.split()))
        pending = ""
    if pending.strip():
        commands.append(" ".join(pending.split()))
    return commands


def target_recipe(text: str, target: str) -> str:
    """Return the tab-indented recipe of one Make target.

    The target line itself may carry a ``##`` help comment. Recipe lines are
    the tab-indented lines that follow, backslash continuations included and
    shell comments removed; a missing target or an empty recipe fails rather
    than returning no recipe, because an empty string would satisfy every
    substring check vacuously.

    Examples
    --------
    >>> target_recipe("build:\\n\\tcargo build\\n", "build")
    'cargo build'
    >>> target_recipe("build:\\n", "build")
    Traceback (most recent call last):
    ...
    ValueError: build has no recipe
    """
    commands = recipe_commands(text, target)
    if not commands:
        message = f"{target} has no recipe"
        raise ValueError(message)
    return " ".join(commands)
