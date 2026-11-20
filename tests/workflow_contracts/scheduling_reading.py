"""Read the scheduling gate out of the Makefile and the Rust sources.

The scheduling contract asserts that two Rust regressions run, and that they
keep running. Those assertions are about text this module reads: the
``test-scheduling-contract`` recipe, and the declarations the recipe names.

The reading is separate from the assertions for the same reasons as
``trim_scoring_reading``: it carries its own cases and doctests, and keeping
both in one file pushed it past the 400-line limit ``AGENTS.md`` sets. Reading
one shell command — where its comment ends and whether it runs nextest at all —
lives in ``scheduling_command_reading`` for the same reason.

Nothing here asserts anything. A reader that cannot read its subject raises
rather than returning an empty answer, so a caller never mistakes a reading
that stopped working for one that found nothing.
"""

from __future__ import annotations

import re
import shlex

from scheduling_command_reading import (
    command_arguments,
    is_nextest_invocation,
    strip_shell_comment,
)
from workflow_support import ROOT

#: The target that runs both scheduling regressions.
TARGET = "test-scheduling-contract"

#: The Makefile that declares the target.
MAKEFILE = ROOT / "Makefile"

#: The exact test names the filter must select.
REGRESSIONS = (
    "insertion_trim_scoring_runs_on_the_insert_mutex_holder",
    "direct_trim_scoring_stays_on_the_calling_worker",
)

#: The fully qualified nextest names, which the recipes must match exactly.
QUALIFIED_REGRESSIONS = {
    "insertion_trim_scoring_runs_on_the_insert_mutex_holder": (
        "hnsw::cpu::trim_tests::insertion_path::"
        "insertion_trim_scoring_runs_on_the_insert_mutex_holder"
    ),
    "direct_trim_scoring_stays_on_the_calling_worker": (
        "hnsw::cpu::trim_tests::direct_trim_scoring_stays_on_the_calling_worker"
    ),
}

#: The nextest flags that carry a filter expression. An unfamiliar spelling
#: yields no expression, which the caller rejects rather than counting as
#: coverage: a filter the reading cannot see is a filter it cannot vouch for.
FILTER_FLAGS = ("-E", "--filter-expr")

#: Every spelling that selects a feature set. The regressions are
#: feature-independent, so the gate must build the default set: a feature flag
#: would tie the evidence to whatever that set compiles, which is the change
#: the gate exists to notice. `--features` also covers its `=` form, and `-F`
#: is Cargo's short spelling.
FEATURE_FLAGS = (
    "--all-features",
    "--no-default-features",
    "--features",
    "-F",
)


def makefile_text() -> str:
    """Return the Makefile source this contract reads.

    Reading once keeps every assertion looking at the same text, so a failure
    names the file once rather than reporting a diagnostic from a different
    read.
    """
    return MAKEFILE.read_text(encoding="utf-8")


def recipe_commands(text: str, target: str) -> list[str]:
    """Return one target's recipe as logical shell commands.

    Backslash continuations are joined into a single command, shell comments
    are removed, and blank lines are dropped. Commands are returned rather
    than the joined recipe so a caller can check an invocation on its own;
    counting matches across the whole recipe would accept one command that
    supplies a match twice beside another that supplies none.

    Examples
    --------
    >>> recipe_commands("build:\\n\\tcargo build --release\\n", "build")
    ['cargo build --release']
    >>> recipe_commands("build:\\n\\t# cargo build\\n", "build")
    []
    """
    pattern = re.compile(
        rf"^{re.escape(target)}:[^\n]*\n(?P<recipe>(?:[ \t]+[^\n]*\n)*)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        message = f"{target} is not declared with a recipe"
        raise ValueError(message)
    commands: list[str] = []
    pending = ""
    for raw in match.group("recipe").splitlines():
        command = strip_shell_comment(raw.strip())
        if not command:
            continue
        if command.endswith("\\"):
            pending = f"{pending}{command[:-1]} "
            continue
        commands.append(" ".join(f"{pending}{command}".split()))
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


def nextest_invocations() -> list[str]:
    """Return each command the scheduling target actually runs.

    Commands are returned separately so that an assertion about an
    invocation's arguments fails when that invocation alone omits them. Only
    a command that runs nextest counts; a mention, an unrelated script, or an
    error-ignoring prefix is not an invocation.
    """
    return [
        command
        for command in recipe_commands(makefile_text(), TARGET)
        if is_nextest_invocation(command)
    ]


def feature_flags(recipe: str) -> list[str]:
    """Return the feature-selection flags `recipe` carries.

    Which features a gate builds is part of what it proves, so the default set
    is a requirement rather than an incidental property of today's recipe.
    Every spelling is reported, including the `--features=name` form that a
    check for the bare flag alone would miss.

    Examples
    --------
    >>> feature_flags("nextest run -p chutoro-core --lib")
    []
    >>> feature_flags("nextest run --all-features")
    ['--all-features']
    >>> feature_flags("nextest run --features=metrics")
    ['--features']
    """
    return [flag for flag in FEATURE_FLAGS if flag in recipe]


def _without_make_comment(line: str) -> str:
    """Return `line` without its Make comment, keeping an escaped ``#``.

    A Make comment starts at an unescaped ``#`` anywhere on the line, not only
    at a word start as a shell comment does, and ``\\#`` is a literal ``#`` in
    a target name.

    Examples
    --------
    >>> _without_make_comment("clean # tidy up")
    'clean '
    >>> escaped = "clean " + chr(92) + "#1"
    >>> _without_make_comment(escaped) == escaped
    True
    """
    index = 0
    while index < len(line):
        if line[index] == "\\" and index + 1 < len(line):
            index += 2
            continue
        if line[index] == "#":
            return line[:index]
        index += 1
    return line


def _declared_names(line: str) -> list[str]:
    """Return the target names one declaration line adds.

    The continuation backslash ends the line rather than naming a target, and
    an escaped ``#`` names a target that contains one.

    Examples
    --------
    >>> _declared_names(" clean " + chr(92))
    ['clean']
    >>> _declared_names(" clean" + chr(92) + "#1")
    ['clean#1']
    """
    uncommented = _without_make_comment(line).rstrip().removesuffix("\\")
    return uncommented.replace("\\#", "#").split()


def phony_targets(text: str) -> set[str]:
    """Return the names a ``.PHONY`` declaration covers.

    GNU make collapses continued declarations, so each declaration is read
    line by line and a continuation backslash only joins the next line's
    names. Comments are removed first: ``.PHONY: clean # test-scheduling-
    contract`` declares ``clean`` alone, and reading the comment would make a
    same-named file seem covered when the target could still be skipped.

    Examples
    --------
    >>> sorted(phony_targets(".PHONY: clean " + chr(35) + " " + TARGET))
    ['clean']
    >>> declaration = ".PHONY: clean " + chr(92) + chr(10) + " " + TARGET
    >>> sorted(phony_targets(declaration))
    ['clean', 'test-scheduling-contract']
    """
    declared = re.findall(r"^\.PHONY:((?:.*\\\n)*.*)$", text, re.MULTILINE)
    names: set[str] = set()
    for declaration in declared:
        for line in declaration.splitlines():
            names.update(_declared_names(line))
    return names


def invocations_missing_flag(invocations: list[str]) -> list[int]:
    """Return the indices of invocations without ``--no-tests=fail``.

    The flag is matched as a whole argument. A substring search accepted
    ``nextest run --no-tests=pass >--no-tests=fail``, where the only match is
    the name of a redirection file: the command carries the opposite setting,
    so an empty selection would pass the gate. The spaced form redirects just
    as surely as the attached one, so redirection targets are removed before
    the check rather than only being matched whole.

    Examples
    --------
    >>> invocations_missing_flag(
    ...     ["nextest run --no-tests=fail", "nextest run"]
    ... )
    [1]
    >>> invocations_missing_flag(["nextest run --no-tests=pass >--no-tests=fail"])
    [0]
    >>> invocations_missing_flag(["nextest run --no-tests=pass > --no-tests=fail"])
    [0]
    """
    return [
        index
        for index, command in enumerate(invocations)
        if "--no-tests=fail" not in command_arguments(command)
    ]


def filter_expressions(command: str) -> list[str]:
    """Return every filter expression `command` passes to nextest.

    The recipe quotes each filter, so the command is split the way a shell
    would split it and the expression arrives without its quotes. ``shlex``
    treats ``#`` as a comment character by default and would stop at one in a
    word, such as a redirection to ``log#1``, losing every flag after it; the
    command's comments are removed before it is read, so the split keeps them.

    Examples
    --------
    >>> filter_expressions("nextest run -E 'test(=a)' --no-tests=fail")
    ['test(=a)']
    >>> filter_expressions("nextest run --no-tests=fail")
    []
    >>> filter_expressions("nextest run >log#1 -E 'test(=a)'")
    ['test(=a)']
    """
    tokens = shlex.split(command, comments=False)
    expressions: list[str] = []
    for index, token in enumerate(tokens):
        if token not in FILTER_FLAGS:
            continue
        if index + 1 == len(tokens):
            message = f"{token} is missing its expression in {command!r}"
            raise ValueError(message)
        expressions.append(tokens[index + 1])
    return expressions


def selected_regression(command: str) -> str:
    """Return the one regression `command` selects with an exact filter.

    The whole expression is compared, not a substring of it. A substring
    search accepts ``test(=name) | all()``: the expression still selects every
    test after the named one stops matching, so ``--no-tests=fail`` never
    fires and the gate quietly stops covering the regression.

    Parameters
    ----------
    command
        One nextest invocation taken from the target's recipe.

    Examples
    --------
    >>> selected_regression(
    ...     "nextest run -E 'test(=hnsw::cpu::trim_tests::"
    ...     "direct_trim_scoring_stays_on_the_calling_worker)'"
    ... )
    'direct_trim_scoring_stays_on_the_calling_worker'

    Returns
    -------
    str
        The regression the expression names.

    Raises
    ------
    ValueError
        When the command passes anything other than exactly one expression,
        or when that expression is not exactly one regression's selector.
    """
    expressions = filter_expressions(command)
    if len(expressions) != 1:
        message = (
            f"{command!r} must pass exactly one filter expression; "
            f"it passes {expressions}"
        )
        raise ValueError(message)
    expression = "".join(expressions[0].split())
    matches = [
        name
        for name in REGRESSIONS
        if expression == f"test(={QUALIFIED_REGRESSIONS[name]})"
    ]
    if not matches:
        message = (
            f"{command!r} must select exactly one regression; "
            f"it filters on {expressions[0]!r}"
        )
        raise ValueError(message)
    return matches[0]
