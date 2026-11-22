"""Read the scheduling gate out of the Makefile and the Rust sources.

The scheduling contract asserts that two Rust regressions run, and that they
keep running. Those assertions are about text this module reads: the
``test-scheduling-contract`` recipe, and the declarations the recipe names.

The reading is separate from the assertions for the same reasons as
``trim_scoring_reading``: it carries its own cases and doctests, and keeping
both in one file pushed it past the 400-line limit ``AGENTS.md`` sets. Reading
one shell command — where its comment ends and whether it runs nextest at all —
lives in ``scheduling_command_reading`` and ``scheduling_invocation_reading``
for the same reason, and reading the target's ``.PHONY`` declaration lives in
``makefile_reading`` because a Make comment and continuation are not shell
syntax.

Nothing here asserts anything. A reader that cannot read its subject raises
rather than returning an empty answer, so a caller never mistakes a reading
that stopped working for one that found nothing.
"""

from __future__ import annotations

import re

from scheduling_comment_reading import strip_shell_comment
from scheduling_invocation_reading import command_arguments, is_nextest_invocation
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

#: The nextest flags that carry a filter expression, in every spelling nextest
#: accepts, including the ``=`` form. Nextest unions two filtersets, so an
#: argument the reader skipped could widen the selection past the one
#: regression an invocation is meant to name. A command carrying one is read
#: in full and rejected when the count is not exactly one, rather than read as
#: if the extra filter were not there.
FILTER_ARGUMENTS = ("-E", "--filter-expr", "--filterset", "--filter")

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
    """
    pattern = re.compile(
        rf"^{re.escape(target)}:[^\n]*\n"
        r"(?P<recipe>(?:[ \t]*#[^\n]*\n|[ \t]+[^\n]*\n|[ \t]*\n)*)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        message = f"{target} is not declared with a recipe"
        raise ValueError(message)
    commands: list[str] = []
    pending = ""
    for raw in match.group("recipe").splitlines():
        command = strip_shell_comment(raw[1:] if raw.startswith("\t") else raw)
        if not command.strip():
            continue
        if not command:
            continue
        if command.endswith("\\"):
            pending = f"{pending}{command[:-1]}"
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
    >>> feature_flags("nextest run > /tmp/nextest-F.log")
    []
    """
    arguments = command_arguments(recipe)
    return [
        flag
        for flag in FEATURE_FLAGS
        if any(
            argument == flag
            or argument.startswith(f"{flag}=")
            or (flag == "-F" and argument.startswith("-F") and argument != "-F")
            for argument in arguments
        )
    ]


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
    would split it and the expression arrives without its quotes. The words
    are read through ``command_arguments``, which drops a redirection and its
    target: a file named ``-Etest(=a)`` is written to, not passed as a filter.

    Examples
    --------
    >>> filter_expressions("nextest run -E 'test(=a)' --no-tests=fail")
    ['test(=a)']
    >>> filter_expressions("nextest run --no-tests=fail")
    []
    >>> filter_expressions("nextest run >log#1 -E 'test(=a)'")
    ['test(=a)']
    >>> filter_expressions("nextest run --filter-expr='test(=a)'")
    ['test(=a)']
    >>> filter_expressions("nextest run --filterset=slow -E 'test(=a)'")
    ['slow', 'test(=a)']
    """
    tokens = command_arguments(command)
    expressions: list[str] = []
    for index, token in enumerate(tokens):
        if not _carries_a_filter(token):
            continue
        value = _filter_value(tokens, index)
        if value is None:
            message = f"{token} is missing its expression in {command!r}"
            raise ValueError(message)
        expressions.append(value)
    return expressions


def _filter_value(tokens: list[str], index: int) -> str | None:
    """Return the value the filter flag at `index` carries, if complete.

    Nextest takes the value as the word after the flag, or after an ``=`` in
    the flag itself. A flag with neither has no value the reader can vouch for.

    Examples
    --------
    >>> _filter_value(["-E", "test(=a)"], 0)
    'test(=a)'
    >>> _filter_value(["--filter-expr=test(=a)"], 0)
    'test(=a)'
    >>> _filter_value(["--filterset=slow"], 0)
    'slow'
    >>> _filter_value(["-Etest(=a)"], 0)
    'test(=a)'
    >>> _filter_value(["-E"], 0) is None
    True
    """
    token = tokens[index]
    if token.startswith("-E") and token != "-E":
        return token[2:]
    if "=" in token:
        return token.split("=", 1)[1]
    return tokens[index + 1] if index + 1 < len(tokens) else None


def _carries_a_filter(token: str) -> bool:
    """Return whether `token` is a nextest flag that carries a filter.

    Nextest accepts a filterset as ``-E expr``, ``--filter-expr expr`` or in
    the ``=`` form of either long flag, and also accepts ``--filterset`` and
    ``--filter``. A reader that skipped an unfamiliar spelling would count the
    named expression alone while nextest ran the union of both.

    Examples
    --------
    >>> _carries_a_filter("-E")
    True
    >>> _carries_a_filter("--filter-expr=test(=a)")
    True
    >>> _carries_a_filter("--filterset")
    True
    >>> _carries_a_filter("-Eall())")
    True
    >>> _carries_a_filter("--no-tests=fail")
    False
    """
    if token.startswith("-E") and token != "-E":
        return True
    name = token.split("=", 1)[0]
    return name in FILTER_ARGUMENTS


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
