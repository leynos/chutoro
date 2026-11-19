"""Read the scheduling gate out of the Makefile and the Rust sources.

The scheduling contract asserts that two Rust regressions run, and that they
keep running. Those assertions are about text this module reads: the
``test-scheduling-contract`` recipe, and the declarations the recipe names.

The reading is separate from the assertions for the same reasons as
``trim_scoring_reading``: it carries its own cases and doctests, and keeping
both in one file pushed it past the 400-line limit ``AGENTS.md`` sets.

Nothing here asserts anything. A reader that cannot read its subject raises
rather than returning an empty answer, so a caller never mistakes a reading
that stopped working for one that found nothing.
"""

from __future__ import annotations

import re
import shlex

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

#: The spellings of the Cargo executable a scheduling invocation may start
#: with. `$(CARGO)` is what the Makefile uses; a literal `cargo` is accepted
#: because it runs the same command. A command that only mentions nextest in
#: an argument, such as `echo $(CARGO) nextest run ...`, runs nothing, and
#: every assertion about its arguments would still pass.
CARGO_COMMANDS = ("$(CARGO)", "cargo")

#: Shell separators that continue a recipe line after the nextest invocation.
#: `|| true` and `; true` make the line succeed after nextest fails, so the
#: target stays green while the regression it exists to run is broken.
SHELL_SEPARATORS = frozenset({";", ";;", "&&", "||", "|", "&"})

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

#: Attributes that register a function with the test harness. Matched on the
#: attribute's path, so `#[rstest(...)]` counts and `#[test_only_helper]` does
#: not.
TEST_ATTRIBUTE = re.compile(r"^#\[\s*(?:test|rstest|tokio::test|async_std::test)\b")


def makefile_text() -> str:
    """Return the Makefile source this contract reads.

    Reading once keeps every assertion looking at the same text, so a failure
    names the file once rather than reporting a diagnostic from a different
    read.
    """
    return MAKEFILE.read_text(encoding="utf-8")


def _strip_shell_comment(line: str) -> str:
    """Return `line` without its trailing shell comment.

    Only an unescaped ``#`` starts a comment, so a literal ``\\#`` survives.
    Comments are removed before any assertion reads a command, because a
    comment that mentioned a flag would otherwise stand in for the flag on an
    invocation that omits it.

    Examples
    --------
    >>> _strip_shell_comment("make test#run the suite")
    'make test'
    """
    comment = re.search(r"(?<!\\)#", line)
    return line if comment is None else line[: comment.start()]


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
        command = _strip_shell_comment(raw.strip())
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


def _shell_tokens(command: str) -> list[str]:
    """Split `command` into words, keeping quoted text whole.

    ``shlex.split`` leaves an unspaced ``||true`` attached to the word before
    it, so a separator check on the resulting tokens misses it. Splitting on
    the separator characters themselves still respects quotes, and leaves a
    word such as ``$(CARGO)`` intact because only separators are punctuation.

    Examples
    --------
    >>> _shell_tokens("$(CARGO) nextest run ||true")
    ['$(CARGO)', 'nextest', 'run', '||', 'true']
    >>> _shell_tokens("nextest run -E 'test(=a) | all()'")
    ['nextest', 'run', '-E', 'test(=a) | all()']
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    return list(lexer)


def is_nextest_invocation(command: str) -> bool:
    """Return whether `command` runs nextest in its executable position.

    A recipe line that only mentions nextest in an argument runs nothing.
    Three other shapes run it and still leave the target green when the
    regression fails: Make's ignore-error prefix, `|| true`, and a second
    command after `;`. Each spelling that is not the plain Cargo command
    leaves the line out of the invocation list, where the
    one-per-regression count rejects it.

    Examples
    --------
    >>> is_nextest_invocation("$(CARGO) nextest run -p chutoro-core --lib")
    True
    >>> is_nextest_invocation("echo $(CARGO) nextest run -p chutoro-core")
    False
    >>> is_nextest_invocation("-$(CARGO) nextest run -p chutoro-core")
    False
    >>> is_nextest_invocation(
    ...     "$(CARGO) nextest run -p chutoro-core || true"
    ... )
    False
    >>> is_nextest_invocation(
    ...     "$(CARGO) nextest run -p chutoro-core ||true"
    ... )
    False
    """
    tokens = _shell_tokens(command)
    if any(token in SHELL_SEPARATORS for token in tokens):
        return False
    return (
        len(tokens) >= 3
        and tokens[0] in CARGO_COMMANDS
        and tokens[1:3] == ["nextest", "run"]
    )


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


def phony_targets(text: str) -> set[str]:
    """Return the names a ``.PHONY`` declaration covers.

    GNU make collapses continued declarations, so the continuation backslashes
    are removed before splitting. A target absent from this set can be skipped
    when a file of the same name exists.
    """
    declared = re.findall(r"^\.PHONY:((?:.*\\\n)*.*)$", text, re.MULTILINE)
    return set(" ".join(declared).replace("\\\n", " ").split())


def invocations_missing_flag(invocations: list[str]) -> list[int]:
    """Return the indices of invocations without ``--no-tests=fail``.

    Examples
    --------
    >>> invocations_missing_flag(
    ...     ["nextest run --no-tests=fail", "nextest run"]
    ... )
    [1]
    """
    return [
        index
        for index, command in enumerate(invocations)
        if "--no-tests=fail" not in command
    ]


def filter_expressions(command: str) -> list[str]:
    """Return every filter expression `command` passes to nextest.

    The recipe quotes each filter, so the command is split the way a shell
    would split it and the expression arrives without its quotes.

    Examples
    --------
    >>> filter_expressions("nextest run -E 'test(=a)' --no-tests=fail")
    ['test(=a)']
    >>> filter_expressions("nextest run --no-tests=fail")
    []
    """
    tokens = shlex.split(command)
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


def declares_test(source: str, name: str) -> bool:
    """Return whether `source` declares `name` as a test.

    A function with the right name is not evidence by itself: the harness only
    runs it when an attribute registers it, and ``--no-tests=fail`` cannot
    fire for a test that was never discovered. The backwards walk stops at the
    first line that is neither an attribute nor a doc comment, so an unrelated
    declaration above the function is not read as its attribute.

    Examples
    --------
    >>> declares_test("#[test]\\nfn verify() {}", "verify")
    True
    >>> declares_test("fn verify() {}", "verify")
    False
    """
    declaration = re.search(rf"fn\s+{re.escape(name)}\s*\(", source)
    if declaration is None:
        return False
    # Start at the declaration's own line: slicing at the `fn` keyword leaves
    # its indentation as a blank-looking line that would end the walk early.
    line_start = source.rfind("\n", 0, declaration.start()) + 1
    for line in reversed(source[:line_start].splitlines()):
        stripped = line.strip()
        if TEST_ATTRIBUTE.match(stripped) is not None:
            return True
        if not stripped.startswith(("#[", "///")):
            return False
    return False