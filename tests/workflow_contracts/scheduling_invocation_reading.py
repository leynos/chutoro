"""Read one shell command for the nextest invocation it runs.

A Makefile recipe line and a CI step are shell command lines, and a line that
mentions nextest is not a line that runs it. ``echo $(CARGO) nextest run``
prints the command, ``nextest run || true`` runs it and reports success
whatever it did, and ``nextest run --no-run`` compiles it without executing a
test. The shell-level reading of quotes, comments, and words lives in
``scheduling_command_reading``; this module reads those words for the
invocation the scheduling gate needs.

Re-use policy: import from ``scheduling_reading`` and
``scheduling_contract_test`` only. The module was split out when the
``AGENTS.md`` 400-line limit was reached, reusing ``shell_tokens`` from
``scheduling_command_reading``. A new rule about which command counts as an
invocation, or about its arguments, belongs here with the shell behaviour it
models stated beside it, so a later reader can check the rule against the
shell rather than against the tests.
"""

from __future__ import annotations

import re

from scheduling_command_reading import is_redirection, shell_tokens

#: The spellings of the Cargo executable a scheduling invocation may start
#: with. `$(CARGO)` is what the Makefile uses; a literal `cargo` is accepted
#: because it runs the same command. A command that only mentions nextest in
#: an argument, such as `echo $(CARGO) nextest run ...`, runs nothing, and
#: every assertion about its arguments would still pass.
CARGO_COMMANDS = ("$(CARGO)", "cargo")

#: Shell separators that continue a recipe line after the nextest invocation.
#: Make runs a recipe line through `sh -c`, whose status is its last command's,
#: so `nextest run ; true` makes the recipe succeed after nextest fails. A
#: `||` after the invocation masks the failure in either a recipe or a CI
#: step, and an `&` backgrounds the invocation and reports success at once. A
#: `;` in a CI step under `errexit` does not mask the failure, but the
#: invocation is then no longer the whole command the line claims to be, and
#: the one-invocation-per-regression count reads each line as the command it
#: runs.
SHELL_SEPARATORS = frozenset({";", ";;", "&&", "||", "|", "&"})

#: Flags that compile without running. `--no-run` builds the tests and executes
#: none of them, so an invocation carrying it produces no runtime evidence even
#: though its selector, count and flags all look right.
COMPILE_ONLY_FLAGS = ("--no-run",)

def _is_redirection(token: str) -> bool:
    """Return whether `token` is a redirection operator.

    Examples
    --------
    >>> _is_redirection(">")
    True
    >>> _is_redirection("2>")
    True
    >>> _is_redirection(">log#1")
    False
    >>> _is_redirection("scheduling>")
    False
    """
    return is_redirection(token)


def command_arguments(command: str) -> list[str]:
    """Return `command`'s words with its redirections and their targets removed.

    A redirection's target is a file name, not an argument. In
    ``nextest run --no-tests=pass > --no-tests=fail`` the only
    ``--no-tests=fail`` is the file written to, and the command still runs
    with ``--no-tests=pass``, so a check that matched the whole word list
    would accept the opposite setting. An attached form such as ``>log#1`` is
    one word and can never be read as a flag, but the spaced form arrives as
    an operator and a separate word, which is why the target is dropped here
    rather than matched as a word.

    Parameters
    ----------
    command
        The command line to read.

    Returns
    -------
    list[str]
        The command's words without redirection operators or their targets.

    Examples
    --------
    >>> command_arguments("nextest run --no-tests=pass > --no-tests=fail")
    ['nextest', 'run', '--no-tests=pass']
    >>> command_arguments("nextest run --no-tests=fail >log#1")
    ['nextest', 'run', '--no-tests=fail', '>log#1']
    >>> command_arguments("nextest run 2> --no-tests=fail --lib")
    ['nextest', 'run', '--lib']
    """
    arguments: list[str] = []
    redirects_next = False
    for token in shell_tokens(command):
        if redirects_next:
            redirects_next = False
            continue
        if _is_redirection(token):
            redirects_next = True
            continue
        arguments.append(token)
    return arguments


def _joins_descriptors(tokens: list[str], index: int) -> bool:
    """Return whether ``tokens[index]`` is the ``&`` of a descriptor copy.

    ``shlex`` splits ``2>&1`` into ``2>``, ``&`` and ``1``. The ``&`` copies a
    descriptor rather than backgrounding the command, so it is not a separator
    that leaves the invocation's failure to no one.

    Examples
    --------
    >>> _joins_descriptors(["true", "2>", "&", "1"], 2)
    True
    >>> _joins_descriptors(["true", "&"], 1)
    False
    """
    if tokens[index] != "&" or index == 0:
        return False
    return is_redirection(tokens[index - 1])


def is_nextest_invocation(command: str) -> bool:
    """Return whether `command` runs nextest in its executable position.

    A recipe line that only mentions nextest in an argument runs nothing.
    Three other shapes run it and still leave the target green when the
    regression fails: Make's ignore-error prefix, `|| true`, and a second
    command after `;`. A fourth runs nothing at all: `--no-run` compiles the
    tests without executing them, so the selector is satisfied and the
    evidence is not produced. Each spelling that is not the plain Cargo
    command leaves the line out of the invocation list, where the
    one-per-regression count rejects it.

    Parameters
    ----------
    command
        The command line to read.

    Returns
    -------
    bool
        Whether the command runs nextest in its executable position.

    Examples
    --------
    >>> is_nextest_invocation("$(CARGO) nextest run -p chutoro-core --lib")
    True
    >>> is_nextest_invocation("echo $(CARGO) nextest run -p chutoro-core")
    False
    >>> is_nextest_invocation("-$(CARGO) nextest run -p chutoro-core")
    False
    >>> is_nextest_invocation("$(CARGO) nextest run -p chutoro-core || true")
    False
    >>> is_nextest_invocation("$(CARGO) nextest run -p chutoro-core ||true")
    False
    >>> is_nextest_invocation("$(CARGO) nextest run -p chutoro-core --no-run")
    False
    >>> is_nextest_invocation("$(CARGO) nextest run -p chutoro-core 2>&1")
    True
    >>> is_nextest_invocation(
    ...     "$(CARGO) nextest run -p chutoro-core > 'scheduling>' & true"
    ... )
    False
    """
    tokens = shell_tokens(command)
    rejected = SHELL_SEPARATORS.union(COMPILE_ONLY_FLAGS)
    if any(
        token in rejected and not _joins_descriptors(tokens, index)
        for index, token in enumerate(tokens)
    ):
        return False
    return (
        len(tokens) >= 3
        and tokens[0] in CARGO_COMMANDS
        and tokens[1:3] == ["nextest", "run"]
    )