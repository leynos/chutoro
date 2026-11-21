"""Read a CI step's script for the commands it actually runs.

The ``build-test`` job installs ``cargo-nextest`` and then runs the scheduling
target. A step can name both and run neither: a command can be printed, or
commented, or its failure masked, a quoted operator can pass for one, or a
later ``||`` or a later line can swallow the status of an ``&&`` chain. Reading
the script as shell does — simple commands, grouped by line, each with the
operator that ends it — is what makes "does this step install nextest, and
would that installation fail the step?" a question about execution rather than
about a substring.

This module is the reading of whether a command installs, and where in its
line it stands. Splitting each line into simple commands is
``scheduling_segment_reading``'s reading, and ``scheduling_ci_test`` holds the
assertions over both, so no file outgrows the 400-line limit ``AGENTS.md``
sets.

Re-use policy: import from ``scheduling_ci_test`` only; this module reuses the
segment reading in ``scheduling_segment_reading`` and the ``errexit`` and
AND/OR readings in ``scheduling_errexit_reading``. A command shape that needs
a new rule belongs here with the shell behaviour it models stated beside it,
so a later reader can check the rule against the shell rather than against the
tests.
"""

from __future__ import annotations

import typing as typ

from scheduling_errexit_reading import (
    set_arguments,
    CARRYING_OPERATOR,
    OWN_LIST_END,
    carries_to_the_step,
    errexit_states,
)
from scheduling_segment_reading import (
    Segment,
    has_a_compound_command,
    read_lines,
)

#: The script that installs ``cargo-nextest`` for the job.
NEXTEST_INSTALLER = "scripts/install-nextest.sh"

#: Words that run the command after them. ``bash scripts/install-nextest.sh``
#: installs; ``echo scripts/install-nextest.sh`` only prints the path.
INTERPRETERS = frozenset({"bash", "sh", "dash"})

#: Operators after which a command is certain to run. Only ``;`` starts a
#: fresh command list unconditionally. A command after ``&&`` runs only when
#: everything before it succeeded, so ``false && installer`` never installs;
#: a command after ``||``, ``|``, ``&`` or ``|&`` may be skipped or deferred,
#: as ``true || installer`` shows.
_UNCONDITIONAL_BEFORE = frozenset({OWN_LIST_END})

def _reaches_the_installer(line: list[Segment], index: int) -> bool:
    """Return whether the line is certain to run `line[index]`.

    A command after ``||``, ``|``, ``&`` or ``|&`` may be skipped while the
    line still carries on: ``true || installer`` never installs anything,
    because ``true`` succeeds. A command after ``&&`` is just as uncertain:
    ``false && installer`` never installs either. Only ``;`` starts a fresh
    command list that runs unconditionally, so ``false && a ; installer``
    does reach the installer, while ``a ; false && installer`` does not.

    Examples
    --------
    >>> _reaches_the_installer(read_lines("a")[0], 0)
    True
    >>> _reaches_the_installer(read_lines("a; b")[0], 1)
    True
    >>> _reaches_the_installer(read_lines("false && a ; b")[0], 2)
    True
    >>> _reaches_the_installer(read_lines("a ; false && b")[0], 2)
    False
    >>> _reaches_the_installer(read_lines("a && b")[0], 1)
    False
    >>> _reaches_the_installer(read_lines("a || b")[0], 1)
    False
    >>> _reaches_the_installer(read_lines("a | b")[0], 1)
    False
    """
    if index == 0:
        return True
    return line[index - 1].follows in _UNCONDITIONAL_BEFORE


def _status_reaches_the_end(line: list[Segment], index: int) -> bool:
    """Return whether a failure of `line[index]` leaves the line failing.

    A line is an AND/OR list, and only ``&&`` passes a failure along it. The
    installer's own operator decides what happens next, but so does every
    operator after it: ``installer && echo installed || true`` records the
    installation attempt and still reports success, so reading the installer's
    operator alone accepted a masked line. A ``;`` is different: it ends the
    installer's own list, and a failing simple command in its own list is not
    exempt from ``errexit``, so ``installer ; make test`` and a trailing
    ``installer ;`` both stop the step when the installer fails. The whole
    line's end has to be read too: a trailing ``&`` backgrounds the last
    command, so the shell moves on before it finishes and its failure belongs
    to no one. That holds whether the backgrounded command is the installer
    (``installer &``) or a later one (``installer && echo done &``), which is
    why the check reads the final segment rather than the installer's own.

    Examples
    --------
    >>> _status_reaches_the_end(read_lines("a && b")[0], 0)
    True
    >>> _status_reaches_the_end(read_lines("a || b")[0], 0)
    False
    >>> _status_reaches_the_end(read_lines("a && b || c")[0], 0)
    False
    >>> _status_reaches_the_end(read_lines("a ; b")[0], 0)
    True
    >>> _status_reaches_the_end(read_lines("a ;")[0], 0)
    True
    >>> _status_reaches_the_end(read_lines("a && b")[0], 1)
    True
    >>> _status_reaches_the_end(read_lines("a &")[0], 0)
    False
    >>> _status_reaches_the_end(read_lines("a && b &")[0], 0)
    False
    """
    if line[index].follows == OWN_LIST_END:
        return True
    if line[index].follows not in {"", CARRYING_OPERATOR}:
        return False
    if line[-1].follows:
        return False
    return all(
        segment.follows == CARRYING_OPERATOR for segment in line[index:-1]
    )


def _takes_an_option_operand(flag: str) -> bool:
    """Return whether `flag` takes an option name in the word after it.

    ``-o``, ``+o``, ``-O`` and ``+O`` do, and so does a cluster whose last
    letter is one of them, as ``-euo`` does for ``pipefail``.

    Examples
    --------
    >>> _takes_an_option_operand("-o")
    True
    >>> _takes_an_option_operand("+O")
    True
    >>> _takes_an_option_operand("-e")
    False
    """
    return flag[:1] in {"-", "+"} and flag[1:].endswith(("o", "O"))


def _enables_noexec(words: list[str]) -> bool:
    """Return whether a ``set`` command leaves ``noexec`` on.

    A non-interactive Bash under ``noexec`` reads the script without running
    it, so an installer after the change installs nothing. The option is set
    as ``-n``, inside a cluster such as ``-eun``, or by name as
    ``-o noexec``, and turned off again by the ``+`` forms.

    Examples
    --------
    >>> _enables_noexec(["set", "-n"])
    True
    >>> _enables_noexec(["set", "-eun", "pipefail"])
    True
    >>> _enables_noexec(["set", "-o", "noexec"])
    True
    >>> _enables_noexec(["set", "-eo", "noexec"])
    True
    >>> _enables_noexec(["set", "-", "-n"])
    False
    >>> _enables_noexec(["builtin", "set", "-n"])
    True
    >>> _enables_noexec(["set", "+o", "noexec"])
    False
    >>> _enables_noexec(["set", "--", "-n"])
    False
    >>> _enables_noexec(["true"])
    False
    """
    words = set_arguments(words)
    if words[:1] != ["set"]:
        return False
    noexec = False
    index = 1
    while index < len(words):
        flag = words[index]
        if flag in {"--", "-"} or flag[:1] not in {"-", "+"}:
            break
        names_noexec = (
            flag[1:].endswith("o")
            and words[index + 1 : index + 2] == ["noexec"]
        )
        if names_noexec:
            noexec = flag[:1] == "-"
        elif "n" in flag[1:]:
            noexec = flag[:1] == "-"
        index += 2 if _takes_an_option_operand(flag) else 1
    return noexec


def _stops_the_script(lines: list[list[Segment]]) -> tuple[int, int] | None:
    """Return the first command that ends the script, if one can.

    ``exit`` ends the shell wherever it runs, so no command after it runs at
    all and none of them can install anything. Every ``exit`` is treated as
    ending the script: one inside a branch that may not be taken still leaves
    a later command's execution unresolved, and reading it as unresolved is
    what keeps the answer evidence rather than a guess. A ``set`` that turns
    on ``noexec`` ends the script's execution the same way: the shell reads
    the rest without running any of it. ``return`` and a sourced script's
    ``exit`` have the same power, but neither appears in this job's step, so
    the reading names only the words that do.

    Examples
    --------
    >>> _stops_the_script(read_lines("a" + chr(10) + "exit 0"))
    (1, 0)
    >>> _stops_the_script(read_lines("a" + chr(10) + "b")) is None
    True
    >>> _stops_the_script(read_lines("a ; exit 0")) == (0, 1)
    True
    >>> _stops_the_script(read_lines("set -n" + chr(10) + "a")) == (0, 0)
    True
    """
    for number, line in enumerate(lines):
        for index, segment in enumerate(line):
            if _ends_execution(segment):
                return number, index
    return None


def _ends_execution(segment: Segment) -> bool:
    """Return whether `segment` ends the script's execution.

    ``exit`` runs nowhere after itself, and a ``set`` that turns on ``noexec``
    leaves the shell reading without running the rest.

    Examples
    --------
    >>> _ends_execution(Segment(["exit", "0"], ""))
    True
    >>> _ends_execution(Segment(["set", "-n"], ""))
    True
    >>> _ends_execution(Segment(["set", "+n"], ""))
    False
    >>> _ends_execution(Segment(["true"], ""))
    False
    """
    return segment.words[:1] == ["exit"] or _enables_noexec(segment.words)


def _runs_installer(words: list[str]) -> bool:
    """Return whether `words` run the installer as a command of their own.

    A shell may run the script, but then the script is the only thing the
    shell is given. ``bash -n scripts/install-nextest.sh`` only parses it, so
    a path anywhere in the argument list read a syntax check as an
    installation.

    Examples
    --------
    >>> _runs_installer(["scripts/install-nextest.sh"])
    True
    >>> _runs_installer(["bash", "scripts/install-nextest.sh"])
    True
    >>> _runs_installer(["bash", "-n", "scripts/install-nextest.sh"])
    False
    """
    if not words:
        return False
    if words[0] == NEXTEST_INSTALLER:
        return True
    return words[0] in INTERPRETERS and words[1:] == [NEXTEST_INSTALLER]


def _counts_as_installation(
    lines: list[list[Segment]],
    states: list[list[bool | None]],
    number: int,
    index: int,
) -> bool:
    """Return whether ``lines[number][index]`` installs nextest, blocking.

    The command has to be a listing of the installer, not a mention of it; it
    has to be certain to run and to run in the foreground; and its failure has
    to be allowed to fail the step, which also means the shell has not turned
    ``errexit`` off before it. ``errexit`` counts only when it is certainly
    on: a state that an untaken branch may have changed is not evidence, so
    the check asks ``is True`` rather than trusting a truthy value. Each
    clause delegates to the function beside it, which carries the shell
    behaviour that decides the answer.

    Examples
    --------
    >>> lines = read_lines("installer")
    >>> _counts_as_installation(lines, [[True]], 0, 0)
    False
    >>> lines = read_lines("scripts/install-nextest.sh")
    >>> _counts_as_installation(lines, [[True]], 0, 0)
    True
    >>> _counts_as_installation(lines, [[False]], 0, 0)
    False
    >>> _counts_as_installation(lines, [[None]], 0, 0)
    False
    """
    segment = lines[number][index]
    checks = (
        states[number][index] is True,
        _runs_installer(segment.words),
        _reaches_the_installer(lines[number], index),
        _status_reaches_the_end(lines[number], index),
        carries_to_the_step(lines, number, segment),
    )
    return all(checks)


def installs_nextest(script: str) -> bool:
    """Return whether `script` executes the nextest installer, blocking.

    The script is read as the shell reads it — continuations resolved,
    comments and here-document bodies dropped, quoted text kept as data,
    ``errexit`` tracked from line to line — and each command it runs is asked
    whether it installs. A substring check accepted
    ``echo scripts/install-nextest.sh`` and a commented-out line, so the
    ordering contract could pass without anything being installed; requiring a
    command that runs and that the step hears fail is what makes the answer
    evidence. A script holding a compound command is refused whole, because a
    body the reading cannot place is not evidence either way.

    Examples
    --------
    >>> installs_nextest("scripts/install-nextest.sh")
    True
    >>> installs_nextest("echo scripts/install-nextest.sh")
    False
    >>> installs_nextest("scripts/install-nextest.sh || true")
    False
    >>> installs_nextest("scripts/install-nextest.sh && make test")
    True
    >>> installs_nextest("scripts/install-nextest.sh &")
    False
    >>> installs_nextest("echo '&&' scripts/install-nextest.sh")
    False
    >>> installs_nextest("scripts/install-nextest.sh ; make test")
    True
    >>> installs_nextest(
    ...     "set +e" + chr(10) + NEXTEST_INSTALLER + chr(10) + "true"
    ... )
    False
    >>> installs_nextest(
    ...     "set +e" + chr(10) + "set -e" + chr(10) + NEXTEST_INSTALLER
    ... )
    True
    >>> installs_nextest(
    ...     "cat <<'EOF'" + chr(10) + NEXTEST_INSTALLER + chr(10) + "EOF"
    ... )
    False
    >>> installs_nextest("exit 0" + chr(10) + NEXTEST_INSTALLER)
    False
    >>> installs_nextest(NEXTEST_INSTALLER + chr(10) + "exit 0")
    True
    >>> installs_nextest(
    ...     "if false; then" + chr(10) + NEXTEST_INSTALLER + chr(10) + "fi"
    ... )
    False
    """
    lines = read_lines(script)
    if has_a_compound_command(lines):
        return False
    states = errexit_states(lines)
    stop = _stops_the_script(lines)
    return any(
        _counts_as_installation(lines, states, number, index)
        for number, line in enumerate(lines)
        for index in range(len(line))
        if stop is None or (number, index) < stop
    )
