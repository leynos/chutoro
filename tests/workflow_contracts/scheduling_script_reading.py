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
    CARRYING_OPERATOR,
    OWN_LIST_END,
    carries_to_the_step,
    errexit_states,
)
from scheduling_segment_reading import Segment, read_lines

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

#: Words that open or close a compound command whose body this reading cannot
#: place. ``if false; then``, ``fi`` and the loop and case keywords are
#: commands of their own to a simple reading, so an installer written inside
#: the body reads exactly like a plain command that certainly ran, while the
#: shell skips the whole body. The reading tracks no block structure, so a
#: script holding one of these yields no evidence at all rather than a guess.
COMPOUND_KEYWORDS = frozenset(
    {
        "if",
        "then",
        "elif",
        "else",
        "fi",
        "for",
        "while",
        "until",
        "do",
        "done",
        "case",
        "esac",
        "select",
        "function",
        "{",
        "}",
        "coproc",
    }
)


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


def _stops_the_script(lines: list[list[Segment]]) -> tuple[int, int] | None:
    """Return the first command that ends the script, if one can.

    ``exit`` ends the shell wherever it runs, so no command after it runs at
    all and none of them can install anything. Every ``exit`` is treated as
    ending the script: one inside a branch that may not be taken still leaves
    a later command's execution unresolved, and reading it as unresolved is
    what keeps the answer evidence rather than a guess. ``return`` and a
    sourced script's ``exit`` have the same power, but neither appears in this
    job's step, so the reading names only the word that does.

    Examples
    --------
    >>> _stops_the_script(read_lines("a" + chr(10) + "exit 0"))
    (1, 0)
    >>> _stops_the_script(read_lines("a" + chr(10) + "b")) is None
    True
    >>> _stops_the_script(read_lines("a ; exit 0")) == (0, 1)
    True
    """
    for number, line in enumerate(lines):
        for index, segment in enumerate(line):
            if segment.words[:1] == ["exit"]:
                return number, index
    return None


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
    return (
        states[number][index] is True
        and _runs_installer(segment.words)
        and _reaches_the_installer(lines[number], index)
        and _status_reaches_the_end(lines[number], index)
        and carries_to_the_step(lines, number, segment)
    )


def _opens_a_compound_command(segment: Segment) -> bool:
    """Return whether `segment` is a keyword this reading cannot place.

    A keyword opens or closes a compound command only in command position, so
    ``echo if`` prints the word rather than opening a branch.

    Examples
    --------
    >>> _opens_a_compound_command(Segment(["if", "false"], ";"))
    True
    >>> _opens_a_compound_command(Segment(["echo", "if"], ""))
    False
    """
    return bool(segment.words) and segment.words[0] in COMPOUND_KEYWORDS


def _has_a_compound_command(lines: list[list[Segment]]) -> bool:
    """Return whether `lines` holds a keyword this reading cannot place.

    A compound command's keywords stand as commands of their own to a simple
    reading, so without this the installer inside an untaken body would count
    as one that certainly ran. The reading has no block structure to say
    which branch was taken, so the whole script is refused instead.

    Examples
    --------
    >>> _has_a_compound_command(read_lines("if false; then" + chr(10) + "fi"))
    True
    >>> _has_a_compound_command(
    ...     read_lines("for job in a b; do" + chr(10) + "done")
    ... )
    True
    >>> _has_a_compound_command(read_lines("true" + chr(10) + "installer"))
    False
    >>> _has_a_compound_command(read_lines("echo if"))
    False
    """
    return any(
        _opens_a_compound_command(segment)
        for line in lines
        for segment in line
    )


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
    if _has_a_compound_command(lines):
        return False
    states = errexit_states(lines)
    stop = _stops_the_script(lines)
    return any(
        _counts_as_installation(lines, states, number, index)
        for number, line in enumerate(lines)
        for index in range(len(line))
        if stop is None or (number, index) < stop
    )
