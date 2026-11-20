"""Read a CI step's script for the commands it actually runs.

The ``build-test`` job installs ``cargo-nextest`` and then runs the scheduling
target. A step can name both and run neither: a command can be printed, or
commented, or its failure masked, or a later ``||`` can swallow the status of
an ``&&`` chain. Reading the script as shell does — simple commands, grouped by
line, each with the operator that ends it — is what makes "does this step
install nextest, and would that installation fail the step?" a question about
execution rather than about a substring.

This module is the reading. ``scheduling_ci_test`` holds the assertions over
it, so neither file outgrows the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_ci_test`` only. A command shape that
needs a new rule belongs here with the shell behaviour it models stated beside
it, so a later reader can check the rule against the shell rather than against
the tests.
"""

from __future__ import annotations

import shlex
import typing as typ

#: The script that installs ``cargo-nextest`` for the job.
NEXTEST_INSTALLER = "scripts/install-nextest.sh"

#: Words that run the command after them. ``bash scripts/install-nextest.sh``
#: installs; ``echo scripts/install-nextest.sh`` only prints the path.
INTERPRETERS = frozenset({"bash", "sh", "dash"})

#: Characters that end one simple command and begin another. ``shlex`` yields
#: any run of them as one token, so ``|&`` arrives whole.
_OPERATOR_CHARS = frozenset(";&|")

#: The only operator that carries the installer's failure to the end of a
#: line. ``installer && make test`` fails when the installer fails; every other
#: operator replaces that status somewhere later on the line, so ``|| true``,
#: ``; true`` and ``installer && echo installed || true`` all install nothing
#: and report success.
_CARRYING_OPERATOR = "&&"

#: Operators a command may follow and still be certain to run. ``&&`` skips it
#: only when the line has already failed, and ``;`` never skips it; every
#: other operator can skip a command whose predecessors succeeded and carry on
#: as if nothing had happened, as ``true || installer`` does.
_UNCONDITIONAL_BEFORE = frozenset({"&&", ";"})


def _joined_lines(script: str) -> list[str]:
    """Return `script`'s lines with backslash continuations joined.

    A workflow script wraps long commands with a trailing backslash. Passing
    one of those lines to ``shlex`` on its own raises ``No escaped
    character``, because the escape has nothing left to escape, so the
    continuations are resolved first.

    Examples
    --------
    The continuation is built with ``chr(92)`` and ``chr(10)``, because a
    docstring holding those escapes would end the example's line:

    >>> _joined_lines(
    ...     "cargo build " + chr(92) + chr(10) + "  --release"
    ... )
    ['cargo build    --release']
    """
    lines: list[str] = []
    pending = ""
    for raw in script.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            pending = f"{pending}{line[:-1]} "
            continue
        lines.append(f"{pending}{line}")
        pending = ""
    if pending.strip():
        lines.append(pending)
    return lines


class _Segment(typ.NamedTuple):
    """One simple command and the operator that ends it.

    The operator is kept because it decides what the job sees when the
    command fails: ``installer || true`` runs the installer and reports
    success, so a segment read without its operator cannot say whether the
    installation was allowed to fail.
    """

    words: list[str]
    follows: str


def _lines(script: str) -> list[list[_Segment]]:
    """Return each line of `script` as the simple commands it runs.

    A step's script can hold several commands, separated by newlines or by
    shell operators, and a comment line names a script without running it.
    Splitting first is what makes "is the installer executed?" a question
    about a command's first word rather than about a substring. The operator
    that follows each command is retained, because an operator decides what
    the line reports when the command fails. Segments are grouped by line
    because a command's status travels only as far as the end of its own line.

    Examples
    --------
    >>> _lines("echo scripts/install-nextest.sh")
    [[_Segment(words=['echo', 'scripts/install-nextest.sh'], follows='')]]
    >>> [segment.words for segment in _lines("a || true")[0]]
    [['a'], ['true']]
    >>> [segment.follows for segment in _lines("a || true")[0]]
    ['||', '']
    >>> _lines("# scripts/install-nextest.sh")
    [[]]
    """
    lines: list[list[_Segment]] = []
    for source_line in _joined_lines(script):
        lexer = shlex.shlex(source_line, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        segments: list[_Segment] = []
        words: list[str] = []
        for token in lexer:
            if set(token) <= _OPERATOR_CHARS:
                if words:
                    segments.append(_Segment(words, token))
                words = []
                continue
            words.append(token)
        if words:
            segments.append(_Segment(words, ""))
        lines.append(segments)
    return lines


def _reaches_the_installer(line: list[_Segment], index: int) -> bool:
    """Return whether the line is certain to run `line[index]`.

    A command after ``||``, ``|``, ``&`` or ``|&`` may be skipped while the
    line still carries on: ``true || installer`` never installs anything,
    because ``true`` succeeds. A command after ``&&`` runs unless the line has
    already failed, and after ``;`` it always runs, so both are reached.

    Examples
    --------
    >>> _reaches_the_installer(_lines("a")[0], 0)
    True
    >>> _reaches_the_installer(_lines("a && b")[0], 1)
    True
    >>> _reaches_the_installer(_lines("a; b")[0], 1)
    True
    >>> _reaches_the_installer(_lines("a || b")[0], 1)
    False
    >>> _reaches_the_installer(_lines("a | b")[0], 1)
    False
    """
    return all(
        segment.follows in _UNCONDITIONAL_BEFORE for segment in line[:index]
    )


def _status_reaches_the_end(line: list[_Segment], index: int) -> bool:
    """Return whether a failure of `line[index]` leaves the line failing.

    A line is an AND/OR list, and only ``&&`` passes a failure along it. The
    installer's own operator decides what happens next, but so does every
    operator after it: ``installer && echo installed || true`` records the
    installation attempt and still reports success, so reading the installer's
    operator alone accepted a masked line. The whole line's end has to be read
    too: a trailing ``&`` backgrounds the last command, so the shell moves on
    before it finishes and its failure belongs to no one. That holds whether
    the backgrounded command is the installer (``installer &``) or a later one
    (``installer && echo done &``), which is why the check reads the final
    segment rather than the installer's own.

    Examples
    --------
    >>> _status_reaches_the_end(_lines("a && b")[0], 0)
    True
    >>> _status_reaches_the_end(_lines("a || b")[0], 0)
    False
    >>> _status_reaches_the_end(_lines("a && b || c")[0], 0)
    False
    >>> _status_reaches_the_end(_lines("a && b")[0], 1)
    True
    >>> _status_reaches_the_end(_lines("a &")[0], 0)
    False
    >>> _status_reaches_the_end(_lines("a && b &")[0], 0)
    False
    """
    if line[index].follows not in {"", _CARRYING_OPERATOR}:
        return False
    if line[-1].follows:
        return False
    return all(
        segment.follows == _CARRYING_OPERATOR for segment in line[index:-1]
    )


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


def installs_nextest(script: str) -> bool:
    """Return whether `script` executes the nextest installer, blocking.

    The path must be a command's first word, or the argument of a shell
    interpreter. A substring check accepted ``echo scripts/install-nextest.sh``
    and a commented-out line, so the ordering contract could pass without
    anything being installed. The command must also be certain to run, run in
    the foreground, and be allowed to fail the step: an earlier ``||`` can skip
    it, a masked failure installs nothing and reports success, the mask can sit
    anywhere later on the line rather than only immediately after the
    installer, and a trailing ``&`` lets the next command start first.

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
    >>> installs_nextest("scripts/install-nextest.sh && echo done || true")
    False
    >>> installs_nextest("true || scripts/install-nextest.sh")
    False
    >>> installs_nextest("scripts/install-nextest.sh &")
    False
    """
    return any(
        _runs_installer(segment.words)
        and _reaches_the_installer(line, index)
        and _status_reaches_the_end(line, index)
        for line in _lines(script)
        for index, segment in enumerate(line)
    )