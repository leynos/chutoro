"""Read a CI step's script for the commands it actually runs.

The ``build-test`` job installs ``cargo-nextest`` and then runs the scheduling
target. A step can name both and run neither: a command can be printed, or
commented, or its failure masked, a quoted operator can pass for one, or a
later ``||`` or a later line can swallow the status of an ``&&`` chain. Reading
the script as shell does — simple commands, grouped by line, each with the
operator that ends it — is what makes "does this step install nextest, and
would that installation fail the step?" a question about execution rather than
about a substring.

This module is the reading. ``scheduling_ci_test`` holds the assertions over
it, so neither file outgrows the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_ci_test`` only; this module reuses the
comment and tokenizing readers in ``scheduling_command_reading`` and the line
joining and quote masking in ``scheduling_line_reading``. A command shape
that needs a new rule belongs here with the shell behaviour it models stated
beside it, so a later reader can check the rule against the shell rather than
against the tests.
"""

from __future__ import annotations

import typing as typ

from scheduling_command_reading import shell_tokens
from scheduling_errexit_reading import errexit_states
from scheduling_line_reading import logical_lines, mask_quotes, restore_quotes

#: The script that installs ``cargo-nextest`` for the job.
NEXTEST_INSTALLER = "scripts/install-nextest.sh"

#: Words that run the command after them. ``bash scripts/install-nextest.sh``
#: installs; ``echo scripts/install-nextest.sh`` only prints the path.
INTERPRETERS = frozenset({"bash", "sh", "dash"})

#: Characters that end one simple command and begin another. ``shlex`` yields
#: any run of them as one token, so ``|&`` arrives whole.
_OPERATOR_CHARS = frozenset(";&|")

#: The only operator that carries the installer's failure to the end of an
#: AND/OR list. ``installer && make test`` fails when the installer fails;
#: every other operator ends the list at the installer, so ``installer ||
#: true`` and ``installer && echo installed || true`` install nothing that can
#: fail the step. A ``;`` is not a mask: it ends the installer's own list, and
#: a failure there is not exempt from ``errexit``.
_CARRYING_OPERATOR = "&&"

#: The operator that ends the installer's own list. A failing simple command
#: in its own list stops the script under ``errexit``, so the installer's
#: status reaches the step whatever later commands would do. This shares the
#: spelling of ``_UNCONDITIONAL_BEFORE`` but answers the other end of the
#: command: whether a failure survives, not whether a command runs.
_OWN_LIST_END = ";"

#: Operators after which a command is certain to run. Only ``;`` starts a
#: fresh command list unconditionally. A command after ``&&`` runs only when
#: everything before it succeeded, so ``false && installer`` never installs;
#: a command after ``||``, ``|``, ``&`` or ``|&`` may be skipped or deferred,
#: as ``true || installer`` shows.
_UNCONDITIONAL_BEFORE = frozenset({";"})


class _Segment(typ.NamedTuple):
    """One simple command and the operator that ends it.

    The operator is kept because it decides what the job sees when the
    command fails: ``installer || true`` runs the installer and reports
    success, so a segment read without its operator cannot say whether the
    installation was allowed to fail.
    """

    words: list[str]
    follows: str


def _segments(tokens: list[str], quoted: list[str]) -> list[_Segment]:
    """Return one line's simple commands, each with the operator that ends it.

    ``shlex`` yields any run of operator characters as one token, so ``|&``
    arrives whole and an unspaced ``||true`` still separates. A word's quoted
    text was set aside before tokenizing and is restored here, so a command is
    read with the words the shell would give it.

    Examples
    --------
    >>> _segments(["a", "&&", "b"], [])
    [_Segment(words=['a'], follows='&&'), _Segment(words=['b'], follows='')]
    >>> _segments(["a", "|&", "b"], [])
    [_Segment(words=['a'], follows='|&'), _Segment(words=['b'], follows='')]
    """
    segments: list[_Segment] = []
    words: list[str] = []
    for token in tokens:
        if set(token) <= _OPERATOR_CHARS:
            if words:
                segments.append(_Segment(words, token))
            words = []
            continue
        words.append(restore_quotes(token, quoted))
    if words:
        segments.append(_Segment(words, ""))
    return segments


def _lines(script: str) -> list[list[_Segment]]:
    """Return each line of `script` as the simple commands it runs.

    A step's script can hold several commands, separated by newlines or by
    shell operators, and a comment line names a script without running it.
    Splitting first is what makes "is the installer executed?" a question
    about a command's first word rather than about a substring. The operator
    that follows each command is retained, because an operator decides what
    the line reports when the command fails. Segments are grouped by line
    because a command's status travels only as far as the end of its own line.

    Lines arrive with their continuations resolved and their comments removed
    by ``scheduling_line_reading``, so a ``#`` inside a redirection stays part
    of its word and a comment line runs nothing. A here-document's body never
    arrives: it is stdin for the command that opened the document, so the
    reader drops it rather than reading it as commands. Each line's quoted
    text is masked, so a quoted operator stays an argument, and the lexer's
    own comment handling is turned off so it cannot cut the line a second time.

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
    >>> [segment.words for segment in _lines("echo '&&' installer")[0]]
    [['echo', '&&', 'installer']]
    >>> [segment.words for segment in _lines("installer >log#1 || true")[0]]
    [['installer', '>log#1'], ['true']]
    >>> [segment.words for segment in _lines("installer # don't")[0]]
    [['installer']]
    >>> [segment.words for segment in _lines("true ||" + chr(10) + "installer")[0]]
    [['true'], ['installer']]
    >>> [segment.follows for segment in _lines("true ||" + chr(10) + "installer")[0]]
    ['||', '']
    >>> escaped = "echo " + chr(92) + "&" + chr(92) + "& installer"
    >>> [segment.words for segment in _lines(escaped)[0]]
    [['echo', '&&', 'installer']]
    >>> len(_lines(escaped)[0])
    1
    """
    lines: list[list[_Segment]] = []
    for source_line in logical_lines(script):
        masked = mask_quotes(source_line)
        tokens = shell_tokens(masked.text)
        lines.append(_segments(tokens, masked.quoted))
    return lines


def _reaches_the_installer(line: list[_Segment], index: int) -> bool:
    """Return whether the line is certain to run `line[index]`.

    A command after ``||``, ``|``, ``&`` or ``|&`` may be skipped while the
    line still carries on: ``true || installer`` never installs anything,
    because ``true`` succeeds. A command after ``&&`` is just as uncertain:
    ``false && installer`` never installs either. Only ``;`` starts a fresh
    command list that runs unconditionally, so ``false && a ; installer``
    does reach the installer, while ``a ; false && installer`` does not.

    Examples
    --------
    >>> _reaches_the_installer(_lines("a")[0], 0)
    True
    >>> _reaches_the_installer(_lines("a; b")[0], 1)
    True
    >>> _reaches_the_installer(_lines("false && a ; b")[0], 2)
    True
    >>> _reaches_the_installer(_lines("a ; false && b")[0], 2)
    False
    >>> _reaches_the_installer(_lines("a && b")[0], 1)
    False
    >>> _reaches_the_installer(_lines("a || b")[0], 1)
    False
    >>> _reaches_the_installer(_lines("a | b")[0], 1)
    False
    """
    if index == 0:
        return True
    return line[index - 1].follows in _UNCONDITIONAL_BEFORE


def _status_reaches_the_end(line: list[_Segment], index: int) -> bool:
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
    >>> _status_reaches_the_end(_lines("a && b")[0], 0)
    True
    >>> _status_reaches_the_end(_lines("a || b")[0], 0)
    False
    >>> _status_reaches_the_end(_lines("a && b || c")[0], 0)
    False
    >>> _status_reaches_the_end(_lines("a ; b")[0], 0)
    True
    >>> _status_reaches_the_end(_lines("a ;")[0], 0)
    True
    >>> _status_reaches_the_end(_lines("a && b")[0], 1)
    True
    >>> _status_reaches_the_end(_lines("a &")[0], 0)
    False
    >>> _status_reaches_the_end(_lines("a && b &")[0], 0)
    False
    """
    if line[index].follows == _OWN_LIST_END:
        return True
    if line[index].follows not in {"", _CARRYING_OPERATOR}:
        return False
    if line[-1].follows:
        return False
    return all(
        segment.follows == _CARRYING_OPERATOR for segment in line[index:-1]
    )


def _ends_the_script(lines: list[list[_Segment]], index: int) -> bool:
    """Return whether ``lines[index]`` is the script's last command line.

    ``errexit`` does not exit for a command that fails inside an AND/OR list
    unless it is the command after the final ``&&``. The installer's failure in
    ``installer && echo installed`` is therefore exempt, and the line reports
    it only through its own status: a later line that succeeds replaces that
    status, so the step stays green. Only the script's last command line
    carries it out to the step, and lines with no command — blanks and
    comments — are not later commands.

    Examples
    --------
    >>> _ends_the_script(_lines("a && b"), 0)
    True
    >>> _ends_the_script(_lines("a && b\\ntrue"), 0)
    False
    >>> _ends_the_script(_lines("a && b\\n# done"), 0)
    True
    """
    commands = [number for number, line in enumerate(lines) if line]
    return bool(commands) and commands[-1] == index


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


def _carries_to_the_step(
    lines: list[list[_Segment]], number: int, segment: _Segment
) -> bool:
    """Return whether a failure of `segment` still fails the step.

    A command that fails inside an AND/OR list is exempt from ``errexit``
    unless it is the command after the final ``&&``, so the installer's
    failure in an ``&&`` chain reaches the step only through the line's own
    status. A later line that succeeds replaces that status, so such a chain
    also has to be the script's last command line. Every other operator leaves
    the failure in the installer's own list, where nothing exempts it.

    Examples
    --------
    >>> _carries_to_the_step(_lines("a && b"), 0, _lines("a && b")[0][0])
    True
    >>> _carries_to_the_step(
    ...     _lines("a && b\\ntrue"), 0, _lines("a && b")[0][0]
    ... )
    False
    >>> _carries_to_the_step(_lines("a ; b"), 0, _lines("a ; b")[0][0])
    True
    """
    if segment.follows == _CARRYING_OPERATOR:
        return _ends_the_script(lines, number)
    return True


def _counts_as_installation(
    lines: list[list[_Segment]],
    states: list[list[bool]],
    number: int,
    index: int,
) -> bool:
    """Return whether ``lines[number][index]`` installs nextest, blocking.

    The command has to be a listing of the installer, not a mention of it; it
    has to be certain to run and to run in the foreground; and its failure has
    to be allowed to fail the step, which also means the shell has not turned
    ``errexit`` off before it. Each clause delegates to the function beside
    it, which carries the shell behaviour that decides the answer.

    Examples
    --------
    >>> lines = _lines("installer")
    >>> _counts_as_installation(lines, [[True]], 0, 0)
    False
    >>> lines = _lines("scripts/install-nextest.sh")
    >>> _counts_as_installation(lines, [[True]], 0, 0)
    True
    >>> _counts_as_installation(lines, [[False]], 0, 0)
    False
    """
    segment = lines[number][index]
    return (
        states[number][index]
        and _runs_installer(segment.words)
        and _reaches_the_installer(lines[number], index)
        and _status_reaches_the_end(lines[number], index)
        and _carries_to_the_step(lines, number, segment)
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
    evidence.

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
    """
    lines = _lines(script)
    states = errexit_states(
        [[segment.words for segment in line] for line in lines]
    )
    return any(
        _counts_as_installation(lines, states, number, index)
        for number, line in enumerate(lines)
        for index in range(len(line))
    )
