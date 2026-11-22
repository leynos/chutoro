"""Read the shell features that stop a step's script being simple commands.

A script that enables ``lastpipe``, installs an ``ERR`` or ``EXIT`` trap, or
holds a compound command reaches commands this reading cannot place: a
pipeline element changes the parent's state, a handler replaces a failure, and
a branch decides whether a command runs at all. Each of those is refused whole
rather than read as if the feature were not there.

This module holds those readings. ``scheduling_script_reading`` applies them
to a step's script, and the split keeps both files inside the 400-line limit
``AGENTS.md`` sets.
"""

from __future__ import annotations

from scheduling_segment_reading import (
    Segment,
    has_a_compound_command,
    read_lines,
)
from scheduling_wrapper_reading import builtin_arguments

#: The ``shopt`` name that runs a pipeline's last element in the parent.
LASTPIPE_OPTION = "lastpipe"


def turns_on_lastpipe(lines: list[list[Segment]]) -> bool:
    """Return whether any command in `lines` turns on ``lastpipe``.

    The reading assumes a pipeline element's state change dies with it, which
    ``lastpipe`` breaks, so a script that enables the option is refused.

    Examples
    --------
    >>> turns_on_lastpipe(read_lines("shopt -s lastpipe"))
    True
    >>> turns_on_lastpipe(read_lines("shopt -u lastpipe"))
    False
    >>> turns_on_lastpipe(read_lines("shopt -s nullglob"))
    False
    """
    return any(
        enables_lastpipe(segment.words)
        for line in lines
        for segment in line
    )


def sets_an_err_trap(lines: list[list[Segment]]) -> bool:
    """Return whether any command installs an ``ERR`` trap.

    Bash runs the handler when a command fails, so ``trap 'exit 0' ERR`` can
    turn an installer failure into a successful exit. The reading cannot tell
    whether a handler preserves the failure, so a script that installs one is
    refused.

    Examples
    --------
    >>> sets_an_err_trap(read_lines("trap 'exit 0' ERR"))
    True
    >>> sets_an_err_trap(read_lines("trap - ERR"))
    False
    >>> sets_an_err_trap(read_lines("trap 'echo x' INT"))
    False
    """
    return any(
        installs_a_trap(segment.words)
        for line in lines
        for segment in line
    )


def installs_a_trap(words: list[str]) -> bool:
    """Return whether `words` install a handler that can replace a failure.

    ``ERR`` handlers run on a failed command, ``DEBUG`` handlers run before
    each one, and ``EXIT`` handlers (or signal ``0``) run as the shell leaves,
    so any of them can turn an installer failure into a successful exit.
    ``trap - ERR`` removes a handler instead.

    Examples
    --------
    >>> installs_a_trap(["trap", "'exit 0'", "ERR"])
    True
    >>> installs_a_trap(["builtin", "trap", "'exit 0'", "ERR"])
    True
    >>> installs_a_trap(["trap", "'exit 0'", "EXIT"])
    True
    >>> installs_a_trap(["trap", "'exit 0'", "0"])
    True
    >>> installs_a_trap(["trap", "'exit 0'", "DEBUG"])
    True
    >>> installs_a_trap(["trap", "-", "ERR"])
    False
    >>> installs_a_trap(["true"])
    False
    """
    words = builtin_arguments(words)
    if words[:1] != ["trap"] or words[1:2] == ["-"]:
        return False
    return any(signal in {"ERR", "EXIT", "DEBUG", "0"} for signal in words[1:])


def cannot_place(lines: list[list[Segment]]) -> bool:
    """Return whether this reading cannot place the script's commands.

    A compound command, a ``lastpipe`` change or an ``ERR`` trap each makes a
    command's reachability depend on Bash semantics the reading does not
    model, so the script yields no evidence at all.

    Examples
    --------
    >>> cannot_place(read_lines("if false; then" + chr(10) + "fi"))
    True
    >>> cannot_place(read_lines("shopt -s lastpipe"))
    True
    >>> cannot_place(read_lines("trap 'exit 0' ERR"))
    True
    >>> cannot_place(read_lines("installer"))
    False
    """
    if has_a_compound_command(lines) or turns_on_lastpipe(lines):
        return True
    return sets_an_err_trap(lines)


def enables_lastpipe(words: cabc.Sequence[str]) -> bool:
    """Return whether a ``shopt`` command turns on ``lastpipe``.

    With job control off, ``lastpipe`` runs a pipeline's last element in the
    parent shell, so a ``set`` written there changes the script's own state.
    The reading cannot place that, so a script that enables the option is
    refused rather than read as if its pipeline elements were isolated.

    Examples
    --------
    >>> enables_lastpipe(["shopt", "-s", "lastpipe"])
    True
    >>> enables_lastpipe(["shopt", "-so", "lastpipe"])
    True
    >>> enables_lastpipe(["shopt", "-u", "lastpipe"])
    False
    >>> enables_lastpipe(["shopt", "lastpipe"])
    False
    >>> enables_lastpipe(["shopt", "-s", "nullglob"])
    False
    """
    words = builtin_arguments(words)
    if words[:1] != ["shopt"]:
        return False
    setting = False
    for word in words[1:]:
        if word.startswith("-"):
            if "s" in word[1:]:
                setting = True
            if "u" in word[1:]:
                setting = False
            continue
        if word == LASTPIPE_OPTION:
            return setting
    return False
