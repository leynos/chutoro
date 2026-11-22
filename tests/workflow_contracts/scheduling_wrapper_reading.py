"""Read the shell wrappers that run a command in this shell.

``builtin`` and ``command`` run the command they name without forking, so a
reader that stops at the wrapper misses the ``set``, ``trap`` or ``exit``
behind it. The wrapper may repeat, carry flags such as ``-p``, and take a
redirection before or after its name; ``command -v`` and ``command -V`` only
look a name up and run nothing.

This module holds that reading. ``scheduling_segment_reading`` uses it to
read a command's arguments, and the split keeps each file inside the 400-line
limit ``AGENTS.md`` sets.
"""

from __future__ import annotations

from scheduling_command_reading import is_redirection, starts_a_redirection


def builtin_arguments(words: list[str]) -> list[str]:
    """Return `words` without an execution wrapper or a redirection.

    ``builtin`` and ``command`` run the command they name in this shell, so
    the reading looks through them; ``command -v`` and ``command -V`` only
    report where a name is found and yield no words at all.

    Examples
    --------
    >>> builtin_arguments(["builtin", "set", "+e"])
    ['set', '+e']
    >>> builtin_arguments(["command", "-p", "set", "+e"])
    ['set', '+e']
    >>> builtin_arguments(["command", "-v", "set"])
    []
    >>> builtin_arguments(["set", ">", "/dev/null", "+e"])
    ['set', '+e']
    """
    arguments = argument_words(words)
    while arguments[:1] in (["builtin"], ["command"]):
        if arguments[:1] == ["builtin"]:
            arguments = arguments[1:]
            continue
        arguments = _after_command_flags(arguments[1:])
        if not arguments:
            return []
    return arguments


def _after_command_flags(arguments: list[str]) -> list[str]:
    """Return the words after ``command`` and its flags, or [] for a lookup.

    Examples
    --------
    >>> _after_command_flags(["-p", "set", "+e"])
    ['set', '+e']
    >>> _after_command_flags(["--", "set", "+e"])
    ['set', '+e']
    >>> _after_command_flags(["-v", "set"])
    []
    """
    while arguments[:1] in (["-p"], ["-v"], ["-V"], ["--"]):
        if arguments[0] in {"-v", "-V"}:
            return []
        arguments = arguments[1:]
    return arguments


def argument_words(words: list[str]) -> list[str]:
    """Return `words` without redirection operators or their targets.

    A redirection configures a file descriptor rather than passing an argument,
    so ``bash scripts/install-nextest.sh 2>&1`` runs the installer from a
    shell and the descriptor copy is not one of that shell's arguments.

    Examples
    --------
    >>> argument_words(["bash", "installer"])
    ['bash', 'installer']
    >>> argument_words(["bash", "installer", "2>", "&", "1"])
    ['bash', 'installer']
    >>> argument_words(["installer", ">", "log"])
    ['installer']
    >>> argument_words(["set", ">/dev/null", "+e"])
    ['set', '+e']
    """
    arguments: list[str] = []
    index = 0
    while index < len(words):
        if is_redirection(words[index]):
            index += 3 if words[index + 1 : index + 2] == ["&"] else 2
            continue
        if starts_a_redirection(words[index]):
            index += 1
            continue
        arguments.append(words[index])
        index += 1
    return arguments
