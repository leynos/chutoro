"""Read the shell options a ``set`` or ``shopt`` command changes.

The scheduling contract reads a CI step as failing when a command fails only
while ``errexit`` holds, and as installing nextest only while the shell runs
its commands at all. Both depend on options the script sets for itself:
``set +e`` takes the first guarantee away, ``set -n`` takes the second, and
``shopt -s lastpipe`` makes a pipeline element's state change reach the
parent.

This module holds those readings alone. ``scheduling_errexit_reading`` applies
the errexit state to the commands it finds, ``scheduling_script_reading``
applies noexec and lastpipe, and the split keeps every file inside the
400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_errexit_reading`` and
``scheduling_script_reading`` only. A new option rule belongs here with the
Bash behaviour it models stated beside it.
"""

from __future__ import annotations

import collections.abc as cabc
import typing as typ

from scheduling_segment_reading import argument_words

#: The option name ``-o`` and ``+o`` take as their operand.
ERREXIT_OPTION = "errexit"

#: The ``shopt`` name that runs a pipeline's last element in the parent.
LASTPIPE_OPTION = "lastpipe"


def _after_wrapper(arguments: list[str]) -> list[str]:
    """Return the words a ``builtin`` or ``command`` wrapper runs.

    ``command -p`` runs the builtin it names, while ``command -v`` and
    ``command -V`` only report where the name is found, so a lookup yields no
    words at all.

    Examples
    --------
    >>> _after_wrapper(["builtin", "set", "+e"])
    ['set', '+e']
    >>> _after_wrapper(["command", "-p", "set", "+e"])
    ['set', '+e']
    >>> _after_wrapper(["command", "-v", "set"])
    []
    >>> _after_wrapper(["set", "+e"])
    ['set', '+e']
    """
    if arguments[:1] == ["builtin"]:
        return arguments[1:]
    if arguments[:1] != ["command"]:
        return arguments
    arguments = arguments[1:]
    while arguments[:1] in (["-p"], ["-v"], ["-V"]):
        if arguments[0] in {"-v", "-V"}:
            return []
        arguments = arguments[1:]
    return arguments


def builtin_arguments(words: cabc.Sequence[str]) -> list[str]:
    """Return `words` without a ``builtin`` wrapper or a redirection.

    Bash runs ``set`` in the current shell through ``builtin`` or ``command``,
    so ``builtin set +e`` disables errexit exactly as ``set +e`` does. A
    redirection configures a file descriptor rather than passing an argument,
    so ``set > /dev/null +e`` still reaches its ``+e``.

    Examples
    --------
    >>> builtin_arguments(["builtin", "set", "+e"])
    ['set', '+e']
    >>> builtin_arguments(["command", "set", "-e"])
    ['set', '-e']
    >>> builtin_arguments(["set", "+e"])
    ['set', '+e']
    >>> builtin_arguments(["set", ">", "/dev/null", "+e"])
    ['set', '+e']
    >>> builtin_arguments([">", "/dev/null", "builtin", "set", "+e"])
    ['set', '+e']
    >>> builtin_arguments(["command", "-p", "set", "+e"])
    ['set', '+e']
    >>> builtin_arguments(["command", "-v", "set"])
    []
    """
    return _after_wrapper(argument_words(list(words)))


def _names_an_option(flag: str) -> bool:
    """Return whether `flag` ends a short cluster with the option-name letter.

    Bash reads a cluster's final ``o`` as the option-name flag, so ``+xo
    errexit`` names the same option as ``+o errexit``, and ``-euo pipefail``
    sets errexit before naming pipefail. ``-O`` names a ``shopt`` option
    instead, so it is not one of these.

    Examples
    --------
    >>> _names_an_option("-o")
    True
    >>> _names_an_option("+xo")
    True
    >>> _names_an_option("-euo")
    True
    >>> _names_an_option("-e")
    False
    """
    return flag[:1] in {"-", "+"} and flag[1:].endswith("o")


def _enables_errexit(flag: str) -> bool:
    """Return whether a short-option cluster `flag` sets errexit.

    Bash reads ``-euo pipefail`` as a cluster, so the ``e`` may sit beside
    other letters rather than standing alone.

    Examples
    --------
    >>> _enables_errexit("-e")
    True
    >>> _enables_errexit("+e")
    True
    >>> _enables_errexit("-euo")
    True
    >>> _enables_errexit("-x")
    False
    """
    return flag[:1] in {"-", "+"} and "e" in flag[1:]


def _takes_a_name(flag: str) -> bool:
    """Return whether the last letter of `flag` takes an option-name operand.

    ``-o``, ``+o``, ``-O`` and ``+O`` name a shell or ``shopt`` option in the
    word that follows. A cluster's final letter does the same, as ``-euo``
    does for ``pipefail``, while the letters before it are still set.

    Examples
    --------
    >>> _takes_a_name("-o")
    True
    >>> _takes_a_name("+O")
    True
    >>> _takes_a_name("-euo")
    True
    >>> _takes_a_name("-e")
    False
    """
    return flag[:1] in {"-", "+"} and flag[1:].endswith(("o", "O"))


def _set_options(words: cabc.Sequence[str]) -> typ.Iterator[tuple[str, str]]:
    """Yield each option word of a ``set`` command with its operand, if any.

    Option processing stops at ``--``, at a lone ``-`` and at the first word
    that does not begin with ``-`` or ``+``: everything from there on is a
    positional parameter. A flag whose last letter names an option takes the
    next word as that name.

    Examples
    --------
    >>> list(_set_options(["set", "-euo", "pipefail", "+e"]))
    [('-euo', 'pipefail'), ('+e', '')]
    >>> list(_set_options(["set", "--", "-e"]))
    []
    >>> list(_set_options(["set", "value", "-e"]))
    []
    """
    index = 1
    while index < len(words):
        flag = words[index]
        if flag in {"--", "-"} or flag[:1] not in {"-", "+"}:
            return
        takes_a_name = _takes_a_name(flag)
        yield flag, words[index + 1] if takes_a_name and index + 1 < len(words) else ""
        index += 2 if takes_a_name else 1


def errexit_setting(words: cabc.Sequence[str]) -> bool | None:
    """Return the errexit state a ``set`` command leaves, if it changes it.

    ``set +e`` turns errexit off, and ``set -e`` turns it back on; both may
    also name the option in full, as ``set -o errexit`` and ``set +o
    errexit``. Any other command, and a ``set`` naming other options, leaves
    the state alone, which ``None`` reports.

    Option processing stops at ``--``, at a lone ``-`` and at the first word
    that does not begin with ``-`` or ``+``: everything from there on is a
    positional parameter, so ``set -- -e``, ``set - -e`` and
    ``set value -e`` set ``$1`` rather than errexit. The word after ``-o``, ``+o``, ``-O`` or ``+O`` is that option's
    operand even though it need not begin with a sign, so ``set -o pipefail
    -e`` is still read to its ``-e``.

    Examples
    --------
    >>> errexit_setting(["set", "+e"])
    False
    >>> errexit_setting(["set", "-e"])
    True
    >>> errexit_setting(["set", "+o", "errexit"])
    False
    >>> errexit_setting(["set", "-euo", "pipefail"])
    True
    >>> errexit_setting(["set", "+xo", "errexit"])
    False
    >>> errexit_setting(["set", "-o", "pipefail", "-e"])
    True
    >>> errexit_setting(["set", "--", "-e"]) is None
    True
    >>> errexit_setting(["set", "-", "-e"]) is None
    True
    >>> errexit_setting(["builtin", "set", "+e"])
    False
    >>> errexit_setting(["set", "value", "-e"]) is None
    True
    >>> errexit_setting(["set", "-x"]) is None
    True
    >>> errexit_setting(["true"]) is None
    True
    """
    words = builtin_arguments(words)
    if words[:1] != ["set"]:
        return None
    setting: bool | None = None
    for flag, operand in _set_options(words):
        names_errexit = operand == ERREXIT_OPTION and _names_an_option(flag)
        if names_errexit or _enables_errexit(flag):
            setting = flag.startswith("-")
    return setting


def enables_noexec(words: cabc.Sequence[str]) -> bool:
    """Return whether a ``set`` command leaves ``noexec`` on.

    A non-interactive Bash under ``noexec`` reads the script without running
    it, so an installer after the change installs nothing. The option is set
    as ``-n``, inside a cluster such as ``-eun``, or by name through a
    cluster ending in ``o``, as ``-o noexec`` and ``-eo noexec`` do, and
    turned off again by the ``+`` forms.

    Examples
    --------
    >>> enables_noexec(["set", "-n"])
    True
    >>> enables_noexec(["set", "-eun", "pipefail"])
    True
    >>> enables_noexec(["set", "-o", "noexec"])
    True
    >>> enables_noexec(["set", "-eo", "noexec"])
    True
    >>> enables_noexec(["set", "+o", "noexec"])
    False
    >>> enables_noexec(["set", "-", "-n"])
    False
    >>> enables_noexec(["builtin", "set", "-n"])
    True
    >>> enables_noexec(["true"])
    False
    """
    words = builtin_arguments(words)
    if words[:1] != ["set"]:
        return False
    noexec = False
    for flag, operand in _set_options(words):
        names_noexec = operand == "noexec" and _names_an_option(flag)
        if names_noexec or "n" in flag[1:]:
            noexec = flag[:1] == "-"
    return noexec


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


