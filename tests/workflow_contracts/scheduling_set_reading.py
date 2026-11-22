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

from scheduling_wrapper_reading import builtin_arguments

#: The option name ``-o`` and ``+o`` take as their operand.
ERREXIT_OPTION = "errexit"

#: The ``set -o`` option name ``shopt -o`` also sets.
NOEXEC_OPTION = "noexec"


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
    >>> errexit_setting(["builtin", "builtin", "set", "+e"])
    False
    >>> errexit_setting(["shopt", "-u", "-o", "errexit"])
    False
    >>> errexit_setting(["set", "value", "-e"]) is None
    True
    >>> errexit_setting(["set", "-x"]) is None
    True
    >>> errexit_setting(["true"]) is None
    True
    """
    words = builtin_arguments(words)
    if words[:1] == ["shopt"]:
        return shopt_setting(words, ERREXIT_OPTION)
    if words[:1] != ["set"]:
        return None
    setting: bool | None = None
    for flag, operand in _set_options(words):
        if _sets_option(flag, operand, ERREXIT_OPTION, "e"):
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
    >>> enables_noexec(["shopt", "-s", "-o", "noexec"])
    True
    """
    words = builtin_arguments(words)
    if words[:1] == ["shopt"]:
        return shopt_setting(words, NOEXEC_OPTION) is True
    if words[:1] != ["set"]:
        return False
    noexec = False
    for flag, operand in _set_options(words):
        if _sets_option(flag, operand, NOEXEC_OPTION, "n"):
            noexec = flag[:1] == "-"
    return noexec


def shopt_setting(words: cabc.Sequence[str], option: str) -> bool | None:
    """Return the state a ``shopt -o`` command leaves for `option`.

    ``shopt -o`` changes the same options ``set -o`` exposes, so
    ``shopt -u -o errexit`` disables errexit and ``shopt -s -o noexec``
    enables noexec. Any other ``shopt`` command leaves the state alone.

    Examples
    --------
    >>> shopt_setting(["shopt", "-u", "-o", "errexit"], "errexit")
    False
    >>> shopt_setting(["shopt", "-s", "-o", "noexec"], "noexec")
    True
    >>> shopt_setting(["shopt", "-s", "lastpipe"], "errexit") is None
    True
    >>> shopt_setting(["shopt", "-o", "errexit"], "errexit") is None
    True
    """
    words = builtin_arguments(words)
    flags = "".join(word[1:] for word in words[1:] if word.startswith("-"))
    if not _names_shopt_option(words, option, flags):
        return None
    if "s" in flags:
        return True
    return False if "u" in flags else None


def _names_shopt_option(
    words: cabc.Sequence[str], option: str, flags: str
) -> bool:
    """Return whether a ``shopt`` command names `option` through ``-o``.

    Examples
    --------
    >>> _names_shopt_option(["shopt", "-u", "-o", "errexit"], "errexit", "uo")
    True
    >>> _names_shopt_option(["shopt", "-s", "lastpipe"], "lastpipe", "s")
    False
    >>> _names_shopt_option(["shopt", "-o", "errexit"], "errexit", "o")
    True
    """
    return words[:1] == ["shopt"] and option in words[1:] and "o" in flags


def _sets_option(flag: str, operand: str, option: str, letter: str) -> bool:
    """Return whether an option word names `option` or sets its letter.

    Examples
    --------
    >>> _sets_option("-o", "errexit", "errexit", "e")
    True
    >>> _sets_option("+e", "", "errexit", "e")
    True
    >>> _sets_option("-x", "", "errexit", "e")
    False
    """
    return (operand == option and _names_an_option(flag)) or letter in flag[1:]
