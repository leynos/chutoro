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

#: The option name ``-o`` and ``+o`` take as their operand.
ERREXIT_OPTION = "errexit"

#: The ``shopt`` name that runs a pipeline's last element in the parent.
LASTPIPE_OPTION = "lastpipe"


def builtin_arguments(words: cabc.Sequence[str]) -> cabc.Sequence[str]:
    """Return `words` with a leading ``builtin`` or ``command`` removed.

    Bash runs ``set`` in the current shell through either word, so
    ``builtin set +e`` disables errexit exactly as ``set +e`` does.

    Examples
    --------
    >>> list(builtin_arguments(["builtin", "set", "+e"]))
    ['set', '+e']
    >>> list(builtin_arguments(["command", "set", "-e"]))
    ['set', '-e']
    >>> list(builtin_arguments(["set", "+e"]))
    ['set', '+e']
    """
    if words[:1] in (["builtin"], ["command"]):
        return words[1:]
    return words


def _names_errexit(words: cabc.Sequence[str], index: int) -> bool:
    """Return whether ``words[index]`` names errexit for ``-o`` or ``+o``.

    Bash reads a cluster's final ``o`` as the option-name flag, so ``+xo
    errexit`` turns errexit off exactly as ``+o errexit`` does, and ``-euo
    pipefail`` sets errexit before naming pipefail.

    Examples
    --------
    >>> _names_errexit(["set", "-o", "errexit"], 1)
    True
    >>> _names_errexit(["set", "+o", "errexit"], 1)
    True
    >>> _names_errexit(["set", "+xo", "errexit"], 1)
    True
    >>> _names_errexit(["set", "-o", "pipefail"], 1)
    False
    >>> _names_errexit(["set", "-e"], 1)
    False
    """
    flag = words[index]
    if flag[:1] not in {"-", "+"} or not flag[1:].endswith("o"):
        return False
    return words[index + 1 : index + 2] == [ERREXIT_OPTION]


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
    index = 1
    while index < len(words):
        flag = words[index]
        if flag in {"--", "-"} or flag[:1] not in {"-", "+"}:
            break
        if _names_errexit(words, index) or _enables_errexit(flag):
            setting = flag.startswith("-")
        index += 2 if _takes_a_name(flag) else 1
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
        index += 2 if _takes_a_name(flag) else 1
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


