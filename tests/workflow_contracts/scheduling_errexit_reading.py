"""Read whether a CI step's script still fails on a failed command.

``scheduling_script_reading`` reads a step's commands as failing the step
while ``errexit`` holds, which is what the runner's ``bash -e {0}`` shell gives
it. A script can turn that off: ``set +e`` makes the shell carry on after a
failure, so a later successful line replaces the installer's status and the
step stays green however the installation went. Reading the script's own
option changes is therefore part of reading whether it installs nextest.

This module holds that reading. ``scheduling_script_reading`` applies it to
the commands it finds, and the split keeps the option vocabulary beside the
shell behaviour it models while leaving that module inside the 400-line limit
``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_script_reading`` only, which applies
the setting to the commands it reads. A new option belongs here with the shell
behaviour it models stated beside it.
"""

from __future__ import annotations

import collections.abc as cabc

#: The option name ``-o`` and ``+o`` take as their operand.
ERREXIT_OPTION = "errexit"


def _names_errexit(words: list[str], index: int) -> bool:
    """Return whether ``words[index]`` names errexit for ``-o`` or ``+o``.

    Examples
    --------
    >>> _names_errexit(["set", "-o", "errexit"], 1)
    True
    >>> _names_errexit(["set", "+o", "errexit"], 1)
    True
    >>> _names_errexit(["set", "-o", "pipefail"], 1)
    False
    >>> _names_errexit(["set", "-e"], 1)
    False
    """
    if words[index] not in {"-o", "+o"}:
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


def errexit_setting(words: list[str]) -> bool | None:
    """Return the errexit state a ``set`` command leaves, if it changes it.

    ``set +e`` turns errexit off, and ``set -e`` turns it back on; both may
    also name the option in full, as ``set -o errexit`` and ``set +o
    errexit``. Any other command, and a ``set`` naming other options, leaves
    the state alone, which ``None`` reports.

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
    >>> errexit_setting(["set", "-x"]) is None
    True
    >>> errexit_setting(["true"]) is None
    True
    """
    if words[:1] != ["set"]:
        return None
    setting: bool | None = None
    for index, flag in enumerate(words[1:], start=1):
        if _names_errexit(words, index):
            setting = flag.startswith("-")
        elif _enables_errexit(flag):
            setting = flag.startswith("-")
    return setting


def errexit_states(
    lines: cabc.Sequence[cabc.Sequence[cabc.Sequence[str]]],
) -> list[list[bool]]:
    """Return whether ``errexit`` is on as each command of `lines` starts.

    `lines` is a script's command lines, each holding the words of its
    commands in the order the shell runs them. A command that changes the
    setting runs under the state it replaces, and the change carries into the
    lines after it, so ``set +e`` joins the line it is written on and turns
    the state off for everything after it.

    Examples
    --------
    >>> errexit_states([[["a"]]])
    [[True]]
    >>> errexit_states([[["set", "+e"]], [["a"]]])
    [[True], [False]]
    >>> errexit_states([[["set", "+e"]], [["set", "-e"]], [["a"]]])
    [[True], [False], [True]]
    >>> errexit_states([[["set", "+o", "errexit"], ["a"]]])
    [[True, False]]
    """
    states: list[list[bool]] = []
    state = True
    for line in lines:
        line_states: list[bool] = []
        for words in line:
            line_states.append(state)
            change = errexit_setting(list(words))
            if change is not None:
                state = change
        states.append(line_states)
    return states
