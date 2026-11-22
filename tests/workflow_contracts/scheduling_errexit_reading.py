"""Read whether a CI step's script still fails when one of its commands fails.

``scheduling_script_reading`` reads a step's commands as failing the step
while ``errexit`` holds, which is what the runner's ``bash -e {0}`` shell gives
it. Two things can take that guarantee away. A script can turn the option off:
``set +e`` makes the shell carry on after a failure, so a later successful
line replaces the installer's status and the step stays green. And a command
inside an AND/OR list is exempt from ``errexit`` unless it is the command after
the final ``&&``, so a failure carried along a chain reaches the step only if
the chain ends the script.

This module holds both readings. ``scheduling_script_reading`` applies them to
the commands it finds, and the split keeps the shell vocabulary beside the
behaviour it models while leaving that module inside the 400-line limit
``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_script_reading`` only, which applies
the readings to the commands it finds. A new shell fact belongs here with the
behaviour it models stated beside it.
"""

from __future__ import annotations

import collections.abc as cabc
import typing as typ

from scheduling_set_reading import errexit_setting

#: The only operator that carries a command's failure to the end of an AND/OR
#: list. ``installer && make test`` fails when the installer fails; every
#: other operator ends the list at the installer, so ``installer || true`` and
#: ``installer && echo installed || true`` install nothing that can fail the
#: step.
CARRYING_OPERATOR = "&&"

#: The operator that ends a command's own list. A failing simple command in
#: its own list stops the script under ``errexit``, so its status reaches the
#: step whatever later commands would do, and a command after it runs whatever
#: came before.
OWN_LIST_END = ";"

#: Operators whose command runs in a subshell of its own. Bash runs every
#: element of a pipeline in a subshell, wherever it stands, so an option a
#: ``set`` changes there dies with it and the enclosing script never sees the
#: change. A backgrounded command is asynchronous for the same reason.
SUBSHELL_OPERATORS = frozenset({"|", "|&", "&"})

#: Operators that make the command after them a pipeline element. A command
#: after a lone ``&`` runs in the parent, so ``&`` is not one of these.
PIPELINE_OPERATORS = frozenset({"|", "|&"})


class Command(typ.NamedTuple):
    """One simple command: its words and the operator that ends it.

    ``scheduling_script_reading`` passes its own segment tuples, which carry
    the same two fields, so the readings here accept either shape.
    """

    words: cabc.Sequence[str]
    follows: str


def _runs_unconditionally(line: cabc.Sequence[Command], index: int) -> bool:
    """Return whether ``line[index]`` runs whatever earlier commands did.

    Only the first command of a line and a command after ``;`` run
    unconditionally. A command after ``&&``, ``||``, ``|``, ``&`` or ``|&``
    may be skipped, so an option change written there may not have happened.

    Examples
    --------
    >>> chain = [Command(["a"], "&&"), Command(["b"], "")]
    >>> _runs_unconditionally(chain, 0)
    True
    >>> _runs_unconditionally(chain, 1)
    False
    >>> _runs_unconditionally([Command(["a"], ";"), Command(["b"], "")], 1)
    True
    """
    if index == 0:
        return True
    return line[index - 1].follows == OWN_LIST_END


def _ends_the_list(line: cabc.Sequence[Command], index: int) -> int:
    """Return the index of the last command in ``line[index]``'s list.

    A list runs from ``;`` to ``;``, and ``&`` ends the list it backgrounds.
    A command in the middle of ``a && b &`` therefore sits in a list whose
    last command is backgrounded, so it runs in a subshell as surely as the
    last one does.

    Examples
    --------
    >>> backgrounded = [Command(["a"], "&&"), Command(["b"], "&")]
    >>> _ends_the_list(backgrounded, 0)
    1
    >>> separate = [Command(["a"], "&"), Command(["b"], "")]
    >>> _ends_the_list(separate, 1)
    1
    """
    end = index
    while end + 1 < len(line) and line[end].follows not in {"", OWN_LIST_END, "&"}:
        end += 1
    return end


def runs_in_a_subshell(line: cabc.Sequence[Command], index: int) -> bool:
    """Return whether ``line[index]`` runs in a subshell of its own.

    Every element of a pipeline runs in a subshell in Bash, so a ``set``
    written there changes nothing the enclosing script sees. A backgrounded
    list is asynchronous in the same way, and ``&`` backgrounds the whole
    AND/OR list it ends rather than only the command before it, so
    ``set -e && true &`` leaves the parent's state alone. The command after a
    lone ``&`` runs in the parent, so only ``|`` and ``|&`` make the command
    after them a pipeline element.

    Examples
    --------
    >>> pipeline = [Command(["set", "-e"], "|"), Command(["true"], "")]
    >>> runs_in_a_subshell(pipeline, 0)
    True
    >>> runs_in_a_subshell(pipeline, 1)
    True
    >>> runs_in_a_subshell([Command(["set", "-e"], "&")], 0)
    True
    >>> async_list = [Command(["set", "-e"], "&&"), Command(["true"], "&")]
    >>> runs_in_a_subshell(async_list, 0)
    True
    >>> after = [Command(["true"], "&"), Command(["set", "-e"], "")]
    >>> runs_in_a_subshell(after, 1)
    False
    """
    if line[index].follows in SUBSHELL_OPERATORS:
        return True
    if index > 0 and line[index - 1].follows in PIPELINE_OPERATORS:
        return True
    return line[_ends_the_list(line, index)].follows == "&"


def errexit_states(
    lines: cabc.Sequence[cabc.Sequence[Command]],
) -> list[list[bool | None]]:
    """Return whether ``errexit`` is certainly on as each command starts.

    ``True`` and ``False`` report the state every change so far has certainly
    left. ``None`` reports that a change may or may not have run: a ``set``
    after ``&&`` or ``||`` runs only on one branch, so applying its change
    either way could claim the next command's failure is carried when the
    branch taken leaves it exempt. A later change that certainly runs reports
    its own state again, because it runs whatever the uncertain branch did.
    A change made in a subshell is ignored rather than applied: Bash runs
    every pipeline element and every backgrounded list in a subshell, so
    ``set -e | true`` and ``set -e && true &`` leave the parent's state
    exactly as it was. A conditional change that leaves the state as it was
    keeps the known state rather than turning it unknown.

    Examples
    --------
    >>> errexit_states([[Command(["a"], "")]])
    [[True]]
    >>> errexit_states([
    ...     [Command(["set", "+e"], "")],
    ...     [Command(["a"], "")],
    ... ])
    [[True], [False]]
    >>> errexit_states([
    ...     [Command(["set", "+e"], "")],
    ...     [Command(["set", "-e"], "")],
    ...     [Command(["a"], "")],
    ... ])
    [[True], [False], [True]]
    >>> errexit_states([
    ...     [Command(["set", "+o", "errexit"], ""), Command(["a"], ";")],
    ... ])
    [[True, False]]
    >>> errexit_states([
    ...     [Command(["false"], ""), Command(["set", "-e"], "&&")],
    ...     [Command(["a"], "")],
    ... ])
    [[True, True], [True]]
    >>> errexit_states([
    ...     [Command(["set", "+e"], "")],
    ...     [Command(["set", "-e"], "|"), Command(["true"], "")],
    ...     [Command(["a"], "")],
    ... ])
    [[True], [False, False], [False]]
    >>> errexit_states([
    ...     [Command(["set", "+e"], "")],
    ...     [Command(["set", "-e"], "&&"), Command(["true"], "&")],
    ...     [Command(["a"], "")],
    ... ])
    [[True], [False, False], [False]]
    """
    states: list[list[bool | None]] = []
    state: bool | None = True
    for line in lines:
        line_states: list[bool | None] = []
        for index, command in enumerate(line):
            line_states.append(state)
            change = errexit_setting(command.words)
            if change is None or runs_in_a_subshell(line, index):
                continue
            certain = _runs_unconditionally(line, index) or change == state
            state = change if certain else None
        states.append(line_states)
    return states


def ends_the_script(
    lines: cabc.Sequence[cabc.Sequence[Command]], index: int
) -> bool:
    """Return whether ``lines[index]`` is the script's last command line.

    ``errexit`` does not exit for a command that fails inside an AND/OR list
    unless it is the command after the final ``&&``. A failure in
    ``installer && echo installed`` is therefore exempt, and the line reports
    it only through its own status: a later line that succeeds replaces that
    status, so the step stays green. Only the script's last command line
    carries it out to the step, and lines with no command — blanks and
    comments — are not later commands.

    Examples
    --------
    >>> chain = [Command(["a"], "&&"), Command(["b"], "")]
    >>> ends_the_script([chain], 0)
    True
    >>> ends_the_script([chain, [Command(["c"], "")]], 0)
    False
    >>> ends_the_script([chain, []], 0)
    True
    """
    commands = [number for number, line in enumerate(lines) if line]
    return bool(commands) and commands[-1] == index


def carries_to_the_step(
    lines: cabc.Sequence[cabc.Sequence[Command]],
    number: int,
    command: Command,
) -> bool:
    """Return whether a failure of `command` still fails the step.

    A command that fails inside an AND/OR list is exempt from ``errexit``
    unless it is the command after the final ``&&``, so a failure carried by
    ``&&`` reaches the step only through the line's own status. A later line
    that succeeds replaces that status, so the chain also has to be the
    script's last command line. Every other operator leaves the failure in the
    command's own list, where nothing exempts it.

    Examples
    --------
    >>> chain = [Command(["a"], "&&"), Command(["b"], "")]
    >>> carries_to_the_step([chain], 0, chain[0])
    True
    >>> carries_to_the_step([chain, [Command(["c"], "")]], 0, chain[0])
    False
    >>> own = [Command(["a"], ";"), Command(["b"], "")]
    >>> carries_to_the_step([own, [Command(["c"], "")]], 0, own[0])
    True
    """
    if command.follows == CARRYING_OPERATOR:
        return ends_the_script(lines, number)
    return True
