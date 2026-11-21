"""Read whether a shell template actually runs a CI step's script.

A custom template such as ``bash -n {0}`` parses the script and executes none
of it, and one such as ``bash {0}`` runs it without ``errexit``, so a later
successful command would absorb a failed gate. Reading the template means
reading its interpreter, its option words, the operand the runner substitutes
the script path into, and every mode that leaves the script unexecuted or
truncated.

This module holds that reading; ``scheduling_shell_reading`` resolves which
template applies. A new shell-template rule belongs here with the GitHub or
Bash behaviour it models stated beside it, so a later reader can check the
rule against the shell rather than against the tests.
"""

from __future__ import annotations

import shlex

from scheduling_shell_option_reading import (
    ERREXIT_OPTION,
    EXECUTING_SHELLS,
    LONG_OPTIONS_WITH_OPERANDS,
    NOEXEC_OPTION,
    NON_EXECUTING_LETTERS,
    NON_EXECUTING_OPTIONS,
    ONESHOT_OPTION,
    enables,
    is_short_cluster,
    names_an_option,
    runs_a_command_string,
    runs_under_a_modelled_interpreter,
    takes_an_operand,
)

def _executes_nothing(options: list[str]) -> bool:
    """Return whether these options leave the script unexecuted.

    Bash reads short options as a cluster, so ``-en`` is ``-e`` plus ``-n``,
    and names one with ``-o``, so ``-o noexec`` reaches the same mode in two
    words. ``-D`` implies ``-n`` while dumping strings, ``-t`` runs one
    command and exits, and ``--help`` or ``--version`` exits without running
    the script at all.

    Examples
    --------
    >>> _executes_nothing(["-n"])
    True
    >>> _executes_nothing(["-en"])
    True
    >>> _executes_nothing(["-eo", "noexec"])
    True
    >>> _executes_nothing(["-eD"])
    True
    >>> _executes_nothing(["-et"])
    True
    >>> _executes_nothing(["-o", "onecmd"])
    True
    >>> _executes_nothing(["--dump-strings", "-e"])
    True
    >>> _executes_nothing(["--help", "-e"])
    True
    >>> _executes_nothing(["-eo", "pipefail"])
    False
    >>> _executes_nothing(["--rcfile", "-n"])
    False
    """
    previous = ""
    for word in options:
        if takes_an_operand(previous):
            if names_an_option(previous, word) and word in {
                NOEXEC_OPTION,
                ONESHOT_OPTION,
            }:
                return True
            previous = ""
            continue
        if word in NON_EXECUTING_OPTIONS or any(
            enables(word, letter) for letter in NON_EXECUTING_LETTERS
        ):
            return True
        previous = word
    return False


def _enables_errexit(options: list[str]) -> bool:
    """Return whether these options make a failing command fail the shell.

    ``-e`` may be spelled inside a cluster such as ``-eo pipefail``, and the
    same setting may be named with ``-o errexit``.

    Examples
    --------
    >>> _enables_errexit(["-e"])
    True
    >>> _enables_errexit(["-eo", "pipefail"])
    True
    >>> _enables_errexit(["-o", "errexit"])
    True
    >>> _enables_errexit(["--noprofile"])
    False
    >>> _enables_errexit(["--rcfile", "-e"])
    False
    """
    previous = ""
    for word in options:
        if takes_an_operand(previous):
            if names_an_option(previous, word) and word == ERREXIT_OPTION:
                return True
            previous = ""
            continue
        if enables(word, "e"):
            return True
        previous = word
    return False


def _reaches_the_script(options: list[str]) -> bool:
    """Return whether ``{0}`` is the script the shell runs, not an argument.

    Bash stops reading options at its first operand and runs that operand as
    the script, so an option word left over before ``{0}`` would run something
    else. ``-c`` and ``-s`` are rejected outright, through
    ``runs_a_command_string``. Option values such as ``pipefail`` after ``-o``
    are skipped.

    Examples
    --------
    >>> _reaches_the_script(["-e"])
    True
    >>> _reaches_the_script(["-eo", "pipefail"])
    True
    >>> _reaches_the_script(["-e", "noexec"])
    False
    >>> _reaches_the_script(["-ec", "run()"])
    False
    >>> _reaches_the_script(["-s"])
    False
    """
    previous = ""
    for word in options:
        if takes_an_operand(previous):
            previous = ""
            continue
        if runs_a_command_string(word):
            return False
        if not word.startswith("-") or word == "-":
            return False
        previous = word
    return True


#: The word the script's path is replaced with before a template is split, so
#: a quoted ``{0}`` is read as the one operand the runner substitutes for.
_SCRIPT_WORD = "__CHUTORO_SCRIPT_PATH__"


def _template_words(shell: str) -> list[str]:
    """Return a shell template's words, with ``{0}`` as one placeholder word.

    GitHub substitutes the script's path before the shell parses the
    template, so a quoted ``"{0}"`` is one operand word rather than a stray
    quote before the operand. An unbalanced template yields no words.

    Examples
    --------
    >>> _template_words("bash -e {0}")
    ['bash', '-e', '__CHUTORO_SCRIPT_PATH__']
    >>> _template_words('bash -e "{0}"')
    ['bash', '-e', '__CHUTORO_SCRIPT_PATH__']
    >>> _template_words('bash -c "run {0}"')
    ['bash', '-c', 'run __CHUTORO_SCRIPT_PATH__']
    >>> _template_words('bash "{0}')
    []
    """
    try:
        return shlex.split(shell.replace("{0}", _SCRIPT_WORD))
    except ValueError:
        return []


def runs_the_script(shell: str | None) -> bool:
    """Return whether `shell` runs the step's script under ``errexit``.

    With no ``shell`` the runner's default applies, which is ``bash -e {0}``
    on Linux; the ``bash`` and ``sh`` shorthands also enable ``errexit``. A
    custom template is read as words: its interpreter must be one whose flags
    this reader models, it must contain ``{0}`` where the script's path is
    substituted, ``{0}`` must be the script operand, and the options before it
    must enable ``errexit`` without selecting a command or stdin mode or an
    option that leaves the script unexecuted.

    Examples
    --------
    >>> runs_the_script(None)
    True
    >>> runs_the_script("bash")
    True
    >>> runs_the_script("bash --noprofile --norc -eo pipefail {0}")
    True
    >>> runs_the_script("bash {0}")
    False
    >>> runs_the_script('bash -e "{0}"')
    True
    >>> runs_the_script("sh {0}")
    False
    >>> runs_the_script("bash -n {0}")
    False
    >>> runs_the_script("bash -en {0}")
    False
    >>> runs_the_script("bash -eo noexec {0}")
    False
    >>> runs_the_script("bash -et {0}")
    False
    >>> runs_the_script("bash --rcfile -e {0}")
    False
    >>> runs_the_script("bash --rcfile errexit {0}")
    False
    >>> runs_the_script("bash -o onecmd -e {0}")
    False
    >>> runs_the_script("bash -eD {0}")
    False
    >>> runs_the_script("bash --dump-strings -e {0}")
    False
    >>> runs_the_script("bash --dump-po-strings -e {0}")
    False
    >>> runs_the_script("bash --help -e {0}")
    False
    >>> runs_the_script("bash --version -e {0}")
    False
    >>> runs_the_script("bash -n")
    False
    >>> runs_the_script("python {0}")
    False
    """
    if shell is None or shell in EXECUTING_SHELLS:
        return True
    if "{0}" not in shell:
        return False
    words = _template_words(shell)
    if _SCRIPT_WORD not in words:
        return False
    command = words[: words.index(_SCRIPT_WORD)]
    if not runs_under_a_modelled_interpreter(command):
        return False
    options = command[1:]
    return (
        _reaches_the_script(options)
        and not _executes_nothing(options)
        and _enables_errexit(options)
    )
