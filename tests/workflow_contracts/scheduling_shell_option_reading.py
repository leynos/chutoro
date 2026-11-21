"""Read the option words of a shell template as Bash reads them.

A template such as ``bash -en {0}`` is a list of option words followed by the
operand the runner substitutes the script path into. Whether an option word is
a cluster of short options, names a further option in the word after it, or is
a long option whose value is the next word decides which words are options at
all — and so whether the template runs the script, runs none of it, or runs
only part of it.

This module holds that vocabulary; ``scheduling_template_reading`` applies it
to a whole template, and ``scheduling_shell_reading`` resolves which template
applies. A new option rule belongs here with the Bash behaviour it models
stated beside it.
"""

from __future__ import annotations


#: Shells whose execution flags this reader models. GitHub runs the default
#: ``run`` shell as ``bash -e {0}`` on Linux, and documents the ``bash`` and
#: ``sh`` shorthands as expanding to shells that enable ``errexit``.
EXECUTING_SHELLS = frozenset({"bash", "sh"})


#: Shell options that make the shell exit without running the script: the
#: syntax-only modes, ``-D`` and its long forms, ``--help`` and ``--version``,
#: and ``--one-command``, which runs one command and exits. ``bash --help -e
#: {0}`` and ``bash -eD {0}`` are no evidence that ``{0}`` ran.
NON_EXECUTING_OPTIONS = frozenset(
    {
        "-n",
        "--noexec",
        "-D",
        "--dump-strings",
        "--dump-po-strings",
        "--help",
        "--version",
        "--one-command",
    }
)

#: Letters that, set in an option word, leave the script unexecuted or run
#: only part of it. ``n`` reads without running, ``D`` implies ``n``, and
#: ``t`` runs one command and exits, so a script of several commands is
#: truncated.
NON_EXECUTING_LETTERS = "nDt"

#: Long options whose value is the word after them, so an option-looking
#: word there is a file name rather than a shell option.
LONG_OPTIONS_WITH_OPERANDS = frozenset({"--rcfile", "--init-file"})

#: Shell option names, written as the word after ``-o`` or ``-O``, that
#: matter to this reader.
NOEXEC_OPTION = "noexec"
ONESHOT_OPTION = "onecmd"
ERREXIT_OPTION = "errexit"


def is_short_cluster(word: str) -> bool:
    """Return whether `word` groups short shell options, as ``-eo`` does.

    A long option such as ``--noprofile`` is not a cluster, and ``-`` is an
    operand rather than an option.

    Examples
    --------
    >>> is_short_cluster("-eo")
    True
    >>> is_short_cluster("--noprofile")
    False
    >>> is_short_cluster("-")
    False
    """
    return word.startswith("-") and not word.startswith("--") and word != "-"


def enables(word: str, letter: str) -> bool:
    """Return whether the short option cluster `word` enables `letter`.

    Examples
    --------
    >>> enables("-eo", "e")
    True
    >>> enables("--errexit", "e")
    False
    >>> enables("pipefail", "o")
    False
    """
    return is_short_cluster(word) and letter in word[1:]


def takes_an_operand(word: str) -> bool:
    """Return whether `word` takes its value as the next word.

    A short option cluster ending in ``o`` or ``O`` names a shell or ``shopt``
    option there, and ``--rcfile`` and ``--init-file`` name a file. The word
    after any of them is a value, so an ``-e`` or ``-n`` written there is not
    an option of its own.

    Examples
    --------
    >>> takes_an_operand("-eo")
    True
    >>> takes_an_operand("--rcfile")
    True
    >>> takes_an_operand("-e")
    False
    """
    if word in LONG_OPTIONS_WITH_OPERANDS:
        return True
    return is_short_cluster(word) and word.endswith(("o", "O"))


def names_an_option(previous: str, word: str) -> bool:
    """Return whether `word` is the option name a short cluster precedes.

    ``-o noexec`` names an option; ``--rcfile noexec`` names a file, so the
    operand of a long option sets nothing.

    Examples
    --------
    >>> names_an_option("-o", "noexec")
    True
    >>> names_an_option("-eo", "pipefail")
    True
    >>> names_an_option("--rcfile", "noexec")
    False
    >>> names_an_option("-e", "noexec")
    False
    """
    return is_short_cluster(previous) and previous.endswith(("o", "O"))


def runs_a_command_string(word: str) -> bool:
    """Return whether `word` selects command-string or stdin mode.

    ``-c`` runs a command string and ``-s`` reads the script from stdin, so in
    either mode the operand substituted for ``{0}`` is an argument rather than
    the script. Both may be spelled inside a cluster, as ``-ec`` is.

    Examples
    --------
    >>> runs_a_command_string("-ec")
    True
    >>> runs_a_command_string("-s")
    True
    >>> runs_a_command_string("--command")
    True
    >>> runs_a_command_string("-e")
    False
    """
    if word == "--command":
        return True
    return any(enables(word, letter) for letter in "cs")


def runs_under_a_modelled_interpreter(command: list[str]) -> bool:
    """Return whether `command` names a shell whose flags this reader models.

    Examples
    --------
    >>> runs_under_a_modelled_interpreter(["/bin/bash", "-e"])
    True
    >>> runs_under_a_modelled_interpreter(["python"])
    False
    >>> runs_under_a_modelled_interpreter([])
    False
    """
    if not command:
        return False
    return command[0].rsplit("/", 1)[-1] in EXECUTING_SHELLS
