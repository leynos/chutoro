"""Read which shell executes a step's script, and whether it runs it.

GitHub lets a workflow, a job, or a step name the shell a ``run`` script is
handed to. A custom template such as ``bash -n {0}`` parses that script and
executes none of it, and one such as ``bash {0}`` runs it without ``errexit``,
so a later successful command would absorb a failed gate. The scheduling
contract reads a step's commands as failing the step when ``errexit`` holds, so
a template that does not enable it is no more evidence than a syntax check.
Resolving the shell that actually applies means reading all three scopes in
override order and then reading the template for an option that never runs the
script — a syntax check, a help or version report, a command or stdin mode —
or for a missing ``errexit``.

This module holds that reading. ``scheduling_shell_test`` holds the assertions
over the precedence and the template forms, and ``scheduling_ci_test`` applies
``runs_the_script`` to the step that runs the scheduling target; the split
keeps each file inside the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_shell_test`` and ``scheduling_ci_test``
only. A new shell-template rule belongs here with the GitHub behaviour it
models stated beside it, so a later reader can check the rule against the
runner rather than against the tests.
"""

from __future__ import annotations

#: Shells whose execution flags this reader models. GitHub runs the default
#: ``run`` shell as ``bash -e {0}`` on Linux, and documents the ``bash`` and
#: ``sh`` shorthands as expanding to shells that enable ``errexit``.
EXECUTING_SHELLS = frozenset({"bash", "sh"})

#: Shell options that make the shell exit without running the script. The
#: syntax-only modes parse it and execute none of it; ``--help`` and
#: ``--version`` answer the request and exit successfully, so ``bash --help -e
#: {0}`` is no evidence that ``{0}`` ran however ``errexit`` is set.
NON_EXECUTING_OPTIONS = frozenset({"-n", "--noexec", "--help", "--version"})

#: Shell option names, written as the word after ``-o`` or ``-O``, that
#: matter to this reader.
_NOEXEC_OPTION = "noexec"
_ERREXIT_OPTION = "errexit"


def _default_run_shell(scope: dict[str, object]) -> str | None:
    """Return the ``defaults.run.shell`` a workflow or job declares, if any.

    Examples
    --------
    >>> _default_run_shell({"defaults": {"run": {"shell": "bash"}}})
    'bash'
    >>> _default_run_shell({}) is None
    True
    """
    defaults = scope.get("defaults")
    run = defaults.get("run") if isinstance(defaults, dict) else None
    shell = run.get("shell") if isinstance(run, dict) else None
    return shell if isinstance(shell, str) else None


def effective_shell(
    step: dict[str, object],
    job_definition: dict[str, object],
    workflow: dict[str, object],
) -> str | None:
    """Return the shell `step` runs under, resolving the three scopes.

    A step's own ``shell`` overrides its job's ``defaults.run.shell``, which
    overrides the workflow's. With none of the three declared, the runner's
    default shell executes the script.

    Examples
    --------
    >>> effective_shell({"shell": "bash -n {0}"}, {}, {})
    'bash -n {0}'
    >>> effective_shell({}, {"defaults": {"run": {"shell": "bash"}}}, {})
    'bash'
    >>> effective_shell({}, {}, {"defaults": {"run": {"shell": "sh"}}})
    'sh'
    >>> effective_shell({}, {}, {}) is None
    True
    """
    for shell in (
        step.get("shell"),
        _default_run_shell(job_definition),
        _default_run_shell(workflow),
    ):
        if isinstance(shell, str):
            return shell
    return None


def _is_short_cluster(word: str) -> bool:
    """Return whether `word` groups short shell options, as ``-eo`` does.

    A long option such as ``--noprofile`` is not a cluster, and the single
    word ``-`` is an operand rather than an option.

    Examples
    --------
    >>> _is_short_cluster("-eo")
    True
    >>> _is_short_cluster("--noprofile")
    False
    >>> _is_short_cluster("-")
    False
    """
    return word.startswith("-") and not word.startswith("--") and word != "-"


def _enables(word: str, letter: str) -> bool:
    """Return whether the short option cluster `word` enables `letter`.

    Examples
    --------
    >>> _enables("-eo", "e")
    True
    >>> _enables("--errexit", "e")
    False
    >>> _enables("pipefail", "o")
    False
    """
    return _is_short_cluster(word) and letter in word[1:]


def _takes_an_operand(word: str) -> bool:
    """Return whether `word` names a shell option whose value follows it.

    ``-o`` and ``-O`` take the option name as the next word, and either may
    sit inside a cluster, as in ``-eo pipefail``.

    Examples
    --------
    >>> _takes_an_operand("-o")
    True
    >>> _takes_an_operand("-eo")
    True
    >>> _takes_an_operand("-e")
    False
    """
    return word in {"-o", "-O"} or _enables(word, "o")


def _executes_nothing(options: list[str]) -> bool:
    """Return whether these options leave the script unexecuted.

    Bash reads short options as a cluster, so ``-en`` is ``-e`` plus ``-n``,
    and names an option with ``-o``, so ``-o noexec`` and ``-eo noexec``
    reach the same mode in two words. ``--help`` and ``--version`` are not
    syntax modes, but Bash reports and exits on either, so the script is just
    as unrun.

    Examples
    --------
    >>> _executes_nothing(["-n"])
    True
    >>> _executes_nothing(["-en"])
    True
    >>> _executes_nothing(["-eo", "noexec"])
    True
    >>> _executes_nothing(["--help", "-e"])
    True
    >>> _executes_nothing(["-eo", "pipefail"])
    False
    """
    previous = ""
    for word in options:
        if word in NON_EXECUTING_OPTIONS or _enables(word, "n"):
            return True
        if _takes_an_operand(previous) and word == _NOEXEC_OPTION:
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
    """
    previous = ""
    for word in options:
        if _enables(word, "e"):
            return True
        if _takes_an_operand(previous) and word == _ERREXIT_OPTION:
            return True
        previous = word
    return False


def _runs_a_command_string(word: str) -> bool:
    """Return whether `word` selects command-string or stdin mode.

    ``-c`` runs a command string and ``-s`` reads the script from stdin, so in
    either mode the operand substituted for ``{0}`` is an argument rather than
    the script. Both may be spelled inside a cluster, as ``-ec`` is.

    Examples
    --------
    >>> _runs_a_command_string("-ec")
    True
    >>> _runs_a_command_string("-s")
    True
    >>> _runs_a_command_string("--command")
    True
    >>> _runs_a_command_string("-e")
    False
    """
    if word == "--command":
        return True
    return any(_enables(word, letter) for letter in "cs")


def _runs_under_a_modelled_interpreter(command: list[str]) -> bool:
    """Return whether `command` names a shell whose flags this reader models.

    Examples
    --------
    >>> _runs_under_a_modelled_interpreter(["/bin/bash", "-e"])
    True
    >>> _runs_under_a_modelled_interpreter(["python"])
    False
    >>> _runs_under_a_modelled_interpreter([])
    False
    """
    if not command:
        return False
    return command[0].rsplit("/", 1)[-1] in EXECUTING_SHELLS


def _reaches_the_script(options: list[str]) -> bool:
    """Return whether ``{0}`` is the script the shell runs, not an argument.

    Bash stops reading options at its first operand and runs that operand as
    the script, so an option word left over before ``{0}`` would run something
    else. ``-c`` and ``-s`` are rejected outright, through
    ``_runs_a_command_string``. Option values such as ``pipefail`` after ``-o``
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
    index = 0
    while index < len(options):
        word = options[index]
        if _runs_a_command_string(word):
            return False
        if not word.startswith("-") or word == "-":
            return False
        index += 2 if _takes_an_operand(word) else 1
    return True


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
    >>> runs_the_script("sh {0}")
    False
    >>> runs_the_script("bash -n {0}")
    False
    >>> runs_the_script("bash -en {0}")
    False
    >>> runs_the_script("bash -eo noexec {0}")
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
    command = shell.split("{0}", 1)[0].split()
    if not _runs_under_a_modelled_interpreter(command):
        return False
    options = command[1:]
    return (
        _reaches_the_script(options)
        and not _executes_nothing(options)
        and _enables_errexit(options)
    )
