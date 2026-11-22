"""Read `rustup toolchain install` invocations out of workflow steps.

`rustup --component` takes one value. Written as `--component clippy
rustfmt`, the second name is parsed as a toolchain and rustup exits 1, so a
lane dies before it does any work. This module owns the one reading of those
invocations: it joins the line continuations the shell joins and groups each
`--component` flag with the values that follow it, so the contracts in
`toolchain_install_test.py` do not each parse the text again.

It also holds the nightly portable SIMD lane's presence contract, because the
reading that contract needs is the same one. Keeping the parser and that
contract together leaves the test module below the 400-line limit.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import itertools as it
import re
import shlex
import typing as typ

from workflow_support import jobs, load_workflow, run_script, steps

#: A `rustup` invocation at the start of an executable command. The script
#: is joined first and quoted text is blanked in the copy the pattern reads,
#: so a command split across lines is one match and a rustup phrase inside
#: `echo` or a multi-line string is none. `echo 'rustup ...'` prints a
#: sentence; it does not resolve a toolchain.
RUSTUP_INVOCATION = re.compile(
    r"(?m)(?:^|[;&|])[ \t]*"
    r"(?P<command>rustup[ \t]+toolchain[ \t]+install(?![^\s;&|])[^\n;&|]*)"
)

#: The flag whose repeated form is the whole point of this module.
COMPONENT_FLAG = "--component"

#: Characters after which a `#` begins a new shell word, and so a comment.
WORD_BOUNDARIES = frozenset(" \t;&|()<>")

#: Quoted shell text, including escaped double quotes. A quoted line break
#: does not begin an executable command, so quoted spans are blanked before
#: command boundaries are read.
QUOTED_TEXT = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"", re.DOTALL)

#: An escaped character. A backslash makes the next character literal, so an
#: escaped separator cannot end a command and is blanked in the copy the
#: boundary pattern reads.
ESCAPED_CHARACTER = re.compile(r"\\.?", re.DOTALL)

#: The lane whose nightly installer this module holds to its purpose.
NIGHTLY_WORKFLOW = "nightly-portable-simd.yml"
NIGHTLY_JOB = "nightly-portable-simd"

#: The toolchain, profile, and components the nightly lane exists to install.
NIGHTLY_TOOLCHAIN = "nightly"
NIGHTLY_PROFILE = "minimal"
NIGHTLY_COMPONENTS = frozenset({"clippy", "rustfmt"})


class ComponentGroup(typ.NamedTuple):
    """One `--component` flag, with enough context to name it in a failure."""

    job_name: str
    invocation: str
    values: list[str]


def _is_escape(character: str, quote: str | None) -> bool:
    """Report whether *character* escapes the next one."""
    if quote == "'":
        return False
    return character == "\\"


def _is_comment_start(character: str, quote: str | None, previous: str) -> bool:
    """Report whether *character* begins a shell comment."""
    if quote is not None or character != "#":
        return False
    return previous in WORD_BOUNDARIES


def _quote_after(character: str, quote: str | None) -> str | None:
    """Return the open quote after an unescaped character."""
    if quote is not None:
        return None if character == quote else quote
    return character if character in {"'", '"'} else None


def _scan_line(line: str, quote: str | None) -> tuple[int | None, str | None]:
    """Return a line's comment start and the quote state it leaves open.

    The scanner carries the quote state in and out, because a shell quote can
    span physical lines. It resets the escape state per line: a backslash at
    the end of a line escapes the newline, not the next line's first
    character.
    """
    escaped = False
    previous = " "
    for index, character in enumerate(line):
        if escaped:
            escaped = False
        elif _is_escape(character, quote):
            escaped = True
        elif _is_comment_start(character, quote, previous):
            return index, quote
        else:
            quote = _quote_after(character, quote)
        previous = character
    return None, quote


def strip_shell_comments(script: str) -> str:
    """Return *script* with unquoted shell comments removed.

    A `#` starts a comment only outside quotes and at the start of a word, so
    a `#` inside a quoted string stays part of the command. Comments are
    dropped before invocations are matched, so a commented-out installer
    cannot satisfy a presence contract while the step installs nothing. The
    text before a comment is kept unchanged, including the whitespace that
    preceded the `#`; that space stops a backslash left at the end of a
    stripped trailing comment from being read as a line continuation.
    """
    stripped = []
    quote: str | None = None
    for line in script.splitlines():
        comment_at, quote = _scan_line(line, quote)
        stripped.append(line if comment_at is None else line[:comment_at])
    return "\n".join(stripped)


def shell_words(command: str) -> list[str]:
    """Split one shell command into decoded words.

    Quotes are removed and escapes are resolved, so `"nightly"` and
    `nightly` are the same word. This is a word splitter, not an evaluator:
    variables, globs, and substitutions are left alone.

    Examples
    --------
    >>> shell_words('rustup toolchain install "nightly" --profile minimal')
    ['rustup', 'toolchain', 'install', 'nightly', '--profile', 'minimal']
    >>> shell_words("rustup toolchain install nightly --profile 'minimal'")
    ['rustup', 'toolchain', 'install', 'nightly', '--profile', 'minimal']
    """
    try:
        return shlex.split(command, comments=False, posix=True)
    except ValueError:
        # An invocation cut mid-quote by the line-based match carries no
        # decodable words; the raw split still lets the contract name it.
        return command.split()


def _continuation_step(
    character: str, quote: str | None, escaped: bool
) -> tuple[str | None, str | None, bool]:
    """Return the text to keep, the open quote, and the escape state."""
    if escaped:
        if character == "\n":
            return None, quote, False
        return "\\" + character, quote, False
    if _is_escape(character, quote):
        return None, quote, True
    if quote is None and character in {"'", '"'}:
        return character, character, False
    if quote == character:
        return character, None, False
    return character, quote, False


def join_continuations(script: str) -> str:
    """Remove unescaped backslash-newline pairs, and only those.

    A backslash escapes the newline and both disappear; an escaped backslash
    leaves the newline in place, so the next line is a new command. The
    backslash is retained for every other escaped character, because shells
    and `shlex` give those escapes meaning that this reader must not erase.
    """
    joined = []
    quote: str | None = None
    escaped = False
    for character in script:
        text, quote, escaped = _continuation_step(character, quote, escaped)
        if text is not None:
            joined.append(text)
    if escaped:
        joined.append("\\")
    return "".join(joined)


def _mask_quoted_match(match: re.Match[str]) -> str:
    """Keep a single quoted command word, or mask the whole span.

    A quoted executable name such as `"rustup"` runs the same command as the
    bare word, so the decoded word stays at the same offset. Quoted text that
    is not one command-like word is masked, so a quoted sentence cannot
    expose a command inside it and a quoted separator cannot become one.
    """
    text = match.group(0)
    try:
        words = shlex.split(text, comments=False, posix=True)
    except ValueError:
        words = []
    if len(words) == 1 and _is_command_word(words[0]):
        return words[0] + " " * (len(text) - len(words[0]))
    return "x" * len(text)


def _is_command_word(word: str) -> bool:
    """Report whether *word* can stand in for an executable name."""
    if not word or "\n" in word:
        return False
    return not any(character in ";&|<>" for character in word)


def _mask_escaped_match(match: re.Match[str]) -> str:
    """Return a word placeholder of the same length as an escaped span."""
    return "x" * len(match.group(0))


def mask_quoted(script: str) -> str:
    """Return *script* with quoted text masked, preserving length.

    Keeping the length keeps the offsets usable against the unmasked script.
    The placeholder is a word character rather than whitespace, so a quoted
    command word still occupies its position and `"echo" rustup ...` cannot
    read as an executable `rustup`.
    """
    return QUOTED_TEXT.sub(_mask_quoted_match, script)


def mask_escaped(script: str) -> str:
    """Return *script* with escaped characters masked, preserving length.

    A backslash makes the next character literal, so an escaped `;` or `|`
    does not end a command. Masking both characters keeps the offsets usable
    while removing the false boundary.
    """
    return ESCAPED_CHARACTER.sub(_mask_escaped_match, script)


#: A here-document operator and its delimiter word, quoted or bare.
HERE_DOCUMENT = re.compile(r"<<-?[ \t]*('[^']*'|\"[^\"]*\"|[^\s;&|]+)")


def _here_document_delimiters(line: str) -> list[str]:
    """Return the here-document delimiter words declared on *line*."""
    return [match.group(1).strip("'\"") for match in HERE_DOCUMENT.finditer(line)]


def mask_here_documents(script: str) -> str:
    """Return *script* with here-document bodies blanked, line for line.

    The body is data for the command that reads it, so an installer line
    inside one is not an execution. The closing delimiter is blanked too; a
    real command after it is left untouched.
    """
    pending: list[str] = []
    masked = []
    for line in script.splitlines():
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
            masked.append(" " * len(line))
            continue
        masked.append(line)
        pending.extend(_here_document_delimiters(line))
    return "\n".join(masked)


def rustup_installs(script: str) -> list[str]:
    """Return each rustup toolchain install in a step's script.

    Comments and here-document bodies are removed first, and genuine
    continuations are deleted, so a command split across lines is one
    invocation rather than none and a commented-out installer is no
    invocation at all. A quoted executable name is decoded in place, while
    `echo 'rustup ...'` and a rustup line inside a multi-line string install
    nothing. A backslash the shell would not treat as a continuation is left
    alone, and the invocation then ends on it, which is what
    `test_no_rustup_install_ends_on_a_dangling_backslash` looks for.

    Examples
    --------
    A quoted `#` stays in the command, while a comment does not:

    >>> rustup_installs("# rustup toolchain install nightly")
    []
    >>> rustup_installs('echo "# kept"; rustup toolchain install nightly')
    ['rustup toolchain install nightly']
    """
    text = mask_here_documents(strip_shell_comments(script))
    joined = join_continuations(text)
    boundaries = mask_escaped(mask_quoted(joined))
    return [
        joined[match.start("command") : match.end("command")]
        for match in RUSTUP_INVOCATION.finditer(boundaries)
    ]


def runs_command(script: str, command: str) -> bool:
    """Report whether *command* starts an executable command in *script*.

    Comments, continuations, quoted text, and escapes are handled as they are
    for the rustup reading, so a printed path such as
    `echo scripts/install-mold.sh` is not an execution.

    Examples
    --------
    >>> runs_command("echo scripts/install-mold.sh", "scripts/install-mold.sh")
    False
    >>> runs_command("scripts/install-mold.sh", "scripts/install-mold.sh")
    True
    """
    text = mask_here_documents(strip_shell_comments(script))
    joined = join_continuations(text)
    boundaries = mask_escaped(mask_quoted(joined))
    pattern = re.compile(
        rf"(?m)(?:^|[;&|])[ \t]*{re.escape(command)}(?![^\s;&|])"
    )
    return pattern.search(boundaries) is not None


def component_values(invocation: str) -> list[list[str]]:
    """Return the tokens following each `--component` flag.

    A value run ends at the next flag, so a `--component` immediately
    followed by one yields an empty list rather than swallowing it. Quoted
    values are decoded before they are grouped.
    """
    tokens = shell_words(invocation)
    return [
        list(it.takewhile(lambda token: not token.startswith("-"), tokens[index + 1 :]))
        for index, token in enumerate(tokens)
        if token == COMPONENT_FLAG
    ]


def rustup_invocations(workflow_name: str) -> typ.Iterator[tuple[str, str]]:
    """Yield every rustup toolchain install in a workflow, with its job name."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in steps(definition):
            for invocation in rustup_installs(run_script(step)):
                yield job_name, invocation


def run_scripts(workflow_name: str) -> typ.Iterator[tuple[str, str]]:
    """Yield every workflow `run` script with its job name."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in steps(definition):
            script = run_script(step)
            if script:
                yield job_name, script


def component_groups_in_step(
    step: dict[str, typ.Any],
) -> typ.Iterator[tuple[str, list[str]]]:
    """Yield each `--component` flag found in one step, with its invocation."""
    for invocation in rustup_installs(run_script(step)):
        for values in component_values(invocation):
            yield invocation, values


def component_groups(workflow_name: str) -> typ.Iterator[ComponentGroup]:
    """Yield every `--component` flag in a workflow, wherever it appears."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for invocation, values in steps_of(definition):
            yield ComponentGroup(job_name, invocation, values)


def steps_of(
    definition: dict[str, typ.Any],
) -> typ.Iterator[tuple[str, list[str]]]:
    """Yield each `--component` flag across one job's steps."""
    for step in steps(definition):
        yield from component_groups_in_step(step)


def installer_invocations(job_definition: dict[str, typ.Any]) -> list[str]:
    """Return each rustup install in one job, in step order."""
    return [
        invocation
        for step in steps(job_definition)
        for invocation in rustup_installs(run_script(step))
    ]


def _toolchain_of(invocation: str) -> str | None:
    """Return the toolchain named after `rustup toolchain install`."""
    tokens = shell_words(invocation)
    return tokens[3] if len(tokens) > 3 else None


def _profile_of(invocation: str) -> str | None:
    """Return the value given to `--profile`, if any."""
    tokens = shell_words(invocation)
    return next(
        (
            tokens[index + 1]
            for index, token in enumerate(tokens[:-1])
            if token == "--profile"
        ),
        None,
    )


def components_of(invocation: str) -> set[str]:
    """Return every component one invocation names, splitting comma lists."""
    return {
        component
        for values in component_values(invocation)
        for value in values
        for component in value.split(",")
        if component
    }


def assert_nightly_installer(job_definition: dict[str, typ.Any]) -> None:
    """Require the nightly lane's one installer to name both components.

    The lane exists to run portable-SIMD tests under a nightly compiler. An
    installer that is absent, names another toolchain, or omits a component
    leaves the lane either unable to start or silently testing the stable
    toolchain, which is the vacuous pass this contract exists to catch.
    Exactly one invocation is required so a second install cannot supply the
    missing component under a different profile or toolchain.

    Parameters
    ----------
    job_definition : dict[str, typing.Any]
        The parsed `nightly-portable-simd` job.

    Examples
    --------
    A comma-separated component list names both components:

    >>> assert_nightly_installer({"steps": [{"run": (
    ...     "rustup toolchain install nightly --profile minimal "
    ...     "--component clippy,rustfmt"
    ... )}]})
    """
    installers = installer_invocations(job_definition)
    assert len(installers) == 1, (
        f"{NIGHTLY_WORKFLOW}:{NIGHTLY_JOB} must install a toolchain exactly "
        f"once, got {installers!r}"
    )
    invocation = installers[0]
    assert _toolchain_of(invocation) == NIGHTLY_TOOLCHAIN, (
        f"{NIGHTLY_WORKFLOW}:{NIGHTLY_JOB} must install {NIGHTLY_TOOLCHAIN}, "
        f"got {_toolchain_of(invocation)!r} in {invocation!r}"
    )
    assert _profile_of(invocation) == NIGHTLY_PROFILE, (
        f"{NIGHTLY_WORKFLOW}:{NIGHTLY_JOB} must install the "
        f"{NIGHTLY_PROFILE} profile, got {_profile_of(invocation)!r} in "
        f"{invocation!r}"
    )
    missing = NIGHTLY_COMPONENTS - components_of(invocation)
    assert not missing, (
        f"{NIGHTLY_WORKFLOW}:{NIGHTLY_JOB} must install "
        f"{sorted(NIGHTLY_COMPONENTS)}, missing {sorted(missing)} in "
        f"{invocation!r}"
    )