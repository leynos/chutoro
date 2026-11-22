"""The scripts that separate a correct installer selector from a weak one.

``installs_nextest`` answers whether a CI step's script really installs
``cargo-nextest``. Every one of these scripts mentions the installer, so a
substring check would accept them all; the expected answer beside each script
is what the shell does with it. The cases live here rather than in
``scheduling_ci_test`` because the table is data, and keeping it apart leaves
both files inside the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_ci_test`` only, the one place that
asserts over these cases, which reads them together with
``scheduling_selector_execution_cases``. A new case belongs here with the shell behaviour it
separates named in its identifier.
"""

from __future__ import annotations

import typing as typ

from scheduling_script_reading import NEXTEST_INSTALLER

#: (script, expected answer, case name) triples for the shapes a command can
#: take and the operators that decide whether it runs. The ``errexit``,
#: container and here-document cases live in
#: ``scheduling_selector_execution_cases``.
SELECTOR_CASES: typ.Final[tuple[tuple[str, bool, str], ...]] = (

    (NEXTEST_INSTALLER, True, "the-installer"),
    (f"bash {NEXTEST_INSTALLER}", True, "through-a-shell"),
    (
        f"make test && {NEXTEST_INSTALLER}",
        False,
        "after-an-unproven-command",
    ),
    (
        f"false && {NEXTEST_INSTALLER}",
        False,
        "skipped-by-a-preceding-failure",
    ),
    (f"echo done ; {NEXTEST_INSTALLER}", True, "after-a-semicolon"),
    (
        f"false && make test ; {NEXTEST_INSTALLER}",
        True,
        "after-a-semicolon-that-resets-the-chain",
    ),
    (
        f"{NEXTEST_INSTALLER} && make test-workflow-contracts",
        True,
        "before-another-command",
    ),
    (f"echo {NEXTEST_INSTALLER}", False, "printed"),
    (f"# {NEXTEST_INSTALLER}", False, "commented"),
    (f"echo hi  # {NEXTEST_INSTALLER}", False, "trailing-comment"),
    (f"{NEXTEST_INSTALLER} || true", False, "failure-masked"),
    (f"{NEXTEST_INSTALLER}; true", True, "a-semicolon-does-not-mask"),
    (
        f"{NEXTEST_INSTALLER} ; make test",
        True,
        "a-semicolon-ends-the-installers-own-list",
    ),
    (f"{NEXTEST_INSTALLER} ;", True, "a-trailing-semicolon"),
    (f"bash -n {NEXTEST_INSTALLER}", False, "syntax-check-only"),
    (f"bash {NEXTEST_INSTALLER} || true", False, "a-shell-that-masks"),
    (
        f"{NEXTEST_INSTALLER} && echo installed || true",
        False,
        "masked-later-on-the-line",
    ),
    (
        f"{NEXTEST_INSTALLER} && make test && echo done",
        True,
        "a-chain-that-carries-the-status",
    ),
    (
        f"true || {NEXTEST_INSTALLER}",
        False,
        "skipped-by-a-preceding-success",
    ),
    (f"{NEXTEST_INSTALLER} &", False, "sent-to-the-background"),
    (f"{NEXTEST_INSTALLER} && echo done &", False, "backgrounded-at-the-end"),
    (f"echo '&&' {NEXTEST_INSTALLER}", False, "a-quoted-operator"),
    (
        "echo " + chr(92) + "&" + chr(92) + "& " + NEXTEST_INSTALLER,
        False,
        "an-escaped-operator",
    ),
    (
        "scripts/install-" + chr(92) + "\nnextest.sh",
        True,
        "a-path-split-by-a-continuation",
    ),
    (
        f"{NEXTEST_INSTALLER} >install#log || true",
        False,
        "a-mask-after-a-word-internal-hash",
    ),
    (
        f"{NEXTEST_INSTALLER} && echo installed\ntrue",
        False,
        "a-chain-swallowed-by-a-later-line",
    ),
    (
        f"{NEXTEST_INSTALLER} && echo installed\n# installed",
        True,
        "a-chain-that-ends-the-script",
    ),
    (
        f"{NEXTEST_INSTALLER} # don't skip this",
        True,
        "a-comment-with-an-apostrophe",
    ),
    (
        f"{NEXTEST_INSTALLER} &&\ntrue",
        True,
        "a-chain-continued-on-the-next-line",
    ),
    (
        f"true ||\n{NEXTEST_INSTALLER}",
        False,
        "skipped-across-a-continued-line",
    ),
    (
        f"{NEXTEST_INSTALLER} &&\n# installed\ntrue",
        True,
        "a-continuation-past-a-comment-line",
    ),
)
