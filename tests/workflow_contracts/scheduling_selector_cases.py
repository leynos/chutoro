"""The scripts that separate a correct installer selector from a weak one.

``installs_nextest`` answers whether a CI step's script really installs
``cargo-nextest``. Every one of these scripts mentions the installer, so a
substring check would accept them all; the expected answer beside each script
is what the shell does with it. The cases live here rather than in
``scheduling_ci_test`` because the table is data, and keeping it apart leaves
both files inside the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_ci_test`` only, the one place that
asserts over these cases. A new case belongs here with the shell behaviour it
separates named in its identifier.
"""

from __future__ import annotations

import typing as typ

from scheduling_script_reading import NEXTEST_INSTALLER

#: (script, expected answer, case name) triples, in reading order: the shapes
#: a command can take, then the operators that decide whether it runs, then
#: ``errexit``, then containers and here-documents.
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
    (
        f"set +e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "errexit-disabled-before-the-installer",
    ),
    (
        f"set -e\n{NEXTEST_INSTALLER}",
        True,
        "errexit-enabled-before-the-installer",
    ),
    (
        f"set +e\nset -e\n{NEXTEST_INSTALLER}",
        True,
        "errexit-restored-before-the-installer",
    ),
    (
        f"set +e\nset -- -e\n{NEXTEST_INSTALLER}",
        False,
        "a-positional-parameter-named-e",
    ),
    (
        f"set +e\nset value -e\n{NEXTEST_INSTALLER}",
        False,
        "a-positional-parameter-before-errexit",
    ),
    (
        f"set +e\nset -o pipefail -e\n{NEXTEST_INSTALLER}",
        True,
        "an-option-operand-before-errexit",
    ),
    (
        f"set +e\nfalse && set -e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "an-errexit-change-on-a-skipped-branch",
    ),
    (
        f"set +e\nset -e | true\n{NEXTEST_INSTALLER}",
        False,
        "an-errexit-change-in-a-pipeline",
    ),
    (
        f"set +e\nset -e &\n{NEXTEST_INSTALLER}",
        False,
        "an-errexit-change-in-the-background",
    ),
    (
        f"if false; then\n{NEXTEST_INSTALLER}\nfi",
        False,
        "an-installer-in-an-untaken-branch",
    ),
    (
        f"if true; then\n{NEXTEST_INSTALLER}\nfi",
        False,
        "an-installer-in-a-branch-the-reading-cannot-place",
    ),
    (f"exit 0\n{NEXTEST_INSTALLER}", False, "an-installer-after-an-exit"),
    (f"{NEXTEST_INSTALLER}\nexit 0", True, "an-installer-before-an-exit"),
    (
        f"cat <<'EOF'\n{NEXTEST_INSTALLER}\nEOF",
        False,
        "a-here-document-body",
    ),
    (
        f"cat <<'EOF'\n{NEXTEST_INSTALLER}\nEOF\n{NEXTEST_INSTALLER}",
        True,
        "after-a-here-document-body",
    ),
    (
        f"cat <<'EOF'\n\tEOF\n{NEXTEST_INSTALLER}\nEOF",
        False,
        "a-tab-indented-terminator-of-an-exact-document",
    ),
    (
        f"cat <<-EOF\n\t{NEXTEST_INSTALLER}\n\tEOF\n{NEXTEST_INSTALLER}",
        True,
        "a-tab-indented-terminator-of-a-stripping-document",
    ),
    (
        f"cat <<-EOF\n  EOF\n{NEXTEST_INSTALLER}\nEOF",
        False,
        "a-space-indented-terminator-of-a-stripping-document",
    ),
    (
        f"cat <<'EOF'\nEO\\\nF\n{NEXTEST_INSTALLER}\nEOF",
        False,
        "a-literal-body-joined-by-a-backslash",
    ),
    (
        f"cat <<EOF\nEO\\\nF\n{NEXTEST_INSTALLER}\nEOF",
        True,
        "an-expanded-body-joined-by-a-backslash",
    ),
    (
        'scripts/install-nextest.sh "$(printf \'%s\' \' #\')" || true',
        False,
        "a-mask-after-a-quoted-substitution",
    ),
    (
        "cat <<\\EOF\n\\EOF\n" + NEXTEST_INSTALLER + "\nEOF",
        False,
        "a-here-document-opened-with-an-escaped-delimiter",
    ),
    (
        f"set +e\nset -e && true &\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "an-errexit-change-in-an-asynchronous-and-list",
    ),
    (
        f"set -n\n{NEXTEST_INSTALLER}",
        False,
        "noexec-enabled-before-the-installer",
    ),
    (
        f"set -o noexec\n{NEXTEST_INSTALLER}",
        False,
        "noexec-named-before-the-installer",
    ),
    (
        f"false && (\n{NEXTEST_INSTALLER}\n)\ntrue",
        False,
        "an-installer-in-a-maybe-skipped-subshell-group",
    ),
    (
        f"cat <<'EOF' &&\necho example; {NEXTEST_INSTALLER}\nEOF\ntrue",
        False,
        "a-here-document-body-inside-a-continued-list",
    ),
)
