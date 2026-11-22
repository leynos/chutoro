"""The ``errexit``, container and here-document scripts the selector must read.

Every one of these scripts mentions the installer, and the expected answer
beside each is what the shell does with it: an option change that leaves the
failure masked, a container whose body may not run, or a here-document whose
body is not a command line. They live apart from ``scheduling_selector_cases``
so both files stay inside the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_ci_test`` only, which reads them
together with the shapes in ``scheduling_selector_cases``.
"""

from __future__ import annotations

import typing as typ

from scheduling_script_reading import NEXTEST_INSTALLER

EXECUTION_CASES: typ.Final[tuple[tuple[str, bool, str], ...]] = (
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
    (
        "# note " + chr(92) + "\n" + NEXTEST_INSTALLER,
        True,
        "a-comment-that-ends-with-a-backslash",
    ),
    (
        f"{NEXTEST_INSTALLER} 2>&1",
        True,
        "a-descriptor-copy-after-the-installer",
    ),
    (
        f"set +xo errexit\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "an-errexit-cluster-naming-errexit",
    ),
    (
        f"echo $'$(' ; {NEXTEST_INSTALLER}",
        True,
        "a-literal-substitution-in-an-ansi-c-string",
    ),
    (
        f"echo '(' ; {NEXTEST_INSTALLER}",
        True,
        "a-quoted-parenthesis",
    ),
    (
        f"false && (true\n{NEXTEST_INSTALLER}\ntrue)",
        False,
        "an-unspaced-parenthesised-group",
    ),
    (
        "set +e\nset - -e\n" + NEXTEST_INSTALLER + "\ntrue",
        False,
        "an-errexit-flag-as-a-positional-parameter",
    ),
    (
        f"set -eo noexec\n{NEXTEST_INSTALLER}",
        False,
        "noexec-named-through-a-cluster",
    ),
    (
        "printf x " + chr(92) * 3 + "\n; " + NEXTEST_INSTALLER,
        True,
        "a-three-backslash-continuation",
    ),
    (
        f"{NEXTEST_INSTALLER} >'install>' & true",
        False,
        "a-background-after-a-quoted-redirection-target",
    ),
    (
        f"builtin set +e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "errexit-disabled-through-builtin-set",
    ),
    (
        f"set $'-n'\n{NEXTEST_INSTALLER}",
        False,
        "noexec-named-by-an-ansi-c-word",
    ),
    (
        'echo "pre' + chr(92) + '\npare" ; ' + NEXTEST_INSTALLER,
        True,
        "a-quoted-continuation-before-the-installer",
    ),
    (
        f"builtin set -n\n{NEXTEST_INSTALLER}",
        False,
        "noexec-enabled-through-builtin-set",
    ),
    (
        "set $" + chr(39) + chr(92) + "x2dn" + chr(39) + "\n" + NEXTEST_INSTALLER,
        False,
        "noexec-spelled-by-an-ansi-c-escape",
    ),
    (
        "echo 'a\nb\nc'\n" + NEXTEST_INSTALLER,
        True,
        "an-installer-after-a-multi-line-quoted-string",
    ),
    (
        f"builtin exit 0\n{NEXTEST_INSTALLER}",
        False,
        "an-exit-through-builtin",
    ),
    (
        f"exec /bin/true\n{NEXTEST_INSTALLER}",
        False,
        "a-shell-replaced-by-exec",
    ),
    (
        f"set -n | true\n{NEXTEST_INSTALLER}",
        True,
        "a-noexec-set-in-a-pipeline",
    ),
    (
        f"shopt -s lastpipe\ntrue | set +e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "lastpipe-enabled",
    ),
    (
        "set $'-" + chr(92) + "u006e" + chr(39) + "\n" + NEXTEST_INSTALLER,
        False,
        "noexec-spelled-by-a-unicode-escape",
    ),
    (
        f"bash {NEXTEST_INSTALLER} 2>&1",
        True,
        "an-interpreter-with-a-descriptor-copy",
    ),
    (
        f"{NEXTEST_INSTALLER} >$(LOG)#1 || true",
        False,
        "a-mask-after-a-make-variable-reference",
    ),
    (
        f"exec > /dev/null\n{NEXTEST_INSTALLER}",
        True,
        "an-exec-that-only-redirects",
    ),
    (
        "set $" + chr(39) + "-" + chr(92) + "u6e" + chr(39) + "\n" + NEXTEST_INSTALLER,
        False,
        "noexec-spelled-by-a-short-unicode-escape",
    ),
    (
        f"set > /dev/null +e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "an-errexit-change-after-a-redirection",
    ),
    (
        "printf x" + chr(92) + "\n#; " + NEXTEST_INSTALLER,
        True,
        "an-installer-after-a-continued-hash",
    ),
    (
        f"! set +e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "a-negated-errexit-change",
    ),
    (
        f"> /dev/null builtin set +e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "an-errexit-change-after-a-leading-redirection",
    ),
    (
        f"set +e\nset '>' ignored -e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "a-quoted-redirection-in-set",
    ),
    (
        "'scripts/install-" + chr(92) + "\nnextest.sh'",
        False,
        "a-backslash-newline-inside-single-quotes",
    ),
    (
        f"command -p set +e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "an-errexit-change-through-command-p",
    ),
    (
        f"trap 'exit 0' ERR\n{NEXTEST_INSTALLER}",
        False,
        "an-err-trap-before-the-installer",
    ),
    (
        f"{NEXTEST_INSTALLER} >| install.log",
        True,
        "a-noclobber-redirection-after-the-installer",
    ),
    (
        "printf %s " + chr(92) + chr(39) + "\n" + NEXTEST_INSTALLER,
        True,
        "an-escaped-apostrophe-before-the-installer",
    ),
    (
        f"set $\"-n\"\n{NEXTEST_INSTALLER}",
        False,
        "noexec-spelled-by-a-locale-quoted-word",
    ),
    (
        f"command -- set +e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "an-errexit-change-after-the-command-terminator",
    ),
    (
        f"builtin trap 'exit 0' ERR\n{NEXTEST_INSTALLER}",
        False,
        "an-err-trap-through-builtin",
    ),
    (
        f"trap 'exit 0' EXIT\n{NEXTEST_INSTALLER}",
        False,
        "an-exit-trap-before-the-installer",
    ),
    (
        f"set >/dev/null +e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "an-errexit-change-after-an-attached-redirection",
    ),
    (
        f"builtin builtin set +e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "an-errexit-change-through-repeated-wrappers",
    ),
    (
        f"set +e\nset '>/dev/null' -e\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "a-quoted-attached-redirection-in-set",
    ),
    (
        f"trap 'exit 0' DEBUG\n{NEXTEST_INSTALLER}",
        False,
        "a-debug-trap-before-the-installer",
    ),
    (
        f"shopt -u -o errexit\n{NEXTEST_INSTALLER}\ntrue",
        False,
        "an-errexit-change-through-shopt-o",
    ),
    (
        f"shopt -s -o noexec\n{NEXTEST_INSTALLER}",
        False,
        "a-noexec-change-through-shopt-o",
    ),
)
