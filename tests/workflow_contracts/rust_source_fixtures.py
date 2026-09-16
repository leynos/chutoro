"""Rust sources written for the discovery reading, not taken from the tree.

Every one of these is a shape this repository's own files do not have. Over its
own sources a file-level reading and a function-level one agree exactly, a
prefix test and a named attribute set agree exactly, and a scanner that reads
literals and one that does not agree exactly. These are what separate them.

Each is named for the reading it discriminates, and the comment above it says
which way the reading fails without it.

The two nextest configurations at the end are here for the same reason: over
this repository's own configuration every override that names a
compile-contract test also extends its allowance, so nothing in the tree
separates a reading that checks for that from one that does not.
"""

import typing as typ

#: Sources whose discovered set is the point, written out so the reading is
#: driven rather than inferred from a tree that happens to agree with it.
SIBLINGS = """
#[test]
fn compiles_the_fixture() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/ok.rs");
}

#[test]
fn asserts_a_value() {
    assert_eq!(2 + 2, 4);
}
"""

RSTEST = """
#[rstest]
fn checks_with_the_repository_attribute() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/ok.rs");
}
"""

BRACE_IN_A_LITERAL = """
#[test]
fn writes_a_brace_then_builds() {
    let pattern = "}";
    let cases = trybuild::TestCases::new();
    cases.pass(pattern);
}
"""

# The marker sits on the same line as the raw string on purpose. A quote
# inside the raw string ends an ordinary-string reading early, exposing the
# `//` that follows; a scanner without raw-string support then blanks the rest
# of that line, and the marker with it.
RAW_STRING_WITH_A_QUOTE = (
    "\n#[test]\nfn documents_a_path_then_builds() {\n"
    '    let note = r#"one " then // a note"#; '
    "let cases = trybuild::TestCases::new();\n"
    "    cases.pass(note);\n}\n"
)

CHAR_LITERAL_BRACE = """
#[test]
fn closes_on_a_char_then_builds() {
    let brace = '}';
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/ok.rs");
}
"""

# A lifetime is a lone quote, not a literal. Reading it as one runs to the next
# quote, or to the end of the file when there is none, and everything after it
# stops being code: the cheap test's body then never closes and absorbs the
# costly test's marker, so the cheap one is reported and the costly one's own
# name may not be.
LIFETIME = """
#[test]
fn borrows_and_asserts() {
    let borrowed: &'static str = "ok";
    assert_eq!(borrowed, "ok");
}

#[test]
fn builds_the_fixture() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/ok.rs");
}
"""

SPAWNER = """
#[test]
fn checks_the_fixture_crate() {
    let output = Command::new(env!("CARGO"))
        .arg("check")
        .output()
        .expect("cargo must run");
}
"""


#: A configuration whose only override for `costly` changes its thread count.
#: nextest supplies an override's settings to the tests its filter matches, so
#: this one leaves `costly` on the sixty-second base allowance.
RETRIES_ONLY = {
    "profile": {
        "default": {
            "slow-timeout": {"period": "60s", "terminate-after": 1},
            "overrides": [
                {"filter": "test(/costly/)", "threads-required": 4},
                {
                    "filter": "test(/other/)",
                    "slow-timeout": {"period": "300s", "terminate-after": 1},
                },
            ],
        }
    }
}

#: The same, with the override actually lengthening the allowance.
EXTENDING = {
    "profile": {
        "default": {
            "slow-timeout": {"period": "60s", "terminate-after": 1},
            "overrides": [
                {
                    "filter": "test(/costly/)",
                    "slow-timeout": {"period": "300s", "terminate-after": 1},
                }
            ],
        }
    }
}
