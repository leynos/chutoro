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

#: The three shapes a narrow declaration matcher drops: an `async fn`, which
#: is the only way a `#[tokio::test]` test is written; a `fn` indented inside
#: `mod tests`; and a name carrying an uppercase character. All three fail in
#: the under-reporting direction, leaving the test bounded by nothing while a
#: pinned set that records only today's tests goes on passing.
NESTED_ASYNC = """
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn builds_the_fixture_crate_V2() {
        let cases = trybuild::TestCases::new();
        cases.pass("tests/ui/ok.rs");
    }

    #[tokio::test]
    async fn asserts_nothing_costly() {
        assert_eq!(1 + 1, 2);
    }
}
"""

#: A public, unsafe and `extern "C"` test declaration, so the qualifiers the
#: matcher steps over are exercised rather than assumed.
QUALIFIED = """
#[rstest]
pub async unsafe fn spawns_a_nested_cargo() {
    let output = Command::new("cargo").arg("check").output();
}
"""

#: A signature lifetime and a later character literal on the same line, which
#: is an ordinary shape and was the one that broke. Reading from `'a` to the
#: `'}'` put the function's opening brace inside a literal span, so the body
#: scan skipped that brace and the costly test was discovered by nothing. The
#: cheap sibling is here so the case discriminates rather than merely counts.
LIFETIME_THEN_CHAR = """
#[rstest]
fn borrows_then_builds(name: &'static str) { assert_eq!(name.ends_with('}'), false);
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/ok.rs");
}

#[rstest]
fn borrows_and_asserts_only(name: &'static str) { assert_eq!(name.ends_with('}'), false);
}
"""

#: Rust permits spacing around `::`, so this is the same attribute as
#: `#[tokio::test]`. The attribute set names `tokio::test`, so without
#: normalising the attribute before matching it the two disagreed over
#: whitespace and the test was discovered by nothing.
SPACED_ATTRIBUTE = """
#[tokio :: test]
async fn builds_under_a_spaced_attribute() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/ok.rs");
}
"""

#: A generic signature, which `#[rstest]` admits and which the declaration
#: matcher must step over. The list nests, so `<T: Into<String>>` is the case
#: that a bracket-free pattern gets wrong.
GENERIC_SIGNATURE = """
#[rstest]
fn builds_for_any_input<T: Into<String>>(value: T) {
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
