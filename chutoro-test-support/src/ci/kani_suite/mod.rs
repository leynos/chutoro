//! Check a Kani run against the checked-in harness inventory.
//!
//! `make kani` and `make kani-full` wrap every `cargo kani` invocation in the
//! `kani_suite` binary. The binary loads `tools/kani/harnesses.toml`, derives
//! the harnesses the requested package and tier must produce, runs the
//! delegated command, parses Kani's regular output, and requires the exact
//! multiset of successful harnesses to match.
//!
//! The reason this exists rather than trusting Kani's exit status is that its
//! status is not a proof count. Kani exits successfully when a filter matches
//! a subset of what was asked, and a proof that stops being declared simply
//! stops appearing. The manifest plus this comparison turns either into a
//! failure, which is what makes the merge gate mean "these proofs ran" rather
//! than "the verifier did not crash".
//!
//! The modules divide by responsibility: [`manifest`] validates the inventory,
//! [`runner`] validates the delegated command, [`parser`] walks its arguments
//! while [`words`] interprets each one, [`process`] executes the command while
//! streaming and capturing its output, [`mod@line`] recognizes each printed line,
//! [`output`] tracks the run those lines describe, and [`verdict`] compares
//! the result with the expected set. Only [`check`] combines them.

pub mod line;
pub mod manifest;
pub mod output;
pub mod parser;
pub mod process;
pub mod runner;
pub mod verdict;
pub mod words;

use std::collections::BTreeSet;
use std::fmt;
use std::path::PathBuf;

use thiserror::Error;

use self::manifest::{Manifest, ManifestError, Tier};
use self::output::OutputError;
use self::process::RunError;
use self::runner::CommandError;
use self::verdict::VerdictError;

/// One checker request, as the command line supplies it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// Path to the checked-in harness manifest.
    pub manifest: PathBuf,
    /// The Cargo package the delegated command verifies.
    pub package: String,
    /// The tier whose harnesses the run must complete.
    pub scope: Tier,
    /// The delegated command, including any leading `NAME=VALUE` words.
    pub command: Vec<String>,
}

/// What a successful check verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verified {
    /// The package that was verified.
    pub package: String,
    /// The tier that was requested.
    pub scope: Tier,
    /// The harnesses that completed successfully, sorted.
    pub harnesses: Vec<String>,
}

impl fmt::Display for Tier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fast => formatter.write_str("fast"),
            Self::Full => formatter.write_str("full"),
        }
    }
}

/// A reason a checked Kani run does not discharge its expected proof surface.
#[derive(Debug, Error)]
pub enum CheckError {
    /// The manifest could not be read or does not hold the expected shape.
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    /// The delegated command is not one the checker can trust.
    #[error(transparent)]
    Command(#[from] CommandError),
    /// The delegated process could not be run.
    #[error(transparent)]
    Run(#[from] RunError),
    /// Kani's output was not a complete regular-format record.
    #[error(transparent)]
    Output(#[from] OutputError),
    /// The run did not match the expected harness set.
    #[error(transparent)]
    Verdict(#[from] VerdictError),
    /// The manifest classifies no harness for the package and tier.
    #[error(
        "the harness manifest classifies no harness for package {package:?} in \
         the {scope} scope"
    )]
    EmptyScope {
        /// The package that was requested.
        package: String,
        /// The tier that was requested.
        scope: Tier,
    },
    /// A selector names a harness the package and tier do not classify.
    #[error(
        "the delegated command selects harnesses outside the {scope} scope for \
         package {package:?}: {names:?}"
    )]
    SelectorOutsideScope {
        /// The package that was requested.
        package: String,
        /// The tier that was requested.
        scope: Tier,
        /// The selectors the manifest does not list in that scope.
        names: Vec<String>,
    },
    /// The delegated verifier itself failed.
    #[error("the delegated Kani command failed with {status}; failed harnesses: {failed:?}")]
    DelegatedFailed {
        /// The process's exit status.
        status: String,
        /// Harnesses the output recorded as failed, when it was parseable.
        failed: Vec<String>,
    },
}

/// Run one checked Kani invocation.
///
/// # Errors
///
/// Returns [`CheckError`] when the manifest is invalid, the delegated command
/// is untrusted, the run cannot start, its output is incomplete, or its
/// results do not match the expected harness set exactly.
///
/// # Examples
///
/// The command is validated before it runs, so a redirected or suppressed run
/// is refused without starting a verifier:
///
/// ```
/// use std::fs;
/// use std::path::PathBuf;
///
/// use chutoro_test_support::ci::kani_suite::manifest::Tier;
/// use chutoro_test_support::ci::kani_suite::{CheckError, Request, check};
///
/// let manifest = std::env::temp_dir().join(format!(
///     "chutoro-kani-check-{}.toml",
///     std::process::id()
/// ));
/// fs::write(
///     &manifest,
///     "version = 1\n[[harness]]\npackage = \"chutoro-core\"\n\
///      pretty_name = \"proofs::verify_one\"\ntier = \"fast\"\n",
/// )
/// .expect("write a manifest");
/// let request = Request {
///     manifest,
///     package: "chutoro-core".to_owned(),
///     scope: Tier::Fast,
///     command: vec![
///         "cargo".to_owned(),
///         "kani".to_owned(),
///         "-p".to_owned(),
///         "chutoro-core".to_owned(),
///         "--lib".to_owned(),
///         "--quiet".to_owned(),
///     ],
/// };
/// assert!(matches!(check(&request), Err(CheckError::Command(_))));
/// ```
pub fn check(request: &Request) -> Result<Verified, CheckError> {
    let manifest = Manifest::load(&request.manifest)?;
    let allowed = manifest.scope(&request.package, request.scope);
    if allowed.is_empty() {
        return Err(CheckError::EmptyScope {
            package: request.package.clone(),
            scope: request.scope,
        });
    }
    let delegated = runner::parse_command(&request.command, &request.package)?;
    let expected = expected_set(request, &delegated, &allowed)?;
    let outcome = process::run(&delegated)?;
    let parsed = output::parse(&outcome.stdout);
    if !outcome.status.success() {
        let failed = parsed
            .as_ref()
            .map(|report| {
                report
                    .unsuccessful()
                    .into_iter()
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        return Err(CheckError::DelegatedFailed {
            status: outcome.status.to_string(),
            failed,
        });
    }
    verdict::evaluate(&expected, &parsed?)?;
    Ok(Verified {
        package: request.package.clone(),
        scope: request.scope,
        harnesses: expected.into_iter().collect(),
    })
}

/// Return the harnesses the run must complete successfully.
///
/// Without selectors the run must cover the whole package tier, which is what
/// makes a package-wide sweep meaningful. With selectors it must cover exactly
/// the named harnesses, each of which must belong to the requested scope.
fn expected_set(
    request: &Request,
    delegated: &runner::Delegated,
    allowed: &BTreeSet<&str>,
) -> Result<BTreeSet<String>, CheckError> {
    if delegated.selectors.is_empty() {
        return Ok(allowed.iter().map(|name| (*name).to_owned()).collect());
    }
    let outside = delegated
        .selectors
        .iter()
        .filter(|selector| !allowed.contains(selector.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !outside.is_empty() {
        return Err(CheckError::SelectorOutsideScope {
            package: request.package.clone(),
            scope: request.scope,
            names: outside,
        });
    }
    Ok(delegated.selectors.iter().cloned().collect())
}
