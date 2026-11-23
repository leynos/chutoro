//! Read and validate the checked-in Kani harness inventory.
//!
//! `tools/kani/harnesses.toml` is the single classification of the proof
//! surface. Both the workflow-contract suite and the `kani_suite` checker read
//! it, so this module accepts exactly the shape the contract writes and
//! rejects everything else: an unknown key, a missing field, an unknown tier,
//! and a duplicated identity are all errors rather than silently ignored
//! entries. A checker that guessed at a malformed manifest would compare a
//! run against a proof surface nobody agreed on.

use std::collections::BTreeSet;
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

/// The only manifest schema version this checker understands.
pub const MANIFEST_VERSION: u32 = 1;

/// The tier a proof is classified into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Runs on every pull request.
    Fast,
    /// Runs only in the nightly package sweep.
    Full,
}

/// One declared Kani harness.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Harness {
    /// The Cargo package the harness is compiled into.
    pub package: String,
    /// The fully qualified path Kani prints for the harness.
    pub pretty_name: String,
    /// The tier that decides when the harness runs.
    pub tier: Tier,
}

/// The on-disk manifest document.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestDocument {
    /// Schema version, read so a newer document is rejected explicitly.
    version: u32,
    /// The declared harnesses.
    harness: Vec<Harness>,
}

/// A reason a manifest cannot be used.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ManifestError {
    /// The file could not be read.
    #[error("cannot read harness manifest {path}: {detail}")]
    Read {
        /// The manifest path.
        path: String,
        /// The underlying read failure.
        detail: String,
    },
    /// The file is not valid TOML for this shape.
    #[error("cannot parse harness manifest {path}: {detail}")]
    Parse {
        /// The manifest path.
        path: String,
        /// The underlying parse failure.
        detail: String,
    },
    /// The document declares a schema this checker does not understand.
    #[error("unsupported harness manifest version {found}; expected {expected}")]
    Version {
        /// The version the document declares.
        found: u32,
        /// The version this checker understands.
        expected: u32,
    },
    /// The document declares no harness.
    #[error("the harness manifest declares no harness")]
    Empty,
    /// One entry leaves a required field blank.
    #[error("harness {index} has an empty {field}")]
    EmptyField {
        /// The entry's zero-based position.
        index: usize,
        /// The field that is empty.
        field: &'static str,
    },
    /// One entry does not name a qualified harness path.
    #[error("harness {index} names {pretty_name:?}, which is not a qualified harness path")]
    Unqualified {
        /// The entry's zero-based position.
        index: usize,
        /// The value that failed the check.
        pretty_name: String,
    },
    /// Two entries name the same package and harness.
    #[error("the harness manifest lists {package}::{pretty_name} more than once")]
    Duplicate {
        /// The duplicated package.
        package: String,
        /// The duplicated harness path.
        pretty_name: String,
    },
}

/// The parsed and validated inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Every declared harness, in document order.
    pub harnesses: Vec<Harness>,
}

impl Manifest {
    /// Read and validate the manifest at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError`] when the file cannot be read or parsed, when
    /// the document is empty, when an entry is malformed, or when an identity
    /// appears twice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs;
    ///
    /// use chutoro_test_support::ci::kani_suite::manifest::Manifest;
    ///
    /// let path = std::env::temp_dir().join(format!(
    ///     "chutoro-kani-manifest-{}.toml",
    ///     std::process::id()
    /// ));
    /// fs::write(
    ///     &path,
    ///     "version = 1\n[[harness]]\npackage = \"pkg\"\n\
    ///      pretty_name = \"proofs::verify_one\"\ntier = \"fast\"\n",
    /// )
    /// .expect("write a manifest");
    /// let manifest = Manifest::load(&path).expect("valid manifest");
    /// assert_eq!(manifest.harnesses.len(), 1);
    /// fs::remove_file(&path).expect("remove the manifest");
    /// ```
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let text = std::fs::read_to_string(path).map_err(|error| ManifestError::Read {
            path: path.display().to_string(),
            detail: error.to_string(),
        })?;
        Self::parse(&text, &path.display().to_string())
    }

    /// Validate a manifest document held in memory.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError`] for the same conditions as [`Self::load`].
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::kani_suite::manifest::Manifest;
    ///
    /// let document = "version = 1\n[[harness]]\npackage = \"pkg\"\n\
    ///                  pretty_name = \"verify_one\"\ntier = \"full\"";
    /// assert!(Manifest::parse(document, "inline").is_err());
    /// ```
    pub fn parse(text: &str, origin: &str) -> Result<Self, ManifestError> {
        let document: ManifestDocument =
            toml::from_str(text).map_err(|error| ManifestError::Parse {
                path: origin.to_owned(),
                detail: error.to_string(),
            })?;
        if document.version != MANIFEST_VERSION {
            return Err(ManifestError::Version {
                found: document.version,
                expected: MANIFEST_VERSION,
            });
        }
        if document.harness.is_empty() {
            return Err(ManifestError::Empty);
        }
        validate_entries(&document.harness)?;
        Ok(Self {
            harnesses: document.harness,
        })
    }

    /// Return every harness in `package` and `tier`, sorted by harness path.
    ///
    /// A [`Tier::Full`] scope returns both tiers, because the nightly sweep
    /// runs the package's whole proof surface rather than only the proofs the
    /// fast tier leaves out.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::kani_suite::manifest::{Manifest, Tier};
    ///
    /// let document = "version = 1\n[[harness]]\npackage = \"pkg\"\n\
    ///                  pretty_name = \"proofs::verify_one\"\ntier = \"fast\"";
    /// let manifest = Manifest::parse(document, "inline").expect("valid");
    /// assert_eq!(manifest.scope("pkg", Tier::Full).len(), 1);
    /// assert!(manifest.scope("other", Tier::Full).is_empty());
    /// ```
    #[must_use]
    pub fn scope(&self, package: &str, tier: Tier) -> BTreeSet<&str> {
        self.harnesses
            .iter()
            .filter(|harness| harness.package == package)
            .filter(|harness| tier == Tier::Full || harness.tier == Tier::Fast)
            .map(|harness| harness.pretty_name.as_str())
            .collect()
    }
}

/// Reject an entry whose fields are blank, unqualified, or duplicated.
fn validate_entries(entries: &[Harness]) -> Result<(), ManifestError> {
    let mut seen: BTreeSet<(&str, &str)> = BTreeSet::new();
    for (index, entry) in entries.iter().enumerate() {
        validate_entry(index, entry)?;
        if !seen.insert((&entry.package, &entry.pretty_name)) {
            return Err(ManifestError::Duplicate {
                package: entry.package.clone(),
                pretty_name: entry.pretty_name.clone(),
            });
        }
    }
    Ok(())
}

/// Reject one entry whose package or harness path is not usable.
fn validate_entry(index: usize, entry: &Harness) -> Result<(), ManifestError> {
    if entry.package.trim().is_empty() {
        return Err(ManifestError::EmptyField {
            index,
            field: "package",
        });
    }
    if entry.pretty_name.trim().is_empty() {
        return Err(ManifestError::EmptyField {
            index,
            field: "pretty_name",
        });
    }
    if !is_qualified(&entry.pretty_name) {
        return Err(ManifestError::Unqualified {
            index,
            pretty_name: entry.pretty_name.clone(),
        });
    }
    Ok(())
}

/// Report whether `name` is a `::`-separated path of Rust identifiers.
fn is_qualified(name: &str) -> bool {
    name.split("::").count() > 1
        && name.split("::").all(|segment| {
            let mut characters = segment.chars();
            characters
                .next()
                .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
                && characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
        })
}

#[cfg(test)]
mod tests {
    //! Unit tests for manifest parsing and validation.

    use super::*;

    /// Return a one-entry manifest document with the supplied tier.
    fn document(tier: &str, package: &str, pretty_name: &str) -> String {
        format!("version = 1\n{}", entry(tier, package, pretty_name))
    }

    /// Return one `[[harness]]` table, without a document header.
    fn entry(tier: &str, package: &str, pretty_name: &str) -> String {
        format!(
            "[[harness]]\npackage = \"{package}\"\n\
             pretty_name = \"{pretty_name}\"\ntier = \"{tier}\"\n"
        )
    }

    #[test]
    fn parses_a_valid_manifest() {
        let manifest = Manifest::parse(&document("fast", "pkg", "proofs::verify_one"), "inline")
            .expect("valid manifest");
        assert_eq!(manifest.harnesses.len(), 1);
        assert_eq!(
            manifest.harnesses.first().expect("one harness").tier,
            Tier::Fast
        );
    }

    #[test]
    fn rejects_an_unknown_tier() {
        assert!(matches!(
            Manifest::parse(&document("nightly", "pkg", "proofs::verify_one"), "inline"),
            Err(ManifestError::Parse { .. })
        ));
    }

    #[test]
    fn rejects_an_unknown_entry_key() {
        let text = concat!(
            "version = 1\n[[harness]]\npackage = \"pkg\"\n",
            "pretty_name = \"proofs::verify_one\"\ntier = \"fast\"\n",
            "owner = \"someone\"\n",
        );
        assert!(matches!(
            Manifest::parse(text, "inline"),
            Err(ManifestError::Parse { .. })
        ));
    }

    #[test]
    fn rejects_a_missing_field() {
        let text = "version = 1\n[[harness]]\npackage = \"pkg\"\ntier = \"fast\"\n";
        assert!(matches!(
            Manifest::parse(text, "inline"),
            Err(ManifestError::Parse { .. })
        ));
    }

    #[test]
    fn rejects_a_duplicate_identity() {
        let text = format!(
            "version = 1\n{}{}",
            entry("fast", "pkg", "proofs::verify_one"),
            entry("full", "pkg", "proofs::verify_one"),
        );
        assert_eq!(
            Manifest::parse(&text, "inline"),
            Err(ManifestError::Duplicate {
                package: "pkg".to_owned(),
                pretty_name: "proofs::verify_one".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_an_unqualified_path() {
        assert_eq!(
            Manifest::parse(&document("fast", "pkg", "verify_one"), "inline"),
            Err(ManifestError::Unqualified {
                index: 0,
                pretty_name: "verify_one".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_an_unsupported_version() {
        let text = "version = 2\n[[harness]]\npackage = \"pkg\"\n\
                    pretty_name = \"proofs::verify_one\"\ntier = \"fast\"\n";
        assert_eq!(
            Manifest::parse(text, "inline"),
            Err(ManifestError::Version {
                found: 2,
                expected: MANIFEST_VERSION,
            })
        );
    }

    #[test]
    fn a_full_scope_includes_fast_harnesses() {
        let text = format!(
            "version = 1\n{}{}",
            entry("fast", "pkg", "proofs::verify_fast"),
            entry("full", "pkg", "proofs::verify_full"),
        );
        let manifest = Manifest::parse(&text, "inline").expect("valid manifest");
        assert_eq!(manifest.scope("pkg", Tier::Fast).len(), 1);
        assert_eq!(manifest.scope("pkg", Tier::Full).len(), 2);
    }

    #[test]
    fn load_reads_a_manifest_from_disk() {
        let path = std::env::temp_dir().join(format!(
            "chutoro-kani-manifest-load-{}.toml",
            std::process::id()
        ));
        std::fs::write(&path, document("fast", "pkg", "proofs::verify_one"))
            .expect("write a manifest");
        let loaded = Manifest::load(&path);
        std::fs::remove_file(&path).expect("remove the manifest");
        assert_eq!(loaded.expect("valid manifest").harnesses.len(), 1);
    }

    #[test]
    fn load_reports_an_unreadable_path() {
        let path = std::env::temp_dir().join("chutoro-kani-manifest-absent.toml");
        assert!(matches!(
            Manifest::load(&path),
            Err(ManifestError::Read { .. })
        ));
    }
}
