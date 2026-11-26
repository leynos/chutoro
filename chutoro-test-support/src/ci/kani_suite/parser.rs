//! Walk a delegated `cargo kani` argument list, rejecting what it must not pass.
//!
//! The walk is separate from the command's construction and validation so the
//! two can be read on their own: this module keeps the state of one walk and
//! records the selections and flags it finds, while [`super::words`] decides
//! what each single argument spells and the parent module decides whether the
//! collected command is one the checker can trust.

use super::runner::{CommandError, Delegated, FORBIDDEN_FLAGS, Invocation};
use super::words;

/// The state of one delegated-command walk.
#[derive(Debug, Default)]
pub(super) struct Parser {
    /// Packages the command selects.
    packages: Vec<String>,
    /// Harness selectors the command passes.
    selectors: Vec<String>,
    /// Whether `--exact` was supplied.
    exact: bool,
    /// Whether `--lib` was supplied.
    lib: bool,
}

impl Parser {
    /// Consume the argument at `index`, returning the next index to read.
    pub(super) fn consume(&mut self, args: &[String], index: usize) -> Result<usize, CommandError> {
        // The caller walks only while `index < args.len()`, so a missing
        // argument ends the walk rather than panicking on a slice index.
        let Some(argument) = args.get(index) else {
            return Ok(args.len());
        };
        if let Some((flag, reason)) = FORBIDDEN_FLAGS
            .iter()
            .find(|(flag, _)| words::matches_forbidden(argument, flag))
        {
            return Err(CommandError::Forbidden {
                flag: (*flag).to_owned(),
                reason,
            });
        }
        let consumed = self.consume_separated(args, index, argument)?;
        if consumed > index + 1 {
            return Ok(consumed);
        }
        match argument.as_str() {
            "--exact" => self.exact = true,
            "--lib" => self.lib = true,
            "--bins" => return Err(CommandError::AllBinaries),
            _ => self.consume_attached(argument)?,
        }
        Ok(index + 1)
    }

    /// Consume an option whose value is the following argument.
    ///
    /// Returns the index to read next, which is `index + 1` when this option is
    /// not one of them, so the caller can fall through to the attached
    /// spellings and the flags.
    pub(super) fn consume_separated(
        &mut self,
        args: &[String],
        index: usize,
        argument: &str,
    ) -> Result<usize, CommandError> {
        let Some(valued) = words::Valued::of(argument) else {
            return Ok(index + 1);
        };
        let Some(value) = args.get(index + 1).cloned() else {
            return Err(CommandError::MissingValue {
                flag: argument.to_owned(),
            });
        };
        valued.store(self, value)?;
        let mut next = index + 2;
        if !valued.repeats() {
            return Ok(next);
        }
        // Cargo's package option takes a list, so every following word that is
        // not an option is another package. Recording only the first would let
        // a second package's harness satisfy the expected set.
        while let Some(extra) = args.get(next) {
            if extra.starts_with('-') {
                break;
            }
            self.push_package(extra.clone());
            next += 1;
        }
        Ok(next)
    }

    /// Consume an option whose value is attached with `=`.
    pub(super) fn consume_attached(&mut self, argument: &str) -> Result<(), CommandError> {
        if let Some(value) = argument.strip_prefix("--output-format=") {
            words::check_output_format(value)?;
        } else if let Some(value) = argument.strip_prefix("--harness=") {
            self.selectors.push(value.to_owned());
        } else if let Some(value) = argument.strip_prefix("--bin=") {
            return Err(CommandError::BinaryTarget {
                name: value.to_owned(),
            });
        } else if let Some(value) = argument
            .strip_prefix("-p=")
            .or_else(|| argument.strip_prefix("--package="))
        {
            self.packages.push(value.to_owned());
        }
        Ok(())
    }

    /// Record one package.
    pub(super) fn push_package(&mut self, package: String) {
        self.packages.push(package);
    }

    /// Record one harness selector.
    pub(super) fn push_selector(&mut self, selector: String) {
        self.selectors.push(selector);
    }

    /// Check the collected state and build the delegated command.
    pub(super) fn finish(
        self,
        invocation: &Invocation,
        package: &str,
    ) -> Result<Delegated, CommandError> {
        if self.packages != [package.to_owned()] {
            return Err(CommandError::Package {
                expected: package.to_owned(),
                found: self.packages,
            });
        }
        if !self.lib {
            return Err(CommandError::MissingLib);
        }
        if !self.selectors.is_empty() && !self.exact {
            return Err(CommandError::NotExact);
        }
        let duplicates = words::repeated(&self.selectors);
        if !duplicates.is_empty() {
            return Err(CommandError::DuplicateSelector { names: duplicates });
        }
        Ok(Delegated {
            env: invocation.env.clone(),
            program: invocation.program.clone(),
            args: invocation.args.clone(),
            package: package.to_owned(),
            selectors: self.selectors,
        })
    }
}
