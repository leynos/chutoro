//! Shared environment-variable name type for benchmark environment policy.
//!
//! Benchmarks read configuration from environment variables. Wrapping the
//! name keeps those keys distinct from arbitrary strings, so policy helpers
//! cannot accidentally receive a report path or other value where a variable
//! name is expected.

/// Name of a benchmark environment variable.
#[derive(Clone, Copy, Debug)]
pub struct EnvVarName<'a>(&'a str);

impl<'a> EnvVarName<'a> {
    /// Wrap the literal name of an environment variable.
    pub(crate) const fn new(name: &'a str) -> Self {
        Self(name)
    }

    /// Borrow the underlying name for environment lookups.
    pub(crate) const fn as_str(&self) -> &'a str {
        self.0
    }
}
