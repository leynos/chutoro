//! Plain payload types carried by [`RecipeError`](super::RecipeError) variants.

use std::{fmt, sync::Arc};

use thiserror::Error;

use crate::{DigestAlgorithm, PortName, SourceUrl};

/// Port failure payload useful for tests and future adapters.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortFailure {
    /// Port that failed.
    pub port: PortName,
    /// Human-readable reason.
    pub reason: Arc<str>,
}

impl fmt::Display for PortFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "port {:?} failed: {}", self.port, self.reason)
    }
}

/// Fetch-size failure payload useful for tests and future adapters.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FetchSizeExceeded {
    /// Source that exceeded the cap.
    pub url: SourceUrl,
    /// Maximum byte count allowed.
    pub limit_bytes: usize,
}

impl fmt::Display for FetchSizeExceeded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "fetch exceeded max_bytes={}: {}",
            self.limit_bytes, self.url
        )
    }
}

/// Failure raised while parsing a source checksum.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum IntegrityFailure {
    /// The value has no `:` separator, or has nothing before it.
    #[error("checksum is missing an algorithm prefix")]
    MissingAlgorithmPrefix,
    /// The algorithm prefix is not recognised; matching is case-sensitive.
    #[error("unsupported checksum algorithm: {prefix}")]
    UnknownAlgorithm {
        /// Prefix exactly as it appeared in the rejected value.
        prefix: Arc<str>,
    },
    /// The digest payload is not the algorithm's expected byte width.
    #[error("invalid {algorithm} digest length: expected {expected} bytes, found {actual}")]
    InvalidDigestLength {
        /// Algorithm whose width was expected.
        algorithm: DigestAlgorithm,
        /// Required payload width in bytes.
        expected: usize,
        /// Payload width supplied, in bytes.
        actual: usize,
    },
    /// A payload byte is not a hexadecimal digit.
    #[error("invalid hexadecimal character at position {position}")]
    InvalidHexCharacter {
        /// Zero-based byte position within the digest payload.
        position: usize,
    },
}
