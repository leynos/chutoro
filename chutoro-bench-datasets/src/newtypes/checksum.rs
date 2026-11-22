//! Checksum value type and digest algorithm selection.
//!
//! A [`Checksum`] pairs a [`DigestAlgorithm`] with the algorithm's fixed-width
//! digest bytes. Its textual form is `<algorithm>:<hex>`, where the only
//! supported algorithm is `sha256` and the payload is 64 bytes of
//! hexadecimal ASCII. Parsing accepts either hexadecimal case, while [`fmt::Display`]
//! always emits the canonical lowercase form so that a checksum read back from
//! a manifest compares equal to one parsed from any case variant.
//!
//! Parsing a checksum does not verify fetched bytes. Fetch-time verification
//! remains part of roadmap item `10.1.2`.

use std::{fmt, sync::Arc};

use crate::{IntegrityFailure, RecipeError};

/// Digest algorithm named by a checksum.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DigestAlgorithm {
    /// SHA-256, whose digest is 32 bytes.
    Sha256,
}

impl DigestAlgorithm {
    /// Return the case-sensitive textual prefix that names this algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::DigestAlgorithm;
    ///
    /// assert_eq!(DigestAlgorithm::Sha256.prefix(), "sha256");
    /// ```
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
        }
    }

    /// Return the digest width in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::DigestAlgorithm;
    ///
    /// assert_eq!(DigestAlgorithm::Sha256.digest_len(), 32);
    /// ```
    #[must_use]
    pub const fn digest_len(self) -> usize {
        match self {
            Self::Sha256 => 32,
        }
    }

    /// Return the hexadecimal payload width in bytes.
    const fn digest_hex_len(self) -> usize {
        self.digest_len() * 2
    }

    /// Resolve a case-sensitive algorithm prefix to its algorithm.
    fn from_prefix(prefix: &str) -> Option<Self> {
        if prefix == Self::Sha256.prefix() {
            Some(Self::Sha256)
        } else {
            None
        }
    }
}

impl fmt::Display for DigestAlgorithm {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.prefix())
    }
}

/// Digest of a source artefact, tagged with the algorithm that produced it.
///
/// # Examples
///
/// ```
/// use chutoro_bench_datasets::{Checksum, DigestAlgorithm};
///
/// let checksum = Checksum::new(DigestAlgorithm::Sha256, [0; 32]);
/// assert_eq!(checksum.algorithm(), DigestAlgorithm::Sha256);
/// assert_eq!(
///     checksum.to_string(),
///     "sha256:0000000000000000000000000000000000000000000000000000000000000000",
/// );
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Checksum {
    /// Algorithm that produced the digest.
    algorithm: DigestAlgorithm,
    /// Raw digest bytes.
    bytes: [u8; 32],
}

impl Checksum {
    /// Construct a checksum from an algorithm and its digest bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{Checksum, DigestAlgorithm};
    ///
    /// let checksum = Checksum::new(DigestAlgorithm::Sha256, [0xab; 32]);
    /// assert_eq!(checksum.as_bytes(), &[0xab; 32]);
    /// ```
    #[must_use]
    pub const fn new(algorithm: DigestAlgorithm, bytes: [u8; 32]) -> Self {
        Self { algorithm, bytes }
    }

    /// Return the digest algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{Checksum, DigestAlgorithm};
    ///
    /// let text = format!("sha256:{}", "ab".repeat(32));
    /// let checksum = Checksum::parse(&text)?;
    /// assert_eq!(checksum.algorithm(), DigestAlgorithm::Sha256);
    /// # Ok::<(), chutoro_bench_datasets::RecipeError>(())
    /// ```
    #[must_use]
    pub const fn algorithm(&self) -> DigestAlgorithm {
        self.algorithm
    }

    /// Borrow the raw digest bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{Checksum, DigestAlgorithm};
    ///
    /// let text = format!("sha256:{}{}", "01", "00".repeat(31));
    /// let checksum = Checksum::parse(&text)?;
    /// assert_eq!(checksum.as_bytes().len(), 32);
    /// assert_eq!(checksum.as_bytes().first(), Some(&0x01));
    /// # Ok::<(), chutoro_bench_datasets::RecipeError>(())
    /// ```
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    /// Parse a checksum in `<algorithm>:<hex>` form.
    ///
    /// Checks run in a fixed order: the value must contain a `:` separator
    /// with a non-empty prefix, the prefix must name a supported algorithm
    /// with the exact case used by [`fmt::Display`], the payload must be the
    /// algorithm's full hexadecimal width in bytes, and every payload byte
    /// must be a hexadecimal digit. Hexadecimal digits may be upper- or
    /// lower-case; [`fmt::Display`] always emits lowercase.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::Checksum;
    ///
    /// let text = concat!(
    ///     "sha256:",
    ///     "0123456789abcdef0123456789abcdef",
    ///     "0123456789abcdef0123456789abcdef",
    /// );
    /// let checksum = Checksum::parse(text)?;
    /// assert_eq!(checksum.to_string(), text);
    /// # Ok::<(), chutoro_bench_datasets::RecipeError>(())
    /// ```
    ///
    /// A malformed digest is rejected with a typed failure:
    ///
    /// ```
    /// use chutoro_bench_datasets::{Checksum, IntegrityFailure, RecipeError};
    ///
    /// let error = Checksum::parse("sha256:abc")
    ///     .expect_err("a short digest is rejected");
    /// assert!(matches!(
    ///     error,
    ///     RecipeError::Integrity(failure)
    ///         if matches!(failure.as_ref(), IntegrityFailure::InvalidDigestLength { .. })
    /// ));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RecipeError::Integrity`] carrying an [`IntegrityFailure`] for
    /// a missing or empty algorithm prefix, an unknown algorithm, a digest of
    /// the wrong width, or a non-hexadecimal character.
    #[must_use = "handle the parsed checksum or its validation error"]
    pub fn parse(value: &str) -> Result<Self, RecipeError> {
        let (prefix, payload) = value
            .split_once(':')
            .ok_or_else(|| RecipeError::integrity(IntegrityFailure::MissingAlgorithmPrefix))?;
        if prefix.is_empty() {
            return Err(RecipeError::integrity(
                IntegrityFailure::MissingAlgorithmPrefix,
            ));
        }
        let algorithm = DigestAlgorithm::from_prefix(prefix).ok_or_else(|| {
            RecipeError::integrity(IntegrityFailure::UnknownAlgorithm {
                prefix: Arc::from(prefix),
            })
        })?;
        if payload.len() != algorithm.digest_hex_len() {
            return Err(RecipeError::integrity(
                IntegrityFailure::InvalidDigestLength {
                    algorithm,
                    expected: algorithm.digest_hex_len(),
                    actual: payload.len(),
                },
            ));
        }
        Ok(Self::new(algorithm, decode_digest(payload)?))
    }
}

impl fmt::Display for Checksum {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.algorithm.prefix())?;
        formatter.write_str(":")?;
        for byte in self.bytes {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Decode a hexadecimal payload whose width has already been validated.
///
/// The caller guarantees the payload is exactly [`DigestAlgorithm`]'s
/// hexadecimal width, so every byte of the destination array receives a pair
/// and no trailing byte is discarded.
fn decode_digest(payload: &str) -> Result<[u8; 32], RecipeError> {
    let mut bytes = [0u8; 32];
    let pairs = payload.as_bytes().as_chunks::<2>().0;
    for (index, (slot, &[high, low])) in bytes.iter_mut().zip(pairs).enumerate() {
        *slot = decode_digest_pair(high, low, index.saturating_mul(2))?;
    }
    Ok(bytes)
}

/// Decode one two-character hexadecimal pair beginning at `position`.
fn decode_digest_pair(high: u8, low: u8, position: usize) -> Result<u8, RecipeError> {
    let high_nibble = decode_hex_digit(high, position)?;
    let low_nibble = decode_hex_digit(low, position.saturating_add(1))?;
    Ok(high_nibble << 4 | low_nibble)
}

/// Decode one hexadecimal digit, reporting its byte position on failure.
fn decode_hex_digit(byte: u8, position: usize) -> Result<u8, RecipeError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(RecipeError::integrity(
            IntegrityFailure::InvalidHexCharacter { position },
        )),
    }
}
