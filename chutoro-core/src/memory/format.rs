//! Human-readable formatting for byte counts.
//!
//! Kept separate from the estimator so each module owns one responsibility.

/// Number of bytes in one kibibyte.
const KIB: u64 = 1024;

/// Number of bytes in one mebibyte.
const MIB: u64 = 1024 * KIB;

/// Number of bytes in one gibibyte.
const GIB: u64 = 1024 * MIB;

/// Number of bytes in one tebibyte.
const TIB: u64 = 1024 * GIB;

/// Selects the appropriate binary unit and divisor for a byte count.
const fn binary_unit(bytes: u64) -> (&'static str, u64) {
    if bytes >= TIB {
        ("TiB", TIB)
    } else if bytes >= GIB {
        ("GiB", GIB)
    } else if bytes >= MIB {
        ("MiB", MIB)
    } else {
        ("KiB", KIB)
    }
}

/// Formats a byte count as a human-readable string using binary units.
///
/// Returns values like `"0 B"`, `"1.0 KiB"`, `"2.4 GiB"`.  The result uses
/// one decimal place for values ≥ 1 KiB.
///
/// # Examples
///
/// ```
/// use chutoro_core::format_bytes;
///
/// assert_eq!(format_bytes(0), "0 B");
/// assert_eq!(format_bytes(1023), "1023 B");
/// assert_eq!(format_bytes(1024), "1.0 KiB");
/// assert_eq!(format_bytes(1_073_741_824), "1.0 GiB");
/// ```
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    if bytes < KIB {
        return format!("{bytes} B");
    }
    let (label, divisor) = binary_unit(bytes);
    let whole = bytes.div_euclid(divisor);
    let scaled_remainder = bytes.rem_euclid(divisor).saturating_mul(10);
    let tenths = scaled_remainder.div_euclid(divisor);
    let remaining_fraction = scaled_remainder.rem_euclid(divisor);
    let half_divisor = divisor.div_euclid(2);
    let should_round_up = remaining_fraction > half_divisor
        || (remaining_fraction == half_divisor && tenths.rem_euclid(2) == 1);

    if should_round_up && tenths == 9 {
        format!("{}.0 {label}", whole.saturating_add(1))
    } else {
        let displayed_tenths = tenths + u64::from(should_round_up);
        format!("{whole}.{displayed_tenths} {label}")
    }
}

#[cfg(test)]
mod tests {
    //! Tests for byte-count formatting.

    use rstest::rstest;

    use super::format_bytes;

    #[rstest]
    #[case::zero(0, "0 B")]
    #[case::small(512, "512 B")]
    #[case::just_below_kib(1023, "1023 B")]
    #[case::one_kib(1024, "1.0 KiB")]
    #[case::rounds_half_to_even_down(1280, "1.2 KiB")]
    #[case::one_and_half_kib(1536, "1.5 KiB")]
    #[case::rounds_half_to_even_up(1792, "1.8 KiB")]
    #[case::one_mib(1_048_576, "1.0 MiB")]
    #[case::one_gib(1_073_741_824, "1.0 GiB")]
    #[case::one_tib(1_099_511_627_776, "1.0 TiB")]
    #[case::two_point_four_gib(2_576_980_378, "2.4 GiB")]
    fn format_bytes_produces_expected_output(#[case] input: u64, #[case] expected: &str) {
        assert_eq!(format_bytes(input), expected);
    }
}
