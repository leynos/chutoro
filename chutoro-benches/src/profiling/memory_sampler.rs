//! Linux process resident-set sampling utilities.
//!
//! These helpers track peak resident set size (RSS) while an operation runs.

use std::{
    fs,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use super::ProfilingError;

/// Memory sampling output captured while running a benchmark operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PeakRssMeasurement {
    /// Wall-clock duration spent executing the profiled operation.
    pub elapsed: Duration,
    /// Maximum observed process resident-set size in bytes.
    pub peak_rss_bytes: u64,
}

/// Measures peak resident-set size while executing `operation`.
///
/// # Errors
///
/// Returns [`ProfilingError`] when sampling cannot be started, `/proc` values
/// cannot be parsed, or sampler thread coordination fails.
pub fn measure_peak_resident_set_size<T>(
    sample_interval: Duration,
    operation: impl FnOnce() -> T,
) -> Result<(T, PeakRssMeasurement), ProfilingError> {
    if sample_interval.is_zero() {
        return Err(ProfilingError::ZeroSamplingInterval);
    }
    #[cfg(target_os = "linux")]
    {
        measure_peak_resident_set_size_linux(sample_interval, operation)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = operation;
        Err(ProfilingError::UnsupportedPlatform {
            os: std::env::consts::OS,
        })
    }
}

#[cfg(target_os = "linux")]
fn measure_peak_resident_set_size_linux<T>(
    sample_interval: Duration,
    operation: impl FnOnce() -> T,
) -> Result<(T, PeakRssMeasurement), ProfilingError> {
    let running = Arc::new(AtomicBool::new(true));
    let peak_bytes = Arc::new(AtomicU64::new(read_vm_rss_bytes()?));
    let background_error = Arc::new(Mutex::new(None::<ProfilingError>));

    let running_handle = Arc::clone(&running);
    let peak_handle = Arc::clone(&peak_bytes);
    let error_handle = Arc::clone(&background_error);
    let sampler = thread::spawn(move || {
        while running_handle.load(Ordering::Relaxed) {
            match read_vm_rss_bytes() {
                Ok(bytes) => {
                    peak_handle.fetch_max(bytes, Ordering::Relaxed);
                }
                Err(err) => {
                    store_background_error(&error_handle, err);
                    break;
                }
            }
            thread::sleep(sample_interval);
        }
    });

    let started = Instant::now();
    let operation_result = operation();
    let elapsed = started.elapsed();

    running.store(false, Ordering::Relaxed);
    sampler
        .join()
        .map_err(|_| ProfilingError::SamplerThreadPanicked)?;

    peak_bytes.fetch_max(read_vm_rss_bytes()?, Ordering::Relaxed);
    let maybe_background_error = background_error
        .lock()
        .map_err(|_| ProfilingError::SamplerLockPoisoned)?
        .take();
    if let Some(err) = maybe_background_error {
        return Err(err);
    }

    Ok((
        operation_result,
        PeakRssMeasurement {
            elapsed,
            peak_rss_bytes: peak_bytes.load(Ordering::Relaxed),
        },
    ))
}

#[cfg(target_os = "linux")]
fn store_background_error(error_slot: &Mutex<Option<ProfilingError>>, error: ProfilingError) {
    if let Ok(mut guard) = error_slot.lock() {
        if guard.is_none() {
            *guard = Some(error);
        }
    }
}

#[cfg(target_os = "linux")]
fn read_vm_rss_bytes() -> Result<u64, ProfilingError> {
    let status = fs::read_to_string("/proc/self/status")?;
    parse_vm_rss_bytes(&status)
}

#[cfg(target_os = "linux")]
fn parse_vm_rss_bytes(status: &str) -> Result<u64, ProfilingError> {
    let field = "VmRSS";
    let line = status
        .lines()
        .find(|candidate| candidate.starts_with("VmRSS:"))
        .ok_or(ProfilingError::MissingProcField { field })?;
    parse_kibibyte_proc_field(line, field)
}

#[cfg(target_os = "linux")]
fn parse_kibibyte_proc_field(line: &str, field: &'static str) -> Result<u64, ProfilingError> {
    let mut parts = line.split_whitespace();
    let _label = parts
        .next()
        .ok_or(ProfilingError::MissingProcField { field })?;
    let value_raw = parts
        .next()
        .ok_or(ProfilingError::MissingProcField { field })?;
    let unit = parts.next().unwrap_or("kB");
    if unit != "kB" {
        return Err(ProfilingError::UnsupportedProcUnit {
            field,
            unit: unit.to_owned(),
        });
    }
    let kibibytes = value_raw
        .parse::<u64>()
        .map_err(|_| ProfilingError::InvalidProcField {
            field,
            value: value_raw.to_owned(),
        })?;
    kibibytes.checked_mul(1024).ok_or(ProfilingError::Overflow {
        context: "vm_rss_kibibytes_to_bytes",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[cfg(target_os = "linux")]
    #[rstest]
    #[case("VmRSS:\t1234 kB", 1_263_616)]
    #[case("VmRSS: 42 kB", 43_008)]
    fn parse_vm_rss_bytes_accepts_valid_lines(#[case] vmrss_line: &str, #[case] expected: u64) {
        let status = format!("{vmrss_line}\nName:\tchutoro\n");
        assert_eq!(
            parse_vm_rss_bytes(&status).expect("valid VmRSS field must parse"),
            expected
        );
    }

    #[cfg(target_os = "linux")]
    #[rstest]
    #[case::missing_field("Name:\tchutoro\n", "VmRSS")]
    #[case::invalid_numeric("VmRSS:\tnot-a-number kB\n", "VmRSS")]
    fn parse_vm_rss_bytes_rejects_invalid_input(#[case] status: &str, #[case] field: &'static str) {
        let err = parse_vm_rss_bytes(status).expect_err("invalid status must fail");
        assert!(
            matches!(err, ProfilingError::MissingProcField { field: actual } if actual == field)
                || matches!(err, ProfilingError::InvalidProcField { field: actual, .. } if actual == field),
        );
    }

    #[cfg(target_os = "linux")]
    #[rstest]
    fn parse_vm_rss_bytes_rejects_unexpected_unit() {
        let status = "VmRSS:\t200 MB\n";
        let err = parse_vm_rss_bytes(status).expect_err("unexpected unit must fail");
        assert!(matches!(
            err,
            ProfilingError::UnsupportedProcUnit { field: "VmRSS", .. }
        ));
    }

    #[rstest]
    fn measure_peak_resident_set_size_rejects_zero_interval() {
        let err = measure_peak_resident_set_size(Duration::from_secs(0), || 1_u8)
            .expect_err("zero interval must fail");
        assert!(matches!(err, ProfilingError::ZeroSamplingInterval));
    }
}
