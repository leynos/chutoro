//! One identifier's evidence within a verdict.

use serde::{Deserialize, Serialize};

use super::CaseClass;

/// Evidence for one benchmark identifier.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CaseVerdict {
    /// Criterion's full benchmark identifier.
    pub id: String,
    /// Verdict class assigned to the identifier.
    pub class: CaseClass,
    /// Lower bound of the mean change interval, when compared.
    pub lower_bound: Option<f64>,
    /// Upper bound of the mean change interval, when compared.
    pub upper_bound: Option<f64>,
    /// Reason the identifier could not be compared, when it could not.
    pub reason: Option<String>,
}

impl CaseVerdict {
    /// Build a verdict entry for a classified interval.
    ///
    ///
    /// # Parameters
    ///
    /// - `id`: Criterion's full benchmark identifier.
    /// - `class`: verdict class for the identifier.
    /// - `lower_bound`: fractional lower bound of the mean change.
    /// - `upper_bound`: fractional upper bound of the mean change.
    ///
    /// # Returns
    ///
    /// A compared case carrying both fractional bounds.
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::CaseVerdict;
    /// use chutoro_test_support::ci::benchmark_baseline::CaseClass;
    ///
    /// let verdict = CaseVerdict::classified("fixture/alpha", CaseClass::WithinNoise, 0.01, 0.02);
    /// assert_eq!(verdict.class, CaseClass::WithinNoise);
    /// ```
    #[must_use]
    pub fn classified(id: &str, class: CaseClass, lower_bound: f64, upper_bound: f64) -> Self {
        Self {
            id: id.to_owned(),
            class,
            lower_bound: Some(lower_bound),
            upper_bound: Some(upper_bound),
            reason: None,
        }
    }

    /// Build a verdict entry for an identifier that was not compared.
    ///
    ///
    /// # Parameters
    ///
    /// - `id`: Criterion's full benchmark identifier.
    /// - `class`: verdict class for the identifier.
    /// - `reason`: why it was not compared.
    ///
    /// # Returns
    ///
    /// An uncompared case carrying its reason and no bounds.
    /// # Examples
    ///
    /// ```
    /// use chutoro_test_support::ci::benchmark_baseline::{CaseClass, CaseVerdict};
    ///
    /// let verdict = CaseVerdict::not_compared("fixture/beta", CaseClass::NewUnverified, "no reference");
    /// assert!(verdict.lower_bound.is_none());
    /// ```
    #[must_use]
    pub fn not_compared(id: &str, class: CaseClass, reason: &str) -> Self {
        Self {
            id: id.to_owned(),
            class,
            lower_bound: None,
            upper_bound: None,
            reason: Some(reason.to_owned()),
        }
    }
}
