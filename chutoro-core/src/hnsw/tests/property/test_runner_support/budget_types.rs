//! Value types for property-test budget selection.
//!
//! These newtypes carry the validated case counts, shrink budgets, and
//! runner stack sizes consumed by `budget_selection`.

/// Number of test cases to execute in a property test run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TestCases(u32);

/// Error returned when test case count is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InvalidTestCasesError;

impl std::fmt::Display for InvalidTestCasesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "test cases must be > 0")
    }
}

impl std::error::Error for InvalidTestCasesError {}

impl TestCases {
    /// Creates a new `TestCases` value, returning an error if invalid.
    ///
    /// # Errors
    /// Returns `InvalidTestCasesError` if `cases` is zero.
    pub(crate) fn try_new(cases: u32) -> Result<Self, InvalidTestCasesError> {
        if cases > 0 {
            Ok(Self(cases))
        } else {
            Err(InvalidTestCasesError)
        }
    }

    /// Returns the number of test cases.
    pub(crate) fn get(self) -> u32 {
        self.0
    }
}

impl From<TestCases> for u32 {
    fn from(cases: TestCases) -> Self {
        cases.0
    }
}

/// Maximum number of shrinking iterations to attempt when minimizing a failing test case.
///
/// A value of 0 disables shrinking entirely. In practice, positive values are used to enable
/// counterexample minimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ShrinkIterations(u32);

impl ShrinkIterations {
    /// Creates a new `ShrinkIterations` value.
    ///
    /// Setting `iterations` to 0 disables shrinking.
    pub(crate) fn new(iterations: u32) -> Self {
        Self(iterations)
    }

    /// Returns the number of shrink iterations.
    ///
    /// A return value of 0 means shrinking is disabled.
    pub(crate) fn get(self) -> u32 {
        self.0
    }
}

impl From<ShrinkIterations> for u32 {
    fn from(iters: ShrinkIterations) -> Self {
        iters.0
    }
}

/// Stack size in bytes for property test runner threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StackSize(usize);

/// Error returned when stack size is below the minimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InvalidStackSizeError {
    provided: usize,
    minimum: usize,
}

impl std::fmt::Display for InvalidStackSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "stack size must be >= {} bytes, got {}",
            self.minimum, self.provided
        )
    }
}

impl std::error::Error for InvalidStackSizeError {}

impl StackSize {
    /// Minimum safe stack size (1 MiB).
    const MIN_STACK_SIZE: usize = 1024 * 1024;

    /// Creates a new `StackSize` value, returning an error if below minimum.
    ///
    /// # Errors
    /// Returns `InvalidStackSizeError` if `size` is below `MIN_STACK_SIZE`.
    pub(crate) fn try_new(size: usize) -> Result<Self, InvalidStackSizeError> {
        if size >= Self::MIN_STACK_SIZE {
            Ok(Self(size))
        } else {
            Err(InvalidStackSizeError {
                provided: size,
                minimum: Self::MIN_STACK_SIZE,
            })
        }
    }

    /// Returns the stack size in bytes.
    pub(crate) fn get(self) -> usize {
        self.0
    }
}

impl From<StackSize> for usize {
    fn from(size: StackSize) -> Self {
        size.0
    }
}
