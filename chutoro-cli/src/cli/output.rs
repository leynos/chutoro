//! Output types and rendering helpers for CLI execution summaries.

use std::io::{self, Write};

use chutoro_core::ClusteringResult;

/// Summarises the outcome of executing a CLI command.
#[derive(Debug, Clone)]
pub struct ExecutionSummary {
    /// Name reported by the data source implementation.
    pub data_source: String,
    /// Cluster assignments produced by the clustering pipeline.
    pub result: ClusteringResult,
}

/// Renders `summary` to `writer` in a human-readable text format.
///
/// # Errors
/// Returns [`io::Error`] if writing to the supplied writer fails.
///
/// # Examples
/// ```
/// # use std::error::Error;
/// # use std::io::Cursor;
/// # use chutoro_cli::cli::{ExecutionSummary, render_summary};
/// # use chutoro_core::{ClusteringResult, ClusterId};
/// #
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let summary = ExecutionSummary {
///     data_source: "demo".into(),
///     result: ClusteringResult::from_assignments(vec![
///         ClusterId::new(0),
///         ClusterId::new(1),
///     ]),
/// };
/// let mut buffer = Cursor::new(Vec::new());
/// render_summary(&summary, &mut buffer)?;
/// assert_eq!(buffer.into_inner().len(), 38);
/// # Ok(())
/// # }
/// ```
pub fn render_summary(summary: &ExecutionSummary, mut writer: impl Write) -> io::Result<()> {
    writeln!(writer, "data source: {}", summary.data_source)?;
    writeln!(writer, "clusters: {}", summary.result.cluster_count())?;
    for (index, cluster) in summary.result.assignments().iter().enumerate() {
        writeln!(writer, "{index}\t{}", cluster.get())?;
    }
    Ok(())
}
