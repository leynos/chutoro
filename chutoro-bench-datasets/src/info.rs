//! Dataset metadata exposed by a recipe.

use std::sync::Arc;

use crate::{RecipeId, RecipeVersion, SourceUrl};

/// Human-readable metadata for a benchmark dataset recipe.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetInfo {
    /// Stable recipe identifier.
    pub id: RecipeId,
    /// Recipe version.
    pub version: RecipeVersion,
    /// Optional dataset homepage.
    pub homepage: Option<SourceUrl>,
    /// Optional citation text.
    pub citation: Option<Arc<str>>,
    /// Optional SPDX licence identifier.
    pub licence_spdx: Option<Arc<str>>,
    /// Short human-readable summary.
    pub summary: Arc<str>,
}

impl DatasetInfo {
    /// Create metadata with empty optional fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{DatasetInfo, RecipeId, RecipeVersion};
    ///
    /// let info = DatasetInfo::new(RecipeId::new("mnist"), RecipeVersion::new(1, 0, 0));
    /// assert_eq!(info.summary.as_ref(), "");
    /// ```
    #[must_use]
    pub fn new(id: RecipeId, version: RecipeVersion) -> Self {
        Self {
            id,
            version,
            homepage: None,
            citation: None,
            licence_spdx: None,
            summary: Arc::from(""),
        }
    }

    /// Set the dataset homepage.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{DatasetInfo, RecipeId, RecipeVersion, SourceUrl};
    ///
    /// let homepage = SourceUrl::parse("https://example.test/datasets/mnist")?;
    /// let info = DatasetInfo::new(RecipeId::new("mnist"), RecipeVersion::new(1, 0, 0))
    ///     .with_homepage(homepage.clone());
    ///
    /// assert_eq!(info.homepage, Some(homepage));
    /// # Ok::<(), chutoro_bench_datasets::RecipeError>(())
    /// ```
    #[must_use]
    pub fn with_homepage(mut self, homepage: SourceUrl) -> Self {
        self.homepage = Some(homepage);
        self
    }

    /// Set the citation text.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{DatasetInfo, RecipeId, RecipeVersion};
    ///
    /// let info = DatasetInfo::new(RecipeId::new("mnist"), RecipeVersion::new(1, 0, 0))
    ///     .with_citation("LeCun et al., 1998");
    ///
    /// assert_eq!(info.citation.as_deref(), Some("LeCun et al., 1998"));
    /// ```
    #[must_use]
    pub fn with_citation(mut self, citation: impl Into<Arc<str>>) -> Self {
        self.citation = Some(citation.into());
        self
    }

    /// Set the SPDX licence identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{DatasetInfo, RecipeId, RecipeVersion};
    ///
    /// let info = DatasetInfo::new(RecipeId::new("mnist"), RecipeVersion::new(1, 0, 0))
    ///     .with_licence_spdx("CC-BY-4.0");
    ///
    /// assert_eq!(info.licence_spdx.as_deref(), Some("CC-BY-4.0"));
    /// ```
    #[must_use]
    pub fn with_licence_spdx(mut self, licence_spdx: impl Into<Arc<str>>) -> Self {
        self.licence_spdx = Some(licence_spdx.into());
        self
    }

    /// Set the short summary.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{DatasetInfo, RecipeId, RecipeVersion};
    ///
    /// let info = DatasetInfo::new(RecipeId::new("mnist"), RecipeVersion::new(1, 0, 0))
    ///     .with_summary("Hand-written digit images");
    ///
    /// assert_eq!(info.summary.as_ref(), "Hand-written digit images");
    /// ```
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<Arc<str>>) -> Self {
        self.summary = summary.into();
        self
    }
}
