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
    #[must_use]
    pub fn with_homepage(mut self, homepage: SourceUrl) -> Self {
        self.homepage = Some(homepage);
        self
    }

    /// Set the citation text.
    #[must_use]
    pub fn with_citation(mut self, citation: impl Into<Arc<str>>) -> Self {
        self.citation = Some(citation.into());
        self
    }

    /// Set the SPDX licence identifier.
    #[must_use]
    pub fn with_licence_spdx(mut self, licence_spdx: impl Into<Arc<str>>) -> Self {
        self.licence_spdx = Some(licence_spdx.into());
        self
    }

    /// Set the short summary.
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<Arc<str>>) -> Self {
        self.summary = summary.into();
        self
    }
}
