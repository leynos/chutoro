//! Pre-flight memory estimation for the CPU clustering pipeline.
//!
//! Provides a conservative estimate of peak memory consumption so callers can
//! reject oversized datasets before any allocation occurs.  The estimate is
//! intentionally pessimistic — it uses a safety multiplier to account for heap
//! fragmentation, Rayon thread-local buffers, and temporary allocations that
//! are difficult to predict statically. Search state is counted once per
//! concurrent search, because the batch core-distance stage runs one search per
//! Rayon worker, and each state is sized to the width that stage searches at.

#[cfg(feature = "cpu")]
mod batch;
mod estimate;
mod format;

#[cfg(feature = "cpu")]
pub use batch::{batch_search_width, estimate_peak_bytes_for_hnsw_params_with_concurrency};
pub use estimate::estimate_peak_bytes;
#[cfg(feature = "cpu")]
pub use estimate::estimate_peak_bytes_for_hnsw_params;
pub use format::format_bytes;
