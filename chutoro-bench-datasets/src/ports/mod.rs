//! Port traits that isolate recipes from infrastructure.

mod fetcher;
mod publisher;
mod storage;

pub use fetcher::Fetcher;
pub use publisher::Publisher;
pub use storage::Storage;
