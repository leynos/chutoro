//! Testing adapters and a stub recipe for dataset recipe consumers.

mod filesystem;
mod in_memory;
mod stub_recipe;

pub use filesystem::FilesystemFetcher;
pub use in_memory::{InMemoryFetcher, InMemoryPublisher, InMemoryStorage};
pub use stub_recipe::StubRecipe;
