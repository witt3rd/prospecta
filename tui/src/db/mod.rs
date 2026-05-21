//! Database access — sqlx pool + per-topic query modules.
//!
//! Each submodule scopes its queries to one schema area (banks, documents,
//! events, etc.) to keep the surface readable. All queries go through `query!`
//! / `query_as!` so they're checked against the live schema at compile time.

pub mod banks;
pub mod documents;
pub mod events;

pub use banks::{Bank, BankSummary};
pub use documents::{Document, MemoryItem};
pub use events::{Event, EventKind};
