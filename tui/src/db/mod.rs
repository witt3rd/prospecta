//! Database access — sqlx pool + per-topic query modules.
//!
//! Each submodule scopes its queries to one schema area (banks, documents,
//! events, etc.) to keep the surface readable. All queries go through `query!`
//! / `query_as!` so they're checked against the live schema at compile time.

pub mod banks;
pub mod dashboard;
pub mod documents;
pub mod events;
pub mod recall_thread;
pub mod retain_thread;

pub use banks::{Bank, BankSummary};
pub use dashboard::{DashboardStats, LlmCallStat, SweepStat};
pub use documents::{Document, MemoryItem};
pub use events::{Event, EventKind};
pub use recall_thread::{FormulateRow, LlmCallRow, RecallRow, RecallThread};
pub use retain_thread::{DocumentRow, IndexTextCall, ItemRow, RetainRow, RetainThread};
