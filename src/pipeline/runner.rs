//! Pipeline runner — orchestrates stage execution and artifact flow.
//!
//! The runner chains stages together, threading artifacts between them and
//! notifying observers at each transition. It owns (or borrows) a
//! [`PipelineWorkspace`](super::artifacts::PipelineWorkspace) for buffer reuse
//! across invocations.
//!
//! Implementation will follow after E2 artifacts and E3 stage traits are
//! finalized (see E4: textranker-ii6).
