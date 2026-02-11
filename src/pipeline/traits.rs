//! Stage trait definitions for the pipeline.
//!
//! Each trait represents one processing stage boundary. Implementations are
//! statically dispatched for performance; trait objects are available behind a
//! feature gate for dynamic composition.
//!
//! Trait signatures will be finalized in E3 (textranker-4a0). This module
//! currently serves as a placeholder to establish the import path.

// Stage traits will be defined here in E3 subtasks:
//   - Preprocessor          (textranker-4a0.1)
//   - CandidateSelector     (textranker-4a0.2)
//   - GraphBuilder          (textranker-4a0.3)
//   - GraphTransform        (textranker-4a0.4)
//   - TeleportBuilder       (textranker-4a0.5)
//   - Ranker                (textranker-4a0.6)
//   - PhraseBuilder         (textranker-4a0.7)
//   - ResultFormatter       (textranker-4a0.8)
