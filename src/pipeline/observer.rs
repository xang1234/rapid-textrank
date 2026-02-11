//! Pipeline observer — hooks for logging, profiling, and debugging.
//!
//! Observers receive notifications at stage boundaries without coupling to
//! stage logic. Use cases include timing stages, capturing intermediate
//! artifacts for debugging, and emitting structured telemetry.
//!
//! Implementation will follow after E2 artifacts and E3 stage traits are
//! finalized (see E4: textranker-ii6).
