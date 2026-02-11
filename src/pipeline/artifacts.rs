//! First-class pipeline artifacts.
//!
//! Each type represents a typed intermediate result flowing between pipeline
//! stages. Artifacts use interned IDs internally; string materialization is
//! deferred to the formatting boundary ([`FormattedResult`]).
//!
//! **Owned vs Borrowed**: Hot-path stage interfaces accept `*Ref<'a>` borrows;
//! the pipeline retains ownership of the corresponding owned artifacts.

// Placeholder types — each will be fleshed out in dedicated subtasks
// (textranker-1e8.2 through textranker-1e8.7).

/// Canonical token stream produced by the preprocessor stage.
///
/// Holds interned text/lemma/POS IDs, offsets, and sentence boundaries.
/// See also [`TokenStreamRef`] for the borrowed view used by downstream stages.
pub struct TokenStream {
    _private: (),
}

/// Borrowed view into a [`TokenStream`].
///
/// Stages accept this to avoid allocation on the hot path.
pub struct TokenStreamRef<'a> {
    _private: std::marker::PhantomData<&'a ()>,
}

/// Set of candidate nodes (word-level or phrase-level) selected for graph
/// construction.
///
/// Supports both word-node candidates (TextRank, PositionRank, etc.) and
/// phrase-level candidates (TopicRank, MultipartiteRank).
pub struct CandidateSet {
    _private: (),
}

/// Borrowed view into a [`CandidateSet`].
pub struct CandidateSetRef<'a> {
    _private: std::marker::PhantomData<&'a ()>,
}

/// Pipeline-level graph artifact wrapping the CSR-backed adjacency + weights.
///
/// Includes a node-index mapping from internal candidate IDs to CSR indices,
/// preserving cache-friendly iteration.
pub struct Graph {
    _private: (),
}

/// PageRank output: per-node scores, convergence info, and optional diagnostics.
pub struct RankOutput {
    _private: (),
}

/// Pre-format phrase collection: scored phrases with interned lemma IDs.
///
/// Surface forms are lazily materialized only when needed for formatting.
pub struct PhraseSet {
    _private: (),
}

/// Borrowed view into a [`PhraseSet`].
pub struct PhraseSetRef<'a> {
    _private: std::marker::PhantomData<&'a ()>,
}

/// Public-facing formatted output — the stability boundary.
///
/// Everything before this type is internal and may change; this type is the
/// public contract exposed to Python and JSON consumers.
pub struct FormattedResult {
    _private: (),
}

/// Reusable scratch buffers for reducing allocator churn across repeated
/// pipeline invocations (common in Python batch processing).
pub struct PipelineWorkspace {
    _private: (),
}
