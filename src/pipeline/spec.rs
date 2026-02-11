//! Pipeline specification types.
//!
//! A [`PipelineSpec`] describes which modules to use for each pipeline stage,
//! runtime execution limits, and strictness settings. These types are the
//! input to the [`super::validation::ValidationEngine`].
//!
//! # JSON shape
//!
//! ```json
//! {
//!   "v": 1,
//!   "preset": "textrank",
//!   "modules": {
//!     "candidates": "word_nodes",
//!     "graph": "cooccurrence_window",
//!     "rank": "standard_pagerank"
//!   },
//!   "runtime": { "max_tokens": 200000 },
//!   "strict": false
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Top-level pipeline specification (v1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSpec {
    /// Spec version (currently `1`).
    pub v: u32,

    /// Optional preset name used as a starting point (e.g., `"textrank"`).
    #[serde(default)]
    pub preset: Option<String>,

    /// Explicit module selections. Omitted modules inherit from the preset.
    #[serde(default)]
    pub modules: ModuleSet,

    /// Runtime execution limits.
    #[serde(default)]
    pub runtime: RuntimeSpec,

    /// If `true`, unrecognized fields are errors; if `false`, warnings.
    #[serde(default)]
    pub strict: bool,

    /// Captures any fields not recognized by the schema.
    /// Used by the strict-mode validation rule.
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, serde_json::Value>,
}

/// The set of modules selected for the pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModuleSet {
    #[serde(default)]
    pub candidates: Option<CandidateModuleType>,

    #[serde(default)]
    pub graph: Option<GraphModuleType>,

    #[serde(default)]
    pub graph_transforms: Vec<GraphTransformType>,

    #[serde(default)]
    pub teleport: Option<TeleportModuleType>,

    #[serde(default)]
    pub clustering: Option<ClusteringModuleType>,

    #[serde(default)]
    pub rank: Option<RankModuleType>,

    /// Captures any fields not recognized by the schema.
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, serde_json::Value>,
}

// ─── Module type enums ──────────────────────────────────────────────────────

/// Candidate selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateModuleType {
    /// Individual word tokens as candidates (standard TextRank family).
    WordNodes,
    /// Noun-phrase chunks as candidates (TopicRank/MultipartiteRank family).
    PhraseCandidates,
}

/// Graph construction strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphModuleType {
    /// Word co-occurrence within a sliding window.
    CooccurrenceWindow,
    /// Topic-level graph where nodes are phrase clusters (TopicRank).
    TopicGraph,
    /// Candidate-level graph with inter-cluster edges (MultipartiteRank).
    CandidateGraph,
}

impl GraphModuleType {
    /// Returns the user-facing name used in JSON and error messages.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CooccurrenceWindow => "cooccurrence_window",
            Self::TopicGraph => "topic_graph",
            Self::CandidateGraph => "candidate_graph",
        }
    }
}

/// Graph post-processing transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphTransformType {
    /// Remove edges between candidates in the same cluster.
    RemoveIntraClusterEdges,
    /// Apply alpha-boost weighting to first-occurring cluster members.
    AlphaBoost,
}

impl GraphTransformType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RemoveIntraClusterEdges => "remove_intra_cluster_edges",
            Self::AlphaBoost => "alpha_boost",
        }
    }
}

/// Teleport (personalization) strategy for PageRank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeleportModuleType {
    /// Uniform distribution (equivalent to no personalization).
    Uniform,
    /// Position-weighted: earlier tokens get higher teleport probability.
    Position,
    /// Focus-terms-biased: specified terms get boosted teleport probability.
    FocusTerms,
    /// Topic-weighted: per-lemma weights from external topic model.
    TopicWeights,
}

/// Clustering strategy for phrase candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusteringModuleType {
    /// Hierarchical agglomerative clustering with Jaccard distance.
    Hac,
}

/// PageRank variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankModuleType {
    /// Standard (unpersonalized) PageRank.
    StandardPagerank,
    /// Personalized PageRank with a teleport distribution.
    PersonalizedPagerank,
}

/// Runtime execution limits (fail-fast guards).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeSpec {
    /// Maximum number of input tokens before rejecting.
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Maximum number of graph nodes before rejecting.
    #[serde(default)]
    pub max_nodes: Option<usize>,

    /// Maximum number of graph edges before rejecting.
    #[serde(default)]
    pub max_edges: Option<usize>,

    /// Captures any fields not recognized by the schema.
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_minimal_spec() {
        let json = r#"{ "v": 1 }"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.v, 1);
        assert!(spec.modules.rank.is_none());
        assert!(!spec.strict);
    }

    #[test]
    fn test_deserialize_full_spec() {
        let json = r#"{
            "v": 1,
            "preset": "textrank",
            "modules": {
                "candidates": "word_nodes",
                "graph": "cooccurrence_window",
                "rank": "personalized_pagerank",
                "teleport": "position"
            },
            "runtime": { "max_tokens": 100000 },
            "strict": true
        }"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.preset.as_deref(), Some("textrank"));
        assert_eq!(spec.modules.rank, Some(RankModuleType::PersonalizedPagerank));
        assert_eq!(spec.modules.teleport, Some(TeleportModuleType::Position));
        assert_eq!(spec.runtime.max_tokens, Some(100000));
        assert!(spec.strict);
    }

    #[test]
    fn test_unknown_fields_captured() {
        let json = r#"{
            "v": 1,
            "bogus_top_level": 42,
            "modules": {
                "rank": "standard_pagerank",
                "bogus_module": "xyz"
            }
        }"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        assert!(spec.unknown_fields.contains_key("bogus_top_level"));
        assert!(spec.modules.unknown_fields.contains_key("bogus_module"));
    }

    #[test]
    fn test_serde_roundtrip() {
        let json = r#"{"v":1,"modules":{"rank":"personalized_pagerank","teleport":"focus_terms"}}"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        let back = serde_json::to_value(&spec).unwrap();
        assert_eq!(back["modules"]["rank"], "personalized_pagerank");
        assert_eq!(back["modules"]["teleport"], "focus_terms");
    }
}
