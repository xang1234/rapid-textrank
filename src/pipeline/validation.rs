//! Validation engine for pipeline specifications.
//!
//! The engine runs all registered [`ValidationRule`]s against a
//! [`PipelineSpec`](super::spec::PipelineSpec) and collects every diagnostic
//! into a [`ValidationReport`] — it never short-circuits on the first error,
//! so users see all problems at once.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use rapid_textrank::pipeline::validation::ValidationEngine;
//!
//! let engine = ValidationEngine::with_defaults();
//! let report = engine.validate(&spec);
//! if report.has_errors() {
//!     for err in report.errors() {
//!         eprintln!("{err}");
//!     }
//! }
//! ```

use serde::Serialize;

use super::error_code::ErrorCode;
use super::errors::PipelineSpecError;
use super::spec::*;

// ─── Severity ───────────────────────────────────────────────────────────────

/// Whether a diagnostic is a hard error or a soft warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
}

// ─── Diagnostic ─────────────────────────────────────────────────────────────

/// A single validation finding — an error or warning attached to a
/// [`PipelineSpecError`] that carries the code, path, message, and hint.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationDiagnostic {
    pub severity: Severity,
    #[serde(flatten)]
    pub error: PipelineSpecError,
}

impl ValidationDiagnostic {
    pub fn error(err: PipelineSpecError) -> Self {
        Self {
            severity: Severity::Error,
            error: err,
        }
    }

    pub fn warning(err: PipelineSpecError) -> Self {
        Self {
            severity: Severity::Warning,
            error: err,
        }
    }
}

// ─── Report ─────────────────────────────────────────────────────────────────

/// Collected diagnostics from running all validation rules.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ValidationReport {
    pub diagnostics: Vec<ValidationDiagnostic>,
}

impl ValidationReport {
    /// Iterate over error-severity diagnostics.
    pub fn errors(&self) -> impl Iterator<Item = &PipelineSpecError> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| &d.error)
    }

    /// Iterate over warning-severity diagnostics.
    pub fn warnings(&self) -> impl Iterator<Item = &PipelineSpecError> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .map(|d| &d.error)
    }

    /// Returns `true` if any diagnostic is an error.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    }

    /// Returns `true` if there are no errors (warnings are acceptable).
    pub fn is_valid(&self) -> bool {
        !self.has_errors()
    }

    /// Total number of diagnostics (errors + warnings).
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// Returns `true` if there are no diagnostics at all.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

// ─── Rule trait ─────────────────────────────────────────────────────────────

/// A single validation rule that inspects a [`PipelineSpec`] and returns
/// zero or more diagnostics.
///
/// Rules are stateless and must be `Send + Sync` so they can be shared
/// across threads (e.g., in a long-lived validation engine).
pub trait ValidationRule: Send + Sync {
    /// Short, stable identifier for this rule (e.g., `"rank_teleport"`).
    fn name(&self) -> &str;

    /// Inspect `spec` and return any findings.
    fn validate(&self, spec: &PipelineSpec) -> Vec<ValidationDiagnostic>;
}

// ─── Engine ─────────────────────────────────────────────────────────────────

/// Runs a set of [`ValidationRule`]s against a [`PipelineSpec`] and collects
/// all diagnostics into a [`ValidationReport`].
pub struct ValidationEngine {
    rules: Vec<Box<dyn ValidationRule>>,
}

impl ValidationEngine {
    /// Create an empty engine with no rules.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Create an engine pre-loaded with the default rule set.
    pub fn with_defaults() -> Self {
        let mut engine = Self::new();
        engine.add_rule(Box::new(RankTeleportRule));
        engine.add_rule(Box::new(TopicGraphDepsRule));
        engine.add_rule(Box::new(GraphTransformDepsRule));
        engine.add_rule(Box::new(RuntimeLimitsRule));
        engine.add_rule(Box::new(UnknownFieldsRule));
        engine
    }

    /// Register an additional rule.
    pub fn add_rule(&mut self, rule: Box<dyn ValidationRule>) {
        self.rules.push(rule);
    }

    /// Run all rules against `spec` and return the collected report.
    pub fn validate(&self, spec: &PipelineSpec) -> ValidationReport {
        let mut report = ValidationReport::default();
        for rule in &self.rules {
            report.diagnostics.extend(rule.validate(spec));
        }
        report
    }
}

impl Default for ValidationEngine {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Concrete rules
// ═══════════════════════════════════════════════════════════════════════════

// ─── 1. personalized_pagerank requires teleport ─────────────────────────────

struct RankTeleportRule;

impl ValidationRule for RankTeleportRule {
    fn name(&self) -> &str {
        "rank_teleport"
    }

    fn validate(&self, spec: &PipelineSpec) -> Vec<ValidationDiagnostic> {
        let is_personalized =
            spec.modules.rank == Some(RankModuleType::PersonalizedPagerank);

        if is_personalized && spec.modules.teleport.is_none() {
            vec![ValidationDiagnostic::error(
                PipelineSpecError::new(
                    ErrorCode::MissingStage,
                    "/modules/teleport",
                    "personalized_pagerank requires a teleport module",
                )
                .with_hint(
                    "Add a teleport module: position, focus_terms, \
                     topic_weights, or uniform",
                ),
            )]
        } else {
            vec![]
        }
    }
}

// ─── 2. topic_graph / candidate_graph require clustering + phrase_candidates ─

struct TopicGraphDepsRule;

impl ValidationRule for TopicGraphDepsRule {
    fn name(&self) -> &str {
        "topic_graph_deps"
    }

    fn validate(&self, spec: &PipelineSpec) -> Vec<ValidationDiagnostic> {
        let graph = match spec.modules.graph {
            Some(g)
                if g == GraphModuleType::TopicGraph
                    || g == GraphModuleType::CandidateGraph =>
            {
                g
            }
            _ => return vec![],
        };

        let mut out = Vec::new();

        if spec.modules.clustering.is_none() {
            out.push(ValidationDiagnostic::error(
                PipelineSpecError::new(
                    ErrorCode::MissingStage,
                    "/modules/clustering",
                    format!("{} requires a clustering module", graph.as_str()),
                )
                .with_hint("Add clustering: \"hac\""),
            ));
        }

        if spec.modules.candidates != Some(CandidateModuleType::PhraseCandidates) {
            out.push(ValidationDiagnostic::error(
                PipelineSpecError::new(
                    ErrorCode::InvalidCombo,
                    "/modules/candidates",
                    format!(
                        "{} requires phrase_candidates, not word_nodes",
                        graph.as_str()
                    ),
                )
                .with_hint("Set candidates to \"phrase_candidates\""),
            ));
        }

        out
    }
}

// ─── 3. remove_intra_cluster_edges requires clustering ──────────────────────

struct GraphTransformDepsRule;

impl ValidationRule for GraphTransformDepsRule {
    fn name(&self) -> &str {
        "graph_transform_deps"
    }

    fn validate(&self, spec: &PipelineSpec) -> Vec<ValidationDiagnostic> {
        let needs_clusters = spec
            .modules
            .graph_transforms
            .iter()
            .any(|t| *t == GraphTransformType::RemoveIntraClusterEdges);

        if needs_clusters && spec.modules.clustering.is_none() {
            vec![ValidationDiagnostic::error(
                PipelineSpecError::new(
                    ErrorCode::MissingStage,
                    "/modules/clustering",
                    "remove_intra_cluster_edges requires a clustering module",
                )
                .with_hint("Add clustering: \"hac\""),
            )]
        } else {
            vec![]
        }
    }
}

// ─── 4. Runtime limits must be positive when set ────────────────────────────

struct RuntimeLimitsRule;

impl ValidationRule for RuntimeLimitsRule {
    fn name(&self) -> &str {
        "runtime_limits"
    }

    fn validate(&self, spec: &PipelineSpec) -> Vec<ValidationDiagnostic> {
        let mut out = Vec::new();

        let checks: &[(&str, Option<usize>)] = &[
            ("max_tokens", spec.runtime.max_tokens),
            ("max_nodes", spec.runtime.max_nodes),
            ("max_edges", spec.runtime.max_edges),
        ];

        for &(field, value) in checks {
            if value == Some(0) {
                out.push(ValidationDiagnostic::error(
                    PipelineSpecError::new(
                        ErrorCode::LimitExceeded,
                        format!("/runtime/{field}"),
                        format!("{field} must be greater than 0"),
                    )
                    .with_hint(format!("Remove {field} to disable the limit, or set it to a positive value")),
                ));
            }
        }

        out
    }
}

// ─── 5. Unknown fields (strict → error, non-strict → warning) ──────────────

struct UnknownFieldsRule;

impl UnknownFieldsRule {
    /// Collect unknown-field diagnostics at the given JSON pointer `path`
    /// from a `HashMap` of extra fields captured by `#[serde(flatten)]`.
    fn check_unknowns(
        path: &str,
        unknowns: &std::collections::HashMap<String, serde_json::Value>,
        strict: bool,
    ) -> Vec<ValidationDiagnostic> {
        unknowns
            .keys()
            .map(|key| {
                let diag_fn = if strict {
                    ValidationDiagnostic::error
                } else {
                    ValidationDiagnostic::warning
                };
                diag_fn(
                    PipelineSpecError::new(
                        ErrorCode::UnknownField,
                        format!("{path}/{key}"),
                        format!("unrecognized field \"{key}\""),
                    )
                    .with_hint("Check spelling or remove this field"),
                )
            })
            .collect()
    }
}

impl ValidationRule for UnknownFieldsRule {
    fn name(&self) -> &str {
        "unknown_fields"
    }

    fn validate(&self, spec: &PipelineSpec) -> Vec<ValidationDiagnostic> {
        let mut out = Vec::new();
        out.extend(Self::check_unknowns("", &spec.unknown_fields, spec.strict));
        out.extend(Self::check_unknowns(
            "/modules",
            &spec.modules.unknown_fields,
            spec.strict,
        ));
        out.extend(Self::check_unknowns(
            "/runtime",
            &spec.runtime.unknown_fields,
            spec.strict,
        ));
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a PipelineSpec from JSON.
    fn spec(json: &str) -> PipelineSpec {
        serde_json::from_str(json).unwrap()
    }

    fn engine() -> ValidationEngine {
        ValidationEngine::with_defaults()
    }

    // ─── Valid specs ────────────────────────────────────────────────────

    #[test]
    fn test_minimal_spec_is_valid() {
        let report = engine().validate(&spec(r#"{ "v": 1 }"#));
        assert!(report.is_valid());
        assert!(report.is_empty());
    }

    #[test]
    fn test_standard_pagerank_without_teleport_is_valid() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "rank": "standard_pagerank" } }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_personalized_with_teleport_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "rank": "personalized_pagerank",
                    "teleport": "position"
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_topic_graph_with_all_deps_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": "phrase_candidates",
                    "graph": "topic_graph",
                    "clustering": "hac",
                    "rank": "standard_pagerank"
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_candidate_graph_with_all_deps_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": "phrase_candidates",
                    "graph": "candidate_graph",
                    "clustering": "hac",
                    "graph_transforms": ["remove_intra_cluster_edges"],
                    "rank": "standard_pagerank"
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_runtime_limits_positive_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "runtime": { "max_tokens": 100000, "max_nodes": 50000, "max_edges": 1000000 }
            }"#,
        ));
        assert!(report.is_valid());
    }

    // ─── Rule: rank_teleport ────────────────────────────────────────────

    #[test]
    fn test_personalized_without_teleport_fails() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "rank": "personalized_pagerank" } }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::MissingStage);
        assert_eq!(errs[0].path, "/modules/teleport");
    }

    #[test]
    fn test_personalized_with_uniform_teleport_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "rank": "personalized_pagerank",
                    "teleport": "uniform"
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    // ─── Rule: topic_graph_deps ─────────────────────────────────────────

    #[test]
    fn test_topic_graph_without_clustering_fails() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": "phrase_candidates",
                    "graph": "topic_graph"
                }
            }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::MissingStage);
        assert!(errs[0].path.contains("clustering"));
    }

    #[test]
    fn test_topic_graph_without_phrase_candidates_fails() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": "word_nodes",
                    "graph": "topic_graph",
                    "clustering": "hac"
                }
            }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::InvalidCombo);
    }

    #[test]
    fn test_topic_graph_missing_both_deps_reports_two_errors() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "graph": "topic_graph" } }"#,
        ));
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn test_candidate_graph_has_same_deps() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "graph": "candidate_graph" } }"#,
        ));
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn test_cooccurrence_window_needs_no_clustering() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "graph": "cooccurrence_window",
                    "candidates": "word_nodes"
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    // ─── Rule: graph_transform_deps ─────────────────────────────────────

    #[test]
    fn test_intra_cluster_removal_without_clustering_fails() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "graph_transforms": ["remove_intra_cluster_edges"]
                }
            }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::MissingStage);
    }

    #[test]
    fn test_alpha_boost_without_clustering_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": { "graph_transforms": ["alpha_boost"] }
            }"#,
        ));
        // alpha_boost doesn't require clustering (it's a weight modifier)
        assert!(
            !report
                .errors()
                .any(|e| e.message.contains("remove_intra_cluster"))
        );
    }

    // ─── Rule: runtime_limits ───────────────────────────────────────────

    #[test]
    fn test_zero_max_tokens_fails() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "runtime": { "max_tokens": 0 } }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::LimitExceeded);
        assert!(errs[0].path.contains("max_tokens"));
    }

    #[test]
    fn test_zero_max_nodes_and_edges_reports_two_errors() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "runtime": { "max_nodes": 0, "max_edges": 0 } }"#,
        ));
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn test_absent_limits_are_fine() {
        let report = engine().validate(&spec(r#"{ "v": 1, "runtime": {} }"#));
        assert!(report.is_valid());
    }

    // ─── Rule: unknown_fields (strict mode) ─────────────────────────────

    #[test]
    fn test_unknown_fields_non_strict_are_warnings() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "strict": false, "bogus": 42 }"#,
        ));
        assert!(report.is_valid()); // warnings don't make it invalid
        let warns: Vec<_> = report.warnings().collect();
        assert_eq!(warns.len(), 1);
        assert_eq!(warns[0].code, ErrorCode::UnknownField);
        assert!(warns[0].path.contains("bogus"));
    }

    #[test]
    fn test_unknown_fields_strict_are_errors() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "strict": true, "bogus": 42 }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::UnknownField);
    }

    #[test]
    fn test_unknown_module_field_strict() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "strict": true,
                "modules": { "bogus_module": "xyz" }
            }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].path.contains("bogus_module"));
    }

    #[test]
    fn test_unknown_runtime_field_strict() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "strict": true,
                "runtime": { "max_threads": 8 }
            }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].path.contains("max_threads"));
    }

    #[test]
    fn test_no_unknown_fields_clean() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "strict": true, "modules": { "rank": "standard_pagerank" } }"#,
        ));
        assert!(report.is_empty());
    }

    // ─── Report helpers ─────────────────────────────────────────────────

    #[test]
    fn test_report_len_and_empty() {
        let report = engine().validate(&spec(r#"{ "v": 1 }"#));
        assert_eq!(report.len(), 0);
        assert!(report.is_empty());

        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "rank": "personalized_pagerank" } }"#,
        ));
        assert_eq!(report.len(), 1);
        assert!(!report.is_empty());
    }

    #[test]
    fn test_multiple_rules_fire_independently() {
        // personalized without teleport + zero max_tokens + unknown field strict
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "strict": true,
                "bogus": true,
                "modules": { "rank": "personalized_pagerank" },
                "runtime": { "max_tokens": 0 }
            }"#,
        ));
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 3);
    }

    // ─── Engine: custom rules ───────────────────────────────────────────

    #[test]
    fn test_custom_rule() {
        struct AlwaysWarnRule;
        impl ValidationRule for AlwaysWarnRule {
            fn name(&self) -> &str {
                "always_warn"
            }
            fn validate(&self, _spec: &PipelineSpec) -> Vec<ValidationDiagnostic> {
                vec![ValidationDiagnostic::warning(PipelineSpecError::new(
                    ErrorCode::ValidationFailed,
                    "",
                    "custom warning",
                ))]
            }
        }

        let mut eng = ValidationEngine::new();
        eng.add_rule(Box::new(AlwaysWarnRule));
        let report = eng.validate(&spec(r#"{ "v": 1 }"#));
        assert!(report.is_valid()); // warnings only
        assert_eq!(report.warnings().count(), 1);
    }

    // ─── Serialization ──────────────────────────────────────────────────

    #[test]
    fn test_report_serializes_to_json() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "rank": "personalized_pagerank" } }"#,
        ));
        let json = serde_json::to_value(&report).unwrap();
        let diags = json["diagnostics"].as_array().unwrap();
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0]["severity"], "error");
        assert_eq!(diags[0]["code"], "missing_stage");
    }
}
