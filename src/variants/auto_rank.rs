//! AutoRank ensemble extractor.
//!
//! AutoRank runs a full pool of eligible keyword extractors, fuses their
//! outputs with reciprocal-rank fusion, and returns a single ranked keyword
//! list plus consensus metadata.

use crate::phrase::extraction::{extract_keyphrases_with_info, ExtractionResult};
use crate::types::{Phrase, TextRankConfig, Token};
use crate::variants::biased_textrank::BiasedTextRank;
use crate::variants::multipartite_rank::MultipartiteRank;
use crate::variants::position_rank::PositionRank;
use crate::variants::single_rank::SingleRank;
use crate::variants::topic_rank::TopicRank;
use crate::variants::topical_pagerank::TopicalPageRank;
use crate::variants::Variant;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

const RRF_K: f64 = 60.0;
const BIASED_WEIGHT: f64 = 1.15;
const TOPICAL_WEIGHT: f64 = 1.15;
const SEMANTIC_BOOST: f64 = 0.20;

/// Per-variant convergence summary attached to AutoRank results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoRankVariantRun {
    pub variant: String,
    pub converged: bool,
    pub iterations: usize,
}

/// Per-phrase support metadata aligned with `ExtractionResult.phrases`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoRankPhraseSupport {
    pub confidence: f64,
    pub support_count: usize,
    pub supporting_variants: Vec<String>,
}

/// Consensus payload produced by AutoRank.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoRankConsensus {
    pub selected_variants: Vec<String>,
    pub selection_reason: String,
    pub variant_runs: Vec<AutoRankVariantRun>,
    pub phrase_support: Vec<AutoRankPhraseSupport>,
}

#[derive(Debug)]
pub struct AutoRank {
    config: TextRankConfig,
    focus_terms: Vec<String>,
    bias_weight: f64,
    semantic_weights: HashMap<String, f64>,
    semantic_min_weight: f64,
    include_topic_rank: bool,
}

#[derive(Debug, Clone)]
struct VariantExecution {
    variant: Variant,
    weight: f64,
    result: ExtractionResult,
}

#[derive(Debug, Clone)]
struct SupportRecord {
    variant: Variant,
    weight: f64,
    phrase: Phrase,
}

#[derive(Debug, Clone)]
struct FusedCandidate {
    lemma: String,
    supports: Vec<SupportRecord>,
    semantic_prior: f64,
}

impl Default for AutoRank {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoRank {
    pub fn new() -> Self {
        Self {
            config: TextRankConfig::default(),
            focus_terms: Vec::new(),
            bias_weight: 5.0,
            semantic_weights: HashMap::new(),
            semantic_min_weight: 0.0,
            include_topic_rank: true,
        }
    }

    pub fn with_config(config: TextRankConfig) -> Self {
        Self {
            config,
            focus_terms: Vec::new(),
            bias_weight: 5.0,
            semantic_weights: HashMap::new(),
            semantic_min_weight: 0.0,
            include_topic_rank: true,
        }
    }

    pub fn with_focus(mut self, terms: &[&str]) -> Self {
        self.focus_terms = terms.iter().map(|s| s.to_lowercase()).collect();
        self
    }

    pub fn with_bias_weight(mut self, weight: f64) -> Self {
        self.bias_weight = weight;
        self
    }

    pub fn with_semantic_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.semantic_weights = weights
            .into_iter()
            .map(|(lemma, weight)| (lemma.to_lowercase(), weight))
            .collect();
        self
    }

    pub fn with_semantic_min_weight(mut self, min_weight: f64) -> Self {
        self.semantic_min_weight = min_weight;
        self
    }

    pub(crate) fn with_topic_rank_enabled(mut self, enabled: bool) -> Self {
        self.include_topic_rank = enabled;
        self
    }

    pub(crate) fn selected_variants(&self) -> Vec<Variant> {
        let mut variants = vec![
            Variant::TextRank,
            Variant::PositionRank,
            Variant::SingleRank,
            Variant::MultipartiteRank,
        ];
        if self.include_topic_rank {
            variants.push(Variant::TopicRank);
        }
        if !self.focus_terms.is_empty() {
            variants.push(Variant::BiasedTextRank);
        }
        if !self.semantic_weights.is_empty() {
            variants.push(Variant::TopicalPageRank);
        }
        variants
    }

    pub fn extract(&self, tokens: &[Token]) -> Vec<Phrase> {
        self.extract_with_info(tokens).phrases
    }

    pub fn extract_with_info(&self, tokens: &[Token]) -> ExtractionResult {
        if !self.has_eligible_tokens(tokens) {
            return ExtractionResult {
                phrases: Vec::new(),
                converged: true,
                iterations: 0,
                debug: None,
                consensus: Some(Self::empty_consensus(
                    "AutoRank skipped execution because no eligible candidates were found."
                        .to_string(),
                )),
            };
        }

        let fusion_top_n = self.config.top_n.saturating_mul(3).max(20);
        let member_config = self.member_config(fusion_top_n);
        let selected_variants = self.selected_variants();
        let total_executed_weight: f64 = selected_variants
            .iter()
            .map(|v| self.variant_weight(*v))
            .sum();
        let selection_reason = self.selection_reason();

        let executions: Vec<VariantExecution> = selected_variants
            .iter()
            .copied()
            .map(|variant| VariantExecution {
                variant,
                weight: self.variant_weight(variant),
                result: self.run_variant(variant, &member_config, tokens),
            })
            .collect();

        let fused = self.fuse_candidates(&executions);
        if fused.is_empty() {
            return ExtractionResult {
                phrases: Vec::new(),
                converged: true,
                iterations: 0,
                debug: None,
                consensus: Some(Self::empty_consensus(
                    "AutoRank skipped execution because no eligible candidates were found."
                        .to_string(),
                )),
            };
        }

        let semantic_bounds = self.semantic_bounds(&fused);
        let mut ranked: Vec<_> = fused
            .into_iter()
            .map(|candidate| {
                self.score_candidate(candidate, total_executed_weight, semantic_bounds)
            })
            .collect();

        ranked.sort_by(|a, b| compare_ranked_candidates(a, b));

        let limit = if self.config.top_n == 0 {
            ranked.len()
        } else {
            ranked.len().min(self.config.top_n)
        };
        ranked.truncate(limit);

        let mut phrases = Vec::with_capacity(ranked.len());
        let mut phrase_support = Vec::with_capacity(ranked.len());
        for (idx, candidate) in ranked.into_iter().enumerate() {
            phrases.push(Phrase {
                text: candidate.display_text,
                lemma: candidate.lemma,
                score: candidate.fused_score,
                count: candidate.count,
                offsets: candidate.offsets,
                rank: idx + 1,
            });
            phrase_support.push(AutoRankPhraseSupport {
                confidence: candidate.confidence,
                support_count: candidate.support_count,
                supporting_variants: candidate.supporting_variants,
            });
        }

        let variant_runs = executions
            .iter()
            .map(|run| AutoRankVariantRun {
                variant: run.variant.canonical_name().to_string(),
                converged: run.result.converged,
                iterations: run.result.iterations,
            })
            .collect::<Vec<_>>();
        let converged = executions.iter().all(|run| run.result.converged);
        let iterations = executions
            .iter()
            .map(|run| run.result.iterations)
            .max()
            .unwrap_or(0);

        ExtractionResult {
            phrases,
            converged,
            iterations,
            debug: None,
            consensus: Some(AutoRankConsensus {
                selected_variants: selected_variants
                    .iter()
                    .map(|variant| variant.canonical_name().to_string())
                    .collect(),
                selection_reason,
                variant_runs,
                phrase_support,
            }),
        }
    }

    fn member_config(&self, fusion_top_n: usize) -> TextRankConfig {
        let mut config = self.config.clone();
        config.top_n = fusion_top_n;
        config.debug_level = crate::pipeline::artifacts::DebugLevel::None;
        config
    }

    fn has_eligible_tokens(&self, tokens: &[Token]) -> bool {
        if tokens.is_empty() {
            return false;
        }

        tokens.iter().any(|token| {
            !token.is_stopword && self.config.include_pos.iter().any(|tag| tag == &token.pos)
        })
    }

    fn variant_weight(&self, variant: Variant) -> f64 {
        match variant {
            Variant::BiasedTextRank => BIASED_WEIGHT,
            Variant::TopicalPageRank => TOPICAL_WEIGHT,
            _ => 1.0,
        }
    }

    fn selection_reason(&self) -> String {
        let mut extras = Vec::new();
        if self.include_topic_rank {
            extras.push("pre-tokenized TopicRank");
        }
        if !self.focus_terms.is_empty() {
            extras.push("focus-driven BiasedTextRank");
        }
        if !self.semantic_weights.is_empty() {
            extras.push("semantic-weight TopicalPageRank");
        }

        if extras.is_empty() {
            "AutoRank ran the default keyword ensemble.".to_string()
        } else {
            format!(
                "AutoRank ran the default keyword ensemble plus {}.",
                extras.join(", ")
            )
        }
    }

    fn run_variant(
        &self,
        variant: Variant,
        config: &TextRankConfig,
        tokens: &[Token],
    ) -> ExtractionResult {
        match variant {
            Variant::TextRank => extract_keyphrases_with_info(tokens, config),
            Variant::PositionRank => {
                PositionRank::with_config(config.clone()).extract_with_info(tokens)
            }
            Variant::BiasedTextRank => BiasedTextRank::with_config(config.clone())
                .with_focus(
                    &self
                        .focus_terms
                        .iter()
                        .map(|term| term.as_str())
                        .collect::<Vec<_>>(),
                )
                .with_bias_weight(self.bias_weight)
                .extract_with_info(tokens),
            Variant::TopicRank => TopicRank::with_config(config.clone()).extract_with_info(tokens),
            Variant::SingleRank => {
                SingleRank::with_config(config.clone()).extract_with_info(tokens)
            }
            Variant::TopicalPageRank => TopicalPageRank::with_config(config.clone())
                .with_topic_weights(self.semantic_weights.clone())
                .with_min_weight(self.semantic_min_weight)
                .extract_with_info(tokens),
            Variant::MultipartiteRank => {
                MultipartiteRank::with_config(config.clone()).extract_with_info(tokens)
            }
            Variant::AutoRank => unreachable!("AutoRank does not execute itself"),
            #[cfg(feature = "sentence-rank")]
            Variant::SentenceRank => unreachable!("SentenceRank is not part of AutoRank"),
        }
    }

    fn fuse_candidates(&self, executions: &[VariantExecution]) -> Vec<FusedCandidate> {
        let mut fused: FxHashMap<String, FusedCandidate> = FxHashMap::default();

        for execution in executions {
            let mut best_by_lemma: FxHashMap<String, Phrase> = FxHashMap::default();
            for phrase in &execution.result.phrases {
                match best_by_lemma.get_mut(&phrase.lemma) {
                    Some(existing) => {
                        if better_support_phrase(phrase, existing) {
                            *existing = phrase.clone();
                        }
                    }
                    None => {
                        best_by_lemma.insert(phrase.lemma.clone(), phrase.clone());
                    }
                }
            }

            for phrase in best_by_lemma.into_values() {
                let lemma = phrase.lemma.clone();
                fused
                    .entry(lemma.clone())
                    .and_modify(|candidate| {
                        candidate.supports.push(SupportRecord {
                            variant: execution.variant,
                            weight: execution.weight,
                            phrase: phrase.clone(),
                        });
                    })
                    .or_insert_with(|| FusedCandidate {
                        semantic_prior: self.semantic_prior_for(&lemma),
                        lemma,
                        supports: vec![SupportRecord {
                            variant: execution.variant,
                            weight: execution.weight,
                            phrase,
                        }],
                    });
            }
        }

        fused.into_values().collect()
    }

    fn semantic_prior_for(&self, lemma: &str) -> f64 {
        if self.semantic_weights.is_empty() {
            return 0.0;
        }

        let tokens: Vec<&str> = lemma.split_whitespace().collect();
        if tokens.is_empty() {
            return self.semantic_min_weight;
        }

        let total: f64 = tokens
            .iter()
            .map(|token| {
                self.semantic_weights
                    .get(*token)
                    .copied()
                    .unwrap_or(self.semantic_min_weight)
            })
            .sum();
        total / tokens.len() as f64
    }

    fn semantic_bounds(&self, candidates: &[FusedCandidate]) -> (f64, f64) {
        if self.semantic_weights.is_empty() || candidates.is_empty() {
            return (0.0, 0.0);
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for candidate in candidates {
            min = min.min(candidate.semantic_prior);
            max = max.max(candidate.semantic_prior);
        }
        (min, max)
    }

    fn score_candidate(
        &self,
        candidate: FusedCandidate,
        total_executed_weight: f64,
        semantic_bounds: (f64, f64),
    ) -> RankedCandidate {
        let representative = candidate
            .supports
            .iter()
            .min_by(|a, b| compare_support_for_display(a, b))
            .expect("fused candidate must have support")
            .phrase
            .clone();
        let count_source = candidate
            .supports
            .iter()
            .min_by(|a, b| compare_support_for_count(a, b))
            .expect("fused candidate must have support")
            .phrase
            .clone();

        let base_score = candidate
            .supports
            .iter()
            .map(|support| support.weight / (RRF_K + support.phrase.rank as f64))
            .sum::<f64>();

        let normalized_prior = normalize_prior(candidate.semantic_prior, semantic_bounds);
        let fused_score = if self.semantic_weights.is_empty() {
            base_score
        } else {
            base_score * (1.0 + SEMANTIC_BOOST * normalized_prior)
        };

        let support_weight: f64 = candidate
            .supports
            .iter()
            .map(|support| support.weight)
            .sum();
        let supporting_variants = candidate
            .supports
            .iter()
            .map(|support| support.variant.canonical_name().to_string())
            .collect::<Vec<_>>();

        RankedCandidate {
            lemma: candidate.lemma,
            display_text: representative.text,
            fused_score,
            count: count_source.count,
            offsets: count_source.offsets,
            support_count: supporting_variants.len(),
            confidence: if total_executed_weight > 0.0 {
                support_weight / total_executed_weight
            } else {
                0.0
            },
            supporting_variants,
        }
    }

    fn empty_consensus(selection_reason: String) -> AutoRankConsensus {
        AutoRankConsensus {
            selected_variants: Vec::new(),
            selection_reason,
            variant_runs: Vec::new(),
            phrase_support: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct RankedCandidate {
    lemma: String,
    display_text: String,
    fused_score: f64,
    count: usize,
    offsets: Vec<(usize, usize)>,
    support_count: usize,
    confidence: f64,
    supporting_variants: Vec<String>,
}

fn compare_ranked_candidates(a: &RankedCandidate, b: &RankedCandidate) -> Ordering {
    b.fused_score
        .partial_cmp(&a.fused_score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| b.support_count.cmp(&a.support_count))
        .then_with(|| b.count.cmp(&a.count))
        .then_with(|| a.lemma.cmp(&b.lemma))
}

fn better_support_phrase(candidate: &Phrase, existing: &Phrase) -> bool {
    compare_support_for_phrase(candidate, existing) == Ordering::Less
}

fn compare_support_for_display(a: &SupportRecord, b: &SupportRecord) -> Ordering {
    compare_support_for_phrase(&a.phrase, &b.phrase)
}

fn compare_support_for_count(a: &SupportRecord, b: &SupportRecord) -> Ordering {
    b.phrase
        .count
        .cmp(&a.phrase.count)
        .then_with(|| compare_support_for_phrase(&a.phrase, &b.phrase))
}

fn compare_support_for_phrase(a: &Phrase, b: &Phrase) -> Ordering {
    a.rank
        .cmp(&b.rank)
        .then_with(|| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal))
        .then_with(|| a.text.len().cmp(&b.text.len()))
        .then_with(|| a.text.cmp(&b.text))
}

fn normalize_prior(prior: f64, bounds: (f64, f64)) -> f64 {
    let (min, max) = bounds;
    if max > min {
        (prior - min) / (max - min)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DeterminismMode, PosTag};

    fn make_tokens() -> Vec<Token> {
        vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("improves", "improve", PosTag::Verb, 17, 25, 0, 2),
            Token::new("search", "search", PosTag::Noun, 26, 32, 0, 3),
            Token::new("ranking", "ranking", PosTag::Noun, 33, 40, 0, 4),
            Token::new("Topic", "topic", PosTag::Noun, 42, 47, 1, 5),
            Token::new("models", "model", PosTag::Noun, 48, 54, 1, 6),
            Token::new("improve", "improve", PosTag::Verb, 55, 62, 1, 7),
            Token::new("keyword", "keyword", PosTag::Noun, 63, 70, 1, 8),
            Token::new("extraction", "extraction", PosTag::Noun, 71, 81, 1, 9),
        ]
    }

    #[test]
    fn test_selected_variants_raw_path() {
        let variants = AutoRank::new()
            .with_topic_rank_enabled(false)
            .selected_variants();
        assert_eq!(
            variants,
            vec![
                Variant::TextRank,
                Variant::PositionRank,
                Variant::SingleRank,
                Variant::MultipartiteRank
            ]
        );
    }

    #[test]
    fn test_selected_variants_with_focus_and_semantics() {
        let mut semantic_weights = HashMap::new();
        semantic_weights.insert("machine".to_string(), 0.9);

        let variants = AutoRank::new()
            .with_focus(&["machine"])
            .with_semantic_weights(semantic_weights)
            .selected_variants();

        assert_eq!(
            variants,
            vec![
                Variant::TextRank,
                Variant::PositionRank,
                Variant::SingleRank,
                Variant::MultipartiteRank,
                Variant::TopicRank,
                Variant::BiasedTextRank,
                Variant::TopicalPageRank
            ]
        );
    }

    #[test]
    fn test_extract_with_consensus_metadata() {
        let mut semantic_weights = HashMap::new();
        semantic_weights.insert("machine".to_string(), 1.0);
        semantic_weights.insert("learning".to_string(), 0.8);

        let result = AutoRank::with_config(TextRankConfig::default().with_top_n(5))
            .with_semantic_weights(semantic_weights)
            .extract_with_info(&make_tokens());

        assert!(!result.phrases.is_empty());
        let consensus = result.consensus.expect("consensus should be present");
        assert_eq!(consensus.phrase_support.len(), result.phrases.len());
        assert!(consensus
            .selected_variants
            .iter()
            .any(|variant| variant == "topical_pagerank"));
    }

    #[test]
    fn test_empty_input_returns_empty_consensus() {
        let result = AutoRank::new().extract_with_info(&[]);
        assert!(result.phrases.is_empty());
        let consensus = result.consensus.expect("consensus should be present");
        assert!(consensus.selected_variants.is_empty());
        assert!(consensus.variant_runs.is_empty());
        assert!(consensus.phrase_support.is_empty());
    }

    #[test]
    fn test_deterministic_output_is_stable() {
        let tokens = make_tokens();
        let config = TextRankConfig {
            determinism: DeterminismMode::Deterministic,
            ..TextRankConfig::default()
        };
        let baseline = AutoRank::with_config(config.clone()).extract_with_info(&tokens);
        for _ in 0..2 {
            let result = AutoRank::with_config(config.clone()).extract_with_info(&tokens);
            assert_eq!(baseline, result);
        }
    }
}
