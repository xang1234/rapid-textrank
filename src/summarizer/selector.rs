//! MMR-based sentence selection for summarization
//!
//! Implements Maximal Marginal Relevance (MMR) for selecting
//! diverse yet relevant sentences for extractive summarization.

use super::unit_vector::{UnitVector, UnitVectorBuilder};
use crate::types::{Phrase, Sentence, Token};

/// Configuration for sentence selection
#[derive(Debug, Clone)]
pub struct SelectorConfig {
    /// Lambda parameter for MMR (0 = diversity only, 1 = relevance only)
    pub lambda: f64,
    /// Number of sentences to select
    pub num_sentences: usize,
    /// Minimum sentence length (in tokens)
    pub min_sentence_length: usize,
    /// Maximum sentence length (in tokens)
    pub max_sentence_length: usize,
}

impl Default for SelectorConfig {
    fn default() -> Self {
        Self {
            lambda: 0.7,
            num_sentences: 3,
            min_sentence_length: 5,
            max_sentence_length: 100,
        }
    }
}

/// Result of sentence selection
#[derive(Debug, Clone)]
pub struct SummaryResult {
    /// Selected sentences in document order
    pub sentences: Vec<SelectedSentence>,
    /// Total relevance score
    pub relevance_score: f64,
    /// Average pairwise diversity
    pub diversity_score: f64,
}

/// A selected sentence with its scores
#[derive(Debug, Clone)]
pub struct SelectedSentence {
    /// The sentence
    pub sentence: Sentence,
    /// Relevance score to query/document
    pub relevance: f64,
    /// MMR score when selected
    pub mmr_score: f64,
}

/// MMR-based sentence selector
#[derive(Debug)]
pub struct SentenceSelector {
    config: SelectorConfig,
}

impl Default for SentenceSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl SentenceSelector {
    /// Create a new selector with default config
    pub fn new() -> Self {
        Self {
            config: SelectorConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: SelectorConfig) -> Self {
        Self { config }
    }

    /// Set lambda (relevance vs diversity tradeoff)
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.config.lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// Set number of sentences to select
    pub fn with_num_sentences(mut self, n: usize) -> Self {
        self.config.num_sentences = n;
        self
    }

    /// Select sentences using MMR
    ///
    /// MMR formula: λ * sim(s, query) - (1-λ) * max_{s' ∈ selected} sim(s, s')
    pub fn select(
        &self,
        sentences: &[Sentence],
        tokens: &[Token],
        phrases: &[Phrase],
    ) -> SummaryResult {
        // Filter sentences by length
        let candidates: Vec<_> = sentences
            .iter()
            .filter(|s| {
                let len = s.end_token - s.start_token;
                len >= self.config.min_sentence_length && len <= self.config.max_sentence_length
            })
            .cloned()
            .collect();

        if candidates.is_empty() {
            return SummaryResult {
                sentences: Vec::new(),
                relevance_score: 0.0,
                diversity_score: 0.0,
            };
        }

        // Build unit vectors
        let builder = UnitVectorBuilder::new(phrases.to_vec());
        let query_vector = builder.build_document_vector();

        let sent_vectors: Vec<_> = candidates
            .iter()
            .map(|s| builder.build_sentence_vector(s, tokens))
            .collect();

        // Calculate relevance scores
        let relevance_scores: Vec<f64> = sent_vectors
            .iter()
            .map(|v| v.cosine_similarity(&query_vector))
            .collect();

        // MMR selection
        let mut selected: Vec<usize> = Vec::new();
        let mut selected_vectors: Vec<&UnitVector> = Vec::new();

        while selected.len() < self.config.num_sentences && selected.len() < candidates.len() {
            let mut best_idx = None;
            let mut best_mmr = f64::NEG_INFINITY;

            for (i, _) in candidates.iter().enumerate() {
                if selected.contains(&i) {
                    continue;
                }

                // Relevance component
                let relevance = relevance_scores[i];

                // Diversity component: max similarity to already selected
                let max_sim = if selected_vectors.is_empty() {
                    0.0
                } else {
                    selected_vectors
                        .iter()
                        .map(|sv| sent_vectors[i].cosine_similarity(sv))
                        .fold(f64::NEG_INFINITY, f64::max)
                };

                // MMR score
                let mmr = self.config.lambda * relevance - (1.0 - self.config.lambda) * max_sim;

                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = Some(i);
                }
            }

            if let Some(idx) = best_idx {
                selected.push(idx);
                selected_vectors.push(&sent_vectors[idx]);
            } else {
                break;
            }
        }

        // Build result
        let mut selected_sentences: Vec<SelectedSentence> = selected
            .iter()
            .map(|&i| SelectedSentence {
                sentence: candidates[i].clone(),
                relevance: relevance_scores[i],
                mmr_score: self.config.lambda * relevance_scores[i],
            })
            .collect();

        // Sort by document order
        selected_sentences.sort_by_key(|s| s.sentence.index);

        // Calculate summary scores
        let relevance_score: f64 = selected_sentences.iter().map(|s| s.relevance).sum();

        // Calculate diversity as 1 - average pairwise similarity
        let diversity_score = if selected_sentences.len() > 1 {
            let mut total_sim = 0.0;
            let mut count = 0;
            for i in 0..selected.len() {
                for j in (i + 1)..selected.len() {
                    total_sim +=
                        sent_vectors[selected[i]].cosine_similarity(&sent_vectors[selected[j]]);
                    count += 1;
                }
            }
            1.0 - (if count > 0 {
                total_sim / count as f64
            } else {
                0.0
            })
        } else {
            1.0
        };

        SummaryResult {
            sentences: selected_sentences,
            relevance_score,
            diversity_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PosTag;

    fn make_test_data() -> (Vec<Sentence>, Vec<Token>, Vec<Phrase>) {
        let sentences = vec![
            Sentence {
                text: "Machine learning is a subset of AI.".to_string(),
                start: 0,
                end: 35,
                index: 0,
                start_token: 0,
                end_token: 7,
                score: 0.0,
            },
            Sentence {
                text: "Deep learning uses neural networks.".to_string(),
                start: 36,
                end: 70,
                index: 1,
                start_token: 7,
                end_token: 12,
                score: 0.0,
            },
            Sentence {
                text: "AI is transforming industries.".to_string(),
                start: 71,
                end: 100,
                index: 2,
                start_token: 12,
                end_token: 16,
                score: 0.0,
            },
        ];

        let tokens = vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("is", "be", PosTag::Verb, 17, 19, 0, 2),
            Token::new("a", "a", PosTag::Determiner, 20, 21, 0, 3),
            Token::new("subset", "subset", PosTag::Noun, 22, 28, 0, 4),
            Token::new("of", "of", PosTag::Preposition, 29, 31, 0, 5),
            Token::new("AI", "ai", PosTag::Noun, 32, 34, 0, 6),
            // Sentence 2
            Token::new("Deep", "deep", PosTag::Adjective, 36, 40, 1, 7),
            Token::new("learning", "learning", PosTag::Noun, 41, 49, 1, 8),
            Token::new("uses", "use", PosTag::Verb, 50, 54, 1, 9),
            Token::new("neural", "neural", PosTag::Adjective, 55, 61, 1, 10),
            Token::new("networks", "network", PosTag::Noun, 62, 70, 1, 11),
            // Sentence 3
            Token::new("AI", "ai", PosTag::Noun, 71, 73, 2, 12),
            Token::new("is", "be", PosTag::Verb, 74, 76, 2, 13),
            Token::new("transforming", "transform", PosTag::Verb, 77, 89, 2, 14),
            Token::new("industries", "industry", PosTag::Noun, 90, 100, 2, 15),
        ];

        let phrases = vec![
            Phrase::new("machine learning", "machine learning", 0.5, 1),
            Phrase::new("AI", "ai", 0.4, 2),
            Phrase::new("neural networks", "neural network", 0.3, 1),
        ];

        (sentences, tokens, phrases)
    }

    #[test]
    fn test_mmr_selection() {
        let (sentences, tokens, phrases) = make_test_data();

        let selector = SentenceSelector::new().with_num_sentences(2);
        let result = selector.select(&sentences, &tokens, &phrases);

        assert_eq!(result.sentences.len(), 2);
    }

    #[test]
    fn test_document_order() {
        let (sentences, tokens, phrases) = make_test_data();

        let selector = SentenceSelector::new().with_num_sentences(3);
        let result = selector.select(&sentences, &tokens, &phrases);

        // Sentences should be in document order
        for i in 1..result.sentences.len() {
            assert!(result.sentences[i].sentence.index > result.sentences[i - 1].sentence.index);
        }
    }

    #[test]
    fn test_lambda_diversity() {
        let (sentences, tokens, phrases) = make_test_data();

        // Low lambda = more diversity
        let selector_diverse = SentenceSelector::new()
            .with_lambda(0.3)
            .with_num_sentences(2);
        let result_diverse = selector_diverse.select(&sentences, &tokens, &phrases);

        // High lambda = more relevance
        let selector_relevant = SentenceSelector::new()
            .with_lambda(0.9)
            .with_num_sentences(2);
        let result_relevant = selector_relevant.select(&sentences, &tokens, &phrases);

        // Both should produce results
        assert!(!result_diverse.sentences.is_empty());
        assert!(!result_relevant.sentences.is_empty());
    }

    #[test]
    fn test_empty_input() {
        let selector = SentenceSelector::new();
        let result = selector.select(&[], &[], &[]);

        assert!(result.sentences.is_empty());
    }

    #[test]
    fn test_min_sentence_length_filter() {
        let sentences = vec![Sentence {
            text: "Short.".to_string(),
            start: 0,
            end: 6,
            index: 0,
            start_token: 0,
            end_token: 2, // Only 2 tokens
            score: 0.0,
        }];

        let tokens = vec![
            Token::new("Short", "short", PosTag::Adjective, 0, 5, 0, 0),
            Token::new(".", ".", PosTag::Punctuation, 5, 6, 0, 1),
        ];

        let phrases = vec![Phrase::new("short", "short", 1.0, 1)];

        let selector = SentenceSelector::new();
        let result = selector.select(&sentences, &tokens, &phrases);

        // Sentence is too short (< 5 tokens)
        assert!(result.sentences.is_empty());
    }
}
