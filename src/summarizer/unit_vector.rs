//! Unit vector representation for sentences
//!
//! Builds sparse vector representations of sentences based on
//! phrase occurrences for similarity calculations.

use crate::types::{Phrase, Sentence, Token};
use rustc_hash::FxHashMap;

/// A sparse unit vector representation of a sentence
#[derive(Debug, Clone, Default)]
pub struct UnitVector {
    /// Non-zero dimensions: phrase lemma -> weight
    pub dimensions: FxHashMap<String, f64>,
    /// L2 norm of the vector
    pub norm: f64,
}

impl UnitVector {
    /// Create a new empty unit vector
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from a map of dimensions
    pub fn from_dimensions(mut dimensions: FxHashMap<String, f64>) -> Self {
        let norm = Self::compute_norm(&dimensions);
        if norm > 0.0 {
            for value in dimensions.values_mut() {
                *value /= norm;
            }
        }
        Self { dimensions, norm }
    }

    /// Compute L2 norm
    fn compute_norm(dimensions: &FxHashMap<String, f64>) -> f64 {
        dimensions.values().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Compute cosine similarity with another vector
    pub fn cosine_similarity(&self, other: &UnitVector) -> f64 {
        // Since vectors are normalized, cosine similarity is just the dot product
        let mut dot = 0.0;
        for (key, value) in &self.dimensions {
            if let Some(other_value) = other.dimensions.get(key) {
                dot += value * other_value;
            }
        }
        dot
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.dimensions.is_empty()
    }
}

/// Build unit vectors for sentences based on phrase occurrences
pub struct UnitVectorBuilder {
    /// Phrases to use for vector dimensions
    phrases: Vec<Phrase>,
    /// Whether to weight by phrase score
    weight_by_score: bool,
}

impl UnitVectorBuilder {
    /// Create a new builder from extracted phrases
    pub fn new(phrases: Vec<Phrase>) -> Self {
        Self {
            phrases,
            weight_by_score: true,
        }
    }

    /// Set whether to weight dimensions by phrase score
    pub fn with_score_weighting(mut self, weight: bool) -> Self {
        self.weight_by_score = weight;
        self
    }

    /// Build a unit vector for a sentence
    pub fn build_sentence_vector(&self, sentence: &Sentence, tokens: &[Token]) -> UnitVector {
        let mut dimensions: FxHashMap<String, f64> = FxHashMap::default();

        // Get tokens in this sentence
        let sent_tokens: Vec<_> = tokens
            .iter()
            .filter(|t| t.sentence_idx == sentence.index)
            .collect();

        // Check which phrases appear in this sentence
        for phrase in &self.phrases {
            // Check if phrase lemma appears in sentence tokens
            let phrase_words: Vec<_> = phrase.lemma.split_whitespace().collect();

            // Simple containment check for each phrase
            let mut found = false;
            for i in 0..=sent_tokens.len().saturating_sub(phrase_words.len()) {
                let mut matches = true;
                for (j, phrase_word) in phrase_words.iter().enumerate() {
                    if i + j >= sent_tokens.len()
                        || sent_tokens[i + j].lemma.to_lowercase() != phrase_word.to_lowercase()
                    {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    found = true;
                    break;
                }
            }

            if found {
                let weight = if self.weight_by_score {
                    phrase.score
                } else {
                    1.0
                };
                *dimensions.entry(phrase.lemma.clone()).or_insert(0.0) += weight;
            }
        }

        UnitVector::from_dimensions(dimensions)
    }

    /// Build a document-level unit vector from all phrases
    pub fn build_document_vector(&self) -> UnitVector {
        let dimensions: FxHashMap<String, f64> = self
            .phrases
            .iter()
            .map(|p| {
                let weight = if self.weight_by_score { p.score } else { 1.0 };
                (p.lemma.clone(), weight)
            })
            .collect();

        UnitVector::from_dimensions(dimensions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let mut dims = FxHashMap::default();
        dims.insert("a".to_string(), 1.0);
        dims.insert("b".to_string(), 2.0);

        let v1 = UnitVector::from_dimensions(dims.clone());
        let v2 = UnitVector::from_dimensions(dims);

        let sim = v1.cosine_similarity(&v2);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut dims1 = FxHashMap::default();
        dims1.insert("a".to_string(), 1.0);

        let mut dims2 = FxHashMap::default();
        dims2.insert("b".to_string(), 1.0);

        let v1 = UnitVector::from_dimensions(dims1);
        let v2 = UnitVector::from_dimensions(dims2);

        let sim = v1.cosine_similarity(&v2);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_empty_vector() {
        let v = UnitVector::new();
        assert!(v.is_empty());
        assert!((v.norm - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_unit_normalization() {
        let mut dims = FxHashMap::default();
        dims.insert("a".to_string(), 3.0);
        dims.insert("b".to_string(), 4.0);

        let v = UnitVector::from_dimensions(dims);

        // Norm should be 1 after normalization
        let actual_norm: f64 = v.dimensions.values().map(|x| x * x).sum::<f64>().sqrt();
        assert!((actual_norm - 1.0).abs() < 1e-6);
    }
}
