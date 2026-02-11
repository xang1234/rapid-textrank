//! Stage trait definitions for the pipeline.
//!
//! Each trait represents one processing stage boundary. Implementations are
//! statically dispatched for performance; trait objects are available behind a
//! feature gate for dynamic composition.

use crate::pipeline::artifacts::TokenStream;
use crate::types::TextRankConfig;

// ============================================================================
// Preprocessor — optional token normalization (stage 0)
// ============================================================================

/// Optional preprocessing / normalization stage.
///
/// Centralizes normalization differences between the built-in tokenizer
/// (Unicode-aware) and spaCy / JSON-provided tokens, without duplicating
/// rules across downstream stages (CandidateSelector, GraphBuilder,
/// PhraseBuilder).
///
/// Most variants don't need custom preprocessing — the provided
/// [`NoopPreprocessor`] is the default.
///
/// # Contract
///
/// - **Input**: a mutable [`TokenStream`] (modify in-place to avoid
///   allocating a second stream).
/// - **Output**: none — the stream is mutated.
/// - **Idempotent**: calling `preprocess` twice should produce the same
///   result as calling it once.
///
/// # Examples
///
/// A concrete preprocessor might:
/// - Re-lemmatize tokens using a different strategy
/// - Override POS tags for known domain terms
/// - Mark additional stopwords based on a custom list
/// - Normalize Unicode forms (NFC/NFKC)
pub trait Preprocessor {
    /// Preprocess the token stream in place.
    fn preprocess(&self, tokens: &mut TokenStream, cfg: &TextRankConfig);
}

/// No-op preprocessor — the default for most pipeline configurations.
///
/// Passes the token stream through unchanged, with zero overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopPreprocessor;

impl Preprocessor for NoopPreprocessor {
    #[inline]
    fn preprocess(&self, _tokens: &mut TokenStream, _cfg: &TextRankConfig) {
        // Intentionally empty.
    }
}

// Future E3 stage traits will be added below:
//   - CandidateSelector     (textranker-4a0.2)
//   - GraphBuilder          (textranker-4a0.3)
//   - GraphTransform        (textranker-4a0.4)
//   - TeleportBuilder       (textranker-4a0.5)
//   - Ranker                (textranker-4a0.6)
//   - PhraseBuilder         (textranker-4a0.7)
//   - ResultFormatter       (textranker-4a0.8)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PosTag, Token};

    fn sample_tokens() -> Vec<Token> {
        vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("is", "be", PosTag::Verb, 17, 19, 0, 2),
        ]
    }

    #[test]
    fn test_noop_preprocessor_preserves_tokens() {
        let tokens = sample_tokens();
        let mut stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let snapshot_len = stream.len();
        let snapshot_text0 = stream.text(&stream.tokens()[0]).to_string();
        let snapshot_pos0 = stream.tokens()[0].pos;

        NoopPreprocessor.preprocess(&mut stream, &cfg);

        assert_eq!(stream.len(), snapshot_len);
        assert_eq!(stream.text(&stream.tokens()[0]), snapshot_text0);
        assert_eq!(stream.tokens()[0].pos, snapshot_pos0);
    }

    #[test]
    fn test_noop_preprocessor_is_idempotent() {
        let tokens = sample_tokens();
        let mut stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        NoopPreprocessor.preprocess(&mut stream, &cfg);
        let after_first: Vec<_> = stream.tokens().to_vec();

        NoopPreprocessor.preprocess(&mut stream, &cfg);
        let after_second: Vec<_> = stream.tokens().to_vec();

        assert_eq!(after_first, after_second);
    }

    #[test]
    fn test_noop_preprocessor_on_empty_stream() {
        let mut stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();

        NoopPreprocessor.preprocess(&mut stream, &cfg);

        assert!(stream.is_empty());
        assert_eq!(stream.num_sentences(), 0);
    }

    /// Test that a custom Preprocessor can mutate tokens via tokens_mut().
    #[test]
    fn test_custom_preprocessor_marks_stopwords() {
        struct MarkVerbsAsStopwords;

        impl Preprocessor for MarkVerbsAsStopwords {
            fn preprocess(&self, tokens: &mut TokenStream, _cfg: &TextRankConfig) {
                for entry in tokens.tokens_mut() {
                    if entry.pos == PosTag::Verb {
                        entry.is_stopword = true;
                    }
                }
            }
        }

        let tokens = sample_tokens();
        let mut stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        // Before: "is" (verb at index 2) is not a stopword.
        assert!(!stream.tokens()[2].is_stopword);

        MarkVerbsAsStopwords.preprocess(&mut stream, &cfg);

        // After: verbs are marked as stopwords.
        assert!(!stream.tokens()[0].is_stopword); // Noun — unchanged
        assert!(!stream.tokens()[1].is_stopword); // Noun — unchanged
        assert!(stream.tokens()[2].is_stopword);  // Verb — now stopword
    }

    /// Test that a custom Preprocessor can re-lemmatize via pool_mut().
    #[test]
    fn test_custom_preprocessor_relemmatize() {
        struct Lowercaser;

        impl Preprocessor for Lowercaser {
            fn preprocess(&self, tokens: &mut TokenStream, _cfg: &TextRankConfig) {
                // Collect texts first to avoid borrow conflict.
                let new_lemmas: Vec<String> = tokens
                    .tokens()
                    .iter()
                    .map(|e| tokens.pool().get(e.lemma_id).unwrap_or("").to_lowercase())
                    .collect();

                for (i, new_lemma) in new_lemmas.into_iter().enumerate() {
                    let new_id = tokens.pool_mut().intern(&new_lemma);
                    tokens.tokens_mut()[i].lemma_id = new_id;
                }
            }
        }

        let tokens = vec![
            Token::new("Machine", "Machine", PosTag::Noun, 0, 7, 0, 0),
        ];
        let mut stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        // Before: lemma is "Machine" (uppercase M).
        assert_eq!(stream.lemma(&stream.tokens()[0]), "Machine");

        Lowercaser.preprocess(&mut stream, &cfg);

        // After: lemma is lowercased.
        assert_eq!(stream.lemma(&stream.tokens()[0]), "machine");
    }

    /// Test trait object usage (dyn Preprocessor).
    #[test]
    fn test_preprocessor_as_trait_object() {
        let preprocessor: Box<dyn Preprocessor> = Box::new(NoopPreprocessor);

        let tokens = sample_tokens();
        let mut stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        preprocessor.preprocess(&mut stream, &cfg);

        assert_eq!(stream.len(), 3);
    }

    /// Test Default derive for NoopPreprocessor.
    #[test]
    fn test_noop_preprocessor_default() {
        let _p = NoopPreprocessor::default();
        // Just verifying it compiles and is usable.
    }
}
