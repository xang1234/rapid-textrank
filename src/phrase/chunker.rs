//! Noun chunk detection
//!
//! Identifies noun phrases using pattern matching on POS tags.
//! Pattern: (DET)? (ADJ)* (NOUN|PROPN)+

use crate::types::{ChunkSpan, PosTag, Token};

/// Configuration for noun chunk detection
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Minimum number of tokens in a chunk
    pub min_length: usize,
    /// Maximum number of tokens in a chunk
    pub max_length: usize,
    /// Whether to include determiners in chunks
    pub include_determiners: bool,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            min_length: 1,
            max_length: 5,
            include_determiners: false,
        }
    }
}

/// Noun chunk detector
#[derive(Debug, Clone)]
pub struct NounChunker {
    config: ChunkerConfig,
}

impl Default for NounChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl NounChunker {
    /// Create a new chunker with default config
    pub fn new() -> Self {
        Self {
            config: ChunkerConfig::default(),
        }
    }

    /// Create a chunker with custom config
    pub fn with_config(config: ChunkerConfig) -> Self {
        Self { config }
    }

    /// Set minimum chunk length
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.config.min_length = min_length;
        self
    }

    /// Set maximum chunk length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.config.max_length = max_length;
        self
    }

    /// Extract noun chunks from tokens
    ///
    /// Pattern: (DET)? (ADJ)* (NOUN|PROPN)+
    pub fn extract_chunks(&self, tokens: &[Token]) -> Vec<ChunkSpan> {
        let mut chunks = Vec::new();

        // Group tokens by sentence
        let mut sentences: Vec<Vec<&Token>> = Vec::new();
        let mut current_sent = Vec::new();
        let mut current_idx = None;

        for token in tokens {
            if current_idx != Some(token.sentence_idx) {
                if !current_sent.is_empty() {
                    sentences.push(std::mem::take(&mut current_sent));
                }
                current_idx = Some(token.sentence_idx);
            }
            current_sent.push(token);
        }
        if !current_sent.is_empty() {
            sentences.push(current_sent);
        }

        // Extract chunks from each sentence
        for sent_tokens in sentences {
            self.extract_chunks_from_sentence(&sent_tokens, &mut chunks);
        }

        chunks
    }

    /// Extract chunks from a single sentence
    fn extract_chunks_from_sentence(&self, tokens: &[&Token], chunks: &mut Vec<ChunkSpan>) {
        let mut i = 0;

        while i < tokens.len() {
            // Try to match a noun phrase pattern
            if let Some(span) = self.match_noun_phrase(tokens, i) {
                // Validate length constraints
                let len = span.end_token - span.start_token;
                if len >= self.config.min_length && len <= self.config.max_length {
                    let next_i = span.end_token - tokens[0].token_idx; // Move past this chunk
                    chunks.push(span);
                    i = next_i;
                    continue;
                }
            }
            i += 1;
        }
    }

    /// Try to match a noun phrase pattern starting at position i
    ///
    /// Pattern: (DET)? (ADJ)* (NOUN|PROPN)+
    fn match_noun_phrase(&self, tokens: &[&Token], start: usize) -> Option<ChunkSpan> {
        if start >= tokens.len() {
            return None;
        }

        let mut end = start;
        let first_token = tokens[start];

        // Optional determiner
        if self.config.include_determiners && tokens[end].pos == PosTag::Determiner {
            end += 1;
        } else if !self.config.include_determiners && tokens[end].pos == PosTag::Determiner {
            // Skip determiner if not including them
            return None;
        }

        if end >= tokens.len() {
            return None;
        }

        // Optional adjectives
        while end < tokens.len() && tokens[end].pos == PosTag::Adjective {
            end += 1;
        }

        // Required: at least one noun
        let noun_start = end;
        while end < tokens.len() && tokens[end].pos.is_noun() {
            end += 1;
        }

        // Must have at least one noun
        if end == noun_start {
            // No nouns found - check if we're on a standalone noun
            if start < tokens.len() && tokens[start].pos.is_noun() {
                return Some(ChunkSpan {
                    start_token: tokens[start].token_idx,
                    end_token: tokens[start].token_idx + 1,
                    start_char: tokens[start].start,
                    end_char: tokens[start].end,
                    sentence_idx: tokens[start].sentence_idx,
                });
            }
            return None;
        }

        let last_token = tokens[end - 1];

        Some(ChunkSpan {
            start_token: first_token.token_idx,
            end_token: tokens[end - 1].token_idx + 1,
            start_char: first_token.start,
            end_char: last_token.end,
            sentence_idx: first_token.sentence_idx,
        })
    }
}

/// Extract the text for a chunk span from the original tokens
pub fn chunk_text(tokens: &[Token], chunk: &ChunkSpan) -> String {
    tokens[chunk.start_token..chunk.end_token]
        .iter()
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract the lemmatized text for a chunk span
pub fn chunk_lemma(tokens: &[Token], chunk: &ChunkSpan) -> String {
    tokens[chunk.start_token..chunk.end_token]
        .iter()
        .map(|t| t.lemma.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokens() -> Vec<Token> {
        // "The quick brown fox jumps over the lazy dog"
        vec![
            Token::new("The", "the", PosTag::Determiner, 0, 3, 0, 0),
            Token::new("quick", "quick", PosTag::Adjective, 4, 9, 0, 1),
            Token::new("brown", "brown", PosTag::Adjective, 10, 15, 0, 2),
            Token::new("fox", "fox", PosTag::Noun, 16, 19, 0, 3),
            Token::new("jumps", "jump", PosTag::Verb, 20, 25, 0, 4),
            Token::new("over", "over", PosTag::Preposition, 26, 30, 0, 5),
            Token::new("the", "the", PosTag::Determiner, 31, 34, 0, 6),
            Token::new("lazy", "lazy", PosTag::Adjective, 35, 39, 0, 7),
            Token::new("dog", "dog", PosTag::Noun, 40, 43, 0, 8),
        ]
    }

    #[test]
    fn test_basic_chunk_extraction() {
        let tokens = make_tokens();
        let chunker = NounChunker::new();
        let chunks = chunker.extract_chunks(&tokens);

        // Should find "quick brown fox" and "lazy dog"
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_text_extraction() {
        let tokens = make_tokens();
        let chunker = NounChunker::new();
        let chunks = chunker.extract_chunks(&tokens);

        // Get text for each chunk
        let texts: Vec<_> = chunks.iter().map(|c| chunk_text(&tokens, c)).collect();

        assert!(texts.iter().any(|t| t.contains("fox")));
        assert!(texts.iter().any(|t| t.contains("dog")));
    }

    #[test]
    fn test_min_max_length() {
        let tokens = make_tokens();
        let chunker = NounChunker::new().with_min_length(2).with_max_length(3);
        let chunks = chunker.extract_chunks(&tokens);

        for chunk in &chunks {
            let len = chunk.end_token - chunk.start_token;
            assert!((2..=3).contains(&len));
        }
    }

    #[test]
    fn test_single_noun() {
        let tokens = vec![Token::new("machine", "machine", PosTag::Noun, 0, 7, 0, 0)];

        let chunker = NounChunker::new();
        let chunks = chunker.extract_chunks(&tokens);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunk_text(&tokens, &chunks[0]), "machine");
    }

    #[test]
    fn test_proper_noun() {
        let tokens = vec![
            Token::new("New", "new", PosTag::ProperNoun, 0, 3, 0, 0),
            Token::new("York", "york", PosTag::ProperNoun, 4, 8, 0, 1),
            Token::new("City", "city", PosTag::ProperNoun, 9, 13, 0, 2),
        ];

        let chunker = NounChunker::new();
        let chunks = chunker.extract_chunks(&tokens);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunk_text(&tokens, &chunks[0]), "New York City");
    }

    #[test]
    fn test_cross_sentence_boundary() {
        let tokens = vec![
            Token::new("machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 1, 1), // Different sentence
        ];

        let chunker = NounChunker::new();
        let chunks = chunker.extract_chunks(&tokens);

        // Should not merge across sentences
        assert_eq!(chunks.len(), 2);
    }
}
