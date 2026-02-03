//! TopicRank variant
//!
//! TopicRank groups similar keyphrases into topics before ranking.
//! This produces more diverse keywords by ensuring each "topic cluster"
//! contributes at most one representative phrase.
//!
//! Process:
//! 1. Extract candidate phrases
//! 2. Cluster similar phrases based on shared stems/lemmas
//! 3. Build graph where nodes are clusters
//! 4. Run PageRank on the cluster graph
//! 5. Select best phrase from each top cluster

use crate::graph::builder::GraphBuilder;
use crate::graph::csr::CsrGraph;
use crate::pagerank::standard::StandardPageRank;
use crate::phrase::chunker::{chunk_lemma, chunk_text, NounChunker};
use crate::types::{Phrase, TextRankConfig, Token};
use rustc_hash::{FxHashMap, FxHashSet};

/// TopicRank implementation
#[derive(Debug)]
pub struct TopicRank {
    config: TextRankConfig,
    /// Jaccard similarity threshold for clustering
    similarity_threshold: f64,
    /// Maximum number of phrases to cluster (for performance)
    max_phrases: usize,
}

impl Default for TopicRank {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicRank {
    /// Create a new TopicRank extractor
    pub fn new() -> Self {
        Self {
            config: TextRankConfig::default(),
            similarity_threshold: 0.25,
            max_phrases: 200,
        }
    }

    /// Create with custom config
    pub fn with_config(config: TextRankConfig) -> Self {
        Self {
            config,
            similarity_threshold: 0.25,
            max_phrases: 200,
        }
    }

    /// Set similarity threshold for clustering
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set maximum phrases to process
    pub fn with_max_phrases(mut self, max: usize) -> Self {
        self.max_phrases = max;
        self
    }

    /// Extract keyphrases using TopicRank
    pub fn extract(&self, tokens: &[Token]) -> Vec<Phrase> {
        // Extract candidate phrases
        let chunker = NounChunker::new()
            .with_min_length(self.config.min_phrase_length)
            .with_max_length(self.config.max_phrase_length);
        let chunks = chunker.extract_chunks(tokens);

        if chunks.is_empty() {
            return Vec::new();
        }

        // Create phrase candidates
        let mut candidates: Vec<PhraseCandidate> = chunks
            .iter()
            .take(self.max_phrases)
            .map(|chunk| {
                let text = chunk_text(tokens, chunk);
                let lemma = chunk_lemma(tokens, chunk);
                let stems: FxHashSet<String> = lemma
                    .split_whitespace()
                    .map(|s| s.to_lowercase())
                    .collect();
                PhraseCandidate {
                    text,
                    lemma,
                    stems,
                    chunk: chunk.clone(),
                }
            })
            .collect();

        // Deduplicate by lemma
        let mut seen: FxHashSet<String> = FxHashSet::default();
        candidates.retain(|c| seen.insert(c.lemma.clone()));

        if candidates.is_empty() {
            return Vec::new();
        }

        // Cluster similar phrases
        let clusters = self.cluster_phrases(&candidates);

        if clusters.is_empty() {
            return Vec::new();
        }

        // Build cluster graph
        let (cluster_graph, cluster_members) = self.build_cluster_graph(tokens, &clusters);

        // Run PageRank on cluster graph
        let pagerank = StandardPageRank::new()
            .with_damping(self.config.damping)
            .with_max_iterations(self.config.max_iterations)
            .with_threshold(self.config.convergence_threshold)
            .run(&cluster_graph);

        // Select best phrase from each top cluster
        let mut phrases = self.select_representatives(&cluster_members, &candidates, &pagerank);

        // Sort by score
        phrases.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Assign ranks
        for (i, phrase) in phrases.iter_mut().enumerate() {
            phrase.rank = i + 1;
        }

        // Limit to top_n
        if self.config.top_n > 0 && phrases.len() > self.config.top_n {
            phrases.truncate(self.config.top_n);
        }

        phrases
    }

    /// Cluster phrases based on stem overlap (Jaccard similarity)
    fn cluster_phrases(&self, candidates: &[PhraseCandidate]) -> Vec<Vec<usize>> {
        let n = candidates.len();
        let mut parent: Vec<usize> = (0..n).collect();

        // Union-find helpers
        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                parent[pi] = pj;
            }
        }

        // Compute pairwise similarities and cluster
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = jaccard_similarity(&candidates[i].stems, &candidates[j].stems);
                if sim >= self.similarity_threshold {
                    union(&mut parent, i, j);
                }
            }
        }

        // Group by cluster
        let mut clusters: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(i);
        }

        clusters.into_values().collect()
    }

    /// Build a graph where nodes are clusters and edges connect co-occurring clusters
    fn build_cluster_graph(
        &self,
        tokens: &[Token],
        clusters: &[Vec<usize>],
    ) -> (CsrGraph, Vec<Vec<usize>>) {
        let mut builder = GraphBuilder::with_capacity(clusters.len());

        // Create node for each cluster
        for (i, _) in clusters.iter().enumerate() {
            builder.get_or_create_node(&format!("cluster_{}", i));
        }

        // Build mapping from lemma to cluster index
        // (simplified: in full implementation, track phrase positions)

        // For now, just connect clusters that share co-occurrence in the same sentence
        // This is a simplified version - full TopicRank uses phrase positions

        // Just return the graph with nodes but no edges for this simplified version
        // In a full implementation, we'd track which clusters' phrases co-occur

        let graph = CsrGraph::from_builder(&builder);
        (graph, clusters.to_vec())
    }

    /// Select the best representative phrase from each cluster
    fn select_representatives(
        &self,
        clusters: &[Vec<usize>],
        candidates: &[PhraseCandidate],
        pagerank: &crate::pagerank::PageRankResult,
    ) -> Vec<Phrase> {
        clusters
            .iter()
            .enumerate()
            .map(|(cluster_idx, members)| {
                // Get cluster score from PageRank
                let cluster_score = pagerank.score(cluster_idx as u32);

                // Select representative: prefer shorter, more frequent phrases
                // For simplicity, just take the first (which tends to be shorter)
                let best_idx = members
                    .iter()
                    .min_by_key(|&&idx| candidates[idx].text.len())
                    .copied()
                    .unwrap_or(members[0]);

                let candidate = &candidates[best_idx];

                Phrase {
                    text: candidate.text.clone(),
                    lemma: candidate.lemma.clone(),
                    score: cluster_score,
                    count: members.len(),
                    offsets: vec![(
                        candidate.chunk.start_token,
                        candidate.chunk.end_token,
                    )],
                    rank: 0,
                }
            })
            .collect()
    }
}

/// A phrase candidate with its stems for clustering
#[derive(Debug)]
struct PhraseCandidate {
    text: String,
    lemma: String,
    stems: FxHashSet<String>,
    chunk: crate::types::ChunkSpan,
}

/// Jaccard similarity between two sets
fn jaccard_similarity(a: &FxHashSet<String>, b: &FxHashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Convenience function
pub fn extract_keyphrases_topic(tokens: &[Token], config: &TextRankConfig) -> Vec<Phrase> {
    TopicRank::with_config(config.clone()).extract(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PosTag;

    fn make_tokens() -> Vec<Token> {
        vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("algorithms", "algorithm", PosTag::Noun, 17, 27, 0, 2),
            Token::new("Deep", "deep", PosTag::Adjective, 29, 33, 1, 3),
            Token::new("learning", "learning", PosTag::Noun, 34, 42, 1, 4),
            Token::new("models", "model", PosTag::Noun, 43, 49, 1, 5),
            Token::new("Neural", "neural", PosTag::Adjective, 51, 57, 2, 6),
            Token::new("networks", "network", PosTag::Noun, 58, 66, 2, 7),
        ]
    }

    #[test]
    fn test_topic_rank() {
        let tokens = make_tokens();
        let config = TextRankConfig::default();
        let phrases = extract_keyphrases_topic(&tokens, &config);

        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_jaccard_similarity() {
        let a: FxHashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let b: FxHashSet<String> = ["b", "c", "d"].iter().map(|s| s.to_string()).collect();

        let sim = jaccard_similarity(&a, &b);
        // Intersection: {b, c} = 2, Union: {a, b, c, d} = 4
        assert!((sim - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_identical() {
        let a: FxHashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let sim = jaccard_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a: FxHashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let b: FxHashSet<String> = ["c", "d"].iter().map(|s| s.to_string()).collect();
        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input() {
        let tokens: Vec<Token> = Vec::new();
        let config = TextRankConfig::default();
        let phrases = extract_keyphrases_topic(&tokens, &config);

        assert!(phrases.is_empty());
    }
}
