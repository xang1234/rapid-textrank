//! Graph builder with efficient edge handling
//!
//! This module provides a mutable graph builder that uses FxHashMap
//! for O(1) edge lookups during construction.

use crate::types::Token;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// A node in the graph builder
#[derive(Debug, Clone)]
pub struct BuilderNode {
    /// The lemma for this node
    pub lemma: String,
    /// Adjacency list: target node ID -> edge weight
    pub edges: FxHashMap<u32, f64>,
}

impl BuilderNode {
    /// Create a new node
    pub fn new(lemma: impl Into<String>) -> Self {
        Self {
            lemma: lemma.into(),
            edges: FxHashMap::default(),
        }
    }
}

/// A mutable graph builder optimized for incremental construction
#[derive(Debug)]
pub struct GraphBuilder {
    /// Maps lemma -> node ID
    lemma_to_id: FxHashMap<String, u32>,
    /// Node storage
    nodes: Vec<BuilderNode>,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder {
    /// Create a new empty graph builder
    pub fn new() -> Self {
        Self {
            lemma_to_id: FxHashMap::default(),
            nodes: Vec::new(),
        }
    }

    /// Create a graph builder with pre-allocated capacity
    pub fn with_capacity(node_capacity: usize) -> Self {
        Self {
            lemma_to_id: FxHashMap::with_capacity_and_hasher(node_capacity, Default::default()),
            nodes: Vec::with_capacity(node_capacity),
        }
    }

    /// Get or create a node for the given lemma, returning its ID
    pub fn get_or_create_node(&mut self, lemma: &str) -> u32 {
        if let Some(&id) = self.lemma_to_id.get(lemma) {
            return id;
        }

        let id = self.nodes.len() as u32;
        self.lemma_to_id.insert(lemma.to_string(), id);
        self.nodes.push(BuilderNode::new(lemma));
        id
    }

    /// Increment the edge weight between two nodes
    ///
    /// If the edge doesn't exist, it's created with weight 1.0
    pub fn increment_edge(&mut self, from: u32, to: u32, weight: f64) {
        if from == to {
            return; // No self-loops
        }

        // Add edge in both directions (undirected graph)
        if let Some(node) = self.nodes.get_mut(from as usize) {
            *node.edges.entry(to).or_insert(0.0) += weight;
        }
        if let Some(node) = self.nodes.get_mut(to as usize) {
            *node.edges.entry(from).or_insert(0.0) += weight;
        }
    }

    /// Build a graph from tokens using a sliding window
    ///
    /// This creates edges between tokens that co-occur within the window.
    pub fn from_tokens(tokens: &[Token], window_size: usize, use_weights: bool) -> Self {
        let mut builder = Self::with_capacity(tokens.len() / 2);

        // Filter to graph candidates (content words, non-stopwords)
        let candidates: Vec<_> = tokens
            .iter()
            .filter(|t| t.is_graph_candidate())
            .collect();

        // Process each sentence separately (don't create edges across sentences)
        let mut i = 0;
        while i < candidates.len() {
            let sent_idx = candidates[i].sentence_idx;

            // Find all candidates in this sentence
            let sent_start = i;
            while i < candidates.len() && candidates[i].sentence_idx == sent_idx {
                i += 1;
            }
            let sent_end = i;

            // Create nodes and edges within the sentence
            for j in sent_start..sent_end {
                let node_j = builder.get_or_create_node(&candidates[j].lemma);

                // Window extends forward
                for k in (j + 1)..std::cmp::min(j + window_size, sent_end) {
                    let node_k = builder.get_or_create_node(&candidates[k].lemma);
                    let weight = if use_weights { 1.0 } else { 1.0 };
                    builder.increment_edge(node_j, node_k, weight);
                }
            }
        }

        builder
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the total number of edges (counting each undirected edge once)
    pub fn edge_count(&self) -> usize {
        self.nodes.iter().map(|n| n.edges.len()).sum::<usize>() / 2
    }

    /// Get a node by ID
    pub fn get_node(&self, id: u32) -> Option<&BuilderNode> {
        self.nodes.get(id as usize)
    }

    /// Get a node ID by lemma
    pub fn get_node_id(&self, lemma: &str) -> Option<u32> {
        self.lemma_to_id.get(lemma).copied()
    }

    /// Get the lemma for a node ID
    pub fn get_lemma(&self, id: u32) -> Option<&str> {
        self.nodes.get(id as usize).map(|n| n.lemma.as_str())
    }

    /// Iterate over all nodes
    pub fn nodes(&self) -> impl Iterator<Item = (u32, &BuilderNode)> {
        self.nodes.iter().enumerate().map(|(i, n)| (i as u32, n))
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Thread-safe counter for parallel graph building
#[derive(Debug, Default)]
pub struct AtomicCounter {
    value: AtomicU32,
}

impl AtomicCounter {
    /// Create a new counter
    pub fn new() -> Self {
        Self {
            value: AtomicU32::new(0),
        }
    }

    /// Increment and return the new value
    pub fn increment(&self) -> u32 {
        self.value.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the current value
    pub fn get(&self) -> u32 {
        self.value.load(Ordering::SeqCst)
    }
}

/// Build a graph from tokens in parallel (for large documents)
///
/// This splits the document into chunks, builds partial graphs in parallel,
/// and then merges them.
pub fn build_graph_parallel(tokens: &[Token], window_size: usize, use_weights: bool) -> GraphBuilder {
    // For small documents, sequential is faster
    if tokens.len() < 1000 {
        return GraphBuilder::from_tokens(tokens, window_size, use_weights);
    }

    // Group tokens by sentence for parallel processing
    let mut sentences: Vec<Vec<&Token>> = Vec::new();
    let mut current_sent = Vec::new();
    let mut current_idx = None;

    for token in tokens.iter().filter(|t| t.is_graph_candidate()) {
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

    // Build partial graphs in parallel
    let partial_graphs: Vec<FxHashMap<(String, String), f64>> = sentences
        .par_iter()
        .map(|sent_tokens| {
            let mut edges = FxHashMap::default();
            for i in 0..sent_tokens.len() {
                for j in (i + 1)..std::cmp::min(i + window_size, sent_tokens.len()) {
                    let (a, b) = if sent_tokens[i].lemma <= sent_tokens[j].lemma {
                        (sent_tokens[i].lemma.clone(), sent_tokens[j].lemma.clone())
                    } else {
                        (sent_tokens[j].lemma.clone(), sent_tokens[i].lemma.clone())
                    };
                    if a != b {
                        let weight = if use_weights { 1.0 } else { 1.0 };
                        *edges.entry((a, b)).or_insert(0.0) += weight;
                    }
                }
            }
            edges
        })
        .collect();

    // Merge partial graphs
    let mut builder = GraphBuilder::new();
    for partial in partial_graphs {
        for ((a, b), weight) in partial {
            let id_a = builder.get_or_create_node(&a);
            let id_b = builder.get_or_create_node(&b);
            builder.increment_edge(id_a, id_b, weight);
        }
    }

    builder
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PosTag;

    fn make_token(text: &str, lemma: &str, sent_idx: usize, tok_idx: usize) -> Token {
        Token {
            text: text.to_string(),
            lemma: lemma.to_string(),
            pos: PosTag::Noun,
            start: 0,
            end: text.len(),
            sentence_idx: sent_idx,
            token_idx: tok_idx,
            is_stopword: false,
        }
    }

    #[test]
    fn test_graph_builder_basic() {
        let mut builder = GraphBuilder::new();

        let id_a = builder.get_or_create_node("machine");
        let id_b = builder.get_or_create_node("learning");
        let id_c = builder.get_or_create_node("machine"); // duplicate

        assert_eq!(id_a, id_c); // Same lemma should get same ID
        assert_ne!(id_a, id_b);
        assert_eq!(builder.node_count(), 2);
    }

    #[test]
    fn test_edge_incrementing() {
        let mut builder = GraphBuilder::new();

        let id_a = builder.get_or_create_node("machine");
        let id_b = builder.get_or_create_node("learning");

        builder.increment_edge(id_a, id_b, 1.0);
        builder.increment_edge(id_a, id_b, 1.0);

        // Should have weight 2.0 in both directions
        assert_eq!(builder.get_node(id_a).unwrap().edges.get(&id_b), Some(&2.0));
        assert_eq!(builder.get_node(id_b).unwrap().edges.get(&id_a), Some(&2.0));
    }

    #[test]
    fn test_from_tokens() {
        let tokens = vec![
            make_token("machine", "machine", 0, 0),
            make_token("learning", "learning", 0, 1),
            make_token("is", "is", 0, 2), // Will be filtered if stopword
            make_token("great", "great", 0, 3),
        ];

        let builder = GraphBuilder::from_tokens(&tokens, 3, true);

        assert_eq!(builder.node_count(), 4);
        // "machine" should be connected to "learning" and "is" (within window)
        let machine_id = builder.get_node_id("machine").unwrap();
        let node = builder.get_node(machine_id).unwrap();
        assert!(!node.edges.is_empty());
    }

    #[test]
    fn test_no_cross_sentence_edges() {
        let tokens = vec![
            make_token("machine", "machine", 0, 0),
            make_token("learning", "learning", 0, 1),
            make_token("deep", "deep", 1, 2), // Different sentence
            make_token("neural", "neural", 1, 3),
        ];

        let builder = GraphBuilder::from_tokens(&tokens, 3, true);

        // "learning" and "deep" should NOT be connected (different sentences)
        let learning_id = builder.get_node_id("learning").unwrap();
        let deep_id = builder.get_node_id("deep").unwrap();
        let learning_node = builder.get_node(learning_id).unwrap();
        assert!(!learning_node.edges.contains_key(&deep_id));
    }

    #[test]
    fn test_self_loops_prevented() {
        let mut builder = GraphBuilder::new();
        let id_a = builder.get_or_create_node("test");

        builder.increment_edge(id_a, id_a, 1.0);

        // No self-loop should be created
        let node = builder.get_node(id_a).unwrap();
        assert!(node.edges.is_empty());
    }
}
