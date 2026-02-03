//! PageRank algorithms
//!
//! This module provides standard and personalized PageRank implementations.

pub mod personalized;
pub mod standard;

/// Result of a PageRank computation
#[derive(Debug, Clone)]
pub struct PageRankResult {
    /// Scores for each node (indexed by node ID)
    pub scores: Vec<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final convergence delta
    pub delta: f64,
    /// Whether the algorithm converged
    pub converged: bool,
}

impl PageRankResult {
    /// Create a new PageRank result
    pub fn new(scores: Vec<f64>, iterations: usize, delta: f64, converged: bool) -> Self {
        Self {
            scores,
            iterations,
            delta,
            converged,
        }
    }

    /// Get top N nodes by score
    pub fn top_n(&self, n: usize) -> Vec<(u32, f64)> {
        let mut indexed: Vec<_> = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i as u32, s))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(n);
        indexed
    }

    /// Get the score for a specific node
    pub fn score(&self, node: u32) -> f64 {
        self.scores.get(node as usize).copied().unwrap_or(0.0)
    }
}
