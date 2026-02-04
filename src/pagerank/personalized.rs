//! Personalized PageRank (PPR) algorithm
//!
//! PPR uses a custom teleport distribution instead of uniform teleportation.
//! This allows biasing the ranking towards specific nodes (e.g., for
//! PositionRank or BiasedTextRank).

use super::PageRankResult;
use crate::graph::csr::CsrGraph;

/// Personalized PageRank implementation
#[derive(Debug, Clone)]
pub struct PersonalizedPageRank {
    /// Damping factor (typically 0.85)
    pub damping: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub threshold: f64,
    /// Personalization vector (bias distribution)
    personalization: Option<Vec<f64>>,
}

impl Default for PersonalizedPageRank {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            threshold: 1e-6,
            personalization: None,
        }
    }
}

impl PersonalizedPageRank {
    /// Create a new PersonalizedPageRank with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the damping factor
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Set the maximum iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the convergence threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the personalization vector (bias distribution)
    ///
    /// The vector should have one entry per node. It will be normalized internally.
    /// Nodes with higher values will be favored during teleportation.
    pub fn with_personalization(mut self, personalization: Vec<f64>) -> Self {
        self.personalization = Some(personalization);
        self
    }

    /// Set personalization from a sparse representation
    ///
    /// Takes a list of (node_id, weight) pairs and the total number of nodes.
    /// Nodes not in the list get weight 0.
    pub fn with_sparse_personalization(mut self, biases: &[(u32, f64)], num_nodes: usize) -> Self {
        let mut personalization = vec![0.0; num_nodes];
        for &(node, weight) in biases {
            if (node as usize) < num_nodes {
                personalization[node as usize] = weight;
            }
        }
        self.personalization = Some(personalization);
        self
    }

    /// Run Personalized PageRank on a graph
    pub fn run(&self, graph: &CsrGraph) -> PageRankResult {
        let n = graph.num_nodes;
        if n == 0 {
            return PageRankResult::new(vec![], 0, 0.0, true);
        }

        // Prepare personalization vector
        let personalization = self.prepare_personalization(n);

        // Initialize scores uniformly
        let initial_score = 1.0 / n as f64;
        let mut scores = vec![initial_score; n];
        let mut new_scores = vec![0.0; n];

        let dangling_nodes = graph.dangling_nodes();
        let mut iterations = 0;
        let mut delta = f64::MAX;

        while iterations < self.max_iterations && delta > self.threshold {
            iterations += 1;

            // Calculate dangling mass (goes to personalization distribution)
            let dangling_mass: f64 = dangling_nodes.iter().map(|&d| scores[d as usize]).sum();

            // Initialize with teleport probability based on personalization
            for i in 0..n {
                new_scores[i] = (1.0 - self.damping) * personalization[i]
                    + self.damping * dangling_mass * personalization[i];
            }

            // Propagate scores through edges
            for node in 0..n {
                let node_score = scores[node];
                let total_weight = graph.node_total_weight(node as u32);

                if total_weight > 0.0 {
                    for (neighbor, weight) in graph.neighbors(node as u32) {
                        let contribution = self.damping * node_score * weight / total_weight;
                        new_scores[neighbor as usize] += contribution;
                    }
                }
            }

            // Calculate convergence delta
            delta = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(old, new)| (old - new).abs())
                .sum();

            std::mem::swap(&mut scores, &mut new_scores);
        }

        // Normalize scores
        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for score in &mut scores {
                *score /= sum;
            }
        }

        PageRankResult::new(scores, iterations, delta, delta <= self.threshold)
    }

    /// Prepare and normalize the personalization vector
    fn prepare_personalization(&self, n: usize) -> Vec<f64> {
        match &self.personalization {
            Some(p) if p.len() == n => {
                // Normalize the provided personalization
                let sum: f64 = p.iter().sum();
                if sum > 0.0 {
                    p.iter().map(|&x| x / sum).collect()
                } else {
                    // Fall back to uniform if sum is 0
                    vec![1.0 / n as f64; n]
                }
            }
            Some(p) => {
                // Resize to match graph size
                let mut result = vec![0.0; n];
                for (i, &v) in p.iter().enumerate().take(n) {
                    result[i] = v;
                }
                let sum: f64 = result.iter().sum();
                if sum > 0.0 {
                    for v in &mut result {
                        *v /= sum;
                    }
                } else {
                    result = vec![1.0 / n as f64; n];
                }
                result
            }
            None => {
                // Uniform distribution (equivalent to standard PageRank)
                vec![1.0 / n as f64; n]
            }
        }
    }
}

/// Create a position-based personalization vector
///
/// Assigns weight 1/(position + 1) to each node's first occurrence.
pub fn position_based_personalization(
    first_positions: &[(u32, usize)],
    num_nodes: usize,
) -> Vec<f64> {
    let mut personalization = vec![0.0; num_nodes];
    for &(node, position) in first_positions {
        if (node as usize) < num_nodes {
            personalization[node as usize] = 1.0 / (position as f64 + 1.0);
        }
    }
    personalization
}

/// Create a focus-based personalization vector for BiasedTextRank
///
/// Assigns `bias_weight` to focus nodes and 1.0 to others.
pub fn focus_based_personalization(
    focus_nodes: &[u32],
    bias_weight: f64,
    num_nodes: usize,
) -> Vec<f64> {
    let mut personalization = vec![1.0; num_nodes];
    for &node in focus_nodes {
        if (node as usize) < num_nodes {
            personalization[node as usize] = bias_weight;
        }
    }
    personalization
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::GraphBuilder;

    fn build_line_graph() -> CsrGraph {
        // A -- B -- C
        let mut builder = GraphBuilder::new();
        let a = builder.get_or_create_node("a");
        let b = builder.get_or_create_node("b");
        let c = builder.get_or_create_node("c");

        builder.increment_edge(a, b, 1.0);
        builder.increment_edge(b, c, 1.0);

        CsrGraph::from_builder(&builder)
    }

    #[test]
    fn test_uniform_personalization_equals_standard() {
        let graph = build_line_graph();

        let standard = crate::pagerank::standard::StandardPageRank::new();
        let result_standard = standard.run(&graph);

        let ppr = PersonalizedPageRank::new();
        let result_ppr = ppr.run(&graph);

        // With uniform personalization, PPR should behave like standard PR
        for (s, p) in result_standard.scores.iter().zip(result_ppr.scores.iter()) {
            assert!((s - p).abs() < 0.01);
        }
    }

    #[test]
    fn test_biased_personalization() {
        let graph = build_line_graph();

        // Heavily bias towards node A
        let ppr = PersonalizedPageRank::new().with_personalization(vec![10.0, 1.0, 1.0]);
        let result = ppr.run(&graph);

        // Node A should have higher score due to bias
        assert!(result.scores[0] > result.scores[2]);
    }

    #[test]
    fn test_sparse_personalization() {
        let graph = build_line_graph();

        // Only bias node C
        let ppr = PersonalizedPageRank::new().with_sparse_personalization(&[(2, 5.0)], 3);
        let result = ppr.run(&graph);

        // Node C should have higher score than without bias
        let ppr_uniform = PersonalizedPageRank::new();
        let result_uniform = ppr_uniform.run(&graph);

        assert!(result.scores[2] > result_uniform.scores[2]);
    }

    #[test]
    fn test_position_based_personalization() {
        let personalization = position_based_personalization(&[(0, 0), (1, 5), (2, 10)], 3);

        // Earlier positions should have higher weights
        assert!(personalization[0] > personalization[1]);
        assert!(personalization[1] > personalization[2]);

        // Check actual values: 1/(0+1) = 1, 1/(5+1) = 0.167, 1/(10+1) = 0.091
        assert!((personalization[0] - 1.0).abs() < 1e-6);
        assert!((personalization[1] - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_focus_based_personalization() {
        let personalization = focus_based_personalization(&[0, 2], 5.0, 3);

        assert!((personalization[0] - 5.0).abs() < 1e-6);
        assert!((personalization[1] - 1.0).abs() < 1e-6);
        assert!((personalization[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_graph() {
        let graph = CsrGraph::default();
        let ppr = PersonalizedPageRank::new();
        let result = ppr.run(&graph);

        assert!(result.converged);
        assert!(result.scores.is_empty());
    }

    #[test]
    fn test_scores_sum_to_one() {
        let graph = build_line_graph();
        let ppr = PersonalizedPageRank::new().with_personalization(vec![5.0, 1.0, 3.0]);
        let result = ppr.run(&graph);

        let sum: f64 = result.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
