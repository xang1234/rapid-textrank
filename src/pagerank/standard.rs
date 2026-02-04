//! Standard PageRank algorithm
//!
//! Implements the classic PageRank with power iteration and proper
//! handling of dangling nodes.

use super::PageRankResult;
use crate::graph::csr::CsrGraph;

/// Standard PageRank implementation
#[derive(Debug, Clone)]
pub struct StandardPageRank {
    /// Damping factor (typically 0.85)
    pub damping: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub threshold: f64,
}

impl Default for StandardPageRank {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            threshold: 1e-6,
        }
    }
}

impl StandardPageRank {
    /// Create a new StandardPageRank with default settings
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

    /// Run PageRank on a graph
    ///
    /// Returns the result even if convergence wasn't achieved, with `converged=false`.
    pub fn run(&self, graph: &CsrGraph) -> PageRankResult {
        let n = graph.num_nodes;
        if n == 0 {
            return PageRankResult::new(vec![], 0, 0.0, true);
        }

        // Initialize scores uniformly
        let initial_score = 1.0 / n as f64;
        let mut scores = vec![initial_score; n];
        let mut new_scores = vec![0.0; n];

        // Precompute dangling node mass contribution
        let dangling_nodes = graph.dangling_nodes();

        let teleport = (1.0 - self.damping) / n as f64;
        let mut iterations = 0;
        let mut delta = f64::MAX;

        while iterations < self.max_iterations && delta > self.threshold {
            iterations += 1;

            // Calculate dangling mass
            let dangling_mass: f64 = dangling_nodes.iter().map(|&d| scores[d as usize]).sum();
            let dangling_contribution = self.damping * dangling_mass / n as f64;

            // Reset new scores
            new_scores.fill(teleport + dangling_contribution);

            // Propagate scores through edges
            for (node, &node_score) in scores.iter().enumerate() {
                let total_weight = graph.node_total_weight(node as u32);

                if total_weight > 0.0 {
                    for (neighbor, weight) in graph.neighbors(node as u32) {
                        let contribution = self.damping * node_score * weight / total_weight;
                        new_scores[neighbor as usize] += contribution;
                    }
                }
            }

            // Calculate convergence delta (L1 norm)
            delta = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(old, new)| (old - new).abs())
                .sum();

            // Swap buffers
            std::mem::swap(&mut scores, &mut new_scores);
        }

        // Normalize scores (they should already sum to ~1, but ensure numerical stability)
        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for score in &mut scores {
                *score /= sum;
            }
        }

        PageRankResult::new(scores, iterations, delta, delta <= self.threshold)
    }

    /// Run PageRank with weighted edges
    ///
    /// Same as `run` but considers edge weights in the propagation.
    pub fn run_weighted(&self, graph: &CsrGraph) -> PageRankResult {
        // The standard run already uses weights - this is an alias for clarity
        self.run(graph)
    }

    /// Run PageRank ignoring edge weights (all edges have weight 1)
    pub fn run_unweighted(&self, graph: &CsrGraph) -> PageRankResult {
        let n = graph.num_nodes;
        if n == 0 {
            return PageRankResult::new(vec![], 0, 0.0, true);
        }

        let initial_score = 1.0 / n as f64;
        let mut scores = vec![initial_score; n];
        let mut new_scores = vec![0.0; n];

        let dangling_nodes = graph.dangling_nodes();
        let teleport = (1.0 - self.damping) / n as f64;
        let mut iterations = 0;
        let mut delta = f64::MAX;

        while iterations < self.max_iterations && delta > self.threshold {
            iterations += 1;

            let dangling_mass: f64 = dangling_nodes.iter().map(|&d| scores[d as usize]).sum();
            let dangling_contribution = self.damping * dangling_mass / n as f64;

            new_scores.fill(teleport + dangling_contribution);

            for (node, &node_score) in scores.iter().enumerate() {
                let degree = graph.degree(node as u32);

                if degree > 0 {
                    let contribution = self.damping * node_score / degree as f64;
                    for (neighbor, _) in graph.neighbors(node as u32) {
                        new_scores[neighbor as usize] += contribution;
                    }
                }
            }

            delta = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(old, new)| (old - new).abs())
                .sum();

            std::mem::swap(&mut scores, &mut new_scores);
        }

        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            for score in &mut scores {
                *score /= sum;
            }
        }

        PageRankResult::new(scores, iterations, delta, delta <= self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::GraphBuilder;

    fn build_triangle_graph() -> CsrGraph {
        let mut builder = GraphBuilder::new();
        let a = builder.get_or_create_node("a");
        let b = builder.get_or_create_node("b");
        let c = builder.get_or_create_node("c");

        builder.increment_edge(a, b, 1.0);
        builder.increment_edge(b, c, 1.0);
        builder.increment_edge(c, a, 1.0);

        CsrGraph::from_builder(&builder)
    }

    fn build_star_graph() -> CsrGraph {
        // Hub connected to 3 spokes
        let mut builder = GraphBuilder::new();
        let hub = builder.get_or_create_node("hub");
        let s1 = builder.get_or_create_node("s1");
        let s2 = builder.get_or_create_node("s2");
        let s3 = builder.get_or_create_node("s3");

        builder.increment_edge(hub, s1, 1.0);
        builder.increment_edge(hub, s2, 1.0);
        builder.increment_edge(hub, s3, 1.0);

        CsrGraph::from_builder(&builder)
    }

    #[test]
    fn test_triangle_graph_equal_scores() {
        let graph = build_triangle_graph();
        let pr = StandardPageRank::new();
        let result = pr.run(&graph);

        assert!(result.converged);
        // All nodes should have equal score in a symmetric graph
        let expected = 1.0 / 3.0;
        for score in &result.scores {
            assert!((score - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_star_graph_hub_highest() {
        let graph = build_star_graph();
        let pr = StandardPageRank::new();
        let result = pr.run(&graph);

        assert!(result.converged);
        // Hub should have highest score (it receives from all spokes)
        let hub_score = result.scores[0];
        for &score in &result.scores[1..] {
            assert!(hub_score >= score);
        }
    }

    #[test]
    fn test_scores_sum_to_one() {
        let graph = build_triangle_graph();
        let pr = StandardPageRank::new();
        let result = pr.run(&graph);

        let sum: f64 = result.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_graph() {
        let graph = CsrGraph::default();
        let pr = StandardPageRank::new();
        let result = pr.run(&graph);

        assert!(result.converged);
        assert!(result.scores.is_empty());
    }

    #[test]
    fn test_max_iterations_returns_partial() {
        let graph = build_triangle_graph();
        let pr = StandardPageRank::new()
            .with_max_iterations(1)
            .with_threshold(0.0); // Never converge

        let result = pr.run(&graph);

        assert_eq!(result.iterations, 1);
        assert!(!result.converged);
        // Should still have valid scores
        assert_eq!(result.scores.len(), 3);
    }

    #[test]
    fn test_damping_factor() {
        let graph = build_star_graph();

        // Lower damping = more teleportation = more uniform scores
        let pr_low = StandardPageRank::new().with_damping(0.5);
        let result_low = pr_low.run(&graph);

        let pr_high = StandardPageRank::new().with_damping(0.95);
        let result_high = pr_high.run(&graph);

        // With higher damping, hub advantage should be more pronounced
        let hub_advantage_low = result_low.scores[0] - result_low.scores[1];
        let hub_advantage_high = result_high.scores[0] - result_high.scores[1];

        assert!(hub_advantage_high > hub_advantage_low);
    }

    #[test]
    fn test_top_n() {
        let graph = build_star_graph();
        let pr = StandardPageRank::new();
        let result = pr.run(&graph);

        let top_2 = result.top_n(2);
        assert_eq!(top_2.len(), 2);
        // Hub should be first
        assert_eq!(top_2[0].0, 0);
    }
}
