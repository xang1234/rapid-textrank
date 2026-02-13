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
            for (node, &node_score) in scores.iter().enumerate() {
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

    /// Run Personalized PageRank, reusing externally-owned score buffers.
    ///
    /// Identical to [`run`](Self::run) but writes into the provided `score_buf`
    /// and `norm_buf` instead of allocating fresh vectors. The final scores are
    /// **cloned** into the returned [`PageRankResult`], so the caller retains
    /// buffer capacity for the next invocation.
    pub fn run_reusing(
        &self,
        graph: &CsrGraph,
        score_buf: &mut Vec<f64>,
        norm_buf: &mut Vec<f64>,
    ) -> PageRankResult {
        let n = graph.num_nodes;
        if n == 0 {
            return PageRankResult::new(vec![], 0, 0.0, true);
        }

        let personalization = self.prepare_personalization(n);

        let initial_score = 1.0 / n as f64;
        score_buf.clear();
        score_buf.resize(n, initial_score);
        norm_buf.clear();
        norm_buf.resize(n, 0.0);

        let dangling_nodes = graph.dangling_nodes();
        let mut iterations = 0;
        let mut delta = f64::MAX;

        while iterations < self.max_iterations && delta > self.threshold {
            iterations += 1;

            let dangling_mass: f64 = dangling_nodes.iter().map(|&d| score_buf[d as usize]).sum();

            for i in 0..n {
                norm_buf[i] = (1.0 - self.damping) * personalization[i]
                    + self.damping * dangling_mass * personalization[i];
            }

            for (node, &node_score) in score_buf.iter().enumerate() {
                let total_weight = graph.node_total_weight(node as u32);
                if total_weight > 0.0 {
                    for (neighbor, weight) in graph.neighbors(node as u32) {
                        let contribution = self.damping * node_score * weight / total_weight;
                        norm_buf[neighbor as usize] += contribution;
                    }
                }
            }

            delta = score_buf
                .iter()
                .zip(norm_buf.iter())
                .map(|(old, new)| (old - new).abs())
                .sum();

            std::mem::swap(score_buf, norm_buf);
        }

        let sum: f64 = score_buf.iter().sum();
        if sum > 0.0 {
            for score in score_buf.iter_mut() {
                *score /= sum;
            }
        }

        PageRankResult::new(score_buf.clone(), iterations, delta, delta <= self.threshold)
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

/// Create a topic-weight-based personalization vector for Topical PageRank
///
/// Each node's teleport probability is proportional to its topic weight.
/// Words not in `topic_weights` receive `min_weight` (PKE uses 0.0 for OOV).
///
/// When `use_pos_in_nodes` is true, the graph keys are `"lemma|POS"`. Each
/// lemma weight is applied to all POS variants present in the graph.
///
/// The returned vector is **not** normalized — `PersonalizedPageRank::run`
/// normalizes internally.
pub fn topic_weight_personalization(
    topic_weights: &std::collections::HashMap<String, f64>,
    graph: &CsrGraph,
    include_pos: &[crate::types::PosTag],
    use_pos_in_nodes: bool,
    min_weight: f64,
) -> Vec<f64> {
    let num_nodes = graph.num_nodes;
    let mut personalization = vec![min_weight; num_nodes];

    if use_pos_in_nodes {
        let default_pos = [
            crate::types::PosTag::Noun,
            crate::types::PosTag::Adjective,
            crate::types::PosTag::ProperNoun,
            crate::types::PosTag::Verb,
        ];
        let pos_tags: &[crate::types::PosTag] = if include_pos.is_empty() {
            &default_pos
        } else {
            include_pos
        };

        for (lemma, &weight) in topic_weights {
            for pos in pos_tags {
                let key = format!("{}|{}", lemma, pos.as_str());
                if let Some(node_id) = graph.get_node_by_lemma(&key) {
                    personalization[node_id as usize] = weight;
                }
            }
        }
    } else {
        for (lemma, &weight) in topic_weights {
            if let Some(node_id) = graph.get_node_by_lemma(lemma) {
                personalization[node_id as usize] = weight;
            }
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
    fn test_run_reusing_matches_run() {
        let graph = build_line_graph();
        let ppr = PersonalizedPageRank::new().with_personalization(vec![5.0, 1.0, 3.0]);

        let result_normal = ppr.run(&graph);

        let mut score_buf = Vec::new();
        let mut norm_buf = Vec::new();
        let result_reusing = ppr.run_reusing(&graph, &mut score_buf, &mut norm_buf);

        assert_eq!(result_normal.iterations, result_reusing.iterations);
        assert_eq!(result_normal.converged, result_reusing.converged);
        for (a, b) in result_normal.scores.iter().zip(result_reusing.scores.iter()) {
            assert!((a - b).abs() < 1e-12, "score mismatch: {a} vs {b}");
        }
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

    #[test]
    fn test_topic_weight_personalization_no_pos() {
        // Graph: a -- b -- c (lemma-only nodes)
        let graph = build_line_graph();

        let mut weights = std::collections::HashMap::new();
        weights.insert("a".to_string(), 0.8);
        weights.insert("c".to_string(), 0.3);
        // "b" not in weights → gets min_weight

        let p = topic_weight_personalization(&weights, &graph, &[], false, 0.0);

        assert!((p[0] - 0.8).abs() < 1e-10); // a
        assert!((p[1] - 0.0).abs() < 1e-10); // b (min_weight)
        assert!((p[2] - 0.3).abs() < 1e-10); // c
    }

    #[test]
    fn test_topic_weight_personalization_with_pos() {
        // Build graph with POS-tagged nodes: "machine|NOUN", "learn|VERB"
        let mut builder = GraphBuilder::new();
        let m = builder.get_or_create_node("machine|NOUN");
        let l = builder.get_or_create_node("learn|VERB");
        builder.increment_edge(m, l, 1.0);
        let graph = CsrGraph::from_builder(&builder);

        let mut weights = std::collections::HashMap::new();
        weights.insert("machine".to_string(), 0.9);
        // "learn" not in weights

        let p = topic_weight_personalization(
            &weights,
            &graph,
            &[crate::types::PosTag::Noun, crate::types::PosTag::Verb],
            true,
            0.1,
        );

        // "machine|NOUN" should get 0.9
        let machine_id = graph.get_node_by_lemma("machine|NOUN").unwrap();
        assert!((p[machine_id as usize] - 0.9).abs() < 1e-10);

        // "learn|VERB" should get min_weight (0.1)
        let learn_id = graph.get_node_by_lemma("learn|VERB").unwrap();
        assert!((p[learn_id as usize] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_topic_weight_personalization_empty_weights() {
        let graph = build_line_graph();
        let weights = std::collections::HashMap::new();

        let p = topic_weight_personalization(&weights, &graph, &[], false, 0.5);

        // All nodes get min_weight
        for &v in &p {
            assert!((v - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_topic_weight_personalization_oov() {
        let graph = build_line_graph();

        let mut weights = std::collections::HashMap::new();
        weights.insert("nonexistent".to_string(), 1.0);

        let p = topic_weight_personalization(&weights, &graph, &[], false, 0.0);

        // All nodes get min_weight since "nonexistent" isn't in the graph
        for &v in &p {
            assert!((v - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_topic_weight_biases_pagerank() {
        let graph = build_line_graph(); // a -- b -- c

        // Bias heavily towards "c"
        let mut weights = std::collections::HashMap::new();
        weights.insert("c".to_string(), 10.0);

        let p = topic_weight_personalization(&weights, &graph, &[], false, 0.0);
        let result = PersonalizedPageRank::new()
            .with_personalization(p)
            .run(&graph);

        // "c" should have higher score than without bias
        let uniform_result = PersonalizedPageRank::new().run(&graph);
        assert!(result.scores[2] > uniform_result.scores[2]);
    }
}
