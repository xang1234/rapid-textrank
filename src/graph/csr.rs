//! Compressed Sparse Row (CSR) graph representation
//!
//! CSR is optimized for iteration over neighbors, which is exactly what
//! PageRank needs during power iteration.

use super::builder::GraphBuilder;

/// A graph in Compressed Sparse Row format
///
/// CSR stores edges contiguously, making iteration over neighbors very fast.
/// This is ideal for PageRank which repeatedly iterates over all edges.
#[derive(Debug, Clone)]
pub struct CsrGraph {
    /// Number of nodes
    pub num_nodes: usize,
    /// Row pointers: node i's edges are at indices row_ptr[i]..row_ptr[i+1]
    pub row_ptr: Vec<usize>,
    /// Column indices (target nodes) for each edge
    pub col_idx: Vec<u32>,
    /// Edge weights
    pub weights: Vec<f64>,
    /// Out-degree for each node
    pub out_degree: Vec<u32>,
    /// Total outgoing weight for each node
    pub total_weight: Vec<f64>,
    /// Lemmas for each node
    pub lemmas: Vec<String>,
}

impl CsrGraph {
    /// Convert a GraphBuilder into CSR format
    pub fn from_builder(builder: &GraphBuilder) -> Self {
        let num_nodes = builder.node_count();
        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();
        let mut out_degree = Vec::with_capacity(num_nodes);
        let mut total_weight = Vec::with_capacity(num_nodes);
        let mut lemmas = Vec::with_capacity(num_nodes);

        row_ptr.push(0);

        for (_, node) in builder.nodes() {
            lemmas.push(node.lemma.clone());

            // Collect and sort edges for deterministic iteration
            let mut edges: Vec<_> = node.edges.iter().map(|(&k, &v)| (k, v)).collect();
            edges.sort_by_key(|(k, _)| *k);

            out_degree.push(edges.len() as u32);
            total_weight.push(edges.iter().map(|(_, w)| w).sum());

            for (target, weight) in edges {
                col_idx.push(target);
                weights.push(weight);
            }

            row_ptr.push(col_idx.len());
        }

        Self {
            num_nodes,
            row_ptr,
            col_idx,
            weights,
            out_degree,
            total_weight,
            lemmas,
        }
    }

    /// Iterate over neighbors of a node
    pub fn neighbors(&self, node: u32) -> impl Iterator<Item = (u32, f64)> + '_ {
        let start = self.row_ptr[node as usize];
        let end = self.row_ptr[node as usize + 1];
        (start..end).map(move |i| (self.col_idx[i], self.weights[i]))
    }

    /// Get the out-degree of a node
    pub fn degree(&self, node: u32) -> u32 {
        self.out_degree[node as usize]
    }

    /// Get the total outgoing weight of a node
    pub fn node_total_weight(&self, node: u32) -> f64 {
        self.total_weight[node as usize]
    }

    /// Get the lemma for a node
    pub fn lemma(&self, node: u32) -> &str {
        &self.lemmas[node as usize]
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.num_nodes == 0
    }

    /// Get the total number of edges (counting each undirected edge twice)
    pub fn num_edges(&self) -> usize {
        self.col_idx.len()
    }

    /// Find dangling nodes (nodes with no outgoing edges)
    pub fn dangling_nodes(&self) -> Vec<u32> {
        (0..self.num_nodes as u32)
            .filter(|&n| self.out_degree[n as usize] == 0)
            .collect()
    }

    /// Get node ID by lemma (linear search - use sparingly)
    pub fn get_node_by_lemma(&self, lemma: &str) -> Option<u32> {
        self.lemmas.iter().position(|l| l == lemma).map(|i| i as u32)
    }
}

impl Default for CsrGraph {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            row_ptr: vec![0],
            col_idx: Vec::new(),
            weights: Vec::new(),
            out_degree: Vec::new(),
            total_weight: Vec::new(),
            lemmas: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_graph() -> GraphBuilder {
        let mut builder = GraphBuilder::new();
        let a = builder.get_or_create_node("a");
        let b = builder.get_or_create_node("b");
        let c = builder.get_or_create_node("c");

        builder.increment_edge(a, b, 1.0);
        builder.increment_edge(b, c, 2.0);
        builder.increment_edge(a, c, 1.5);

        builder
    }

    #[test]
    fn test_csr_conversion() {
        let builder = build_test_graph();
        let csr = CsrGraph::from_builder(&builder);

        assert_eq!(csr.num_nodes, 3);
        assert_eq!(csr.lemmas, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_neighbor_iteration() {
        let builder = build_test_graph();
        let csr = CsrGraph::from_builder(&builder);

        // Node "a" (id 0) should have neighbors "b" and "c"
        let neighbors: Vec<_> = csr.neighbors(0).collect();
        assert_eq!(neighbors.len(), 2);

        // Check that weights are correct
        let b_neighbor = neighbors.iter().find(|(n, _)| *n == 1);
        assert!(b_neighbor.is_some());
        assert!((b_neighbor.unwrap().1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_degree_and_weight() {
        let builder = build_test_graph();
        let csr = CsrGraph::from_builder(&builder);

        // Node "a" has degree 2 (connected to b and c)
        assert_eq!(csr.degree(0), 2);

        // Total weight should be 1.0 + 1.5 = 2.5
        assert!((csr.node_total_weight(0) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_empty_graph() {
        let builder = GraphBuilder::new();
        let csr = CsrGraph::from_builder(&builder);

        assert!(csr.is_empty());
        assert_eq!(csr.num_edges(), 0);
    }

    #[test]
    fn test_dangling_nodes() {
        // Create a graph with a dangling node
        let mut builder = GraphBuilder::new();
        let a = builder.get_or_create_node("a");
        let b = builder.get_or_create_node("b");
        let c = builder.get_or_create_node("c"); // No edges from c
        builder.increment_edge(a, b, 1.0);
        let _ = c; // Unused, but node exists

        let csr = CsrGraph::from_builder(&builder);

        // In an undirected graph with edges a-b, both a and b have degree 1, c has degree 0
        // But since we add edges in both directions, a and b each have 1 edge
        // c has no edges, so it's dangling
        let dangling = csr.dangling_nodes();
        assert!(dangling.contains(&2)); // c is dangling
    }

    #[test]
    fn test_get_node_by_lemma() {
        let builder = build_test_graph();
        let csr = CsrGraph::from_builder(&builder);

        assert_eq!(csr.get_node_by_lemma("a"), Some(0));
        assert_eq!(csr.get_node_by_lemma("b"), Some(1));
        assert_eq!(csr.get_node_by_lemma("z"), None);
    }
}
