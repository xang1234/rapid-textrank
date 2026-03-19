# Algorithm Variants

rapid_textrank ships with seven core algorithm variants plus `AutoRank`, a smart ensemble that fuses the full eligible keyword pool for a document. The core variants all share the same graph-and-PageRank foundation, but they differ in how they construct the graph, weight edges, and post-process results.

Choose the variant that best matches your document type and extraction goal. If you are unsure where to start, use `AutoRank` or see [Choosing a Variant](choosing-a-variant.md).

## Variant Comparison

| Variant | Best For | Description |
|---------|----------|-------------|
| [BaseTextRank](base-textrank.md) | General text | Standard TextRank implementation |
| [PositionRank](position-rank.md) | Academic papers, news | Favors words appearing early in the document |
| [BiasedTextRank](biased-textrank.md) | Topic-focused extraction | Biases results toward specified focus terms |
| [TopicRank](topic-rank.md) | Multi-topic documents | Clusters similar phrases into topics and ranks the topics |
| [SingleRank](single-rank.md) | Longer documents | Uses weighted co-occurrence edges and cross-sentence windowing |
| [TopicalPageRank](topical-pagerank.md) | Topic-model-guided extraction | Biases SingleRank towards topically important words via personalized PageRank |
| [MultipartiteRank](multipartite-rank.md) | Multi-topic documents | Builds a k-partite graph removing intra-topic edges; boosts first-occurring variants |

## Learn More

- [How TextRank Works](how-textrank-works.md) -- the three-step pipeline shared by all variants.
- [Choosing a Variant](choosing-a-variant.md) -- a decision flowchart and scenario table to help you pick the right one.
- `AutoRank` -- the recommended keyword default when you do not want to choose a single variant manually.

!!! tip "Interactive Notebook"
    Compare all variants side-by-side in the [Algorithm Variants Notebook](https://github.com/xang1234/rapid-textrank/blob/main/notebooks/02_algorithm_variants.ipynb).
