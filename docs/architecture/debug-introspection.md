# Debug and Introspection Outputs

rapid_textrank provides a tiered debug system that lets you see inside the pipeline — from graph statistics to per-iteration convergence residuals. This is useful when you want to understand *why* certain keywords were ranked higher than others. `AutoRank` preserves this contract by aggregating the per-variant debug payloads from the ensemble run into a single result-level `debug` object.

---

## Quick Start

### Via `TextRankConfig`

Set `debug_level` to control what debug data is attached to results:

```python
from rapid_textrank import TextRankConfig, PositionRank

config = TextRankConfig(debug_level="top_nodes")
result = PositionRank().extract_with_info(tokens, config)
print(result.debug)  # contains graph_stats, convergence_summary, node_scores
```

### Via `PipelineSpec` (JSON interface)

Add an `expose` section to your pipeline spec for fine-grained control:

```json
{
  "v": 1,
  "preset": "textrank",
  "expose": {
    "node_scores": { "top_k": 20 },
    "graph_stats": true,
    "stage_timings": true
  }
}
```

---

## Debug Levels

Debug output is organized into four cumulative levels. Each level is a strict superset of the previous:

```
None ⊂ Stats ⊂ TopNodes ⊂ Full
```

| Level | Serialized | What's Included | Overhead |
|-------|-----------|-----------------|----------|
| `None` | `"none"` | Nothing (default) | Zero — compiled away |
| `Stats` | `"stats"` | Graph statistics, convergence summary, stage timings | Negligible |
| `TopNodes` | `"top_nodes"` | Everything in Stats + top-K node scores | Allocates + sorts score vector |
| `Full` | `"full"` | Everything in TopNodes + per-iteration residuals + cluster memberships | Can be substantial on large graphs |

The level ordering is enforced at the type level via `PartialOrd`:

```rust
assert!(DebugLevel::Stats < DebugLevel::TopNodes);
assert!(DebugLevel::TopNodes < DebugLevel::Full);
```

---

## Debug Payload Fields

The `DebugPayload` struct contains all introspection data. Fields are individually optional — you only pay for what you ask for.

### `graph_stats` (available at `Stats` and above)

Summary statistics for the co-occurrence graph.

| Field | Type | Description |
|-------|------|-------------|
| `num_nodes` | `usize` | Number of nodes (unique candidates) in the graph |
| `num_edges` | `usize` | Number of edges (co-occurrence links) |
| `is_transformed` | `bool` | Whether a `GraphTransform` stage modified the graph |

**Interpretation:** For word-graph variants, `num_nodes` is the number of unique (lemma, POS) pairs that survived filtering. For topic variants, it's the number of clusters or candidates depending on the variant. A high edge-to-node ratio indicates a dense graph where many candidates co-occur.

```json
{
  "graph_stats": {
    "num_nodes": 42,
    "num_edges": 156,
    "is_transformed": false
  }
}
```

### `convergence_summary` (available at `Stats` and above)

PageRank convergence metadata.

| Field | Type | Description |
|-------|------|-------------|
| `iterations` | `u32` | Number of power-iteration steps performed |
| `converged` | `bool` | Whether the L1-norm delta dropped below `convergence_threshold` |
| `final_delta` | `f64` | L1-norm delta between the last two iteration vectors |

**Interpretation:** If `converged` is `false`, results may be approximate. Check `final_delta` to gauge how close to convergence the algorithm got. Increasing `max_iterations` in `TextRankConfig` may help. A very small graph (< 5 nodes) often converges in 2–3 iterations.

```json
{
  "convergence_summary": {
    "iterations": 28,
    "converged": true,
    "final_delta": 3.7e-8
  }
}
```

### `stage_timings` (available at `Stats` and above)

Wall-clock duration of each pipeline stage in milliseconds.

```json
{
  "stage_timings": [
    ["preprocess", 0.012],
    ["candidates", 0.089],
    ["graph", 0.234],
    ["graph_transform", 0.001],
    ["teleport", 0.003],
    ["rank", 0.156],
    ["phrases", 0.045],
    ["format", 0.018]
  ]
}
```

**Interpretation:** Use this to identify bottlenecks. For most documents, `graph` and `rank` dominate. If `candidates` is slow, you may have a very large document — consider truncating input. The `graph_transform` stage is `0.0` for non-MultipartiteRank variants (the NoopGraphTransform is compiled away).

### `node_scores` (available at `TopNodes` and above)

Top-K graph nodes sorted by PageRank score descending. Ties are broken by lemma ascending for stability.

```json
{
  "node_scores": [
    ["machine|NOUN", 0.0523],
    ["learning|NOUN", 0.0487],
    ["neural|ADJ", 0.0312],
    ["network|NOUN", 0.0298]
  ]
}
```

**Interpretation:** These are the raw PageRank scores *before* phrase assembly. Comparing node scores helps you understand:

- **Why a phrase ranks high:** Its constituent words all have high individual scores.
- **Why a phrase ranks low:** One of its words has very low score, dragging down the aggregate.
- **How teleport biasing works:** In PositionRank, early words get inflated scores. In BiasedTextRank, focus terms dominate.

Node labels follow the `"lemma|POS"` format when `use_pos_in_nodes` is enabled (the default). With it disabled, labels are just the lemma.

The number of nodes returned is bounded by `top_k`:

| Source | Priority | Default |
|--------|----------|---------|
| `expose.node_scores.top_k` | 1 (highest) | — |
| `runtime.max_debug_top_k` | 2 | — |
| `DebugLevel::DEFAULT_TOP_K` | 3 (fallback) | **50** |

### `residuals` (available at `Full` only)

Per-iteration L1-norm convergence residuals from PageRank.

```json
{
  "residuals": [0.8234, 0.3912, 0.1856, 0.0423, 0.0089, 0.0012, 0.00003]
}
```

**Interpretation:** This shows how fast PageRank is converging. A healthy convergence curve drops roughly exponentially. If the curve plateaus early, the graph may have a structural issue (e.g., disconnected components, very low damping). Only populated when PageRank diagnostics are enabled internally — if absent even at `Full` level, diagnostics were not captured for this run.

### `cluster_memberships` (available at `Full` only, topic-family variants)

Cluster membership arrays for TopicRank and MultipartiteRank. `cluster_memberships[i]` lists the candidate indices belonging to cluster `i`.

```json
{
  "cluster_memberships": [
    [0, 1, 4],
    [2, 3],
    [5]
  ]
}
```

**Interpretation:** This shows how phrase candidates were grouped by the HAC clusterer. Candidates in the same cluster share overlapping words (Jaccard similarity above the threshold). For TopicRank, each cluster becomes a single graph node. For MultipartiteRank, intra-cluster edges are removed to create a k-partite structure.

This field is `null` for word-graph variants (BaseTextRank, PositionRank, etc.) and SentenceRank since they don't use clustering.

---

## The `expose` Spec (JSON Interface)

The `expose` field in a `PipelineSpec` provides fine-grained control over which debug fields to populate. It maps declaratively to `DebugLevel`:

```
┌─────────────────────────┬──────────────────────┐
│ expose field            │ Minimum DebugLevel   │
├─────────────────────────┼──────────────────────┤
│ graph_stats: true       │ Stats                │
│ stage_timings: true     │ Stats                │
│ pagerank: {}            │ Stats                │
│ node_scores: {}         │ TopNodes             │
│ node_scores: {top_k: N} │ TopNodes             │
│ pagerank: {residuals: T}│ Full                 │
│ clusters: true          │ Full                 │
└─────────────────────────┴──────────────────────┘
```

When multiple fields are requested, the **highest** required level wins. For example, `{ "graph_stats": true, "clusters": true }` resolves to `Full`.

### Full Expose Example

```json
{
  "v": 1,
  "preset": "multipartite_rank",
  "expose": {
    "node_scores": { "top_k": 50 },
    "graph_stats": true,
    "pagerank": { "residuals": true },
    "clusters": true,
    "stage_timings": true
  },
  "runtime": {
    "max_debug_top_k": 100
  }
}
```

This requests everything at `Full` level: graph stats, convergence summary, stage timings, top-50 node scores, per-iteration residuals, and cluster memberships.

### Minimal Expose (Stats Only)

```json
{
  "v": 1,
  "preset": "textrank",
  "expose": {
    "graph_stats": true,
    "stage_timings": true
  }
}
```

### Empty Expose (Disabled)

An empty `expose: {}` enables nothing — each sub-field must be explicitly set:

```json
{
  "v": 1,
  "expose": {}
}
```

---

## Pipeline Observer (Rust API)

For Rust users, the `PipelineObserver` trait provides real-time callbacks at every stage boundary — more powerful than the static `DebugPayload`.

### Available Callbacks

| Method | Called After | Receives |
|--------|------------|----------|
| `on_stage_start(stage)` | — | Stage name (before execution) |
| `on_stage_end(stage, report)` | Every stage | `StageReport` with duration + optional metrics |
| `on_tokens(tokens)` | Preprocess | Full `TokenStream` |
| `on_candidates(candidates)` | Candidate selection | `CandidateSet` |
| `on_graph(graph)` | Graph build + transform | `Graph` (CSR, with cluster assignments if topic-family) |
| `on_rank(rank)` | Ranking | `RankOutput` (scores + convergence) |
| `on_phrases(phrases)` | Phrase building | `PhraseSet` |

### Stage Names

Stage names are `&'static str` constants:

| Constant | Value |
|----------|-------|
| `STAGE_PREPROCESS` | `"preprocess"` |
| `STAGE_CANDIDATES` | `"candidates"` |
| `STAGE_GRAPH` | `"graph"` |
| `STAGE_GRAPH_TRANSFORM` | `"graph_transform"` |
| `STAGE_TELEPORT` | `"teleport"` |
| `STAGE_RANK` | `"rank"` |
| `STAGE_PHRASES` | `"phrases"` |
| `STAGE_FORMAT` | `"format"` |

### Built-in Observers

| Observer | Purpose |
|----------|---------|
| `NoopObserver` | Default. Zero-sized, compiled away entirely. |
| `StageTimingObserver` | Collects `(stage_name, StageReport)` pairs. Call `.reports()` after the run to inspect timings. Call `.total_duration_ms()` for aggregate wall time. |

### Custom Observer Example

```rust
use rapid_textrank::pipeline::observer::*;

struct MyObserver;

impl PipelineObserver for MyObserver {
    fn on_stage_end(&mut self, stage: &'static str, report: &StageReport) {
        println!("{stage}: {:.3}ms", report.duration_ms());
        if let Some(nodes) = report.nodes() {
            println!("  graph: {nodes} nodes, {} edges", report.edges().unwrap_or(0));
        }
        if let Some(iters) = report.iterations() {
            println!("  PageRank: {iters} iterations, converged={}",
                     report.converged().unwrap_or(false));
        }
    }
}
```

### StageReport Fields

Each `StageReport` always has `duration_us`. Other fields depend on which stage produced it:

| Field | Type | Populated By |
|-------|------|-------------|
| `duration_us()` | `u64` | All stages |
| `duration_ms()` | `f64` | All stages |
| `nodes()` | `Option<usize>` | GraphBuilder |
| `edges()` | `Option<usize>` | GraphBuilder |
| `iterations()` | `Option<u32>` | Ranker |
| `converged()` | `Option<bool>` | Ranker |
| `residual()` | `Option<f64>` | Ranker |

---

## Troubleshooting with Debug Output

### "Why is keyword X missing?"

1. Set `debug_level: "top_nodes"` and check `node_scores`. If the word appears with a low score, it was considered but ranked below the `top_n` cutoff.
2. If the word doesn't appear in `node_scores` at all, it was likely filtered out during candidate selection (wrong POS tag, or in the stopword list).

### "Why did PageRank not converge?"

1. Check `convergence_summary.iterations` — did it hit `max_iterations`?
2. Check `convergence_summary.final_delta` — how far from the threshold is it?
3. At `Full` level, inspect `residuals` to see the convergence curve. If it plateaus, try increasing `damping` (default: 0.85).

### "Why are two similar phrases both in the output?"

1. Check `cluster_memberships` (TopicRank/MultipartiteRank) — if they're in different clusters, the similarity threshold wasn't met.
2. For word-graph variants, similar phrases with different lemmas won't be grouped. Consider TopicRank if you need deduplication.

### "Which stage is slow?"

1. Set `expose.stage_timings: true` (or use `StageTimingObserver` in Rust).
2. Look for the dominant stage. Common patterns:
   - **`graph` dominates**: Large document, many candidates, dense co-occurrence.
   - **`rank` dominates**: Large graph, slow convergence. Try reducing `max_iterations` or increasing `convergence_threshold`.
   - **`candidates` dominates**: Very long document. Consider truncating input.
