# TextRankConfig

`TextRankConfig` controls every tunable aspect of the TextRank algorithm. Pass it to any extractor class via the `config` parameter.

## Parameter Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `damping` | `float` | `0.85` | PageRank damping factor (0-1). Higher values give more weight to graph structure vs. uniform distribution. |
| `max_iterations` | `int` | `100` | Maximum number of PageRank iterations. |
| `convergence_threshold` | `float` | `1e-6` | PageRank convergence threshold. Iteration stops when the score change between iterations falls below this value. |
| `window_size` | `int` | `3` | Co-occurrence window size. Two words are connected in the graph if they appear within this many words of each other. |
| `top_n` | `int` | `10` | Number of top-scoring phrases to return. Set `0` to return all phrases. |
| `min_phrase_length` | `int` | `1` | Minimum number of words in a phrase. Set to `2` to exclude single-word results. |
| `max_phrase_length` | `int` | `4` | Maximum number of words in a phrase. |
| `score_aggregation` | `str` | `"sum"` | How to combine individual word scores into a phrase score. Options: `"sum"`, `"mean"`, `"max"`, `"rms"` (root mean square). |
| `language` | `str` | `"en"` | Language code for built-in stopword filtering. See [Supported Languages](supported-languages.md). |
| `use_edge_weights` | `bool` | `True` | Whether to use weighted edges in the co-occurrence graph. When `False`, all edges have weight 1. |
| `include_pos` | `list[str]` | `["NOUN","ADJ","PROPN","VERB"]` | POS tags to include in the graph. Only words with these POS tags become graph nodes. |
| `stopwords` | `list[str]` | `[]` | Additional stopwords that extend the built-in list for the selected language. |
| `use_pos_in_nodes` | `bool` | `True` | If `True`, graph nodes are keyed by `"lemma|POS"` (e.g., `"learning|NOUN"`). If `False`, nodes are keyed by lemma only. |
| `phrase_grouping` | `str` | `"scrubbed_text"` | How to group phrase variants. `"scrubbed_text"` groups by lowercased surface form. `"lemma"` groups by lemmatized form. |

## Full Example

```python
from rapid_textrank import TextRankConfig, BaseTextRank

config = TextRankConfig(
    damping=0.85,
    max_iterations=100,
    convergence_threshold=1e-6,
    window_size=3,
    top_n=10,
    min_phrase_length=1,
    max_phrase_length=4,
    score_aggregation="sum",
    language="en",
    use_edge_weights=True,
    include_pos=["NOUN", "ADJ", "PROPN", "VERB"],
    use_pos_in_nodes=True,
    phrase_grouping="scrubbed_text",
    stopwords=["custom", "terms"],
)

extractor = BaseTextRank(config=config)
result = extractor.extract_keywords(text)
```

## Common Tuning Patterns

### SEO-style multi-word phrases

Force 2-4 word phrases, noun-heavy, with scrubbed-text grouping:

```python
config = TextRankConfig(
    min_phrase_length=2,
    max_phrase_length=4,
    include_pos=["NOUN", "ADJ", "PROPN"],
    phrase_grouping="scrubbed_text",
)
```

### Larger co-occurrence window

A wider window captures longer-range relationships:

```python
config = TextRankConfig(window_size=6)
```

### Stricter convergence

More iterations with a tighter threshold can improve score stability on long documents:

```python
config = TextRankConfig(
    max_iterations=200,
    convergence_threshold=1e-8,
)
```

### Adding domain-specific stopwords

Extend the built-in stopword list with terms that are too common in your domain:

```python
config = TextRankConfig(
    language="en",
    stopwords=["data", "system", "model", "2024"],
)
```

## Notes

- The `include_pos` parameter expects Universal POS tags as strings (the same tags spaCy uses): `"NOUN"`, `"VERB"`, `"ADJ"`, `"ADV"`, `"PROPN"`, etc.
- The `stopwords` parameter extends the built-in list -- it does not replace it. To use only your custom stopwords without built-in ones, you would need to use the JSON interface with `is_stopword` flags on individual tokens.
- `TextRankConfig` is validated on construction. Invalid combinations (e.g., negative damping) raise a `ValueError`.
