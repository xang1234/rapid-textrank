# Extractor Classes

The class-based API gives you reusable extractor instances with full configuration control. AutoRank is the recommended default when you do not want to choose a single algorithm manually. TopicRank is available only through the [JSON interface](json-interface.md).

## Class Summary

| Class | Extra Constructor Params | Extra Methods |
|---|---|---|
| `BaseTextRank` | -- | `extract_keywords(text)` |
| `PositionRank` | -- | `extract_keywords(text)` |
| `BiasedTextRank` | `focus_terms`, `bias_weight=5.0` | `set_focus(terms)`, `extract_keywords(text, focus_terms=None)` |
| `SingleRank` | -- | `extract_keywords(text)` |
| `TopicalPageRank` | `topic_weights`, `min_weight=0.0` | `set_topic_weights(w)`, `extract_keywords(text, topic_weights=None)` |
| `AutoRank` | `focus_terms`, `bias_weight=5.0`, `semantic_weights`, `semantic_min_weight=0.0` | `set_focus(terms)`, `set_semantic_weights(w)`, `extract_keywords(text)` |
| `MultipartiteRank` | `similarity_threshold=0.26`, `alpha=1.1` | `extract_keywords(text)` |

All classes share these constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `TextRankConfig` | `None` | Full configuration object (see [TextRankConfig](textrank-config.md)). Overrides all defaults. |
| `top_n` | `int` | `None` | Number of results. Overrides `config.top_n` if both are provided. |
| `language` | `str` | `None` | Language for stopword filtering. Overrides `config.language` if both are provided. |

All classes return a [`TextRankResult`](result-objects.md#textrankresult) from `extract_keywords()`.

## BaseTextRank

The standard TextRank implementation. A good starting point for general-purpose keyword extraction.

```python
from rapid_textrank import BaseTextRank

extractor = BaseTextRank(top_n=10, language="en")
result = extractor.extract_keywords(text)

for phrase in result.phrases:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

## PositionRank

Weights words by their position in the document -- earlier appearances score higher. Useful for academic papers, news articles, and executive summaries where key information appears early.

```python
from rapid_textrank import PositionRank

extractor = PositionRank(top_n=10, language="en")
result = extractor.extract_keywords("""
Quantum Computing Advances in 2024

Researchers have made significant breakthroughs in quantum error correction.
The quantum computing field continues to evolve rapidly...
""")

# "quantum computing" ranks higher due to early position
for phrase in result.phrases:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

## BiasedTextRank

Steers extraction toward specific topics using focus terms. The `bias_weight` parameter controls how strongly results favor the focus terms.

### Constructor

```python
BiasedTextRank(
    focus_terms=None,       # list[str] -- terms to bias toward
    bias_weight=5.0,        # float -- higher = stronger bias
    config=None,
    top_n=None,
    language=None,
)
```

### Usage

```python
from rapid_textrank import BiasedTextRank

extractor = BiasedTextRank(
    focus_terms=["security", "privacy"],
    bias_weight=5.0,
    top_n=10,
    language="en",
)

result = extractor.extract_keywords("""
Modern web applications must balance user experience with security.
Privacy regulations require careful data handling. Performance
optimizations should not compromise security measures.
""")

# Results will favor security/privacy-related phrases
for phrase in result.phrases:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

### Updating focus terms

You can update focus terms on an existing extractor in two ways:

```python
# Option 1: set_focus() method
extractor.set_focus(["encryption", "audit"])
result = extractor.extract_keywords(text)

# Option 2: pass focus_terms per call
result = extractor.extract_keywords(text, focus_terms=["neural", "network"])
```

## SingleRank

Extends TextRank with weighted co-occurrence edges and cross-sentence windowing. Edges are weighted by co-occurrence count, and the sliding window ignores sentence boundaries.

```python
from rapid_textrank import SingleRank

extractor = SingleRank(top_n=10, language="en")
result = extractor.extract_keywords("""
Machine learning is a powerful tool. Deep learning is a subset of
machine learning. Neural networks power deep learning systems.
""")

# Cross-sentence co-occurrences strengthen "machine learning" edges
for phrase in result.phrases:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

## TopicalPageRank

Extends SingleRank by biasing the random walk toward topically important words. Supply per-word topic weights as a `{lemma: weight}` dictionary -- typically from a topic model (LDA, TF-IDF, etc.), but any source of word importance scores works.

### Constructor

```python
TopicalPageRank(
    topic_weights=None,     # dict[str, float] -- per-lemma importance weights
    min_weight=0.0,         # float -- floor for out-of-vocabulary words
    config=None,
    top_n=None,
    language=None,
)
```

### Usage

```python
from rapid_textrank import TopicalPageRank

topic_weights = {
    "neural": 0.9,
    "network": 0.8,
    "learning": 0.7,
    "deep": 0.6,
}

extractor = TopicalPageRank(
    topic_weights=topic_weights,
    min_weight=0.01,
    top_n=10,
    language="en",
)

result = extractor.extract_keywords("""
Deep learning is a subset of machine learning that uses artificial neural
networks. Neural networks with many layers can learn complex patterns.
Convolutional neural networks excel at image recognition tasks.
""")

for phrase in result.phrases:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

### Updating topic weights

```python
# Option 1: set_topic_weights() method
extractor.set_topic_weights({"machine": 0.9, "data": 0.8})
result = extractor.extract_keywords(text)

# Option 2: pass topic_weights per call
result = extractor.extract_keywords(text, topic_weights={"machine": 0.9, "data": 0.8})
```

See [Topic Utilities](topic-utilities.md) for computing topic weights from LDA.

## AutoRank

Runs the full eligible keyword ensemble for the document and fuses the results into one ranked list. Use it when you want a strong default without manual variant selection.

### Constructor

```python
AutoRank(
    focus_terms=None,
    bias_weight=5.0,
    semantic_weights=None,
    semantic_min_weight=0.0,
    config=None,
    top_n=None,
    language=None,
)
```

### Usage

```python
from rapid_textrank import AutoRank

extractor = AutoRank(
    top_n=10,
    semantic_weights={"machine": 1.0, "learning": 0.8},
)
result = extractor.extract_keywords(text)

for phrase, support in zip(result.phrases, result.consensus.phrase_support):
    print(phrase.text, support.confidence, support.supporting_variants)
```

## MultipartiteRank

Builds a k-partite directed graph where candidates from different topic clusters are connected. Intra-topic edges are removed to reduce competition between similar candidates. An `alpha` weight adjustment boosts the first-occurring variant in each topic cluster, encoding positional preference.

### Constructor

```python
MultipartiteRank(
    similarity_threshold=0.26,  # float -- Jaccard threshold for topic clustering
    alpha=1.1,                  # float -- position boost strength (0 = disabled)
    config=None,
    top_n=None,
    language=None,
)
```

### Usage

```python
from rapid_textrank import MultipartiteRank

extractor = MultipartiteRank(
    similarity_threshold=0.26,
    alpha=1.1,
    top_n=10,
    language="en",
)

result = extractor.extract_keywords("""
Machine learning is a powerful tool for data analysis. Deep learning
is a subset of machine learning. Neural networks power deep learning
systems. Convolutional neural networks excel at image recognition.
""")

for phrase in result.phrases:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

## TopicRank (JSON only)

TopicRank clusters similar candidate phrases into topics, ranks the topics, then selects representatives. It is available only through the [JSON interface](json-interface.md) with `variant="topic_rank"`. This is because TopicRank works best with external tokenization (e.g., spaCy), which provides accurate POS tags and lemmatization for the clustering step.

## Using TextRankConfig with Extractor Classes

Any extractor class accepts a `TextRankConfig` for full control:

```python
from rapid_textrank import BaseTextRank, TextRankConfig

config = TextRankConfig(
    damping=0.85,
    window_size=4,
    top_n=15,
    min_phrase_length=2,
    max_phrase_length=4,
    include_pos=["NOUN", "ADJ", "PROPN"],
    phrase_grouping="scrubbed_text",
    language="en",
)

extractor = BaseTextRank(config=config)
result = extractor.extract_keywords(text)
```

The `top_n` and `language` shortcut parameters override the corresponding values in `config` if both are provided. See [TextRankConfig](textrank-config.md) for the full parameter reference.
