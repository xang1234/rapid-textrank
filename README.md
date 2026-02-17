# rapid_textrank

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

**High-performance TextRank keyword extraction in Rust with Python bindings.**

Extract keywords and key phrases from text up to 10-100x faster than pure Python implementations, with 7 algorithm variants and stopword support for 18 languages. Computation runs in Rust; the Python GIL is released during extraction.

## Install

```bash
pip install rapid_textrank
```

Optional extras: `pip install rapid_textrank[spacy]` for spaCy tokenization, `pip install rapid_textrank[topic]` for gensim LDA utilities.

## Quick Start

```python
from rapid_textrank import extract_keywords

text = """
Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience. Deep learning, a type of
machine learning, uses neural networks with many layers.
"""

keywords = extract_keywords(text, top_n=5, language="en")
for phrase in keywords:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

Output:

```
machine learning: 0.2341
deep learning: 0.1872
artificial intelligence: 0.1654
neural networks: 0.1432
systems: 0.0891
```

## Performance

Extraction latency (single document, end-to-end including tokenization):

| Document size | rapid_textrank | pytextrank + spaCy | Speedup |
|---|---|---|---|
| ~20 words | ~0.1 ms | ~5 ms | ~50x |
| ~100 words | ~0.3 ms | ~15 ms | ~50x |
| ~1,000 words | ~2 ms | ~80 ms | ~40x |

See [benchmarks](https://xang1234.github.io/rapid-textrank/performance/benchmarks/) for methodology and full results.

## Steering Extraction with BiasedTextRank

Use focus terms to pull results toward a specific domain. Here we steer toward security/privacy phrases in a mixed document:

```python
from rapid_textrank import BiasedTextRank

text = """
We encrypt data at rest using AES-256 and enforce TLS 1.2+ for data in transit.
Access to production is gated by MFA and short-lived credentials. Audit logs are
retained for 180 days and monitored for anomalous access patterns. Personal data
processing is limited to the declared purpose, and retention follows a documented
schedule. We support DSAR workflows and apply data minimization by default.
"""

extractor = BiasedTextRank(
    focus_terms=["privacy", "encrypt", "tls", "mfa", "audit", "retention"],
    bias_weight=8.0,
    top_n=10,
    language="en",
)

result = extractor.extract_keywords(text)
for phrase in result.phrases[:5]:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

You can override focus terms per call without creating a new extractor:

```python
result = extractor.extract_keywords(text, focus_terms=["privacy", "retention", "dsar"])
```

## Choosing an Algorithm

| Algorithm | Best for | Key idea |
|---|---|---|
| **BaseTextRank** | General-purpose keyword extraction | Standard co-occurrence graph + PageRank |
| **PositionRank** | Short/structured text (abstracts, news) | Biases toward early-occurring terms |
| **BiasedTextRank** | Domain-focused extraction | Steers PageRank toward user-supplied focus terms |
| **SingleRank** | Long technical documents | Weighted edges + cross-sentence co-occurrence window |
| **TopicRank** | Multi-topic documents needing diversity | Clusters candidates into topics, ranks topics |
| **TopicalPageRank** | LDA-guided extraction | Personalization vector from per-word topic weights |
| **MultipartiteRank** | Fine-grained topic diversity | Multipartite graph separates candidates by topic cluster |

Start with **BaseTextRank** if unsure. See the [algorithm guide](https://xang1234.github.io/rapid-textrank/algorithms/choosing-a-variant/) for a decision flowchart.

## Learn More

| Topic | Link |
|---|---|
| Getting Started | [Installation, quickstart, recipes](https://xang1234.github.io/rapid-textrank/getting-started/) |
| Algorithms | [How TextRank works + all 7 variants](https://xang1234.github.io/rapid-textrank/algorithms/) |
| API Reference | [extract_keywords(), extractor classes, JSON interface](https://xang1234.github.io/rapid-textrank/api/) |
| Performance | [Benchmarks and comparison with alternatives](https://xang1234.github.io/rapid-textrank/performance/) |
| Development | [Contributing guide](https://xang1234.github.io/rapid-textrank/development/contributing/) |

## See Also

- [pytextrank](https://github.com/DerwenAI/pytextrank/) — Python TextRank implementation built on spaCy
- [KeyBERT](https://github.com/MaartenGr/KeyBERT) — keyword extraction using BERT embeddings

## License

MIT
