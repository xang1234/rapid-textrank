# Comparison with Alternatives

This page compares rapid_textrank with other popular keyword and keyphrase extraction libraries to help you choose the right tool for your use case.

## Feature Comparison

| Feature | rapid_textrank | pytextrank | YAKE | KeyBERT | pke | Rake-NLTK |
|---|---|---|---|---|---|---|
| Speed | Very fast (Rust) | Moderate (Python) | Fast (Python) | Slow (transformer) | Moderate | Fast |
| Algorithm variants | 7 | 3 (TextRank, BiasedTR, TopicRank) | 1 (YAKE) | 1 (BERT-based) | 5+ (TextRank, TFIDF, etc.) | 1 (RAKE) |
| Language support | 18 languages | Via spaCy | 25+ | Any (via embeddings) | Via spaCy | English-focused |
| spaCy dependency | Optional | Required | None | None | Required | None |
| Pre-tokenized input | Yes (JSON API) | Via spaCy | No | No | Via spaCy | No |
| API style | Classes + JSON | spaCy pipeline | Function | Class | Classes | Class |

## When to Use Each Tool

### rapid_textrank

Best for **speed-critical pipelines, batch processing, multi-variant exploration, and a smart default ensemble**.

Choose rapid_textrank when latency matters -- real-time APIs, high-throughput batch jobs, or interactive applications where users expect instant results. The seven core algorithm variants (BaseTextRank, PositionRank, BiasedTextRank, TopicRank, SingleRank, TopicalPageRank, MultipartiteRank) let you experiment with different ranking strategies without switching libraries, and `AutoRank` gives you a high-quality default when you do not want to choose manually. The JSON API is ideal for pipelines that already tokenize with spaCy or another NLP tool.

### pytextrank

Best when you are **already in a spaCy pipeline and need spaCy's NER/dependency parsing**.

If your application already loads a spaCy model for named entity recognition, dependency parsing, or other linguistic features, pytextrank integrates as a native pipeline component. It leverages spaCy's tokenization and POS tagging directly. The tradeoff is speed -- pytextrank is pure Python and includes the full spaCy pipeline overhead.

### YAKE

Best for **lightweight extraction without graph computation**.

YAKE (Yet Another Keyword Extractor) uses statistical features (word frequency, position, co-occurrence) without building a graph. It is fast, unsupervised, and language-independent. Choose YAKE when you need a simple, dependency-free solution and do not need graph-based ranking or multiple algorithm variants.

### KeyBERT

Best when **semantic understanding matters more than speed**.

KeyBERT uses transformer embeddings (BERT, RoBERTa, etc.) to find keywords that are semantically similar to the document. It captures meaning beyond surface-level co-occurrence, making it strong for documents where important concepts are expressed with varied vocabulary. The tradeoff is speed -- transformer inference is orders of magnitude slower than graph-based methods.

### pke

Best for **academic research and access to many classical keyphrase methods**.

pke (Python Keyphrase Extraction) implements a wide range of keyphrase extraction algorithms (TextRank, SingleRank, TopicRank, TFIDF, KP-Miner, and more) in a unified framework. It is designed for reproducible research and benchmarking across methods. Choose pke when you need to compare many algorithms or need methods not available elsewhere.

### Rake-NLTK

Best for a **quick RAKE implementation with minimal dependencies**.

Rake-NLTK implements the Rapid Automatic Keyword Extraction (RAKE) algorithm, which uses word co-occurrence within phrases delimited by stopwords and punctuation. It is simple, fast, and easy to understand. Choose it for quick prototyping or when you specifically want the RAKE algorithm.

## Summary

If speed is your primary concern and you want flexibility across algorithm variants, rapid_textrank is the strongest choice. If you want the library to make the variant choice for you, use `AutoRank`. If you need deeper semantic understanding than graph methods can provide on their own, look at KeyBERT. If you need minimal dependencies and a simple statistical approach, consider YAKE or Rake-NLTK. If you are building on spaCy, pytextrank or pke integrate naturally into that ecosystem.
