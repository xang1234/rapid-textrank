# API Reference

rapid_textrank provides three API layers, each suited to different use cases. Pick the one that matches your workflow.

## 1. Convenience Functions

The convenience functions in [`extract-keywords()`](extract-keywords.md) are the simplest way to extract keywords. `extract_keywords()` keeps the stable BaseTextRank one-liner, while `extract_keywords_auto()` gives you the AutoRank ensemble in one call.

```python
from rapid_textrank import extract_keywords_auto
phrases = extract_keywords_auto("Your text here.", top_n=10, language="en")
```

## 2. Extractor Classes

The [extractor classes](extractor-classes.md) give you more control over the algorithm variant and configuration. Create a reusable instance with a [`TextRankConfig`](textrank-config.md), then call `extract_keywords()` on any number of documents. Native Python objects are available for BaseTextRank, PositionRank, BiasedTextRank, SingleRank, TopicalPageRank, AutoRank, MultipartiteRank, and SentenceRank.

```python
from rapid_textrank import PositionRank
extractor = PositionRank(top_n=10, language="en")
result = extractor.extract_keywords(text)
```

## 3. JSON Interface

The [JSON interface](json-interface.md) accepts pre-tokenized input as a JSON string and returns results as JSON. This is the right choice when you are tokenizing with spaCy (or another NLP pipeline) and want to pass tokens directly to the Rust core, or when you need batch processing. It is also the only way to use TopicRank.

```python
from rapid_textrank import extract_from_json
result_json = extract_from_json(json_string)
```

## Supporting Pages

- [TextRankConfig](textrank-config.md) -- full parameter reference for fine-tuning the algorithm.
- [Result Objects](result-objects.md) -- attributes of `TextRankResult` and `Phrase`.
- [spaCy Integration](spacy-integration.md) -- drop-in pipeline component for spaCy.
- [Topic Utilities](topic-utilities.md) -- computing topic weights from LDA for TopicalPageRank.
- [Supported Languages](supported-languages.md) -- the 18 languages available for stopword filtering.
