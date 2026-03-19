# Result Objects

All extractor classes return a `TextRankResult` containing a list of `Phrase` objects. This page documents both.

## TextRankResult

Returned by every `extract_keywords()` call on an extractor class.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `phrases` | `list[Phrase]` | List of extracted phrases, sorted by score in descending order. |
| `converged` | `bool` | Whether the PageRank iteration converged within `max_iterations`. |
| `iterations` | `int` | Number of PageRank iterations actually run. |
| `consensus` | `ConsensusPayload \| None` | AutoRank-only metadata about the executed variant pool and per-phrase agreement. |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `as_tuples()` | `list[tuple[str, float]]` | Returns phrases as `[(text, score), ...]` tuples. Useful for quick inspection or serialization. |
| `__len__()` | `int` | Number of phrases. Supports `len(result)`. |
| `__getitem__(idx)` | `Phrase` | Index into the phrase list. Supports `result[0]`. |

### Example

```python
from rapid_textrank import BaseTextRank

extractor = BaseTextRank(top_n=5, language="en")
result = extractor.extract_keywords("Machine learning is a subset of artificial intelligence.")

# Check convergence
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")

# Iterate over phrases
for phrase in result.phrases:
    print(f"{phrase.text}: {phrase.score:.4f}")

# Quick tuple output
print(result.as_tuples())
# [('machine learning', 0.2341), ('artificial intelligence', 0.1654), ...]

# Length and indexing
print(len(result))     # 5
print(result[0].text)  # 'machine learning'
```

## Phrase

Each item in `TextRankResult.phrases` is a `Phrase` object.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The phrase text as it appears in the document (e.g., `"machine learning"`). |
| `lemma` | `str` | Lemmatized form of the phrase (e.g., `"machine learning"`). Useful for deduplication. |
| `score` | `float` | TextRank score. Higher is more important. Scores are not normalized to any fixed range. |
| `count` | `int` | Number of times this phrase (or its variants) appears in the text. |
| `rank` | `int` | 1-indexed rank position. The top-scoring phrase has `rank=1`. |

### Example

```python
result = extractor.extract_keywords(text)

for phrase in result.phrases:
    print(f"Text:  {phrase.text}")
    print(f"Lemma: {phrase.lemma}")
    print(f"Score: {phrase.score:.4f}")
    print(f"Count: {phrase.count}")
    print(f"Rank:  {phrase.rank}")
    print()
```

### String representation

`Phrase` supports both `repr()` and `str()`:

```python
phrase = result.phrases[0]

repr(phrase)  # "Phrase(text='machine learning', score=0.2341, rank=1)"
str(phrase)   # "machine learning"
```

## ConsensusPayload

Present only for `AutoRank` results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `selected_variants` | `list[str]` | Canonical names of the executed member variants. |
| `selection_reason` | `str` | Human-readable summary of why the pool included focus-driven, semantic-driven, or pre-tokenized members. |
| `variant_runs` | `list[VariantRun]` | Per-variant convergence summaries. |
| `phrase_support` | `list[PhraseSupport]` | Per-phrase agreement metadata aligned with `TextRankResult.phrases`. |

## VariantRun

| Attribute | Type | Description |
|-----------|------|-------------|
| `variant` | `str` | Canonical variant name. |
| `converged` | `bool` | Whether that member extractor converged. |
| `iterations` | `int` | Iterations used by that member extractor. |

## PhraseSupport

| Attribute | Type | Description |
|-----------|------|-------------|
| `confidence` | `float` | Agreement score in `[0, 1]`, based on the supporting variant weights. |
| `support_count` | `int` | Number of member variants that supported the phrase. |
| `supporting_variants` | `list[str]` | Canonical names of the supporting variants. |

## JSON Result Format

When using the [JSON interface](json-interface.md), results are returned as a JSON string with the same structure:

```json
{
    "phrases": [
        {
            "text": "machine learning",
            "lemma": "machine learning",
            "score": 0.2341,
            "count": 2,
            "rank": 1
        }
    ],
    "converged": true,
    "iterations": 12
}
```
