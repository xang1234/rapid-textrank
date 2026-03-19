# JSON Interface

The JSON interface accepts pre-tokenized input as JSON strings and returns results as JSON strings. This minimizes Python-to-Rust overhead when you already have tokenized data (e.g., from spaCy) and enables batch processing. It is also the only way to use TopicRank.

## Functions

### extract_from_json

Process a single document.

```python
from rapid_textrank import extract_from_json
import json

result_json = extract_from_json(json_str)
result = json.loads(result_json)
```

**Signature:** `extract_from_json(json_input: str) -> str`

- **json_input** -- a JSON string containing a single `JsonDocument` object.
- **Returns** -- a JSON string containing the extraction result (phrases, converged, iterations).

### extract_batch_from_json

Process multiple documents in a single call. Documents are processed sequentially in the Rust core.

```python
from rapid_textrank import extract_batch_from_json
import json

results_json = extract_batch_from_json(json_str)
results = json.loads(results_json)  # list of result objects
```

**Signature:** `extract_batch_from_json(json_input: str) -> str`

- **json_input** -- a JSON string containing an array of `JsonDocument` objects.
- **Returns** -- a JSON string containing an array of result objects.

## Input Schema

### JsonDocument

The top-level object for a single document:

```json
{
    "tokens": [ ... ],
    "variant": "textrank",
    "config": { ... }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tokens` | `array[JsonToken]` | Yes | Array of pre-tokenized tokens. |
| `variant` | `string` | No | Algorithm variant (default: `"textrank"`). See variant table below. |
| `config` | `object` | No | Configuration parameters. Accepts all [TextRankConfig](textrank-config.md) fields plus variant-specific fields. |

### JsonToken

Each token in the `tokens` array:

```json
{
    "text": "Machine",
    "lemma": "machine",
    "pos": "NOUN",
    "start": 0,
    "end": 7,
    "sentence_idx": 0,
    "token_idx": 0,
    "is_stopword": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Surface form of the token. |
| `lemma` | `string` | Lemmatized form. |
| `pos` | `string` | Universal POS tag (e.g., `"NOUN"`, `"VERB"`, `"ADJ"`). |
| `start` | `int` | Character start offset in the original text. |
| `end` | `int` | Character end offset in the original text. |
| `sentence_idx` | `int` | 0-based sentence index. |
| `token_idx` | `int` | 0-based token index within the document. |
| `is_stopword` | `bool` | Whether this token is a stopword. Defaults to `false` if omitted. |

## Variant Strings

| Variant | Accepted String Values |
|---|---|
| BaseTextRank | `"textrank"` (default), `"text_rank"`, `"base"` |
| PositionRank | `"position_rank"`, `"positionrank"`, `"position"` |
| BiasedTextRank | `"biased_textrank"`, `"biased"`, `"biasedtextrank"` |
| TopicRank | `"topic_rank"`, `"topicrank"`, `"topic"` |
| SingleRank | `"single_rank"`, `"singlerank"`, `"single"` |
| TopicalPageRank | `"topical_pagerank"`, `"topicalpagerank"`, `"tpr"`, `"single_tpr"` |
| MultipartiteRank | `"multipartite_rank"`, `"multipartiterank"`, `"multipartite"`, `"mpr"` |
| AutoRank | `"auto_rank"`, `"autorank"`, `"auto"` |

## Variant-Specific Config Fields

In addition to the standard [TextRankConfig](textrank-config.md) fields, each variant accepts additional config parameters:

### biased_textrank

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `focus_terms` | `list[str]` | `[]` | Terms to bias extraction toward. |
| `bias_weight` | `float` | `5.0` | Strength of the bias toward focus terms. |

### topic_rank

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `topic_similarity_threshold` | `float` | `0.25` | Similarity threshold for topic clustering. Higher values produce fewer, larger topics. |
| `topic_edge_weight` | `float` | `1.0` | Weight for edges between topic nodes. |

### topical_pagerank

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `topic_weights` | `dict[str, float]` | `{}` | Per-lemma importance weights (e.g., from LDA). |
| `topic_min_weight` | `float` | `0.0` | Floor weight for words not in `topic_weights`. |

### multipartite_rank

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `multipartite_alpha` | `float` | `1.1` | Position boost strength. Set to `0` to disable. |
| `multipartite_similarity_threshold` | `float` | `0.26` | Jaccard threshold for topic clustering. |

### auto_rank

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `focus_terms` | `list[str]` | `[]` | Optional focus vocabulary enabling BiasedTextRank inside AutoRank. |
| `bias_weight` | `float` | `5.0` | Bias strength for the focus-driven member extractor. |
| `semantic_weights` | `dict[str, float]` | `{}` | Optional lemma weights enabling semantic priors and TopicalPageRank inside AutoRank. |
| `semantic_min_weight` | `float` | `0.0` | Fallback weight for missing lemmas in the AutoRank semantic prior. |
| `topic_weights` | `dict[str, float]` | `{}` | Backward-compatible alias for `semantic_weights` when `variant="auto_rank"`. |
| `topic_min_weight` | `float` | `0.0` | Backward-compatible alias for `semantic_min_weight` when `variant="auto_rank"`. |

## Single Document Example

```python
import json
from rapid_textrank import extract_from_json

doc = {
    "tokens": [
        {
            "text": "Machine",
            "lemma": "machine",
            "pos": "NOUN",
            "start": 0,
            "end": 7,
            "sentence_idx": 0,
            "token_idx": 0,
            "is_stopword": False,
        },
        {
            "text": "learning",
            "lemma": "learning",
            "pos": "NOUN",
            "start": 8,
            "end": 16,
            "sentence_idx": 0,
            "token_idx": 1,
            "is_stopword": False,
        },
        # ... more tokens
    ],
    "variant": "textrank",
    "config": {
        "top_n": 10,
        "language": "en",
        "stopwords": ["nlp", "transformers"],
    },
}

result_json = extract_from_json(json.dumps(doc))
result = json.loads(result_json)

for phrase in result["phrases"]:
    print(f"{phrase['text']}: {phrase['score']:.4f}")
```

## Batch Processing Example

```python
import json
from rapid_textrank import extract_batch_from_json

docs = [
    {
        "tokens": tokens_doc1,
        "variant": "textrank",
        "config": {"top_n": 5},
    },
    {
        "tokens": tokens_doc2,
        "variant": "position_rank",
        "config": {"top_n": 10},
    },
    {
        "tokens": tokens_doc3,
        "variant": "biased_textrank",
        "config": {
            "top_n": 10,
            "focus_terms": ["security", "privacy"],
            "bias_weight": 5.0,
        },
    },
]

results_json = extract_batch_from_json(json.dumps(docs))
results = json.loads(results_json)

for i, result in enumerate(results):
    print(f"Document {i}: {len(result['phrases'])} phrases")
    for phrase in result["phrases"]:
        print(f"  {phrase['text']}: {phrase['score']:.4f}")
```

## TopicRank via JSON

TopicRank is only available through the JSON interface. This example uses spaCy for tokenization:

```python
import json
import spacy
from rapid_textrank import extract_from_json

nlp = spacy.load("en_core_web_sm")
doc = nlp("Your text here...")

tokens = []
for sent_idx, sent in enumerate(doc.sents):
    for token in sent:
        tokens.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "start": token.idx,
            "end": token.idx + len(token.text),
            "sentence_idx": sent_idx,
            "token_idx": token.i,
            "is_stopword": token.is_stop,
        })

payload = {
    "tokens": tokens,
    "variant": "topic_rank",
    "config": {
        "top_n": 10,
        "language": "en",
        "topic_similarity_threshold": 0.25,
        "topic_edge_weight": 1.0,
    },
}

result = json.loads(extract_from_json(json.dumps(payload)))
for phrase in result["phrases"]:
    print(f"{phrase['text']}: {phrase['score']:.4f}")
```

## Stopword Handling

## AutoRank Result Metadata

When `variant="auto_rank"`, the JSON result includes a `consensus` object with:

- `selected_variants`
- `selection_reason`
- `variant_runs`
- `phrase_support`

The JSON interface supports two complementary mechanisms for stopword filtering:

1. **Per-token `is_stopword` field** -- set this to `true` on individual tokens (e.g., using `token.is_stop` from spaCy). This gives you full control over which tokens are treated as stopwords.

2. **`config.language` and `config.stopwords`** -- when `config.stopwords` is a non-empty list, the Rust core loads the built-in stopword list for the configured language, extends it with your custom stopwords, and marks any matching tokens as stopwords (in addition to any tokens already marked via `is_stopword`).

Both mechanisms can be used together. A token is treated as a stopword if `is_stopword` is `true` on the token itself OR if it matches the built-in + custom stopword list.
