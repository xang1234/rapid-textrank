"""
Tests for the rapid_textrank Python API.
"""

import pytest
import json


def test_import():
    """Test that the module can be imported."""
    import rapid_textrank

    assert hasattr(rapid_textrank, "__version__")
    assert hasattr(rapid_textrank, "BaseTextRank")
    assert hasattr(rapid_textrank, "PositionRank")
    assert hasattr(rapid_textrank, "BiasedTextRank")
    assert hasattr(rapid_textrank, "SingleRank")


def _find_exports_toml():
    """Walk up from this file to find exports.toml at the repo root."""
    import pathlib

    d = pathlib.Path(__file__).resolve().parent
    for _ in range(10):
        candidate = d / "exports.toml"
        if candidate.exists():
            return candidate
        d = d.parent
    raise FileNotFoundError("exports.toml not found in any parent directory")


def _load_manifest():
    """Parse exports.toml and return (all_names, feature_gated_names)."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    path = _find_exports_toml()
    with open(path, "rb") as f:
        data = tomllib.load(f)

    all_names = set()
    feature_gated = set()
    for section in ("classes", "functions", "constants"):
        for name, meta in data.get(section, {}).items():
            all_names.add(name)
            if meta.get("feature"):
                feature_gated.add(name)
    return all_names, feature_gated


def test_all_rust_symbols_match_manifest():
    """Bidirectional check: manifest <-> _rust module symbols."""
    import rapid_textrank
    import rapid_textrank._rust as _rust

    manifest_names, feature_gated = _load_manifest()
    init_names = set(dir(rapid_textrank))
    rust_names = {n for n in dir(_rust) if not n.startswith("_")}

    # Direction 1: manifest -> Python
    # Every manifest entry must be importable. Feature-gated items that are
    # absent from _rust are tolerated (the feature may be compiled out).
    missing_from_init = set()
    for name in manifest_names:
        if name in feature_gated and name not in rust_names:
            continue  # feature compiled out, acceptable
        if name not in init_names:
            missing_from_init.add(name)
    assert not missing_from_init, (
        f"In exports.toml but not importable from rapid_textrank: {missing_from_init}"
    )

    # Direction 2: _rust -> manifest
    # Every public symbol in _rust must be in the manifest.
    # Module-intrinsic dunders (__name__, __doc__, etc.) are excluded via
    # an allowlist; all other dunders (e.g. __version__) are checked.
    _module_dunders = frozenset({
        "__name__", "__doc__", "__file__", "__loader__",
        "__package__", "__spec__", "__all__", "__builtins__",
        "__cached__", "__path__",
    })
    rust_public = {n for n in dir(_rust) if not n.startswith("_")}
    rust_dunders = {n for n in dir(_rust)
                    if n.startswith("__") and n.endswith("__")
                    } - _module_dunders
    missing_from_manifest = (rust_public | rust_dunders) - manifest_names
    assert not missing_from_manifest, (
        f"In rapid_textrank._rust but not in exports.toml: {missing_from_manifest}"
    )


def test_version():
    """Test version string is valid."""
    from rapid_textrank import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) >= 2


class TestBaseTextRank:
    """Tests for BaseTextRank extractor."""

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=5)
        result = extractor.extract_keywords(
            "Machine learning is a subset of artificial intelligence. "
            "Deep learning is a type of machine learning."
        )

        assert len(result.phrases) > 0
        assert result.converged
        assert all(p.score > 0 for p in result.phrases)

    def test_extract_keywords_ranking(self):
        """Test that phrases are properly ranked."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=10)
        result = extractor.extract_keywords(
            "Machine learning algorithms process data. "
            "Machine learning is used in many applications. "
            "Data science relies on machine learning."
        )

        # Ranks should be sequential starting from 1
        for i, phrase in enumerate(result.phrases):
            assert phrase.rank == i + 1

    def test_empty_input(self):
        """Test handling of empty input."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank()
        result = extractor.extract_keywords("")

        assert len(result.phrases) == 0

    def test_top_n_limit(self):
        """Test that top_n limits results."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=3)
        result = extractor.extract_keywords(
            "Machine learning, deep learning, natural language processing, "
            "computer vision, and neural networks are all important topics."
        )

        assert len(result.phrases) <= 3

    def test_phrase_attributes(self):
        """Test phrase object attributes."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=1)
        result = extractor.extract_keywords("Machine learning is fascinating.")

        if result.phrases:
            phrase = result.phrases[0]
            assert hasattr(phrase, "text")
            assert hasattr(phrase, "lemma")
            assert hasattr(phrase, "score")
            assert hasattr(phrase, "count")
            assert hasattr(phrase, "rank")


class TestPositionRank:
    """Tests for PositionRank extractor."""

    def test_position_bias(self):
        """Test that early words are favored."""
        from rapid_textrank import PositionRank

        extractor = PositionRank(top_n=10)

        # "Important" appears first, "secondary" appears later
        result = extractor.extract_keywords(
            "Important topic is discussed first. "
            "Then we talk about secondary topic. "
            "Important topic appears again."
        )

        assert len(result.phrases) > 0


class TestBiasedTextRank:
    """Tests for BiasedTextRank extractor."""

    def test_focus_terms(self):
        """Test extraction with focus terms."""
        from rapid_textrank import BiasedTextRank

        extractor = BiasedTextRank(
            focus_terms=["neural"], bias_weight=10.0, top_n=10
        )

        result = extractor.extract_keywords(
            "Machine learning uses algorithms. "
            "Deep learning uses neural networks. "
            "Neural networks are powerful."
        )

        assert len(result.phrases) > 0

    def test_change_focus(self):
        """Test changing focus terms."""
        from rapid_textrank import BiasedTextRank

        extractor = BiasedTextRank(focus_terms=["machine"], top_n=10)
        result1 = extractor.extract_keywords("Machine learning and neural networks.")

        extractor.set_focus(["neural"])
        result2 = extractor.extract_keywords("Machine learning and neural networks.")

        # Both should produce results
        assert len(result1.phrases) > 0
        assert len(result2.phrases) > 0


class TestSingleRank:
    """Tests for SingleRank extractor."""

    def test_extract_keywords_basic(self):
        """Test basic SingleRank keyword extraction."""
        from rapid_textrank import SingleRank

        extractor = SingleRank(top_n=5)
        result = extractor.extract_keywords(
            "Machine learning is a subset of artificial intelligence. "
            "Deep learning is a type of machine learning. "
            "Neural networks are used in deep learning."
        )

        assert len(result.phrases) > 0
        assert result.converged
        assert all(p.score > 0 for p in result.phrases)

    def test_empty_input(self):
        """Test handling of empty input."""
        from rapid_textrank import SingleRank

        extractor = SingleRank()
        result = extractor.extract_keywords("")

        assert len(result.phrases) == 0

    def test_cross_sentence_keywords(self):
        """Test that cross-sentence co-occurrences are captured."""
        from rapid_textrank import SingleRank

        extractor = SingleRank(top_n=10)
        # "data science" spans sentence boundaries
        result = extractor.extract_keywords(
            "Modern data science is evolving. "
            "Science and data drive decisions. "
            "Data science applications are growing."
        )

        assert len(result.phrases) > 0
        top_texts = [p.text.lower() for p in result.phrases]
        assert any("data" in t or "science" in t for t in top_texts)


class TestSingleRankJson:
    """Tests for SingleRank via JSON interface."""

    def test_json_variant_single_rank(self):
        """Test SingleRank through the JSON API."""
        from rapid_textrank import extract_from_json

        doc = {
            "tokens": [
                {"text": "Machine", "lemma": "machine", "pos": "NOUN",
                 "start": 0, "end": 7, "sentence_idx": 0, "token_idx": 0,
                 "is_stopword": False},
                {"text": "learning", "lemma": "learning", "pos": "NOUN",
                 "start": 8, "end": 16, "sentence_idx": 0, "token_idx": 1,
                 "is_stopword": False},
                {"text": "deep", "lemma": "deep", "pos": "ADJ",
                 "start": 18, "end": 22, "sentence_idx": 1, "token_idx": 2,
                 "is_stopword": False},
                {"text": "learning", "lemma": "learning", "pos": "NOUN",
                 "start": 23, "end": 31, "sentence_idx": 1, "token_idx": 3,
                 "is_stopword": False},
            ],
            "variant": "single_rank",
            "config": {"top_n": 5},
        }

        result_json = extract_from_json(json.dumps(doc))
        result = json.loads(result_json)

        assert "phrases" in result
        assert "converged" in result
        assert len(result["phrases"]) > 0


class TestTextRankConfig:
    """Tests for TextRankConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from rapid_textrank import TextRankConfig

        config = TextRankConfig()
        assert config is not None

    def test_custom_config(self):
        """Test custom configuration."""
        from rapid_textrank import TextRankConfig, BaseTextRank

        config = TextRankConfig(
            damping=0.9,
            window_size=5,
            top_n=15,
            score_aggregation="mean",
        )

        extractor = BaseTextRank(config=config)
        result = extractor.extract_keywords("Test text for configuration.")

        assert result is not None

    def test_invalid_config(self):
        """Test that invalid config raises error."""
        from rapid_textrank import TextRankConfig
        import pytest

        with pytest.raises(ValueError):
            TextRankConfig(damping=2.0)  # Invalid: must be 0-1


class TestJsonInterface:
    """Tests for the JSON interface."""

    def test_extract_from_json(self):
        """Test JSON-based extraction."""
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
            ],
            "config": {"top_n": 5},
        }

        result_json = extract_from_json(json.dumps(doc))
        result = json.loads(result_json)

        assert "phrases" in result
        assert "converged" in result
        assert "iterations" in result

    def test_batch_from_json(self):
        """Test batch JSON extraction."""
        from rapid_textrank import extract_batch_from_json

        docs = [
            {
                "tokens": [
                    {
                        "text": "First",
                        "lemma": "first",
                        "pos": "ADJ",
                        "start": 0,
                        "end": 5,
                        "sentence_idx": 0,
                        "token_idx": 0,
                        "is_stopword": False,
                    },
                ],
            },
            {
                "tokens": [
                    {
                        "text": "Second",
                        "lemma": "second",
                        "pos": "ADJ",
                        "start": 0,
                        "end": 6,
                        "sentence_idx": 0,
                        "token_idx": 0,
                        "is_stopword": False,
                    },
                ],
            },
        ]

        results_json = extract_batch_from_json(json.dumps(docs))
        results = json.loads(results_json)

        assert isinstance(results, list)
        assert len(results) == 2


class TestConvenienceFunction:
    """Tests for the extract_keywords convenience function."""

    def test_extract_keywords(self):
        """Test the convenience function."""
        from rapid_textrank import extract_keywords

        phrases = extract_keywords(
            "Machine learning is transforming industries.", top_n=5
        )

        assert isinstance(phrases, list)


class TestMaxThreads:
    """Tests for per-extractor thread pool via max_threads."""

    SAMPLE_TEXT = (
        "Machine learning is a subset of artificial intelligence. "
        "Deep learning is a type of machine learning. "
        "Neural networks are used in deep learning."
    )

    def test_max_threads_constructor(self):
        """Create extractor with max_threads, verify getter."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=5, max_threads=2)
        assert extractor.max_threads == 2

    def test_max_threads_none_by_default(self):
        """Without max_threads, getter returns None (global pool)."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=5)
        assert extractor.max_threads is None

    def test_max_threads_extraction(self):
        """Extraction works correctly with a dedicated pool."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=5, max_threads=2)
        result = extractor.extract_keywords(self.SAMPLE_TEXT)

        assert len(result.phrases) > 0
        assert result.converged

    def test_set_max_threads(self):
        """Setter replaces the pool."""
        from rapid_textrank import SingleRank

        extractor = SingleRank(top_n=5)
        assert extractor.max_threads is None

        extractor.set_max_threads(3)
        assert extractor.max_threads == 3

        # Revert to global pool
        extractor.set_max_threads(None)
        assert extractor.max_threads is None

    def test_max_threads_zero_rejected(self):
        """max_threads=0 raises ValueError."""
        from rapid_textrank import PositionRank

        with pytest.raises(ValueError, match="max_threads must be >= 1"):
            PositionRank(top_n=5, max_threads=0)

    def test_set_max_threads_zero_rejected(self):
        """set_max_threads(0) also raises ValueError."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=5, max_threads=2)
        with pytest.raises(ValueError, match="max_threads must be >= 1"):
            extractor.set_max_threads(0)

    def test_all_variants_support_max_threads(self):
        """Every extractor class accepts max_threads."""
        from rapid_textrank import (
            BaseTextRank,
            PositionRank,
            BiasedTextRank,
            SingleRank,
            TopicalPageRank,
            MultipartiteRank,
        )

        classes = [
            lambda: BaseTextRank(top_n=3, max_threads=1),
            lambda: PositionRank(top_n=3, max_threads=1),
            lambda: BiasedTextRank(top_n=3, max_threads=1),
            lambda: SingleRank(top_n=3, max_threads=1),
            lambda: TopicalPageRank(top_n=3, max_threads=1),
            lambda: MultipartiteRank(top_n=3, max_threads=1),
        ]

        for factory in classes:
            ext = factory()
            assert ext.max_threads == 1
            result = ext.extract_keywords(self.SAMPLE_TEXT)
            assert len(result.phrases) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
