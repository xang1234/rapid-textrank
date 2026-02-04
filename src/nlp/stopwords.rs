//! Stopword filtering
//!
//! This module provides multi-language stopword filtering using the `stop-words` crate
//! with support for custom stopword lists.

use rustc_hash::FxHashSet;
use stop_words::{get, LANGUAGE};

/// A filter for removing stopwords from text
#[derive(Debug, Clone)]
pub struct StopwordFilter {
    /// Set of stopwords (lowercase)
    stopwords: FxHashSet<String>,
    /// Whether the filter is case-sensitive
    case_sensitive: bool,
}

impl Default for StopwordFilter {
    fn default() -> Self {
        Self::new("en")
    }
}

impl StopwordFilter {
    /// Create a new stopword filter for the given language
    ///
    /// Supported languages: en, de, fr, es, it, pt, nl, ru, ja, zh, ar, hi
    pub fn new(language: &str) -> Self {
        let stopwords = Self::load_stopwords(language);
        Self {
            stopwords,
            case_sensitive: false,
        }
    }

    /// Create an empty stopword filter (no filtering)
    pub fn empty() -> Self {
        Self {
            stopwords: FxHashSet::default(),
            case_sensitive: false,
        }
    }

    /// Create a stopword filter from a custom list
    pub fn from_list(words: &[&str]) -> Self {
        let stopwords: FxHashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
        Self {
            stopwords,
            case_sensitive: false,
        }
    }

    /// Set case sensitivity
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Add additional stopwords to the filter
    pub fn add_stopwords(&mut self, words: &[&str]) {
        for word in words {
            self.stopwords.insert(word.to_lowercase());
        }
    }

    /// Remove stopwords from the filter
    pub fn remove_stopwords(&mut self, words: &[&str]) {
        for word in words {
            self.stopwords.remove(&word.to_lowercase());
        }
    }

    /// Check if a word is a stopword
    pub fn is_stopword(&self, word: &str) -> bool {
        if self.case_sensitive {
            self.stopwords.contains(word)
        } else {
            self.stopwords.contains(&word.to_lowercase())
        }
    }

    /// Get the number of stopwords in the filter
    pub fn len(&self) -> usize {
        self.stopwords.len()
    }

    /// Check if the filter is empty
    pub fn is_empty(&self) -> bool {
        self.stopwords.is_empty()
    }

    /// Load stopwords for a language
    fn load_stopwords(language: &str) -> FxHashSet<String> {
        let lang = match language.to_lowercase().as_str() {
            "en" | "english" => LANGUAGE::English,
            "de" | "german" => LANGUAGE::German,
            "fr" | "french" => LANGUAGE::French,
            "es" | "spanish" => LANGUAGE::Spanish,
            "it" | "italian" => LANGUAGE::Italian,
            "pt" | "portuguese" => LANGUAGE::Portuguese,
            "nl" | "dutch" => LANGUAGE::Dutch,
            "ru" | "russian" => LANGUAGE::Russian,
            "sv" | "swedish" => LANGUAGE::Swedish,
            "no" | "norwegian" => LANGUAGE::Norwegian,
            "da" | "danish" => LANGUAGE::Danish,
            "fi" | "finnish" => LANGUAGE::Finnish,
            "hu" | "hungarian" => LANGUAGE::Hungarian,
            "tr" | "turkish" => LANGUAGE::Turkish,
            "pl" | "polish" => LANGUAGE::Polish,
            "ar" | "arabic" => LANGUAGE::Arabic,
            "zh" | "chinese" => {
                // Chinese doesn't have a standard stopword list in the crate
                // Return common Chinese stopwords
                return Self::chinese_stopwords();
            }
            "ja" | "japanese" => {
                // Japanese stopwords
                return Self::japanese_stopwords();
            }
            _ => {
                // Default to English for unknown languages
                LANGUAGE::English
            }
        };

        get(lang).iter().map(|s| s.to_string()).collect()
    }

    /// Common Chinese stopwords
    fn chinese_stopwords() -> FxHashSet<String> {
        [
            "的", "是", "在", "有", "和", "与", "或", "不", "了", "也", "就", "都", "而", "及",
            "这", "那", "个", "为", "以", "等", "但", "被", "给", "让", "把", "从", "到", "对",
            "将", "于", "能", "会", "可", "要", "很", "还", "更", "最", "只", "已", "又", "再",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Common Japanese stopwords
    fn japanese_stopwords() -> FxHashSet<String> {
        [
            "の",
            "に",
            "は",
            "を",
            "た",
            "が",
            "で",
            "て",
            "と",
            "し",
            "れ",
            "さ",
            "ある",
            "いる",
            "も",
            "する",
            "から",
            "な",
            "こと",
            "として",
            "い",
            "や",
            "など",
            "なっ",
            "ない",
            "この",
            "ため",
            "その",
            "あっ",
            "よう",
            "また",
            "もの",
            "という",
            "あり",
            "まで",
            "られ",
            "なる",
            "へ",
            "か",
            "だ",
            "これ",
            "によって",
            "により",
            "おり",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_english_stopwords() {
        let filter = StopwordFilter::new("en");

        assert!(filter.is_stopword("the"));
        assert!(filter.is_stopword("The")); // case insensitive
        assert!(filter.is_stopword("is"));
        assert!(filter.is_stopword("a"));
        assert!(!filter.is_stopword("machine"));
        assert!(!filter.is_stopword("learning"));
    }

    #[test]
    fn test_custom_stopwords() {
        let mut filter = StopwordFilter::from_list(&["custom", "words"]);

        assert!(filter.is_stopword("custom"));
        assert!(filter.is_stopword("words"));
        assert!(!filter.is_stopword("the"));

        filter.add_stopwords(&["extra"]);
        assert!(filter.is_stopword("extra"));

        filter.remove_stopwords(&["custom"]);
        assert!(!filter.is_stopword("custom"));
    }

    #[test]
    fn test_empty_filter() {
        let filter = StopwordFilter::empty();

        assert!(!filter.is_stopword("the"));
        assert!(!filter.is_stopword("a"));
        assert!(filter.is_empty());
    }

    #[test]
    fn test_german_stopwords() {
        let filter = StopwordFilter::new("de");

        assert!(filter.is_stopword("der"));
        assert!(filter.is_stopword("die"));
        assert!(filter.is_stopword("und"));
        assert!(!filter.is_stopword("machine"));
    }

    #[test]
    fn test_case_sensitivity() {
        let filter = StopwordFilter::new("en").with_case_sensitive(true);

        // The stop-words crate returns lowercase words
        assert!(filter.is_stopword("the"));
        assert!(!filter.is_stopword("The")); // Case sensitive now
    }

    #[test]
    fn test_chinese_stopwords() {
        let filter = StopwordFilter::new("zh");

        assert!(filter.is_stopword("的"));
        assert!(filter.is_stopword("是"));
        assert!(!filter.is_stopword("机器"));
    }

    #[test]
    fn test_japanese_stopwords() {
        let filter = StopwordFilter::new("ja");

        assert!(filter.is_stopword("の"));
        assert!(filter.is_stopword("は"));
        assert!(!filter.is_stopword("機械"));
    }
}
