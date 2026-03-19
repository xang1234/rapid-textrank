//! TextRank variants
//!
//! This module provides specialized TextRank variants:
//! - PositionRank: Biases towards words appearing earlier in the document
//! - BiasedTextRank: Allows focusing on specific topic words
//! - TopicRank: Clusters similar phrases before ranking
//! - SingleRank: TextRank with forced weighted edges and cross-sentence windowing
//! - TopicalPageRank: SingleRank graph + topic-weight-biased personalized PageRank

pub mod auto_rank;
pub mod biased_textrank;
pub mod multipartite_rank;
pub mod position_rank;
pub mod single_rank;
pub mod topic_rank;
pub mod topical_pagerank;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    TextRank,
    PositionRank,
    BiasedTextRank,
    TopicRank,
    SingleRank,
    TopicalPageRank,
    MultipartiteRank,
    AutoRank,
    #[cfg(feature = "sentence-rank")]
    SentenceRank,
}

impl Variant {
    fn parse(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "textrank" | "text_rank" | "base" => Variant::TextRank,
            "position_rank" | "positionrank" | "position" => Variant::PositionRank,
            "biased_textrank" | "biased" | "biasedtextrank" => Variant::BiasedTextRank,
            "topic_rank" | "topicrank" | "topic" => Variant::TopicRank,
            "single_rank" | "singlerank" | "single" => Variant::SingleRank,
            "topical_pagerank" | "topicalpagerank" | "single_tpr" | "tpr" => {
                Variant::TopicalPageRank
            }
            "multipartite_rank" | "multipartiterank" | "multipartite" | "mpr" => {
                Variant::MultipartiteRank
            }
            "auto_rank" | "autorank" | "auto" => Variant::AutoRank,
            #[cfg(feature = "sentence-rank")]
            "sentence_rank" | "sentencerank" | "sentence" => Variant::SentenceRank,
            _ => Variant::TextRank,
        }
    }

    pub fn canonical_name(&self) -> &'static str {
        match self {
            Variant::TextRank => "textrank",
            Variant::PositionRank => "position_rank",
            Variant::BiasedTextRank => "biased_textrank",
            Variant::TopicRank => "topic_rank",
            Variant::SingleRank => "single_rank",
            Variant::TopicalPageRank => "topical_pagerank",
            Variant::MultipartiteRank => "multipartite_rank",
            Variant::AutoRank => "auto_rank",
            #[cfg(feature = "sentence-rank")]
            Variant::SentenceRank => "sentence_rank",
        }
    }
}

impl std::str::FromStr for Variant {
    type Err = std::convert::Infallible;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Variant::parse(value))
    }
}
