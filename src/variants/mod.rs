//! TextRank variants
//!
//! This module provides specialized TextRank variants:
//! - PositionRank: Biases towards words appearing earlier in the document
//! - BiasedTextRank: Allows focusing on specific topic words
//! - TopicRank: Clusters similar phrases before ranking

pub mod biased_textrank;
pub mod position_rank;
pub mod topic_rank;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    TextRank,
    PositionRank,
    BiasedTextRank,
    TopicRank,
}

impl Variant {
    fn parse(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "position_rank" | "positionrank" | "position" => Variant::PositionRank,
            "biased_textrank" | "biased" | "biasedtextrank" => Variant::BiasedTextRank,
            "topic_rank" | "topicrank" | "topic" => Variant::TopicRank,
            _ => Variant::TextRank,
        }
    }
}

impl std::str::FromStr for Variant {
    type Err = std::convert::Infallible;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Variant::parse(value))
    }
}
