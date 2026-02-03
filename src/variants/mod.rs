//! TextRank variants
//!
//! This module provides specialized TextRank variants:
//! - PositionRank: Biases towards words appearing earlier in the document
//! - BiasedTextRank: Allows focusing on specific topic words
//! - TopicRank: Clusters similar phrases before ranking

pub mod biased_textrank;
pub mod position_rank;
pub mod topic_rank;
