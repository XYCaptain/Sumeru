//! Layer 2: 事务与暂存协议 (Transaction & Staging Protocol)
//!
//! 负责延迟命令缓冲 `CommandBuffer` 与实体流式构建器 `EntityCommands`。

pub(crate) mod commands;

pub use commands::{BatchUpdateItem, CommandBuffer, CommandLimits, EntityCommands};
