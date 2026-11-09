//! # `sumeru` (Sumeru)
//!
//! **Sumeru** — 高性能纯连续内存列式状态引擎与关系型 ECS 内核 (Open-Source Core Edition, MIT License)。
//!
//! 公开导出：
//! - 稳定标识与 Schema 元数据：[`id`]、[`schema`]、[`error`]、[`binding`]、[`context`]
//! - 连续列式存储与关系型 ECS 门面：[`Store`]、[`Entity`]、[`EntityRef`]、[`EntityMut`]、
//!   [`Pair`]、[`Tag`]、[`Singleton`]、[`SelectBuilder`]、[`QueryBuilder`]、[`QueryMutBuilder`]、
//!   [`Query`]、[`CommandBuffer`]、[`CommitReceipt`]、[`PageAlignedBuffer`]、[`ColumnarStoreSnapshot`]

extern crate self as sumeru;

pub mod al;
pub mod binding;
pub mod context;
pub mod error;
pub mod id;
pub mod schedule;
pub mod schema;
pub mod storage;

pub use binding::*;
pub use context::*;
pub use error::*;
pub use id::*;
pub use schedule::*;
pub use schema::*;
pub use storage::*;
#[cfg(feature = "gpu")]
pub use sumeru_macros::{gpu, gpu_component, gpu_kernel, GpuComponent, GpuShaderStruct};

#[doc(hidden)]
pub mod __private {
    pub use bytemuck;
    pub use rkyv;
    pub use serde;
}

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

#[cfg(doctest)]
#[doc = include_str!("../README_zh-CN.md")]
pub struct ReadmeZhCnDoctests;

