//! 连续列式存储内核与关系型 ECS 门面 (Sumeru Storage Core)
//!
//! # 访问契约
//!
//! 统一权威 `Store` 提供单所有者强类型列存、关系对 `Pair<R, Target>`、零大小标签 `Tag<N>`、
//! 原位流式查询 `query` / `query_mut`、持久化查询计划 `Query<Q, F>` 以及延迟命令缓冲 `CommandBuffer`。
//!
//! ## 查询与借用
//!
//! - `Store::query` / `Store::query_mut` 提供基于原型表连续切片的零分配行迭代与块迭代（`chunks` / `chunks_mut`）。
//! - `Query` 保存查询形态及原型匹配缓存（含 Level-2 预绑定列槽位），随 `archetype_set_generation` 自适应增量刷新。
//! - `EntityRef` 与 `EntityMut` 提供单实体零开销游标与 1..=16 元组联合借用访问。
//!
//! ## 批量提交与命令缓冲
//!
//! - `StoreBatch`、`EncodedStoreBatch` 与 `CommandBuffer` 支持延迟实体创建、销毁与组件结构变更。
//! - 提交时执行两阶段原子预校验与物理应用，按需产出轻量级帧间增量收据 `CommitReceipt`（别名 `BatchReceipt`）。
//! - 物理变更过程受 `CorePoisonGuard` 保护，若发生异常 panic 自动标记 `Store` 为故障污染态（Poisoned）。
//!
//! 已删除的兼容查询入口不属于公共 API：
//! ```compile_fail
//! use sumeru::Store;
//! let store = Store::new();
//! let _ = store.select::<&u64>();
//! ```
//! ```compile_fail
//! use sumeru::Store;
//! let mut store = Store::new();
//! let _ = store.select_mut::<&mut u64>();
//! ```
//!
//! Physical backend types are deliberately not available to external callers.
//! ```compile_fail
//! use sumeru::storage::{EntityAllocator, EntityLocation, AnyColumn, TypedColumn, Archetype};
//! ```

pub use crate::al as contracts;

// 核心子模块声明 (包内实现私有化，统一通过 storage 平坦重导出)
pub(crate) mod batch;
pub(crate) mod core;
pub(crate) mod entity;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod primitives;
pub(crate) mod query;
pub(crate) mod stage;

// 平坦导出各层公共 API 与包内共享核心符号
pub use crate::al::{EntityHandle, Pair, StateSpace, Tag, WorldEngine};
pub use self::batch::*;
pub use self::core::*;
#[allow(unused_imports)]
pub(crate) use self::core::{
    allocator, archetype, column, entity_index, modify, registry, store, AnyColumn,
    Archetype, ColumnBacking, ColumnBlock, ColumnDescriptor, ColumnIter, ColumnValueOps, DirectBlock,
    EntityAllocator, EntityLocation, InPlaceReplacements, PlannedInPlacePatch,
    RawColumn, TypedColumn,
};
pub use self::entity::*;
#[cfg(feature = "gpu")]
pub use self::gpu::*;
pub use self::primitives::*;
pub use self::query::*;
pub use self::stage::*;

#[cfg(test)]
mod benches;
