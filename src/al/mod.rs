//! # StateFabric Abstraction Layer (`al`)
//!
//! 将 `sumeru` 的原生接口体系（`Store`、`EntityRef<'a>`、`EntityMut<'a>`、`CommandBuffer`、
//! `EntityCommands<'a>`、`Pair<R, E>`、`Tag<N>`、连续切片块查询与并行块迭代）直接提炼为标准契约：
//! - [`StateSpace`]（别名 [`WorldEngine`]）：通用状态空间契约（实体句柄 `EntityHandle`、快速探针与两阶段形式化诊断、
//!   即时生效 vs 两阶段延迟命令缓冲、链式实体门面 `EntityRefAccess` / `EntityMutAccess`、块级切片迭代与并行迭代、
//!   一等公民 `Pair` / `Tag`）。

pub mod store_impl;
pub mod traits;

#[cfg(test)]
mod tests;

pub use traits::{
    All, ColumnChunkProbe, CommandBufferAccess, Component, EntityCommandsAccess, EntityHandle,
    EntityMutAccess, EntityRefAccess, GenericCommandBuffer, GenericEntityCommands,
    GenericEntityMut, GenericEntityRef, LayoutReport, Pair, QueryFilter, StateSpace, Tag, With,
    Without, WorldEngine,
};
