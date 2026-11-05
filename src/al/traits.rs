//! # StateFabric Abstraction Layer (AL) — Core Traits & GAT Contracts
//!
//! 抽象层直接将 `sumeru` 的原生公开接口（`Store`、`EntityRef<'a>`、`EntityMut<'a>`、
//! `CommandBuffer`、`EntityCommands<'a>`、多维切片查询、并行块迭代与关系对 `Pair<R, E>`）
//! 升格为标准 Trait + GAT 关联类型契约 [`StateSpace`]（别名 [`WorldEngine`]）。
//!
//! - 对于 `sumeru::Store`，关联类型直接绑定其原生句柄类型，**零二次包装、零旁路钩子**；
//! - 对于外部对比引擎（如 `hecs`、`bevy_ecs`、`flecs`），通过本模块提供的通用轻量门面实现同一契约。

use std::fmt::{Debug, Display};
use std::hash::Hash;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::context::ModelComponent;

// 统一复用 `storage::query` 的过滤组合子与块级探针，消除双轨重复定义
pub use crate::storage::query::{All, ColumnChunkProbe, QueryFilter, With, Without};

/// 抽象层统一组件契约：自动为所有满足 `ModelComponent + Serialize + DeserializeOwned + Clone` 的类型实现。
pub trait Component:
    ModelComponent + Serialize + DeserializeOwned + Clone + Send + Sync + 'static
{
}

impl<T> Component for T where
    T: ModelComponent + Serialize + DeserializeOwned + Clone + Send + Sync + 'static
{
}

// ============================================================================
// 0. 实体句柄契约 (EntityHandle) 与零大小标签/关系对原语 (Tag / Pair)
// ============================================================================

/// 具备明确（1-based 槽位索引 + 代际版本号）语义的通用实体句柄契约。
///
/// 统一提供：
/// - `from_parts(index, generation)`：从 1-based 槽位索引与代际号构造实体句柄；
/// - `index(self)` / `generation(self)`：提取槽位索引与代际号；
/// - `to_bits_u64(self)` / `from_bits_u64(bits)`：标准 64 位紧凑编码（高 32 位代际，低 32 位索引）；
/// - `NULL` / `is_null(self)`：空实体哨兵。
pub trait EntityHandle:
    Copy
    + Clone
    + Eq
    + PartialEq
    + Ord
    + PartialOrd
    + Hash
    + Debug
    + Display
    + Serialize
    + DeserializeOwned
    + Send
    + Sync
    + 'static
{
    /// 空实体常量（`index == 0, generation == 0`）。
    const NULL: Self;

    /// 从 1-based 槽位索引与代际号构造实体句柄。
    fn from_parts(index: u32, generation: u32) -> Self;

    /// 获取 1-based 槽位索引（`NULL` 实体为 `0`）。
    fn index(self) -> u32;

    /// 获取代际版本号（有效实体 `>= 1`）。
    fn generation(self) -> u32;

    /// 判断是否为空实体句柄。
    #[inline]
    fn is_null(self) -> bool {
        self == Self::NULL
    }

    /// 编码为 64 位无符号整数：`((generation as u64) << 32) | (index as u64)`。
    #[inline]
    fn to_bits_u64(self) -> u64 {
        ((self.generation() as u64) << 32) | (self.index() as u64)
    }

    /// 从 64 位紧凑位模式还原实体句柄。
    #[inline]
    fn from_bits_u64(bits: u64) -> Self {
        if bits == 0 {
            return Self::NULL;
        }
        let idx = (bits & 0xFFFF_FFFF) as u32;
        let g = (bits >> 32) as u32;
        Self::from_parts(idx, if g == 0 { 1 } else { g })
    }
}

impl EntityHandle for crate::EntityId {
    const NULL: Self = crate::EntityId::from_u128(0);

    #[inline]
    fn from_parts(index: u32, generation: u32) -> Self {
        if index == 0 && generation == 0 {
            return Self::NULL;
        }
        let g = if generation == 0 { 1 } else { generation };
        let bits = ((g as u64) << 32) | (index as u64);
        crate::EntityId::from_u128(bits as u128)
    }

    #[inline]
    fn index(self) -> u32 {
        self.to_u128() as u32
    }

    #[inline]
    fn generation(self) -> u32 {
        let g = (self.to_u128() >> 32) as u32;
        if g == 0 && !self.is_nil() { 1 } else { g }
    }
}

/// 零大小编译期编号标签组件 (`Tag<N>`)。
///
/// 在 Archetype 引擎中仅驱动原型迁移，不产生有载荷列内存拷贝。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct Tag<const N: u8>;

/// 一等公民关系对组件 (`Pair<R, E>`)，建模 `(Relation R, Target Entity E)` 有向图边。
///
/// 默认目标实体句柄参数 `E = crate::EntityId`，使原生调用方可直接简写 `Pair<R>`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Pair<R, E = crate::EntityId> {
    pub relation: R,
    pub target: E,
}

impl<R, E> Pair<R, E> {
    #[inline]
    pub const fn new(relation: R, target: E) -> Self {
        Self { relation, target }
    }
}

// ============================================================================
// 1. 原生句柄门面契约 (EntityRefAccess / EntityMutAccess / CommandBufferAccess)
// ============================================================================

/// 单实体只读访问门面契约（直接对应 `sumeru::EntityRef<'a>`）。
pub trait EntityRefAccess<'a> {
    type Entity: EntityHandle;
    type Error: Debug + Display + Send + Sync + 'static;

    /// 获取当前实体 ID。
    fn id(&self) -> Self::Entity;

    /// 判断实体是否持有组件 `T`。
    fn has<T: Component>(&self) -> bool;

    /// 零开销探针：获取组件 `T` 的只读借用（若缺失则返回 `None`）。
    fn get<T: Component>(&self) -> Option<&'a T>;

    /// 形式化诊断：获取组件 `T` 的只读借用（若缺失则返回结构化错误）。
    fn try_get<T: Component>(&self) -> Result<&'a T, Self::Error>;
}

/// 单实体独占可变访问门面契约（直接对应 `sumeru::EntityMut<'a>`，支持链式与原位修改）。
pub trait EntityMutAccess<'a>: Sized {
    type Entity: EntityHandle;
    type Error: Debug + Display + Send + Sync + 'static;

    /// 获取当前实体 ID。
    fn id(&self) -> Self::Entity;

    /// 判断实体是否持有组件 `T`。
    fn has<T: Component>(&self) -> bool;

    /// 零开销探针：获取组件 `T` 的只读借用（若缺失则返回 `None`）。
    fn get<T: Component>(&self) -> Option<&T>;

    /// 形式化诊断：获取组件 `T` 的只读借用（若缺失则返回结构化错误）。
    fn try_get<T: Component>(&self) -> Result<&T, Self::Error>;

    /// 零开销探针：获取组件 `T` 的可变借用（若缺失则返回 `None`）。
    fn get_mut<T: Component>(&mut self) -> Option<&mut T>;

    /// 插入或替换组件 `T`（即时生效，返回旧值）。
    fn set<T: Component>(&mut self, value: T) -> Result<Option<T>, Self::Error>;

    /// 新增组件 `T`（若已存在则返回错误，即时生效）。
    fn add<T: Component>(&mut self, value: T) -> Result<(), Self::Error>;

    /// 新增组件 `T`（`add` 的兼容别名）。
    #[inline]
    fn insert<T: Component>(&mut self, value: T) -> Result<(), Self::Error> {
        self.add(value)
    }

    /// 替换已有组件 `T`（若不存在则返回错误，即时生效）。
    fn replace<T: Component>(&mut self, value: T) -> Result<T, Self::Error>;

    /// 移除组件 `T` 并返回旧值（即时生效）。
    fn remove<T: Component>(&mut self) -> Result<T, Self::Error>;

    /// 原位闭包修改组件 `T`（即时生效）。
    fn modify<T: Component, R>(
        &mut self,
        f: impl FnOnce(&mut T) -> R,
    ) -> Result<R, Self::Error>;

    /// 销毁当前实体并消费可变门面（即时生效）。
    fn despawn(self) -> Result<(), Self::Error>;

    /// 销毁当前实体并消费可变门面（`despawn` 的兼容别名）。
    #[inline]
    fn delete(self) -> Result<(), Self::Error> {
        self.despawn()
    }
}

/// 单实体延迟命令流式构建器契约（直接对应 `sumeru::EntityCommands<'a>`）。
pub trait EntityCommandsAccess<'a>: Sized {
    type Entity: EntityHandle;

    /// 获取目标实体 ID。
    fn id(&self) -> Self::Entity;

    /// 排队新增组件 `T`。
    fn add<T: Component>(&mut self, value: T) -> &mut Self;

    /// 排队新增组件 `T`（`add` 的兼容别名）。
    #[inline]
    fn insert<T: Component>(&mut self, value: T) -> &mut Self {
        self.add(value)
    }

    /// 排队替换已有组件 `T`。
    fn replace<T: Component>(&mut self, value: T) -> &mut Self;

    /// 排队设置（插入或覆盖）组件 `T`（同一批次内多次 `set` 以最后一次为准，且支持对同批次 `create_empty` 的实体追加组件）。
    fn set<T: Component>(&mut self, value: T) -> &mut Self;

    /// 排队插入或覆盖组件 `T`（`set` 的兼容别名）。
    #[inline]
    fn upsert<T: Component>(&mut self, value: T) -> &mut Self {
        self.set(value)
    }

    /// 排队移除组件 `T`。
    fn remove<T: Component>(&mut self) -> &mut Self;

    /// 排队设置指向 `target` 的关系对 `Pair<R, Self::Entity>`。
    #[inline]
    fn set_pair<R: Component>(&mut self, relation: R, target: Self::Entity) -> &mut Self {
        self.set(Pair::new(relation, target))
    }

    /// 排队移除关系对 `Pair<R, Self::Entity>`。
    #[inline]
    fn remove_pair<R: Component>(&mut self) -> &mut Self {
        self.remove::<Pair<R, Self::Entity>>()
    }

    /// 排队添加零大小编译期槽位标签 `Tag<N>`。
    #[inline]
    fn add_tag<const N: u8>(&mut self) -> &mut Self {
        self.set(Tag::<N>)
    }

    /// 排队移除零大小编译期槽位标签 `Tag<N>`。
    #[inline]
    fn remove_tag<const N: u8>(&mut self) -> &mut Self {
        self.remove::<Tag<N>>()
    }

    /// 排队销毁当前实体。
    fn despawn(&mut self) -> &mut Self;

    /// 排队销毁当前实体（`despawn` 的兼容别名）。
    #[inline]
    fn delete(&mut self) -> &mut Self {
        self.despawn()
    }
}

/// 两阶段延迟命令缓冲契约（直接对应 `sumeru::CommandBuffer`）。
///
/// 第一阶段在 `&S` 只读借用或独立状态下录制命令，第二阶段通过 `apply_discard(&mut S)` 统一提交生效。
pub trait CommandBufferAccess<S: ?Sized>: Sized {
    type Entity: EntityHandle;
    type Error: Debug + Display + Send + Sync + 'static;
    type EntityCommands<'a>: EntityCommandsAccess<'a, Entity = Self::Entity>
    where
        Self: 'a;

    /// 获取针对指定实体 `id` 的流式命令构建器。
    fn entity(&mut self, id: Self::Entity) -> Self::EntityCommands<'_>;

    /// 排队创建一个指定 ID 的空实体（可紧接着通过 `self.entity(id).set(comp)` 追加组件）。
    fn create_empty(&mut self, id: Self::Entity) -> &mut Self;

    /// 排队销毁指定实体。
    fn despawn(&mut self, id: Self::Entity) -> &mut Self;

    /// 排队销毁指定实体（`despawn` 的兼容别名）。
    #[inline]
    fn delete_entity(&mut self, id: Self::Entity) -> &mut Self {
        self.despawn(id)
    }

    /// 判断命令缓冲是否为空。
    fn is_empty(&self) -> bool;

    /// 提交缓冲中的所有延迟命令至目标引擎 `store`（不保留撤销收据）。
    fn apply_discard(self, store: &mut S) -> Result<(), Self::Error>;
}

// ============================================================================
// 2. 供外部引擎直接复用的零擦除通用句柄门面 (GenericEntityRef / GenericEntityMut / GenericCommandBuffer)
// ============================================================================

/// 供外部引擎直接复用的只读实体门面（基于 `StateSpace` 点查，零额外包装开销）。
pub struct GenericEntityRef<'a, E: StateSpace> {
    engine: &'a E,
    id: E::Entity,
    missing_err: fn(E::Entity, &'static str) -> E::Error,
}

impl<'a, E: StateSpace> GenericEntityRef<'a, E> {
    #[inline]
    pub fn new(
        engine: &'a E,
        id: E::Entity,
        missing_err: fn(E::Entity, &'static str) -> E::Error,
    ) -> Self {
        Self {
            engine,
            id,
            missing_err,
        }
    }
}

impl<'a, E: StateSpace> EntityRefAccess<'a> for GenericEntityRef<'a, E> {
    type Entity = E::Entity;
    type Error = E::Error;

    #[inline]
    fn id(&self) -> Self::Entity {
        self.id
    }

    #[inline]
    fn has<T: Component>(&self) -> bool {
        self.engine.has::<T>(self.id)
    }

    #[inline]
    fn get<T: Component>(&self) -> Option<&'a T> {
        self.engine.get::<T>(self.id)
    }

    #[inline]
    fn try_get<T: Component>(&self) -> Result<&'a T, Self::Error> {
        self.engine
            .get::<T>(self.id)
            .ok_or_else(|| (self.missing_err)(self.id, std::any::type_name::<T>()))
    }
}

/// 供外部引擎直接复用的独占可变实体门面（基于 `StateSpace` 即时修改方法）。
pub struct GenericEntityMut<'a, E: StateSpace> {
    engine: &'a mut E,
    id: E::Entity,
    missing_err: fn(E::Entity, &'static str) -> E::Error,
}

impl<'a, E: StateSpace> GenericEntityMut<'a, E> {
    #[inline]
    pub fn new(
        engine: &'a mut E,
        id: E::Entity,
        missing_err: fn(E::Entity, &'static str) -> E::Error,
    ) -> Self {
        Self {
            engine,
            id,
            missing_err,
        }
    }
}

impl<'a, E: StateSpace> EntityMutAccess<'a> for GenericEntityMut<'a, E> {
    type Entity = E::Entity;
    type Error = E::Error;

    #[inline]
    fn id(&self) -> Self::Entity {
        self.id
    }

    #[inline]
    fn has<T: Component>(&self) -> bool {
        self.engine.has::<T>(self.id)
    }

    #[inline]
    fn get<T: Component>(&self) -> Option<&T> {
        self.engine.get::<T>(self.id)
    }

    #[inline]
    fn try_get<T: Component>(&self) -> Result<&T, Self::Error> {
        self.engine
            .get::<T>(self.id)
            .ok_or_else(|| (self.missing_err)(self.id, std::any::type_name::<T>()))
    }

    #[inline]
    fn get_mut<T: Component>(&mut self) -> Option<&mut T> {
        self.engine.get_mut::<T>(self.id)
    }

    #[inline]
    fn set<T: Component>(&mut self, value: T) -> Result<Option<T>, Self::Error> {
        let old = self.engine.get::<T>(self.id).cloned();
        self.engine.set::<T>(self.id, value)?;
        Ok(old)
    }

    #[inline]
    fn add<T: Component>(&mut self, value: T) -> Result<(), Self::Error> {
        self.engine.add::<T>(self.id, value)
    }

    #[inline]
    fn replace<T: Component>(&mut self, value: T) -> Result<T, Self::Error> {
        self.engine.replace::<T>(self.id, value)
    }

    #[inline]
    fn remove<T: Component>(&mut self) -> Result<T, Self::Error> {
        self.engine
            .remove::<T>(self.id)?
            .ok_or_else(|| (self.missing_err)(self.id, std::any::type_name::<T>()))
    }

    #[inline]
    fn modify<T: Component, R>(
        &mut self,
        f: impl FnOnce(&mut T) -> R,
    ) -> Result<R, Self::Error> {
        self.engine.modify::<T, R>(self.id, f)
    }

    #[inline]
    fn despawn(self) -> Result<(), Self::Error> {
        self.engine.despawn(self.id)
    }
}

/// 供外部引擎直接复用的强类型闭包延迟命令缓冲（零 `StoreBatch`、零 `ErasedComponentValue` 绕行）。
pub struct GenericCommandBuffer<E: StateSpace> {
    ops: Vec<Box<dyn FnOnce(&mut E) -> Result<(), E::Error> + Send>>,
}

impl<E: StateSpace> Default for GenericCommandBuffer<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: StateSpace> GenericCommandBuffer<E> {
    #[inline]
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }
}

/// 供外部引擎直接复用的单实体延迟命令流式构建器。
pub struct GenericEntityCommands<'a, E: StateSpace> {
    buf: &'a mut GenericCommandBuffer<E>,
    id: E::Entity,
}

impl<'a, E: StateSpace> EntityCommandsAccess<'a> for GenericEntityCommands<'a, E> {
    type Entity = E::Entity;

    #[inline]
    fn id(&self) -> Self::Entity {
        self.id
    }

    #[inline]
    fn add<T: Component>(&mut self, value: T) -> &mut Self {
        let id = self.id;
        self.buf
            .ops
            .push(Box::new(move |engine: &mut E| engine.add::<T>(id, value)));
        self
    }

    #[inline]
    fn replace<T: Component>(&mut self, value: T) -> &mut Self {
        let id = self.id;
        self.buf.ops.push(Box::new(move |engine: &mut E| {
            engine.replace::<T>(id, value).map(|_| ())
        }));
        self
    }

    #[inline]
    fn set<T: Component>(&mut self, value: T) -> &mut Self {
        let id = self.id;
        self.buf
            .ops
            .push(Box::new(move |engine: &mut E| engine.set::<T>(id, value)));
        self
    }

    #[inline]
    fn remove<T: Component>(&mut self) -> &mut Self {
        let id = self.id;
        self.buf.ops.push(Box::new(move |engine: &mut E| {
            engine.remove::<T>(id).map(|_| ())
        }));
        self
    }

    #[inline]
    fn despawn(&mut self) -> &mut Self {
        let id = self.id;
        self.buf
            .ops
            .push(Box::new(move |engine: &mut E| engine.despawn(id)));
        self
    }
}

impl<E: StateSpace> CommandBufferAccess<E> for GenericCommandBuffer<E> {
    type Entity = E::Entity;
    type Error = E::Error;
    type EntityCommands<'a>
        = GenericEntityCommands<'a, E>
    where
        Self: 'a;

    #[inline]
    fn entity(&mut self, id: Self::Entity) -> Self::EntityCommands<'_> {
        GenericEntityCommands { buf: self, id }
    }

    #[inline]
    fn create_empty(&mut self, id: Self::Entity) -> &mut Self {
        self.ops.push(Box::new(move |engine: &mut E| {
            engine.spawn_at(id).map(|_| ())
        }));
        self
    }

    #[inline]
    fn despawn(&mut self, id: Self::Entity) -> &mut Self {
        self.ops
            .push(Box::new(move |engine: &mut E| engine.despawn(id)));
        self
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    #[inline]
    fn apply_discard(self, store: &mut E) -> Result<(), Self::Error> {
        for op in self.ops {
            op(store)?;
        }
        Ok(())
    }
}

// ============================================================================
// 3. Layer 1: StateSpace (WorldEngine) — 通用状态空间数据平面契约
// ============================================================================

/// 通用状态空间数据平面契约（直接对应 `sumeru::Store` 的公开接口定义）。
///
/// 严格区分：
/// 1. **零开销快速探针路径（Fast-Path Probe，`Option<T>` / `bool`）**：
///    `entity(id)`、`get::<T>(id)`、`get_mut::<T>(id)`、`has::<T>(id)`、`pair_target::<R>(id)`；
/// 2. **两阶段形式化诊断路径（Formal Two-Stage Diagnostic，`Result<T, Self::Error>`）**：
///    `try_entity(id)`、`try_get::<T>(id)`、`try_get_mut::<T>(id)`、`try_has::<T>(id)`、`try_pair_target::<R>(id)`；
/// 3. **即时生效写路径（Immediate Execution，`&mut self`）** 与 **两阶段延迟命令缓冲路径（`commands(&self)`）**。
pub trait StateSpace: Send + Sync + 'static {
    /// 引擎可读标识名称。
    const ENGINE_NAME: &'static str;

    /// 后端可读标识名称（与 `ENGINE_NAME` 一致，对标实施方案 §1.4）。
    const BACKEND_NAME: &'static str = Self::ENGINE_NAME;

    /// 引擎原生实体句柄类型（实现 `EntityHandle`）。
    type Entity: EntityHandle;

    /// 引擎错误类型。
    type Error: Debug + Display + Send + Sync + 'static;

    /// 单实体只读视图（在 `Store` 上直接绑定为 `sumeru::EntityRef<'a>`）。
    type EntityRef<'a>: EntityRefAccess<'a, Entity = Self::Entity, Error = Self::Error>
    where
        Self: 'a;

    /// 单实体独占可变视图（在 `Store` 上直接绑定为 `sumeru::EntityMut<'a>`）。
    type EntityMut<'a>: EntityMutAccess<'a, Entity = Self::Entity, Error = Self::Error>
    where
        Self: 'a;

    /// 两阶段延迟命令缓冲（在 `Store` 上直接绑定为 `sumeru::CommandBuffer`）。
    type CommandBuffer: CommandBufferAccess<Self, Entity = Self::Entity, Error = Self::Error>;

    // ── 容量、存活状态与空间生命周期 ──────────────────────────────────────────

    /// 当前空间中存活的实体总数。
    fn entity_count(&self) -> usize;

    /// 当前空间中存活的实体总数（与 `Store::len` 一致）。
    #[inline]
    fn len(&self) -> usize {
        self.entity_count()
    }

    /// 当前空间是否不包含任何存活实体。
    #[inline]
    fn is_empty(&self) -> bool {
        self.entity_count() == 0
    }

    /// 检查指定实体 `entity` 当前是否在空间中存活（与 `Store::contains_entity` 一致）。
    fn contains_entity(&self, entity: Self::Entity) -> bool;

    /// 检查指定实体 `entity` 当前是否存活（与 `Store::is_alive` 一致）。
    #[inline]
    fn is_alive(&self, entity: Self::Entity) -> bool {
        self.contains_entity(entity)
    }

    /// 枚举当前空间中所有存活实体的句柄列表。
    fn alive_entities(&self) -> Vec<Self::Entity>;

    /// 枚举当前空间中所有存活实体的句柄列表（与 `alive_entities` 一致）。
    #[inline]
    fn entities(&self) -> Vec<Self::Entity> {
        self.alive_entities()
    }

    /// 清空空间中的所有实体与组件状态。
    fn clear(&mut self);

    // ── 即时生效生命周期管理 (&mut self) ─────────────────────────────────────

    /// 【即时生效】自动分配并立即在空间中创建一个新的空实体（与 `Store::spawn_auto` 对应）。
    fn spawn(&mut self) -> Result<Self::Entity, Self::Error>;

    /// 【即时生效】自动分配并创建一个新的空实体（`spawn` 的同义方法）。
    #[inline]
    fn spawn_auto(&mut self) -> Result<Self::Entity, Self::Error> {
        self.spawn()
    }

    /// 【即时生效】按指定实体句柄 `entity` 立即创建（或覆写复活）一个空实体。
    fn spawn_at(&mut self, entity: Self::Entity) -> Result<Self::Entity, Self::Error>;

    /// 【即时生效】立即从空间中彻底销毁指定实体及其挂载的全部组件（与 `Store::despawn` 一致）。
    fn despawn(&mut self, entity: Self::Entity) -> Result<(), Self::Error>;

    /// 【即时生效】立即从空间中彻底删除指定实体及其全部组件（与 `Store::delete_entity` 一致）。
    #[inline]
    fn delete_entity(&mut self, entity: Self::Entity) -> Result<(), Self::Error> {
        self.despawn(entity)
    }

    /// 【即时生效】批量销毁指定实体列表，返回实际成功销毁的存活实体数量。
    fn despawn_batch(&mut self, entities: &[Self::Entity]) -> Result<usize, Self::Error> {
        let mut removed = 0;
        for &id in entities {
            if self.contains_entity(id) {
                self.despawn(id)?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    // ── 原生实体门面访问 (EntityRef / EntityMut) ──────────────────────────────

    /// 零开销探针：获取指定实体的只读视图门面 `Self::EntityRef<'_>`（若实体不存在或已销毁则返回 `None`）。
    fn entity(&self, id: Self::Entity) -> Option<Self::EntityRef<'_>>;

    /// 获取指定实体的独占可变视图门面 `Self::EntityMut<'_>`（若实体不存在则返回错误）。
    fn entity_mut(&mut self, id: Self::Entity) -> Result<Self::EntityMut<'_>, Self::Error>;

    // ── 即时生效单组件 CRUD (&self / &mut self) ───────────────────────────────

    /// 零开销探针：获取实体上组件 `T` 的只读引用。
    fn get<T: Component>(&self, entity: Self::Entity) -> Option<&T>;

    /// 零开销探针：获取实体上组件 `T` 的可变引用。
    fn get_mut<T: Component>(&mut self, entity: Self::Entity) -> Option<&mut T>;

    /// 零开销探针：判断实体当前是否持有组件 `T`。
    #[inline]
    fn has<T: Component>(&self, entity: Self::Entity) -> bool {
        self.get::<T>(entity).is_some()
    }

    /// 【即时生效】为实体设置（插入或原位覆盖）组件 `T`。
    fn set<T: Component>(&mut self, entity: Self::Entity, component: T) -> Result<(), Self::Error>;

    /// 【即时生效】为实体新增组件 `T`（在 `Store` 上严格校验组件未存在；默认委托至 `set`）。
    #[inline]
    fn add<T: Component>(&mut self, entity: Self::Entity, component: T) -> Result<(), Self::Error> {
        self.set(entity, component)
    }

    /// 【即时生效】为实体新增组件 `T`（`add` 的兼容别名）。
    #[inline]
    fn insert_component<T: Component>(
        &mut self,
        entity: Self::Entity,
        component: T,
    ) -> Result<(), Self::Error> {
        self.add(entity, component)
    }

    /// 【即时生效】替换实体已有的组件 `T` 并返回旧值。
    fn replace<T: Component>(
        &mut self,
        entity: Self::Entity,
        component: T,
    ) -> Result<T, Self::Error>;

    /// 【即时生效】替换实体已有的组件 `T` 并返回旧值（`replace` 的兼容别名）。
    #[inline]
    fn replace_component<T: Component>(
        &mut self,
        entity: Self::Entity,
        component: T,
    ) -> Result<T, Self::Error> {
        self.replace(entity, component)
    }

    /// 【即时生效】移除实体上的组件 `T` 并返回旧值（若组件不存在则返回 `Ok(None)`）。
    fn remove<T: Component>(&mut self, entity: Self::Entity) -> Result<Option<T>, Self::Error>;

    /// 【即时生效】原位闭包修改实体上的组件 `T`。
    fn modify<T: Component, R>(
        &mut self,
        entity: Self::Entity,
        f: impl FnOnce(&mut T) -> R,
    ) -> Result<R, Self::Error>;

    // ── 两阶段延迟命令缓冲 (&self -> apply_discard) ──────────────────────────

    /// 【两阶段延迟提交】创建绑定当前空间的延迟命令缓冲 `Self::CommandBuffer`。
    fn commands(&self) -> Self::CommandBuffer;

    // ── 一等公民关系对 (Pair<R, Self::Entity>) ────────────────────────────────

    /// 为源实体 `source` 设置指向 `target` 的关系对 `Pair<R, Self::Entity>`（校验源实体与目标实体均存活）。
    #[inline]
    fn set_pair<R: Component>(
        &mut self,
        source: Self::Entity,
        relation: R,
        target: Self::Entity,
    ) -> Result<(), Self::Error> {
        if !self.contains_entity(target) {
            let _ = self.entity_mut(target)?;
        }
        self.set::<Pair<R, Self::Entity>>(source, Pair::new(relation, target))
    }

    /// 为源实体 `source` 挂载指向 `target` 的关系对（`set_pair` 的兼容别名）。
    #[inline]
    fn add_pair<R: Component>(
        &mut self,
        source: Self::Entity,
        target: Self::Entity,
        relation: R,
    ) -> Result<(), Self::Error> {
        self.set_pair(source, relation, target)
    }

    /// 移除源实体 `source` 上指向 `target` 的关系 `R`（若目标匹配则移除并返回 `Ok(true)`）。
    #[inline]
    fn remove_pair<R: Component>(
        &mut self,
        source: Self::Entity,
        target: Self::Entity,
    ) -> Result<bool, Self::Error> {
        let matches = self
            .get::<Pair<R, Self::Entity>>(source)
            .is_some_and(|p| p.target == target);
        if matches {
            let _ = self.remove::<Pair<R, Self::Entity>>(source)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// 零开销探针：查询源实体 `source` 在关系 `R` 下指向的目标实体（若目标已销毁则自动返回 `None`）。
    #[inline]
    fn pair_target<R: Component>(&self, source: Self::Entity) -> Option<Self::Entity> {
        let target = self.get::<Pair<R, Self::Entity>>(source)?.target;
        if self.contains_entity(target) {
            Some(target)
        } else {
            None
        }
    }

    /// 反查所有通过关系 `R` 指向 `target` 的源实体集合。
    #[inline]
    fn sources_with_pair<R: Component>(&self, target: Self::Entity) -> Vec<Self::Entity> {
        if !self.contains_entity(target) {
            return Vec::new();
        }
        let mut out = Vec::new();
        self.for_each_chunk1::<Pair<R, Self::Entity>, All>(|entities, pairs| {
            for (i, p) in pairs.iter().enumerate() {
                if p.target == target {
                    out.push(entities[i]);
                }
            }
        });
        out
    }

    /// 【即时生效】为实体挂载零大小编译期槽位标签 `Tag<N>`。
    #[inline]
    fn add_tag<const N: u8>(&mut self, entity: Self::Entity) -> Result<(), Self::Error> {
        self.set(entity, Tag::<N>)
    }

    /// 【即时生效】移除实体上的零大小编译期槽位标签 `Tag<N>`。
    #[inline]
    fn remove_tag<const N: u8>(&mut self, entity: Self::Entity) -> Result<bool, Self::Error> {
        Ok(self.remove::<Tag<N>>(entity)?.is_some())
    }

    /// 零开销探针：判断实体是否持有零大小编译期槽位标签 `Tag<N>`。
    #[inline]
    fn has_tag<const N: u8>(&self, entity: Self::Entity) -> bool {
        self.has::<Tag<N>>(entity)
    }

    // ── 连续切片只读/可变查询与并行块迭代 (Chunk Slice Queries) ──────────────

    /// 按谓词检索第一个匹配的实体。
    fn find_entity_by<T: Component>(&self, pred: impl Fn(&T) -> bool) -> Option<Self::Entity> {
        let mut found = None;
        self.for_each_chunk1::<T, All>(|entities, col| {
            if found.is_some() {
                return;
            }
            for (i, item) in col.iter().enumerate() {
                if pred(item) {
                    found = Some(entities[i]);
                    return;
                }
            }
        });
        found
    }

    /// 单组件只读块切片遍历（带过滤组合子 `F`）。
    fn for_each_chunk1<A: Component, F: QueryFilter>(&self, f: impl FnMut(&[Self::Entity], &[A]));

    /// 双组件只读块切片遍历（带过滤组合子 `F`）。
    fn for_each_chunk2<A: Component, B: Component, F: QueryFilter>(
        &self,
        f: impl FnMut(&[Self::Entity], &[A], &[B]),
    );

    /// 三组件只读块切片遍历（带过滤组合子 `F`）。
    fn for_each_chunk3<A: Component, B: Component, C: Component, F: QueryFilter>(
        &self,
        f: impl FnMut(&[Self::Entity], &[A], &[B], &[C]),
    );

    /// 四组件只读块切片遍历（带过滤组合子 `F`）。
    fn for_each_chunk4<A: Component, B: Component, C: Component, D: Component, F: QueryFilter>(
        &self,
        f: impl FnMut(&[Self::Entity], &[A], &[B], &[C], &[D]),
    );

    /// 五组件只读块切片遍历（带过滤组合子 `F`）。
    fn for_each_chunk5<
        A: Component,
        B: Component,
        C: Component,
        D: Component,
        G: Component,
        F: QueryFilter,
    >(
        &self,
        f: impl FnMut(&[Self::Entity], &[A], &[B], &[C], &[D], &[G]),
    );

    /// 单组件可变块切片遍历（带过滤组合子 `F`）。
    fn for_each_chunk_mut1<A: Component, F: QueryFilter>(
        &mut self,
        f: impl FnMut(&[Self::Entity], &mut [A]),
    );

    /// `1 写 + 1 读` 块切片遍历（带过滤组合子 `F`）。
    fn for_each_chunk_mut2<A: Component, B: Component, F: QueryFilter>(
        &mut self,
        f: impl FnMut(&[Self::Entity], &mut [A], &[B]),
    );

    /// `2 写 + 1 读` 块切片遍历（带过滤组合子 `F`）。
    fn for_each_chunk_mut3<A: Component, B: Component, C: Component, F: QueryFilter>(
        &mut self,
        f: impl FnMut(&[Self::Entity], &mut [A], &mut [B], &[C]),
    );

    /// 多核并行单组件可变块切片遍历（带过滤组合子 `F`）。
    fn par_for_each_chunk_mut1<A: Component, F: QueryFilter>(
        &mut self,
        f: impl Fn(&[Self::Entity], &mut [A]) + Send + Sync,
    );

    /// 多核并行 `1 写 + 1 读` 块切片遍历（带过滤组合子 `F`）。
    fn par_for_each_chunk_mut2<A: Component, B: Component, F: QueryFilter>(
        &mut self,
        f: impl Fn(&[Self::Entity], &mut [A], &[B]) + Send + Sync,
    );

    /// 单组件只读行遍历（基于 `for_each_chunk1` 的零开销内联展开）。
    #[inline]
    fn for_each1<A: Component, F: QueryFilter>(&self, mut f: impl FnMut(Self::Entity, &A)) {
        self.for_each_chunk1::<A, F>(|ids, a| {
            for (i, &id) in ids.iter().enumerate() {
                f(id, &a[i]);
            }
        });
    }

    /// 双组件只读行遍历（基于 `for_each_chunk2` 的零开销内联展开）。
    #[inline]
    fn for_each2<A: Component, B: Component, F: QueryFilter>(
        &self,
        mut f: impl FnMut(Self::Entity, &A, &B),
    ) {
        self.for_each_chunk2::<A, B, F>(|ids, a, b| {
            for (i, &id) in ids.iter().enumerate() {
                f(id, &a[i], &b[i]);
            }
        });
    }

    /// 三组件只读行遍历（基于 `for_each_chunk3` 的零开销内联展开）。
    #[inline]
    fn for_each3<A: Component, B: Component, C: Component, F: QueryFilter>(
        &self,
        mut f: impl FnMut(Self::Entity, &A, &B, &C),
    ) {
        self.for_each_chunk3::<A, B, C, F>(|ids, a, b, c| {
            for (i, &id) in ids.iter().enumerate() {
                f(id, &a[i], &b[i], &c[i]);
            }
        });
    }

    /// 单组件可变行遍历（基于 `for_each_chunk_mut1` 的零开销内联展开）。
    #[inline]
    fn for_each_mut1<A: Component, F: QueryFilter>(
        &mut self,
        mut f: impl FnMut(Self::Entity, &mut A),
    ) {
        self.for_each_chunk_mut1::<A, F>(|ids, a| {
            for (i, &id) in ids.iter().enumerate() {
                f(id, &mut a[i]);
            }
        });
    }

    /// `1 写 + 1 读` 可变行遍历（基于 `for_each_chunk_mut2` 的零开销内联展开）。
    #[inline]
    fn for_each_mut2<A: Component, B: Component, F: QueryFilter>(
        &mut self,
        mut f: impl FnMut(Self::Entity, &mut A, &B),
    ) {
        self.for_each_chunk_mut2::<A, B, F>(|ids, a, b| {
            for (i, &id) in ids.iter().enumerate() {
                f(id, &mut a[i], &b[i]);
            }
        });
    }

    /// `2 写 + 1 读` 可变行遍历（基于 `for_each_chunk_mut3` 的零开销内联展开）。
    #[inline]
    fn for_each_mut3<A: Component, B: Component, C: Component, F: QueryFilter>(
        &mut self,
        mut f: impl FnMut(Self::Entity, &mut A, &mut B, &C),
    ) {
        self.for_each_chunk_mut3::<A, B, C, F>(|ids, a, b, c| {
            for (i, &id) in ids.iter().enumerate() {
                f(id, &mut a[i], &mut b[i], &c[i]);
            }
        });
    }
}

/// `StateSpace` 的兼容性别名。
pub use StateSpace as WorldEngine;

/// 分组布局重排统计报告。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayoutReport {
    pub changed: bool,
    pub selected_entities: usize,
    pub moved_entities: usize,
}
