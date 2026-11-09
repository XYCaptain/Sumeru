use super::*;
use super::sealed;
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use crate::storage::archetype::Archetype;
use crate::storage::column::ColumnDescriptor;
use crate::storage::store::Store;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::marker::PhantomData;
use std::ops::Range;


mod builder;
mod select;
mod shape;

pub use builder::*;
pub use select::*;
pub use shape::*;

/// 提取查询行元素中的纯组件元组部分（剥离外层 `EntityId`），以支持 `.for_each(|(a, b)| ...)` 直接闭包类型推断。
pub trait QueryRowItem {
    /// 剥离 `EntityId` 后的组件借用或元组类型。
    type Comps;
    /// 提取组件元组。
    fn into_comps(self) -> Self::Comps;
}

impl<C> QueryRowItem for (EntityId, C) {
    type Comps = C;

    #[inline(always)]
    fn into_comps(self) -> C {
        self.1
    }
}

/// 闭包算子标记类型（用于 `|(pos, vel)| ...`）。
pub enum ClosureOpMarker {}

/// `#[gpu]` 内核算子标记类型（用于直接传入 `integrate_sim_motion`）。
pub enum KernelOpMarker {}

/// CPU 单线程遍历算子统一抽象（支持闭包与 `#[gpu]` 算子 `op`）。
pub trait QueryForEachOp<Item, Marker = ClosureOpMarker> {
    /// 对单行元素执行算子。
    fn call_row(&mut self, item: Item);
}

impl<Comp, F: FnMut(Comp)> QueryForEachOp<(EntityId, Comp), ClosureOpMarker> for F {
    #[inline(always)]
    fn call_row(&mut self, (_entity, comp): (EntityId, Comp)) {
        (self)(comp);
    }
}

/// CPU 多核并行 (`par_for_each` / `par_run`) 算子统一抽象（支持 `Fn` 闭包与 `#[gpu]` 算子）。
pub trait QueryParForEachOp<Item, Marker = ClosureOpMarker>: Sync + Send {
    /// 在并行工作线程上对单行元素无锁执行算子。
    fn call_row_par(&self, item: Item);
}

impl<Comp, F: Fn(Comp) + Sync + Send> QueryParForEachOp<(EntityId, Comp), ClosureOpMarker> for F {
    #[inline(always)]
    fn call_row_par(&self, (_entity, comp): (EntityId, Comp)) {
        (self)(comp);
    }
}

