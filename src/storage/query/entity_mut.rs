//! 实体独占可变借用门面 (EntityMut)
//!
//! 独占借用底层的 `Store`，在维持严格借用检查与生命周期约束的前提下，
//! 为目标实体提供直观便捷的原位更新与组件级 CRUD 操作。
//!
//! ```compile_fail
//! // 负例 1：同一 Store 上无法同时存活两个独占可写句柄 EntityMut (E0499)
//! use sumeru::{Entity, EntityId, Store};
//! let mut store = Store::new();
//! let id1 = EntityId::from_uuid(uuid::Uuid::new_v4());
//! let id2 = EntityId::from_uuid(uuid::Uuid::new_v4());
//! store.insert(Entity::new(id1)).unwrap();
//! store.insert(Entity::new(id2)).unwrap();
//! let mut e1 = store.entity_mut(id1).unwrap();
//! let mut e2 = store.entity_mut(id2).unwrap();
//! let _ = (e1.id(), e2.id());
//! ```
//!
//! ```compile_fail
//! // 负例 2：从 EntityMut 派生的组件引用与后续写操作冲突 (E0502)
//! use sumeru::{Store, EntityId, Entity};
//! #[derive(Clone, serde::Serialize, serde::Deserialize)]
//! struct Pos(f64);
//! let mut store = Store::new();
//! let id = EntityId::from_uuid(uuid::Uuid::new_v4());
//! store.insert(Entity::new(id).with(Pos(1.0))).unwrap();
//! let mut entity = store.entity_mut(id).unwrap();
//! let pos_ref = entity.get::<Pos>().unwrap();
//! entity.set(Pos(2.0)).unwrap();
//! let _ = pos_ref.0;
//! ```
//!
//! ```compile_fail
//! // 负例 3：从 EntityMut 派生的 EntityRef 只读视图与后续结构写冲突 (E0502)
//! use sumeru::{Store, EntityId, Entity};
//! #[derive(Clone, serde::Serialize, serde::Deserialize)]
//! struct Pos(f64);
//! let mut store = Store::new();
//! let id = EntityId::from_uuid(uuid::Uuid::new_v4());
//! store.insert(Entity::new(id).with(Pos(1.0))).unwrap();
//! let mut entity = store.entity_mut(id).unwrap();
//! let entity_ref = entity.as_ref().unwrap();
//! entity.remove::<Pos>().unwrap();
//! let _ = entity_ref.has::<Pos>();
//! ```

use crate::storage::allocator::EntityLocation;
use super::entity_ref::EntityRef;
use crate::storage::store::Store;
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::{ComponentTypeId, LocalSlotId};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// 实体独占可变借用门面 (EntityMut)。
pub struct EntityMut<'a> {
    pub(crate) store: &'a mut Store,
    pub(crate) id: EntityId,
    pub(crate) local_slot: Option<LocalSlotId>,
    pub(crate) location: EntityLocation,
}

impl<'a> std::fmt::Debug for EntityMut<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EntityMut")
            .field("id", &self.id)
            .field("location", &self.location)
            .finish()
    }
}

impl<'a> EntityMut<'a> {
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn new(store: &'a mut Store, id: EntityId, location: EntityLocation) -> Self {
        Self {
            store,
            id,
            local_slot: None,
            location,
        }
    }

    #[inline]
    pub(crate) fn with_slot(
        store: &'a mut Store,
        id: EntityId,
        local_slot: LocalSlotId,
        location: EntityLocation,
    ) -> Self {
        Self {
            store,
            id,
            local_slot: Some(local_slot),
            location,
        }
    }

    /// 获取实体 ID
    #[inline]
    pub fn id(&self) -> EntityId {
        self.id
    }

    /// 为实体设置组件值：
    /// 1. 若实体已有该组件，直接原位替换（0 次全局查表、0 次 Archetype 迁移，复用统一替换内核）；
    /// 2. 若实体无该组件，结构性插入并自动刷新本地物理位置。
    #[inline]
    pub fn set<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        value: T,
    ) -> Result<Option<T>, StateError> {
        let (old, new_loc) =
            self.store
                .upsert_component_at_location(self.id, self.local_slot, self.location, value)?;
        self.location = new_loc;
        Ok(old)
    }

    /// 为实体移除强类型组件：成功时移除并返回拥有所有权的旧值，并自动刷新物理位置。
    #[inline]
    pub fn remove<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
    ) -> Result<T, StateError> {
        let (old, new_loc) =
            self.store
                .remove_component_at_location::<T>(self.id, self.local_slot, self.location)?;
        self.location = new_loc;
        Ok(old)
    }

    /// 原位闭包修改组件值（直接利用缓存物理位置直通访问原型列，0 次 allocator 寻址）。
    #[inline]
    pub fn modify<T: ModelComponent + Serialize + DeserializeOwned + 'static, R>(
        &mut self,
        f: impl FnOnce(&mut T) -> R,
    ) -> Result<R, StateError> {
        self.store.modify_at::<T, R>(self.id, self.location, f)
    }

    /// 原位多组件元组联合借用与修改（支持 1..16 元组宏，如 (Pos, Vel) 或 (Pos, Vel, Mass)）。
    ///
    /// 单次原型有效性检索、单次行对齐校验、单次代际推进、0 动态分配，直接传入多组件独占引用元组。
    #[inline]
    pub fn modify_tuple<T: for<'x> crate::storage::modify::ModifyTarget<'x>, R>(
        &mut self,
        f: impl FnOnce(<T as crate::storage::modify::sealed::ModifyFetch<'_>>::View) -> R,
    ) -> Result<R, StateError> {
        self.store.modify_tuple_at::<T, R>(self.id, self.location, f)
    }

    /// 为实体添加新组件：实体必须已存在且未拥有该组件，并自动刷新物理位置。
    #[inline]
    pub fn add<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        value: T,
    ) -> Result<(), StateError> {
        self.location =
            self.store
                .insert_component_at_location(self.id, self.local_slot, self.location, value)?;
        Ok(())
    }

    /// 为实体插入新组件（`add` 的兼容别名）：实体必须已存在且未拥有该组件，并自动刷新物理位置。
    #[inline]
    pub fn insert<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        value: T,
    ) -> Result<(), StateError> {
        self.add(value)
    }

    /// 替换实体的已有组件：直接利用缓存物理位置就地替换并返回旧值，0 次额外查表。
    #[inline]
    pub fn replace<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        value: T,
    ) -> Result<T, StateError> {
        self.store.replace_at(self.id, self.location, value)
    }

    /// 为当前实体设置关系对 `(R, target)`（校验目标实体 `target` 存活）。
    #[inline]
    pub fn set_pair<R: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        relation: R,
        target: EntityId,
    ) -> Result<Option<crate::al::Pair<R>>, StateError> {
        if !self.store.contains_entity(target) {
            return Err(StateError::EntityNotFound(target));
        }
        self.set(crate::al::Pair::new(relation, target))
    }

    /// 移除当前实体的关系对 `R`。
    #[inline]
    pub fn remove_pair<R: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
    ) -> Result<crate::al::Pair<R>, StateError> {
        self.remove::<crate::al::Pair<R>>()
    }

    /// 查询当前实体上关系 `R` 指向的目标实体 ID（若目标实体已销毁则返回 `None`）。
    #[inline]
    pub fn pair_target<R: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> Option<EntityId> {
        self.as_ref()?.pair_target::<R>()
    }

    /// 立即销毁当前实体并释放句柄。
    #[inline]
    pub fn despawn(self) -> Result<(), StateError> {
        self.store.delete_entity(self.id)
    }

    /// 立即销毁当前实体并释放句柄（`despawn` 的兼容别名）。
    #[inline]
    pub fn delete(self) -> Result<(), StateError> {
        self.despawn()
    }

    /// 获取实体的只读借用视图 (`EntityRef`)。
    #[inline]
    pub fn as_ref(&self) -> Option<EntityRef<'_>> {
        let arch = self.store.archetypes.get(self.location.archetype_id as usize)?;
        if arch.entities().get(self.location.row_index as usize) != Some(&self.id) {
            return None;
        }
        Some(EntityRef::new(self.store, self.id, self.location))
    }

    /// 检查实体是否拥有特定组件类型（直接检查当前物理原型，0 次全局查找）。
    #[inline]
    pub fn has<T: ModelComponent + 'static>(&self) -> bool {
        let type_id = ComponentTypeId::of::<T>();
        self.store
            .archetypes
            .get(self.location.archetype_id as usize)
            .map_or(false, |arch| arch.component_types_slice().contains(&type_id))
    }

    /// 检查实体在当前原型行上的非迁移状态位（`0..=255`）是否已置位（0 次全局查找）。
    #[inline]
    pub fn has_flag<const FLAG: u8>(&self) -> bool {
        self.store
            .archetypes
            .get(self.location.archetype_id as usize)
            .map_or(false, |arch| {
                arch.get_row_flag(self.location.row_index as usize, FLAG)
            })
    }

    /// 设置实体在当前原型行上的非迁移状态位（`0..=255`），不触发跨表迁移，返回修改前的旧状态值。
    #[inline]
    pub fn set_flag<const FLAG: u8>(&mut self, enabled: bool) -> Result<bool, StateError> {
        self.store.check_poisoned()?;
        let arch = self
            .store
            .archetypes
            .get_mut(self.location.archetype_id as usize)
            .ok_or(StateError::EntityNotFound(self.id))?;
        let row = self.location.row_index as usize;
        if arch.entities().get(row) != Some(&self.id) {
            return Err(StateError::EntityNotFound(self.id));
        }
        Ok(arch.set_row_flag(row, FLAG, enabled))
    }

    /// 获取强类型组件不可变引用（缺失或类型不匹配时返回 `None`，零堆分配、零拷贝、直通访问）。
    #[inline]
    pub fn get<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> Option<&T> {
        self.as_ref()?.get::<T>()
    }

    /// 获取强类型组件不可变引用，缺失或类型不匹配时返回详细的 `StateError`。
    #[inline]
    pub fn try_get<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> Result<&T, StateError> {
        self.as_ref()
            .ok_or(StateError::EntityNotFound(self.id))?
            .try_get::<T>()
    }

    /// 获取强类型组件可变引用（直接利用缓存物理位置直通访问原型列，自动推进代际，缺失时返回 `None`）。
    #[inline]
    pub fn get_mut<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
    ) -> Option<&mut T> {
        if self.store.check_poisoned().is_err() {
            return None;
        }
        let next_epoch = self.store.next_epoch().ok()?;
        let type_id = ComponentTypeId::of::<T>();
        let arch = self.store.archetypes.get_mut(self.location.archetype_id as usize)?;
        let row = self.location.row_index as usize;
        if arch.entities().get(row) != Some(&self.id) {
            return None;
        }
        let val_ref = arch
            .get_typed_column_row_mut_at_epoch::<T>(type_id, self.id, row, next_epoch)
            .ok()??;
        self.store.epoch = next_epoch;
        Some(val_ref)
    }

    /// 形式化诊断：获取强类型组件可变引用（缺失或类型不匹配时返回详细的 `StateError`，成功时自动推进代际）。
    #[inline]
    pub fn try_get_mut<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
    ) -> Result<&mut T, StateError> {
        self.store.check_poisoned()?;
        let next_epoch = self.store.next_epoch()?;
        let type_id = ComponentTypeId::of::<T>();
        let arch = self
            .store
            .archetypes
            .get_mut(self.location.archetype_id as usize)
            .ok_or(StateError::EntityNotFound(self.id))?;
        let row = self.location.row_index as usize;
        if arch.entities().get(row) != Some(&self.id) {
            return Err(StateError::EntityNotFound(self.id));
        }
        let val_ref = arch
            .get_typed_column_row_mut_at_epoch::<T>(type_id, self.id, row, next_epoch)?
            .ok_or_else(|| {
                StateError::AccessDenied(format!(
                    "Entity {} row {} out of bounds in archetype {}",
                    self.id, row, self.location.archetype_id
                ))
            })?;
        self.store.epoch = next_epoch;
        Ok(val_ref)
    }

    /// 联合借用提取多元组组件可选引用（任一组件缺失时返回 `None`，零错误对象分配）。
    #[inline]
    pub fn get_tuple<'b, E: super::entity_ref::EntityData<'b>>(&'b self) -> Option<E::Item> {
        self.as_ref()?.get_tuple::<E>()
    }

    /// 形式化诊断：联合借用提取多元组组件引用（任一组件缺失时返回 `StateError`）。
    #[inline]
    pub fn extract<'b, E: super::entity_ref::EntityData<'b>>(
        &'b self,
    ) -> Result<E::Item, StateError> {
        self.as_ref()
            .ok_or(StateError::EntityNotFound(self.id))?
            .extract::<E>()
    }
}

impl Store {
    /// 获取实体的独占可变借用门面 (`EntityMut`)。
    ///
    /// 若实体不存在返回 `StateError::EntityNotFound(id)`。
    #[inline]
    pub fn entity_mut(&mut self, id: EntityId) -> Result<EntityMut<'_>, StateError> {
        self.check_poisoned()?;
        let (slot, loc) = self
            .allocator
            .local_slot_and_location(id)
            .ok_or_else(|| StateError::EntityNotFound(id))?;
        Ok(EntityMut::with_slot(self, id, slot, loc))
    }
}
