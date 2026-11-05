//! 实体销毁与批量生命周期管理 (Entity Despawn & Lifecycle)

use super::Store;
use crate::storage::allocator::EntityLocation;
use crate::storage::archetype::Archetype;
use crate::id::ArchetypeId;
use crate::storage::ErasedComponentValue;
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::BTreeSet;



impl Store {
    /// 按指定实体 ID 在空组件原型（Archetype 0）中创建空实体。
    #[inline]
    pub fn spawn_with_id(&mut self, id: EntityId) -> Result<EntityId, StateError> {
        let next_epoch = self.next_epoch()?;
        let row = u32::try_from(self.archetypes[0].len()).map_err(|_| {
            StateError::CapacityExhausted("Archetype row capacity exceeds u32".into())
        })?;
        let loc = EntityLocation::new(0, row);
        let slot = self.allocator.allocate_with_location(id, loc)?;
        self.archetypes[0].push_entity_with_slot(id, slot);
        self.epoch = next_epoch;
        Ok(id)
    }

    /// 计算下一个不冲突的自动分配代际实体 ID（纯计算，不修改分配器状态）。
    #[inline]
    pub(crate) fn next_auto_entity_id(&self) -> Result<EntityId, StateError> {
        let (slot, generation) =
            if let Some(&free_slot) = self.allocator.snapshot_free_slots().last() {
                let slot_gen = self.allocator.slot_generation(free_slot).unwrap_or(1);
                (free_slot, slot_gen)
            } else {
                let slot = u32::try_from(self.allocator.total_slots())
                    .map_err(|_| StateError::CapacityExhausted("Entity slots exhausted".into()))?;
                (slot, 1u32)
            };
        let low = (slot.wrapping_add(1)) as u64;
        let id = if self.allocator.has_no_custom_identities() {
            EntityId::from_u128((((generation as u64) << 32) | low) as u128)
        } else {
            let mut high = generation as u64;
            loop {
                let packed = (high << 32) | low;
                let candidate = EntityId::from_u128(packed as u128);
                if !self.is_alive(candidate) {
                    break candidate;
                }
                high = high.wrapping_add(1).max(1);
            }
        };
        Ok(id)
    }

    /// 创建新实体并一步写入目标原型（0 次中间迁移）：
    /// - `store.spawn(())`：自动分配实体 ID 并创建空实体；
    /// - `store.spawn((Pos, Vel))`：自动分配实体 ID 并一步写入 `(Pos, Vel)` 原型；
    /// - `store.spawn(bundle)`：自动分配实体 ID 并写入 `EntityBundle`；
    /// - `store.spawn(id)`：在显式指定的 `EntityId` 上创建空实体。
    #[inline]
    pub fn spawn<B: crate::storage::entity::SpawnBundle>(
        &mut self,
        bundle: B,
    ) -> Result<EntityId, StateError> {
        bundle.spawn_in_store(self)
    }

    /// 自动分配不冲突的代际编码实体 ID 并在空组件原型中生成实体。
    #[inline]
    pub fn spawn_auto(&mut self) -> Result<EntityId, StateError> {
        let id = self.next_auto_entity_id()?;
        self.spawn_with_id(id)
    }

    /// 在指定实体 ID 位置幂等生成空实体（若该 ID 已有活跃实体则先销毁旧实体再重新生成）。
    #[inline]
    pub fn spawn_at(&mut self, id: EntityId) -> Result<EntityId, StateError> {
        if self.contains_entity(id) {
            self.delete_entity(id)?;
        }
        self.spawn_with_id(id)
    }

    /// 销毁并回收实体
    pub fn despawn(&mut self, id: EntityId) -> Result<(), StateError> {
        self.delete_entity(id)
    }

    /// 检查实体是否活跃
    #[inline]
    pub fn is_alive(&self, id: EntityId) -> bool {
        self.allocator.is_alive(id)
    }

    /// 获取活跃实体总数
    #[inline]
    pub fn active_entities(&self) -> usize {
        self.allocator.active_count()
    }

    /// 获取活跃实体总数（别名）
    #[inline]
    pub fn active_count(&self) -> usize {
        self.allocator.active_count()
    }

    /// 获取或创建指定的 Archetype
    pub(crate) fn get_or_create_archetype(&mut self, types: BTreeSet<ComponentTypeId>) -> u32 {
        if let Some(&id) = self.archetype_index.get(&types) {
            return id;
        }
        self.check_archetype_set_generation_capacity()
            .expect("archetype set generation should not overflow");
        let id = self.archetypes.len() as u32;
        let arch = Archetype::new(ArchetypeId(id), types.clone());
        self.archetypes.push(arch);
        self.archetype_index.insert(types, id);
        self.bump_archetype_set_generation()
            .expect("archetype set generation should not overflow");
        id
    }

    /// 兼容旧式单组件写入接口（支持 Upsert 语义）
    pub fn insert_component_value(
        &mut self,
        entity: EntityId,
        component: ErasedComponentValue,
    ) -> Result<(), StateError> {
        self.upsert_component_erased(entity, component).map(|_| ())
    }



    /// 强类型只读查询实体的组件 (单实体点查，建议优先使用 `store.entity(id)?.get::<T>()`)
    #[inline]
    pub fn get<T: ModelComponent + Serialize + DeserializeOwned>(
        &self,
        entity: EntityId,
    ) -> Option<&T> {
        let location = self.allocator.get_location(entity)?;
        self.archetypes
            .get(location.archetype_id as usize)?
            .get_column::<T>()?
            .get(location.row_index as usize)
    }

    /// Resolve a stable identity afresh and verify its component row mapping.
    pub(crate) fn resolve_component_location(
        &self,
        entity: EntityId,
        component_type_id: ComponentTypeId,
    ) -> Result<EntityLocation, StateError> {
        let location = self
            .allocator
            .get_location(entity)
            .ok_or_else(|| StateError::EntityNotFound(entity))?;
        self.validate_component_location(entity, component_type_id, location)
    }

    /// Validate a location resolved during the current store borrow. This is not
    /// a persistent location cache: callers must resolve again after migration.
    /// Keep the same layout, native column and row checks as identity-based
    /// replacement, but do not look up the UUID twice in one point operation.
    pub(crate) fn validate_component_location(
        &self,
        entity: EntityId,
        component_type_id: ComponentTypeId,
        location: EntityLocation,
    ) -> Result<EntityLocation, StateError> {
        self.validate_component_location_with_name(entity, component_type_id, location, |name| {
            ComponentTypeId::from_name(name) == component_type_id
        })
    }

    /// Share the complete preflight; typed callers can establish the name/hash
    /// relation by equality with their known type name before hashing it again.
    fn validate_component_location_with_name(
        &self,
        entity: EntityId,
        component_type_id: ComponentTypeId,
        location: EntityLocation,
        matches_name: impl FnOnce(&'static str) -> bool,
    ) -> Result<EntityLocation, StateError> {
        let archetype = self
            .archetypes
            .get(location.archetype_id as usize)
            .ok_or_else(|| {
                StateError::AccessDenied(format!("Entity {entity} references an unknown archetype"))
            })?;
        if archetype.entities().get(location.row_index as usize) != Some(&entity) {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype row"
            )));
        }
        if archetype.id().0 != location.archetype_id
            || self.archetype_index.len() != self.archetypes.len()
            || archetype.columns().len() != archetype.component_types().len()
        {
            return Err(StateError::ComponentNotFoundInEntity {
                entity,
                type_name: format!("{component_type_id}"),
            });
        }
        let Some(column) = archetype.columns().get(&component_type_id) else {
            if archetype.component_types().contains(&component_type_id) {
                return Err(StateError::ComponentNotFound(format!("{component_type_id}")));
            }
            return Err(StateError::ComponentNotFoundInEntity {
                entity,
                type_name: format!("{component_type_id}"),
            });
        };
        if column.len() != archetype.len()
            || column.as_ref().type_id() != component_type_id
            || !matches_name(column.type_name())
            || column.rust_type_id().is_none()
        {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent component column"
            )));
        }
        Ok(location)
    }

    /// Clone one native component for read/prepare operations, without encoding.
    pub fn clone_component_value(
        &self,
        entity: EntityId,
        component_type_id: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        let location = self.resolve_component_location(entity, component_type_id)?;
        self.archetypes[location.archetype_id as usize].columns()[&component_type_id]
            .clone_erased(location.row_index as usize)
    }

    /// Explicit restore boundary: decode using the installed native column prototype.
    pub fn decode_component_value(
        &self,
        entity: EntityId,
        component_type_id: ComponentTypeId,
        bytes: &[u8],
    ) -> Result<ErasedComponentValue, StateError> {
        let location = self.resolve_component_location(entity, component_type_id)?;
        self.archetypes[location.archetype_id as usize].columns()[&component_type_id]
            .decode_erased(bytes)
    }

    /// Validate a replacement by stable entity identity and both component types.
    pub fn validate_component_value(
        &self,
        entity: EntityId,
        component_type_id: ComponentTypeId,
        value: &ErasedComponentValue,
    ) -> Result<(), StateError> {
        let location = self.resolve_component_location(entity, component_type_id)?;
        self.archetypes[location.archetype_id as usize].columns()[&component_type_id]
            .validate_erased(location.row_index as usize, value)
    }

    /// Replace one typed value, returning the previous value for stable-ID undo.
    pub fn replace_component_value(
        &mut self,
        entity: EntityId,
        component_type_id: ComponentTypeId,
        value: ErasedComponentValue,
    ) -> Result<ErasedComponentValue, StateError> {
        let location = self.resolve_component_location(entity, component_type_id)?;
        self.archetypes[location.archetype_id as usize].columns()[&component_type_id]
            .validate_erased(location.row_index as usize, &value)?;
        let next_epoch = self.next_epoch()?;
        let old = self.archetypes[location.archetype_id as usize]
            .columns_mut()
            .get_mut(&component_type_id)
            .expect("validated component column")
            .replace_erased_infallible(location.row_index as usize, value);
        self.epoch = next_epoch;
        Ok(old)
    }

    /// Apply after successful validation and without intervening structure/type changes.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn replace_component_value_infallible(
        &mut self,
        entity: EntityId,
        component_type_id: ComponentTypeId,
        value: ErasedComponentValue,
    ) -> ErasedComponentValue {
        let location = self
            .resolve_component_location(entity, component_type_id)
            .expect("validate_component_value must precede replace_component_value_infallible");
        self.archetypes[location.archetype_id as usize]
            .columns_mut()
            .get_mut(&component_type_id)
            .expect("validated component column")
            .replace_erased_infallible(location.row_index as usize, value)
    }



}
