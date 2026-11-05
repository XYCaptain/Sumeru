//! 两阶段批次提交与一致性契约引擎 (Two-Phase Batch Engine)

use crate::storage::core::store::{CorePoisonGuard, Store};
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::storage::allocator::EntityLocation;
use crate::storage::batch::{EntityBundle, EntityPatch, StoreBatch};
use serde::de::DeserializeOwned;
use serde::Serialize;

pub(crate) mod prepared;
pub(crate) mod scratch;
pub use prepared::PreparedBatch;
pub(crate) use prepared::{PlannedOp, PreparedCommitMode};
pub(crate) use crate::storage::core::column::PlannedInPlacePatch;
pub(crate) use scratch::{
    CommandEntries, CommandEntry, NativeDirectOps, NativeLane, NativeLanes,
    NativeType, PlanScratch,
};
use crate::storage::batch::BatchReceipt;
use crate::storage::ErasedComponentValue;
use crate::error::StateError;
use crate::id::ComponentTypeId;

pub(crate) mod commit;
pub(crate) mod planning;
pub(crate) mod prepare;
pub(crate) mod validation;

#[cfg(test)]
mod tests;

impl Store {
    /// 一次性创建实体及其初始携带的全部组件。
    ///
    /// 针对已存在原型的常见单实体创建场景，走直接预检与就地追加快路径（0 临时 Batch，0 计划容器）；
    /// 若目标原型尚不存在，安全回退到完整两阶段批次路径创建新原型。
    /// 若实体已存在，严格返回 `EntityAlreadyExists(id)`。
    pub fn create_entity(&mut self, id: EntityId, bundle: EntityBundle) -> Result<(), StateError> {
        self.check_poisoned()?;
        if bundle.components().is_empty()
            && !self.archetypes.is_empty()
            && self.archetypes[0].component_types().is_empty()
        {
            return self.spawn(id).map(|_| ());
        }
        // 1. 实体身份预检：严禁 nil 身份
        if id.is_nil() {
            return Err(StateError::AccessDenied(
                "Cannot install a nil entity identity".into(),
            ));
        }

        let target_arch_id = if bundle.components().is_empty()
            && !self.archetypes.is_empty()
            && self.archetypes[0].component_types().is_empty()
        {
            Some(0)
        } else if let Some(cached_id) = self.last_created_archetype {
            if (cached_id as usize) < self.archetypes.len() {
                let arch = &self.archetypes[cached_id as usize];
                if arch.component_types().len() == bundle.components().len()
                    && arch.component_types().iter().copied().eq(bundle.components().keys().copied())
                {
                    Some(cached_id)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let target_arch_id = match target_arch_id {
            Some(id) => Some(id),
            None => {
                let types: std::collections::BTreeSet<ComponentTypeId> =
                    bundle.components().keys().copied().collect();
                let found = self.archetype_index.get(&types).copied();
                if let Some(id) = found {
                    self.last_created_archetype = Some(id);
                }
                found
            }
        };

        if let Some(target_arch_id) = target_arch_id {
            let target_arch = &self.archetypes[target_arch_id as usize];
            // 2. 容量预检：检查原型行数上限与代际递增预检
            let row_usize = target_arch.len();
            if row_usize >= u32::MAX as usize {
                return Err(StateError::CapacityExhausted(
                    "Archetype rows exhausted".into(),
                ));
            }
            let next_epoch = self.next_epoch()?;

            // 2a. 平坦连续原型快路径：锁步校验与直写 flat_columns，免除 BTreeMap 查找与 flat_columns 重建
            if !target_arch.may_be_segmented_or_unaligned()
                && target_arch.flat_columns().len() == bundle.components().len()
            {
                for (entry, (_, value)) in target_arch
                    .flat_columns()
                    .iter()
                    .zip(bundle.components().iter())
                {
                    unsafe { (&*entry.col_ptr).validate_erased_for_push(value)? };
                }
                let local_slot = self.allocator.allocate_unique_with_location(
                    id,
                    EntityLocation::new(target_arch_id, row_usize as u32),
                )?;
                let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
                let target_arch = &mut self.archetypes[target_arch_id as usize];
                for (entry, (_, value)) in target_arch
                    .flat_columns()
                    .iter()
                    .zip(bundle.into_components())
                {
                    unsafe { (&mut *entry.col_ptr).push_erased_infallible(value) };
                }
                target_arch.bump_columns_version_in_place();
                let row = target_arch.push_entity_with_slot(id, local_slot) as u32;
                debug_assert_eq!(row, row_usize as u32);
                self.epoch = next_epoch;
                poison_guard.disarm();
                return Ok(());
            }

            if self.contains_entity(id) {
                return Err(StateError::EntityAlreadyExists(id));
            }
            self.allocator.validate_allocate(id)?;
            self.allocator.prepare_allocations(1)?;

            // 3. 列数据契约前置校验（不可变借用只读校验，零副作用）
            for (type_id, value) in bundle.components() {
                let col = target_arch
                    .columns()
                    .get(type_id)
                    .expect("column exists in archetype");
                col.validate_erased_for_push(value)?;
            }

            // 5. 内存容量物理预留（确保安装阶段彻底杜绝重分配与失败）
            self.archetypes[target_arch_id as usize].reserve(1);

            // ================== 进入不可失败的原子安装阶段 ==================
            let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
            let local_slot = self.allocator.allocate(id).expect("pre-validated allocation");
            let target_arch = &mut self.archetypes[target_arch_id as usize];
            for (type_id, value) in bundle.into_components() {
                target_arch
                    .columns_mut()
                    .get_mut(&type_id)
                    .expect("column exists in archetype")
                    .push_erased_infallible(value);
            }
            let row = target_arch.push_entity_with_slot(id, local_slot) as u32;
            target_arch.rebuild_flat_columns();
            self.allocator.set_location_by_slot_infallible(
                local_slot,
                EntityLocation::new(target_arch_id, row),
            );
            self.epoch = next_epoch;
            poison_guard.disarm();
            Ok(())
        } else {
            // 目标原型尚不存在，回退到批次路径安全初始化新原型与架构元数据
            let mut batch = StoreBatch::new();
            batch.create_entity(id, bundle)?;
            self.apply_batch(batch)?;
            if let Some(loc) = self.allocator.get_location(id) {
                self.last_created_archetype = Some(loc.archetype_id);
            }
            Ok(())
        }
    }


    /// 多组件原子更新补丁，未涉及的组件绝对保持原样。实体不存在报错 `EntityNotFound(id)`。
    pub fn patch_entity(
        &mut self,
        id: EntityId,
        patch: EntityPatch,
    ) -> Result<BatchReceipt, StateError> {
        let prepared = self.prepare_input((id, CommandEntry::Patch(patch)))?;
        self.commit_batch(prepared)
    }

    // =========================================================================
    // 单组件细分读写接口 (Single Component Operations)
    // =========================================================================

    /// 插入新组件：实体必须已存在；若实体已拥有该组件，严格返回 `ComponentAlreadyExists`
    #[inline]
    pub fn insert_component<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        entity: EntityId,
        component: T,
    ) -> Result<(), StateError> {
        let (local_slot, location) = self
            .allocator
            .local_slot_and_location(entity)
            .ok_or_else(|| {
                if self.is_poisoned() {
                    StateError::SessionPoisoned
                } else {
                    StateError::EntityNotFound(entity)
                }
            })?;
        self.insert_component_at_location(entity, Some(local_slot), location, component)?;
        Ok(())
    }

    /// 插入新类型擦除组件：实体必须已存在；若实体已拥有该组件，严格返回 `ComponentAlreadyExists`
    pub fn insert_component_erased(
        &mut self,
        entity: EntityId,
        component: ErasedComponentValue,
    ) -> Result<(), StateError> {
        let mut patch = EntityPatch::new();
        patch.insert_erased(component)?;
        self.patch_entity(entity, patch)?;
        Ok(())
    }


    /// Upsert 组件：若已有则替换并返回 Some(旧值)，若无则插入并返回 None
    #[inline]
    pub fn upsert_component<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        entity: EntityId,
        component: T,
    ) -> Result<Option<T>, StateError> {
        let (local_slot, location) = self
            .allocator
            .local_slot_and_location(entity)
            .ok_or_else(|| {
                if self.is_poisoned() {
                    StateError::SessionPoisoned
                } else {
                    StateError::EntityNotFound(entity)
                }
            })?;
        let (old, _) =
            self.upsert_component_at_location(entity, Some(local_slot), location, component)?;
        Ok(old)
    }

    /// Upsert 类型擦除组件：若已有则替换并返回 Some(旧值)，若无则插入并返回 None
    pub fn upsert_component_erased(
        &mut self,
        entity: EntityId,
        component: ErasedComponentValue,
    ) -> Result<Option<ErasedComponentValue>, StateError> {
        let type_id = component.component_type_id();
        if self.has_component(entity, type_id) {
            return self.replace_component_erased(entity, component).map(Some);
        }
        let mut patch = EntityPatch::new();
        patch.upsert_erased(component)?;
        self.patch_entity(entity, patch)?;
        Ok(None)
    }

    /// 为实体移除强类型组件：成功时移除并返回拥有所有权的旧值
    #[inline]
    pub fn remove_component<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        entity: EntityId,
    ) -> Result<T, StateError> {
        let (local_slot, location) = self
            .allocator
            .local_slot_and_location(entity)
            .ok_or_else(|| {
                if self.is_poisoned() {
                    StateError::SessionPoisoned
                } else {
                    StateError::EntityNotFound(entity)
                }
            })?;
        let (old, _) = self.remove_component_at_location::<T>(entity, Some(local_slot), location)?;
        Ok(old)
    }

    /// 为实体移除组件：成功时移除并返回拥有所有权的旧值
    pub fn remove_component_erased(
        &mut self,
        entity: EntityId,
        type_id: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        self.check_poisoned()?;
        let old = self
            .entity(entity)
            .and_then(|e| e.clone_erased(type_id).ok());
        let mut patch = EntityPatch::new();
        patch.remove_by_id(type_id, "Component")?;
        self.patch_entity(entity, patch)?;
        old.ok_or_else(|| StateError::ComponentNotFound(format!("{type_id}")))
    }

    /// Archive a logical batch using the registered column capabilities. This
    /// borrows private candidate values and never reads/re-captures Current.
    pub fn encode_batch(&self, batch: &crate::storage::batch::StoreBatch)
        -> Result<crate::storage::batch::EncodedStoreBatch, StateError>
    {
        self.check_poisoned()?;
        batch.encode_with_registry(&self.component_registry)
    }

    /// Validate all metadata, decode all columns privately, then return one
    /// native batch. Neither an early nor a late refusal modifies Current.
    pub fn decode_batch(&self, encoded: &crate::storage::batch::EncodedStoreBatch)
        -> Result<crate::storage::batch::StoreBatch, StateError>
    {
        self.check_poisoned()?;
        encoded.decode_with_registry(&self.component_registry)
    }

    /// Validate all metadata and decode a pure value-replacement batch directly
    /// into sorted `(EntityId, ErasedComponentValue)` pairs without constructing
    /// intermediate `EntityPatch` or `StoreBatch` maps.
    pub fn decode_value_batch(
        &self,
        encoded: &crate::storage::batch::EncodedStoreBatch,
    ) -> Result<Vec<(crate::EntityId, crate::storage::ErasedComponentValue)>, StateError> {
        self.check_poisoned()?;
        encoded.decode_values_with_registry(&self.component_registry)
    }

    /// Decode a pure value-replacement batch while preserving disjoint sorted entity runs
    /// when all column runs share the exact same entity layout.
    pub fn decode_value_batch_preserving_runs(
        &self,
        encoded: &crate::storage::batch::EncodedStoreBatch,
    ) -> Result<Vec<(crate::EntityId, crate::storage::ErasedComponentValue)>, StateError> {
        self.check_poisoned()?;
        encoded.decode_values_preserving_runs_with_registry(&self.component_registry)
    }

    /// Decode completely before invoking the existing single prepare/commit.
    /// Even an empty payload must pass the explicit format/version checks.
    pub fn apply_encoded_batch(&mut self, encoded: &crate::storage::batch::EncodedStoreBatch)
        -> Result<crate::storage::batch::BatchReceipt, StateError>
    {
        let batch = self.decode_batch(encoded)?;
        self.apply_batch(batch)
    }

}
