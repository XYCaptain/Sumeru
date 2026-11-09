//! 快照序列化与受检还原 (Snapshot & Restore)

use super::Store;
use super::CorePoisonGuard;
use crate::storage::allocator::{EntityAllocator, EntityLocation};
use crate::storage::archetype::Archetype;
use crate::id::ArchetypeId;
use crate::storage::column::{AnyColumn, ArchivedColumnData};
use crate::binding::EntityId;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod binary;

/// 当前存储快照格式版本
pub const CURRENT_SNAPSHOT_VERSION: u32 = 1;

fn default_snapshot_version() -> u32 {
    CURRENT_SNAPSHOT_VERSION
}

/// Store 结构快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnarStoreSnapshot {
    #[serde(default = "default_snapshot_version")]
    pub(crate) format_version: u32,
    pub(crate) allocator: EntityAllocator,
    pub(crate) archetypes_data: Vec<ArchetypeSnapshotData>,
}

impl ColumnarStoreSnapshot {
    /// 获取快照格式版本号
    #[inline]
    pub fn format_version(&self) -> u32 {
        self.format_version
    }
    pub fn len(&self) -> usize {
        self.allocator.active_count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn entities(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.archetypes_data
            .iter()
            .flat_map(|a| a.entities.iter().copied())
    }
    /// Logical component sets only; no physical row or allocator handles.
    pub fn component_sets(&self) -> impl Iterator<Item = (&[EntityId], &[ComponentTypeId])> + '_ {
        self.archetypes_data
            .iter()
            .map(|a| (a.entities.as_slice(), a.component_types.as_slice()))
    }

    /// 仅提取包含目标组件 `cid` 的原型实体列表与归档列数据（跳过所有无关组件列）。
    pub fn extract_component_archived(
        &self,
        cid: ComponentTypeId,
    ) -> Vec<(&[EntityId], &ArchivedColumnData)> {
        self.archetypes_data
            .iter()
            .filter_map(|arch| {
                arch.columns
                    .get(&cid)
                    .map(|col| (arch.entities.as_slice(), col))
            })
            .collect()
    }

    /// 在快照基线上仅针对单一组件（由 `registration` 指定）重放一组 L2 `EncodedStoreBatch`（正向或逆向），
    /// 返回按 `EntityId` 升序排列的存活 `(EntityId, ErasedComponentValue)` 集合及实际重放的批次数。
    ///
    /// - **表级裁剪**：完全跳过快照与增量批次中所有非目标组件 `cid` 的列数据反序列化；
    /// - **表级 Delta 短路**：若某个 `EncodedStoreBatch` 未触碰 `cid`（`!batch.affects_component(cid)`），直接跳过。
    pub fn replay_component_with_registration(
        &self,
        registration: &crate::storage::column::ComponentRegistration,
        batches: &[crate::storage::batch::EncodedStoreBatch],
    ) -> Result<(Vec<(EntityId, crate::storage::ErasedComponentValue)>, usize), StateError> {
        let proto = &registration.column;
        let cid = proto.type_id();
        let mut state_map: BTreeMap<EntityId, crate::storage::ErasedComponentValue> =
            BTreeMap::new();

        for arch in &self.archetypes_data {
            let Some(archived_col) = arch.columns.get(&cid) else {
                continue;
            };
            if archived_col.row_count as usize != arch.entities.len() {
                return Err(StateError::AccessDenied(format!(
                    "Snapshot component {cid} row count does not match its archetype"
                )));
            }
            let mut col = proto.empty_clone();
            col.restore_archived_column(archived_col)?;
            for i in (0..arch.entities.len()).rev() {
                let entity_id = arch.entities[i];
                let val = col.swap_remove_erased(col.len() - 1);
                state_map.insert(entity_id, val);
            }
        }

        let mut applied_batches = 0usize;
        for batch in batches {
            if !batch.affects_component(cid) {
                continue;
            }
            applied_batches += 1;
            for &del_id in &batch.deleted_entities {
                state_map.remove(&del_id);
            }
            for &(rem_id, rem_cid) in &batch.removed_components {
                if rem_cid == cid {
                    state_map.remove(&rem_id);
                }
            }
            for run in &batch.column_runs {
                if run.archived.component_type_id != cid {
                    continue;
                }
                let entities = run.entities_slice();
                if entities.len() != run.archived.row_count as usize {
                    return Err(StateError::CodecError(
                        "Encoded batch column row count mismatch".into(),
                    ));
                }
                let mut col = proto.empty_clone();
                col.restore_archived_column(&run.archived)?;
                for i in (0..entities.len()).rev() {
                    let entity_id = entities[i];
                    let val = col.swap_remove_erased(col.len() - 1);
                    state_map.insert(entity_id, val);
                }
            }
        }

        Ok((state_map.into_iter().collect(), applied_batches))
    }

    /// 在快照基线上仅针对强类型组件 `T` 重放一组 L2 `EncodedStoreBatch`，
    /// 返回按 `EntityId` 升序排列的 `(EntityId, T)` 列表及实际重放的批次数。
    pub fn replay_component_typed<T>(
        &self,
        batches: &[crate::storage::batch::EncodedStoreBatch],
    ) -> Result<(Vec<(EntityId, T)>, usize), StateError>
    where
        T: crate::storage::column::ArchiveComponent + Clone,
    {
        let reg = crate::storage::column::ComponentRegistration::of::<T>();
        let (erased, applied_batches) = self.replay_component_with_registration(&reg, batches)?;
        let mut typed = Vec::with_capacity(erased.len());
        for (entity_id, val) in erased {
            let item = val
                .downcast_ref::<T>()
                .ok_or_else(|| StateError::TypeMismatch {
                    expected: std::any::type_name::<T>().into(),
                    actual: "erased component value".into(),
                })?
                .clone();
            typed.push((entity_id, item));
        }
        Ok((typed, applied_batches))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ArchetypeSnapshotData {
    pub archetype_id: u32,
    pub component_types: Vec<ComponentTypeId>,
    #[serde(with = "crate::storage::core::entity_index::entity_slice_serde")]
    pub entities: Vec<EntityId>,
    pub columns: BTreeMap<ComponentTypeId, ArchivedColumnData>,
}



impl Store {
    #[inline]
    pub fn take_snapshot(&self) -> Result<ColumnarStoreSnapshot, StateError> {
        self.snapshot()
    }

    pub fn snapshot(&self) -> Result<ColumnarStoreSnapshot, StateError> {
        let mut archetypes_data = Vec::new();

        for arch in &self.archetypes {
            let mut columns = BTreeMap::new();
            for (&type_id, col) in arch.columns().iter() {
                let archived_col = col.archive_column()?;
                columns.insert(type_id, archived_col);
            }

            archetypes_data.push(ArchetypeSnapshotData {
                archetype_id: arch.id().0,
                component_types: arch.component_types().iter().copied().collect(),
                entities: arch.entities().to_vec(),
                columns,
            });
        }

        Ok(ColumnarStoreSnapshot {
            format_version: CURRENT_SNAPSHOT_VERSION,
            allocator: self.allocator.clone(),
            archetypes_data,
        })
    }

    /// 从全量快照中恢复状态
    pub fn restore_snapshot(&mut self, snapshot: &ColumnarStoreSnapshot) -> Result<(), StateError> {
        self.check_poisoned()?;
        if snapshot.format_version == 0 || snapshot.format_version > CURRENT_SNAPSHOT_VERSION {
            return Err(StateError::CodecError(format!(
                "Unsupported snapshot format version: {} (current supported version is {})",
                snapshot.format_version, CURRENT_SNAPSHOT_VERSION
            )));
        }
        self.next_epoch()?;
        snapshot.allocator.validate()?;
        if snapshot.archetypes_data.is_empty() {
            return Err(StateError::AccessDenied(
                "Snapshot is missing the empty archetype".into(),
            ));
        }

        let mut column_prototypes: BTreeMap<ComponentTypeId, Box<dyn AnyColumn>> = BTreeMap::new();
        for &type_id in self.component_registry.keys() {
            if let Some(proto) = self.component_registry.spawn_column(type_id) {
                column_prototypes.insert(type_id, proto);
            }
        }

        let mut restored_archetypes = Vec::with_capacity(snapshot.archetypes_data.len());
        let mut restored_index = BTreeMap::new();
        let mut total_entities = 0usize;

        for (index, arch_data) in snapshot.archetypes_data.iter().enumerate() {
            if usize::try_from(arch_data.archetype_id).ok() != Some(index) {
                return Err(StateError::AccessDenied(
                    "Snapshot archetype IDs must match their unique physical positions".into(),
                ));
            }
            let component_types: BTreeSet<ComponentTypeId> =
                arch_data.component_types.iter().copied().collect();
            if component_types.len() != arch_data.component_types.len()
                || (index == 0 && !component_types.is_empty())
                || restored_index.contains_key(&component_types)
                || arch_data
                    .columns
                    .keys()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    != component_types
            {
                return Err(StateError::AccessDenied(format!(
                    "Snapshot archetype {index} has duplicate or mismatched component columns"
                )));
            }
            let mut arch =
                Archetype::new(ArchetypeId(arch_data.archetype_id), component_types.clone());

            let mut slot_hint = 0usize;
            for (row, &entity) in arch_data.entities.iter().enumerate() {
                let row = u32::try_from(row).map_err(|_| {
                    StateError::CapacityExhausted(
                        "Snapshot archetype row capacity exceeds u32".into(),
                    )
                })?;
                if snapshot
                    .allocator
                    .resolve_with_hint(entity, &mut slot_hint)
                    .map(|(_, loc)| loc)
                    != Some(EntityLocation::new(arch_data.archetype_id, row))
                {
                    return Err(StateError::AccessDenied(format!(
                        "Snapshot entity {entity} has a duplicate or inconsistent identity/slot/row mapping"
                    )));
                }
                arch.push_entity(entity);
            }
            total_entities = total_entities
                .checked_add(arch_data.entities.len())
                .ok_or_else(|| {
                    StateError::CapacityExhausted("Snapshot total entity count overflow".into())
                })?;

            for (&type_id, archived_col) in &arch_data.columns {
                if archived_col.row_count as usize != arch_data.entities.len() {
                    return Err(StateError::AccessDenied(format!(
                        "Snapshot component {type_id} row count does not match its archetype"
                    )));
                }
                let proto = column_prototypes.get(&type_id).ok_or_else(|| {
                    StateError::ComponentNotFound(format!(
                        "No registered column prototype for {type_id}"
                    ))
                })?;
                let mut col = proto.empty_clone();
                col.restore_archived_column(archived_col)?;
                if col.len() != arch_data.entities.len() || col.len() != archived_col.row_count as usize {
                    return Err(StateError::AccessDenied(format!(
                        "Restored column {type_id} length {} does not match header row_count {} or archetype entities {}",
                        col.len(), archived_col.row_count, arch_data.entities.len()
                    )));
                }
                arch.columns_mut().insert(type_id, col);
            }

            restored_archetypes.push(arch);
            restored_index.insert(component_types, arch_data.archetype_id);
        }

        if total_entities != snapshot.allocator.active_count() {
            return Err(StateError::AccessDenied(
                "Snapshot allocator identities and archetype rows are not bijective".into(),
            ));
        }

        let next_epoch = self.next_epoch()?;
        self.check_archetype_set_generation_capacity()?;

        let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
        self.allocator = snapshot.allocator.clone();
        self.archetypes = restored_archetypes;
        self.archetype_index = restored_index;
        self.normalize_archetypes_to_registry(Some(next_epoch));
        self.rebuild_component_archetypes_on_topology_reset();
        self.epoch = next_epoch;
        self.bump_archetype_set_generation()?;
        poison_guard.disarm();
        Ok(())
    }

    /// 从全量快照中按值移动恢复状态（消除大规模 allocator 与实体列表克隆）
    pub fn restore_snapshot_owned(&mut self, snapshot: ColumnarStoreSnapshot) -> Result<(), StateError> {
        self.check_poisoned()?;
        if snapshot.format_version == 0 || snapshot.format_version > CURRENT_SNAPSHOT_VERSION {
            return Err(StateError::CodecError(format!(
                "Unsupported snapshot format version: {} (current supported version is {})",
                snapshot.format_version, CURRENT_SNAPSHOT_VERSION
            )));
        }
        self.next_epoch()?;
        snapshot.allocator.validate()?;
        if snapshot.archetypes_data.is_empty() {
            return Err(StateError::AccessDenied(
                "Snapshot is missing the empty archetype".into(),
            ));
        }

        let mut column_prototypes: BTreeMap<ComponentTypeId, Box<dyn AnyColumn>> = BTreeMap::new();
        for &type_id in self.component_registry.keys() {
            if let Some(proto) = self.component_registry.spawn_column(type_id) {
                column_prototypes.insert(type_id, proto);
            }
        }

        let mut restored_archetypes = Vec::with_capacity(snapshot.archetypes_data.len());
        let mut restored_index = BTreeMap::new();
        let mut total_entities = 0usize;

        for (index, arch_data) in snapshot.archetypes_data.into_iter().enumerate() {
            if usize::try_from(arch_data.archetype_id).ok() != Some(index) {
                return Err(StateError::AccessDenied(
                    "Snapshot archetype IDs must match their unique physical positions".into(),
                ));
            }
            let component_types: BTreeSet<ComponentTypeId> =
                arch_data.component_types.iter().copied().collect();
            if component_types.len() != arch_data.component_types.len()
                || (index == 0 && !component_types.is_empty())
                || restored_index.contains_key(&component_types)
                || arch_data
                    .columns
                    .keys()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    != component_types
            {
                return Err(StateError::AccessDenied(format!(
                    "Snapshot archetype {index} has duplicate or mismatched component columns"
                )));
            }
            let mut arch =
                Archetype::new(ArchetypeId(arch_data.archetype_id), component_types.clone());

            let entity_len = arch_data.entities.len();
            let mut slot_hint = 0usize;
            for (row, &entity) in arch_data.entities.iter().enumerate() {
                let row = u32::try_from(row).map_err(|_| {
                    StateError::CapacityExhausted(
                        "Snapshot archetype row capacity exceeds u32".into(),
                    )
                })?;
                if snapshot
                    .allocator
                    .resolve_with_hint(entity, &mut slot_hint)
                    .map(|(_, loc)| loc)
                    != Some(EntityLocation::new(arch_data.archetype_id, row))
                {
                    return Err(StateError::AccessDenied(format!(
                        "Snapshot entity {entity} has a duplicate or inconsistent identity/slot/row mapping"
                    )));
                }
            }
            *arch.entities_mut() = arch_data.entities;
            total_entities = total_entities
                .checked_add(entity_len)
                .ok_or_else(|| {
                    StateError::CapacityExhausted("Snapshot total entity count overflow".into())
                })?;

            for (&type_id, archived_col) in &arch_data.columns {
                if archived_col.row_count as usize != entity_len {
                    return Err(StateError::AccessDenied(format!(
                        "Snapshot component {type_id} row count does not match its archetype"
                    )));
                }
                let proto = column_prototypes.get(&type_id).ok_or_else(|| {
                    StateError::ComponentNotFound(format!(
                        "No registered column prototype for {type_id}"
                    ))
                })?;
                let mut col = proto.empty_clone();
                col.restore_archived_column(archived_col)?;
                if self.layout_optimization == crate::storage::LayoutOptimization::PageAlignedGpu
                    && !col.is_page_aligned()
                {
                    col.promote_to_page_aligned();
                }
                if col.len() != entity_len || col.len() != archived_col.row_count as usize {
                    return Err(StateError::AccessDenied(format!(
                        "Restored column {type_id} length {} does not match header row_count {} or archetype entities {}",
                        col.len(), archived_col.row_count, entity_len
                    )));
                }
                arch.columns_mut().insert(type_id, col);
            }

            restored_archetypes.push(arch);
            restored_index.insert(component_types, arch_data.archetype_id);
        }

        if total_entities != snapshot.allocator.active_count() {
            return Err(StateError::AccessDenied(
                "Snapshot allocator identities and archetype rows are not bijective".into(),
            ));
        }

        let next_epoch = self.next_epoch()?;
        self.check_archetype_set_generation_capacity()?;

        let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
        self.allocator = snapshot.allocator;
        self.archetypes = restored_archetypes;
        self.archetype_index = restored_index;
        self.normalize_archetypes_to_registry(Some(next_epoch));
        self.rebuild_component_archetypes_on_topology_reset();
        self.epoch = next_epoch;
        self.bump_archetype_set_generation()?;
        poison_guard.disarm();
        Ok(())
    }

}
