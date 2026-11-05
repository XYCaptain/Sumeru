//! Archetype 原型表管理与 Swap-Remove 同步 (Archetype)

use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::{ArchetypeId, ComponentTypeId, LocalSlotId};
use crate::storage::column::{AnyColumn, TypedColumn};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::{BTreeMap, BTreeSet};

/// 与 `component_types_slice` 1:1 对齐的扁平列指针缓存项（消除 BTreeMap 节点追逐与虚表 downcast）
#[derive(Clone, Copy)]
pub(crate) struct FlatColumnEntry {
    pub type_id: ComponentTypeId,
    pub rust_type: Option<std::any::TypeId>,
    pub col_ptr: *mut dyn AnyColumn,
    pub move_row_fn: unsafe fn(*mut (), usize, *mut ()),
    pub swap_remove_fn: unsafe fn(*mut (), usize),
    pub push_erased_fn: unsafe fn(*mut (), &mut crate::storage::ErasedComponentValue),
}

// SAFETY: `col_ptr` 指向当前 `Archetype` 自身 `columns` 持有的列实例。
// 只读访问（`&Archetype`）仅将其视作不可变引用，可变访问要求持有 `&mut Archetype`。
unsafe impl Send for FlatColumnEntry {}
unsafe impl Sync for FlatColumnEntry {}

/// Archetype 结构体，持有具有相同组件集合的所有实体及其列数据
pub(crate) struct Archetype {
    id: ArchetypeId,
    component_types: BTreeSet<ComponentTypeId>,
    component_types_slice: Vec<ComponentTypeId>,
    entities: Vec<EntityId>,
    /// 与 `entities` 同步维护的行级代际槽位表（支持 swap_remove 尾部实体 O(1) 槽位反查；若外部直接修改 entities_mut 则自动清空回退）
    entity_slots: Vec<LocalSlotId>,
    columns: BTreeMap<ComponentTypeId, Box<dyn AnyColumn>>,
    /// 与 `component_types_slice` 1:1 对应的扁平列指针表
    flat_columns: smallvec::SmallVec<[FlatColumnEntry; 8]>,
    /// 标记列是否可能处于分段态、列长未对齐或扁平列指针缓存待重建
    may_be_segmented_or_unaligned: bool,
    /// 列容器结构与可变借用版本号（任何可能替换/移除/分段化 Box<dyn AnyColumn> 的操作均递增）
    columns_version: u64,
    /// 单组件插入原型转移边缓存 (ComponentTypeId, Rust TypeId -> (Target ArchetypeId, Target Column Index))
    add_edges: smallvec::SmallVec<[(ComponentTypeId, std::any::TypeId, u32, u16); 4]>,
    /// 单组件移除原型转移边缓存 (ComponentTypeId, Rust TypeId -> (Target ArchetypeId, Source Column Index))
    remove_edges: smallvec::SmallVec<[(ComponentTypeId, std::any::TypeId, u32, u16); 4]>,
}

#[allow(dead_code)]
impl Archetype {
    pub fn new(id: ArchetypeId, component_types: BTreeSet<ComponentTypeId>) -> Self {
        let component_types_slice = component_types.iter().copied().collect();
        let may_be_segmented_or_unaligned = !component_types.is_empty();
        Self {
            id,
            component_types,
            component_types_slice,
            entities: Vec::new(),
            entity_slots: Vec::new(),
            columns: BTreeMap::new(),
            flat_columns: smallvec::SmallVec::new(),
            may_be_segmented_or_unaligned,
            columns_version: 1,
            add_edges: smallvec::SmallVec::new(),
            remove_edges: smallvec::SmallVec::new(),
        }
    }

    pub fn empty_with_schema(&self) -> Self {
        let columns = self
            .columns
            .iter()
            .map(|(&type_id, col)| (type_id, col.empty_clone()))
            .collect();
        let mut arch = Self {
            id: self.id,
            component_types: self.component_types.clone(),
            component_types_slice: self.component_types_slice.clone(),
            entities: Vec::new(),
            entity_slots: Vec::new(),
            columns,
            flat_columns: smallvec::SmallVec::new(),
            may_be_segmented_or_unaligned: true,
            columns_version: 1,
            add_edges: self.add_edges.clone(),
            remove_edges: self.remove_edges.clone(),
        };
        arch.rebuild_flat_columns();
        arch
    }

    /// 重建扁平连续列指针表并核验列非分段与行长一致性
    pub(crate) fn rebuild_flat_columns(&mut self) {
        self.flat_columns.clear();
        let expected_len = self.entities.len();
        let mut dirty = self.columns.len() != self.component_types_slice.len();
        for (&type_id, col) in &mut self.columns {
            if col.is_segmented()
                || col.len() != expected_len
                || col.as_ref().type_id() != type_id
                || ComponentTypeId::from_name(col.type_name()) != type_id
                || col.rust_type_id().is_none()
            {
                dirty = true;
                break;
            }
            let (Some(move_row_fn), Some(swap_remove_fn), Some(push_erased_fn)) = (
                col.raw_move_row_direct_fn(),
                col.raw_swap_remove_direct_fn(),
                col.raw_push_erased_direct_fn(),
            ) else {
                dirty = true;
                break;
            };
            let rust_type = col.rust_type_id();
            let col_ptr: *mut dyn AnyColumn = col.as_mut();
            self.flat_columns.push(FlatColumnEntry {
                type_id,
                rust_type,
                col_ptr,
                move_row_fn,
                swap_remove_fn,
                push_erased_fn,
            });
        }
        if !dirty
            && !self
                .flat_columns
                .iter()
                .map(|e| e.type_id)
                .eq(self.component_types_slice.iter().copied())
        {
            dirty = true;
        }
        if dirty {
            self.flat_columns.clear();
        }
        self.may_be_segmented_or_unaligned = dirty;
    }

    #[inline(always)]
    pub(crate) fn may_be_segmented_or_unaligned(&self) -> bool {
        self.may_be_segmented_or_unaligned
    }

    #[inline(always)]
    pub(crate) fn flat_columns(&self) -> &[FlatColumnEntry] {
        &self.flat_columns
    }

    #[inline(always)]
    pub(crate) fn add_edge(
        &self,
        type_id: ComponentTypeId,
        rust_type: std::any::TypeId,
    ) -> Option<(u32, u16)> {
        self.add_edges
            .iter()
            .find_map(|&(cid, rtid, target, col_idx)| {
                (cid == type_id && rtid == rust_type).then_some((target, col_idx))
            })
    }

    #[inline]
    pub(crate) fn set_add_edge(
        &mut self,
        type_id: ComponentTypeId,
        rust_type: std::any::TypeId,
        target: u32,
        dst_col_idx: u16,
    ) {
        if let Some((_, existing_rtid, existing_target, existing_col)) =
            self.add_edges.iter_mut().find(|(cid, _, _, _)| *cid == type_id)
        {
            *existing_rtid = rust_type;
            *existing_target = target;
            *existing_col = dst_col_idx;
        } else {
            self.add_edges
                .push((type_id, rust_type, target, dst_col_idx));
        }
    }

    #[inline(always)]
    pub(crate) fn remove_edge(
        &self,
        type_id: ComponentTypeId,
        rust_type: std::any::TypeId,
    ) -> Option<(u32, u16)> {
        self.remove_edges
            .iter()
            .find_map(|&(cid, rtid, target, col_idx)| {
                (cid == type_id && rtid == rust_type).then_some((target, col_idx))
            })
    }

    #[inline]
    pub(crate) fn set_remove_edge(
        &mut self,
        type_id: ComponentTypeId,
        rust_type: std::any::TypeId,
        target: u32,
        src_col_idx: u16,
    ) {
        if let Some((_, existing_rtid, existing_target, existing_col)) =
            self.remove_edges.iter_mut().find(|(cid, _, _, _)| *cid == type_id)
        {
            *existing_rtid = rust_type;
            *existing_target = target;
            *existing_col = src_col_idx;
        } else {
            self.remove_edges
                .push((type_id, rust_type, target, src_col_idx));
        }
    }

    #[inline(always)]
    pub(crate) fn has_segmented_columns(&self) -> bool {
        if !self.may_be_segmented_or_unaligned {
            return false;
        }
        self.columns.values().any(|col| col.is_segmented())
    }

    #[inline]
    pub fn id(&self) -> ArchetypeId {
        self.id
    }

    #[inline]
    pub fn component_types(&self) -> &BTreeSet<ComponentTypeId> {
        &self.component_types
    }

    #[inline]
    pub fn component_types_slice(&self) -> &[ComponentTypeId] {
        &self.component_types_slice
    }

    #[inline]
    pub fn entities(&self) -> &[EntityId] {
        &self.entities
    }

    #[inline(always)]
    pub(crate) fn entity_slot(&self, row: usize) -> Option<LocalSlotId> {
        if self.entity_slots.len() == self.entities.len() {
            self.entity_slots.get(row).copied()
        } else {
            None
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    #[inline]
    pub(crate) fn columns(&self) -> &BTreeMap<ComponentTypeId, Box<dyn AnyColumn>> {
        &self.columns
    }

    #[inline]
    pub(crate) fn columns_version(&self) -> u64 {
        self.columns_version
    }

    #[inline]
    pub(crate) fn columns_mut(&mut self) -> &mut BTreeMap<ComponentTypeId, Box<dyn AnyColumn>> {
        self.columns_version = self.columns_version.wrapping_add(1);
        self.may_be_segmented_or_unaligned = true;
        self.flat_columns.clear();
        self.add_edges.clear();
        self.remove_edges.clear();
        &mut self.columns
    }

    /// 获取强类型列裸指针（供静态多元组插入缓存校验使用，不改变列集合结构）
    #[inline]
    pub(crate) fn raw_typed_column_ptr<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
    ) -> Option<*mut TypedColumn<T>> {
        let type_id = ComponentTypeId::of::<T>();
        let col = self.columns.get_mut(&type_id)?;
        if col.type_name() != std::any::type_name::<T>() || col.is_segmented() {
            return None;
        }
        let typed = col.as_any_mut().downcast_mut::<TypedColumn<T>>()?;
        Some(typed as *mut TypedColumn<T>)
    }

    /// 确保当前 Archetype 拥有源 Archetype 的所有列结构 (克隆空列)
    pub fn ensure_columns_from(&mut self, src: &Archetype) {
        self.columns_version = self.columns_version.wrapping_add(1);
        for (&type_id, col) in &src.columns {
            self.columns
                .entry(type_id)
                .or_insert_with(|| col.empty_clone());
            self.component_types.insert(type_id);
        }
        self.component_types_slice = self.component_types.iter().copied().collect();
        self.add_edges.clear();
        self.remove_edges.clear();
        self.rebuild_flat_columns();
    }

    /// 注册一列强类型组件存储
    pub fn register_column<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        type_name: &'static str,
    ) {
        self.columns_version = self.columns_version.wrapping_add(1);
        let type_id = ComponentTypeId::from_name(type_name);
        self.component_types.insert(type_id);
        self.component_types_slice = self.component_types.iter().copied().collect();
        self.columns
            .entry(type_id)
            .or_insert_with(|| Box::new(TypedColumn::<T>::new(type_name)));
        self.add_edges.clear();
        self.remove_edges.clear();
        self.rebuild_flat_columns();
    }

    /// 清空原型内所有实体与组件行数据，保留底层列容量与原型转移边缓存
    #[inline]
    pub(crate) fn clear_rows(&mut self) {
        self.columns_version = self.columns_version.wrapping_add(1);
        self.entities.clear();
        self.entity_slots.clear();
        for col in self.columns.values_mut() {
            col.clear_rows();
        }
        self.rebuild_flat_columns();
    }

    #[inline]
    pub(crate) fn entities_mut(&mut self) -> &mut Vec<EntityId> {
        self.entity_slots.clear();
        self.may_be_segmented_or_unaligned = true;
        &mut self.entities
    }

    /// 安全拆分实体列表切片与组件列可变映射，供连续可变查询使用
    #[inline]
    pub(crate) fn split_for_query_mut(
        &mut self,
    ) -> (
        &[EntityId],
        &[ComponentTypeId],
        &[FlatColumnEntry],
        bool,
        &mut BTreeMap<ComponentTypeId, Box<dyn AnyColumn>>,
    ) {
        self.columns_version = self.columns_version.wrapping_add(1);
        (
            &self.entities,
            &self.component_types_slice,
            &self.flat_columns,
            self.may_be_segmented_or_unaligned,
            &mut self.columns,
        )
    }

    #[inline]
    pub(crate) fn bump_columns_version_in_place(&mut self) {
        self.columns_version = self.columns_version.wrapping_add(1);
    }

    /// 向当前 Archetype 添加一个实体
    #[inline]
    pub fn push_entity(&mut self, entity: EntityId) -> usize {
        self.entity_slots.clear();
        let row = self.entities.len();
        self.entities.push(entity);
        row
    }

    /// 向当前 Archetype 同步添加实体及其代际槽位
    #[inline]
    pub(crate) fn push_entity_with_slot(&mut self, entity: EntityId, slot: LocalSlotId) -> usize {
        let row = self.entities.len();
        if self.entity_slots.len() == row {
            self.entity_slots.push(slot);
        } else {
            self.entity_slots.clear();
        }
        self.entities.push(entity);
        row
    }

    /// 仅对实体与行槽位执行同步 swap_remove（由已完成全列同步迁移的受控路径调用）
    #[inline]
    pub(crate) fn swap_remove_entity_only(&mut self, row: usize) {
        if self.entity_slots.len() == self.entities.len() {
            self.entity_slots.swap_remove(row);
        } else {
            self.entity_slots.clear();
        }
        self.entities.swap_remove(row);
    }

    /// 获取强类型组件列不可变引用
    #[inline]
    pub fn get_column<T: ModelComponent + Serialize + DeserializeOwned>(
        &self,
    ) -> Option<&TypedColumn<T>> {
        let type_id = ComponentTypeId::of::<T>();
        if !self.may_be_segmented_or_unaligned
            && self.flat_columns.len() == self.component_types_slice.len()
        {
            let pos = self
                .component_types_slice
                .iter()
                .position(|&id| id == type_id)?;
            let entry = unsafe { self.flat_columns.get_unchecked(pos) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let typed_ptr = entry.col_ptr as *const () as *const TypedColumn<T>;
                return Some(unsafe { &*typed_ptr });
            }
            return None;
        }
        self.columns
            .get(&type_id)
            .and_then(|col| col.as_any().downcast_ref::<TypedColumn<T>>())
    }

    /// 获取强类型组件列可变引用（保持 flat_columns 缓存有效，因为不改变列集合与 Box 地址）
    #[inline]
    pub fn get_column_mut<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
    ) -> Option<&mut TypedColumn<T>> {
        let type_id = ComponentTypeId::of::<T>();
        if !self.may_be_segmented_or_unaligned
            && self.flat_columns.len() == self.component_types_slice.len()
        {
            let pos = self
                .component_types_slice
                .iter()
                .position(|&id| id == type_id)?;
            let entry = unsafe { *self.flat_columns.get_unchecked(pos) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                self.columns_version = self.columns_version.wrapping_add(1);
                let typed_ptr = entry.col_ptr as *mut () as *mut TypedColumn<T>;
                return Some(unsafe { &mut *typed_ptr });
            }
        }
        self.columns_version = self.columns_version.wrapping_add(1);
        self.columns
            .get_mut(&type_id)
            .and_then(|c| c.as_any_mut().downcast_mut::<TypedColumn<T>>())
    }

    /// 按 ComponentTypeId 获取强类型组件列可变引用，并校验类型一致性
    #[inline]
    pub fn get_typed_column_mut<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        cid: ComponentTypeId,
        entity: EntityId,
    ) -> Result<&mut TypedColumn<T>, StateError> {
        if !self.may_be_segmented_or_unaligned
            && self.flat_columns.len() == self.component_types_slice.len()
        {
            let pos = self
                .component_types_slice
                .iter()
                .position(|&id| id == cid)
                .ok_or_else(|| StateError::ComponentNotFoundInEntity {
                    entity,
                    type_name: std::any::type_name::<T>().to_string(),
                })?;
            let entry = unsafe { *self.flat_columns.get_unchecked(pos) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let typed_ptr = entry.col_ptr as *mut () as *mut TypedColumn<T>;
                return Ok(unsafe { &mut *typed_ptr });
            }
        }
        let col =
            self.columns
                .get_mut(&cid)
                .ok_or_else(|| StateError::ComponentNotFoundInEntity {
                    entity,
                    type_name: std::any::type_name::<T>().to_string(),
                })?;
        let type_name = col.type_name();
        col.as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .ok_or_else(|| StateError::TypeMismatch {
                expected: std::any::type_name::<T>().to_string(),
                actual: type_name.to_string(),
            })
    }

    /// 按 ComponentTypeId 获取强类型组件列裸指针（用于多列互斥借用，调用方必须保证 cid 互不相同）
    #[inline]
    pub fn get_typed_column_mut_ptr<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        cid: ComponentTypeId,
        entity: EntityId,
    ) -> Result<*mut TypedColumn<T>, StateError> {
        if !self.may_be_segmented_or_unaligned
            && self.flat_columns.len() == self.component_types_slice.len()
        {
            let pos = self
                .component_types_slice
                .iter()
                .position(|&id| id == cid)
                .ok_or_else(|| StateError::ComponentNotFoundInEntity {
                    entity,
                    type_name: std::any::type_name::<T>().to_string(),
                })?;
            let entry = unsafe { *self.flat_columns.get_unchecked(pos) };
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                return Ok(entry.col_ptr as *mut () as *mut TypedColumn<T>);
            }
        }
        let col =
            self.columns
                .get_mut(&cid)
                .ok_or_else(|| StateError::ComponentNotFoundInEntity {
                    entity,
                    type_name: std::any::type_name::<T>().to_string(),
                })?;
        let type_name = col.type_name();
        let typed = col
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .ok_or_else(|| StateError::TypeMismatch {
                expected: std::any::type_name::<T>().to_string(),
                actual: type_name.to_string(),
            })?;
        Ok(typed as *mut TypedColumn<T>)
    }

    /// 为实体列表及所有组件列预留容量
    pub fn reserve(&mut self, additional: usize) {
        self.entities.reserve(additional);
        if self.entity_slots.len() == self.entities.len() {
            self.entity_slots.reserve(additional);
        }
        for col in self.columns.values_mut() {
            col.reserve(additional);
        }
    }

    /// Prepare every fallible Clone before publishing any replacement. Existing
    /// Direct columns allocate nothing; uniquely owned frozen blocks are moved.
    #[inline]
    pub fn co_materialize_segmented_columns(&mut self) -> Result<(), StateError> {
        let mut replacements =
            smallvec::SmallVec::<[(ComponentTypeId, Box<dyn AnyColumn>); 4]>::new();
        let mut move_unique = false;
        for (&cid, col) in &mut self.columns {
            if let Some(prepared) = col.prepare_contiguous()? {
                replacements.push((cid, prepared));
            } else {
                move_unique = true;
            }
        }
        if !replacements.is_empty() || move_unique {
            self.columns_version = self.columns_version.wrapping_add(1);
            for (cid, replacement) in &mut replacements {
                std::mem::swap(
                    self.columns.get_mut(cid).expect("prepared column exists"),
                    replacement,
                );
            }
            if move_unique {
                for col in self.columns.values_mut() {
                    col.ensure_contiguous()?;
                }
            }
            drop(replacements);
            self.rebuild_flat_columns();
        }
        Ok(())
    }

    /// Swap-Remove 移除指定行实体，并返回被移动到该位置填补空缺的实体（若发生移动）
    pub fn swap_remove(&mut self, row: usize) -> Result<Option<EntityId>, StateError> {
        if row >= self.entities.len() {
            return Err(StateError::AccessDenied(format!(
                "Row {} out of bounds in archetype {}",
                row, self.id.0
            )));
        }
        self.co_materialize_segmented_columns()?;
        Ok(self.swap_remove_materialized(row))
    }

    /// 在前置校验保证行号有效的前提下，执行不可失败的 Swap-Remove
    pub fn swap_remove_infallible(&mut self, row: usize) -> Option<EntityId> {
        if self.may_be_segmented_or_unaligned {
            self.co_materialize_segmented_columns()
                .expect("validated structural column materialization");
        }
        self.swap_remove_materialized(row)
    }

    fn swap_remove_materialized(&mut self, row: usize) -> Option<EntityId> {
        let is_last = row == self.entities.len() - 1;
        if self.entity_slots.len() == self.entities.len() {
            self.entity_slots.swap_remove(row);
        } else {
            self.entity_slots.clear();
        }
        self.entities.swap_remove(row);

        if !self.may_be_segmented_or_unaligned && self.flat_columns.len() == self.columns.len() {
            for entry in &self.flat_columns {
                unsafe {
                    (entry.swap_remove_fn)(entry.col_ptr as *mut (), row);
                }
            }
        } else {
            for col in self.columns.values_mut() {
                col.swap_remove(row);
            }
        }

        if is_last {
            None
        } else {
            // 被移到当前 row 的实体
            Some(self.entities[row])
        }
    }
}

impl Clone for Archetype {
    fn clone(&self) -> Self {
        let columns = self
            .columns
            .iter()
            .map(|(&type_id, col)| (type_id, col.clone_box()))
            .collect();
        let mut arch = Self {
            id: self.id,
            component_types: self.component_types.clone(),
            component_types_slice: self.component_types_slice.clone(),
            entities: self.entities.clone(),
            entity_slots: self.entity_slots.clone(),
            columns,
            flat_columns: smallvec::SmallVec::new(),
            may_be_segmented_or_unaligned: true,
            columns_version: 1,
            add_edges: self.add_edges.clone(),
            remove_edges: self.remove_edges.clone(),
        };
        arch.rebuild_flat_columns();
        arch
    }
}

impl std::fmt::Debug for Archetype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Archetype")
            .field("id", &self.id)
            .field("component_types", &self.component_types)
            .field("entities_count", &self.entities.len())
            .field("columns_count", &self.columns.len())
            .finish()
    }
}

/// 按 ArchetypeId 快速查找原型（优先按索引直接定位，失败后降级线性遍历）
#[inline]
pub(crate) fn find_archetype_by_id<'a>(
    archetypes: &'a [Archetype],
    id: ArchetypeId,
) -> Option<&'a Archetype> {
    archetypes
        .get(id.0 as usize)
        .filter(|a| a.id() == id)
        .or_else(|| archetypes.iter().find(|a| a.id() == id))
}

#[cfg(test)]
#[path = "archetype_tests.rs"]
mod materialization_tests;
