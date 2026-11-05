//! 单实体 CRUD 与直接修改 (Entity CRUD & Direct Mutation)

use super::Store;
use super::CorePoisonGuard;
use crate::storage::core::allocator::EntityLocation;
use crate::storage::column::TypedColumn;
use crate::storage::ErasedComponentValue;
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::{ComponentTypeId, LocalSlotId};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// 全局单例组件包装器 (`Singleton<T>`)。
///
/// 挂载于保留的全局单例宿主实体 [`Store::SINGLETON_ENTITY`] 上，在复用列式存储、
/// 快照与并发捕获基础设施的同时，避免普通实体组件 `T` 查询误匹配到全局单例。
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Singleton<T>(pub T);

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub(crate) enum ReplaceInPlaceResult<T> {
    /// 替换成功，返回被替换出的旧值
    Replaced(T),
    /// 原型中不包含该组件，返回未消费的原组件值（保证未发生任何数据突变与代际推进）
    ComponentNotPresent(T),
}

impl Store {
    /// 非事务性原地修改组件值（直接在列存储内原地执行闭包并更新，不进行 Archetype 原型迁移）。
    ///
    /// # 保证与语义
    /// 1. **严格三段式借用与生命周期**：
    ///    - 阶段 1（只读预检）：查验实体存在性、定位 Archetype、检查列存在性、下转型与行边界合法性。
    ///      若类型不匹配或越界立即返回 Err，成功路径 0 堆分配、0 克隆、0 序列化；预检失败绝不推进 epoch。
    ///    - 阶段 2（推进代际）：检查容量并在独占借用前推进代际 (`epoch`)，使历史批次与缓冲立即失效。
    ///    - 阶段 3（独占列借用）：切片直接索引，向闭包暴露 `&mut T` 并就地生效。
    /// 2. **Panic 不承诺回滚**：专为轻量高频局部物理计算设计。若闭包发生 panic，部分字段修改不自动回滚。
    ///
    /// A borrow cannot escape the modify closure:
    /// ```compile_fail
    /// use sumer_state::{EntityId, Store};
    /// let mut store = Store::new();
    /// let id = EntityId::from_u128(1);
    /// let _ = store.modify::<u64, _>(id, |v| v);
    /// ```
    pub(crate) fn modify_at<T: ModelComponent + Serialize + DeserializeOwned + 'static, R>(
        &mut self,
        entity: EntityId,
        location: EntityLocation,
        modify_fn: impl FnOnce(&mut T) -> R,
    ) -> Result<R, StateError> {
        self.check_poisoned()?;
        let next_epoch = self.next_epoch()?;
        let type_id = ComponentTypeId::of::<T>();
        let arch = self
            .archetypes
            .get_mut(location.archetype_id as usize)
            .ok_or_else(|| {
                StateError::AccessDenied(format!("Entity {entity} references an unknown archetype"))
            })?;

        let row = location.row_index as usize;
        if arch.entities().get(row) != Some(&entity) {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype row"
            )));
        }

        let typed = arch.get_typed_column_mut::<T>(type_id, entity)?;
        let val_ref = typed.get_mut(row).ok_or_else(|| {
            StateError::AccessDenied(format!(
                "Entity {entity} row {row} out of bounds in archetype {}",
                location.archetype_id
            ))
        })?;

        self.epoch = next_epoch;
        Ok(modify_fn(val_ref))
    }

    pub fn modify<T: ModelComponent + Serialize + DeserializeOwned + 'static, R>(
        &mut self,
        entity: EntityId,
        modify_fn: impl FnOnce(&mut T) -> R,
    ) -> Result<R, StateError> {
        let location = self
            .allocator
            .get_location(entity)
            .ok_or_else(|| StateError::EntityNotFound(entity))?;
        self.modify_at(entity, location, modify_fn)
    }

    /// 原位多组件元组联合借用与修改底层实现（支持 1..16 元组宏）。
    pub(crate) fn modify_tuple_at<T: for<'a> crate::storage::modify::ModifyTarget<'a>, R>(
        &mut self,
        entity: EntityId,
        location: EntityLocation,
        modify_fn: impl FnOnce(<T as crate::storage::modify::sealed::ModifyFetch<'_>>::View) -> R,
    ) -> Result<R, StateError> {
        self.check_poisoned()?;
        let next_epoch = self.next_epoch()?;
        let arch = self
            .archetypes
            .get_mut(location.archetype_id as usize)
            .ok_or_else(|| {
                StateError::AccessDenied(format!("Entity {entity} references an unknown archetype"))
            })?;

        let row = location.row_index as usize;
        if arch.entities().get(row) != Some(&entity) {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype row"
            )));
        }

        let view = <T as crate::storage::modify::sealed::ModifyFetch<'_>>::fetch_mut(arch, row, entity)?;
        self.epoch = next_epoch;
        Ok(modify_fn(view))
    }

    /// 原位多组件元组联合借用与修改（支持 1..16 元组宏，如 (A, B) 或 (A, B, C)）。
    pub fn modify_tuple<T: for<'a> crate::storage::modify::ModifyTarget<'a>, R>(
        &mut self,
        entity: EntityId,
        modify_fn: impl FnOnce(<T as crate::storage::modify::sealed::ModifyFetch<'_>>::View) -> R,
    ) -> Result<R, StateError> {
        let location = self
            .allocator
            .get_location(entity)
            .ok_or_else(|| StateError::EntityNotFound(entity))?;
        self.modify_tuple_at::<T, R>(entity, location, modify_fn)
    }

    /// 包内专用测试与寻址差分辅助：获取实体的 Store 本地代际槽位。
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn local_slot(&self, entity: EntityId) -> Option<LocalSlotId> {
        self.allocator.local_slot(entity)
    }

    /// 包内专用测试与寻址差分辅助：通过已有代际槽位执行联合借用修改（Layer B 路径）。
    ///
    /// 验证槽位存在、代际匹配、活跃状态和物理位置存在后，执行修改；跳过 UUID B-Tree 查找。
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn modify_tuple_by_slot<T: for<'a> crate::storage::modify::ModifyTarget<'a>, R>(
        &mut self,
        entity: EntityId,
        slot: LocalSlotId,
        modify_fn: impl FnOnce(<T as crate::storage::modify::sealed::ModifyFetch<'_>>::View) -> R,
    ) -> Result<R, StateError> {
        let location = self
            .allocator
            .get_location_by_slot(entity, slot)
            .ok_or_else(|| StateError::EntityNotFound(entity))?;
        self.modify_tuple_at::<T, R>(entity, location, modify_fn)
    }

    /// 判断实体是否存在：稳定 ID 索引查询为 O(log N)，不分配、不克隆组件。
    #[inline]
    pub fn contains_entity(&self, id: EntityId) -> bool {
        self.allocator.is_alive(id)
    }

    /// 借用迭代器，安全枚举当前所有活跃实体，零堆分配
    #[inline]
    pub fn entities(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.archetypes
            .iter()
            .flat_map(|arch| arch.entities().iter().copied())
    }

    /// 当前存储库管理的活跃实体总数，O(1)
    #[inline]
    pub fn len(&self) -> usize {
        self.allocator.active_count()
    }

    /// 当前存储库是否无活跃实体
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.allocator.active_count() == 0
    }

    /// 彻底删除实体及其全部组件。若实体不存在报错 `EntityNotFound(id)`。
    pub fn delete_entity(&mut self, id: EntityId) -> Result<(), StateError> {
        self.check_poisoned()?;
        let (local_slot, location) = self
            .allocator
            .take_slot_and_location(id)
            .ok_or(StateError::EntityNotFound(id))?;
        let next_epoch = match self.next_epoch() {
            Ok(ep) => ep,
            Err(e) => {
                self.allocator.restore_index_entry(id, local_slot);
                return Err(e);
            }
        };
        if let Err(e) = self.allocator.validate_deallocate_slot(local_slot) {
            self.allocator.restore_index_entry(id, local_slot);
            return Err(e);
        }
        if let Err(e) = self.validate_archetype_layout_mut(location.archetype_id as usize) {
            self.allocator.restore_index_entry(id, local_slot);
            return Err(e);
        }
        let arch = &self.archetypes[location.archetype_id as usize];
        let row = location.row_index as usize;
        if arch.entities().get(row) != Some(&id) {
            self.allocator.restore_index_entry(id, local_slot);
            return Err(StateError::AccessDenied("Inconsistent entity row".into()));
        }
        let moved_tail = if row + 1 < arch.len() {
            let last_idx = arch.len() - 1;
            let last = arch.entities()[last_idx];
            let expected_tail_loc = EntityLocation::new(
                location.archetype_id,
                last_idx as u32,
            );
            if let Some(tail_slot) = arch.entity_slot(last_idx).filter(|&s| {
                self.allocator.get_location_by_slot(last, s) == Some(expected_tail_loc)
            }) {
                Some(tail_slot)
            } else {
                match self.allocator.local_slot_and_location(last) {
                    Some((tail_slot, tail_loc)) if tail_loc == expected_tail_loc => Some(tail_slot),
                    _ => {
                        self.allocator.restore_index_entry(id, local_slot);
                        return Err(StateError::AccessDenied(
                            "Inconsistent swap-remove tail".into(),
                        ));
                    }
                }
            }
        } else {
            None
        };
        let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
        let _moved = self.archetypes[location.archetype_id as usize]
            .swap_remove_infallible(row);
        if let Some(tail_slot) = moved_tail {
            self.allocator
                .set_location_by_slot_infallible(tail_slot, location);
        }
        self.allocator.finish_deallocate_slot_infallible(local_slot);
        self.epoch = next_epoch;
        poison_guard.disarm();
        Ok(())
    }


    /// 统一的高性能强类型就地替换内核
    ///
    /// 在单次 Archetype 获取与单次列检索下完成全量防御校验并原位替换组件值：
    /// 1. 索引 Archetype 并核对行对应实体身份（防 swap-remove 错位，返回 AccessDenied）
    /// 2. 原型 ID、拓扑索引一致性与包含关系核验（不一致返回 AccessDenied）
    /// 3. 若原型不包含目标组件，原样归还 component，绝不推进代际、不产生任何写入（返回 ComponentNotPresent）
    /// 4. B 树列定位与列长度一致性校验、类型名匹配校验、原生类型存在性校验（不匹配返回 AccessDenied）
    /// 5. 强类型 Rust downcast 校验（类型不匹配返回 TypeMismatch）
    /// 6. 行边界防卫（越界返回 AccessDenied）与代际溢出预检（CapacityExhausted）
    /// 7. 校验全量通过后，就地 mem::replace 原位写入，原子推进代际并返回 Replaced(old)
    pub(crate) fn try_replace_component_in_place<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        entity: EntityId,
        location: EntityLocation,
        component: T,
        matches_name: impl FnOnce(&'static str) -> bool,
    ) -> Result<ReplaceInPlaceResult<T>, StateError> {
        self.check_poisoned()?;
        let next_epoch = self.next_epoch()?;
        let index_len = self.archetype_index.len();
        let arch_count = self.archetypes.len();

        let arch = self
            .archetypes
            .get_mut(location.archetype_id as usize)
            .ok_or_else(|| {
                StateError::AccessDenied(format!("Entity {entity} references an unknown archetype"))
            })?;

        let row = location.row_index as usize;
        if arch.entities().get(row) != Some(&entity) {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype row"
            )));
        }

        let type_id = ComponentTypeId::of::<T>();
        if arch.id().0 != location.archetype_id
            || index_len != arch_count
            || arch.columns().len() != arch.component_types().len()
        {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype layout"
            )));
        }

        if !arch.may_be_segmented_or_unaligned()
            && arch.flat_columns().len() == arch.component_types_slice().len()
        {
            let Some(pos) = arch
                .component_types_slice()
                .iter()
                .position(|&id| id == type_id)
            else {
                return Ok(ReplaceInPlaceResult::ComponentNotPresent(component));
            };
            let entry = arch.flat_columns()[pos];
            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                let typed = unsafe { &mut *(entry.col_ptr as *mut () as *mut TypedColumn<T>) };
                let val_mut = typed.get_mut(row).ok_or_else(|| {
                    StateError::AccessDenied(format!(
                        "Entity {entity} row {row} out of bounds in archetype {}",
                        location.archetype_id
                    ))
                })?;
                let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
                let old = std::mem::replace(val_mut, component);
                self.epoch = next_epoch;
                poison_guard.disarm();
                return Ok(ReplaceInPlaceResult::Replaced(old));
            }
        }

        let arch_len = arch.len();
        let Some(col) = arch.columns_mut().get_mut(&type_id) else {
            if !arch.columns().is_empty()
                && arch.component_types_slice().binary_search(&type_id).is_ok()
            {
                return Err(StateError::ComponentNotFound(format!("{type_id}")));
            }
            return Ok(ReplaceInPlaceResult::ComponentNotPresent(component));
        };

        let Some(typed) = col.as_any_mut().downcast_mut::<TypedColumn<T>>() else {
            if col.len() != arch_len
                || col.as_ref().type_id() != type_id
                || !matches_name(col.type_name())
                || col.rust_type_id().is_none()
            {
                return Err(StateError::AccessDenied(format!(
                    "Entity {entity} has invalid component layout"
                )));
            }
            return Err(StateError::TypeMismatch {
                expected: std::any::type_name::<T>().into(),
                actual: col.type_name().into(),
            });
        };

        if typed.len() != arch_len
            || typed.type_id != type_id
            || (!std::ptr::eq(typed.type_name, std::any::type_name::<T>())
                && !matches_name(typed.type_name))
        {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has invalid component layout"
            )));
        }

        let val_mut = typed.get_mut(row).ok_or_else(|| {
            StateError::AccessDenied(format!(
                "Entity {entity} row {row} out of bounds in archetype {}",
                location.archetype_id
            ))
        })?;

        let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
        let old = std::mem::replace(val_mut, component);
        self.epoch = next_epoch;
        poison_guard.disarm();
        Ok(ReplaceInPlaceResult::Replaced(old))
    }

    /// 单实体强类型组件插入与原型迁移快路径（命中已存在目标原型时零装箱、零临时 Batch、零收据构建）
    pub(crate) fn insert_component_at_location<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        entity: EntityId,
        local_slot: Option<LocalSlotId>,
        location: EntityLocation,
        component: T,
    ) -> Result<EntityLocation, StateError> {
        self.check_poisoned()?;
        let next_epoch = self.next_epoch()?;
        let type_id = ComponentTypeId::of::<T>();
        let rust_type = std::any::TypeId::of::<T>();
        let src_arch_id = location.archetype_id;
        let src_row = location.row_index as usize;

        if self.archetype_index.len() != self.archetypes.len() {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype layout"
            )));
        }

        let src_arch = self
            .archetypes
            .get(src_arch_id as usize)
            .ok_or_else(|| {
                StateError::AccessDenied(format!("Entity {entity} references an unknown archetype"))
            })?;

        if src_arch.entities().get(src_row) != Some(&entity) {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype row"
            )));
        }

        if src_arch.id().0 != src_arch_id
            || (src_arch.may_be_segmented_or_unaligned()
                && src_arch.columns().len() != src_arch.component_types().len())
        {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype layout"
            )));
        }

        let (dst_arch_id, dst_col_idx) =
            if let Some((cached_dst, cached_col_idx)) = src_arch.add_edge(type_id, rust_type) {
                let dst_arch = &self.archetypes[cached_dst as usize];
                if dst_arch.len() >= u32::MAX as usize {
                    return Err(StateError::CapacityExhausted(
                        "Archetype rows exhausted".into(),
                    ));
                }
                (cached_dst, cached_col_idx as usize)
            } else {
                if src_arch.component_types_slice().binary_search(&type_id).is_ok() {
                    return Err(StateError::ComponentAlreadyExists {
                        entity,
                        type_name: std::any::type_name::<T>().to_string(),
                    });
                }

                let mut target_types = src_arch.component_types().clone();
                target_types.insert(type_id);
                let Some(&found_dst) = self.archetype_index.get(&target_types) else {
                    self.insert_component_erased(entity, ErasedComponentValue::new(component))?;
                    let new_loc = self
                        .allocator
                        .get_location(entity)
                        .ok_or(StateError::EntityNotFound(entity))?;
                    let col_idx = self
                        .archetypes
                        .get(new_loc.archetype_id as usize)
                        .and_then(|a| a.component_types_slice().binary_search(&type_id).ok())
                        .and_then(|idx| u16::try_from(idx).ok());
                    if let Some(col_idx) = col_idx {
                        if let Some(src_arch) = self.archetypes.get_mut(src_arch_id as usize) {
                            src_arch.set_add_edge(type_id, rust_type, new_loc.archetype_id, col_idx);
                        }
                        if let Some(dst_arch) =
                            self.archetypes.get_mut(new_loc.archetype_id as usize)
                        {
                            dst_arch.set_remove_edge(type_id, rust_type, src_arch_id, col_idx);
                        }
                    }
                    return Ok(new_loc);
                };

                if src_arch_id == found_dst {
                    return Err(StateError::AccessDenied(format!(
                        "Invalid self-loop transition edge on archetype {src_arch_id}"
                    )));
                }
                self.validate_archetype_layout_mut(src_arch_id as usize)?;
                self.validate_archetype_layout_mut(found_dst as usize)?;

                let src_arch = &self.archetypes[src_arch_id as usize];
                let dst_arch = &self.archetypes[found_dst as usize];

                if dst_arch.columns().len() != src_arch.columns().len() + 1 {
                    return Err(StateError::AccessDenied(format!(
                        "Invalid transition edge from archetype {src_arch_id} to {found_dst}"
                    )));
                }

                if dst_arch.len() >= u32::MAX as usize {
                    return Err(StateError::CapacityExhausted(
                        "Archetype rows exhausted".into(),
                    ));
                }

                let dst_col = dst_arch
                    .columns()
                    .get(&type_id)
                    .ok_or_else(|| StateError::ComponentNotFound(format!("{type_id}")))?;
                if dst_col.type_name() != std::any::type_name::<T>()
                    || dst_col.as_any().downcast_ref::<TypedColumn<T>>().is_none()
                {
                    return Err(StateError::AccessDenied(format!(
                        "Native component {} does not match destination column {}",
                        std::any::type_name::<T>(),
                        dst_col.type_name(),
                    )));
                }

                if src_arch.has_segmented_columns() || dst_arch.has_segmented_columns() {
                    self.insert_component_erased(entity, ErasedComponentValue::new(component))?;
                    return self
                        .allocator
                        .get_location(entity)
                        .ok_or(StateError::EntityNotFound(entity));
                }

                let col_idx = dst_arch
                    .component_types_slice()
                    .binary_search(&type_id)
                    .map_err(|_| StateError::ComponentNotFound(format!("{type_id}")))?;
                let col_idx_u16 = u16::try_from(col_idx)
                    .map_err(|_| StateError::CapacityExhausted("Archetype columns exhausted".into()))?;
                self.archetypes[src_arch_id as usize]
                    .set_add_edge(type_id, rust_type, found_dst, col_idx_u16);
                self.archetypes[found_dst as usize]
                    .set_remove_edge(type_id, rust_type, src_arch_id, col_idx_u16);
                (found_dst, col_idx)
            };

        let src_arch = &self.archetypes[src_arch_id as usize];

        let moved_tail = if src_row + 1 < src_arch.len() {
            let last_idx = src_arch.len() - 1;
            let last_entity = src_arch.entities()[last_idx];
            let expected_tail_loc = EntityLocation::new(
                src_arch_id,
                last_idx as u32,
            );
            if let Some(tail_slot) = src_arch.entity_slot(last_idx).filter(|&s| {
                self.allocator.get_location_by_slot(last_entity, s) == Some(expected_tail_loc)
            }) {
                Some(tail_slot)
            } else {
                match self.allocator.local_slot_and_location(last_entity) {
                    Some((tail_slot, tail_loc)) if tail_loc == expected_tail_loc => Some(tail_slot),
                    _ => {
                        return Err(StateError::AccessDenied(
                            "Inconsistent swap-remove tail".into(),
                        ));
                    }
                }
            }
        } else {
            None
        };

        let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
        let src_idx = src_arch_id as usize;
        let dst_idx = dst_arch_id as usize;
        let (src_arch, dst_arch) = if src_idx < dst_idx {
            let (left, right) = self.archetypes.split_at_mut(dst_idx);
            (&mut left[src_idx], &mut right[0])
        } else {
            let (left, right) = self.archetypes.split_at_mut(src_idx);
            (&mut right[0], &mut left[dst_idx])
        };

        if !src_arch.may_be_segmented_or_unaligned()
            && !dst_arch.may_be_segmented_or_unaligned()
        {
            let src_flat = src_arch.flat_columns();
            let dst_flat = dst_arch.flat_columns();
            debug_assert_eq!(src_flat.len() + 1, dst_flat.len());
            debug_assert!(dst_col_idx < dst_flat.len());
            unsafe {
                for i in 0..dst_col_idx {
                    let s = src_flat.get_unchecked(i);
                    let d = dst_flat.get_unchecked(i);
                    debug_assert_eq!(s.type_id, d.type_id);
                    (s.move_row_fn)(s.col_ptr as *mut (), src_row, d.col_ptr as *mut ());
                }
                let dst_entry = dst_flat.get_unchecked(dst_col_idx);
                debug_assert_eq!(dst_entry.type_id, type_id);
                let typed_col = &mut *(dst_entry.col_ptr as *mut () as *mut TypedColumn<T>);
                typed_col.push_direct_owned_unchecked(component);
                for i in dst_col_idx..src_flat.len() {
                    let s = src_flat.get_unchecked(i);
                    let d = dst_flat.get_unchecked(i + 1);
                    debug_assert_eq!(s.type_id, d.type_id);
                    (s.move_row_fn)(s.col_ptr as *mut (), src_row, d.col_ptr as *mut ());
                }
            }
        } else {
            let mut component_opt = Some(component);
            let mut src_cols = src_arch.columns_mut().iter_mut();
            for (&dst_cid, dst_col) in dst_arch.columns_mut().iter_mut() {
                if dst_cid == type_id {
                    let comp = component_opt.take().expect("component pushed once");
                    debug_assert!(dst_col.as_any().is::<TypedColumn<T>>());
                    let typed_col = unsafe {
                        &mut *(dst_col.as_mut() as *mut dyn crate::storage::column::AnyColumn
                            as *mut TypedColumn<T>)
                    };
                    typed_col.push(comp);
                } else {
                    let (src_cid, src_col) = src_cols
                        .next()
                        .expect("source archetype has all non-inserted columns");
                    debug_assert_eq!(*src_cid, dst_cid);
                    src_col.move_row_to_infallible(src_row, dst_col.as_mut());
                }
            }
        }

        src_arch.swap_remove_entity_only(src_row);
        let resolved_slot = local_slot.or_else(|| self.allocator.local_slot(entity));
        let dst_row = if let Some(slot) = resolved_slot {
            dst_arch.push_entity_with_slot(entity, slot) as u32
        } else {
            dst_arch.push_entity(entity) as u32
        };
        let new_location = EntityLocation::new(dst_arch_id, dst_row);

        if let Some(tail_slot) = moved_tail {
            self.allocator.set_location_by_slot_infallible(
                tail_slot,
                EntityLocation::new(src_arch_id, src_row as u32),
            );
        }
        if let Some(slot) = resolved_slot {
            self.allocator
                .set_location_by_slot_infallible(slot, new_location);
        } else {
            self.allocator.set_location_infallible(entity, new_location);
        }
        self.epoch = next_epoch;
        poison_guard.disarm();
        Ok(new_location)
    }

    /// 单实体强类型组件 Upsert 快路径（单次预检：若已存在于平坦列则 O(1) 原位替换，否则走转移边缓存 O(1) 迁移）
    #[inline]
    pub(crate) fn upsert_component_at_location<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        entity: EntityId,
        local_slot: Option<LocalSlotId>,
        location: EntityLocation,
        component: T,
    ) -> Result<(Option<T>, EntityLocation), StateError> {
        if self.check_poisoned().is_ok() {
            if let Some(next_epoch) = self.epoch.checked_add(1) {
                let src_arch_id = location.archetype_id;
                let src_row = location.row_index as usize;
                let index_len = self.archetype_index.len();
                let arch_count = self.archetypes.len();
                if let Some(src_arch) = self.archetypes.get_mut(src_arch_id as usize) {
                    if src_arch.id().0 == src_arch_id
                        && index_len == arch_count
                        && !src_arch.may_be_segmented_or_unaligned()
                        && src_arch.entities().get(src_row) == Some(&entity)
                        && src_arch.flat_columns().len() == src_arch.component_types_slice().len()
                    {
                        let type_id = ComponentTypeId::of::<T>();
                        if let Some(pos) = src_arch
                            .component_types_slice()
                            .iter()
                            .position(|&id| id == type_id)
                        {
                            let entry = unsafe { *src_arch.flat_columns().get_unchecked(pos) };
                            if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
                                let typed = unsafe {
                                    &mut *(entry.col_ptr as *mut () as *mut TypedColumn<T>)
                                };
                                if let Some(val_mut) = typed.get_mut(src_row) {
                                    let old = std::mem::replace(val_mut, component);
                                    self.epoch = next_epoch;
                                    return Ok((Some(old), location));
                                }
                            }
                        }
                    }
                }
            }
        }
        self.upsert_component_slow_at_location(entity, local_slot, location, component)
    }

    #[inline(never)]
    fn upsert_component_slow_at_location<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        entity: EntityId,
        local_slot: Option<LocalSlotId>,
        location: EntityLocation,
        component: T,
    ) -> Result<(Option<T>, EntityLocation), StateError> {
        self.check_poisoned()?;
        let next_epoch = self.next_epoch()?;
        let type_id = ComponentTypeId::of::<T>();
        let rust_type = std::any::TypeId::of::<T>();
        let src_arch_id = location.archetype_id;
        let src_row = location.row_index as usize;
        let index_len = self.archetype_index.len();
        let arch_count = self.archetypes.len();

        let src_arch = self
            .archetypes
            .get_mut(src_arch_id as usize)
            .ok_or_else(|| {
                StateError::AccessDenied(format!("Entity {entity} references an unknown archetype"))
            })?;

        if src_arch.entities().get(src_row) != Some(&entity) {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype row"
            )));
        }

        if src_arch.id().0 != src_arch_id
            || index_len != arch_count
            || (src_arch.may_be_segmented_or_unaligned()
                && src_arch.columns().len() != src_arch.component_types().len())
        {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype layout"
            )));
        }

        let cached_edge = src_arch.add_edge(type_id, rust_type);
        if cached_edge.is_none()
            && (src_arch.may_be_segmented_or_unaligned()
                || src_arch.component_types_slice().contains(&type_id))
        {
            let arch_len = src_arch.len();
            if let Some(col) = src_arch.columns_mut().get_mut(&type_id) {
                let matches_name = |name: &'static str| {
                    name == std::any::type_name::<T>()
                        || ComponentTypeId::from_name(name) == type_id
                };
                let Some(typed) = col.as_any_mut().downcast_mut::<TypedColumn<T>>() else {
                    if col.len() != arch_len
                        || col.as_ref().type_id() != type_id
                        || !matches_name(col.type_name())
                        || col.rust_type_id().is_none()
                    {
                        return Err(StateError::AccessDenied(format!(
                            "Entity {entity} has invalid component layout"
                        )));
                    }
                    return Err(StateError::TypeMismatch {
                        expected: std::any::type_name::<T>().into(),
                        actual: col.type_name().into(),
                    });
                };

                if typed.len() != arch_len
                    || typed.type_id != type_id
                    || (!std::ptr::eq(typed.type_name, std::any::type_name::<T>())
                        && !matches_name(typed.type_name))
                {
                    return Err(StateError::AccessDenied(format!(
                        "Entity {entity} has invalid component layout"
                    )));
                }

                let val_mut = typed.get_mut(src_row).ok_or_else(|| {
                    StateError::AccessDenied(format!(
                        "Entity {entity} row {src_row} out of bounds in archetype {src_arch_id}"
                    ))
                })?;

                let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
                let old = std::mem::replace(val_mut, component);
                self.epoch = next_epoch;
                poison_guard.disarm();
                return Ok((Some(old), location));
            }
            if src_arch.component_types_slice().binary_search(&type_id).is_ok() {
                return Err(StateError::ComponentNotFound(format!("{type_id}")));
            }
        }

        let (dst_arch_id, dst_col_idx) = if let Some((cached_dst, cached_col_idx)) = cached_edge {
            let dst_arch = &self.archetypes[cached_dst as usize];
            if dst_arch.len() >= u32::MAX as usize {
                return Err(StateError::CapacityExhausted(
                    "Archetype rows exhausted".into(),
                ));
            }
            (cached_dst, cached_col_idx as usize)
        } else {
            let src_arch = &self.archetypes[src_arch_id as usize];
            let mut target_types = src_arch.component_types().clone();
            target_types.insert(type_id);
            let Some(&found_dst) = self.archetype_index.get(&target_types) else {
                self.insert_component_erased(entity, ErasedComponentValue::new(component))?;
                let new_loc = self
                    .allocator
                    .get_location(entity)
                    .ok_or(StateError::EntityNotFound(entity))?;
                let col_idx = self
                    .archetypes
                    .get(new_loc.archetype_id as usize)
                    .and_then(|a| a.component_types_slice().binary_search(&type_id).ok())
                    .and_then(|idx| u16::try_from(idx).ok());
                if let Some(col_idx) = col_idx {
                    if let Some(src_arch) = self.archetypes.get_mut(src_arch_id as usize) {
                        src_arch.set_add_edge(type_id, rust_type, new_loc.archetype_id, col_idx);
                    }
                    if let Some(dst_arch) =
                        self.archetypes.get_mut(new_loc.archetype_id as usize)
                    {
                        dst_arch.set_remove_edge(type_id, rust_type, src_arch_id, col_idx);
                    }
                }
                return Ok((None, new_loc));
            };

            if src_arch_id == found_dst {
                return Err(StateError::AccessDenied(format!(
                    "Invalid self-loop transition edge on archetype {src_arch_id}"
                )));
            }
            self.validate_archetype_layout_mut(src_arch_id as usize)?;
            self.validate_archetype_layout_mut(found_dst as usize)?;

            let src_arch = &self.archetypes[src_arch_id as usize];
            let dst_arch = &self.archetypes[found_dst as usize];

            if dst_arch.columns().len() != src_arch.columns().len() + 1 {
                return Err(StateError::AccessDenied(format!(
                    "Invalid transition edge from archetype {src_arch_id} to {found_dst}"
                )));
            }

            if dst_arch.len() >= u32::MAX as usize {
                return Err(StateError::CapacityExhausted(
                    "Archetype rows exhausted".into(),
                ));
            }

            let dst_col = dst_arch
                .columns()
                .get(&type_id)
                .ok_or_else(|| StateError::ComponentNotFound(format!("{type_id}")))?;
            if dst_col.type_name() != std::any::type_name::<T>()
                || dst_col.as_any().downcast_ref::<TypedColumn<T>>().is_none()
            {
                return Err(StateError::AccessDenied(format!(
                    "Native component {} does not match destination column {}",
                    std::any::type_name::<T>(),
                    dst_col.type_name(),
                )));
            }

            if src_arch.has_segmented_columns() || dst_arch.has_segmented_columns() {
                self.insert_component_erased(entity, ErasedComponentValue::new(component))?;
                let new_loc = self
                    .allocator
                    .get_location(entity)
                    .ok_or(StateError::EntityNotFound(entity))?;
                return Ok((None, new_loc));
            }

            let col_idx = dst_arch
                .component_types_slice()
                .binary_search(&type_id)
                .map_err(|_| StateError::ComponentNotFound(format!("{type_id}")))?;
            let col_idx_u16 = u16::try_from(col_idx)
                .map_err(|_| StateError::CapacityExhausted("Archetype columns exhausted".into()))?;
            self.archetypes[src_arch_id as usize]
                .set_add_edge(type_id, rust_type, found_dst, col_idx_u16);
            self.archetypes[found_dst as usize]
                .set_remove_edge(type_id, rust_type, src_arch_id, col_idx_u16);
            (found_dst, col_idx)
        };

        let src_arch = &self.archetypes[src_arch_id as usize];

        let moved_tail = if src_row + 1 < src_arch.len() {
            let last_idx = src_arch.len() - 1;
            let last_entity = src_arch.entities()[last_idx];
            let expected_tail_loc = EntityLocation::new(
                src_arch_id,
                last_idx as u32,
            );
            if let Some(tail_slot) = src_arch.entity_slot(last_idx).filter(|&s| {
                self.allocator.get_location_by_slot(last_entity, s) == Some(expected_tail_loc)
            }) {
                Some(tail_slot)
            } else {
                match self.allocator.local_slot_and_location(last_entity) {
                    Some((tail_slot, tail_loc)) if tail_loc == expected_tail_loc => Some(tail_slot),
                    _ => {
                        return Err(StateError::AccessDenied(
                            "Inconsistent swap-remove tail".into(),
                        ));
                    }
                }
            }
        } else {
            None
        };

        let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
        let src_idx = src_arch_id as usize;
        let dst_idx = dst_arch_id as usize;
        let (src_arch, dst_arch) = if src_idx < dst_idx {
            let (left, right) = self.archetypes.split_at_mut(dst_idx);
            (&mut left[src_idx], &mut right[0])
        } else {
            let (left, right) = self.archetypes.split_at_mut(src_idx);
            (&mut right[0], &mut left[dst_idx])
        };

        if !src_arch.may_be_segmented_or_unaligned()
            && !dst_arch.may_be_segmented_or_unaligned()
        {
            let src_flat = src_arch.flat_columns();
            let dst_flat = dst_arch.flat_columns();
            debug_assert_eq!(src_flat.len() + 1, dst_flat.len());
            debug_assert!(dst_col_idx < dst_flat.len());
            unsafe {
                for i in 0..dst_col_idx {
                    let s = src_flat.get_unchecked(i);
                    let d = dst_flat.get_unchecked(i);
                    debug_assert_eq!(s.type_id, d.type_id);
                    (s.move_row_fn)(s.col_ptr as *mut (), src_row, d.col_ptr as *mut ());
                }
                let dst_entry = dst_flat.get_unchecked(dst_col_idx);
                debug_assert_eq!(dst_entry.type_id, type_id);
                let typed_col = &mut *(dst_entry.col_ptr as *mut () as *mut TypedColumn<T>);
                typed_col.push_direct_owned_unchecked(component);
                for i in dst_col_idx..src_flat.len() {
                    let s = src_flat.get_unchecked(i);
                    let d = dst_flat.get_unchecked(i + 1);
                    debug_assert_eq!(s.type_id, d.type_id);
                    (s.move_row_fn)(s.col_ptr as *mut (), src_row, d.col_ptr as *mut ());
                }
            }
        } else {
            let mut component_opt = Some(component);
            let mut src_cols = src_arch.columns_mut().iter_mut();
            for (&dst_cid, dst_col) in dst_arch.columns_mut().iter_mut() {
                if dst_cid == type_id {
                    let comp = component_opt.take().expect("component pushed once");
                    debug_assert!(dst_col.as_any().is::<TypedColumn<T>>());
                    let typed_col = unsafe {
                        &mut *(dst_col.as_mut() as *mut dyn crate::storage::column::AnyColumn
                            as *mut TypedColumn<T>)
                    };
                    typed_col.push(comp);
                } else {
                    let (src_cid, src_col) = src_cols
                        .next()
                        .expect("source archetype has all non-inserted columns");
                    debug_assert_eq!(*src_cid, dst_cid);
                    src_col.move_row_to_infallible(src_row, dst_col.as_mut());
                }
            }
        }

        src_arch.swap_remove_entity_only(src_row);
        let resolved_slot = local_slot.or_else(|| self.allocator.local_slot(entity));
        let dst_row = if let Some(slot) = resolved_slot {
            dst_arch.push_entity_with_slot(entity, slot) as u32
        } else {
            dst_arch.push_entity(entity) as u32
        };
        let new_location = EntityLocation::new(dst_arch_id, dst_row);

        if let Some(tail_slot) = moved_tail {
            self.allocator.set_location_by_slot_infallible(
                tail_slot,
                EntityLocation::new(src_arch_id, src_row as u32),
            );
        }
        if let Some(slot) = resolved_slot {
            self.allocator
                .set_location_by_slot_infallible(slot, new_location);
        } else {
            self.allocator.set_location_infallible(entity, new_location);
        }
        self.epoch = next_epoch;
        poison_guard.disarm();
        Ok((None, new_location))
    }

    /// 单实体强类型组件移除与原型迁移快路径（命中已存在目标原型时零装箱、零临时 Batch、零收据构建）
    pub(crate) fn remove_component_at_location<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        entity: EntityId,
        local_slot: Option<LocalSlotId>,
        location: EntityLocation,
    ) -> Result<(T, EntityLocation), StateError> {
        self.check_poisoned()?;
        let next_epoch = self.next_epoch()?;
        let type_id = ComponentTypeId::of::<T>();
        let rust_type = std::any::TypeId::of::<T>();
        let src_arch_id = location.archetype_id;
        let src_row = location.row_index as usize;

        if self.archetype_index.len() != self.archetypes.len() {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype layout"
            )));
        }

        let src_arch = self
            .archetypes
            .get(src_arch_id as usize)
            .ok_or_else(|| {
                StateError::AccessDenied(format!("Entity {entity} references an unknown archetype"))
            })?;

        if src_arch.entities().get(src_row) != Some(&entity) {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype row"
            )));
        }

        if src_arch.id().0 != src_arch_id
            || (src_arch.may_be_segmented_or_unaligned()
                && src_arch.columns().len() != src_arch.component_types().len())
        {
            return Err(StateError::AccessDenied(format!(
                "Entity {entity} has an inconsistent archetype layout"
            )));
        }

        let (dst_arch_id, src_col_idx) =
            if let Some((cached_dst, cached_col_idx)) = src_arch.remove_edge(type_id, rust_type) {
                let dst_arch = &self.archetypes[cached_dst as usize];
                if dst_arch.len() >= u32::MAX as usize {
                    return Err(StateError::CapacityExhausted(
                        "Archetype rows exhausted".into(),
                    ));
                }
                (cached_dst, cached_col_idx as usize)
            } else {
                let src_target_col = src_arch
                    .columns()
                    .get(&type_id)
                    .ok_or_else(|| StateError::ComponentNotFoundInEntity {
                        entity,
                        type_name: std::any::type_name::<T>().to_string(),
                    })?;
                let src_typed = src_target_col
                    .as_any()
                    .downcast_ref::<TypedColumn<T>>()
                    .ok_or_else(|| StateError::TypeMismatch {
                        expected: std::any::type_name::<T>().to_string(),
                        actual: src_target_col.type_name().to_string(),
                    })?;
                if src_typed.get(src_row).is_none() {
                    return Err(StateError::AccessDenied(format!(
                        "Entity {entity} row {src_row} out of bounds in archetype {src_arch_id}"
                    )));
                }

                let mut target_types = src_arch.component_types().clone();
                target_types.remove(&type_id);
                let Some(&found_dst) = self.archetype_index.get(&target_types) else {
                    let col_idx = src_arch
                        .component_types_slice()
                        .binary_search(&type_id)
                        .ok()
                        .and_then(|idx| u16::try_from(idx).ok());
                    let erased = self.remove_component_erased(entity, type_id)?;
                    let new_loc = self
                        .allocator
                        .get_location(entity)
                        .ok_or(StateError::EntityNotFound(entity))?;
                    if let Some(col_idx) = col_idx {
                        if let Some(src_arch) = self.archetypes.get_mut(src_arch_id as usize) {
                            src_arch.set_remove_edge(
                                type_id,
                                rust_type,
                                new_loc.archetype_id,
                                col_idx,
                            );
                        }
                        if let Some(dst_arch) =
                            self.archetypes.get_mut(new_loc.archetype_id as usize)
                        {
                            dst_arch.set_add_edge(type_id, rust_type, src_arch_id, col_idx);
                        }
                    }
                    let val = erased
                        .into_typed::<T>()
                        .expect("validated native removal type");
                    return Ok((val, new_loc));
                };

                if src_arch_id == found_dst {
                    return Err(StateError::AccessDenied(format!(
                        "Invalid self-loop transition edge on archetype {src_arch_id}"
                    )));
                }
                self.validate_archetype_layout_mut(src_arch_id as usize)?;
                self.validate_archetype_layout_mut(found_dst as usize)?;

                let src_arch = &self.archetypes[src_arch_id as usize];
                let dst_arch = &self.archetypes[found_dst as usize];

                if dst_arch.columns().len() + 1 != src_arch.columns().len() {
                    return Err(StateError::AccessDenied(format!(
                        "Invalid transition edge from archetype {src_arch_id} to {found_dst}"
                    )));
                }

                if dst_arch.len() >= u32::MAX as usize {
                    return Err(StateError::CapacityExhausted(
                        "Archetype rows exhausted".into(),
                    ));
                }

                if src_arch.has_segmented_columns() || dst_arch.has_segmented_columns() {
                    let erased = self.remove_component_erased(entity, type_id)?;
                    let new_loc = self
                        .allocator
                        .get_location(entity)
                        .ok_or(StateError::EntityNotFound(entity))?;
                    let val = erased
                        .into_typed::<T>()
                        .expect("validated native removal type");
                    return Ok((val, new_loc));
                }

                let col_idx = src_arch
                    .component_types_slice()
                    .binary_search(&type_id)
                    .map_err(|_| StateError::ComponentNotFound(format!("{type_id}")))?;
                let col_idx_u16 = u16::try_from(col_idx)
                    .map_err(|_| StateError::CapacityExhausted("Archetype columns exhausted".into()))?;
                self.archetypes[src_arch_id as usize]
                    .set_remove_edge(type_id, rust_type, found_dst, col_idx_u16);
                self.archetypes[found_dst as usize]
                    .set_add_edge(type_id, rust_type, src_arch_id, col_idx_u16);
                (found_dst, col_idx)
            };

        let src_arch = &self.archetypes[src_arch_id as usize];

        let moved_tail = if src_row + 1 < src_arch.len() {
            let last_idx = src_arch.len() - 1;
            let last_entity = src_arch.entities()[last_idx];
            let expected_tail_loc = EntityLocation::new(
                src_arch_id,
                last_idx as u32,
            );
            if let Some(tail_slot) = src_arch.entity_slot(last_idx).filter(|&s| {
                self.allocator.get_location_by_slot(last_entity, s) == Some(expected_tail_loc)
            }) {
                Some(tail_slot)
            } else {
                match self.allocator.local_slot_and_location(last_entity) {
                    Some((tail_slot, tail_loc)) if tail_loc == expected_tail_loc => Some(tail_slot),
                    _ => {
                        return Err(StateError::AccessDenied(
                            "Inconsistent swap-remove tail".into(),
                        ));
                    }
                }
            }
        } else {
            None
        };

        let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
        let src_idx = src_arch_id as usize;
        let dst_idx = dst_arch_id as usize;
        let (src_arch, dst_arch) = if src_idx < dst_idx {
            let (left, right) = self.archetypes.split_at_mut(dst_idx);
            (&mut left[src_idx], &mut right[0])
        } else {
            let (left, right) = self.archetypes.split_at_mut(src_idx);
            (&mut right[0], &mut left[dst_idx])
        };

        let removed_val: T = if !src_arch.may_be_segmented_or_unaligned()
            && !dst_arch.may_be_segmented_or_unaligned()
        {
            let src_flat = src_arch.flat_columns();
            let dst_flat = dst_arch.flat_columns();
            debug_assert_eq!(dst_flat.len() + 1, src_flat.len());
            debug_assert!(src_col_idx < src_flat.len());
            unsafe {
                for i in 0..src_col_idx {
                    let s = src_flat.get_unchecked(i);
                    let d = dst_flat.get_unchecked(i);
                    debug_assert_eq!(s.type_id, d.type_id);
                    (s.move_row_fn)(s.col_ptr as *mut (), src_row, d.col_ptr as *mut ());
                }
                let src_entry = src_flat.get_unchecked(src_col_idx);
                debug_assert_eq!(src_entry.type_id, type_id);
                let typed_col = &mut *(src_entry.col_ptr as *mut () as *mut TypedColumn<T>);
                let val = typed_col.swap_remove_direct_owned_unchecked(src_row);
                for i in (src_col_idx + 1)..src_flat.len() {
                    let s = src_flat.get_unchecked(i);
                    let d = dst_flat.get_unchecked(i - 1);
                    debug_assert_eq!(s.type_id, d.type_id);
                    (s.move_row_fn)(s.col_ptr as *mut (), src_row, d.col_ptr as *mut ());
                }
                val
            }
        } else {
            let mut extracted: Option<T> = None;
            let mut dst_cols = dst_arch.columns_mut().iter_mut();
            for (&src_cid, src_col) in src_arch.columns_mut().iter_mut() {
                if src_cid == type_id {
                    debug_assert!(src_col.as_any().is::<TypedColumn<T>>());
                    let typed_col = unsafe {
                        &mut *(src_col.as_mut() as *mut dyn crate::storage::column::AnyColumn
                            as *mut TypedColumn<T>)
                    };
                    let val = typed_col
                        .swap_remove(src_row)
                        .expect("validated source row");
                    extracted = Some(val);
                } else {
                    let (dst_cid, dst_col) = dst_cols
                        .next()
                        .expect("destination archetype has all remaining columns");
                    debug_assert_eq!(src_cid, *dst_cid);
                    src_col.move_row_to_infallible(src_row, dst_col.as_mut());
                }
            }
            extracted.expect("removed component extracted")
        };

        src_arch.swap_remove_entity_only(src_row);
        let resolved_slot = local_slot.or_else(|| self.allocator.local_slot(entity));
        let dst_row = if let Some(slot) = resolved_slot {
            dst_arch.push_entity_with_slot(entity, slot) as u32
        } else {
            dst_arch.push_entity(entity) as u32
        };
        let new_location = EntityLocation::new(dst_arch_id, dst_row);

        if let Some(tail_slot) = moved_tail {
            self.allocator.set_location_by_slot_infallible(
                tail_slot,
                EntityLocation::new(src_arch_id, src_row as u32),
            );
        }
        if let Some(slot) = resolved_slot {
            self.allocator
                .set_location_by_slot_infallible(slot, new_location);
        } else {
            self.allocator.set_location_infallible(entity, new_location);
        }
        self.epoch = next_epoch;
        poison_guard.disarm();
        Ok((removed_val, new_location))
    }

    pub(crate) fn replace_component_in_place<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        entity: EntityId,
        location: EntityLocation,
        component: T,
        matches_name: impl FnOnce(&'static str) -> bool,
    ) -> Result<T, StateError> {
        let type_id = ComponentTypeId::of::<T>();
        match self.try_replace_component_in_place(entity, location, component, matches_name)? {
            ReplaceInPlaceResult::Replaced(old) => Ok(old),
            ReplaceInPlaceResult::ComponentNotPresent(_) => {
                Err(StateError::ComponentNotFoundInEntity {
                    entity,
                    type_name: format!("{type_id}"),
                })
            }
        }
    }

    #[inline]
    pub(crate) fn replace_at<T: ModelComponent + Serialize + DeserializeOwned>(
        &mut self,
        entity: EntityId,
        location: EntityLocation,
        component: T,
    ) -> Result<T, StateError> {
        let expected_name = std::any::type_name::<T>();
        self.replace_component_in_place(entity, location, component, |registered| {
            registered == expected_name
        })
    }

    #[inline]
    pub fn replace_component<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        entity: EntityId,
        component: T,
    ) -> Result<T, StateError> {
        self.replace(entity, component)
    }

    /// 替换已有类型擦除组件：成功时就地替换并返回拥有所有权的旧值
    pub fn replace_component_erased(
        &mut self,
        entity: EntityId,
        component: ErasedComponentValue,
    ) -> Result<ErasedComponentValue, StateError> {
        self.replace_component_value(entity, component.component_type_id(), component)
    }


    /// 按 [`ComponentTypeId`] 直接从实体所在列的物理槽位克隆类型擦除组件值，
    /// 不触发 `patch_entity` 批处理或跨 Archetype 搬迁（专用于只读提取或即将丢弃的临时 `Store` 拆装）。
    pub fn clone_component_erased(
        &self,
        entity: EntityId,
        type_id: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        let loc = self
            .allocator
            .get_location(entity)
            .ok_or(StateError::EntityNotFound(entity))?;
        let arch = self
            .archetypes
            .get(loc.archetype_id as usize)
            .ok_or(StateError::EntityNotFound(entity))?;
        let col = arch
            .columns()
            .get(&type_id)
            .ok_or_else(|| StateError::ComponentNotFound(format!("{type_id}")))?;
        col.clone_erased(loc.row_index as usize)
    }


    #[inline]
    pub fn has_component(&self, entity: EntityId, type_id: ComponentTypeId) -> bool {
        let Some(loc) = self.allocator.get_location(entity) else {
            return false;
        };
        let Some(arch) = self.archetypes.get(loc.archetype_id as usize) else {
            return false;
        };
        if arch.entities().get(loc.row_index as usize) != Some(&entity) {
            return false;
        }
        arch.component_types_slice().contains(&type_id)
    }

    /// 检查指定实体是否挂载了强类型泛型 T 组件
    #[inline]
    pub fn has_component_type<T: ModelComponent>(&self, entity: EntityId) -> bool {
        let type_id = ComponentTypeId::of::<T>();
        self.has_component(entity, type_id)
    }

    /// 检查指定实体是否持有组件 `T`（`has_component_type` 的简洁原生别名）
    #[inline]
    pub fn has<T: ModelComponent>(&self, entity: EntityId) -> bool {
        self.has_component_type::<T>(entity)
    }

    /// 形式化诊断：获取实体只读视图（若实体不存在或已销毁则返回 `StateError::EntityNotFound`）
    #[inline]
    pub fn try_entity(&self, entity: EntityId) -> Result<crate::EntityRef<'_>, StateError> {
        self.entity(entity).ok_or(StateError::EntityNotFound(entity))
    }

    /// 形式化两阶段诊断：检查实体是否持有组件 `T`（若实体不存在则返回 `StateError::EntityNotFound`）
    #[inline]
    pub fn try_has<T: ModelComponent>(&self, entity: EntityId) -> Result<bool, StateError> {
        if !self.contains_entity(entity) {
            return Err(StateError::EntityNotFound(entity));
        }
        Ok(self.has::<T>(entity))
    }

    /// 形式化两阶段诊断：获取实体上组件 `T` 的只读借用（精确区分 `EntityNotFound` 与 `ComponentNotFoundInEntity`）
    #[inline]
    pub fn try_get<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
        entity: EntityId,
    ) -> Result<&T, StateError> {
        let entity_ref = self.try_entity(entity)?;
        entity_ref.try_get::<T>()
    }

    /// 形式化两阶段诊断：获取实体上组件 `T` 的可变借用（精确区分 `EntityNotFound` 与 `ComponentNotFoundInEntity`）
    #[inline]
    pub fn try_get_mut<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        entity: EntityId,
    ) -> Result<&mut T, StateError> {
        let loc = self
            .allocator
            .get_location(entity)
            .ok_or(StateError::EntityNotFound(entity))?;
        let next_epoch = self.next_epoch()?;
        let cid = ComponentTypeId::of::<T>();
        let arch = self
            .archetypes
            .get_mut(loc.archetype_id as usize)
            .ok_or(StateError::EntityNotFound(entity))?;
        let row = loc.row_index as usize;
        if arch.entities().get(row) != Some(&entity) {
            return Err(StateError::EntityNotFound(entity));
        }
        let typed = arch.get_typed_column_mut::<T>(cid, entity)?;
        let val = typed
            .get_mut(row)
            .ok_or(StateError::EntityNotFound(entity))?;
        self.epoch = next_epoch;
        Ok(val)
    }

    /// 形式化三阶段诊断：查询源实体 `source` 在关系 `R` 下指向的目标实体（精确区分源实体缺失、关系缺失、目标实体已销毁）
    #[inline]
    pub fn try_pair_target<R: ModelComponent + Serialize + DeserializeOwned + Clone + 'static>(
        &self,
        source: EntityId,
    ) -> Result<EntityId, StateError> {
        let pair = self.try_get::<crate::al::Pair<R, EntityId>>(source)?;
        if self.contains_entity(pair.target) {
            Ok(pair.target)
        } else {
            Err(StateError::EntityNotFound(pair.target))
        }
    }

    /// 获取指定实体上组件 `T` 的独占可变借用（推进存储 Epoch，无结构迁移）
    #[inline]
    pub fn get_mut<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        entity: EntityId,
    ) -> Option<&mut T> {
        let loc = self.allocator.get_location(entity)?;
        let next_epoch = self.next_epoch().ok()?;
        let cid = ComponentTypeId::of::<T>();
        let arch = self.archetypes.get_mut(loc.archetype_id as usize)?;
        let row = loc.row_index as usize;
        if arch.entities().get(row) != Some(&entity) {
            return None;
        }
        let typed = arch.get_typed_column_mut::<T>(cid, entity).ok()?;
        let val = typed.get_mut(row)?;
        self.epoch = next_epoch;
        Some(val)
    }

    /// 即时插入或覆盖实体上的组件 `T`（Upsert，返回旧值）
    #[inline]
    pub fn set<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        entity: EntityId,
        component: T,
    ) -> Result<Option<T>, StateError> {
        let (slot, loc) = self
            .allocator
            .local_slot_and_location(entity)
            .ok_or(StateError::EntityNotFound(entity))?;
        let (old, _) = self.upsert_component_at_location(entity, Some(slot), loc, component)?;
        Ok(old)
    }

    /// 即时为实体新增组件 `T`（若组件已存在则返回 `DuplicateComponentInEntity` 错误）
    #[inline]
    pub fn add<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        entity: EntityId,
        component: T,
    ) -> Result<(), StateError> {
        let (slot, loc) = self
            .allocator
            .local_slot_and_location(entity)
            .ok_or(StateError::EntityNotFound(entity))?;
        self.insert_component_at_location(entity, Some(slot), loc, component)?;
        Ok(())
    }

    /// 即时替换实体上已有的组件 `T` 并返回旧值（若组件不存在则返回 `ComponentNotFoundInEntity` 错误）
    #[inline]
    pub fn replace<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        entity: EntityId,
        component: T,
    ) -> Result<T, StateError> {
        let loc = self
            .allocator
            .get_location(entity)
            .ok_or(StateError::EntityNotFound(entity))?;
        self.replace_at(entity, loc, component)
    }

    /// 即时移除实体上的组件 `T` 并返回旧值（若组件不存在则返回 `Ok(None)`，若实体不存在则返回 `EntityNotFound`）
    #[inline]
    pub fn remove<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        entity: EntityId,
    ) -> Result<Option<T>, StateError> {
        let Some((slot, loc)) = self.allocator.local_slot_and_location(entity) else {
            return Err(StateError::EntityNotFound(entity));
        };
        match self.remove_component_at_location::<T>(entity, Some(slot), loc) {
            Ok((removed, _)) => Ok(Some(removed)),
            Err(StateError::ComponentNotFoundInEntity { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// 保留的全局单例宿主实体 ID。
    pub const SINGLETON_ENTITY: EntityId =
        EntityId::from_u128(0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFE);

    /// 设置（插入或覆盖）全局单例组件 `T`，返回被替换的旧值（若首次设置则返回 `Ok(None)`）。
    #[inline]
    pub fn set_singleton<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        value: T,
    ) -> Result<Option<T>, StateError> {
        if !self.contains_entity(Self::SINGLETON_ENTITY) {
            self.spawn_with_id(Self::SINGLETON_ENTITY)?;
        }
        Ok(self
            .set(Self::SINGLETON_ENTITY, Singleton(value))?
            .map(|s| s.0))
    }

    /// 零开销探针：获取全局单例组件 `T` 的只读引用（未设置时返回 `None`）。
    #[inline]
    pub fn singleton<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> Option<&T> {
        self.get::<Singleton<T>>(Self::SINGLETON_ENTITY)
            .map(|s| &s.0)
    }

    /// 形式化诊断：获取全局单例组件 `T` 的只读引用（未设置时返回 `StateError`）。
    #[inline]
    pub fn try_singleton<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> Result<&T, StateError> {
        self.try_get::<Singleton<T>>(Self::SINGLETON_ENTITY)
            .map(|s| &s.0)
    }

    /// 零开销探针：获取全局单例组件 `T` 的可变引用（未设置时返回 `None`，自动推进 Epoch）。
    #[inline]
    pub fn singleton_mut<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
    ) -> Option<&mut T> {
        self.get_mut::<Singleton<T>>(Self::SINGLETON_ENTITY)
            .map(|s| &mut s.0)
    }

    /// 形式化诊断：获取全局单例组件 `T` 的可变引用（未设置时返回 `StateError`，自动推进 Epoch）。
    #[inline]
    pub fn try_singleton_mut<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
    ) -> Result<&mut T, StateError> {
        self.try_get_mut::<Singleton<T>>(Self::SINGLETON_ENTITY)
            .map(|s| &mut s.0)
    }

    /// 移除全局单例组件 `T` 并返回旧值（未设置时返回 `Ok(None)`）。
    #[inline]
    pub fn remove_singleton<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
    ) -> Result<Option<T>, StateError> {
        if !self.contains_entity(Self::SINGLETON_ENTITY) {
            return Ok(None);
        }
        Ok(self
            .remove::<Singleton<T>>(Self::SINGLETON_ENTITY)?
            .map(|s| s.0))
    }

    /// 判断当前是否已设置全局单例组件 `T`。
    #[inline]
    pub fn has_singleton<T: ModelComponent + 'static>(&self) -> bool {
        self.has::<Singleton<T>>(Self::SINGLETON_ENTITY)
    }

    /// 为源实体 `source` 设置指向 `target` 的关系对 `Pair<R, EntityId>`（校验 `source` 与 `target` 均存活）
    #[inline]
    pub fn set_pair<R: ModelComponent + Serialize + DeserializeOwned + Clone + 'static>(
        &mut self,
        source: EntityId,
        relation: R,
        target: EntityId,
    ) -> Result<Option<crate::al::Pair<R, EntityId>>, StateError> {
        if !self.contains_entity(target) {
            return Err(StateError::EntityNotFound(target));
        }
        self.set(source, crate::al::Pair::new(relation, target))
    }

    /// 移除源实体 `source` 上指向 `target` 的关系对 `Pair<R, EntityId>`
    #[inline]
    pub fn remove_pair<R: ModelComponent + Serialize + DeserializeOwned + Clone + 'static>(
        &mut self,
        source: EntityId,
        target: EntityId,
    ) -> Result<Option<crate::al::Pair<R, EntityId>>, StateError> {
        let matches = self
            .get::<crate::al::Pair<R, EntityId>>(source)
            .is_some_and(|p| p.target == target);
        if matches {
            self.remove::<crate::al::Pair<R, EntityId>>(source)
        } else if !self.contains_entity(source) {
            Err(StateError::EntityNotFound(source))
        } else {
            Ok(None)
        }
    }

    /// 查询源实体 `source` 在关系 `R` 下指向的目标实体（若目标已销毁则返回 `None`）
    #[inline]
    pub fn pair_target<R: ModelComponent + Serialize + DeserializeOwned + Clone + 'static>(
        &self,
        source: EntityId,
    ) -> Option<EntityId> {
        let target = self.get::<crate::al::Pair<R, EntityId>>(source)?.target;
        if self.contains_entity(target) {
            Some(target)
        } else {
            None
        }
    }

    /// 反查所有通过关系 `R` 指向 `target` 的存活源实体列表
    pub fn sources_with_pair<R: ModelComponent + Serialize + DeserializeOwned + Clone + 'static>(
        &self,
        target: EntityId,
    ) -> Vec<EntityId> {
        if !self.contains_entity(target) {
            return Vec::new();
        }
        let cid = ComponentTypeId::of::<crate::al::Pair<R, EntityId>>();
        let mut out = Vec::new();
        for arch in &self.archetypes {
            let len = arch.len();
            if len == 0 || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            if let Some(col) = arch.get_column::<crate::al::Pair<R, EntityId>>() {
                let entities = arch.entities();
                for (i, pair) in col.as_slice()[..len].iter().enumerate() {
                    if pair.target == target {
                        out.push(entities[i]);
                    }
                }
            }
        }
        out
    }

    /// 按谓词扫描并返回第一个匹配的实体 ID
    pub fn find_entity_by<C: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
        predicate: impl Fn(&C) -> bool,
    ) -> Option<EntityId> {
        let cid = ComponentTypeId::of::<C>();
        for arch in &self.archetypes {
            let len = arch.len();
            if len == 0 || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            if let Some(col) = arch.get_column::<C>() {
                let entities = arch.entities();
                for (i, item) in col.as_slice()[..len].iter().enumerate() {
                    if predicate(item) {
                        return Some(entities[i]);
                    }
                }
            }
        }
        None
    }

    /// 批量替换已有实体的单组件或多组件元组值（支持 `(EntityId, T0)` 至 `(EntityId, T0..T15)`）并返回提交收据 `BatchReceipt`
    #[inline]
    pub fn update_batch<Item: crate::storage::stage::commands::BatchUpdateItem>(
        &mut self,
        items: impl IntoIterator<Item = Item>,
    ) -> Result<crate::BatchReceipt, StateError> {
        let mut cmd = self.commands();
        cmd.update_batch(items);
        cmd.apply(self)
    }

    /// 批量替换已有实体的单组件或多组件元组值（支持 `(EntityId, T0)` 至 `(EntityId, T0..T15)`，走极速直写通道且不保留回滚收据）
    #[inline]
    pub fn update_batch_discard<
        Item: crate::storage::core::store::direct::TypedDirectBatchUpdate,
    >(
        &mut self,
        items: impl IntoIterator<Item = Item>,
    ) -> Result<(), StateError> {
        self.update_batch_typed_tuple_discard(items)
    }

    /// 按给定实体顺序重排其所在原型的物理行顺序，返回发生位置移动的实体数量。
    pub fn reorder_entities(&mut self, entities: &[EntityId]) -> Result<usize, StateError> {
        let Some(&first) = entities.first() else {
            return Ok(0);
        };
        let arch_id = self
            .allocator
            .get_location(first)
            .ok_or(StateError::EntityNotFound(first))?
            .archetype_id;
        let arch_len = self.archetypes[arch_id as usize].len();
        let original = self.archetypes[arch_id as usize].entities().to_vec();

        let mut seen = std::collections::BTreeSet::new();
        let mut order = Vec::with_capacity(arch_len);
        for &id in entities {
            if !seen.insert(id) {
                return Err(StateError::BatchConflict(
                    "Duplicate entity in reorder_entities".into(),
                ));
            }
            let loc = self
                .allocator
                .get_location(id)
                .ok_or(StateError::EntityNotFound(id))?;
            if loc.archetype_id != arch_id {
                return Err(StateError::RangeNotContiguous);
            }
            order.push(loc.row_index as usize);
        }
        order.extend(
            original
                .iter()
                .enumerate()
                .filter_map(|(row, id)| (!seen.contains(id)).then_some(row)),
        );

        let moved_entities = order
            .iter()
            .enumerate()
            .filter(|&(i, &row)| i != row)
            .count();
        let slots_aligned = self.allocator.are_entities_consecutive_slots(&original);
        let arch = &mut self.archetypes[arch_id as usize];
        if moved_entities == 0 && slots_aligned && !arch.may_be_segmented_or_unaligned() {
            return Ok(0);
        }

        for col in arch.columns_mut().values_mut() {
            *col = col.reordered_column(&order, None)?;
        }
        let new_entities: Vec<EntityId> = order.iter().map(|&row| original[row]).collect();
        self.allocator
            .align_archetype_slots(arch_id, &original, &new_entities);
        let arch = &mut self.archetypes[arch_id as usize];
        *arch.entities_mut() = new_entities;
        arch.rebuild_flat_columns();
        self.epoch = self.epoch.saturating_add(1);

        Ok(moved_entities)
    }

    /// 获取指定实体挂载的所有 ComponentTypeId 列表
    pub fn component_types(&self, entity: EntityId) -> Option<Vec<ComponentTypeId>> {
        let loc = self.allocator.get_location(entity)?;
        let arch = self.archetypes.get(loc.archetype_id as usize)?;
        if arch.entities().get(loc.row_index as usize) != Some(&entity) {
            return None;
        }
        Some(arch.component_types().iter().copied().collect())
    }

    /// 获取底层分配器的堆内存估算字节数（包含槽位向量、空闲槽位向量与身份哈希索引）
    #[inline]
    pub fn allocator_heap_bytes(&self) -> usize {
        self.allocator.heap_bytes_estimate()
    }

    /// 获取底层分配器身份索引哈希表当前的预分配容量
    #[inline]
    pub fn allocator_index_capacity(&self) -> usize {
        self.allocator.index_capacity()
    }

    /// 预留指定数量的实体分配容量，避免后续批量插入时哈希表和槽位数组重新分配
    #[inline]
    pub fn reserve_entities(&mut self, additional: usize) -> Result<(), StateError> {
        self.allocator.prepare_allocations(additional)
    }
}

