//! 阶段 2：批次提交与流水线管道 (Phase 2: Commit & Pipeline Execution)

use super::*;
use super::Store;
use super::CorePoisonGuard;
use crate::storage::allocator::EntityLocation;
use crate::storage::batch::{
    BatchReceipt,
    PreparedBatch, StoreBatch,
};
use crate::storage::core::column::InPlaceReplacements;
use crate::storage::batch::planned_patch_values::{
    InPlaceValueSource, InPlaceValuesIter, MixedInPlaceValuesIter,
};
use crate::storage::column::AnyColumn;
use crate::error::StateError;
use crate::id::ComponentTypeId;

use super::scratch::NativeDirectOps;

impl Store {
    /// 阶段 2：消费执行计划并集中提交（前置版本校验，失败不可发生）
    pub fn commit_batch(&mut self, prepared: PreparedBatch) -> Result<BatchReceipt, StateError> {
        self.commit_batch_with_receipt::<true>(prepared)
    }

    /// 阶段 2（丢弃收据，仅限包内与测试专用）：消费执行计划并集中提交，不保留收据（前置版本校验，失败不可发生）
    #[inline]
    pub(crate) fn commit_batch_discard(&mut self, prepared: PreparedBatch) -> Result<(), StateError> {
        self.commit_batch_with_receipt::<false>(prepared).map(|_| ())
    }

    fn commit_batch_with_receipt<const KEEP: bool>(
        &mut self,
        prepared: PreparedBatch,
    ) -> Result<BatchReceipt, StateError> {
        if prepared.in_place_values.has_owned() {
            self.commit_batch_with_values::<KEEP, MixedInPlaceValuesIter>(prepared)
        } else {
            self.commit_batch_with_values::<KEEP, InPlaceValuesIter>(prepared)
        }
    }

    fn commit_batch_with_values<const KEEP: bool, Values: InPlaceValueSource>(
        &mut self,
        prepared: PreparedBatch,
    ) -> Result<BatchReceipt, StateError> {
        self.check_poisoned()?;
        let expected_mode = if KEEP {
            crate::storage::batch::PreparedCommitMode::KeepReceipt
        } else {
            crate::storage::batch::PreparedCommitMode::DiscardReceipt
        };
        if prepared.commit_mode != expected_mode {
            return Err(StateError::AccessDenied(format!(
                "Mismatched PreparedBatch commit mode: expected {expected_mode:?}, found {:?}",
                prepared.commit_mode
            )));
        }
        if prepared.store_id != self.store_id {
            return Err(StateError::StoreMismatch {
                expected_store: prepared.store_id,
                actual_store: self.store_id,
            });
        }
        if prepared.epoch != self.epoch {
            return Err(StateError::StaleBatch {
                expected_epoch: prepared.epoch,
                actual_epoch: self.epoch,
            });
        }

        let changes = prepared.native_plan_count != 0
            || prepared
                .plan_scratch.in_place_plans
                .iter()
                .any(|p| !p.replacements.is_empty())
            || !prepared.plans.is_empty();
        if !changes {
            let mut receipt = prepared.receipt;
            receipt.epoch = self.epoch;
            return Ok(receipt);
        }
        let has_new_archetypes = !prepared.detached.is_empty();
        if has_new_archetypes {
            self.check_archetype_set_generation_capacity()?;
        }
        self.next_epoch()?;
        // Direct assignment may execute user Drop. Invalidate previously
        // prepared plans even if a destructor unwinds after a partial write.
        self.bump_epoch();
        let commit_epoch = self.epoch;

        let PreparedBatch {
            mut plan_scratch,
            in_place_values,
            mut native_plans,
            mut structural_values,
            mut plans,
            detached,
            new_definitions,
            mut receipt,
            mut discarded_values,
            ..
        } = prepared;
        if KEEP {
            receipt.epoch = self.epoch;
        }
        let super::PlanScratch {
            in_place_plans, native_references,
        } = &mut plan_scratch;
        let mut in_place_values = Values::from_values(in_place_values);

        let mut poison_guard = CorePoisonGuard::new(&self.poisoned);
        if !new_definitions.is_empty() {
            self.component_registry.extend(new_definitions);
        }
        for mut arch in detached {
            arch.rebuild_flat_columns();
            let arch_id = arch.id().0;
            let comp_types = arch.component_types().clone();
            self.register_archetype_in_component_index(arch_id, &comp_types);
            self.archetype_index.insert(comp_types, arch_id);
            self.archetypes.push(arch);
        }
        if has_new_archetypes {
            self.bump_archetype_set_generation()?;
        }

        let total_plans = in_place_plans.len();
        let mut idx = 0;
        // Standalone native plans commit through their own prepared scratch.
        // Only linked native operations need this per-segment lane list.
        let mut bound_lanes: Vec<usize> = Vec::with_capacity(
            if native_references.is_empty() { 0 } else { native_plans.len().min(128) },
        );
        // Reuse capacity across segments, not column bindings. Allocate lazily
        // so standalone lanes and one-row fallbacks pay no table allocation.
        let mut direct_lanes = smallvec::SmallVec::<[Option<NativeDirectOps>; 8]>::new();
        while idx < total_plans {
            let arch_id = in_place_plans[idx].location.archetype_id as usize;
            match in_place_plans[idx].replacements {
                InPlaceReplacements::Single(type_id, _) => {
                    let current_type_id = type_id;
                    let start = idx;
                    let mut end = start + 1;
                    while end < total_plans {
                        let p = &in_place_plans[end];
                        if p.location.archetype_id as usize == arch_id {
                            if let InPlaceReplacements::Single(t, _) = p.replacements {
                                if t == current_type_id {
                                    end += 1;
                                    continue;
                                }
                            }
                        }
                        break;
                    }
                    let arch = &mut self.archetypes[arch_id];
                    let col = arch
                        .get_erased_column_mut(current_type_id)
                        .expect("validated component column");
                    col.discard_in_place_single_chunk(&mut in_place_plans[start..end]);
                    for p in &in_place_plans[start..end] {
                        arch.mark_column_row_epoch(
                            current_type_id,
                            p.location.row_index as usize,
                            commit_epoch,
                        );
                    }
                    idx = end;
                }
                InPlaceReplacements::Multiple(_) => {
                    let plan = &mut in_place_plans[idx];
                    let row = plan.location.row_index as usize;
                    if let InPlaceReplacements::Multiple(replacements) =
                        std::mem::replace(&mut plan.replacements, InPlaceReplacements::Empty)
                    {
                        let arch = &mut self.archetypes[arch_id];
                        in_place_values.for_each_range(replacements, |(type_id, op)| {
                            let value = op.into_value();
                            unsafe {
                                arch.get_erased_column_mut(type_id)
                                    .expect("validated component column")
                                    .assign_exclusive_unchecked(row, value);
                            }
                            arch.mark_column_row_epoch(type_id, row, commit_epoch);
                        });
                    }
                    idx += 1;
                }
                InPlaceReplacements::NativeMultiple(_) => {
                    // 识别连续同 arch_id 的 NativeMultiple 段
                    let start = idx;
                    let mut end = start + 1;
                    while end < total_plans {
                        let p = &in_place_plans[end];
                        if p.location.archetype_id as usize == arch_id {
                            if matches!(p.replacements, InPlaceReplacements::NativeMultiple(_)) {
                                end += 1;
                                continue;
                            }
                        }
                        break;
                    }
                    let segment_len = end - start;

                    if segment_len == 1 {
                        // 单实体回退到原始路径，避免绑定开销
                        let plan = &mut in_place_plans[idx];
                        if let InPlaceReplacements::NativeMultiple(range) =
                            std::mem::replace(&mut plan.replacements, InPlaceReplacements::Empty)
                        {
                            let row = plan.location.row_index as usize;
                            let arch = &mut self.archetypes[arch_id];
                            for (cid, reference) in &native_references[range] {
                                native_plans[reference.lane()]
                                    .assign_linked(reference.row(), row, arch);
                                arch.mark_column_row_epoch(*cid, row, commit_epoch);
                            }
                        }
                    } else {
                        // 多实体段：预绑定所有涉及的 Lane 的列切片指针与块纪元指针
                        bound_lanes.clear();
                        let mut seen_lanes = [0u64; 4];
                        for i in start..end {
                            if let InPlaceReplacements::NativeMultiple(ref range) = in_place_plans[i].replacements {
                                for (_, reference) in &native_references[range.clone()] {
                                    let lane_idx = reference.lane();
                                    let word = lane_idx / 64;
                                    let is_new = if word < 4 {
                                        let mask = 1u64 << (lane_idx % 64);
                                        let unseen = (seen_lanes[word] & mask) == 0;
                                        seen_lanes[word] |= mask;
                                        unseen
                                    } else {
                                        !bound_lanes.contains(&lane_idx)
                                    };
                                    if is_new {
                                        bound_lanes.push(lane_idx);
                                    }
                                }
                            }
                        }

                        {
                            let arch = &mut self.archetypes[arch_id];
                            for &lane_idx in &bound_lanes {
                                native_plans[lane_idx].bind_linked_column(arch);
                            }
                        }

                        if direct_lanes.is_empty() {
                            direct_lanes.resize(native_plans.len(), None);
                        }
                        for &lane_idx in &bound_lanes {
                            let ops = native_plans[lane_idx].direct_ops();
                            direct_lanes[lane_idx] = Some(ops);
                        }

                        for i in start..end {
                            let plan = &mut in_place_plans[i];
                            if let InPlaceReplacements::NativeMultiple(range) =
                                std::mem::replace(&mut plan.replacements, InPlaceReplacements::Empty)
                            {
                                let row = plan.location.row_index as usize;
                                for (_, reference) in &native_references[range] {
                                    unsafe {
                                        direct_lanes[reference.lane()]
                                            .as_ref().expect("lane bound for current segment")
                                            .assign(reference.row(), row, commit_epoch);
                                    }
                                }
                            }
                        }

                        // 解绑所有 Lane
                        for &lane_idx in &bound_lanes {
                            direct_lanes[lane_idx] = None;
                            native_plans[lane_idx].unbind_linked_column();
                        }
                    }
                    idx = end;
                }
                InPlaceReplacements::Takeover(_, _) => {
                    unreachable!("takeover replacements are not used in contiguous mode");
                }
                InPlaceReplacements::Empty => {
                    idx += 1;
                }
            }
        }

        // Remaining standalone targets belong to distinct, non-structural entities.
        // Linked targets have already executed in the entity-major loop above.
        // Run before any migration can invalidate their prepared locations.
        for lane in native_plans.iter_mut() {
            lane.commit_discard(&mut self.archetypes, commit_epoch);
        }
        // Recycle while the poison guard is still armed, including user Drop.
        drop(native_plans);

        for plan in plans.drain(..) {
            match plan {
                PlannedOp::Create {
                    entity,
                    target_archetype,
                    components,
                } => {
                    let target_arch = &mut self.archetypes[target_archetype as usize];
                    let comp_slice = structural_values.take_slice_mut(&components);
                    if target_arch.flat_columns().is_empty() && !comp_slice.is_empty() {
                        target_arch.rebuild_flat_columns();
                    }
                    let flat_cols = target_arch.flat_columns();
                    debug_assert_eq!(flat_cols.len(), comp_slice.len());
                    for (&col_entry, (_, val)) in flat_cols.iter().zip(comp_slice.iter_mut()) {
                        let value = val.as_mut().expect("validated component value");
                        unsafe {
                            (col_entry.push_erased_fn)(col_entry.col_ptr as *mut (), value);
                        }
                    }
                    let row = target_arch.len() as u32;
                    let local_slot = self
                        .allocator
                        .allocate_with_location(
                            entity,
                            EntityLocation::new(target_archetype, row),
                        )
                        .expect("validated allocation");
                    target_arch.push_entity_with_slot(entity, local_slot);
                    target_arch.mark_all_columns_row_epoch(row as usize, commit_epoch);
                }
                PlannedOp::CreatePooled {
                    entity,
                    target_archetype,
                    pool_start,
                    perm,
                    len,
                } => {
                    let target_arch = &mut self.archetypes[target_archetype as usize];
                    let len = len as usize;
                    if target_arch.flat_columns().is_empty() && len > 0 {
                        target_arch.rebuild_flat_columns();
                    }
                    let flat_cols = target_arch.flat_columns();
                    debug_assert_eq!(flat_cols.len(), len);
                    let pool_slice = structural_values.pool_slice_mut(pool_start as usize, len);
                    for (col_entry, &p) in flat_cols.iter().zip(perm[..len].iter()) {
                        let pe = &mut pool_slice[p as usize];
                        let value = match &mut pe.op {
                            crate::storage::batch::PatchOp::Insert(v)
                            | crate::storage::batch::PatchOp::Replace(v)
                            | crate::storage::batch::PatchOp::Upsert(v) => v,
                            crate::storage::batch::PatchOp::Remove(_) => unreachable!(),
                        };
                        unsafe {
                            (col_entry.push_erased_fn)(col_entry.col_ptr as *mut (), value);
                        }
                    }
                    let row = target_arch.len() as u32;
                    let local_slot = self
                        .allocator
                        .allocate_with_location(
                            entity,
                            EntityLocation::new(target_archetype, row),
                        )
                        .expect("validated allocation");
                    target_arch.push_entity_with_slot(entity, local_slot);
                    target_arch.mark_all_columns_row_epoch(row as usize, commit_epoch);
                }
                PlannedOp::StructuralPatch {
                    entity,
                    location,
                    target_archetype,
                    new_components,
                    replaced_components,
                    removed_components,
                    ..
                } => {
                    let loc = self.allocator.get_location(entity).unwrap_or(location);
                    let src_arch_idx = loc.archetype_id as usize;
                    let src_row = loc.row_index as usize;

                    let dst_arch_idx = target_archetype as usize;

                    let (src_arch, dst_arch) = if src_arch_idx < dst_arch_idx {
                        let (left, right) = self.archetypes.split_at_mut(dst_arch_idx);
                        (&mut left[src_arch_idx], &mut right[0])
                    } else {
                        let (left, right) = self.archetypes.split_at_mut(src_arch_idx);
                        (&mut right[0], &mut left[dst_arch_idx])
                    } ;

                    for (&t_id, dst_col) in dst_arch.columns_mut().iter_mut() {
                        if let Some(new_val) = structural_values.take(&new_components, t_id) {
                            dst_col.push_erased_infallible(new_val);
                        } else if let Some(rep_val) = structural_values.take(&replaced_components, t_id) {
                            let src_col = src_arch
                                .columns_mut()
                                .get_mut(&t_id)
                                .expect("src column exists");
                            let old_val = src_col.replace_erased_infallible(src_row, rep_val);
                            src_col.move_row_to_infallible(src_row, dst_col.as_mut());
                            discarded_values.push(old_val);
                        } else if let Some(src_col) = src_arch.columns_mut().get_mut(&t_id) {
                            src_col.move_row_to_infallible(src_row, dst_col.as_mut());
                        } else {
                            panic!("Column {t_id} missing in migration");
                        }
                    }

                    for &t_id in structural_values.removed(removed_components.clone()) {
                        if let Some(src_col) = src_arch.columns_mut().get_mut(&t_id) {
                            let old_val = src_col.swap_remove_erased(src_row);
                            discarded_values.push(old_val);
                        }
                    }

                    let entity_slot = self.allocator.local_slot(entity);
                    let dst_row = if let Some(slot) = entity_slot {
                        dst_arch.push_entity_with_slot(entity, slot) as u32
                    } else {
                        dst_arch.push_entity(entity) as u32
                    };
                    src_arch.copy_row_flags_to(src_row, dst_arch, dst_row as usize);
                    dst_arch.mark_all_columns_row_epoch(dst_row as usize, commit_epoch);

                    let is_last = src_row == src_arch.entities().len() - 1;
                    src_arch.swap_remove_entity_only_at_epoch(src_row, commit_epoch);
                    if !is_last {
                        let moved = src_arch.entities()[src_row];
                        self.allocator.set_location_infallible(
                            moved,
                            EntityLocation::new(src_arch_idx as u32, src_row as u32),
                        );
                    }

                    if let Some(slot) = entity_slot {
                        self.allocator.set_location_by_slot_infallible(
                            slot,
                            EntityLocation::new(dst_arch_idx as u32, dst_row),
                        );
                    } else {
                        self.allocator.set_location_infallible(
                            entity,
                            EntityLocation::new(dst_arch_idx as u32, dst_row),
                        );
                    }
                    src_arch.rebuild_flat_columns();
                    dst_arch.rebuild_flat_columns();
                }
                PlannedOp::Delete { entity, location, .. } => {
                    let loc = self.allocator.get_location(entity).unwrap_or(location);
                    let arch_id = loc.archetype_id;
                    let arch = &mut self.archetypes[arch_id as usize];
                    for (&_cid, col) in arch.columns_mut().iter_mut() {
                        discarded_values.push(col.swap_remove_erased(loc.row_index as usize));
                    }

                    let is_last = loc.row_index as usize == arch.entities().len() - 1;
                    arch.swap_remove_entity_only_at_epoch(loc.row_index as usize, commit_epoch);
                    if !is_last {
                        let moved = arch.entities()[loc.row_index as usize];
                        self.allocator.set_location_infallible(moved, loc);
                    }
                    arch.rebuild_flat_columns();
                    self.allocator.deallocate_infallible(entity);
                }
            }
            // No user destructor runs while a structural row is half moved.
            discarded_values.clear();
        }
        structural_values.recycle_plans(plans);

        // 毒化保护覆盖到清理结束：
        // 所有被替换值的析构必须在 poison_guard 武装范围内完成！
        // 若任何用户析构器抛出 panic，Store 立即被标记为永久中毒。
        drop(discarded_values);
        drop(plan_scratch);
        poison_guard.disarm();
        Ok(receipt)
    }

    /// 快捷接口：内部执行 prepare 并立即 commit
    pub fn apply_batch(&mut self, batch: StoreBatch) -> Result<BatchReceipt, StateError> {
        let prepared = self.prepare_batch(batch)?;
        self.commit_batch(prepared)
    }

    // =========================================================================
    // 内部 Archetype 管理与辅助例程
    // =========================================================================

    #[inline]
    pub(crate) fn find_prototype(&self, type_id: ComponentTypeId) -> Option<Box<dyn AnyColumn>> {
        self.component_registry.spawn_column(type_id)
    }
}
