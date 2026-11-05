//! Homogeneous recording owned by CommandBuffer. No separate transaction,
//! store index, version domain, commit engine, or receipt implementation.
use crate::storage::ErasedComponentValue;
use crate::storage::{
    allocator::EntityLocation,
    archetype::Archetype,
    column::TypedColumn,
};
use super::*;
use crate::ComponentTypeId;
use std::sync::atomic::AtomicPtr;

#[derive(Debug)]
struct NativeTarget {
    // KEEP borrows the ID already owned by BatchReceipt::updated_entities.
    // Discard never reads this index. Do not duplicate the 128-bit identity.
    receipt_index: u32,
    row_index: u32,
    value_row: u32,
    lane_arch_idx: u16,
}

struct NativeRow<T> {
    value: Option<T>,
    // Links only operations on this same entity; not a second entity index.
    next: Option<NativeUpdateRef>,
}

#[derive(Debug)]
struct ArchetypeValidationCache {
    mask_lo: u64,
    mask_hi: u64,
    // u32::MAX means type-validated only; real bound indices still fit u16.
    // Linked targets must not consume the single-target pointer index space.
    arch_to_lane_idx: [u32; 128],
    overflow: Option<std::collections::HashMap<u32, u32>>,
}

impl Default for ArchetypeValidationCache {
    fn default() -> Self {
        Self {
            mask_lo: 0,
            mask_hi: 0,
            arch_to_lane_idx: [0; 128],
            overflow: None,
        }
    }
}

impl ArchetypeValidationCache {
    #[inline]
    fn get_lane_arch_idx(&self, arch_id: u32) -> Option<u16> {
        self.get_validated_entry(arch_id).and_then(|index| u16::try_from(index).ok())
    }

    #[inline]
    fn get_validated_entry(&self, arch_id: u32) -> Option<u32> {
        if arch_id < 64 {
            if (self.mask_lo & (1u64 << arch_id)) != 0 {
                Some(self.arch_to_lane_idx[arch_id as usize])
            } else {
                None
            }
        } else if arch_id < 128 {
            if (self.mask_hi & (1u64 << (arch_id - 64))) != 0 {
                Some(self.arch_to_lane_idx[arch_id as usize])
            } else {
                None
            }
        } else {
            self.overflow.as_ref().and_then(|m| m.get(&arch_id).copied())
        }
    }

    #[inline]
    fn register_validated(&mut self, arch_id: u32, lane_arch_idx: u32) {
        if arch_id < 64 {
            self.mask_lo |= 1u64 << arch_id;
            self.arch_to_lane_idx[arch_id as usize] = lane_arch_idx;
        } else if arch_id < 128 {
            self.mask_hi |= 1u64 << (arch_id - 64);
            self.arch_to_lane_idx[arch_id as usize] = lane_arch_idx;
        } else {
            self.overflow
                .get_or_insert_with(std::collections::HashMap::new)
                .insert(arch_id, lane_arch_idx);
        }
    }
}

struct Values<T> {
    type_id: ComponentTypeId,
    type_verified: bool,
    last_arch_id: u32,
    last_lane_arch_idx: u16,
    // Pure native targets (single or linked) stay typed through commit. Mixed
    // ordinary/structural patches use the shared erased planner.
    rows: Vec<NativeRow<T>>,
    targets: Vec<NativeTarget>,
    validated_archetypes: ArchetypeValidationCache,
    touched_archetypes: Vec<u32>,
    // Prepared scratch contains ONLY nulls; bind addresses after exclusive commit
    // begins. AtomicPtr keeps the owned plan Send + Sync without unsafe impls.
    // All access uses get_mut(), so there are no atomic loads/stores in the loop.
    pointer_scratch: Vec<AtomicPtr<T>>,
    // NativeMultiple 路径中用于预绑定列切片指针。
    // bind_linked_column 设置非空值，unbind_linked_column 清零。
    linked_bound_ptr: AtomicPtr<T>,
}

impl<T> Values<T> {
    #[inline]
    fn new_lane(type_id: ComponentTypeId) -> Self {
        Self {
            type_id,
            type_verified: false,
            last_arch_id: u32::MAX,
            last_lane_arch_idx: 0,
            rows: Vec::new(),
            targets: Vec::new(),
            validated_archetypes: ArchetypeValidationCache::default(),
            touched_archetypes: Vec::new(),
            pointer_scratch: Vec::new(),
            linked_bound_ptr: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

impl<T> std::fmt::Debug for Values<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeUpdates")
            .field("rows", &self.rows.len())
            .field("touched_archetypes", &self.touched_archetypes.len())
            .finish()
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned> NativeLane for Values<T> {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn clear_for_reuse(&mut self) {
        self.rows.clear();
        self.targets.clear();
        self.touched_archetypes.clear();
        self.pointer_scratch.clear();
        let cap = 65_536;
        if self.rows.capacity() > cap { self.rows = Vec::new(); }
        if self.targets.capacity() > cap { self.targets = Vec::new(); }
        if self.touched_archetypes.capacity() > cap { self.touched_archetypes = Vec::new(); }
        if self.pointer_scratch.capacity() > cap { self.pointer_scratch = Vec::new(); }
        self.type_verified = false;
        self.last_arch_id = u32::MAX;
        self.last_lane_arch_idx = 0;
        self.validated_archetypes.mask_lo = 0;
        self.validated_archetypes.mask_hi = 0;
        self.validated_archetypes.overflow = None;
        *self.linked_bound_ptr.get_mut() = std::ptr::null_mut();
    }

    fn cached_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.rows.capacity() * std::mem::size_of::<NativeRow<T>>()
            + self.targets.capacity() * std::mem::size_of::<NativeTarget>()
            + self.touched_archetypes.capacity() * std::mem::size_of::<u32>()
            + self.pointer_scratch.capacity() * std::mem::size_of::<AtomicPtr<T>>()
    }
    fn metadata(&self) -> NativeType {
        NativeType {
            type_id: self.type_id,
            rust_type_id: std::any::TypeId::of::<T>(),
            name: std::any::type_name::<T>(),
        }
    }
    fn empty_column(&self) -> Box<dyn crate::storage::column::AnyColumn> {
        Box::new(crate::storage::column::TypedColumn::<T>::new(
            std::any::type_name::<T>(),
        ))
    }
    fn next(&self, row: usize) -> Option<NativeUpdateRef> {
        self.rows[row].next
    }
    fn take(&mut self, row: usize) -> (ErasedComponentValue, Option<NativeUpdateRef>) {
        (
            ErasedComponentValue::with_component_type_id(
                self.type_id,
                self.rows[row].value.take().expect("owned native row"),
            ),
            self.rows[row].next,
        )
    }

    fn try_plan_single(
        &mut self,
        row: usize,
        entity: EntityId,
        location: EntityLocation,
        archetype: &Archetype,
        receipt_index: usize,
    ) -> Result<bool, StateError> {
        // Decide eligibility inside the same typed call as planning. A linked
        // multi-component entity remains entirely in the shared general plan.
        if self.rows[row].next.is_some() {
            return Ok(false);
        }
        // Reusing a type lane across calls must not turn the compact binding
        // index into a new batch-wide limit. The existing linked planner uses
        // full archetype IDs and also supports a single component.
        if self.touched_archetypes.len() > u16::MAX as usize
            && self.validated_archetypes.get_lane_arch_idx(location.archetype_id).is_none()
        {
            return Ok(false);
        }
        self.plan_columnar_member(row, entity, location, archetype, receipt_index)?;
        Ok(true)
    }

    fn plan_columnar_member(
        &mut self, row: usize, entity: EntityId, location: EntityLocation,
        archetype: &Archetype, receipt_index: usize,
    ) -> Result<(), StateError> {
        let receipt_index = u32::try_from(receipt_index).map_err(|_| {
            StateError::CapacityExhausted("Native receipt index exceeds u32 limit".into())
        })?;
        let lane_arch_idx = self.validate_target(entity, location, archetype)?;

        if self.targets.is_empty() {
            self.targets
                .try_reserve_exact(self.rows.len())
                .map_err(|e| StateError::CapacityExhausted(e.to_string()))?;
        }
        let value_row = u32::try_from(row).map_err(|_| {
            StateError::CapacityExhausted("Native value row index exceeds u32 limit".into())
        })?;
        self.targets.push(NativeTarget {
            receipt_index,
            row_index: location.row_index,
            value_row,
            lane_arch_idx,
        });
        Ok(())
    }

    fn plan_linked(&mut self, value_row: usize, entity: EntityId, location: EntityLocation, archetype: &Archetype)
        -> Result<(ComponentTypeId, Option<NativeUpdateRef>), StateError>
    {
        // Share type validation, but do not allocate pointer scratch or consume
        // compact indices: linked execution binds via full archetype IDs.
        if self.validated_archetypes.get_validated_entry(location.archetype_id).is_none() {
            self.validate_column(entity, archetype)?;
            self.validated_archetypes.register_validated(location.archetype_id, u32::MAX);
        }
        Ok((self.type_id, self.rows[value_row].next))
    }

    fn replace_linked(&mut self, value_row: usize, row: usize, archetype: &mut Archetype)
        -> ErasedComponentValue
    {
        let dst = archetype.columns_mut().get_mut(&self.type_id)
            .expect("validated native column").as_any_mut()
            .downcast_mut::<TypedColumn<T>>().expect("validated native type");
        let new = self.rows[value_row].value.take().expect("owned native value");
        let old = std::mem::replace(&mut dst.as_mut_slice()[row], new);
        ErasedComponentValue::with_component_type_id(self.type_id, old)
    }

    fn assign_linked(&mut self, value_row: usize, row: usize, archetype: &mut Archetype) {
        let dst = archetype.columns_mut().get_mut(&self.type_id)
            .expect("validated native column").as_any_mut()
            .downcast_mut::<TypedColumn<T>>().expect("validated native type");
        if !std::mem::needs_drop::<T>() {
            let val_ref = self.rows[value_row].value.as_ref().expect("owned native value");
            unsafe {
                let dst_ptr = dst.as_mut_slice().as_mut_ptr().add(row);
                std::ptr::copy_nonoverlapping(val_ref as *const T, dst_ptr, 1);
            }
        } else {
            let new_val = self.rows[value_row].value.take().expect("owned native value");
            let old = std::mem::replace(&mut dst.as_mut_slice()[row], new_val);
            drop(old);
        }
    }

    fn bind_linked_column(&mut self, archetype: &mut Archetype) {
        let column = archetype.columns_mut().get_mut(&self.type_id)
            .expect("validated native column");
        let data = column.as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .expect("validated typed batch column")
            .as_mut_slice();
        // SAFETY: 裸指针存储到 AtomicPtr，在 unbind 前有效。
        // 安全契约：
        // - 提交段内已局部验证所有绑定的 type_id 互斥（disjoint），同一物理列仅建立唯一的绑定来源；
        // - 列切片在绑定有效期内不会发生重分配、扩容或结构迁移；
        // - 各实体操作互不相交的 row_index，行级物理隔离无别名冲突；
        // - 所有访问使用 get_mut() 在独占借用下完成，无原子开销。
        *self.linked_bound_ptr.get_mut() = data.as_mut_ptr();
    }

    fn unbind_linked_column(&mut self) {
        *self.linked_bound_ptr.get_mut() = std::ptr::null_mut();
    }

    fn direct_ops(&mut self) -> NativeDirectOps {
        NativeDirectOps {
            col_ptr: *self.linked_bound_ptr.get_mut() as *mut (),
            rows_ptr: self.rows.as_mut_ptr() as *mut (),
            type_id: self.type_id,
            assign_fn: Self::raw_assign_discard_impl,
            replace_fn: Self::raw_replace_keep_impl,
        }
    }

    fn planned_len(&self) -> usize {
        self.targets.len()
    }

    fn planned_archetypes(&self) -> &[u32] { &self.touched_archetypes }

    fn commit_discard(&mut self, archetypes: &mut [Archetype]) {
        self.commit_values(archetypes);
    }

    fn mark_type_verified(&mut self) {
        self.type_verified = true;
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned> Values<T> {
    #[inline(always)]
    unsafe fn raw_assign_discard_impl(
        col_ptr: *mut (),
        rows_ptr: *mut (),
        value_row: usize,
        target_row: usize,
    ) {
        unsafe {
            let dst_ptr = (col_ptr as *mut T).add(target_row);
            let src_row = (rows_ptr as *mut NativeRow<T>).add(value_row);

            if !std::mem::needs_drop::<T>() {
                // POD 极致单向写入路径：
                // 1. 无需写回 None 到暂存区（暂存区在批次结束后随 CommandBuffer 一次性整体释放）；
                // 2. 目标物理列无需读取旧值（消除 Load），单向写入新值。
                match (*src_row).value.as_ref() {
                    Some(val_ref) => {
                        std::ptr::copy_nonoverlapping(val_ref as *const T, dst_ptr, 1);
                    }
                    None => {
                        #[cfg(debug_assertions)]
                        panic!("unexpected None in native row during discard commit");
                        #[cfg(not(debug_assertions))]
                        std::hint::unreachable_unchecked();
                    }
                }
            } else {
                // Non-POD 严格异常安全路径：
                // 必须使用 std::ptr::replace 先行置换，保证目标物理列始终持有有效的新值。
                // 若旧值在 drop 时触发 panic，目标列依然处于合法已初始化状态，彻底杜绝 store 析构时的 double-free。
                match (*src_row).value.take() {
                    Some(new_val) => {
                        let old = std::ptr::replace(dst_ptr, new_val);
                        drop(old);
                    }
                    None => {
                        #[cfg(debug_assertions)]
                        panic!("unexpected None in native row during non-pod discard commit");
                        #[cfg(not(debug_assertions))]
                        std::hint::unreachable_unchecked();
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn raw_replace_keep_impl(
        col_ptr: *mut (),
        rows_ptr: *mut (),
        value_row: usize,
        target_row: usize,
        type_id: ComponentTypeId,
    ) -> ErasedComponentValue {
        unsafe {
            let dst_ptr = (col_ptr as *mut T).add(target_row);
            let src_row = (rows_ptr as *mut NativeRow<T>).add(value_row);

            // Keep 模式必须转移所有权并读取旧值纳管进 ErasedComponentValue 以供回滚
            match (*src_row).value.take() {
                Some(new_val) => {
                    let old_val = std::ptr::replace(dst_ptr, new_val);
                    ErasedComponentValue::with_component_type_id(type_id, old_val)
                }
                None => {
                    #[cfg(debug_assertions)]
                    panic!("unexpected None in native row during keep commit");
                    #[cfg(not(debug_assertions))]
                    std::hint::unreachable_unchecked();
                }
            }
        }
    }

    fn validate_column(&self, entity: EntityId, archetype: &Archetype) -> Result<(), StateError> {
        let column = archetype.columns().get(&self.type_id).ok_or_else(|| {
            StateError::ComponentNotFoundInEntity {
                entity, type_name: std::any::type_name::<T>().into(),
            }
        })?;
        if !self.type_verified {
            if column.as_any().downcast_ref::<TypedColumn<T>>().is_none()
                || column.type_id() != self.type_id
                || column.type_name() != std::any::type_name::<T>()
            {
                return Err(StateError::AccessDenied("Native batch column type mismatch".into()));
            }
        }
        Ok(())
    }

    /// Both single and linked targets reuse one concrete-type cache. The common
    /// planner validates the entity and aligned row layout on every entity.
    #[inline(always)]
    fn validate_target(&mut self, entity: EntityId, location: EntityLocation, archetype: &Archetype)
        -> Result<u16, StateError>
    {
        if self.last_arch_id == location.archetype_id {
            return Ok(self.last_lane_arch_idx);
        }
        let lane_arch_idx = match self.validated_archetypes.get_lane_arch_idx(location.archetype_id) {
            Some(idx) => idx,
            None => {
                if self.validated_archetypes.get_validated_entry(location.archetype_id).is_none() {
                    self.validate_column(entity, archetype)?;
                }
                let idx = u16::try_from(self.touched_archetypes.len()).map_err(|_| {
                    StateError::CapacityExhausted("Lane archetype count exhausted".into())
                })?;
                let touched_len = self.touched_archetypes.len() + 1;
                if touched_len > 64 {
                    self.pointer_scratch
                        .try_reserve(touched_len - self.pointer_scratch.len())
                        .map_err(|e| StateError::CapacityExhausted(e.to_string()))?;
                    self.pointer_scratch.resize_with(touched_len, || AtomicPtr::new(std::ptr::null_mut()));
                }
                self.touched_archetypes.push(location.archetype_id);
                self.validated_archetypes.register_validated(location.archetype_id, u32::from(idx));
                idx
            }
        };

        self.last_arch_id = location.archetype_id;
        self.last_lane_arch_idx = lane_arch_idx;
        Ok(lane_arch_idx)
    }
    fn commit_values(
        &mut self,
        archetypes: &mut [Archetype],
    ) {
        let touched_len = self.touched_archetypes.len();
        if self.targets.is_empty() {
            return;
        }

        if touched_len == 1 {
            let arch_id = self.touched_archetypes[0] as usize;
            let column = archetypes[arch_id]
                .columns_mut()
                .get_mut(&self.type_id)
                .expect("validated native column");
            let data = column
                .as_any_mut()
                .downcast_mut::<TypedColumn<T>>()
                .expect("validated typed batch column")
                .as_mut_slice();

            for target in self.targets.drain(..) {
                let row = target.row_index as usize;
                let new = self.rows[target.value_row as usize]
                    .value
                    .take()
                    .expect("owned native value");
                data[row] = new;
            }
            return;
        }

        let mut stack_ptrs = [const { AtomicPtr::<T>::new(std::ptr::null_mut()) }; 64];
        let col_ptrs: &mut [AtomicPtr<T>] = if touched_len <= 64 {
            &mut stack_ptrs[..touched_len]
        } else {
            &mut self.pointer_scratch[..touched_len]
        };

        for (lane_idx, &arch_id) in self.touched_archetypes.iter().enumerate() {
            let column = archetypes[arch_id as usize]
                .columns_mut()
                .get_mut(&self.type_id)
                .expect("validated native column");
            let data = column
                .as_any_mut()
                .downcast_mut::<TypedColumn<T>>()
                .expect("validated typed batch column")
                .as_mut_slice();
            *col_ptrs[lane_idx].get_mut() = data.as_mut_ptr();
        }

        for target in self.targets.drain(..) {
            let ptr = *col_ptrs[target.lane_arch_idx as usize].get_mut();
            let row = target.row_index as usize;
            let new = self.rows[target.value_row as usize]
                .value
                .take()
                .expect("owned native value");
            unsafe {
                let dst = ptr.add(row);
                *dst = new;
            }
        }
    }
}

pub(crate) fn contains_type(
    head: Option<NativeUpdateRef>,
    lanes: &[Box<dyn NativeLane>],
    type_id: ComponentTypeId,
) -> bool {
    contains_type_recording(head, lanes, type_id, |_| None)
}

fn contains_type_recording(
    mut head: Option<NativeUpdateRef>,
    lanes: &[Box<dyn NativeLane>],
    type_id: ComponentTypeId,
    active_next: impl Fn(NativeUpdateRef) -> Option<Option<NativeUpdateRef>>,
) -> bool {
    while let Some(reference) = head {
        let lane = &lanes[reference.lane()];
        if lane.metadata().type_id == type_id {
            return true;
        }
        head = active_next(reference).unwrap_or_else(|| lane.next(reference.row()));
    }
    false
}

struct RecordingRows<T> {
    lane: usize,
    type_id: ComponentTypeId,
    rows: Vec<NativeRow<T>>,
}

impl<T: ModelComponent + Serialize + DeserializeOwned> RecordingRows<T> {
    fn take(buffer: &mut CommandBuffer, lane: usize) -> Self {
        let values = buffer.native_values[lane].as_any_mut()
            .downcast_mut::<Values<T>>().expect("lane indexed by Rust type");
        Self { lane, type_id: values.type_id, rows: std::mem::take(&mut values.rows) }
    }
    fn restore(&mut self, buffer: &mut CommandBuffer) {
        let values = buffer.native_values[self.lane].as_any_mut()
            .downcast_mut::<Values<T>>().expect("recording does not replace lane types");
        values.rows = std::mem::take(&mut self.rows);
    }
}

trait RestoreRecordingRows {
    fn restore(&mut self, buffer: &mut CommandBuffer);
}

/// Temporarily owns typed rows for a monomorphized recording loop. This avoids
/// per-value downcasts and restores every lane on early error or iterator panic.
struct RecordingGuard<'a, R: RestoreRecordingRows> {
    buffer: &'a mut CommandBuffer,
    rows: R,
}
impl<R: RestoreRecordingRows> Drop for RecordingGuard<'_, R> {
    fn drop(&mut self) {
        self.rows.restore(self.buffer);
        if std::thread::panicking() {
            self.buffer.record_error(StateError::AccessDenied("Native recording iterator panicked".into()));
        }
    }
}

impl CommandBuffer {
    fn recording_lane<T: ModelComponent + Serialize + DeserializeOwned>(&mut self) -> usize {
        let rust_type = std::any::TypeId::of::<T>();
        // One lookup per batch call/type, not one lookup per entity/value.
        if let Some(i) = self.native_values.iter().position(|lane| lane.metadata().rust_type_id == rust_type) {
            return i;
        }
        let lane = Box::new(Values::<T>::new_lane(ComponentTypeId::of::<T>()));
        let index = self.native_values.len();
        self.native_values.push(lane);
        index
    }
}

/// 支持批量更新的元组项特征。
///
/// 统一抽象了单组件 `(EntityId, T0)` 到 16 元组件 `(EntityId, T0, ..., T15)` 的同批次原子批量录入机制。
/// 在录入端对同一实体仅执行一次哈希查找与冲突检查，消除多次重复录入的哈希膨胀；
/// 内部通过紧凑的 `NativeRow.next` 链表关联多个组件，并在准备与提交阶段保持完整的事务边界与原子回滚。
pub trait BatchUpdateItem: Sized + 'static {
    fn record_batch(
        buffer: &mut CommandBuffer,
        updates: impl Iterator<Item = Self>,
    );
}

macro_rules! impl_batch_update_item {
    ($($idx:tt: $T:ident $val:ident $r:ident),+) => {
        impl<$($T: ModelComponent + Serialize + DeserializeOwned),+> RestoreRecordingRows
            for ($(RecordingRows<$T>,)+)
        {
            fn restore(&mut self, buffer: &mut CommandBuffer) {
                $(self.$idx.restore(buffer);)+
            }
        }
        impl<$($T),+> BatchUpdateItem for (EntityId, $($T,)+)
        where
            $($T: ModelComponent + Serialize + DeserializeOwned + 'static,)+
        {
            fn record_batch(
                buffer: &mut CommandBuffer,
                updates: impl Iterator<Item = Self>,
            ) {
                const K: usize = 0 $(+ { let _ = stringify!($idx); 1 })+;
                if K > 1 {
                    let type_ids = [$(ComponentTypeId::of::<$T>()),+];
                    let type_names = [$(std::any::type_name::<$T>()),+];
                    for i in 0..type_ids.len() {
                        for j in (i + 1)..type_ids.len() {
                            if type_ids[i] == type_ids[j] {
                                buffer.record_error(StateError::ComponentDuplicateInPatch {
                                    entity: EntityId::from_u128(0),
                                    type_name: type_names[i].into(),
                                });
                                return;
                            }
                        }
                    }
                }

                // Preserve the caller's concrete iterator hot loop. Peekable
                // adds per-item state merely to detect an empty input.
                let iter = updates;
                if iter.size_hint().1 == Some(0) { return; }
                let indices = [$(buffer.recording_lane::<$T>()),+];
                let rows = ($(RecordingRows::<$T>::take(buffer, indices[$idx]),)+);
                let mut recording = RecordingGuard { buffer, rows };
                let buffer = &mut *recording.buffer;
                let lanes = &mut recording.rows;
                let reserve = iter
                    .size_hint()
                    .0
                    .min(buffer.limits.max_entities.unwrap_or(u32::MAX as usize))
                    .min(
                        buffer
                            .limits
                            .max_component_ops
                            .unwrap_or(usize::MAX)
                            .saturating_sub(buffer.component_op_count)
                            / K,
                    );

                $(
                    if let Err(error) = lanes.$idx.rows.try_reserve(reserve) {
                        buffer.record_error(StateError::CapacityExhausted(error.to_string()));
                        return;
                    }
                )+

                let entry_reserve = reserve.min(
                    buffer
                        .limits
                        .max_entities
                        .unwrap_or(u32::MAX as usize)
                        .saturating_sub(buffer.entries.len()),
                );
                if let Err(error) = buffer.entries.try_reserve(entry_reserve) {
                    buffer.record_error(StateError::CapacityExhausted(error.to_string()));
                    return;
                }
                super::sync_index(&buffer.entries, &mut buffer.index);
                if let Err(error) = buffer.index.try_reserve(entry_reserve) {
                    buffer.record_error(StateError::CapacityExhausted(error.to_string()));
                    return;
                }

                for (id, $($val,)+) in iter {
                    let new_ops = match buffer.component_op_count.checked_add(K) {
                        Some(n) if !buffer.limits.max_component_ops.is_some_and(|max| n > max) => n,
                        _ => {
                            buffer.record_error(StateError::CapacityExhausted(
                                "Command buffer component operations limit exceeded".into(),
                            ));
                            break;
                        }
                    };

                    let refs = match (
                        $(
                            NativeUpdateRef::new(lanes.$idx.lane, lanes.$idx.rows.len()),
                        )+
                    ) {
                        ($(Ok($r),)+) => [$($r),+],
                        _ => {
                            buffer.record_error(StateError::CapacityExhausted(
                                "Native value row index exceeds u32 limit".into(),
                            ));
                            break;
                        }
                    };

                    match buffer.index.entry(id) {
                        Entry::Vacant(vacant) => {
                            if buffer
                                .limits
                                .max_entities
                                .is_some_and(|max| buffer.entries.len() >= max)
                            {
                                buffer.record_error(StateError::CapacityExhausted(
                                    "Command buffer entity limit exceeded".into(),
                                ));
                                break;
                            }
                            let entry_idx = buffer.entries.len();
                            $(
                                let next = refs.get($idx + 1).copied();
                                lanes.$idx.rows.push(NativeRow {
                                    value: Some($val),
                                    next,
                                });
                            )+
                            buffer.entries.push((id, CommandEntry::NativeUpdate(refs[0])));
                            vacant.insert(entry_idx);
                            buffer.component_op_count = new_ops;
                            buffer.in_place_patches += 1;
                        }
                        Entry::Occupied(occupied) => {
                            let idx = *occupied.get();
                            let entry = &buffer.entries[idx].1;
                            let head = entry.native_head();
                            let contains = |type_id| contains_type_recording(
                                head, &buffer.native_values, type_id, |reference| {
                                    $(
                                        if reference.lane() == lanes.$idx.lane {
                                            return Some(lanes.$idx.rows[reference.row()].next);
                                        }
                                    )+
                                    None
                                },
                            );
                            let mut duplicate_name: Option<&'static str> = None;
                            $(
                                if duplicate_name.is_none() {
                                    let dup = head.is_some_and(|r| r.lane() == lanes.$idx.lane)
                                        || contains(lanes.$idx.type_id)
                                        || match entry {
                                            CommandEntry::Patch(patch) => {
                                                patch.ops().contains_key_pooled(&lanes.$idx.type_id, &buffer.pool)
                                            }
                                            _ => false,
                                        };
                                    if dup {
                                        duplicate_name = Some(std::any::type_name::<$T>());
                                    }
                                }
                            )+
                            if let Some(type_name) = duplicate_name {
                                buffer.record_error(StateError::ComponentDuplicateInPatch {
                                    entity: id,
                                    type_name: type_name.into(),
                                });
                            } else if matches!(entry, CommandEntry::Create(_) | CommandEntry::Delete) {
                                $(
                                    buffer.entity(id).replace($val);
                                )+
                            } else {
                                $(
                                    let next = refs.get($idx + 1).copied().or(head);
                                    lanes.$idx.rows.push(NativeRow {
                                        value: Some($val),
                                        next,
                                    });
                                )+

                                let old = std::mem::replace(
                                    &mut buffer.entries[idx].1,
                                    CommandEntry::NativeUpdate(refs[0]),
                                );
                                match old {
                                    CommandEntry::Patch(mut patch) => {
                                        patch.native_head = Some(refs[0]);
                                        buffer.entries[idx].1 = CommandEntry::Patch(patch);
                                    }
                                    CommandEntry::NativeUpdate(_) => {}
                                    _ => unreachable!("checked patch entry"),
                                }
                                buffer.component_op_count = new_ops;
                            }
                            if buffer.deferred_error.is_some() {
                                break;
                            }
                        }
                    }
                }

                // RecordingGuard restores the accumulated rows to their stable
                // lane indices; future calls append to the same typed vectors.
            }
        }
    };
}

impl_batch_update_item!(0: T0 v0 r0);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4, 5: T5 v5 r5);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4, 5: T5 v5 r5, 6: T6 v6 r6);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4, 5: T5 v5 r5, 6: T6 v6 r6, 7: T7 v7 r7);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4, 5: T5 v5 r5, 6: T6 v6 r6, 7: T7 v7 r7, 8: T8 v8 r8);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4, 5: T5 v5 r5, 6: T6 v6 r6, 7: T7 v7 r7, 8: T8 v8 r8, 9: T9 v9 r9);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4, 5: T5 v5 r5, 6: T6 v6 r6, 7: T7 v7 r7, 8: T8 v8 r8, 9: T9 v9 r9, 10: T10 v10 r10);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4, 5: T5 v5 r5, 6: T6 v6 r6, 7: T7 v7 r7, 8: T8 v8 r8, 9: T9 v9 r9, 10: T10 v10 r10, 11: T11 v11 r11);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4, 5: T5 v5 r5, 6: T6 v6 r6, 7: T7 v7 r7, 8: T8 v8 r8, 9: T9 v9 r9, 10: T10 v10 r10, 11: T11 v11 r11, 12: T12 v12 r12);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4, 5: T5 v5 r5, 6: T6 v6 r6, 7: T7 v7 r7, 8: T8 v8 r8, 9: T9 v9 r9, 10: T10 v10 r10, 11: T11 v11 r11, 12: T12 v12 r12, 13: T13 v13 r13);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4, 5: T5 v5 r5, 6: T6 v6 r6, 7: T7 v7 r7, 8: T8 v8 r8, 9: T9 v9 r9, 10: T10 v10 r10, 11: T11 v11 r11, 12: T12 v12 r12, 13: T13 v13 r13, 14: T14 v14 r14);
impl_batch_update_item!(0: T0 v0 r0, 1: T1 v1 r1, 2: T2 v2 r2, 3: T3 v3 r3, 4: T4 v4 r4, 5: T5 v5 r5, 6: T6 v6 r6, 7: T7 v7 r7, 8: T8 v8 r8, 9: T9 v9 r9, 10: T10 v10 r10, 11: T11 v11 r11, 12: T12 v12 r12, 13: T13 v13 r13, 14: T14 v14 r14, 15: T15 v15 r15);

impl CommandBuffer {
    /// 统一批量更新接口：支持单组件 `(EntityId, T0)` 与多组件元组 `(EntityId, T0, T1, ...)`。
    ///
    /// 在录入端对同一实体仅执行一次哈希查找与冲突校验，消除多次录入的哈希膨胀；
    /// 内部串联各组件的 NativeLane 引用，并在准备与提交阶段保持完整的事务边界与原子回滚。
    pub fn update_batch<Item: BatchUpdateItem>(
        &mut self,
        updates: impl IntoIterator<Item = Item>,
    ) -> &mut Self {
        if self.deferred_error.is_some() {
            return self;
        }
        Item::record_batch(self, updates.into_iter());
        self
    }

    pub(super) fn lower_native_updates(&mut self) {
        // Compatibility adapter for explicit, crate-private into_batch only.
        // prepare/apply never call it.
        for (_, entry) in &mut self.entries {
            let Some(mut head) = entry.native_head() else {
                continue;
            };
            let old = std::mem::replace(entry, CommandEntry::Delete);
            let mut patch = match old {
                CommandEntry::Patch(mut patch) => {
                    patch.native_head = None;
                    patch.detach_pool(&mut self.pool);
                    patch
                }
                CommandEntry::NativeUpdate(_) => EntityPatch::new(),
                _ => unreachable!(),
            };
            loop {
                let lane = &mut self.native_values[head.lane()];
                let (value, next) = lane.take(head.row());
                patch
                    .replace_erased(value)
                    .expect("recorded disjoint operations");
                match next {
                    Some(next) => head = next,
                    None => break,
                }
            }
            *entry = CommandEntry::Patch(patch);
        }
        drop(std::mem::take(&mut self.native_values));
    }
}

#[cfg(test)]
#[path = "updates_tests.rs"]
mod tests;
