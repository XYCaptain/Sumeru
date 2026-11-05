use crate::error::StateError;
use crate::id::{ComponentTypeId, EntityId};
use crate::storage::ErasedComponentValue;
use smallvec::SmallVec;
use std::ops::Range;
use std::sync::Arc;

use super::any::{BorrowedArchiveRows, ErasedBlock, sealed};

impl ErasedBlock for Vec<ErasedComponentValue> {
    fn borrow_archive_rows<'a>(&'a self, rows: &mut BorrowedArchiveRows<'a>, range: Range<usize>) -> Result<(), StateError> {
        let values = self.get(range).ok_or_else(|| StateError::BatchConflict("Invalid erased archive range".into()))?;
        for value in values { rows.push_erased(value)?; }
        Ok(())
    }
    fn internal_controlled_token(&self) -> Option<sealed::InternalControlledToken> {
        Some(sealed::InternalControlledToken::new())
    }
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn into_any_box(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }

    fn into_any_arc(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync> {
        self
    }

    fn estimated_bytes(&self) -> usize {
        self.capacity() * std::mem::size_of::<ErasedComponentValue>()
            + self.iter().map(|v| v.size()).sum::<usize>()
    }

    fn into_values(
        mut self: Box<Self>,
        _component: ComponentTypeId,
        range: Range<usize>,
    ) -> Result<Vec<ErasedComponentValue>, StateError> {
        if range.start > range.end || range.end > self.len() {
            return Err(StateError::BatchConflict("Invalid owned block range".into()));
        }
        Ok(self.drain(range).collect())
    }

    fn clone_value(
        &self,
        row: usize,
        _component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        self.get(row).cloned().ok_or_else(|| {
            StateError::BatchConflict("Row out of bounds in ErasedBlock".into())
        })
    }

    fn take_value(
        &mut self,
        row: usize,
        _component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        if row < self.len() {
            Ok(self.swap_remove(row))
        } else {
            Err(StateError::BatchConflict("Row out of bounds in ErasedBlock".into()))
        }
    }

    fn type_info(&self, row: usize, _component: ComponentTypeId) -> Option<(std::any::TypeId, &'static str)> {
        self.get(row).map(|v| (v.rust_type_id(), v.type_name()))
    }
}

/// 独占类型擦除物理块 (ErasedOwnedBlock)
///
/// 用于在 prepare 阶段预先分配容量，并在 commit 阶段零分配追加旧值。
#[derive(Debug, Default)]
pub struct ErasedOwnedBlock {
    pub values: SmallVec<[ErasedComponentValue; 1]>,
}

impl ErasedOwnedBlock {
    #[inline]
    pub fn new() -> Self {
        Self { values: SmallVec::new() }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: SmallVec::with_capacity(capacity),
        }
    }
}

impl ErasedBlock for ErasedOwnedBlock {
    fn borrow_archive_rows<'a>(&'a self, rows: &mut BorrowedArchiveRows<'a>, range: Range<usize>) -> Result<(), StateError> {
        let values = self.values.get(range).ok_or_else(|| StateError::BatchConflict("Invalid erased archive range".into()))?;
        for value in values { rows.push_erased(value)?; }
        Ok(())
    }
    fn internal_controlled_token(&self) -> Option<sealed::InternalControlledToken> {
        Some(sealed::InternalControlledToken::new())
    }
    fn len(&self) -> usize {
        self.values.len()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn into_any_box(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }

    fn into_any_arc(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync> {
        self
    }

    fn estimated_bytes(&self) -> usize {
        if self.values.spilled() {
            self.values.capacity() * std::mem::size_of::<ErasedComponentValue>()
                + self.values.iter().map(|v| v.size()).sum::<usize>()
        } else {
            self.values.iter().map(|v| v.size()).sum::<usize>()
        }
    }

    fn into_values(
        mut self: Box<Self>,
        _component: ComponentTypeId,
        range: Range<usize>,
    ) -> Result<Vec<ErasedComponentValue>, StateError> {
        if range.start > range.end || range.end > self.len() {
            return Err(StateError::BatchConflict("Invalid owned block range".into()));
        }
        if range.start == 0 && range.end == self.values.len() {
            Ok(self.values.into_vec())
        } else {
            Ok(self.values.drain(range).collect())
        }
    }

    fn clone_value(
        &self,
        row: usize,
        _component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        self.values.get(row).cloned().ok_or_else(|| {
            StateError::BatchConflict("Row out of bounds in ErasedBlock".into())
        })
    }

    fn take_value(
        &mut self,
        row: usize,
        _component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        if row < self.len() {
            Ok(self.values.swap_remove(row))
        } else {
            Err(StateError::BatchConflict("Row out of bounds in ErasedBlock".into()))
        }
    }

    fn type_info(&self, row: usize, _component: ComponentTypeId) -> Option<(std::any::TypeId, &'static str)> {
        self.values.get(row).map(|v| (v.rust_type_id(), v.type_name()))
    }

    fn clone_erased_arc(&self) -> Result<Arc<dyn ErasedBlock>, StateError> {
        Ok(Arc::new(Self {
            values: self.values.clone(),
        }))
    }
}

struct PooledVecAllocation {
    ptr: usize,
    cap: usize,
    drop_fn: unsafe fn(usize, usize),
}

impl Drop for PooledVecAllocation {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                (self.drop_fn)(self.ptr, self.cap);
            }
        }
    }
}

static TYPED_VEC_POOL: std::sync::OnceLock<
    std::sync::Mutex<Vec<(std::any::TypeId, Vec<PooledVecAllocation>)>>,
> = std::sync::OnceLock::new();

unsafe fn drop_pooled_vec<T>(ptr: usize, cap: usize) {
    unsafe {
        drop(Vec::<T>::from_raw_parts(ptr as *mut T, 0, cap));
    }
}

/// Acquire a typed `Vec<T>` from the thread-safe buffer recycling pool when
/// `capacity` is within `[1024, 262_144]` and `<= 4 MiB`, avoiding OS page faults
/// and large-zone allocator contention on repeated batch preparations.
#[inline]
pub fn take_pooled_typed_vec<
    T: crate::context::ModelComponent + serde::Serialize + serde::de::DeserializeOwned,
>(
    capacity: usize,
) -> Vec<T> {
    let elem_size = std::mem::size_of::<T>();
    if elem_size > 0
        && (1024..=262_144).contains(&capacity)
        && capacity.saturating_mul(elem_size) <= (4 << 20)
    {
        let pool = TYPED_VEC_POOL.get_or_init(|| std::sync::Mutex::new(Vec::with_capacity(8)));
        if let Ok(mut guard) = pool.lock() {
            let tid = std::any::TypeId::of::<T>();
            if let Some((_, bucket)) = guard.iter_mut().find(|(id, _)| *id == tid) {
                if let Some(idx) = bucket
                    .iter()
                    .rposition(|entry| entry.cap == capacity)
                    .or_else(|| {
                        bucket.iter().rposition(|entry| {
                            entry.cap >= capacity && entry.cap <= capacity.saturating_mul(2)
                        })
                    })
                {
                    let mut entry = bucket.swap_remove(idx);
                    let ptr = entry.ptr;
                    let cap = entry.cap;
                    entry.ptr = 0;
                    return unsafe { Vec::from_raw_parts(ptr as *mut T, 0, cap) };
                }
            }
        }
    }
    Vec::with_capacity(capacity)
}

/// Return a typed `Vec<T>` to the thread-safe buffer recycling pool if its capacity
/// is within `[1024, 262_144]` and `<= 4 MiB` and its type bucket has `< 16` buffers.
#[inline]
pub fn return_pooled_typed_vec<
    T: crate::context::ModelComponent + serde::Serialize + serde::de::DeserializeOwned,
>(
    mut vec: Vec<T>,
) {
    let elem_size = std::mem::size_of::<T>();
    let cap = vec.capacity();
    if elem_size > 0
        && (1024..=262_144).contains(&cap)
        && cap.saturating_mul(elem_size) <= (4 << 20)
    {
        vec.clear();
        let pool = TYPED_VEC_POOL.get_or_init(|| std::sync::Mutex::new(Vec::with_capacity(8)));
        if let Ok(mut guard) = pool.lock() {
            let tid = std::any::TypeId::of::<T>();
            let bucket = if let Some((_, b)) = guard.iter_mut().find(|(id, _)| *id == tid) {
                b
            } else {
                guard.push((tid, Vec::with_capacity(16)));
                &mut guard.last_mut().unwrap().1
            };
            if bucket.len() < 16 {
                let mut v = std::mem::ManuallyDrop::new(vec);
                let ptr = v.as_mut_ptr() as usize;
                bucket.push(PooledVecAllocation {
                    ptr,
                    cap,
                    drop_fn: drop_pooled_vec::<T>,
                });
                return;
            }
        }
    }
}

/// 独占强类型连续物理块 (TypedOwnedBlock<T>)
///
/// 在 prepare 阶段直接以连续 `Vec<T>` 预分配并在 commit 阶段单次虚表分发完成整列原位交换。
#[derive(Debug)]
pub struct TypedOwnedBlock<
    T: crate::context::ModelComponent + serde::Serialize + serde::de::DeserializeOwned,
> {
    pub component: ComponentTypeId,
    pub values: Vec<T>,
}

impl<T: crate::context::ModelComponent + serde::Serialize + serde::de::DeserializeOwned> Drop
    for TypedOwnedBlock<T>
{
    fn drop(&mut self) {
        let values = std::mem::take(&mut self.values);
        if values.capacity() > 0 {
            return_pooled_typed_vec(values);
        }
    }
}

impl<T: crate::context::ModelComponent + serde::Serialize + serde::de::DeserializeOwned>
    TypedOwnedBlock<T>
{
    #[inline]
    pub fn with_capacity(component: ComponentTypeId, capacity: usize) -> Self {
        let values = take_pooled_typed_vec::<T>(capacity);
        Self { component, values }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }
}

impl<T: crate::context::ModelComponent + serde::Serialize + serde::de::DeserializeOwned> ErasedBlock
    for TypedOwnedBlock<T>
{
    fn borrow_archive_rows<'a>(
        &'a self,
        rows: &mut BorrowedArchiveRows<'a>,
        range: Range<usize>,
    ) -> Result<(), StateError> {
        let slice = self
            .values
            .get(range)
            .ok_or_else(|| StateError::BatchConflict("Invalid typed owned archive range".into()))?;
        rows.push_slice(slice)
    }

    fn internal_controlled_token(&self) -> Option<sealed::InternalControlledToken> {
        Some(sealed::InternalControlledToken::new())
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn into_any_box(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }

    fn into_any_arc(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync> {
        self
    }

    fn estimated_bytes(&self) -> usize {
        self.values.capacity() * std::mem::size_of::<T>()
    }

    fn into_values(
        mut self: Box<Self>,
        component: ComponentTypeId,
        range: Range<usize>,
    ) -> Result<Vec<ErasedComponentValue>, StateError> {
        if range.start > range.end || range.end > self.values.len() {
            return Err(StateError::BatchConflict(
                "Invalid owned block range".into(),
            ));
        }
        Ok(self
            .values
            .drain(range)
            .map(|val| ErasedComponentValue::with_component_type_id(component, val))
            .collect())
    }

    fn clone_value(
        &self,
        row: usize,
        component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        self.values
            .get(row)
            .cloned()
            .map(|val| ErasedComponentValue::with_component_type_id(component, val))
            .ok_or_else(|| {
                StateError::BatchConflict("Row out of bounds in TypedOwnedBlock".into())
            })
    }

    fn take_value(
        &mut self,
        row: usize,
        component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        if row < self.values.len() {
            Ok(ErasedComponentValue::with_component_type_id(
                component,
                self.values.swap_remove(row),
            ))
        } else {
            Err(StateError::BatchConflict(
                "Row out of bounds in TypedOwnedBlock".into(),
            ))
        }
    }

    fn type_info(
        &self,
        row: usize,
        component: ComponentTypeId,
    ) -> Option<(std::any::TypeId, &'static str)> {
        if row < self.values.len() && component == self.component {
            Some((std::any::TypeId::of::<T>(), std::any::type_name::<T>()))
        } else {
            None
        }
    }

    fn is_native_column_block(&self) -> bool {
        true
    }

    fn clone_native_range(&self, range: Range<usize>) -> Option<Box<dyn ErasedBlock>> {
        let slice = self.values.get(range)?;
        let mut values = take_pooled_typed_vec::<T>(slice.len());
        values.extend_from_slice(slice);
        Some(Box::new(Self {
            component: self.component,
            values,
        }))
    }

    fn value_any_ref(&self, row: usize) -> Option<&(dyn std::any::Any + Send + Sync)> {
        self.values
            .get(row)
            .map(|v| v as &(dyn std::any::Any + Send + Sync))
    }

    fn row_size_bytes(&self, _row: usize) -> usize {
        std::mem::size_of::<T>()
    }

    fn clone_erased_arc(&self) -> Result<Arc<dyn ErasedBlock>, StateError> {
        let mut values = take_pooled_typed_vec::<T>(self.values.len());
        values.extend_from_slice(&self.values);
        Ok(Arc::new(Self {
            component: self.component,
            values,
        }))
    }

    fn push_from_erased(&mut self, value: &mut ErasedComponentValue) -> Result<(), StateError> {
        let typed = value.take_typed::<T>()?;
        self.values.push(typed);
        Ok(())
    }

    fn extend_from_erased_pairs(
        &mut self,
        values: &mut [(EntityId, ErasedComponentValue)],
    ) -> Result<(), StateError> {
        self.values.reserve(values.len());
        for (_, value) in values.iter_mut() {
            let typed = value.take_typed::<T>()?;
            self.values.push(typed);
        }
        Ok(())
    }

    fn extend_from_erased_indexed(
        &mut self,
        values: &mut [(EntityId, ErasedComponentValue)],
        indices: &[usize],
    ) -> Result<(), StateError> {
        self.values.reserve(indices.len());
        for &idx in indices {
            let typed = values[idx].1.take_typed::<T>()?;
            self.values.push(typed);
        }
        Ok(())
    }

    fn drain_into_erased(
        &mut self,
        range: Range<usize>,
        component: ComponentTypeId,
    ) -> Result<Vec<ErasedComponentValue>, StateError> {
        if range.start > range.end || range.end > self.values.len() {
            return Err(StateError::BatchConflict(
                "Invalid owned typed block range".into(),
            ));
        }
        Ok(self
            .values
            .drain(range)
            .map(|val| ErasedComponentValue::with_component_type_id(component, val))
            .collect())
    }

    fn reserve_exact_total(&mut self, total_capacity: usize) {
        if self.values.capacity() < total_capacity {
            let mut new_vec = take_pooled_typed_vec::<T>(total_capacity);
            new_vec.append(&mut self.values);
            let old = std::mem::replace(&mut self.values, new_vec);
            if old.capacity() > 0 {
                return_pooled_typed_vec(old);
            }
        }
    }

    fn append_from_other(&mut self, other: &mut dyn ErasedBlock) -> Result<usize, StateError> {
        let other_typed = other
            .as_any_mut()
            .downcast_mut::<Self>()
            .ok_or_else(|| StateError::BatchConflict("Type mismatch in append_from_other".into()))?;
        self.values.append(&mut other_typed.values);
        Ok(self.values.len())
    }

    #[allow(private_interfaces)]
    unsafe fn swap_prepared_column_rows(
        &mut self,
        descriptor: &crate::storage::core::column::any::ColumnDescriptor,
        rows: &[crate::storage::core::column::any::PreparedRow],
    ) {
        let base = descriptor.raw_ptr as *mut T;
        for (row, slot_val) in rows.iter().zip(self.values.iter_mut()) {
            debug_assert!(row.row < descriptor.len, "row index out of bounds");
            debug_assert!(
                descriptor.writable,
                "read-only backing cannot be written"
            );
            unsafe {
                std::mem::swap(&mut *base.add(row.row), slot_val);
            }
        }
    }

    #[allow(private_interfaces)]
    unsafe fn swap_strided_column_rows(
        &mut self,
        descriptor: &crate::storage::core::column::any::ColumnDescriptor,
        first_row: usize,
        count: usize,
    ) {
        debug_assert_eq!(self.values.len(), count);
        if count == 0 {
            return;
        }
        unsafe {
            let col_slice = std::slice::from_raw_parts_mut(
                (descriptor.raw_ptr as *mut T).add(first_row),
                self.values.len(),
            );
            col_slice.swap_with_slice(&mut self.values);
        }
    }

    #[allow(private_interfaces)]
    unsafe fn write_strided_column_rows(
        &self,
        descriptor: &crate::storage::core::column::any::ColumnDescriptor,
        first_row: usize,
        count: usize,
    ) {
        debug_assert_eq!(self.values.len(), count);
        if count == 0 {
            return;
        }
        unsafe {
            let col_slice = std::slice::from_raw_parts_mut(
                (descriptor.raw_ptr as *mut T).add(first_row),
                self.values.len(),
            );
            col_slice.clone_from_slice(&self.values);
        }
    }
}

/// 旧值物理载体
pub enum FragmentPayload {
    /// 共享物理块引用 (Arc 零拷贝保活物理块，持有块内有效范围)
    SharedBlock {
        block: Arc<dyn ErasedBlock>,
        block_range: Range<usize>,
    },
    /// 独占旧值块 (用于 replace_sparse / replace_range 移出的旧值)
    OwnedBlock {
        block: Box<dyn ErasedBlock>,
        block_range: Range<usize>,
    },
    /// 内联类型擦除旧值块 (受控共享所有权：独占时通过 Arc::get_mut / Arc::try_unwrap 零拷贝，后台并发归档时共享只读引用)
    OwnedErased {
        block: Arc<ErasedOwnedBlock>,
        block_range: Range<usize>,
    },
    /// 强类型连续旧值块 (受控共享所有权：独占时零拷贝原地置换/直交，后台并发归档时共享 Arc<TypedOwnedBlock<T>> 连续切片)
    OwnedTyped {
        block: Arc<dyn ErasedBlock>,
        block_range: Range<usize>,
    },
    /// 已被消费或移交的空载荷
    Consumed,
}

impl FragmentPayload {
    pub fn estimated_bytes(&self) -> usize {
        match self {
            Self::OwnedBlock { block, .. } => block.estimated_bytes(),
            Self::SharedBlock { block, .. } => block.estimated_bytes(),
            Self::OwnedErased { block, .. } => block.estimated_bytes(),
            Self::OwnedTyped { block, .. } => block.estimated_bytes(),
            Self::Consumed => 0,
        }
    }

    pub(crate) fn take_owned_block(&mut self) -> Option<(Box<dyn ErasedBlock>, Range<usize>)> {
        match std::mem::replace(self, Self::Consumed) {
            Self::OwnedBlock { block, block_range } => Some((block, block_range)),
            other => {
                *self = other;
                None
            }
        }
    }

    pub(crate) fn take_shared_block(&mut self) -> Option<(Arc<dyn ErasedBlock>, Range<usize>)> {
        match std::mem::replace(self, Self::Consumed) {
            Self::SharedBlock { block, block_range } => Some((block, block_range)),
            other => {
                *self = other;
                None
            }
        }
    }

    pub(crate) fn clone_shared_block(&self) -> Option<(Arc<dyn ErasedBlock>, Range<usize>)> {
        match self {
            Self::SharedBlock { block, block_range } => Some((Arc::clone(block), block_range.clone())),
            _ => None,
        }
    }
}
