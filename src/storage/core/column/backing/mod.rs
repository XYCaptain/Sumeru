//! 列物理后备存储 (ColumnBacking)
//!
//! 统一封装强类型连续列的底层内存形态：
//! - `Direct(DirectBlock<T>)`：单块连续内存（`Owned(Vec<T>)` 或 `Frozen(Arc<ColumnBlock<T>>)`），支持零拷贝切片与 COW 写时复制。

use std::any::Any;
use std::ops::{Deref, Range};
use std::sync::Arc;

use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use crate::storage::column::any::{BorrowedArchiveRows, ErasedBlock, ErasedComponentValue, sealed};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// 连续物理列内存块，支持 `Arc<ColumnBlock<T>>` 零拷贝共享与唯一持有者原地复用。
#[derive(Debug)]
pub struct ColumnBlock<T> {
    pub(crate) data: Vec<T>,
}

impl<T> ColumnBlock<T> {
    #[inline]
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    #[inline]
    pub(crate) fn get_mut_if_unique(this: &mut Arc<Self>) -> Option<&mut Vec<T>> {
        Arc::get_mut(this).map(|block| &mut block.data)
    }
}

impl<T: Clone> Clone for ColumnBlock<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<T: PartialEq> PartialEq for ColumnBlock<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Eq> Eq for ColumnBlock<T> {}

impl<T> Deref for ColumnBlock<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> AsRef<[T]> for ColumnBlock<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned> ErasedBlock for ColumnBlock<T> {
    fn borrow_archive_rows<'a>(
        &'a self,
        rows: &mut BorrowedArchiveRows<'a>,
        range: Range<usize>,
    ) -> Result<(), StateError> {
        let values = self
            .data
            .get(range)
            .ok_or_else(|| StateError::BatchConflict("Invalid native archive range".into()))?;
        rows.push_slice(values)
    }

    fn clone_native_range(&self, range: Range<usize>) -> Option<Box<dyn ErasedBlock>> {
        let source = self.data.get(range)?;
        Some(Box::new(Self {
            data: source.to_vec(),
        }))
    }

    fn internal_controlled_token(&self) -> Option<sealed::InternalControlledToken> {
        Some(sealed::InternalControlledToken::new())
    }

    fn is_native_column_block(&self) -> bool {
        true
    }

    fn into_values(
        mut self: Box<Self>,
        component: ComponentTypeId,
        range: Range<usize>,
    ) -> Result<Vec<ErasedComponentValue>, StateError> {
        if range.start > range.end || range.end > self.data.len() {
            return Err(StateError::BatchConflict(
                "Invalid owned block range".into(),
            ));
        }
        Ok(self
            .data
            .drain(range)
            .map(|value| ErasedComponentValue::with_component_type_id(component, value))
            .collect())
    }

    fn clone_value(
        &self,
        row: usize,
        component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        self.data.clone_value(row, component)
    }

    fn take_value(
        &mut self,
        row: usize,
        component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        if row < self.data.len() {
            let val = self.data.swap_remove(row);
            Ok(ErasedComponentValue::with_component_type_id(component, val))
        } else {
            Err(StateError::BatchConflict(
                "Row out of bounds in ErasedBlock".into(),
            ))
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_any_box(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn into_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn estimated_bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<T>()
    }

    fn type_info(
        &self,
        _row: usize,
        _component: ComponentTypeId,
    ) -> Option<(std::any::TypeId, &'static str)> {
        Some((std::any::TypeId::of::<T>(), std::any::type_name::<T>()))
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned> ErasedBlock for Vec<T> {
    fn borrow_archive_rows<'a>(
        &'a self,
        rows: &mut BorrowedArchiveRows<'a>,
        range: Range<usize>,
    ) -> Result<(), StateError> {
        let values = self
            .get(range)
            .ok_or_else(|| StateError::BatchConflict("Invalid native archive range".into()))?;
        rows.push_slice(values)
    }

    fn internal_controlled_token(&self) -> Option<sealed::InternalControlledToken> {
        Some(sealed::InternalControlledToken::new())
    }

    fn is_native_column_block(&self) -> bool {
        true
    }

    fn into_values(
        self: Box<Self>,
        component: ComponentTypeId,
        range: Range<usize>,
    ) -> Result<Vec<ErasedComponentValue>, StateError> {
        if range.start > range.end || range.end > self.len() {
            return Err(StateError::BatchConflict(
                "Invalid owned block range".into(),
            ));
        }
        Ok((*self)
            .into_iter()
            .skip(range.start)
            .take(range.len())
            .map(|value| ErasedComponentValue::with_component_type_id(component, value))
            .collect())
    }

    fn clone_value(
        &self,
        row: usize,
        component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        let value = self
            .get(row)
            .ok_or_else(|| StateError::AccessDenied("Inverse block row out of bounds".into()))?;
        Ok(ErasedComponentValue::with_component_type_id(
            component,
            value.clone(),
        ))
    }

    fn take_value(
        &mut self,
        row: usize,
        component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        if row < self.len() {
            let val = self.swap_remove(row);
            Ok(ErasedComponentValue::with_component_type_id(component, val))
        } else {
            Err(StateError::BatchConflict(
                "Row out of bounds in ErasedBlock".into(),
            ))
        }
    }

    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_any_box(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn into_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn estimated_bytes(&self) -> usize {
        self.capacity() * std::mem::size_of::<T>()
    }

    fn type_info(
        &self,
        _row: usize,
        _component: ComponentTypeId,
    ) -> Option<(std::any::TypeId, &'static str)> {
        Some((std::any::TypeId::of::<T>(), std::any::type_name::<T>()))
    }
}

/// 单块连续内存形态：独占可变向量或引用计数共享块。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectBlock<T> {
    Owned(Vec<T>),
    Frozen(Arc<ColumnBlock<T>>),
}

impl<T> DirectBlock<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match self {
            DirectBlock::Owned(vec) => vec.as_slice(),
            DirectBlock::Frozen(block) => block.as_slice(),
        }
    }

    #[inline]
    pub fn make_owned(&mut self) -> &mut Vec<T>
    where
        T: Clone,
    {
        if matches!(self, DirectBlock::Frozen(_)) {
            let prev = std::mem::replace(self, DirectBlock::Owned(Vec::new()));
            if let DirectBlock::Frozen(arc) = prev {
                let vec = Arc::try_unwrap(arc)
                    .map(|b| b.into_vec())
                    .unwrap_or_else(|arc| arc.as_slice().to_vec());
                *self = DirectBlock::Owned(vec);
            }
        }
        match self {
            DirectBlock::Owned(vec) => vec,
            DirectBlock::Frozen(_) => unreachable!(),
        }
    }

    #[inline]
    pub fn freeze(&mut self) -> Arc<ColumnBlock<T>> {
        match self {
            DirectBlock::Owned(vec) => {
                let taken = std::mem::take(vec);
                let block = Arc::new(ColumnBlock::new(taken));
                *self = DirectBlock::Frozen(Arc::clone(&block));
                block
            }
            DirectBlock::Frozen(block) => Arc::clone(block),
        }
    }

    #[inline]
    pub(crate) fn is_uniquely_writable(&self) -> bool {
        match self {
            DirectBlock::Owned(_) => true,
            DirectBlock::Frozen(block) => {
                Arc::strong_count(block) == 1 && Arc::weak_count(block) == 0
            }
        }
    }

    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T]
    where
        T: Clone,
    {
        if matches!(
            self,
            DirectBlock::Frozen(block) if Arc::strong_count(block) != 1 || Arc::weak_count(block) != 0
        ) {
            return self.make_owned().as_mut_slice();
        }
        match self {
            DirectBlock::Owned(vec) => vec.as_mut_slice(),
            DirectBlock::Frozen(block) => ColumnBlock::get_mut_if_unique(block)
                .expect("unique Arc")
                .as_mut_slice(),
        }
    }
}

/// 强类型列物理后备存储（纯连续内存版）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnBacking<T> {
    Direct(DirectBlock<T>),
}

impl<T> Default for ColumnBacking<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ColumnBacking<T> {
    #[inline]
    pub fn new() -> Self {
        ColumnBacking::Direct(DirectBlock::Owned(Vec::new()))
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        ColumnBacking::Direct(DirectBlock::Owned(Vec::with_capacity(capacity)))
    }

    #[inline]
    pub fn from_vec(vec: Vec<T>) -> Self {
        ColumnBacking::Direct(DirectBlock::Owned(vec))
    }

    #[inline]
    pub fn from_block(block: Arc<ColumnBlock<T>>) -> Self {
        ColumnBacking::Direct(DirectBlock::Frozen(block))
    }

    #[inline]
    pub fn len(&self) -> usize {
        let ColumnBacking::Direct(block) = self;
        block.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn is_segmented(&self) -> bool {
        false
    }

    #[inline]
    pub fn is_frozen(&self) -> bool {
        matches!(self, ColumnBacking::Direct(DirectBlock::Frozen(_)))
    }

    #[inline]
    pub fn is_owned(&self) -> bool {
        matches!(self, ColumnBacking::Direct(DirectBlock::Owned(_)))
    }

    #[inline]
    pub fn segment_count(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            1
        }
    }

    #[inline]
    pub fn try_as_contiguous_slice(&self) -> Option<&[T]> {
        Some(self.as_contiguous_slice())
    }

    #[inline]
    pub fn as_contiguous_slice(&self) -> &[T] {
        let ColumnBacking::Direct(block) = self;
        block.as_slice()
    }

    #[inline]
    pub fn as_direct_slice(&self) -> Option<&[T]> {
        Some(self.as_contiguous_slice())
    }

    #[inline]
    pub fn as_vec(&self) -> Option<&Vec<T>> {
        match self {
            ColumnBacking::Direct(DirectBlock::Owned(vec)) => Some(vec),
            ColumnBacking::Direct(DirectBlock::Frozen(_)) => None,
        }
    }

    #[inline]
    pub fn make_contiguous_owned(&mut self) -> &mut Vec<T>
    where
        T: Clone,
    {
        let ColumnBacking::Direct(block) = self;
        block.make_owned()
    }

    #[inline]
    pub fn as_vec_mut(&mut self) -> &mut Vec<T>
    where
        T: Clone,
    {
        self.make_contiguous_owned()
    }

    #[inline]
    pub fn ensure_contiguous(&mut self) -> &mut Vec<T>
    where
        T: Clone,
    {
        self.make_contiguous_owned()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T]
    where
        T: Clone,
    {
        let ColumnBacking::Direct(block) = self;
        block.as_mut_slice()
    }

    #[inline]
    pub fn get_clone(&self, row: usize) -> Option<T>
    where
        T: Clone,
    {
        self.as_contiguous_slice().get(row).cloned()
    }

    #[inline]
    pub fn get_ref(&self, row: usize) -> Option<&T> {
        self.as_contiguous_slice().get(row)
    }

    #[inline]
    pub fn get(&self, row: usize) -> Option<&T> {
        self.get_ref(row)
    }

    #[inline]
    pub fn get_mut(&mut self, row: usize) -> Option<&mut T>
    where
        T: Clone,
    {
        self.as_mut_slice().get_mut(row)
    }

    #[inline]
    pub fn borrow_row_mut(&mut self, row: usize) -> Option<&mut T>
    where
        T: Clone,
    {
        self.as_mut_slice().get_mut(row)
    }

    #[inline]
    pub fn replace_row_infallible(&mut self, row: usize, val: T) -> T
    where
        T: Clone,
    {
        let slice = self.as_mut_slice();
        std::mem::replace(&mut slice[row], val)
    }

    #[inline]
    pub fn freeze_contiguous(&mut self) -> Arc<ColumnBlock<T>> {
        let ColumnBacking::Direct(block) = self;
        block.freeze()
    }

    #[inline]
    pub fn freeze(&mut self) -> Arc<ColumnBlock<T>> {
        self.freeze_contiguous()
    }

    #[inline]
    pub fn slice_from(&self, start: usize) -> Option<&[T]> {
        let slice = self.as_contiguous_slice();
        if start <= slice.len() {
            Some(&slice[start..])
        } else {
            None
        }
    }

    #[inline]
    pub fn slice_from_mut(&mut self, start: usize) -> Option<&mut [T]>
    where
        T: Clone,
    {
        let slice = self.as_mut_slice();
        if start <= slice.len() {
            Some(&mut slice[start..])
        } else {
            None
        }
    }

    #[inline]
    pub fn push(&mut self, val: T) -> usize
    where
        T: Clone,
    {
        let vec = self.make_contiguous_owned();
        let row = vec.len();
        vec.push(val);
        row
    }

    #[inline]
    pub fn swap_remove(&mut self, row: usize) -> Option<T>
    where
        T: Clone,
    {
        if row < self.len() {
            Some(self.make_contiguous_owned().swap_remove(row))
        } else {
            None
        }
    }

    #[inline]
    pub fn swap_remove_and_drop(&mut self, row: usize)
    where
        T: Clone,
    {
        let _ = self.swap_remove(row);
    }

    #[inline]
    pub fn swap_remove_batch_sorted_and_drop(&mut self, sorted_desc_rows: &[usize])
    where
        T: Clone,
    {
        if sorted_desc_rows.is_empty() {
            return;
        }
        let vec = self.make_contiguous_owned();
        for &row in sorted_desc_rows {
            let _ = vec.swap_remove(row);
        }
    }

    #[inline]
    pub fn truncate(&mut self, new_len: usize)
    where
        T: Clone,
    {
        if new_len >= self.len() {
            return;
        }
        self.make_contiguous_owned().truncate(new_len);
    }

    #[inline]
    pub fn clear(&mut self) {
        let ColumnBacking::Direct(block) = self;
        match block {
            DirectBlock::Owned(vec) => vec.clear(),
            DirectBlock::Frozen(arc) => {
                if let Some(vec) = ColumnBlock::get_mut_if_unique(arc) {
                    vec.clear();
                    *block = DirectBlock::Owned(std::mem::take(vec));
                } else {
                    *block = DirectBlock::Owned(Vec::new());
                }
            }
        }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize)
    where
        T: Clone,
    {
        self.make_contiguous_owned().reserve(additional);
    }

    #[inline]
    pub fn empty_with_origin(&self) -> Self {
        Self::new()
    }

    #[inline]
    pub fn physical_version(&self) -> u64 {
        0
    }

    #[inline]
    pub fn is_direct_owned(&self) -> bool {
        self.is_owned()
    }

    #[inline]
    pub(crate) fn access_parts(&mut self) -> (*mut u8, usize, bool)
    where
        T: Clone,
    {
        let slice = self.as_mut_slice();
        (slice.as_mut_ptr() as *mut u8, slice.len(), true)
    }

    #[inline(always)]
    pub(crate) unsafe fn push_direct_owned_unchecked(&mut self, value: T)
    where
        T: Clone,
    {
        self.make_contiguous_owned().push(value);
    }

    #[inline(always)]
    pub(crate) unsafe fn swap_remove_direct_owned_unchecked(&mut self, row: usize) -> T
    where
        T: Clone,
    {
        self.make_contiguous_owned().swap_remove(row)
    }

    #[inline]
    pub(crate) unsafe fn write_exclusive_row_unchecked(&mut self, row: usize, val: T) -> T
    where
        T: Clone,
    {
        self.replace_row_infallible(row, val)
    }

    #[inline]
    pub fn freeze_erased(&mut self) -> Arc<dyn ErasedBlock>
    where
        T: ModelComponent + Serialize + DeserializeOwned,
    {
        self.freeze_contiguous()
    }

    #[inline]
    pub(crate) fn is_uniquely_writable(&self) -> bool {
        let ColumnBacking::Direct(block) = self;
        block.is_uniquely_writable()
    }

    #[inline]
    pub(crate) fn prepare_in_place_write(&mut self) -> Result<(), StateError>
    where
        T: Clone,
    {
        let _ = self.as_mut_slice();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
