//! 列级操作虚表与内存描述符实现 (ColumnDescriptor & ErasedComponentValue methods)

use super::any::{
    AnyColumn, BorrowedArchiveRows, ColumnDescriptor, ErasedBlock, ErasedComponentValue,
    HasInlineVTable, INLINE_ALIGN, INLINE_SIZE, InlineBuffer, ValueRepr,
};
use super::archive::{ArchivedColumnData, RawArchiveSlice, RawColumnArchive};
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use crate::schema::ComponentSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;
use smallvec::SmallVec;
use std::any::TypeId;
use std::marker::PhantomData;
use std::ops::Range;

impl ErasedComponentValue {
    /// Metadata for a borrowed archive row. No reference escapes or T clone;
    /// the internal collector ties this pointer to the owning value's borrow.
    pub(crate) fn borrowed_archive_slice(
        &self,
        archive: &RawColumnArchive,
    ) -> Result<RawArchiveSlice, StateError> {
        if self.component_type_id() != archive.component || self.rust_type_id() != archive.rust_type
        {
            return Err(StateError::AccessDenied(
                "Inverse erased archive type mismatch".into(),
            ));
        }
        let base = match &self.repr {
            ValueRepr::Inline { buffer, .. } => buffer.bytes.as_ptr().cast(),
            ValueRepr::Heap(value) => value.as_any() as *const dyn std::any::Any as *const u8,
            ValueRepr::Placeholder { .. } => {
                return Err(StateError::AccessDenied(
                    "Cannot archive an unfilled inverse placeholder".into(),
                ))
            }
        };
        Ok(RawArchiveSlice {
            base,
            offset: 0,
            len: 1,
        })
    }
}

impl ErasedComponentValue {
    #[inline]
    pub fn new<T: ModelComponent + Serialize + DeserializeOwned>(value: T) -> Self {
        Self::with_component_type_id(ComponentTypeId::of::<T>(), value)
    }

    /// Construct a zero-allocation empty/dummy placeholder value (unit type `()`).
    /// Useful for taking ownership of values via `std::mem::replace`.
    #[inline]
    pub fn empty() -> Self {
        Self::new(())
    }

    /// Use a registered schema identity. Columns still check the Rust type,
    /// so assigning another component's identity cannot bypass validation.
    #[inline]
    pub fn with_component_type_id<T: ModelComponent + Serialize + DeserializeOwned>(
        component_type_id: ComponentTypeId,
        value: T,
    ) -> Self {
        let repr = if std::mem::size_of::<T>() <= INLINE_SIZE
            && std::mem::align_of::<T>() <= INLINE_ALIGN
        {
            let mut buffer = InlineBuffer::uninit();
            unsafe {
                std::ptr::write(buffer.bytes.as_mut_ptr() as *mut T, value);
            }
            ValueRepr::Inline {
                buffer,
                vtable: &<T as HasInlineVTable>::VTABLE,
            }
        } else {
            ValueRepr::Heap(Box::new(value))
        };

        Self {
            component_type_id,
            type_name: std::any::type_name::<T>(),
            repr,
        }
    }

    #[inline]
    pub fn component_type_id(&self) -> ComponentTypeId {
        self.component_type_id
    }

    #[inline]
    pub fn rust_type_id(&self) -> TypeId {
        match &self.repr {
            ValueRepr::Inline { vtable, .. } => vtable.rust_type_id,
            ValueRepr::Heap(boxed) => boxed.as_any().type_id(),
            ValueRepr::Placeholder { rust_type_id } => *rust_type_id,
        }
    }

    /// Concrete native Rust type, independent of any supplied schema identity.
    #[inline]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    #[inline]
    pub fn size(&self) -> usize {
        match &self.repr {
            ValueRepr::Inline { vtable, .. } => vtable.size,
            ValueRepr::Heap(boxed) => std::mem::size_of_val(boxed.as_any()),
            ValueRepr::Placeholder { .. } => 0,
        }
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        match &self.repr {
            ValueRepr::Inline { vtable, .. } => vtable.align,
            ValueRepr::Heap(boxed) => std::mem::align_of_val(boxed.as_any()),
            ValueRepr::Placeholder { .. } => 1,
        }
    }

    /// Compare declared codec/type/layout metadata to the actual boxed type.
    /// Computing native schema metadata never invokes the component's codec.
    pub fn validate_schema(&self, schema: &ComponentSchema) -> Result<(), StateError> {
        let matches = match &self.repr {
            ValueRepr::Inline { vtable, .. } => {
                self.component_type_id == schema.type_id
                    && (vtable.schema)(schema.version) == *schema
            }
            ValueRepr::Heap(boxed) => {
                self.component_type_id == schema.type_id
                    && boxed.schema(schema.version) == *schema
            }
            ValueRepr::Placeholder { .. } => self.component_type_id == schema.type_id,
        };

        if !matches {
            return Err(StateError::AccessDenied(format!(
                "Native component {} ({}, size {}, alignment {}) does not match declared schema {} ({}, size {}, alignment {})",
                self.type_name(),
                self.component_type_id(),
                self.size(),
                self.alignment(),
                schema.type_name,
                schema.type_id,
                schema.size,
                schema.alignment,
            )));
        }
        Ok(())
    }

    #[inline]
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        match &self.repr {
            ValueRepr::Inline { buffer, vtable } => {
                if vtable.rust_type_id == TypeId::of::<T>() {
                    unsafe { Some(&*(buffer.bytes.as_ptr() as *const T)) }
                } else {
                    None
                }
            }
            ValueRepr::Heap(boxed) => boxed.as_any().downcast_ref::<T>(),
            ValueRepr::Placeholder { .. } => None,
        }
    }

    #[inline]
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        match &mut self.repr {
            ValueRepr::Inline { buffer, vtable } => {
                if vtable.rust_type_id == TypeId::of::<T>() {
                    unsafe { Some(&mut *(buffer.bytes.as_mut_ptr() as *mut T)) }
                } else {
                    None
                }
            }
            ValueRepr::Heap(boxed) => boxed.as_any_mut().downcast_mut::<T>(),
            ValueRepr::Placeholder { .. } => None,
        }
    }

    #[inline]
    pub fn clone_value<T: Clone + 'static>(&self) -> Option<T> {
        self.downcast_ref::<T>().cloned()
    }

    /// Explicitly encode this value for a persistence or transport boundary.
    pub fn encode(&self) -> Result<Vec<u8>, StateError> {
        match &self.repr {
            ValueRepr::Inline { buffer, vtable } => unsafe {
                (vtable.encode)(buffer.bytes.as_ptr() as *const u8)
            },
            ValueRepr::Heap(boxed) => boxed.encode(),
            ValueRepr::Placeholder { .. } => Err(StateError::AccessDenied(
                "Cannot encode placeholder component value".into(),
            )),
        }
    }

    #[inline]
    pub(crate) fn empty_column(&self) -> Box<dyn AnyColumn> {
        match &self.repr {
            ValueRepr::Inline { vtable, .. } => (vtable.empty_column)(),
            ValueRepr::Heap(boxed) => boxed.empty_column(),
            ValueRepr::Placeholder { .. } => {
                panic!("Cannot create empty column from placeholder component value")
            }
        }
    }

    pub(crate) fn take_typed<T: 'static>(&mut self) -> Result<T, StateError> {
        if self.is_placeholder() {
            return Err(StateError::AccessDenied(format!(
                "Erased value {} is a placeholder and cannot be downcast to concrete type {}",
                self.type_name,
                std::any::type_name::<T>()
            )));
        }
        if self.rust_type_id() != TypeId::of::<T>() {
            return Err(StateError::AccessDenied(format!(
                "Erased value {} cannot be downcast to {}",
                self.type_name,
                std::any::type_name::<T>()
            )));
        }
        let rust_type_id = TypeId::of::<T>();
        match std::mem::replace(&mut self.repr, ValueRepr::Placeholder { rust_type_id }) {
            ValueRepr::Inline { buffer, .. } => {
                Ok(unsafe { std::ptr::read(buffer.bytes.as_ptr() as *const T) })
            }
            ValueRepr::Heap(boxed) => boxed
                .into_any()
                .downcast::<T>()
                .map(|value| *value)
                .map_err(|_| {
                    StateError::AccessDenied(format!(
                        "Erased value {} cannot be downcast to {}",
                        self.type_name,
                        std::any::type_name::<T>()
                    ))
                }),
            ValueRepr::Placeholder { .. } => unreachable!(),
        }
    }

    #[inline]
    pub(crate) fn into_typed<T: 'static>(mut self) -> Result<T, StateError> {
        self.take_typed::<T>()
    }

    #[inline(always)]
    pub(crate) unsafe fn take_typed_unchecked<T: 'static>(&mut self) -> T {
        if !std::mem::needs_drop::<T>()
            && std::mem::size_of::<T>() <= INLINE_SIZE
            && std::mem::align_of::<T>() <= INLINE_ALIGN
        {
            if let ValueRepr::Inline { buffer, .. } = &self.repr {
                return unsafe { std::ptr::read(buffer.bytes.as_ptr() as *const T) };
            }
        }
        let rust_type_id = TypeId::of::<T>();
        match std::mem::replace(&mut self.repr, ValueRepr::Placeholder { rust_type_id }) {
            ValueRepr::Inline { buffer, .. } => unsafe {
                std::ptr::read(buffer.bytes.as_ptr() as *const T)
            },
            ValueRepr::Heap(boxed) => *boxed
                .into_any()
                .downcast::<T>()
                .expect("prevalidated component type"),
            ValueRepr::Placeholder { .. } => unreachable!(),
        }
    }

    /// Construct a typed placeholder value with zero memory allocation and no-op Drop.
    #[inline]
    pub fn placeholder(
        component_type_id: ComponentTypeId,
        rust_type_id: TypeId,
        type_name: &'static str,
    ) -> Self {
        Self {
            component_type_id,
            type_name,
            repr: ValueRepr::Placeholder { rust_type_id },
        }
    }

    /// Check if this erased value is a placeholder.
    #[inline]
    pub fn is_placeholder(&self) -> bool {
        matches!(self.repr, ValueRepr::Placeholder { .. })
    }
}

impl<'a> BorrowedArchiveRows<'a> {
    pub(crate) fn from_capability(archive: RawColumnArchive) -> Self {
        Self {
            archive,
            slices: SmallVec::new(),
            _borrow: PhantomData,
        }
    }

    pub(crate) fn capability(&self) -> &RawColumnArchive {
        &self.archive
    }

    pub(crate) fn push_slice<T: 'static>(&mut self, values: &'a [T]) -> Result<(), StateError> {
        if std::any::TypeId::of::<T>() != self.archive.rust_type {
            return Err(StateError::AccessDenied(
                "Inverse archive native type mismatch".into(),
            ));
        }
        self.slices.push(RawArchiveSlice {
            base: values.as_ptr().cast(),
            offset: 0,
            len: values.len(),
        });
        Ok(())
    }

    pub(crate) fn push_erased(
        &mut self,
        value: &'a ErasedComponentValue,
    ) -> Result<(), StateError> {
        self.slices
            .push(value.borrowed_archive_slice(&self.archive)?);
        Ok(())
    }

    /// SAFETY: each extent has the capability's concrete type, lies in ONE
    /// physical allocation, remains alive for 'a, and is immutable/protected.
    pub(crate) unsafe fn extend_raw(&mut self, slices: impl IntoIterator<Item = RawArchiveSlice>) {
        self.slices.extend(slices);
    }

    pub(crate) fn finish(self) -> Result<ArchivedColumnData, StateError> {
        unsafe {
            (self.archive.encode)(
                self.archive.component,
                self.archive.schema_version,
                &self.slices,
            )
        }
    }
}

pub(crate) fn encode_erased_block_with_archive(
    block: &dyn ErasedBlock,
    range: Range<usize>,
    archive: RawColumnArchive,
) -> Result<ArchivedColumnData, StateError> {
    let mut rows = BorrowedArchiveRows::from_capability(archive);
    block.borrow_archive_rows(&mut rows, range)?;
    rows.finish()
}

impl ColumnDescriptor {
    /// 从动态列安全构造描述符
    #[inline]
    pub(crate) fn from_column(col: &mut dyn AnyColumn) -> Self {
        let (raw_ptr, len) = unsafe { col.raw_parts_mut() };
        Self {
            raw_ptr,
            len,
            rust_type_id: col.rust_type_id(),
            type_name: col.type_name(),
            ops: col.ops_vtable(),
            archive: col.raw_archive_capability(),
            writable: true,
            _allocation_pin: None,
            physical_version: col.physical_version(),
        }
    }

    /// 从只读动态列安全构造描述符（无 &mut 要求）
    #[inline]
    pub(crate) fn from_column_ref(col: &dyn AnyColumn) -> Self {
        let (raw_ptr, len) = unsafe { col.raw_parts_read() };
        Self {
            raw_ptr: raw_ptr as *mut u8,
            len,
            rust_type_id: col.rust_type_id(),
            type_name: col.type_name(),
            ops: col.ops_vtable(),
            archive: col.raw_archive_capability(),
            writable: false,
            _allocation_pin: None,
            physical_version: col.physical_version(),
        }
    }

    /// Exclusive metadata refresh: derive mutable pointers ONLY from unique
    /// backing allocations. Shared portions stay read-only; never trigger CoW.
    pub(crate) fn from_column_access(col: &mut dyn AnyColumn) -> Self {
        let (raw_ptr, len, writable) = col.access_parts();
        Self {
            raw_ptr,
            len,
            writable,
            _allocation_pin: None,
            physical_version: col.physical_version(),
            rust_type_id: col.rust_type_id(),
            type_name: col.type_name(),
            ops: col.ops_vtable(),
            archive: col.raw_archive_capability(),
        }
    }
}

impl ColumnDescriptor {
    #[inline]
    pub(crate) fn write_ready(&self, range: std::ops::Range<usize>) -> bool {
        if range.start > range.end || range.end > self.len {
            return false;
        }
        if range.is_empty() {
            return true;
        }
        self.writable
    }

    #[inline]
    pub(crate) fn locate(&self, row: usize) -> (*mut u8, usize, usize) {
        (self.raw_ptr, row, self.len)
    }

    #[inline]
    pub(crate) fn locate_with_cursor(
        &self,
        row: usize,
        _cursor: &mut usize,
    ) -> (*mut u8, usize, usize) {
        (self.raw_ptr, row, self.len)
    }

    /// 从受保护的单行内存安全派生组件只读引用
    #[inline]
    pub(crate) unsafe fn get_unchecked<'a, T: 'static>(&self, row: usize) -> &'a T {
        debug_assert!(row < self.len, "row index out of bounds");
        let (ptr, offset, _) = self.locate(row);
        unsafe { &*(ptr as *const T).add(offset) }
    }

    /// 在指定行原位交换擦除组件值并返回旧值
    #[inline]
    pub(crate) unsafe fn replace_unchecked(
        &self,
        row: usize,
        value: ErasedComponentValue,
    ) -> ErasedComponentValue {
        debug_assert!(row < self.len, "row index out of bounds");
        debug_assert!(
            self.write_ready(row..row + 1),
            "read-only backing cannot be written"
        );
        let (ptr, offset, _) = self.locate(row);
        unsafe { (self.ops.replace_raw)(ptr, offset, value) }
    }

    #[inline]
    pub(crate) unsafe fn replace_unchecked_with_cursor(
        &self,
        row: usize,
        value: ErasedComponentValue,
        cursor: &mut usize,
    ) -> ErasedComponentValue {
        debug_assert!(row < self.len, "row index out of bounds");
        debug_assert!(
            self.write_ready(row..row + 1),
            "read-only backing cannot be written"
        );
        let (ptr, offset, _) = self.locate_with_cursor(row, cursor);
        unsafe { (self.ops.replace_raw)(ptr, offset, value) }
    }

    /// 在指定行原位覆盖擦除组件值并释放旧值
    #[inline]
    pub(crate) unsafe fn assign_unchecked(&self, row: usize, value: ErasedComponentValue) {
        debug_assert!(row < self.len, "row index out of bounds");
        debug_assert!(
            self.write_ready(row..row + 1),
            "read-only backing cannot be written"
        );
        let (ptr, offset, _) = self.locate(row);
        unsafe { (self.ops.assign_raw)(ptr, offset, value) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_ready_validates_contiguous_ranges() {
        let column = super::super::TypedColumn::<u64>::new("u64");
        let mut desc = ColumnDescriptor::from_column_ref(&column);
        desc.len = 10;
        for writable in [false, true] {
            desc.writable = writable;
            for start in 0..=12 {
                for end in 0..=12 {
                    let expected = if start > end || end > 10 {
                        false
                    } else if start == end {
                        true
                    } else {
                        writable
                    };
                    assert_eq!(desc.write_ready(start..end), expected);
                }
            }
        }
    }
}
