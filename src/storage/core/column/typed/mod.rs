//! 强类型连续列存储 (TypedColumn)

use super::any::{
    AnyColumn, ColumnExperimentOps, ColumnLayout, ColumnOpsVTable, ColumnRows, ColumnSnapshot,
    ColumnValueOps, ComponentValue, ErasedComponentValue, HasInlineVTable, InlineVTable,
};
use super::archive::{
    ArchivedColumnData, ColumnArchiveVTable, ColumnCodecId, RkyvDefaultDeserializer,
    RkyvDefaultSerializer, RkyvDefaultValidator,
};
use super::backing::{ColumnBacking, ColumnBlock};
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use serde::de::DeserializeOwned;
use serde::Serialize;
use smallvec::SmallVec;
use std::any::Any;
use std::sync::Arc;

mod any_column;
mod iter;
mod ops;
#[cfg(test)]
mod tests;

pub use iter::ColumnIter;

/// 强类型连续列存储 (TypedColumn)
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TypedColumn<T: ModelComponent + Serialize + DeserializeOwned> {
    pub(crate) type_id: ComponentTypeId,
    pub(crate) type_name: &'static str,
    pub(super) backing: ColumnBacking<T>,
    pub(super) archive_vtable: ColumnArchiveVTable<T>,
}

#[allow(dead_code)]
impl<T: ModelComponent + Serialize + DeserializeOwned> TypedColumn<T> {
    pub fn new(type_name: &'static str) -> Self {
        Self::new_unsupported(type_name)
    }

    pub fn new_unsupported(type_name: &'static str) -> Self {
        Self {
            type_id: ComponentTypeId::from_name(type_name),
            type_name,
            backing: ColumnBacking::new(),
            archive_vtable: ColumnArchiveVTable::new_unsupported(),
        }
    }

    pub fn new_with_vtable(
        type_name: &'static str,
        archive_vtable: ColumnArchiveVTable<T>,
    ) -> Self {
        Self {
            type_id: ComponentTypeId::from_name(type_name),
            type_name,
            backing: ColumnBacking::new(),
            archive_vtable,
        }
    }

    pub fn new_rkyv(type_name: &'static str) -> Self
    where
        T: for<'a> rkyv::Serialize<RkyvDefaultSerializer<'a>>,
        Vec<T>: for<'a> rkyv::Serialize<RkyvDefaultSerializer<'a>>
            + rkyv::Archive<Archived = rkyv::vec::ArchivedVec<rkyv::Archived<T>>>,
        rkyv::vec::ArchivedVec<rkyv::Archived<T>>: rkyv::Portable
            + for<'a> rkyv::bytecheck::CheckBytes<RkyvDefaultValidator<'a>>
            + rkyv::Deserialize<Vec<T>, RkyvDefaultDeserializer>,
    {
        Self {
            type_id: ComponentTypeId::from_name(type_name),
            type_name,
            backing: ColumnBacking::new(),
            archive_vtable: ColumnArchiveVTable::new_rkyv(),
        }
    }

    pub fn new_arrow_columnar(type_name: &'static str) -> Self
    where
        T: super::archive::ArrowColumnarComponent,
    {
        Self {
            type_id: ComponentTypeId::from_name(type_name),
            type_name,
            backing: ColumnBacking::new(),
            archive_vtable: ColumnArchiveVTable::new_arrow_columnar(),
        }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.backing.reserve(additional);
    }

    #[inline]
    pub fn push(&mut self, value: T) -> usize {
        self.backing.push(value)
    }

    #[inline(always)]
    pub(crate) unsafe fn push_direct_owned_unchecked(&mut self, value: T) {
        unsafe {
            self.backing.push_direct_owned_unchecked(value);
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn swap_remove_direct_owned_unchecked(&mut self, row: usize) -> T {
        unsafe { self.backing.swap_remove_direct_owned_unchecked(row) }
    }

    #[inline]
    pub fn get(&self, row: usize) -> Option<&T> {
        self.backing.get(row)
    }

    #[inline]
    pub fn get_mut(&mut self, row: usize) -> Option<&mut T> {
        self.backing.get_mut(row)
    }

    #[inline]
    pub fn physical_version(&self) -> u64 {
        self.backing.physical_version()
    }

    #[inline]
    pub fn borrow_row_mut(&mut self, row: usize) -> Option<&mut T> {
        self.backing.borrow_row_mut(row)
    }

    #[inline]
    pub fn replace_row_infallible(&mut self, row: usize, val: T) -> T {
        self.backing.replace_row_infallible(row, val)
    }

    #[inline]
    pub(crate) unsafe fn write_exclusive_row_unchecked(&mut self, row: usize, val: T) -> T {
        unsafe { self.backing.write_exclusive_row_unchecked(row, val) }
    }

    #[inline]
    pub fn set(&mut self, row: usize, value: T) -> Result<(), StateError> {
        if let Some(slot) = self.backing.get_mut(row) {
            *slot = value;
            Ok(())
        } else {
            Err(StateError::AccessDenied(format!(
                "Row {} out of bounds for column {}",
                row, self.type_name
            )))
        }
    }

    #[inline]
    pub fn swap_remove(&mut self, row: usize) -> Option<T> {
        self.backing.swap_remove(row)
    }

    #[inline]
    pub fn swap_remove_and_drop(&mut self, row: usize) {
        self.backing.swap_remove_and_drop(row);
    }

    #[inline]
    pub fn swap_remove_batch_sorted_and_drop(&mut self, sorted_desc_rows: &[usize]) {
        self.backing.swap_remove_batch_sorted_and_drop(sorted_desc_rows);
    }

    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        self.backing.truncate(new_len);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.backing.clear();
    }

    #[inline]
    pub fn clone_empty(&self) -> Self {
        Self {
            type_id: self.type_id,
            type_name: self.type_name,
            backing: ColumnBacking::new(),
            archive_vtable: self.archive_vtable,
        }
    }

    #[inline]
    pub fn merge_from(&mut self, other: &Self) {
        let src = other.as_slice();
        if !src.is_empty() {
            self.backing.make_contiguous_owned().extend_from_slice(src);
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.backing.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.backing.is_empty()
    }

    #[inline]
    pub fn is_segmented(&self) -> bool {
        false
    }

    #[inline]
    pub fn segment_count(&self) -> usize {
        self.backing.segment_count()
    }

    #[inline]
    pub fn ensure_contiguous(&mut self) -> &mut [T] {
        self.backing.ensure_contiguous()
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        self.backing.as_contiguous_slice()
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut Vec<T> {
        self.backing.make_contiguous_owned()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.backing.as_contiguous_slice()
    }

    #[inline]
    pub fn as_direct_slice(&self) -> Option<&[T]> {
        self.backing.as_direct_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.backing.as_mut_slice()
    }

    #[inline]
    pub fn backing(&self) -> &ColumnBacking<T> {
        &self.backing
    }

    #[inline]
    pub fn backing_mut(&mut self) -> &mut ColumnBacking<T> {
        &mut self.backing
    }

    #[inline]
    pub fn iter(&self) -> ColumnIter<'_, T> {
        self.iter_from(0)
    }

    pub fn iter_from(&self, start: usize) -> ColumnIter<'_, T> {
        let slice = self.backing.as_contiguous_slice();
        if start >= slice.len() {
            ColumnIter::Direct([].iter())
        } else {
            ColumnIter::Direct(slice[start..].iter())
        }
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned> ColumnLayout for TypedColumn<T> {
    #[inline]
    fn type_id(&self) -> ComponentTypeId {
        self.type_id
    }

    #[inline]
    fn type_name(&self) -> &'static str {
        self.type_name
    }

    #[inline]
    fn rust_type_id(&self) -> Option<std::any::TypeId> {
        Some(std::any::TypeId::of::<T>())
    }

    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    #[inline]
    unsafe fn raw_parts_mut(&mut self) -> (*mut u8, usize) {
        let slice = self.backing.as_mut_slice();
        (slice.as_mut_ptr() as *mut u8, slice.len())
    }

    #[inline]
    unsafe fn raw_parts_read(&self) -> (*const u8, usize) {
        let slice = self.as_slice();
        (slice.as_ptr() as *const u8, slice.len())
    }

    fn access_parts(&mut self) -> (*mut u8, usize, bool) {
        self.backing.access_parts()
    }

    #[inline]
    fn ops_vtable(&self) -> ColumnOpsVTable {
        Self::OPS_VTABLE
    }

    fn shared_clone_raw(
        &self,
    ) -> Option<unsafe fn(*const u8, usize, ComponentTypeId) -> ErasedComponentValue> {
        unsafe fn clone_row<T: ModelComponent + Serialize + DeserializeOwned>(
            p: *const u8,
            row: usize,
            cid: ComponentTypeId,
        ) -> ErasedComponentValue {
            // SAFETY: caller protects this row; no unrelated reference is formed.
            let value = unsafe { (&*p.cast::<T>().add(row)).clone() };
            ErasedComponentValue::with_component_type_id(cid, value)
        }
        Some(clone_row::<T>)
    }

    fn shared_push_clone_raw(
        &self,
    ) -> Option<unsafe fn(*const u8, usize, &mut dyn AnyColumn) -> Result<(), StateError>> {
        unsafe fn push_clone_row<T: ModelComponent + Serialize + DeserializeOwned>(
            p: *const u8,
            row: usize,
            dst: &mut dyn AnyColumn,
        ) -> Result<(), StateError> {
            let dst_typed = dst
                .as_any_mut()
                .downcast_mut::<TypedColumn<T>>()
                .ok_or_else(|| {
                    StateError::AccessDenied("Destination column type mismatch".into())
                })?;
            // SAFETY: caller protects this row; no unrelated reference is formed.
            let value = unsafe { (&*p.cast::<T>().add(row)).clone() };
            dst_typed.push(value);
            Ok(())
        }
        Some(push_clone_row::<T>)
    }

    fn shared_push_clone_range_raw(
        &self,
    ) -> Option<unsafe fn(*const u8, usize, usize, &mut dyn AnyColumn) -> Result<(), StateError>> {
        unsafe fn push_clone_range<T: ModelComponent + Serialize + DeserializeOwned>(
            p: *const u8,
            start_row: usize,
            count: usize,
            dst: &mut dyn AnyColumn,
        ) -> Result<(), StateError> {
            let dst_typed = dst
                .as_any_mut()
                .downcast_mut::<TypedColumn<T>>()
                .ok_or_else(|| {
                    StateError::AccessDenied("Destination column type mismatch".into())
                })?;
            let vec = dst_typed.backing_mut().as_vec_mut();
            let base = unsafe { p.cast::<T>().add(start_row) };
            vec.extend((0..count).map(|i| unsafe { (&*base.add(i)).clone() }));
            Ok(())
        }
        Some(push_clone_range::<T>)
    }

    fn shared_element_size(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned> ColumnRows for TypedColumn<T> {
    #[inline]
    fn len(&self) -> usize {
        self.backing.len()
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.backing.reserve(additional);
    }

    #[inline]
    fn clear_rows(&mut self) {
        self.backing.clear();
    }

    #[inline]
    fn swap_remove(&mut self, row: usize) {
        let _ = self.backing.swap_remove(row);
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned + 'static> TypedColumn<T> {
    const OPS_VTABLE: ColumnOpsVTable = ColumnOpsVTable {
        assign_raw: Self::assign_raw_impl,
        replace_raw: Self::replace_raw_impl,
        new_typed_owned_block: Self::new_typed_owned_block_impl,
    };

    fn new_typed_owned_block_impl(
        component: ComponentTypeId,
        capacity: usize,
    ) -> Arc<dyn crate::storage::column::ErasedBlock> {
        Arc::new(crate::storage::column::TypedOwnedBlock::<T>::with_capacity(
            component,
            capacity,
        ))
    }

    unsafe fn assign_raw_impl(raw_ptr: *mut u8, row: usize, value: ErasedComponentValue) {
        let typed_val = value.into_typed::<T>().expect("validated component type");
        unsafe {
            let ptr = (raw_ptr as *mut T).add(row);
            *ptr = typed_val;
        }
    }

    unsafe fn replace_raw_impl(
        raw_ptr: *mut u8,
        row: usize,
        mut value: ErasedComponentValue,
    ) -> ErasedComponentValue {
        let replacement = value.downcast_mut::<T>().expect("validated component type");
        unsafe {
            let ptr = (raw_ptr as *mut T).add(row);
            std::mem::swap(&mut *ptr, replacement);
        }
        value
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned> ComponentValue for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn ComponentValue> {
        Box::new(self.clone())
    }

    fn encode(&self) -> Result<Vec<u8>, StateError> {
        bincode::serialize(self).map_err(|error| StateError::CodecError(error.to_string()))
    }

    fn empty_column(&self) -> Box<dyn AnyColumn> {
        Box::new(TypedColumn::<T>::new(std::any::type_name::<T>()))
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn schema(&self, version: crate::id::SchemaVersion) -> crate::schema::ComponentSchema {
        crate::schema::ComponentSchema::of::<T>(version)
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned + 'static> HasInlineVTable for T {
    const VTABLE: InlineVTable = InlineVTable {
        rust_type_id: std::any::TypeId::of::<T>(),
        size: std::mem::size_of::<T>(),
        align: std::mem::align_of::<T>(),
        drop: if std::mem::needs_drop::<T>() {
            Some(|ptr| unsafe {
                std::ptr::drop_in_place(ptr as *mut T);
            })
        } else {
            None
        },
        clone: |src, dst| unsafe {
            let val = (&*(src as *const T)).clone();
            std::ptr::write(dst as *mut T, val);
        },
        encode: |ptr| unsafe {
            let val = &*(ptr as *const T);
            bincode::serialize(val).map_err(|error| StateError::CodecError(error.to_string()))
        },
        empty_column: || Box::new(TypedColumn::<T>::new(std::any::type_name::<T>())),
        schema: |version| crate::schema::ComponentSchema::of::<T>(version),
    };
}

mod sealed {
    pub trait Sealed {}
    impl<T> Sealed for T
    where
        T: crate::context::ModelComponent
            + serde::Serialize
            + serde::de::DeserializeOwned
            + for<'a> rkyv::Serialize<super::RkyvDefaultSerializer<'a>>
            + 'static,
        Vec<T>: for<'a> rkyv::Serialize<super::RkyvDefaultSerializer<'a>>
            + rkyv::Archive<Archived = rkyv::vec::ArchivedVec<rkyv::Archived<T>>>,
        rkyv::vec::ArchivedVec<rkyv::Archived<T>>: rkyv::Portable
            + for<'a> rkyv::bytecheck::CheckBytes<super::RkyvDefaultValidator<'a>>
            + rkyv::Deserialize<Vec<T>, super::RkyvDefaultDeserializer>,
    {
    }
}

/// 标识具备整列 rkyv 快照归档能力的组件契约。
///
/// 本 trait 采用 Sealed Trait 模式封闭，严禁外部手动实现。
/// 满足内存组件/serde 契约，且元素支持 rkyv 序列化、`Vec<Self>` 支持
/// 受检 rkyv 恢复的类型自动实现此 trait；普通内存接口不要求此能力。
pub trait ArchiveComponent:
    sealed::Sealed + ModelComponent + Serialize + DeserializeOwned + 'static
{
    /// 构造该组件绑定的强类型 rkyv 整列注册信息
    #[doc(hidden)]
    fn registration() -> ComponentRegistration;
}

impl<T> ArchiveComponent for T
where
    T: ModelComponent
        + Serialize
        + DeserializeOwned
        + for<'a> rkyv::Serialize<RkyvDefaultSerializer<'a>>,
    Vec<T>: for<'a> rkyv::Serialize<RkyvDefaultSerializer<'a>>
        + rkyv::Archive<Archived = rkyv::vec::ArchivedVec<rkyv::Archived<T>>>,
    rkyv::vec::ArchivedVec<rkyv::Archived<T>>: rkyv::Portable
        + for<'a> rkyv::bytecheck::CheckBytes<RkyvDefaultValidator<'a>>
        + rkyv::Deserialize<Vec<T>, RkyvDefaultDeserializer>,
{
    fn registration() -> ComponentRegistration {
        ComponentRegistration {
            column: Box::new(TypedColumn::<T>::new_rkyv(std::any::type_name::<T>())),
        }
    }
}

/// Trusted native component schema without exposing physical column operations.
pub struct ComponentRegistration {
    pub(crate) column: Box<dyn AnyColumn>,
}

impl ComponentRegistration {
    /// 统一注册组件（整列统一且仅使用 rkyv 归档）
    pub fn of<T: ArchiveComponent>() -> Self {
        let reg = T::registration();
        debug_assert!(
            reg.has_archive_capability(),
            "ArchiveComponent must produce a column with archive capability"
        );
        reg
    }

    /// 显式注册 rkyv 整列归档路径（与 of 等价）
    pub fn of_rkyv<T: ArchiveComponent>() -> Self {
        Self::of::<T>()
    }

    /// 显式注册 `ColumnCodecId::ArrowColumnar = 3` 列式归档路径
    pub fn of_arrow_columnar<T>() -> Self
    where
        T: super::archive::ArrowColumnarComponent + Serialize + DeserializeOwned + 'static,
    {
        Self {
            column: Box::new(TypedColumn::<T>::new_arrow_columnar(
                std::any::type_name::<T>(),
            )),
        }
    }

    /// 检查该组件原型是否具备整列快照归档能力
    #[inline]
    pub fn has_archive_capability(&self) -> bool {
        self.column.has_archive_capability()
    }

    /// 注册纯内存组件列原型（无整列快照归档能力）
    pub fn of_unsupported<T>() -> Self
    where
        T: ModelComponent + Serialize + DeserializeOwned + 'static,
    {
        Self {
            column: Box::new(TypedColumn::<T>::new_unsupported(
                std::any::type_name::<T>(),
            )),
        }
    }
}

impl ErasedComponentValue {
    /// Empty type prototype for trusted schema installation; no values are
    /// copied and no component codec is invoked.
    pub fn registration(&self) -> ComponentRegistration {
        ComponentRegistration {
            column: self.empty_column(),
        }
    }
}
