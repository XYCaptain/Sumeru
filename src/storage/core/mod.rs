//! 第一层：存储核心层 (Core Layer - 纯数据与物理布局)
//!
//! 职责：实体槽位分配、类型擦除与强类型连续列、原型表 (Archetype)、组件注册表、底层 Store 数据容器与快照。

pub(crate) mod allocator;
pub(crate) mod archetype;
pub(crate) mod column;
pub(crate) mod entity_index;
pub(crate) mod modify;
pub(crate) mod registry;
pub(crate) mod store;

pub(crate) use column::{
    AnyColumn, ColumnBacking, ColumnBlock, ColumnDescriptor, ColumnIter, ColumnValueOps,
    DirectBlock, RawColumn, TypedColumn,
};
pub use column::{
    archived_column_to_opaque_binary_array, archived_column_to_typed_arrow_array,
    arrow_archive_slices, arrow_restore_column, arrow_validate_column,
    build_record_batch_from_arrays, column_block_to_arrow_buffer_zero_copy,
    column_block_to_primitive_array_zero_copy, column_block_to_scalar_buffer_zero_copy,
    parse_arrow_columnar_header, return_pooled_typed_vec, take_pooled_typed_vec,
    write_arrow_columnar_header, ArchiveComponent, ArchivedColumnData, ArrowColumnarComponent,
    ArrowScalarField, ColumnCodecId, ComponentRegistration, ErasedBlock, ErasedComponentValue,
    ErasedOwnedBlock, FragmentPayload, PageAlignedBuffer, RkyvDefaultDeserializer,
    RkyvDefaultSerializer, RkyvDefaultValidator, TypedOwnedBlock,
};
pub use modify::ModifyTarget;
#[allow(unused_imports)]
pub(crate) use registry::ComponentRegistry;
pub use store::{
    ColumnarStoreSnapshot, LayoutOptimization, Singleton, Store, TypedBatch,
    TypedDirectBatchUpdate, TypedReceipt, CURRENT_SNAPSHOT_VERSION,
};

#[cfg(test)]
mod allocator_tests;
#[cfg(test)]
mod archetype_tests;
#[cfg(test)]
mod tests;
