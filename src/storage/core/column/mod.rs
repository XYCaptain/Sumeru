//! 强类型列缓冲区与组件列存储 (TypedColumn & ColumnBuffer)

pub(crate) mod any;
pub(crate) mod archive;
pub mod backing;
pub mod block;
pub(crate) mod descriptor;
pub mod page_aligned;
pub(crate) mod raw;
pub(crate) mod typed;

#[allow(unused_imports)]
pub use backing::{ColumnBacking, ColumnBlock, DirectBlock};
#[allow(unused_imports)]
pub use page_aligned::PageAlignedBuffer;

pub use any::ErasedComponentValue;
#[allow(unused_imports)]
pub(crate) use any::{
    AllocationPin, AnyColumn, ColumnDescriptor, ColumnExperimentOps, ColumnLayout, ColumnOpsVTable,
    ColumnRows, ColumnSnapshot, ColumnValueOps, ComponentValue, HasInlineVTable, INLINE_ALIGN,
    INLINE_SIZE, InPlaceOp, InPlaceReplacements, InlineBuffer, InlineVTable, PlannedInPlacePatch,
    PreparedRow, ValueRepr, sealed,
};
#[allow(unused_imports)]
pub use any::{BorrowedArchiveRows, ErasedBlock};
#[allow(unused_imports)]
pub use block::{
    ErasedOwnedBlock, FragmentPayload, TypedOwnedBlock, return_pooled_typed_vec,
    take_pooled_typed_vec,
};
#[allow(unused_imports)]
pub(crate) use descriptor::encode_erased_block_with_archive;
#[allow(unused_imports)]
pub(crate) use raw::RawColumn;
#[allow(unused_imports)]
pub(crate) use typed::TypedColumn;
pub use typed::{ArchiveComponent, ColumnIter, ComponentRegistration};

#[allow(unused_imports)]
pub use archive::{
    archived_column_to_opaque_binary_array, archived_column_to_typed_arrow_array,
    arrow_archive_slices, arrow_restore_column, arrow_validate_column,
    build_record_batch_from_arrays, column_block_to_arrow_buffer_zero_copy,
    column_block_to_primitive_array_zero_copy, column_block_to_scalar_buffer_zero_copy,
    parse_arrow_columnar_header, write_arrow_columnar_header, ArchivedColumnData,
    ArrowColumnarComponent, ArrowScalarField, ColumnCodecId,
    RkyvDefaultDeserializer, RkyvDefaultSerializer, RkyvDefaultValidator,
};

#[allow(unused_imports)]
pub(crate) use archive::{
    rkyv_archive_slices, rkyv_restore_column, rkyv_restore_column_from_slice, rkyv_validate_column,
    validate_column_header, ColumnArchiveVTable,
};
