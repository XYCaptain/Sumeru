//! L1 内存列 (`ColumnBlock<T>` / `TypedColumn<T>`) 与 Apache Arrow 列式生态的零拷贝及向量化桥接层
//!
//! 承载时光机列式数据库在 L1 热态内存层的双轨编码策略：
//! 1. **快路径（$O(1)$ 物理内存零拷贝）**：
//!    当 `ColumnBlock<T>` 承载原生标量类型（`T: ArrowNativeType`）时，通过
//!    `arrow::buffer::Buffer::from_custom_allocation` 直接将 `Arc<ColumnBlock<T>>` 托管给 Arrow `Buffer` /
//!    `ScalarBuffer<T>` / `PrimitiveArray<P>`。Arrow 数组底层物理地址与 `ColumnBlock<T>::as_slice().as_ptr()`
//!    严格相等，零内存分配、零字节拷贝；当 Arrow `RecordBatch` 生命周期结束时，自动触发 `ColumnBlock<T>`
//!    安全释放底层连续内存。
//! 2. **向量化结构列路径（`ColumnCodecId::ArrowColumnar = 3`）**：
//!    针对多字段物理组件或需下沉至 L3 Tonbo Parquet / DataFusion SQL 的结构化组件，提供
//!    `ArrowColumnarComponent` 契约与 `impl_arrow_struct_component!` 宏，实现从物理切片 `&[&[T]]`
//!    到 Arrow `StructArray` / `PrimitiveArray` 的紧致小端序向量化转换及双向无损恢复。

use std::panic::AssertUnwindSafe;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow::alloc::Allocation;
use arrow::array::{
    Array, ArrayRef, BinaryArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, PrimitiveArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{
    ArrowNativeType, ArrowPrimitiveType, DataType, Float32Type, Float64Type, Int16Type, Int32Type,
    Int64Type, Int8Type, SchemaRef, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use arrow::record_batch::RecordBatch;
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::{validate_column_header, ArchivedColumnData, ColumnCodecId, RawArchiveSlice};
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use crate::storage::column::ColumnBlock;

/// Arrow 列式组件归档魔数（8 字节对齐头）
const ARROW_COLUMNAR_MAGIC: &[u8; 4] = b"ARWC";
/// Arrow 列式组件归档格式版本
const ARROW_COLUMNAR_FORMAT_V1: u32 = 1;
/// Arrow 列式归档固定头部字节数：`magic(4) + version(4) + row_count(4) + field_count(4) = 16B`
const ARROW_COLUMNAR_HEADER_BYTES: usize = 16;

/// 支持 `ColumnCodecId::ArrowColumnar = 3` 列式归档与 Arrow 向量化直通的组件契约
pub trait ArrowColumnarComponent: ModelComponent + Sized {
    /// 对应的 Arrow 逻辑数据类型（标量 `DataType::Float64` 或复合结构 `DataType::Struct(...)`）
    fn arrow_data_type() -> DataType;

    /// 每行在 Arrow 列式归档中的固定字节宽度（定长组件返回 `Some(size)`）
    fn fixed_row_bytes() -> Option<usize>;

    /// 将多个物理连续段 `&[&[Self]]` 直接向量化转换为 Arrow `ArrayRef`
    fn slices_to_arrow_array(slices: &[&[Self]]) -> Result<ArrayRef, StateError>;

    /// 从 Arrow `Array` 向量化还原追加至 `Vec<Self>`
    fn arrow_array_to_vec(array: &dyn Array, out: &mut Vec<Self>) -> Result<(), StateError>;

    /// 将多个物理连续段编码为 `ColumnCodecId::ArrowColumnar` 紧致列存载荷
    fn encode_arrow_columnar_bytes(slices: &[&[Self]]) -> Result<(u32, Vec<u8>), StateError>;

    /// 校验 `ColumnCodecId::ArrowColumnar` 载荷完整性（零反序列化开销）
    fn validate_arrow_columnar_bytes(row_count: u32, payload: &[u8]) -> Result<(), StateError>;

    /// 从 `ColumnCodecId::ArrowColumnar` 载荷恢复至目标 `Vec<Self>`
    fn decode_arrow_columnar_bytes(
        row_count: u32,
        payload: &[u8],
        out: &mut Vec<Self>,
    ) -> Result<(), StateError>;
}

// ============================================================================
// 1. O(1) 零拷贝快路径：Arc<ColumnBlock<T>> -> Arrow Buffer / ScalarBuffer / PrimitiveArray
// ============================================================================

/// 将冻结的 `Arc<ColumnBlock<T>>` 以 $O(1)$ 零拷贝方式封装为 Arrow [`Buffer`]。
///
/// # 内存安全与生命周期保证
/// 1. `ColumnBlock<T>` 一旦封装进 `Arc`，其内部 `Vec<T>` 在整个共享生命周期内只读且绝不发生重分配。
/// 2. `Buffer::from_custom_allocation` 持有 `Arc<ColumnBlock<T>>` 作为自定义 `owner`。
/// 3. 当 Arrow `Buffer` / `ScalarBuffer<T>` / `PrimitiveArray` 及其派生的所有零拷贝切片全部释放后，
///    `Arc<ColumnBlock<T>>` 引用计数归零，自动释放底层连续内存。
pub fn column_block_to_arrow_buffer_zero_copy<T>(block: Arc<ColumnBlock<T>>) -> Buffer
where
    T: ModelComponent + Serialize + DeserializeOwned + ArrowNativeType,
{
    let slice = block.as_slice();
    let byte_len = slice.len().saturating_mul(std::mem::size_of::<T>());
    let ptr = NonNull::new(slice.as_ptr() as *mut u8).unwrap_or_else(NonNull::dangling);
    let owner: Arc<dyn Allocation> = Arc::new(AssertUnwindSafe(block));
    // SAFETY:
    // - `ptr` 指向 `Arc<ColumnBlock<T>>` 内部不可变的 `Vec<T>` 首地址，满足 `T` 的对齐要求；
    // - `byte_len` 严格等于 `slice.len() * size_of::<T>()`，不越界；
    // - `owner` 延长了 `Arc<ColumnBlock<T>>` 的生命周期，直至返回的 `Buffer` 及其所有切片销毁。
    unsafe { Buffer::from_custom_allocation(ptr, byte_len, owner) }
}

/// 将冻结的 `Arc<ColumnBlock<T>>` 以 $O(1)$ 零拷贝方式封装为 Arrow [`ScalarBuffer<T>`]。
#[inline]
pub fn column_block_to_scalar_buffer_zero_copy<T>(block: Arc<ColumnBlock<T>>) -> ScalarBuffer<T>
where
    T: ModelComponent + Serialize + DeserializeOwned + ArrowNativeType,
{
    let len = block.len();
    let buffer = column_block_to_arrow_buffer_zero_copy(block);
    ScalarBuffer::new(buffer, 0, len)
}

/// 将冻结的 `Arc<ColumnBlock<P::Native>>` 以 $O(1)$ 零拷贝方式直接构造为 Arrow [`PrimitiveArray<P>`]。
#[inline]
pub fn column_block_to_primitive_array_zero_copy<P>(
    block: Arc<ColumnBlock<P::Native>>,
) -> PrimitiveArray<P>
where
    P: ArrowPrimitiveType,
    P::Native: ModelComponent + Serialize + DeserializeOwned,
{
    let scalar_buf = column_block_to_scalar_buffer_zero_copy(block);
    PrimitiveArray::<P>::new(scalar_buf, None)
}

// ============================================================================
// 2. ColumnCodecId::ArrowColumnar = 3 虚表算子实现
// ============================================================================

/// 将逻辑列的物理切片集合归档为 `ColumnCodecId::ArrowColumnar` 格式的 [`ArchivedColumnData`]
pub fn arrow_archive_slices<T>(
    type_id: ComponentTypeId,
    schema_version: u32,
    slices: &[&[T]],
) -> Result<ArchivedColumnData, StateError>
where
    T: ArrowColumnarComponent,
{
    let (row_count, payload) = T::encode_arrow_columnar_bytes(slices)?;
    Ok(ArchivedColumnData::new(
        type_id,
        schema_version,
        row_count,
        ColumnCodecId::ArrowColumnar,
        payload,
    ))
}

/// # Safety
/// 调用方必须保证每个 `RawArchiveSlice` 在本次调用期间均指向合法、对齐且受保护的 `T` 连续物理内存。
pub(crate) unsafe fn arrow_archive_raw_slices<T>(
    type_id: ComponentTypeId,
    schema_version: u32,
    raw: &[RawArchiveSlice],
) -> Result<ArchivedColumnData, StateError>
where
    T: ArrowColumnarComponent,
{
    let slices: smallvec::SmallVec<[&[T]; 4]> = raw
        .iter()
        .map(|slice| unsafe {
            std::slice::from_raw_parts(slice.base.cast::<T>().add(slice.offset), slice.len)
        })
        .collect();
    arrow_archive_slices(type_id, schema_version, &slices)
}

/// 校验 `ColumnCodecId::ArrowColumnar` 归档列的头部与字节载荷完整性
pub fn arrow_validate_column<T>(
    type_id: ComponentTypeId,
    schema_version: u32,
    archived: &ArchivedColumnData,
) -> Result<(), StateError>
where
    T: ArrowColumnarComponent,
{
    validate_column_header(
        type_id,
        schema_version,
        ColumnCodecId::ArrowColumnar,
        archived,
    )?;
    T::validate_arrow_columnar_bytes(archived.row_count(), archived.payload())
}

/// 从 `ColumnCodecId::ArrowColumnar` 归档列恢复强类型 `Vec<T>`
pub fn arrow_restore_column<T>(
    type_id: ComponentTypeId,
    schema_version: u32,
    archived: &ArchivedColumnData,
    target: &mut Vec<T>,
) -> Result<(), StateError>
where
    T: ArrowColumnarComponent,
{
    validate_column_header(
        type_id,
        schema_version,
        ColumnCodecId::ArrowColumnar,
        archived,
    )?;
    let mut restored = Vec::with_capacity(archived.row_count() as usize);
    T::decode_arrow_columnar_bytes(archived.row_count(), archived.payload(), &mut restored)?;
    if restored.len() != archived.row_count() as usize {
        return Err(StateError::AccessDenied(format!(
            "Restored ArrowColumnar length {} does not match header row_count {}",
            restored.len(),
            archived.row_count()
        )));
    }
    *target = restored;
    Ok(())
}

/// 将任意已归档列 (`ArchivedColumnData`) 转换为 Arrow `ArrayRef`：
/// - 若提供强类型 `T: ArrowColumnarComponent` 且列编码为 `ArrowColumnar`，还原为强类型 Arrow 数组；
/// - 若为 `Rkyv` 或 `RawBytes` 编码，则封装为单元素/整列 `BinaryArray` 载荷以供通用快照透传。
pub fn archived_column_to_typed_arrow_array<T>(
    archived: &ArchivedColumnData,
) -> Result<ArrayRef, StateError>
where
    T: ArrowColumnarComponent,
{
    if archived.codec() != ColumnCodecId::ArrowColumnar {
        return Err(StateError::CodecError(format!(
            "Expected ArrowColumnar codec, got {:?}",
            archived.codec()
        )));
    }
    let mut values = Vec::with_capacity(archived.row_count() as usize);
    T::decode_arrow_columnar_bytes(archived.row_count(), archived.payload(), &mut values)?;
    let slice = values.as_slice();
    T::slices_to_arrow_array(&[slice])
}

/// 将不透明归档列封装为单行二进制 `BinaryArray`（用于 `Rkyv = 1` 复杂嵌套状态列在 Arrow 批次中的零损透传）
pub fn archived_column_to_opaque_binary_array(archived: &ArchivedColumnData) -> BinaryArray {
    BinaryArray::from_iter_values([archived.payload()])
}

/// 将一组 Arrow 列组装为 [`RecordBatch`]
pub fn build_record_batch_from_arrays(
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
) -> Result<RecordBatch, StateError> {
    RecordBatch::try_new(schema, columns)
        .map_err(|e| StateError::CodecError(format!("Arrow RecordBatch construction failed: {e}")))
}

// ============================================================================
// 3. 标量与多字段结构体的 ArrowColumnarComponent 实现宏
// ============================================================================

/// 校验并解析 16 字节 `ARROW_COLUMNAR_MAGIC` 头部，返回 `(row_count, field_count, body_slice)`
#[inline]
pub fn parse_arrow_columnar_header(
    expected_rows: u32,
    expected_fields: u32,
    payload: &[u8],
) -> Result<&[u8], StateError> {
    if payload.len() < ARROW_COLUMNAR_HEADER_BYTES {
        return Err(StateError::CodecError(
            "ArrowColumnar payload shorter than 16-byte header".into(),
        ));
    }
    if &payload[0..4] != ARROW_COLUMNAR_MAGIC {
        return Err(StateError::CodecError(
            "Invalid ArrowColumnar magic header".into(),
        ));
    }
    let version = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    if version != ARROW_COLUMNAR_FORMAT_V1 {
        return Err(StateError::CodecError(format!(
            "Unsupported ArrowColumnar format version {version}"
        )));
    }
    let rows = u32::from_le_bytes(payload[8..12].try_into().unwrap());
    if rows != expected_rows {
        return Err(StateError::CodecError(format!(
            "ArrowColumnar row count mismatch: header has {expected_rows}, payload has {rows}"
        )));
    }
    let fields = u32::from_le_bytes(payload[12..16].try_into().unwrap());
    if fields != expected_fields {
        return Err(StateError::CodecError(format!(
            "ArrowColumnar field count mismatch: expected {expected_fields}, got {fields}"
        )));
    }
    Ok(&payload[ARROW_COLUMNAR_HEADER_BYTES..])
}

/// 写入 16 字节 `ARROW_COLUMNAR_MAGIC` 头部
#[inline]
pub fn write_arrow_columnar_header(out: &mut Vec<u8>, row_count: u32, field_count: u32) {
    out.extend_from_slice(ARROW_COLUMNAR_MAGIC);
    out.extend_from_slice(&ARROW_COLUMNAR_FORMAT_V1.to_le_bytes());
    out.extend_from_slice(&row_count.to_le_bytes());
    out.extend_from_slice(&field_count.to_le_bytes());
}

/// 标量字段类型辅助 trait：用于将结构体字段以 Structure-of-Arrays (SoA) 紧致列布局写入/读出 Arrow 数组与字节缓冲
pub trait ArrowScalarField: Copy + Send + Sync + 'static {
    type ArrowType: ArrowPrimitiveType<Native = Self>;
    const DATA_TYPE: DataType;
    const BYTE_WIDTH: usize = std::mem::size_of::<Self>();

    fn to_le_bytes_vec(values: &[Self], out: &mut Vec<u8>);
    fn from_le_bytes_slice(bytes: &[u8], count: usize, out: &mut Vec<Self>) -> Result<(), StateError>;
    fn values_to_arrow_array(values: Vec<Self>) -> ArrayRef;
    fn arrow_array_to_values(array: &dyn Array) -> Result<Vec<Self>, StateError>;
}

macro_rules! impl_arrow_scalar_field {
    ($rust_ty:ty, $arrow_ty:ty, $array_ty:ty, $data_type:expr) => {
        impl ArrowScalarField for $rust_ty {
            type ArrowType = $arrow_ty;
            const DATA_TYPE: DataType = $data_type;

            #[inline]
            fn to_le_bytes_vec(values: &[Self], out: &mut Vec<u8>) {
                out.reserve(values.len() * Self::BYTE_WIDTH);
                for &v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }

            #[inline]
            fn from_le_bytes_slice(
                bytes: &[u8],
                count: usize,
                out: &mut Vec<Self>,
            ) -> Result<(), StateError> {
                let expected = count.checked_mul(Self::BYTE_WIDTH).ok_or_else(|| {
                    StateError::CodecError("ArrowScalarField byte length overflow".into())
                })?;
                if bytes.len() < expected {
                    return Err(StateError::CodecError(format!(
                        "ArrowScalarField truncated buffer: need {expected} bytes, got {}",
                        bytes.len()
                    )));
                }
                out.reserve(count);
                for chunk in bytes[..expected].chunks_exact(Self::BYTE_WIDTH) {
                    let arr: [u8; std::mem::size_of::<$rust_ty>()] = chunk.try_into().unwrap();
                    out.push(<$rust_ty>::from_le_bytes(arr));
                }
                Ok(())
            }

            #[inline]
            fn values_to_arrow_array(values: Vec<Self>) -> ArrayRef {
                Arc::new(<$array_ty>::from(values))
            }

            #[inline]
            fn arrow_array_to_values(array: &dyn Array) -> Result<Vec<Self>, StateError> {
                let typed = array.as_any().downcast_ref::<$array_ty>().ok_or_else(|| {
                    StateError::CodecError(format!(
                        "Expected Arrow {}, got {:?}",
                        stringify!($array_ty),
                        array.data_type()
                    ))
                })?;
                Ok(typed.values().iter().copied().collect())
            }
        }
    };
}

impl_arrow_scalar_field!(f64, Float64Type, Float64Array, DataType::Float64);
impl_arrow_scalar_field!(f32, Float32Type, Float32Array, DataType::Float32);
impl_arrow_scalar_field!(i64, Int64Type, Int64Array, DataType::Int64);
impl_arrow_scalar_field!(i32, Int32Type, Int32Array, DataType::Int32);
impl_arrow_scalar_field!(i16, Int16Type, Int16Array, DataType::Int16);
impl_arrow_scalar_field!(i8, Int8Type, Int8Array, DataType::Int8);
impl_arrow_scalar_field!(u64, UInt64Type, UInt64Array, DataType::UInt64);
impl_arrow_scalar_field!(u32, UInt32Type, UInt32Array, DataType::UInt32);
impl_arrow_scalar_field!(u16, UInt16Type, UInt16Array, DataType::UInt16);
impl_arrow_scalar_field!(u8, UInt8Type, UInt8Array, DataType::UInt8);

/// 为由定长标量字段组成的领域组件结构体快速实现 [`ArrowColumnarComponent`]。
///
/// 在 `ColumnCodecId::ArrowColumnar = 3` 下，该宏将结构体切片 `&[&[Struct]]` 拆解为按字段连续排列的
/// SoA (Structure-of-Arrays) 小端序字节流，并在 Arrow 内存层映射为标准 `StructArray`（子列均为原生 `PrimitiveArray`）。
#[macro_export]
macro_rules! impl_arrow_struct_component {
    ($struct_ty:ty { $( $field_ident:ident : $field_ty:ty ),+ $(,)? }) => {
        impl $crate::storage::ArrowColumnarComponent for $struct_ty {
            fn arrow_data_type() -> arrow::datatypes::DataType {
                let fields = vec![
                    $(
                        arrow::datatypes::Field::new(
                            stringify!($field_ident),
                            <$field_ty as $crate::storage::ArrowScalarField>::DATA_TYPE,
                            false,
                        ),
                    )+
                ];
                arrow::datatypes::DataType::Struct(arrow::datatypes::Fields::from(fields))
            }

            fn fixed_row_bytes() -> Option<usize> {
                Some(0usize $( + std::mem::size_of::<$field_ty>() )+)
            }

            fn slices_to_arrow_array(
                slices: &[&[Self]],
            ) -> Result<arrow::array::ArrayRef, $crate::error::StateError> {
                let total_rows: usize = slices.iter().map(|s| s.len()).sum();
                $(
                    let mut $field_ident: Vec<$field_ty> = Vec::with_capacity(total_rows);
                )+
                for slice in slices {
                    for item in *slice {
                        $(
                            $field_ident.push(item.$field_ident);
                        )+
                    }
                }
                let fields = vec![
                    $(
                        std::sync::Arc::new(arrow::datatypes::Field::new(
                            stringify!($field_ident),
                            <$field_ty as $crate::storage::ArrowScalarField>::DATA_TYPE,
                            false,
                        )),
                    )+
                ];
                let arrays: Vec<arrow::array::ArrayRef> = vec![
                    $(
                        <$field_ty as $crate::storage::ArrowScalarField>::values_to_arrow_array(
                            $field_ident
                        ),
                    )+
                ];
                let struct_array = arrow::array::StructArray::try_new(
                    arrow::datatypes::Fields::from(fields),
                    arrays,
                    None,
                )
                .map_err(|e| $crate::error::StateError::CodecError(format!("StructArray error: {e}")))?;
                Ok(std::sync::Arc::new(struct_array))
            }

            fn arrow_array_to_vec(
                array: &dyn arrow::array::Array,
                out: &mut Vec<Self>,
            ) -> Result<(), $crate::error::StateError> {
                let struct_arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::StructArray>()
                    .ok_or_else(|| {
                        $crate::error::StateError::CodecError(format!(
                            "Expected Arrow StructArray for {}, got {:?}",
                            stringify!($struct_ty),
                            array.data_type()
                        ))
                    })?;
                let row_count = arrow::array::Array::len(struct_arr);
                $(
                    let $field_ident: Vec<$field_ty> = {
                        let col = struct_arr
                            .column_by_name(stringify!($field_ident))
                            .ok_or_else(|| {
                                $crate::error::StateError::CodecError(format!(
                                    "Missing StructArray field {}",
                                    stringify!($field_ident)
                                ))
                            })?;
                        <$field_ty as $crate::storage::ArrowScalarField>::arrow_array_to_values(
                            col.as_ref(),
                        )?
                    };
                )+
                out.reserve(row_count);
                for idx in 0..row_count {
                    out.push(Self {
                        $(
                            $field_ident: $field_ident[idx],
                        )+
                    });
                }
                Ok(())
            }

            fn encode_arrow_columnar_bytes(
                slices: &[&[Self]],
            ) -> Result<(u32, Vec<u8>), $crate::error::StateError> {
                let total_rows = slices
                    .iter()
                    .try_fold(0usize, |acc, s| acc.checked_add(s.len()))
                    .ok_or_else(|| {
                        $crate::error::StateError::CodecError("Column row count overflow".into())
                    })?;
                let row_count = u32::try_from(total_rows).map_err(|_| {
                    $crate::error::StateError::CodecError("Column row count exceeds u32".into())
                })?;
                let field_count = 0u32 $( + { let _ = stringify!($field_ident); 1u32 } )+;
                let row_bytes = Self::fixed_row_bytes().unwrap_or(0);
                let mut out = Vec::with_capacity(16 + total_rows.saturating_mul(row_bytes));
                $crate::storage::write_arrow_columnar_header(&mut out, row_count, field_count);

                // 按字段 SoA 连续编码，兼顾缓存局部性与列存压缩比
                $(
                    for slice in slices {
                        for item in *slice {
                            out.extend_from_slice(&item.$field_ident.to_le_bytes());
                        }
                    }
                )+
                Ok((row_count, out))
            }

            fn validate_arrow_columnar_bytes(
                row_count: u32,
                payload: &[u8],
            ) -> Result<(), $crate::error::StateError> {
                let field_count = 0u32 $( + { let _ = stringify!($field_ident); 1u32 } )+;
                let body = $crate::storage::parse_arrow_columnar_header(
                    row_count,
                    field_count,
                    payload,
                )?;
                let expected_body_bytes = (row_count as usize)
                    .checked_mul(Self::fixed_row_bytes().unwrap_or(0))
                    .ok_or_else(|| {
                        $crate::error::StateError::CodecError("ArrowColumnar body size overflow".into())
                    })?;
                if body.len() != expected_body_bytes {
                    return Err($crate::error::StateError::CodecError(format!(
                        "ArrowColumnar payload size mismatch: expected {expected_body_bytes} body bytes, got {}",
                        body.len()
                    )));
                }
                Ok(())
            }

            fn decode_arrow_columnar_bytes(
                row_count: u32,
                payload: &[u8],
                out: &mut Vec<Self>,
            ) -> Result<(), $crate::error::StateError> {
                Self::validate_arrow_columnar_bytes(row_count, payload)?;
                let field_count = 0u32 $( + { let _ = stringify!($field_ident); 1u32 } )+;
                let body = $crate::storage::parse_arrow_columnar_header(
                    row_count,
                    field_count,
                    payload,
                )?;
                let count = row_count as usize;
                let mut offset = 0usize;
                $(
                    let mut $field_ident: Vec<$field_ty> = Vec::with_capacity(count);
                    let field_bytes = count * std::mem::size_of::<$field_ty>();
                    <$field_ty as $crate::storage::ArrowScalarField>::from_le_bytes_slice(
                        &body[offset..offset + field_bytes],
                        count,
                        &mut $field_ident,
                    )?;
                    offset += field_bytes;
                )+
                let _ = offset;
                out.reserve(count);
                for idx in 0..count {
                    out.push(Self {
                        $(
                            $field_ident: $field_ident[idx],
                        )+
                    });
                }
                Ok(())
            }
        }
    };
}

impl<T: ArrowColumnarComponent + Serialize + DeserializeOwned> super::ColumnArchiveVTable<T> {
    /// 构造基于 `ColumnCodecId::ArrowColumnar = 3` 的列归档虚表
    pub fn new_arrow_columnar() -> Self {
        Self {
            codec: ColumnCodecId::ArrowColumnar,
            schema_version: 1,
            has_capability: true,
            fixed_archived_row_bytes: T::fixed_row_bytes(),
            archive_raw_slices: Some(arrow_archive_raw_slices::<T>),
            archive_slices: arrow_archive_slices::<T>,
            validate: arrow_validate_column::<T>,
            restore: arrow_restore_column::<T>,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::EntityId;
    use crate::storage::{ColumnarStoreSnapshot, ComponentRegistration, Store};
    use crate::Entity;
    use arrow::datatypes::{Field, Schema};

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
    struct TrainKinematicsComponent {
        pub s_m: f64,
        pub v_mps: f64,
        pub a_mps2: f32,
        pub segment_id: u32,
    }

    impl_arrow_struct_component!(TrainKinematicsComponent {
        s_m: f64,
        v_mps: f64,
        a_mps2: f32,
        segment_id: u32,
    });

    #[test]
    fn zero_copy_column_block_to_arrow_shares_exact_memory_pointer() {
        let raw_data: Vec<f64> = (0..1024).map(|i| i as f64 * 1.25).collect();
        let expected_ptr = raw_data.as_ptr();
        let block = Arc::new(ColumnBlock::new(raw_data));

        // 零拷贝转换为 Arrow Float64Array
        let arrow_array: Float64Array =
            column_block_to_primitive_array_zero_copy::<Float64Type>(Arc::clone(&block));

        assert_eq!(arrow_array.len(), 1024);
        // 核心零拷贝断言：Arrow PrimitiveArray 底层物理指针与 L1 ColumnBlock<f64> 完全一致！
        assert_eq!(
            arrow_array.values().as_ptr(),
            expected_ptr,
            "Arrow Float64Array must share the exact physical memory address with L1 ColumnBlock<f64>"
        );
        assert_eq!(arrow_array.value(0), 0.0);
        assert_eq!(arrow_array.value(1023), 1023.0 * 1.25);

        // 封装进 RecordBatch 后指针依然保持零拷贝
        let schema = Arc::new(Schema::new(vec![Field::new(
            "speed_mps",
            DataType::Float64,
            false,
        )]));
        let batch =
            build_record_batch_from_arrays(schema, vec![Arc::new(arrow_array)]).expect("batch");
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(col.values().as_ptr(), expected_ptr);
    }

    #[test]
    fn arrow_columnar_codec_roundtrips_struct_slices_and_arrow_struct_array() {
        let slice_a = [
            TrainKinematicsComponent {
                s_m: 100.5,
                v_mps: 28.0,
                a_mps2: 0.5,
                segment_id: 1,
            },
            TrainKinematicsComponent {
                s_m: 250.0,
                v_mps: 32.5,
                a_mps2: -0.25,
                segment_id: 2,
            },
        ];
        let slice_b = [TrainKinematicsComponent {
            s_m: 512.75,
            v_mps: 0.0,
            a_mps2: -1.2,
            segment_id: 5,
        }];

        let type_id = ComponentTypeId::of::<TrainKinematicsComponent>();
        let archived =
            arrow_archive_slices(type_id, 1, &[&slice_a, &slice_b]).expect("arrow archive");
        assert_eq!(archived.codec(), ColumnCodecId::ArrowColumnar);
        assert_eq!(archived.row_count(), 3);

        // 1. 校验与恢复 Vec<TrainKinematicsComponent>
        arrow_validate_column::<TrainKinematicsComponent>(type_id, 1, &archived)
            .expect("arrow validate");
        let mut restored = Vec::new();
        arrow_restore_column::<TrainKinematicsComponent>(type_id, 1, &archived, &mut restored)
            .expect("arrow restore");
        assert_eq!(restored.len(), 3);
        assert_eq!(restored[0], slice_a[0]);
        assert_eq!(restored[1], slice_a[1]);
        assert_eq!(restored[2], slice_b[0]);

        // 2. 双向转换 Arrow StructArray <-> Vec<TrainKinematicsComponent>
        let array_ref =
            archived_column_to_typed_arrow_array::<TrainKinematicsComponent>(&archived)
                .expect("to arrow array");
        assert_eq!(array_ref.len(), 3);
        let mut from_arrow = Vec::new();
        TrainKinematicsComponent::arrow_array_to_vec(array_ref.as_ref(), &mut from_arrow)
            .expect("from arrow array");
        assert_eq!(from_arrow, restored);
    }

    #[test]
    fn store_snapshot_and_binary_envelope_roundtrip_with_arrow_columnar_codec() {
        let mut store = Store::new();
        store
            .register_components([ComponentRegistration::of_arrow_columnar::<
                TrainKinematicsComponent,
            >()])
            .expect("register ArrowColumnar prototype");

        let type_id = ComponentTypeId::of::<TrainKinematicsComponent>();
        assert_eq!(
            store.component_codec(type_id),
            Some(ColumnCodecId::ArrowColumnar)
        );

        let e1 = EntityId::from_u128(1001);
        let e2 = EntityId::from_u128(1002);
        let k1 = TrainKinematicsComponent {
            s_m: 10.0,
            v_mps: 5.0,
            a_mps2: 1.0,
            segment_id: 10,
        };
        let k2 = TrainKinematicsComponent {
            s_m: 20.0,
            v_mps: 15.0,
            a_mps2: 0.0,
            segment_id: 11,
        };
        store
            .insert(Entity::new(e1).with(k1))
            .expect("insert e1");
        store
            .insert(Entity::new(e2).with(k2))
            .expect("insert e2");

        // 快照与二进制信封序列化往返
        let snap = store.snapshot().expect("snapshot with ArrowColumnar");
        let bytes = snap.into_binary().expect("snapshot into_binary");

        let mut restored_store = store.empty_with_schema();
        let decoded_snap =
            ColumnarStoreSnapshot::from_binary(&bytes).expect("snapshot from_binary");
        restored_store
            .restore_snapshot(&decoded_snap)
            .expect("restore snapshot");

        let r1 = restored_store
            .entity(e1)
            .expect("e1 present")
            .get::<TrainKinematicsComponent>()
            .expect("get e1");
        let r2 = restored_store
            .entity(e2)
            .expect("e2 present")
            .get::<TrainKinematicsComponent>()
            .expect("get e2");
        assert_eq!(*r1, k1);
        assert_eq!(*r2, k2);
    }
}

