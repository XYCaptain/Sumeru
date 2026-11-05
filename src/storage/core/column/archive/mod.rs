//! 列快照归档、版本控制与 rkyv 编解码模型

use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub mod arrow_bridge;
pub use arrow_bridge::{
    archived_column_to_opaque_binary_array, archived_column_to_typed_arrow_array,
    arrow_archive_slices, arrow_restore_column, arrow_validate_column,
    build_record_batch_from_arrays, column_block_to_arrow_buffer_zero_copy,
    column_block_to_primitive_array_zero_copy, column_block_to_scalar_buffer_zero_copy,
    parse_arrow_columnar_header, write_arrow_columnar_header, ArrowColumnarComponent,
    ArrowScalarField,
};

/// 列编码标识
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ColumnCodecId {
    /// 纯内存列：无快照归档能力
    None = 0,
    /// 高性能零拷贝编码：rkyv 连续整列归档
    Rkyv = 1,
    /// 原始字节列编码：用于未定型/RawColumn 内部暂存
    RawBytes = 2,
    /// Arrow 标准列存编码：支持 L1 ColumnBlock 零拷贝直通 Arrow Buffer / RecordBatch 与列存恢复
    ArrowColumnar = 3,
}

/// 统一列快照归档数据载荷 (ArchivedColumnData)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchivedColumnData {
    pub component_type_id: ComponentTypeId,
    pub schema_version: u32,
    pub row_count: u32,
    pub codec: ColumnCodecId,
    #[serde(with = "byte_vec_serde")]
    pub payload: Vec<u8>,
}

mod byte_vec_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    const COMPACT_COL_MAGIC: &[u8; 8] = b"\xFDENSCOL1";
    const RAW_ESCAPE_MAGIC: &[u8; 8] = b"\xFDENSCOL0";
    const TAG_STRIDE: u8 = 0x01;
    const TAG_LITERAL: u8 = 0x02;
    const MAX_COMPACT_WORDS: usize = 33_554_432;

    #[inline(always)]
    fn word_at(bytes: &[u8], idx: usize) -> u64 {
        let off = idx * 8;
        u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
    }

    fn try_compress_words(bytes: &[u8]) -> Option<Vec<u8>> {
        if bytes.len() < 128 || bytes.len() % 8 != 0 {
            return None;
        }
        let word_count = bytes.len() / 8;
        if word_count > MAX_COMPACT_WORDS {
            return None;
        }
        let max_compressed_len = bytes.len().saturating_sub(16);
        let mut out = Vec::with_capacity(64.min(bytes.len()));
        out.extend_from_slice(COMPACT_COL_MAGIC);
        out.extend_from_slice(&(word_count as u32).to_le_bytes());
        out.extend_from_slice(&bytes[0..8]);

        let mut idx = 1usize;
        while idx < word_count {
            let prev = word_at(bytes, idx - 1);
            let curr = word_at(bytes, idx);
            let stride = curr.wrapping_sub(prev);
            let mut run_len = 1usize;
            while idx + run_len < word_count {
                let w_prev = word_at(bytes, idx + run_len - 1);
                let w_curr = word_at(bytes, idx + run_len);
                if w_curr.wrapping_sub(w_prev) == stride {
                    run_len += 1;
                } else {
                    break;
                }
            }
            if run_len >= 2 {
                if out.len() + 13 > max_compressed_len {
                    return None;
                }
                out.push(TAG_STRIDE);
                out.extend_from_slice(&(run_len as u32).to_le_bytes());
                out.extend_from_slice(&stride.to_le_bytes());
                idx += run_len;
            } else {
                let mut lit_end = idx + 1;
                while lit_end < word_count {
                    if lit_end + 1 < word_count {
                        let w0 = word_at(bytes, lit_end - 1);
                        let w1 = word_at(bytes, lit_end);
                        let w2 = word_at(bytes, lit_end + 1);
                        if w1.wrapping_sub(w0) == w2.wrapping_sub(w1) {
                            break;
                        }
                    }
                    lit_end += 1;
                }
                let lit_len = lit_end - idx;
                if out.len() + 5 + lit_len * 8 > max_compressed_len {
                    return None;
                }
                out.push(TAG_LITERAL);
                out.extend_from_slice(&(lit_len as u32).to_le_bytes());
                out.extend_from_slice(&bytes[idx * 8..lit_end * 8]);
                idx = lit_end;
            }
        }
        Some(out)
    }

    fn decode_compact_bytes<E: serde::de::Error>(v: &[u8]) -> Result<Vec<u8>, E> {
        if v.len() < 20 {
            return Err(E::custom("Invalid compact column header"));
        }
        let word_count = u32::from_le_bytes(v[8..12].try_into().unwrap()) as usize;
        if !(16..=MAX_COMPACT_WORDS).contains(&word_count) {
            return Err(E::custom("Invalid compact column word count"));
        }
        let mut out = Vec::with_capacity(word_count * 8);
        out.extend_from_slice(&v[12..20]);
        let mut current = u64::from_le_bytes(v[12..20].try_into().unwrap());
        let mut emitted = 1usize;
        let mut pos = 20usize;

        while pos < v.len() {
            let tag = v[pos];
            pos += 1;
            if pos + 4 > v.len() {
                return Err(E::custom("Truncated compact column segment header"));
            }
            let seg_len = u32::from_le_bytes(v[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if seg_len == 0 || emitted + seg_len > word_count {
                return Err(E::custom("Compact column segment length overflow"));
            }
            match tag {
                TAG_STRIDE => {
                    if pos + 8 > v.len() {
                        return Err(E::custom("Truncated compact column stride"));
                    }
                    let stride = u64::from_le_bytes(v[pos..pos + 8].try_into().unwrap());
                    pos += 8;
                    for _ in 0..seg_len {
                        current = current.wrapping_add(stride);
                        out.extend_from_slice(&current.to_le_bytes());
                    }
                    emitted += seg_len;
                }
                TAG_LITERAL => {
                    let byte_len = seg_len * 8;
                    if pos + byte_len > v.len() {
                        return Err(E::custom("Truncated compact column literal"));
                    }
                    out.extend_from_slice(&v[pos..pos + byte_len]);
                    current = u64::from_le_bytes(
                        v[pos + byte_len - 8..pos + byte_len].try_into().unwrap(),
                    );
                    pos += byte_len;
                    emitted += seg_len;
                }
                _ => return Err(E::custom("Unknown compact column segment tag")),
            }
        }
        if emitted != word_count {
            return Err(E::custom("Compact column word count mismatch"));
        }
        Ok(out)
    }

    pub(super) fn serialize<S: Serializer>(
        bytes: &Vec<u8>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            bytes.serialize(serializer)
        } else if let Some(compressed) = try_compress_words(bytes) {
            serializer.serialize_bytes(&compressed)
        } else if bytes.starts_with(b"\xFDENSCOL") {
            let mut escaped = Vec::with_capacity(8 + bytes.len());
            escaped.extend_from_slice(RAW_ESCAPE_MAGIC);
            escaped.extend_from_slice(bytes);
            serializer.serialize_bytes(&escaped)
        } else {
            serializer.serialize_bytes(bytes)
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<u8>, D::Error> {
        if deserializer.is_human_readable() {
            Vec::<u8>::deserialize(deserializer)
        } else {
            struct ByteVecVisitor;
            impl<'de> serde::de::Visitor<'de> for ByteVecVisitor {
                type Value = Vec<u8>;
                fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str("a byte buffer")
                }
                fn visit_byte_buf<E: serde::de::Error>(
                    self,
                    v: Vec<u8>,
                ) -> Result<Self::Value, E> {
                    if v.starts_with(RAW_ESCAPE_MAGIC) {
                        Ok(v[8..].to_vec())
                    } else if v.starts_with(COMPACT_COL_MAGIC) {
                        decode_compact_bytes(&v)
                    } else {
                        Ok(v)
                    }
                }
                fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                    if v.starts_with(RAW_ESCAPE_MAGIC) {
                        Ok(v[8..].to_vec())
                    } else if v.starts_with(COMPACT_COL_MAGIC) {
                        decode_compact_bytes(v)
                    } else {
                        Ok(v.to_vec())
                    }
                }
                fn visit_seq<A: serde::de::SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> Result<Self::Value, A::Error> {
                    let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(4096));
                    while let Some(b) = seq.next_element()? {
                        out.push(b);
                    }
                    if out.starts_with(RAW_ESCAPE_MAGIC) {
                        Ok(out[8..].to_vec())
                    } else if out.starts_with(COMPACT_COL_MAGIC) {
                        decode_compact_bytes(&out)
                    } else {
                        Ok(out)
                    }
                }
            }
            deserializer.deserialize_byte_buf(ByteVecVisitor)
        }
    }
}

impl ArchivedColumnData {
    pub fn new(
        component_type_id: ComponentTypeId,
        schema_version: u32,
        row_count: u32,
        codec: ColumnCodecId,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            component_type_id,
            schema_version,
            row_count,
            codec,
            payload,
        }
    }

    #[inline]
    pub fn component_type_id(&self) -> ComponentTypeId {
        self.component_type_id
    }

    #[inline]
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[inline]
    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    #[inline]
    pub fn codec(&self) -> ColumnCodecId {
        self.codec
    }

    #[inline]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

/// rkyv 0.8 默认高阶序列化器类型别名
pub type RkyvDefaultSerializer<'a> = rkyv::api::high::HighSerializer<
    rkyv::util::AlignedVec,
    rkyv::ser::allocator::ArenaHandle<'a>,
    rkyv::rancor::BoxedError,
>;

/// rkyv 0.8 默认高阶反序列化器类型别名
pub type RkyvDefaultDeserializer = rkyv::api::high::HighDeserializer<rkyv::rancor::BoxedError>;

/// rkyv 0.8 默认高阶校验器类型别名
pub type RkyvDefaultValidator<'a> = rkyv::api::high::HighValidator<'a, rkyv::rancor::BoxedError>;

// Match the default HighSerializer AlignedVec. Checking only the ArchivedVec
// root (typically alignment 4) misses nested archived fields requiring 8/16.
const ARCHIVE_BUFFER_ALIGNMENT: usize = 16;

/// Private metadata only. The caller separately proves type, physical bounds,
/// lifetime and compatible access permission for each slice.
#[derive(Clone, Copy)]
pub(crate) struct RawArchiveSlice {
    pub base: *const u8,
    pub offset: usize,
    pub len: usize,
}

pub(crate) type ArchiveRawSlices =
    unsafe fn(ComponentTypeId, u32, &[RawArchiveSlice]) -> Result<ArchivedColumnData, StateError>;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RawColumnArchive {
    pub component: ComponentTypeId,
    pub schema_version: u32,
    pub rust_type: std::any::TypeId,
    pub fixed_archived_row_bytes: Option<usize>,
    pub encode: ArchiveRawSlices,
}

/// 列归档操作虚表 (ColumnArchiveVTable)
///
/// 纯函数指针表，类型擦除列持有它以在不特判任何业务类型的情况下执行整列归档、校验与恢复。
pub(crate) struct ColumnArchiveVTable<T> {
    pub codec: ColumnCodecId,
    pub schema_version: u32,
    pub has_capability: bool,
    pub fixed_archived_row_bytes: Option<usize>,
    pub archive_slices: fn(ComponentTypeId, u32, &[&[T]]) -> Result<ArchivedColumnData, StateError>,
    pub archive_raw_slices: Option<ArchiveRawSlices>,
    pub validate: fn(ComponentTypeId, u32, &ArchivedColumnData) -> Result<(), StateError>,
    pub restore:
        fn(ComponentTypeId, u32, &ArchivedColumnData, &mut Vec<T>) -> Result<(), StateError>,
}

impl<T> Clone for ColumnArchiveVTable<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ColumnArchiveVTable<T> {}

impl<T> std::fmt::Debug for ColumnArchiveVTable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnArchiveVTable")
            .field("codec", &self.codec)
            .field("schema_version", &self.schema_version)
            .field("has_capability", &self.has_capability)
            .finish()
    }
}

impl<T> PartialEq for ColumnArchiveVTable<T> {
    fn eq(&self, other: &Self) -> bool {
        self.codec == other.codec
            && self.schema_version == other.schema_version
            && self.has_capability == other.has_capability
            && self.archive_slices as usize == other.archive_slices as usize
            && self.archive_raw_slices.map(|f| f as usize)
                == other.archive_raw_slices.map(|f| f as usize)
            && self.validate as usize == other.validate as usize
            && self.restore as usize == other.restore as usize
    }
}

impl<T> Eq for ColumnArchiveVTable<T> {}

/// Borrowed logical rows, preserving physical slice boundaries. Cloning this
/// iterator copies only cursor metadata, never a component or its allocation.
struct SliceRows<'view, 'data, T> {
    slices: std::slice::Iter<'view, &'data [T]>,
    current: std::slice::Iter<'data, T>,
    remaining: usize,
}

impl<T> Clone for SliceRows<'_, '_, T> {
    fn clone(&self) -> Self {
        Self {
            slices: self.slices.clone(),
            current: self.current.clone(),
            remaining: self.remaining,
        }
    }
}

impl<'view, 'data, T> SliceRows<'view, 'data, T> {
    fn new(slices: &'view [&'data [T]], len: usize) -> Self {
        Self {
            slices: slices.iter(),
            current: [].iter(),
            remaining: len,
        }
    }
}

impl<'data, T> Iterator for SliceRows<'_, 'data, T> {
    type Item = &'data T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(row) = self.current.next() {
                self.remaining -= 1;
                return Some(row);
            }
            self.current = self.slices.next()?.iter();
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for SliceRows<'_, '_, T> {}

struct ColumnSlices<'view, 'data, T> {
    slices: &'view [&'data [T]],
    len: usize,
}

impl<T: rkyv::Archive> rkyv::Archive for ColumnSlices<'_, '_, T> {
    // Deliberately identical to Vec<T>, so existing column readers need no
    // format/version change. Only the source representation differs.
    type Archived = rkyv::vec::ArchivedVec<rkyv::Archived<T>>;
    type Resolver = rkyv::vec::VecResolver;

    fn resolve(&self, resolver: Self::Resolver, out: rkyv::Place<Self::Archived>) {
        Self::Archived::resolve_from_len(self.len, resolver, out);
    }
}

impl<T, S> rkyv::Serialize<S> for ColumnSlices<'_, '_, T>
where
    T: rkyv::Serialize<S>,
    S: rkyv::rancor::Fallible + rkyv::ser::Allocator + rkyv::ser::Writer + ?Sized,
{
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        if let [slice] = self.slices {
            Self::Archived::serialize_from_slice(slice, serializer)
        } else {
            // Flatten<...> is not ExactSizeIterator. The checked, exact cursor
            // supports nested payloads too (unlike unknown_length_iter).
            Self::Archived::serialize_from_iter::<T, _, _>(
                SliceRows::new(self.slices, self.len),
                serializer,
            )
        }
    }
}

fn checked_row_count(mut lengths: impl Iterator<Item = usize>) -> Result<u32, StateError> {
    let len = lengths
        .try_fold(0usize, |sum, len| sum.checked_add(len))
        .ok_or_else(|| StateError::CodecError("Column row count overflow".into()))?;
    u32::try_from(len).map_err(|_| StateError::CodecError("Column row count exceeds u32".into()))
}

/// Archive real physical slices as one logical column without cloning T or
/// materializing a Vec<T>. The caller supplies valid, protected shared slices;
/// this safe helper neither resolves raw pointers nor changes Current.
pub(crate) fn rkyv_archive_slices<T>(
    type_id: ComponentTypeId,
    schema_version: u32,
    slices: &[&[T]],
) -> Result<ArchivedColumnData, StateError>
where
    T: for<'a> rkyv::Serialize<RkyvDefaultSerializer<'a>>,
{
    let row_count = checked_row_count(slices.iter().map(|slice| slice.len()))?;
    let column = ColumnSlices {
        slices,
        len: row_count as usize,
    };
    let estimated_bytes = (row_count as usize)
        .saturating_mul(std::mem::size_of::<rkyv::Archived<T>>())
        .saturating_add(32);
    let aligned = rkyv::api::high::to_bytes_in::<_, rkyv::rancor::BoxedError>(
        &column,
        rkyv::util::AlignedVec::with_capacity(estimated_bytes),
    )
    .map_err(|e| StateError::CodecError(format!("rkyv archive failed: {e}")))?;
    Ok(ArchivedColumnData {
        component_type_id: type_id,
        schema_version,
        row_count,
        codec: ColumnCodecId::Rkyv,
        payload: aligned.into_vec(),
    })
}

/// # Safety
/// Every range must be valid shared T data within ONE physical allocation,
/// pinned and protected from mutation for this call. In particular, a logical
/// extent crossing descriptor segments must already have been split. Never
/// recreate a parent Vec/column reference to satisfy these requirements.
unsafe fn rkyv_archive_raw_slices<T>(
    type_id: ComponentTypeId,
    schema_version: u32,
    raw: &[RawArchiveSlice],
) -> Result<ArchivedColumnData, StateError>
where
    T: for<'a> rkyv::Serialize<RkyvDefaultSerializer<'a>>,
{
    let slices: smallvec::SmallVec<[&[T]; 4]> = raw
        .iter()
        .map(|slice| {
            // SAFETY: the private caller proves the typed, protected physical span.
            unsafe {
                std::slice::from_raw_parts(slice.base.cast::<T>().add(slice.offset), slice.len)
            }
        })
        .collect();
    rkyv_archive_slices(type_id, schema_version, &slices)
}

/// 对泛型 `Vec<T>` 归档列数据执行元数据与二进制访问校验
pub fn rkyv_validate_column<T>(
    type_id: ComponentTypeId,
    schema_version: u32,
    archived: &ArchivedColumnData,
) -> Result<(), StateError>
where
    T: DeserializeOwned + rkyv::Archive,
    Vec<T>: rkyv::Archive<Archived = rkyv::vec::ArchivedVec<rkyv::Archived<T>>>,
    rkyv::vec::ArchivedVec<rkyv::Archived<T>>:
        rkyv::Portable + for<'a> rkyv::bytecheck::CheckBytes<RkyvDefaultValidator<'a>>,
{
    validate_column_header(type_id, schema_version, ColumnCodecId::Rkyv, archived)?;
    let align = ARCHIVE_BUFFER_ALIGNMENT;
    let len = if (archived.payload.as_ptr() as usize) % align == 0 {
        let archived_vec = rkyv::access::<
            rkyv::vec::ArchivedVec<rkyv::Archived<T>>,
            rkyv::rancor::BoxedError,
        >(&archived.payload)
        .map_err(|e| StateError::CodecError(format!("rkyv validate failed: {e}")))?;
        archived_vec.len()
    } else {
        let mut aligned = rkyv::util::AlignedVec::<ARCHIVE_BUFFER_ALIGNMENT>::new();
        aligned.extend_from_slice(&archived.payload);
        let archived_vec = rkyv::access::<
            rkyv::vec::ArchivedVec<rkyv::Archived<T>>,
            rkyv::rancor::BoxedError,
        >(&aligned)
        .map_err(|e| StateError::CodecError(format!("rkyv validate failed: {e}")))?;
        archived_vec.len()
    };
    if len != archived.row_count as usize {
        return Err(StateError::CodecError(format!(
            "Archived row count mismatch: header has {}, payload has {}",
            archived.row_count, len
        )));
    }
    Ok(())
}

/// 校验归档列数据的轻量头部元数据（类型 ID、模式版本、编解码标识），零反序列化开销
pub fn validate_column_header(
    expected_type_id: ComponentTypeId,
    expected_schema_version: u32,
    expected_codec: ColumnCodecId,
    archived: &ArchivedColumnData,
) -> Result<(), StateError> {
    if archived.component_type_id != expected_type_id {
        return Err(StateError::AccessDenied(format!(
            "Column type mismatch: expected {expected_type_id}, got {}",
            archived.component_type_id
        )));
    }
    if archived.schema_version != expected_schema_version {
        return Err(StateError::CodecError(format!(
            "Schema version mismatch: expected {expected_schema_version}, got {}",
            archived.schema_version
        )));
    }
    if archived.codec != expected_codec {
        return Err(StateError::CodecError(format!(
            "Column codec mismatch: expected {:?}, got {:?}",
            expected_codec, archived.codec
        )));
    }
    Ok(())
}

/// 对泛型 `Vec<T>` 连续切片执行受检反序列化恢复（含指针物理对齐自愈与行数一致性刚性校验）
pub fn rkyv_restore_column_from_slice<T>(
    _type_id: ComponentTypeId,
    _schema_version: u32,
    codec: ColumnCodecId,
    row_count: u32,
    payload: &[u8],
    target: &mut Vec<T>,
) -> Result<(), StateError>
where
    T: DeserializeOwned + rkyv::Archive,
    Vec<T>: rkyv::Archive<Archived = rkyv::vec::ArchivedVec<rkyv::Archived<T>>>,
    rkyv::vec::ArchivedVec<rkyv::Archived<T>>: rkyv::Portable
        + for<'a> rkyv::bytecheck::CheckBytes<RkyvDefaultValidator<'a>>
        + rkyv::Deserialize<Vec<T>, RkyvDefaultDeserializer>,
{
    if codec != ColumnCodecId::Rkyv {
        return Err(StateError::CodecError(format!(
            "Column codec mismatch: expected Rkyv, got {:?}",
            codec
        )));
    }
    let align = ARCHIVE_BUFFER_ALIGNMENT;
    let restored: Vec<T> = if (payload.as_ptr() as usize) % align == 0 {
        rkyv::from_bytes::<Vec<T>, rkyv::rancor::BoxedError>(payload)
            .map_err(|e| StateError::CodecError(format!("rkyv restore failed: {e}")))?
    } else {
        let mut aligned = rkyv::util::AlignedVec::<ARCHIVE_BUFFER_ALIGNMENT>::new();
        aligned.extend_from_slice(payload);
        rkyv::from_bytes::<Vec<T>, rkyv::rancor::BoxedError>(&aligned)
            .map_err(|e| StateError::CodecError(format!("rkyv restore failed: {e}")))?
    };

    if restored.len() != row_count as usize {
        return Err(StateError::AccessDenied(format!(
            "Restored column length {} does not match header row_count {}",
            restored.len(),
            row_count
        )));
    }

    *target = restored;
    Ok(())
}

/// 对泛型 `Vec<T>` 归档列数据执行单次受检反序列化恢复
pub fn rkyv_restore_column<T>(
    type_id: ComponentTypeId,
    schema_version: u32,
    archived: &ArchivedColumnData,
    target: &mut Vec<T>,
) -> Result<(), StateError>
where
    T: DeserializeOwned + rkyv::Archive,
    Vec<T>: rkyv::Archive<Archived = rkyv::vec::ArchivedVec<rkyv::Archived<T>>>,
    rkyv::vec::ArchivedVec<rkyv::Archived<T>>: rkyv::Portable
        + for<'a> rkyv::bytecheck::CheckBytes<RkyvDefaultValidator<'a>>
        + rkyv::Deserialize<Vec<T>, RkyvDefaultDeserializer>,
{
    validate_column_header(type_id, schema_version, ColumnCodecId::Rkyv, archived)?;
    rkyv_restore_column_from_slice(
        type_id,
        schema_version,
        archived.codec,
        archived.row_count,
        &archived.payload,
        target,
    )
}

impl<T: ModelComponent + Serialize + DeserializeOwned> ColumnArchiveVTable<T> {
    pub fn new_unsupported() -> Self {
        Self {
            codec: ColumnCodecId::None,
            schema_version: 1,
            has_capability: false,
            fixed_archived_row_bytes: None,
            archive_raw_slices: None,
            archive_slices: |type_id, _, _| {
                Err(StateError::AccessDenied(format!(
                    "Component prototype {type_id} has no registered archive capability"
                )))
            },
            validate: |type_id, _, _| {
                Err(StateError::AccessDenied(format!(
                    "Component prototype {type_id} has no registered archive capability"
                )))
            },
            restore: |type_id, _, _, _| {
                Err(StateError::AccessDenied(format!(
                    "Component prototype {type_id} has no registered archive capability"
                )))
            },
        }
    }

    pub fn new_rkyv() -> Self
    where
        T: for<'a> rkyv::Serialize<RkyvDefaultSerializer<'a>>,
        Vec<T>: for<'a> rkyv::Serialize<RkyvDefaultSerializer<'a>>
            + rkyv::Archive<Archived = rkyv::vec::ArchivedVec<rkyv::Archived<T>>>,
        rkyv::vec::ArchivedVec<rkyv::Archived<T>>: rkyv::Portable
            + for<'a> rkyv::bytecheck::CheckBytes<RkyvDefaultValidator<'a>>
            + rkyv::Deserialize<Vec<T>, RkyvDefaultDeserializer>,
    {
        Self {
            codec: ColumnCodecId::Rkyv,
            schema_version: 1,
            has_capability: true,
            fixed_archived_row_bytes: (!std::mem::needs_drop::<T>())
                .then_some(std::mem::size_of::<rkyv::Archived<T>>().max(std::mem::size_of::<T>())),
            archive_raw_slices: Some(rkyv_archive_raw_slices::<T>),
            archive_slices: rkyv_archive_slices::<T>,
            validate: rkyv_validate_column::<T>,
            restore: rkyv_restore_column::<T>,
        }
    }
}

#[cfg(test)]
mod slice_tests;

#[cfg(test)]
mod compact_serde_tests {
    use super::*;

    #[test]
    fn compact_column_serde_compresses_50k_strided_and_equal_words() {
        let mut payload = Vec::with_capacity(50_001 * 8);
        for i in 0..50_000u64 {
            payload.extend_from_slice(&(1000 + i * 7).to_le_bytes());
        }
        // Simulate rkyv ArchivedVec trailer word
        payload.extend_from_slice(&0x0000_c350_fff9_e578u64.to_le_bytes());

        let col = ArchivedColumnData::new(
            ComponentTypeId::new(42),
            1,
            50_000,
            ColumnCodecId::Rkyv,
            payload.clone(),
        );
        let encoded = bincode::serialize(&col).expect("bincode serialize");
        assert!(
            encoded.len() < 128,
            "50k strided words should compress to < 128B, got {}",
            encoded.len()
        );
        let decoded: ArchivedColumnData =
            bincode::deserialize(&encoded).expect("bincode deserialize");
        assert_eq!(decoded, col);
    }

    #[test]
    fn compact_column_serde_roundtrips_mixed_runs_and_escaped_magic() {
        let mut payload = Vec::new();
        for i in 0..32u64 {
            payload.extend_from_slice(&(i * 10).to_le_bytes());
        }
        payload.extend_from_slice(&0xdead_beef_cafe_babeu64.to_le_bytes());
        payload.extend_from_slice(&0x0123_4567_89ab_cdefu64.to_le_bytes());
        for _ in 0..32u64 {
            payload.extend_from_slice(&999u64.to_le_bytes());
        }
        let col = ArchivedColumnData::new(
            ComponentTypeId::new(7),
            1,
            66,
            ColumnCodecId::Rkyv,
            payload,
        );
        let encoded = bincode::serialize(&col).expect("serialize mixed");
        let decoded: ArchivedColumnData = bincode::deserialize(&encoded).expect("deserialize mixed");
        assert_eq!(decoded, col);

        // Raw escape test: incompressible payload starting with \xFDENSCOL1
        let mut raw_magic_payload = b"\xFDENSCOL1".to_vec();
        for i in 0..25u64 {
            raw_magic_payload.extend_from_slice(&(i * i * 131 + 17).to_le_bytes());
        }
        let raw_col = ArchivedColumnData::new(
            ComponentTypeId::new(8),
            1,
            26,
            ColumnCodecId::RawBytes,
            raw_magic_payload,
        );
        let raw_encoded = bincode::serialize(&raw_col).expect("serialize escaped");
        let raw_decoded: ArchivedColumnData =
            bincode::deserialize(&raw_encoded).expect("deserialize escaped");
        assert_eq!(raw_decoded, raw_col);
    }

    #[test]
    fn compact_column_serde_boundaries_wrapping_stride_and_corrupted_rejection() {
        // 1. Boundary < 128 bytes (e.g. 24 bytes starting with RAW_ESCAPE_MAGIC b"\xFDENSCOL0")
        // and 131 bytes (>= 128 bytes, not divisible by 8, starting with COMPACT_MAGIC b"\xFDENSCOL1")
        for payload in [
            b"\xFDENSCOL0_short_escape_24".to_vec(),
            b"\xFDENSCOL1_short_magic_24_".to_vec(),
            {
                let mut v = b"\xFDENSCOL1".to_vec();
                v.extend((0..123u8).map(|b| b.wrapping_mul(37)));
                assert_eq!(v.len(), 131);
                v
            },
            {
                // Exactly 128 bytes (16 u64 words), incompressible quadratic sequence
                let mut v = Vec::with_capacity(128);
                for i in 0..16u64 {
                    v.extend_from_slice(&(i * i * 9973 + i * 31 + 7).to_le_bytes());
                }
                v
            },
        ] {
            let col = ArchivedColumnData::new(
                ComponentTypeId::new(11),
                1,
                16,
                ColumnCodecId::RawBytes,
                payload,
            );
            let bytes = bincode::serialize(&col).unwrap();
            let decoded: ArchivedColumnData = bincode::deserialize(&bytes).unwrap();
            assert_eq!(decoded, col);
        }

        // 2. Negative wrapping stride (u64::MAX, i.e. -1 mod 2^64) across 64 words
        let mut decreasing = Vec::with_capacity(64 * 8);
        let mut val = 10_000u64;
        for _ in 0..64 {
            decreasing.extend_from_slice(&val.to_le_bytes());
            val = val.wrapping_sub(1);
        }
        let dec_col = ArchivedColumnData::new(
            ComponentTypeId::new(12),
            1,
            64,
            ColumnCodecId::Rkyv,
            decreasing,
        );
        let dec_bytes = bincode::serialize(&dec_col).unwrap();
        assert!(dec_bytes.len() < 128);
        let dec_decoded: ArchivedColumnData = bincode::deserialize(&dec_bytes).unwrap();
        assert_eq!(dec_decoded, dec_col);

        // 3. Corrupted COMPACT_MAGIC payloads must return Err on deserialize
        #[derive(Serialize)]
        struct FakeArchivedColumnData {
            component_type_id: ComponentTypeId,
            schema_version: u32,
            row_count: u32,
            codec: ColumnCodecId,
            payload: Vec<u8>,
        }

        // (a) Truncated compact header (< 12 bytes)
        let truncated_header = FakeArchivedColumnData {
            component_type_id: ComponentTypeId::new(1),
            schema_version: 1,
            row_count: 1,
            codec: ColumnCodecId::Rkyv,
            payload: b"\xFDENSCOL1\x01\x02".to_vec(),
        };
        assert!(
            bincode::deserialize::<ArchivedColumnData>(
                &bincode::serialize(&truncated_header).unwrap()
            )
            .is_err()
        );

        // (b) Unknown opcode tag (0x99)
        let mut bad_opcode = b"\xFDENSCOL1".to_vec();
        bad_opcode.extend_from_slice(&4u32.to_le_bytes());
        bad_opcode.push(0x99);
        let unknown_op = FakeArchivedColumnData {
            component_type_id: ComponentTypeId::new(1),
            schema_version: 1,
            row_count: 4,
            codec: ColumnCodecId::Rkyv,
            payload: bad_opcode,
        };
        assert!(
            bincode::deserialize::<ArchivedColumnData>(&bincode::serialize(&unknown_op).unwrap())
                .is_err()
        );

        // (c) Word count mismatch (header claims 10 words, OP_STRIDED_RUN provides only 4)
        let mut count_mismatch = b"\xFDENSCOL1".to_vec();
        count_mismatch.extend_from_slice(&10u32.to_le_bytes());
        count_mismatch.push(1); // OP_STRIDED_RUN
        count_mismatch.extend_from_slice(&4u32.to_le_bytes());
        count_mismatch.extend_from_slice(&100u64.to_le_bytes());
        count_mismatch.extend_from_slice(&1u64.to_le_bytes());
        let mismatch_col = FakeArchivedColumnData {
            component_type_id: ComponentTypeId::new(1),
            schema_version: 1,
            row_count: 10,
            codec: ColumnCodecId::Rkyv,
            payload: count_mismatch,
        };
        assert!(
            bincode::deserialize::<ArchivedColumnData>(&bincode::serialize(&mismatch_col).unwrap())
                .is_err()
        );

        // (d) Truncated OP_LITERAL_WORDS body (claims 4 literal words = 32 bytes, only 8 bytes present)
        let mut truncated_lit = b"\xFDENSCOL1".to_vec();
        truncated_lit.extend_from_slice(&4u32.to_le_bytes());
        truncated_lit.push(2); // OP_LITERAL_WORDS
        truncated_lit.extend_from_slice(&4u32.to_le_bytes());
        truncated_lit.extend_from_slice(&42u64.to_le_bytes());
        let trunc_lit_col = FakeArchivedColumnData {
            component_type_id: ComponentTypeId::new(1),
            schema_version: 1,
            row_count: 4,
            codec: ColumnCodecId::Rkyv,
            payload: truncated_lit,
        };
        assert!(
            bincode::deserialize::<ArchivedColumnData>(
                &bincode::serialize(&trunc_lit_col).unwrap()
            )
            .is_err()
        );
    }
}

