//! 未定型原始字节列 (RawColumn)

use super::any::{
    AnyColumn, ColumnExperimentOps, ColumnLayout, ColumnRows, ColumnSnapshot, ColumnValueOps,
};
use super::archive::{validate_column_header, ArchivedColumnData, ColumnCodecId};
use super::any::ColumnOpsVTable;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use crate::storage::ErasedComponentValue;
use std::any::Any;

/// 类型擦除的未定型原始字节列 (用于无原型反序列化和字节暂存)
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawColumn {
    type_id: ComponentTypeId,
    type_name: &'static str,
    data: Vec<Vec<u8>>,
}

#[allow(dead_code)]
impl RawColumn {
    pub fn new(type_id: ComponentTypeId, data: Vec<Vec<u8>>) -> Self {
        Self {
            type_id,
            type_name: "RawColumn",
            data,
        }
    }

    #[inline]
    pub fn data(&self) -> &[Vec<u8>] {
        &self.data
    }
}

impl ColumnLayout for RawColumn {
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
        None
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
        (std::ptr::null_mut(), self.data.len())
    }

    #[inline]
    unsafe fn raw_parts_read(&self) -> (*const u8, usize) {
        (std::ptr::null(), self.data.len())
    }

    #[inline]
    fn ops_vtable(&self) -> ColumnOpsVTable {
        Self::OPS_VTABLE
    }
}

impl ColumnRows for RawColumn {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    #[inline]
    fn clear_rows(&mut self) {
        self.data.clear();
    }

    #[inline]
    fn swap_remove(&mut self, row: usize) {
        if row < self.data.len() {
            self.data.swap_remove(row);
        }
    }
}

impl ColumnValueOps for RawColumn {
    #[inline]
    fn value_ref(&self, _row: usize) -> Option<&(dyn Any + Send + Sync)> {
        None
    }

    fn clone_erased(&self, _row: usize) -> Result<ErasedComponentValue, StateError> {
        Err(StateError::AccessDenied(
            "Raw columns have no typed component value".into(),
        ))
    }

    fn validate_erased(
        &self,
        _row: usize,
        _value: &ErasedComponentValue,
    ) -> Result<(), StateError> {
        Err(StateError::AccessDenied(
            "Raw columns cannot accept typed component values".into(),
        ))
    }

    fn validate_erased_for_push(&self, _value: &ErasedComponentValue) -> Result<(), StateError> {
        Err(StateError::AccessDenied(
            "Raw snapshot columns cannot accept native component values".into(),
        ))
    }

    fn push_erased_infallible(&mut self, _value: ErasedComponentValue) {
        panic!("raw snapshot columns cannot accept native component values")
    }

    fn replace_erased_infallible(
        &mut self,
        _row: usize,
        _value: ErasedComponentValue,
    ) -> ErasedComponentValue {
        panic!("RawColumn cannot pass validate_erased")
    }

    fn swap_erased_infallible(&mut self, _row: usize, _slot: &mut ErasedComponentValue) {
        panic!("RawColumn cannot pass validate_erased")
    }

    fn swap_remove_erased(&mut self, _row: usize) -> ErasedComponentValue {
        panic!("RawColumn cannot produce typed ErasedComponentValue")
    }
}

impl ColumnSnapshot for RawColumn {
    fn archive_column(&self) -> Result<ArchivedColumnData, StateError> {
        let payload = bincode::serialize(&self.data)
            .map_err(|e| StateError::CodecError(format!("raw column archive failed: {e}")))?;
        Ok(ArchivedColumnData {
            component_type_id: self.type_id,
            schema_version: 1,
            row_count: self.data.len() as u32,
            codec: ColumnCodecId::RawBytes,
            payload,
        })
    }

    fn validate_archived_column(&self, archived: &ArchivedColumnData) -> Result<(), StateError> {
        if archived.component_type_id != self.type_id {
            return Err(StateError::AccessDenied(format!(
                "Column type mismatch: expected {:?}, got {:?}",
                self.type_id, archived.component_type_id
            )));
        }
        if archived.codec != ColumnCodecId::RawBytes {
            return Err(StateError::CodecError(format!(
                "Column codec mismatch: expected {:?}, got {:?}",
                ColumnCodecId::RawBytes,
                archived.codec
            )));
        }
        let restored: Vec<Vec<u8>> = bincode::deserialize(&archived.payload)
            .map_err(|e| StateError::CodecError(format!("raw column validate failed: {e}")))?;
        if restored.len() != archived.row_count as usize {
            return Err(StateError::CodecError(format!(
                "Raw column row count mismatch: header has {}, payload has {}",
                archived.row_count,
                restored.len()
            )));
        }
        Ok(())
    }

    fn restore_archived_column(&mut self, archived: &ArchivedColumnData) -> Result<(), StateError> {
        validate_column_header(
            self.type_id,
            1,
            ColumnCodecId::RawBytes,
            archived,
        )?;
        let restored: Vec<Vec<u8>> = bincode::deserialize(&archived.payload)
            .map_err(|e| StateError::CodecError(format!("raw column restore failed: {e}")))?;
        if restored.len() != archived.row_count as usize {
            return Err(StateError::CodecError(format!(
                "Raw column row count mismatch: header has {}, payload has {}",
                archived.row_count,
                restored.len()
            )));
        }
        self.data = restored;
        Ok(())
    }

    #[inline]
    fn codec(&self) -> ColumnCodecId {
        ColumnCodecId::RawBytes
    }

    #[inline]
    fn schema_version(&self) -> u32 {
        1
    }

    #[inline]
    fn has_archive_capability(&self) -> bool {
        true
    }

    fn serialize_row(&self, row: usize) -> Result<Vec<u8>, StateError> {
        self.data
            .get(row)
            .cloned()
            .ok_or_else(|| StateError::AccessDenied(format!("Row {} not found", row)))
    }

    fn push_deserialized(&mut self, bytes: &[u8]) -> Result<(), StateError> {
        self.data.push(bytes.to_vec());
        Ok(())
    }

    fn set_deserialized(&mut self, row: usize, bytes: &[u8]) -> Result<(), StateError> {
        if let Some(slot) = self.data.get_mut(row) {
            *slot = bytes.to_vec();
            Ok(())
        } else {
            Err(StateError::AccessDenied(format!(
                "Row {} out of bounds for column {}",
                row, self.type_name
            )))
        }
    }

    fn validate_deserialized(&self, row: usize, _bytes: &[u8]) -> Result<(), StateError> {
        if row >= self.data.len() {
            Err(StateError::AccessDenied(format!(
                "Row {} out of bounds for column {}",
                row, self.type_name
            )))
        } else {
            Ok(())
        }
    }

    fn decode_erased(&self, _bytes: &[u8]) -> Result<ErasedComponentValue, StateError> {
        Err(StateError::AccessDenied(
            "Raw columns have no typed decoder".into(),
        ))
    }
}

impl ColumnExperimentOps for RawColumn {}

impl AnyColumn for RawColumn {
    fn clone_box(&self) -> Box<dyn AnyColumn> {
        Box::new(self.clone())
    }

    fn empty_clone(&self) -> Box<dyn AnyColumn> {
        Box::new(Self::new(self.type_id, Vec::new()))
    }

    fn validate_copy_to(&self, src_row: usize, dst: &dyn AnyColumn) -> Result<(), StateError> {
        if src_row >= self.data.len() {
            return Err(StateError::AccessDenied(format!(
                "Row {} out of bounds for column {}",
                src_row, self.type_name
            )));
        }
        if dst.type_id() != self.type_id || !dst.as_any().is::<RawColumn>() {
            return Err(StateError::AccessDenied(format!(
                "Destination column type mismatch: expected {:?}, got {:?}",
                self.type_id,
                dst.type_id()
            )));
        }
        Ok(())
    }

    fn copy_row_to_infallible(&self, src_row: usize, dst: &mut dyn AnyColumn) {
        let bytes = &self.data[src_row];
        let dst_raw = dst
            .as_any_mut()
            .downcast_mut::<RawColumn>()
            .expect("validate_copy_to must be called prior to copy_row_to_infallible");
        dst_raw.data.push(bytes.clone());
    }

    fn move_row_to_infallible(&mut self, src_row: usize, dst: &mut dyn AnyColumn) {
        let bytes = self.data.swap_remove(src_row);
        let dst_raw = dst
            .as_any_mut()
            .downcast_mut::<RawColumn>()
            .expect("validate_copy_to must be called prior to move_row_to_infallible");
        dst_raw.data.push(bytes);
    }
}

impl RawColumn {
    const OPS_VTABLE: ColumnOpsVTable = ColumnOpsVTable {
        assign_raw: Self::assign_raw_panic,
        replace_raw: Self::replace_raw_panic,
        new_typed_owned_block: Self::new_typed_owned_block_panic,
    };

    unsafe fn assign_raw_panic(_raw_ptr: *mut u8, _row: usize, _value: ErasedComponentValue) {
        panic!("RawColumn does not support typed raw assign");
    }

    unsafe fn replace_raw_panic(_raw_ptr: *mut u8, _row: usize, _value: ErasedComponentValue) -> ErasedComponentValue {
        panic!("RawColumn does not support typed raw replace");
    }

    fn new_typed_owned_block_panic(
        _component: ComponentTypeId,
        _capacity: usize,
    ) -> std::sync::Arc<dyn crate::storage::column::ErasedBlock> {
        panic!("RawColumn does not support TypedOwnedBlock");
    }
}
