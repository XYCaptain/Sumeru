use super::*;
use crate::storage::core::column::archive;

impl<T: ModelComponent + Serialize + DeserializeOwned> ColumnValueOps for TypedColumn<T> {
    #[inline]
    fn value_ref(&self, row: usize) -> Option<&(dyn Any + Send + Sync)> {
        self.get(row).map(|value| value as &(dyn Any + Send + Sync))
    }

    fn clone_erased(&self, row: usize) -> Result<ErasedComponentValue, StateError> {
        let value = self.get(row).ok_or_else(|| {
            StateError::AccessDenied(format!(
                "Row {row} out of bounds for column {}",
                self.type_name
            ))
        })?;
        Ok(ErasedComponentValue::with_component_type_id(
            self.type_id,
            value.clone(),
        ))
    }

    fn validate_erased(&self, row: usize, value: &ErasedComponentValue) -> Result<(), StateError> {
        if row >= self.backing.len() {
            return Err(StateError::AccessDenied(format!(
                "Row {row} out of bounds for column {}",
                self.type_name
            )));
        }
        self.validate_erased_for_push(value)
    }

    fn validate_erased_for_push(&self, value: &ErasedComponentValue) -> Result<(), StateError> {
        if value.component_type_id() != self.type_id
            || value.rust_type_id() != std::any::TypeId::of::<T>()
            || self.type_name != std::any::type_name::<T>()
            || value.type_name() != self.type_name
        {
            return Err(StateError::AccessDenied(format!(
                "Native component {} ({:?}) does not match column {} ({:?})",
                value.type_name(),
                value.rust_type_id(),
                self.type_name,
                std::any::TypeId::of::<T>(),
            )));
        }
        Ok(())
    }

    #[inline]
    fn push_erased_infallible(&mut self, value: ErasedComponentValue) {
        self.backing.push(
            value
                .into_typed::<T>()
                .expect("validate_erased_for_push must precede push_erased_infallible"),
        );
    }

    #[inline]
    fn replace_erased_infallible(
        &mut self,
        row: usize,
        mut value: ErasedComponentValue,
    ) -> ErasedComponentValue {
        assert_eq!(
            value.component_type_id(),
            self.type_id,
            "validate_erased must precede replace_erased_infallible"
        );
        let replacement = value
            .downcast_mut::<T>()
            .expect("validate_erased must precede replace_erased_infallible");
        let slot = self
            .backing
            .borrow_row_mut(row)
            .expect("row in bounds for replace_erased_infallible");
        std::mem::swap(slot, replacement);
        value
    }

    fn assign_erased_infallible(&mut self, row: usize, value: ErasedComponentValue) {
        debug_assert_eq!(value.component_type_id(), self.type_id);
        let typed_val = value.into_typed::<T>().expect("validated component type");
        let slot = self
            .backing
            .borrow_row_mut(row)
            .expect("row in bounds for assign_erased_infallible");
        *slot = typed_val;
    }

    #[inline]
    fn swap_erased_infallible(&mut self, row: usize, slot: &mut ErasedComponentValue) {
        debug_assert_eq!(slot.component_type_id(), self.type_id);
        let replacement = slot
            .downcast_mut::<T>()
            .expect("validate_erased must precede swap_erased_infallible");
        let target = self
            .backing
            .borrow_row_mut(row)
            .expect("row in bounds for swap_erased_infallible");
        std::mem::swap(target, replacement);
    }

    #[inline]
    fn swap_remove_erased(&mut self, row: usize) -> ErasedComponentValue {
        let val = self.backing.swap_remove(row).expect("row in bounds");
        ErasedComponentValue::with_component_type_id(self.type_id, val)
    }

    fn discard_in_place_single_chunk(
        &mut self,
        chunk: &mut [crate::storage::column::PlannedInPlacePatch],
    ) {
        let type_id = self.type_id;
        let data = self.backing.as_mut_slice();
        for plan in chunk {
            let crate::storage::column::InPlaceReplacements::Single(t, op) = std::mem::replace(
                &mut plan.replacements,
                crate::storage::column::InPlaceReplacements::Empty,
            ) else {
                unreachable!("chunk must only contain InPlaceReplacements::Single");
            };
            let value = op.into_value();
            debug_assert_eq!(t, type_id);
            data[plan.location.row_index as usize] =
                value.into_typed::<T>().expect("validated component type");
        }
    }

    unsafe fn replace_exclusive_unchecked(
        &mut self,
        row: usize,
        value: ErasedComponentValue,
    ) -> ErasedComponentValue {
        debug_assert_eq!(value.component_type_id(), self.type_id);
        let val = value.into_typed::<T>().expect("validated component type");
        let old = unsafe { self.backing.write_exclusive_row_unchecked(row, val) };
        ErasedComponentValue::with_component_type_id(self.type_id, old)
    }

    unsafe fn assign_exclusive_unchecked(&mut self, row: usize, value: ErasedComponentValue) {
        debug_assert_eq!(value.component_type_id(), self.type_id);
        let val = value.into_typed::<T>().expect("validated component type");
        unsafe {
            drop(self.backing.write_exclusive_row_unchecked(row, val));
        }
    }
}

impl<T: ModelComponent + Serialize + DeserializeOwned> ColumnSnapshot for TypedColumn<T> {
    fn raw_archive_capability(&self) -> Option<archive::RawColumnArchive> {
        self.archive_vtable
            .archive_raw_slices
            .map(|encode| archive::RawColumnArchive {
                component: self.type_id,
                schema_version: self.archive_vtable.schema_version,
                rust_type: std::any::TypeId::of::<T>(),
                fixed_archived_row_bytes: self.archive_vtable.fixed_archived_row_bytes,
                encode,
            })
    }

    fn archive_erased_values(
        &self,
        values: &[&ErasedComponentValue],
    ) -> Result<ArchivedColumnData, StateError> {
        let slices: SmallVec<[&[T]; 4]> = values
            .iter()
            .map(|value| {
                if value.component_type_id() != self.type_id {
                    return Err(StateError::AccessDenied(
                        "Archive component identity mismatch".into(),
                    ));
                }
                value
                    .downcast_ref::<T>()
                    .map(std::slice::from_ref)
                    .ok_or_else(|| StateError::TypeMismatch {
                        expected: self.type_name.into(),
                        actual: value.type_name().into(),
                    })
            })
            .collect::<Result<_, _>>()?;
        (self.archive_vtable.archive_slices)(
            self.type_id,
            self.archive_vtable.schema_version,
            &slices,
        )
    }

    fn archive_column(&self) -> Result<ArchivedColumnData, StateError> {
        let slice = self.backing.as_contiguous_slice();
        let slices: SmallVec<[&[T]; 4]> = smallvec::smallvec![slice];
        (self.archive_vtable.archive_slices)(
            self.type_id,
            self.archive_vtable.schema_version,
            &slices,
        )
    }

    fn validate_archived_column(&self, archived: &ArchivedColumnData) -> Result<(), StateError> {
        if archived.component_type_id != self.type_id {
            return Err(StateError::AccessDenied(format!(
                "Column type mismatch: expected {:?}, got {:?}",
                self.type_id, archived.component_type_id
            )));
        }
        (self.archive_vtable.validate)(
            self.type_id,
            self.archive_vtable.schema_version,
            archived,
        )
    }

    fn restore_archived_column(&mut self, archived: &ArchivedColumnData) -> Result<(), StateError> {
        let was_paged = self.backing.is_page_aligned();
        let vec = self.backing.as_vec_mut();
        let res = (self.archive_vtable.restore)(
            self.type_id,
            self.archive_vtable.schema_version,
            archived,
            vec,
        );
        if was_paged && res.is_ok() {
            self.backing.promote_to_page_aligned();
        }
        res
    }

    #[inline]
    fn codec(&self) -> ColumnCodecId {
        self.archive_vtable.codec
    }

    #[inline]
    fn schema_version(&self) -> u32 {
        self.archive_vtable.schema_version
    }

    #[inline]
    fn has_archive_capability(&self) -> bool {
        self.archive_vtable.has_capability
    }

    fn serialize_row(&self, row: usize) -> Result<Vec<u8>, StateError> {
        let item = self
            .get(row)
            .ok_or_else(|| StateError::AccessDenied(format!("Row {} not found", row)))?;
        bincode::serialize(item).map_err(|e| StateError::CodecError(e.to_string()))
    }

    fn push_deserialized(&mut self, bytes: &[u8]) -> Result<(), StateError> {
        let item: T =
            bincode::deserialize(bytes).map_err(|e| StateError::CodecError(e.to_string()))?;
        self.push(item);
        Ok(())
    }

    fn set_deserialized(&mut self, row: usize, bytes: &[u8]) -> Result<(), StateError> {
        let item: T =
            bincode::deserialize(bytes).map_err(|e| StateError::CodecError(e.to_string()))?;
        self.set(row, item)
    }

    fn validate_deserialized(&self, row: usize, bytes: &[u8]) -> Result<(), StateError> {
        if row >= self.backing.len() {
            return Err(StateError::AccessDenied(format!(
                "Row {} out of bounds for column {}",
                row, self.type_name
            )));
        }
        bincode::deserialize::<T>(bytes)
            .map(|_| ())
            .map_err(|e| StateError::CodecError(e.to_string()))
    }

    fn decode_erased(&self, bytes: &[u8]) -> Result<ErasedComponentValue, StateError> {
        let value: T = bincode::deserialize(bytes)
            .map_err(|error| StateError::CodecError(error.to_string()))?;
        Ok(ErasedComponentValue::with_component_type_id(
            self.type_id,
            value,
        ))
    }
}

#[cfg(test)]
impl<T: ModelComponent + Serialize + DeserializeOwned> ColumnExperimentOps for TypedColumn<T> {
    fn experiment_swap_retained(&mut self, row: usize, value: &mut ErasedComponentValue) {
        let data = self.backing.as_mut_slice();
        std::mem::swap(&mut data[row], value.downcast_mut::<T>().unwrap());
    }

    fn experiment_assign_retained(&mut self, row: usize, value: &ErasedComponentValue) {
        let data = self.backing.as_mut_slice();
        data[row] = value.downcast_ref::<T>().unwrap().clone();
    }
}

#[cfg(not(test))]
impl<T: ModelComponent + Serialize + DeserializeOwned> ColumnExperimentOps for TypedColumn<T> {}
