use super::*;
use crate::storage::core::column::backing;

impl<T: ModelComponent + Serialize + DeserializeOwned> AnyColumn for TypedColumn<T> {
    fn clone_box(&self) -> Box<dyn AnyColumn> {
        Box::new(self.clone())
    }

    fn empty_clone(&self) -> Box<dyn AnyColumn> {
        Box::new(Self {
            type_id: self.type_id,
            type_name: self.type_name,
            backing: self.backing.empty_with_origin(),
            archive_vtable: self.archive_vtable,
        })
    }

    fn validate_copy_to(&self, src_row: usize, dst: &dyn AnyColumn) -> Result<(), StateError> {
        if src_row >= self.backing.len() {
            return Err(StateError::AccessDenied(format!(
                "Row {} out of bounds for column {}",
                src_row, self.type_name
            )));
        }
        if dst.type_id() != self.type_id
            || self.type_name != std::any::type_name::<T>()
            || dst.type_name() != self.type_name
        {
            return Err(StateError::AccessDenied(format!(
                "Destination column type mismatch: expected {}, got {}",
                self.type_name,
                dst.type_name()
            )));
        }
        if dst.as_any().downcast_ref::<TypedColumn<T>>().is_none() {
            return Err(StateError::AccessDenied(format!(
                "Destination column downcast failed for {}",
                self.type_name
            )));
        }
        Ok(())
    }

    fn copy_row_to_infallible(&self, src_row: usize, dst: &mut dyn AnyColumn) {
        let val = self
            .backing
            .get(src_row)
            .expect("validate_copy_to must precede copy_row_to_infallible");
        let dst_typed = dst
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .expect("validate_copy_to must be called prior to copy_row_to_infallible");
        dst_typed.push(val.clone());
    }

    #[inline]
    fn move_row_to_infallible(&mut self, src_row: usize, dst: &mut dyn AnyColumn) {
        let val = self
            .backing
            .swap_remove(src_row)
            .expect("validate_copy_to must precede move_row_to_infallible");
        debug_assert!(
            dst.as_any().is::<TypedColumn<T>>(),
            "validate_copy_to must be called prior to move_row_to_infallible"
        );
        let dst_typed = unsafe { &mut *(dst as *mut dyn AnyColumn as *mut TypedColumn<T>) };
        dst_typed.push(val);
    }

    #[inline]
    fn raw_move_row_direct_fn(&self) -> Option<unsafe fn(*mut (), usize, *mut ())> {
        unsafe fn move_row_direct<T: ModelComponent + Serialize + DeserializeOwned>(
            src_ptr: *mut (),
            src_row: usize,
            dst_ptr: *mut (),
        ) {
            let src = unsafe { &mut *(src_ptr as *mut TypedColumn<T>) };
            let dst = unsafe { &mut *(dst_ptr as *mut TypedColumn<T>) };
            unsafe {
                let val = src.backing.swap_remove_direct_owned_unchecked(src_row);
                dst.backing.push_direct_owned_unchecked(val);
            }
        }
        Some(move_row_direct::<T>)
    }

    #[inline]
    fn raw_swap_remove_direct_fn(&self) -> Option<unsafe fn(*mut (), usize)> {
        unsafe fn swap_remove_direct<T: ModelComponent + Serialize + DeserializeOwned>(
            col_ptr: *mut (),
            row: usize,
        ) {
            let col = unsafe { &mut *(col_ptr as *mut TypedColumn<T>) };
            unsafe {
                let _ = col.backing.swap_remove_direct_owned_unchecked(row);
            }
        }
        Some(swap_remove_direct::<T>)
    }

    #[inline]
    fn raw_push_erased_direct_fn(&self) -> Option<unsafe fn(*mut (), &mut ErasedComponentValue)> {
        unsafe fn push_erased_direct<T: ModelComponent + Serialize + DeserializeOwned>(
            col_ptr: *mut (),
            val: &mut ErasedComponentValue,
        ) {
            let col = unsafe { &mut *(col_ptr as *mut TypedColumn<T>) };
            unsafe {
                let typed = val.take_typed_unchecked::<T>();
                col.backing.push_direct_owned_unchecked(typed);
            }
        }
        Some(push_erased_direct::<T>)
    }

    fn upgrade_archive_capability_from(&mut self, source: &dyn AnyColumn) -> bool {
        if !self.archive_vtable.has_capability && source.has_archive_capability() {
            if let Some(src_typed) = source.as_any().downcast_ref::<TypedColumn<T>>() {
                self.archive_vtable = src_typed.archive_vtable;
                return true;
            }
        }
        false
    }

    #[inline]
    fn physical_version(&self) -> u64 {
        self.backing.physical_version()
    }

    #[inline]
    fn is_segmented(&self) -> bool {
        false
    }

    #[inline]
    fn segment_count(&self) -> usize {
        self.backing.segment_count()
    }

    #[inline]
    fn ensure_contiguous(&mut self) -> Result<(), StateError> {
        self.backing.ensure_contiguous();
        Ok(())
    }

    fn prepare_contiguous(&mut self) -> Result<Option<Box<dyn AnyColumn>>, StateError> {
        if self.backing.is_uniquely_writable() {
            return Ok(None);
        }
        let mut detached = self.clone();
        detached.backing.ensure_contiguous();
        Ok(Some(Box::new(detached)))
    }

    #[inline]
    fn supports_freeze(&self) -> bool {
        true
    }

    fn column_from_frozen(
        &self,
        block: Arc<dyn crate::storage::column::ErasedBlock>,
    ) -> Result<Box<dyn AnyColumn>, StateError> {
        let block: Arc<dyn Any + Send + Sync> = block.into_any_arc();
        let typed_block = Arc::downcast::<ColumnBlock<T>>(block).map_err(|_| {
            StateError::TypeMismatch {
                expected: self.type_name.into(),
                actual: "frozen column payload".into(),
            }
        })?;
        Ok(Box::new(Self {
            type_id: self.type_id,
            type_name: self.type_name,
            backing: ColumnBacking::from_block(typed_block),
            archive_vtable: self.archive_vtable,
        }))
    }

    fn column_from_typed_owned(
        &self,
        block: Arc<dyn crate::storage::column::ErasedBlock>,
    ) -> Result<Box<dyn AnyColumn>, StateError> {
        let any_arc: Arc<dyn Any + Send + Sync> = block.into_any_arc();
        let typed_arc = Arc::downcast::<crate::storage::column::TypedOwnedBlock<T>>(any_arc)
            .map_err(|_| StateError::TypeMismatch {
                expected: self.type_name.into(),
                actual: "typed owned column payload".into(),
            })?;
        let values = match Arc::try_unwrap(typed_arc) {
            Ok(mut owned) => std::mem::take(&mut owned.values),
            Err(shared) => shared.values.clone(),
        };
        Ok(Box::new(Self {
            type_id: self.type_id,
            type_name: self.type_name,
            backing: ColumnBacking::from_vec(values),
            archive_vtable: self.archive_vtable,
        }))
    }

    fn freeze_column(
        &mut self,
    ) -> Result<std::sync::Arc<dyn crate::storage::column::ErasedBlock>, StateError> {
        Ok(self.backing.freeze_erased())
    }

    fn restore_row_from_erased_block(
        &mut self,
        target_row: usize,
        block: &dyn crate::storage::column::ErasedBlock,
        source_idx: usize,
    ) -> Result<(), StateError> {
        let col_block = block
            .as_any()
            .downcast_ref::<backing::ColumnBlock<T>>()
            .ok_or_else(|| StateError::TypeMismatch {
                expected: std::any::type_name::<T>().into(),
                actual: "Unknown ErasedBlock".into(),
            })?;
        let slice = self.backing.as_mut_slice();
        if target_row >= slice.len() || source_idx >= col_block.len() {
            return Err(StateError::AccessDenied(
                "Row out of bounds in restore_row".into(),
            ));
        }
        slice[target_row] = col_block.data[source_idx].clone();
        Ok(())
    }

    fn restore_range_from_erased_block(
        &mut self,
        target_start: usize,
        len: usize,
        block: &dyn crate::storage::column::ErasedBlock,
        source_start: usize,
    ) -> Result<(), StateError> {
        let col_block = block
            .as_any()
            .downcast_ref::<backing::ColumnBlock<T>>()
            .ok_or_else(|| StateError::TypeMismatch {
                expected: std::any::type_name::<T>().into(),
                actual: "Unknown ErasedBlock".into(),
            })?;
        let slice = self.backing.as_mut_slice();
        if target_start + len > slice.len() || source_start + len > col_block.len() {
            return Err(StateError::AccessDenied(
                "Range out of bounds in restore_range".into(),
            ));
        }
        slice[target_start..target_start + len]
            .clone_from_slice(&col_block.data[source_start..source_start + len]);
        Ok(())
    }

    fn apply_native_block_scatter(
        &mut self,
        block: Box<dyn crate::storage::column::ErasedBlock>,
        block_range: std::ops::Range<usize>,
        target_rows: &[usize],
    ) -> Result<(), StateError> {
        let any_box = block.into_any_box();
        let mut data: Vec<T> = match any_box.downcast::<backing::ColumnBlock<T>>() {
            Ok(mut col_block) => std::mem::take(&mut col_block.data),
            Err(any_box) => match any_box.downcast::<Vec<T>>() {
                Ok(boxed_vec) => *boxed_vec,
                Err(any_box) => {
                    match any_box.downcast::<crate::storage::column::TypedOwnedBlock<T>>() {
                        Ok(mut typed_block) => std::mem::take(&mut typed_block.values),
                        Err(_) => {
                            return Err(StateError::BatchConflict(
                                "Type mismatch in native block scatter".into(),
                            ))
                        }
                    }
                }
            },
        };
        if block_range.start > block_range.end || block_range.end > data.len() {
            return Err(StateError::BatchConflict(
                "Invalid block range for scatter".into(),
            ));
        }
        if block_range.len() != target_rows.len() {
            return Err(StateError::BatchConflict(
                "Block range len != target rows len".into(),
            ));
        }
        for (val, &row) in data.drain(block_range).zip(target_rows) {
            let _old = self.backing.replace_row_infallible(row, val);
        }
        Ok(())
    }

    fn apply_native_block_range(
        &mut self,
        block: Box<dyn crate::storage::column::ErasedBlock>,
        block_range: std::ops::Range<usize>,
        target_row_start: usize,
    ) -> Result<(), StateError> {
        let any_box = block.into_any_box();
        let mut data: Vec<T> = match any_box.downcast::<backing::ColumnBlock<T>>() {
            Ok(mut typed_block) => std::mem::take(&mut typed_block.data),
            Err(any_box) => match any_box.downcast::<Vec<T>>() {
                Ok(boxed_vec) => *boxed_vec,
                Err(any_box) => {
                    match any_box.downcast::<crate::storage::column::TypedOwnedBlock<T>>() {
                        Ok(mut typed_block) => std::mem::take(&mut typed_block.values),
                        Err(_) => {
                            return Err(StateError::BatchConflict(
                                "Type mismatch in native block range".into(),
                            ))
                        }
                    }
                }
            },
        };
        if block_range.start > block_range.end || block_range.end > data.len() {
            return Err(StateError::BatchConflict(
                "Invalid block range for range handoff".into(),
            ));
        }
        if target_row_start + block_range.len() > self.backing.len() {
            return Err(StateError::BatchConflict(
                "Target range out of bounds for range handoff".into(),
            ));
        }
        for (idx, val) in data.drain(block_range).enumerate() {
            let _old = self
                .backing
                .replace_row_infallible(target_row_start + idx, val);
        }
        Ok(())
    }

    fn apply_native_shared_block_range(
        &mut self,
        block: std::sync::Arc<dyn crate::storage::column::ErasedBlock>,
        block_range: std::ops::Range<usize>,
        target_row_start: usize,
    ) -> Result<(), StateError> {
        let any_arc = block.into_any_arc();
        match any_arc.downcast::<backing::ColumnBlock<T>>() {
            Ok(arc_block) => {
                if block_range.start > block_range.end || block_range.end > arc_block.len() {
                    return Err(StateError::BatchConflict(
                        "Invalid block range for shared range handoff".into(),
                    ));
                }
                if target_row_start + block_range.len() > self.backing.len() {
                    return Err(StateError::BatchConflict(
                        "Target range out of bounds for shared range handoff".into(),
                    ));
                }
                let dst = self.backing.as_mut_slice();
                dst[target_row_start..target_row_start + block_range.len()]
                    .clone_from_slice(&arc_block.as_slice()[block_range]);
                Ok(())
            }
            Err(_) => Err(StateError::BatchConflict(
                "Type mismatch in native shared block range".into(),
            )),
        }
    }

    fn new_append_column(&self, capacity: usize) -> Option<(Box<dyn AnyColumn>, *mut u8, usize)> {
        if capacity == 0 {
            return None;
        }
        let mut vec = Vec::<T>::new();
        let (ptr, actual_cap) = if std::mem::size_of::<T>() == 0 {
            (vec.as_mut_ptr() as *mut u8, capacity)
        } else {
            vec.reserve_exact(capacity);
            (vec.as_mut_ptr() as *mut u8, vec.capacity())
        };
        let col = Box::new(Self {
            type_id: self.type_id,
            type_name: self.type_name,
            backing: backing::ColumnBacking::from_vec(vec),
            archive_vtable: self.archive_vtable,
        });
        Some((col, ptr, actual_cap))
    }

    unsafe fn move_into_append_ptr(
        &self,
        dst_base: *mut u8,
        dst_slot: usize,
        src: &mut dyn AnyColumn,
        src_row: usize,
    ) {
        let src_typed = src
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .expect("matching TypedColumn for append move");
        let backing::ColumnBacking::Direct(backing::DirectBlock::Owned(ref mut vec)) =
            src_typed.backing
        else {
            unreachable!("append move source must be DirectBlock::Owned");
        };
        if !vec.is_empty() {
            unsafe {
                vec.set_len(0);
            }
        }
        debug_assert!(src_row < vec.capacity());
        unsafe {
            let val = std::ptr::read(vec.as_ptr().add(src_row));
            std::ptr::write(dst_base.cast::<T>().add(dst_slot), val);
        }
    }

    unsafe fn reclaim_append_tail_ptr(
        &self,
        base: *mut u8,
        slot_start: usize,
        count: usize,
    ) -> Option<Box<dyn AnyColumn>> {
        if count == 0 || !std::mem::needs_drop::<T>() {
            return None;
        }
        let mut retired_vec = Vec::with_capacity(count);
        for i in 0..count {
            let val = unsafe { std::ptr::read(base.cast::<T>().add(slot_start + i)) };
            retired_vec.push(val);
        }
        Some(Box::new(Self {
            type_id: self.type_id,
            type_name: self.type_name,
            backing: backing::ColumnBacking::from_vec(retired_vec),
            archive_vtable: self.archive_vtable,
        }))
    }

    unsafe fn sync_append_len(&mut self, initialized_len: usize) {
        if let backing::ColumnBacking::Direct(backing::DirectBlock::Owned(ref mut vec)) =
            self.backing
        {
            debug_assert!(initialized_len <= vec.capacity());
            unsafe {
                vec.set_len(initialized_len);
            }
        }
    }

    fn reordered_column(
        &self,
        order: &[usize],
        _group_spans: Option<&[std::ops::Range<usize>]>,
    ) -> Result<Box<dyn AnyColumn>, StateError> {
        if order.len() != self.backing.len() {
            return Err(StateError::AccessDenied(
                "Reorder permutation length mismatch".into(),
            ));
        }
        let mut values = Vec::with_capacity(order.len());
        for &src_row in order {
            let val = self
                .backing
                .get(src_row)
                .ok_or_else(|| StateError::AccessDenied("Reorder row out of bounds".into()))?;
            values.push(val.clone());
        }
        Ok(Box::new(Self {
            type_id: self.type_id,
            type_name: self.type_name,
            backing: backing::ColumnBacking::from_vec(values),
            archive_vtable: self.archive_vtable,
        }))
    }
}
