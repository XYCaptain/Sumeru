use super::*;

#[test]
fn slice_archive_single_slice_matches_existing_vec_bytes() {
    let values = vec![1u64, 20, 300];
    let archive = rkyv_archive_slices(ComponentTypeId::of::<u64>(), 7, &[&values]).unwrap();
    let old = rkyv::to_bytes::<rkyv::rancor::BoxedError>(&values).unwrap();
    assert_eq!(archive.payload, old.as_slice());
    assert_eq!(archive.row_count, 3);
    assert_eq!(archive.schema_version, 7);
    let mut restored = vec![];
    rkyv_restore_column::<u64>(ComponentTypeId::of::<u64>(), 7, &archive, &mut restored).unwrap();
    assert_eq!(restored, values);
}

// Intentionally not Clone: the archiver requires serialization, not Clone.
#[derive(
    Debug,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
struct Nested {
    label: String,
    samples: Vec<Vec<u64>>,
}

#[test]
fn slice_archive_multiple_allocations_support_nested_values_without_clone() {
    let left = vec![Nested {
        label: "left".repeat(12),
        samples: vec![vec![1, 2], vec![]],
    }];
    let right = vec![Nested {
        label: "right".repeat(10),
        samples: vec![vec![], vec![3, 4]],
    }];
    let archive = rkyv_archive_slices(
        ComponentTypeId::of::<Nested>(),
        1,
        &[&[], &left, &[], &right, &[]],
    )
    .unwrap();
    let mut restored = vec![];
    rkyv_restore_column::<Nested>(ComponentTypeId::of::<Nested>(), 1, &archive, &mut restored)
        .unwrap();
    assert_eq!(restored[0], left[0]);
    assert_eq!(restored[1], right[0]);
    assert_eq!(archive.row_count, 2);
}

#[test]
fn slice_archive_empty_sources_have_existing_vec_layout() {
    let old = rkyv::to_bytes::<rkyv::rancor::BoxedError>(&Vec::<u32>::new()).unwrap();
    for slices in [&[][..], &[&[][..]][..], &[&[][..], &[][..]][..]] {
        let archive = rkyv_archive_slices::<u32>(ComponentTypeId::of::<u32>(), 1, slices).unwrap();
        assert_eq!(archive.payload, old.as_slice());
        assert_eq!(archive.row_count, 0);
    }
}

#[test]
fn slice_archive_zero_sized_rows_keep_logical_count() {
    let left = [(); 3];
    let right = [(); 2];
    let archive = rkyv_archive_slices(ComponentTypeId::of::<()>(), 1, &[&left, &right]).unwrap();
    let mut restored = vec![];
    rkyv_restore_column::<()>(ComponentTypeId::of::<()>(), 1, &archive, &mut restored).unwrap();
    assert_eq!(restored.len(), 5);
    assert_eq!(archive.row_count, 5);
}

#[test]
fn slice_archive_cursor_clone_preserves_exact_remaining_count() {
    let left = [1, 2];
    let right = [3];
    let slices: &[&[i32]] = &[&[], &left, &[], &right, &[]];
    let mut rows = SliceRows::new(slices, 3);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows.next(), Some(&1));
    assert_eq!(rows.size_hint(), (2, Some(2)));
    let mut copy = rows.clone();
    assert_eq!(copy.next(), Some(&2));
    assert_eq!(copy.len(), 1);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows.copied().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(copy.next(), Some(&3));
    assert_eq!(copy.len(), 0);
    assert_eq!(copy.next(), None);
    assert_eq!(copy.next(), None);
}

#[test]
fn slice_archive_row_count_rejects_overflow_before_serializing() {
    assert_eq!(
        checked_row_count([u32::MAX as usize].into_iter()).unwrap(),
        u32::MAX
    );
    assert!(checked_row_count([u32::MAX as usize, 1].into_iter()).is_err());
    assert!(checked_row_count([usize::MAX, 1].into_iter()).is_err());
}

#[derive(Clone, serde::Serialize, serde::Deserialize, rkyv::Archive, rkyv::Deserialize)]
struct RejectArchive(String);

impl<S: rkyv::rancor::Fallible + ?Sized> rkyv::Serialize<S> for RejectArchive
where
    S::Error: rkyv::rancor::Source,
{
    fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
        Err(rkyv::rancor::Source::new(std::io::Error::other(
            "injected archive refusal",
        )))
    }
}

#[test]
fn slice_archive_serialization_error_does_not_modify_column() {
    use crate::storage::column::{ColumnSnapshot, TypedColumn};
    let mut column = TypedColumn::<RejectArchive>::new_rkyv(std::any::type_name::<RejectArchive>());
    column.push(RejectArchive("unchanged".into()));
    let version = column.physical_version();
    assert!(matches!(
        column.archive_column(),
        Err(StateError::CodecError(_))
    ));
    assert_eq!(column.physical_version(), version);
    assert_eq!(column.get(0).unwrap().0, "unchanged");
}

#[test]
fn rkyv_vec_dependency_control_roundtrip() {
    // Dependency-only control: no borrowed-column wrapper, vtable, Store or
    // byte Vec conversion. Keep this visible when assessing Miri diagnostics.
    let values = vec![1u64, 20, 300];
    let bytes = rkyv::to_bytes::<rkyv::rancor::BoxedError>(&values).unwrap();
    let restored: Vec<u64> = rkyv::from_bytes::<_, rkyv::rancor::BoxedError>(&bytes).unwrap();
    assert_eq!(restored, values);
}

// Frozen default rkyv-0.8 wire: two Arc<[u8]> rows share the four-byte
// pointee. Only the second pointer's length differs in the malformed case.
// All offsets and bounds are valid, so a bounds-only validator is insufficient.
fn shared_slice_column(second_len: u32) -> ArchivedColumnData {
    let mut payload = vec![
        9, 8, 7, 6, 0xfc, 0xff, 0xff, 0xff, 4, 0, 0, 0, 0xf4, 0xff, 0xff, 0xff, 4, 0, 0, 0, 0xf0,
        0xff, 0xff, 0xff, 2, 0, 0, 0,
    ];
    payload[16..20].copy_from_slice(&second_len.to_le_bytes());
    ArchivedColumnData::new(
        ComponentTypeId::of::<std::sync::Arc<[u8]>>(),
        1,
        2,
        ColumnCodecId::Rkyv,
        payload,
    )
}

#[test]
fn slice_archive_rejects_inconsistent_shared_metadata_before_restore() {
    use std::sync::Arc;
    let archive = shared_slice_column(2);
    let cid = archive.component_type_id;
    assert!(matches!(
        rkyv_validate_column::<Arc<[u8]>>(cid, 1, &archive),
        Err(StateError::CodecError(_))
    ));
    let original: Arc<[u8]> = Arc::from([255u8, 254]);
    let mut target = vec![original.clone()];
    assert!(matches!(
        rkyv_restore_column::<Arc<[u8]>>(cid, 1, &archive, &mut target),
        Err(StateError::CodecError(_))
    ));
    assert_eq!(target.len(), 1);
    assert!(Arc::ptr_eq(&target[0], &original));
    assert_eq!(&*target[0], &[255, 254]);
}

#[test]
fn slice_archive_preserves_valid_shared_rows_and_existing_wire() {
    use std::sync::Arc;
    let archive = shared_slice_column(4);
    let cid = archive.component_type_id;
    let value: Arc<[u8]> = Arc::from([9u8, 8, 7, 6]);
    let rows = [value.clone(), value];
    let encoded = rkyv_archive_slices(cid, 1, &[&rows[..1], &rows[1..]]).unwrap();
    assert_eq!(encoded.payload, archive.payload);
    rkyv_validate_column::<Arc<[u8]>>(cid, 1, &archive).unwrap();
    let mut restored = vec![];
    rkyv_restore_column::<Arc<[u8]>>(cid, 1, &archive, &mut restored).unwrap();
    assert_eq!(restored.len(), 2);
    assert_eq!(&*restored[0], &[9, 8, 7, 6]);
    assert!(Arc::ptr_eq(&restored[0], &restored[1]));
}

#[test]
fn slice_archive_restore_repairs_nested_alignment_not_just_root() {
    let values = vec![Nested {
        label: "nested".repeat(8),
        samples: vec![vec![1u64, 2, 3]],
    }];
    let archive = rkyv_archive_slices(ComponentTypeId::of::<Nested>(), 1, &[&values]).unwrap();
    let mut shifted = rkyv::util::AlignedVec::<16>::new();
    shifted.extend_from_slice(&[0; 4]);
    shifted.extend_from_slice(&archive.payload);
    let payload = &shifted[4..];
    assert_eq!((payload.as_ptr() as usize) % 4, 0);
    assert_ne!((payload.as_ptr() as usize) % 8, 0);
    let mut restored = vec![];
    rkyv_restore_column_from_slice::<Nested>(
        ComponentTypeId::of::<Nested>(),
        1,
        ColumnCodecId::Rkyv,
        1,
        payload,
        &mut restored,
    )
    .unwrap();
    assert_eq!(restored, values);
}
