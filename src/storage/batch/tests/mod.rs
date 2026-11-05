//! Batch 模块白盒单元测试套件

mod commit_mode_tests;

#[test]
fn pooled_links_reserve_sentinel_and_reject_overflow_before_mutation() {
    use super::{CompactPatchOps, PatchOp};
    use crate::{ComponentTypeId, StateError};
    let max = u32::MAX as usize;
    assert!(CompactPatchOps::check_pool_growth(max - 3, 3).is_ok());
    assert!(CompactPatchOps::check_pool_growth(max - 1, 1).is_ok());
    for (current, additional) in [(max - 2, 3), (max, 1), (usize::MAX, 1)] {
        assert!(matches!(CompactPatchOps::check_pool_growth(current, additional), Err(StateError::CapacityExhausted(_))));
    }
    let mut ops = CompactPatchOps::Pooled { head: 0, tail: 0, len: max };
    let mut pool = Vec::new();
    let ty = ComponentTypeId::of::<u64>();
    assert!(matches!(ops.insert_op_pooled(ty, PatchOp::Remove(ty), &mut pool), Err(StateError::CapacityExhausted(_))));
    assert!(matches!(ops, CompactPatchOps::Pooled { head: 0, tail: 0, len } if len == max));
    assert!(pool.is_empty());
}

#[test]
fn encoded_column_run_binary_and_json_roundtrip_and_unaligned_byte_len_rejection() {
    use crate::storage::{ArchivedColumnData, ColumnCodecId, EncodedColumnOpKind, EncodedColumnRun};
    use crate::{ComponentTypeId, EntityId};
    use bincode::Options;

    let entities = vec![
        EntityId::from_u128(0xfeed_face_cafe_beef_0001_0002_0003_0004),
        EntityId::from_u128(0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10),
        EntityId::from_u128(0xffff_0000_aaaa_5555_1234_5678_9abc_def0),
    ];
    let run = EncodedColumnRun {
        entities: Vec::new(),
        shared_entities: Some(entities.clone().into()),
        op_kind: EncodedColumnOpKind::Replace,
        archived: ArchivedColumnData::new(
            ComponentTypeId::of::<u64>(),
            1,
            3,
            ColumnCodecId::Rkyv,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        ),
    };

    let opts = bincode::DefaultOptions::new().with_fixint_encoding();
    let bytes = opts.serialize(&run).expect("bincode serialize");
    // Leading u64 length must be 3 * 16 = 48 bytes (not 3 elements of 24 bytes each).
    assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 48);
    let decoded: EncodedColumnRun = opts.deserialize(&bytes).expect("bincode deserialize");
    assert_eq!(decoded.entities, entities);
    assert_eq!(decoded, run);

    // JSON (human-readable) roundtrip must also succeed.
    let json = serde_json::to_string(&run).expect("json serialize");
    let decoded_json: EncodedColumnRun = serde_json::from_str(&json).expect("json deserialize");
    assert_eq!(decoded_json.entities, entities);
    assert_eq!(decoded_json, run);

    // Corrupting the entities byte length to a non-multiple of 16 (e.g. 47) must be rejected.
    let mut corrupted = bytes.clone();
    corrupted[0..8].copy_from_slice(&47u64.to_le_bytes());
    assert!(opts.deserialize::<EncodedColumnRun>(&corrupted).is_err());
}
