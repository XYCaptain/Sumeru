use super::*;
use crate::storage::batch::{BatchOp, EntityBundle};
use crate::storage::store::Store;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct DummyPos(i32);

fn id(n: u128) -> EntityId {
    EntityId::from_u128(n)
}

#[test]
fn chained_entry_index_is_reused_and_separate_handles_merge() {
    let mut store = Store::new();
    store.insert((id(1), (1u64, 2u32, String::from("old")))).unwrap();
    let mut cmd = store.commands();
    {
        let mut entity = cmd.entity(id(1));
        assert_eq!(entity.entry_index, None);
        entity.replace(10u64);
        assert_eq!(entity.entry_index, Some(0));
        entity.replace(20u32);
        assert_eq!(entity.entry_index, Some(0));
    }
    cmd.entity(id(1)).replace(String::from("new"));
    assert_eq!(cmd.len(), 1);
    assert_eq!(cmd.component_op_count, 3);
    let receipt = cmd.apply(&mut store).unwrap();
    assert_eq!(receipt.updated_entities, vec![id(1)]);
    assert_eq!(*store.extract::<&u64>(id(1)).unwrap(), 10);
    assert_eq!(*store.extract::<&u32>(id(1)).unwrap(), 20);
    assert_eq!(store.extract::<&String>(id(1)).unwrap(), "new");
}

#[test]
fn cached_entry_preserves_first_error_limits_and_native_conflicts() {
    let mut store = Store::new();
    store.insert((id(1), (1u64, 2u32))).unwrap();
    let epoch = store.epoch();
    let mut cmd = store.commands();
    cmd.entity(id(1)).replace(10u64).replace(11u64).delete();
    assert!(matches!(cmd.apply(&mut store), Err(StateError::ComponentDuplicateInPatch { .. })));
    let mut cmd = store.commands_with_limits(CommandLimits {
        max_entities: Some(1), max_component_ops: Some(1),
    });
    cmd.entity(id(1)).replace(10u64).replace(20u32);
    assert!(matches!(cmd.apply(&mut store), Err(StateError::CapacityExhausted(_))));
    let mut cmd = store.commands();
    cmd.update_batch([(id(1), 10u64)]);
    cmd.entity(id(1)).replace(20u32).replace(30u64);
    assert!(matches!(cmd.apply(&mut store), Err(StateError::ComponentDuplicateInPatch { .. })));
    let mut cmd = store.commands();
    cmd.entity(id(1)).delete().replace(10u64);
    assert!(matches!(cmd.apply(&mut store), Err(StateError::BatchConflict(_))));
    assert_eq!(store.epoch(), epoch);
    assert_eq!(*store.extract::<&u64>(id(1)).unwrap(), 1);
}

#[test]
fn shared_in_place_ranges_commit_and_discard() {
    for keep in [true, false] {
        let mut store = Store::new();
        for n in 1..=64 {
            store.insert((id(n), (n as u64, n as u32, format!("old-{n}")))).unwrap();
        }
        let mut cmd = store.commands();
        // Reverse order and a different number of updates per entity.
        for n in (1..=64).rev() {
            let mut entity = cmd.entity(id(n));
            entity.replace(format!("new-{n}")).upsert(n as u64 + 100);
            if n % 2 == 0 { entity.replace(n as u32 + 100); }
        }
        let plan = if keep { cmd.prepare(&mut store) } else { cmd.prepare_discard(&mut store) }.unwrap();
        let mut end = 0;
        for entity in &plan.plan_scratch.in_place_plans {
            match &entity.replacements {
                crate::storage::core::column::InPlaceReplacements::Multiple(range) => {
                    assert_eq!(range.start, end);
                    end = range.end;
                }
                _ => panic!("multi-component plan must use shared storage or direct operations"),
            }
        }
        assert_eq!(end, 160);
        let receipt = if keep { Some(store.commit_batch(plan).unwrap()) } else {
            store.commit_batch_discard(plan).unwrap(); None
        };
        for n in 1..=64 {
            assert_eq!(*store.extract::<&u64>(id(n)).unwrap(), n as u64 + 100);
            assert_eq!(store.extract::<&String>(id(n)).unwrap(), &format!("new-{n}"));
        }
        if let Some(receipt) = receipt {
            assert_eq!(receipt.updated_entities.len(), 64);
        }
        store.validate_invariants().unwrap();
    }
}

#[test]
fn recoverable_failure_returns_shared_values_with_original_operations() {
    let mut store = Store::new();
    store.insert((id(1), (1u64, String::from("old")))).unwrap();
    let epoch = store.epoch();
    let mut batch = StoreBatch::new();
    let mut first = EntityPatch::new();
    first.upsert(11u64).unwrap().replace(String::from("new")).unwrap();
    batch.patch_entity(id(1), first).unwrap();
    let mut missing = EntityPatch::new();
    missing.replace(22u64).unwrap();
    batch.patch_entity(id(2), missing).unwrap();
    let (err, recovered) = store.prepare_batch_recoverable(batch).unwrap_err();
    assert!(matches!(err, StateError::EntityNotFound(_)));
    assert_eq!(store.epoch(), epoch);
    assert_eq!(store.extract::<&String>(id(1)).unwrap(), "old");
    let BatchOp::PatchEntity { patch, .. } = &recovered.ops()[&id(1)] else { panic!() };
    patch.for_each_op(|_, op| match op {
        crate::storage::batch::PatchOp::Upsert(v) => assert_eq!(v.downcast_ref::<u64>(), Some(&11)),
        crate::storage::batch::PatchOp::Replace(v) => assert_eq!(v.downcast_ref::<String>().unwrap(), "new"),
        _ => panic!("operation kind changed during recovery"),
    });
    store.insert((id(2), (2u64,))).unwrap();
    store.apply_batch(recovered).unwrap();
    assert_eq!(*store.extract::<&u64>(id(2)).unwrap(), 22);
    assert_eq!(store.extract::<&String>(id(1)).unwrap(), "new");
}

#[test]
fn discard_prepare_has_no_receipt_buffers_or_in_place_destruction_scratch() {
    for native in [false, true] {
        let mut store = Store::new();
        for n in 1..=32 {
            store.insert((id(n), (n as u64,))).unwrap();
        }
        let mut cmd = store.commands();
        if native {
            cmd.update_batch((1..=32).map(|n| (id(n), n as u64 + 100)));
        } else {
            for n in 1..=32 {
                cmd.entity(id(n)).replace(n as u64 + 100);
            }
        }
        let plan = cmd.prepare_with_receipt::<false>(&mut store).unwrap();
        assert_eq!(plan.receipt.created_entities.capacity(), 0);
        assert_eq!(plan.receipt.updated_entities.capacity(), 0);
        assert_eq!(plan.receipt.deleted_entities.capacity(), 0);
        assert_eq!(plan.discarded_values.capacity(), 0);
        store.commit_batch_discard(plan).unwrap();
        assert_eq!(*store.extract::<&u64>(id(32)).unwrap(), 132);
    }
}

#[test]
fn discard_structural_scratch_is_bounded_by_one_row_not_batch_size() {
    let mut store = Store::new();
    for n in 1..=32 {
        store.insert((id(n), (n as u64, true))).unwrap();
    }
    let mut cmd = store.commands();
    for n in 1..=32 {
        cmd.delete_entity(id(n));
    }
    let plan = cmd.prepare_with_receipt::<false>(&mut store).unwrap();
    assert_eq!(plan.discarded_values.capacity(), 2);
    assert_eq!(plan.receipt.deleted_entities.capacity(), 0);
    store.commit_batch_discard(plan).unwrap();
    assert!(store.is_empty());
    store.validate_invariants().unwrap();
}

#[test]
fn discard_empty_at_exhausted_epoch_succeeds_but_writes_fail() {
    let mut store = Store::new();
    store.insert((id(1), (1u64,))).unwrap();
    store.epoch = u64::MAX;
    store.commands().apply_discard(&mut store).unwrap();
    let mut cmd = store.commands();
    cmd.entity(id(1)); // Empty patch is also a no-op.
    cmd.apply_discard(&mut store).unwrap();
    for native in [false, true] {
        let mut cmd = store.commands();
        if native {
            cmd.update_batch([(id(1), 2u64)]);
        } else {
            cmd.entity(id(1)).replace(2u64);
        }
        assert!(matches!(
            cmd.apply_discard(&mut store),
            Err(StateError::CapacityExhausted(_))
        ));
        assert_eq!(*store.extract::<&u64>(id(1)).unwrap(), 1);
        assert_eq!(store.epoch(), u64::MAX);
    }
}

#[test]
fn test_command_limits_component_ops_overflow_injection_internal() {
    let mut store = Store::new();
    let e1 = id(1);
    let mut b = EntityBundle::new();
    b.insert(DummyPos(10)).unwrap();
    store.create_entity(e1, b).unwrap();
    let initial_epoch = store.epoch();

    // 1. 测试 EntityCommands::replace 中的 checked_add 溢出
    let mut cmd = store.commands();
    cmd.component_op_count = usize::MAX;

    cmd.entity(e1).replace(DummyPos(9999));

    let err = cmd.apply(&mut store).unwrap_err();
    assert!(
        matches!(err, StateError::CapacityExhausted(ref msg) if msg.contains("operations limit exceeded")),
        "Expected CapacityExhausted on op count overflow, got: {err:?}"
    );
    assert_eq!(store.epoch(), initial_epoch);
    assert_eq!(store.extract::<&DummyPos>(e1).unwrap(), &DummyPos(10));

    // 2. 测试 create_entity 中的 checked_add 溢出
    let mut cmd2 = store.commands();
    cmd2.component_op_count = usize::MAX;
    cmd2.create_entity(
        id(999),
        EntityBundle::new().with(DummyPos(123)).unwrap(),
    );
    let err2 = cmd2.apply(&mut store).unwrap_err();
    assert!(matches!(err2, StateError::CapacityExhausted(_)));
    assert_eq!(store.epoch(), initial_epoch);
    assert!(!store.contains_entity(id(999)));
}
