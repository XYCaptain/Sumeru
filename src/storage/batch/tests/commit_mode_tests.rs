//! PreparedBatch 提交模式防呆与接口安全测试
//!
//! 验证：
//! 1. Discard 模式准备的批次传给保留收据的 commit_batch 会被严格拦截拒绝，杜绝残缺收据；
//! 2. Keep 模式准备的批次传给丢弃收据的 commit_batch_discard 会被严格拦截拒绝；
//! 3. 正常匹配的准备与提交路径保证语义与收据一致性。

use crate::binding::EntityId;
use crate::error::StateError;
use crate::storage::core::store::Store;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Pos(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Vel(u64);

#[test]
fn linked_typed_plans_keep_lane_identity_and_complete_entity_receipts() {
    use crate::storage::core::column::InPlaceReplacements;
    for discard in [false, true] {
        let mut store = Store::new();
        for n in 1..=3 {
            store.insert((EntityId::from_u128(n), (Pos(n as u64), Vel(10)))).unwrap();
        }
        let id = EntityId::from_u128;
        let mut commands = store.commands();
        // Lane 0 contains a mixed ordinary patch, linked and standalone targets.
        commands.update_batch([(id(1), Pos(100))]);
        commands.entity(id(1)).replace(Vel(200));
        // The second Pos call reuses lane 0; Vel is lane 1.
        commands.update_batch([(id(2), Pos(100)), (id(3), Pos(100))]);
        commands.update_batch([(id(2), Vel(200))]);
        let plan = if discard { commands.prepare_discard(&mut store) }
            else { commands.prepare(&mut store) }.unwrap();
        assert_eq!(plan.plan_scratch.native_references.len(), 2);
        assert_eq!(plan.native_plans.len(), 2, "one stable lane per Rust type");
        assert!(matches!(plan.plan_scratch.in_place_plans[1].replacements, InPlaceReplacements::NativeMultiple(_)));
        if discard {
            store.commit_batch_discard(plan).unwrap();
        } else {
            let receipt = store.commit_batch(plan).unwrap();
            assert_eq!(receipt.updated_entities, vec![id(1), id(2), id(3)]);
        }
        for n in 1..=3 {
            assert_eq!(store.get::<Pos>(id(n)), Some(&Pos(100)));
            assert_eq!(store.get::<Vel>(id(n)), Some(&Vel(if n != 3 { 200 } else { 10 })));
        }
        store.validate_invariants().unwrap();
    }
}

#[test]
fn conditional_structural_buffers_follow_actual_migrations() {
    for inserts in [0, 1, 128] {
        for discard in [false, true] {
            let mut store = Store::new();
            for n in 1..=128 {
                store.insert((EntityId::from_u128(n), (Pos(n as u64),))).unwrap();
            }
            let mut commands = store.commands();
            for n in 1..=128 {
                let mut entity = commands.entity(EntityId::from_u128(n));
                entity.upsert(Pos(200));
                if n <= inserts { entity.upsert(Vel(300)); }
            }
            let plan = if discard {
                commands.prepare_discard(&mut store)
            } else { commands.prepare(&mut store) }.unwrap();
            assert_eq!(plan.plans.len(), inserts as usize);
            if inserts == 0 {
                assert_eq!(plan.plans.capacity(), 0);
            } else if inserts == 1 {
                assert!(plan.plans.capacity() <= 16);
            }
            if discard {
                store.commit_batch_discard(plan).unwrap();
            } else {
                let receipt = store.commit_batch(plan).unwrap();
                assert_eq!(receipt.updated_entities.len(), 128);
            }
            for n in 1..=128 {
                assert_eq!(store.get::<Pos>(EntityId::from_u128(n)), Some(&Pos(200)));
                assert_eq!(store.get::<Vel>(EntityId::from_u128(n)), (n <= inserts).then_some(&Vel(300)));
            }
            store.validate_invariants().unwrap();
        }
    }
}

#[test]
fn test_discard_prepared_batch_rejected_by_commit_batch() {
    let mut store = Store::new();
    let eid = EntityId::from_u128(101);

    // 录入创建实体与初始组件
    let mut cmd = store.commands();
    cmd.insert((eid, (Pos(10), Vel(20))));

    // 使用 discard 模式准备（不收集实体创建收据）
    let prep = cmd.prepare_discard(&mut store).unwrap();

    // 试图交给保留收据的 commit_batch 提交
    let res = store.commit_batch(prep);
    match res {
        Err(StateError::AccessDenied(msg)) => {
            assert!(
                msg.contains("Mismatched PreparedBatch commit mode"),
                "Expected mismatched mode error message, got: {msg}"
            );
        }
        other => panic!("Expected StateError::AccessDenied, got: {other:?}"),
    }

    // 确认 store 保持未提交状态（实体不存在）
    assert!(!store.contains_entity(eid));
}

#[test]
fn test_keep_prepared_batch_rejected_by_commit_batch_discard() {
    let mut store = Store::new();
    let eid = EntityId::from_u128(202);

    let mut cmd = store.commands();
    cmd.insert((eid, (Pos(10), Vel(20))));

    // 使用 keep 模式准备（保留收据）
    let prep = cmd.prepare(&mut store).unwrap();

    // 试图交给 discard 的提交接口
    let res = store.commit_batch_discard(prep);
    match res {
        Err(StateError::AccessDenied(msg)) => {
            assert!(
                msg.contains("Mismatched PreparedBatch commit mode"),
                "Expected mismatched mode error message, got: {msg}"
            );
        }
        other => panic!("Expected StateError::AccessDenied, got: {other:?}"),
    }

    assert!(!store.contains_entity(eid));
}

#[test]
fn test_matched_modes_execute_correctly() {
    let mut store = Store::new();
    let eid = EntityId::from_u128(404);
    let eid2 = EntityId::from_u128(405);

    // 1. Keep 模式正常提交
    let mut cmd = store.commands();
    cmd.insert((eid, (Pos(1), Vel(2))));
    let prep = cmd.prepare(&mut store).unwrap();
    let receipt = store.commit_batch(prep).unwrap();

    assert!(store.contains_entity(eid));
    assert_eq!(receipt.created_entities, vec![eid]);

    // 2. Discard 模式正常提交
    let mut cmd2 = store.commands();
    cmd2.insert((eid2, (Pos(10), Vel(20))));
    let prep_discard = cmd2.prepare_discard(&mut store).unwrap();
    store.commit_batch_discard(prep_discard).unwrap();
    assert!(store.contains_entity(eid2));
}
