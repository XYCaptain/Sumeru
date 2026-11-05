use crate::storage::{
    ColumnValueOps, Store, ErasedComponentValue, RawColumn, TypedColumn,
};
use crate::{ComponentTypeId, EntityId};
use serde::{Deserialize, Serialize, Serializer};
use std::any::TypeId;

fn component_id<T>() -> ComponentTypeId {
    ComponentTypeId::from_name(std::any::type_name::<T>())
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
struct RejectEncoding(u64);

impl Serialize for RejectEncoding {
    fn serialize<S: Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom(
            "encoding is forbidden in typed operations",
        ))
    }
}

#[test]
fn typed_value_clone_downcast_and_replace_do_not_encode() {
    let mut value = ErasedComponentValue::new(RejectEncoding(2));
    assert_eq!(value.component_type_id(), component_id::<RejectEncoding>());
    assert_eq!(value.rust_type_id(), TypeId::of::<RejectEncoding>());
    assert!(value.downcast_ref::<u64>().is_none());
    value.downcast_mut::<RejectEncoding>().unwrap().0 = 3;
    assert_eq!(
        value.clone().downcast_ref::<RejectEncoding>(),
        Some(&RejectEncoding(3))
    );

    let entity = EntityId::from_u128(1);
    let mut store = Store::new();
    store.spawn(entity).unwrap();
    store.insert_component(entity, RejectEncoding(1)).unwrap();
    let id = component_id::<RejectEncoding>();
    store.validate_component_value(entity, id, &value).unwrap();
    let old = store.replace_component_value_infallible(entity, id, value);
    assert_eq!(
        old.downcast_ref::<RejectEncoding>(),
        Some(&RejectEncoding(1))
    );
    assert_eq!(
        store
            .clone_component_value(entity, id)
            .unwrap()
            .downcast_ref::<RejectEncoding>(),
        Some(&RejectEncoding(3))
    );
    let replaced = store.replace_component_value(entity, id, old).unwrap();
    assert_eq!(
        replaced.downcast_ref::<RejectEncoding>(),
        Some(&RejectEncoding(3))
    );
    assert_eq!(
        store.get::<RejectEncoding>(entity),
        Some(&RejectEncoding(1))
    );
    assert!(replaced.encode().is_err());
}

#[test]
fn validation_checks_schema_rust_type_row_and_entity_without_mutation() {
    let entity = EntityId::from_u128(1);
    let mut store = Store::new();
    store.spawn(entity).unwrap();
    store.insert_component(entity, 10_u64).unwrap();
    let id = component_id::<u64>();
    let forged_rust = ErasedComponentValue::with_component_type_id(id, 99_i64);
    let forged_schema =
        ErasedComponentValue::with_component_type_id(component_id::<bool>(), 99_u64);
    for invalid in [forged_rust, forged_schema] {
        assert!(
            store
                .validate_component_value(entity, id, &invalid)
                .is_err()
        );
        assert!(store.replace_component_value(entity, id, invalid).is_err());
        assert_eq!(store.get::<u64>(entity), Some(&10));
    }
    assert!(
        store
            .replace_component_value(
                EntityId::from_u128(2),
                id,
                ErasedComponentValue::new(99_u64)
            )
            .is_err()
    );
    assert!(
        store
            .replace_component_value(
                entity,
                component_id::<bool>(),
                ErasedComponentValue::new(false)
            )
            .is_err()
    );

    let column = TypedColumn::<u64>::new(std::any::type_name::<u64>());
    assert!(
        column
            .validate_erased(0, &ErasedComponentValue::new(1_u64))
            .is_err()
    );
    let raw = RawColumn::new(id, vec![vec![1]]);
    assert!(
        raw.validate_erased(0, &ErasedComponentValue::new(1_u64))
            .is_err()
    );
    assert!(raw.clone_erased(0).is_err());
}

#[test]
fn undo_resolves_entity_after_swap_remove_and_archetype_move() {
    let mut store = Store::new();
    let first = EntityId::from_u128(1);
    let target = EntityId::from_u128(2);
    for (entity, value) in [(first, 10_u64), (target, 20_u64)] {
        store.spawn(entity).unwrap();
        store.insert_component(entity, value).unwrap();
    }
    let id = component_id::<u64>();
    let old = store
        .replace_component_value(target, id, ErasedComponentValue::new(21_u64))
        .unwrap();
    let original_location = store.get_entity_location(target).unwrap();
    store.despawn(first).unwrap();
    assert_ne!(
        store.get_entity_location(target).unwrap(),
        original_location
    );
    store.insert_component(target, true).unwrap();
    store.replace_component_value(target, id, old).unwrap();
    assert_eq!(store.get::<u64>(target), Some(&20));
    assert_eq!(store.get::<bool>(target), Some(&true));

    let old = store.clone_component_value(target, id).unwrap();
    store.despawn(target).unwrap();
    let replacement = EntityId::from_u128(3);
    store.spawn(replacement).unwrap();
    store.insert_component(replacement, 30_u64).unwrap();
    assert!(store.replace_component_value(target, id, old).is_err());
    assert_eq!(store.get::<u64>(replacement), Some(&30));
}

#[derive(
    Debug,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
struct RejectClone(u64);

impl Clone for RejectClone {
    fn clone(&self) -> Self {
        panic!("schema copying must never clone values")
    }
}

#[test]
fn empty_schema_and_snapshot_restore_do_not_clone_live_values() {
    let mut source = Store::new();
    source.register_component::<RejectClone>().unwrap();
    let entity = EntityId::from_u128(1);
    source.spawn(entity).unwrap();
    source.insert_component(entity, RejectClone(7)).unwrap();
    let old = source
        .replace_component_value(
            entity,
            component_id::<RejectClone>(),
            ErasedComponentValue::new(RejectClone(8)),
        )
        .unwrap();
    assert_eq!(old.downcast_ref::<RejectClone>().unwrap().0, 7);
    source
        .replace_component_value(entity, component_id::<RejectClone>(), old)
        .unwrap();
    let snapshot = source.snapshot().unwrap();
    let mut candidate = source.empty_with_schema();
    assert_eq!(candidate.active_entities(), 0);
    assert_eq!(candidate.len(), 0);
    candidate.restore_snapshot(&snapshot).unwrap();
    assert_eq!(candidate.get::<RejectClone>(entity).unwrap().0, 7);
    source.restore_snapshot(&snapshot).unwrap();
    assert_eq!(source.get::<RejectClone>(entity).unwrap().0, 7);
}

#[test]
fn explicit_encoding_matches_component_codec() {
    let value = ErasedComponentValue::new(42_u64);
    assert_eq!(
        value.encode().unwrap(),
        bincode::serialize(&42_u64).unwrap()
    );
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DropMutationTracker {
    #[serde(skip)]
    drop_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    mutable_state: u64,
}

impl Drop for DropMutationTracker {
    fn drop(&mut self) {
        // Mutate self in drop to ensure mutable pointer is valid
        self.mutable_state = self.mutable_state.wrapping_add(100);
        self.drop_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LargeDropMutationTracker {
    #[serde(skip)]
    drop_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    data: ([u8; 32], [u8; 16]),
}

impl Drop for LargeDropMutationTracker {
    fn drop(&mut self) {
        self.data.0[0] = self.data.0[0].wrapping_add(1);
        self.drop_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[test]
fn test_svo_into_typed_success_transfers_ownership_without_premature_drop() {
    let drop_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tracker = DropMutationTracker {
        drop_count: drop_counter.clone(),
        mutable_state: 42,
    };
    let erased = ErasedComponentValue::new(tracker);
    assert_eq!(drop_counter.load(std::sync::atomic::Ordering::SeqCst), 0);

    // 成功提取所有权
    let extracted: DropMutationTracker = erased.into_typed().expect("must downcast to DropMutationTracker");
    // 关键断言：erased 容器 drop 不应导致内部值被提前析构
    assert_eq!(drop_counter.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(extracted.mutable_state, 42);

    // 提取出来的实际值被 drop 后，计数器才递增为 1
    drop(extracted);
    assert_eq!(drop_counter.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn test_svo_into_typed_error_branch_drops_safely_with_mutable_pointer() {
    let drop_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tracker = DropMutationTracker {
        drop_count: drop_counter.clone(),
        mutable_state: 10,
    };
    let erased = ErasedComponentValue::new(tracker);
    assert_eq!(drop_counter.load(std::sync::atomic::Ordering::SeqCst), 0);

    // 尝试提取为不匹配的类型 (u64)，进入错误分支
    let res: Result<u64, _> = erased.into_typed();
    assert!(res.is_err());
    // 关键断言：错误分支必须安全且完整地执行析构，且计数恰好为 1
    assert_eq!(drop_counter.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn test_svo_into_typed_heap_error_and_success_lifecycle() {
    let drop_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let large = LargeDropMutationTracker {
        drop_count: drop_counter.clone(),
        data: ([1; 32], [1; 16]),
    };
    let erased = ErasedComponentValue::new(large);
    assert_eq!(drop_counter.load(std::sync::atomic::Ordering::SeqCst), 0);

    // 堆回退类型的错误提取分支
    let res: Result<u64, _> = erased.into_typed();
    assert!(res.is_err());
    assert_eq!(drop_counter.load(std::sync::atomic::Ordering::SeqCst), 1);

    // 堆回退类型的成功提取分支
    let large2 = LargeDropMutationTracker {
        drop_count: drop_counter.clone(),
        data: ([2; 32], [2; 16]),
    };
    let erased2 = ErasedComponentValue::new(large2);
    let extracted: LargeDropMutationTracker = erased2.into_typed().expect("heap downcast must succeed");
    assert_eq!(drop_counter.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(extracted.data.0[0], 2);
    drop(extracted);
    assert_eq!(drop_counter.load(std::sync::atomic::Ordering::SeqCst), 2);
}

