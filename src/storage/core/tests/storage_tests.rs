//! Storage Core 单元与集成测试套件 (EntityAllocator, Archetype, Store, Snapshot)

use crate::EntityId;
use crate::storage::Store;
use crate::storage::EntityAllocator;
use serde::{Deserialize, Serialize};

#[derive(
    Debug,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
struct Position {
    pub x: f64,
    pub y: f64,
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
struct Velocity {
    pub vx: f64,
    pub vy: f64,
}

#[test]
fn point_upsert_preserves_replacement_preflight_on_corrupt_state() {
    use crate::id::ComponentTypeId;
    use crate::storage::allocator::EntityLocation;
    use crate::storage::column::TypedColumn;

    let id = EntityId::from_u128(8301);
    let mut base = Store::new();
    base.insert((id, (11u64, 22i64))).unwrap();
    let location = base.allocator.get_location(id).unwrap();
    let cid = ComponentTypeId::of::<u64>();
    for case in 0..7 {
        let mut store = base.clone();
        match case {
            0 => store.allocator.set_location(id, EntityLocation::new(location.archetype_id, 999)).unwrap(),
            1 => store.archetype_index.clear(),
            2 => { store.archetypes[location.archetype_id as usize].columns_mut().remove(&cid); }
            3 => {
                let col = store.archetypes[location.archetype_id as usize].columns_mut().get_mut(&cid).unwrap();
                col.as_any_mut().downcast_mut::<TypedColumn<u64>>().unwrap().swap_remove(0);
            }
            4 => {
                let mut wrong = TypedColumn::<i64>::new(std::any::type_name::<u64>());
                wrong.push(99);
                store.archetypes[location.archetype_id as usize].columns_mut().insert(cid, Box::new(wrong));
            }
            5 => {
                // Correct native T is not enough: forged logical metadata must
                // still fail, even when the typed upsert has a known type name.
                let mut wrong = TypedColumn::<u64>::new("forged::logical::name");
                wrong.push(99);
                store.archetypes[location.archetype_id as usize].columns_mut().insert(cid, Box::new(wrong));
            }
            _ => store.epoch = u64::MAX,
        }
        let epoch = store.epoch();
        // Untouched replacement entry retains the original hash-based preflight.
        let mut reference = store.clone();
        // Clone deliberately starts a new store epoch; reproduce the tested
        // revision explicitly so the exhaustion case compares the same state.
        reference.epoch = epoch;
        let expected_error = reference.replace_component(id, 100u64).unwrap_err();
        let actual_error = store.upsert_component(id, 100u64).unwrap_err();
        assert_eq!(actual_error, expected_error, "error parity: {case}");
        assert_eq!(reference.epoch(), epoch, "reference preflight: {case}");
        assert_eq!(store.epoch(), epoch, "preflight failure must not advance epoch: {case}");
    }
}

#[test]
fn test_allocator_generation_aba_prevention() {
    let mut alloc = EntityAllocator::new();

    // 1. 分配第一个实体
    let e1 = EntityId::from_u128(1);
    let local1 = alloc.allocate(e1).unwrap();
    assert_eq!(local1.slot, 0);
    assert_eq!(local1.generation, 1);
    assert!(alloc.is_alive(e1));

    // 2. 释放 e1
    let loc = alloc.deallocate(e1).unwrap();
    assert_eq!(loc, None);
    assert!(!alloc.is_alive(e1));

    // 3. 再次分配，复用 slot 0，但代际递增为 2
    let e2 = EntityId::from_u128(2);
    let local2 = alloc.allocate(e2).unwrap();
    assert_eq!(local2.slot, 0);
    assert_eq!(local2.generation, 2);
    assert!(alloc.is_alive(e2));

    // 4. 使用旧的 e1 句柄访问必须失败
    assert!(!alloc.is_alive(e1));
    assert!(alloc.deallocate(e1).is_err());
    assert_eq!(alloc.entity_for_slot(local1), None);
    assert_eq!(alloc.entity_for_slot(local2), Some(e2));
}

#[test]
fn test_columnar_store_archetype_migration_and_queries() {
    let mut store = Store::new();

    let e1 = store
        .spawn(crate::EntityId::from_uuid(uuid::Uuid::new_v4()))
        .unwrap();
    let e2 = store
        .spawn(crate::EntityId::from_uuid(uuid::Uuid::new_v4()))
        .unwrap();

    // 1. 插入 Position 组件 (从 Archetype 0 -> Archetype {Position})
    store
        .insert_component(e1, Position { x: 10.0, y: 20.0 })
        .unwrap();
    store
        .insert_component(e2, Position { x: 30.0, y: 40.0 })
        .unwrap();

    assert_eq!(
        store.get::<Position>(e1),
        Some(&Position { x: 10.0, y: 20.0 })
    );
    assert_eq!(
        store.get::<Position>(e2),
        Some(&Position { x: 30.0, y: 40.0 })
    );
    assert_eq!(store.get::<Velocity>(e1), None);

    // 2. 为 e1 插入 Velocity 组件 (触发 Archetype 迁移: {Position} -> {Position, Velocity})
    store
        .insert_component(e1, Velocity { vx: 1.0, vy: 2.0 })
        .unwrap();

    assert_eq!(
        store.get::<Position>(e1),
        Some(&Position { x: 10.0, y: 20.0 })
    );
    assert_eq!(
        store.get::<Velocity>(e1),
        Some(&Velocity { vx: 1.0, vy: 2.0 })
    );
    assert_eq!(store.get::<Velocity>(e2), None);
}

#[test]
fn test_columnar_store_swap_remove_location_repair() {
    let mut store = Store::new();

    let e1 = store
        .spawn(crate::EntityId::from_uuid(uuid::Uuid::new_v4()))
        .unwrap();
    let e2 = store
        .spawn(crate::EntityId::from_uuid(uuid::Uuid::new_v4()))
        .unwrap();
    let e3 = store
        .spawn(crate::EntityId::from_uuid(uuid::Uuid::new_v4()))
        .unwrap();

    store
        .insert_component(e1, Position { x: 1.0, y: 1.0 })
        .unwrap();
    store
        .insert_component(e2, Position { x: 2.0, y: 2.0 })
        .unwrap();
    store
        .insert_component(e3, Position { x: 3.0, y: 3.0 })
        .unwrap();

    // 销毁中间实体 e2，e3 将被 swap-remove 移动到 e2 的行位置
    store.despawn(e2).unwrap();
    assert!(!store.is_alive(e2));
    assert_eq!(store.get::<Position>(e2), None);

    // 验证 e1 与 e3 依然能精确查询
    assert_eq!(
        store.get::<Position>(e1),
        Some(&Position { x: 1.0, y: 1.0 })
    );
    assert_eq!(
        store.get::<Position>(e3),
        Some(&Position { x: 3.0, y: 3.0 })
    );
}

#[test]
fn test_columnar_store_snapshot_serialization_roundtrip() {
    let mut store = Store::new();
    store.register_component::<Position>().unwrap();
    store.register_component::<Velocity>().unwrap();

    let e1 = store
        .spawn(crate::EntityId::from_uuid(uuid::Uuid::new_v4()))
        .unwrap();
    let e2 = store
        .spawn(crate::EntityId::from_uuid(uuid::Uuid::new_v4()))
        .unwrap();

    store
        .insert_component(e1, Position { x: 100.0, y: 200.0 })
        .unwrap();
    store
        .insert_component(e2, Position { x: 300.0, y: 400.0 })
        .unwrap();
    store
        .insert_component(e2, Velocity { vx: 5.0, vy: 6.0 })
        .unwrap();

    let snapshot = store.snapshot().unwrap();
    assert_eq!(snapshot.allocator.active_count(), 2);

    // 序列化为二进制
    let encoded = bincode::serialize(&snapshot).unwrap();
    assert!(!encoded.is_empty());

    // 反序列化恢复
    let decoded: crate::storage::ColumnarStoreSnapshot = bincode::deserialize(&encoded).unwrap();
    assert_eq!(decoded.allocator.active_count(), 2);
    assert_eq!(decoded.archetypes_data.len(), 3); // 0(Empty), 1({Pos}), 2({Pos, Vel})
}

#[test]
fn test_allocator_generation_overflow_fail_closed() {
    use crate::error::StateError;

    let mut alloc = EntityAllocator::new();
    let e1 = EntityId::from_u128(3);
    let local = alloc.allocate(e1).unwrap();
    assert_eq!(local.generation, 1);

    // 序列化后修改代际值为 u32::MAX
    let encoded = bincode::serialize(&alloc).unwrap();
    let mut modified = encoded.clone();
    // bincode 序列化布局：slots vec 长度 (8 bytes) + slots[0].generation (4 bytes)
    modified[8..12].copy_from_slice(&u32::MAX.to_le_bytes());

    let mut alloc_max: EntityAllocator = bincode::deserialize(&modified).unwrap();
    assert!(alloc_max.is_alive(e1));
    assert_eq!(alloc_max.local_slot(e1).unwrap().generation, u32::MAX);
    assert_eq!(alloc_max.active_count(), 1);

    // 执行 deallocate 必须触发 GenerationOverflow 错误
    let before = bincode::serialize(&alloc_max).unwrap();
    let res = alloc_max.deallocate(e1);
    assert_eq!(res, Err(StateError::GenerationOverflow));

    // 状态零突变验证 (Fail-Closed: 实体依然保持活跃，活跃数不变，未进入空闲列表)
    assert!(alloc_max.is_alive(e1));
    assert_eq!(alloc_max.active_count(), 1);
    assert_eq!(bincode::serialize(&alloc_max).unwrap(), before);
}

#[test]
fn test_columnar_store_remove_component_archetype_migration() {
    let mut store = Store::new();

    let e1 = store
        .spawn(crate::EntityId::from_uuid(uuid::Uuid::new_v4()))
        .unwrap();
    let e2 = store
        .spawn(crate::EntityId::from_uuid(uuid::Uuid::new_v4()))
        .unwrap();

    store
        .insert_component(e1, Position { x: 10.0, y: 20.0 })
        .unwrap();
    store
        .insert_component(e1, Velocity { vx: 1.0, vy: 2.0 })
        .unwrap();
    store
        .insert_component(e2, Position { x: 30.0, y: 40.0 })
        .unwrap();

    assert_eq!(
        store.get::<Position>(e1),
        Some(&Position { x: 10.0, y: 20.0 })
    );
    assert_eq!(
        store.get::<Velocity>(e1),
        Some(&Velocity { vx: 1.0, vy: 2.0 })
    );

    // 移除 e1 的 Velocity 组件 (触发 Archetype 迁移: {Position, Velocity} -> {Position})
    store.remove_component::<Velocity>(e1).unwrap();
    assert_eq!(
        store.get::<Position>(e1),
        Some(&Position { x: 10.0, y: 20.0 })
    );
    assert_eq!(store.get::<Velocity>(e1), None);

    // 移除不存在的组件返回 ComponentNotFound
    assert!(store.remove_component::<Velocity>(e1).is_err());

    // 移除 e1 的 Position 组件 (触发 Archetype 迁移: {Position} -> {})
    store.remove_component::<Position>(e1).unwrap();
    assert_eq!(store.get::<Position>(e1), None);
    assert!(store.is_alive(e1));
}

#[test]
fn stable_identity_has_independent_locations_and_components_in_two_stores() {
    let entity = EntityId::from_u128(100);
    let other = EntityId::from_u128(101);
    let mut service = Store::new();
    let mut model = Store::new();
    service.register_component::<Position>().unwrap();
    model.register_component::<Position>().unwrap();
    service.spawn(entity).unwrap();
    model.spawn(other).unwrap();
    model.spawn(entity).unwrap();
    service
        .insert_component(entity, Position { x: 1.0, y: 2.0 })
        .unwrap();
    model
        .upsert_component(entity, Position { x: 10.0, y: 20.0 })
        .unwrap();

    let service_local = service
        .snapshot()
        .unwrap()
        .allocator
        .local_slot(entity)
        .unwrap();
    let model_local = model
        .snapshot()
        .unwrap()
        .allocator
        .local_slot(entity)
        .unwrap();
    assert_ne!(service_local.slot, model_local.slot);
    service.despawn(entity).unwrap();
    assert!(!service.is_alive(entity));
    assert_eq!(
        model.get::<Position>(entity),
        Some(&Position { x: 10.0, y: 20.0 })
    );

    // Recreating the service facet preserves identity, not its old physical handle.
    assert_eq!(service.spawn(entity).unwrap(), entity);
    let recreated = service.snapshot().unwrap().allocator;
    let new_local = recreated.local_slot(entity).unwrap();
    assert_eq!(new_local.slot, service_local.slot);
    assert!(new_local.generation > service_local.generation);
    assert_eq!(recreated.entity_for_slot(service_local), None);
    assert_eq!(recreated.entity_for_slot(new_local), Some(entity));
    assert_eq!(service.get::<Position>(entity), None);
    assert_eq!(
        model.snapshot().unwrap().allocator.local_slot(entity),
        Some(model_local)
    );
}

#[test]
fn allocator_rejects_nil_duplicate_and_foreign_id_without_consuming_a_slot() {
    let mut allocator = EntityAllocator::new();
    let before = bincode::serialize(&allocator).unwrap();
    assert!(allocator.allocate(EntityId::default()).is_err());
    assert_eq!(bincode::serialize(&allocator).unwrap(), before);
    let entity = EntityId::from_u128(200);
    let local = allocator.allocate(entity).unwrap();
    let before = bincode::serialize(&allocator).unwrap();
    assert!(allocator.allocate(entity).is_err());
    assert!(allocator.deallocate(EntityId::from_u128(201)).is_err());
    assert_eq!(allocator.local_slot(entity), Some(local));
    assert_eq!(bincode::serialize(&allocator).unwrap(), before);
}

#[test]
fn allocator_snapshot_rejects_duplicate_id_free_overlap_and_zero_generation() {
    let mut allocator = EntityAllocator::new();
    allocator.allocate(EntityId::from_u128(301)).unwrap();
    allocator.allocate(EntityId::from_u128(302)).unwrap();
    let original = serde_json::to_value(&allocator).unwrap();
    let mut duplicate = original.clone();
    duplicate["slots"][1]["entity"] = duplicate["slots"][0]["entity"].clone();
    let mut free_overlap = original.clone();
    free_overlap["free_slots"] = serde_json::json!([0]);
    let mut zero_generation = original.clone();
    zero_generation["slots"][0]["generation"] = serde_json::json!(0);
    let mut unlisted_free = original.clone();
    unlisted_free["slots"][0]["entity"] = serde_json::Value::Null;
    for corrupted in [duplicate, free_overlap, zero_generation, unlisted_free] {
        assert!(serde_json::from_value::<EntityAllocator>(corrupted).is_err());
    }
    let decoded: EntityAllocator = serde_json::from_value(original).unwrap();
    assert_eq!(decoded, allocator);
}

#[test]
fn typed_snapshot_restore_is_detached_and_validates_identity_column_bijections() {
    use crate::id::ComponentTypeId;
    use crate::storage::EntityLocation;

    let mut store = Store::new();
    store.register_component::<Position>().unwrap();
    let first = EntityId::from_u128(401);
    let second = EntityId::from_u128(402);
    store.spawn(first).unwrap();
    store.spawn(second).unwrap();
    store
        .insert_component(first, Position { x: 1.0, y: 2.0 })
        .unwrap();
    store
        .insert_component(second, Position { x: 3.0, y: 4.0 })
        .unwrap();
    let valid = store.snapshot().unwrap();
    let type_id = ComponentTypeId::from_name(std::any::type_name::<Position>());
    let before = bincode::serialize(&valid).unwrap();
    let mut corruptions = Vec::new();

    let mut bad = valid.clone();
    bad.allocator
        .set_location(first, EntityLocation::new(1, 1))
        .unwrap();
    corruptions.push(bad);
    let mut bad = valid.clone();
    bad.archetypes_data[1].entities[1] = first;
    corruptions.push(bad);
    let mut bad = valid.clone();
    bad.archetypes_data[1]
        .columns
        .get_mut(&type_id)
        .unwrap()
        .row_count -= 1;
    corruptions.push(bad);
    let mut bad = valid.clone();
    bad.archetypes_data[1].columns.clear();
    corruptions.push(bad);
    let mut bad = valid.clone();
    bad.archetypes_data[1].columns.insert(
        ComponentTypeId::new(991),
        crate::storage::ArchivedColumnData::new(
            ComponentTypeId::new(991),
            1,
            2,
            crate::storage::ColumnCodecId::Rkyv,
            vec![1, 2],
        ),
    );
    corruptions.push(bad);
    let mut bad = valid.clone();
    bad.archetypes_data[1].component_types.push(type_id);
    corruptions.push(bad);
    let mut bad = valid.clone();
    bad.archetypes_data[1].archetype_id = 0;
    corruptions.push(bad);
    let mut bad = valid.clone();
    bad.allocator.allocate(EntityId::from_u128(403)).unwrap();
    corruptions.push(bad);
    let mut bad = valid.clone();
    bad.archetypes_data[1]
        .columns
        .get_mut(&type_id)
        .unwrap()
        .payload = vec![0xFF, 0xFF, 0xFF, 0xFF];
    corruptions.push(bad);

    for bad in corruptions {
        assert!(store.restore_snapshot(&bad).is_err());
        assert_eq!(
            bincode::serialize(&store.snapshot().unwrap()).unwrap(),
            before
        );
    }
    let mut cold_without_prototype = Store::new();
    assert!(cold_without_prototype.restore_snapshot(&valid).is_err());
    assert!(cold_without_prototype.is_empty());
    assert_eq!(cold_without_prototype.len(), 0);

    store
        .replace_component(first, Position { x: 99.0, y: 100.0 })
        .unwrap();
    store.restore_snapshot(&valid).unwrap();
    assert_eq!(
        bincode::serialize(&store.snapshot().unwrap()).unwrap(),
        before
    );
}

#[test]
fn test_modify_epoch_exhausted_rejects() {
    let mut store = Store::new();
    let e1 = store
        .spawn(EntityId::from_u128(1))
        .unwrap();
    store.insert_component(e1, Position { x: 1.0, y: 2.0 }).unwrap();

    // 强行将代际设为 u64::MAX
    store.epoch = u64::MAX;

    let mut called = false;
    let res = store.modify::<Position, _>(e1, |_| {
        called = true;
    });

    assert!(!called, "Closure must never be executed on capacity exhausted");
    assert!(matches!(res, Err(crate::error::StateError::CapacityExhausted(_))));
    assert_eq!(store.epoch, u64::MAX);
}

#[test]
fn test_modify_corrupted_row_or_short_column_access_denied() {
    use crate::storage::allocator::EntityLocation;
    use crate::storage::column::TypedColumn;
    use crate::id::ComponentTypeId;

    let mut store = Store::new();
    let e1 = store
        .spawn(EntityId::from_u128(1))
        .unwrap();
    store.insert_component(e1, Position { x: 10.0, y: 20.0 }).unwrap();

    let initial_epoch = store.epoch();

    // 1. 注入损坏行号 (inconsistent row)
    store.allocator.set_location(e1, EntityLocation::new(1, 9999)).unwrap();
    let mut called1 = false;
    let res1 = store.modify::<Position, _>(e1, |_| {
        called1 = true;
    });
    assert!(!called1, "Closure must never be called when row is inconsistent");
    assert!(matches!(res1, Err(crate::error::StateError::AccessDenied(ref msg)) if msg.contains("inconsistent archetype row")));
    assert_eq!(store.epoch(), initial_epoch, "Epoch must not advance on precheck failure");

    // 2. 注入损坏的 archetype_id (unknown archetype)
    store.allocator.set_location(e1, EntityLocation::new(999, 0)).unwrap();
    let mut called2 = false;
    let res2 = store.modify::<Position, _>(e1, |_| {
        called2 = true;
    });
    assert!(!called2, "Closure must never be called when archetype is unknown");
    assert!(matches!(res2, Err(crate::error::StateError::AccessDenied(ref msg)) if msg.contains("unknown archetype")));
    assert_eq!(store.epoch(), initial_epoch, "Epoch must not advance on precheck failure");

    // 3. 注入短列 (short column: archetype 有实体，但底层 TypedColumn 被截断为空)
    store.allocator.set_location(e1, EntityLocation::new(1, 0)).unwrap();
    let col = store.archetypes[1].columns_mut().get_mut(&ComponentTypeId::of::<Position>()).unwrap();
    let typed = col.as_any_mut().downcast_mut::<TypedColumn<Position>>().unwrap();
    typed.swap_remove(0); // 截断列数据，使其长度为 0
    assert_eq!(typed.len(), 0);
    assert_eq!(store.archetypes[1].entities().len(), 1);

    let mut called3 = false;
    let res3 = store.modify::<Position, _>(e1, |_| {
        called3 = true;
    });
    assert!(!called3, "Closure must never be called on short column");
    assert!(matches!(res3, Err(crate::error::StateError::AccessDenied(_))));
    assert_eq!(store.epoch(), initial_epoch, "Epoch must not advance on precheck failure");
}

#[test]
fn test_columnar_snapshot_archived_column_metadata_and_corruption_checks() {
    use crate::id::ComponentTypeId;
    use crate::storage::{ColumnCodecId, ComponentRegistration};

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct CustomRkyvData {
        val: u64,
    }

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct CustomSerdeData {
        text: String,
    }

    let mut store = Store::new();
    store
        .register_components([
            ComponentRegistration::of_rkyv::<CustomRkyvData>(),
            ComponentRegistration::of::<CustomSerdeData>(),
        ])
        .unwrap();

    let e1 = EntityId::from_u128(901);
    let e2 = EntityId::from_u128(902);
    store.spawn(e1).unwrap();
    store.spawn(e2).unwrap();
    store
        .insert_component(e1, CustomRkyvData { val: 42 })
        .unwrap();
    store
        .insert_component(e2, CustomRkyvData { val: 84 })
        .unwrap();
    store
        .insert_component(
            e1,
            CustomSerdeData {
                text: "hello".into(),
            },
        )
        .unwrap();
    store
        .insert_component(
            e2,
            CustomSerdeData {
                text: "world".into(),
            },
        )
        .unwrap();

    let snap = store.snapshot().unwrap();
    let rkyv_type = ComponentTypeId::from_name(std::any::type_name::<CustomRkyvData>());
    let serde_type = ComponentTypeId::from_name(std::any::type_name::<CustomSerdeData>());

    // 1. 验证快照元数据结构正确性
    let (arch_idx, arch_data) = snap
        .archetypes_data
        .iter()
        .enumerate()
        .find(|(_, a)| a.component_types.contains(&rkyv_type) && a.component_types.contains(&serde_type))
        .unwrap();

    let rkyv_col = arch_data.columns.get(&rkyv_type).unwrap();
    assert_eq!(rkyv_col.codec, ColumnCodecId::Rkyv);
    assert_eq!(rkyv_col.row_count, 2);
    assert_eq!(rkyv_col.schema_version, 1);

    let serde_col = arch_data.columns.get(&serde_type).unwrap();
    assert_eq!(serde_col.codec, ColumnCodecId::Rkyv);
    assert_eq!(serde_col.row_count, 2);
    assert_eq!(serde_col.schema_version, 1);

    // 2. 破坏性测试：篡改 row_count
    let mut bad_row = snap.clone();
    bad_row.archetypes_data[arch_idx]
        .columns
        .get_mut(&rkyv_type)
        .unwrap()
        .row_count = 1;
    assert!(store.restore_snapshot(&bad_row).is_err());

    // 3. 破坏性测试：损坏 payload 字节
    let mut bad_payload = snap.clone();
    bad_payload.archetypes_data[arch_idx]
        .columns
        .get_mut(&rkyv_type)
        .unwrap()
        .payload = vec![0xBA, 0xAD, 0xF0, 0x0D];
    assert!(store.restore_snapshot(&bad_payload).is_err());

    // 4. 正常恢复校验
    let mut clean_store = Store::new();
    clean_store
        .register_components([
            ComponentRegistration::of_rkyv::<CustomRkyvData>(),
            ComponentRegistration::of::<CustomSerdeData>(),
        ])
        .unwrap();
    clean_store.restore_snapshot(&snap).unwrap();

    assert_eq!(
        clean_store.get::<CustomRkyvData>(e1),
        Some(&CustomRkyvData { val: 42 })
    );
    assert_eq!(
        clean_store.get::<CustomSerdeData>(e2),
        Some(&CustomSerdeData {
            text: "world".into()
        })
    );
}

#[test]
fn test_registration_rejects_conflicting_component_schema_version() {
    use crate::storage::column::TypedColumn;
    use crate::storage::{ComponentRegistration, Store};

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct RkyvPos {
        x: f64,
    }

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct SerdeVel {
        v: f64,
    }

    let mut store = Store::new();
    // 第一次注册：以默认模式版本 (1) 注册 RkyvPos
    store
        .register_components([ComponentRegistration::of_rkyv::<RkyvPos>()])
        .unwrap();

    // 第二次注册：试图在包含 [RkyvPos (版本 2), SerdeVel] 的原型组合中注册 RkyvPos
    let mut vtable_v2 = crate::storage::column::ColumnArchiveVTable::new_rkyv();
    vtable_v2.schema_version = 2;
    let conflict_pos = ComponentRegistration {
        column: Box::new(TypedColumn::<RkyvPos>::new_with_vtable(
            std::any::type_name::<RkyvPos>(),
            vtable_v2,
        )),
    };

    let err = store
        .register_components([
            conflict_pos,
            ComponentRegistration::of::<SerdeVel>(),
        ])
        .unwrap_err();

    let msg = err.to_string();
    assert!(
        msg.contains("conflicts with existing registered version"),
        "Unexpected error message: {msg}"
    );

    // 验证 Store 未被污染（依然只有第 0 个空原型，注册表仅有首个组件）
    assert_eq!(store.archetypes.len(), 1);
    assert_eq!(store.component_registry.len(), 1);
}

#[test]
fn test_multi_archetype_same_component_snapshot_roundtrip() {
    use crate::storage::{ComponentRegistration, Store};

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct MultiArchPos {
        x: f64,
    }

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct MultiArchVel {
        v: f64,
    }

    let mut store = Store::new();
    // 统一契约：多个原型使用一致的 codec (rkyv) 注册同一组件
    store
        .register_components([ComponentRegistration::of_rkyv::<MultiArchPos>()])
        .unwrap();
    store
        .register_components([
            ComponentRegistration::of_rkyv::<MultiArchPos>(),
            ComponentRegistration::of::<MultiArchVel>(),
        ])
        .unwrap();

    let e1 = EntityId::from_u128(1001);
    let e2 = EntityId::from_u128(1002);

    store.spawn(e1).unwrap();
    store
        .insert_component(e1, MultiArchPos { x: 111.0 })
        .unwrap();

    store.spawn(e2).unwrap();
    store
        .insert_component(e2, MultiArchPos { x: 222.0 })
        .unwrap();
    store
        .insert_component(e2, MultiArchVel { v: 333.0 })
        .unwrap();

    // 生成快照
    let snapshot = store.snapshot().unwrap();

    // 在同结构新 Store 中往返恢复
    let mut clean_store = Store::new();
    clean_store
        .register_components([ComponentRegistration::of_rkyv::<MultiArchPos>()])
        .unwrap();
    clean_store
        .register_components([
            ComponentRegistration::of_rkyv::<MultiArchPos>(),
            ComponentRegistration::of::<MultiArchVel>(),
        ])
        .unwrap();

    clean_store.restore_snapshot(&snapshot).unwrap();

    assert_eq!(
        clean_store.get::<MultiArchPos>(e1),
        Some(&MultiArchPos { x: 111.0 })
    );
    assert_eq!(
        clean_store.get::<MultiArchPos>(e2),
        Some(&MultiArchPos { x: 222.0 })
    );
    assert_eq!(
        clean_store.get::<MultiArchVel>(e2),
        Some(&MultiArchVel { v: 333.0 })
    );
}

static SERDE_DESERIALIZE_COUNTER: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
struct CountingSerdeComponent {
    val: u64,
}

impl<'de> serde::Deserialize<'de> for CountingSerdeComponent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawPayload {
            val: u64,
        }
        let raw = RawPayload::deserialize(deserializer)?;
        SERDE_DESERIALIZE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(CountingSerdeComponent { val: raw.val })
    }
}

#[test]
fn test_serde_snapshot_restore_single_deserialization_count() {
    use crate::storage::{ComponentRegistration, Store};

    let mut store = Store::new();
    store
        .register_components([ComponentRegistration::of::<CountingSerdeComponent>()])
        .unwrap();

    let n: usize = 10;
    for i in 0..n {
        let e = EntityId::from_u128(2000 + i as u128);
        store.spawn(e).unwrap();
        store
            .insert_component(e, CountingSerdeComponent { val: i as u64 })
            .unwrap();
    }

    let snapshot = store.snapshot().unwrap();

    let mut clean_store = store.empty_with_schema();
    clean_store.restore_snapshot(&snapshot).unwrap();

    // 验证恢复数据正确性
    for i in 0..n {
        let e = EntityId::from_u128(2000 + i as u128);
        assert_eq!(
            clean_store.get::<CountingSerdeComponent>(e),
            Some(&CountingSerdeComponent { val: i as u64 })
        );
    }
}

#[test]
fn test_serde_snapshot_restore_failure_leaves_store_unmutated() {
    use crate::id::ComponentTypeId;
    use crate::storage::{ComponentRegistration, Store};

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct FailureTestComponent {
        text: String,
    }

    let mut store = Store::new();
    store
        .register_components([ComponentRegistration::of::<FailureTestComponent>()])
        .unwrap();

    let e = EntityId::from_u128(9999);
    store.spawn(e).unwrap();
    store
        .insert_component(
            e,
            FailureTestComponent {
                text: "initial".into(),
            },
        )
        .unwrap();

    let original_epoch = store.epoch();
    let original_val = store.get::<FailureTestComponent>(e).cloned();

    // 构造损坏 payload 的快照
    let mut bad_snap = store.snapshot().unwrap();
    let serde_type = ComponentTypeId::from_name(std::any::type_name::<FailureTestComponent>());
    bad_snap.archetypes_data[1]
        .columns
        .get_mut(&serde_type)
        .unwrap()
        .payload = vec![0xFF, 0xFF, 0x00];

    // 恢复必须失败
    assert!(store.restore_snapshot(&bad_snap).is_err());

    // 核心断言：原 Store 逐字节绝对零突变，epoch 不变，数据不受污染
    assert_eq!(store.epoch(), original_epoch);
    assert_eq!(store.get::<FailureTestComponent>(e).cloned(), original_val);
}

#[test]
fn test_unified_replace_in_place_existing_and_epoch() {
    use crate::storage::{ComponentRegistration, Store};

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct Score(u32);

    let mut store = Store::new();
    store.register_components([ComponentRegistration::of::<Score>()]).unwrap();

    let e = EntityId::from_u128(5001);
    store.spawn(e).unwrap();
    store.insert_component(e, Score(10)).unwrap();

    let epoch0 = store.epoch();

    // 1. replace_component 测试
    let old1 = store.replace_component(e, Score(20)).unwrap();
    assert_eq!(old1, Score(10));
    assert_eq!(store.get::<Score>(e), Some(&Score(20)));
    assert_eq!(store.epoch(), epoch0 + 1);

    // 2. upsert_component (已有组件分支) 测试
    let old2 = store.upsert_component(e, Score(30)).unwrap();
    assert_eq!(old2, Some(Score(20)));
    assert_eq!(store.get::<Score>(e), Some(&Score(30)));
    assert_eq!(store.epoch(), epoch0 + 2);

    // 3. EntityMut::set (已有组件分支) 测试
    {
        let mut em = store.entity_mut(e).unwrap();
        let old3 = em.set(Score(40)).unwrap();
        assert_eq!(old3, Some(Score(30)));
    }
    assert_eq!(store.get::<Score>(e), Some(&Score(40)));
    assert_eq!(store.epoch(), epoch0 + 3);
}

#[test]
fn test_unified_upsert_and_entity_mut_set_absent_fallback_to_insert() {
    use crate::storage::{ComponentRegistration, Store};

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct Level(u16);

    let mut store = Store::new();
    store.register_components([ComponentRegistration::of::<Level>()]).unwrap();

    let e1 = EntityId::from_u128(6001);
    let e2 = EntityId::from_u128(6002);
    store.spawn(e1).unwrap();
    store.spawn(e2).unwrap();

    let epoch0 = store.epoch();

    // 1. upsert_component 未持有组件安全回退至 insert
    let res1 = store.upsert_component(e1, Level(1)).unwrap();
    assert_eq!(res1, None);
    assert_eq!(store.get::<Level>(e1), Some(&Level(1)));
    assert_eq!(store.epoch(), epoch0 + 1);

    // 2. EntityMut::set 未持有组件安全回退至 insert 并刷新内部物理位置
    {
        let mut em = store.entity_mut(e2).unwrap();
        let res2 = em.set(Level(5)).unwrap();
        assert_eq!(res2, None);
    }
    assert_eq!(store.get::<Level>(e2), Some(&Level(5)));
    assert_eq!(store.epoch(), epoch0 + 2);
}

#[test]
fn test_unified_replace_component_failure_preserves_store_zero_mutation() {
    use crate::storage::{ComponentRegistration, Store};

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct Stamina(u32);

    let mut store = Store::new();
    store.register_components([ComponentRegistration::of::<Stamina>()]).unwrap();

    let e = EntityId::from_u128(7001);
    store.spawn(e).unwrap();

    let epoch_before = store.epoch();

    // 未持有组件调用 replace_component 必须报错 ComponentNotFoundInEntity
    let err = store.replace_component(e, Stamina(100));
    assert!(matches!(err, Err(crate::error::StateError::ComponentNotFoundInEntity { .. })));

    // 严格断言：代际不变，Store 零突变
    assert_eq!(store.epoch(), epoch_before);
    assert_eq!(store.get::<Stamina>(e), None);

    // 不存在实体调用 replace_component 必须报错 EntityNotFound
    let non_existent = EntityId::from_u128(99999);
    let err2 = store.replace_component(non_existent, Stamina(100));
    assert!(matches!(err2, Err(crate::error::StateError::EntityNotFound(_))));
    assert_eq!(store.epoch(), epoch_before);
}

#[test]
fn test_unified_replace_component_value_erased_single_resolution() {
    use crate::storage::{ComponentRegistration, Store};
    use crate::storage::ErasedComponentValue;

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct Speed(f32);

    let mut store = Store::new();
    store.register_components([ComponentRegistration::of::<Speed>()]).unwrap();

    let e = EntityId::from_u128(8001);
    store.spawn(e).unwrap();
    store.insert_component(e, Speed(1.5)).unwrap();

    let epoch0 = store.epoch();
    let type_id = crate::id::ComponentTypeId::of::<Speed>();

    // 成功替换
    let new_val = ErasedComponentValue::new(Speed(3.0));
    let old_val = store.replace_component_value(e, type_id, new_val).unwrap();
    assert_eq!(old_val.into_typed::<Speed>().unwrap(), Speed(1.5));
    assert_eq!(store.get::<Speed>(e), Some(&Speed(3.0)));
    assert_eq!(store.epoch(), epoch0 + 1);

    // 错误替换（实体不存在）
    let bad_e = EntityId::from_u128(99999);
    let bad_val = ErasedComponentValue::new(Speed(4.0));
    assert!(store.replace_component_value(bad_e, type_id, bad_val).is_err());
    assert_eq!(store.epoch(), epoch0 + 1);
}

#[test]
fn test_registration_staging_all_or_nothing_zero_mutation() {
    use crate::storage::column::TypedColumn;
    use crate::storage::{ComponentRegistration, Store};

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct StagedPos {
        x: f64,
    }

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct StagedVel {
        v: f64,
    }

    let mut store = Store::new();
    // 预先注册 StagedPos 为默认模式版本 (1)
    store
        .register_components([ComponentRegistration::of::<StagedPos>()])
        .unwrap();
    assert_eq!(store.component_registry.len(), 1);

    // 构造版本冲突原型 (schema_version = 2)
    let mut vtable_v2 = crate::storage::column::ColumnArchiveVTable::new_rkyv();
    vtable_v2.schema_version = 2;
    let conflict_pos = ComponentRegistration {
        column: Box::new(TypedColumn::<StagedPos>::new_with_vtable(
            std::any::type_name::<StagedPos>(),
            vtable_v2,
        )),
    };

    // 尝试原子注册批次：[全新的 StagedVel, 冲突的 StagedPos (version 2)]
    // StagedPos 与已有版本冲突，整个批次必须原子回滚，StagedVel 绝不能残留在注册表中
    let err = store.register_components([
        ComponentRegistration::of::<StagedVel>(),
        conflict_pos,
    ]);
    assert!(err.is_err());

    // 刚性断言：注册表维持且仅有 1 项，StagedVel 零突变零残留
    assert_eq!(store.component_registry.len(), 1);
    let vel_type = crate::id::ComponentTypeId::of::<StagedVel>();
    assert!(!store.component_registry.contains(&vel_type));
}

#[test]
fn test_snapshot_restore_non_aligned_rkyv_payload_succeeds() {
    use crate::storage::column::{rkyv_restore_column_from_slice, ColumnCodecId};
    use crate::storage::{ComponentRegistration, Store};

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct AlignedProbe {
        val: u64,
    }

    let mut store = Store::new();
    store
        .register_components([ComponentRegistration::of_rkyv::<AlignedProbe>()])
        .unwrap();

    let e = EntityId::from_u128(9001);
    store.spawn(e).unwrap();
    store.insert_component(e, AlignedProbe { val: 0xDEADBEEFCAFE }).unwrap();

    let snapshot = store.snapshot().unwrap();
    let probe_type = crate::id::ComponentTypeId::of::<AlignedProbe>();
    let original_payload = &snapshot.archetypes_data[1].columns[&probe_type].payload;

    // 1. 构造一个具有强制 16 字节对齐的大缓冲区，并在第 1 个字节处前置偏移 1 字节
    let mut aligned_buffer = rkyv::util::AlignedVec::<16>::new();
    aligned_buffer.push(0xAA); // 插入 1 字节破坏对齐
    aligned_buffer.extend_from_slice(original_payload);

    let align = core::mem::align_of::<rkyv::vec::ArchivedVec<rkyv::Archived<AlignedProbe>>>();
    let misaligned_slice = &aligned_buffer[1..];

    // 刚性断言前置条件：切片首地址模对齐值严格不为 0，物理证明输入处于非对齐状态！
    assert_ne!(
        misaligned_slice.as_ptr() as usize % align,
        0,
        "Precondition: slice pointer ({:p}) must not be aligned to {align}",
        misaligned_slice.as_ptr()
    );

    // 2. 通过包内解码辅助切片入口直接调用，验证走入 AlignedVec 缓冲自愈分支并成功解码
    let mut restored = Vec::new();
    rkyv_restore_column_from_slice::<AlignedProbe>(
        probe_type,
        1,
        ColumnCodecId::Rkyv,
        1,
        misaligned_slice,
        &mut restored,
    )
    .unwrap();

    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0], AlignedProbe { val: 0xDEADBEEFCAFE });

    // 3. 同时验证自然对齐切片正常恢复
    let natural_slice = original_payload.as_slice();
    let mut restored_natural = Vec::new();
    rkyv_restore_column_from_slice::<AlignedProbe>(
        probe_type,
        1,
        ColumnCodecId::Rkyv,
        1,
        natural_slice,
        &mut restored_natural,
    )
    .unwrap();
    assert_eq!(restored_natural, restored);
}

#[test]
fn test_snapshot_restore_tampered_row_count_and_entity_mismatch_rejected_zero_mutation() {
    use crate::storage::{ComponentRegistration, Store};

    #[derive(
        Debug,
        Clone,
        PartialEq,
        serde::Serialize,
        serde::Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct StrictRowData {
        val: u32,
    }

    let mut store = Store::new();
    store
        .register_components([ComponentRegistration::of_rkyv::<StrictRowData>()])
        .unwrap();

    let e1 = EntityId::from_u128(9101);
    let e2 = EntityId::from_u128(9102);
    store.spawn(e1).unwrap();
    store.spawn(e2).unwrap();
    store.insert_component(e1, StrictRowData { val: 1 }).unwrap();
    store.insert_component(e2, StrictRowData { val: 2 }).unwrap();

    let valid_snap = store.snapshot().unwrap();
    let original_epoch = store.epoch();
    let type_id = crate::id::ComponentTypeId::of::<StrictRowData>();

    // 1. 篡改 row_count 为 3 (与实际行数 2 和实体数 2 不一致)
    let mut bad_row_count = valid_snap.clone();
    bad_row_count.archetypes_data[1]
        .columns
        .get_mut(&type_id)
        .unwrap()
        .row_count = 3;
    assert!(store.restore_snapshot(&bad_row_count).is_err());
    assert_eq!(store.epoch(), original_epoch);

    // 2. 篡改 entities 实体数多出 1 个 (与列长 2 不一致)
    let mut bad_entities = valid_snap.clone();
    bad_entities.archetypes_data[1]
        .entities
        .push(EntityId::from_u128(9103));
    assert!(store.restore_snapshot(&bad_entities).is_err());
    assert_eq!(store.epoch(), original_epoch);

    // 验证原 Store 数据完好无损
    assert_eq!(store.get::<StrictRowData>(e1), Some(&StrictRowData { val: 1 }));
    assert_eq!(store.get::<StrictRowData>(e2), Some(&StrictRowData { val: 2 }));
}

#[test]
fn test_validate_archetype_layout_failure_does_not_cache_corrupted_state() {
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
    #[rkyv(crate = rkyv)]
    struct SampleVal(u64);

    let mut store = crate::storage::Store::new();
    store.register_components([crate::storage::ComponentRegistration::of_rkyv::<SampleVal>()]).unwrap();
    let e = EntityId::from_u128(1001);
    store.spawn(e).unwrap();
    store.insert_component(e, SampleVal(42)).unwrap();

    // Archetype 1 是有效的
    assert!(store.validate_archetype_layout(1).is_ok());
    // 再次调用，命中缓存，依然 ok
    assert!(store.validate_archetype_layout(1).is_ok());

    // 针对不存在的 Archetype
    assert!(store.validate_archetype_layout(999).is_err());
    // 再次调用，绝不因提前标记而放行
    assert!(store.validate_archetype_layout(999).is_err());
}


