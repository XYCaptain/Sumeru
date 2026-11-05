use super::*;
use crate::storage::EntityBundle;


#[test]
fn test_create_entity_failure_direct_archetype_inspection() {
    let mut store = Store::new();
    let nil_id = EntityId::from_u128(0);
    assert!(nil_id.is_nil());

    // 1. 向空 Store 插入 nil ID (使用空 bundle，对应 Archetype 0)
    let err = store.create_entity(nil_id, EntityBundle::new()).unwrap_err();
    assert!(matches!(err, StateError::AccessDenied(_)));

    // 白盒直接检查 store.archetypes：
    // Archetype 0 必须存在，但其实体行数和所有列必须保持为 0
    assert!(!store.archetypes.is_empty());
    assert_eq!(store.archetypes[0].len(), 0);
    assert_eq!(store.archetypes[0].entities().len(), 0);
    for col in store.archetypes[0].columns().values() {
        assert_eq!(col.len(), 0);
    }

    // 2. 预先注册一个包含有效组件的原型 (如 (TestPos, TestVel))
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct TestPos(pub i32);
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct TestVel(pub i32);

    let valid_id = EntityId::from_u128(999);
    store.insert((valid_id, (TestPos(10), TestVel(20)))).unwrap();
    assert_eq!(store.len(), 1);

    let loc = store.allocator.get_location(valid_id).unwrap();
    let target_arch_id = loc.archetype_id as usize;
    let initial_arch_len = store.archetypes[target_arch_id].len();
    assert_eq!(initial_arch_len, 1);

    // 3. 再次向该已知原型尝试插入 nil ID
    let err = store.insert((nil_id, (TestPos(99), TestVel(99)))).unwrap_err();
    assert!(matches!(err, StateError::AccessDenied(_)));

    // 白盒断言：目标 Archetype 的 entities 长度、物理列深度必须完全等于 initial_arch_len (1)
    assert_eq!(store.archetypes[target_arch_id].len(), initial_arch_len);
    assert_eq!(store.archetypes[target_arch_id].entities().len(), initial_arch_len);
    for col in store.archetypes[target_arch_id].columns().values() {
        assert_eq!(col.len(), initial_arch_len);
    }
}

#[test]
fn test_local_slot_real_store_migration_tracking() {
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct CompA(pub i32);
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct CompB(pub i32);

    let mut store = Store::new();
    let entity = EntityId::from_u128(777);

    // 1. 创建仅包含 CompA 的实体
    store.insert((entity, (CompA(100),))).unwrap();
    let slot = store.local_slot(entity).expect("must have slot");
    let loc1 = store.allocator.get_location_by_slot(entity, slot).expect("must have loc1");

    // 2. 真实插入 CompB，触发端到端 Archetype 结构迁移
    store.insert_component(entity, CompB(200)).unwrap();

    // 3. 验证同一 LocalSlotId 必须立即解析到新的物理位置，且 archetype 发生变更
    let loc2 = store.allocator.get_location_by_slot(entity, slot).expect("must have loc2");
    assert_ne!(loc1.archetype_id, loc2.archetype_id, "Archetype must change upon component insertion");

    // 4. 验证新位置的组件数据真实正确
    let view = store.entity(entity).unwrap();
    assert_eq!(view.get::<CompA>().unwrap(), &CompA(100));
    assert_eq!(view.get::<CompB>().unwrap(), &CompB(200));

    // 5. 使用 modify_tuple_by_slot 依然能够正常操作迁移后的新位置
    store
        .modify_tuple_by_slot::<(CompA, CompB), _>(entity, slot, |(a, b)| {
            a.0 += 1;
            b.0 += 2;
        })
        .unwrap();

    let view2 = store.entity(entity).unwrap();
    assert_eq!(view2.get::<CompA>().unwrap(), &CompA(101));
    assert_eq!(view2.get::<CompB>().unwrap(), &CompB(202));
}
