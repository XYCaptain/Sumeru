use super::*;


#[test]
fn revision_exhaustion_rejects_every_write_before_mutation() {
    let mut store = Store::new();
    store.register_component::<u64>().unwrap();
    let entity = EntityId::from_u128(1);
    store
        .create_entity(entity, EntityBundle::new().with(1u64).unwrap())
        .unwrap();
    store.epoch = u64::MAX;
    let before = bincode::serialize(&store.snapshot().unwrap()).unwrap();
    assert!(matches!(
        store.replace_component(entity, 2u64),
        Err(StateError::CapacityExhausted(_))
    ));
    assert!(store.delete_entity(entity).is_err());
    assert!(store.spawn(EntityId::from_u128(2)).is_err());
    assert!(store.register_component::<bool>().is_err());
    let mut batch = StoreBatch::new();
    batch.delete_entity(entity).unwrap();
    assert!(store.prepare_batch(batch).is_err());
    assert!(store.restore_snapshot(&store.snapshot().unwrap()).is_err());
    assert_eq!(
        bincode::serialize(&store.snapshot().unwrap()).unwrap(),
        before
    );
    store.validate_invariants().unwrap();
}

#[test]
fn epoch_exhaustion_allows_empty_ops_matrix_and_rejects_modifications() {
    let mut store = Store::new();
    let entity = EntityId::from_u128(1);
    store
        .create_entity(entity, EntityBundle::new().with(100u64).unwrap())
        .unwrap();

    // 强行模拟代际耗尽至 u64::MAX
    store.epoch = u64::MAX;

    // 1. 空 StoreBatch 允许成功，且 epoch 保持不变
    let empty_batch = StoreBatch::new();
    let prepared = store.prepare_batch(empty_batch).unwrap();
    let receipt = store.commit_batch(prepared).unwrap();
    assert!(receipt.created_entities.is_empty());
    assert_eq!(store.epoch(), u64::MAX);

    // 2. 合法已有实体的空 patch 允许成功，且 epoch 保持不变
    let mut empty_patch = StoreBatch::new();
    empty_patch
        .patch_entity(entity, EntityPatch::new())
        .unwrap();
    let prepared = store.prepare_batch(empty_patch).unwrap();
    let receipt = store.commit_batch(prepared).unwrap();
    assert!(receipt.updated_entities.is_empty());
    assert_eq!(store.epoch(), u64::MAX);

    // 3. 未知实体的空 patch 拒绝并报错 EntityNotFound
    let mut unknown_patch = StoreBatch::new();
    unknown_patch
        .patch_entity(EntityId::from_u128(9999), EntityPatch::new())
        .unwrap();
    let err = store.prepare_batch(unknown_patch).unwrap_err();
    assert!(matches!(err, StateError::EntityNotFound(..)));
    assert_eq!(store.epoch(), u64::MAX);

    // 4. 有变更的批次（如新建实体）必须报错 CapacityExhausted，且不改变状态
    let mut change_batch = StoreBatch::new();
    change_batch
        .create_entity(
            EntityId::from_u128(2),
            EntityBundle::new().with(200u64).unwrap(),
        )
        .unwrap();
    let err = store.prepare_batch(change_batch).unwrap_err();
    assert!(matches!(err, StateError::CapacityExhausted(_)));
    assert!(!store.contains_entity(EntityId::from_u128(2)));
    assert_eq!(store.epoch(), u64::MAX);

    // 5. modify 原地修改（即使是 no-op 闭包）在 epoch == u64::MAX 时必须报错 CapacityExhausted，不执行闭包
    let mut closure_invoked = false;
    let err = store
        .modify::<u64, _>(entity, |v| {
            closure_invoked = true;
            *v += 1;
        })
        .unwrap_err();
    assert!(!closure_invoked);
    assert!(matches!(err, StateError::CapacityExhausted(_)));
    assert_eq!(store.extract::<&u64>(entity).unwrap(), &100u64);
    // 6. query_mut 在 epoch == u64::MAX 时必须报错 CapacityExhausted，且状态未突变
    let err = store.query_mut::<&mut u64>().unwrap_err();
    assert!(matches!(err, StateError::CapacityExhausted(_)));
    assert_eq!(store.extract::<&u64>(entity).unwrap(), &100u64);
    assert_eq!(store.epoch(), u64::MAX);

    store.validate_invariants().unwrap();
}

#[test]
fn test_archetype_set_generation_overflow_pre_check_has_no_side_effects() {
    let mut store = Store::new();
    store.register_component::<u64>().unwrap();
    let entity = EntityId::from_u128(1);
    store
        .create_entity(entity, EntityBundle::new().with(100u64).unwrap())
        .unwrap();

    let initial_archetypes_len = store.archetypes.len();
    let initial_index_len = store.archetype_index.len();
    let initial_epoch = store.epoch();

    let _initial_snapshot = bincode::serialize(&store.snapshot().unwrap()).unwrap();

    // 强行模拟原型代际达到最大值 u64::MAX
    store.archetype_set_generation = u64::MAX;

    // 1. 代际耗尽在 prepare 阶段拒绝，输入仍可交还，不修改 archetypes 和 index。
    let mut batch = StoreBatch::new();
    let entity2 = EntityId::from_u128(2);
    // 新增包含新组件类型 (f64) 的实体，这会导致生成新原型
    batch
        .create_entity(
            entity2,
            EntityBundle::new().with(std::f64::consts::PI).unwrap(),
        )
        .unwrap();
    let (err, batch) = store.prepare_batch_recoverable(batch).unwrap_err();
    assert!(matches!(err, StateError::ArchetypeGenerationOverflow));
    assert_eq!(batch.len(), 1);
    assert_eq!(store.epoch(), initial_epoch);
    assert!(!store.contains_entity(entity2));

    // commit 仍保留防御：准备完成后若容量状态改变，也必须在发布前拒绝。
    store.archetype_set_generation = u64::MAX - 1;
    let prepared = store.prepare_batch(batch).unwrap();
    store.archetype_set_generation = u64::MAX;
    assert!(
        !prepared.detached.is_empty(),
        "必须包含新原型以验证前置检查"
    );

    let err = store.commit_batch(prepared).unwrap_err();
    assert!(
        matches!(err, StateError::ArchetypeGenerationOverflow),
        "预期 ArchetypeGenerationOverflow，实际为 {:?}",
        err
    );
    // 验证状态 100% 零污染
    assert_eq!(store.archetypes.len(), initial_archetypes_len);
    assert_eq!(store.archetype_index.len(), initial_index_len);
    assert_eq!(store.epoch(), initial_epoch);
    assert_eq!(store.archetype_set_generation, u64::MAX);
    assert!(!store.contains_entity(entity2));

    // 2. 测试 register_component_prototypes 在代际耗尽时被拦截且零副作用
    let initial_registry_len = store.component_registry.len();
    let proto: Box<dyn AnyColumn> = Box::new(TypedColumn::<f32>::new("f32"));
    let err = store
        .register_component_prototypes(vec![proto])
        .unwrap_err();
    assert!(matches!(err, StateError::ArchetypeGenerationOverflow));
    assert_eq!(store.archetypes.len(), initial_archetypes_len);
    assert_eq!(store.archetype_index.len(), initial_index_len);
    assert_eq!(store.epoch(), initial_epoch);
    assert_eq!(store.component_registry.len(), initial_registry_len);
    assert!(
        !store
            .component_registry
            .contains(&crate::id::ComponentTypeId::of::<f32>())
    );

    // 3. 测试 restore_snapshot 在代际耗尽时被前置拦截且零副作用
    let snap = store.snapshot().unwrap();
    let err = store.restore_snapshot(&snap).unwrap_err();
    assert!(matches!(err, StateError::ArchetypeGenerationOverflow));
    assert_eq!(store.archetypes.len(), initial_archetypes_len);
    assert_eq!(store.epoch(), initial_epoch);

    store.validate_invariants().unwrap();
}

#[test]
fn test_in_place_restore_snapshot_preserves_store_id() {
    let mut store = Store::new();
    store.register_component::<u64>().unwrap();
    let entity1 = EntityId::from_u128(101);
    store
        .create_entity(entity1, EntityBundle::new().with(42u64).unwrap())
        .unwrap();

    let initial_store_id = store.store_id();

    // 1. Capture snapshot before any mutations
    let snapshot = store.snapshot().unwrap();

    // Sample epoch and generation immediately before restore to verify progression
    let epoch_before_restore = store.epoch();
    let generation_before_restore = store.archetype_set_generation();

    // 2. In-place restore from snapshot
    store.restore_snapshot(&snapshot).unwrap();

    // 3. Assert in-place instance identity strictly preserved
    assert_eq!(
        store.store_id(),
        initial_store_id,
        "In-place restore must preserve store_id instance identity"
    );

    // 4. Assert epoch and archetype_set_generation have progressed (anti-ABA)
    assert!(
        store.epoch() > epoch_before_restore,
        "Epoch must advance after in-place restore strictly beyond the pre-restore point (anti-ABA)"
    );
    assert!(
        store.archetype_set_generation() > generation_before_restore,
        "Archetype set generation must advance after in-place restore strictly beyond the pre-restore point"
    );

    // 5. Assert restored content integrity
    assert!(store.contains_entity(entity1));

    // 6. Contrast with clone: clone allocates a brand-new store_id and resets epoch
    let cloned = store.clone();
    assert_ne!(
        cloned.store_id(),
        store.store_id(),
        "Clone must allocate a distinct store_id"
    );
    assert_eq!(
        cloned.epoch(),
        1,
        "Clone resets epoch to initial baseline 1"
    );

    store.validate_invariants().unwrap();
    cloned.validate_invariants().unwrap();
}
