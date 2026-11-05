//! 专项验证组件注册表 (ComponentRegistry) 与列原型复用体系的 5 项核心约定
//!
//! 1. 候选定义隔离与两阶段提交：prepare 暂存，commit 发布，drop/fail 零污染；
//! 2. 注册定义与写值校验分离：已注册 rkyv 类型接收普通写值，不报 Bincode 冲突且物理列保持 rkyv；
//! 3. 原子批注册 (register_components)：All-or-Nothing 原子性，批次中途失败绝不残留部分定义；
//! 4. 快照与恢复一致性：模式继承与恢复后注册表同步。

use crate::binding::EntityId;
use crate::id::ComponentTypeId;
use crate::storage::column::ColumnCodecId;
use crate::storage::{
    ComponentRegistration, EntityBundle, Store, StoreBatch,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CompIsolatedA(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CompIsolatedB(pub u64);

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
struct RkyvOnlyPos {
    x: f32,
    y: f32,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct NormalSerdeData(pub String);

#[test]
fn test_candidate_definition_isolation_and_two_phase_commit() {
    let mut store = Store::new();
    let type_id_a = ComponentTypeId::of::<CompIsolatedA>();

    // 初始状态：未注册 CompIsolatedA
    assert!(!store.component_registry.contains(&type_id_a));

    let e1 = EntityId::from_u128(1001);
    let bundle = EntityBundle::new().with(CompIsolatedA(42)).unwrap();

    let mut batch = StoreBatch::new();
    batch.create_entity(e1, bundle).unwrap();

    // 1. Prepare 阶段：构造候选批次，新定义暂存在 PreparedBatch 中
    let prepared = store.prepare_batch(batch).unwrap();
    // 断言：正式 Store 注册表中仍无此定义（完全隔离）
    assert!(!store.component_registry.contains(&type_id_a));
    // 断言：候选批次持有该新定义
    assert!(prepared.new_definitions.contains_key(&type_id_a));

    // 2. 丢弃批次（模拟业务放弃提交或网络超时）：验证零污染
    drop(prepared);
    assert!(!store.component_registry.contains(&type_id_a));
    assert_eq!(store.len(), 0);

    // 3. 再次准备并 Commit：两阶段正式发布
    let bundle2 = EntityBundle::new().with(CompIsolatedA(99)).unwrap();
    let mut batch2 = StoreBatch::new();
    batch2.create_entity(e1, bundle2).unwrap();

    let prepared2 = store.prepare_batch(batch2).unwrap();
    assert!(!store.component_registry.contains(&type_id_a));

    store.commit_batch(prepared2).unwrap();

    // 断言：Commit 后新定义原子并入正式注册表
    assert!(store.component_registry.contains(&type_id_a));
    let entity = store.entity(e1).expect("entity must exist");
    assert_eq!(entity.get::<CompIsolatedA>().unwrap(), &CompIsolatedA(99));
}

#[test]
fn test_prepare_failure_leaves_registry_untouched() {
    let mut store = Store::new();
    let type_id_b = ComponentTypeId::of::<CompIsolatedB>();

    let e1 = EntityId::from_u128(2001);
    // 先创建一个实体
    store.spawn(e1).unwrap();

    // 试图再次以相同 ID 创建实体，触发 prepare 阶段的主键冲突失败
    let bundle = EntityBundle::new().with(CompIsolatedB(12345)).unwrap();

    let mut batch = StoreBatch::new();
    batch.create_entity(e1, bundle).unwrap();

    let err = store.prepare_batch(batch).unwrap_err();
    assert!(matches!(err, crate::error::StateError::EntityAlreadyExists(_)));

    // 验证：虽然 bundle 携带了全新的 CompIsolatedB，但由于 prepare 失败，正式注册表绝无残留
    assert!(!store.component_registry.contains(&type_id_b));
}

#[test]
fn test_registered_rkyv_receives_normal_write_without_bincode_conflict() {
    let mut store = Store::new();
    let type_id_rkyv = ComponentTypeId::of::<RkyvOnlyPos>();

    // 1. 显式注册为高性能零拷贝 rkyv 列
    store
        .register_components([ComponentRegistration::of_rkyv::<RkyvOnlyPos>()])
        .unwrap();

    let def = store.component_registry.get(&type_id_rkyv).unwrap();
    assert_eq!(def.codec(), ColumnCodecId::Rkyv);

    // 2. 业务使用普通 Rust 原生值写入（不提供任何列编码配置）
    let e1 = EntityId::from_u128(3001);
    store.spawn(e1).unwrap();

    // 普通 insert_component 写值
    store
        .insert_component(e1, RkyvOnlyPos { x: 10.0, y: 20.0 })
        .expect("writing native value into registered rkyv component must succeed");

    // 3. 验证物理列保持为 Rkyv，且未被降级或冲突
    let loc = store.allocator.get_location(e1).unwrap();
    let arch = &store.archetypes[loc.archetype_id as usize];
    let col = arch.columns().get(&type_id_rkyv).unwrap();
    assert_eq!(col.codec(), ColumnCodecId::Rkyv);

    // 4. 验证就地更新与读取
    let entity = store.entity(e1).unwrap();
    assert_eq!(entity.get::<RkyvOnlyPos>().unwrap(), &RkyvOnlyPos { x: 10.0, y: 20.0 });

    store
        .modify::<RkyvOnlyPos, _>(e1, |pos| {
            pos.x += 5.0;
            pos.y += 10.0;
        })
        .unwrap();

    let entity2 = store.entity(e1).unwrap();
    assert_eq!(entity2.get::<RkyvOnlyPos>().unwrap(), &RkyvOnlyPos { x: 15.0, y: 30.0 });

    // 5. 验证快照往返保持 rkyv 编解码一致
    let snapshot = store.snapshot().unwrap();
    let mut restored = store.empty_with_schema();
    restored.restore_snapshot(&snapshot).unwrap();

    let restored_ent = restored.entity(e1).unwrap();
    assert_eq!(restored_ent.get::<RkyvOnlyPos>().unwrap(), &RkyvOnlyPos { x: 15.0, y: 30.0 });
}


#[test]
fn test_register_components_all_or_nothing_atomicity() {
    use crate::storage::column::TypedColumn;

    let mut store = Store::new();

    #[derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Serialize,
        Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct CompGood(pub u32);

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
    struct CompConflict(pub f64);

    let type_id_good = ComponentTypeId::of::<CompGood>();
    let type_id_conflict = ComponentTypeId::of::<CompConflict>();

    // 先注册默认版本 (schema_version = 1) 的 CompConflict
    store
        .register_components([ComponentRegistration::of_rkyv::<CompConflict>()])
        .unwrap();
    assert_eq!(store.component_registry.len(), 1);

    // 构造版本冲突的原型 (schema_version = 2)
    let mut vtable_v2 = crate::storage::column::ColumnArchiveVTable::new_rkyv();
    vtable_v2.schema_version = 2;
    let conflict_v2 = ComponentRegistration {
        column: Box::new(TypedColumn::<CompConflict>::new_with_vtable(
            std::any::type_name::<CompConflict>(),
            vtable_v2,
        )),
    };

    // 尝试在单次调用中同时注册 [CompGood, CompConflict (version 2)]
    // CompGood 本身无冲突，但 CompConflict 存在模式版本冲突
    let err = store
        .register_components([
            ComponentRegistration::of::<CompGood>(),
            conflict_v2,
        ])
        .unwrap_err();

    assert!(err.to_string().contains("conflicts with existing registered version"));

    // 核心断言：All-or-Nothing 原子性防线生效，CompGood 绝不能被部分留在注册表中！
    assert!(
        !store.component_registry.contains(&type_id_good),
        "CompGood must not be registered when the batch has a conflict"
    );
    assert_eq!(
        store.component_registry.len(),
        1,
        "Only the pre-existing component should remain in registry"
    );
    assert_eq!(
        store.component_registry.get(&type_id_conflict).unwrap().schema_version(),
        1
    );
    assert_eq!(
        store.component_registry.get(&type_id_conflict).unwrap().codec(),
        ColumnCodecId::Rkyv
    );
}

#[test]
fn test_registration_failure_on_epoch_overflow_leaves_registry_untouched() {
    let mut store = Store::new();
    store.epoch = u64::MAX;

    #[derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Serialize,
        Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct CompEpochOverflow(pub u32);

    let type_id = ComponentTypeId::of::<CompEpochOverflow>();
    assert!(!store.component_registry.contains(&type_id));

    let err = store
        .register_components([ComponentRegistration::of::<CompEpochOverflow>()])
        .unwrap_err();

    assert!(matches!(err, crate::error::StateError::CapacityExhausted(_)));
    assert!(
        !store.component_registry.contains(&type_id),
        "Registry must not be polluted when epoch capacity is exhausted"
    );
    assert_eq!(store.component_registry.len(), 0);
}

#[test]
fn test_registration_failure_on_archetype_generation_overflow_leaves_registry_untouched() {
    let mut store = Store::new();
    store.archetype_set_generation = u64::MAX;

    #[derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Serialize,
        Deserialize,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    struct CompGenOverflow(pub u64);

    let type_id = ComponentTypeId::of::<CompGenOverflow>();
    assert!(!store.component_registry.contains(&type_id));

    let err = store
        .register_components([ComponentRegistration::of::<CompGenOverflow>()])
        .unwrap_err();

    assert!(matches!(err, crate::error::StateError::ArchetypeGenerationOverflow));
    assert!(
        !store.component_registry.contains(&type_id),
        "Registry must not be polluted when archetype generation overflows"
    );
    assert_eq!(store.component_registry.len(), 0);
}

