use super::*;
use crate::storage::BatchOpCounts;
use crate::storage::batch::PatchSemantics;
use crate::storage::EntityBundle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct CompA(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct CompB(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct CompC(pub u64);

#[test]
fn test_empty_patch_semantics() {
    let mut store = Store::new();
    let id1 = EntityId::from_u128(1);
    store.insert((id1, (CompA(10),))).unwrap();

    // 1. 对不存在实体的空补丁返回 EntityNotFound
    let mut batch1 = StoreBatch::new();
    let ghost_id = EntityId::from_u128(999);
    batch1.patch_entity(ghost_id, EntityPatch::new()).unwrap();
    assert_eq!(batch1.counts().empty_patches, 1);
    let err = store.prepare_batch(batch1).unwrap_err();
    assert!(matches!(err, StateError::EntityNotFound(id) if id == ghost_id));

    // 2. 对已存在实体的空补丁，不分配 plans、in_place_plans、updated_entities 容量
    let mut batch2 = StoreBatch::new();
    batch2.patch_entity(id1, EntityPatch::new()).unwrap();
    assert_eq!(batch2.counts().empty_patches, 1);
    let prepared = store.prepare_batch(batch2).unwrap();
    assert_eq!(prepared.in_place_plans_capacity(), 0);
    assert_eq!(prepared.plans_capacity(), 0);
    assert_eq!(prepared.receipt_updated_capacity(), 0);
    let receipt = store.commit_batch(prepared).unwrap();
    assert_eq!(receipt.updated_entities.len(), 0);

    // 3. 批次内先删后空补丁报错 BatchConflict
    let mut batch3 = StoreBatch::new();
    batch3.delete_entity(id1).unwrap();
    let err3 = batch3.patch_entity(id1, EntityPatch::new()).unwrap_err();
    assert!(matches!(err3, StateError::BatchConflict(_)));

    // 4. 重复组件操作失败不改变语义计数
    let mut patch = EntityPatch::new();
    patch.insert(CompA(10)).unwrap();
    assert_eq!(patch.semantics(), PatchSemantics::Structural);
    let err_dup = patch.insert(CompA(20)).unwrap_err();
    assert!(matches!(err_dup, StateError::ComponentDuplicateInPatch { .. }));
    assert_eq!(patch.semantics(), PatchSemantics::Structural);
}

#[test]
fn test_pure_workloads_zero_idle_capacity() {
    let mut store = Store::new();
    let count = 100;
    store.reserve_entities(count).unwrap();

    // 1. 纯创建
    let mut cmd_create = store.commands();
    for i in 1..=count as u128 {
        let bundle = EntityBundle::new().with(CompA(i as u64)).unwrap();
        cmd_create.create_entity(EntityId::from_u128(i), bundle);
    }
    let prep_create = cmd_create.prepare(&mut store).unwrap();
    assert_eq!(prep_create.plans_capacity(), count);
    assert_eq!(prep_create.in_place_plans_capacity(), 0);
    assert_eq!(prep_create.receipt_created_capacity(), count);
    assert_eq!(prep_create.receipt_updated_capacity(), 0);
    store.commit_batch(prep_create).unwrap();

    // 2. 纯原位替换
    let mut cmd_update = store.commands();
    for i in 1..=count as u128 {
        cmd_update.entity(EntityId::from_u128(i)).replace(CompA(i as u64 * 10));
    }
    let prep_update = cmd_update.prepare(&mut store).unwrap();
    assert_eq!(prep_update.plans_capacity(), 0);
    assert_eq!(prep_update.in_place_plans_capacity(), count);
    assert_eq!(prep_update.receipt_updated_capacity(), count);
    store.commit_batch(prep_update).unwrap();

    // 3. 纯结构迁移（添加 CompB）
    let mut cmd_migrate = store.commands();
    for i in 1..=count as u128 {
        cmd_migrate.entity(EntityId::from_u128(i)).insert(CompB(i as u64 * 20));
    }
    let prep_migrate = cmd_migrate.prepare(&mut store).unwrap();
    assert_eq!(prep_migrate.plans_capacity(), count);
    assert_eq!(prep_migrate.in_place_plans_capacity(), 0);
    assert_eq!(prep_migrate.receipt_updated_capacity(), count);
    store.commit_batch(prep_migrate).unwrap();
}

#[test]
fn test_migration_ordering_zero_idle_capacity() {
    let mut store = Store::new();
    let total = 10_000;
    store.reserve_entities(total).unwrap();
    for i in 1..=total as u128 {
        store.insert((EntityId::from_u128(i), (CompA(i as u64),))).unwrap();
    }

    // 1. 迁移在最前：1 个迁移 + 9999 个原位替换
    let mut cmd1 = store.commands();
    cmd1.entity(EntityId::from_u128(1)).insert(CompB(100));
    for i in 2..=total as u128 {
        cmd1.entity(EntityId::from_u128(i)).replace(CompA(i as u64 + 1));
    }
    let prep1 = cmd1.prepare(&mut store).unwrap();
    assert_eq!(prep1.plans_capacity(), 1, "plans capacity must be exactly 1");
    assert_eq!(prep1.in_place_plans_capacity(), 9999, "in_place_plans capacity must be exactly 9999");
    assert_eq!(prep1.receipt_updated_capacity(), total);
    store.commit_batch(prep1).unwrap();

    // 2. 迁移在最后：9999 个原位替换 + 1 个迁移
    let mut cmd2 = store.commands();
    for i in 2..=total as u128 {
        cmd2.entity(EntityId::from_u128(i)).replace(CompA(i as u64 + 2));
    }
    cmd2.entity(EntityId::from_u128(1)).remove::<CompB>();
    let prep2 = cmd2.prepare(&mut store).unwrap();
    assert_eq!(prep2.plans_capacity(), 1, "plans capacity must be exactly 1");
    assert_eq!(prep2.in_place_plans_capacity(), 9999, "in_place_plans capacity must be exactly 9999");
    assert_eq!(prep2.receipt_updated_capacity(), total);
    store.commit_batch(prep2).unwrap();
}

#[test]
fn test_upsert_conditional_bounded_idle_capacity() {
    let mut store = Store::new();
    let n_inplace = 50;
    let n_upsert = 20;
    let n_create = 30;
    let total_existing = n_inplace + n_upsert;
    store.reserve_entities(total_existing + n_create).unwrap();

    for i in 1..=total_existing as u128 {
        store.insert((EntityId::from_u128(i), (CompA(i as u64),))).unwrap();
    }

    let mut cmd = store.commands();
    // 50 个原位替换
    for i in 1..=n_inplace as u128 {
        cmd.entity(EntityId::from_u128(i)).replace(CompA(i as u64 * 10));
    }
    // 10 个 Upsert CompA（已存在 → 原位）
    for i in 1..=10 as u128 {
        let id = EntityId::from_u128(n_inplace as u128 + i);
        cmd.entity(id).upsert(CompA(999));
    }
    // 10 个 Upsert CompB（不存在 → 迁移）
    for i in 11..=n_upsert as u128 {
        let id = EntityId::from_u128(n_inplace as u128 + i);
        cmd.entity(id).upsert(CompB(888));
    }
    // 30 个创建
    for i in 1..=n_create as u128 {
        let id = EntityId::from_u128(total_existing as u128 + i);
        let bundle = EntityBundle::new().with(CompA(i as u64)).unwrap();
        cmd.create_entity(id, bundle);
    }

    let prep = cmd.prepare(&mut store).unwrap();

    // 验证容量上界
    // Conditional migrations grow lazily: capacity follows actual structural
    // plans (with geometric growth), not the number of possible migrations.
    assert!(prep.plans_capacity() >= prep.plans_len());
    assert!(prep.plans_capacity() <= prep.plans_len().saturating_mul(2).max(4));
    assert_eq!(prep.in_place_plans_capacity(), n_inplace + n_upsert,
        "in_place capacity = in_place + conditional_patches");

    // 验证实际计划数量（非硬编码）
    // 10 个 Upsert CompA 是原位 → in_place_plans；10 个 Upsert CompB 是迁移 → plans
    let actual_in_place = prep.in_place_plan_count();
    let actual_plans = prep.plans_len();
    assert_eq!(actual_in_place, n_inplace + 10,
        "60 实际原位计划：50 Replace + 10 Upsert(已存在)");
    assert_eq!(actual_plans, n_create + 10,
        "40 实际结构计划：30 Create + 10 Upsert(新增)");

    // 用实际 len 计算闲置
    let idle_in_place = prep.in_place_plans_capacity() - actual_in_place;
    let idle_plans = prep.plans_capacity() - actual_plans;
    let total_idle = idle_in_place + idle_plans;
    assert!(total_idle <= n_upsert + actual_plans,
        "idle capacity is bounded by conditional in-place reserve plus structural growth");

    // 验证 receipt 预分配（prepare 阶段即填充 created/updated）
    assert_eq!(prep.receipt_created_len(), n_create,
        "prepare 阶段 created = 30 个创建");
    assert_eq!(prep.receipt_updated_len(), actual_in_place + 10,
        "prepare 阶段 updated = 原位 + 迁移中的更新");

    // 提交并验证组件值
    let receipt = store.commit_batch(prep).unwrap();
    assert_eq!(receipt.created_entities.len(), n_create);
    assert_eq!(receipt.updated_entities.len(), total_existing);

    // 验证 Upsert CompA（原位）的值
    let view_upsert_inplace = store.entity(EntityId::from_u128(51)).unwrap();
    assert_eq!(view_upsert_inplace.get::<CompA>().unwrap().0, 999);

    // 验证 Upsert CompB（迁移）的值
    let id_upsert_migrate = EntityId::from_u128(61);
    let view_upsert_migrate = store.entity(id_upsert_migrate).unwrap();
    assert_eq!(view_upsert_migrate.get::<CompB>().unwrap().0, 888);
    assert!(view_upsert_migrate.get::<CompA>().is_some(), "原有 CompA 应保留");

    // 验证创建的值
    let view_created = store.entity(EntityId::from_u128(71)).unwrap();
    assert_eq!(view_created.get::<CompA>().unwrap().0, 1);

    store.validate_invariants().unwrap();
}

#[test]
fn test_classification_upgrade_multi_op_per_entity() {
    // 验证同实体连续操作的分类升级：Replace → Upsert → Insert
    let mut store = Store::new();
    let id1 = EntityId::from_u128(1);
    store.insert((id1, (CompA(10),))).unwrap();

    let mut cmd = store.commands();
    cmd.entity(id1)
        .replace(CompA(20))   // Empty → InPlace
        .upsert(CompB(30))    // InPlace → Conditional
        .insert(CompB(40));   // 重复组件，应失败

    // 重复组件应产生延迟错误
    let err = cmd.prepare(&mut store).unwrap_err();
    assert!(matches!(err, StateError::ComponentDuplicateInPatch { .. }));
}

#[test]
fn test_classification_failure_preserves_counts() {
    // 验证操作失败不改变分类计数，完整比对 6 项，且总和始终与 cmd.len() 一致
    let mut store = Store::new();
    let id1 = EntityId::from_u128(1);
    let id2 = EntityId::from_u128(2);
    store.insert((id1, (CompA(10),))).unwrap();
    store.insert((id2, (CompA(20),))).unwrap();

    let mut cmd = store.commands();
    // 初始状态：0 条目，全 0 计数
    let initial_counts = BatchOpCounts::default();
    assert_eq!(cmd.counts(), initial_counts);
    assert_eq!(cmd.len(), 0);

    // 成功：id1 Replace → InPlace
    cmd.entity(id1).replace(CompA(100));
    let expected_counts = BatchOpCounts {
        creates: 0,
        deletes: 0,
        in_place_patches: 1,
        structural_patches: 0,
        conditional_patches: 0,
        empty_patches: 0,
    };
    let counts_after_success = cmd.counts();
    assert_eq!(counts_after_success, expected_counts, "全部 6 项精确匹配 InPlace");
    assert_eq!(cmd.len(), 1);
    let sum_success = counts_after_success.creates
        + counts_after_success.deletes
        + counts_after_success.in_place_patches
        + counts_after_success.structural_patches
        + counts_after_success.conditional_patches
        + counts_after_success.empty_patches;
    assert_eq!(sum_success, cmd.len());

    // 失败：id1 再次 Insert CompA → 重复组件错误，应触发延迟错误
    // 但全部 6 项分类计数完全不变
    cmd.entity(id1).insert(CompA(200));
    let counts_after_failure = cmd.counts();
    assert_eq!(counts_after_failure, expected_counts, "失败后全部 6 项精确不变");
    assert_eq!(cmd.len(), 1);
    let sum_failure = counts_after_failure.creates
        + counts_after_failure.deletes
        + counts_after_failure.in_place_patches
        + counts_after_failure.structural_patches
        + counts_after_failure.conditional_patches
        + counts_after_failure.empty_patches;
    assert_eq!(sum_failure, cmd.len());

    // 场景 B：同批次冲突失败（先删除再修改）
    let mut cmd2 = store.commands();
    cmd2.delete_entity(id2);
    let expected_after_delete = BatchOpCounts {
        creates: 0,
        deletes: 1,
        in_place_patches: 0,
        structural_patches: 0,
        conditional_patches: 0,
        empty_patches: 0,
    };
    assert_eq!(cmd2.counts(), expected_after_delete);
    assert_eq!(cmd2.len(), 1);

    cmd2.entity(id2).replace(CompA(999)); // 冲突失败
    assert_eq!(cmd2.counts(), expected_after_delete, "冲突失败后全部 6 项精确不变");
    assert_eq!(cmd2.len(), 1);
    let sum_conflict = cmd2.counts().creates
        + cmd2.counts().deletes
        + cmd2.counts().in_place_patches
        + cmd2.counts().structural_patches
        + cmd2.counts().conditional_patches
        + cmd2.counts().empty_patches;
    assert_eq!(sum_conflict, cmd2.len());
}

#[test]
fn test_replace_then_insert_failure_keeps_inplace() {
    // 单独验证：Replace A 成功后 Insert A 失败，语义保持 InPlace
    let mut patch = EntityPatch::new();
    patch.replace(CompA(10)).unwrap();
    assert_eq!(patch.semantics(), PatchSemantics::InPlace);

    // Insert 同类型失败
    let err = patch.insert(CompA(20)).unwrap_err();
    assert!(matches!(err, StateError::ComponentDuplicateInPatch { .. }));
    // 语义不变
    assert_eq!(patch.semantics(), PatchSemantics::InPlace);
}

#[test]
fn test_replace_upsert_insert_classification_chain() {
    // 1. 底层 EntityPatch 的分类升级链：Replace A → Upsert B → Insert C
    let mut patch = EntityPatch::new();
    patch.replace(CompA(10)).unwrap();
    assert_eq!(patch.semantics(), PatchSemantics::InPlace);

    patch.upsert(CompB(20)).unwrap();
    assert_eq!(patch.semantics(), PatchSemantics::Conditional);

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    struct CompC(pub u64);
    patch.insert(CompC(30)).unwrap();
    assert_eq!(patch.semantics(), PatchSemantics::Structural);

    // 2. CommandBuffer 录入期间的 6 项计数动态转移全链路
    let mut store = Store::new();
    let id1 = EntityId::from_u128(100);
    store.insert((id1, (CompA(10),))).unwrap();

    let mut cmd = store.commands();
    assert_eq!(cmd.len(), 0);
    assert_eq!(cmd.counts(), BatchOpCounts::default());

    // 步骤一：Replace A → InPlace (in_place: 1)
    cmd.entity(id1).replace(CompA(11));
    assert_eq!(cmd.len(), 1);
    assert_eq!(
        cmd.counts(),
        BatchOpCounts {
            creates: 0,
            deletes: 0,
            in_place_patches: 1,
            structural_patches: 0,
            conditional_patches: 0,
            empty_patches: 0,
        }
    );
    let sum1 = cmd.counts().creates + cmd.counts().deletes
        + cmd.counts().in_place_patches + cmd.counts().structural_patches
        + cmd.counts().conditional_patches + cmd.counts().empty_patches;
    assert_eq!(sum1, cmd.len());

    // 步骤二：同实体追加 Upsert B → 计数从 InPlace 转移至 Conditional (in_place: 0, conditional: 1)
    cmd.entity(id1).upsert(CompB(22));
    assert_eq!(cmd.len(), 1, "同实体操作，cmd.len() 仍为 1");
    assert_eq!(
        cmd.counts(),
        BatchOpCounts {
            creates: 0,
            deletes: 0,
            in_place_patches: 0,
            structural_patches: 0,
            conditional_patches: 1,
            empty_patches: 0,
        },
        "计数由 in_place 成功转移至 conditional"
    );
    let sum2 = cmd.counts().creates + cmd.counts().deletes
        + cmd.counts().in_place_patches + cmd.counts().structural_patches
        + cmd.counts().conditional_patches + cmd.counts().empty_patches;
    assert_eq!(sum2, cmd.len());

    // 步骤三：同实体追加 Insert C → 计数从 Conditional 转移至 Structural (conditional: 0, structural: 1)
    cmd.entity(id1).insert(CompC(33));
    assert_eq!(cmd.len(), 1, "同实体操作，cmd.len() 仍为 1");
    assert_eq!(
        cmd.counts(),
        BatchOpCounts {
            creates: 0,
            deletes: 0,
            in_place_patches: 0,
            structural_patches: 1,
            conditional_patches: 0,
            empty_patches: 0,
        },
        "计数由 conditional 成功转移至 structural"
    );
    let sum3 = cmd.counts().creates + cmd.counts().deletes
        + cmd.counts().in_place_patches + cmd.counts().structural_patches
        + cmd.counts().conditional_patches + cmd.counts().empty_patches;
    assert_eq!(sum3, cmd.len());

    // 验证可正常提交且数据正确写入
    let prepared = cmd.prepare(&mut store).unwrap();
    assert_eq!(prepared.plans_len(), 1, "Structural patch 生成 1 个迁移计划");
    assert_eq!(prepared.in_place_plan_count(), 0);
    store.commit_batch(prepared).unwrap();

    let view = store.entity(id1).unwrap();
    assert_eq!(view.get::<CompA>().unwrap().0, 11);
    assert_eq!(view.get::<CompB>().unwrap().0, 22);
    assert_eq!(view.get::<CompC>().unwrap().0, 33);
}

#[test]
fn test_count_sum_equals_entry_count() {
    // 验证 6 项计数之和始终等于条目总数，且严格等于 cmd.len()
    let mut store = Store::new();
    store.reserve_entities(200).unwrap();
    for i in 1..=100u128 {
        store.insert((EntityId::from_u128(i), (CompA(i as u64),))).unwrap();
    }

    let mut cmd = store.commands();
    // 20 Replace（InPlace）
    for i in 1..=20u128 {
        cmd.entity(EntityId::from_u128(i)).replace(CompA(i as u64 * 10));
    }
    // 10 Insert CompB（Structural）
    for i in 21..=30u128 {
        cmd.entity(EntityId::from_u128(i)).insert(CompB(i as u64));
    }
    // 15 Upsert CompB（Conditional）
    for i in 31..=45u128 {
        cmd.entity(EntityId::from_u128(i)).upsert(CompB(i as u64));
    }
    // 5 Delete
    for i in 46..=50u128 {
        cmd.delete_entity(EntityId::from_u128(i));
    }
    // 25 Create
    for i in 101..=125u128 {
        let bundle = EntityBundle::new().with(CompA(i as u64)).unwrap();
        cmd.create_entity(EntityId::from_u128(i), bundle);
    }

    let counts = cmd.counts();
    let total_entries = counts.creates + counts.deletes
        + counts.in_place_patches + counts.structural_patches
        + counts.conditional_patches + counts.empty_patches;
    assert_eq!(total_entries, 75, "6 项计数之和 = 条目总数");
    assert_eq!(total_entries, cmd.len(), "6 项计数之和严格等于 cmd.len()");
    assert_eq!(counts.creates, 25);
    assert_eq!(counts.deletes, 5);
    assert_eq!(counts.in_place_patches, 20);
    assert_eq!(counts.structural_patches, 10);
    assert_eq!(counts.conditional_patches, 15);
    assert_eq!(counts.empty_patches, 0);
}

#[test]
fn test_mixed_1_update_9999_creates_exact_capacity() {
    // 从被删除的 mixed_batch_allocation_test.rs 迁入
    let mut store = Store::new();
    let update_count = 1;
    let create_count = 9_999;

    store.reserve_entities(update_count + create_count).unwrap();
    let update_id = EntityId::from_u128(1);
    store.insert((update_id, (CompA(10), CompB(20)))).unwrap();

    let create_ids: Vec<EntityId> = (2..=(create_count + 1) as u128)
        .map(EntityId::from_u128)
        .collect();

    let mut cmd = store.commands();
    cmd.reserve(update_count + create_count);

    // 1 原位替换
    cmd.entity(update_id).replace(CompA(999));

    // 9,999 创建
    for (idx, &id) in create_ids.iter().enumerate() {
        let bundle = EntityBundle::new()
            .with(CompA(idx as u64))
            .unwrap()
            .with(CompB(idx as u64 * 2))
            .unwrap();
        cmd.create_entity(id, bundle);
    }

    let prepared = cmd.prepare(&mut store).unwrap();

    assert_eq!(prepared.receipt_created_capacity(), create_count);
    assert_eq!(prepared.receipt_updated_capacity(), update_count);
    assert_eq!(prepared.in_place_plans_capacity(), update_count);
    assert_eq!(prepared.plans_capacity(), create_count);
    assert_eq!(prepared.in_place_plan_count(), update_count);
    assert_eq!(prepared.plans_len(), create_count);

    let receipt = store.commit_batch(prepared).unwrap();
    assert_eq!(receipt.created_entities.len(), create_count);
    assert_eq!(receipt.updated_entities.len(), update_count);
    assert_eq!(store.len(), update_count + create_count);
    store.validate_invariants().unwrap();
}

#[test]
fn test_store_batch_mixed_exact_capacity() {
    // 从被删除的 mixed_batch_allocation_test.rs 迁入
    let mut store = Store::new();
    let update_count = 500;
    let create_count = 500;

    store.reserve_entities(update_count + create_count).unwrap();
    let update_ids: Vec<EntityId> = (1..=update_count as u128)
        .map(EntityId::from_u128)
        .collect();

    for &id in &update_ids {
        store.insert((id, (CompA(10), CompB(20)))).unwrap();
    }

    let mut batch = StoreBatch::new();

    // 500 原位替换
    for (idx, &id) in update_ids.iter().enumerate() {
        let mut patch = EntityPatch::new();
        patch.replace(CompA(200 + idx as u64)).unwrap();
        batch.patch_entity(id, patch).unwrap();
    }

    // 500 创建
    for i in 1..=create_count as u128 {
        let id = EntityId::from_u128(10_000 + i);
        let bundle = EntityBundle::new()
            .with(CompA(i as u64))
            .unwrap()
            .with(CompB(i as u64 * 3))
            .unwrap();
        batch.create_entity(id, bundle).unwrap();
    }

    let prepared = store.prepare_batch(batch).unwrap();

    assert_eq!(prepared.receipt_created_capacity(), create_count);
    assert_eq!(prepared.receipt_updated_capacity(), update_count);
    assert_eq!(prepared.in_place_plans_capacity(), update_count);
    assert_eq!(prepared.plans_capacity(), create_count);
    assert_eq!(prepared.in_place_plan_count(), update_count);
    assert_eq!(prepared.plans_len(), create_count);

    let receipt = store.commit_batch(prepared).unwrap();
    assert_eq!(receipt.created_entities.len(), create_count);
    assert_eq!(receipt.updated_entities.len(), update_count);
    assert_eq!(store.len(), update_count + create_count);
    store.validate_invariants().unwrap();
}

