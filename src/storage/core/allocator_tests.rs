use crate::storage::core::allocator::SlotMeta;
use super::*;
use crate::{EntityId, LocalSlotId, StateError};
use serde::Serialize;

#[test]
fn test_local_slot_aba_invalidation() {
    let mut alloc = EntityAllocator::new();
    let e1 = EntityId::from_u128(101);
    let e2 = EntityId::from_u128(102);

    // 1. 分配 e1，获取 slot
    let slot1 = alloc.allocate(e1).unwrap();
    let loc1 = EntityLocation::new(1, 0);
    alloc.set_location(e1, loc1).unwrap();

    assert_eq!(alloc.get_location_by_slot(e1, slot1), Some(loc1));

    // 2. 释放 e1
    alloc.deallocate(e1).unwrap();

    // 此时 slot1 对应 meta.entity == None，调用 get_location_by_slot 必须拒绝
    assert_eq!(alloc.get_location_by_slot(e1, slot1), None);

    // 3. 分配 e2，复用同一 slot 索引，但 generation 自增
    let slot2 = alloc.allocate(e2).unwrap();
    assert_eq!(slot2.slot, slot1.slot);
    assert_ne!(slot2.generation, slot1.generation);
    let loc2 = EntityLocation::new(2, 5);
    alloc.set_location(e2, loc2).unwrap();

    // 用 e2 + slot2 查询成功
    assert_eq!(alloc.get_location_by_slot(e2, slot2), Some(loc2));

    // 旧的 e1 + slot1 必须被拒绝（代际不匹配且 entity 不匹配）
    assert_eq!(alloc.get_location_by_slot(e1, slot1), None);
    // 即使传入 e2 + 旧 slot1 也必须被拒绝（防 ABA）
    assert_eq!(alloc.get_location_by_slot(e2, slot1), None);
}

#[test]
fn test_local_slot_migration_tracking() {
    let mut alloc = EntityAllocator::new();
    let e1 = EntityId::from_u128(201);
    let slot1 = alloc.allocate(e1).unwrap();
    let loc_init = EntityLocation::new(1, 0);
    alloc.set_location(e1, loc_init).unwrap();

    assert_eq!(alloc.get_location_by_slot(e1, slot1), Some(loc_init));

    // 模拟实体发生结构迁移（搬移至新 Archetype 3，行号 8）
    let loc_migrated = EntityLocation::new(3, 8);
    alloc.set_location(e1, loc_migrated).unwrap();

    // 同一有效 LocalSlotId 必须立即解析为迁移后的新位置
    assert_eq!(alloc.get_location_by_slot(e1, slot1), Some(loc_migrated));
}

#[test]
fn test_allocator_snapshot_rebuild_index_and_validate() {
    let mut alloc = EntityAllocator::new();
    let e1 = EntityId::from_u128(301);
    let e2 = EntityId::from_u128(302);
    let e3 = EntityId::from_u128(303);

    let _s1 = alloc.allocate(e1).unwrap();
    let s2 = alloc.allocate(e2).unwrap();
    let _s3 = alloc.allocate(e3).unwrap();

    alloc.deallocate(e2).unwrap();
    assert_eq!(alloc.active_count(), 2);
    assert!(alloc.validate().is_ok());

    // 模拟序列化与反序列化（验证 rebuild_index 逻辑正确重建 HashMap）
    let encoded = bincode::serialize(&alloc).expect("serialize allocator");
    let decoded: EntityAllocator = bincode::deserialize(&encoded).expect("deserialize allocator");

    assert_eq!(decoded.active_count(), 2);
    assert!(decoded.is_alive(e1));
    assert!(!decoded.is_alive(e2));
    assert!(decoded.is_alive(e3));
    assert_eq!(decoded.local_slot(e1).unwrap().slot, 0);
    assert_eq!(decoded.local_slot(e3).unwrap().slot, 2);
    assert!(decoded.validate().is_ok());

    // 重新分配复用槽位 1
    let mut alloc_resumed = decoded;
    let e4 = EntityId::from_u128(304);
    let s4 = alloc_resumed.allocate(e4).unwrap();
    assert_eq!(s4.slot, s2.slot);
    assert_eq!(s4.generation, s2.generation + 1);
    assert!(alloc_resumed.validate().is_ok());
}

#[test]
fn test_legacy_snapshot_payload_compatibility() {
    // 构造等价于旧版本的持久化二进制 Payload：
    // 旧版本 EntityAllocator 序列化内容为 slots: Vec<SlotMeta> 与 free_slots: Vec<u32>
    // by_entity 为 #[serde(skip)]。
    let e1 = EntityId::from_u128(9001);
    let e2 = EntityId::from_u128(9002);
    let legacy_slots = vec![
        SlotMeta {
            generation: 1,
            entity: Some(e1),
            location: Some(EntityLocation::new(1, 0)),
        },
        SlotMeta {
            generation: 2,
            entity: None,
            location: None,
        },
        SlotMeta {
            generation: 3,
            entity: Some(e2),
            location: Some(EntityLocation::new(2, 5)),
        },
    ];
    let legacy_free_slots = vec![1u32];

    #[derive(Serialize)]
    struct LegacyAllocatorPayload {
        slots: Vec<SlotMeta>,
        free_slots: Vec<u32>,
    }

    let legacy_payload = LegacyAllocatorPayload {
        slots: legacy_slots,
        free_slots: legacy_free_slots,
    };

    let encoded = bincode::serialize(&legacy_payload).expect("encode legacy payload");

    // 使用当前 EntityAllocator 进行反序列化解码
    let decoded: EntityAllocator =
        bincode::deserialize(&encoded).expect("deserialize into current EntityAllocator");

    // 断言解码后状态与重建的 HashMap 索引完全正确
    assert_eq!(decoded.active_count(), 2);
    assert!(decoded.is_alive(e1));
    assert!(!decoded.is_alive(EntityId::from_u128(9999)));
    assert!(decoded.is_alive(e2));
    assert_eq!(decoded.local_slot(e1).unwrap(), LocalSlotId::new(0, 1));
    assert_eq!(decoded.local_slot(e2).unwrap(), LocalSlotId::new(2, 3));
    assert_eq!(
        decoded.get_location(e1),
        Some(EntityLocation::new(1, 0))
    );
    assert_eq!(
        decoded.get_location(e2),
        Some(EntityLocation::new(2, 5))
    );
    assert!(decoded.validate().is_ok());
}

#[test]
fn test_compact_allocator_bincode_runs_and_literals_roundtrip() {
    let mut alloc = EntityAllocator::new();
    for i in 0..10_000u32 {
        let id = EntityId::from_u128(10_000 + (i as u128) * 8);
        alloc
            .allocate_fresh_with_location(id, EntityLocation::new(1, i))
            .unwrap();
    }
    // Deallocate one middle entity so we exercise both ActiveSlotRun and literal free slot
    let hole_id = EntityId::from_u128(10_000 + 500 * 8);
    alloc.deallocate(hole_id).unwrap();

    let encoded = bincode::serialize(&alloc).unwrap();
    assert!(
        encoded.len() < 512,
        "Expected compact allocator bincode < 512 bytes, got {}",
        encoded.len()
    );
    let decoded: EntityAllocator = bincode::deserialize(&encoded).unwrap();
    assert_eq!(decoded, alloc);
    assert!(decoded.validate().is_ok());
}

#[test]
fn test_slots_serde_fragmented_generations_and_out_of_order_rows_roundtrip() {
    let mut alloc = EntityAllocator::new();
    // 1. Short run (< 4 slots) in archetype 0 -> encoded as literals
    for i in 0..3u32 {
        let id = EntityId::from_u128(100 + i as u128);
        alloc
            .allocate_fresh_with_location(id, EntityLocation::new(0, i))
            .unwrap();
    }

    // 2. Medium run (8 slots) in archetype 1, then deallocate slot 5 and respawn a new entity
    // into that free slot (bumping its generation from 1 -> 2), breaking the run in the middle.
    for i in 0..8u32 {
        let id = EntityId::from_u128(200 + i as u128);
        alloc
            .allocate_fresh_with_location(id, EntityLocation::new(1, i))
            .unwrap();
    }
    let hole_entity = EntityId::from_u128(202); // slot 5
    alloc.deallocate(hole_entity).unwrap();
    let respawned = EntityId::from_u128(999);
    let respawn_slot = alloc.allocate(respawned).unwrap();
    assert_eq!(respawn_slot, LocalSlotId::new(5, 2));
    alloc
        .set_location(respawned, EntityLocation::new(1, 2))
        .unwrap();

    // 3. Run of 6 slots in archetype 2 where two middle rows are swapped (simulating swap_remove),
    // breaking row_index contiguity.
    for i in 0..6u32 {
        let id = EntityId::from_u128(300 + i as u128);
        alloc
            .allocate_fresh_with_location(id, EntityLocation::new(2, i))
            .unwrap();
    }
    alloc
        .set_location(EntityId::from_u128(302), EntityLocation::new(2, 5))
        .unwrap();
    alloc
        .set_location(EntityId::from_u128(305), EntityLocation::new(2, 2))
        .unwrap();

    // 4. Unlocated active slot (location == None) and a deallocated hole at the tail.
    let unlocated = EntityId::from_u128(400);
    alloc.allocate(unlocated).unwrap();
    let tail_hole = EntityId::from_u128(401);
    alloc.allocate(tail_hole).unwrap();
    alloc.deallocate(tail_hole).unwrap();

    let encoded = bincode::serialize(&alloc).unwrap();
    let decoded: EntityAllocator = bincode::deserialize(&encoded).unwrap();
    assert_eq!(decoded, alloc);
    assert!(decoded.validate().is_ok());
    assert_eq!(decoded.local_slot(respawned), Some(LocalSlotId::new(5, 2)));
    assert_eq!(
        decoded.get_location(EntityId::from_u128(302)),
        Some(EntityLocation::new(2, 5))
    );
    assert_eq!(decoded.get_location(unlocated), None);
    assert!(!decoded.is_alive(tail_hole));
}

#[test]
fn test_resolve_with_hint_monotonic_fast_path_and_fallback() {
    let mut alloc = EntityAllocator::new();
    let ids: Vec<EntityId> = (1..=8).map(EntityId::from_u128).collect();
    alloc
        .allocate_batch_with_contiguous_rows(&ids, 3, 10)
        .unwrap();

    // Monotonic forward scan hits the O(1) slot_hint fast path and advances hint by 1 each step.
    let mut hint = 0usize;
    for (i, &id) in ids.iter().enumerate() {
        let (local, loc) = alloc.resolve_with_hint(id, &mut hint).unwrap();
        assert_eq!(local, LocalSlotId::new(i as u32, 1));
        assert_eq!(loc, EntityLocation::new(3, 10 + i as u32));
        assert_eq!(hint, i + 1);
    }

    // Reverse / random order misses the hint and falls back to HashMap, updating hint to actual + 1.
    let mut random_hint = 999usize; // Out-of-bounds hint
    for &idx in &[5usize, 1, 7, 0, 3] {
        let (local, loc) = alloc.resolve_with_hint(ids[idx], &mut random_hint).unwrap();
        assert_eq!(local, LocalSlotId::new(idx as u32, 1));
        assert_eq!(loc, EntityLocation::new(3, 10 + idx as u32));
        assert_eq!(random_hint, idx + 1);
    }

    // Unlocated active entity (location == None) must return None even when hint points to its slot.
    let unlocated = EntityId::from_u128(99);
    let unlocated_slot = alloc.allocate(unlocated).unwrap();
    let mut unlocated_hint = unlocated_slot.slot as usize;
    assert_eq!(alloc.resolve_with_hint(unlocated, &mut unlocated_hint), None);

    // Deallocated entity and unknown entity must return None.
    alloc.deallocate(ids[2]).unwrap();
    let mut dead_hint = 2usize;
    assert_eq!(alloc.resolve_with_hint(ids[2], &mut dead_hint), None);
    assert_eq!(
        alloc.resolve_with_hint(EntityId::from_u128(123456), &mut dead_hint),
        None
    );
}

#[test]
fn test_allocate_batch_with_contiguous_rows_with_free_slots_and_duplicate_rejection() {
    let mut alloc = EntityAllocator::new();
    let initial: Vec<EntityId> = (1..=4).map(EntityId::from_u128).collect();
    alloc
        .allocate_batch_with_contiguous_rows(&initial, 0, 0)
        .unwrap();

    // Reject nil in allocate_fresh_with_location and allocate_batch_with_contiguous_rows
    assert!(matches!(
        alloc.allocate_fresh_with_location(EntityId::from_u128(0), EntityLocation::new(0, 4)),
        Err(StateError::AccessDenied(_))
    ));
    assert!(matches!(
        alloc.allocate_batch_with_contiguous_rows(&[EntityId::from_u128(0)], 0, 4),
        Err(StateError::AccessDenied(_))
    ));

    // Reject duplicate in allocate_fresh_with_location and verify existing entity mapping stays intact
    let orig_slot_1 = alloc.local_slot(initial[1]).unwrap();
    let orig_loc_1 = alloc.get_location(initial[1]).unwrap();
    assert!(matches!(
        alloc.allocate_fresh_with_location(initial[1], EntityLocation::new(9, 99)),
        Err(StateError::EntityAlreadyExists(id)) if id == initial[1]
    ));
    assert_eq!(alloc.local_slot(initial[1]), Some(orig_slot_1));
    assert_eq!(alloc.get_location(initial[1]), Some(orig_loc_1));
    assert!(alloc.validate().is_ok());

    // Reject duplicate in allocate_batch_with_contiguous_rows (when free_slots is empty)
    assert!(matches!(
        alloc.allocate_batch_with_contiguous_rows(&[initial[2]], 9, 50),
        Err(StateError::EntityAlreadyExists(id)) if id == initial[2]
    ));
    assert_eq!(alloc.local_slot(initial[2]), Some(LocalSlotId::new(2, 1)));
    assert_eq!(alloc.get_location(initial[2]), Some(EntityLocation::new(0, 2)));
    assert!(alloc.validate().is_ok());

    // Deallocate two entities so free_slots is non-empty, then batch-allocate 4 new entities
    // (2 reuse free slots with generation 2, 2 allocate fresh slots at the end).
    alloc.deallocate(initial[1]).unwrap(); // slot 1
    alloc.deallocate(initial[3]).unwrap(); // slot 3

    let batch2: Vec<EntityId> = (10..=13).map(EntityId::from_u128).collect();
    alloc
        .allocate_batch_with_contiguous_rows(&batch2, 5, 20)
        .unwrap();
    assert!(alloc.validate().is_ok());
    for (i, &id) in batch2.iter().enumerate() {
        assert_eq!(
            alloc.get_location(id),
            Some(EntityLocation::new(5, 20 + i as u32))
        );
    }
    // Popped LIFO from free_slots: batch2[0] -> slot 3 (gen 2), batch2[1] -> slot 1 (gen 2)
    assert_eq!(alloc.local_slot(batch2[0]), Some(LocalSlotId::new(3, 2)));
    assert_eq!(alloc.local_slot(batch2[1]), Some(LocalSlotId::new(1, 2)));
    assert_eq!(alloc.local_slot(batch2[2]), Some(LocalSlotId::new(4, 1)));
    assert_eq!(alloc.local_slot(batch2[3]), Some(LocalSlotId::new(5, 1)));
}

