use super::*;

#[test]
fn repeated_native_calls_share_types_and_preserve_conflicts() {
    let id = EntityId::from_u128;
    let mut store = Store::new();
    for n in 1..=3 { store.insert((id(n), (0u64, false, 0i32))).unwrap(); }
    let mut cmd = store.commands();
    cmd.update_batch([(id(1), 10u64, true)]);
    cmd.update_batch([(id(2), true, 20u64)]);
    cmd.update_batch([(id(3), 30u64)]);
    cmd.update_batch([(id(1), 100i32), (id(2), 200i32)]);
    assert_eq!(cmd.native_values.len(), 3);
    let receipt = cmd.apply(&mut store).unwrap();
    assert_eq!(receipt.updated_entities, vec![id(1), id(2), id(3)]);
    assert_eq!(store.get::<u64>(id(2)), Some(&20));
    assert_eq!(store.get::<i32>(id(1)), Some(&100));

    let mut store = Store::new();
    for n in 1..=3 { store.insert((id(n), (0u64, false, 0i32))).unwrap(); }
    let mut cmd = store.commands();
    cmd.update_batch([(id(1), 10u64, true)]);
    cmd.update_batch([(id(2), 20u64, true)]);
    // The existing chain contains both a taken-out active lane and another lane.
    cmd.update_batch([(id(1), 100i32)]);
    cmd.update_batch([(id(1), 11u64)]);
    assert!(matches!(cmd.apply(&mut store), Err(StateError::ComponentDuplicateInPatch { .. })));
    assert_eq!(store.get::<u64>(id(1)), Some(&0));
}

#[test]
fn native_cancellation_failure_and_iterator_unwind() {
    let id = EntityId::from_u128;
    let mut store = Store::new();
    store.insert((id(1), (String::from("old"),))).unwrap();
    for mode in 0..5 {
        let mut cmd = store.commands();
        cmd.update_batch([(id(if mode == 2 { 99 } else { 1 }), String::from("new"))]);
        match mode {
            0 => drop(cmd),
            1 => drop(cmd.prepare(&mut store).unwrap()),
            2 => assert!(cmd.prepare(&mut store).is_err()),
            3 => {
                let plan = cmd.prepare(&mut store).unwrap();
                assert!(store.commit_batch_discard(plan).is_err());
            }
            _ => {
                let plan = cmd.prepare(&mut store).unwrap();
                store.replace_component(id(1), String::from("old")).unwrap();
                assert!(store.commit_batch(plan).is_err());
            }
        }
        assert_eq!(store.get::<String>(id(1)).unwrap(), "old");
    }
    let mut cmd = store.commands();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cmd.update_batch((0..2).map(|n| {
            assert_eq!(n, 0, "iterator panic");
            (id(1), String::from("not committed"))
        }));
    }));
    assert!(result.is_err());
    cmd.update_batch(std::iter::from_fn(|| -> Option<(EntityId, String)> {
        panic!("first error must freeze subsequent input");
    }));
    assert!(cmd.apply_discard(&mut store).is_err());
    assert_eq!(store.get::<String>(id(1)).unwrap(), "old");
}

#[test]
fn native_target_reuses_receipt_ids_with_checked_index() {
    assert_eq!(std::mem::size_of::<NativeTarget>(), 16);
    let entity = EntityId::from_u128(1);
    let mut store = Store::new();
    store.insert((entity, (1u64,))).unwrap();
    let loc = store.allocator.get_location(entity).unwrap();
    let arch = &store.archetypes[loc.archetype_id as usize];
    let mut values = Values::<u64>::new_lane(ComponentTypeId::of::<u64>());
    values.rows.push(NativeRow { value: Some(2), next: None });
    if let Some(overflow) = (u32::MAX as usize).checked_add(1) {
        assert!(matches!(values.try_plan_single(0, entity, loc, arch, overflow),
            Err(StateError::CapacityExhausted(_))));
        assert!(values.targets.is_empty());
        assert_eq!(values.rows[0].value, Some(2));
    }
    assert!(values.try_plan_single(0, entity, loc, arch, u32::MAX as usize).unwrap());
    assert_eq!(values.targets[0].receipt_index, u32::MAX);
    assert_eq!(values.targets[0].row_index, loc.row_index);
}

#[test]
fn linked_targets_do_not_inherit_compact_lane_index_limit() {
    let id = EntityId::from_u128(1);
    let mut store = Store::new();
    store.insert((id, (1u64,))).unwrap();
    let location = store.allocator.get_location(id).unwrap();
    let mut values = Values {
        type_id: ComponentTypeId::of::<u64>(),
        type_verified: false,
        last_arch_id: u32::MAX,
        last_lane_arch_idx: 0,
        rows: vec![NativeRow { value: Some(2u64), next: None }],
        targets: Vec::new(),
        validated_archetypes: ArchetypeValidationCache::default(),
        touched_archetypes: vec![0; u16::MAX as usize + 1],
        pointer_scratch: Vec::new(),
        epoch_pointer_scratch: Vec::new(),
        linked_bound_ptr: AtomicPtr::new(std::ptr::null_mut()),
        linked_chunk_ep_ptr: AtomicPtr::new(std::ptr::null_mut()),
    };
    let arch = &store.archetypes[location.archetype_id as usize];
    assert!(values.plan_linked(0, id, location, arch).is_ok());
    assert_eq!(values.touched_archetypes.len(), u16::MAX as usize + 1);
    // Compact indices remain u16; excess tables use the existing full-ID
    // linked planner, instead of rejecting merged same-type calls.
    assert!(!values.try_plan_single(0, id, location, arch, 0).unwrap());
    assert!(values.targets.is_empty());
    assert_eq!(values.rows[0].value, Some(2));
    assert!(values.plan_linked(0, id, location, arch).is_ok());
    values.touched_archetypes.clear();
    assert!(values.try_plan_single(0, id, location, arch, 0).unwrap());
    assert_eq!(values.touched_archetypes, vec![location.archetype_id]);
}

#[test]
fn receipt_covers_native_general_structural_and_empty_deletes() {
    let id = EntityId::from_u128;
    let mut store = Store::new();
    for n in 1..=8 {
        store.insert((id(n), (n as u64, true, 3u16))).unwrap();
    }
    store.insert((id(9), ())).unwrap();
    let mut cmd = store.commands();
    cmd.update_batch([(id(1), 10u64), (id(2), 20u64)]);
    cmd.update_batch([(id(2), false)]);
    cmd.entity(id(3)).replace(30u64).replace(false);
    cmd.entity(id(4)).upsert(40u64);
    cmd.entity(id(5)).upsert(50u64);
    cmd.entity(id(6))
        .replace(60u64)
        .remove::<bool>()
        .insert(6u32);
    cmd.entity(id(7)).upsert(7u32);
    cmd.delete_entity(id(8));
    cmd.delete_entity(id(9));
    cmd.insert((id(10), (10u64, true, 3u16)));
    let prepared = cmd.prepare(&mut store).unwrap();
    assert_eq!(prepared.native_plan_count, 1);
    let receipt = store.commit_batch(prepared).unwrap();
    assert_eq!(receipt.created_entities, vec![id(10)]);
    assert_eq!(receipt.updated_entities.len(), 7);
    assert_eq!(receipt.deleted_entities, vec![id(8), id(9)]);
    assert!(!store.contains_entity(id(8)));
    assert!(!store.contains_entity(id(9)));
    assert!(store.contains_entity(id(10)));
    store.validate_invariants().unwrap();
}

#[test]
fn single_native_targets_do_not_build_erased_in_place_plans() {
    let mut store = Store::new();
    let ids = [3, 1, 2].map(EntityId::from_u128);
    for (n, id) in ids.into_iter().enumerate() {
        store.insert((id, (n as u64,))).unwrap();
    }
    let mut cmd = store.commands();
    cmd.update_batch(ids.into_iter().map(|id| (id, 42u64)));
    let prepared = cmd.prepare(&mut store).unwrap();
    assert_eq!(prepared.native_plan_count, 3);
    assert_eq!(prepared.native_plans.len(), 1);
    assert_eq!(prepared.native_plans[0].planned_len(), 3);
    assert!(prepared.plan_scratch.in_place_plans.is_empty());
    assert!(prepared.plans.is_empty());
    assert_eq!(prepared.in_place_plan_count(), 3);
    for (n, id) in ids.into_iter().enumerate() {
        assert_eq!(*store.entity(id).unwrap().get::<u64>().unwrap(), n as u64);
    }
    let receipt = store.commit_batch(prepared).unwrap();
    assert_eq!(receipt.updated_entities.len(), 3);
    for id in ids {
        assert_eq!(*store.entity(id).unwrap().get::<u64>().unwrap(), 42);
    }
}

#[test]
fn native_links_are_compact_and_checked_before_index_conversion() {
    #[allow(dead_code)]
    enum PreviousEntryLayout {
        Create(EntityBundle),
        Patch(EntityPatch),
        Delete,
        NativeUpdate([usize; 2]),
    }
    assert_eq!(
        std::mem::size_of::<CommandEntry>(),
        std::mem::size_of::<PreviousEntryLayout>()
    );
    assert_eq!(std::mem::size_of::<CommandEntry>(), 240);
    assert_eq!(std::mem::size_of::<Option<NativeUpdateRef>>(), 8);
    assert_eq!(std::mem::size_of::<NativeRow<u64>>(), 24);
    let last = NativeUpdateRef::new(u32::MAX as usize, u32::MAX as usize - 1).unwrap();
    assert_eq!(last.row(), u32::MAX as usize - 1);
    assert_eq!(last.lane(), u32::MAX as usize);
    assert!(NativeUpdateRef::new(0, u32::MAX as usize).is_err());
    assert!(NativeUpdateRef::new(usize::MAX, 0).is_err());
}

#[test]
fn bulk_values_stay_native_when_ordinary_operations_are_attached() {
    let mut store = Store::new();
    let id = EntityId::from_u128(1);
    store.insert((id, (10u64, true))).unwrap();
    let mut cmd = store.commands();
    cmd.update_batch([(id, 20u64)]);
    assert!(matches!(cmd.entries[0].1, CommandEntry::NativeUpdate(_)));
    assert_eq!(cmd.native_values.len(), 1);
    assert_eq!(cmd.counts().in_place_patches, 1);
    cmd.entity(id).remove::<bool>();
    match &cmd.entries[0].1 {
        CommandEntry::Patch(patch) => {
            assert!(patch.native_head.is_some());
            assert_eq!(patch.len(), 1);
            assert!(!patch.ops().contains_key(&ComponentTypeId::of::<u64>()));
        }
        _ => panic!("native replacement must not become a Patch"),
    }
    assert_eq!(cmd.counts().in_place_patches, 0);
    assert_eq!(cmd.counts().structural_patches, 1);
    assert_eq!(cmd.component_op_count, 2);
    let p = cmd.prepare(&mut store).unwrap();
    assert_eq!(p.plan_scratch.in_place_plans.len(), 0);
    assert_eq!(p.plans.len(), 1);
}

#[test]
fn multiple_native_components_and_both_mixed_orders_keep_lane_references() {
    for ordinary_first in [false, true] {
        let mut store = Store::new();
        let id = EntityId::from_u128(1);
        store.insert((id, (1u64, 2u32, true))).unwrap();
        let mut cmd = store.commands();
        if ordinary_first {
            cmd.entity(id).remove::<bool>();
        }
        cmd.update_batch([(id, 10u64)]);
        cmd.update_batch([(id, 20u32)]);
        if !ordinary_first {
            cmd.entity(id).remove::<bool>();
        }
        assert_eq!(cmd.native_values.len(), 2);
        let head = cmd.entries[0].1.native_head();
        assert!(contains_type(
            head,
            &cmd.native_values,
            ComponentTypeId::of::<u64>()
        ));
        assert!(contains_type(
            head,
            &cmd.native_values,
            ComponentTypeId::of::<u32>()
        ));
        match &cmd.entries[0].1 {
            CommandEntry::Patch(patch) => {
                assert!(patch.native_head.is_some());
                assert_eq!(patch.len(), 1);
            }
            _ => panic!("ordinary patch must not own native values"),
        }
        let prepared = cmd.prepare(&mut store).unwrap();
        assert_eq!(prepared.plans.len(), 1);
        assert_eq!(prepared.plan_scratch.in_place_plans.len(), 0);
        let receipt = store.commit_batch(prepared).unwrap();
        assert_eq!(receipt.updated_entities, vec![id]);
        assert_eq!(*store.entity(id).unwrap().get::<u64>().unwrap(), 10);
        assert_eq!(*store.entity(id).unwrap().get::<u32>().unwrap(), 20);
        assert!(!store.entity(id).unwrap().has::<bool>());
    }
}

#[test]
fn native_unknown_definition_failure_does_not_publish_metadata() {
    let mut store = Store::new();
    let id = EntityId::from_u128(1);
    store.insert((id, (1u64,))).unwrap();
    let mut cmd = store.commands();
    cmd.update_batch([(id, 10u64)]);
    cmd.update_batch([(id, 20u32)]);
    let epoch = store.epoch();
    assert!(matches!(
        cmd.prepare(&mut store),
        Err(StateError::ComponentNotFoundInEntity { .. })
    ));
    assert!(
        store
            .component_registry
            .get(&ComponentTypeId::of::<u32>())
            .is_none()
    );
    assert_eq!(store.epoch(), epoch);
    assert_eq!(*store.entity(id).unwrap().get::<u64>().unwrap(), 1);
}

#[test]
fn explicit_internal_into_batch_preserves_native_chains() {
    let mut store = Store::new();
    let id = EntityId::from_u128(1);
    store.insert((id, (1u64, 2u32, true))).unwrap();
    let mut cmd = store.commands();
    cmd.update_batch([(id, 10u64)]);
    cmd.entity(id).remove::<bool>();
    cmd.update_batch([(id, 20u32)]);
    let batch = cmd.into_batch().unwrap();
    let receipt = store.apply_batch(batch).unwrap();
    assert_eq!(receipt.updated_entities, vec![id]);
    assert_eq!(*store.entity(id).unwrap().get::<u64>().unwrap(), 10);
    assert_eq!(*store.entity(id).unwrap().get::<u32>().unwrap(), 20);
    assert!(!store.entity(id).unwrap().has::<bool>());
}

#[test]
fn bulk_empty_at_exhaustion_and_corrupt_preflight_do_not_write() {
    let mut store = Store::new();
    let a = EntityId::from_u128(1);
    let b = EntityId::from_u128(2);
    store.insert((a, (1u64,))).unwrap();
    store.insert((b, (2u64,))).unwrap();
    store.epoch = u64::MAX;
    let mut cmd = store.commands();
    cmd.update_batch::<(EntityId, u64)>([]);
    assert_eq!(cmd.native_values.len(), 0);
    assert!(cmd.apply(&mut store).unwrap().updated_entities.is_empty());
    let mut cmd = store.commands();
    cmd.update_batch([(a, 10u64)]);
    assert!(matches!(
        cmd.apply(&mut store),
        Err(StateError::CapacityExhausted(_))
    ));
    store.epoch = 10;
    let loc = store.allocator.get_location(b).unwrap();
    store.archetypes[loc.archetype_id as usize].entities_mut()[loc.row_index as usize] = a;
    let mut cmd = store.commands();
    cmd.update_batch([(a, 10u64), (b, 20u64)]);
    assert!(cmd.prepare(&mut store).is_err());
    assert_eq!(*store.entity(a).unwrap().get::<u64>().unwrap(), 1);
    assert_eq!(store.epoch(), 10);
}

#[test]
fn test_native_batch_type_witness_bypass_and_missing_component_safety() {
    let mut store = Store::new();
    let e1 = EntityId::from_u128(101);
    let e2 = EntityId::from_u128(102);
    store.insert((e1, (100u64,))).unwrap();
    store.insert((e2, (true,))).unwrap();

    let mut cmd = store.commands();
    cmd.update_batch([(e1, 200u64)]);
    let _receipt = cmd.apply(&mut store).unwrap();
    assert_eq!(*store.entity(e1).unwrap().get::<u64>().unwrap(), 200);

    let mut cmd2 = store.commands();
    cmd2.update_batch([(e2, 300u64)]);
    let err = cmd2.apply(&mut store).unwrap_err();
    match err {
        StateError::ComponentNotFoundInEntity { entity, .. } => {
            assert_eq!(entity, e2);
        }
        other => panic!("Expected ComponentNotFoundInEntity, got: {:?}", other),
    }
}

#[test]
fn test_multicomponent_tuple_update_recording() {
    let mut store = Store::new();
    let e1 = EntityId::from_u128(1);
    let e2 = EntityId::from_u128(2);
    let e3 = EntityId::from_u128(3);
    store.insert((e1, (10u64, 1.0f32))).unwrap();
    store.insert((e2, (20u64, 2.0f32))).unwrap();
    store.insert((e3, (30u64, 3.0f32))).unwrap();

    let mut cmd = store.commands();
    cmd.update_batch([(e1, 100u64, 10.0f32), (e2, 200u64, 20.0f32), (e3, 300u64, 30.0f32)]);

    assert_eq!(cmd.native_values.len(), 2);
    assert_eq!(cmd.index.len(), 3);
    assert_eq!(cmd.entries.len(), 3);
    assert_eq!(cmd.component_op_count, 6);

    let receipt = cmd.apply(&mut store).unwrap();
    assert_eq!(receipt.updated_entities, vec![e1, e2, e3]);
    assert_eq!(*store.entity(e1).unwrap().get::<u64>().unwrap(), 100);
    assert_eq!(*store.entity(e1).unwrap().get::<f32>().unwrap(), 10.0);
    assert_eq!(*store.entity(e2).unwrap().get::<u64>().unwrap(), 200);
    assert_eq!(*store.entity(e2).unwrap().get::<f32>().unwrap(), 20.0);
    assert_eq!(*store.entity(e3).unwrap().get::<u64>().unwrap(), 300);
    assert_eq!(*store.entity(e3).unwrap().get::<f32>().unwrap(), 30.0);
}

#[test]
fn test_multicomponent_tuple_quota_limits() {
    let mut store = Store::new();
    let e1 = EntityId::from_u128(1);
    let e2 = EntityId::from_u128(2);
    store.insert((e1, (10u64, 1.0f32))).unwrap();
    store.insert((e2, (20u64, 2.0f32))).unwrap();

    let limits = CommandLimits {
        max_entities: None,
        max_component_ops: Some(3),
    };
    let mut cmd = store.commands_with_limits(limits);
    cmd.update_batch([(e1, 100u64, 10.0f32), (e2, 200u64, 20.0f32)]);
    assert!(matches!(
        cmd.apply(&mut store),
        Err(StateError::CapacityExhausted(_))
    ));

    let limits = CommandLimits {
        max_entities: Some(1),
        max_component_ops: None,
    };
    let mut cmd2 = store.commands_with_limits(limits);
    cmd2.update_batch([(e1, 100u64, 10.0f32), (e2, 200u64, 20.0f32)]);
    assert!(matches!(
        cmd2.apply(&mut store),
        Err(StateError::CapacityExhausted(_))
    ));
}

#[test]
fn test_multicomponent_tuple_conflict_detection() {
    let mut store = Store::new();
    let e1 = EntityId::from_u128(1);
    store.insert((e1, (10u64, 1.0f32))).unwrap();

    let mut cmd = store.commands();
    cmd.update_batch([(e1, 100u64, 10.0f32), (e1, 200u64, 20.0f32)]);
    assert!(matches!(
        cmd.apply(&mut store),
        Err(StateError::ComponentDuplicateInPatch { .. })
    ));

    let mut cmd2 = store.commands();
    cmd2.update_batch([(e1, 100u64, 200u64)]);
    assert!(matches!(
        cmd2.apply(&mut store),
        Err(StateError::ComponentDuplicateInPatch { .. })
    ));
}
