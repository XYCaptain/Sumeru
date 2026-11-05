use super::*;
use crate::{ComponentRegistration, EntityBundle};

fn codecs(store: &mut Store) {
    store
        .register_components([
            ComponentRegistration::of::<u64>(),
            ComponentRegistration::of::<f64>(),
            ComponentRegistration::of::<String>(),
        ])
        .unwrap();
}

#[test]
fn roundtrip_preserves_full_ids_slot_generations_payloads_and_unaligned_input() {
    let mut source = Store::new();
    codecs(&mut source);
    let first = EntityId::from_u128(u128::MAX);
    source
        .create_entity(
            first,
            EntityBundle::new()
                .with(u64::MAX)
                .unwrap()
                .with(f64::NAN)
                .unwrap()
                .with("列车🚄".to_owned())
                .unwrap(),
        )
        .unwrap();
    source.spawn(EntityId::from_u128(1)).unwrap();
    source.delete_entity(EntityId::from_u128(1)).unwrap();
    source
        .create_entity(
            EntityId::from_u128(2),
            EntityBundle::new().with(42u64).unwrap(),
        )
        .unwrap();
    let snapshot = source.snapshot().unwrap();
    let expected_metadata = serde_json::to_value(&snapshot).unwrap();
    let encoded = snapshot.into_binary().unwrap();
    let mut unaligned = vec![0];
    unaligned.extend_from_slice(&encoded);
    let decoded = ColumnarStoreSnapshot::from_binary(&unaligned[1..]).unwrap();
    assert_eq!(serde_json::to_value(&decoded).unwrap(), expected_metadata);
    assert_eq!(decoded.clone().into_binary().unwrap(), encoded);
    let mut restored = Store::new();
    codecs(&mut restored);
    restored.restore_snapshot(&decoded).unwrap();
    assert_eq!(restored.get::<u64>(first), Some(&u64::MAX));
    assert_eq!(restored.get::<String>(first).unwrap(), "列车🚄");
    assert!(restored.get::<f64>(first).unwrap().is_nan());
    assert_eq!(restored.get::<u64>(EntityId::from_u128(2)), Some(&42));
    assert!(!restored.contains_entity(EntityId::from_u128(1)));
}

#[test]
fn malformed_envelope_and_unsupported_versions_are_rejected() {
    let encoded = Store::new().snapshot().unwrap().into_binary().unwrap();
    for stop in 0..encoded.len() {
        assert!(ColumnarStoreSnapshot::from_binary(&encoded[..stop]).is_err());
    }
    let mut wrong_magic = encoded.clone();
    wrong_magic[0] ^= 1;
    assert!(ColumnarStoreSnapshot::from_binary(&wrong_magic).is_err());
    let mut extra = encoded.clone();
    extra.push(0);
    assert!(ColumnarStoreSnapshot::from_binary(&extra).is_err());
    let mut corrupt_root = encoded.clone();
    corrupt_root[PREFIX..].fill(0xff);
    assert!(ColumnarStoreSnapshot::from_binary(&corrupt_root).is_err());
    let mut aligned = rkyv::util::AlignedVec::<16>::new();
    aligned.extend_from_slice(&encoded[PREFIX..]);
    let mut envelope =
        rkyv::from_bytes::<Envelope, rkyv::rancor::BoxedError>(&aligned).unwrap();
    envelope.version = CURRENT_SNAPSHOT_VERSION + 1;
    let bad = rkyv::to_bytes::<rkyv::rancor::BoxedError>(&envelope).unwrap();
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&(bad.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&bad);
    assert!(ColumnarStoreSnapshot::from_binary(&bytes).is_err());
    assert!(ColumnarStoreSnapshot::from_binary(br#"{"format_version":1}"#).is_err());
}

#[test]
fn envelope_does_not_repair_forged_column_identity_or_bypass_native_validation() {
    let mut source = Store::new();
    codecs(&mut source);
    source.insert((EntityId::from_u128(1), (1u64,))).unwrap();
    let mut snapshot = source.snapshot().unwrap();
    snapshot
        .archetypes_data
        .iter_mut()
        .flat_map(|table| table.columns.values_mut())
        .next()
        .unwrap()
        .component_type_id = ComponentTypeId::new(0);
    let encoded = snapshot.into_binary().unwrap();
    let decoded = ColumnarStoreSnapshot::from_binary(&encoded).unwrap();
    assert_eq!(
        decoded
            .archetypes_data
            .iter()
            .flat_map(|table| table.columns.values())
            .next()
            .unwrap()
            .component_type_id,
        ComponentTypeId::new(0)
    );
    let mut restored = Store::new();
    codecs(&mut restored);
    restored.insert((EntityId::from_u128(9), (9u64,))).unwrap();
    assert!(restored.restore_snapshot(&decoded).is_err());
    assert_eq!(restored.get::<u64>(EntityId::from_u128(9)), Some(&9));
    assert!(!restored.contains_entity(EntityId::from_u128(1)));
}

#[test]
fn bincode_snapshot_compact_metadata_under_1kb_for_10k_strided_entities() {
    let mut source = Store::new();
    codecs(&mut source);
    for i in 0..10_000u64 {
        let id = EntityId::from_u128(1_000 + (i as u128) * 4);
        source.insert((id, (i,))).unwrap();
    }

    let snapshot = source.snapshot().unwrap();
    let encoded = bincode::serialize(&snapshot).unwrap();
    let column_payload_bytes: usize = snapshot
        .archetypes_data
        .iter()
        .flat_map(|arch| arch.columns.values())
        .map(|col| col.payload.len())
        .sum();
    let metadata_bytes = encoded.len().saturating_sub(column_payload_bytes);
    assert!(
        metadata_bytes < 1024,
        "Expected compact snapshot metadata < 1024 bytes, got {metadata_bytes} bytes"
    );

    let decoded: ColumnarStoreSnapshot = bincode::deserialize(&encoded).unwrap();
    let mut restored = Store::new();
    codecs(&mut restored);
    restored.restore_snapshot(&decoded).unwrap();
    assert_eq!(restored.len(), 10_000);
    for i in [0u64, 1, 4999, 9999] {
        let id = EntityId::from_u128(1_000 + (i as u128) * 4);
        assert_eq!(restored.get::<u64>(id), Some(&i));
    }
}

#[test]
fn test_store_merge_disjoint_fast_and_general_paths_and_rejection() {
    // 1. Fast path: empty target store with total_slots == 0 adopts source directly
    let mut source = Store::new();
    codecs(&mut source);
    let e1 = EntityId::from_u128(1);
    let e2 = EntityId::from_u128(2);
    source.insert((e1, (10u64,))).unwrap();
    source.insert((e2, (20u64, "two".to_string()))).unwrap();

    let mut target = Store::new();
    assert!(target.can_merge_disjoint(&source));
    target.merge_disjoint_from(source).unwrap();
    assert_eq!(target.len(), 2);
    assert_eq!(target.get::<u64>(e1), Some(&10));
    assert_eq!(target.get::<u64>(e2), Some(&20));
    assert_eq!(target.get::<String>(e2).map(String::as_str), Some("two"));

    // 2. General path with free slots in target (after deleting an entity so total_slots > 0),
    // overlapping archetype (u64,) AND new disjoint archetype (String,)
    let temp_id = EntityId::from_u128(99);
    target.insert((temp_id, (990u64,))).unwrap();
    target.delete_entity(temp_id).unwrap(); // creates a free slot in target.allocator

    let mut other = Store::new();
    codecs(&mut other);
    let e3 = EntityId::from_u128(3);
    let e4 = EntityId::from_u128(4);
    let e5 = EntityId::from_u128(5);
    // Overlapping archetype (u64,) with 2 entities (one reuses free slot, one allocates fresh)
    other.insert((e3, (30u64,))).unwrap();
    other.insert((e4, (40u64,))).unwrap();
    // Disjoint archetype (String,) not yet present as solo archetype in target
    other.insert((e5, ("five".to_string(),))).unwrap();

    assert!(target.can_merge_disjoint(&other));
    target.merge_disjoint_from(other).unwrap();
    assert_eq!(target.len(), 5);
    assert_eq!(target.get::<u64>(e1), Some(&10));
    assert_eq!(target.get::<u64>(e2), Some(&20));
    assert_eq!(target.get::<u64>(e3), Some(&30));
    assert_eq!(target.get::<u64>(e4), Some(&40));
    assert_eq!(target.get::<String>(e5).map(String::as_str), Some("five"));

    // 3. General path where target had all entities deleted (is_empty() == true, total_slots() > 0)
    let mut emptied_target = Store::new();
    let e_dead = EntityId::from_u128(500);
    emptied_target.insert((e_dead, (500u64,))).unwrap();
    emptied_target.delete_entity(e_dead).unwrap();
    assert!(emptied_target.is_empty());

    let mut other2 = Store::new();
    let e6 = EntityId::from_u128(6);
    other2.insert((e6, (60u64,))).unwrap();
    assert!(emptied_target.can_merge_disjoint(&other2));
    emptied_target.merge_disjoint_from(other2).unwrap();
    assert_eq!(emptied_target.len(), 1);
    assert_eq!(emptied_target.get::<u64>(e6), Some(&60));

    // 4. Reject overlapping EntityId without mutating target
    let mut conflicting = Store::new();
    conflicting.insert((e3, (333u64,))).unwrap();
    assert!(!target.can_merge_disjoint(&conflicting));
    assert!(matches!(
        target.merge_disjoint_from(conflicting),
        Err(StateError::AccessDenied(_))
    ));
    assert_eq!(target.get::<u64>(e3), Some(&30));
}

#[test]
fn test_restore_snapshot_owned_with_holes_respawns_and_swapped_rows() {
    let mut source = Store::new();
    codecs(&mut source);

    // Populate two archetypes:
    // Archetype A: (u64,) with 6 entities
    let a_ids: Vec<EntityId> = (1..=6).map(EntityId::from_u128).collect();
    for (i, &id) in a_ids.iter().enumerate() {
        source.insert((id, ((i as u64 + 1) * 10,))).unwrap();
    }
    // Delete a_ids[1] (triggers swap_remove with a_ids[5], making archetype entities unsorted)
    // and delete a_ids[3] (creates another free slot)
    source.delete_entity(a_ids[1]).unwrap();
    source.delete_entity(a_ids[3]).unwrap();

    // Respawn a new entity into one of the free slots (generation = 2) in Archetype B: (u64, String)
    let respawned = EntityId::from_u128(100);
    source
        .insert((respawned, (1000u64, "respawned".to_string())))
        .unwrap();
    let b2 = EntityId::from_u128(101);
    source.insert((b2, (1010u64, "b2".to_string()))).unwrap();

    // Take snapshot, roundtrip through bincode, and restore via restore_snapshot_owned
    let snap = source.snapshot().unwrap();
    let bytes = bincode::serialize(&snap).unwrap();
    let decoded: ColumnarStoreSnapshot = bincode::deserialize(&bytes).unwrap();

    let mut restored = Store::new();
    codecs(&mut restored);
    // Insert pre-existing data that restore_snapshot_owned must cleanly replace
    restored
        .insert((EntityId::from_u128(999), (999u64,)))
        .unwrap();
    restored.restore_snapshot_owned(decoded).unwrap();

    assert!(!restored.contains_entity(EntityId::from_u128(999)));
    assert!(!restored.contains_entity(a_ids[1]));
    assert!(!restored.contains_entity(a_ids[3]));
    assert_eq!(restored.len(), 6);
    assert_eq!(restored.get::<u64>(a_ids[0]), Some(&10));
    assert_eq!(restored.get::<u64>(a_ids[2]), Some(&30));
    assert_eq!(restored.get::<u64>(a_ids[4]), Some(&50));
    assert_eq!(restored.get::<u64>(a_ids[5]), Some(&60));
    assert_eq!(restored.get::<u64>(respawned), Some(&1000));
    assert_eq!(
        restored.get::<String>(respawned).map(String::as_str),
        Some("respawned")
    );
    assert_eq!(restored.get::<u64>(b2), Some(&1010));
    assert_eq!(restored.get::<String>(b2).map(String::as_str), Some("b2"));
}

