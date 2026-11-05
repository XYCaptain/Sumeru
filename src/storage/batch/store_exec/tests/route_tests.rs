use super::*;
use serde::{Deserialize, Serialize};

#[test]
fn prescan_rejects_late_wrong_rust_type_without_writes() {
    use crate::binding::EntityId;
    use crate::error::StateError;
    use crate::storage::batch::{EntityPatch, StoreBatch};
    use crate::storage::ErasedComponentValue;
    use crate::storage::core::store::Store;

    for discard in [false, true] {
        let mut store = Store::new();
        let mut batch = StoreBatch::new();
        for n in 1..=301 {
            let id = EntityId::from_u128(n);
            store.insert((id, (10u64, false))).unwrap();
            let mut patch = EntityPatch::new();
            patch.replace(true).unwrap();
            if n == 301 {
                patch
                    .replace_erased(ErasedComponentValue::with_component_type_id(
                        ComponentTypeId::of::<u64>(),
                        -1i64,
                    ))
                    .unwrap();
            } else {
                patch.replace(20u64).unwrap();
            }
            batch.patch_entity(id, patch).unwrap();
        }
        let epoch = store.epoch();
        let result = if discard {
            store.prepare_batch_discard(batch)
        } else {
            store.prepare_batch(batch)
        };
        assert!(matches!(result, Err(StateError::AccessDenied(_))));
        assert_eq!(store.epoch(), epoch);
        for n in 1..=301 {
            let id = EntityId::from_u128(n);
            assert_eq!(store.get::<u64>(id), Some(&10));
            assert_eq!(store.get::<bool>(id), Some(&false));
        }
        store.validate_invariants().unwrap();
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Tag<const N: u8>;

fn id(n: u128) -> EntityId { EntityId::from_u128(n) }

#[test]
fn repeated_route_builds_one_signature_and_preserves_native_values() {
    for discard in [false, true] {
        for existing_target in [false, true] {
            let mut store = Store::new();
            for n in 1..=128 { store.insert((id(n), (n as u64, false))).unwrap(); }
            if existing_target { store.insert((id(999), (0u64, String::new()))).unwrap(); }
            let mut cmd = store.commands();
            cmd.update_batch((1..=128).map(|n| (id(n), n as u64 + 1000)));
            for n in 1..=128 {
                if n % 2 == 0 {
                    cmd.entity(id(n)).remove::<bool>().insert(format!("new-{n}"));
                } else {
                    cmd.entity(id(n)).upsert(format!("new-{n}")).remove::<bool>();
                }
            }
            ROUTE_SIGNATURE_BUILDS.with(|c| c.set(0));
            let plan = if discard { cmd.prepare_discard(&mut store) }
                else { cmd.prepare(&mut store) }.unwrap();
            assert_eq!(ROUTE_SIGNATURE_BUILDS.with(|c| c.get()), 1);
            let receipt = if discard { store.commit_batch_discard(plan).unwrap(); None }
                else { Some(store.commit_batch(plan).unwrap()) };
            for n in 1..=128 {
                assert_eq!(store.get::<u64>(id(n)), Some(&(n as u64 + 1000)));
                assert_eq!(store.get::<String>(id(n)), Some(&format!("new-{n}")));
                assert!(store.get::<bool>(id(n)).is_none());
            }
            if let Some(receipt) = receipt {
                assert_eq!(receipt.updated_entities.len(), 128);
            }
            store.validate_invariants().unwrap();
        }
    }
}

#[test]
fn route_hits_do_not_bypass_late_validation_or_version_checks() {
    for discard in [false, true] {
        let mut store = Store::new();
        for n in 1..=3 { store.insert((id(n), (n as u64,))).unwrap(); }
        let epoch = store.epoch();
        let mut cmd = store.commands();
        cmd.entity(id(1)).insert(true);
        cmd.entity(id(2)).insert(false).replace(String::from("missing"));
        let result = if discard { cmd.prepare_discard(&mut store) } else { cmd.prepare(&mut store) };
        assert!(matches!(result, Err(StateError::ComponentNotFoundInEntity { entity, .. }) if entity == id(2)));
        assert_eq!(store.epoch(), epoch);
        for n in 1..=3 { assert!(store.get::<bool>(id(n)).is_none()); }
        let mut cmd = store.commands();
        cmd.entity(id(1)).insert(true);
        cmd.entity(id(2)).insert(false);
        let plan = if discard { cmd.prepare_discard(&mut store) } else { cmd.prepare(&mut store) }.unwrap();
        store.replace_component(id(3), 300u64).unwrap();
        let result = if discard { store.commit_batch_discard(plan) }
            else { store.commit_batch(plan).map(|_| ()) };
        assert!(matches!(result, Err(StateError::StaleBatch { .. })));
        assert!(store.get::<bool>(id(1)).is_none());
        store.validate_invariants().unwrap();
    }
}

#[test]
fn route_cache_is_source_specific_and_bounded() {
    let mut store = Store::new();
    // 65 source layouts, each visited twice; first 64 routes are retained.
    for n in 0..130u128 {
        let pattern = n % 65;
        let mut bundle = EntityBundle::new().with(n as u64).unwrap();
        macro_rules! tag { ($bit:literal) => { if pattern & (1 << $bit) != 0 { bundle.insert(Tag::<$bit>).unwrap(); } }; }
        tag!(0); tag!(1); tag!(2); tag!(3); tag!(4); tag!(5); tag!(6);
        store.create_entity(id(n + 1), bundle).unwrap();
    }
    let mut cmd = store.commands();
    for n in 1..=130 { cmd.entity(id(n)).insert(true); }
    ROUTE_SIGNATURE_BUILDS.with(|c| c.set(0));
    let plan = cmd.prepare(&mut store).unwrap();
    assert_eq!(ROUTE_SIGNATURE_BUILDS.with(|c| c.get()), 66);
    let receipt = store.commit_batch(plan).unwrap();
    assert_eq!(receipt.updated_entities.len(), 130);
    for n in 0..130u128 {
        assert_eq!(store.get::<u64>(id(n + 1)), Some(&(n as u64)));
        assert_eq!(store.get::<bool>(id(n + 1)), Some(&true));
        assert_eq!(store.get::<Tag<0>>(id(n + 1)).is_some(), n % 65 & 1 != 0);
    }
    store.validate_invariants().unwrap();
}

#[test]
fn large_deltas_fall_back_without_losing_changes() {
    for discard in [false, true] {
        let mut store = Store::new();
        for n in 1..=2 { store.insert((id(n), (n as u64,))).unwrap(); }
        let mut cmd = store.commands();
        for n in 1..=2 {
            cmd.entity(id(n)).insert(Tag::<0>).insert(Tag::<1>).insert(Tag::<2>)
                .insert(Tag::<3>).insert(Tag::<4>);
        }
        ROUTE_SIGNATURE_BUILDS.with(|c| c.set(0));
        let plan = if discard { cmd.prepare_discard(&mut store) } else { cmd.prepare(&mut store) }.unwrap();
        assert_eq!(ROUTE_SIGNATURE_BUILDS.with(|c| c.get()), 2);
        if discard { store.commit_batch_discard(plan).unwrap(); }
        else {
            let receipt = store.commit_batch(plan).unwrap();
            assert_eq!(store.entity(id(1)).unwrap().component_types().len(), 6);
            assert_eq!(receipt.updated_entities.len(), 2);
        }
        for n in 1..=2 {
            assert!(store.get::<Tag<4>>(id(n)).is_some());
            assert_eq!(store.get::<u64>(id(n)), Some(&(n as u64)));
        }
        store.validate_invariants().unwrap();
    }
}
