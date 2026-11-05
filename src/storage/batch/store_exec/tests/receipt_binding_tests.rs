//! Mechanism gates: receipt pool indices and column cache tracking.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Value<const N: usize>(u64);

fn id(i: usize) -> EntityId {
    EntityId::from_u128(i as u128 + 1)
}

#[test]
fn receipt_pool_endpoint_check_matches_linear_reference_for_monotonic_indices() {
    for indices in [
        vec![], vec![0], vec![u32::MAX], vec![1, 2, 3],
        vec![0, 2, 4], vec![0, u32::MAX],
        vec![u32::MAX - 2, u32::MAX - 1, u32::MAX],
    ] {
        let plans: Vec<_> = indices.iter().enumerate().map(|(i, &pool_index)| {
            PlannedInPlacePatch {
                entity: id(i),
                location: EntityLocation { archetype_id: 0, row_index: i as u32 },
                replacements: InPlaceReplacements::Empty,
                pool_index,
            }
        }).collect();
        for start in 0..=plans.len() {
            for end in start..=plans.len() {
                let segment = &plans[start..end];
                let expected = segment.first().is_none_or(|first| {
                    segment.iter().enumerate().all(|(offset, plan)| {
                        first.pool_index.checked_add(offset as u32) == Some(plan.pool_index)
                    })
                });
                assert_eq!(super::super::validation::receipt_pool_range_is_contiguous::<false>(segment), expected);
                assert_eq!(super::super::validation::receipt_pool_range_is_contiguous::<true>(segment), expected);
            }
        }
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "prepared receipt pool indices must increase strictly")]
fn receipt_pool_endpoint_check_rejects_duplicate_indices_in_debug() {
    let plans: Vec<_> = [0, 2, 2].into_iter().enumerate().map(|(i, pool_index)| {
        PlannedInPlacePatch {
            entity: id(i),
            location: EntityLocation { archetype_id: 0, row_index: i as u32 },
            replacements: InPlaceReplacements::Empty,
            pool_index,
        }
    }).collect();
    super::super::validation::receipt_pool_range_is_contiguous::<false>(&plans);
}

#[test]
fn receipt_pool_indices_stay_increasing_across_native_structural_and_single_updates() {
    let mut store = Store::new();
    for i in 0..10 {
        store.insert((id(i), (Value::<0>(0), Value::<1>(1), Value::<2>(2), 0u64))).unwrap();
    }
    let mut commands = store.commands();
    commands.entity(id(8)).replace(Value::<0>(10)).replace(Value::<1>(11)).replace(Value::<2>(12));
    commands.update_batch([(id(1), 10u64)]);
    commands.entity(id(2)).replace(Value::<0>(20)).replace(Value::<1>(21)).replace(Value::<2>(22));
    commands.entity(id(3)).replace(Value::<0>(30)).insert(Value::<99>(99));
    commands.entity(id(4)).replace(Value::<0>(40)).replace(Value::<1>(41)).replace(Value::<2>(42));
    commands.update_batch([(id(5), 50u64)]);
    commands.entity(id(5)).replace(Value::<0>(50)).replace(Value::<1>(51));
    commands.update_batch([(id(6), 60u64)]);
    commands.update_batch([(id(6), Value::<0>(60))]);
    commands.delete_entity(id(7));
    commands.entity(id(0)).replace(Value::<0>(100));
    let prepared = commands.prepare(&mut store).unwrap();
    let plans = &prepared.plan_scratch.in_place_plans;
    assert!(plans.windows(2).all(|p| p[0].pool_index < p[1].pool_index));
    assert!(plans.windows(2).any(|p| p[1].pool_index - p[0].pool_index > 1));
    assert!(plans.iter().any(|p| matches!(p.replacements, InPlaceReplacements::NativeMultiple(_))));
    for start in 0..plans.len() {
        for end in start + 1..=plans.len() {
            let expected = plans[start..end].iter().enumerate().all(|(offset, plan)| {
                plans[start].pool_index + offset as u32 == plan.pool_index
            });
            assert_eq!(super::super::validation::receipt_pool_range_is_contiguous::<false>(&plans[start..end]), expected);
            assert_eq!(super::super::validation::receipt_pool_range_is_contiguous::<true>(&plans[start..end]), expected);
        }
    }
    let receipt = store.commit_batch(prepared).unwrap();
    assert_eq!(store.get::<u64>(id(6)), Some(&60));
    assert!(!store.contains_entity(id(7)));
    assert_eq!(receipt.deleted_entities, vec![id(7)]);
    store.validate_invariants().unwrap();
}

#[test]
fn ordinary_receipt_segments_preserve_permuted_and_unequal_component_sets() {
    for upsert in [false, true] {
        let mut store = Store::new();
        for i in 0..4 {
            store.insert((id(i), (Value::<0>(0), Value::<1>(1), Value::<2>(2), Value::<3>(3)))).unwrap();
        }
        let mut commands = store.commands();
        for i in 0..4 {
            let mut entity = commands.entity(id(i));
            macro_rules! write { ($k:literal) => {
                if upsert { entity.upsert(Value::<$k>(10 + $k)); }
                else { entity.replace(Value::<$k>(10 + $k)); }
            }; }
            if i == 1 { write!(2); write!(1); write!(0); }
            else { write!(0); write!(1); write!(2); }
            if i == 2 { write!(3); }
        }
        let prepared = commands.prepare(&mut store).unwrap();
        let receipt = store.commit_batch(prepared).unwrap();
        assert_eq!(receipt.updated_entities.len(), 4);
        for i in 0..4 {
            assert_eq!(store.get::<Value<0>>(id(i)), Some(&Value(10)));
            assert_eq!(store.get::<Value<2>>(id(i)), Some(&Value(12)));
            assert_eq!(store.get::<Value<3>>(id(i)), Some(&Value(if i == 2 { 13 } else { 3 })));
        }
        store.validate_invariants().unwrap();
    }
}

#[test]
fn forward_column_cache_tracks_only_touched_backings_and_first_visits() {
    let mut store = Store::new();
    store.insert((id(0), (0u64,))).unwrap();
    store.insert((id(1), (0u64, true))).unwrap();
    let loc0 = store.allocator.get_location(id(0)).unwrap();
    let loc1 = store.allocator.get_location(id(1)).unwrap();
    let cid = ComponentTypeId::of::<u64>();
    for enabled in [false, true] {
        let mut cache = ColumnCacheTracker::new(enabled);
        cache.visit_archetype(loc1.archetype_id);
        cache.validate_or_lookup(&store.archetypes[loc1.archetype_id as usize], cid, &store.component_registry).unwrap();
        cache.visit_archetype(loc0.archetype_id);
        cache.get_or_insert_column(&store.archetypes[loc0.archetype_id as usize], cid).unwrap();
    }
    let mut cache = ColumnCacheTracker::new(true);
    let mut first_visits = 0;
    for _ in 0..64 {
        for loc in [loc0, loc1] {
            cache.visit_archetype(loc.archetype_id);
            first_visits += usize::from(cache.visit_column(&store.archetypes[loc.archetype_id as usize], cid).unwrap().1);
            cache.deactivate();
        }
    }
    assert_eq!(first_visits, 2, "one binding per table/column, including direct columns");
}
