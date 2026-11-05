//! Diagnostic subpaths, NOT additive time attribution or public handle APIs.
//! Pre-resolved positions exist only in this cfg(test) module, in a fixture
//! with no structural changes. Each mode gets its own store and warmup.
use crate::id::{ComponentTypeId, EntityId};
use crate::storage::core::allocator::EntityLocation;
use crate::storage::core::store::Store;
use serde::{Deserialize, Serialize};
use std::{hint::black_box, time::Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Value(u64);
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Untouched(u64);

fn sample(
    scale: usize,
    events: &[usize],
    mode: &str,
    round: usize,
    writes: bool,
    mut operation: impl FnMut(&mut Store, EntityId, EntityLocation, u64) -> u64,
) {
    let mut store = Store::new();
    store.reserve_entities(scale).unwrap();
    let ids: Vec<_> = (1..=scale).map(|i| EntityId::from_u128(
        (i as u128).wrapping_mul(0x9e3779b97f4a7c15d1b54a32d192ed03)
    )).collect();
    for (i, &id) in ids.iter().enumerate() {
        store.insert((id, (Value(0), Untouched(i as u64)))).unwrap();
    }
    let locations: Vec<_> = ids.iter().map(|&id| store.allocator.get_location(id).unwrap()).collect();
    for (&id, &location) in ids.iter().zip(&locations) {
        assert_eq!(store.archetypes[location.archetype_id as usize].entities()[location.row_index as usize], id);
    }
    for &i in events.iter().take(4096) {
        black_box(operation(&mut store, ids[i], locations[i], 0));
    }
    let mut expected_values = vec![0u64; scale];
    let mut expected_sum = 0u64;
    for (step, &i) in events.iter().enumerate() {
        if writes {
            expected_sum += expected_values[i];
            expected_values[i] = step as u64 + 1;
        } else {
            expected_sum += locations[i].row_index as u64;
        }
    }
    let epoch = store.epoch();
    let started = Instant::now();
    let mut sum = 0u64;
    for (step, &i) in events.iter().enumerate() {
        sum += operation(&mut store, ids[i], locations[i], step as u64 + 1);
    }
    black_box(sum);
    let ns = started.elapsed().as_nanos() as f64 / events.len() as f64;
    assert_eq!(sum, expected_sum);
    assert_eq!(store.epoch(), epoch + if writes { events.len() as u64 } else { 0 });
    assert_eq!(store.len(), scale);
    for (i, &id) in ids.iter().enumerate() {
        let view = store.entity(id).unwrap();
        assert_eq!(view.get::<Value>().unwrap().0, expected_values[i]);
        assert_eq!(view.get::<Untouched>().unwrap().0, i as u64);
    }
    println!("STAGE_RESULT {{\"scale\":{scale},\"events\":{},\"round\":{round},\"mode\":\"{mode}\",\"ns_per_event\":{ns},\"oracle\":true}}", events.len());
}

#[test]
#[ignore = "in-crate internal benchmark"]
fn point_write_stages() {
    let scale: usize = std::env::var("STATE_POINT_SCALE")
        .map(|v| v.parse().unwrap()).unwrap_or(100_000);
    assert!(scale > 0);
    let mut rng = 12345u64;
    let events: Vec<_> = (0..200_000).map(|_| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng as usize % scale
    }).collect();
    for round in 0..3 {
        let mut modes = ["lookup", "validate_at", "replace_at", "validated_replace_at", "upsert", "entity_set"];
        modes.rotate_left(round * 2);
        for mode in modes {
            match mode {
                "lookup" => sample(scale, &events, mode, round, false, |s, id, _, _| {
                    s.allocator.get_location(id).unwrap().row_index as u64
                }),
                "validate_at" => sample(scale, &events, mode, round, false, |s, id, loc, _| {
                    s.validate_component_location(id, ComponentTypeId::of::<Value>(), loc).unwrap().row_index as u64
                }),
                "replace_at" => sample(scale, &events, mode, round, true, |s, id, loc, value| {
                    s.replace_at(id, loc, Value(value)).unwrap().0
                }),
                "validated_replace_at" => sample(scale, &events, mode, round, true, |s, id, loc, value| {
                    s.validate_component_location(id, ComponentTypeId::of::<Value>(), loc).unwrap();
                    s.replace_at(id, loc, Value(value)).unwrap().0
                }),
                "upsert" => sample(scale, &events, mode, round, true, |s, id, _, value| {
                    s.upsert_component(id, Value(value)).unwrap().unwrap().0
                }),
                _ => sample(scale, &events, mode, round, true, |s, id, _, value| {
                    s.entity_mut(id).unwrap().set(Value(value)).unwrap().unwrap().0
                }),
            }
        }
    }
}
