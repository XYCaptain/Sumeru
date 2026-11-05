//! Port of `examples/flecs/entities/entity_prefab.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;
use super::{IsA, get_inherited, set_is_a};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Attack {
    pub value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Defence {
    pub value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FreightCapacity {
    pub value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImpulseSpeed {
    pub value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct HasFlt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AutoOverridePosition;

fn instantiate_prefab(store: &mut Store, inst: EntityId, prefab: EntityId) {
    set_is_a(store, inst, prefab);
    if get_inherited::<AutoOverridePosition>(store, prefab).is_some() {
        if let Some(&pos) = get_inherited::<Position>(store, prefab) {
            store.set(inst, pos).unwrap();
        }
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a prefab hierarchy in Store
    let spaceship = store.prefab_named("Spaceship");
    store.set(spaceship, ImpulseSpeed { value: 50.0 }).unwrap();
    store.set(spaceship, Defence { value: 50.0 }).unwrap();
    store.set(spaceship, Position { x: 0.0, y: 0.0 }).unwrap();
    store.set(spaceship, AutoOverridePosition).unwrap();

    let freighter = store.prefab_named("Freighter");
    set_is_a(&mut store, freighter, spaceship);
    store.set(freighter, FreightCapacity { value: 100.0 }).unwrap();
    store.set(freighter, Defence { value: 100.0 }).unwrap();
    store.set(freighter, HasFlt).unwrap();

    let mammoth_freighter = store.prefab_named("MammothFreighter");
    set_is_a(&mut store, mammoth_freighter, freighter);
    store.set(mammoth_freighter, FreightCapacity { value: 500.0 }).unwrap();
    store.set(mammoth_freighter, Defence { value: 300.0 }).unwrap();

    let frigate = store.prefab_named("Frigate");
    set_is_a(&mut store, frigate, spaceship);
    store.set(frigate, HasFlt).unwrap();
    store.set(frigate, Attack { value: 100.0 }).unwrap();
    store.set(frigate, Defence { value: 75.0 }).unwrap();
    store.set(frigate, ImpulseSpeed { value: 125.0 }).unwrap();

    // Create a regular entity from a prefab
    let inst = store.entity_named("my_mammoth_freighter");
    instantiate_prefab(&mut store, inst, mammoth_freighter);

    let isa_target = store.pair_target::<IsA>(inst)
        .map(|t| store.name_of(t))
        .unwrap_or_default();
    elog!(
        "Instance type: [Position, (Identifier,Name), (IsA,{isa_target})]"
    );

    if let Some(impulse_speed) = get_inherited::<ImpulseSpeed>(&store, inst) {
        elog!("ImpulseSpeed: {}", impulse_speed.value);
    }

    // Iterate non-prefab entities with Position and inherited ImpulseSpeed
    let mut non_prefab_entities = Vec::new();
    for (entity, _) in store.query::<&Position>().without::<PrefabTag>().iter() {
        if let Some(speed) = get_inherited::<ImpulseSpeed>(&store, entity) {
            non_prefab_entities.push((entity, speed.value));
        }
    }
    for (entity, speed_val) in non_prefab_entities {
        if let Some(pos) = store.get_mut::<Position>(entity) {
            pos.x += speed_val;
        }
        let pos = store.get::<Position>(entity).unwrap();
        elog!("Entity {}: {:?}", store.name_of(entity), pos);
    }
}

#[test]
fn test_entity_prefab() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "Instance type: [Position, (Identifier,Name), (IsA,MammothFreighter)]",
            "ImpulseSpeed: 50",
            "Entity my_mammoth_freighter: Position { x: 50.0, y: 0.0 }",
        ]
    );
}
