use crate::common::*;
use super::{add_is_a, base_of_entity, get_inherited};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Attack {
    pub value: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Defence {
    pub value: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Damage {
    pub value: f32,
}

fn format_override_type(store: &Store, entity: EntityId) -> String {
    let mut parts = Vec::new();
    if store.has::<Attack>(entity) {
        parts.push("Attack".to_string());
    }
    if store.has::<Defence>(entity) {
        parts.push("Defence".to_string());
    }
    if store.has::<Damage>(entity) {
        parts.push("Damage".to_string());
    }
    if store.has::<EntityName>(entity) {
        parts.push("(Identifier,Name)".to_string());
    }
    if let Some(base) = base_of_entity(store, entity) {
        parts.push(format!("(IsA,{})", store.name_of(base)));
    }
    parts.join(", ")
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a prefab with Attack, Defence, and auto-overridden Damage components
    let spaceship = store.prefab_named("SpaceShip");
    store.set(spaceship, Attack { value: 75.0 }).unwrap();
    store.set(spaceship, Defence { value: 100.0 }).unwrap();
    store.set(spaceship, Damage { value: 50.0 }).unwrap();

    // Create a prefab instance. The Damage component is overridden automatically.
    let inst = store.entity_named("my_spaceship");
    add_is_a(&mut store, inst, spaceship);
    if let Some(dmg) = get_inherited::<Damage>(&store, inst).copied() {
        store.set(inst, dmg).unwrap();
    }

    // The entity will now have a private copy of the Damage component,
    // because of the is_a relationship, will share the Attack and Defense components.
    elog!("{}", format_override_type(&store, inst));

    // We can also manually override a component for an instance with set_auto_override.
    // When a component is overridden, the value is initialized from the prefab:
    if let Some(atk) = get_inherited::<Attack>(&store, inst).copied() {
        store.set(inst, atk).unwrap();
    }
    elog!("{}", format_override_type(&store, inst));

    let attack = get_inherited::<Attack>(&store, inst).unwrap();
    let defence = get_inherited::<Defence>(&store, inst).unwrap();
    let damage = get_inherited::<Damage>(&store, inst).unwrap();

    elog!("attack: {}", attack.value);
    elog!("defence: {}", defence.value);
    elog!("damage: {}", damage.value);
}

#[test]
fn test_prefab_override() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "Damage, (Identifier,Name), (IsA,SpaceShip)\n",
            "Attack, Damage, (Identifier,Name), (IsA,SpaceShip)\n",
            "attack: 75\n",
            "defence: 100\n",
            "damage: 50"
        )
    );
}
