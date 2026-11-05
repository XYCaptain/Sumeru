use crate::common::*;
use super::{
    add_child_of, add_is_a, base_of_entity, find_direct_child, get_inherited,
    instantiate_prefab_hierarchy, parent_of_entity,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TirePressure {
    pub value: f32,
}

fn format_wheel_type(store: &Store, entity: EntityId) -> String {
    let mut parts = Vec::new();
    if store.has::<TirePressure>(entity) {
        parts.push("TirePressure".to_string());
    }
    if store.has::<EntityName>(entity) {
        parts.push("(Identifier,Name)".to_string());
    }
    if let Some(parent) = parent_of_entity(store, entity) {
        parts.push(format!("(ChildOf,{})", store.name_of(parent)));
    }
    if let Some(base) = base_of_entity(store, entity) {
        parts.push(format!("(IsA,{})", store.name_of(base)));
    }
    parts.join(", ")
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a Wheel prefab, make sure each instantiated wheel has a private
    // copy of the TirePressure component.
    let wheel = store.prefab_named("Wheel");
    store.set(wheel, TirePressure { value: 32.0 }).unwrap();

    // Create a Car prefab with four wheels.
    let car = store.prefab_named("Car");
    for wheel_name in ["FrontLeft", "FrontRight", "BackLeft", "BackRight"] {
        let w = store.prefab_named(wheel_name);
        add_child_of(&mut store, w, car);
        add_is_a(&mut store, w, wheel);
    }

    // Create a prefab instance.
    let inst = store.entity_named("my_car");
    instantiate_prefab_hierarchy(&mut store, inst, car);

    // Apply auto-override for TirePressure onto each instantiated wheel child
    for child in super::children_of_entity(&store, inst) {
        if let Some(tp) = get_inherited::<TirePressure>(&store, child).copied() {
            store.set(child, tp).unwrap();
        }
    }

    // Lookup the FrontLeft wheel
    if let Some(fl) = find_direct_child(&store, inst, "FrontLeft") {
        // The type shows that the child has a private copy of the TirePressure
        // component, and an IsA relationship to the Wheel prefab.
        elog!("{}", format_wheel_type(&store, fl));

        // Get the TirePressure component and log its value
        if let Some(p) = store.get::<TirePressure>(fl) {
            elog!("pressure: {}", p.value);
        }
    } else {
        elog!("entity lookup failed");
    }
}

#[test]
fn test_prefab_nested() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "TirePressure, (Identifier,Name), (ChildOf,my_car), (IsA,Wheel)\n",
            "pressure: 32"
        )
    );
}
