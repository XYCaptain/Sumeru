use crate::common::*;
use super::{add_is_a, full_path, get_inherited};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Defence {
    pub value: f32,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a SpaceShip prefab with a Defense component.
    let spaceship = store.prefab_named("SpaceShip");
    store.set(spaceship, Defence { value: 50.0 }).unwrap();

    // Create a prefab instance
    let inst = store.entity_named("my_spaceship");
    add_is_a(&mut store, inst, spaceship);

    // Because of the IsA relationship, the instance now shares the Defense
    // component with the prefab, and can be retrieved as a regular component:
    if let Some(d_inst) = get_inherited::<Defence>(&store, inst) {
        elog!("{d_inst:?}");
        // Because the component is shared, changing the value on the prefab will
        // also change the value for the instance:
        store.set(spaceship, Defence { value: 100.0 }).unwrap();
    }

    if let Some(d_inst) = get_inherited::<Defence>(&store, inst) {
        elog!("after set: {d_inst:?}");
    }

    // Prefab components can also be iterated for non-prefab instances:
    for e in store.entities() {
        if !store.has::<PrefabTag>(e) {
            if let Some(d) = get_inherited::<Defence>(&store, e) {
                elog!("{}: defence: {}", full_path(&store, e), d.value);
            }
        }
    }
}

#[test]
fn test_prefab_basics() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "Defence { value: 50.0 }\n",
            "after set: Defence { value: 100.0 }\n",
            "::my_spaceship: defence: 100"
        )
    );
}
