use crate::common::*;
use super::{add_child_of, full_path, instantiate_prefab_hierarchy, target_for_slot};

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create the same prefab hierarchy as from the hierarchy example, Issue #292, now
    // with the SlotOf relationship.
    let spaceship = store.prefab_named("SpaceShip");
    let engine = store.prefab_named("Engine");
    add_child_of(&mut store, engine, spaceship);
    store.set_pair(engine, SlotOf, spaceship).unwrap();

    let cockpit = store.prefab_named("Cockpit");
    add_child_of(&mut store, cockpit, spaceship);
    store.set_pair(cockpit, SlotOf, spaceship).unwrap();

    // Add an additional child to the Cockpit prefab to demonstrate how
    // slots can be different from the parent. This slot could have been
    // added to the Cockpit prefab, Issue #292, instead we register it on the top
    // level SpaceShip prefab.
    let pilot_seat = store.prefab_named("PilotSeat");
    add_child_of(&mut store, pilot_seat, cockpit);
    store.set_pair(pilot_seat, SlotOf, spaceship).unwrap();

    // Create a prefab instance.
    let inst = store.entity_named("my_spaceship");
    instantiate_prefab_hierarchy(&mut store, inst, spaceship);

    // Get the instantiated entities for the prefab slots
    let inst_engine = target_for_slot(&store, inst, engine).unwrap();
    let inst_cockpit = target_for_slot(&store, inst, cockpit).unwrap();
    let inst_seat = target_for_slot(&store, inst, pilot_seat).unwrap();

    elog!("instance engine: {}", full_path(&store, inst_engine));
    elog!("instance cockpit: {}", full_path(&store, inst_cockpit));
    elog!("instance seat: {}", full_path(&store, inst_seat));
}

#[test]
fn test_prefab_slots() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "instance engine: ::my_spaceship::Engine\n",
            "instance cockpit: ::my_spaceship::Cockpit\n",
            "instance seat: ::my_spaceship::Cockpit::PilotSeat"
        )
    );
}
