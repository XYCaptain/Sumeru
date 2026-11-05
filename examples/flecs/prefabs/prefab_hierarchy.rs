use crate::common::*;
use super::{add_child_of, find_direct_child, full_path, instantiate_prefab_hierarchy};

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a prefab hierarchy.
    let spaceship = store.prefab_named("SpaceShip");
    let engine = store.prefab_named("Engine");
    add_child_of(&mut store, engine, spaceship);
    let cockpit = store.prefab_named("Cockpit");
    add_child_of(&mut store, cockpit, spaceship);

    // Instantiate the prefab. This also creates instances for the child
    // prefabs.
    let inst = store.entity_named("my_spaceship");
    instantiate_prefab_hierarchy(&mut store, inst, spaceship);
    let inst_engine = find_direct_child(&store, inst, "Engine").unwrap();
    let inst_cockpit = find_direct_child(&store, inst, "Cockpit").unwrap();

    elog!("instance engine:  {:?}", full_path(&store, inst_engine));
    elog!("instance cockpit: {:?}", full_path(&store, inst_cockpit));
}

#[test]
fn test_prefab_hierarchy() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "instance engine:  \"::my_spaceship::Engine\"\n",
            "instance cockpit: \"::my_spaceship::Cockpit\""
        )
    );
}
