use crate::common::*;
use super::{
    add_child_of, add_is_a, full_path, instantiate_prefab_hierarchy, target_for_slot,
};

// Create types that mirror the prefab hierarchy.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Turret;
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Base;
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Head;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Railgun;
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Beam;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Register types for prefab hierarchy.
    let turret = store.prefab_named("Turret");
    store.set(turret, Turret).unwrap();

    let base = store.prefab_named("Base");
    store.set(base, Base).unwrap();
    add_child_of(&mut store, base, turret);
    store.set_pair(base, SlotOf, turret).unwrap();

    let head = store.prefab_named("Head");
    store.set(head, Head).unwrap();
    add_child_of(&mut store, head, turret);
    store.set_pair(head, SlotOf, turret).unwrap();

    let railgun = store.prefab_named("Railgun");
    store.set(railgun, Railgun).unwrap();
    add_is_a(&mut store, railgun, turret);

    let beam = store.prefab_named("Beam");
    store.set(beam, Beam).unwrap();
    add_child_of(&mut store, beam, railgun);
    store.set_pair(beam, SlotOf, railgun).unwrap();

    // Create a prefab instance.
    let inst = store.entity_named("my_railgun");
    instantiate_prefab_hierarchy(&mut store, inst, railgun);

    // Get entities for slots by their prefab entity handles
    let inst_base = target_for_slot(&store, inst, base).unwrap();
    let inst_head = target_for_slot(&store, inst, head).unwrap();
    let inst_beam = target_for_slot(&store, inst, beam).unwrap();

    elog!("instance base: {}", full_path(&store, inst_base));
    elog!("instance head: {}", full_path(&store, inst_head));
    elog!("instance beam: {}", full_path(&store, inst_beam));
}

#[test]
fn test_prefab_typed() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "instance base: ::my_railgun::Base\n",
            "instance head: ::my_railgun::Head\n",
            "instance beam: ::my_railgun::Beam"
        )
    );
}
