use crate::common::*;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Platoon;

fn in_platoon(store: &Store, unit: EntityId, platoon: EntityId) -> bool {
    store.pair_target::<Platoon>(unit) == Some(platoon)
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create two platoons
    let platoon_1 = store.spawn_empty();
    let platoon_2 = store.spawn_empty();

    // Create a unit
    let unit = store.spawn_empty();

    // Add unit to platoon 1
    store.set_pair(unit, Platoon, platoon_1).unwrap();

    // Log platoon of unit
    elog!("Unit in platoon 1: {}", in_platoon(&store, unit, platoon_1));
    elog!("Unit in platoon 2: {}", in_platoon(&store, unit, platoon_2));
    elog!();

    // Add unit to platoon 2. Because Pair<Platoon, EntityId> is exclusive per relation type,
    // this both removes (Platoon, platoon_1) and adds (Platoon, platoon_2) in a
    // single operation.
    store.set_pair(unit, Platoon, platoon_2).unwrap();

    elog!("Unit in platoon 1: {}", in_platoon(&store, unit, platoon_1));
    elog!("Unit in platoon 2: {}", in_platoon(&store, unit, platoon_2));
}

#[test]
fn test_relationships_exclusive() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "Unit in platoon 1: true\n",
            "Unit in platoon 2: false\n",
            "\n",
            "Unit in platoon 1: false\n",
            "Unit in platoon 2: true"
        )
    );
}
