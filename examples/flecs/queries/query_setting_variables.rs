use crate::common::*;
use crate::elog;

// Extending the component inheritance example, this example shows how we can
// use a single query to match units from a player and a platoon by setting
// query variables before iterating.
//
// Query:
//   (UnitClass, $class) && is_a($class, RangedUnit) && (Platoon, $platoon) && (Player, $player)[src=$platoon]

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Player;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Platoon;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct UnitClass;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MyPlayer;

const PLAYER_COUNT: usize = 100;
const PLATOONS_PER_PLAYER: usize = 3;

fn inherits_from(store: &Store, class: EntityId, ancestor: EntityId) -> bool {
    if class == ancestor {
        return true;
    }
    let mut current = store.pair_target::<IsA>(class);
    while let Some(parent) = current {
        if parent == ancestor {
            return true;
        }
        current = store.pair_target::<IsA>(parent);
    }
    false
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Class inheritance hierarchy
    let unit = store.entity_named("Unit");
    let combat_unit = store.entity_named("CombatUnit");
    store.add_is_a(combat_unit, unit);

    let melee_unit = store.entity_named("MeleeUnit");
    store.add_is_a(melee_unit, combat_unit);

    let ranged_unit = store.entity_named("RangedUnit");
    store.add_is_a(ranged_unit, combat_unit);

    let warrior = store.entity_named("Warrior");
    store.add_is_a(warrior, melee_unit);

    let wizard = store.entity_named("Wizard");
    store.add_is_a(wizard, ranged_unit);

    let marksman = store.entity_named("Marksman");
    store.add_is_a(marksman, ranged_unit);

    // Populate store with players, platoons and units
    for p in 0..PLAYER_COUNT {
        let player = if p == 0 {
            let e = store.entity_named("MyPlayer");
            store.set(e, MyPlayer);
            e
        } else {
            store.spawn_empty()
        };

        for _pl in 0..PLATOONS_PER_PLAYER {
            let platoon = store.spawn_empty();
            store.set_pair(platoon, Player, player).unwrap();

            let u_wizard = store.spawn_empty();
            store.set_pair(u_wizard, UnitClass, wizard).unwrap();
            store.set_pair(u_wizard, Platoon, platoon).unwrap();

            let u_marksman = store.spawn_empty();
            store.set_pair(u_marksman, UnitClass, marksman).unwrap();
            store.set_pair(u_marksman, Platoon, platoon).unwrap();

            let u_warrior = store.spawn_empty();
            store.set_pair(u_warrior, UnitClass, warrior).unwrap();
            store.set_pair(u_warrior, Platoon, platoon).unwrap();
        }
    }

    // Find MyPlayer
    let my_player = store
        .query::<&MyPlayer>()
        .iter()
        .next()
        .map(|(e, _)| e)
        .unwrap();

    // Set variable $player = my_player and iterate matching ranged units
    for target_class in [wizard, marksman] {
        for (unit_entity, (class_pair, platoon_pair)) in store
            .query::<(&Pair<UnitClass, EntityId>, &Pair<Platoon, EntityId>)>()
            .iter()
        {
            if class_pair.target != target_class
                || !inherits_from(&store, class_pair.target, ranged_unit)
            {
                continue;
            }
            let platoon = platoon_pair.target;
            if store.pair_target::<Player>(platoon) == Some(my_player) {
                elog!(
                    "Unit {} of class {} in platoon {} for player {}",
                    EntityHandle::index(unit_entity),
                    store.name_of(class_pair.target),
                    EntityHandle::index(platoon),
                    store.name_of(my_player)
                );
            }
        }
    }
}

#[test]
fn test_query_setting_variables() {
    main();
    let log = take_log();
    assert_eq!(log.len(), 6);
    assert!(log[0].contains("of class Wizard in platoon") && log[0].ends_with("for player MyPlayer"));
    assert!(log[1].contains("of class Wizard in platoon") && log[1].ends_with("for player MyPlayer"));
    assert!(log[2].contains("of class Wizard in platoon") && log[2].ends_with("for player MyPlayer"));
    assert!(
        log[3].contains("of class Marksman in platoon") && log[3].ends_with("for player MyPlayer")
    );
    assert!(
        log[4].contains("of class Marksman in platoon") && log[4].ends_with("for player MyPlayer")
    );
    assert!(
        log[5].contains("of class Marksman in platoon") && log[5].ends_with("for player MyPlayer")
    );
}
