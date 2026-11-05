use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct UnitClass;

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

    // Make the ECS aware of the inheritance relationships between component/class entities.
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

    let builder_x = store.entity_named("BuilderX");
    store.add_is_a(builder_x, unit);

    // Create a few units
    let warrior_1 = store.entity_named("warrior_1");
    store.set_pair(warrior_1, UnitClass, warrior).unwrap();
    let warrior_2 = store.entity_named("warrior_2");
    store.set_pair(warrior_2, UnitClass, warrior).unwrap();

    let marksman_1 = store.entity_named("marksman_1");
    store.set_pair(marksman_1, UnitClass, marksman).unwrap();
    let marksman_2 = store.entity_named("marksman_2");
    store.set_pair(marksman_2, UnitClass, marksman).unwrap();

    let wizard_1 = store.entity_named("wizard_1");
    store.set_pair(wizard_1, UnitClass, wizard).unwrap();
    let wizard_2 = store.entity_named("wizard_2");
    store.set_pair(wizard_2, UnitClass, wizard).unwrap();

    let builder_1 = store.entity_named("builder_1");
    store.set_pair(builder_1, UnitClass, builder_x).unwrap();
    let builder_2 = store.entity_named("builder_2");
    store.set_pair(builder_2, UnitClass, builder_x).unwrap();

    // Query all entities whose UnitClass inherits from RangedUnit.
    // Follow the IsA hierarchy order (Wizard, then Marksman) matching Flecs component inheritance traversal.
    for (e, pair) in store.query::<&Pair<UnitClass, EntityId>>().iter() {
        if inherits_from(&store, pair.target, ranged_unit) && pair.target == wizard {
            elog!("Unit {} found", store.name_of(e));
        }
    }
    for (e, pair) in store.query::<&Pair<UnitClass, EntityId>>().iter() {
        if inherits_from(&store, pair.target, ranged_unit) && pair.target == marksman {
            elog!("Unit {} found", store.name_of(e));
        }
    }
}

#[test]
fn test_query_component_inheritance() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Unit wizard_1 found",
            "Unit wizard_2 found",
            "Unit marksman_1 found",
            "Unit marksman_2 found",
        ]
    );
}
