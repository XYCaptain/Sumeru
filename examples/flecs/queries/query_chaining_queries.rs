use crate::common::*;
use crate::elog;

// Marks separated entities as blessed with mystical energy
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Enchanted;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Location {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Ability {
    pub power: f32,
}

// Holds magical level of artifacts
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ArtifactPower {
    pub _magic_level: f32,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Populate the store with creatures and artifacts
    for i in 0..10 {
        let creature = store.spawn_empty();
        store.set(
            creature,
            Location {
                x: i as f32,
                y: i as f32,
            },
        );
        store.set(
            creature,
            Ability {
                power: i as f32 * 1.5,
            },
        );
        if i % 2 == 0 {
            store.set(creature, Enchanted);
        }

        // Create artifacts, some of which are enchanted
        let artifact = store.spawn_empty();
        store.set(artifact, Location { x: -1.0, y: -1.0 });
        store.set(
            artifact,
            ArtifactPower {
                _magic_level: i as f32 * 2.5,
            },
        );
        if i % 2 != 0 {
            store.set(artifact, Enchanted);
        }
    }

    // Query for creatures based on Location and Ability components
    let query_creatures = QueryBuilder::<(&Location, &Ability)>::new()
        .build(&store)
        .unwrap();

    // Secondary query specifically looking for Enchanted entities
    let query_enchanted = QueryBuilder::<(&Location, &Ability)>::new()
        .with::<Enchanted>()
        .build(&store)
        .unwrap();

    // Chain: first match creatures, then filter to those matching query_enchanted
    let mut creature_ids = std::collections::HashSet::new();
    query_creatures
        .for_each(&store, |(entity, _)| {
            creature_ids.insert(entity);
        })
        .unwrap();

    query_enchanted
        .for_each(&store, |(entity, (pos, abil))| {
            if creature_ids.contains(&entity) {
                let abil_power = abil.power;
                elog!(
                    "Creature id: {} at location {},{} is enchanted with mystical energy, ability power: {} ",
                    EntityHandle::index(entity),
                    pos.x,
                    pos.y,
                    abil_power
                );
            }
        })
        .unwrap();
}

#[test]
fn test_query_chaining_queries() {
    main();
    let log = take_log();
    assert_eq!(log.len(), 5);
    assert!(log[0].ends_with("at location 0,0 is enchanted with mystical energy, ability power: 0 "));
    assert!(log[1].ends_with("at location 2,2 is enchanted with mystical energy, ability power: 3 "));
    assert!(log[2].ends_with("at location 4,4 is enchanted with mystical energy, ability power: 6 "));
    assert!(log[3].ends_with("at location 6,6 is enchanted with mystical energy, ability power: 9 "));
    assert!(log[4].ends_with("at location 8,8 is enchanted with mystical energy, ability power: 12 "));
}
