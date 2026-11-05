//! Port of `examples/flecs/entities/entity_basics.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Walking;

fn format_archetype(store: &Store, entity: EntityId) -> String {
    let mut parts = Vec::new();
    if store.has::<Position>(entity) {
        parts.push("Position");
    }
    if store.has::<Walking>(entity) {
        parts.push("Walking");
    }
    if store.has::<EntityName>(entity) {
        parts.push("(Identifier,Name)");
    }
    parts.join(", ")
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create an entity with name Bob
    let bob = store.entity_named("Bob");
    store.set(bob, Position { x: 10.0, y: 20.0 }).unwrap();
    store.set(bob, Walking).unwrap();

    // Get the value for the Position component
    if let Some(pos) = store.get::<Position>(bob) {
        elog!("Bob's position: {pos:?}");
    }

    // Overwrite the value of the Position component
    store.set(bob, Position { x: 20.0, y: 30.0 }).unwrap();

    // Create another named entity
    let alice = store.entity_named("Alice");
    store.set(alice, Position { x: 10.0, y: 20.0 }).unwrap();

    // Add a tag after entity is created
    store.set(alice, Walking).unwrap();

    // Print all of the components the entity has
    elog!("[{}]", format_archetype(&store, alice));

    // Remove tag
    store.remove::<Walking>(alice).unwrap();

    // Iterate all entities with Position
    let mut rows: Vec<(String, Position)> = store
        .query::<&Position>()
        .iter()
        .map(|(entity, &pos)| (store.name_of(entity), pos))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, pos) in rows {
        elog!("{name} has {pos:?}");
    }
}

#[test]
fn test_entity_basics() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "Bob's position: Position { x: 10.0, y: 20.0 }",
            "[Position, Walking, (Identifier,Name)]",
            "Alice has Position { x: 10.0, y: 20.0 }",
            "Bob has Position { x: 20.0, y: 30.0 }",
        ]
    );
}
