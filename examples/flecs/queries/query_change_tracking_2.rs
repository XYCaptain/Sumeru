use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dirty {
    pub value: bool,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create two prefabs with a Dirty component.
    let p1 = store.prefab_named("p1");
    store.set(p1, Dirty { value: false });

    let p2 = store.prefab_named("p2");
    store.set(p2, Dirty { value: true });

    // Create instances of p1 and p2. Because the entities have different
    // prefabs, they end up in different tables.
    let e1 = store.entity_named("e1");
    store.add_is_a(e1, p1);
    store.set(e1, Position { x: 10.0, y: 20.0 });

    let e2 = store.entity_named("e2");
    store.add_is_a(e2, p1);
    store.set(e2, Position { x: 30.0, y: 40.0 });

    let e3 = store.entity_named("e3");
    store.add_is_a(e3, p2);
    store.set(e3, Position { x: 40.0, y: 50.0 });

    let e4 = store.entity_named("e4");
    store.add_is_a(e4, p2);
    store.set(e4, Position { x: 50.0, y: 60.0 });

    let tables = [(p1, [e1, e2]), (p2, [e3, e4])];

    // Step 1: Write query that checks inherited Dirty flag and only mutates
    // Position when Dirty.value == true, capturing the native BatchReceipt.
    let mut cmd = store.commands();
    for (prefab, entities) in &tables {
        let dirty = store.get::<Dirty>(*prefab).copied().unwrap_or_default();
        if !dirty.value {
            // Skip mutating this table's Position column
            continue;
        }
        for &e in entities {
            if let Some(pos) = store.get::<Position>(e).copied() {
                cmd.entity(e).replace(Position {
                    x: pos.x + 1.0,
                    y: pos.y + 1.0,
                });
            }
        }
    }
    let receipt = cmd.apply(&mut store).unwrap();

    // Step 2: Read query that checks if a table's Position column has changed
    // in the committed batch receipt.
    for (prefab, entities) in &tables {
        let table_changed = entities
            .iter()
            .any(|e| receipt.updated_entities.contains(e));
        let table_desc = format!(
            "Position, (Identifier,Name), (IsA,{})",
            store.name_of(*prefab)
        );
        if !table_changed {
            elog!("it.changed() is false for [{}]", table_desc);
            continue;
        }
        elog!("it.changed() is true for [{}]", table_desc);
        for &e in entities {
            let pos = store.get::<Position>(e).unwrap();
            elog!("{}: {:?}", store.name_of(e), pos);
        }
    }
}

#[test]
fn test_query_change_tracking_2() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "it.changed() is false for [Position, (Identifier,Name), (IsA,p1)]",
            "it.changed() is true for [Position, (Identifier,Name), (IsA,p2)]",
            "e3: Position { x: 41.0, y: 51.0 }",
            "e4: Position { x: 51.0, y: 61.0 }",
        ]
    );
}
