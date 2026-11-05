use crate::common::*;
use crate::elog;

// Tags for local/world position pairs: Pair<Position, Local> and Pair<Position, WorldX>
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Local;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WorldX;

fn propagate_recursive(store: &mut Store, entity: EntityId, parent_world: Option<Position>) {
    let local = store
        .get::<Pair<Position, Local>>(entity)
        .map(|p| p.relation)
        .unwrap_or_default();
    let world_pos = if let Some(pw) = parent_world {
        Position {
            x: local.x + pw.x,
            y: local.y + pw.y,
        }
    } else {
        local
    };
    store.set(entity, Pair::new(world_pos, WorldX));

    let children = store.children(entity);
    for child in children {
        propagate_recursive(store, child, Some(world_pos));
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a hierarchy. In the hierarchy we use (Position, Local) and
    // (Position, WorldX) pairs to store local and world positions.
    let sun = store.entity_named("Sun");
    store.set(sun, Pair::new(Position { x: 1.0, y: 1.0 }, WorldX));
    store.set(sun, Pair::new(Position { x: 1.0, y: 1.0 }, Local));

    let mercury = store.entity_named("Mercury");
    store.set_parent(mercury, sun);
    store.set(mercury, Pair::new(Position { x: 0.0, y: 0.0 }, WorldX));
    store.set(mercury, Pair::new(Position { x: 1.0, y: 1.0 }, Local));

    let venus = store.entity_named("Venus");
    store.set_parent(venus, sun);
    store.set(venus, Pair::new(Position { x: 0.0, y: 0.0 }, WorldX));
    store.set(venus, Pair::new(Position { x: 2.0, y: 2.0 }, Local));

    let earth = store.entity_named("Earth");
    store.set_parent(earth, sun);
    store.set(earth, Pair::new(Position { x: 0.0, y: 0.0 }, WorldX));
    store.set(earth, Pair::new(Position { x: 3.0, y: 3.0 }, Local));

    let moon = store.entity_named("Moon");
    store.set_parent(moon, earth);
    store.set(moon, Pair::new(Position { x: 0.0, y: 0.0 }, WorldX));
    store.set(moon, Pair::new(Position { x: 0.1, y: 0.1 }, Local));

    // Top-down breadth-first cascade from root entities
    let roots: Vec<EntityId> = store
        .query::<&Pair<Position, Local>>()
        .iter()
        .filter_map(|(e, _)| if store.parent(e).is_none() { Some(e) } else { None })
        .collect();

    for root in roots {
        propagate_recursive(&mut store, root, None);
    }

    // Print world positions for all entities that have (Position, WorldX)
    for (e, pair) in store.query::<&Pair<Position, WorldX>>().iter() {
        let p = &pair.relation;
        elog!("Entity {} is at ({}, {})", store.name_of(e), p.x, p.y);
    }
}

#[test]
fn test_query_hierarchy() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Entity Sun is at (1, 1)",
            "Entity Mercury is at (2, 2)",
            "Entity Venus is at (3, 3)",
            "Entity Earth is at (4, 4)",
            "Entity Moon is at (4.1, 4.1)",
        ]
    );
}
