//! Port of `examples/flecs/entities/entity_hierarchy.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;
use super::{children_of_entity, has_child_of, hierarchy_path_of, parent_of_entity, set_child_of};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Star;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Planet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Moon;

fn format_hierarchy_archetype(store: &Store, entity: EntityId) -> String {
    let mut parts = Vec::new();
    if store.has::<Position>(entity) {
        parts.push("Position".to_string());
    }
    if store.has::<Planet>(entity) {
        parts.push("Planet".to_string());
    }
    if store.has::<Moon>(entity) {
        parts.push("Moon".to_string());
    }
    if store.has::<EntityName>(entity) {
        parts.push("(Identifier,Name)".to_string());
    }
    if let Some(parent) = parent_of_entity(store, entity) {
        let parent_label = if let Some(grand) = parent_of_entity(store, parent) {
            format!("{}.{}", store.name_of(grand), store.name_of(parent))
        } else {
            store.name_of(parent)
        };
        parts.push(format!("(ChildOf,{parent_label})"));
    }
    parts.join(", ")
}

fn iterate_tree(store: &Store, entity: EntityId, position_parent: &Position) {
    elog!(
        "{} [{}]",
        hierarchy_path_of(store, entity),
        format_hierarchy_archetype(store, entity)
    );

    let pos = store.get::<Position>(entity).expect("entity has Position");
    let pos_actual = Position {
        x: pos.x + position_parent.x,
        y: pos.y + position_parent.y,
    };

    elog!("{pos_actual:?}");

    for child in children_of_entity(store, entity) {
        iterate_tree(store, child, &pos_actual);
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let sun = store.entity_named("Sun");
    store.set(sun, Position { x: 1.0, y: 1.0 }).unwrap();

    let mercury = store.entity_named("Mercury");
    store.set(mercury, Position { x: 1.0, y: 1.0 }).unwrap();
    store.set(mercury, Planet).unwrap();
    set_child_of(&mut store, mercury, sun);

    let venus = store.entity_named("Venus");
    store.set(venus, Position { x: 2.0, y: 2.0 }).unwrap();
    store.set(venus, Planet).unwrap();
    set_child_of(&mut store, venus, sun);

    let earth = store.entity_named("Earth");
    store.set(earth, Position { x: 3.0, y: 3.0 }).unwrap();
    store.set(earth, Planet).unwrap();
    set_child_of(&mut store, earth, sun);

    let moon = store.entity_named("Moon");
    store.set(moon, Position { x: 0.1, y: 0.1 }).unwrap();
    store.set(moon, Moon).unwrap();
    set_child_of(&mut store, moon, earth);

    let is_child = has_child_of(&store, moon, earth);
    elog!("Is the Moon a child of the Earth? {is_child} / {is_child}");
    elog!();

    iterate_tree(&store, sun, &Position { x: 0.0, y: 0.0 });
}

#[test]
fn test_entity_hierarchy() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "Is the Moon a child of the Earth? true / true",
            "",
            "::Sun [Position, (Identifier,Name)]",
            "Position { x: 1.0, y: 1.0 }",
            "::Sun::Mercury [Position, Planet, (Identifier,Name), (ChildOf,Sun)]",
            "Position { x: 2.0, y: 2.0 }",
            "::Sun::Venus [Position, Planet, (Identifier,Name), (ChildOf,Sun)]",
            "Position { x: 3.0, y: 3.0 }",
            "::Sun::Earth [Position, Planet, (Identifier,Name), (ChildOf,Sun)]",
            "Position { x: 4.0, y: 4.0 }",
            "::Sun::Earth::Moon [Position, Moon, (Identifier,Name), (ChildOf,Sun.Earth)]",
            "Position { x: 4.1, y: 4.1 }",
        ]
    );
}
