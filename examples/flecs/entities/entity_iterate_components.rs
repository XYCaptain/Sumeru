//! Port of `examples/flecs/entities/entity_iterate_components.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;
use super::{ChildOf, set_child_of};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct Eats;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Human;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ComponentMeta;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolName(pub String);

enum ComponentEntry {
    Entity(String),
    Pair { rel: String, target: String },
}

impl ComponentEntry {
    fn short_label(&self) -> String {
        match self {
            ComponentEntry::Entity(name) => name.clone(),
            ComponentEntry::Pair { rel, target } => format!("({rel},{target})"),
        }
    }
}

fn collect_entries(store: &Store, entity: EntityId) -> Vec<ComponentEntry> {
    let mut out = Vec::new();
    let Some(eref) = store.entity(entity) else {
        return out;
    };
    let cids = eref.component_types();

    if cids.contains(&ComponentTypeId::of::<ComponentMeta>()) {
        out.push(ComponentEntry::Entity("Component".to_string()));
    }
    if cids.contains(&ComponentTypeId::of::<Position>()) {
        out.push(ComponentEntry::Entity("Position".to_string()));
    }
    if cids.contains(&ComponentTypeId::of::<Velocity>()) {
        out.push(ComponentEntry::Entity("Velocity".to_string()));
    }
    if cids.contains(&ComponentTypeId::of::<Human>()) {
        out.push(ComponentEntry::Entity("Human".to_string()));
    }
    if cids.contains(&ComponentTypeId::of::<EntityName>()) {
        out.push(ComponentEntry::Pair {
            rel: "Identifier".to_string(),
            target: "Name".to_string(),
        });
    }
    if cids.contains(&ComponentTypeId::of::<SymbolName>()) {
        out.push(ComponentEntry::Pair {
            rel: "Identifier".to_string(),
            target: "Symbol".to_string(),
        });
    }
    if let Some(target) = store.pair_target::<Eats>(entity) {
        out.push(ComponentEntry::Pair {
            rel: "Eats".to_string(),
            target: store.name_of(target),
        });
    }
    if let Some(parent) = store.pair_target::<ChildOf>(entity) {
        out.push(ComponentEntry::Pair {
            rel: "ChildOf".to_string(),
            target: store.name_of(parent),
        });
    }
    out
}

fn iterate_components(store: &Store, entity: EntityId) {
    let entries = collect_entries(store, entity);
    let summary: Vec<String> = entries.iter().map(ComponentEntry::short_label).collect();
    elog!("[{}]", summary.join(", "));
    elog!();

    for (i, entry) in entries.iter().enumerate() {
        elog!("{i}: {}", entry.short_label());
    }
    elog!();

    for (i, entry) in entries.iter().enumerate() {
        match entry {
            ComponentEntry::Entity(name) => elog!("{i}: entity: {name}"),
            ComponentEntry::Pair { rel, target } => elog!("{i}: rel: {rel}, target: {target}"),
        }
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let apples = store.entity_named("Apples");
    let bob = store.entity_named("Bob");
    store.set(bob, Position { x: 10.0, y: 20.0 }).unwrap();
    store.set(bob, Velocity { x: 1.0, y: 1.0 }).unwrap();
    store.set(bob, Human).unwrap();
    store.set_pair(bob, Eats, apples).unwrap();

    elog!("Bob's components:");
    iterate_components(&store, bob);
    elog!();

    let common_scope = store.entity_named("common");
    let pos_component = store.entity_named("Position");
    store.set(pos_component, ComponentMeta).unwrap();
    store.set(pos_component, SymbolName("Position".to_string())).unwrap();
    set_child_of(&mut store, pos_component, common_scope);

    elog!("Position's components:");
    iterate_components(&store, pos_component);
}

#[test]
fn test_entity_iterate_components() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "Bob's components:",
            "[Position, Velocity, Human, (Identifier,Name), (Eats,Apples)]",
            "",
            "0: Position",
            "1: Velocity",
            "2: Human",
            "3: (Identifier,Name)",
            "4: (Eats,Apples)",
            "",
            "0: entity: Position",
            "1: entity: Velocity",
            "2: entity: Human",
            "3: rel: Identifier, target: Name",
            "4: rel: Eats, target: Apples",
            "",
            "Position's components:",
            "[Component, (Identifier,Name), (Identifier,Symbol), (ChildOf,common)]",
            "",
            "0: Component",
            "1: (Identifier,Name)",
            "2: (Identifier,Symbol)",
            "3: (ChildOf,common)",
            "",
            "0: entity: Component",
            "1: rel: Identifier, target: Name",
            "2: rel: Identifier, target: Symbol",
            "3: rel: ChildOf, target: common",
        ]
    );
}
