//! Entity examples implemented on `sumeru::Store`.

use crate::common::*;

pub mod component_attributes;
pub mod entity_basics;
pub mod entity_hierarchy;
pub mod entity_hooks;
pub mod entity_iterate_components;
pub mod entity_prefab;

/// Hierarchy relation (`(ChildOf, Parent)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct ChildOf;

/// Inheritance relation (`(IsA, BasePrefab)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct IsA;

/// Attaches a `(ChildOf, parent)` relationship pair in `Store`.
pub fn set_child_of(store: &mut Store, child: EntityId, parent: EntityId) {
    store.set_pair(child, ChildOf, parent).expect("add ChildOf pair");
}

/// Checks whether `child` has `(ChildOf, parent)` in `Store`.
pub fn has_child_of(store: &Store, child: EntityId, parent: EntityId) -> bool {
    store.pair_target::<ChildOf>(child) == Some(parent)
}

/// Returns the parent of `entity` via `ChildOf` in `Store`.
pub fn parent_of_entity(store: &Store, entity: EntityId) -> Option<EntityId> {
    store.pair_target::<ChildOf>(entity)
}

/// Returns all direct children of `parent` in `Store`, sorted by entity slot order.
pub fn children_of_entity(store: &Store, parent: EntityId) -> Vec<EntityId> {
    let mut children = store.sources_with_pair::<ChildOf>(parent);
    children.sort_by_key(|e| EntityHandle::index(*e));
    children
}

/// Computes the Flecs-style `::Root::Child` path of `entity` using `ChildOf` in `Store`.
pub fn hierarchy_path_of(store: &Store, entity: EntityId) -> String {
    let mut chain = Vec::new();
    let mut cur = Some(entity);
    while let Some(e) = cur {
        chain.push(store.name_of(e));
        cur = parent_of_entity(store, e);
    }
    chain.reverse();
    format!("::{}", chain.join("::"))
}

/// Attaches an `(IsA, prefab)` relationship pair in `Store`.
pub fn set_is_a(store: &mut Store, entity: EntityId, prefab: EntityId) {
    store.set_pair(entity, IsA, prefab).expect("add IsA pair");
}

/// Resolves a component `T` on `entity` or recursively up its `IsA` prefab chain in `Store`.
pub fn get_inherited<T: Component>(store: &Store, entity: EntityId) -> Option<&T> {
    let mut cur = Some(entity);
    while let Some(e) = cur {
        if let Some(val) = store.get::<T>(e) {
            return Some(val);
        }
        cur = store.pair_target::<IsA>(e);
    }
    None
}

pub fn run_all() {
    entity_basics::main();
    entity_hierarchy::main();
    entity_hooks::main();
    entity_iterate_components::main();
    entity_prefab::main();
    component_attributes::main();
}
