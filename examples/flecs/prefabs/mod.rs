use crate::common::*;

pub mod prefab_basics;
pub mod prefab_hierarchy;
pub mod prefab_nested;
pub mod prefab_override;
pub mod prefab_slots;
pub mod prefab_typed;
pub mod prefab_variant;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SlotBindings(pub Vec<(EntityId, EntityId)>);

pub fn add_child_of(store: &mut Store, child: EntityId, parent: EntityId) {
    store.set_pair(child, ChildOf, parent).unwrap();
}

pub fn parent_of_entity(store: &Store, entity: EntityId) -> Option<EntityId> {
    store.pair_target::<ChildOf>(entity)
}

pub fn children_of_entity(store: &Store, parent: EntityId) -> Vec<EntityId> {
    let mut children = store.sources_with_pair::<ChildOf>(parent);
    children.sort();
    children
}

pub fn add_is_a(store: &mut Store, entity: EntityId, prefab: EntityId) {
    store.set_pair(entity, IsA, prefab).unwrap();
}

pub fn base_of_entity(store: &Store, entity: EntityId) -> Option<EntityId> {
    store.pair_target::<IsA>(entity)
}

pub fn get_inherited<'a, T: Component>(store: &'a Store, mut entity: EntityId) -> Option<&'a T> {
    loop {
        if let Some(val) = store.get::<T>(entity) {
            return Some(val);
        }
        entity = base_of_entity(store, entity)?;
    }
}

pub fn full_path(store: &Store, entity: EntityId) -> String {
    let mut parts = vec![store.name_of(entity)];
    let mut cur = entity;
    while let Some(parent) = parent_of_entity(store, cur) {
        parts.push(store.name_of(parent));
        cur = parent;
    }
    parts.reverse();
    format!("::{}", parts.join("::"))
}

pub fn find_direct_child(store: &Store, parent: EntityId, name: &str) -> Option<EntityId> {
    for child in children_of_entity(store, parent) {
        if store.name_of(child) == name {
            return Some(child);
        }
    }
    None
}

pub fn bind_slot(store: &mut Store, instance: EntityId, slot: EntityId, target: EntityId) {
    let mut bindings = store
        .get::<SlotBindings>(instance)
        .cloned()
        .unwrap_or_default();
    if let Some(entry) = bindings.0.iter_mut().find(|(s, _)| *s == slot) {
        entry.1 = target;
    } else {
        bindings.0.push((slot, target));
    }
    store.set(instance, bindings).unwrap();
}

pub fn target_for_slot(store: &Store, instance: EntityId, slot: EntityId) -> Option<EntityId> {
    store
        .get::<SlotBindings>(instance)
        .and_then(|b| b.0.iter().find(|(s, _)| *s == slot).map(|(_, t)| *t))
}

pub fn instantiate_prefab_hierarchy(store: &mut Store, instance: EntityId, prefab: EntityId) {
    add_is_a(store, instance, prefab);
    if let Some(base) = base_of_entity(store, prefab) {
        clone_children_recursive(store, instance, base, instance);
    }
    clone_children_recursive(store, instance, prefab, instance);
}

fn clone_children_recursive(
    store: &mut Store,
    root_instance: EntityId,
    prefab_parent: EntityId,
    instance_parent: EntityId,
) {
    let children = children_of_entity(store, prefab_parent);
    for prefab_child in children {
        let name = store.name_of(prefab_child);
        let inst_child = store.entity_named(&name);
        add_child_of(store, inst_child, instance_parent);
        if let Some(child_base) = base_of_entity(store, prefab_child) {
            add_is_a(store, inst_child, child_base);
        }
        if store.pair_target::<SlotOf>(prefab_child).is_some() {
            bind_slot(store, root_instance, prefab_child, inst_child);
        }
        clone_children_recursive(store, root_instance, prefab_child, inst_child);
    }
}

pub fn run_all() {
    prefab_basics::main();
    prefab_hierarchy::main();
    prefab_nested::main();
    prefab_override::main();
    prefab_slots::main();
    prefab_typed::main();
    prefab_variant::main();
}
