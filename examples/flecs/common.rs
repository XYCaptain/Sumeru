//! Shared helpers for the 85 Flecs-canonical examples implemented on `sumeru::Store`.
//!
//! All state storage, entity lifecycle, columnar queries, command buffers, relationships,
//! change tracking, and snapshots execute directly on `sumeru::Store`.

use std::cell::RefCell;

pub use sumeru::al::{
    All, ColumnChunkProbe, CommandBufferAccess, Component, EntityCommandsAccess, EntityHandle,
    EntityMutAccess, EntityRefAccess, Pair, QueryFilter, StateSpace, Tag, With, Without,
};
pub use sumeru::{
    ArchetypeId, BatchReceipt, ColumnarStoreSnapshot, CommandBuffer, ComponentField,
    ComponentRegistration, ComponentSchema, ComponentTypeId, DomainTypeDefinition, DomainTypeId,
    DomainTypeKey, EntityCommands, EntityId, EntityMut, EntityRef, IntoEntity, ModelComponent,
    ModelSchema, Query, QueryBuilder, StateError, Store,
};
pub use serde::{Deserialize, Serialize};

use std::sync::atomic::{AtomicBool, Ordering};

static BENCH_QUIET: AtomicBool = AtomicBool::new(false);

thread_local! {
    static EXAMPLE_LOG: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Enables or disables quiet benchmark mode (suppresses stdout and log vector allocation).
pub fn set_bench_quiet(quiet: bool) {
    BENCH_QUIET.store(quiet, Ordering::Relaxed);
}

/// Clears the thread-local example output buffer before running an example.
pub fn clear_log() {
    if !BENCH_QUIET.load(Ordering::Relaxed) {
        EXAMPLE_LOG.with(|log| log.borrow_mut().clear());
    }
}

/// Appends a formatted line to both stdout and the thread-local test log.
pub fn log_line(line: impl Into<String>) {
    if BENCH_QUIET.load(Ordering::Relaxed) {
        std::hint::black_box(line.into());
        return;
    }
    let s = line.into();
    println!("{s}");
    EXAMPLE_LOG.with(|log| log.borrow_mut().push(s));
}

/// Returns all lines logged during the current example execution.
pub fn take_log() -> Vec<String> {
    EXAMPLE_LOG.with(|log| log.borrow().clone())
}

#[macro_export]
macro_rules! elog {
    () => {
        $crate::common::log_line("")
    };
    ($($arg:tt)*) => {
        $crate::common::log_line(format!($($arg)*))
    };
}

/// Standard entity name component stored directly inside `sumeru::Store`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityName(pub String);

/// Marker tag component for prefab entities in `Store`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PrefabTag;

/// Builtin hierarchical parent relation (`Pair<ChildOf, EntityId>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct ChildOf;

/// Builtin prefab inheritance relation (`Pair<IsA, EntityId>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct IsA;

/// Marker relation for prefab slots (`Pair<SlotOf, PrefabEntity>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct SlotOf;

/// Common 2D position component used across Flecs examples.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

/// Common 2D velocity component used across Flecs examples.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Velocity {
    pub x: f32,
    pub y: f32,
}

/// Ergonomic extension trait over `sumeru::Store` for named entities and hierarchical paths.
pub trait StoreExampleExt {
    /// Spawns a new empty entity in `Store`.
    fn spawn_empty(&mut self) -> EntityId;

    /// Spawns a new entity with `EntityName(name)` in `Store`.
    fn entity_named(&mut self, name: &str) -> EntityId;

    /// Finds an existing entity with `EntityName(name)` or spawns a new one.
    fn ensure_entity_named(&mut self, name: &str) -> EntityId;

    /// Spawns a prefab entity with `EntityName(name)` and `PrefabTag`.
    fn prefab_named(&mut self, name: &str) -> EntityId;

    /// Returns the `EntityName` string of `entity`, or `"#<index>"` if unnamed.
    fn name_of(&self, entity: EntityId) -> String;

    /// Returns the hierarchical path of `entity` (e.g. `"::Parent::Child"` or `"::Bob"`).
    fn path_of(&self, entity: EntityId) -> String;

    /// Sets `(ChildOf, parent)` on `child`.
    fn set_child_of(&mut self, child: EntityId, parent: EntityId);

    /// Alias for `set_child_of(child, parent)`.
    fn set_parent(&mut self, child: EntityId, parent: EntityId);

    /// Sets `(IsA, prefab)` on `instance`.
    fn set_is_a(&mut self, instance: EntityId, prefab: EntityId);

    /// Alias for `set_is_a(instance, prefab)`.
    fn add_is_a(&mut self, instance: EntityId, prefab: EntityId);

    /// Returns the `ChildOf` parent of `entity`, if any.
    fn parent_of(&self, entity: EntityId) -> Option<EntityId>;

    /// Alias for `parent_of(entity)`.
    fn parent(&self, entity: EntityId) -> Option<EntityId>;

    /// Returns all direct children with `(ChildOf, parent)`.
    fn children_of(&self, parent: EntityId) -> Vec<EntityId>;

    /// Alias for `children_of(parent)`.
    fn children(&self, parent: EntityId) -> Vec<EntityId>;

    /// Returns all `IsA` prefab targets of `entity`.
    fn prefabs_of(&self, entity: EntityId) -> Vec<EntityId>;

    /// Looks up a direct child of `parent` by `EntityName`.
    fn lookup_child(&self, parent: EntityId, name: &str) -> Option<EntityId>;

    /// Recursively looks up a descendant of `parent` (or its `IsA` prefab) by `EntityName`.
    fn lookup_recursive(&self, parent: EntityId, name: &str) -> Option<EntityId>;

    /// Finds a top-level or arbitrary entity by `EntityName`.
    fn find_by_name(&self, name: &str) -> Option<EntityId>;
}

impl StoreExampleExt for Store {
    fn spawn_empty(&mut self) -> EntityId {
        self.spawn_auto().expect("spawn empty")
    }

    fn entity_named(&mut self, name: &str) -> EntityId {
        self.spawn((EntityName(name.to_string()),))
            .expect("spawn named entity")
    }

    fn ensure_entity_named(&mut self, name: &str) -> EntityId {
        if let Some(id) = self.find_by_name(name) {
            return id;
        }
        self.entity_named(name)
    }

    fn prefab_named(&mut self, name: &str) -> EntityId {
        self.spawn((EntityName(name.to_string()), PrefabTag))
            .expect("spawn named prefab")
    }

    fn name_of(&self, entity: EntityId) -> String {
        self.get::<EntityName>(entity)
            .map(|n| n.0.clone())
            .unwrap_or_else(|| format!("#{}", EntityHandle::index(entity)))
    }

    fn path_of(&self, entity: EntityId) -> String {
        let mut chain = Vec::new();
        let mut cur = Some(entity);
        while let Some(e) = cur {
            chain.push(self.name_of(e));
            cur = self.parent_of(e);
        }
        chain.reverse();
        format!("::{}", chain.join("::"))
    }

    fn set_child_of(&mut self, child: EntityId, parent: EntityId) {
        self.set_pair(child, ChildOf, parent).expect("set ChildOf");
    }

    fn set_parent(&mut self, child: EntityId, parent: EntityId) {
        self.set_child_of(child, parent);
    }

    fn set_is_a(&mut self, instance: EntityId, prefab: EntityId) {
        self.set_pair(instance, IsA, prefab).expect("set IsA");
    }

    fn add_is_a(&mut self, instance: EntityId, prefab: EntityId) {
        self.set_is_a(instance, prefab);
    }

    fn parent_of(&self, entity: EntityId) -> Option<EntityId> {
        self.pair_target::<ChildOf>(entity)
    }

    fn parent(&self, entity: EntityId) -> Option<EntityId> {
        self.parent_of(entity)
    }

    fn children_of(&self, parent: EntityId) -> Vec<EntityId> {
        let mut children = self.sources_with_pair::<ChildOf>(parent);
        children.sort();
        children
    }

    fn children(&self, parent: EntityId) -> Vec<EntityId> {
        self.children_of(parent)
    }

    fn prefabs_of(&self, entity: EntityId) -> Vec<EntityId> {
        self.pair_target::<IsA>(entity).into_iter().collect()
    }

    fn lookup_child(&self, parent: EntityId, name: &str) -> Option<EntityId> {
        for child in self.children_of(parent) {
            if self.name_of(child) == name {
                return Some(child);
            }
        }
        None
    }

    fn lookup_recursive(&self, parent: EntityId, name: &str) -> Option<EntityId> {
        if let Some(c) = self.lookup_child(parent, name) {
            return Some(c);
        }
        for child in self.children_of(parent) {
            if let Some(found) = self.lookup_recursive(child, name) {
                return Some(found);
            }
        }
        for prefab in self.prefabs_of(parent) {
            if let Some(found) = self.lookup_recursive(prefab, name) {
                return Some(found);
            }
        }
        None
    }

    fn find_by_name(&self, name: &str) -> Option<EntityId> {
        self.find_entity_by::<EntityName>(|n| n.0 == name)
    }
}
