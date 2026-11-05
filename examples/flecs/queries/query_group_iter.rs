use crate::common::*;
use crate::elog;
use std::collections::BTreeMap;

// A world cell relationship with four cells
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WorldCell;

// Npc tags
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Npc;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Merchant;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Soldier;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Beggar;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mage;

fn role_name(store: &Store, entity: EntityId) -> &'static str {
    if store.has::<Merchant>(entity) {
        "Merchant"
    } else if store.has::<Soldier>(entity) {
        "Soldier"
    } else if store.has::<Beggar>(entity) {
        "Beggar"
    } else if store.has::<Mage>(entity) {
        "Mage"
    } else {
        "Unknown"
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let cell_0_0 = store.entity_named("Cell_0_0");
    let cell_0_1 = store.entity_named("Cell_0_1");
    let cell_1_0 = store.entity_named("Cell_1_0");
    let cell_1_1 = store.entity_named("Cell_1_1");

    // Create NPCs in world cell 0_0
    let e1 = store.spawn_empty();
    store.set_pair(e1, WorldCell, cell_0_0).unwrap();
    store.set(e1, Merchant);
    store.set(e1, Npc);

    let e2 = store.spawn_empty();
    store.set_pair(e2, WorldCell, cell_0_0).unwrap();
    store.set(e2, Merchant);
    store.set(e2, Npc);

    // Create NPCs in world cell 0_1
    let e3 = store.spawn_empty();
    store.set_pair(e3, WorldCell, cell_0_1).unwrap();
    store.set(e3, Beggar);
    store.set(e3, Npc);

    let e4 = store.spawn_empty();
    store.set_pair(e4, WorldCell, cell_0_1).unwrap();
    store.set(e4, Soldier);
    store.set(e4, Npc);

    // Create NPCs in world cell 1_0
    let e5 = store.spawn_empty();
    store.set_pair(e5, WorldCell, cell_1_0).unwrap();
    store.set(e5, Mage);
    store.set(e5, Npc);

    let e6 = store.spawn_empty();
    store.set_pair(e6, WorldCell, cell_1_0).unwrap();
    store.set(e6, Beggar);
    store.set(e6, Npc);

    // Create NPCs in world cell 1_1
    let e7 = store.spawn_empty();
    store.set_pair(e7, WorldCell, cell_1_1).unwrap();
    store.set(e7, Soldier);
    store.set(e7, Npc);

    // Build grouped table descriptors ordered by cell entity index and creation order of tables
    let mut matched: Vec<(EntityId, EntityId)> = store
        .query::<&Pair<WorldCell, EntityId>>()
        .with::<Npc>()
        .iter()
        .map(|(entity, pair)| (entity, pair.target))
        .collect();
    matched.sort_by_key(|(entity, _)| EntityHandle::index(*entity));

    let mut groups: BTreeMap<u32, (EntityId, Vec<String>)> = BTreeMap::new();
    for (entity, cell) in matched {
        let role = role_name(&store, entity);
        let cell_name = store.name_of(cell);
        let table_str = format!("Npc, {role}, (WorldCell,{cell_name})");
        let tables = &mut groups
            .entry(EntityHandle::index(cell))
            .or_insert_with(|| (cell, Vec::new()))
            .1;
        if !tables.contains(&table_str) {
            tables.push(table_str);
        }
    }

    elog!("All tables:");
    for (_idx, (cell, tables)) in &groups {
        let group_path = store.path_of(*cell);
        for table_str in tables {
            elog!(" - group {:?}: table [{}]", group_path, table_str);
        }
    }

    elog!("");

    elog!("Tables for cell 1_0:");
    if let Some((cell, tables)) = groups.get(&EntityHandle::index(cell_1_0)) {
        let group_path = store.path_of(*cell);
        for table_str in tables {
            elog!(" - group {:?}: table [{}]", group_path, table_str);
        }
    }
}

#[test]
fn test_query_group_iter() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "All tables:",
            " - group \"::Cell_0_0\": table [Npc, Merchant, (WorldCell,Cell_0_0)]",
            " - group \"::Cell_0_1\": table [Npc, Beggar, (WorldCell,Cell_0_1)]",
            " - group \"::Cell_0_1\": table [Npc, Soldier, (WorldCell,Cell_0_1)]",
            " - group \"::Cell_1_0\": table [Npc, Mage, (WorldCell,Cell_1_0)]",
            " - group \"::Cell_1_0\": table [Npc, Beggar, (WorldCell,Cell_1_0)]",
            " - group \"::Cell_1_1\": table [Npc, Soldier, (WorldCell,Cell_1_1)]",
            "",
            "Tables for cell 1_0:",
            " - group \"::Cell_1_0\": table [Npc, Mage, (WorldCell,Cell_1_0)]",
            " - group \"::Cell_1_0\": table [Npc, Beggar, (WorldCell,Cell_1_0)]",
        ]
    );
}
