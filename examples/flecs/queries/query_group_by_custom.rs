use crate::common::*;
use crate::elog;
use std::collections::BTreeMap;

// Dummy tag to put entities in different tables
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tag;

// Relationship used for the group_by mechanism
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Group;

fn custom_group_by(store: &Store, entity: EntityId) -> u64 {
    store
        .pair_target::<Group>(entity)
        .map(|target| EntityHandle::index(target) as u64)
        .unwrap_or(0)
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let first = store.entity_named("First");
    let second = store.entity_named("Second");
    let third = store.entity_named("Third");

    let e1 = store.spawn_empty();
    store.set_pair(e1, Group, third).unwrap();
    store.set(e1, Position { x: 1.0, y: 1.0 });

    let e2 = store.spawn_empty();
    store.set_pair(e2, Group, second).unwrap();
    store.set(e2, Position { x: 2.0, y: 2.0 });

    let e3 = store.spawn_empty();
    store.set_pair(e3, Group, first).unwrap();
    store.set(e3, Position { x: 3.0, y: 3.0 });

    let e4 = store.spawn_empty();
    store.set_pair(e4, Group, third).unwrap();
    store.set(e4, Position { x: 4.0, y: 4.0 });
    store.set(e4, Tag);

    let e5 = store.spawn_empty();
    store.set_pair(e5, Group, second).unwrap();
    store.set(e5, Position { x: 5.0, y: 5.0 });
    store.set(e5, Tag);

    let e6 = store.spawn_empty();
    store.set_pair(e6, Group, first).unwrap();
    store.set(e6, Position { x: 6.0, y: 6.0 });
    store.set(e6, Tag);

    let mut grouped_tables: BTreeMap<(u64, bool), (EntityId, Vec<Position>)> = BTreeMap::new();
    for (e, (pos, group_pair)) in store
        .query::<(&Position, &Pair<Group, EntityId>)>()
        .iter()
    {
        let group_id = custom_group_by(&store, e);
        let has_tag = store.has::<Tag>(e);
        grouped_tables
            .entry((group_id, has_tag))
            .or_insert_with(|| (group_pair.target, Vec::new()))
            .1
            .push(*pos);
    }

    for ((_group_id, has_tag), (group_entity, positions)) in grouped_tables {
        let group_path = store.path_of(group_entity);
        let group_name = store.name_of(group_entity);
        let table_str = if has_tag {
            format!("Position, Tag, (Group,{group_name})")
        } else {
            format!("Position, (Group,{group_name})")
        };
        elog!("Group: {:?} - Table: [{}]", group_path, table_str);
        for pos in positions {
            elog!(" [{pos:?}]");
        }
        elog!("");
    }
}

#[test]
fn test_query_group_by_custom() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Group: \"::First\" - Table: [Position, (Group,First)]",
            " [Position { x: 3.0, y: 3.0 }]",
            "",
            "Group: \"::First\" - Table: [Position, Tag, (Group,First)]",
            " [Position { x: 6.0, y: 6.0 }]",
            "",
            "Group: \"::Second\" - Table: [Position, (Group,Second)]",
            " [Position { x: 2.0, y: 2.0 }]",
            "",
            "Group: \"::Second\" - Table: [Position, Tag, (Group,Second)]",
            " [Position { x: 5.0, y: 5.0 }]",
            "",
            "Group: \"::Third\" - Table: [Position, (Group,Third)]",
            " [Position { x: 1.0, y: 1.0 }]",
            "",
            "Group: \"::Third\" - Table: [Position, Tag, (Group,Third)]",
            " [Position { x: 4.0, y: 4.0 }]",
            "",
        ]
    );
}
