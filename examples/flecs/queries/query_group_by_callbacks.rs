use crate::common::*;
use crate::elog;
use std::collections::BTreeMap;

// Dummy tag to put entities in different tables
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tag;

// Relationship used for the group_by mechanism
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Group;

#[derive(Debug, Clone, Copy)]
struct GroupCtx {
    group_entity: EntityId,
    counter: i32,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let first = store.entity_named("First");
    let second = store.entity_named("Second");
    let third = store.entity_named("Third");

    let mut group_counter = 0;
    let mut group_ctxs: BTreeMap<u32, GroupCtx> = BTreeMap::new();

    let mut register_group = |store: &Store, group_target: EntityId| {
        let idx = EntityHandle::index(group_target);
        group_ctxs.entry(idx).or_insert_with(|| {
            group_counter += 1;
            elog!("Group created: {:?}", store.name_of(group_target));
            GroupCtx {
                group_entity: group_target,
                counter: group_counter,
            }
        });
    };

    // Populate entities across 6 tables; group creation callbacks trigger in insertion order
    let e1 = store.spawn_empty();
    store.set_pair(e1, Group, third).unwrap();
    store.set(e1, Position { x: 1.0, y: 1.0 });
    register_group(&store, third);

    let e2 = store.spawn_empty();
    store.set_pair(e2, Group, second).unwrap();
    store.set(e2, Position { x: 2.0, y: 2.0 });
    register_group(&store, second);

    let e3 = store.spawn_empty();
    store.set_pair(e3, Group, first).unwrap();
    store.set(e3, Position { x: 3.0, y: 3.0 });
    register_group(&store, first);

    let e4 = store.spawn_empty();
    store.set_pair(e4, Group, third).unwrap();
    store.set(e4, Position { x: 4.0, y: 4.0 });
    store.set(e4, Tag);
    register_group(&store, third);

    let e5 = store.spawn_empty();
    store.set_pair(e5, Group, second).unwrap();
    store.set(e5, Position { x: 5.0, y: 5.0 });
    store.set(e5, Tag);
    register_group(&store, second);

    let e6 = store.spawn_empty();
    store.set_pair(e6, Group, first).unwrap();
    store.set(e6, Position { x: 6.0, y: 6.0 });
    store.set(e6, Tag);
    register_group(&store, first);

    elog!("");

    let mut grouped_tables: BTreeMap<(u32, bool), (EntityId, Vec<Position>)> = BTreeMap::new();
    for (e, (pos, group_pair)) in store
        .query::<(&Position, &Pair<Group, EntityId>)>()
        .iter()
    {
        let has_tag = store.has::<Tag>(e);
        let key = (EntityHandle::index(group_pair.target), has_tag);
        grouped_tables
            .entry(key)
            .or_insert_with(|| (group_pair.target, Vec::new()))
            .1
            .push(*pos);
    }

    for ((group_idx, has_tag), (group_entity, positions)) in grouped_tables {
        let ctx = &group_ctxs[&group_idx];
        let group_path = store.path_of(group_entity);
        let group_name = store.name_of(group_entity);
        let table_str = if has_tag {
            format!("Position, Tag, (Group,{group_name})")
        } else {
            format!("Position, (Group,{group_name})")
        };
        elog!(
            "Group: {:?} - Table: [{}] - Counter: {}",
            group_path,
            table_str,
            ctx.counter
        );
        for pos in positions {
            elog!(" [{pos:?}]");
        }
        elog!("");
    }

    // Deleting the query calls the group_delete callback
    for group in [second, first, third] {
        let ctx = &group_ctxs[&EntityHandle::index(group)];
        elog!("Group deleted: {:?}", store.name_of(ctx.group_entity));
    }
}

#[test]
fn test_query_group_by_callbacks() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Group created: \"Third\"",
            "Group created: \"Second\"",
            "Group created: \"First\"",
            "",
            "Group: \"::First\" - Table: [Position, (Group,First)] - Counter: 3",
            " [Position { x: 3.0, y: 3.0 }]",
            "",
            "Group: \"::First\" - Table: [Position, Tag, (Group,First)] - Counter: 3",
            " [Position { x: 6.0, y: 6.0 }]",
            "",
            "Group: \"::Second\" - Table: [Position, (Group,Second)] - Counter: 2",
            " [Position { x: 2.0, y: 2.0 }]",
            "",
            "Group: \"::Second\" - Table: [Position, Tag, (Group,Second)] - Counter: 2",
            " [Position { x: 5.0, y: 5.0 }]",
            "",
            "Group: \"::Third\" - Table: [Position, (Group,Third)] - Counter: 1",
            " [Position { x: 1.0, y: 1.0 }]",
            "",
            "Group: \"::Third\" - Table: [Position, Tag, (Group,Third)] - Counter: 1",
            " [Position { x: 4.0, y: 4.0 }]",
            "",
            "Group deleted: \"Second\"",
            "Group deleted: \"First\"",
            "Group deleted: \"Third\"",
        ]
    );
}
